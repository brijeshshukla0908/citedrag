# llm_handler.py
"""
LLM Handler Module
==================
Handles LLM interactions via Groq API for answer generation with citations.

Key Functions:
- generate_answer(): Generate answer from retrieved context
- build_prompt(): Create prompt with context and instructions
- extract_citations(): Parse citations from LLM response
"""

import sys
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import re
from groq import Groq
from loguru import logger

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import config


class LLMHandler:
    """
    Handles LLM interactions for generating answers with citations.
    Uses Groq API for fast inference.
    """
    
    def __init__(
        self,
        api_key: str = config.GROQ_API_KEY,
        model: str = config.LLM_MODEL,
        temperature: float = config.LLM_TEMPERATURE,
        max_tokens: int = config.LLM_MAX_TOKENS
    ):
        """
        Initialize LLM handler.
        
        Args:
            api_key: Groq API key
            model: Model name (e.g., llama-3.1-8b-instant)
            temperature: Sampling temperature (0-2)
            max_tokens: Maximum tokens in response
        """
        if not api_key:
            raise ValueError("GROQ_API_KEY is required. Set it in .env file.")
        
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Initialize Groq client
        self.client = Groq(api_key=api_key)
        
        logger.info(f"LLMHandler initialized with model: {model}")
    
    
    def generate_answer(
        self,
        query: str,
        context_chunks: List[Dict],
        system_prompt: Optional[str] = None
    ) -> Dict:
        """
        Generate answer from query and retrieved context.
        
        Args:
            query: User's question
            context_chunks: Retrieved chunks from hybrid search
            system_prompt: Optional system prompt (uses default if None)
            
        Returns:
            Dictionary with answer, citations, and metadata
        """
        logger.info(f"Generating answer for: '{query[:100]}'")
        
        # Build prompt
        if system_prompt is None:
            system_prompt = self._get_default_system_prompt()
        
        user_prompt = self._build_user_prompt(query, context_chunks)
        
        logger.debug(f"System prompt length: {len(system_prompt)} chars")
        logger.debug(f"User prompt length: {len(user_prompt)} chars")
        
        # Call Groq API
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            # Extract response
            answer = response.choices[0].message.content
            
            # Get token usage
            token_usage = {
                'prompt_tokens': response.usage.prompt_tokens,
                'completion_tokens': response.usage.completion_tokens,
                'total_tokens': response.usage.total_tokens
            }
            
            logger.info(f"✅ Answer generated ({token_usage['completion_tokens']} tokens)")
            
            # Extract citations
            citations = self._extract_citations(answer, context_chunks)
            
            return {
                'answer': answer,
                'citations': citations,
                'token_usage': token_usage,
                'model': self.model,
                'query': query,
                'num_context_chunks': len(context_chunks)
            }
            
        except Exception as e:
            logger.error(f"Failed to generate answer: {str(e)}")
            raise RuntimeError(f"LLM generation failed: {str(e)}")
    
    
    def _get_default_system_prompt(self) -> str:
        """
        Get default system prompt for answer generation.
        
        Returns:
            System prompt string
        """
        return """You are a helpful AI assistant that answers questions based on provided context.

INSTRUCTIONS:
1. Answer the question using ONLY the information from the provided context
2. If the answer cannot be found in the context, say "I cannot find this information in the provided documents"
3. Always cite your sources using [Chunk X] notation when referencing information
4. Be concise and accurate
5. If multiple chunks support your answer, cite all relevant chunks
6. Do not make up information or use knowledge outside the provided context

CITATION FORMAT:
- Use [Chunk X] at the end of sentences where you reference information from Chunk X
- Example: "Employees are entitled to 15 days of leave [Chunk 3]."
- For information from multiple chunks: "The policy covers remote work and leave benefits [Chunk 2][Chunk 5]."

Remember: ONLY use information from the provided context chunks."""
    
    
    def _build_user_prompt(
        self,
        query: str,
        context_chunks: List[Dict]
    ) -> str:
        """
        Build user prompt with query and context.
        
        Args:
            query: User's question
            context_chunks: Retrieved chunks
            
        Returns:
            Formatted user prompt
        """
        prompt_parts = []
        
        # Add context
        prompt_parts.append("CONTEXT:\n")
        
        for i, chunk_data in enumerate(context_chunks):
            chunk = chunk_data.get('chunk', {})
            chunk_id = chunk.get('chunk_id', i)
            page = chunk.get('page_number', 'unknown')
            text = chunk.get('text', '')
            
            prompt_parts.append(f"[Chunk {chunk_id}] (Page {page})")
            prompt_parts.append(text)
            prompt_parts.append("")  # Blank line between chunks
        
        # Add question
        prompt_parts.append("---\n")
        prompt_parts.append(f"QUESTION: {query}\n")
        prompt_parts.append("ANSWER:")
        
        return "\n".join(prompt_parts)
    
    
    def _extract_citations(
        self,
        answer: str,
        context_chunks: List[Dict]
    ) -> List[Dict]:
        """
        Extract citation information from answer.
        
        Args:
            answer: Generated answer text
            context_chunks: Original context chunks
            
        Returns:
            List of citation dictionaries
        """
        # Find all [Chunk X] citations in answer
        citation_pattern = r'\[Chunk (\d+)\]'
        matches = re.findall(citation_pattern, answer)
        
        if not matches:
            return []
        
        # Build citation map
        chunk_map = {}
        for chunk_data in context_chunks:
            chunk = chunk_data.get('chunk', {})
            chunk_id = chunk.get('chunk_id')
            if chunk_id is not None:
                chunk_map[chunk_id] = chunk_data
        
        # Extract unique citations
        citations = []
        seen_chunks = set()
        
        for chunk_id_str in matches:
            chunk_id = int(chunk_id_str)
            
            if chunk_id in seen_chunks:
                continue
            
            seen_chunks.add(chunk_id)
            
            # Get chunk data
            chunk_data = chunk_map.get(chunk_id)
            if chunk_data:
                chunk = chunk_data.get('chunk', {})
                citations.append({
                    'chunk_id': chunk_id,
                    'page_number': chunk.get('page_number', 'unknown'),
                    'document_name': chunk.get('document_name', 'unknown'),
                    'text_preview': chunk.get('text', '')[:100] + "..." if len(chunk.get('text', '')) > 100 else chunk.get('text', ''),
                    'retrieval_score': chunk_data.get('hybrid_score', 0.0)
                })
        
        return citations
    
    
    def generate_streaming_answer(
        self,
        query: str,
        context_chunks: List[Dict],
        system_prompt: Optional[str] = None
    ):
        """
        Generate answer with streaming (for real-time display).
        
        Args:
            query: User's question
            context_chunks: Retrieved chunks
            system_prompt: Optional system prompt
            
        Yields:
            Answer text chunks as they're generated
        """
        if system_prompt is None:
            system_prompt = self._get_default_system_prompt()
        
        user_prompt = self._build_user_prompt(query, context_chunks)
        
        try:
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stream=True
            )
            
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        
        except Exception as e:
            logger.error(f"Streaming generation failed: {str(e)}")
            raise RuntimeError(f"Streaming failed: {str(e)}")


# ==========================================
# Utility Functions
# ==========================================

def format_answer_with_citations(result: Dict) -> str:
    """
    Format answer with citations for display.
    
    Args:
        result: Result dictionary from generate_answer()
        
    Returns:
        Formatted string
    """
    output = []
    
    output.append("\n" + "="*60)
    output.append("ANSWER")
    output.append("="*60 + "\n")
    
    output.append(result['answer'])
    
    if result['citations']:
        output.append("\n" + "="*60)
        output.append("CITATIONS")
        output.append("="*60 + "\n")
        
        for i, citation in enumerate(result['citations']):
            output.append(f"{i+1}. [Chunk {citation['chunk_id']}] - Page {citation['page_number']}")
            output.append(f"   Document: {citation['document_name']}")
            output.append(f"   Preview: {citation['text_preview']}")
            output.append(f"   Score: {citation['retrieval_score']:.4f}")
            output.append("")
    
    output.append("="*60)
    output.append("METADATA")
    output.append("="*60)
    output.append(f"Model: {result['model']}")
    output.append(f"Context chunks: {result['num_context_chunks']}")
    output.append(f"Tokens used: {result['token_usage']['total_tokens']} (prompt: {result['token_usage']['prompt_tokens']}, completion: {result['token_usage']['completion_tokens']})")
    output.append("="*60 + "\n")
    
    return "\n".join(output)


# ==========================================
# Testing & Demo
# ==========================================

if __name__ == "__main__":
    """
    Test LLM handler with sample context.
    """
    import sys
    
    # Setup logging
    logger.remove()
    logger.add(sys.stdout, level="INFO")
    
    print("\n" + "="*60)
    print("LLM Handler Test")
    print("="*60 + "\n")
    
    # Check API key
    if not config.GROQ_API_KEY or config.GROQ_API_KEY == "your_groq_api_key_here":
        print("❌ Error: GROQ_API_KEY not set")
        print("Please set your API key in .env file")
        sys.exit(1)
    
    # Initialize handler
    print("🔄 Initializing LLM handler...")
    llm = LLMHandler()
    print(f"✅ Handler initialized with model: {llm.model}")
    
    # Create sample context
    print("\n📝 Creating sample context...")
    sample_chunks = [
        {
            'chunk': {
                'chunk_id': 0,
                'text': 'Employees are entitled to 15 days of paid annual leave per year. Leave must be approved by the manager.',
                'page_number': 5,
                'document_name': 'employee_handbook.pdf'
            },
            'hybrid_score': 0.85
        },
        {
            'chunk': {
                'chunk_id': 1,
                'text': 'Remote work is allowed up to 3 days per week. Employees must maintain regular communication with their team.',
                'page_number': 12,
                'document_name': 'employee_handbook.pdf'
            },
            'hybrid_score': 0.72
        }
    ]
    
    # Test question
    query = "How many days of leave do employees get?"
    
    print(f"\n❓ Question: {query}")
    print("\n🤖 Generating answer...")
    
    try:
        result = llm.generate_answer(query, sample_chunks)
        
        print("\n" + format_answer_with_citations(result))
        
        print("✅ Test completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
