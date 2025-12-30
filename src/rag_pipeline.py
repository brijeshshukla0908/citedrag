"""
RAG Pipeline Module
===================
Complete RAG pipeline with caching and rate limiting integrated.

This is the main entry point for the RAG system.
"""

import sys
from pathlib import Path
from typing import Dict, Optional, Tuple
from loguru import logger

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.document_processor import DocumentProcessor
from src.embeddings import EmbeddingGenerator
from src.vector_store import VectorStore
from src.keyword_search import BM25Search
from src.retrieval import HybridRetriever
from src.llm_handler import LLMHandler
from src.cache_manager import CacheManager
from src.rate_limiter import RateLimiter
from src.demo_questions import DemoQuestionsManager
import config


class RAGPipeline:
    """
    Complete RAG pipeline with all components integrated.
    Includes caching, rate limiting, and demo questions.
    """
    
    def __init__(
        self,
        enable_cache: bool = config.CACHE_ENABLED,
        enable_rate_limit: bool = True
    ):
        """
        Initialize RAG pipeline.
        
        Args:
            enable_cache: Enable response caching
            enable_rate_limit: Enable rate limiting
        """
        logger.info("Initializing RAG Pipeline...")
        
        # Core components
        self.document_processor = None
        self.embedding_generator = None
        self.vector_store = None
        self.bm25_search = None
        self.retriever = None
        self.llm_handler = None
        
        # Production components
        self.cache_manager = CacheManager() if enable_cache else None
        self.rate_limiter = RateLimiter() if enable_rate_limit else None
        self.demo_manager = DemoQuestionsManager()
        
        # State
        self.current_document = None
        self.is_ready = False
        
        logger.info("✅ RAG Pipeline initialized")
    
    
    def load_document(self, pdf_path: str) -> Dict:
        """
        Load and process a PDF document.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Processing summary dictionary
        """
        logger.info(f"Loading document: {pdf_path}")
        
        try:
            # Step 1: Process document
            self.document_processor = DocumentProcessor()
            chunks, metadata = self.document_processor.process_document(pdf_path)
            
            # Step 2: Generate embeddings
            if self.embedding_generator is None:
                self.embedding_generator = EmbeddingGenerator()
            
            chunks_with_embeddings = self.embedding_generator.batch_generate_embeddings(
                chunks,
                show_progress=True
            )
            
            # Step 3: Store in vector database
            document_name = Path(pdf_path).name
            self.vector_store = VectorStore(reset=True)
            self.vector_store.add_chunks(chunks_with_embeddings, document_name=document_name)
            
            # Step 4: Build BM25 index
            self.bm25_search = BM25Search()
            self.bm25_search.build_index(chunks)
            
            # Step 5: Initialize retriever
            self.retriever = HybridRetriever(
                vector_store=self.vector_store,
                bm25_search=self.bm25_search,
                embedding_generator=self.embedding_generator
            )
            
            # Step 6: Initialize LLM handler
            if self.llm_handler is None:
                self.llm_handler = LLMHandler()
            
            # Update state
            self.current_document = document_name
            self.is_ready = True
            
            # Clear cache for this document
            if self.cache_manager:
                self.cache_manager.clear_cache(document_name=document_name)
            
            summary = {
                'document_name': document_name,
                'total_pages': metadata['total_pages'],
                'total_chunks': len(chunks),
                'total_tokens': metadata['estimated_tokens'],
                'status': 'ready'
            }
            
            logger.info(f"✅ Document loaded successfully: {document_name}")
            return summary
            
        except Exception as e:
            logger.error(f"Failed to load document: {str(e)}")
            raise RuntimeError(f"Document loading failed: {str(e)}")
    
    
    def query(
        self,
        question: str,
        user_id: str,
        use_cache: bool = True,
        is_demo: bool = False
    ) -> Tuple[Dict, str]:
        """
        Query the RAG system.
        
        Args:
            question: User's question
            user_id: Unique user identifier
            use_cache: Use cached response if available
            is_demo: Whether this is a demo question (skip rate limiting)
            
        Returns:
            Tuple of (response_dict, status_message)
        """
        if not self.is_ready:
            return None, "No document loaded. Please load a document first."
        
        # Check rate limit (skip for demo questions)
        if not is_demo and self.rate_limiter:
            allowed, reason = self.rate_limiter.check_limit(user_id)
            if not allowed:
                logger.warning(f"Rate limit exceeded for user: {user_id}")
                return None, reason
        
        # Check cache
        if use_cache and self.cache_manager:
            cached_response = self.cache_manager.get_cached_response(
                question,
                document_name=self.current_document
            )
            
            if cached_response:
                logger.info(f"✅ Returning cached response")
                return cached_response, "cached"
        
        try:
            # Retrieve relevant chunks
            logger.info(f"Retrieving chunks for: {question[:50]}...")
            context_chunks = self.retriever.hybrid_search(question, top_k=config.TOP_K_RETRIEVAL)
            
            # Generate answer
            logger.info("Generating answer...")
            response = self.llm_handler.generate_answer(question, context_chunks)
            
            # Cache response
            if use_cache and self.cache_manager:
                self.cache_manager.cache_response(
                    question,
                    response,
                    document_name=self.current_document
                )
            
            # Record request (skip for demo)
            if not is_demo and self.rate_limiter:
                self.rate_limiter.record_request(user_id)
            
            logger.info("✅ Query completed successfully")
            return response, "success"
            
        except Exception as e:
            logger.error(f"Query failed: {str(e)}")
            return None, f"Error: {str(e)}"
    
    
    def get_demo_question_response(
        self,
        question_id: int,
        user_id: str
    ) -> Tuple[Dict, str]:
        """
        Get response for a demo question (pre-computed or generate new).
        
        Args:
            question_id: Demo question ID
            user_id: User identifier
            
        Returns:
            Tuple of (response_dict, status_message)
        """
        if not self.is_ready:
            return None, "No document loaded."
        
        # Get demo question
        demo_question = self.demo_manager.get_question_by_id(question_id)
        if not demo_question:
            return None, f"Demo question {question_id} not found."
        
        question_text = demo_question['question']
        
        # Check if current document is the sample document
        is_sample_doc = self.current_document in ["test.pdf", "test.pdf (Sample)"]
        
        # If sample document, try to use pre-computed response
        if is_sample_doc and demo_question.get('has_response', False):
            logger.info(f"✅ Returning pre-computed demo response for sample document")
            return demo_question['response'], "demo_precomputed"
        
        # For custom documents or missing responses, generate fresh
        if not is_sample_doc:
            logger.info(f"🔄 Generating fresh response for custom document: {self.current_document}")
        
        # Generate response (mark as demo to skip rate limiting)
        response, status = self.query(question_text, user_id, use_cache=True, is_demo=True)
        
        # Don't save to demo responses (to avoid overwriting sample doc responses)
        # Only cache in regular cache for this specific document
        
        return response, "demo_generated"

    
    
    def get_user_quota(self, user_id: str) -> Optional[Dict]:
        """
        Get remaining quota for user.
        
        Args:
            user_id: User identifier
            
        Returns:
            Quota dictionary
        """
        if not self.rate_limiter:
            return None
        
        return self.rate_limiter.get_remaining_quota(user_id)
    
    
    def get_pipeline_stats(self) -> Dict:
        """
        Get statistics about the pipeline.
        
        Returns:
            Statistics dictionary
        """
        stats = {
            'document_loaded': self.is_ready,
            'current_document': self.current_document
        }
        
        if self.cache_manager:
            stats['cache'] = self.cache_manager.get_cache_stats()
        
        if self.rate_limiter:
            stats['rate_limiter'] = self.rate_limiter.get_stats()
        
        if self.demo_manager:
            stats['demo_questions'] = self.demo_manager.get_statistics()
        
        if self.vector_store and self.is_ready:
            stats['vector_store'] = self.vector_store.get_collection_stats()
        
        if self.bm25_search and self.is_ready:
            stats['bm25'] = self.bm25_search.get_stats()
        
        return stats


# ==========================================
# Testing & Demo
# ==========================================

if __name__ == "__main__":
    """
    Test RAG pipeline with all components.
    """
    
    # Setup logging
    logger.remove()
    logger.add(sys.stdout, level="INFO")
    
    print("\n" + "="*70)
    print("RAG PIPELINE WITH CACHING & RATE LIMITING TEST")
    print("="*70 + "\n")
    
    # Check API key
    if not config.GROQ_API_KEY or config.GROQ_API_KEY == "your_groq_api_key_here":
        print("❌ Error: GROQ_API_KEY not set")
        sys.exit(1)
    
    # Initialize pipeline
    print("🔄 Initializing RAG Pipeline...")
    pipeline = RAGPipeline(enable_cache=True, enable_rate_limit=True)
    print("✅ Pipeline initialized\n")
    
    # Load document
    pdf_path = "data/sample_documents/test.pdf"
    if not Path(pdf_path).exists():
        print(f"❌ Error: PDF not found at {pdf_path}")
        sys.exit(1)
    
    print("📄 Loading document...")
    summary = pipeline.load_document(pdf_path)
    print(f"✅ Document loaded: {summary['document_name']}")
    print(f"   Pages: {summary['total_pages']}, Chunks: {summary['total_chunks']}\n")
    
    # Test queries
    user_id = "test_user_001"
    
    print("="*70)
    print("TEST 1: First Query (Cache Miss)")
    print("="*70)
    question1 = "What is the Nagarro Constitution about?"
    response1, status1 = pipeline.query(question1, user_id)
    print(f"\n✅ Status: {status1}")
    print(f"Answer: {response1['answer'][:200]}...")
    print(f"Citations: {len(response1['citations'])}")
    print(f"Tokens: {response1['token_usage']['total_tokens']}\n")
    
    print("="*70)
    print("TEST 2: Same Query (Cache Hit)")
    print("="*70)
    response2, status2 = pipeline.query(question1, user_id)
    print(f"\n✅ Status: {status2}")
    assert status2 == "cached", "Should be cache hit"
    print("✅ Cache working correctly!\n")
    
    print("="*70)
    print("TEST 3: Check User Quota")
    print("="*70)
    quota = pipeline.get_user_quota(user_id)
    print(f"\nHourly: {quota['hourly']['used']}/{quota['hourly']['limit']} used")
    print(f"Daily: {quota['daily']['used']}/{quota['daily']['limit']} used")
    print(f"Remaining: {quota['hourly']['remaining']} queries this hour\n")
    
    print("="*70)
    print("TEST 4: Pipeline Statistics")
    print("="*70)
    stats = pipeline.get_pipeline_stats()
    print(f"\nDocument: {stats['current_document']}")
    print(f"Cache entries: {stats['cache']['total_entries']}")
    print(f"Active users: {stats['rate_limiter']['active_users']}")
    print(f"Demo questions: {stats['demo_questions']['total_questions']}")
    
    print("\n" + "="*70)
    print("✅ ALL TESTS PASSED - PRODUCTION-READY PIPELINE!")
    print("="*70 + "\n")
