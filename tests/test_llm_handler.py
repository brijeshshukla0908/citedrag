# test_llm_handler.py
"""
Unit Tests for LLM Handler
===========================
Tests for LLM answer generation and citation extraction.
"""

import pytest
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.llm_handler import LLMHandler
import config


class TestLLMHandler:
    """Test suite for LLMHandler class"""
    
    @pytest.fixture
    def sample_chunks(self):
        """Sample chunks for testing"""
        return [
            {
                'chunk': {
                    'chunk_id': 0,
                    'text': 'Employees get 15 days of leave annually.',
                    'page_number': 5,
                    'document_name': 'handbook.pdf'
                },
                'hybrid_score': 0.85
            },
            {
                'chunk': {
                    'chunk_id': 1,
                    'text': 'Remote work allowed 3 days per week.',
                    'page_number': 12,
                    'document_name': 'handbook.pdf'
                },
                'hybrid_score': 0.72
            }
        ]
    
    
    def test_initialization(self):
        """Test handler initialization"""
        # Skip if no API key
        if not config.GROQ_API_KEY or config.GROQ_API_KEY == "your_groq_api_key_here":
            pytest.skip("GROQ_API_KEY not set")
        
        llm = LLMHandler()
        assert llm.client is not None
        assert llm.model == config.LLM_MODEL
    
    
    def test_build_user_prompt(self, sample_chunks):
        """Test prompt building"""
        if not config.GROQ_API_KEY or config.GROQ_API_KEY == "your_groq_api_key_here":
            pytest.skip("GROQ_API_KEY not set")
        
        llm = LLMHandler()
        prompt = llm._build_user_prompt("Test question?", sample_chunks)
        
        assert "CONTEXT:" in prompt
        assert "QUESTION:" in prompt
        assert "Chunk 0" in prompt
        assert "Chunk 1" in prompt
    
    
    def test_extract_citations(self, sample_chunks):
        """Test citation extraction"""
        if not config.GROQ_API_KEY or config.GROQ_API_KEY == "your_groq_api_key_here":
            pytest.skip("GROQ_API_KEY not set")
        
        llm = LLMHandler()
        answer = "Employees get 15 days leave [Chunk 0]. Remote work is allowed [Chunk 1]."
        
        citations = llm._extract_citations(answer, sample_chunks)
        
        assert len(citations) == 2
        assert citations[0]['chunk_id'] == 0
        assert citations[1]['chunk_id'] == 1
    
    
    def test_generate_answer(self, sample_chunks):
        """Test answer generation"""
        if not config.GROQ_API_KEY or config.GROQ_API_KEY == "your_groq_api_key_here":
            pytest.skip("GROQ_API_KEY not set")
        
        llm = LLMHandler()
        result = llm.generate_answer("How many leave days?", sample_chunks)
        
        assert 'answer' in result
        assert 'citations' in result
        assert 'token_usage' in result
        assert len(result['answer']) > 0
        assert result['token_usage']['total_tokens'] > 0


# Run tests if executed directly
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
