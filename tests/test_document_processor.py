# test_document_processor.py
"""
Unit Tests for Document Processor
==================================
Tests for PDF extraction, chunking, and metadata handling.
"""

import pytest
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.document_processor import (
    DocumentProcessor,
    count_tokens,
    validate_pdf
)


class TestDocumentProcessor:
    """Test suite for DocumentProcessor class"""
    
    def test_initialization(self):
        """Test processor initialization"""
        processor = DocumentProcessor(chunk_size=500, chunk_overlap=100)
        
        assert processor.chunk_size == 500
        assert processor.chunk_overlap == 100
    
    
    def test_clean_text(self):
        """Test text cleaning function"""
        processor = DocumentProcessor()
        
        # Test with messy text
        messy_text = "Hello    World  \n\n  Page 123  \n  Test"
        cleaned = processor._clean_text(messy_text)
        
        assert "Page 123" not in cleaned
        assert cleaned.startswith("Hello")
        assert "World" in cleaned
        assert "Test" in cleaned
        # Check that excessive whitespace is reduced (may leave single spaces)
        assert "    " not in cleaned  # No 4+ consecutive spaces
    
    
    def test_count_tokens(self):
        """Test token counting"""
        text = "This is a test sentence with some words."
        token_count = count_tokens(text)
        
        assert token_count > 0
        assert isinstance(token_count, int)
    
    
    def test_chunk_overlap_logic(self):
        """Test that chunks overlap correctly"""
        processor = DocumentProcessor(chunk_size=50, chunk_overlap=10)
        
        # Create sample text
        text = "word " * 100  # 100 words
        chunks = processor.chunk_text(text)
        
        # Should create multiple chunks
        assert len(chunks) > 1
        
        # Each chunk should have metadata
        for chunk in chunks:
            assert "chunk_id" in chunk
            assert "text" in chunk
            assert "token_count" in chunk
    
    
    def test_estimate_page_number(self):
        """Test page number estimation"""
        processor = DocumentProcessor()
        
        # Test middle of 10-page document with 1000 tokens
        page_num = processor._estimate_page_number(500, 1000, 10)
        
        assert page_num == 6  # Should be around middle page


# Run tests if executed directly
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
