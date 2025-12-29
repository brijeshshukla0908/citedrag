"""
Unit Tests for Keyword Search
==============================
Tests for BM25 indexing and search.
"""

import pytest
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.keyword_search import BM25Search


class TestBM25Search:
    """Test suite for BM25Search class"""
    
    @pytest.fixture
    def sample_chunks(self):
        """Sample chunks for testing"""
        return [
            {
                'chunk_id': 0,
                'text': 'The quick brown fox jumps over the lazy dog.',
                'page_number': 1
            },
            {
                'chunk_id': 1,
                'text': 'A fast brown fox leaps across a sleeping canine.',
                'page_number': 2
            },
            {
                'chunk_id': 2,
                'text': 'Dogs and foxes are both mammals.',
                'page_number': 3
            }
        ]
    
    
    def test_initialization(self):
        """Test BM25 initialization"""
        bm25 = BM25Search()
        assert bm25.bm25_index is None
        assert len(bm25.chunks) == 0
    
    
    def test_tokenization(self):
        """Test text tokenization"""
        bm25 = BM25Search()
        
        text = "Hello, World! This is a TEST."
        tokens = bm25._tokenize(text)
        
        assert 'hello' in tokens
        assert 'world' in tokens
        assert 'test' in tokens
        assert len(tokens) > 0
    
    
    def test_build_index(self, sample_chunks):
        """Test building BM25 index"""
        bm25 = BM25Search()
        indexed = bm25.build_index(sample_chunks)
        
        assert indexed == 3
        assert bm25.bm25_index is not None
        assert len(bm25.chunks) == 3
    
    
    def test_search(self, sample_chunks):
        """Test BM25 search"""
        bm25 = BM25Search()
        bm25.build_index(sample_chunks)
        
        results = bm25.search("fox brown", top_k=2)
        
        assert len(results) >= 1
        assert 'bm25_score' in results[0]
        assert 'chunk' in results[0]
        assert results[0]['bm25_score'] > 0
    
    
    def test_get_scores(self, sample_chunks):
        """Test getting scores for all chunks"""
        bm25 = BM25Search()
        bm25.build_index(sample_chunks)
        
        scores = bm25.get_scores("fox dog")
        
        assert len(scores) == 3
        assert all(score >= 0 for score in scores)
    
    
    def test_get_stats(self, sample_chunks):
        """Test index statistics"""
        bm25 = BM25Search()
        
        # Before indexing
        stats = bm25.get_stats()
        assert stats['indexed'] == False
        
        # After indexing
        bm25.build_index(sample_chunks)
        stats = bm25.get_stats()
        
        assert stats['indexed'] == True
        assert stats['total_chunks'] == 3
        assert stats['total_tokens'] > 0
    
    
    def test_save_load(self, sample_chunks, tmp_path):
        """Test saving and loading index"""
        # Build and save
        bm25 = BM25Search(cache_dir=str(tmp_path))
        bm25.build_index(sample_chunks)
        bm25.save_index("test_index.pkl")
        
        # Load in new instance
        bm25_new = BM25Search(cache_dir=str(tmp_path))
        loaded = bm25_new.load_index("test_index.pkl")
        
        assert loaded == True
        assert len(bm25_new.chunks) == 3
        
        # Verify search works
        results = bm25_new.search("fox", top_k=1)
        assert len(results) >= 1


# Run tests if executed directly
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
