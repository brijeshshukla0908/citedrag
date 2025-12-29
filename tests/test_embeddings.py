# test_embeddings.py
"""
Unit Tests for Embeddings Module
=================================
Tests for embedding generation, caching, and similarity.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.embeddings import (
    EmbeddingGenerator,
    cosine_similarity
)


class TestEmbeddingGenerator:
    """Test suite for EmbeddingGenerator class"""
    
    @pytest.fixture
    def generator(self):
        """Create generator instance for tests"""
        return EmbeddingGenerator()
    
    
    def test_initialization(self, generator):
        """Test generator initialization"""
        assert generator.model is not None
        assert generator.get_embedding_dimension() == 768
    
    
    def test_single_embedding(self, generator):
        """Test single text embedding generation"""
        text = "This is a test sentence."
        embedding = generator.generate_embedding(text)
        
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (768,)
        assert not np.all(embedding == 0)  # Should not be all zeros
    
    
    def test_batch_embeddings(self, generator):
        """Test batch embedding generation"""
        chunks = [
            {"chunk_id": 0, "text": "First chunk of text."},
            {"chunk_id": 1, "text": "Second chunk of text."},
            {"chunk_id": 2, "text": "Third chunk of text."}
        ]
        
        result = generator.batch_generate_embeddings(
            chunks,
            show_progress=False
        )
        
        assert len(result) == 3
        for chunk in result:
            assert 'embedding' in chunk
            assert chunk['embedding'].shape == (768,)
    
    
    def test_caching(self, generator):
        """Test embedding caching"""
        text = "Test text for caching."
        
        # Generate twice
        emb1 = generator.generate_embedding(text, use_cache=True)
        emb2 = generator.generate_embedding(text, use_cache=True)
        
        # Should be identical (cached)
        np.testing.assert_array_equal(emb1, emb2)
    
    
    def test_cosine_similarity(self):
        """Test cosine similarity calculation"""
        vec1 = np.array([1.0, 0.0, 0.0])
        vec2 = np.array([1.0, 0.0, 0.0])
        vec3 = np.array([0.0, 1.0, 0.0])
        
        # Identical vectors
        assert abs(cosine_similarity(vec1, vec2) - 1.0) < 0.01
        
        # Orthogonal vectors
        assert abs(cosine_similarity(vec1, vec3)) < 0.01
    
    
    def test_cache_stats(self, generator):
        """Test cache statistics"""
        # Generate some embeddings
        generator.generate_embedding("Test 1")
        generator.generate_embedding("Test 2")
        
        stats = generator.get_cache_stats()
        
        assert 'memory_cache_entries' in stats
        assert 'disk_cache_entries' in stats
        assert stats['embedding_dimension'] == 768


# Run tests if executed directly
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
