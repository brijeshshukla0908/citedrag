"""
Unit Tests for Vector Store
============================
Tests for ChromaDB operations and vector search.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.vector_store import VectorStore, format_search_results


class TestVectorStore:
    """Test suite for VectorStore class"""
    
    @pytest.fixture
    def vector_store(self):
        """Create fresh vector store for each test"""
        return VectorStore(collection_name="test_collection", reset=True)
    
    
    @pytest.fixture
    def sample_chunks(self):
        """Sample chunks with embeddings"""
        return [
            {
                'chunk_id': 0,
                'text': 'Sample text one about policies.',
                'embedding': np.random.rand(768).tolist(),
                'page_number': 1,
                'token_count': 6
            },
            {
                'chunk_id': 1,
                'text': 'Sample text two about procedures.',
                'embedding': np.random.rand(768).tolist(),
                'page_number': 2,
                'token_count': 6
            }
        ]
    
    
    def test_initialization(self, vector_store):
        """Test vector store initialization"""
        assert vector_store.collection is not None
        assert vector_store.collection.count() == 0
    
    
    def test_add_chunks(self, vector_store, sample_chunks):
        """Test adding chunks to store"""
        added = vector_store.add_chunks(sample_chunks, document_name="test.pdf")
        
        assert added == 2
        assert vector_store.collection.count() == 2
    
    
    def test_similarity_search(self, vector_store, sample_chunks):
        """Test similarity search"""
        # Add chunks
        vector_store.add_chunks(sample_chunks, document_name="test.pdf")
        
        # Search with random query
        query_embedding = np.random.rand(768)
        results = vector_store.similarity_search(query_embedding, top_k=2)
        
        assert len(results) == 2
        assert 'similarity_score' in results[0]
        assert 'text' in results[0]
        assert 'metadata' in results[0]
    
    
    def test_get_by_metadata(self, vector_store, sample_chunks):
        """Test metadata filtering"""
        # Add chunks
        vector_store.add_chunks(sample_chunks, document_name="test.pdf")
        
        # Filter by page number
        results = vector_store.get_by_metadata({'page_number': 1})
        
        assert len(results) >= 1
        assert results[0]['metadata']['page_number'] == 1
    
    
    def test_collection_stats(self, vector_store, sample_chunks):
        """Test collection statistics"""
        # Add chunks
        vector_store.add_chunks(sample_chunks, document_name="test.pdf")
        
        stats = vector_store.get_collection_stats()
        
        assert stats['total_chunks'] == 2
        assert stats['unique_documents'] == 1
        assert 'test.pdf' in stats['document_names']
    
    
    def test_delete_by_document(self, vector_store, sample_chunks):
        """Test document deletion"""
        # Add chunks
        vector_store.add_chunks(sample_chunks, document_name="test.pdf")
        
        # Delete
        deleted = vector_store.delete_by_document("test.pdf")
        
        assert deleted == 2
        assert vector_store.collection.count() == 0


# Run tests if executed directly
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
