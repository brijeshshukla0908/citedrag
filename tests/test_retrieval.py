# test_retrieval.py
"""
Unit Tests for Hybrid Retrieval
================================
Tests for hybrid search combining BM25 and vector search.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.retrieval import HybridRetriever
from src.keyword_search import BM25Search
from src.vector_store import VectorStore
from src.embeddings import EmbeddingGenerator


class TestHybridRetriever:
    """Test suite for HybridRetriever class"""
    
    @pytest.fixture
    def sample_chunks(self):
        """Sample chunks for testing"""
        return [
            {
                'chunk_id': 0,
                'text': 'Employees are entitled to annual leave benefits.',
                'page_number': 1,
                'token_count': 8
            },
            {
                'chunk_id': 1,
                'text': 'Remote work policy allows flexible arrangements.',
                'page_number': 2,
                'token_count': 7
            },
            {
                'chunk_id': 2,
                'text': 'Code of conduct for all employees.',
                'page_number': 3,
                'token_count': 6
            }
        ]
    
    
    @pytest.fixture
    def retriever(self, sample_chunks):
        """Create retriever with sample data"""
        # Generate embeddings
        generator = EmbeddingGenerator()
        chunks_with_embeddings = generator.batch_generate_embeddings(
            sample_chunks,
            show_progress=False
        )
        
        # Setup vector store
        vector_store = VectorStore(
            collection_name="test_hybrid_collection",
            reset=True
        )
        vector_store.add_chunks(chunks_with_embeddings, document_name="test.pdf")
        
        # Setup BM25
        bm25 = BM25Search()
        bm25.build_index(sample_chunks)
        
        # Create retriever
        return HybridRetriever(
            vector_store=vector_store,
            bm25_search=bm25,
            embedding_generator=generator
        )
    
    
    def test_initialization(self, retriever):
        """Test retriever initialization"""
        assert retriever.vector_store is not None
        assert retriever.bm25_search is not None
        assert retriever.embedding_generator is not None
        assert retriever.bm25_weight == 0.5
        assert retriever.vector_weight == 0.5
    
    
    def test_hybrid_search(self, retriever):
        """Test hybrid search"""
        results = retriever.hybrid_search("employee leave policy", top_k=2)
        
        assert len(results) >= 1
        assert 'hybrid_score' in results[0]
        assert 'bm25_score' in results[0]
        assert 'vector_score' in results[0]
        assert 'chunk' in results[0]
    
    
    def test_normalize_scores(self, retriever):
        """Test score normalization"""
        scores = [10.0, 20.0, 30.0]
        normalized = retriever._normalize_scores(scores)
        
        assert len(normalized) == 3
        assert normalized[0] == 0.0  # Min
        assert normalized[-1] == 1.0  # Max
        assert all(0 <= s <= 1 for s in normalized)
    
    
    def test_get_retrieval_method(self, retriever):
        """Test retrieval method labeling"""
        assert retriever._get_retrieval_method(True, True) == 'both'
        assert retriever._get_retrieval_method(True, False) == 'bm25_only'
        assert retriever._get_retrieval_method(False, True) == 'vector_only'
    
    
    def test_get_context(self, retriever):
        """Test context generation"""
        context, chunks = retriever.get_context("leave policy", top_k=2)
        
        assert isinstance(context, str)
        assert len(context) > 0
        assert len(chunks) >= 1
        assert 'Chunk' in context
        assert 'Page' in context
    
    
    def test_get_retrieval_stats(self, retriever):
        """Test retrieval statistics"""
        stats = retriever.get_retrieval_stats()
        
        assert 'bm25' in stats
        assert 'vector' in stats
        assert 'weights' in stats
        assert stats['bm25']['total_chunks'] == 3
        assert stats['vector']['total_chunks'] == 3


# Run tests if executed directly
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
