# retrieval.py
"""
Hybrid Retrieval Module
=======================
Combines BM25 keyword search and vector semantic search for optimal retrieval.

Key Functions:
- hybrid_search(): Combine BM25 and vector search results
- ensemble_ranking(): Merge and rank results from both methods
- get_context(): Format retrieved chunks for LLM context
"""

import sys
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import numpy as np
from loguru import logger

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.keyword_search import BM25Search
from src.vector_store import VectorStore
from src.embeddings import EmbeddingGenerator
import config


class HybridRetriever:
    """
    Combines BM25 keyword search and vector semantic search
    for optimal document retrieval.
    """
    
    def __init__(
        self,
        vector_store: VectorStore,
        bm25_search: BM25Search,
        embedding_generator: EmbeddingGenerator,
        bm25_weight: float = config.BM25_WEIGHT,
        vector_weight: float = config.VECTOR_WEIGHT
    ):
        """
        Initialize HybridRetriever.
        
        Args:
            vector_store: Initialized VectorStore instance
            bm25_search: Initialized BM25Search instance with built index
            embedding_generator: EmbeddingGenerator instance
            bm25_weight: Weight for BM25 scores (0-1)
            vector_weight: Weight for vector scores (0-1)
        """
        self.vector_store = vector_store
        self.bm25_search = bm25_search
        self.embedding_generator = embedding_generator
        self.bm25_weight = bm25_weight
        self.vector_weight = vector_weight
        
        # Validate weights
        if not (0 <= bm25_weight <= 1 and 0 <= vector_weight <= 1):
            raise ValueError("Weights must be between 0 and 1")
        
        if abs((bm25_weight + vector_weight) - 1.0) > 0.001:
            logger.warning(f"Weights don't sum to 1.0: BM25={bm25_weight}, Vector={vector_weight}")
        
        logger.info(f"HybridRetriever initialized (BM25={bm25_weight}, Vector={vector_weight})")
    
    
    def hybrid_search(
        self,
        query: str,
        top_k: int = config.TOP_K_RETRIEVAL,
        rerank: bool = config.RERANK_ENABLED
    ) -> List[Dict]:
        """
        Perform hybrid search combining BM25 and vector results.
        
        Args:
            query: Search query
            top_k: Number of final results to return
            rerank: Whether to apply re-ranking
            
        Returns:
            List of chunks with hybrid scores
        """
        logger.info(f"Hybrid search for: '{query[:100]}'")
        
        # Get more results from each method for better ensemble
        retrieval_k = top_k * 2
        
        # 1. BM25 keyword search
        logger.debug("Performing BM25 search...")
        bm25_results = self.bm25_search.search(query, top_k=retrieval_k)
        
        # 2. Vector semantic search
        logger.debug("Performing vector search...")
        vector_results = self.vector_store.search_by_text(
            query,
            self.embedding_generator,
            top_k=retrieval_k
        )
        
        # 3. Ensemble ranking
        logger.debug("Combining results...")
        combined_results = self._ensemble_ranking(
            bm25_results,
            vector_results,
            top_k=top_k
        )
        
        # 4. Optional re-ranking
        if rerank and len(combined_results) > 0:
            combined_results = self._rerank_results(combined_results, query)
        
        logger.info(f"✅ Hybrid search complete: {len(combined_results)} results")
        
        return combined_results
    
    
    def _ensemble_ranking(
        self,
        bm25_results: List[Dict],
        vector_results: List[Dict],
        top_k: int
    ) -> List[Dict]:
        """
        Combine BM25 and vector results using weighted scoring.
        
        Args:
            bm25_results: Results from BM25 search
            vector_results: Results from vector search
            top_k: Number of results to return
            
        Returns:
            Merged and ranked results
        """
        # Normalize BM25 scores
        bm25_scores_normalized = self._normalize_scores(
            [r['bm25_score'] for r in bm25_results]
        )
        
        # Normalize vector scores (similarity_score already 0-1)
        vector_scores_normalized = [r['similarity_score'] for r in vector_results]
        
        # Create lookup dictionaries
        bm25_lookup = {}
        for i, result in enumerate(bm25_results):
            chunk_id = result['chunk'].get('chunk_id')
            bm25_lookup[chunk_id] = {
                'chunk': result['chunk'],
                'bm25_score': bm25_scores_normalized[i],
                'bm25_raw_score': result['bm25_score']
            }
        
        vector_lookup = {}
        for i, result in enumerate(vector_results):
            chunk_id = result['metadata'].get('chunk_id')
            vector_lookup[chunk_id] = {
                'text': result['text'],
                'metadata': result['metadata'],
                'vector_score': vector_scores_normalized[i],
                'vector_raw_score': result['similarity_score']
            }
        
        # Combine results
        all_chunk_ids = set(list(bm25_lookup.keys()) + list(vector_lookup.keys()))
        
        combined = []
        for chunk_id in all_chunk_ids:
            bm25_data = bm25_lookup.get(chunk_id, {})
            vector_data = vector_lookup.get(chunk_id, {})
            
            # Calculate hybrid score
            bm25_score = bm25_data.get('bm25_score', 0.0)
            vector_score = vector_data.get('vector_score', 0.0)
            
            hybrid_score = (
                self.bm25_weight * bm25_score +
                self.vector_weight * vector_score
            )
            
            # Get chunk data (prefer BM25 chunk if available)
            chunk_data = bm25_data.get('chunk') or {
                'text': vector_data.get('text', ''),
                'chunk_id': chunk_id,
                'page_number': vector_data.get('metadata', {}).get('page_number', 0),
                'document_name': vector_data.get('metadata', {}).get('document_name', 'unknown')
            }
            
            result = {
                'chunk': chunk_data,
                'hybrid_score': hybrid_score,
                'bm25_score': bm25_data.get('bm25_raw_score', 0.0),
                'vector_score': vector_data.get('vector_raw_score', 0.0),
                'bm25_normalized': bm25_score,
                'vector_normalized': vector_score,
                'retrieval_method': self._get_retrieval_method(
                    chunk_id in bm25_lookup,
                    chunk_id in vector_lookup
                )
            }
            
            combined.append(result)
        
        # Sort by hybrid score
        combined.sort(key=lambda x: x['hybrid_score'], reverse=True)
        
        # Return top-k
        return combined[:top_k]
    
    
    def _normalize_scores(self, scores: List[float]) -> List[float]:
        """
        Normalize scores to 0-1 range using min-max scaling.
        
        Args:
            scores: List of scores
            
        Returns:
            Normalized scores
        """
        if not scores:
            return []
        
        min_score = min(scores)
        max_score = max(scores)
        
        # Avoid division by zero
        if max_score - min_score < 1e-10:
            return [1.0] * len(scores)
        
        normalized = [
            (score - min_score) / (max_score - min_score)
            for score in scores
        ]
        
        return normalized
    
    
    def _get_retrieval_method(self, in_bm25: bool, in_vector: bool) -> str:
        """
        Determine which retrieval method(s) found the chunk.
        
        Args:
            in_bm25: Whether chunk was in BM25 results
            in_vector: Whether chunk was in vector results
            
        Returns:
            Retrieval method label
        """
        if in_bm25 and in_vector:
            return 'both'
        elif in_bm25:
            return 'bm25_only'
        elif in_vector:
            return 'vector_only'
        else:
            return 'unknown'
    
    
    def _rerank_results(
        self,
        results: List[Dict],
        query: str
    ) -> List[Dict]:
        """
        Apply simple re-ranking based on query term presence.
        
        Args:
            results: Initial ranked results
            query: Original query
            
        Returns:
            Re-ranked results
        """
        query_terms = set(query.lower().split())
        
        for result in results:
            text = result['chunk'].get('text', '').lower()
            
            # Count query term matches
            term_matches = sum(1 for term in query_terms if term in text)
            
            # Boost score based on term matches
            boost = term_matches * 0.05  # 5% boost per matching term
            result['hybrid_score'] = min(result['hybrid_score'] + boost, 1.0)
            result['reranked'] = True
        
        # Re-sort after boosting
        results.sort(key=lambda x: x['hybrid_score'], reverse=True)
        
        return results
    
    
    def get_context(
        self,
        query: str,
        top_k: int = config.TOP_K_RETRIEVAL,
        max_tokens: Optional[int] = None
    ) -> Tuple[str, List[Dict]]:
        """
        Get formatted context string for LLM from retrieved chunks.
        
        Args:
            query: Search query
            top_k: Number of chunks to retrieve
            max_tokens: Optional maximum tokens for context
            
        Returns:
            Tuple of (formatted_context_string, retrieved_chunks)
        """
        # Retrieve chunks
        results = self.hybrid_search(query, top_k=top_k)
        
        if not results:
            return "No relevant information found.", []
        
        # Format context
        context_parts = []
        total_tokens = 0
        included_chunks = []
        
        for i, result in enumerate(results):
            chunk = result['chunk']
            text = chunk.get('text', '')
            page = chunk.get('page_number', 'unknown')
            chunk_id = chunk.get('chunk_id', i)
            
            # Format chunk with citation
            chunk_text = f"[Chunk {chunk_id}, Page {page}]\n{text}\n"
            
            # Check token limit if specified
            if max_tokens:
                chunk_tokens = len(chunk_text.split()) * 1.3  # Rough token estimate
                if total_tokens + chunk_tokens > max_tokens:
                    break
                total_tokens += chunk_tokens
            
            context_parts.append(chunk_text)
            included_chunks.append(result)
        
        context = "\n---\n\n".join(context_parts)
        
        logger.info(f"Context generated: {len(included_chunks)} chunks, ~{total_tokens:.0f} tokens")
        
        return context, included_chunks
    
    
    def get_retrieval_stats(self) -> Dict:
        """
        Get statistics about retrieval components.
        
        Returns:
            Dictionary with retrieval statistics
        """
        bm25_stats = self.bm25_search.get_stats()
        vector_stats = self.vector_store.get_collection_stats()
        
        return {
            'bm25': bm25_stats,
            'vector': vector_stats,
            'weights': {
                'bm25_weight': self.bm25_weight,
                'vector_weight': self.vector_weight
            }
        }


# ==========================================
# Utility Functions
# ==========================================

def format_retrieval_results(results: List[Dict]) -> str:
    """
    Format retrieval results for display.
    
    Args:
        results: List of retrieval results
        
    Returns:
        Formatted string
    """
    if not results:
        return "No results found."
    
    output = []
    output.append(f"\n{'='*60}")
    output.append(f"Retrieved {len(results)} chunks")
    output.append(f"{'='*60}\n")
    
    for i, result in enumerate(results):
        chunk = result['chunk']
        
        output.append(f"--- Result {i+1} ---")
        output.append(f"Hybrid Score: {result['hybrid_score']:.4f}")
        output.append(f"  BM25: {result['bm25_score']:.4f} (normalized: {result['bm25_normalized']:.4f})")
        output.append(f"  Vector: {result['vector_score']:.4f} (normalized: {result['vector_normalized']:.4f})")
        output.append(f"Method: {result['retrieval_method']}")
        output.append(f"Page: {chunk.get('page_number', 'unknown')}")
        
        text_preview = chunk.get('text', '')[:150] + "..." if len(chunk.get('text', '')) > 150 else chunk.get('text', '')
        output.append(f"Text: {text_preview}")
        output.append("")
    
    return "\n".join(output)


# ==========================================
# Testing & Demo
# ==========================================

if __name__ == "__main__":
    """
    Test hybrid retrieval with sample data.
    """
    import sys
    from src.document_processor import DocumentProcessor
    
    # Setup logging
    logger.remove()
    logger.add(sys.stdout, level="INFO")
    
    print("\n" + "="*60)
    print("Hybrid Retrieval Test")
    print("="*60 + "\n")
    
    pdf_path = "data/sample_documents/test.pdf"
    
    if not Path(pdf_path).exists():
        print(f"❌ Error: PDF not found at {pdf_path}")
        print("Please run previous pipeline tests first.")
        sys.exit(1)
    
    try:
        # Step 1: Process document
        print("📄 Step 1: Processing document...")
        processor = DocumentProcessor()
        chunks, metadata = processor.process_document(pdf_path)
        print(f"✅ Processed {len(chunks)} chunks")
        
        # Step 2: Generate embeddings
        print("\n🔄 Step 2: Generating embeddings...")
        generator = EmbeddingGenerator()
        chunks_with_embeddings = generator.batch_generate_embeddings(
            chunks,
            show_progress=False
        )
        print(f"✅ Generated embeddings")
        
        # Step 3: Setup vector store
        print("\n💾 Step 3: Setting up vector store...")
        vector_store = VectorStore(reset=True)
        vector_store.add_chunks(chunks_with_embeddings, document_name="test.pdf")
        print(f"✅ Vector store ready")
        
        # Step 4: Build BM25 index
        print("\n🔍 Step 4: Building BM25 index...")
        bm25 = BM25Search()
        bm25.build_index(chunks)
        print(f"✅ BM25 index ready")
        
        # Step 5: Initialize hybrid retriever
        print("\n🚀 Step 5: Initializing hybrid retriever...")
        retriever = HybridRetriever(
            vector_store=vector_store,
            bm25_search=bm25,
            embedding_generator=generator
        )
        print(f"✅ Hybrid retriever ready")
        
        # Step 6: Test hybrid search
        print("\n🔍 Step 6: Testing Hybrid Search")
        
        test_queries = [
            "employee conduct policy",
            "nagarro constitution rules"
        ]
        
        for query in test_queries:
            print(f"\n{'='*60}")
            print(f"Query: '{query}'")
            print(f"{'='*60}")
            
            results = retriever.hybrid_search(query, top_k=3)
            
            print(f"\n✅ Found {len(results)} results\n")
            
            for i, result in enumerate(results):
                print(f"Result {i+1}:")
                print(f"  Hybrid: {result['hybrid_score']:.4f} | BM25: {result['bm25_score']:.4f} | Vector: {result['vector_score']:.4f}")
                print(f"  Method: {result['retrieval_method']}")
                print(f"  Page: {result['chunk']['page_number']}")
                print(f"  Text: {result['chunk']['text'][:100]}...\n")
        
        # Step 7: Test context generation
        print(f"\n{'='*60}")
        print("Step 7: Testing Context Generation")
        print(f"{'='*60}\n")
        
        context, chunks = retriever.get_context("employee policy", top_k=3)
        print(f"✅ Generated context with {len(chunks)} chunks")
        print(f"\nContext preview (first 300 chars):")
        print(context[:300] + "...")
        
        # Step 8: Get stats
        print(f"\n{'='*60}")
        print("Step 8: Retrieval Statistics")
        print(f"{'='*60}\n")
        
        stats = retriever.get_retrieval_stats()
        print(f"BM25 chunks indexed: {stats['bm25']['total_chunks']}")
        print(f"Vector chunks stored: {stats['vector']['total_chunks']}")
        print(f"Weights: BM25={stats['weights']['bm25_weight']}, Vector={stats['weights']['vector_weight']}")
        
        print("\n" + "="*60)
        print("✅ All hybrid retrieval tests passed!")
        print("="*60 + "\n")
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
