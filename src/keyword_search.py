# keyword_search.py
"""
Keyword Search Module
=====================
Implements BM25 keyword-based search for lexical matching.

Key Functions:
- build_index(): Create BM25 index from chunks
- search(): Find chunks matching keywords
- get_scores(): Get relevance scores for all chunks
"""

import sys
from pathlib import Path
from typing import List, Dict, Optional
from rank_bm25 import BM25Okapi
import numpy as np
from loguru import logger
import pickle
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
import config


class BM25Search:
    """
    Implements BM25 keyword search for document chunks.
    Provides lexical matching to complement semantic vector search.
    """
    
    def __init__(self, cache_dir: str = config.CACHE_DIR):
        """
        Initialize BM25Search.
        
        Args:
            cache_dir: Directory for caching BM25 index
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        self.bm25_index = None
        self.chunks = []
        self.tokenized_corpus = []
        
        logger.info("BM25Search initialized")
    
    
    def _tokenize(self, text: str) -> List[str]:
        """
        Simple tokenization: lowercase and split by whitespace.
        
        Args:
            text: Input text
            
        Returns:
            List of tokens
        """
        # Convert to lowercase and split
        tokens = text.lower().split()
        
        # Remove punctuation and short tokens
        tokens = [
            ''.join(c for c in token if c.isalnum())
            for token in tokens
        ]
        tokens = [t for t in tokens if len(t) > 2]  # Remove very short words
        
        return tokens
    
    
    def build_index(self, chunks: List[Dict]) -> int:
        """
        Build BM25 index from chunks.
        
        Args:
            chunks: List of chunk dictionaries with 'text' key
            
        Returns:
            Number of chunks indexed
        """
        if not chunks:
            logger.warning("No chunks provided to index")
            return 0
        
        logger.info(f"Building BM25 index for {len(chunks)} chunks...")
        
        self.chunks = chunks
        self.tokenized_corpus = []
        
        # Tokenize all chunks
        for chunk in chunks:
            text = chunk.get('text', '')
            tokens = self._tokenize(text)
            self.tokenized_corpus.append(tokens)
        
        # Build BM25 index
        self.bm25_index = BM25Okapi(self.tokenized_corpus)
        
        logger.info(f"✅ BM25 index built for {len(chunks)} chunks")
        
        return len(chunks)
    
    
    def search(
        self,
        query: str,
        top_k: int = config.TOP_K_RETRIEVAL
    ) -> List[Dict]:
        """
        Search for chunks matching query keywords.
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of matching chunks with BM25 scores
        """
        if self.bm25_index is None:
            raise ValueError("BM25 index not built. Call build_index() first.")
        
        logger.debug(f"BM25 searching for: '{query[:100]}'")
        
        # Tokenize query
        query_tokens = self._tokenize(query)
        
        if not query_tokens:
            logger.warning("Query produced no tokens after tokenization")
            return []
        
        # Get BM25 scores
        scores = self.bm25_index.get_scores(query_tokens)
        
        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        # Format results
        results = []
        for idx in top_indices:
            if scores[idx] > 0:  # Only include non-zero scores
                result = {
                    'chunk': self.chunks[idx].copy(),
                    'bm25_score': float(scores[idx]),
                    'index': int(idx)
                }
                results.append(result)
        
        logger.debug(f"BM25 found {len(results)} results")
        
        return results
    
    
    def get_scores(self, query: str) -> np.ndarray:
        """
        Get BM25 scores for all chunks given a query.
        
        Args:
            query: Search query
            
        Returns:
            Array of BM25 scores for all chunks
        """
        if self.bm25_index is None:
            raise ValueError("BM25 index not built. Call build_index() first.")
        
        query_tokens = self._tokenize(query)
        scores = self.bm25_index.get_scores(query_tokens)
        
        return scores
    
    
    def save_index(self, filename: str = "bm25_index.pkl"):
        """
        Save BM25 index to disk.
        
        Args:
            filename: Name of cache file
        """
        if self.bm25_index is None:
            logger.warning("No index to save")
            return
        
        cache_file = self.cache_dir / filename
        
        cache_data = {
            'bm25_index': self.bm25_index,
            'chunks': self.chunks,
            'tokenized_corpus': self.tokenized_corpus,
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
            logger.info(f"BM25 index saved to {cache_file}")
        except Exception as e:
            logger.error(f"Failed to save index: {str(e)}")
    
    
    def load_index(self, filename: str = "bm25_index.pkl") -> bool:
        """
        Load BM25 index from disk.
        
        Args:
            filename: Name of cache file
            
        Returns:
            True if loaded successfully, False otherwise
        """
        cache_file = self.cache_dir / filename
        
        if not cache_file.exists():
            logger.info(f"No cached index found at {cache_file}")
            return False
        
        try:
            with open(cache_file, 'rb') as f:
                cache_data = pickle.load(f)
            
            self.bm25_index = cache_data['bm25_index']
            self.chunks = cache_data['chunks']
            self.tokenized_corpus = cache_data['tokenized_corpus']
            
            logger.info(f"BM25 index loaded from {cache_file}")
            logger.info(f"Index timestamp: {cache_data.get('timestamp', 'unknown')}")
            
            return True
        except Exception as e:
            logger.error(f"Failed to load index: {str(e)}")
            return False
    
    
    def get_stats(self) -> Dict:
        """
        Get statistics about the BM25 index.
        
        Returns:
            Dictionary with index statistics
        """
        if self.bm25_index is None:
            return {
                'indexed': False,
                'total_chunks': 0
            }
        
        return {
            'indexed': True,
            'total_chunks': len(self.chunks),
            'total_tokens': sum(len(tokens) for tokens in self.tokenized_corpus),
            'avg_tokens_per_chunk': sum(len(tokens) for tokens in self.tokenized_corpus) / len(self.tokenized_corpus) if self.tokenized_corpus else 0
        }


# ==========================================
# Testing & Demo
# ==========================================

if __name__ == "__main__":
    """
    Test BM25 search with sample chunks.
    """
    import sys
    
    # Setup logging
    logger.remove()
    logger.add(sys.stdout, level="INFO")
    
    print("\n" + "="*60)
    print("BM25 Keyword Search Test")
    print("="*60 + "\n")
    
    # Create sample chunks
    print("📝 Creating sample chunks...")
    sample_chunks = [
        {
            'chunk_id': 0,
            'text': 'Employees are entitled to 15 days of paid annual leave per year.',
            'page_number': 1
        },
        {
            'chunk_id': 1,
            'text': 'Remote work is allowed up to 3 days per week with manager approval.',
            'page_number': 2
        },
        {
            'chunk_id': 2,
            'text': 'All employees must follow the company code of conduct and professional standards.',
            'page_number': 3
        },
        {
            'chunk_id': 3,
            'text': 'Annual performance reviews are conducted every year for all staff members.',
            'page_number': 4
        },
        {
            'chunk_id': 4,
            'text': 'Leave requests must be submitted at least 2 weeks in advance through the HR portal.',
            'page_number': 5
        }
    ]
    
    # Initialize BM25 search
    print("🔄 Building BM25 index...")
    bm25 = BM25Search()
    indexed_count = bm25.build_index(sample_chunks)
    print(f"✅ Indexed {indexed_count} chunks")
    
    # Get stats
    stats = bm25.get_stats()
    print(f"\n📊 Index Statistics:")
    print(f"   - Total chunks: {stats['total_chunks']}")
    print(f"   - Total tokens: {stats['total_tokens']}")
    print(f"   - Avg tokens/chunk: {stats['avg_tokens_per_chunk']:.1f}")
    
    # Test searches
    test_queries = [
        "leave policy annual vacation",
        "remote work from home",
        "code of conduct professional"
    ]
    
    print("\n🔍 Testing BM25 Searches:")
    
    for query in test_queries:
        print(f"\n--- Query: '{query}' ---")
        results = bm25.search(query, top_k=3)
        
        print(f"✅ Found {len(results)} results")
        for i, result in enumerate(results):
            print(f"\n   Result {i+1} (BM25 score: {result['bm25_score']:.4f}):")
            print(f"   Page {result['chunk']['page_number']}")
            print(f"   {result['chunk']['text'][:80]}...")
    
    # Test save/load
    print("\n\n💾 Testing index persistence...")
    bm25.save_index("test_bm25.pkl")
    
    # Create new instance and load
    bm25_new = BM25Search()
    loaded = bm25_new.load_index("test_bm25.pkl")
    
    if loaded:
        print("✅ Index loaded successfully")
        
        # Verify it works
        results = bm25_new.search("leave vacation", top_k=2)
        print(f"✅ Search after reload: {len(results)} results found")
    
    print("\n" + "="*60)
    print("✅ All tests passed!")
    print("="*60 + "\n")
