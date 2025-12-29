# embeddings.py
"""
Embeddings Module
=================
Handles text embedding generation using Sentence Transformers.

Key Functions:
- load_embedding_model(): Load pre-trained embedding model
- generate_embeddings(): Create embeddings for text chunks
- batch_generate_embeddings(): Process multiple chunks efficiently
"""

import sys
from pathlib import Path
from typing import List, Dict, Union
import numpy as np
from sentence_transformers import SentenceTransformer
from loguru import logger
import hashlib
import pickle
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
import config


class EmbeddingGenerator:
    """
    Handles embedding generation using Sentence Transformers.
    Supports caching to avoid re-embedding same text.
    """
    
    def __init__(
        self,
        model_name: str = config.EMBEDDING_MODEL,
        device: str = config.EMBEDDING_DEVICE,
        cache_dir: str = config.CACHE_DIR
    ):
        """
        Initialize EmbeddingGenerator.
        
        Args:
            model_name: Name of Sentence Transformer model
            device: Device to run model on ('cpu' or 'cuda')
            cache_dir: Directory for caching embeddings
        """
        self.model_name = model_name
        self.device = device
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        logger.info(f"Initializing EmbeddingGenerator with model: {model_name}")
        logger.info(f"Device: {device}")
        
        # Load model
        self.model = self._load_model()
        
        # Embedding cache (in-memory for current session)
        self._embedding_cache = {}
        
        logger.info(f"EmbeddingGenerator initialized successfully")
    
    
    def _load_model(self) -> SentenceTransformer:
        """
        Load Sentence Transformer model.
        
        Returns:
            Loaded SentenceTransformer model
        """
        try:
            logger.info(f"Loading model: {self.model_name}")
            model = SentenceTransformer(self.model_name, device=self.device)
            
            # Get model info
            embedding_dim = model.get_sentence_embedding_dimension()
            logger.info(f"Model loaded successfully. Embedding dimension: {embedding_dim}")
            
            return model
            
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise RuntimeError(f"Model loading failed: {str(e)}")
    
    
    def generate_embedding(self, text: str, use_cache: bool = True) -> np.ndarray:
        """
        Generate embedding for a single text.
        
        Args:
            text: Input text
            use_cache: Whether to use cached embedding if available
            
        Returns:
            Embedding vector as numpy array
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")
        
        # Generate cache key
        cache_key = self._get_cache_key(text)
        
        # Check in-memory cache
        if use_cache and cache_key in self._embedding_cache:
            logger.debug(f"Cache hit for text: {text[:50]}...")
            return self._embedding_cache[cache_key]
        
        # Check disk cache
        if use_cache:
            cached_embedding = self._load_from_disk_cache(cache_key)
            if cached_embedding is not None:
                logger.debug(f"Disk cache hit for text: {text[:50]}...")
                self._embedding_cache[cache_key] = cached_embedding
                return cached_embedding
        
        # Generate new embedding
        logger.debug(f"Generating embedding for text: {text[:50]}...")
        embedding = self.model.encode(
            text,
            convert_to_numpy=True,
            show_progress_bar=False
        )
        
        # Cache the embedding
        if use_cache:
            self._embedding_cache[cache_key] = embedding
            self._save_to_disk_cache(cache_key, embedding)
        
        return embedding
    
    
    def batch_generate_embeddings(
        self,
        chunks: List[Dict],
        batch_size: int = config.EMBEDDING_BATCH_SIZE,
        use_cache: bool = True,
        show_progress: bool = True
    ) -> List[Dict]:
        """
        Generate embeddings for multiple chunks efficiently.
        
        Args:
            chunks: List of chunk dictionaries with 'text' key
            batch_size: Number of chunks to process at once
            use_cache: Whether to use cached embeddings
            show_progress: Whether to show progress bar
            
        Returns:
            List of chunks with added 'embedding' key
        """
        logger.info(f"Generating embeddings for {len(chunks)} chunks (batch_size={batch_size})")
        
        # Separate cached and uncached chunks
        chunks_to_process = []
        chunk_indices = []
        
        for i, chunk in enumerate(chunks):
            text = chunk.get('text', '')
            cache_key = self._get_cache_key(text)
            
            # Check cache
            if use_cache and cache_key in self._embedding_cache:
                chunk['embedding'] = self._embedding_cache[cache_key]
                chunk['embedding_cached'] = True
            elif use_cache:
                cached_embedding = self._load_from_disk_cache(cache_key)
                if cached_embedding is not None:
                    chunk['embedding'] = cached_embedding
                    chunk['embedding_cached'] = True
                    self._embedding_cache[cache_key] = cached_embedding
                else:
                    chunks_to_process.append(text)
                    chunk_indices.append(i)
            else:
                chunks_to_process.append(text)
                chunk_indices.append(i)
        
        # Log cache statistics
        cached_count = len(chunks) - len(chunks_to_process)
        logger.info(f"Cache hits: {cached_count}/{len(chunks)} ({cached_count/len(chunks)*100:.1f}%)")
        
        # Generate embeddings for uncached chunks
        if chunks_to_process:
            logger.info(f"Generating {len(chunks_to_process)} new embeddings...")
            
            embeddings = self.model.encode(
                chunks_to_process,
                batch_size=batch_size,
                convert_to_numpy=True,
                show_progress_bar=show_progress
            )
            
            # Add embeddings to chunks and cache them
            for idx, embedding in zip(chunk_indices, embeddings):
                chunks[idx]['embedding'] = embedding
                chunks[idx]['embedding_cached'] = False
                
                # Cache the embedding
                if use_cache:
                    text = chunks[idx]['text']
                    cache_key = self._get_cache_key(text)
                    self._embedding_cache[cache_key] = embedding
                    self._save_to_disk_cache(cache_key, embedding)
        
        # Add embedding metadata
        for chunk in chunks:
            chunk['embedding_model'] = self.model_name
            chunk['embedding_dimension'] = len(chunk['embedding'])
            chunk['embedding_timestamp'] = datetime.now().isoformat()
        
        logger.info(f"✅ All embeddings generated successfully")
        
        return chunks
    
    
    def _get_cache_key(self, text: str) -> str:
        """
        Generate cache key for text.
        
        Args:
            text: Input text
            
        Returns:
            MD5 hash of text
        """
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    
    def _save_to_disk_cache(self, cache_key: str, embedding: np.ndarray):
        """
        Save embedding to disk cache.
        
        Args:
            cache_key: Cache key
            embedding: Embedding vector
        """
        try:
            cache_file = self.cache_dir / f"{cache_key}.pkl"
            with open(cache_file, 'wb') as f:
                pickle.dump(embedding, f)
        except Exception as e:
            logger.warning(f"Failed to save to disk cache: {str(e)}")
    
    
    def _load_from_disk_cache(self, cache_key: str) -> Union[np.ndarray, None]:
        """
        Load embedding from disk cache.
        
        Args:
            cache_key: Cache key
            
        Returns:
            Cached embedding or None if not found
        """
        try:
            cache_file = self.cache_dir / f"{cache_key}.pkl"
            if cache_file.exists():
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
        except Exception as e:
            logger.warning(f"Failed to load from disk cache: {str(e)}")
        
        return None
    
    
    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of embeddings produced by the model.
        
        Returns:
            Embedding dimension
        """
        return self.model.get_sentence_embedding_dimension()
    
    
    def clear_cache(self, disk_only: bool = False):
        """
        Clear embedding cache.
        
        Args:
            disk_only: If True, only clear disk cache, not memory cache
        """
        if not disk_only:
            self._embedding_cache.clear()
            logger.info("Memory cache cleared")
        
        # Clear disk cache
        try:
            cache_files = list(self.cache_dir.glob("*.pkl"))
            for cache_file in cache_files:
                cache_file.unlink()
            logger.info(f"Disk cache cleared: {len(cache_files)} files removed")
        except Exception as e:
            logger.warning(f"Failed to clear disk cache: {str(e)}")
    
    
    def get_cache_stats(self) -> Dict:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        memory_cache_size = len(self._embedding_cache)
        
        try:
            cache_files = list(self.cache_dir.glob("*.pkl"))
            disk_cache_size = len(cache_files)
            disk_cache_size_mb = sum(f.stat().st_size for f in cache_files) / (1024 * 1024)
        except:
            disk_cache_size = 0
            disk_cache_size_mb = 0
        
        return {
            "memory_cache_entries": memory_cache_size,
            "disk_cache_entries": disk_cache_size,
            "disk_cache_size_mb": round(disk_cache_size_mb, 2),
            "model": self.model_name,
            "embedding_dimension": self.get_embedding_dimension()
        }


# ==========================================
# Utility Functions
# ==========================================

def cosine_similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    """
    Calculate cosine similarity between two embeddings.
    
    Args:
        embedding1: First embedding vector
        embedding2: Second embedding vector
        
    Returns:
        Cosine similarity score (0 to 1)
    """
    dot_product = np.dot(embedding1, embedding2)
    norm1 = np.linalg.norm(embedding1)
    norm2 = np.linalg.norm(embedding2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return dot_product / (norm1 * norm2)


# ==========================================
# Testing & Demo
# ==========================================

if __name__ == "__main__":
    """
    Test embedding generator with sample texts.
    """
    import sys
    
    # Setup logging
    logger.remove()
    logger.add(sys.stdout, level="INFO")
    
    print("\n" + "="*60)
    print("Embedding Generator Test")
    print("="*60 + "\n")
    
    # Initialize generator
    print("🔄 Loading embedding model...")
    generator = EmbeddingGenerator()
    
    print(f"✅ Model loaded: {generator.model_name}")
    print(f"📊 Embedding dimension: {generator.get_embedding_dimension()}")
    
    # Test single embedding
    print("\n--- Test 1: Single Embedding ---")
    test_text = "This is a test sentence for embedding generation."
    embedding = generator.generate_embedding(test_text)
    print(f"Text: {test_text}")
    print(f"Embedding shape: {embedding.shape}")
    print(f"Embedding preview: {embedding[:5]}...")
    
    # Test batch embeddings
    print("\n--- Test 2: Batch Embeddings ---")
    test_chunks = [
        {"chunk_id": 0, "text": "Employees are entitled to 15 days of leave."},
        {"chunk_id": 1, "text": "Remote work is allowed 3 days per week."},
        {"chunk_id": 2, "text": "All employees must follow the code of conduct."},
    ]
    
    chunks_with_embeddings = generator.batch_generate_embeddings(
        test_chunks,
        show_progress=False
    )
    
    print(f"✅ Generated embeddings for {len(chunks_with_embeddings)} chunks")
    for chunk in chunks_with_embeddings:
        print(f"  Chunk {chunk['chunk_id']}: {chunk['embedding'].shape}")
    
    # Test similarity
    print("\n--- Test 3: Similarity Calculation ---")
    emb1 = chunks_with_embeddings[0]['embedding']
    emb2 = chunks_with_embeddings[1]['embedding']
    emb3 = chunks_with_embeddings[2]['embedding']
    
    sim_1_2 = cosine_similarity(emb1, emb2)
    sim_1_3 = cosine_similarity(emb1, emb3)
    
    print(f"Similarity (Chunk 0 vs Chunk 1): {sim_1_2:.4f}")
    print(f"Similarity (Chunk 0 vs Chunk 2): {sim_1_3:.4f}")
    
    # Test caching
    print("\n--- Test 4: Caching ---")
    print("Generating embedding (first time)...")
    _ = generator.generate_embedding(test_text)
    
    print("Generating same embedding (should be cached)...")
    _ = generator.generate_embedding(test_text)
    
    cache_stats = generator.get_cache_stats()
    print(f"\n📊 Cache Statistics:")
    print(f"  Memory cache: {cache_stats['memory_cache_entries']} entries")
    print(f"  Disk cache: {cache_stats['disk_cache_entries']} entries")
    print(f"  Disk cache size: {cache_stats['disk_cache_size_mb']} MB")
    
    print("\n" + "="*60)
    print("✅ All tests passed!")
    print("="*60 + "\n")
