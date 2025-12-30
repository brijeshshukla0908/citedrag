# cache_manager.py
"""
Cache Manager Module
====================
Handles response caching to reduce API calls and improve performance.

Key Functions:
- cache_response(): Store query responses
- get_cached_response(): Retrieve cached responses
- clear_cache(): Manage cache size
"""

import sys
from pathlib import Path
from typing import Optional, Dict, Any
import hashlib
import json
import pickle
from datetime import datetime, timedelta
from loguru import logger

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import config


class CacheManager:
    """
    Manages caching for query responses to reduce API calls
    and improve response times.
    """
    
    def __init__(
        self,
        cache_dir: str = config.CACHE_DIR,
        ttl_hours: int = config.CACHE_TTL_HOURS,
        max_size: int = config.CACHE_MAX_SIZE
    ):
        """
        Initialize CacheManager.
        
        Args:
            cache_dir: Directory for cache storage
            ttl_hours: Time-to-live for cache entries in hours
            max_size: Maximum number of cached entries
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        
        self.ttl_hours = ttl_hours
        self.max_size = max_size
        
        # In-memory cache for fast access
        self._memory_cache = {}
        
        # Cache metadata
        self.metadata_file = self.cache_dir / "cache_metadata.json"
        self.metadata = self._load_metadata()
        
        logger.info(f"CacheManager initialized (TTL: {ttl_hours}h, Max: {max_size})")
    
    
    def _generate_cache_key(self, query: str, document_name: str = "default") -> str:
        """
        Generate unique cache key for query + document.
        
        Args:
            query: User query
            document_name: Document identifier
            
        Returns:
            MD5 hash key
        """
        content = f"{query}:{document_name}".lower().strip()
        return hashlib.md5(content.encode('utf-8')).hexdigest()
    
    
    def get_cached_response(
        self,
        query: str,
        document_name: str = "default"
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve cached response if available and not expired.
        
        Args:
            query: User query
            document_name: Document identifier
            
        Returns:
            Cached response dict or None
        """
        cache_key = self._generate_cache_key(query, document_name)
        
        # Check memory cache first
        if cache_key in self._memory_cache:
            cached_data = self._memory_cache[cache_key]
            
            # Check expiration
            if self._is_expired(cached_data['timestamp']):
                logger.debug(f"Cache expired for key: {cache_key[:8]}...")
                del self._memory_cache[cache_key]
                self._remove_from_disk(cache_key)
                return None
            
            logger.info(f"✅ Cache HIT (memory): {query[:50]}...")
            self._update_access_time(cache_key)
            return cached_data['response']
        
        # Check disk cache
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                
                # Check expiration
                if self._is_expired(cached_data['timestamp']):
                    logger.debug(f"Cache expired (disk) for key: {cache_key[:8]}...")
                    cache_file.unlink()
                    return None
                
                # Load into memory cache
                self._memory_cache[cache_key] = cached_data
                
                logger.info(f"✅ Cache HIT (disk): {query[:50]}...")
                self._update_access_time(cache_key)
                return cached_data['response']
                
            except Exception as e:
                logger.warning(f"Failed to load cache from disk: {str(e)}")
                cache_file.unlink()
                return None
        
        logger.debug(f"Cache MISS: {query[:50]}...")
        return None
    
    
    def cache_response(
        self,
        query: str,
        response: Dict[str, Any],
        document_name: str = "default"
    ):
        """
        Cache a query response.
        
        Args:
            query: User query
            response: Response dictionary to cache
            document_name: Document identifier
        """
        cache_key = self._generate_cache_key(query, document_name)
        
        cached_data = {
            'query': query,
            'response': response,
            'timestamp': datetime.now().isoformat(),
            'document_name': document_name
        }
        
        # Store in memory
        self._memory_cache[cache_key] = cached_data
        
        # Store on disk
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(cached_data, f)
            
            # Update metadata
            self.metadata[cache_key] = {
                'query': query[:100],  # Truncated for metadata
                'timestamp': cached_data['timestamp'],
                'last_accessed': cached_data['timestamp'],
                'document_name': document_name
            }
            self._save_metadata()
            
            logger.info(f"✅ Response cached: {query[:50]}...")
            
            # Enforce max cache size
            self._enforce_cache_size()
            
        except Exception as e:
            logger.error(f"Failed to cache response: {str(e)}")
    
    
    def _is_expired(self, timestamp_str: str) -> bool:
        """
        Check if cached entry is expired.
        
        Args:
            timestamp_str: ISO format timestamp
            
        Returns:
            True if expired
        """
        try:
            timestamp = datetime.fromisoformat(timestamp_str)
            expiry_time = timestamp + timedelta(hours=self.ttl_hours)
            return datetime.now() > expiry_time
        except:
            return True
    
    
    def _update_access_time(self, cache_key: str):
        """
        Update last accessed time for cache entry.
        
        Args:
            cache_key: Cache key
        """
        if cache_key in self.metadata:
            self.metadata[cache_key]['last_accessed'] = datetime.now().isoformat()
            self._save_metadata()
    
    
    def _enforce_cache_size(self):
        """
        Ensure cache doesn't exceed max size by removing oldest entries.
        """
        if len(self.metadata) <= self.max_size:
            return
        
        logger.info(f"Cache size ({len(self.metadata)}) exceeds max ({self.max_size}), pruning...")
        
        # Sort by last accessed time
        sorted_keys = sorted(
            self.metadata.keys(),
            key=lambda k: self.metadata[k].get('last_accessed', ''),
            reverse=False
        )
        
        # Remove oldest entries
        num_to_remove = len(self.metadata) - self.max_size
        for cache_key in sorted_keys[:num_to_remove]:
            self._remove_from_disk(cache_key)
            if cache_key in self._memory_cache:
                del self._memory_cache[cache_key]
            del self.metadata[cache_key]
        
        self._save_metadata()
        logger.info(f"✅ Pruned {num_to_remove} old cache entries")
    
    
    def _remove_from_disk(self, cache_key: str):
        """
        Remove cache file from disk.
        
        Args:
            cache_key: Cache key
        """
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        if cache_file.exists():
            cache_file.unlink()
    
    
    def clear_cache(self, document_name: Optional[str] = None):
        """
        Clear cache entries.
        
        Args:
            document_name: If provided, only clear entries for this document
        """
        if document_name:
            # Clear specific document cache
            keys_to_remove = [
                k for k, v in self.metadata.items()
                if v.get('document_name') == document_name
            ]
            
            for cache_key in keys_to_remove:
                self._remove_from_disk(cache_key)
                if cache_key in self._memory_cache:
                    del self._memory_cache[cache_key]
                del self.metadata[cache_key]
            
            logger.info(f"✅ Cleared {len(keys_to_remove)} cache entries for: {document_name}")
        else:
            # Clear all cache
            self._memory_cache.clear()
            
            for cache_file in self.cache_dir.glob("*.pkl"):
                cache_file.unlink()
            
            self.metadata.clear()
            logger.info("✅ All cache cleared")
        
        self._save_metadata()
    
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache stats
        """
        total_entries = len(self.metadata)
        memory_entries = len(self._memory_cache)
        
        # Calculate cache size on disk
        total_size_mb = sum(
            f.stat().st_size for f in self.cache_dir.glob("*.pkl")
        ) / (1024 * 1024)
        
        # Count expired entries
        expired_count = sum(
            1 for v in self.metadata.values()
            if self._is_expired(v['timestamp'])
        )
        
        return {
            'total_entries': total_entries,
            'memory_entries': memory_entries,
            'disk_size_mb': round(total_size_mb, 2),
            'expired_entries': expired_count,
            'max_size': self.max_size,
            'ttl_hours': self.ttl_hours
        }
    
    
    def _load_metadata(self) -> Dict:
        """
        Load cache metadata from disk.
        
        Returns:
            Metadata dictionary
        """
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            except:
                logger.warning("Failed to load cache metadata, starting fresh")
        
        return {}
    
    
    def _save_metadata(self):
        """
        Save cache metadata to disk.
        """
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save cache metadata: {str(e)}")


# ==========================================
# Testing & Demo
# ==========================================

if __name__ == "__main__":
    """
    Test cache manager functionality.
    """
    import time
    
    # Setup logging
    logger.remove()
    logger.add(sys.stdout, level="INFO")
    
    print("\n" + "="*60)
    print("Cache Manager Test")
    print("="*60 + "\n")
    
    # Initialize cache manager
    print("🔄 Initializing cache manager...")
    cache = CacheManager(ttl_hours=1, max_size=100)
    
    # Test 1: Cache miss
    print("\n--- Test 1: Cache Miss ---")
    query1 = "What is the leave policy?"
    result = cache.get_cached_response(query1)
    print(f"Query: {query1}")
    print(f"Result: {result}")
    assert result is None, "Should be cache miss"
    
    # Test 2: Cache response
    print("\n--- Test 2: Cache Response ---")
    mock_response = {
        'answer': 'Employees get 15 days of leave.',
        'citations': [{'chunk_id': 0}],
        'token_usage': {'total_tokens': 100}
    }
    cache.cache_response(query1, mock_response)
    print(f"✅ Cached response for: {query1}")
    
    # Test 3: Cache hit
    print("\n--- Test 3: Cache Hit ---")
    result = cache.get_cached_response(query1)
    print(f"Query: {query1}")
    print(f"Result: {result is not None}")
    assert result is not None, "Should be cache hit"
    assert result['answer'] == mock_response['answer']
    print("✅ Cache hit successful")
    
    # Test 4: Different document
    print("\n--- Test 4: Different Document ---")
    result = cache.get_cached_response(query1, document_name="other_doc.pdf")
    print(f"Result for different document: {result}")
    assert result is None, "Should be cache miss for different document"
    
    # Test 5: Cache stats
    print("\n--- Test 5: Cache Statistics ---")
    stats = cache.get_cache_stats()
    print(f"Total entries: {stats['total_entries']}")
    print(f"Memory entries: {stats['memory_entries']}")
    print(f"Disk size: {stats['disk_size_mb']} MB")
    print(f"Expired entries: {stats['expired_entries']}")
    
    # Test 6: Clear cache
    print("\n--- Test 6: Clear Cache ---")
    cache.clear_cache()
    stats = cache.get_cache_stats()
    print(f"✅ Cache cleared. Remaining entries: {stats['total_entries']}")
    
    print("\n" + "="*60)
    print("✅ All cache tests passed!")
    print("="*60 + "\n")
