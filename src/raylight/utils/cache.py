"""LRU cache for mmap'd state dictionaries.

Provides bounded caching for model state dicts loaded via mmap,
evicting least-recently-used entries when capacity is exceeded.
"""
from collections import OrderedDict
from typing import Dict, Any, Optional


class LRUStateCache:
    """LRU cache for mmap state dicts with configurable max size."""
    
    def __init__(self, max_size: int = 2):
        """
        Args:
            max_size: Maximum number of state dicts to cache.
                      When exceeded, least-recently-used entry is evicted.
        """
        self.max_size = max_size
        self._cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()
    
    def get(self, key: str) -> Optional[Dict[str, Any]]:
        """Get cached state dict, marking it as recently used.
        
        Returns:
            State dict if cached, None otherwise.
        """
        if key in self._cache:
            self._cache.move_to_end(key)  # Mark as recently used
            return self._cache[key]
        return None
    
    def put(self, key: str, value: Dict[str, Any]) -> Optional[str]:
        """Add or update state dict in cache.
        
        Args:
            key: Cache key (typically file path)
            value: State dict to cache
            
        Returns:
            Key of evicted entry if eviction occurred, None otherwise.
        """
        evicted_key = None
        
        if key in self._cache:
            self._cache.move_to_end(key)
            self._cache[key] = value
        else:
            if len(self._cache) >= self.max_size:
                evicted_key, _ = self._cache.popitem(last=False)
                print(f"[LRUStateCache] Evicted: {evicted_key}")
            self._cache[key] = value
        
        return evicted_key
    
    def clear(self):
        """Clear all cached entries."""
        self._cache.clear()
    
    def __contains__(self, key: str) -> bool:
        """Check if key is in cache (does not affect LRU order)."""
        return key in self._cache
    
    def __len__(self) -> int:
        """Return number of cached entries."""
        return len(self._cache)
    
    def keys(self):
        """Return cache keys in LRU order (oldest first)."""
        return self._cache.keys()
