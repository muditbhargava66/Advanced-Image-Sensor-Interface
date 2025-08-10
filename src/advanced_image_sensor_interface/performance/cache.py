"""
Caching utilities for performance optimization.
"""

import threading
import time
from collections import OrderedDict
from typing import Any, Optional, TypeVar

T = TypeVar("T")


class LRUCache:
    """Least Recently Used cache implementation."""

    def __init__(self, max_size: int = 128):
        """Initialize LRU cache."""
        self.max_size = max_size
        self.cache: OrderedDict = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: str) -> Optional[Any]:
        """Get item from cache."""
        with self._lock:
            if key in self.cache:
                # Move to end (most recently used)
                self.cache.move_to_end(key)
                return self.cache[key]
            return None

    def put(self, key: str, value: Any) -> None:
        """Put item in cache."""
        with self._lock:
            if key in self.cache:
                self.cache.move_to_end(key)
            elif len(self.cache) >= self.max_size:
                # Remove least recently used
                self.cache.popitem(last=False)
            self.cache[key] = value


class MemoryCache:
    """Simple memory cache with TTL support."""

    def __init__(self, default_ttl: float = 300.0):
        """Initialize memory cache."""
        self.default_ttl = default_ttl
        self.cache: dict[str, tuple[Any, float]] = {}
        self._lock = threading.Lock()

    def get(self, key: str) -> Optional[Any]:
        """Get item from cache."""
        with self._lock:
            if key in self.cache:
                value, expiry = self.cache[key]
                if time.time() < expiry:
                    return value
                else:
                    del self.cache[key]
            return None

    def put(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """Put item in cache."""
        ttl = ttl or self.default_ttl
        expiry = time.time() + ttl

        with self._lock:
            self.cache[key] = (value, expiry)


class CacheManager:
    """Manages multiple cache instances."""

    def __init__(self):
        """Initialize cache manager."""
        self.caches: dict[str, Any] = {}

    def get_cache(self, name: str, cache_type: str = "lru", **kwargs) -> Any:
        """Get or create a cache instance."""
        if name not in self.caches:
            if cache_type == "lru":
                self.caches[name] = LRUCache(**kwargs)
            elif cache_type == "memory":
                self.caches[name] = MemoryCache(**kwargs)
            else:
                raise ValueError(f"Unknown cache type: {cache_type}")

        return self.caches[name]
