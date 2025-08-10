"""
Advanced Buffer Management for High-Performance Image Processing

This module provides efficient buffer management with pooling, recycling,
and memory optimization for high-throughput image sensor operations.
"""

import logging
import threading
import time
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class BufferStats:
    """Buffer pool statistics."""

    total_allocations: int = 0
    total_deallocations: int = 0
    pool_hits: int = 0
    pool_misses: int = 0
    current_pool_size: int = 0
    peak_pool_size: int = 0
    total_memory_allocated: int = 0
    peak_memory_usage: int = 0


class BufferManager:
    """
    High-performance buffer manager with pooling and recycling.

    Provides efficient memory management for image processing operations
    with automatic buffer pooling, size-based allocation, and memory tracking.
    """

    def __init__(self, pool_size: int = 10, max_buffer_size: int = 100 * 1024 * 1024, enable_stats: bool = True):  # 100MB
        """
        Initialize buffer manager.

        Args:
            pool_size: Maximum number of buffers to keep in pool
            max_buffer_size: Maximum size for a single buffer
            enable_stats: Whether to collect statistics
        """
        self._pool_size = pool_size
        self._max_buffer_size = max_buffer_size
        self._enable_stats = enable_stats

        # Size-based buffer pools (size -> list of buffers)
        self._buffer_pools: dict[int, list[bytearray]] = {}
        self._lock = threading.RLock()

        # Statistics tracking
        self._stats = BufferStats() if enable_stats else None

        # Track active buffers using their id (since bytearray can't be weakly referenced or hashed)
        self._active_buffers: dict[int, int] = {}  # id -> size mapping

        logger.info(f"Buffer manager initialized: pool_size={pool_size}, max_size={max_buffer_size}")

    def get_buffer(self, size: int) -> bytearray:
        """
        Get a buffer of the specified size.

        Args:
            size: Required buffer size in bytes

        Returns:
            bytearray: Buffer of requested size

        Raises:
            ValueError: If size is invalid or too large
        """
        if size <= 0:
            raise ValueError(f"Buffer size must be positive, got {size}")

        if size > self._max_buffer_size:
            raise ValueError(f"Buffer size {size} exceeds maximum {self._max_buffer_size}")

        with self._lock:
            # Try to get from pool first
            if self._buffer_pools.get(size):
                buffer = self._buffer_pools[size].pop()
                if self._stats:
                    self._stats.pool_hits += 1
                    self._stats.current_pool_size -= 1

                # Clear the buffer for reuse
                buffer[:] = b"\x00" * len(buffer)
                self._active_buffers[id(buffer)] = size

                logger.debug(f"Buffer retrieved from pool: {size} bytes")
                return buffer

            # Create new buffer
            buffer = bytearray(size)
            self._active_buffers[id(buffer)] = size

            if self._stats:
                self._stats.pool_misses += 1
                self._stats.total_allocations += 1
                self._stats.total_memory_allocated += size
                if self._stats.total_memory_allocated > self._stats.peak_memory_usage:
                    self._stats.peak_memory_usage = self._stats.total_memory_allocated

            logger.debug(f"New buffer allocated: {size} bytes")
            return buffer

    def return_buffer(self, buffer: bytearray) -> bool:
        """
        Return a buffer to the pool for reuse.

        Args:
            buffer: Buffer to return

        Returns:
            bool: True if buffer was returned to pool, False otherwise
        """
        if not isinstance(buffer, bytearray):
            logger.warning("Attempted to return non-bytearray buffer")
            return False

        size = len(buffer)

        with self._lock:
            # Remove from active buffers tracking
            self._active_buffers.pop(id(buffer), None)

            # Initialize pool for this size if needed
            if size not in self._buffer_pools:
                self._buffer_pools[size] = []

            # Only return to pool if we have space
            if len(self._buffer_pools[size]) < self._pool_size:
                # Clear sensitive data
                buffer[:] = b"\x00" * len(buffer)
                self._buffer_pools[size].append(buffer)

                if self._stats:
                    self._stats.total_deallocations += 1
                    self._stats.current_pool_size += 1
                    if self._stats.current_pool_size > self._stats.peak_pool_size:
                        self._stats.peak_pool_size = self._stats.current_pool_size

                logger.debug(f"Buffer returned to pool: {size} bytes")
                return True
            else:
                # Pool is full, let buffer be garbage collected
                if self._stats:
                    self._stats.total_deallocations += 1
                    self._stats.total_memory_allocated -= size

                logger.debug(f"Buffer discarded (pool full): {size} bytes")
                return False

    def get_stats(self) -> Optional[BufferStats]:
        """
        Get buffer pool statistics.

        Returns:
            BufferStats: Current statistics or None if disabled
        """
        if not self._stats:
            return None

        with self._lock:
            # Update current pool size
            self._stats.current_pool_size = sum(len(pool) for pool in self._buffer_pools.values())
            return self._stats

    def clear_pools(self) -> int:
        """
        Clear all buffer pools and free memory.

        Returns:
            int: Number of buffers cleared
        """
        with self._lock:
            total_cleared = sum(len(pool) for pool in self._buffer_pools.values())
            self._buffer_pools.clear()

            if self._stats:
                self._stats.current_pool_size = 0
                self._stats.total_memory_allocated = sum(self._active_buffers.values())

            logger.info(f"Cleared {total_cleared} buffers from pools")
            return total_cleared

    def optimize_pools(self) -> dict[str, int]:
        """
        Optimize buffer pools by removing unused sizes and consolidating.

        Returns:
            dict: Optimization statistics
        """
        with self._lock:
            original_pools = len(self._buffer_pools)
            original_buffers = sum(len(pool) for pool in self._buffer_pools.values())

            # Remove empty pools
            empty_pools = [size for size, pool in self._buffer_pools.items() if not pool]
            for size in empty_pools:
                del self._buffer_pools[size]

            # Trim oversized pools
            trimmed_buffers = 0
            for size, pool in self._buffer_pools.items():
                if len(pool) > self._pool_size:
                    excess = len(pool) - self._pool_size
                    del pool[self._pool_size :]
                    trimmed_buffers += excess

            final_pools = len(self._buffer_pools)
            final_buffers = sum(len(pool) for pool in self._buffer_pools.values())

            optimization_stats = {
                "pools_removed": original_pools - final_pools,
                "buffers_trimmed": trimmed_buffers,
                "memory_freed": (original_buffers - final_buffers) * 1024,  # Estimate
                "final_pool_count": final_pools,
                "final_buffer_count": final_buffers,
            }

            logger.info(f"Pool optimization completed: {optimization_stats}")
            return optimization_stats

    def get_memory_usage(self) -> dict[str, int]:
        """
        Get detailed memory usage information.

        Returns:
            dict: Memory usage statistics
        """
        with self._lock:
            pool_memory = 0
            pool_breakdown = {}

            for size, pool in self._buffer_pools.items():
                pool_size_memory = size * len(pool)
                pool_memory += pool_size_memory
                pool_breakdown[f"size_{size}"] = {"count": len(pool), "memory": pool_size_memory}

            active_count = len(self._active_buffers)

            return {
                "pool_memory": pool_memory,
                "active_buffers": active_count,
                "total_pools": len(self._buffer_pools),
                "pool_breakdown": pool_breakdown,
                "estimated_active_memory": sum(self._active_buffers.values()),
            }


# Global buffer manager instance
_global_buffer_manager: BufferManager | None = None


# Context manager for automatic buffer management
class ManagedBuffer:
    """Context manager for automatic buffer lifecycle management."""

    def __init__(self, size: int, manager: Optional[BufferManager] = None):
        """
        Initialize managed buffer.

        Args:
            size: Buffer size
            manager: Buffer manager (uses global if None)
        """
        self.size = size
        self.manager = manager or get_buffer_manager()
        self.buffer: Optional[bytearray] = None

    def __enter__(self) -> bytearray:
        """Get buffer on context entry."""
        self.buffer = self.manager.get_buffer(self.size)
        return self.buffer

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Return buffer on context exit."""
        if self.buffer is not None:
            self.manager.return_buffer(self.buffer)
            self.buffer = None


# Example usage and testing
if __name__ == "__main__":
    print("Testing Buffer Manager...")

    # Create buffer manager
    manager = BufferManager(pool_size=5, max_buffer_size=1024 * 1024)

    # Test basic allocation and return
    print("\n1. Basic Buffer Operations:")
    print("-" * 30)

    buffer1 = manager.get_buffer(1024)
    print(f"Allocated buffer: {len(buffer1)} bytes")

    returned = manager.return_buffer(buffer1)
    print(f"Buffer returned: {returned}")

    # Test pool reuse
    buffer2 = manager.get_buffer(1024)
    print(f"Reused buffer: {len(buffer2)} bytes")

    # Test statistics
    stats = manager.get_stats()
    if stats:
        print(f"Pool hits: {stats.pool_hits}")
        print(f"Pool misses: {stats.pool_misses}")

    # Test context manager
    print("\n2. Context Manager Test:")
    print("-" * 30)

    with ManagedBuffer(2048, manager) as buf:
        print(f"Context buffer: {len(buf)} bytes")
        buf[:10] = b"test data!"

    print("Buffer automatically returned")

    # Test memory usage
    print("\n3. Memory Usage:")
    print("-" * 30)

    memory_info = manager.get_memory_usage()
    print(f"Pool memory: {memory_info['pool_memory']} bytes")
    print(f"Active buffers: {memory_info['active_buffers']}")

    print("\nBuffer manager tests completed!")


class AsyncBufferManager(BufferManager):
    """
    Async-compatible buffer manager.

    This extends BufferManager with async methods and background optimization.
    """

    def __init__(self, pool_size: int = 10, max_buffer_size: int = 1024 * 1024 * 1024):
        """Initialize async buffer manager."""
        super().__init__(pool_size=pool_size, max_buffer_size=max_buffer_size)
        self._optimization_interval = 60.0  # seconds
        self._last_optimization = time.time()

    async def get_buffer_async(self, size: int) -> bytearray:
        """
        Asynchronously get a buffer of the specified size.

        Args:
            size: Size of buffer in bytes

        Returns:
            bytearray: Buffer of requested size
        """
        # Check if background optimization is needed
        if time.time() - self._last_optimization > self._optimization_interval:
            self._background_optimize()

        # Use synchronous method for now (can be made truly async later)
        return self.get_buffer(size)

    async def return_buffer_async(self, buffer: bytearray) -> bool:
        """
        Asynchronously return a buffer to the pool.

        Args:
            buffer: Buffer to return

        Returns:
            bool: True if buffer was returned to pool
        """
        # Use synchronous method for now (can be made truly async later)
        return self.return_buffer(buffer)

    def _background_optimize(self) -> None:
        """Perform background optimization."""
        self.optimize_pools()
        self._last_optimization = time.time()


# Global buffer manager instance for singleton pattern
_global_buffer_manager = None


def get_buffer_manager(max_buffer_size: int | None = None, pool_size: int = 10, async_mode: bool = False) -> BufferManager:
    """
    Factory function to create a BufferManager with appropriate settings.
    Uses singleton pattern to return the same instance, but recreates if parameters change.

    Args:
        max_buffer_size: Maximum buffer size in bytes (defaults to 100MB for image data)
        pool_size: Number of buffers to keep in each pool
        async_mode: If True, returns AsyncBufferManager instance

    Returns:
        BufferManager: Configured buffer manager instance
    """
    global _global_buffer_manager

    # Default to 100MB for image data if not specified
    if max_buffer_size is None:
        max_buffer_size = 100 * 1024 * 1024  # 100MB

    # Check if we need to create a new instance (different parameters or no existing instance)
    needs_new_instance = (
        _global_buffer_manager is None
        or _global_buffer_manager._max_buffer_size != max_buffer_size
        or _global_buffer_manager._pool_size != pool_size
        or (async_mode and not isinstance(_global_buffer_manager, AsyncBufferManager))
        or (not async_mode and isinstance(_global_buffer_manager, AsyncBufferManager))
    )

    if needs_new_instance:
        # Create appropriate manager type
        if async_mode:
            _global_buffer_manager = AsyncBufferManager(pool_size=pool_size, max_buffer_size=max_buffer_size)
        else:
            _global_buffer_manager = BufferManager(pool_size=pool_size, max_buffer_size=max_buffer_size)

    return _global_buffer_manager
