"""
Tests for buffer management system.
"""

import threading
import time

import pytest
from advanced_image_sensor_interface.utils.buffer_manager import (
    AsyncBufferManager,
    BufferManager,
    BufferStats,
    ManagedBuffer,
    get_buffer_manager,
)


class TestBufferStats:
    """Test buffer statistics."""

    def test_initialization(self):
        """Test buffer stats initialization."""
        stats = BufferStats()

        assert stats.total_allocations == 0
        assert stats.total_deallocations == 0
        assert stats.pool_hits == 0
        assert stats.pool_misses == 0
        assert stats.current_pool_size == 0
        assert stats.peak_pool_size == 0
        assert stats.total_memory_allocated == 0
        assert stats.peak_memory_usage == 0


class TestBufferManager:
    """Test buffer manager functionality."""

    def test_initialization(self):
        """Test buffer manager initialization."""
        manager = BufferManager(pool_size=5, max_buffer_size=1024)

        assert manager._pool_size == 5
        assert manager._max_buffer_size == 1024
        assert len(manager._buffer_pools) == 0
        assert manager._stats is not None

    def test_get_buffer_basic(self):
        """Test basic buffer allocation."""
        manager = BufferManager(pool_size=5, max_buffer_size=1024)

        buffer = manager.get_buffer(512)

        assert isinstance(buffer, bytearray)
        assert len(buffer) == 512
        assert id(buffer) in manager._active_buffers

    def test_get_buffer_invalid_size(self):
        """Test buffer allocation with invalid sizes."""
        manager = BufferManager(max_buffer_size=1024)

        # Test zero size
        with pytest.raises(ValueError, match="Buffer size must be positive"):
            manager.get_buffer(0)

        # Test negative size
        with pytest.raises(ValueError, match="Buffer size must be positive"):
            manager.get_buffer(-1)

        # Test size too large
        with pytest.raises(ValueError, match="exceeds maximum"):
            manager.get_buffer(2048)

    def test_return_buffer(self):
        """Test buffer return to pool."""
        manager = BufferManager(pool_size=5)

        buffer = manager.get_buffer(512)
        returned = manager.return_buffer(buffer)

        assert returned is True
        assert 512 in manager._buffer_pools
        assert len(manager._buffer_pools[512]) == 1

    def test_buffer_reuse(self):
        """Test buffer pool reuse."""
        manager = BufferManager(pool_size=5)

        # Allocate and return buffer
        buffer1 = manager.get_buffer(512)
        manager.return_buffer(buffer1)

        # Get another buffer of same size
        buffer2 = manager.get_buffer(512)

        # Should reuse the same buffer
        assert buffer1 is buffer2

        # Check statistics
        stats = manager.get_stats()
        assert stats.pool_hits == 1
        assert stats.pool_misses == 1

    def test_pool_size_limit(self):
        """Test pool size limits."""
        manager = BufferManager(pool_size=2)

        # Create and return multiple buffers
        buffers = [manager.get_buffer(512) for _ in range(5)]

        # Return all buffers
        returned_count = 0
        for buffer in buffers:
            if manager.return_buffer(buffer):
                returned_count += 1

        # Only pool_size buffers should be kept
        assert returned_count == 2  # Only pool_size (2) buffers returned to pool
        assert len(manager._buffer_pools[512]) == 2  # Pool size limit enforced

    def test_return_invalid_buffer(self):
        """Test returning invalid buffer types."""
        manager = BufferManager()

        # Test non-bytearray
        result = manager.return_buffer("not a buffer")
        assert result is False

        # Test None
        result = manager.return_buffer(None)
        assert result is False

    def test_statistics_tracking(self):
        """Test statistics tracking."""
        manager = BufferManager(pool_size=2, enable_stats=True)

        # Allocate some buffers
        buffer1 = manager.get_buffer(512)
        buffer2 = manager.get_buffer(1024)
        buffer3 = manager.get_buffer(512)  # Should be pool miss

        # Return buffers
        manager.return_buffer(buffer1)
        manager.return_buffer(buffer2)
        manager.return_buffer(buffer3)  # Should be pool hit when reallocated

        # Get buffer again (should be pool hit)
        buffer4 = manager.get_buffer(512)

        stats = manager.get_stats()
        assert stats.total_allocations == 3
        assert stats.total_deallocations == 3
        assert stats.pool_hits == 1
        assert stats.pool_misses == 3

    def test_clear_pools(self):
        """Test clearing all buffer pools."""
        manager = BufferManager(pool_size=5)

        # Create and return some buffers
        for size in [512, 1024, 2048]:
            buffer = manager.get_buffer(size)
            manager.return_buffer(buffer)

        assert len(manager._buffer_pools) == 3

        cleared_count = manager.clear_pools()

        assert cleared_count == 3
        assert len(manager._buffer_pools) == 0

    def test_optimize_pools(self):
        """Test pool optimization."""
        manager = BufferManager(pool_size=2)

        # Create oversized pools by manually adding to pool
        manager._buffer_pools[512] = []
        for _ in range(5):
            buffer = bytearray(512)
            manager._buffer_pools[512].append(buffer)

        # Create empty pool entry
        manager._buffer_pools[1024] = []

        optimization_stats = manager.optimize_pools()

        assert optimization_stats["pools_removed"] == 1  # Empty pool removed
        assert optimization_stats["buffers_trimmed"] == 3  # 5 - 2 = 3 trimmed
        assert len(manager._buffer_pools[512]) == 2  # Pool size limit
        assert 1024 not in manager._buffer_pools  # Empty pool removed

    def test_memory_usage_tracking(self):
        """Test memory usage tracking."""
        manager = BufferManager(pool_size=3)

        # Create buffers of different sizes
        buffer1 = manager.get_buffer(512)
        buffer2 = manager.get_buffer(1024)

        # Return to pools
        manager.return_buffer(buffer1)
        manager.return_buffer(buffer2)

        memory_info = manager.get_memory_usage()

        assert memory_info["pool_memory"] == 512 + 1024
        assert memory_info["total_pools"] == 2
        assert "size_512" in memory_info["pool_breakdown"]
        assert "size_1024" in memory_info["pool_breakdown"]
        assert memory_info["pool_breakdown"]["size_512"]["count"] == 1
        assert memory_info["pool_breakdown"]["size_512"]["memory"] == 512

    def test_thread_safety(self):
        """Test thread safety of buffer manager."""
        manager = BufferManager(pool_size=10)
        results = []
        errors = []

        def worker():
            try:
                for _ in range(100):
                    buffer = manager.get_buffer(512)
                    time.sleep(0.001)  # Small delay to increase contention
                    returned = manager.return_buffer(buffer)
                    results.append(returned)
            except Exception as e:
                errors.append(e)

        # Start multiple threads
        threads = [threading.Thread(target=worker) for _ in range(5)]
        for thread in threads:
            thread.start()

        # Wait for completion
        for thread in threads:
            thread.join()

        # Check results
        assert len(errors) == 0, f"Thread safety errors: {errors}"
        assert len(results) == 500  # 5 threads * 100 operations
        assert all(results)  # All returns should be successful


class TestAsyncBufferManager:
    """Test async buffer manager functionality."""

    def test_initialization(self):
        """Test async buffer manager initialization."""
        manager = AsyncBufferManager(pool_size=5, max_buffer_size=1024)

        assert manager._pool_size == 5
        assert manager._max_buffer_size == 1024
        assert manager._optimization_interval == 60.0

    @pytest.mark.asyncio
    async def test_async_operations(self):
        """Test async buffer operations."""
        manager = AsyncBufferManager(pool_size=5)

        # Test async get_buffer
        buffer = await manager.get_buffer_async(512)
        assert isinstance(buffer, bytearray)
        assert len(buffer) == 512

        # Test async return_buffer
        returned = await manager.return_buffer_async(buffer)
        assert returned is True

    @pytest.mark.asyncio
    async def test_background_optimization(self):
        """Test background optimization trigger."""
        manager = AsyncBufferManager(pool_size=2)

        # Set last optimization to trigger background optimization
        manager._last_optimization = time.time() - 70  # 70 seconds ago

        # Create some buffers to optimize
        for _ in range(5):
            buffer = manager.get_buffer(512)
            manager.return_buffer(buffer)

        # This should trigger background optimization
        buffer = await manager.get_buffer_async(1024)

        # Check that optimization was triggered (pool should be trimmed)
        assert len(manager._buffer_pools[512]) <= 2


class TestManagedBuffer:
    """Test managed buffer context manager."""

    def test_context_manager_basic(self):
        """Test basic context manager functionality."""
        manager = BufferManager(pool_size=5)

        with ManagedBuffer(512, manager) as buffer:
            assert isinstance(buffer, bytearray)
            assert len(buffer) == 512

            # Buffer should be active
            assert id(buffer) in manager._active_buffers

        # Buffer should be returned to pool after context exit
        assert 512 in manager._buffer_pools
        assert len(manager._buffer_pools[512]) == 1

    def test_context_manager_with_exception(self):
        """Test context manager behavior with exceptions."""
        manager = BufferManager(pool_size=5)

        try:
            with ManagedBuffer(512, manager) as buffer:
                assert len(buffer) == 512
                raise ValueError("Test exception")
        except ValueError:
            pass

        # Buffer should still be returned to pool
        assert 512 in manager._buffer_pools
        assert len(manager._buffer_pools[512]) == 1

    def test_context_manager_default_manager(self):
        """Test context manager with default global manager."""
        with ManagedBuffer(512) as buffer:
            assert isinstance(buffer, bytearray)
            assert len(buffer) == 512


class TestGlobalBufferManager:
    """Test global buffer manager functionality."""

    def test_get_buffer_manager_singleton(self):
        """Test that get_buffer_manager returns singleton."""
        # Clear any existing global instance
        import advanced_image_sensor_interface.utils.buffer_manager as buffer_module
        buffer_module._global_buffer_manager = None

        manager1 = get_buffer_manager()
        manager2 = get_buffer_manager()

        assert manager1 is manager2
        assert isinstance(manager1, BufferManager)

    def test_get_buffer_manager_async(self):
        """Test getting async buffer manager."""
        # Clear any existing global instance
        import advanced_image_sensor_interface.utils.buffer_manager as buffer_module
        buffer_module._global_buffer_manager = None

        manager = get_buffer_manager(async_mode=True)

        assert isinstance(manager, AsyncBufferManager)

    def test_get_buffer_manager_with_params(self):
        """Test get_buffer_manager with custom parameters."""
        # Clear any existing global instance
        import advanced_image_sensor_interface.utils.buffer_manager as buffer_module
        buffer_module._global_buffer_manager = None

        manager = get_buffer_manager(pool_size=20, max_buffer_size=2048)

        assert manager._pool_size == 20
        assert manager._max_buffer_size == 2048


class TestBufferManagerPerformance:
    """Test buffer manager performance characteristics."""

    def test_large_buffer_handling(self):
        """Test handling of large buffers."""
        manager = BufferManager(max_buffer_size=10 * 1024 * 1024)  # 10MB

        # Allocate large buffer
        large_buffer = manager.get_buffer(5 * 1024 * 1024)  # 5MB
        assert len(large_buffer) == 5 * 1024 * 1024

        # Return and reuse
        manager.return_buffer(large_buffer)
        reused_buffer = manager.get_buffer(5 * 1024 * 1024)

        assert large_buffer is reused_buffer

    def test_memory_efficiency(self):
        """Test memory efficiency with multiple buffer sizes."""
        manager = BufferManager(pool_size=5)

        # Create buffers of various sizes
        sizes = [128, 256, 512, 1024, 2048]
        buffers = []

        for size in sizes:
            buffer = manager.get_buffer(size)
            buffers.append(buffer)

        # Return all buffers
        for buffer in buffers:
            manager.return_buffer(buffer)

        # Check that pools are created for each size
        assert len(manager._buffer_pools) == len(sizes)
        for size in sizes:
            assert size in manager._buffer_pools
            assert len(manager._buffer_pools[size]) == 1

        # Check memory usage tracking
        memory_info = manager.get_memory_usage()
        expected_memory = sum(sizes)
        assert memory_info["pool_memory"] == expected_memory

    def test_pool_optimization_performance(self):
        """Test performance impact of pool optimization."""
        manager = BufferManager(pool_size=3)

        # Create many buffers to trigger optimization need
        # Manually create oversized pools
        for size in [512, 1024, 2048]:
            manager._buffer_pools[size] = []
            for _ in range(10):  # Create more than pool_size (3)
                buffer = bytearray(size)
                manager._buffer_pools[size].append(buffer)

        # Measure optimization time
        start_time = time.time()
        optimization_stats = manager.optimize_pools()
        optimization_time = time.time() - start_time

        # Optimization should be fast (< 0.1 seconds)
        assert optimization_time < 0.1

        # Should have trimmed excess buffers
        assert optimization_stats["buffers_trimmed"] > 0

        # All pools should now be at or below pool_size
        for pool in manager._buffer_pools.values():
            assert len(pool) <= manager._pool_size
