"""
Performance benchmarks for buffer management systems.
"""

import threading
import time
from typing import Any

import numpy as np
import pytest

from advanced_image_sensor_interface.utils.buffer_manager import AsyncBufferManager, BufferManager


class BufferManagerBenchmarks:
    """Comprehensive benchmarks for buffer management performance."""

    @pytest.mark.benchmark(group="buffer_allocation")
    def test_buffer_allocation_performance(self, benchmark):
        """Benchmark buffer allocation performance."""
        manager = BufferManager(max_pool_size=100)

        def allocate_and_return():
            buffer = manager.get_buffer(1024 * 1024)  # 1MB buffer
            if buffer:
                manager.return_buffer(buffer)

        benchmark(allocate_and_return)

        # Verify performance metrics
        stats = manager.get_statistics()
        assert stats["pool_hit_rate"] > 0.8  # Should have good hit rate

    @pytest.mark.benchmark(group="buffer_allocation")
    def test_large_buffer_allocation(self, benchmark):
        """Benchmark large buffer allocation."""
        manager = BufferManager(max_pool_size=50)

        def allocate_large_buffer():
            buffer = manager.get_buffer(10 * 1024 * 1024)  # 10MB buffer
            if buffer:
                manager.return_buffer(buffer)

        benchmark(allocate_large_buffer)

    @pytest.mark.benchmark(group="buffer_threading")
    def test_concurrent_buffer_access(self, benchmark):
        """Benchmark concurrent buffer access performance."""
        manager = BufferManager(max_pool_size=200)
        num_threads = 10
        operations_per_thread = 100

        def worker():
            for _ in range(operations_per_thread):
                buffer = manager.get_buffer(512 * 1024)  # 512KB
                if buffer:
                    # Simulate some work
                    time.sleep(0.0001)
                    manager.return_buffer(buffer)

        def run_concurrent_test():
            threads = []
            for _ in range(num_threads):
                thread = threading.Thread(target=worker)
                threads.append(thread)

            for thread in threads:
                thread.start()

            for thread in threads:
                thread.join()

        benchmark(run_concurrent_test)

        # Verify thread safety
        stats = manager.get_statistics()
        expected_operations = num_threads * operations_per_thread
        assert stats["total_allocated"] == expected_operations
        assert stats["total_returned"] == expected_operations

    @pytest.mark.benchmark(group="buffer_async")
    @pytest.mark.asyncio
    async def test_async_buffer_performance(self, benchmark):
        """Benchmark async buffer manager performance."""
        manager = AsyncBufferManager(max_pool_size=100)

        async def async_allocate_and_return():
            buffer = await manager.get_buffer_async(1024 * 1024)
            if buffer:
                await manager.return_buffer_async(buffer)

        await benchmark(async_allocate_and_return)

    @pytest.mark.benchmark(group="buffer_optimization")
    def test_pool_optimization_performance(self, benchmark):
        """Benchmark pool optimization performance."""
        manager = BufferManager(max_pool_size=1000, enable_optimization=True)

        # Pre-populate with various sizes
        buffers = []
        for size in [1024, 2048, 4096, 8192, 16384]:
            for _ in range(20):
                buffer = manager.get_buffer(size)
                if buffer:
                    buffers.append(buffer)

        # Return all buffers
        for buffer in buffers:
            manager.return_buffer(buffer)

        def optimize_pools():
            manager.optimize_pools()

        benchmark(optimize_pools)

    @pytest.mark.benchmark(group="buffer_memory")
    def test_memory_efficiency(self, benchmark):
        """Benchmark memory efficiency of buffer pools."""
        manager = BufferManager(max_pool_size=500)

        def memory_test():
            # Allocate many buffers of different sizes
            buffers = []
            sizes = [1024, 2048, 4096, 8192, 16384, 32768]

            for size in sizes:
                for _ in range(50):
                    buffer = manager.get_buffer(size)
                    if buffer:
                        buffers.append(buffer)

            # Return all buffers
            for buffer in buffers:
                manager.return_buffer(buffer)

            return len(buffers)

        benchmark(memory_test)

        # Check memory efficiency
        stats = manager.get_statistics()
        memory_usage = stats.get("total_memory_mb", 0)
        assert memory_usage > 0  # Should track memory usage

    @pytest.mark.benchmark(group="buffer_stress")
    def test_high_frequency_allocation(self, benchmark):
        """Benchmark high-frequency buffer allocation/deallocation."""
        manager = BufferManager(max_pool_size=1000)

        def high_frequency_test():
            for _ in range(1000):
                buffer = manager.get_buffer(4096)
                if buffer:
                    manager.return_buffer(buffer)

        benchmark(high_frequency_test)

        # Verify pool efficiency
        stats = manager.get_statistics()
        assert stats["pool_hit_rate"] > 0.9  # Should have very high hit rate

    @pytest.mark.benchmark(group="buffer_patterns")
    def test_realistic_usage_pattern(self, benchmark):
        """Benchmark realistic buffer usage patterns."""
        manager = BufferManager(max_pool_size=200)

        def realistic_pattern():
            # Simulate camera frame processing
            frame_buffers = []
            processing_buffers = []

            # Allocate frame buffers (typical camera resolutions)
            frame_sizes = [1920 * 1080 * 2, 2048 * 1536 * 2, 4096 * 3072 * 2]  # 1080p 16-bit  # 3MP 16-bit  # 12MP 16-bit

            for size in frame_sizes:
                for _ in range(10):
                    buffer = manager.get_buffer(size)
                    if buffer:
                        frame_buffers.append(buffer)

            # Allocate processing buffers
            for _ in range(20):
                buffer = manager.get_buffer(1024 * 1024)  # 1MB processing buffer
                if buffer:
                    processing_buffers.append(buffer)

            # Return buffers in mixed order (realistic usage)
            all_buffers = frame_buffers + processing_buffers
            np.random.shuffle(all_buffers)

            for buffer in all_buffers:
                manager.return_buffer(buffer)

        benchmark(realistic_pattern)

    @pytest.mark.benchmark(group="buffer_comparison")
    def test_vs_standard_allocation(self, benchmark):
        """Compare buffer manager vs standard memory allocation."""

        def standard_allocation():
            buffers = []
            for _ in range(100):
                # Standard Python bytes allocation
                buffer = bytearray(1024 * 1024)  # 1MB
                buffers.append(buffer)
            # Let buffers go out of scope for GC

        benchmark(standard_allocation)

    @pytest.mark.benchmark(group="buffer_comparison")
    def test_buffer_manager_allocation(self, benchmark):
        """Benchmark buffer manager allocation for comparison."""
        manager = BufferManager(max_pool_size=200)

        def manager_allocation():
            buffers = []
            for _ in range(100):
                buffer = manager.get_buffer(1024 * 1024)  # 1MB
                if buffer:
                    buffers.append(buffer)

            # Return all buffers
            for buffer in buffers:
                manager.return_buffer(buffer)

        benchmark(manager_allocation)


# Utility functions for benchmark analysis
def analyze_benchmark_results(results: dict[str, Any]) -> dict[str, Any]:
    """Analyze benchmark results and provide insights."""
    analysis = {"performance_summary": {}, "recommendations": [], "bottlenecks": []}

    # Analyze allocation performance
    if "buffer_allocation" in results:
        allocation_times = results["buffer_allocation"]
        avg_time = sum(allocation_times) / len(allocation_times)

        analysis["performance_summary"]["allocation_avg_ms"] = avg_time * 1000

        if avg_time > 0.001:  # > 1ms
            analysis["bottlenecks"].append("Buffer allocation is slow")
            analysis["recommendations"].append("Increase buffer pool size")

    # Analyze threading performance
    if "buffer_threading" in results:
        threading_time = results["buffer_threading"]

        if threading_time > 1.0:  # > 1 second for concurrent test
            analysis["bottlenecks"].append("Poor concurrent performance")
            analysis["recommendations"].append("Review thread safety implementation")

    return analysis


def generate_benchmark_report(results: dict[str, Any]) -> str:
    """Generate a comprehensive benchmark report."""
    report_lines = ["Buffer Manager Performance Benchmark Report", "=" * 50, ""]

    # Performance summary
    if "performance_summary" in results:
        report_lines.extend(["Performance Summary:", "-" * 20])

        for metric, value in results["performance_summary"].items():
            report_lines.append(f"{metric}: {value}")

        report_lines.append("")

    # Recommendations
    if results.get("recommendations"):
        report_lines.extend(["Recommendations:", "-" * 15])

        for i, rec in enumerate(results["recommendations"], 1):
            report_lines.append(f"{i}. {rec}")

        report_lines.append("")

    # Bottlenecks
    if results.get("bottlenecks"):
        report_lines.extend(["Identified Bottlenecks:", "-" * 22])

        for i, bottleneck in enumerate(results["bottlenecks"], 1):
            report_lines.append(f"{i}. {bottleneck}")

    return "\n".join(report_lines)
