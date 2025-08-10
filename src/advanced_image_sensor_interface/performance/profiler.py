"""
Performance profiling utilities for identifying bottlenecks and optimization opportunities.
"""

import functools
import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Optional, TypeVar

import psutil

from ..types import Percentage, Seconds

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])
AF = TypeVar("AF", bound=Callable[..., Any])


@dataclass
class ProfileResult:
    """Result of a performance profiling operation."""

    function_name: str
    execution_time: Seconds
    memory_usage_mb: float
    cpu_usage_percent: Percentage
    call_count: int = 1
    min_time: Seconds = field(default_factory=lambda: float("inf"))
    max_time: Seconds = 0.0
    avg_time: Seconds = 0.0
    total_time: Seconds = 0.0

    def update(self, execution_time: Seconds, memory_mb: float, cpu_percent: Percentage) -> None:
        """Update profile result with new measurement."""
        self.call_count += 1
        self.total_time += execution_time
        self.min_time = min(self.min_time, execution_time)
        self.max_time = max(self.max_time, execution_time)
        self.avg_time = self.total_time / self.call_count
        self.execution_time = execution_time
        self.memory_usage_mb = memory_mb
        self.cpu_usage_percent = cpu_percent


class PerformanceProfiler:
    """
    Advanced performance profiler for monitoring function execution metrics.

    Provides detailed profiling including execution time, memory usage,
    CPU utilization, and call frequency analysis.
    """

    def __init__(self, enable_memory_profiling: bool = True, enable_cpu_profiling: bool = True):
        """Initialize the performance profiler."""
        self.enable_memory_profiling = enable_memory_profiling
        self.enable_cpu_profiling = enable_cpu_profiling
        self.results: dict[str, ProfileResult] = {}
        self.active_profiles: dict[str, float] = {}
        self._lock = threading.Lock()
        self._process = psutil.Process()

    def start_profile(self, name: str) -> None:
        """Start profiling a named operation."""
        with self._lock:
            self.active_profiles[name] = time.perf_counter()

    def end_profile(self, name: str) -> Optional[ProfileResult]:
        """End profiling and record results."""
        end_time = time.perf_counter()

        with self._lock:
            if name not in self.active_profiles:
                logger.warning(f"No active profile found for: {name}")
                return None

            start_time = self.active_profiles.pop(name)
            execution_time = end_time - start_time

            # Measure memory usage
            memory_mb = 0.0
            if self.enable_memory_profiling:
                try:
                    memory_info = self._process.memory_info()
                    memory_mb = memory_info.rss / (1024 * 1024)  # Convert to MB
                except Exception as e:
                    logger.debug(f"Failed to get memory info: {e}")

            # Measure CPU usage
            cpu_percent = 0.0
            if self.enable_cpu_profiling:
                try:
                    cpu_percent = self._process.cpu_percent()
                except Exception as e:
                    logger.debug(f"Failed to get CPU info: {e}")

            # Update or create profile result
            if name in self.results:
                self.results[name].update(execution_time, memory_mb, cpu_percent)
            else:
                self.results[name] = ProfileResult(
                    function_name=name,
                    execution_time=execution_time,
                    memory_usage_mb=memory_mb,
                    cpu_usage_percent=cpu_percent,
                    min_time=execution_time,
                    max_time=execution_time,
                    avg_time=execution_time,
                    total_time=execution_time,
                )

            return self.results[name]

    def profile_function(self, func: F) -> F:
        """Decorator to profile function execution."""

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            func_name = f"{func.__module__}.{func.__qualname__}"
            self.start_profile(func_name)
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                self.end_profile(func_name)

        return wrapper

    def profile_async_function(self, func: AF) -> AF:
        """Decorator to profile async function execution."""

        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            func_name = f"{func.__module__}.{func.__qualname__}"
            self.start_profile(func_name)
            try:
                result = await func(*args, **kwargs)
                return result
            finally:
                self.end_profile(func_name)

        return wrapper

    def get_results(self) -> dict[str, ProfileResult]:
        """Get all profiling results."""
        with self._lock:
            return self.results.copy()

    def get_top_functions(self, n: int = 10, sort_by: str = "total_time") -> list[ProfileResult]:
        """Get top N functions by specified metric."""
        with self._lock:
            results = list(self.results.values())

        if sort_by == "total_time":
            results.sort(key=lambda x: x.total_time, reverse=True)
        elif sort_by == "avg_time":
            results.sort(key=lambda x: x.avg_time, reverse=True)
        elif sort_by == "call_count":
            results.sort(key=lambda x: x.call_count, reverse=True)
        elif sort_by == "memory_usage":
            results.sort(key=lambda x: x.memory_usage_mb, reverse=True)
        else:
            raise ValueError(f"Invalid sort_by parameter: {sort_by}")

        return results[:n]

    def generate_report(self) -> str:
        """Generate a detailed profiling report."""
        with self._lock:
            if not self.results:
                return "No profiling data available."

            report_lines = ["Performance Profiling Report", "=" * 50, f"Total functions profiled: {len(self.results)}", ""]

            # Top functions by total time
            top_by_time = self.get_top_functions(5, "total_time")
            report_lines.extend(["Top 5 Functions by Total Time:", "-" * 30])

            for result in top_by_time:
                report_lines.append(
                    f"{result.function_name}: "
                    f"{result.total_time:.4f}s total, "
                    f"{result.avg_time:.4f}s avg, "
                    f"{result.call_count} calls"
                )

            report_lines.append("")

            # Top functions by average time
            top_by_avg = self.get_top_functions(5, "avg_time")
            report_lines.extend(["Top 5 Functions by Average Time:", "-" * 30])

            for result in top_by_avg:
                report_lines.append(
                    f"{result.function_name}: "
                    f"{result.avg_time:.4f}s avg, "
                    f"min: {result.min_time:.4f}s, "
                    f"max: {result.max_time:.4f}s"
                )

            report_lines.append("")

            # Memory usage summary
            if self.enable_memory_profiling:
                avg_memory = sum(r.memory_usage_mb for r in self.results.values()) / len(self.results)
                max_memory = max(r.memory_usage_mb for r in self.results.values())

                report_lines.extend(
                    [
                        "Memory Usage Summary:",
                        "-" * 20,
                        f"Average memory usage: {avg_memory:.2f} MB",
                        f"Peak memory usage: {max_memory:.2f} MB",
                        "",
                    ]
                )

            return "\n".join(report_lines)

    def clear_results(self) -> None:
        """Clear all profiling results."""
        with self._lock:
            self.results.clear()
            self.active_profiles.clear()


# Global profiler instance
_global_profiler = PerformanceProfiler()


def profile_function(func: F) -> F:
    """Global function decorator for performance profiling."""
    return _global_profiler.profile_function(func)


def profile_async_function(func: AF) -> AF:
    """Global async function decorator for performance profiling."""
    return _global_profiler.profile_async_function(func)


def get_profiler() -> PerformanceProfiler:
    """Get the global profiler instance."""
    return _global_profiler


def generate_global_report() -> str:
    """Generate a report from the global profiler."""
    return _global_profiler.generate_report()


def clear_global_results() -> None:
    """Clear results from the global profiler."""
    _global_profiler.clear_results()


class ContextProfiler:
    """Context manager for profiling code blocks."""

    def __init__(self, name: str, profiler: Optional[PerformanceProfiler] = None):
        """Initialize context profiler."""
        self.name = name
        self.profiler = profiler or _global_profiler
        self.result: Optional[ProfileResult] = None

    def __enter__(self) -> "ContextProfiler":
        """Enter profiling context."""
        self.profiler.start_profile(self.name)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit profiling context."""
        self.result = self.profiler.end_profile(self.name)


# Convenience function for context profiling
def profile_context(name: str, profiler: Optional[PerformanceProfiler] = None) -> ContextProfiler:
    """Create a context manager for profiling."""
    return ContextProfiler(name, profiler)
