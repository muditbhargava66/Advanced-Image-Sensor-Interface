"""
Comprehensive benchmarking suite for the Advanced Image Sensor Interface.

This module provides performance benchmarks for all major components
including buffer management, protocol drivers, image processing,
and system integration.
"""

from .buffer_benchmarks import BufferManagerBenchmarks
from .noise_analysis import run_noise_analysis
from .performance_benchmark import PerformanceBenchmark
from .speed_tests import PerformanceProfiler, BenchmarkSuite, run_performance_profile

__all__ = [
    "BufferManagerBenchmarks",
    "run_noise_analysis",
    "PerformanceBenchmark",
    "PerformanceProfiler",
    "BenchmarkSuite",
    "run_performance_profile",
]
