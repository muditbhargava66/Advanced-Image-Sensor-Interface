"""
Performance Benchmarks and Speed Tests

This module provides realistic performance benchmarks for the Advanced Image Sensor
Interface simulation framework. All measurements are clearly marked as simulation
performance, not hardware throughput.

Classes:
    PerformanceProfiler: Main class for performance profiling
    BenchmarkSuite: Comprehensive benchmark suite

Functions:
    benchmark_mipi_simulation: Benchmark MIPI protocol simulation
    benchmark_signal_processing: Benchmark image processing pipeline
    benchmark_power_modeling: Benchmark power management simulation
"""

import logging
import platform
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import psutil

from advanced_image_sensor_interface.sensor_interface.mipi_driver import MIPIConfig, MIPIDriver
from advanced_image_sensor_interface.sensor_interface.power_management import PowerConfig, PowerManager
from advanced_image_sensor_interface.sensor_interface.signal_processing import SignalConfig, SignalProcessor

logger = logging.getLogger(__name__)


@dataclass
class SystemInfo:
    """System information for benchmark context."""

    cpu_model: str
    cpu_cores: int
    cpu_threads: int
    memory_gb: float
    python_version: str
    numpy_version: str
    platform: str


@dataclass
class BenchmarkResult:
    """Result of a performance benchmark."""

    test_name: str
    frames_per_second: float
    processing_time_ms: float
    memory_usage_mb: float
    cpu_usage_percent: float
    system_info: SystemInfo
    configuration: dict
    notes: str = ""


class PerformanceProfiler:
    """
    Performance profiler for simulation components.

    IMPORTANT: All measurements are simulation performance in Python,
    not hardware throughput. Real hardware would have different characteristics.
    """

    def __init__(self):
        """Initialize the performance profiler."""
        self.system_info = self._get_system_info()
        self.results: list[BenchmarkResult] = []

    def _get_system_info(self) -> SystemInfo:
        """Get system information for benchmark context."""
        import numpy

        return SystemInfo(
            cpu_model=platform.processor() or "Unknown",
            cpu_cores=psutil.cpu_count(logical=False) or 1,
            cpu_threads=psutil.cpu_count(logical=True) or 1,
            memory_gb=psutil.virtual_memory().total / (1024**3),
            python_version=platform.python_version(),
            numpy_version=numpy.__version__,
            platform=platform.platform(),
        )

    def benchmark_mipi_simulation(self, config: MIPIConfig, data_sizes: list[int], iterations: int = 100) -> BenchmarkResult:
        """
        Benchmark MIPI protocol simulation performance.

        Args:
            config: MIPI configuration
            data_sizes: List of data sizes to test (bytes)
            iterations: Number of iterations per test

        Returns:
            BenchmarkResult with simulation performance metrics
        """
        driver = MIPIDriver(config)

        # Warm up
        test_data = b"0" * 1024
        for _ in range(10):
            driver.send_data(test_data)

        total_time = 0.0
        total_bytes = 0
        memory_before = psutil.Process().memory_info().rss / (1024**2)
        cpu_before = psutil.cpu_percent()

        start_time = time.perf_counter()

        for data_size in data_sizes:
            test_data = b"0" * data_size
            for _ in range(iterations):
                driver.send_data(test_data)
                total_bytes += data_size

        end_time = time.perf_counter()
        total_time = end_time - start_time

        memory_after = psutil.Process().memory_info().rss / (1024**2)
        cpu_after = psutil.cpu_percent()

        # Calculate metrics (simulation performance, not hardware)
        throughput_mbps = (total_bytes / (1024**2)) / total_time
        frames_per_second = (len(data_sizes) * iterations) / total_time

        result = BenchmarkResult(
            test_name="MIPI Simulation",
            frames_per_second=frames_per_second,
            processing_time_ms=total_time * 1000,
            memory_usage_mb=memory_after - memory_before,
            cpu_usage_percent=cpu_after - cpu_before,
            system_info=self.system_info,
            configuration={
                "lanes": config.lanes,
                "data_rate_gbps": config.data_rate,
                "channel": config.channel,
                "data_sizes": data_sizes,
                "iterations": iterations,
                "throughput_mbps_simulated": throughput_mbps,
            },
            notes="Simulation performance in Python, not hardware throughput",
        )

        self.results.append(result)
        return result

    def benchmark_signal_processing(
        self, resolutions: list[tuple[int, int, int]], bit_depths: list[int], iterations: int = 10
    ) -> BenchmarkResult:
        """
        Benchmark signal processing pipeline performance.

        Args:
            resolutions: List of (height, width, channels) tuples
            bit_depths: List of bit depths to test
            iterations: Number of iterations per test

        Returns:
            BenchmarkResult with processing performance metrics
        """
        total_frames = 0
        total_time = 0.0
        memory_before = psutil.Process().memory_info().rss / (1024**2)

        start_time = time.perf_counter()

        for height, width, channels in resolutions:
            for bit_depth in bit_depths:
                # Create processor
                config = SignalConfig(bit_depth=bit_depth, noise_reduction_strength=0.1, color_correction_matrix=np.eye(3))
                processor = SignalProcessor(config)

                # Generate test frame
                max_val = (2**bit_depth) - 1
                if channels == 1:
                    frame = np.random.randint(0, max_val + 1, (height, width), dtype=np.uint16)
                else:
                    frame = np.random.randint(0, max_val + 1, (height, width, channels), dtype=np.uint16)

                # Process frames
                for _ in range(iterations):
                    _ = processor.process_frame(frame)
                    total_frames += 1

        end_time = time.perf_counter()
        total_time = end_time - start_time

        memory_after = psutil.Process().memory_info().rss / (1024**2)

        frames_per_second = total_frames / total_time
        avg_processing_time = (total_time / total_frames) * 1000  # ms per frame

        result = BenchmarkResult(
            test_name="Signal Processing",
            frames_per_second=frames_per_second,
            processing_time_ms=avg_processing_time,
            memory_usage_mb=memory_after - memory_before,
            cpu_usage_percent=psutil.cpu_percent(),
            system_info=self.system_info,
            configuration={
                "resolutions": resolutions,
                "bit_depths": bit_depths,
                "iterations": iterations,
                "total_frames": total_frames,
            },
            notes="Pure Python processing performance, not optimized for real-time",
        )

        self.results.append(result)
        return result

    def benchmark_power_modeling(self, iterations: int = 1000) -> BenchmarkResult:
        """
        Benchmark power management simulation performance.

        Args:
            iterations: Number of power calculations to perform

        Returns:
            BenchmarkResult with power modeling performance
        """
        config = PowerConfig(voltage_main=1.8, voltage_io=3.3, current_limit=1.0)
        power_manager = PowerManager(config)

        memory_before = psutil.Process().memory_info().rss / (1024**2)
        start_time = time.perf_counter()

        for _ in range(iterations):
            _ = power_manager.get_power_status()
            power_manager.set_voltage("main", 1.8 + np.random.uniform(-0.1, 0.1))

        end_time = time.perf_counter()
        total_time = end_time - start_time

        memory_after = psutil.Process().memory_info().rss / (1024**2)

        calculations_per_second = iterations / total_time

        result = BenchmarkResult(
            test_name="Power Modeling",
            frames_per_second=calculations_per_second,  # calculations per second
            processing_time_ms=total_time * 1000,
            memory_usage_mb=memory_after - memory_before,
            cpu_usage_percent=psutil.cpu_percent(),
            system_info=self.system_info,
            configuration={
                "iterations": iterations,
                "voltage_main": config.voltage_main,
                "voltage_io": config.voltage_io,
                "current_limit": config.current_limit,
            },
            notes="Power modeling simulation, not actual hardware measurements",
        )

        self.results.append(result)
        return result

    def generate_report(self) -> dict:
        """Generate a comprehensive performance report."""
        if not self.results:
            return {"error": "No benchmark results available"}

        report = {
            "system_info": {
                "cpu_model": self.system_info.cpu_model,
                "cpu_cores": self.system_info.cpu_cores,
                "cpu_threads": self.system_info.cpu_threads,
                "memory_gb": self.system_info.memory_gb,
                "python_version": self.system_info.python_version,
                "numpy_version": self.system_info.numpy_version,
                "platform": self.system_info.platform,
            },
            "benchmark_results": [],
            "summary": {
                "total_tests": len(self.results),
                "disclaimer": "All measurements are Python simulation performance, not hardware throughput",
            },
        }

        for result in self.results:
            report["benchmark_results"].append(
                {
                    "test_name": result.test_name,
                    "frames_per_second": result.frames_per_second,
                    "processing_time_ms": result.processing_time_ms,
                    "memory_usage_mb": result.memory_usage_mb,
                    "cpu_usage_percent": result.cpu_usage_percent,
                    "configuration": result.configuration,
                    "notes": result.notes,
                }
            )

        return report


class BenchmarkSuite:
    """Comprehensive benchmark suite for the simulation framework."""

    def __init__(self):
        """Initialize the benchmark suite."""
        self.profiler = PerformanceProfiler()

    def run_all_benchmarks(self) -> dict:
        """
        Run all performance benchmarks.

        Returns:
            Complete benchmark report
        """
        logger.info("Starting comprehensive benchmark suite...")

        # MIPI simulation benchmark
        mipi_config = MIPIConfig(lanes=4, data_rate=2.5, channel=0)
        data_sizes = [1024, 4096, 16384, 65536]  # 1KB to 64KB
        self.profiler.benchmark_mipi_simulation(mipi_config, data_sizes, iterations=50)

        # Signal processing benchmark
        resolutions = [(480, 640, 3), (720, 1280, 3), (1080, 1920, 3)]  # VGA  # HD  # Full HD
        bit_depths = [8, 12, 16]
        self.profiler.benchmark_signal_processing(resolutions, bit_depths, iterations=5)

        # Power modeling benchmark
        self.profiler.benchmark_power_modeling(iterations=1000)

        logger.info("Benchmark suite completed")
        return self.profiler.generate_report()


def run_performance_profile(output_file: Optional[str] = None) -> dict:
    """
    Run performance profiling and optionally save results.

    Args:
        output_file: Optional file to save results

    Returns:
        Performance report dictionary
    """
    suite = BenchmarkSuite()
    report = suite.run_all_benchmarks()

    if output_file:
        import json

        with open(output_file, "w") as f:
            json.dump(report, f, indent=2)
        logger.info(f"Performance report saved to {output_file}")

    return report


# Example usage and CLI
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run performance benchmarks")
    parser.add_argument("--output", "-o", help="Output file for results")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.INFO)

    print("Running Advanced Image Sensor Interface Performance Benchmarks")
    print("=" * 60)
    print("IMPORTANT: These are Python simulation performance metrics,")
    print("not hardware throughput measurements.")
    print("=" * 60)

    report = run_performance_profile(args.output)

    # Print summary
    print(f"\nSystem: {report['system_info']['cpu_model']}")
    print(f"CPU: {report['system_info']['cpu_cores']} cores, {report['system_info']['cpu_threads']} threads")
    print(f"Memory: {report['system_info']['memory_gb']:.1f} GB")
    print(f"Python: {report['system_info']['python_version']}")
    print(f"NumPy: {report['system_info']['numpy_version']}")

    print("\nBenchmark Results:")
    print("-" * 40)
    for result in report["benchmark_results"]:
        print(f"{result['test_name']}: {result['frames_per_second']:.1f} ops/sec")
        print(f"  Processing time: {result['processing_time_ms']:.2f} ms")
        print(f"  Memory usage: {result['memory_usage_mb']:.1f} MB")
        print(f"  Notes: {result['notes']}")
        print()

    print("Benchmark completed successfully!")
