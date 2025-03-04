"""
Speed Tests for Advanced Image Sensor Interface

This module contains benchmark tests for measuring the processing speed
and throughput of various components in the Advanced Image Sensor Interface project.

Functions:
    benchmark_mipi_driver: Measure MIPI driver data transfer speeds.
    benchmark_signal_processing: Measure signal processing pipeline speed.
    benchmark_power_management: Measure power management operations speed.
    run_speed_benchmarks: Run all speed benchmarks and report results.

Usage:
    from benchmarks.speed_tests import run_speed_benchmarks
    run_speed_benchmarks()
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import json
import time
from typing import Any

import numpy as np

from src.sensor_interface.mipi_driver import MIPIConfig, MIPIDriver
from src.sensor_interface.power_management import PowerConfig, PowerManager
from src.sensor_interface.signal_processing import SignalConfig, SignalProcessor


def benchmark_mipi_driver(data_sizes: list[int] = [1024, 1024*1024, 10*1024*1024]) -> dict[str, float]:
    """
    Benchmark MIPI driver data transfer speeds.

    Args:
    ----
        data_sizes (list[int]): List of data sizes to test in bytes.

    Returns:
    -------
        Dict[str, float]: Dictionary of data sizes and their corresponding transfer speeds in MB/s.

    """
    config = MIPIConfig(lanes=4, data_rate=2.5, channel=0)
    driver = MIPIDriver(config)
    results = {}

    for size in data_sizes:
        data = b'0' * size
        start_time = time.time()
        driver.send_data(data)
        end_time = time.time()
        speed = size / (end_time - start_time) / 1024 / 1024  # MB/s
        results[f"{size/1024/1024:.2f}MB"] = speed

    return results

def benchmark_signal_processing(image_sizes: list[tuple[int, int]] = [(1920, 1080), (3840, 2160), (7680, 4320)]) -> dict[str, float]:
    """
    Benchmark signal processing pipeline speed.

    Args:
    ----
        image_sizes (list[tuple[int, int]]): List of image sizes to test.

    Returns:
    -------
        Dict[str, float]: Dictionary of image sizes and their corresponding processing speeds in FPS.

    """
    config = SignalConfig(bit_depth=12, noise_reduction_strength=0.1, color_correction_matrix=np.eye(3))
    processor = SignalProcessor(config)
    results = {}

    for width, height in image_sizes:
        image = np.random.randint(0, 4096, (height, width, 3), dtype=np.uint16)
        start_time = time.time()
        for _ in range(10):  # Process 10 frames for more accurate measurement
            processor.process_frame(image)
        end_time = time.time()
        fps = 10 / (end_time - start_time)
        results[f"{width}x{height}"] = fps

    return results

def benchmark_power_management(num_operations: int = 1000) -> dict[str, float]:
    """
    Benchmark power management operations speed.

    Args:
    ----
        num_operations (int): Number of power management operations to perform.

    Returns:
    -------
        Dict[str, float]: Dictionary of operation types and their corresponding speeds in operations/second.

    """
    config = PowerConfig(voltage_main=1.8, voltage_io=3.3, current_limit=1.0)
    power_manager = PowerManager(config)
    results = {}

    # Benchmark voltage setting
    start_time = time.time()
    for _ in range(num_operations):
        power_manager.set_voltage('main', 1.5)
    end_time = time.time()
    results['voltage_set_ops_per_second'] = num_operations / (end_time - start_time)

    # Benchmark power status retrieval
    start_time = time.time()
    for _ in range(num_operations):
        power_manager.get_power_status()
    end_time = time.time()
    results['power_status_ops_per_second'] = num_operations / (end_time - start_time)

    return results

def run_speed_benchmarks() -> dict[str, Any]:
    """
    Run all speed benchmarks and compile results.

    Returns
    -------
        Dict[str, Any]: Dictionary containing all benchmark results.

    """
    results = {}
    results['mipi_driver'] = benchmark_mipi_driver()
    results['signal_processing'] = benchmark_signal_processing()
    results['power_management'] = benchmark_power_management()
    return results

if __name__ == "__main__":
    benchmark_results = run_speed_benchmarks()
    print("Speed Benchmark Results:")
    print(json.dumps(benchmark_results, indent=2))

    # Calculate and print overall performance improvements
    mipi_speeds = list(benchmark_results['mipi_driver'].values())
    mipi_improvement = (mipi_speeds[-1] - mipi_speeds[0]) / mipi_speeds[0] * 100

    signal_processing_speeds = list(benchmark_results['signal_processing'].values())
    signal_improvement = (signal_processing_speeds[-1] - signal_processing_speeds[0]) / signal_processing_speeds[0] * 100

    print(f"\nOverall MIPI Driver speed improvement: {mipi_improvement:.2f}%")
    print(f"Overall Signal Processing speed improvement: {signal_improvement:.2f}%")
