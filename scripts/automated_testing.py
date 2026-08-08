"""
Automated Testing Suite for Advanced Image Sensor Interface

This script provides a comprehensive automated testing suite for the
Advanced Image Sensor Interface project, including unit tests, integration tests,
and performance benchmarks.

Usage:
    python automated_testing.py [options]

Options:
    --unit-tests              Run unit tests
    --integration-tests       Run integration tests
    --benchmarks              Run performance benchmarks
    --output OUTPUT           Output file for test results (default: test_results.json)

Example:
    python automated_testing.py --unit-tests --integration-tests --benchmarks
"""

import argparse
import json
import os
import sys
import time
import unittest
from typing import Any

import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from advanced_image_sensor_interface.sensor_interface.protocol.mipi.driver import MIPIConfig, MIPIProtocolDriver
from advanced_image_sensor_interface.sensor_interface.power_management import PowerConfig, PowerManager
from advanced_image_sensor_interface.sensor_interface.signal_processing import SignalConfig, SignalProcessor


class TestMIPIDriver(unittest.TestCase):
    """Tests for MIPI CSI-2 protocol driver."""

    def setUp(self) -> None:
        self.config = MIPIConfig(lanes=4, data_rate_mbps=2500.0, pixel_format="RAW10", resolution=(1920, 1080))
        self.driver = MIPIProtocolDriver(self.config)

    def test_initialization(self) -> None:
        self.assertEqual(self.driver.mipi_config.lanes, 4)
        self.assertEqual(self.driver.mipi_config.data_rate_mbps, 2500.0)

    def test_connect_disconnect(self) -> None:
        self.assertTrue(self.driver.connect())
        self.assertTrue(self.driver.is_connected)
        self.assertTrue(self.driver.disconnect())
        self.assertFalse(self.driver.is_connected)

    def test_streaming_lifecycle(self) -> None:
        self.driver.connect()
        self.assertTrue(self.driver.start_streaming())
        self.assertTrue(self.driver.is_streaming)
        self.assertTrue(self.driver.stop_streaming())
        self.assertFalse(self.driver.is_streaming)
        self.driver.disconnect()

    def test_frame_capture(self) -> None:
        self.driver.connect()
        self.driver.start_streaming()
        frame = self.driver.capture_frame()
        self.assertIsNotNone(frame)
        self.driver.stop_streaming()
        self.driver.disconnect()

    def test_get_status(self) -> None:
        self.driver.connect()
        stats = self.driver.get_statistics()
        self.assertIn("frames_captured", stats)
        self.driver.disconnect()


class TestSignalProcessor(unittest.TestCase):
    """Tests for signal processing pipeline."""

    def setUp(self) -> None:
        self.config = SignalConfig(bit_depth=12, noise_reduction_strength=0.1, color_correction_matrix=np.eye(3))
        self.processor = SignalProcessor(self.config)

    def test_process_frame(self) -> None:
        test_frame = np.random.randint(0, 4096, (1080, 1920), dtype=np.uint16)
        processed_frame = self.processor.process_frame(test_frame)
        self.assertEqual(processed_frame.shape, test_frame.shape)
        self.assertEqual(processed_frame.dtype, test_frame.dtype)

    def test_noise_reduction(self) -> None:
        noisy_frame = np.random.randint(0, 4096, (1080, 1920), dtype=np.uint16)
        processed_frame = self.processor.process_frame(noisy_frame)
        self.assertLessEqual(np.std(processed_frame), np.std(noisy_frame) * 1.1)


class TestPowerManager(unittest.TestCase):
    """Tests for power management system."""

    def setUp(self) -> None:
        self.config = PowerConfig(voltage_main=1.8, voltage_io=3.3, current_limit=1.0)
        self.manager = PowerManager(self.config)

    def test_set_voltage(self) -> None:
        self.assertTrue(self.manager.set_voltage("main", 1.5))
        self.assertEqual(self.manager.config.voltage_main, 1.5)

    def test_get_power_status(self) -> None:
        status = self.manager.get_power_status()
        self.assertIn("voltage_main", status)
        self.assertIn("voltage_io", status)
        self.assertIn("power_consumption", status)


class IntegrationTests(unittest.TestCase):
    """End-to-end integration tests."""

    def setUp(self) -> None:
        self.mipi_config = MIPIConfig(lanes=4, data_rate_mbps=2500.0, pixel_format="RAW10", resolution=(1920, 1080))
        self.mipi_driver = MIPIProtocolDriver(self.mipi_config)

        self.signal_config = SignalConfig(bit_depth=12, noise_reduction_strength=0.1, color_correction_matrix=np.eye(3))
        self.signal_processor = SignalProcessor(self.signal_config)

        self.power_config = PowerConfig(voltage_main=1.8, voltage_io=3.3, current_limit=1.0)
        self.power_manager = PowerManager(self.power_config)

    def test_end_to_end_processing(self) -> None:
        # Connect and capture
        self.mipi_driver.connect()
        self.mipi_driver.start_streaming()

        frame = self.mipi_driver.capture_frame()
        self.assertIsNotNone(frame)

        # Convert to numpy array for processing
        frame_array = np.frombuffer(frame, dtype=np.uint16)
        if frame_array.size >= 1920 * 1080:
            frame_array = frame_array[: 1920 * 1080].reshape((1080, 1920))
            processed_frame = self.signal_processor.process_frame(frame_array)
            self.assertEqual(processed_frame.shape, frame_array.shape)

        # Check power
        power_status = self.power_manager.get_power_status()
        self.assertGreaterEqual(power_status["power_consumption"], 0)

        self.mipi_driver.stop_streaming()
        self.mipi_driver.disconnect()


def run_performance_benchmarks() -> dict[str, Any]:
    """Run performance benchmarks for the entire system."""
    mipi_config = MIPIConfig(lanes=4, data_rate_mbps=2500.0, pixel_format="RAW10", resolution=(1920, 1080))
    mipi_driver = MIPIProtocolDriver(mipi_config)

    signal_config = SignalConfig(bit_depth=12, noise_reduction_strength=0.1, color_correction_matrix=np.eye(3))
    signal_processor = SignalProcessor(signal_config)

    power_config = PowerConfig(voltage_main=1.8, voltage_io=3.3, current_limit=1.0)
    power_manager = PowerManager(power_config)

    results: dict[str, Any] = {}

    # Benchmark MIPI connection and streaming
    mipi_driver.connect()
    mipi_driver.start_streaming()

    start_time = time.time()
    frames_captured = 0
    for _ in range(10):
        frame = mipi_driver.capture_frame()
        if frame:
            frames_captured += 1
    capture_time = time.time() - start_time

    results["frame_capture_rate"] = frames_captured / capture_time if capture_time > 0 else 0
    results["frames_captured"] = frames_captured

    # Benchmark signal processing
    test_frame = np.random.randint(0, 4096, (1080, 1920), dtype=np.uint16)
    start_time = time.time()
    for _ in range(5):
        signal_processor.process_frame(test_frame)
    processing_time = time.time() - start_time

    results["frame_processing_time"] = processing_time / 5
    results["frame_processing_rate"] = 5 / processing_time if processing_time > 0 else 0

    # Power efficiency
    power_status = power_manager.get_power_status()
    results["power_consumption"] = power_status["power_consumption"]

    mipi_driver.stop_streaming()
    mipi_driver.disconnect()

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Automated Testing Suite for Advanced Image Sensor Interface")
    parser.add_argument("--unit-tests", action="store_true", help="Run unit tests")
    parser.add_argument("--integration-tests", action="store_true", help="Run integration tests")
    parser.add_argument("--benchmarks", action="store_true", help="Run performance benchmarks")
    parser.add_argument("--output", default="test_results.json", help="Output file for test results")

    args = parser.parse_args()

    results: dict[str, Any] = {}

    if args.unit_tests:
        print("Running unit tests...")
        unit_suite = unittest.TestSuite()
        unit_suite.addTest(unittest.makeSuite(TestMIPIDriver))
        unit_suite.addTest(unittest.makeSuite(TestSignalProcessor))
        unit_suite.addTest(unittest.makeSuite(TestPowerManager))

        runner = unittest.TextTestRunner(verbosity=2)
        unit_result = runner.run(unit_suite)
        results["unit_tests"] = {
            "total": unit_result.testsRun,
            "failures": len(unit_result.failures),
            "errors": len(unit_result.errors),
        }

    if args.integration_tests:
        print("Running integration tests...")
        integration_suite = unittest.TestSuite()
        integration_suite.addTest(unittest.makeSuite(IntegrationTests))

        runner = unittest.TextTestRunner(verbosity=2)
        integration_result = runner.run(integration_suite)
        results["integration_tests"] = {
            "total": integration_result.testsRun,
            "failures": len(integration_result.failures),
            "errors": len(integration_result.errors),
        }

    if args.benchmarks:
        print("Running performance benchmarks...")
        benchmark_results = run_performance_benchmarks()
        results["benchmarks"] = benchmark_results

        print("Benchmark Results:")
        for key, value in benchmark_results.items():
            print(f"  {key}: {value}")

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Test results saved to {args.output}")


if __name__ == "__main__":
    main()
