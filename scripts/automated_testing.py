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
    python automated_testing.py --unit-tests --integration-tests --benchmarks --output full_test_results.json
"""

import unittest
import json
import argparse
import time
import numpy as np
from typing import Dict, Any

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.sensor_interface.mipi_driver import MIPIDriver, MIPIConfig
from src.sensor_interface.signal_processing import SignalProcessor, SignalConfig
from src.sensor_interface.power_management import PowerManager, PowerConfig
from src.utils.performance_metrics import calculate_snr, calculate_dynamic_range, calculate_color_accuracy

class TestMIPIDriver(unittest.TestCase):
    def setUp(self):
        self.config = MIPIConfig(lanes=4, data_rate=2.5, channel=0)
        self.driver = MIPIDriver(self.config)
    
    def test_initialization(self):
        self.assertEqual(self.driver.config.lanes, 4)
        self.assertEqual(self.driver.config.data_rate, 2.5)
        self.assertEqual(self.driver.config.channel, 0)
    
    def test_send_receive_data(self):
        test_data = b'Hello, MIPI!'
        self.assertTrue(self.driver.send_data(test_data))
        received_data = self.driver.receive_data(len(test_data))
        self.assertEqual(received_data, test_data)
    
    def test_get_status(self):
        status = self.driver.get_status()
        self.assertIn('error_rate', status)
        self.assertIn('throughput', status)

class TestSignalProcessor(unittest.TestCase):
    def setUp(self):
        self.config = SignalConfig(bit_depth=12, noise_reduction_strength=0.1, color_correction_matrix=np.eye(3))
        self.processor = SignalProcessor(self.config)
    
    def test_process_frame(self):
        test_frame = np.random.randint(0, 4096, (1080, 1920), dtype=np.uint16)
        processed_frame = self.processor.process_frame(test_frame)
        self.assertEqual(processed_frame.shape, test_frame.shape)
        self.assertEqual(processed_frame.dtype, test_frame.dtype)
    
    def test_noise_reduction(self):
        noisy_frame = np.random.randint(0, 4096, (1080, 1920), dtype=np.uint16)
        processed_frame = self.processor.process_frame(noisy_frame)
        self.assertLess(np.std(processed_frame), np.std(noisy_frame))

class TestPowerManager(unittest.TestCase):
    def setUp(self):
        self.config = PowerConfig(voltage_main=1.8, voltage_io=3.3, current_limit=1.0)
        self.manager = PowerManager(self.config)
    
    def test_set_voltage(self):
        self.assertTrue(self.manager.set_voltage('main', 1.5))
        self.assertEqual(self.manager.config.voltage_main, 1.5)
    
    def test_get_power_status(self):
        status = self.manager.get_power_status()
        self.assertIn('voltage_main', status)
        self.assertIn('voltage_io', status)
        self.assertIn('power_consumption', status)

class IntegrationTests(unittest.TestCase):
    def setUp(self):
        self.mipi_config = MIPIConfig(lanes=4, data_rate=2.5, channel=0)
        self.mipi_driver = MIPIDriver(self.mipi_config)
        
        self.signal_config = SignalConfig(bit_depth=12, noise_reduction_strength=0.1, color_correction_matrix=np.eye(3))
        self.signal_processor = SignalProcessor(self.signal_config)
        
        self.power_config = PowerConfig(voltage_main=1.8, voltage_io=3.3, current_limit=1.0)
        self.power_manager = PowerManager(self.power_config)
    
    def test_end_to_end_processing(self):
        # Generate test frame
        test_frame = np.random.randint(0, 4096, (1080, 1920), dtype=np.uint16)
        
        # Simulate MIPI transfer
        self.mipi_driver.send_data(test_frame.tobytes())
        received_data = self.mipi_driver.receive_data(test_frame.nbytes)
        received_frame = np.frombuffer(received_data, dtype=np.uint16).reshape(test_frame.shape)
        
        # Process frame
        processed_frame = self.signal_processor.process_frame(received_frame)
        
        # Check results
        self.assertEqual(processed_frame.shape, test_frame.shape)
        self.assertLess(np.std(processed_frame), np.std(test_frame))
        
        # Check power consumption
        power_status = self.power_manager.get_power_status()
        self.assertGreater(power_status['power_consumption'], 0)

def run_performance_benchmarks() -> Dict[str, Any]:
    """Run performance benchmarks for the entire system."""
    mipi_config = MIPIConfig(lanes=4, data_rate=2.5, channel=0)
    mipi_driver = MIPIDriver(mipi_config)
    
    signal_config = SignalConfig(bit_depth=12, noise_reduction_strength=0.1, color_correction_matrix=np.eye(3))
    signal_processor = SignalProcessor(signal_config)
    
    power_config = PowerConfig(voltage_main=1.8, voltage_io=3.3, current_limit=1.0)
    power_manager = PowerManager(power_config)
    
    results = {}
    
    # Benchmark MIPI transfer speed
    test_data = b'0' * (1920 * 1080 * 2)  # 2 bytes per pixel for 12-bit depth
    start_time = time.time()
    mipi_driver.send_data(test_data)
    mipi_time = time.time() - start_time
    results['mipi_transfer_rate'] = len(test_data) / mipi_time / 1e6  # MB/s
    
    # Benchmark signal processing speed
    test_frame = np.random.randint(0, 4096, (1080, 1920), dtype=np.uint16)
    start_time = time.time()
    signal_processor.process_frame(test_frame)
    processing_time = time.time() - start_time
    results['frame_processing_time'] = processing_time
    results['frame_processing_rate'] = 1 / processing_time  # FPS
    
    # Benchmark power efficiency
    power_status = power_manager.get_power_status()
    results['power_consumption'] = power_status['power_consumption']
    results['power_efficiency'] = len(test_data) / power_status['power_consumption'] / mipi_time  # MB/J
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Automated Testing Suite for Advanced Image Sensor Interface")
    parser.add_argument('--unit-tests', action='store_true', help='Run unit tests')
    parser.add_argument('--integration-tests', action='store_true', help='Run integration tests')
    parser.add_argument('--benchmarks', action='store_true', help='Run performance benchmarks')
    parser.add_argument('--output', default='test_results.json', help='Output file for test results')
    
    args = parser.parse_args()
    
    results = {}
    
    if args.unit_tests:
        print("Running unit tests...")
        unit_suite = unittest.TestSuite()
        unit_suite.addTest(unittest.makeSuite(TestMIPIDriver))
        unit_suite.addTest(unittest.makeSuite(TestSignalProcessor))
        unit_suite.addTest(unittest.makeSuite(TestPowerManager))
        
        runner = unittest.TextTestRunner(verbosity=2)
        unit_result = runner.run(unit_suite)
        results['unit_tests'] = {
            'total': unit_result.testsRun,
            'failures': len(unit_result.failures),
            'errors': len(unit_result.errors)
        }
    
    if args.integration_tests:
        print("Running integration tests...")
        integration_suite = unittest.TestSuite()
        integration_suite.addTest(unittest.makeSuite(IntegrationTests))
        
        runner = unittest.TextTestRunner(verbosity=2)
        integration_result = runner.run(integration_suite)
        results['integration_tests'] = {
            'total': integration_result.testsRun,
            'failures': len(integration_result.failures),
            'errors': len(integration_result.errors)
        }
    
    if args.benchmarks:
        print("Running performance benchmarks...")
        benchmark_results = run_performance_benchmarks()
        results['benchmarks'] = benchmark_results
        
        print("Benchmark Results:")
        for key, value in benchmark_results.items():
            print(f"{key}: {value}")
    
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Test results saved to {args.output}")

if __name__ == "__main__":
    main()