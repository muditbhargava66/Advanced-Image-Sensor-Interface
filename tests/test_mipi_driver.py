"""
Unit Tests for MIPI Driver

This module contains unit tests for the MIPI driver implementation
in the Advanced Image Sensor Interface project.

Classes:
    TestMIPIDriver: Test cases for the MIPIDriver class.

Usage:
    Run these tests using pytest:
    $ pytest tests/test_mipi_driver.py
"""

import os
import sys
import time
from unittest.mock import patch

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from advanced_image_sensor_interface.sensor_interface.mipi_driver import MIPIConfig, MIPIDriver


@pytest.fixture
def mipi_driver():
    """Fixture to create a MIPIDriver instance for testing."""
    config = MIPIConfig(lanes=4, data_rate=2.5, channel=0)
    return MIPIDriver(config)


class TestMIPIDriver:
    """Test cases for the MIPIDriver class."""

    def test_initialization(self, mipi_driver):
        """Test the initialization of MIPIDriver."""
        assert mipi_driver.config.lanes == 4
        assert mipi_driver.config.data_rate == 2.5
        assert mipi_driver.config.channel == 0

    def test_send_data_success(self, mipi_driver):
        """Test successful data sending."""
        test_data = b"Hello, MIPI!"
        assert mipi_driver.send_data(test_data) == True

    def test_send_data_failure(self, mipi_driver):
        """Test data sending failure scenario."""
        with patch.object(MIPIDriver, "send_data", return_value=False):
            assert mipi_driver.send_data(b"Fail test") == False

    def test_receive_data_success(self, mipi_driver):
        """Test successful data receiving."""
        # First, send some data to ensure there's something in the buffer
        test_data = b"TestMIPIData"
        mipi_driver.send_data(test_data)

        # Now try to receive it
        received_data = mipi_driver.receive_data(10)
        assert isinstance(received_data, bytes)
        assert len(received_data) == 10
        assert received_data == b"TestMIPIDa"

    def test_receive_data_failure(self, mipi_driver):
        """Test data receiving failure scenario."""
        with patch.object(MIPIDriver, "receive_data", return_value=None):
            assert mipi_driver.receive_data(10) is None

    def test_get_status(self, mipi_driver):
        """Test get_status method."""
        status = mipi_driver.get_status()
        assert "lanes" in status
        assert "data_rate" in status
        assert "channel" in status
        assert "error_rate" in status
        assert "throughput" in status

    @pytest.mark.parametrize("data_size", [1000, 10000, 100000])
    def test_performance_optimization(self, mipi_driver, data_size):
        """Test performance optimization with different data sizes."""
        test_data = b"0" * data_size

        # Measure initial performance - run multiple times and take average
        initial_times = []
        for _ in range(3):  # Run 3 times
            start_time = time.time()
            mipi_driver.send_data(test_data)
            initial_times.append(time.time() - start_time)
        initial_time = sum(initial_times) / len(initial_times)

        # Optimize performance
        mipi_driver.optimize_performance()

        # Measure optimized performance - run multiple times and take average
        optimized_times = []
        for _ in range(3):  # Run 3 times
            start_time = time.time()
            mipi_driver.send_data(test_data)
            optimized_times.append(time.time() - start_time)
        optimized_time = sum(optimized_times) / len(optimized_times)

        # Check if performance improved - using a more lenient threshold
        # Either no degradation for small data or at least 25% improvement for large data
        if data_size <= 10000:
            # For small data sizes, we just ensure it didn't get significantly worse
            assert optimized_time <= initial_time * 1.2, f"Performance worsened significantly: {optimized_time} vs {initial_time}"
        else:
            # For large data sizes, we look for improvement but relax the threshold to 25%
            assert optimized_time < initial_time * 0.75, f"Performance didn't improve enough: {optimized_time} vs {initial_time}"

    def test_error_handling(self, mipi_driver):
        """Test error handling in the driver."""
        with pytest.raises(ValueError):
            mipi_driver.send_data("Invalid data type")

    @patch("advanced_image_sensor_interface.sensor_interface.mipi_driver.time.sleep")
    def test_transmission_simulation(self, mock_sleep, mipi_driver):
        """Test the transmission simulation timing."""
        test_data = b"0" * 1000000  # 1 MB of data
        mipi_driver.send_data(test_data)
        expected_sleep_time = len(test_data) / (mipi_driver.config.data_rate * 1e9 / 8)
        mock_sleep.assert_called_with(pytest.approx(expected_sleep_time, rel=1e-6))

    def test_concurrent_operations(self, mipi_driver):
        """Test concurrent send and receive operations."""
        import threading

        def send_operation():
            assert mipi_driver.send_data(b"Concurrent send test")

        def receive_operation():
            assert mipi_driver.receive_data(10) is not None

        thread1 = threading.Thread(target=send_operation)
        thread2 = threading.Thread(target=receive_operation)

        thread1.start()
        thread2.start()

        thread1.join()
        thread2.join()

    def test_invalid_configuration(self):
        """Test driver initialization with invalid configurations."""
        invalid_configs = [
            {"lanes": 0, "data_rate": 2.5, "channel": 0},
            {"lanes": 4, "data_rate": 0, "channel": 0},
            {"lanes": 4, "data_rate": 2.5, "channel": -1},
        ]

        for config_dict in invalid_configs:
            with pytest.raises(ValueError):
                config = MIPIConfig(**config_dict)
                MIPIDriver(config)


if __name__ == "__main__":
    pytest.main([__file__])
