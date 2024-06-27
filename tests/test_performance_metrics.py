"""
Unit Tests for Performance Metrics

This module contains unit tests for the performance metrics calculations
in the Advanced Image Sensor Interface project.

Functions tested:
    calculate_snr: Signal-to-Noise Ratio calculation
    calculate_dynamic_range: Dynamic Range calculation
    calculate_color_accuracy: Color Accuracy calculation

Usage:
    Run these tests using pytest:
    $ pytest tests/test_performance_metrics.py
"""

import pytest
import numpy as np

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.utils.performance_metrics import calculate_snr, calculate_dynamic_range, calculate_color_accuracy

class TestPerformanceMetrics:
    """Test cases for performance metrics calculations."""

    def test_calculate_snr(self):
        """Test SNR calculation."""
        signal = np.ones((100, 100)) * 100
        noise = np.random.normal(0, 10, (100, 100))
        snr = calculate_snr(signal, noise)
        assert 20 <= snr <= 30  # Expected SNR range for given signal and noise

    def test_calculate_snr_zero_noise(self):
        """Test SNR calculation with zero noise."""
        signal = np.ones((100, 100)) * 100
        noise = np.zeros((100, 100))
        snr = calculate_snr(signal, noise)
        assert np.isinf(snr)

    def test_calculate_snr_error(self):
        """Test SNR calculation error handling."""
        signal = np.ones((100, 100))
        noise = np.ones((50, 50))
        with pytest.raises(ValueError):
            calculate_snr(signal, noise)

    def test_calculate_dynamic_range(self):
        """Test dynamic range calculation."""
        image = np.array([1, 10, 100, 1000], dtype=np.uint16)
        dr = calculate_dynamic_range(image)
        assert 59 <= dr <= 61  # Expected dynamic range for given image

    def test_calculate_dynamic_range_zero_min(self):
        """Test dynamic range calculation with zero minimum value."""
        image = np.array([0, 10, 100, 1000], dtype=np.uint16)
        dr = calculate_dynamic_range(image)
        assert dr == 0

    def test_calculate_color_accuracy(self):
        """Test color accuracy calculation."""
        reference_colors = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]], dtype=np.uint8)
        measured_colors = np.array([[250, 10, 10], [10, 250, 10], [10, 10, 250]], dtype=np.uint8)
        mean_delta_e, delta_e_values = calculate_color_accuracy(reference_colors, measured_colors)
        assert 0 < mean_delta_e < 10  # Expected range for given color differences
        assert len(delta_e_values) == 3

    def test_calculate_color_accuracy_error(self):
        """Test color accuracy calculation error handling."""
        reference_colors = np.array([[255, 0, 0], [0, 255, 0]], dtype=np.uint8)
        measured_colors = np.array([[250, 10, 10], [10, 250, 10], [10, 10, 250]], dtype=np.uint8)
        with pytest.raises(ValueError):
            calculate_color_accuracy(reference_colors, measured_colors)

    @pytest.mark.parametrize("signal,noise,expected_snr", [
        (np.ones((10, 10)) * 100, np.ones((10, 10)) * 10, 20),
        (np.ones((10, 10)) * 100, np.ones((10, 10)) * 1, 40),
        (np.ones((10, 10)) * 10, np.ones((10, 10)) * 1, 20)
    ])
    def test_snr_parametrized(self, signal, noise, expected_snr):
        """Parametrized test for SNR calculation with different signal and noise levels."""
        calculated_snr = calculate_snr(signal, noise)
        assert np.isclose(calculated_snr, expected_snr, atol=0.1)

    @pytest.mark.parametrize("image,expected_dr", [
        (np.array([1, 10, 100, 1000], dtype=np.uint16), 60),
        (np.array([10, 100, 1000], dtype=np.uint16), 40),
        (np.array([1, 2, 4, 8, 16], dtype=np.uint16), 24)
    ])
    def test_dynamic_range_parametrized(self, image, expected_dr):
        """Parametrized test for dynamic range calculation with different image data."""
        calculated_dr = calculate_dynamic_range(image)
        assert np.isclose(calculated_dr, expected_dr, atol=0.1)

    def test_snr_with_random_data(self):
        """Test SNR calculation with random data."""
        for _ in range(10):  # Run 10 random tests
            signal = np.random.uniform(50, 200, (100, 100))
            noise = np.random.normal(0, 10, (100, 100))
            snr = calculate_snr(signal, noise)
            theoretical_snr = 20 * np.log10(np.mean(signal) / np.std(noise))
            assert np.isclose(snr, theoretical_snr, rtol=0.1)

    def test_dynamic_range_with_random_data(self):
        """Test dynamic range calculation with random data."""
        for _ in range(10):  # Run 10 random tests
            image = np.random.randint(1, 1000, 1000, dtype=np.uint16)
            dr = calculate_dynamic_range(image)
            theoretical_dr = 20 * np.log10(np.max(image) / np.min(image))
            assert np.isclose(dr, theoretical_dr, rtol=0.1)

    def test_color_accuracy_perfect_match(self):
        """Test color accuracy calculation with perfectly matching colors."""
        colors = np.random.randint(0, 256, (10, 3), dtype=np.uint8)
        mean_delta_e, delta_e_values = calculate_color_accuracy(colors, colors)
        assert mean_delta_e == 0
        assert np.all(delta_e_values == 0)

    def test_performance_metrics_integration(self):
        """Integration test for all performance metrics."""
        # Generate test image data
        signal = np.random.uniform(50, 200, (100, 100, 3)).astype(np.uint8)
        noise = np.random.normal(0, 10, (100, 100, 3)).astype(np.uint8)
        noisy_signal = np.clip(signal + noise, 0, 255).astype(np.uint8)

        # Calculate all metrics
        snr = calculate_snr(signal, noisy_signal - signal)
        dr = calculate_dynamic_range(signal)
        mean_delta_e, _ = calculate_color_accuracy(signal, noisy_signal)

        # Assert that all metrics are within reasonable ranges
        assert 10 <= snr <= 50
        assert 20 <= dr <= 80
        assert 0 <= mean_delta_e <= 20

if __name__ == "__main__":
    pytest.main([__file__])