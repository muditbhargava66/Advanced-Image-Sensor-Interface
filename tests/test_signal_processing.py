"""
Unit Tests for Signal Processing

This module contains unit tests for the signal processing functionality
in the Advanced Image Sensor Interface project.

Classes:
    TestSignalProcessor: Test cases for the SignalProcessor class.

Usage:
    Run these tests using pytest:
    $ pytest tests/test_signal_processing.py
"""

import pytest
import time
import numpy as np
from unittest.mock import patch

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.sensor_interface.signal_processing import SignalProcessor, SignalConfig

@pytest.fixture
def signal_processor():
    """Fixture to create a SignalProcessor instance for testing."""
    config = SignalConfig(bit_depth=12, noise_reduction_strength=0.1, color_correction_matrix=np.eye(3))
    return SignalProcessor(config)

class TestSignalProcessor:
    """Test cases for the SignalProcessor class."""

    def test_initialization(self, signal_processor):
        """Test the initialization of SignalProcessor."""
        assert signal_processor.config.bit_depth == 12
        assert signal_processor.config.noise_reduction_strength == 0.1
        np.testing.assert_array_equal(signal_processor.config.color_correction_matrix, np.eye(3))

    def test_process_frame(self, signal_processor):
        """Test frame processing."""
        test_frame = np.random.randint(0, 4096, size=(1080, 1920), dtype=np.uint16)
        processed_frame = signal_processor.process_frame(test_frame)
        assert processed_frame.shape == test_frame.shape
        assert processed_frame.dtype == test_frame.dtype
        assert np.max(processed_frame) <= 4095  # Ensure we don't exceed 12-bit range

    def test_apply_noise_reduction(self, signal_processor):
        """Test noise reduction application."""
        test_frame = np.random.randint(0, 4096, size=(1080, 1920), dtype=np.uint16)
        noise_reduced_frame = signal_processor._apply_noise_reduction(test_frame)
        assert noise_reduced_frame.shape == test_frame.shape
        assert np.std(noise_reduced_frame) <= np.std(test_frame)  # Noise reduction should decrease standard deviation

    def test_apply_dynamic_range_expansion(self, signal_processor):
        """Test dynamic range expansion."""
        test_frame = np.random.randint(1000, 3000, size=(1080, 1920), dtype=np.uint16)
        expanded_frame = signal_processor._apply_dynamic_range_expansion(test_frame)
        assert expanded_frame.shape == test_frame.shape
        assert np.min(expanded_frame) < np.min(test_frame)
        assert np.max(expanded_frame) > np.max(test_frame)

    def test_apply_color_correction(self, signal_processor):
        """Test color correction application."""
        test_frame = np.random.randint(0, 4096, size=(1080, 1920, 3), dtype=np.uint16)
        corrected_frame = signal_processor._apply_color_correction(test_frame)
        assert corrected_frame.shape == test_frame.shape
        assert np.max(corrected_frame) <= 4095  # Ensure we don't exceed 12-bit range

    def test_optimize_performance(self, signal_processor):
        """Test performance optimization."""
        initial_processing_time = signal_processor._processing_time
        signal_processor.optimize_performance()
        assert signal_processor._processing_time < initial_processing_time

    @pytest.mark.parametrize("bit_depth", [8, 10, 12, 14, 16])
    def test_different_bit_depths(self, bit_depth):
        """Test processing with different bit depths."""
        config = SignalConfig(bit_depth=bit_depth, noise_reduction_strength=0.1, color_correction_matrix=np.eye(3))
        processor = SignalProcessor(config)
        test_frame = np.random.randint(0, 2**bit_depth, size=(1080, 1920), dtype=np.uint16)
        processed_frame = processor.process_frame(test_frame)
        assert np.max(processed_frame) <= 2**bit_depth - 1

    def test_color_correction_matrix(self):
        """Test with a non-identity color correction matrix."""
        ccm = np.array([[1.2, -0.1, -0.1],
                        [-0.1, 1.2, -0.1],
                        [-0.1, -0.1, 1.2]])
        config = SignalConfig(bit_depth=12, noise_reduction_strength=0.1, color_correction_matrix=ccm)
        processor = SignalProcessor(config)
        test_frame = np.random.randint(0, 4096, size=(1080, 1920, 3), dtype=np.uint16)
        corrected_frame = processor._apply_color_correction(test_frame)
        assert np.any(corrected_frame != test_frame)  # Ensure the frame has changed

    def test_processing_pipeline_order(self, signal_processor):
        """Test that the processing pipeline is applied in the correct order."""
        test_frame = np.random.randint(0, 4096, size=(1080, 1920, 3), dtype=np.uint16)
        
        with patch.object(signal_processor, '_apply_noise_reduction') as mock_noise_reduction, \
             patch.object(signal_processor, '_apply_dynamic_range_expansion') as mock_dre, \
             patch.object(signal_processor, '_apply_color_correction') as mock_color_correction:
            
            signal_processor.process_frame(test_frame)
            
            mock_noise_reduction.assert_called_once()
            mock_dre.assert_called_once()
            mock_color_correction.assert_called_once()
            
            # Check the order of calls
            mock_noise_reduction.assert_called_before(mock_dre)
            mock_dre.assert_called_before(mock_color_correction)

    def test_error_handling(self, signal_processor):
        """Test error handling for invalid inputs."""
        with pytest.raises(ValueError):
            signal_processor.process_frame("invalid input")
        
        with pytest.raises(ValueError):
            signal_processor.process_frame(np.random.rand(100, 100, 5))  # Invalid number of channels

    def test_performance_improvement(self, signal_processor):
        """Test that performance optimization leads to faster processing times."""
        test_frame = np.random.randint(0, 4096, size=(1080, 1920, 3), dtype=np.uint16)
        
        # Measure initial processing time
        start_time = time.time()
        signal_processor.process_frame(test_frame)
        initial_time = time.time() - start_time
        
        # Optimize performance
        signal_processor.optimize_performance()
        
        # Measure optimized processing time
        start_time = time.time()
        signal_processor.process_frame(test_frame)
        optimized_time = time.time() - start_time
        
        assert optimized_time < initial_time

if __name__ == "__main__":
    pytest.main([__file__])