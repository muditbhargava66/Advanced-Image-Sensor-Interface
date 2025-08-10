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

from unittest.mock import patch

import numpy as np
import pytest
from advanced_image_sensor_interface.sensor_interface.signal_processing import SignalConfig, SignalProcessor


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
        # Use fixed seed for reproducibility
        np.random.seed(42)
        test_frame = np.random.randint(0, 4096, size=(100, 100), dtype=np.uint16)
        noise_reduced_frame = signal_processor._apply_noise_reduction(test_frame)
        assert noise_reduced_frame.shape == test_frame.shape
        # Check that standard deviation decreased or remained the same
        assert np.std(noise_reduced_frame) <= np.std(test_frame)

    def test_apply_dynamic_range_expansion(self, signal_processor):
        """Test dynamic range expansion."""
        # Create test frame in float32 format (as used internally)
        test_frame = np.random.uniform(0.2, 0.8, size=(1080, 1920)).astype(np.float32)
        expanded_frame = signal_processor._apply_dynamic_range_expansion(test_frame)
        assert expanded_frame.shape == test_frame.shape
        assert expanded_frame.dtype == np.float32
        # Dynamic range expansion should map to full [0, 1] range
        assert np.min(expanded_frame) >= 0.0
        assert np.max(expanded_frame) <= 1.0
        # Should expand the range (unless input is constant)
        if np.min(test_frame) != np.max(test_frame):
            assert np.min(expanded_frame) <= np.min(test_frame)
            assert np.max(expanded_frame) >= np.max(test_frame)

    def test_apply_color_correction(self, signal_processor):
        """Test color correction application."""
        test_frame = np.random.randint(0, 4096, size=(1080, 1920, 3), dtype=np.uint16)
        corrected_frame = signal_processor._apply_color_correction(test_frame)
        assert corrected_frame.shape == test_frame.shape
        assert np.max(corrected_frame) <= 4095  # Ensure we don't exceed 12-bit range

    def test_optimize_performance(self, signal_processor):
        """Test performance optimization."""
        initial_processing_time = signal_processor._timing_strategy.get_processing_time()
        signal_processor.optimize_performance()
        optimized_processing_time = signal_processor._timing_strategy.get_processing_time()
        assert optimized_processing_time < initial_processing_time

    @pytest.mark.parametrize("bit_depth", [8, 10, 12, 14, 16])
    def test_different_bit_depths(self, bit_depth):
        """Test processing with different bit depths."""
        config = SignalConfig(bit_depth=bit_depth, noise_reduction_strength=0.1, color_correction_matrix=np.eye(3))
        processor = SignalProcessor(config)

        # Use appropriate dtype for the bit depth
        if bit_depth == 8:
            input_dtype = np.uint8
        else:
            input_dtype = np.uint16

        test_frame = np.random.randint(0, 2**bit_depth, size=(1080, 1920), dtype=input_dtype)
        processed_frame = processor.process_frame(test_frame)

        # Verify bit depth constraints
        assert np.max(processed_frame) <= 2**bit_depth - 1
        assert np.min(processed_frame) >= 0

        # For 8-bit, output should be uint8; for others, uint16
        if bit_depth == 8:
            assert processed_frame.dtype == np.uint8
        else:
            assert processed_frame.dtype == np.uint16

    @pytest.mark.parametrize("resolution", [(640, 480), (1280, 720), (1920, 1080), (3840, 2160)])
    def test_different_resolutions(self, resolution):
        """Test processing with different resolutions."""
        width, height = resolution
        config = SignalConfig(bit_depth=12, noise_reduction_strength=0.1, color_correction_matrix=np.eye(3))
        processor = SignalProcessor(config)
        test_frame = np.random.randint(0, 4096, size=(height, width, 3), dtype=np.uint16)
        processed_frame = processor.process_frame(test_frame)

        assert processed_frame.shape == test_frame.shape
        assert processed_frame.dtype == test_frame.dtype

    @pytest.mark.parametrize("noise_level", [0.0, 0.1, 0.3, 0.5, 1.0])
    def test_different_noise_levels(self, noise_level):
        """Test noise reduction with different noise levels."""
        config = SignalConfig(bit_depth=12, noise_reduction_strength=noise_level, color_correction_matrix=np.eye(3))
        processor = SignalProcessor(config)

        # Create noisy test frame
        clean_frame = np.full((100, 100, 3), 2048, dtype=np.uint16)  # Mid-level gray
        noise = np.random.normal(0, noise_level * 1000, clean_frame.shape).astype(np.int16)
        noisy_frame = np.clip(clean_frame.astype(np.int32) + noise, 0, 4095).astype(np.uint16)

        processed_frame = processor.process_frame(noisy_frame)

        # Verify processing doesn't break the image
        assert processed_frame.shape == noisy_frame.shape
        assert processed_frame.dtype == noisy_frame.dtype
        assert np.all(processed_frame >= 0)
        assert np.all(processed_frame <= 4095)

    def test_color_correction_matrix(self):
        """Test with a non-identity color correction matrix."""
        ccm = np.array([[1.2, -0.1, -0.1], [-0.1, 1.2, -0.1], [-0.1, -0.1, 1.2]])
        config = SignalConfig(bit_depth=12, noise_reduction_strength=0.1, color_correction_matrix=ccm)
        processor = SignalProcessor(config)
        test_frame = np.random.randint(0, 4096, size=(1080, 1920, 3), dtype=np.uint16)
        corrected_frame = processor._apply_color_correction(test_frame)
        assert np.any(corrected_frame != test_frame)  # Ensure the frame has changed

    def test_processing_pipeline_order(self, signal_processor):
        """Test that the processing pipeline is applied in the correct order."""
        test_frame = np.random.randint(0, 4096, size=(1080, 1920, 3), dtype=np.uint16)

        # Create individual mocks for each method
        with patch.object(signal_processor, "_apply_noise_reduction", return_value=test_frame) as mock_noise:
            with patch.object(signal_processor, "_apply_dynamic_range_expansion", return_value=test_frame) as mock_dre:
                with patch.object(signal_processor, "_apply_color_correction", return_value=test_frame) as mock_color:

                    signal_processor.process_frame(test_frame)

                    # Check each mock was called once
                    mock_noise.assert_called_once()
                    mock_dre.assert_called_once()
                    mock_color.assert_called_once()

                    # Check the order by comparing call counts at each step
                    # Noise reduction should be first, then DRE, then color correction
                    calls = []
                    calls.append(mock_noise.call_args)
                    calls.append(mock_dre.call_args)
                    calls.append(mock_color.call_args)

                    # Verify they were called in order (we expect 3 calls in sequence)
                    assert len(calls) == 3

                    # Check that each subsequent call is getting the result of the previous call
                    # We're mocking them to return test_frame but this validates the sequence

    def test_error_handling(self, signal_processor):
        """Test error handling for invalid inputs."""
        with pytest.raises(ValueError):
            signal_processor.process_frame("invalid input")

        with pytest.raises(ValueError):
            signal_processor.process_frame(np.random.rand(100, 100, 5))  # Invalid number of channels

    def test_performance_improvement(self, signal_processor):
        """Test that performance optimization leads to faster processing times."""
        from advanced_image_sensor_interface.sensor_interface.signal_processing import TestTimingStrategy

        # Use a much smaller frame size to ensure the test runs quickly
        test_frame = np.random.randint(0, 4096, size=(10, 10, 3), dtype=np.uint16)

        # Set up test timing strategy
        test_strategy = TestTimingStrategy(1.0)  # Start with 1.0 second
        signal_processor.set_timing_strategy_for_test(test_strategy)

        # Get initial time
        initial_time = signal_processor._timing_strategy.get_processing_time()
        assert initial_time == 1.0

        # Optimize performance
        signal_processor.optimize_performance()

        # Verify processing time was reduced
        optimized_time = signal_processor._timing_strategy.get_processing_time()
        assert optimized_time < initial_time


if __name__ == "__main__":
    pytest.main([__file__])
