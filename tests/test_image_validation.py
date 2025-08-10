"""
Unit Tests for Image Validation

This module contains unit tests for the image validation and processing
functionality in the Advanced Image Sensor Interface project.

Classes:
    TestImageValidator: Test cases for the ImageValidator class.
    TestBitDepthConverter: Test cases for the BitDepthConverter class.
    TestSafeImageProcessor: Test cases for the SafeImageProcessor class.

Usage:
    Run these tests using pytest:
    $ pytest tests/test_image_validation.py
"""

import numpy as np
import pytest
from advanced_image_sensor_interface.sensor_interface.image_validation import (
    BitDepthConverter,
    ImageFormat,
    ImageValidator,
    SafeImageProcessor,
    SupportedDType,
)


class TestImageValidator:
    """Test cases for the ImageValidator class."""

    def test_validate_mono_image(self):
        """Test validation of monochrome images."""
        validator = ImageValidator()

        # 8-bit mono
        img = np.random.randint(0, 256, (480, 640), dtype=np.uint8)
        fmt = validator.validate_image(img)
        assert fmt.height == 480
        assert fmt.width == 640
        assert fmt.channels == 1
        assert fmt.bit_depth == 8
        assert fmt.dtype == SupportedDType.UINT8

    def test_validate_rgb_image(self):
        """Test validation of RGB images."""
        validator = ImageValidator()

        # 12-bit RGB
        img = np.random.randint(0, 4096, (480, 640, 3), dtype=np.uint16)
        fmt = validator.validate_image(img)
        assert fmt.height == 480
        assert fmt.width == 640
        assert fmt.channels == 3
        assert fmt.bit_depth == 12
        assert fmt.dtype == SupportedDType.UINT12_IN_UINT16

    def test_validate_different_bit_depths(self):
        """Test validation of different bit depths."""
        validator = ImageValidator()

        test_cases = [
            (255, 8, SupportedDType.UINT8),
            (1023, 10, SupportedDType.UINT10_IN_UINT16),
            (4095, 12, SupportedDType.UINT12_IN_UINT16),
            (16383, 14, SupportedDType.UINT14_IN_UINT16),
            (65535, 16, SupportedDType.UINT16),
        ]

        for max_val, expected_depth, expected_dtype in test_cases:
            img = np.random.randint(0, max_val + 1, (100, 100), dtype=np.uint16)
            fmt = validator.validate_image(img)
            assert fmt.bit_depth == expected_depth
            assert fmt.dtype == expected_dtype

    def test_validate_invalid_images(self):
        """Test validation of invalid images."""
        validator = ImageValidator()

        # Empty array
        with pytest.raises(ValueError, match="Image cannot be empty"):
            validator.validate_image(np.array([]))

        # Wrong dimensions
        with pytest.raises(ValueError, match="Image must be 2D or 3D"):
            validator.validate_image(np.random.rand(10, 10, 10, 10))

        # Invalid channels
        with pytest.raises(ValueError, match="Invalid number of channels"):
            validator.validate_image(np.random.rand(10, 10, 5))

        # Non-array input
        with pytest.raises(ValueError, match="Image must be a numpy array"):
            validator.validate_image("not an array")

    def test_validate_color_correction_matrix(self):
        """Test color correction matrix validation."""
        validator = ImageValidator()

        # Valid 3x3 matrix for RGB
        matrix = np.eye(3)
        validator.validate_color_correction_matrix(matrix, 3)  # Should not raise

        # Invalid matrix shape for RGB
        with pytest.raises(ValueError, match="Color correction matrix must be 3x3"):
            validator.validate_color_correction_matrix(np.eye(2), 3)

        # Non-array matrix
        with pytest.raises(ValueError, match="Color correction matrix must be a numpy array"):
            validator.validate_color_correction_matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]], 3)


class TestBitDepthConverter:
    """Test cases for the BitDepthConverter class."""

    def test_convert_to_float32(self):
        """Test conversion to float32."""
        converter = BitDepthConverter()

        # 8-bit to float32
        img_8bit = np.array([0, 127, 255], dtype=np.uint8)
        float_img = converter.convert_to_float32(img_8bit, 8)
        expected = np.array([0.0, 127 / 255, 1.0], dtype=np.float32)
        np.testing.assert_array_almost_equal(float_img, expected)

    def test_convert_from_float32(self):
        """Test conversion from float32."""
        converter = BitDepthConverter()

        # Float32 to 8-bit
        float_img = np.array([0.0, 0.5, 1.0], dtype=np.float32)
        img_8bit = converter.convert_from_float32(float_img, 8, np.uint8)
        expected = np.array([0, 128, 255], dtype=np.uint8)
        np.testing.assert_array_equal(img_8bit, expected)

    def test_convert_bit_depth(self):
        """Test bit depth conversion."""
        converter = BitDepthConverter()

        # 8-bit to 12-bit
        img_8bit = np.array([0, 128, 255], dtype=np.uint8)
        img_12bit = converter.convert_bit_depth(img_8bit, 8, 12)

        # Check scaling is correct
        assert img_12bit[0] == 0
        assert img_12bit[2] == 4095  # 255 * (4095/255)
        assert img_12bit.dtype == np.uint16

    def test_same_bit_depth_conversion(self):
        """Test conversion with same source and target bit depth."""
        converter = BitDepthConverter()

        img = np.random.randint(0, 256, (10, 10), dtype=np.uint8)
        converted = converter.convert_bit_depth(img, 8, 8)

        np.testing.assert_array_equal(img, converted)
        assert converted is not img  # Should be a copy

    def test_clamping_in_conversion(self):
        """Test that values are properly clamped during conversion."""
        converter = BitDepthConverter()

        # Create float image with out-of-range values
        float_img = np.array([-0.1, 0.5, 1.1], dtype=np.float32)
        img_8bit = converter.convert_from_float32(float_img, 8, np.uint8)

        # Values should be clamped to [0, 255]
        assert img_8bit[0] == 0
        assert img_8bit[1] == 128
        assert img_8bit[2] == 255


class TestImageFormat:
    """Test cases for the ImageFormat class."""

    def test_valid_image_format(self):
        """Test creation of valid image formats."""
        fmt = ImageFormat(480, 640, 3, 12, SupportedDType.UINT12_IN_UINT16)
        assert fmt.height == 480
        assert fmt.width == 640
        assert fmt.channels == 3
        assert fmt.bit_depth == 12
        assert fmt.max_value == 4095
        assert fmt.shape == (480, 640, 3)

    def test_invalid_image_format(self):
        """Test creation of invalid image formats."""
        # Invalid dimensions
        with pytest.raises(ValueError, match="Image dimensions must be positive"):
            ImageFormat(0, 640, 3, 12, SupportedDType.UINT12_IN_UINT16)

        # Invalid channels
        with pytest.raises(ValueError, match="Channels must be 1 \\(mono\\), 3 \\(RGB\\), or 4 \\(RGBA\\)"):
            ImageFormat(480, 640, 2, 12, SupportedDType.UINT12_IN_UINT16)

        # Invalid bit depth
        with pytest.raises(ValueError, match="Bit depth must be 8, 10, 12, 14, or 16"):
            ImageFormat(480, 640, 3, 9, SupportedDType.UINT12_IN_UINT16)

        # Incompatible dtype and bit depth
        with pytest.raises(ValueError, match="8-bit images must use UINT8 dtype"):
            ImageFormat(480, 640, 3, 8, SupportedDType.UINT16)

    def test_mono_image_shape(self):
        """Test shape property for mono images."""
        fmt = ImageFormat(480, 640, 1, 8, SupportedDType.UINT8)
        assert fmt.shape == (480, 640)


class TestSafeImageProcessor:
    """Test cases for the SafeImageProcessor class."""

    def test_preprocess_image(self):
        """Test image preprocessing."""
        target_format = ImageFormat(480, 640, 3, 12, SupportedDType.UINT12_IN_UINT16)
        processor = SafeImageProcessor(target_format)

        # Test with matching format
        img = np.random.randint(0, 4096, (480, 640, 3), dtype=np.uint16)
        float_img, original_format = processor.preprocess_image(img)

        assert float_img.dtype == np.float32
        assert 0.0 <= float_img.min() <= float_img.max() <= 1.0
        assert original_format.bit_depth == 12

    def test_postprocess_image(self):
        """Test image postprocessing."""
        target_format = ImageFormat(480, 640, 3, 12, SupportedDType.UINT12_IN_UINT16)
        processor = SafeImageProcessor(target_format)

        # Create float image
        float_img = np.random.rand(480, 640, 3).astype(np.float32)
        original_format = ImageFormat(480, 640, 3, 12, SupportedDType.UINT12_IN_UINT16)

        # Postprocess
        result = processor.postprocess_image(float_img, original_format)

        assert result.dtype == np.uint16
        assert 0 <= result.min() <= result.max() <= 4095

    def test_safe_process(self):
        """Test safe processing pipeline."""
        target_format = ImageFormat(480, 640, 3, 12, SupportedDType.UINT12_IN_UINT16)
        processor = SafeImageProcessor(target_format)

        # Define a simple processing function
        def simple_processing(img):
            return img * 0.8  # Reduce brightness by 20%

        # Test processing
        original_img = np.random.randint(0, 4096, (480, 640, 3), dtype=np.uint16)
        processed_img = processor.safe_process(original_img, simple_processing)

        # Check that processing was applied and format preserved
        assert processed_img.shape == original_img.shape
        assert processed_img.dtype == original_img.dtype
        assert np.mean(processed_img) < np.mean(original_img)  # Should be darker

    def test_channel_conversion(self):
        """Test automatic channel conversion."""
        target_format = ImageFormat(480, 640, 3, 8, SupportedDType.UINT8)
        processor = SafeImageProcessor(target_format)

        # Test mono to RGB conversion
        mono_img = np.random.randint(0, 256, (480, 640), dtype=np.uint8)
        float_img, _ = processor.preprocess_image(mono_img)

        assert float_img.shape == (480, 640, 3)  # Should be converted to RGB

        # Test RGB to mono conversion
        target_format_mono = ImageFormat(480, 640, 1, 8, SupportedDType.UINT8)
        processor_mono = SafeImageProcessor(target_format_mono)

        rgb_img = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        float_img, _ = processor_mono.preprocess_image(rgb_img)

        assert float_img.shape == (480, 640)  # Should be converted to mono

    def test_invalid_channel_conversion(self):
        """Test invalid channel conversion."""
        target_format = ImageFormat(480, 640, 3, 8, SupportedDType.UINT8)
        processor = SafeImageProcessor(target_format)

        # RGBA to RGB conversion should fail
        rgba_img = np.random.randint(0, 256, (480, 640, 4), dtype=np.uint8)

        with pytest.raises(ValueError, match="Cannot convert from 4 to 3 channels"):
            processor.preprocess_image(rgba_img)


if __name__ == "__main__":
    pytest.main([__file__])
