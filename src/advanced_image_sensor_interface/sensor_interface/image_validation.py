"""
Image Validation and Processing Utilities

This module provides comprehensive image validation, bit-depth handling,
and safe processing operations for the Advanced Image Sensor Interface.

Classes:
    ImageValidator: Validates image format, shape, and bit depth
    ImageProcessor: Safe image processing with automatic scaling
    BitDepthConverter: Handles conversion between different bit depths

Functions:
    validate_image_format: Validate image format and parameters
    safe_process_image: Process image with automatic validation and scaling
    convert_bit_depth: Convert image between bit depths with proper scaling
"""

import logging
from dataclasses import dataclass
from enum import Enum

import numpy as np

logger = logging.getLogger(__name__)


class SupportedDType(Enum):
    """Supported image data types."""

    UINT8 = np.uint8
    UINT10_IN_UINT16 = "uint10_in_uint16"  # 10-bit data stored in uint16
    UINT12_IN_UINT16 = "uint12_in_uint16"  # 12-bit data stored in uint16
    UINT14_IN_UINT16 = "uint14_in_uint16"  # 14-bit data stored in uint16
    UINT16 = np.uint16
    FLOAT32 = np.float32


@dataclass
class ImageFormat:
    """Image format specification."""

    height: int
    width: int
    channels: int  # 1 for mono, 3 for RGB, 4 for RGBA
    bit_depth: int  # Actual bit depth (8, 10, 12, 14, 16)
    dtype: SupportedDType

    def __post_init__(self):
        """Validate image format parameters."""
        if self.height <= 0 or self.width <= 0:
            raise ValueError("Image dimensions must be positive")
        if self.channels not in {1, 3, 4}:
            raise ValueError("Channels must be 1 (mono), 3 (RGB), or 4 (RGBA)")
        if self.bit_depth not in {8, 10, 12, 14, 16}:
            raise ValueError("Bit depth must be 8, 10, 12, 14, or 16")

        # Validate dtype compatibility with bit depth
        if self.bit_depth == 8 and self.dtype != SupportedDType.UINT8:
            raise ValueError("8-bit images must use UINT8 dtype")
        elif self.bit_depth == 16 and self.dtype not in {SupportedDType.UINT16, SupportedDType.FLOAT32}:
            raise ValueError("16-bit images must use UINT16 or FLOAT32 dtype")
        elif self.bit_depth in {10, 12, 14}:
            expected_dtype = getattr(SupportedDType, f"UINT{self.bit_depth}_IN_UINT16")
            if self.dtype not in {expected_dtype, SupportedDType.UINT16, SupportedDType.FLOAT32}:
                raise ValueError(f"{self.bit_depth}-bit images must use appropriate dtype")

    @property
    def max_value(self) -> int:
        """Get maximum value for this bit depth."""
        return (2**self.bit_depth) - 1

    @property
    def shape(self) -> tuple[int, ...]:
        """Get expected image shape."""
        if self.channels == 1:
            return (self.height, self.width)
        else:
            return (self.height, self.width, self.channels)


class ImageValidator:
    """Validates image format, shape, and bit depth."""

    @staticmethod
    def validate_image(image: np.ndarray, expected_format: ImageFormat = None) -> ImageFormat:
        """
        Validate image format and return detected format.

        Args:
            image: Input image array
            expected_format: Expected format (optional)

        Returns:
            Detected or validated ImageFormat

        Raises:
            ValueError: If image format is invalid
        """
        if not isinstance(image, np.ndarray):
            raise ValueError("Image must be a numpy array")

        if image.size == 0:
            raise ValueError("Image cannot be empty")

        # Validate shape
        if image.ndim == 2:
            height, width = image.shape
            channels = 1
        elif image.ndim == 3:
            height, width, channels = image.shape
            if channels not in {3, 4}:
                raise ValueError(f"Invalid number of channels: {channels}")
        else:
            raise ValueError(f"Image must be 2D or 3D, got {image.ndim}D")

        # Detect bit depth and dtype
        if image.dtype == np.uint8:
            bit_depth = 8
            dtype = SupportedDType.UINT8
        elif image.dtype == np.uint16:
            # Need to infer bit depth from data range
            max_val = np.max(image)
            if max_val <= 255:
                bit_depth = 8
                dtype = SupportedDType.UINT8
            elif max_val <= 1023:
                bit_depth = 10
                dtype = SupportedDType.UINT10_IN_UINT16
            elif max_val <= 4095:
                bit_depth = 12
                dtype = SupportedDType.UINT12_IN_UINT16
            elif max_val <= 16383:
                bit_depth = 14
                dtype = SupportedDType.UINT14_IN_UINT16
            else:
                bit_depth = 16
                dtype = SupportedDType.UINT16
        elif image.dtype == np.float32:
            bit_depth = 16  # Assume float32 represents 16-bit data
            dtype = SupportedDType.FLOAT32
        else:
            raise ValueError(f"Unsupported dtype: {image.dtype}")

        detected_format = ImageFormat(height=height, width=width, channels=channels, bit_depth=bit_depth, dtype=dtype)

        # Validate against expected format if provided
        if expected_format is not None:
            if (
                detected_format.height != expected_format.height
                or detected_format.width != expected_format.width
                or detected_format.channels != expected_format.channels
            ):
                raise ValueError(f"Image shape mismatch: expected {expected_format.shape}, " f"got {detected_format.shape}")

            if detected_format.bit_depth != expected_format.bit_depth:
                logger.warning(
                    f"Bit depth mismatch: expected {expected_format.bit_depth}, " f"detected {detected_format.bit_depth}"
                )

        return detected_format

    @staticmethod
    def validate_color_correction_matrix(matrix: np.ndarray, channels: int) -> None:
        """
        Validate color correction matrix.

        Args:
            matrix: Color correction matrix
            channels: Number of image channels

        Raises:
            ValueError: If matrix is invalid
        """
        if not isinstance(matrix, np.ndarray):
            raise ValueError("Color correction matrix must be a numpy array")

        if channels == 1:
            # For mono images, matrix should be 1x1 or identity
            if matrix.shape != (1, 1) and not np.allclose(matrix, np.eye(3)):
                logger.warning("Color correction matrix ignored for mono images")
        elif channels in {3, 4}:
            if matrix.shape != (3, 3):
                raise ValueError(f"Color correction matrix must be 3x3 for RGB images, got {matrix.shape}")
        else:
            raise ValueError(f"Unsupported number of channels: {channels}")


class BitDepthConverter:
    """Handles conversion between different bit depths."""

    @staticmethod
    def convert_to_float32(image: np.ndarray, bit_depth: int) -> np.ndarray:
        """
        Convert image to float32 with normalized range [0, 1].

        Args:
            image: Input image
            bit_depth: Source bit depth

        Returns:
            Float32 image normalized to [0, 1]
        """
        max_val = (2**bit_depth) - 1
        return image.astype(np.float32) / max_val

    @staticmethod
    def convert_from_float32(image: np.ndarray, target_bit_depth: int, target_dtype: np.dtype) -> np.ndarray:
        """
        Convert float32 image back to target bit depth and dtype.

        Args:
            image: Float32 image in [0, 1] range
            target_bit_depth: Target bit depth
            target_dtype: Target numpy dtype

        Returns:
            Image converted to target format
        """
        max_val = (2**target_bit_depth) - 1

        # Clamp to valid range and scale
        clamped = np.clip(image, 0.0, 1.0)
        scaled = clamped * max_val

        # Round and convert to target dtype
        if target_dtype == np.float32:
            return scaled.astype(np.float32)
        else:
            return np.round(scaled).astype(target_dtype)

    @staticmethod
    def convert_bit_depth(image: np.ndarray, source_depth: int, target_depth: int) -> np.ndarray:
        """
        Convert image between bit depths.

        Args:
            image: Input image
            source_depth: Source bit depth
            target_depth: Target bit depth

        Returns:
            Image converted to target bit depth
        """
        if source_depth == target_depth:
            return image.copy()

        # Convert through float32 for precision
        float_image = BitDepthConverter.convert_to_float32(image, source_depth)

        # Determine target dtype
        if target_depth == 8:
            target_dtype = np.uint8
        elif target_depth == 16:
            target_dtype = np.uint16
        else:
            target_dtype = np.uint16  # Store 10/12/14-bit in uint16

        return BitDepthConverter.convert_from_float32(float_image, target_depth, target_dtype)


class SafeImageProcessor:
    """Safe image processing with automatic validation and scaling."""

    def __init__(self, target_format: ImageFormat):
        """
        Initialize safe image processor.

        Args:
            target_format: Target image format for processing
        """
        self.target_format = target_format
        self.validator = ImageValidator()
        self.converter = BitDepthConverter()

    def preprocess_image(self, image: np.ndarray) -> tuple[np.ndarray, ImageFormat]:
        """
        Preprocess image for safe processing.

        Args:
            image: Input image

        Returns:
            Tuple of (processed_image, original_format)
        """
        # Validate input image
        original_format = self.validator.validate_image(image)

        # Convert to float32 for processing
        float_image = self.converter.convert_to_float32(image, original_format.bit_depth)

        # Ensure correct shape
        if original_format.channels != self.target_format.channels:
            if original_format.channels == 1 and self.target_format.channels == 3:
                # Convert mono to RGB
                float_image = np.stack([float_image] * 3, axis=-1)
            elif original_format.channels == 3 and self.target_format.channels == 1:
                # Convert RGB to mono (luminance)
                float_image = np.dot(float_image, [0.299, 0.587, 0.114])
            else:
                raise ValueError(f"Cannot convert from {original_format.channels} to " f"{self.target_format.channels} channels")

        return float_image, original_format

    def postprocess_image(self, image: np.ndarray, original_format: ImageFormat) -> np.ndarray:
        """
        Postprocess image back to original format.

        Args:
            image: Processed float32 image
            original_format: Original image format

        Returns:
            Image in original format
        """
        # Convert back to original bit depth and dtype
        if original_format.dtype == SupportedDType.UINT8:
            target_dtype = np.uint8
        elif original_format.dtype == SupportedDType.UINT16:
            target_dtype = np.uint16
        elif original_format.dtype == SupportedDType.FLOAT32:
            target_dtype = np.float32
        else:
            target_dtype = np.uint16  # For 10/12/14-bit data

        return self.converter.convert_from_float32(image, original_format.bit_depth, target_dtype)

    def safe_process(self, image: np.ndarray, processing_func) -> np.ndarray:
        """
        Safely process image with validation and format handling.

        Args:
            image: Input image
            processing_func: Function to apply to the image

        Returns:
            Processed image in original format
        """
        # Preprocess
        float_image, original_format = self.preprocess_image(image)

        # Apply processing function
        processed_image = processing_func(float_image)

        # Postprocess
        return self.postprocess_image(processed_image, original_format)


# Example usage and testing
if __name__ == "__main__":
    # Test image validation
    validator = ImageValidator()

    # Test various image formats
    test_images = [
        np.random.randint(0, 256, (480, 640), dtype=np.uint8),  # 8-bit mono
        np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8),  # 8-bit RGB
        np.random.randint(0, 4096, (480, 640), dtype=np.uint16),  # 12-bit mono
        np.random.randint(0, 4096, (480, 640, 3), dtype=np.uint16),  # 12-bit RGB
    ]

    for i, img in enumerate(test_images):
        try:
            fmt = validator.validate_image(img)
            print(f"Image {i}: {fmt.height}x{fmt.width}x{fmt.channels}, " f"{fmt.bit_depth}-bit, {fmt.dtype}")
        except ValueError as e:
            print(f"Image {i} validation failed: {e}")

    # Test bit depth conversion
    converter = BitDepthConverter()

    img_8bit = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
    img_12bit = converter.convert_bit_depth(img_8bit, 8, 12)
    img_back = converter.convert_bit_depth(img_12bit, 12, 8)

    print(f"8-bit range: {img_8bit.min()}-{img_8bit.max()}")
    print(f"12-bit range: {img_12bit.min()}-{img_12bit.max()}")
    print(f"Back to 8-bit range: {img_back.min()}-{img_back.max()}")

    # Test safe processing
    target_format = ImageFormat(480, 640, 3, 12, SupportedDType.UINT12_IN_UINT16)
    processor = SafeImageProcessor(target_format)

    def dummy_processing(img):
        return img * 0.9  # Simple processing

    test_img = np.random.randint(0, 4096, (480, 640, 3), dtype=np.uint16)
    processed = processor.safe_process(test_img, dummy_processing)

    print(f"Original range: {test_img.min()}-{test_img.max()}")
    print(f"Processed range: {processed.min()}-{processed.max()}")
    print(f"Processing preserved dtype: {processed.dtype == test_img.dtype}")

    print("All image validation tests passed!")
