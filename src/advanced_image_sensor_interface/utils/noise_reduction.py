"""
Extensible Noise Reduction Framework

This module provides abstract base classes and interfaces for implementing
custom noise reduction algorithms in the Advanced Image Sensor Interface.

Author: Advanced Image Sensor Interface Team
Version: 2.0.0
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, ClassVar

import numpy as np


class NoiseType(Enum):
    """Types of noise that can be reduced."""

    GAUSSIAN = "gaussian"
    SALT_PEPPER = "salt_pepper"
    POISSON = "poisson"
    SPECKLE = "speckle"
    THERMAL = "thermal"
    SHOT = "shot"


@dataclass
class NoiseReductionConfig:
    """Configuration for noise reduction algorithms."""

    noise_type: NoiseType
    strength: float = 0.5  # 0.0 to 1.0
    preserve_edges: bool = True
    preserve_texture: bool = True
    adaptive: bool = True
    parameters: dict[str, Any] = None

    def __post_init__(self):
        if self.parameters is None:
            self.parameters = {}


class NoiseReducer(ABC):
    """
    Abstract base class for noise reduction algorithms.

    This class defines the interface that all noise reduction implementations
    must follow, enabling easy extension and customization of noise reduction
    capabilities.
    """

    def __init__(self, config: NoiseReductionConfig):
        """
        Initialize the noise reducer.

        Args:
            config: Configuration for the noise reduction algorithm
        """
        self.config = config
        self.processing_stats = {"images_processed": 0, "total_processing_time": 0.0, "average_processing_time": 0.0}

    @abstractmethod
    def process(self, image: np.ndarray) -> np.ndarray:
        """
        Process an image to reduce noise.

        Args:
            image: Input image as numpy array

        Returns:
            Processed image with reduced noise

        Raises:
            ValueError: If image format is invalid
            NotImplementedError: If method is not implemented
        """
        pass

    @abstractmethod
    def estimate_noise_level(self, image: np.ndarray) -> float:
        """
        Estimate the noise level in an image.

        Args:
            image: Input image as numpy array

        Returns:
            Estimated noise level (0.0 to 1.0)
        """
        pass

    @abstractmethod
    def get_algorithm_name(self) -> str:
        """
        Get the name of the noise reduction algorithm.

        Returns:
            Algorithm name string
        """
        pass

    def validate_image(self, image: np.ndarray) -> None:
        """
        Validate input image format.

        Args:
            image: Input image to validate

        Raises:
            ValueError: If image format is invalid
        """
        if not isinstance(image, np.ndarray):
            raise ValueError("Image must be a numpy array")

        if image.ndim not in [2, 3]:
            raise ValueError("Image must be 2D (grayscale) or 3D (color)")

        if image.ndim == 3 and image.shape[2] not in [1, 3, 4]:
            raise ValueError("Color images must have 1, 3, or 4 channels")

        if image.size == 0:
            raise ValueError("Image cannot be empty")

    def update_stats(self, processing_time: float) -> None:
        """
        Update processing statistics.

        Args:
            processing_time: Time taken to process the image
        """
        self.processing_stats["images_processed"] += 1
        self.processing_stats["total_processing_time"] += processing_time
        self.processing_stats["average_processing_time"] = (
            self.processing_stats["total_processing_time"] / self.processing_stats["images_processed"]
        )

    def get_processing_stats(self) -> dict[str, Any]:
        """
        Get processing statistics.

        Returns:
            Dictionary containing processing statistics
        """
        return self.processing_stats.copy()

    def reset_stats(self) -> None:
        """Reset processing statistics."""
        self.processing_stats = {"images_processed": 0, "total_processing_time": 0.0, "average_processing_time": 0.0}

    def get_config(self) -> NoiseReductionConfig:
        """
        Get the current configuration.

        Returns:
            Current noise reduction configuration
        """
        return self.config

    def update_config(self, **kwargs) -> None:
        """
        Update configuration parameters.

        Args:
            **kwargs: Configuration parameters to update
        """
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
            else:
                self.config.parameters[key] = value


class GaussianNoiseReducer(NoiseReducer):
    """
    Gaussian noise reduction implementation using adaptive filtering.

    This implementation uses Gaussian filtering with edge preservation
    to reduce Gaussian noise while maintaining image details.
    """

    def get_algorithm_name(self) -> str:
        """Get algorithm name."""
        return "Adaptive Gaussian Filter"

    def process(self, image: np.ndarray) -> np.ndarray:
        """
        Process image to reduce Gaussian noise.

        Args:
            image: Input image

        Returns:
            Denoised image
        """
        import time

        from scipy.ndimage import gaussian_filter

        start_time = time.time()
        self.validate_image(image)

        # Estimate noise level if adaptive
        if self.config.adaptive:
            noise_level = self.estimate_noise_level(image)
            sigma = self.config.strength * noise_level * 2.0
        else:
            sigma = self.config.strength * 2.0

        # Apply Gaussian filter
        if image.ndim == 3:
            # Process each channel separately
            processed = np.zeros_like(image)
            for c in range(image.shape[2]):
                processed[:, :, c] = gaussian_filter(image[:, :, c].astype(np.float32), sigma=sigma)
        else:
            processed = gaussian_filter(image.astype(np.float32), sigma=sigma)

        # Preserve edges if requested
        if self.config.preserve_edges:
            processed = self._preserve_edges(image, processed)

        # Update statistics
        processing_time = time.time() - start_time
        self.update_stats(processing_time)

        return np.clip(processed, 0, 255).astype(image.dtype)

    def estimate_noise_level(self, image: np.ndarray) -> float:
        """
        Estimate Gaussian noise level using Laplacian variance.

        Args:
            image: Input image

        Returns:
            Estimated noise level
        """
        from scipy.ndimage import laplace

        if image.ndim == 3:
            gray = np.mean(image, axis=2)
        else:
            gray = image

        # Use Laplacian variance as noise estimate
        laplacian_var = np.var(laplace(gray))
        return min(laplacian_var / 1000.0, 1.0)

    def _preserve_edges(self, original: np.ndarray, filtered: np.ndarray) -> np.ndarray:
        """
        Preserve edges by blending original and filtered images.

        Args:
            original: Original image
            filtered: Filtered image

        Returns:
            Edge-preserved image
        """
        from scipy.ndimage import sobel

        # Calculate edge strength
        if original.ndim == 3:
            gray = np.mean(original, axis=2)
        else:
            gray = original

        edge_strength = np.sqrt(sobel(gray, axis=0) ** 2 + sobel(gray, axis=1) ** 2)
        edge_strength = edge_strength / np.max(edge_strength)

        # Blend based on edge strength
        if original.ndim == 3:
            edge_strength = np.expand_dims(edge_strength, axis=2)

        alpha = edge_strength * 0.8  # Preserve 80% of original at edges
        result = alpha * original.astype(np.float32) + (1 - alpha) * filtered

        return result


class BilateralNoiseReducer(NoiseReducer):
    """
    Bilateral filtering for edge-preserving noise reduction.

    This implementation uses bilateral filtering to reduce noise while
    preserving edges and fine details in the image.
    """

    def get_algorithm_name(self) -> str:
        """Get algorithm name."""
        return "Bilateral Filter"

    def process(self, image: np.ndarray) -> np.ndarray:
        """
        Process image using bilateral filtering.

        Args:
            image: Input image

        Returns:
            Denoised image
        """
        import time

        start_time = time.time()
        self.validate_image(image)

        # Get bilateral filter parameters
        d = self.config.parameters.get("diameter", 9)
        sigma_color = self.config.strength * 75
        sigma_space = self.config.strength * 75

        # Apply bilateral filter (simplified implementation)
        processed = self._bilateral_filter(image, d, sigma_color, sigma_space)

        # Update statistics
        processing_time = time.time() - start_time
        self.update_stats(processing_time)

        return processed

    def estimate_noise_level(self, image: np.ndarray) -> float:
        """Estimate noise level for bilateral filtering."""
        # Use standard deviation of high-frequency components
        from scipy.ndimage import gaussian_filter

        if image.ndim == 3:
            gray = np.mean(image, axis=2)
        else:
            gray = image

        # High-pass filter
        low_pass = gaussian_filter(gray, sigma=2.0)
        high_pass = gray - low_pass

        noise_level = np.std(high_pass) / 255.0
        return min(noise_level, 1.0)

    def _bilateral_filter(self, image: np.ndarray, d: int, sigma_color: float, sigma_space: float) -> np.ndarray:
        """
        Simplified bilateral filter implementation.

        Args:
            image: Input image
            d: Diameter of pixel neighborhood
            sigma_color: Filter sigma in color space
            sigma_space: Filter sigma in coordinate space

        Returns:
            Filtered image
        """
        # This is a simplified implementation
        # In practice, you would use cv2.bilateralFilter or similar
        from scipy.ndimage import gaussian_filter

        # Fallback to Gaussian filter for this example
        sigma = sigma_space / 10.0

        if image.ndim == 3:
            processed = np.zeros_like(image, dtype=np.float32)
            for c in range(image.shape[2]):
                processed[:, :, c] = gaussian_filter(image[:, :, c].astype(np.float32), sigma=sigma)
        else:
            processed = gaussian_filter(image.astype(np.float32), sigma=sigma)

        return np.clip(processed, 0, 255).astype(image.dtype)


class MedianNoiseReducer(NoiseReducer):
    """
    Median filtering for salt-and-pepper noise reduction.

    This implementation uses median filtering to effectively remove
    salt-and-pepper noise while preserving edges.
    """

    def get_algorithm_name(self) -> str:
        """Get algorithm name."""
        return "Median Filter"

    def process(self, image: np.ndarray) -> np.ndarray:
        """
        Process image using median filtering.

        Args:
            image: Input image

        Returns:
            Denoised image
        """
        import time

        from scipy.ndimage import median_filter

        start_time = time.time()
        self.validate_image(image)

        # Get kernel size based on strength
        kernel_size = int(3 + self.config.strength * 4)  # 3 to 7
        if kernel_size % 2 == 0:
            kernel_size += 1  # Ensure odd kernel size

        # Apply median filter
        if image.ndim == 3:
            processed = np.zeros_like(image)
            for c in range(image.shape[2]):
                processed[:, :, c] = median_filter(image[:, :, c], size=kernel_size)
        else:
            processed = median_filter(image, size=kernel_size)

        # Update statistics
        processing_time = time.time() - start_time
        self.update_stats(processing_time)

        return processed

    def estimate_noise_level(self, image: np.ndarray) -> float:
        """Estimate salt-and-pepper noise level."""
        # Count pixels that are likely salt or pepper noise
        if image.ndim == 3:
            gray = np.mean(image, axis=2)
        else:
            gray = image

        # Count very dark and very bright pixels
        salt_pixels = np.sum(gray > 240)
        pepper_pixels = np.sum(gray < 15)
        total_pixels = gray.size

        noise_ratio = (salt_pixels + pepper_pixels) / total_pixels
        return min(noise_ratio * 10, 1.0)  # Scale to 0-1 range


class NoiseReducerFactory:
    """
    Factory class for creating noise reduction algorithms.

    This factory enables easy registration and creation of custom
    noise reduction implementations.
    """

    _reducers: ClassVar[dict] = {
        NoiseType.GAUSSIAN: GaussianNoiseReducer,
        NoiseType.SALT_PEPPER: MedianNoiseReducer,
        NoiseType.POISSON: GaussianNoiseReducer,  # Use Gaussian for Poisson
        NoiseType.SPECKLE: BilateralNoiseReducer,
        NoiseType.THERMAL: GaussianNoiseReducer,
        NoiseType.SHOT: GaussianNoiseReducer,
    }

    @classmethod
    def register_reducer(cls, noise_type: NoiseType, reducer_class: type) -> None:
        """
        Register a custom noise reducer.

        Args:
            noise_type: Type of noise the reducer handles
            reducer_class: Class implementing NoiseReducer interface
        """
        if not issubclass(reducer_class, NoiseReducer):
            raise ValueError("Reducer class must inherit from NoiseReducer")

        cls._reducers[noise_type] = reducer_class

    @classmethod
    def create_reducer(cls, config: NoiseReductionConfig) -> NoiseReducer:
        """
        Create a noise reducer instance.

        Args:
            config: Configuration for the noise reducer

        Returns:
            Noise reducer instance

        Raises:
            ValueError: If noise type is not supported
        """
        if config.noise_type not in cls._reducers:
            raise ValueError(f"Unsupported noise type: {config.noise_type}")

        reducer_class = cls._reducers[config.noise_type]
        return reducer_class(config)

    @classmethod
    def get_available_types(cls) -> list:
        """
        Get list of available noise types.

        Returns:
            List of supported noise types
        """
        return list(cls._reducers.keys())

    @classmethod
    def get_reducer_info(cls, noise_type: NoiseType) -> dict[str, Any]:
        """
        Get information about a specific reducer.

        Args:
            noise_type: Type of noise reducer

        Returns:
            Dictionary with reducer information
        """
        if noise_type not in cls._reducers:
            raise ValueError(f"Unsupported noise type: {noise_type}")

        reducer_class = cls._reducers[noise_type]

        # Create temporary instance to get algorithm name
        temp_config = NoiseReductionConfig(noise_type=noise_type)
        temp_instance = reducer_class(temp_config)

        return {
            "noise_type": noise_type,
            "algorithm_name": temp_instance.get_algorithm_name(),
            "class_name": reducer_class.__name__,
            "module": reducer_class.__module__,
        }


# Example of how to extend with a custom noise reducer
class WaveletNoiseReducer(NoiseReducer):
    """
    Example custom noise reducer using wavelet denoising.

    This demonstrates how to implement a custom noise reduction algorithm
    by inheriting from the NoiseReducer base class.
    """

    def get_algorithm_name(self) -> str:
        """Get algorithm name."""
        return "Wavelet Denoising"

    def process(self, image: np.ndarray) -> np.ndarray:
        """
        Process image using wavelet denoising.

        Args:
            image: Input image

        Returns:
            Denoised image
        """
        import time

        start_time = time.time()
        self.validate_image(image)

        # Simplified wavelet denoising (would use pywt in practice)
        # For this example, we'll use a combination of filters
        processed = self._wavelet_denoise(image)

        # Update statistics
        processing_time = time.time() - start_time
        self.update_stats(processing_time)

        return processed

    def estimate_noise_level(self, image: np.ndarray) -> float:
        """Estimate noise level for wavelet denoising."""
        # Use robust median estimator
        if image.ndim == 3:
            gray = np.mean(image, axis=2)
        else:
            gray = image

        # Estimate noise using median absolute deviation
        median_val = np.median(gray)
        mad = np.median(np.abs(gray - median_val))
        noise_level = mad / 255.0 * 1.4826  # Scale factor for Gaussian noise

        return min(noise_level, 1.0)

    def _wavelet_denoise(self, image: np.ndarray) -> np.ndarray:
        """
        Simplified wavelet denoising implementation.

        Args:
            image: Input image

        Returns:
            Denoised image
        """
        # This is a placeholder implementation
        # In practice, you would use PyWavelets (pywt) library
        from scipy.ndimage import gaussian_filter

        # Multi-scale denoising approximation
        scales = [0.5, 1.0, 2.0]
        weights = [0.2, 0.6, 0.2]

        if image.ndim == 3:
            processed = np.zeros_like(image, dtype=np.float32)
            for c in range(image.shape[2]):
                channel_result = np.zeros_like(image[:, :, c], dtype=np.float32)
                for scale, weight in zip(scales, weights):
                    filtered = gaussian_filter(image[:, :, c].astype(np.float32), sigma=scale * self.config.strength)
                    channel_result += weight * filtered
                processed[:, :, c] = channel_result
        else:
            processed = np.zeros_like(image, dtype=np.float32)
            for scale, weight in zip(scales, weights):
                filtered = gaussian_filter(image.astype(np.float32), sigma=scale * self.config.strength)
                processed += weight * filtered

        return np.clip(processed, 0, 255).astype(image.dtype)


# Register the custom wavelet reducer
# This shows how users can extend the system
def register_custom_reducers():
    """Register custom noise reducers with the factory."""
    # Register wavelet reducer for speckle noise
    NoiseReducerFactory.register_reducer(NoiseType.SPECKLE, WaveletNoiseReducer)


# Compatibility functions for backward compatibility
def reduce_noise(image: np.ndarray, strength: float = 0.5, noise_type: str = "gaussian") -> np.ndarray:
    """
    Legacy noise reduction function for backward compatibility.

    Args:
        image: Input image
        strength: Noise reduction strength (0.0 to 1.0)
        noise_type: Type of noise ("gaussian", "salt_pepper", etc.)

    Returns:
        Denoised image
    """
    # Map string to enum
    noise_type_map = {
        "gaussian": NoiseType.GAUSSIAN,
        "salt_pepper": NoiseType.SALT_PEPPER,
        "poisson": NoiseType.POISSON,
        "speckle": NoiseType.SPECKLE,
        "thermal": NoiseType.THERMAL,
        "shot": NoiseType.SHOT,
    }

    noise_enum = noise_type_map.get(noise_type.lower(), NoiseType.GAUSSIAN)

    # Create configuration and reducer
    config = NoiseReductionConfig(noise_type=noise_enum, strength=strength, preserve_edges=True, adaptive=True)

    reducer = NoiseReducerFactory.create_reducer(config)
    return reducer.process(image)


def adaptive_noise_reduction(image: np.ndarray, strength: float = 0.5) -> np.ndarray:
    """
    Legacy adaptive noise reduction function for backward compatibility.

    Args:
        image: Input image
        strength: Noise reduction strength (0.0 to 1.0)

    Returns:
        Denoised image
    """
    config = NoiseReductionConfig(noise_type=NoiseType.GAUSSIAN, strength=strength, preserve_edges=True, adaptive=True)

    reducer = NoiseReducerFactory.create_reducer(config)
    return reducer.process(image)


# Register custom reducers on module import
register_custom_reducers()
