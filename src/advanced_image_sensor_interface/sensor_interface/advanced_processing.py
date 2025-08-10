"""
Advanced image processing algorithms and pipelines.

This module provides sophisticated image processing capabilities including
advanced noise reduction, edge enhancement, color processing, and optimization
algorithms for high-quality image output.
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


class ProcessingMode(Enum):
    """Image processing modes."""

    FAST = "fast"
    BALANCED = "balanced"
    QUALITY = "quality"
    CUSTOM = "custom"


class NoiseReductionMethod(Enum):
    """Noise reduction methods."""

    GAUSSIAN = "gaussian"
    BILATERAL = "bilateral"
    NON_LOCAL_MEANS = "non_local_means"
    WAVELET = "wavelet"
    ADAPTIVE = "adaptive"


@dataclass
class AdvancedProcessingConfig:
    """Configuration for advanced image processing."""

    # Processing mode
    mode: ProcessingMode = ProcessingMode.BALANCED

    # Noise reduction
    enable_noise_reduction: bool = True
    noise_reduction_method: NoiseReductionMethod = NoiseReductionMethod.BILATERAL
    noise_reduction_strength: float = 0.5

    # Edge enhancement
    enable_edge_enhancement: bool = True
    edge_enhancement_strength: float = 0.3
    edge_preservation: bool = True

    # Color processing
    enable_color_enhancement: bool = True
    saturation_boost: float = 1.1
    contrast_enhancement: float = 1.05

    # Advanced features
    enable_adaptive_processing: bool = True
    enable_multi_scale: bool = True
    enable_temporal_filtering: bool = False

    # Performance settings
    use_gpu_acceleration: bool = False
    num_threads: int = 4

    def __post_init__(self):
        """Validate configuration parameters."""
        if not 0.0 <= self.noise_reduction_strength <= 1.0:
            raise ValueError("Noise reduction strength must be between 0.0 and 1.0")
        if not 0.0 <= self.edge_enhancement_strength <= 1.0:
            raise ValueError("Edge enhancement strength must be between 0.0 and 1.0")


class AdvancedImageProcessor:
    """
    Advanced image processing pipeline with sophisticated algorithms.

    This processor provides high-quality image enhancement using advanced
    algorithms for noise reduction, edge enhancement, and color processing.
    """

    def __init__(self, config: Optional[AdvancedProcessingConfig] = None):
        """Initialize advanced image processor."""
        self.config = config or AdvancedProcessingConfig()
        self.processing_stats = {
            "frames_processed": 0,
            "total_processing_time": 0.0,
            "average_processing_time": 0.0,
            "quality_improvements": [],
        }

        # Initialize processing kernels
        self._initialize_kernels()

        logger.info(f"Advanced image processor initialized with mode: {self.config.mode.value}")

    def _initialize_kernels(self):
        """Initialize processing kernels and filters."""
        # Edge detection kernels
        self.sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
        self.sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)

        # Gaussian kernels for different scales
        self.gaussian_kernels = {}
        for sigma in [0.5, 1.0, 1.5, 2.0]:
            size = int(6 * sigma + 1)
            if size % 2 == 0:
                size += 1
            kernel = self._create_gaussian_kernel(size, sigma)
            self.gaussian_kernels[sigma] = kernel

        # Laplacian kernel for edge enhancement
        self.laplacian_kernel = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=np.float32)

    def _create_gaussian_kernel(self, size: int, sigma: float) -> np.ndarray:
        """Create Gaussian kernel for filtering."""
        kernel = np.zeros((size, size), dtype=np.float32)
        center = size // 2

        for i in range(size):
            for j in range(size):
                x, y = i - center, j - center
                kernel[i, j] = np.exp(-(x * x + y * y) / (2 * sigma * sigma))

        return kernel / np.sum(kernel)

    def process_image(self, image: np.ndarray, metadata: Optional[dict[str, Any]] = None) -> np.ndarray:
        """
        Process image with advanced algorithms.

        Args:
            image: Input image array
            metadata: Optional metadata for adaptive processing

        Returns:
            Processed image array
        """
        import time

        start_time = time.time()

        try:
            # Validate input
            if image is None or image.size == 0:
                raise ValueError("Invalid input image")

            # Convert to float for processing
            processed = image.astype(np.float32)

            # Apply processing pipeline based on mode
            if self.config.mode == ProcessingMode.FAST:
                processed = self._fast_processing_pipeline(processed, metadata)
            elif self.config.mode == ProcessingMode.QUALITY:
                processed = self._quality_processing_pipeline(processed, metadata)
            else:  # BALANCED or CUSTOM
                processed = self._balanced_processing_pipeline(processed, metadata)

            # Convert back to original dtype
            processed = np.clip(processed, 0, 255).astype(image.dtype)

            # Update statistics
            processing_time = time.time() - start_time
            self._update_statistics(processing_time)

            logger.debug(f"Image processed in {processing_time:.3f}s")
            return processed

        except Exception as e:
            logger.error(f"Image processing failed: {e}")
            return image  # Return original image on error

    def _fast_processing_pipeline(self, image: np.ndarray, metadata: Optional[dict[str, Any]]) -> np.ndarray:
        """Fast processing pipeline optimized for speed."""
        processed = image.copy()

        # Basic noise reduction
        if self.config.enable_noise_reduction:
            processed = self._apply_gaussian_blur(processed, sigma=0.5)

        # Simple edge enhancement
        if self.config.enable_edge_enhancement:
            processed = self._apply_simple_edge_enhancement(processed)

        # Basic color enhancement
        if self.config.enable_color_enhancement and len(image.shape) == 3:
            processed = self._apply_basic_color_enhancement(processed)

        return processed

    def _balanced_processing_pipeline(self, image: np.ndarray, metadata: Optional[dict[str, Any]]) -> np.ndarray:
        """Balanced processing pipeline with good quality/speed tradeoff."""
        processed = image.copy()

        # Advanced noise reduction
        if self.config.enable_noise_reduction:
            processed = self._apply_bilateral_filter(processed)

        # Edge-preserving enhancement
        if self.config.enable_edge_enhancement:
            processed = self._apply_edge_preserving_enhancement(processed)

        # Color processing
        if self.config.enable_color_enhancement and len(image.shape) == 3:
            processed = self._apply_advanced_color_enhancement(processed)

        # Adaptive processing if enabled
        if self.config.enable_adaptive_processing:
            processed = self._apply_adaptive_enhancement(processed, metadata)

        return processed

    def _quality_processing_pipeline(self, image: np.ndarray, metadata: Optional[dict[str, Any]]) -> np.ndarray:
        """High-quality processing pipeline with best results."""
        processed = image.copy()

        # Multi-scale noise reduction
        if self.config.enable_noise_reduction:
            if self.config.enable_multi_scale:
                processed = self._apply_multiscale_noise_reduction(processed)
            else:
                processed = self._apply_non_local_means_denoising(processed)

        # Advanced edge enhancement
        if self.config.enable_edge_enhancement:
            processed = self._apply_advanced_edge_enhancement(processed)

        # Professional color processing
        if self.config.enable_color_enhancement and len(image.shape) == 3:
            processed = self._apply_professional_color_processing(processed)

        # Adaptive and temporal processing
        if self.config.enable_adaptive_processing:
            processed = self._apply_adaptive_enhancement(processed, metadata)

        return processed

    def _apply_gaussian_blur(self, image: np.ndarray, sigma: float = 1.0) -> np.ndarray:
        """Apply Gaussian blur for noise reduction."""
        kernel = self.gaussian_kernels.get(sigma)
        if kernel is None:
            kernel = self._create_gaussian_kernel(int(6 * sigma + 1), sigma)

        return self._convolve_2d(image, kernel)

    def _apply_bilateral_filter(self, image: np.ndarray) -> np.ndarray:
        """Apply bilateral filter for edge-preserving noise reduction."""
        # Simplified bilateral filter implementation
        # In practice, this would use optimized implementations

        filtered = np.zeros_like(image)
        h, w = image.shape[:2]

        # Bilateral filter parameters
        spatial_sigma = 5.0
        intensity_sigma = 50.0
        kernel_size = 9

        for i in range(kernel_size // 2, h - kernel_size // 2):
            for j in range(kernel_size // 2, w - kernel_size // 2):
                # Extract neighborhood
                neighborhood = image[
                    i - kernel_size // 2 : i + kernel_size // 2 + 1, j - kernel_size // 2 : j + kernel_size // 2 + 1
                ]

                # Compute bilateral weights
                center_intensity = image[i, j]
                weights = self._compute_bilateral_weights(neighborhood, center_intensity, spatial_sigma, intensity_sigma)

                # Apply weighted average
                if len(image.shape) == 3:
                    for c in range(image.shape[2]):
                        filtered[i, j, c] = np.sum(weights * neighborhood[:, :, c]) / np.sum(weights)
                else:
                    filtered[i, j] = np.sum(weights * neighborhood) / np.sum(weights)

        return filtered

    def _compute_bilateral_weights(
        self, neighborhood: np.ndarray, center_intensity: float, spatial_sigma: float, intensity_sigma: float
    ) -> np.ndarray:
        """Compute bilateral filter weights."""
        h, w = neighborhood.shape[:2]
        weights = np.zeros((h, w))

        center_y, center_x = h // 2, w // 2

        for i in range(h):
            for j in range(w):
                # Spatial distance
                spatial_dist = np.sqrt((i - center_y) ** 2 + (j - center_x) ** 2)
                spatial_weight = np.exp(-(spatial_dist**2) / (2 * spatial_sigma**2))

                # Intensity difference
                if len(neighborhood.shape) == 3:
                    intensity_diff = np.linalg.norm(neighborhood[i, j] - center_intensity)
                else:
                    intensity_diff = abs(neighborhood[i, j] - center_intensity)

                intensity_weight = np.exp(-(intensity_diff**2) / (2 * intensity_sigma**2))

                weights[i, j] = spatial_weight * intensity_weight

        return weights

    def _apply_simple_edge_enhancement(self, image: np.ndarray) -> np.ndarray:
        """Apply simple edge enhancement."""
        # Detect edges using Laplacian
        edges = self._convolve_2d(image, self.laplacian_kernel)

        # Enhance edges
        strength = self.config.edge_enhancement_strength
        enhanced = image + strength * edges

        return np.clip(enhanced, 0, 255)

    def _apply_edge_preserving_enhancement(self, image: np.ndarray) -> np.ndarray:
        """Apply edge-preserving enhancement."""
        # Compute edge map
        if len(image.shape) == 3:
            gray = np.mean(image, axis=2)
        else:
            gray = image

        edges_x = self._convolve_2d(gray, self.sobel_x)
        edges_y = self._convolve_2d(gray, self.sobel_y)
        edge_magnitude = np.sqrt(edges_x**2 + edges_y**2)

        # Create edge mask
        edge_threshold = np.percentile(edge_magnitude, 75)
        edge_mask = edge_magnitude > edge_threshold

        # Apply enhancement only to edge regions
        enhanced = image.copy()
        strength = self.config.edge_enhancement_strength

        if len(image.shape) == 3:
            for c in range(image.shape[2]):
                laplacian = self._convolve_2d(image[:, :, c], self.laplacian_kernel)
                enhanced[:, :, c] = np.where(edge_mask, image[:, :, c] + strength * laplacian, image[:, :, c])
        else:
            laplacian = self._convolve_2d(image, self.laplacian_kernel)
            enhanced = np.where(edge_mask, image + strength * laplacian, image)

        return np.clip(enhanced, 0, 255)

    def _apply_basic_color_enhancement(self, image: np.ndarray) -> np.ndarray:
        """Apply basic color enhancement."""
        if len(image.shape) != 3:
            return image

        enhanced = image.copy().astype(np.float32)

        # Saturation boost
        if self.config.saturation_boost != 1.0:
            # Convert to HSV-like processing
            # Simplified saturation enhancement
            mean_intensity = np.mean(enhanced, axis=2, keepdims=True)
            enhanced = mean_intensity + self.config.saturation_boost * (enhanced - mean_intensity)

        # Contrast enhancement
        if self.config.contrast_enhancement != 1.0:
            mean_val = np.mean(enhanced)
            enhanced = mean_val + self.config.contrast_enhancement * (enhanced - mean_val)

        return np.clip(enhanced, 0, 255)

    def _apply_advanced_color_enhancement(self, image: np.ndarray) -> np.ndarray:
        """Apply advanced color enhancement."""
        if len(image.shape) != 3:
            return image

        enhanced = image.copy().astype(np.float32)

        # Advanced saturation enhancement with luminance preservation
        # Convert to YUV-like color space
        weights = np.array([0.299, 0.587, 0.114])  # Luminance weights
        luminance = np.dot(enhanced, weights)

        # Enhance chrominance
        for c in range(3):
            chrominance = enhanced[:, :, c] - luminance
            enhanced[:, :, c] = luminance + self.config.saturation_boost * chrominance

        # Adaptive contrast enhancement
        local_mean = self._apply_gaussian_blur(enhanced, sigma=2.0)
        enhanced = local_mean + self.config.contrast_enhancement * (enhanced - local_mean)

        return np.clip(enhanced, 0, 255)

    def _apply_professional_color_processing(self, image: np.ndarray) -> np.ndarray:
        """Apply professional-grade color processing."""
        if len(image.shape) != 3:
            return image

        enhanced = image.copy().astype(np.float32)

        # Color balance adjustment
        enhanced = self._apply_color_balance(enhanced)

        # Selective color enhancement
        enhanced = self._apply_selective_color_enhancement(enhanced)

        # Tone curve adjustment
        enhanced = self._apply_tone_curve(enhanced)

        return np.clip(enhanced, 0, 255)

    def _apply_adaptive_enhancement(self, image: np.ndarray, metadata: Optional[dict[str, Any]]) -> np.ndarray:
        """Apply adaptive enhancement based on image content and metadata."""
        # Analyze image characteristics
        characteristics = self._analyze_image_characteristics(image)

        # Adapt processing based on characteristics
        if characteristics["noise_level"] > 0.3:
            # High noise - apply stronger noise reduction
            image = self._apply_gaussian_blur(image, sigma=1.5)

        if characteristics["edge_density"] < 0.2:
            # Low edge density - apply stronger edge enhancement
            strength = min(1.0, self.config.edge_enhancement_strength * 1.5)
            temp_config = self.config
            temp_config.edge_enhancement_strength = strength
            image = self._apply_edge_preserving_enhancement(image)

        return image

    def _analyze_image_characteristics(self, image: np.ndarray) -> dict[str, float]:
        """Analyze image characteristics for adaptive processing."""
        if len(image.shape) == 3:
            gray = np.mean(image, axis=2)
        else:
            gray = image

        # Estimate noise level
        laplacian_var = np.var(self._convolve_2d(gray, self.laplacian_kernel))
        noise_level = min(1.0, laplacian_var / 1000.0)  # Normalize

        # Estimate edge density
        edges_x = self._convolve_2d(gray, self.sobel_x)
        edges_y = self._convolve_2d(gray, self.sobel_y)
        edge_magnitude = np.sqrt(edges_x**2 + edges_y**2)
        edge_density = np.mean(edge_magnitude > np.percentile(edge_magnitude, 80))

        # Estimate contrast
        contrast = np.std(gray) / 255.0

        return {
            "noise_level": noise_level,
            "edge_density": edge_density,
            "contrast": contrast,
            "brightness": np.mean(gray) / 255.0,
        }

    def _convolve_2d(self, image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """Apply 2D convolution."""
        # Simplified convolution implementation
        # In practice, this would use optimized implementations like scipy.ndimage

        if len(image.shape) == 3:
            result = np.zeros_like(image)
            for c in range(image.shape[2]):
                result[:, :, c] = self._convolve_2d_single(image[:, :, c], kernel)
            return result
        else:
            return self._convolve_2d_single(image, kernel)

    def _convolve_2d_single(self, image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """Apply 2D convolution to single channel."""
        h, w = image.shape
        kh, kw = kernel.shape

        # Pad image
        pad_h, pad_w = kh // 2, kw // 2
        padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode="reflect")

        # Apply convolution
        result = np.zeros_like(image)
        for i in range(h):
            for j in range(w):
                result[i, j] = np.sum(padded[i : i + kh, j : j + kw] * kernel)

        return result

    def _apply_multiscale_noise_reduction(self, image: np.ndarray) -> np.ndarray:
        """Apply multi-scale noise reduction."""
        # Process at multiple scales
        scales = [0.5, 1.0, 1.5]
        processed_scales = []

        for sigma in scales:
            denoised = self._apply_gaussian_blur(image, sigma)
            processed_scales.append(denoised)

        # Combine scales with weights
        weights = [0.2, 0.6, 0.2]  # Emphasize middle scale
        result = np.zeros_like(image, dtype=np.float32)

        for scale, weight in zip(processed_scales, weights):
            result += weight * scale.astype(np.float32)

        return result

    def _apply_non_local_means_denoising(self, image: np.ndarray) -> np.ndarray:
        """Apply non-local means denoising (simplified version)."""
        # This is a simplified implementation
        # Real implementation would be much more sophisticated
        return self._apply_bilateral_filter(image)

    def _apply_advanced_edge_enhancement(self, image: np.ndarray) -> np.ndarray:
        """Apply advanced edge enhancement with multiple operators."""
        # Combine multiple edge detection operators
        if len(image.shape) == 3:
            gray = np.mean(image, axis=2)
        else:
            gray = image

        # Sobel edges
        edges_x = self._convolve_2d(gray, self.sobel_x)
        edges_y = self._convolve_2d(gray, self.sobel_y)
        sobel_edges = np.sqrt(edges_x**2 + edges_y**2)

        # Laplacian edges
        laplacian_edges = np.abs(self._convolve_2d(gray, self.laplacian_kernel))

        # Combine edge maps
        combined_edges = 0.7 * sobel_edges + 0.3 * laplacian_edges

        # Apply enhancement
        strength = self.config.edge_enhancement_strength
        if len(image.shape) == 3:
            enhanced = image.copy().astype(np.float32)
            for c in range(image.shape[2]):
                enhanced[:, :, c] += strength * combined_edges
        else:
            enhanced = image.astype(np.float32) + strength * combined_edges

        return np.clip(enhanced, 0, 255)

    def _apply_color_balance(self, image: np.ndarray) -> np.ndarray:
        """Apply automatic color balance."""
        # Gray world assumption
        mean_r, mean_g, mean_b = np.mean(image, axis=(0, 1))
        gray_mean = (mean_r + mean_g + mean_b) / 3

        # Calculate correction factors
        r_factor = gray_mean / mean_r if mean_r > 0 else 1.0
        g_factor = gray_mean / mean_g if mean_g > 0 else 1.0
        b_factor = gray_mean / mean_b if mean_b > 0 else 1.0

        # Apply correction
        balanced = image.copy()
        balanced[:, :, 0] *= r_factor
        balanced[:, :, 1] *= g_factor
        balanced[:, :, 2] *= b_factor

        return balanced

    def _apply_selective_color_enhancement(self, image: np.ndarray) -> np.ndarray:
        """Apply selective color enhancement."""
        # Enhance specific color ranges
        enhanced = image.copy()

        # Enhance reds
        red_mask = (image[:, :, 0] > image[:, :, 1]) & (image[:, :, 0] > image[:, :, 2])
        enhanced[:, :, 0] = np.where(red_mask, np.minimum(255, image[:, :, 0] * 1.1), image[:, :, 0])

        # Enhance greens
        green_mask = (image[:, :, 1] > image[:, :, 0]) & (image[:, :, 1] > image[:, :, 2])
        enhanced[:, :, 1] = np.where(green_mask, np.minimum(255, image[:, :, 1] * 1.05), image[:, :, 1])

        return enhanced

    def _apply_tone_curve(self, image: np.ndarray) -> np.ndarray:
        """Apply tone curve adjustment."""
        # S-curve for contrast enhancement
        normalized = image / 255.0

        # Apply S-curve: y = 3x² - 2x³ (for x in [0,1])
        s_curve = 3 * normalized**2 - 2 * normalized**3

        return s_curve * 255.0

    def _update_statistics(self, processing_time: float):
        """Update processing statistics."""
        self.processing_stats["frames_processed"] += 1
        self.processing_stats["total_processing_time"] += processing_time
        self.processing_stats["average_processing_time"] = (
            self.processing_stats["total_processing_time"] / self.processing_stats["frames_processed"]
        )

    def get_statistics(self) -> dict[str, Any]:
        """Get processing statistics."""
        return self.processing_stats.copy()

    def reset_statistics(self):
        """Reset processing statistics."""
        self.processing_stats = {
            "frames_processed": 0,
            "total_processing_time": 0.0,
            "average_processing_time": 0.0,
            "quality_improvements": [],
        }
