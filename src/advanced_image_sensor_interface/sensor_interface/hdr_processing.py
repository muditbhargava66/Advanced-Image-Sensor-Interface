"""HDR Image Processing Pipeline for v2.0.0.

This module provides comprehensive HDR (High Dynamic Range) image processing
capabilities including tone mapping, exposure fusion, and HDR reconstruction.
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np
from skimage import exposure, filters

logger = logging.getLogger(__name__)


class ToneMappingMethod(Enum):
    """Available tone mapping methods."""

    REINHARD = "reinhard"
    DRAGO = "drago"
    MANTIUK = "mantiuk"
    DURAND = "durand"
    ADAPTIVE = "adaptive"
    GAMMA = "gamma"


class ExposureFusionMethod(Enum):
    """Exposure fusion methods."""

    MERTENS = "mertens"
    WEIGHTED_AVERAGE = "weighted_average"
    LAPLACIAN_PYRAMID = "laplacian_pyramid"
    GRADIENT_DOMAIN = "gradient_domain"


@dataclass
class HDRParameters:
    """HDR processing parameters."""

    # Tone mapping parameters
    tone_mapping_method: ToneMappingMethod = ToneMappingMethod.REINHARD
    gamma: float = 2.2
    exposure_compensation: float = 0.0

    # Reinhard parameters
    reinhard_intensity: float = -1.0  # Auto if -1
    reinhard_light_adapt: float = 1.0
    reinhard_color_adapt: float = 0.0

    # Drago parameters
    drago_bias: float = 0.85
    drago_saturation: float = 1.0

    # Exposure fusion parameters
    fusion_method: ExposureFusionMethod = ExposureFusionMethod.MERTENS
    contrast_weight: float = 1.0
    saturation_weight: float = 1.0
    exposure_weight: float = 1.0

    # General parameters
    preserve_color: bool = True
    apply_gamma_correction: bool = True
    output_bit_depth: int = 8

    def __post_init__(self):
        """Validate parameters."""
        if not 0.1 <= self.gamma <= 5.0:
            raise ValueError("Gamma must be between 0.1 and 5.0")

        if not -5.0 <= self.exposure_compensation <= 5.0:
            raise ValueError("Exposure compensation must be between -5.0 and 5.0")

        if self.output_bit_depth not in [8, 16, 32]:
            raise ValueError("Output bit depth must be 8, 16, or 32")


class HDRProcessor:
    """HDR image processing pipeline."""

    def __init__(self, parameters: Optional[HDRParameters] = None):
        """Initialize HDR processor.

        Args:
            parameters: HDR processing parameters
        """
        self.parameters = parameters or HDRParameters()
        logger.info(f"HDR processor initialized with {self.parameters.tone_mapping_method.value} tone mapping")

    def process_single_image(self, image: np.ndarray, exposure_value: float = 0.0) -> np.ndarray:
        """Process a single image with HDR techniques.

        Args:
            image: Input image (can be 8-bit, 16-bit, or float)
            exposure_value: Exposure value for the image

        Returns:
            HDR processed image
        """
        try:
            # Convert to float32 for processing
            if image.dtype == np.uint8:
                img_float = image.astype(np.float32) / 255.0
            elif image.dtype == np.uint16:
                img_float = image.astype(np.float32) / 65535.0
            else:
                img_float = image.astype(np.float32)

            # Apply exposure compensation
            exposure_factor = 2.0 ** (self.parameters.exposure_compensation + exposure_value)
            img_float = np.clip(img_float * exposure_factor, 0.0, 1.0)

            # Apply tone mapping
            tone_mapped = self._apply_tone_mapping(img_float)

            # Apply gamma correction if requested
            if self.parameters.apply_gamma_correction:
                tone_mapped = np.power(tone_mapped, 1.0 / self.parameters.gamma)

            # Convert to output format
            return self._convert_to_output_format(tone_mapped)

        except Exception as e:
            logger.error(f"HDR processing failed: {e}")
            return image

    def process_exposure_stack(self, images: list[np.ndarray], exposure_values: list[float]) -> np.ndarray:
        """Process a stack of images with different exposures.

        Args:
            images: List of input images
            exposure_values: List of exposure values for each image

        Returns:
            HDR fused image
        """
        if len(images) != len(exposure_values):
            raise ValueError("Number of images must match number of exposure values")

        if len(images) < 2:
            logger.warning("Only one image provided, using single image processing")
            return self.process_single_image(images[0], exposure_values[0])

        try:
            # Convert all images to float32
            float_images = []
            for img in images:
                if img.dtype == np.uint8:
                    float_images.append(img.astype(np.float32) / 255.0)
                elif img.dtype == np.uint16:
                    float_images.append(img.astype(np.float32) / 65535.0)
                else:
                    float_images.append(img.astype(np.float32))

            # Apply exposure fusion
            fused_image = self._apply_exposure_fusion(float_images, exposure_values)

            # Apply tone mapping to the fused result
            tone_mapped = self._apply_tone_mapping(fused_image)

            # Apply gamma correction if requested
            if self.parameters.apply_gamma_correction:
                tone_mapped = np.power(tone_mapped, 1.0 / self.parameters.gamma)

            # Convert to output format
            return self._convert_to_output_format(tone_mapped)

        except Exception as e:
            logger.error(f"Exposure stack processing failed: {e}")
            return images[len(images) // 2]  # Return middle exposure as fallback

    def _apply_tone_mapping(self, image: np.ndarray) -> np.ndarray:
        """Apply tone mapping to HDR image."""
        if self.parameters.tone_mapping_method == ToneMappingMethod.REINHARD:
            return self._reinhard_tone_mapping(image)
        elif self.parameters.tone_mapping_method == ToneMappingMethod.DRAGO:
            return self._drago_tone_mapping(image)
        elif self.parameters.tone_mapping_method == ToneMappingMethod.ADAPTIVE:
            return self._adaptive_tone_mapping(image)
        elif self.parameters.tone_mapping_method == ToneMappingMethod.GAMMA:
            return self._gamma_tone_mapping(image)
        else:
            logger.warning(f"Unsupported tone mapping method: {self.parameters.tone_mapping_method}")
            return self._reinhard_tone_mapping(image)

    def _reinhard_tone_mapping(self, image: np.ndarray) -> np.ndarray:
        """Apply Reinhard tone mapping."""
        # Convert to luminance
        if len(image.shape) == 3:
            luminance = 0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]
        else:
            luminance = image

        # Calculate log average luminance
        epsilon = 1e-6
        log_avg_lum = np.exp(np.mean(np.log(luminance + epsilon)))

        # Use provided intensity or calculate automatically
        if self.parameters.reinhard_intensity < 0:
            key_value = 1.03 - 2.0 / (2.0 + np.log10(log_avg_lum + 1.0))
        else:
            key_value = self.parameters.reinhard_intensity

        # Scale luminance
        scaled_lum = (key_value / log_avg_lum) * luminance

        # Apply Reinhard operator
        tone_mapped_lum = scaled_lum / (1.0 + scaled_lum)

        # Apply to color channels if RGB
        if len(image.shape) == 3:
            # Preserve color ratios
            result = np.zeros_like(image)
            for c in range(3):
                result[:, :, c] = image[:, :, c] * (tone_mapped_lum / (luminance + epsilon))
            return np.clip(result, 0.0, 1.0)
        else:
            return np.clip(tone_mapped_lum, 0.0, 1.0)

    def _drago_tone_mapping(self, image: np.ndarray) -> np.ndarray:
        """Apply Drago tone mapping."""
        # Convert to luminance
        if len(image.shape) == 3:
            luminance = 0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]
        else:
            luminance = image

        # Parameters
        bias = self.parameters.drago_bias
        max_lum = np.max(luminance)

        if max_lum <= 0:
            return image

        # Apply Drago operator
        log_lum = np.log10(luminance + 1e-6)
        log_max = np.log10(max_lum)

        tone_mapped_lum = (log_lum / log_max) / (
            np.log10(2.0 + 8.0 * ((luminance / max_lum) ** (np.log10(bias) / np.log10(0.5))))
        )
        tone_mapped_lum = np.clip(tone_mapped_lum, 0.0, 1.0)

        # Apply to color channels if RGB
        if len(image.shape) == 3:
            result = np.zeros_like(image)
            for c in range(3):
                result[:, :, c] = image[:, :, c] * (tone_mapped_lum / (luminance + 1e-6))
            return np.clip(result, 0.0, 1.0)
        else:
            return tone_mapped_lum

    def _adaptive_tone_mapping(self, image: np.ndarray) -> np.ndarray:
        """Apply adaptive tone mapping using local adaptation."""
        # Use adaptive histogram equalization
        if len(image.shape) == 3:
            # Process each channel separately
            result = np.zeros_like(image)
            for c in range(3):
                result[:, :, c] = exposure.equalize_adapthist(image[:, :, c], clip_limit=0.03)
            return result
        else:
            return exposure.equalize_adapthist(image, clip_limit=0.03)

    def _gamma_tone_mapping(self, image: np.ndarray) -> np.ndarray:
        """Apply simple gamma tone mapping."""
        return np.power(image, 1.0 / self.parameters.gamma)

    def _apply_exposure_fusion(self, images: list[np.ndarray], exposure_values: list[float]) -> np.ndarray:
        """Apply exposure fusion to combine multiple exposures."""
        if self.parameters.fusion_method == ExposureFusionMethod.MERTENS:
            return self._mertens_fusion(images)
        elif self.parameters.fusion_method == ExposureFusionMethod.WEIGHTED_AVERAGE:
            return self._weighted_average_fusion(images, exposure_values)
        else:
            logger.warning(f"Unsupported fusion method: {self.parameters.fusion_method}")
            return self._mertens_fusion(images)

    def _mertens_fusion(self, images: list[np.ndarray]) -> np.ndarray:
        """Apply Mertens exposure fusion algorithm."""
        weights = []

        for img in images:
            # Calculate contrast weight
            if len(img.shape) == 3:
                gray = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
            else:
                gray = img

            contrast = np.abs(filters.laplace(gray))

            # Calculate saturation weight
            if len(img.shape) == 3:
                mean_rgb = np.mean(img, axis=2)
                saturation = np.sqrt(np.sum((img - mean_rgb[:, :, np.newaxis]) ** 2, axis=2) / 3.0)
            else:
                saturation = np.zeros_like(gray)

            # Calculate well-exposedness weight
            sigma = 0.2
            well_exposed = np.exp(-0.5 * ((gray - 0.5) / sigma) ** 2)

            # Combine weights
            weight = (
                contrast**self.parameters.contrast_weight
                * saturation**self.parameters.saturation_weight
                * well_exposed**self.parameters.exposure_weight
            )

            weights.append(weight)

        # Normalize weights
        total_weight = np.sum(weights, axis=0)
        total_weight[total_weight == 0] = 1e-6

        # Fuse images
        result = np.zeros_like(images[0])
        for i, img in enumerate(images):
            if len(img.shape) == 3:
                for c in range(3):
                    result[:, :, c] += img[:, :, c] * weights[i] / total_weight
            else:
                result += img * weights[i] / total_weight

        return np.clip(result, 0.0, 1.0)

    def _weighted_average_fusion(self, images: list[np.ndarray], exposure_values: list[float]) -> np.ndarray:
        """Apply weighted average fusion based on exposure values."""
        # Calculate weights based on exposure values
        weights = []
        for ev in exposure_values:
            # Weight based on distance from optimal exposure (0 EV)
            weight = np.exp(-0.5 * (ev / 2.0) ** 2)
            weights.append(weight)

        # Normalize weights
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]

        # Fuse images
        result = np.zeros_like(images[0])
        for i, img in enumerate(images):
            result += img * weights[i]

        return np.clip(result, 0.0, 1.0)

    def _convert_to_output_format(self, image: np.ndarray) -> np.ndarray:
        """Convert processed image to output format."""
        if self.parameters.output_bit_depth == 8:
            return (np.clip(image, 0.0, 1.0) * 255).astype(np.uint8)
        elif self.parameters.output_bit_depth == 16:
            return (np.clip(image, 0.0, 1.0) * 65535).astype(np.uint16)
        else:  # 32-bit float
            return np.clip(image, 0.0, 1.0).astype(np.float32)

    def get_processing_stats(self) -> dict:
        """Get HDR processing statistics.

        Returns:
            Dictionary containing processing statistics
        """
        return {
            "tone_mapping_method": self.parameters.tone_mapping_method.value,
            "fusion_method": self.parameters.fusion_method.value,
            "gamma": self.parameters.gamma,
            "exposure_compensation": self.parameters.exposure_compensation,
            "output_bit_depth": self.parameters.output_bit_depth,
            "preserve_color": self.parameters.preserve_color,
        }


def create_hdr_processor_for_automotive() -> HDRProcessor:
    """Create HDR processor optimized for automotive applications.

    Returns:
        HDRProcessor configured for automotive use
    """
    params = HDRParameters(
        tone_mapping_method=ToneMappingMethod.ADAPTIVE,
        gamma=2.2,
        exposure_compensation=0.5,
        fusion_method=ExposureFusionMethod.MERTENS,
        contrast_weight=1.5,
        saturation_weight=1.0,
        exposure_weight=1.2,
        output_bit_depth=8,
    )
    return HDRProcessor(params)
