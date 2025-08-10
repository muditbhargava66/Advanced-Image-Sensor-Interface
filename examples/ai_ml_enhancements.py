#!/usr/bin/env python3
"""
AI and Machine Learning Enhancements Example

This example demonstrates the AI/ML features implemented in the Advanced Image Sensor Interface,
including intelligent noise reduction, adaptive processing, and machine learning-based optimizations.

Author: Advanced Image Sensor Interface Team
Version: 2.0.0
"""

import argparse
import logging
import sys
import time
from typing import Any

import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

try:
    from advanced_image_sensor_interface import EnhancedSensorInterface, HDRProcessor, RAWProcessor
    from advanced_image_sensor_interface.sensor_interface.gpu_acceleration import GPUAccelerator
    from advanced_image_sensor_interface.utils.performance_metrics import calculate_dynamic_range, calculate_snr
except ImportError as e:
    logger.error(f"Failed to import required modules: {e}")
    logger.error("Please install the package with: pip install -e .")
    sys.exit(1)


class AIEnhancedProcessor:
    """
    AI-Enhanced Image Processor with machine learning capabilities.

    This class demonstrates AI/ML enhancements including:
    - Intelligent noise reduction using adaptive algorithms
    - Scene-aware processing parameter optimization
    - Predictive quality assessment
    - Adaptive HDR tone mapping
    """

    def __init__(self):
        """Initialize the AI-enhanced processor."""
        self.gpu_accelerator = GPUAccelerator()
        self.processing_history = []
        self.scene_classifier = SceneClassifier()
        self.noise_predictor = NoisePredictor()
        self.quality_assessor = QualityAssessor()

        logger.info("AI-Enhanced Processor initialized")

    def intelligent_noise_reduction(self, image: np.ndarray, scene_type: str = "auto") -> np.ndarray:
        """
        Apply intelligent noise reduction based on scene analysis.

        Args:
            image: Input image array
            scene_type: Scene type ("auto", "portrait", "landscape", "night", "sports")

        Returns:
            Noise-reduced image
        """
        logger.info(f"Applying intelligent noise reduction (scene: {scene_type})")

        # Analyze scene if auto mode
        if scene_type == "auto":
            scene_type = self.scene_classifier.classify_scene(image)
            logger.info(f"Auto-detected scene type: {scene_type}")

        # Predict optimal noise reduction parameters
        noise_params = self.noise_predictor.predict_parameters(image, scene_type)
        logger.info(f"Predicted noise parameters: {noise_params}")

        # Apply scene-specific noise reduction
        if scene_type == "portrait":
            # Preserve skin texture while reducing noise
            processed = self._portrait_noise_reduction(image, noise_params)
        elif scene_type == "landscape":
            # Preserve fine details in textures
            processed = self._landscape_noise_reduction(image, noise_params)
        elif scene_type == "night":
            # Aggressive noise reduction for low light
            processed = self._night_noise_reduction(image, noise_params)
        elif scene_type == "sports":
            # Fast processing with motion blur consideration
            processed = self._sports_noise_reduction(image, noise_params)
        else:
            # General purpose noise reduction
            processed = self._general_noise_reduction(image, noise_params)

        # Assess quality improvement
        quality_improvement = self.quality_assessor.assess_improvement(image, processed)
        logger.info(f"Quality improvement: {quality_improvement:.2f}%")

        return processed

    def adaptive_hdr_processing(self, images: list, scene_analysis: bool = True) -> np.ndarray:
        """
        Apply adaptive HDR processing with ML-based tone mapping.

        Args:
            images: List of exposure-bracketed images
            scene_analysis: Whether to perform scene analysis for optimization

        Returns:
            HDR processed image
        """
        logger.info("Starting adaptive HDR processing")

        if scene_analysis:
            # Analyze scene characteristics
            scene_info = self.scene_classifier.analyze_hdr_scene(images)
            logger.info(f"HDR scene analysis: {scene_info}")

            # Optimize HDR parameters based on scene
            hdr_params = self._optimize_hdr_parameters(scene_info)
        else:
            hdr_params = self._default_hdr_parameters()

        # Initialize HDR processor with optimized parameters
        from advanced_image_sensor_interface.sensor_interface.hdr_processing import HDRParameters, ToneMappingMethod

        # Map string to enum
        tone_mapping_map = {
            "adaptive": ToneMappingMethod.ADAPTIVE,
            "reinhard": ToneMappingMethod.REINHARD,
            "drago": ToneMappingMethod.DRAGO,
            "mantiuk": ToneMappingMethod.MANTIUK,
        }

        tone_mapping_method = tone_mapping_map.get(hdr_params["tone_mapping"], ToneMappingMethod.ADAPTIVE)

        hdr_parameters = HDRParameters(
            tone_mapping_method=tone_mapping_method,
            gamma=hdr_params["gamma"],
            exposure_compensation=hdr_params["exposure_compensation"],
        )
        hdr_processor = HDRProcessor(hdr_parameters)

        # Process HDR with GPU acceleration if available
        exposure_values = [-2.0, 0.0, 2.0]  # EV values for the images

        # Check if GPU is available by checking backend
        from advanced_image_sensor_interface.sensor_interface.gpu_acceleration import GPUBackend

        if self.gpu_accelerator.backend != GPUBackend.NONE:
            logger.info("Using GPU acceleration for HDR processing")
            processed = self._gpu_hdr_processing(hdr_processor, images, exposure_values)
        else:
            logger.info("Using CPU for HDR processing")
            processed = hdr_processor.process_exposure_stack(images, exposure_values)

        return processed

    def predictive_quality_assessment(self, image: np.ndarray) -> dict[str, float]:
        """
        Perform predictive quality assessment using ML models.

        Args:
            image: Input image for quality assessment

        Returns:
            Dictionary of quality metrics and predictions
        """
        logger.info("Performing predictive quality assessment")

        # Calculate traditional metrics
        snr = calculate_snr(image.astype(np.float32), np.random.normal(0, 5, image.shape).astype(np.float32))
        dynamic_range = calculate_dynamic_range(image)

        # ML-based quality predictions
        sharpness_score = self.quality_assessor.predict_sharpness(image)
        noise_level = self.quality_assessor.predict_noise_level(image)
        color_accuracy = self.quality_assessor.predict_color_accuracy(image)
        overall_quality = self.quality_assessor.predict_overall_quality(image)

        # Predict potential improvements
        improvement_potential = self.quality_assessor.predict_improvement_potential(image)

        quality_metrics = {
            "snr_db": snr,
            "dynamic_range_db": dynamic_range,
            "sharpness_score": sharpness_score,
            "noise_level": noise_level,
            "color_accuracy": color_accuracy,
            "overall_quality": overall_quality,
            "improvement_potential": improvement_potential,
        }

        logger.info(f"Quality assessment complete: {quality_metrics}")
        return quality_metrics

    def adaptive_processing_pipeline(self, image: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        """
        Run adaptive processing pipeline that adjusts based on image content.

        Args:
            image: Input image

        Returns:
            Tuple of (processed_image, processing_stats)
        """
        logger.info("Starting adaptive processing pipeline")
        start_time = time.time()

        # Initial quality assessment
        initial_quality = self.predictive_quality_assessment(image)

        # Scene classification
        scene_type = self.scene_classifier.classify_scene(image)

        # Adaptive noise reduction
        denoised = self.intelligent_noise_reduction(image, scene_type)

        # Adaptive sharpening based on content
        sharpened = self._adaptive_sharpening(denoised, scene_type)

        # Adaptive color enhancement
        color_enhanced = self._adaptive_color_enhancement(sharpened, scene_type)

        # Final quality assessment
        final_quality = self.predictive_quality_assessment(color_enhanced)

        processing_time = time.time() - start_time

        processing_stats = {
            "scene_type": scene_type,
            "initial_quality": initial_quality,
            "final_quality": final_quality,
            "processing_time": processing_time,
            "quality_improvement": final_quality["overall_quality"] - initial_quality["overall_quality"],
        }

        logger.info(f"Adaptive processing complete in {processing_time:.3f}s")
        logger.info(f"Quality improvement: {processing_stats['quality_improvement']:.2f}")

        return color_enhanced, processing_stats

    # Private helper methods
    def _portrait_noise_reduction(self, image: np.ndarray, params: dict) -> np.ndarray:
        """Portrait-specific noise reduction preserving skin texture."""
        # Simulate advanced portrait noise reduction
        params.get("kernel_size", 3)
        strength = params.get("strength", 0.3)

        # Apply bilateral filter to preserve edges (skin texture)
        from scipy.ndimage import gaussian_filter

        smoothed = gaussian_filter(image.astype(np.float32), sigma=strength)

        # Blend with original to preserve texture
        alpha = 0.7
        result = alpha * smoothed + (1 - alpha) * image.astype(np.float32)

        return np.clip(result, 0, 255).astype(image.dtype)

    def _landscape_noise_reduction(self, image: np.ndarray, params: dict) -> np.ndarray:
        """Landscape-specific noise reduction preserving fine details."""
        # Simulate edge-preserving noise reduction
        from scipy.ndimage import median_filter

        strength = params.get("strength", 0.2)
        kernel_size = params.get("kernel_size", 3)

        # Use median filter to preserve edges
        filtered = median_filter(image, size=kernel_size)

        # Blend based on edge strength
        alpha = strength
        result = alpha * filtered + (1 - alpha) * image

        return result.astype(image.dtype)

    def _night_noise_reduction(self, image: np.ndarray, params: dict) -> np.ndarray:
        """Night-specific aggressive noise reduction."""
        from scipy.ndimage import gaussian_filter

        strength = params.get("strength", 0.8)

        # Aggressive smoothing for night scenes
        smoothed = gaussian_filter(image.astype(np.float32), sigma=strength * 2)

        return np.clip(smoothed, 0, 255).astype(image.dtype)

    def _sports_noise_reduction(self, image: np.ndarray, params: dict) -> np.ndarray:
        """Sports-specific fast noise reduction."""
        # Fast processing for sports scenes
        strength = params.get("strength", 0.1)

        # Use gaussian filter for fast processing
        from scipy.ndimage import gaussian_filter

        if image.ndim == 3:
            smoothed = np.zeros_like(image, dtype=np.float32)
            for c in range(image.shape[2]):
                smoothed[:, :, c] = gaussian_filter(image[:, :, c].astype(np.float32), sigma=strength)
        else:
            smoothed = gaussian_filter(image.astype(np.float32), sigma=strength)

        alpha = strength
        result = alpha * smoothed + (1 - alpha) * image.astype(np.float32)

        return np.clip(result, 0, 255).astype(image.dtype)

    def _general_noise_reduction(self, image: np.ndarray, params: dict) -> np.ndarray:
        """General purpose noise reduction."""
        from scipy.ndimage import gaussian_filter

        strength = params.get("strength", 0.4)
        smoothed = gaussian_filter(image.astype(np.float32), sigma=strength)

        return np.clip(smoothed, 0, 255).astype(image.dtype)

    def _optimize_hdr_parameters(self, scene_info: dict) -> dict:
        """Optimize HDR parameters based on scene analysis."""
        base_params = self._default_hdr_parameters()

        # Adjust based on scene characteristics
        if scene_info.get("high_contrast", False):
            base_params["tone_mapping"] = "adaptive"
            base_params["gamma"] = 2.4

        if scene_info.get("low_light", False):
            base_params["exposure_compensation"] = 0.8

        return base_params

    def _default_hdr_parameters(self) -> dict:
        """Default HDR processing parameters."""
        return {"tone_mapping": "adaptive", "gamma": 2.2, "exposure_compensation": 0.5}

    def _gpu_hdr_processing(self, hdr_processor, images: list, exposure_values: list) -> np.ndarray:
        """GPU-accelerated HDR processing."""
        # Simulate GPU acceleration
        return hdr_processor.process_exposure_stack(images, exposure_values)

    def _adaptive_sharpening(self, image: np.ndarray, scene_type: str) -> np.ndarray:
        """Apply adaptive sharpening based on scene type."""
        from scipy.ndimage import laplace

        # Adjust sharpening strength based on scene
        strength_map = {"portrait": 0.2, "landscape": 0.5, "night": 0.1, "sports": 0.3, "general": 0.3}

        strength = strength_map.get(scene_type, 0.3)

        # Apply unsharp masking
        laplacian = laplace(image.astype(np.float32))
        sharpened = image.astype(np.float32) - strength * laplacian

        return np.clip(sharpened, 0, 255).astype(image.dtype)

    def _adaptive_color_enhancement(self, image: np.ndarray, scene_type: str) -> np.ndarray:
        """Apply adaptive color enhancement based on scene type."""
        # Scene-specific color enhancement
        if scene_type == "portrait":
            # Warm up skin tones
            enhanced = self._warm_color_enhancement(image)
        elif scene_type == "landscape":
            # Enhance greens and blues
            enhanced = self._landscape_color_enhancement(image)
        elif scene_type == "night":
            # Reduce color noise
            enhanced = self._night_color_enhancement(image)
        else:
            # General color enhancement
            enhanced = self._general_color_enhancement(image)

        return enhanced

    def _warm_color_enhancement(self, image: np.ndarray) -> np.ndarray:
        """Warm color enhancement for portraits."""
        if len(image.shape) == 3:
            enhanced = image.copy().astype(np.float32)
            enhanced[:, :, 0] *= 1.1  # Enhance red channel
            enhanced[:, :, 1] *= 1.05  # Slightly enhance green
            return np.clip(enhanced, 0, 255).astype(image.dtype)
        return image

    def _landscape_color_enhancement(self, image: np.ndarray) -> np.ndarray:
        """Color enhancement for landscapes."""
        if len(image.shape) == 3:
            enhanced = image.copy().astype(np.float32)
            enhanced[:, :, 1] *= 1.15  # Enhance green channel
            enhanced[:, :, 2] *= 1.1  # Enhance blue channel
            return np.clip(enhanced, 0, 255).astype(image.dtype)
        return image

    def _night_color_enhancement(self, image: np.ndarray) -> np.ndarray:
        """Color enhancement for night scenes."""
        # Reduce color saturation to minimize noise
        if len(image.shape) == 3:
            gray = np.mean(image, axis=2, keepdims=True)
            enhanced = 0.8 * image + 0.2 * gray
            return enhanced.astype(image.dtype)
        return image

    def _general_color_enhancement(self, image: np.ndarray) -> np.ndarray:
        """General color enhancement."""
        if len(image.shape) == 3:
            enhanced = image.copy().astype(np.float32)
            enhanced *= 1.05  # Slight overall enhancement
            return np.clip(enhanced, 0, 255).astype(image.dtype)
        return image


class SceneClassifier:
    """ML-based scene classifier for adaptive processing."""

    def classify_scene(self, image: np.ndarray) -> str:
        """
        Classify scene type based on image characteristics.

        Args:
            image: Input image

        Returns:
            Scene type string
        """
        # Simulate ML-based scene classification
        # In a real implementation, this would use a trained CNN

        # Simple heuristic-based classification for demo
        if len(image.shape) == 3:
            # Analyze color distribution
            mean_brightness = np.mean(image)
            color_variance = np.var(image, axis=(0, 1))

            if mean_brightness < 50:
                return "night"
            elif np.mean(color_variance) < 100:
                return "portrait"
            elif color_variance[1] > color_variance[0] and color_variance[1] > color_variance[2]:
                return "landscape"
            else:
                return "general"
        else:
            return "general"

    def analyze_hdr_scene(self, images: list) -> dict[str, Any]:
        """
        Analyze HDR scene characteristics.

        Args:
            images: List of exposure-bracketed images

        Returns:
            Scene analysis dictionary
        """
        if not images:
            return {}

        # Analyze exposure range
        brightnesses = [np.mean(img) for img in images]
        exposure_range = max(brightnesses) - min(brightnesses)

        # Detect high contrast scenes
        high_contrast = exposure_range > 100

        # Detect low light scenes
        low_light = min(brightnesses) < 30

        return {
            "high_contrast": high_contrast,
            "low_light": low_light,
            "exposure_range": exposure_range,
            "num_exposures": len(images),
        }


class NoisePredictor:
    """ML-based noise parameter predictor."""

    def predict_parameters(self, image: np.ndarray, scene_type: str) -> dict[str, float]:
        """
        Predict optimal noise reduction parameters.

        Args:
            image: Input image
            scene_type: Scene type

        Returns:
            Dictionary of predicted parameters
        """
        # Simulate ML-based parameter prediction
        # In reality, this would use a trained regression model

        # Estimate noise level
        noise_level = self._estimate_noise_level(image)

        # Scene-specific parameter adjustment
        base_strength = min(noise_level / 50.0, 1.0)

        scene_adjustments = {
            "portrait": {"strength": base_strength * 0.7, "kernel_size": 3},
            "landscape": {"strength": base_strength * 0.5, "kernel_size": 3},
            "night": {"strength": base_strength * 1.2, "kernel_size": 5},
            "sports": {"strength": base_strength * 0.3, "kernel_size": 3},
            "general": {"strength": base_strength, "kernel_size": 3},
        }

        return scene_adjustments.get(scene_type, scene_adjustments["general"])

    def _estimate_noise_level(self, image: np.ndarray) -> float:
        """Estimate noise level in image."""
        # Simple noise estimation using Laplacian variance
        from scipy.ndimage import laplace

        if len(image.shape) == 3:
            gray = np.mean(image, axis=2)
        else:
            gray = image

        laplacian_var = np.var(laplace(gray))
        return min(laplacian_var, 100.0)  # Cap at reasonable value


class QualityAssessor:
    """ML-based image quality assessor."""

    def predict_sharpness(self, image: np.ndarray) -> float:
        """Predict image sharpness score."""
        from scipy.ndimage import laplace

        if len(image.shape) == 3:
            gray = np.mean(image, axis=2)
        else:
            gray = image

        # Use Laplacian variance as sharpness metric
        sharpness = np.var(laplace(gray))
        return min(sharpness / 1000.0, 1.0)  # Normalize to 0-1

    def predict_noise_level(self, image: np.ndarray) -> float:
        """Predict noise level score."""
        # Estimate noise using high-frequency content
        from scipy.ndimage import gaussian_filter

        if len(image.shape) == 3:
            gray = np.mean(image, axis=2)
        else:
            gray = image

        # High-pass filter to isolate noise
        smoothed = gaussian_filter(gray, sigma=1.0)
        noise = gray - smoothed
        noise_level = np.std(noise)

        return min(noise_level / 50.0, 1.0)  # Normalize to 0-1

    def predict_color_accuracy(self, image: np.ndarray) -> float:
        """Predict color accuracy score."""
        if len(image.shape) != 3:
            return 0.5  # Neutral score for grayscale

        # Simple color balance assessment
        channel_means = np.mean(image, axis=(0, 1))
        color_balance = 1.0 - np.std(channel_means) / np.mean(channel_means)

        return max(0.0, min(1.0, color_balance))

    def predict_overall_quality(self, image: np.ndarray) -> float:
        """Predict overall image quality score."""
        sharpness = self.predict_sharpness(image)
        noise_level = self.predict_noise_level(image)
        color_accuracy = self.predict_color_accuracy(image)

        # Weighted combination
        overall = 0.4 * sharpness + 0.3 * (1.0 - noise_level) + 0.3 * color_accuracy
        return overall

    def predict_improvement_potential(self, image: np.ndarray) -> float:
        """Predict potential for quality improvement."""
        current_quality = self.predict_overall_quality(image)

        # Higher potential for lower quality images
        potential = 1.0 - current_quality
        return potential

    def assess_improvement(self, original: np.ndarray, processed: np.ndarray) -> float:
        """Assess quality improvement between original and processed images."""
        original_quality = self.predict_overall_quality(original)
        processed_quality = self.predict_overall_quality(processed)

        improvement = ((processed_quality - original_quality) / original_quality) * 100
        return improvement


def demo_ai_noise_reduction(resolution: tuple[int, int] = (1920, 1080)):
    """Demonstrate AI-enhanced noise reduction."""
    logger.info("=== AI-Enhanced Noise Reduction Demo ===")

    width, height = resolution
    processor = AIEnhancedProcessor()

    # Generate test images for different scenes
    scenes = ["portrait", "landscape", "night", "sports"]

    for scene in scenes:
        logger.info(f"Testing {scene} scene...")

        # Generate scene-appropriate test image
        if scene == "portrait":
            # Skin-tone dominated image
            base_color = [200, 150, 120]  # Skin tone
        elif scene == "landscape":
            # Green/blue dominated image
            base_color = [100, 180, 120]  # Nature colors
        elif scene == "night":
            # Dark image with noise
            base_color = [30, 30, 40]  # Dark scene
        else:  # sports
            # High contrast image
            base_color = [150, 150, 150]  # Neutral

        # Create test image
        test_image = np.full((height, width, 3), base_color, dtype=np.uint8)

        # Add realistic noise
        noise_level = 20 if scene == "night" else 10
        noise = np.random.normal(0, noise_level, test_image.shape)
        noisy_image = np.clip(test_image.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        # Apply AI-enhanced noise reduction
        start_time = time.time()
        denoised = processor.intelligent_noise_reduction(noisy_image, scene)
        processing_time = time.time() - start_time

        # Calculate improvement
        original_snr = calculate_snr(test_image.astype(np.float32), noise.astype(np.float32))
        denoised_noise = noisy_image.astype(np.float32) - denoised.astype(np.float32)
        improved_snr = calculate_snr(denoised.astype(np.float32), denoised_noise)

        logger.info(f"  Processing time: {processing_time:.3f}s")
        logger.info(f"  Original SNR: {original_snr:.2f} dB")
        logger.info(f"  Improved SNR: {improved_snr:.2f} dB")
        logger.info(f"  SNR improvement: {improved_snr - original_snr:.2f} dB")


def demo_adaptive_hdr(resolution: tuple[int, int] = (1920, 1080)):
    """Demonstrate adaptive HDR processing."""
    logger.info("=== Adaptive HDR Processing Demo ===")

    width, height = resolution
    processor = AIEnhancedProcessor()

    # Generate exposure-bracketed images
    exposures = [-2, 0, 2]  # EV values
    test_images = []

    for ev in exposures:
        # Create exposure-varied image
        base_intensity = 128 + (ev * 40)
        base_intensity = np.clip(base_intensity, 0, 255)

        # Add some scene structure
        x, y = np.meshgrid(np.linspace(0, 1, width), np.linspace(0, 1, height))
        pattern = np.sin(x * 8) * np.cos(y * 8) * 60 + base_intensity

        image = np.clip(pattern, 0, 255).astype(np.uint8)
        rgb_image = np.stack([image, image, image], axis=2)
        test_images.append(rgb_image)

        logger.info(f"Generated exposure {ev:+d} EV: mean={np.mean(rgb_image):.1f}")

    # Process with adaptive HDR
    start_time = time.time()
    hdr_result = processor.adaptive_hdr_processing(test_images, scene_analysis=True)
    processing_time = time.time() - start_time

    # Calculate metrics
    dynamic_range = calculate_dynamic_range(hdr_result)

    logger.info(f"HDR processing completed in {processing_time:.3f}s")
    logger.info(f"Output shape: {hdr_result.shape}, dtype: {hdr_result.dtype}")
    logger.info(f"Dynamic range: {dynamic_range:.2f} dB")


def demo_quality_assessment(resolution: tuple[int, int] = (1920, 1080)):
    """Demonstrate predictive quality assessment."""
    logger.info("=== Predictive Quality Assessment Demo ===")

    width, height = resolution
    processor = AIEnhancedProcessor()

    # Generate test images with different quality characteristics
    test_cases = [
        ("high_quality", {"noise": 5, "blur": 0}),
        ("noisy", {"noise": 25, "blur": 0}),
        ("blurry", {"noise": 5, "blur": 2}),
        ("low_quality", {"noise": 30, "blur": 3}),
    ]

    for case_name, params in test_cases:
        logger.info(f"Testing {case_name} image...")

        # Generate test image
        base_image = np.random.randint(50, 200, (height, width, 3), dtype=np.uint8)

        # Add noise
        if params["noise"] > 0:
            noise = np.random.normal(0, params["noise"], base_image.shape)
            base_image = np.clip(base_image.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        # Add blur
        if params["blur"] > 0:
            from scipy.ndimage import gaussian_filter

            for c in range(3):
                base_image[:, :, c] = gaussian_filter(base_image[:, :, c], sigma=params["blur"])

        # Assess quality
        quality_metrics = processor.predictive_quality_assessment(base_image)

        logger.info(f"  Quality metrics for {case_name}:")
        for metric, value in quality_metrics.items():
            logger.info(f"    {metric}: {value:.3f}")


def demo_adaptive_pipeline(resolution: tuple[int, int] = (1920, 1080)):
    """Demonstrate full adaptive processing pipeline."""
    logger.info("=== Adaptive Processing Pipeline Demo ===")

    width, height = resolution
    processor = AIEnhancedProcessor()

    # Generate test image with mixed characteristics
    test_image = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)

    # Add some realistic degradations
    # Add noise
    noise = np.random.normal(0, 15, test_image.shape)
    test_image = np.clip(test_image.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    # Add slight blur
    from scipy.ndimage import gaussian_filter

    for c in range(3):
        test_image[:, :, c] = gaussian_filter(test_image[:, :, c], sigma=0.5)

    logger.info(f"Input image: {test_image.shape}, dtype: {test_image.dtype}")

    # Run adaptive pipeline
    processed_image, stats = processor.adaptive_processing_pipeline(test_image)

    logger.info("Pipeline Results:")
    logger.info(f"  Scene type: {stats['scene_type']}")
    logger.info(f"  Processing time: {stats['processing_time']:.3f}s")
    logger.info(f"  Quality improvement: {stats['quality_improvement']:.3f}")
    logger.info(f"  Initial quality: {stats['initial_quality']['overall_quality']:.3f}")
    logger.info(f"  Final quality: {stats['final_quality']['overall_quality']:.3f}")


def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(
        description="AI and Machine Learning Enhancements Demo", formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument("--resolution", type=str, default="1920x1080", help="Image resolution in WIDTHxHEIGHT format")

    parser.add_argument("--demo", choices=["noise", "hdr", "quality", "pipeline", "all"], default="all", help="Which demo to run")

    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        # Parse resolution
        width_str, height_str = args.resolution.split("x")
        resolution = (int(width_str), int(height_str))

        logger.info("AI and Machine Learning Enhancements Demo")
        logger.info("=" * 50)
        logger.info(f"Resolution: {resolution[0]}x{resolution[1]}")
        logger.info("=" * 50)

        # Run selected demos
        if args.demo in ["noise", "all"]:
            demo_ai_noise_reduction(resolution)

        if args.demo in ["hdr", "all"]:
            demo_adaptive_hdr(resolution)

        if args.demo in ["quality", "all"]:
            demo_quality_assessment(resolution)

        if args.demo in ["pipeline", "all"]:
            demo_adaptive_pipeline(resolution)

        logger.info("=" * 50)
        logger.info("✓ AI/ML Enhancement demos completed successfully!")

    except Exception as e:
        logger.error(f"✗ Demo failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
