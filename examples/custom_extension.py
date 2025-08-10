#!/usr/bin/env python3
"""
Custom Extension Example

This example demonstrates how to extend the Advanced Image Sensor Interface
with custom implementations, including custom MIPI drivers, noise reducers,
and processing algorithms.

Author: Advanced Image Sensor Interface Team
Version: 2.0.0
"""

import logging
import sys
import time
from typing import Any, Optional

import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

try:
    from advanced_image_sensor_interface import MIPIConfig, MIPIDriver
    from advanced_image_sensor_interface.utils.noise_reduction import (
        NoiseReducer,
        NoiseReducerFactory,
        NoiseReductionConfig,
        NoiseType,
    )
except ImportError as e:
    logger.error(f"Failed to import required modules: {e}")
    logger.error("Please install the package with: pip install -e .")
    sys.exit(1)


# Example 1: Custom MIPI Driver Extension
class HighSpeedMIPIDriver(MIPIDriver):
    """
    Custom MIPI driver with high-speed optimizations.

    This example shows how to extend the base MIPIDriver class
    to add custom functionality for specific hardware or use cases.
    """

    def __init__(self, config: MIPIConfig, optimization_level: int = 1):
        """
        Initialize high-speed MIPI driver.

        Args:
            config: MIPI configuration
            optimization_level: Optimization level (1-3)
        """
        super().__init__(config)
        self.optimization_level = optimization_level
        self.burst_mode = False
        self.compression_enabled = False

        logger.info(f"High-speed MIPI driver initialized (optimization level: {optimization_level})")

    def enable_burst_mode(self, enable: bool = True) -> bool:
        """
        Enable/disable burst mode for high-speed transfers.

        Args:
            enable: Whether to enable burst mode

        Returns:
            True if successful
        """
        try:
            self.burst_mode = enable
            if enable:
                # Simulate burst mode configuration
                logger.info("✓ Burst mode enabled - increased throughput by 25%")
                # In real implementation, this would configure hardware registers
            else:
                logger.info("✓ Burst mode disabled")
            return True
        except Exception as e:
            logger.error(f"✗ Failed to configure burst mode: {e}")
            return False

    def enable_compression(self, enable: bool = True, ratio: float = 0.7) -> bool:
        """
        Enable/disable data compression.

        Args:
            enable: Whether to enable compression
            ratio: Compression ratio (0.1 to 1.0)

        Returns:
            True if successful
        """
        try:
            self.compression_enabled = enable
            self.compression_ratio = ratio if enable else 1.0

            if enable:
                logger.info(f"✓ Compression enabled (ratio: {ratio:.1f})")
            else:
                logger.info("✓ Compression disabled")
            return True
        except Exception as e:
            logger.error(f"✗ Failed to configure compression: {e}")
            return False

    def send_data(self, data: bytes) -> bool:
        """
        Enhanced send_data with optimizations.

        Args:
            data: Data to send

        Returns:
            True if successful
        """
        try:
            # Apply compression if enabled
            if self.compression_enabled:
                # Simulate compression
                compressed_size = int(len(data) * self.compression_ratio)
                logger.debug(f"Compressed data: {len(data)} -> {compressed_size} bytes")

            # Apply burst mode optimizations
            if self.burst_mode:
                # Simulate burst transfer
                transfer_time = len(data) / (self.config.data_rate * 1e9 / 8) * 0.75  # 25% faster
            else:
                transfer_time = len(data) / (self.config.data_rate * 1e9 / 8)

            # Simulate transfer delay
            time.sleep(min(transfer_time, 0.001))  # Cap simulation delay

            # Call parent implementation
            return super().send_data(data)

        except Exception as e:
            logger.error(f"✗ Enhanced send_data failed: {e}")
            return False

    def get_enhanced_status(self) -> dict[str, Any]:
        """
        Get enhanced status including custom features.

        Returns:
            Enhanced status dictionary
        """
        status = self.get_status()
        status.update(
            {
                "optimization_level": self.optimization_level,
                "burst_mode": self.burst_mode,
                "compression_enabled": self.compression_enabled,
                "compression_ratio": getattr(self, "compression_ratio", 1.0),
                "driver_type": "HighSpeedMIPIDriver",
            }
        )
        return status


# Example 2: Custom Noise Reducer
class AINoiseReducer(NoiseReducer):
    """
    AI-based noise reducer using machine learning techniques.

    This example demonstrates how to implement a custom noise reduction
    algorithm by inheriting from the NoiseReducer base class.
    """

    def __init__(self, config: NoiseReductionConfig):
        """Initialize AI noise reducer."""
        super().__init__(config)
        self.model_loaded = False
        self.training_data = []

        # Simulate model loading
        self._load_model()

    def get_algorithm_name(self) -> str:
        """Get algorithm name."""
        return "AI-Based Adaptive Noise Reduction"

    def _load_model(self) -> None:
        """Load or initialize the AI model."""
        try:
            # Simulate model loading
            logger.info("Loading AI noise reduction model...")
            time.sleep(0.1)  # Simulate loading time

            # In a real implementation, you would load a trained model here
            # For example: self.model = torch.load('noise_reduction_model.pth')

            self.model_loaded = True
            logger.info("✓ AI model loaded successfully")
        except Exception as e:
            logger.error(f"✗ Failed to load AI model: {e}")
            self.model_loaded = False

    def process(self, image: np.ndarray) -> np.ndarray:
        """
        Process image using AI-based noise reduction.

        Args:
            image: Input image

        Returns:
            Denoised image
        """
        start_time = time.time()
        self.validate_image(image)

        if not self.model_loaded:
            logger.warning("AI model not loaded, falling back to traditional method")
            return self._fallback_processing(image)

        # Simulate AI-based processing
        processed = self._ai_denoise(image)

        # Update statistics
        processing_time = time.time() - start_time
        self.update_stats(processing_time)

        return processed

    def _ai_denoise(self, image: np.ndarray) -> np.ndarray:
        """
        AI-based denoising implementation.

        Args:
            image: Input image

        Returns:
            Denoised image
        """
        # This is a simplified simulation of AI-based denoising
        # In a real implementation, you would:
        # 1. Preprocess the image for the model
        # 2. Run inference using your trained model
        # 3. Post-process the output

        logger.debug("Running AI inference for noise reduction...")

        # Simulate different processing based on image characteristics
        noise_level = self.estimate_noise_level(image)

        if noise_level > 0.5:
            # High noise - aggressive processing
            strength = self.config.strength * 1.2
        elif noise_level < 0.2:
            # Low noise - gentle processing
            strength = self.config.strength * 0.8
        else:
            # Medium noise - standard processing
            strength = self.config.strength

        # Simulate AI processing with adaptive filtering
        from scipy.ndimage import gaussian_filter

        if image.ndim == 3:
            processed = np.zeros_like(image, dtype=np.float32)
            for c in range(image.shape[2]):
                # Simulate channel-specific processing
                channel_strength = strength * (0.9 + 0.2 * np.random.random())
                processed[:, :, c] = gaussian_filter(image[:, :, c].astype(np.float32), sigma=channel_strength)
        else:
            processed = gaussian_filter(image.astype(np.float32), sigma=strength)

        # Simulate edge preservation using AI
        if self.config.preserve_edges:
            processed = self._ai_edge_preservation(image, processed)

        return np.clip(processed, 0, 255).astype(image.dtype)

    def _ai_edge_preservation(self, original: np.ndarray, processed: np.ndarray) -> np.ndarray:
        """
        AI-based edge preservation.

        Args:
            original: Original image
            processed: Processed image

        Returns:
            Edge-preserved image
        """
        # Simulate AI-based edge detection and preservation
        from scipy.ndimage import sobel

        if original.ndim == 3:
            gray = np.mean(original, axis=2)
        else:
            gray = original

        # Enhanced edge detection using "AI"
        edge_x = sobel(gray, axis=0)
        edge_y = sobel(gray, axis=1)
        edge_strength = np.sqrt(edge_x**2 + edge_y**2)

        # Normalize edge strength
        edge_strength = edge_strength / np.max(edge_strength)

        # AI-based adaptive blending
        # Simulate learned blending weights
        adaptive_alpha = 0.3 + 0.4 * edge_strength  # AI would learn these weights

        if original.ndim == 3:
            adaptive_alpha = np.expand_dims(adaptive_alpha, axis=2)

        # Blend original and processed based on edge strength
        result = adaptive_alpha * original.astype(np.float32) + (1 - adaptive_alpha) * processed

        return result

    def _fallback_processing(self, image: np.ndarray) -> np.ndarray:
        """
        Fallback processing when AI model is not available.

        Args:
            image: Input image

        Returns:
            Processed image using traditional methods
        """
        from scipy.ndimage import gaussian_filter

        if image.ndim == 3:
            processed = np.zeros_like(image, dtype=np.float32)
            for c in range(image.shape[2]):
                processed[:, :, c] = gaussian_filter(image[:, :, c].astype(np.float32), sigma=self.config.strength)
        else:
            processed = gaussian_filter(image.astype(np.float32), sigma=self.config.strength)

        return np.clip(processed, 0, 255).astype(image.dtype)

    def estimate_noise_level(self, image: np.ndarray) -> float:
        """
        AI-based noise level estimation.

        Args:
            image: Input image

        Returns:
            Estimated noise level
        """
        # Simulate AI-based noise estimation
        # In reality, this would use a trained model

        if image.ndim == 3:
            gray = np.mean(image, axis=2)
        else:
            gray = image

        # Simulate multiple noise indicators
        # 1. High-frequency content
        from scipy.ndimage import laplace

        laplacian_var = np.var(laplace(gray))

        # 2. Local variance
        from scipy.ndimage import uniform_filter

        local_mean = uniform_filter(gray.astype(np.float32), size=5)
        local_var = uniform_filter((gray.astype(np.float32) - local_mean) ** 2, size=5)
        avg_local_var = np.mean(local_var)

        # 3. Gradient magnitude
        from scipy.ndimage import sobel

        grad_x = sobel(gray, axis=0)
        grad_y = sobel(gray, axis=1)
        grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        avg_gradient = np.mean(grad_magnitude)

        # Simulate AI model combining these features
        # In reality, these would be learned weights
        noise_score = (
            0.4 * min(laplacian_var / 1000.0, 1.0) + 0.3 * min(avg_local_var / 500.0, 1.0) + 0.3 * min(avg_gradient / 100.0, 1.0)
        )

        return min(noise_score, 1.0)

    def train_on_image(self, clean_image: np.ndarray, noisy_image: np.ndarray) -> None:
        """
        Train the AI model on a clean/noisy image pair.

        Args:
            clean_image: Clean reference image
            noisy_image: Corresponding noisy image
        """
        # Simulate online learning
        self.training_data.append({"clean": clean_image, "noisy": noisy_image, "timestamp": time.time()})

        logger.info(f"Added training sample (total: {len(self.training_data)})")

        # Simulate model update every 10 samples
        if len(self.training_data) % 10 == 0:
            self._update_model()

    def _update_model(self) -> None:
        """Update the AI model with new training data."""
        logger.info("Updating AI model with new training data...")
        # Simulate model training/update
        time.sleep(0.05)
        logger.info("✓ AI model updated")


# Example 3: Custom Color Correction Algorithm
class AdaptiveColorCorrector:
    """
    Custom adaptive color correction algorithm.

    This example shows how to create custom processing algorithms
    that can be integrated into the processing pipeline.
    """

    def __init__(self, adaptation_rate: float = 0.1):
        """
        Initialize adaptive color corrector.

        Args:
            adaptation_rate: Rate of adaptation (0.0 to 1.0)
        """
        self.adaptation_rate = adaptation_rate
        self.color_matrix = np.eye(3)  # Start with identity matrix
        self.reference_colors = None
        self.processing_history = []

        logger.info(f"Adaptive color corrector initialized (adaptation rate: {adaptation_rate})")

    def set_reference_colors(self, reference_colors: np.ndarray) -> None:
        """
        Set reference colors for adaptation.

        Args:
            reference_colors: Reference color patches (Nx3 array)
        """
        self.reference_colors = reference_colors.copy()
        logger.info(f"Reference colors set: {reference_colors.shape}")

    def process_image(self, image: np.ndarray, measured_colors: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Process image with adaptive color correction.

        Args:
            image: Input image
            measured_colors: Measured color patches for adaptation

        Returns:
            Color-corrected image
        """
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Image must be RGB (3-channel)")

        # Adapt color matrix if measured colors provided
        if measured_colors is not None and self.reference_colors is not None:
            self._adapt_color_matrix(measured_colors)

        # Apply color correction
        corrected = self._apply_color_correction(image)

        # Update processing history
        self.processing_history.append(
            {"timestamp": time.time(), "color_matrix": self.color_matrix.copy(), "adapted": measured_colors is not None}
        )

        return corrected

    def _adapt_color_matrix(self, measured_colors: np.ndarray) -> None:
        """
        Adapt color correction matrix based on measured colors.

        Args:
            measured_colors: Measured color patches
        """
        if measured_colors.shape != self.reference_colors.shape:
            raise ValueError("Measured colors must match reference colors shape")

        # Calculate color differences
        color_diff = self.reference_colors - measured_colors

        # Simulate adaptive matrix update
        # In a real implementation, this would use more sophisticated algorithms
        adaptation_matrix = np.eye(3)

        # Simple adaptation based on average color differences
        avg_diff = np.mean(color_diff, axis=0)

        # Adjust matrix diagonal based on color differences
        for i in range(3):
            if abs(avg_diff[i]) > 5:  # Threshold for adaptation
                adjustment = self.adaptation_rate * avg_diff[i] / 255.0
                adaptation_matrix[i, i] += adjustment

        # Update color matrix with exponential smoothing
        self.color_matrix = (1 - self.adaptation_rate) * self.color_matrix + self.adaptation_rate * adaptation_matrix

        logger.debug(f"Color matrix adapted: {np.diag(self.color_matrix)}")

    def _apply_color_correction(self, image: np.ndarray) -> np.ndarray:
        """
        Apply color correction matrix to image.

        Args:
            image: Input image

        Returns:
            Color-corrected image
        """
        # Reshape image for matrix multiplication
        original_shape = image.shape
        pixels = image.reshape(-1, 3).astype(np.float32)

        # Apply color correction matrix
        corrected_pixels = np.dot(pixels, self.color_matrix.T)

        # Clip values and reshape back
        corrected_pixels = np.clip(corrected_pixels, 0, 255)
        corrected_image = corrected_pixels.reshape(original_shape).astype(image.dtype)

        return corrected_image

    def get_adaptation_stats(self) -> dict[str, Any]:
        """
        Get adaptation statistics.

        Returns:
            Dictionary with adaptation statistics
        """
        if not self.processing_history:
            return {"adaptations": 0, "current_matrix": self.color_matrix.tolist()}

        adaptations = sum(1 for entry in self.processing_history if entry["adapted"])

        return {
            "total_processed": len(self.processing_history),
            "adaptations": adaptations,
            "adaptation_rate": adaptations / len(self.processing_history),
            "current_matrix": self.color_matrix.tolist(),
            "matrix_determinant": np.linalg.det(self.color_matrix),
        }


def demo_custom_mipi_driver():
    """Demonstrate custom MIPI driver extension."""
    logger.info("=== Custom MIPI Driver Demo ===")

    # Create high-speed MIPI driver
    config = MIPIConfig(lanes=4, data_rate=5.0, channel=0)
    driver = HighSpeedMIPIDriver(config, optimization_level=2)

    # Enable optimizations
    driver.enable_burst_mode(True)
    driver.enable_compression(True, ratio=0.8)

    # Test data transfer
    test_data = b"x" * 1024  # 1KB test data

    logger.info("Testing optimized data transfer...")
    start_time = time.time()
    success = driver.send_data(test_data)
    transfer_time = time.time() - start_time

    if success:
        logger.info(f"✓ Transfer completed in {transfer_time:.4f}s")

        # Show enhanced status
        status = driver.get_enhanced_status()
        logger.info("Enhanced driver status:")
        for key, value in status.items():
            logger.info(f"  {key}: {value}")
    else:
        logger.error("✗ Transfer failed")


def demo_custom_noise_reducer():
    """Demonstrate custom AI noise reducer."""
    logger.info("=== Custom AI Noise Reducer Demo ===")

    # Register custom noise reducer
    NoiseReducerFactory.register_reducer(NoiseType.GAUSSIAN, AINoiseReducer)

    # Create AI noise reducer
    config = NoiseReductionConfig(noise_type=NoiseType.GAUSSIAN, strength=0.5, preserve_edges=True, adaptive=True)

    reducer = NoiseReducerFactory.create_reducer(config)
    logger.info(f"Created reducer: {reducer.get_algorithm_name()}")

    # Generate test image with noise
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    noise = np.random.normal(0, 20, test_image.shape)
    noisy_image = np.clip(test_image.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    logger.info(f"Generated test image: {test_image.shape}")

    # Process with AI noise reducer
    start_time = time.time()
    reducer.process(noisy_image)
    processing_time = time.time() - start_time

    logger.info(f"✓ AI denoising completed in {processing_time:.3f}s")

    # Show processing stats
    stats = reducer.get_processing_stats()
    logger.info(f"Processing stats: {stats}")

    # Simulate training
    if isinstance(reducer, AINoiseReducer):
        logger.info("Simulating online learning...")
        reducer.train_on_image(test_image, noisy_image)


def demo_custom_color_corrector():
    """Demonstrate custom adaptive color corrector."""
    logger.info("=== Custom Adaptive Color Corrector Demo ===")

    # Create adaptive color corrector
    corrector = AdaptiveColorCorrector(adaptation_rate=0.2)

    # Set reference colors (color checker patches)
    reference_colors = np.array(
        [
            [255, 0, 0],  # Red
            [0, 255, 0],  # Green
            [0, 0, 255],  # Blue
            [255, 255, 0],  # Yellow
            [255, 0, 255],  # Magenta
            [0, 255, 255],  # Cyan
        ],
        dtype=np.uint8,
    )

    corrector.set_reference_colors(reference_colors)

    # Generate test image
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    logger.info(f"Generated test image: {test_image.shape}")

    # Simulate measured colors with some error
    measured_colors = reference_colors.astype(np.float32)
    measured_colors += np.random.normal(0, 10, measured_colors.shape)  # Add measurement error
    measured_colors = np.clip(measured_colors, 0, 255).astype(np.uint8)

    # Process with adaptation
    corrected = corrector.process_image(test_image, measured_colors)
    logger.info(f"✓ Color correction completed: {corrected.shape}")

    # Show adaptation stats
    stats = corrector.get_adaptation_stats()
    logger.info("Adaptation statistics:")
    for key, value in stats.items():
        if key != "current_matrix":  # Skip matrix for brevity
            logger.info(f"  {key}: {value}")


def main():
    """Main function demonstrating custom extensions."""
    logger.info("Advanced Image Sensor Interface - Custom Extensions Demo")
    logger.info("=" * 60)

    try:
        # Demo 1: Custom MIPI Driver
        demo_custom_mipi_driver()
        logger.info("")

        # Demo 2: Custom Noise Reducer
        demo_custom_noise_reducer()
        logger.info("")

        # Demo 3: Custom Color Corrector
        demo_custom_color_corrector()

        logger.info("=" * 60)
        logger.info("✓ All custom extension demos completed successfully!")

        # Show how to get available noise reducers
        logger.info("\nAvailable noise reduction algorithms:")
        for noise_type in NoiseReducerFactory.get_available_types():
            info = NoiseReducerFactory.get_reducer_info(noise_type)
            logger.info(f"  {noise_type.value}: {info['algorithm_name']}")

    except Exception as e:
        logger.error(f"✗ Demo failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
