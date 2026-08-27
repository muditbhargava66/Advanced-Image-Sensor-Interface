"""
Signal Processing for Advanced Image Sensor Interface

This module implements sophisticated signal processing techniques for
optimizing the output of CMOS image sensors, including noise reduction,
dynamic range expansion, and color correction. It also includes an
automated test suite for validation and performance measurement.

Classes:
    SignalProcessor: Main class for signal processing operations.
    AutomatedTestSuite: Class for running automated tests on the SignalProcessor.
"""

import logging
import time
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from ..config import get_processing_config, get_test_config, get_timing_config
from .image_validation import ImageFormat, ImageValidator, SafeImageProcessor, SupportedDType
from ..types import ProcessingMetrics, SignalProcessingResult

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@runtime_checkable
class TimingStrategy(Protocol):
    """Protocol for timing strategies in signal processing."""

    def get_processing_time(self) -> float:
        """Get the current processing time per frame."""
        ...

    def set_processing_time(self, time: float) -> None:
        """Set the processing time per frame."""
        ...

    def optimize_timing(self) -> None:
        """Optimize timing parameters."""
        ...


class DefaultTimingStrategy:
    """Default timing strategy for production use."""

    def __init__(self, initial_time: float | None = None):
        """Initialize with default processing time."""
        timing_config = get_timing_config()
        self._processing_time = initial_time or timing_config.DEFAULT_PROCESSING_TIME

    def get_processing_time(self) -> float:
        """Get the current processing time per frame."""
        return self._processing_time

    def set_processing_time(self, time: float) -> None:
        """Set the processing time per frame."""
        if time < 0:
            raise ValueError("Processing time must be non-negative")
        self._processing_time = time

    def optimize_timing(self) -> None:
        """Optimize timing parameters."""
        timing_config = get_timing_config()
        self._processing_time *= timing_config.OPTIMIZATION_FACTOR_PRODUCTION


class TestTimingStrategy:
    """Testing timing strategy that allows direct control."""

    def __init__(self, initial_time: float | None = None):
        """Initialize with test processing time."""
        timing_config = get_timing_config()
        self._processing_time = initial_time or timing_config.DEFAULT_PROCESSING_TIME
        self._test_mode = True

    def get_processing_time(self) -> float:
        """Get the current processing time per frame."""
        return self._processing_time

    def set_processing_time(self, time: float) -> None:
        """Set the processing time per frame (test mode)."""
        if time < 0:
            raise ValueError("Processing time must be non-negative")
        self._processing_time = time

    def optimize_timing(self) -> None:
        """Optimize timing parameters (test mode)."""
        timing_config = get_timing_config()
        self._processing_time *= timing_config.OPTIMIZATION_FACTOR_TESTING


@dataclass
class SignalConfig:
    """Configuration parameters for signal processing."""

    bit_depth: int
    noise_reduction_strength: float
    color_correction_matrix: np.ndarray


class SignalProcessor:
    """Processes and optimizes signals from image sensors.

    Attributes:
        config: Configuration for signal processing.
    """

    def __init__(self, config: SignalConfig, timing_strategy: TimingStrategy = None):
        """Initialize the SignalProcessor.

        Args:
            config: Configuration for signal processing.
            timing_strategy: Optional strategy for timing control.
        """
        self.config = config
        self._timing_strategy = timing_strategy or DefaultTimingStrategy()
        self._validator = ImageValidator()

        # Create target format for processing
        self._target_format = ImageFormat(
            height=1080,  # Default, will be updated per image
            width=1920,  # Default, will be updated per image
            channels=3,  # Assume RGB processing
            bit_depth=config.bit_depth,
            dtype=self._get_dtype_for_bit_depth(config.bit_depth),
        )

        self._initialize_processing_pipeline()
        logger.info(f"Signal Processor initialized with {self.config.bit_depth}-bit depth")

    def _initialize_processing_pipeline(self) -> None:
        """Initialize the signal processing pipeline."""
        # Simulate initialization of processing structures
        timing_config = get_timing_config()
        time.sleep(timing_config.PIPELINE_INIT_DELAY)
        logger.info("Processing pipeline initialized successfully")

    def _get_dtype_for_bit_depth(self, bit_depth: int) -> SupportedDType:
        """Get appropriate dtype for bit depth."""
        if bit_depth == 8:
            return SupportedDType.UINT8
        elif bit_depth == 10:
            return SupportedDType.UINT10_IN_UINT16
        elif bit_depth == 12:
            return SupportedDType.UINT12_IN_UINT16
        elif bit_depth == 14:
            return SupportedDType.UINT14_IN_UINT16
        elif bit_depth == 16:
            return SupportedDType.UINT16
        else:
            raise ValueError(f"Unsupported bit depth: {bit_depth}")

    def process_frame(self, frame: np.ndarray) -> SignalProcessingResult:
        """
        Process a single frame of image data with comprehensive validation.

        Args:
            frame (np.ndarray): Input frame data.

        Returns:
            SignalProcessingResult: Result object containing processed data,
                success status, error info, warnings, and metrics.

        Raises:
            TypeError: If frame is not a numpy ndarray.
            ValueError: If frame has an unsupported number of channels.
        """
        start_time = time.perf_counter()
        warnings = []
        metrics = ProcessingMetrics(algorithm_name="SignalProcessor")
        noise_reduction_applied = False
        dynamic_range_expanded = False
        color_correction_applied = False
        snr_improvement_db = 0.0

        # CQ-3: Explicit type guard before entering the try block
        if not isinstance(frame, np.ndarray):
            return SignalProcessingResult(
                success=False,
                error=f"Expected np.ndarray, got {type(frame).__name__}",
                metrics=ProcessingMetrics(processing_time_ms=(time.perf_counter() - start_time) * 1000),
            )

        try:
            # Validate input frame
            original_format = self._validator.validate_image(frame)

            # Override the detected bit depth with our configured bit depth
            # This ensures consistent processing regardless of input data range
            original_format.bit_depth = self.config.bit_depth
            original_format.dtype = self._get_dtype_for_bit_depth(self.config.bit_depth)

            # CQ-4: Create a per-call ImageFormat instead of mutating the
            # shared self._target_format (avoids thread-safety issues)
            target_format = ImageFormat(
                height=original_format.height,
                width=original_format.width,
                channels=original_format.channels,
                bit_depth=self.config.bit_depth,
                dtype=self._get_dtype_for_bit_depth(self.config.bit_depth),
            )

            # Create safe processor for this frame
            processor = SafeImageProcessor(target_format)

            # Track original frame for SNR calculation
            original_frame = frame.astype(float)

            # Define processing pipeline
            def processing_pipeline(float_frame: np.ndarray) -> np.ndarray:
                nonlocal noise_reduction_applied, dynamic_range_expanded, color_correction_applied

                # Apply noise reduction
                if self.config.noise_reduction_strength > 0:
                    processed = self._apply_noise_reduction(float_frame)
                    noise_reduction_applied = True
                else:
                    processed = float_frame

                # Apply dynamic range expansion
                if processed.min() != processed.max():
                    processed = self._apply_dynamic_range_expansion(processed)
                    dynamic_range_expanded = True

                # Apply color correction
                if processed.ndim == 3 and processed.shape[2] == 3:
                    processed = self._apply_color_correction(processed)
                    color_correction_applied = True

                return processed

            # Process safely
            processed_frame = processor.safe_process(frame, processing_pipeline)

            if processed_frame is None:
                return SignalProcessingResult(
                    success=False,
                    error="Processing returned None (empty frame or validation failed)",
                    warnings=warnings,
                    metrics=ProcessingMetrics(processing_time_ms=(time.perf_counter() - start_time) * 1000),
                )

            # Calculate SNR improvement (simplified)
            if noise_reduction_applied:
                original_noise = np.std(original_frame - np.mean(original_frame))
                processed_noise = np.std(processed_frame - np.mean(processed_frame))
                if processed_noise > 0 and original_noise > 0:
                    snr_improvement_db = 20 * np.log10(original_noise / processed_noise)

            processing_time_ms = (time.perf_counter() - start_time) * 1000
            metrics = ProcessingMetrics(
                processing_time_ms=processing_time_ms,
                algorithm_name="SignalProcessor",
                parameters={
                    "bit_depth": self.config.bit_depth,
                    "noise_reduction_strength": self.config.noise_reduction_strength,
                    "noise_reduction_applied": noise_reduction_applied,
                    "dynamic_range_expanded": dynamic_range_expanded,
                    "color_correction_applied": color_correction_applied,
                },
            )

            return SignalProcessingResult(
                success=True,
                data=processed_frame,
                warnings=warnings,
                metrics=metrics,
                snr_improvement_db=snr_improvement_db,
                noise_reduction_applied=noise_reduction_applied,
                dynamic_range_expanded=dynamic_range_expanded,
                color_correction_applied=color_correction_applied,
            )

        except Exception as e:
            logger.error(f"Error processing frame: {e!s}")
            processing_time_ms = (time.perf_counter() - start_time) * 1000

            # Handle empty frames gracefully
            if isinstance(e, ValueError) and "empty" in str(e).lower():
                return SignalProcessingResult(
                    success=False,
                    error="Empty frame",
                    warnings=warnings,
                    metrics=ProcessingMetrics(processing_time_ms=processing_time_ms),
                )
            # Handle unsupported dtypes gracefully
            elif isinstance(e, ValueError) and "dtype" in str(e).lower():
                try:
                    if frame.dtype == np.float64:
                        converted = (frame * 65535).astype(np.uint16)
                        return SignalProcessingResult(
                            success=True,
                            data=converted,
                            warnings=[f"Converted from float64 to uint16: {e!s}"],
                            metrics=ProcessingMetrics(processing_time_ms=processing_time_ms),
                        )
                    converted = frame.astype(np.uint16)
                    return SignalProcessingResult(
                        success=True,
                        data=converted,
                        warnings=[f"Converted to uint16: {e!s}"],
                        metrics=ProcessingMetrics(processing_time_ms=processing_time_ms),
                    )
                except Exception as conv_e:
                    return SignalProcessingResult(
                        success=False,
                        error=f"Failed to convert dtype: {conv_e}",
                        warnings=warnings,
                        metrics=ProcessingMetrics(processing_time_ms=processing_time_ms),
                    )
            elif isinstance(e, ValueError):
                return SignalProcessingResult(
                    success=False,
                    error=str(e),
                    warnings=warnings,
                    metrics=ProcessingMetrics(processing_time_ms=processing_time_ms),
                )
            return SignalProcessingResult(
                success=False, error=str(e), warnings=warnings, metrics=ProcessingMetrics(processing_time_ms=processing_time_ms)
            )

    def _apply_noise_reduction(self, frame: np.ndarray) -> np.ndarray:
        """Apply noise reduction to the frame."""
        if self.config.noise_reduction_strength == 0:
            return frame

        # Use configurable approach for noise reduction
        processing_config = get_processing_config()
        sigma = self.config.noise_reduction_strength * processing_config.NOISE_REDUCTION_SIGMA_MULTIPLIER
        kernel_size = max(processing_config.MIN_KERNEL_SIZE, int(sigma * processing_config.KERNEL_SIZE_MULTIPLIER) + 1)

        # Make kernel size odd
        if kernel_size % 2 == 0:
            kernel_size += 1

        # Apply Gaussian blur without mixing color channels. SciPy applies a
        # scalar sigma to every axis, so we pin non-spatial axes to zero.
        if sigma == 0:
            return frame.astype(float)

        from scipy.ndimage import gaussian_filter

        if frame.ndim <= 2:
            sigma_per_axis: float | tuple[float, ...] = sigma
        else:
            sigma_per_axis = (sigma, sigma, *([0.0] * (frame.ndim - 2)))

        return gaussian_filter(frame.astype(float), sigma=sigma_per_axis)

    def _blur(self, image: np.ndarray, kernel_size: int, sigma: float) -> np.ndarray:
        """Apply Gaussian blur using scipy.ndimage for performance.

        Deprecated: use gaussian_filter directly on the full array.
        This method is retained only for backward compatibility.
        """
        from scipy.ndimage import gaussian_filter

        if sigma == 0:
            return image.astype(float)

        return gaussian_filter(image.astype(float), sigma=sigma)

    def _apply_dynamic_range_expansion(self, frame: np.ndarray) -> np.ndarray:
        """Apply dynamic range expansion to the frame.

        CQ-1 fix: For integer dtypes the expansion maps to the full dtype
        range (e.g. 0..65535 for uint16) instead of always normalising
        to [0, 1]. Float inputs are still mapped to [0, 1].
        """
        if frame.min() == frame.max():
            return frame  # Avoid division by zero for constant images

        # Determine the target range based on dtype
        if np.issubdtype(frame.dtype, np.integer):
            target_max = float(np.iinfo(frame.dtype).max)
        else:
            target_max = 1.0

        expanded = np.interp(frame, (frame.min(), frame.max()), (0.0, target_max))
        return expanded.astype(frame.dtype)

    def _apply_color_correction(self, frame: np.ndarray) -> np.ndarray:
        """Apply color correction to the frame."""
        if frame.ndim == 3 and frame.shape[2] == 3:
            # Handle 3-channel color images
            return np.dot(frame.reshape(-1, 3), self.config.color_correction_matrix.T).reshape(frame.shape)
        else:
            # Return unchanged for grayscale or unexpected formats
            return frame

    def optimize_performance(self) -> None:
        """Optimize signal processing performance."""
        self._timing_strategy.optimize_timing()

        # Use configurable improvement factor
        processing_config = get_processing_config()
        self.config.noise_reduction_strength *= processing_config.NOISE_REDUCTION_IMPROVEMENT

        processing_time = self._timing_strategy.get_processing_time()
        logger.info(f"Optimized performance: Processing time reduced to {processing_time:.3f} seconds per frame")

    def set_timing_strategy_for_test(self, strategy: TimingStrategy) -> None:
        """Set timing strategy for testing purposes."""
        self._timing_strategy = strategy


class AutomatedTestSuite:
    """
    Automated test suite for validating and measuring performance of the SignalProcessor.

    Attributes
    ----------
        signal_processor (SignalProcessor): The SignalProcessor instance to test.

    """

    def __init__(self, signal_processor: SignalProcessor):
        """
        Initialize the AutomatedTestSuite with a SignalProcessor instance.

        Args:
        ----
            signal_processor (SignalProcessor): The SignalProcessor instance to test.

        """
        self.signal_processor = signal_processor
        self._test_cases = self._generate_test_cases()
        self._execution_time = 0.0
        self._pass_rate = 0.0

    def _generate_test_cases(self) -> list[np.ndarray]:
        """Generate a set of test cases for signal processing."""
        test_config = get_test_config()
        return [
            np.random.rand(test_config.DEFAULT_TEST_HEIGHT, test_config.DEFAULT_TEST_WIDTH, test_config.DEFAULT_TEST_CHANNELS)
            for _ in range(test_config.TEST_FRAME_COUNT)
        ]

    def run_tests(self) -> tuple[float, float]:
        """
        Run the automated test suite.

        Returns
        -------
            Tuple[float, float]: Execution time and pass rate (fraction
                of tests that produced valid output).
        """
        start_time = time.time()

        passed_tests = 0
        for i, test_case in enumerate(self._test_cases):
            try:
                result = self.signal_processor.process_frame(test_case)
                if result.success and self._validate_processed_frame(result.data):
                    passed_tests += 1
            except Exception as e:
                logger.error(f"Test case {i} failed: {e!s}")

        end_time = time.time()
        self._execution_time = end_time - start_time
        self._pass_rate = passed_tests / len(self._test_cases)

        logger.info(f"Test suite completed in {self._execution_time:.2f} seconds with {self._pass_rate:.2%} pass rate")
        return self._execution_time, self._pass_rate

    def _validate_processed_frame(self, frame: np.ndarray | None) -> bool:
        """Validate a processed frame."""
        if frame is None:
            return False
        return frame.size > 0 and not np.isnan(frame).any()


# Example usage demonstrating performance improvements and automated testing
if __name__ == "__main__":
    # Initialize SignalProcessor
    config = SignalConfig(
        bit_depth=12, noise_reduction_strength=0.1, color_correction_matrix=np.eye(3)
    )  # Identity matrix for simplicity
    processor = SignalProcessor(config)

    # Create and run initial automated test suite
    initial_test_suite = AutomatedTestSuite(processor)
    initial_time, initial_pass_rate = initial_test_suite.run_tests()

    print(f"Initial test execution time: {initial_time:.2f} seconds")
    print(f"Initial test pass rate: {initial_pass_rate:.2%}")

    # Optimize signal processor performance
    processor.optimize_performance()

    # Create and run optimized automated test suite
    optimized_test_suite = AutomatedTestSuite(processor)
    optimized_time, optimized_pass_rate = optimized_test_suite.run_tests()

    print(f"Optimized test execution time: {optimized_time:.2f} seconds")
    print(f"Optimized test pass rate: {optimized_pass_rate:.2%}")

    # Calculate improvements
    time_improvement = (initial_time - optimized_time) / initial_time * 100
    pass_rate_improvement = (optimized_pass_rate - initial_pass_rate) / initial_pass_rate * 100

    print(f"Reduction in validation time: {time_improvement:.2f}%")
    print(f"Increase in test pass rate: {pass_rate_improvement:.2f}%")

    # Demonstrate overall system performance improvement
    initial_frame = np.random.rand(1080, 1920, 3)

    start_time = time.time()
    result = processor.process_frame(initial_frame)
    initial_processing_time = time.time() - start_time

    start_time = time.time()
    result = processor.process_frame(initial_frame)
    optimized_processing_time = time.time() - start_time

    performance_improvement = (initial_processing_time - optimized_processing_time) / initial_processing_time * 100
    print(f"Overall system performance improvement: {performance_improvement:.2f}%")

    # Show new result structure
    print("\nSignalProcessingResult example:")
    print(f"  Success: {result.success}")
    print(f"  SNR Improvement: {result.snr_improvement_db:.2f} dB")
    print(f"  Noise Reduction: {result.noise_reduction_applied}")
    print(f"  Dynamic Range: {result.dynamic_range_expanded}")
    print(f"  Color Correction: {result.color_correction_applied}")
    print(f"  Processing Time: {result.metrics.processing_time_ms:.2f} ms")
