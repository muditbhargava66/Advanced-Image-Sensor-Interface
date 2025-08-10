#!/usr/bin/env python3
"""
Basic Usage Example for Advanced Image Sensor Interface

This example demonstrates the fundamental usage patterns with proper error handling,
input validation, and realistic parameter definitions.

Author: Advanced Image Sensor Interface Team
Version: 2.0.0
"""

import argparse
import logging
import sys

import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

try:
    from advanced_image_sensor_interface import EnhancedSensorInterface, HDRProcessor, PowerConfig, PowerManager, RAWProcessor
    from advanced_image_sensor_interface.utils.performance_metrics import (
        calculate_color_accuracy,
        calculate_dynamic_range,
        calculate_snr,
    )
except ImportError as e:
    logger.error(f"Failed to import required modules: {e}")
    logger.error("Please install the package with: pip install -e .")
    sys.exit(1)


def validate_resolution(resolution_str: str) -> tuple[int, int]:
    """
    Validate and parse resolution string.

    Args:
        resolution_str: Resolution in format "WIDTHxHEIGHT" (e.g., "1920x1080")

    Returns:
        Tuple of (width, height)

    Raises:
        ValueError: If resolution format is invalid
    """
    try:
        width_str, height_str = resolution_str.split("x")
        width, height = int(width_str), int(height_str)

        if width <= 0 or height <= 0:
            raise ValueError("Width and height must be positive")
        if width > 8192 or height > 8192:
            raise ValueError("Resolution too large (max 8192x8192)")

        return width, height
    except ValueError as e:
        if "not enough values to unpack" in str(e):
            raise ValueError("Resolution must be in format WIDTHxHEIGHT (e.g., 1920x1080)")
        raise


def validate_noise_level(noise_level: float) -> float:
    """
    Validate noise level parameter.

    Args:
        noise_level: Noise level between 0.0 and 1.0

    Returns:
        Validated noise level

    Raises:
        ValueError: If noise level is out of range
    """
    if not 0.0 <= noise_level <= 1.0:
        raise ValueError("Noise level must be between 0.0 and 1.0")
    return noise_level


def basic_sensor_demo(resolution: tuple[int, int], frame_rate: float = 30.0) -> None:
    """
    Demonstrate basic sensor interface usage.

    Args:
        resolution: Sensor resolution as (width, height)
        frame_rate: Target frame rate in fps
    """
    logger.info("=== Basic Sensor Interface Demo ===")

    try:
        # Create sensor configuration with explicit parameters
        width, height = resolution

        logger.info(f"Initializing sensor: {width}x{height} @ {frame_rate} fps")

        # Create sensor configuration
        from advanced_image_sensor_interface.sensor_interface.enhanced_sensor import SensorConfiguration, SensorResolution

        sensor_config = SensorConfiguration(
            resolution=SensorResolution.CUSTOM, custom_resolution=(width, height), frame_rate=frame_rate, bit_depth=12
        )

        # Initialize enhanced sensor interface
        sensor = EnhancedSensorInterface(sensor_config)
        logger.info("✓ Sensor configured successfully")

        # Start streaming with error handling
        try:
            sensor.start_streaming()
            logger.info("✓ Streaming started")

            # Capture a few test frames
            frames_captured = 0
            target_frames = 5

            for i in range(target_frames):
                try:
                    frame = sensor.capture_frame(0)
                    if frame is not None:
                        frames_captured += 1
                        logger.info(f"✓ Captured frame {i+1}: {frame.shape}, dtype: {frame.dtype}")
                    else:
                        logger.warning(f"✗ Failed to capture frame {i+1}")
                except Exception as e:
                    logger.error(f"✗ Error capturing frame {i+1}: {e}")

            logger.info(f"Successfully captured {frames_captured}/{target_frames} frames")

        finally:
            # Always stop streaming
            sensor.stop_streaming()
            logger.info("✓ Streaming stopped")

    except Exception as e:
        logger.error(f"✗ Basic sensor demo failed: {e}")
        raise


def hdr_processing_demo(resolution: tuple[int, int], noise_level: float = 0.1) -> None:
    """
    Demonstrate HDR processing with proper error handling.

    Args:
        resolution: Image resolution as (width, height)
        noise_level: Noise level for synthetic test data
    """
    logger.info("=== HDR Processing Demo ===")

    try:
        width, height = resolution

        # Initialize HDR processor
        from advanced_image_sensor_interface.sensor_interface.hdr_processing import HDRParameters, ToneMappingMethod

        hdr_params = HDRParameters(tone_mapping_method=ToneMappingMethod.ADAPTIVE)
        hdr_processor = HDRProcessor(hdr_params)
        logger.info("✓ HDR processor initialized")

        # Generate synthetic HDR test images with different exposures
        exposures = [-2.0, 0.0, 2.0]  # EV values
        test_images = []

        for ev in exposures:
            # Create synthetic image with exposure variation
            base_intensity = 128 + (ev * 30)  # Adjust base intensity by EV
            base_intensity = np.clip(base_intensity, 0, 255)

            # Generate image with some structure
            x, y = np.meshgrid(np.linspace(0, 1, width), np.linspace(0, 1, height))
            pattern = np.sin(x * 10) * np.cos(y * 10) * 50 + base_intensity

            # Add noise
            noise = np.random.normal(0, noise_level * 255, (height, width))
            image = np.clip(pattern + noise, 0, 255).astype(np.uint8)

            # Convert to 3-channel RGB
            rgb_image = np.stack([image, image, image], axis=2)
            test_images.append(rgb_image)

            logger.info(f"✓ Generated test image for EV {ev:+.1f}: {rgb_image.shape}")

        # Process single image HDR
        try:
            single_hdr = hdr_processor.process_single_image(test_images[1])  # Use middle exposure
            logger.info(f"✓ Single image HDR: {single_hdr.shape}, dtype: {single_hdr.dtype}")
        except Exception as e:
            logger.error(f"✗ Single image HDR failed: {e}")

        # Process exposure stack HDR
        try:
            stack_hdr = hdr_processor.process_exposure_stack(test_images)
            logger.info(f"✓ Exposure stack HDR: {stack_hdr.shape}, dtype: {stack_hdr.dtype}")

            # Get processing statistics
            stats = hdr_processor.get_processing_stats()
            logger.info(f"HDR processing stats: {stats}")

        except Exception as e:
            logger.error(f"✗ Exposure stack HDR failed: {e}")

    except Exception as e:
        logger.error(f"✗ HDR processing demo failed: {e}")
        raise


def raw_processing_demo(resolution: tuple[int, int]) -> None:
    """
    Demonstrate RAW image processing.

    Args:
        resolution: Image resolution as (width, height)
    """
    logger.info("=== RAW Processing Demo ===")

    try:
        width, height = resolution

        # Initialize RAW processor with RGGB Bayer pattern
        from advanced_image_sensor_interface.sensor_interface.raw_processing import BayerPattern, DemosaicMethod, RAWParameters

        raw_params = RAWParameters(bayer_pattern=BayerPattern.RGGB, demosaic_method=DemosaicMethod.MALVAR)
        raw_processor = RAWProcessor(raw_params)
        logger.info(f"✓ RAW processor initialized with {raw_params.bayer_pattern.value} pattern")

        # Generate synthetic RAW data (Bayer pattern)
        raw_data = np.random.randint(0, 4095, (height, width), dtype=np.uint16)

        # Add some structure to make it more realistic
        x, y = np.meshgrid(np.linspace(0, 1, width), np.linspace(0, 1, height))
        pattern = (np.sin(x * 5) * np.cos(y * 5) * 1000 + 2048).astype(np.uint16)
        raw_data = np.clip(raw_data + pattern, 0, 4095).astype(np.uint16)

        logger.info(f"✓ Generated synthetic RAW data: {raw_data.shape}, dtype: {raw_data.dtype}")

        # Process RAW to RGB
        try:
            rgb_image = raw_processor.process_raw_to_rgb(raw_data)
            logger.info(f"✓ RAW to RGB conversion: {rgb_image.shape}, dtype: {rgb_image.dtype}")

            # Get processing statistics
            stats = raw_processor.get_processing_stats()
            logger.info(f"RAW processing stats: {stats}")

        except Exception as e:
            logger.error(f"✗ RAW processing failed: {e}")

    except Exception as e:
        logger.error(f"✗ RAW processing demo failed: {e}")
        raise


def power_management_demo() -> None:
    """
    Demonstrate power management with realistic parameters.
    """
    logger.info("=== Power Management Demo ===")

    try:
        # Initialize power manager with realistic embedded system values
        power_config = PowerConfig(
            voltage_main=1.8, voltage_io=3.3, current_limit=2.0  # Core voltage  # I/O voltage  # 2A current limit
        )

        power_manager = PowerManager(power_config)
        logger.info("✓ Power manager initialized")

        # Test voltage setting with validation
        test_voltages = [1.8, 2.5, 3.3]

        for voltage in test_voltages:
            try:
                success = power_manager.set_voltage("main", voltage)
                if success:
                    logger.info(f"✓ Set main voltage to {voltage}V")
                else:
                    logger.warning(f"✗ Failed to set main voltage to {voltage}V")
            except Exception as e:
                logger.error(f"✗ Error setting voltage to {voltage}V: {e}")

        # Get power status
        try:
            status = power_manager.get_power_status()
            logger.info("Power Status:")
            for key, value in status.items():
                if isinstance(value, float):
                    logger.info(f"  {key}: {value:.3f}")
                else:
                    logger.info(f"  {key}: {value}")
        except Exception as e:
            logger.error(f"✗ Failed to get power status: {e}")

        # Test power optimization
        try:
            power_manager.optimize_noise_reduction()
            logger.info("✓ Power optimization completed")
        except Exception as e:
            logger.error(f"✗ Power optimization failed: {e}")

    except Exception as e:
        logger.error(f"✗ Power management demo failed: {e}")
        raise


def performance_metrics_demo(resolution: tuple[int, int]) -> None:
    """
    Demonstrate performance metrics calculation.

    Args:
        resolution: Image resolution as (width, height)
    """
    logger.info("=== Performance Metrics Demo ===")

    try:
        width, height = resolution

        # Generate test images for metrics calculation
        clean_signal = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
        noise_component = np.random.normal(0, 10, (height, width, 3)).astype(np.float32)

        logger.info(f"✓ Generated test images: {clean_signal.shape}")

        # Calculate SNR
        try:
            snr = calculate_snr(clean_signal.astype(np.float32), noise_component)
            logger.info(f"✓ SNR: {snr:.2f} dB")
        except Exception as e:
            logger.error(f"✗ SNR calculation failed: {e}")

        # Calculate dynamic range
        try:
            dynamic_range = calculate_dynamic_range(clean_signal)
            logger.info(f"✓ Dynamic Range: {dynamic_range:.2f} dB")
        except Exception as e:
            logger.error(f"✗ Dynamic range calculation failed: {e}")

        # Calculate color accuracy
        try:
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

            # Simulate measured colors with small errors
            measured_colors = reference_colors + np.random.randint(-5, 6, reference_colors.shape)
            measured_colors = np.clip(measured_colors, 0, 255).astype(np.uint8)

            color_accuracy, delta_e_values = calculate_color_accuracy(reference_colors, measured_colors)
            logger.info(f"✓ Color Accuracy (Mean ΔE): {color_accuracy:.2f}")
            logger.info(f"  Individual ΔE values: {delta_e_values}")

        except Exception as e:
            logger.error(f"✗ Color accuracy calculation failed: {e}")

    except Exception as e:
        logger.error(f"✗ Performance metrics demo failed: {e}")
        raise


def main():
    """
    Main function with comprehensive argument parsing and validation.
    """
    parser = argparse.ArgumentParser(
        description="Basic Usage Example for Advanced Image Sensor Interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --resolution 1920x1080 --frame-rate 30
  %(prog)s --resolution 3840x2160 --noise-level 0.05
  %(prog)s --resolution 640x480 --frame-rate 60 --verbose
        """,
    )

    parser.add_argument(
        "--resolution", type=str, default="1920x1080", help="Sensor resolution in WIDTHxHEIGHT format (default: 1920x1080)"
    )

    parser.add_argument("--frame-rate", type=float, default=30.0, help="Target frame rate in fps (default: 30.0)")

    parser.add_argument("--noise-level", type=float, default=0.1, help="Noise level for synthetic data (0.0-1.0, default: 0.1)")

    parser.add_argument("--skip-sensor", action="store_true", help="Skip sensor interface demo")

    parser.add_argument("--skip-hdr", action="store_true", help="Skip HDR processing demo")

    parser.add_argument("--skip-raw", action="store_true", help="Skip RAW processing demo")

    parser.add_argument("--skip-power", action="store_true", help="Skip power management demo")

    parser.add_argument("--skip-metrics", action="store_true", help="Skip performance metrics demo")

    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        # Validate arguments
        resolution = validate_resolution(args.resolution)
        noise_level = validate_noise_level(args.noise_level)

        if args.frame_rate <= 0:
            raise ValueError("Frame rate must be positive")
        if args.frame_rate > 240:
            logger.warning("High frame rate requested, may not be achievable")

        logger.info("Advanced Image Sensor Interface - Basic Usage Example")
        logger.info("=" * 60)
        logger.info(f"Resolution: {resolution[0]}x{resolution[1]}")
        logger.info(f"Frame Rate: {args.frame_rate} fps")
        logger.info(f"Noise Level: {noise_level}")
        logger.info("=" * 60)

        # Run demos based on arguments
        if not args.skip_sensor:
            basic_sensor_demo(resolution, args.frame_rate)

        if not args.skip_hdr:
            hdr_processing_demo(resolution, noise_level)

        if not args.skip_raw:
            raw_processing_demo(resolution)

        if not args.skip_power:
            power_management_demo()

        if not args.skip_metrics:
            performance_metrics_demo(resolution)

        logger.info("=" * 60)
        logger.info("✓ All demos completed successfully!")

    except ValueError as e:
        logger.error(f"✗ Invalid argument: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Demo interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"✗ Demo failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
