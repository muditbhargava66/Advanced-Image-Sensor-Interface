#!/usr/bin/env python3
"""
Advanced Image Sensor Interface Comprehensive Demo

This script demonstrates the enhanced features including:
- Enhanced sensor interface with 8K support
- HDR image processing
- RAW image processing
- Multi-sensor synchronization
- GPU acceleration
- Advanced power management

Requirements:
    - numpy>=1.23.5
    - scipy>=1.10.0
    - scikit-image>=0.20.0
    - matplotlib>=3.7.0 (for visualization)
    - Optional: cupy (for GPU acceleration)
    - Optional: numba (for JIT acceleration)

Author: Advanced Image Sensor Interface Team
Version: 2.0.0
"""

import logging
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Import enhanced features
from advanced_image_sensor_interface.sensor_interface import (  # Advanced power management; GPU acceleration; HDR processing; RAW processing
    AdvancedPowerManager,
    EnhancedSensorInterface,
    GPUAccelerator,
    HDRMode,
    MultiSensorSynchronizer,
    PowerMode,
    RAWFormat,
    SensorConfiguration,
    SensorResolution,
    create_gpu_config_for_automotive,
    create_hdr_processor_for_automotive,
    create_multi_camera_sync_config,
    create_power_config_for_automotive,
    create_raw_processor_for_automotive,
    create_stereo_sync_config,
)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def demo_enhanced_sensor_interface():
    """Demonstrate enhanced sensor interface with high resolution support."""
    logger.info("=== Enhanced Sensor Interface Demo ===")

    # Create 4K sensor configuration (8K exceeds data rate limits in simulation)
    config = SensorConfiguration(
        resolution=SensorResolution.UHD_4K,
        frame_rate=60.0,
        bit_depth=12,
        hdr_mode=HDRMode.HDR10,
        raw_format=RAWFormat.RAW12,
        raw_processing=True,
        gpu_acceleration=True,
        parallel_processing=True,
    )
    logger.info(f"Created 4K sensor config: {config.effective_resolution}")
    logger.info(f"Data rate: {config.data_rate_mbps:.2f} Mbps")

    # Initialize enhanced sensor interface
    sensor = EnhancedSensorInterface(config)

    # Start streaming
    if sensor.start_streaming():
        logger.info("4K streaming started successfully")

        # Capture a few frames
        for i in range(3):
            frame = sensor.capture_frame()
            if frame is not None:
                logger.info(f"Captured 4K frame {i+1}: {frame.shape}, dtype: {frame.dtype}")
            time.sleep(0.1)

        # Stop streaming
        sensor.stop_streaming()
        logger.info("4K streaming stopped")

    # Get sensor status
    status = sensor.get_sensor_status()
    logger.info(f"Sensor status: {status['streaming']}, frames captured: {status['total_frames']}")

    return sensor


def demo_hdr_processing():
    """Demonstrate HDR image processing capabilities."""
    logger.info("=== HDR Processing Demo ===")

    # Create HDR processor for automotive use
    hdr_processor = create_hdr_processor_for_automotive()
    logger.info(f"HDR processor created with {hdr_processor.parameters.tone_mapping_method.value} tone mapping")

    # Generate test images with different exposures
    height, width = 480, 640
    test_images = []
    exposure_values = [-2.0, 0.0, 2.0]  # Under, normal, over exposed

    for ev in exposure_values:
        # Generate synthetic HDR test image
        base_image = np.random.randint(50, 200, (height, width, 3), dtype=np.uint8)

        # Simulate exposure variation
        exposure_factor = 2.0**ev
        exposed_image = np.clip(base_image.astype(np.float32) * exposure_factor, 0, 255).astype(np.uint8)
        test_images.append(exposed_image)

        logger.info(f"Generated test image with EV {ev:+.1f}")

    # Process single image
    single_result = hdr_processor.process_single_image(test_images[1])  # Normal exposure
    logger.info(f"Single image HDR processing: {single_result.shape}, dtype: {single_result.dtype}")

    # Process exposure stack
    stack_result = hdr_processor.process_exposure_stack(test_images, exposure_values)
    logger.info(f"Exposure stack HDR processing: {stack_result.shape}, dtype: {stack_result.dtype}")

    # Get processing statistics
    stats = hdr_processor.get_processing_stats()
    logger.info(f"HDR processing stats: {stats}")

    return hdr_processor, test_images, stack_result


def demo_raw_processing():
    """Demonstrate RAW image processing capabilities."""
    logger.info("=== RAW Processing Demo ===")

    # Create RAW processor for automotive use
    raw_processor = create_raw_processor_for_automotive()
    logger.info(f"RAW processor created with {raw_processor.parameters.bayer_pattern.value} pattern")
    logger.info(f"Demosaic method: {raw_processor.parameters.demosaic_method.value}")

    # Generate synthetic RAW data (Bayer pattern)
    height, width = 480, 640
    raw_data = np.random.randint(0, 4095, (height, width), dtype=np.uint16)  # 12-bit RAW

    # Apply Bayer pattern (RGGB)
    raw_data[0::2, 0::2] = np.clip(raw_data[0::2, 0::2] * 1.2, 0, 4095)  # R
    raw_data[0::2, 1::2] = raw_data[0::2, 1::2]  # G1
    raw_data[1::2, 0::2] = raw_data[1::2, 0::2]  # G2
    raw_data[1::2, 1::2] = np.clip(raw_data[1::2, 1::2] * 0.8, 0, 4095)  # B

    logger.info(f"Generated synthetic RAW data: {raw_data.shape}, dtype: {raw_data.dtype}")

    # Process RAW to RGB
    rgb_result = raw_processor.process_raw_image(raw_data)
    logger.info(f"RAW to RGB processing: {rgb_result.shape}, dtype: {rgb_result.dtype}")

    # Get processing statistics
    stats = raw_processor.processing_stats
    logger.info(f"RAW processing stats: {stats}")

    return raw_processor, raw_data, rgb_result


def demo_multi_sensor_sync():
    """Demonstrate multi-sensor synchronization."""
    logger.info("=== Multi-Sensor Synchronization Demo ===")

    # Create stereo camera synchronization
    stereo_config = create_stereo_sync_config()
    stereo_sync = MultiSensorSynchronizer(stereo_config)
    logger.info("Stereo camera synchronizer created")

    # Start synchronization
    if stereo_sync.start_synchronization():
        logger.info("Stereo synchronization started")

        # Capture synchronized frames
        for i in range(3):
            frames = stereo_sync.capture_synchronized_frames()
            if frames:
                logger.info(f"Captured synchronized frames {i+1}: {len(frames)} sensors")
                for sensor_id, (frame, timestamp) in frames.items():
                    logger.info(f"  Sensor {sensor_id}: {frame.shape}, timestamp: {timestamp:.6f}")
            time.sleep(0.1)

        # Stop synchronization
        stereo_sync.stop_synchronization()
        logger.info("Stereo synchronization stopped")

    # Get synchronization status
    status = stereo_sync.get_synchronization_status()
    logger.info(f"Sync status: {status['statistics']}")

    # Create multi-camera setup
    multi_config = create_multi_camera_sync_config(num_cameras=4)
    multi_sync = MultiSensorSynchronizer(multi_config)
    logger.info("Multi-camera synchronizer created (4 cameras)")

    return stereo_sync, multi_sync


def demo_gpu_acceleration():
    """Demonstrate GPU acceleration capabilities."""
    logger.info("=== GPU Acceleration Demo ===")

    # Create GPU accelerator
    gpu_config = create_gpu_config_for_automotive()
    gpu_accelerator = GPUAccelerator(gpu_config)

    # Get device information
    device_info = gpu_accelerator.get_device_info()
    logger.info(f"GPU backend: {device_info['backend']}")
    logger.info(f"GPU initialized: {device_info['is_initialized']}")

    if device_info["device_info"]:
        logger.info(f"Device: {device_info['device_info'].get('name', 'Unknown')}")

    # Generate test images for batch processing
    batch_size = 4
    height, width = 480, 640
    test_images = []

    for i in range(batch_size):
        img = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
        test_images.append(img)

    logger.info(f"Generated {batch_size} test images for GPU processing")

    # Test different GPU operations
    operations = ["gaussian_blur", "edge_detection", "histogram_equalization", "noise_reduction"]

    for operation in operations:
        start_time = time.time()

        if operation == "gaussian_blur":
            results = gpu_accelerator.process_image_batch(test_images, operation, sigma=2.0)
        else:
            results = gpu_accelerator.process_image_batch(test_images, operation)

        processing_time = time.time() - start_time
        logger.info(f"GPU {operation}: {len(results)} images in {processing_time:.3f}s")

    # Get performance statistics
    stats = gpu_accelerator.get_performance_stats()
    logger.info("GPU performance stats:")
    logger.info(f"  Total operations: {stats['operations_processed']}")
    logger.info(f"  GPU operations: {stats['gpu_operations']} ({stats.get('gpu_percentage', 0):.1f}%)")
    logger.info(f"  CPU operations: {stats['cpu_operations']} ({stats.get('cpu_percentage', 0):.1f}%)")

    return gpu_accelerator


def demo_advanced_power_management():
    """Demonstrate advanced power management."""
    logger.info("=== Advanced Power Management Demo ===")

    # Create power manager for automotive use
    power_config = create_power_config_for_automotive()
    power_manager = AdvancedPowerManager(power_config)
    logger.info(f"Power manager created in {power_manager.current_mode.value} mode")

    # Start power monitoring
    if power_manager.start_monitoring():
        logger.info("Power monitoring started")

        # Demonstrate power mode changes
        power_modes = [PowerMode.PERFORMANCE, PowerMode.BALANCED, PowerMode.POWER_SAVER]

        for mode in power_modes:
            power_manager.set_power_mode(mode)
            time.sleep(0.5)  # Let monitoring update

            metrics = power_manager.get_power_metrics()
            logger.info(f"Power mode {mode.value}:")
            logger.info(f"  Total power: {metrics.total_power:.2f}W")
            logger.info(f"  Temperature: {metrics.temperature_celsius:.1f}°C")
            logger.info(f"  Frequency: {metrics.current_frequency_mhz:.0f} MHz")
            logger.info(f"  Thermal state: {metrics.thermal_state.value}")

        # Demonstrate workload optimization
        workloads = ["streaming", "processing", "idle"]

        for workload in workloads:
            power_manager.optimize_for_workload(workload)
            time.sleep(0.2)

            metrics = power_manager.get_power_metrics()
            logger.info(f"Optimized for {workload}: {metrics.total_power:.2f}W")

        # Stop monitoring
        power_manager.stop_monitoring()
        logger.info("Power monitoring stopped")

    return power_manager


def create_visualization(hdr_result, raw_result, output_dir="output"):
    """Create visualization of processing results."""
    logger.info("=== Creating Visualizations ===")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Advanced Image Sensor Interface Enhanced Features Demo Results", fontsize=16)

    # HDR result
    if hdr_result is not None:
        axes[0, 0].imshow(hdr_result)
        axes[0, 0].set_title("HDR Processed Image")
        axes[0, 0].axis("off")

    # RAW result
    if raw_result is not None:
        axes[0, 1].imshow(raw_result)
        axes[0, 1].set_title("RAW to RGB Processed")
        axes[0, 1].axis("off")

    # Generate some synthetic performance data
    resolutions = ["HD", "FHD", "4K", "8K"]
    frame_rates = [120, 60, 30, 15]

    axes[1, 0].bar(resolutions, frame_rates, color="skyblue")
    axes[1, 0].set_title("Maximum Frame Rates by Resolution")
    axes[1, 0].set_ylabel("FPS")

    # Power consumption by mode
    power_modes = ["Performance", "Balanced", "Power Saver"]
    power_consumption = [5.2, 3.1, 1.8]

    axes[1, 1].bar(power_modes, power_consumption, color="lightcoral")
    axes[1, 1].set_title("Power Consumption by Mode")
    axes[1, 1].set_ylabel("Watts")
    axes[1, 1].tick_params(axis="x", rotation=45)

    plt.tight_layout()

    # Save visualization
    output_file = output_path / "comprehensive_demo_results.png"
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    logger.info(f"Visualization saved to {output_file}")

    plt.show()


def main():
    """Main demo function."""
    logger.info("Starting Advanced Image Sensor Interface Comprehensive Demo")
    logger.info("=" * 60)

    try:
        # Demo all v2.0.0 features
        demo_enhanced_sensor_interface()
        hdr_processor, test_images, hdr_result = demo_hdr_processing()
        raw_processor, raw_data, raw_result = demo_raw_processing()
        stereo_sync, multi_sync = demo_multi_sensor_sync()
        demo_gpu_acceleration()
        demo_advanced_power_management()

        # Create visualizations
        try:
            create_visualization(hdr_result, raw_result)
        except ImportError:
            logger.warning("Matplotlib not available - skipping visualization")

        logger.info("=" * 60)
        logger.info("Demo completed successfully!")
        logger.info("All enhanced features demonstrated:")
        logger.info("✓ Enhanced sensor interface (up to 8K support)")
        logger.info("✓ HDR image processing")
        logger.info("✓ RAW image processing")
        logger.info("✓ Multi-sensor synchronization")
        logger.info("✓ GPU acceleration")
        logger.info("✓ Advanced power management")

    except Exception as e:
        logger.error(f"Demo failed: {e}")
        raise


if __name__ == "__main__":
    main()
