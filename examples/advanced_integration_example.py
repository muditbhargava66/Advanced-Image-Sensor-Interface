"""
Advanced Integration Example for the Advanced Image Sensor Interface v2.0.0

This example demonstrates the enhanced features including:
- Type-safe operations with comprehensive type annotations
- Performance profiling and monitoring
- Enhanced error handling with recovery strategies
- USB3 Vision protocol support
- Real-time performance monitoring
- Advanced buffer management with optimization
"""

import asyncio
import logging
import time
from typing import Any, Optional

from advanced_image_sensor_interface.error_handling.exceptions import (
    ConnectionError,
    SensorError,
    create_connection_timeout_error,
)
from advanced_image_sensor_interface.error_handling.recovery import ErrorRecoveryManager, RecoveryStrategy
from advanced_image_sensor_interface.performance.monitor import ApplicationMetrics, PerformanceMonitor
from advanced_image_sensor_interface.performance.profiler import PerformanceProfiler, profile_async_function, profile_function
from advanced_image_sensor_interface.sensor_interface.protocol.usb3.driver import USB3VisionConfig, USB3VisionDriver

# Core imports with enhanced type safety
from advanced_image_sensor_interface.types import (
    BufferManagerInterface,
    ImageArray,
    PerformanceMetrics,
    PixelFormat,
    PowerMetrics,
    ProtocolInterface,
    Resolution,
)

# Enhanced components
from advanced_image_sensor_interface.utils.buffer_manager import AsyncBufferManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AdvancedSensorSystem:
    """
    Advanced sensor system demonstrating v2.0.0 enhancements.

    Features:
    - Type-safe operations
    - Performance monitoring
    - Error recovery
    - Multiple protocol support
    - Real-time optimization
    """

    def __init__(self):
        """Initialize the advanced sensor system."""
        # Performance monitoring
        self.profiler = PerformanceProfiler(enable_memory_profiling=True, enable_cpu_profiling=True)
        self.monitor = PerformanceMonitor(sampling_interval=1.0, enable_alerts=True)

        # Error handling
        self.recovery_manager = ErrorRecoveryManager()

        # Buffer management
        self.buffer_manager: Optional[AsyncBufferManager] = None

        # Protocol drivers
        self.usb3_driver: Optional[USB3VisionDriver] = None

        # System state
        self.is_running = False
        self.frame_count = 0
        self.start_time: Optional[float] = None

        # Setup alert callbacks
        self.monitor.add_alert_callback(self._handle_performance_alert)

    def _handle_performance_alert(self, alert_type: str, alert_data: dict[str, Any]) -> None:
        """Handle performance alerts."""
        logger.warning(f"Performance alert: {alert_type} - {alert_data}")

        # Implement automatic recovery strategies
        if alert_type == "high_memory":
            self._optimize_memory_usage()
        elif alert_type == "high_cpu":
            self._reduce_processing_load()

    def _optimize_memory_usage(self) -> None:
        """Optimize memory usage in response to alerts."""
        if self.buffer_manager:
            # Trigger buffer pool optimization
            optimization_task = asyncio.create_task(self.buffer_manager.optimize_pools_async())
            # Store reference to prevent garbage collection
            if not hasattr(self, "_background_tasks"):
                self._background_tasks = set()
            self._background_tasks.add(optimization_task)
            optimization_task.add_done_callback(self._background_tasks.discard)
            logger.info("Triggered buffer pool optimization")

    def _reduce_processing_load(self) -> None:
        """Reduce processing load in response to high CPU usage."""
        # Could reduce frame rate, disable non-essential processing, etc.
        logger.info("Reducing processing load due to high CPU usage")

    @profile_function
    def initialize_system(self) -> bool:
        """Initialize the sensor system with enhanced error handling."""
        try:
            logger.info("Initializing Advanced Sensor System v2.0.0...")

            # Initialize buffer manager
            self.buffer_manager = AsyncBufferManager(max_pool_size=200, enable_optimization=True)

            # Initialize USB3 Vision driver
            usb3_config = USB3VisionConfig(
                pixel_format="Mono16", resolution=(2048, 1536), frame_rate=60.0, usb_speed="SuperSpeedPlus", buffer_count=20
            )

            self.usb3_driver = USB3VisionDriver(usb3_config)

            # Start performance monitoring
            self.monitor.start_monitoring()

            logger.info("System initialization completed successfully")
            return True

        except Exception as e:
            logger.error(f"System initialization failed: {e}")
            return False

    @profile_async_function
    async def connect_devices(self) -> bool:
        """Connect to sensor devices with retry logic."""
        max_retries = 3
        retry_delay = 2.0

        for attempt in range(max_retries):
            try:
                logger.info(f"Connecting to devices (attempt {attempt + 1}/{max_retries})...")

                if self.usb3_driver:
                    if not self.usb3_driver.connect():
                        raise ConnectionError("Failed to connect to USB3 Vision device")

                logger.info("All devices connected successfully")
                return True

            except ConnectionError as e:
                logger.warning(f"Connection attempt {attempt + 1} failed: {e}")

                if attempt < max_retries - 1:
                    logger.info(f"Retrying in {retry_delay} seconds...")
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 1.5  # Exponential backoff
                else:
                    logger.error("All connection attempts failed")
                    return False

        return False

    @profile_async_function
    async def start_streaming(self) -> bool:
        """Start streaming with performance monitoring."""
        try:
            if not self.usb3_driver:
                raise SensorError("USB3 driver not initialized")

            # Start streaming
            if not self.usb3_driver.start_streaming():
                raise SensorError("Failed to start streaming")

            self.is_running = True
            self.start_time = time.time()

            logger.info("Streaming started successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to start streaming: {e}")
            return False

    @profile_async_function
    async def capture_and_process_frames(self, num_frames: int = 100) -> dict[str, Any]:
        """Capture and process frames with performance tracking."""
        if not self.is_running or not self.usb3_driver:
            raise SensorError("System not running or driver not available")

        results = {"frames_captured": 0, "frames_processed": 0, "errors": 0, "average_fps": 0.0, "processing_time_ms": 0.0}

        processing_start = time.time()

        try:
            for i in range(num_frames):
                # Capture frame
                frame_data = self.usb3_driver.capture_frame()

                if frame_data:
                    results["frames_captured"] += 1

                    # Get buffer for processing
                    buffer = await self.buffer_manager.get_buffer_async(len(frame_data))

                    if buffer:
                        try:
                            # Simulate frame processing
                            await self._process_frame_async(frame_data, buffer)
                            results["frames_processed"] += 1

                        finally:
                            # Always return buffer
                            await self.buffer_manager.return_buffer_async(buffer)
                    else:
                        logger.warning("Failed to get processing buffer")

                else:
                    results["errors"] += 1

                # Update application metrics
                if i % 10 == 0:  # Every 10 frames
                    await self._update_application_metrics()

                # Small delay to prevent overwhelming the system
                await asyncio.sleep(0.001)

        except Exception as e:
            logger.error(f"Error during frame processing: {e}")
            results["errors"] += 1

        # Calculate final statistics
        processing_time = time.time() - processing_start
        results["processing_time_ms"] = processing_time * 1000

        if processing_time > 0:
            results["average_fps"] = results["frames_captured"] / processing_time

        return results

    async def _process_frame_async(self, frame_data: bytes, buffer: memoryview) -> None:
        """Process a single frame asynchronously."""
        # Copy frame data to buffer
        buffer[: len(frame_data)] = frame_data

        # Simulate processing (could be actual image processing)
        await asyncio.sleep(0.001)  # 1ms processing time

        self.frame_count += 1

    async def _update_application_metrics(self) -> None:
        """Update application performance metrics."""
        if not self.start_time:
            return

        current_time = time.time()
        elapsed = current_time - self.start_time

        # Calculate current FPS
        fps = self.frame_count / elapsed if elapsed > 0 else 0.0

        # Get buffer utilization
        buffer_stats = self.buffer_manager.get_statistics() if self.buffer_manager else {}
        buffer_utilization = buffer_stats.get("pool_utilization_percent", 0.0)

        # Create application metrics
        metrics = ApplicationMetrics(
            timestamp=current_time,
            frames_per_second=fps,
            buffer_utilization=buffer_utilization,
            active_connections=1 if self.usb3_driver and self.usb3_driver.is_connected else 0,
            error_rate=0.01,  # 1% error rate (example)
            latency_ms=5.0,  # 5ms latency (example)
            throughput_mbps=fps * 2048 * 1536 * 2 / (1024 * 1024),  # Approximate throughput
        )

        # Record metrics
        self.monitor.record_application_metrics(metrics)

    def generate_performance_report(self) -> str:
        """Generate comprehensive performance report."""
        report_lines = ["Advanced Sensor System Performance Report", "=" * 50, ""]

        # Profiling results
        profiler_report = self.profiler.generate_report()
        report_lines.extend(["Profiling Results:", "-" * 18, profiler_report, ""])

        # System monitoring results
        system_stats = self.monitor.get_summary_stats(window_minutes=5)
        report_lines.extend(["System Monitoring (Last 5 minutes):", "-" * 35])

        if "system" in system_stats:
            sys_stats = system_stats["system"]
            report_lines.extend(
                [
                    f"Average CPU Usage: {sys_stats.get('cpu_percent_avg', 0):.1f}%",
                    f"Peak CPU Usage: {sys_stats.get('cpu_percent_max', 0):.1f}%",
                    f"Average Memory Usage: {sys_stats.get('memory_percent_avg', 0):.1f}%",
                    f"Peak Memory Usage: {sys_stats.get('memory_percent_max', 0):.1f}%",
                ]
            )

        if "application" in system_stats:
            app_stats = system_stats["application"]
            report_lines.extend(
                [
                    "",
                    "Application Performance:",
                    f"Average FPS: {app_stats.get('fps_avg', 0):.1f}",
                    f"Average Latency: {app_stats.get('latency_ms_avg', 0):.1f}ms",
                    f"Average Throughput: {app_stats.get('throughput_mbps_avg', 0):.1f} MB/s",
                    f"Error Rate: {app_stats.get('error_rate_avg', 0):.3f}",
                ]
            )

        return "\n".join(report_lines)

    async def shutdown(self) -> None:
        """Shutdown the system gracefully."""
        logger.info("Shutting down Advanced Sensor System...")

        self.is_running = False

        # Stop streaming
        if self.usb3_driver and self.usb3_driver.is_streaming:
            self.usb3_driver.stop_streaming()

        # Disconnect devices
        if self.usb3_driver and self.usb3_driver.is_connected:
            self.usb3_driver.disconnect()

        # Stop monitoring
        self.monitor.stop_monitoring()

        logger.info("System shutdown completed")


async def main():
    """Main demonstration function."""
    system = AdvancedSensorSystem()

    try:
        # Initialize system
        if not system.initialize_system():
            logger.error("Failed to initialize system")
            return

        # Connect devices
        if not await system.connect_devices():
            logger.error("Failed to connect devices")
            return

        # Start streaming
        if not await system.start_streaming():
            logger.error("Failed to start streaming")
            return

        # Capture and process frames
        logger.info("Starting frame capture and processing...")
        results = await system.capture_and_process_frames(num_frames=50)

        # Display results
        logger.info("Frame processing completed:")
        logger.info(f"  Frames captured: {results['frames_captured']}")
        logger.info(f"  Frames processed: {results['frames_processed']}")
        logger.info(f"  Errors: {results['errors']}")
        logger.info(f"  Average FPS: {results['average_fps']:.1f}")
        logger.info(f"  Processing time: {results['processing_time_ms']:.1f}ms")

        # Wait a bit for monitoring data
        await asyncio.sleep(2.0)

        # Generate performance report
        report = system.generate_performance_report()
        print("\n" + report)

    except Exception as e:
        logger.error(f"Error in main execution: {e}")

    finally:
        # Shutdown system
        await system.shutdown()


if __name__ == "__main__":
    # Run the advanced integration example
    asyncio.run(main())
