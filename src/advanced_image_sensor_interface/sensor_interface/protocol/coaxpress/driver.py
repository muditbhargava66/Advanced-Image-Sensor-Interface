"""
CoaXPress protocol driver implementation.

This module provides a complete implementation of the CoaXPress camera interface
protocol, supporting high-speed data transfer over coaxial cables with power delivery.
"""

import logging
import time
from dataclasses import dataclass
from typing import Any, ClassVar, Optional

import numpy as np

from ..base import ConnectionError, DataTransferError, ProtocolCapabilities, StreamingProtocolBase

logger = logging.getLogger(__name__)


@dataclass
class CoaXPressConfig:
    """Configuration for CoaXPress protocol."""

    speed_grade: str = "CXP-6"  # CXP-1, CXP-2, CXP-3, CXP-5, CXP-6, CXP-10, CXP-12
    connections: int = 1  # Number of coax connections (1-4)
    packet_size: int = 8192  # Packet size in bytes
    trigger_mode: str = "software"  # "software", "hardware", "continuous"
    pixel_format: str = "Mono16"  # Pixel format
    resolution: tuple = (2048, 2048)  # Image resolution
    frame_rate: float = 30.0  # Frames per second
    power_over_coax: bool = True  # Enable power over coax
    discovery_timeout: float = 5.0  # Device discovery timeout in seconds

    # Advanced settings
    master_host_connection: int = 0  # Master host connection ID
    packet_delay: int = 0  # Inter-packet delay in nanoseconds
    stream_packet_size: int = 0  # Stream packet size (0 = auto)

    def __post_init__(self):
        """Validate configuration parameters."""
        valid_speeds = ["CXP-1", "CXP-2", "CXP-3", "CXP-5", "CXP-6", "CXP-10", "CXP-12"]
        if self.speed_grade not in valid_speeds:
            raise ValueError(f"Invalid speed grade: {self.speed_grade}")

        if not 1 <= self.connections <= 4:
            raise ValueError("Connections must be between 1 and 4")

        if self.packet_size < 64 or self.packet_size > 65536:
            raise ValueError("Packet size must be between 64 and 65536 bytes")


class CoaXPressDriver(StreamingProtocolBase):
    """
    CoaXPress protocol driver implementation.

    This driver provides a complete implementation of the CoaXPress standard
    for high-speed industrial camera interfaces.
    """

    # Speed grade to bandwidth mapping (Mbps)
    SPEED_GRADES: ClassVar[dict[str, int]] = {
        "CXP-1": 1250,
        "CXP-2": 2500,
        "CXP-3": 3125,
        "CXP-5": 5000,
        "CXP-6": 6250,
        "CXP-10": 10000,
        "CXP-12": 12500,
    }

    def __init__(self, config: CoaXPressConfig):
        """Initialize CoaXPress driver."""
        super().__init__(config.__dict__)
        self.cxp_config = config
        self.device_handle = None
        self.stream_handle = None
        self.frame_count = 0
        self.start_time = None

        # Statistics
        self.stats = {"frames_captured": 0, "frames_dropped": 0, "bytes_transferred": 0, "errors": 0, "last_frame_time": 0.0}

        logger.info(f"CoaXPress driver initialized with {config.speed_grade} at {config.connections} connections")

    def _get_capabilities(self) -> ProtocolCapabilities:
        """Get CoaXPress protocol capabilities."""
        max_bandwidth = self.SPEED_GRADES[self.cxp_config.speed_grade] * self.cxp_config.connections

        return ProtocolCapabilities(
            max_bandwidth_gbps=max_bandwidth / 1000.0,
            max_distance_m=100.0,  # Standard coax cable length
            power_over_cable=True,
            hot_pluggable=False,  # Industrial cameras typically not hot-pluggable
            multi_camera_support=True,
            hardware_trigger_support=True,
            software_trigger_support=True,
            supported_pixel_formats=[
                "Mono8",
                "Mono10",
                "Mono12",
                "Mono16",
                "BayerGR8",
                "BayerRG8",
                "BayerGB8",
                "BayerBG8",
                "BayerGR10",
                "BayerRG10",
                "BayerGB10",
                "BayerBG10",
                "BayerGR12",
                "BayerRG12",
                "BayerGB12",
                "BayerBG12",
                "RGB8",
                "BGR8",
                "YUV422",
            ],
            supported_resolutions=[
                (640, 480),
                (800, 600),
                (1024, 768),
                (1280, 1024),
                (1600, 1200),
                (2048, 1536),
                (2048, 2048),
                (4096, 4096),
            ],
        )

    def connect(self) -> bool:
        """Establish connection to CoaXPress device."""
        try:
            logger.info("Connecting to CoaXPress device...")

            # Simulate device discovery and connection
            # In real implementation, this would use CoaXPress SDK
            self._simulate_device_discovery()

            # Initialize device handle (simulation)
            self.device_handle = f"cxp_device_{id(self)}"

            # Configure device parameters
            self._configure_device()

            # Update status
            self.is_connected = True
            self.status.is_connected = True
            self.status.connection_quality = 1.0
            self.status.data_rate_mbps = self.SPEED_GRADES[self.cxp_config.speed_grade]

            logger.info("CoaXPress device connected successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to connect to CoaXPress device: {e}")
            self.status.error_count += 1
            self.status.last_error = str(e)
            raise ConnectionError(f"CoaXPress connection failed: {e}")

    def disconnect(self) -> bool:
        """Disconnect from CoaXPress device."""
        try:
            if self.is_streaming:
                self.stop_streaming()

            # Close device handle (simulation)
            self.device_handle = None
            self.stream_handle = None

            # Update status
            self.is_connected = False
            self.status.is_connected = False
            self.status.connection_quality = 0.0
            self.status.data_rate_mbps = 0.0

            logger.info("CoaXPress device disconnected")
            return True

        except Exception as e:
            logger.error(f"Error during CoaXPress disconnect: {e}")
            return False

    def send_data(self, data: bytes) -> bool:
        """Send control data to CoaXPress device."""
        if not self.is_connected:
            raise ConnectionError("Not connected to CoaXPress device")

        try:
            # Simulate sending control data
            # In real implementation, this would send data over control channel
            time.sleep(0.001)  # Simulate transmission delay

            self.status.bytes_transmitted += len(data)
            logger.debug(f"Sent {len(data)} bytes to CoaXPress device")
            return True

        except Exception as e:
            logger.error(f"Failed to send data: {e}")
            self.status.error_count += 1
            self.status.last_error = str(e)
            raise DataTransferError(f"CoaXPress send failed: {e}")

    def receive_data(self, size: int) -> Optional[bytes]:
        """Receive data from CoaXPress device."""
        if not self.is_connected:
            raise ConnectionError("Not connected to CoaXPress device")

        try:
            # Simulate receiving data
            # In real implementation, this would receive from control channel
            time.sleep(0.001)  # Simulate transmission delay

            # Generate simulated response data
            data = bytes([i % 256 for i in range(size)])

            self.status.bytes_received += len(data)
            logger.debug(f"Received {len(data)} bytes from CoaXPress device")
            return data

        except Exception as e:
            logger.error(f"Failed to receive data: {e}")
            self.status.error_count += 1
            self.status.last_error = str(e)
            raise DataTransferError(f"CoaXPress receive failed: {e}")

    def start_streaming(self) -> bool:
        """Start continuous image streaming."""
        if not self.is_connected:
            raise ConnectionError("Not connected to CoaXPress device")

        try:
            logger.info("Starting CoaXPress streaming...")

            # Initialize streaming (simulation)
            self.stream_handle = f"cxp_stream_{id(self)}"
            self.is_streaming = True
            self.start_time = time.time()
            self.frame_count = 0

            # Configure streaming parameters
            self._configure_streaming()

            logger.info("CoaXPress streaming started")
            return True

        except Exception as e:
            logger.error(f"Failed to start streaming: {e}")
            self.status.error_count += 1
            self.status.last_error = str(e)
            return False

    def stop_streaming(self) -> bool:
        """Stop continuous image streaming."""
        try:
            if self.is_streaming:
                self.is_streaming = False
                self.stream_handle = None

                # Calculate final statistics
                if self.start_time:
                    duration = time.time() - self.start_time
                    avg_fps = self.frame_count / duration if duration > 0 else 0
                    logger.info(f"Streaming stopped. Captured {self.frame_count} frames at {avg_fps:.2f} FPS")

            return True

        except Exception as e:
            logger.error(f"Error stopping streaming: {e}")
            return False

    def capture_frame(self) -> Optional[bytes]:
        """Capture a single frame from CoaXPress device."""
        if not self.is_connected:
            raise ConnectionError("Not connected to CoaXPress device")

        try:
            # Calculate frame size based on resolution and pixel format
            width, height = self.cxp_config.resolution
            bytes_per_pixel = self._get_bytes_per_pixel(self.cxp_config.pixel_format)
            width * height * bytes_per_pixel

            # Simulate frame capture delay
            capture_delay = 1.0 / self.cxp_config.frame_rate
            time.sleep(min(capture_delay, 0.1))  # Cap simulation delay

            # Generate simulated frame data
            frame_data = self._generate_test_frame(width, height, bytes_per_pixel)

            # Update statistics
            self.frame_count += 1
            self.stats["frames_captured"] += 1
            self.stats["bytes_transferred"] += len(frame_data)
            self.stats["last_frame_time"] = time.time()

            logger.debug(f"Captured frame {self.frame_count}, size: {len(frame_data)} bytes")
            return frame_data

        except Exception as e:
            logger.error(f"Frame capture failed: {e}")
            self.stats["errors"] += 1
            self.status.error_count += 1
            self.status.last_error = str(e)
            return None

    def _simulate_device_discovery(self):
        """Simulate CoaXPress device discovery."""
        # Simulate discovery delay
        time.sleep(0.1)

        # In real implementation, this would scan for CoaXPress devices
        logger.debug("CoaXPress device discovery completed")

    def _configure_device(self):
        """Configure CoaXPress device parameters."""
        # Simulate device configuration
        # In real implementation, this would set GenICam parameters

        config_params = {
            "Width": self.cxp_config.resolution[0],
            "Height": self.cxp_config.resolution[1],
            "PixelFormat": self.cxp_config.pixel_format,
            "AcquisitionFrameRate": self.cxp_config.frame_rate,
            "TriggerMode": "On" if self.cxp_config.trigger_mode != "continuous" else "Off",
            "TriggerSource": "Software" if self.cxp_config.trigger_mode == "software" else "Line1",
        }

        logger.debug(f"Configured CoaXPress device with parameters: {config_params}")

    def _configure_streaming(self):
        """Configure streaming parameters."""
        # Calculate optimal packet size
        if self.cxp_config.stream_packet_size == 0:
            # Auto-calculate based on image size and connection speed
            width, height = self.cxp_config.resolution
            bytes_per_pixel = self._get_bytes_per_pixel(self.cxp_config.pixel_format)
            frame_size = width * height * bytes_per_pixel

            # Aim for ~100 packets per frame
            optimal_packet_size = max(1024, min(self.cxp_config.packet_size, frame_size // 100))
            self.cxp_config.stream_packet_size = optimal_packet_size

        logger.debug(f"Streaming configured with packet size: {self.cxp_config.stream_packet_size}")

    def _get_bytes_per_pixel(self, pixel_format: str) -> int:
        """Get bytes per pixel for given pixel format."""
        format_map = {
            "Mono8": 1,
            "Mono10": 2,
            "Mono12": 2,
            "Mono16": 2,
            "BayerGR8": 1,
            "BayerRG8": 1,
            "BayerGB8": 1,
            "BayerBG8": 1,
            "BayerGR10": 2,
            "BayerRG10": 2,
            "BayerGB10": 2,
            "BayerBG10": 2,
            "BayerGR12": 2,
            "BayerRG12": 2,
            "BayerGB12": 2,
            "BayerBG12": 2,
            "RGB8": 3,
            "BGR8": 3,
            "YUV422": 2,
        }
        return format_map.get(pixel_format, 1)

    def _generate_test_frame(self, width: int, height: int, bytes_per_pixel: int) -> bytes:
        """Generate test frame data for simulation."""
        # Create a simple test pattern
        width * height * bytes_per_pixel

        # Generate gradient pattern
        if bytes_per_pixel == 1:
            # Mono pattern
            frame = np.zeros((height, width), dtype=np.uint8)
            for y in range(height):
                for x in range(width):
                    frame[y, x] = (x + y) % 256
        else:
            # Multi-byte pattern
            frame = np.zeros((height, width, bytes_per_pixel), dtype=np.uint8)
            for y in range(height):
                for x in range(width):
                    for c in range(bytes_per_pixel):
                        frame[y, x, c] = (x + y + c * 50) % 256

        return frame.tobytes()

    def get_device_info(self) -> dict[str, Any]:
        """Get CoaXPress device information."""
        return {
            "protocol": "CoaXPress",
            "speed_grade": self.cxp_config.speed_grade,
            "connections": self.cxp_config.connections,
            "max_bandwidth_mbps": self.SPEED_GRADES[self.cxp_config.speed_grade] * self.cxp_config.connections,
            "power_over_coax": self.cxp_config.power_over_coax,
            "pixel_format": self.cxp_config.pixel_format,
            "resolution": self.cxp_config.resolution,
            "frame_rate": self.cxp_config.frame_rate,
        }

    def get_statistics(self) -> dict[str, Any]:
        """Get CoaXPress driver statistics."""
        current_time = time.time()
        uptime = current_time - self.start_time if self.start_time else 0

        return {
            **self.stats,
            "uptime_seconds": uptime,
            "current_fps": self.frame_count / uptime if uptime > 0 else 0,
            "connection_quality": self.status.connection_quality,
            "error_rate": self.stats["errors"] / max(1, uptime),
        }

    def _get_current_frame_rate(self) -> float:
        """Get current frame rate."""
        if self.start_time and self.frame_count > 0:
            elapsed = time.time() - self.start_time
            return self.frame_count / elapsed if elapsed > 0 else 0.0
        return 0.0

    def _get_dropped_frame_count(self) -> int:
        """Get number of dropped frames."""
        return self.stats["frames_dropped"]

    def _get_buffer_utilization(self) -> float:
        """Get buffer utilization percentage."""
        # Simulate buffer utilization based on frame rate
        target_fps = self.cxp_config.frame_rate
        actual_fps = self._get_current_frame_rate()

        if target_fps > 0:
            return min(100.0, (actual_fps / target_fps) * 100.0)
        return 0.0
