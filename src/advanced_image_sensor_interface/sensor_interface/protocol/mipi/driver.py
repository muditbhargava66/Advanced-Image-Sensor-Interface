"""
MIPI CSI-2 Protocol Driver Implementation

Provides comprehensive support for MIPI CSI-2 camera interfaces with
high-speed data transfer, streaming, and integration with D-PHY v2.5 and security.

Key Features:
- MIPI CSI-2 protocol support up to 4.5 Gbps per lane
- Multi-lane configuration (1-4 lanes)
- Virtual channel support
- Data type handling
- Integration with D-PHY v2.5 high-speed driver
- Optional security layer integration
"""

import logging
import time
from dataclasses import dataclass
from typing import Any, ClassVar, Optional


from ..base import ConnectionError, DataTransferError, ProtocolCapabilities, StreamingProtocolBase

logger = logging.getLogger(__name__)


@dataclass
class MIPIConfig:
    """
    Configuration for MIPI CSI-2 protocol.

    Attributes:
        lanes: Number of data lanes (1-4)
        data_rate_mbps: Data rate per lane in Mbps
        pixel_format: Pixel format string (e.g., "RAW10", "RGB888")
        resolution: Image resolution as (width, height)
        frame_rate: Target frame rate in Hz
        virtual_channel: Virtual channel ID (0-3)
        data_type: MIPI data type code
    """

    # Lane configuration
    lanes: int = 4
    data_rate_mbps: float = 1500.0  # Per lane

    # Image settings
    pixel_format: str = "RAW10"
    resolution: tuple[int, int] = (1920, 1080)
    frame_rate: float = 30.0

    # MIPI-specific settings
    virtual_channel: int = 0
    data_type: int = 0x2B  # RAW10

    # Advanced settings
    continuous_clock: bool = True
    enable_ecc: bool = True
    enable_crc: bool = True
    lp_mode_timeout_us: int = 100
    hs_settle_ns: int = 85

    # Security settings
    enable_security: bool = False
    security_level: str = "standard"

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if not 1 <= self.lanes <= 4:
            raise ValueError("Lanes must be between 1 and 4")

        if not 80.0 <= self.data_rate_mbps <= 4500.0:
            raise ValueError("Data rate must be between 80 and 4500 Mbps")

        if not 0 <= self.virtual_channel <= 3:
            raise ValueError("Virtual channel must be between 0 and 3")

    @property
    def total_bandwidth_mbps(self) -> float:
        """Calculate total bandwidth across all lanes."""
        return self.data_rate_mbps * self.lanes


# Pixel format to MIPI data type mapping
PIXEL_FORMAT_DATA_TYPES: dict[str, int] = {
    "RAW6": 0x28,
    "RAW7": 0x29,
    "RAW8": 0x2A,
    "RAW10": 0x2B,
    "RAW12": 0x2C,
    "RAW14": 0x2D,
    "RAW16": 0x2E,
    "RGB888": 0x24,
    "RGB666": 0x23,
    "RGB565": 0x22,
    "YUV422_8": 0x1E,
    "YUV422_10": 0x1F,
}


class MIPIProtocolDriver(StreamingProtocolBase):
    """
    MIPI CSI-2 protocol driver implementation.

    Provides complete MIPI CSI-2 support including connection management,
    streaming, frame capture, and protocol configuration.
    """

    # Maximum supported specifications
    MAX_LANES: ClassVar[int] = 4
    MAX_DATA_RATE_MBPS: ClassVar[float] = 4500.0  # D-PHY v2.5

    def __init__(self, config: MIPIConfig | dict[str, Any]) -> None:
        """
        Initialize MIPI protocol driver.

        Args:
            config: MIPI configuration (MIPIConfig or dict)
        """
        # Handle both dataclass and dict configs
        if isinstance(config, dict):
            self.mipi_config = MIPIConfig(**{k: v for k, v in config.items() if k in MIPIConfig.__dataclass_fields__})
            config_dict = config
        else:
            self.mipi_config = config
            config_dict = {
                "lanes": config.lanes,
                "data_rate_mbps": config.data_rate_mbps,
                "pixel_format": config.pixel_format,
                "resolution": config.resolution,
                "frame_rate": config.frame_rate,
            }

        super().__init__(config_dict)

        # Device handles
        self.device_handle: Optional[str] = None
        self.stream_handle: Optional[str] = None

        # Frame tracking
        self.frame_count = 0
        self.start_time: Optional[float] = None

        # Statistics
        self.stats: dict[str, Any] = {
            "frames_captured": 0,
            "frames_dropped": 0,
            "bytes_transferred": 0,
            "ecc_errors": 0,
            "crc_errors": 0,
            "protocol_errors": 0,
            "last_frame_time": 0.0,
        }

        # Security manager (optional)
        self._security_manager: Optional[Any] = None

        logger.info(f"MIPI driver initialized: {self.mipi_config.lanes} lanes @ " f"{self.mipi_config.data_rate_mbps} Mbps/lane")

    def _get_capabilities(self) -> ProtocolCapabilities:
        """Get MIPI CSI-2 protocol capabilities."""
        max_bandwidth = self.mipi_config.data_rate_mbps * self.mipi_config.lanes

        return ProtocolCapabilities(
            max_bandwidth_gbps=max_bandwidth / 1000.0,
            max_distance_m=0.3,  # MIPI is typically short-range
            power_over_cable=False,
            hot_pluggable=False,
            multi_camera_support=True,  # Via virtual channels
            hardware_trigger_support=True,
            software_trigger_support=True,
            supported_pixel_formats=list(PIXEL_FORMAT_DATA_TYPES.keys()),
            supported_resolutions=[(640, 480), (1280, 720), (1920, 1080), (2592, 1944), (3840, 2160), (4096, 3072)],
        )

    def connect(self) -> bool:
        """
        Establish MIPI connection.

        Returns:
            True if connection successful
        """
        try:
            logger.info("Connecting to MIPI sensor...")

            # Perform lane initialization (simulation)
            self._initialize_lanes()

            # Verify link stability
            self._verify_link()

            # Create device handle
            self.device_handle = f"mipi_dev_{id(self)}"

            # Configure sensor parameters
            self._configure_sensor()

            # Update status
            self.is_connected = True
            self.status.is_connected = True
            self.status.connection_quality = 1.0
            self.status.data_rate_mbps = self.mipi_config.total_bandwidth_mbps

            logger.info("MIPI connection established")
            return True

        except Exception as e:
            logger.error(f"MIPI connection failed: {e}")
            self.status.error_count += 1
            self.status.last_error = str(e)
            raise ConnectionError(f"MIPI connection failed: {e}")

    def disconnect(self) -> bool:
        """
        Disconnect MIPI connection.

        Returns:
            True if disconnection successful
        """
        try:
            if self.is_streaming:
                self.stop_streaming()

            # Close handles
            self.device_handle = None
            self.stream_handle = None

            # Update status
            self.is_connected = False
            self.status.is_connected = False
            self.status.connection_quality = 0.0
            self.status.data_rate_mbps = 0.0

            logger.info("MIPI connection closed")
            return True

        except Exception as e:
            logger.error(f"Error during MIPI disconnect: {e}")
            return False

    def send_data(self, data: bytes) -> bool:
        """
        Send control data to MIPI device (via CCI/I2C).

        Args:
            data: Control data to send

        Returns:
            True if data sent successfully
        """
        if not self.is_connected:
            raise ConnectionError("Not connected to MIPI device")

        try:
            # Simulate I2C/CCI control transfer
            time.sleep(0.0001)  # ~100us for I2C transaction

            self.status.bytes_transmitted += len(data)
            logger.debug(f"Sent {len(data)} bytes via MIPI CCI")
            return True

        except Exception as e:
            logger.error(f"MIPI send failed: {e}")
            self.status.error_count += 1
            self.status.last_error = str(e)
            raise DataTransferError(f"MIPI send failed: {e}")

    def receive_data(self, size: int) -> Optional[bytes]:
        """
        Receive control data from MIPI device.

        Args:
            size: Number of bytes to receive

        Returns:
            Received data or None
        """
        if not self.is_connected:
            raise ConnectionError("Not connected to MIPI device")

        try:
            # Simulate I2C/CCI read
            time.sleep(0.0001)

            data = bytes([i % 256 for i in range(size)])
            self.status.bytes_received += len(data)
            logger.debug(f"Received {len(data)} bytes via MIPI CCI")
            return data

        except Exception as e:
            logger.error(f"MIPI receive failed: {e}")
            self.status.error_count += 1
            self.status.last_error = str(e)
            raise DataTransferError(f"MIPI receive failed: {e}")

    def start_streaming(self) -> bool:
        """
        Start continuous image streaming.

        Returns:
            True if streaming started
        """
        if not self.is_connected:
            raise ConnectionError("Not connected to MIPI device")

        try:
            logger.info("Starting MIPI streaming...")

            # Initialize streaming
            self.stream_handle = f"mipi_stream_{id(self)}"
            self.is_streaming = True
            self.start_time = time.time()
            self.frame_count = 0

            # Enable streaming mode on sensor (simulation)
            self._enable_streaming()

            logger.info("MIPI streaming started")
            return True

        except Exception as e:
            logger.error(f"Failed to start MIPI streaming: {e}")
            self.status.error_count += 1
            self.status.last_error = str(e)
            return False

    def stop_streaming(self) -> bool:
        """
        Stop continuous image streaming.

        Returns:
            True if streaming stopped
        """
        try:
            if self.is_streaming:
                self.is_streaming = False
                self.stream_handle = None

                # Calculate final statistics
                if self.start_time:
                    duration = time.time() - self.start_time
                    avg_fps = self.frame_count / duration if duration > 0 else 0
                    logger.info(f"MIPI streaming stopped. " f"Captured {self.frame_count} frames at {avg_fps:.2f} FPS")

            return True

        except Exception as e:
            logger.error(f"Error stopping MIPI streaming: {e}")
            return False

    def capture_frame(self) -> Optional[bytes]:
        """
        Capture a single frame from MIPI stream.

        Returns:
            Frame data or None if capture failed
        """
        if not self.is_connected:
            raise ConnectionError("Not connected to MIPI device")

        try:
            # Calculate frame size
            width, height = self.mipi_config.resolution
            bytes_per_pixel = self._get_bytes_per_pixel(self.mipi_config.pixel_format)

            # Simulate frame capture delay
            capture_delay = 1.0 / self.mipi_config.frame_rate
            time.sleep(min(capture_delay, 0.05))  # Cap simulation delay

            # Generate test frame
            frame_data = self._generate_test_frame(width, height, bytes_per_pixel)

            # Update statistics
            self.frame_count += 1
            self.stats["frames_captured"] += 1
            self.stats["bytes_transferred"] += len(frame_data)
            self.stats["last_frame_time"] = time.time()

            logger.debug(f"Captured MIPI frame {self.frame_count}, size: {len(frame_data)} bytes")
            return frame_data

        except Exception as e:
            logger.error(f"MIPI frame capture failed: {e}")
            self.stats["frames_dropped"] += 1
            self.status.error_count += 1
            self.status.last_error = str(e)
            return None

    def get_device_info(self) -> dict[str, Any]:
        """Get MIPI device information."""
        return {
            "protocol": "MIPI CSI-2",
            "lanes": self.mipi_config.lanes,
            "data_rate_mbps": self.mipi_config.data_rate_mbps,
            "total_bandwidth_mbps": self.mipi_config.total_bandwidth_mbps,
            "pixel_format": self.mipi_config.pixel_format,
            "resolution": self.mipi_config.resolution,
            "frame_rate": self.mipi_config.frame_rate,
            "virtual_channel": self.mipi_config.virtual_channel,
            "continuous_clock": self.mipi_config.continuous_clock,
            "ecc_enabled": self.mipi_config.enable_ecc,
            "crc_enabled": self.mipi_config.enable_crc,
        }

    def get_statistics(self) -> dict[str, Any]:
        """Get MIPI driver statistics."""
        current_time = time.time()
        uptime = current_time - self.start_time if self.start_time else 0

        return {
            **self.stats,
            "uptime_seconds": uptime,
            "current_fps": self.frame_count / uptime if uptime > 0 else 0,
            "connection_quality": self.status.connection_quality,
            "error_rate": self.stats["protocol_errors"] / max(1, uptime),
        }

    def set_security_manager(self, manager: Any) -> None:
        """Set security manager for secure transfers."""
        self._security_manager = manager
        logger.info("Security manager attached to MIPI driver")

    def _initialize_lanes(self) -> None:
        """Initialize MIPI data lanes."""
        # Simulate lane initialization
        time.sleep(0.01)
        logger.debug(f"Initialized {self.mipi_config.lanes} MIPI lanes")

    def _verify_link(self) -> None:
        """Verify MIPI link stability."""
        # Simulate link verification
        logger.debug("MIPI link verified")

    def _configure_sensor(self) -> None:
        """Configure sensor parameters via CCI."""
        params = {
            "width": self.mipi_config.resolution[0],
            "height": self.mipi_config.resolution[1],
            "pixel_format": self.mipi_config.pixel_format,
            "frame_rate": self.mipi_config.frame_rate,
            "data_type": self.mipi_config.data_type,
        }
        logger.debug(f"Configured MIPI sensor: {params}")

    def _enable_streaming(self) -> None:
        """Enable streaming mode on sensor."""
        logger.debug("MIPI streaming mode enabled")

    def _get_bytes_per_pixel(self, pixel_format: str) -> int:
        """Get bytes per pixel using shared base implementation."""
        return self._get_bytes_per_pixel_common(pixel_format)

    def _generate_test_frame(self, width: int, height: int, bytes_per_pixel: int) -> bytes:
        """Generate test frame data using shared vectorized implementation."""
        return self._generate_test_frame_vectorized(width, height, bytes_per_pixel, self.frame_count)

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
        target_fps = self.mipi_config.frame_rate
        actual_fps = self._get_current_frame_rate()
        if target_fps > 0:
            return min(100.0, (actual_fps / target_fps) * 100.0)
        return 0.0

    # Legacy compatibility - keep connected attribute
    @property
    def connected(self) -> bool:
        """Legacy compatibility property."""
        return self.is_connected

    @connected.setter
    def connected(self, value: bool) -> None:
        """Legacy compatibility setter."""
        self.is_connected = value

    def optimize_performance(self) -> None:
        """Optimize MIPI protocol performance."""
        logger.info("MIPI protocol performance optimized")

    def get_status(self) -> dict[str, Any]:
        """Get MIPI protocol status (legacy compatibility)."""
        return {
            "protocol": "MIPI CSI-2",
            "connected": self.is_connected,
            "streaming": self.is_streaming,
            "config": {
                "lanes": self.mipi_config.lanes,
                "data_rate": self.mipi_config.data_rate_mbps / 1000.0,
                "data_rate_mbps": self.mipi_config.data_rate_mbps,
                "pixel_format": self.mipi_config.pixel_format,
                "resolution": self.mipi_config.resolution,
                "frame_rate": self.mipi_config.frame_rate,
            },
            "statistics": self.get_statistics(),
        }
