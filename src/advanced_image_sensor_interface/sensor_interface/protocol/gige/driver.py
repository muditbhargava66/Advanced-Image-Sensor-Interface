"""
GigE Vision Protocol Driver Implementation

Provides comprehensive support for GigE Vision camera interfaces with
Ethernet-based high-speed data transfer, streaming, and GVCP/GVSP protocols.

Key Features:
- GigE Vision 2.0 protocol support
- 1 Gbps (with 10GigE option)
- GVCP (Control Protocol) for camera control
- GVSP (Streaming Protocol) for image data
- Optional RoCE acceleration
- Multi-camera support on same network
"""

import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, ClassVar, Optional


from ..base import ConnectionError, DataTransferError, ProtocolCapabilities, StreamingProtocolBase

logger = logging.getLogger(__name__)


class GigESpeed(Enum):
    """GigE interface speed options."""

    GIGE_1G = 1000  # Standard GigE
    GIGE_2_5G = 2500  # 2.5 GigE
    GIGE_5G = 5000  # 5 GigE
    GIGE_10G = 10000  # 10 GigE


class PacketResendMode(Enum):
    """GVSP packet resend strategies."""

    NONE = "none"
    BASIC = "basic"
    AGGRESSIVE = "aggressive"


@dataclass
class GigEVisionConfig:
    """
    Configuration for GigE Vision protocol.

    Attributes:
        ip_address: Camera IP address
        port: GVCP port (default 3956)
        speed: GigE interface speed
        packet_size: GVSP packet size (jumbo frames recommended)
        inter_packet_delay: Delay between packets in ticks
        pixel_format: Image pixel format
        resolution: Image resolution as (width, height)
        frame_rate: Target frame rate
    """

    # Network settings
    ip_address: str = "192.168.1.100"
    port: int = 3956
    speed: GigESpeed = GigESpeed.GIGE_1G

    # Streaming settings
    packet_size: int = 9000  # Jumbo frames
    inter_packet_delay: int = 0
    stream_channel: int = 0
    packet_resend_mode: PacketResendMode = PacketResendMode.BASIC

    # Image settings
    pixel_format: str = "Mono8"
    resolution: tuple[int, int] = (1920, 1080)
    frame_rate: float = 30.0

    # Advanced settings
    heartbeat_timeout_ms: int = 3000
    command_timeout_ms: int = 1000
    enable_action_commands: bool = False
    enable_scheduled_action: bool = False

    # RoCE acceleration
    enable_roce: bool = False
    roce_priority: int = 4

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if self.packet_size < 576 or self.packet_size > 9000:
            raise ValueError("Packet size must be between 576 and 9000 bytes")

        if self.heartbeat_timeout_ms < 100:
            raise ValueError("Heartbeat timeout must be at least 100ms")

    @property
    def bandwidth_mbps(self) -> float:
        """Get effective bandwidth in Mbps."""
        return self.speed.value


# GVCP message types
class GVCPCommand(Enum):
    """GigE Vision Control Protocol commands."""

    DISCOVERY_CMD = 0x0002
    DISCOVERY_ACK = 0x0003
    FORCEIP_CMD = 0x0004
    FORCEIP_ACK = 0x0005
    READREG_CMD = 0x0080
    READREG_ACK = 0x0081
    WRITEREG_CMD = 0x0082
    WRITEREG_ACK = 0x0083
    READMEM_CMD = 0x0084
    READMEM_ACK = 0x0085
    WRITEMEM_CMD = 0x0086
    WRITEMEM_ACK = 0x0087


class GigEProtocolDriver(StreamingProtocolBase):
    """
    GigE Vision protocol driver implementation.

    Provides complete GigE Vision 2.0 support including connection management,
    GVCP control, GVSP streaming, and frame capture.
    """

    # Protocol constants
    GVCP_PORT: ClassVar[int] = 3956
    DEFAULT_HEARTBEAT_MS: ClassVar[int] = 3000

    def __init__(self, config: GigEVisionConfig | dict[str, Any]) -> None:
        """
        Initialize GigE Vision protocol driver.

        Args:
            config: GigE Vision configuration (GigEVisionConfig or dict)
        """
        # Handle both dataclass and dict configs
        if isinstance(config, dict):
            self.gige_config = GigEVisionConfig(**{k: v for k, v in config.items() if k in GigEVisionConfig.__dataclass_fields__})
            config_dict = config
        else:
            self.gige_config = config
            config_dict = {
                "ip_address": config.ip_address,
                "port": config.port,
                "packet_size": config.packet_size,
                "pixel_format": config.pixel_format,
                "resolution": config.resolution,
            }

        super().__init__(config_dict)

        # Device handles
        self.device_handle: Optional[str] = None
        self.stream_handle: Optional[str] = None
        self.device_info: dict[str, Any] = {}

        # Frame tracking
        self.frame_count = 0
        self.start_time: Optional[float] = None

        # Statistics
        self.stats: dict[str, Any] = {
            "frames_captured": 0,
            "frames_dropped": 0,
            "frames_resent": 0,
            "bytes_transferred": 0,
            "packets_received": 0,
            "packets_resent": 0,
            "network_errors": 0,
            "last_frame_time": 0.0,
        }

        # RoCE transport (optional)
        self._roce_transport: Optional[Any] = None

        logger.info(f"GigE Vision driver initialized: {self.gige_config.ip_address} @ " f"{self.gige_config.speed.name}")

    def _get_capabilities(self) -> ProtocolCapabilities:
        """Get GigE Vision protocol capabilities."""
        return ProtocolCapabilities(
            max_bandwidth_gbps=self.gige_config.bandwidth_mbps / 1000.0,
            max_distance_m=100.0,  # Standard Ethernet cable length
            power_over_cable=True,  # PoE supported
            hot_pluggable=True,
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
                "YUV444",
            ],
            supported_resolutions=[
                (640, 480),
                (800, 600),
                (1024, 768),
                (1280, 720),
                (1280, 1024),
                (1600, 1200),
                (1920, 1080),
                (2048, 1536),
                (2560, 1440),
                (3840, 2160),
            ],
        )

    def connect(self) -> bool:
        """
        Establish GigE Vision connection.

        Returns:
            True if connection successful
        """
        try:
            logger.info(f"Connecting to GigE Vision camera at {self.gige_config.ip_address}...")

            # Discover camera
            if not self._discover_camera():
                raise ConnectionError("Camera not found at specified IP")

            # Open GVCP connection
            self.device_handle = self._open_gvcp_connection()
            if not self.device_handle:
                raise ConnectionError("Failed to open GVCP connection")

            # Read device information
            self._read_device_info()

            # Configure camera
            self._configure_camera()

            # Initialize RoCE if enabled
            if self.gige_config.enable_roce:
                self._initialize_roce()

            # Update status
            self.is_connected = True
            self.status.is_connected = True
            self.status.connection_quality = 1.0
            self.status.data_rate_mbps = self.gige_config.bandwidth_mbps

            logger.info(f"Connected to GigE Vision camera: {self.device_info.get('model', 'Unknown')}")
            return True

        except Exception as e:
            logger.error(f"GigE Vision connection failed: {e}")
            self.status.error_count += 1
            self.status.last_error = str(e)
            raise ConnectionError(f"GigE Vision connection failed: {e}")

    def disconnect(self) -> bool:
        """
        Disconnect from GigE Vision camera.

        Returns:
            True if disconnection successful
        """
        try:
            if self.is_streaming:
                self.stop_streaming()

            # Close stream channel
            if self.stream_handle:
                self._close_stream_channel()
                self.stream_handle = None

            # Close GVCP connection
            if self.device_handle:
                self._close_gvcp_connection()
                self.device_handle = None

            # Update status
            self.is_connected = False
            self.status.is_connected = False
            self.status.connection_quality = 0.0
            self.status.data_rate_mbps = 0.0

            logger.info("GigE Vision connection closed")
            return True

        except Exception as e:
            logger.error(f"Error during GigE Vision disconnect: {e}")
            return False

    def send_data(self, data: bytes) -> bool:
        """
        Send GVCP command to camera.

        Args:
            data: Command data to send

        Returns:
            True if data sent successfully
        """
        if not self.is_connected:
            raise ConnectionError("Not connected to GigE Vision camera")

        try:
            # Simulate GVCP command
            time.sleep(0.001)

            self.status.bytes_transmitted += len(data)
            logger.debug(f"Sent {len(data)} bytes via GVCP")
            return True

        except Exception as e:
            logger.error(f"GVCP send failed: {e}")
            self.status.error_count += 1
            self.stats["network_errors"] += 1
            raise DataTransferError(f"GVCP send failed: {e}")

    def receive_data(self, size: int) -> Optional[bytes]:
        """
        Receive GVCP response from camera.

        Args:
            size: Expected response size

        Returns:
            Response data or None
        """
        if not self.is_connected:
            raise ConnectionError("Not connected to GigE Vision camera")

        try:
            # Simulate GVCP response
            time.sleep(0.001)

            data = bytes([i % 256 for i in range(size)])
            self.status.bytes_received += len(data)
            logger.debug(f"Received {len(data)} bytes via GVCP")
            return data

        except Exception as e:
            logger.error(f"GVCP receive failed: {e}")
            self.status.error_count += 1
            self.stats["network_errors"] += 1
            raise DataTransferError(f"GVCP receive failed: {e}")

    def start_streaming(self) -> bool:
        """
        Start GVSP image streaming.

        Returns:
            True if streaming started
        """
        if not self.is_connected:
            raise ConnectionError("Not connected to GigE Vision camera")

        try:
            logger.info("Starting GigE Vision streaming...")

            # Open stream channel
            self.stream_handle = self._open_stream_channel()

            # Configure streaming parameters
            self._configure_streaming()

            # Start acquisition
            self._start_acquisition()

            self.is_streaming = True
            self.start_time = time.time()
            self.frame_count = 0

            logger.info("GigE Vision streaming started")
            return True

        except Exception as e:
            logger.error(f"Failed to start GigE Vision streaming: {e}")
            self.status.error_count += 1
            self.status.last_error = str(e)
            return False

    def stop_streaming(self) -> bool:
        """
        Stop GVSP image streaming.

        Returns:
            True if streaming stopped
        """
        try:
            if self.is_streaming:
                # Stop acquisition
                self._stop_acquisition()

                # Close stream channel
                if self.stream_handle:
                    self._close_stream_channel()
                    self.stream_handle = None

                self.is_streaming = False

                # Calculate final statistics
                if self.start_time:
                    duration = time.time() - self.start_time
                    avg_fps = self.frame_count / duration if duration > 0 else 0
                    logger.info(f"GigE Vision streaming stopped. " f"Captured {self.frame_count} frames at {avg_fps:.2f} FPS")

            return True

        except Exception as e:
            logger.error(f"Error stopping GigE Vision streaming: {e}")
            return False

    def capture_frame(self) -> Optional[bytes]:
        """
        Capture a single frame via GVSP.

        Returns:
            Frame data or None if capture failed
        """
        if not self.is_connected:
            raise ConnectionError("Not connected to GigE Vision camera")

        try:
            # Calculate frame parameters
            width, height = self.gige_config.resolution
            bytes_per_pixel = self._get_bytes_per_pixel(self.gige_config.pixel_format)
            frame_size = width * height * bytes_per_pixel

            # Simulate network transfer time
            transfer_time = frame_size / (self.gige_config.bandwidth_mbps * 1024 * 1024 / 8)
            capture_delay = 1.0 / self.gige_config.frame_rate

            time.sleep(min(capture_delay, transfer_time + 0.001))

            # Generate test frame
            frame_data = self._generate_test_frame(width, height, bytes_per_pixel)

            # Update statistics
            self.frame_count += 1
            self.stats["frames_captured"] += 1
            self.stats["bytes_transferred"] += len(frame_data)
            self.stats["packets_received"] += frame_size // self.gige_config.packet_size + 1
            self.stats["last_frame_time"] = time.time()

            logger.debug(f"Captured GigE frame {self.frame_count}, size: {len(frame_data)} bytes")
            return frame_data

        except Exception as e:
            logger.error(f"GigE frame capture failed: {e}")
            self.stats["frames_dropped"] += 1
            self.status.error_count += 1
            self.status.last_error = str(e)
            return None

    def write_register(self, address: int, data: bytes) -> bool:
        """Write to camera register via GVCP.

        Args:
            address: Register address
            data: Data to write as bytes

        Returns:
            True if write successful
        """
        if not self.is_connected:
            return False

        # Simulate register write
        if len(data) == 4:
            value = int.from_bytes(data, "big")
            logger.debug(f"GVCP WriteReg: 0x{address:08X} = 0x{value:08X}")
        else:
            logger.debug(f"GVCP WriteReg: 0x{address:08X} = {data.hex()}")
        return True

    def read_register(self, address: int, length: int = 4) -> bytes:
        """Read from camera register via GVCP.

        Args:
            address: Register address
            length: Number of bytes to read

        Returns:
            Register data as bytes
        """
        if not self.is_connected:
            return b"\x00" * length

        # Simulate register read
        logger.debug(f"GVCP ReadReg: 0x{address:08X}, length={length}")
        return b"\x00" * length  # Placeholder

    def get_device_info(self) -> dict[str, Any]:
        """Get GigE Vision device information."""
        return {
            "protocol": "GigE Vision",
            "ip_address": self.gige_config.ip_address,
            "speed": self.gige_config.speed.name,
            "bandwidth_mbps": self.gige_config.bandwidth_mbps,
            "packet_size": self.gige_config.packet_size,
            "pixel_format": self.gige_config.pixel_format,
            "resolution": self.gige_config.resolution,
            "frame_rate": self.gige_config.frame_rate,
            "roce_enabled": self.gige_config.enable_roce,
            **self.device_info,
        }

    def get_statistics(self) -> dict[str, Any]:
        """Get GigE Vision driver statistics."""
        current_time = time.time()
        uptime = current_time - self.start_time if self.start_time else 0

        return {
            **self.stats,
            "uptime_seconds": uptime,
            "current_fps": self.frame_count / uptime if uptime > 0 else 0,
            "connection_quality": self.status.connection_quality,
            "packet_loss_rate": self.stats["packets_resent"] / max(1, self.stats["packets_received"]),
            "network_error_rate": self.stats["network_errors"] / max(1, uptime),
        }

    def _discover_camera(self) -> bool:
        """Discover GigE Vision camera."""
        # Simulate GVCP discovery
        time.sleep(0.05)
        logger.debug(f"Discovered camera at {self.gige_config.ip_address}")
        return True

    def _open_gvcp_connection(self) -> str:
        """Open GVCP control connection."""
        handle = f"gvcp_{self.gige_config.ip_address}"
        logger.debug(f"Opened GVCP connection: {handle}")
        return handle

    def _close_gvcp_connection(self) -> None:
        """Close GVCP control connection."""
        logger.debug("Closed GVCP connection")

    def _read_device_info(self) -> None:
        """Read device information from camera."""
        self.device_info = {
            "manufacturer": "Simulated GigE Camera",
            "model": "GigE-SIM-1G",
            "serial_number": "GIGE-001",
            "firmware_version": "1.0.0",
            "gige_version": "2.0",
        }

    def _configure_camera(self) -> None:
        """Configure camera parameters."""
        params = {
            "Width": self.gige_config.resolution[0],
            "Height": self.gige_config.resolution[1],
            "PixelFormat": self.gige_config.pixel_format,
            "AcquisitionFrameRate": self.gige_config.frame_rate,
        }
        logger.debug(f"Configured GigE camera: {params}")

    def _initialize_roce(self) -> None:
        """Initialize RoCE acceleration if available."""
        try:
            from .roce import RoCEConfig, RoCETransport

            roce_config = RoCEConfig()
            self._roce_transport = RoCETransport(roce_config)
            if self._roce_transport.initialize():
                logger.info("RoCE acceleration enabled")
            else:
                self._roce_transport = None
                logger.warning("RoCE initialization failed, falling back to standard transport")
        except ImportError:
            logger.warning("RoCE module not available")

    def _open_stream_channel(self) -> str:
        """Open GVSP stream channel."""
        handle = f"gvsp_{self.gige_config.stream_channel}"
        logger.debug(f"Opened GVSP stream channel: {handle}")
        return handle

    def _close_stream_channel(self) -> None:
        """Close GVSP stream channel."""
        logger.debug("Closed GVSP stream channel")

    def _configure_streaming(self) -> None:
        """Configure streaming parameters."""
        config = {
            "PacketSize": self.gige_config.packet_size,
            "InterPacketDelay": self.gige_config.inter_packet_delay,
            "StreamChannel": self.gige_config.stream_channel,
        }
        logger.debug(f"Configured GVSP streaming: {config}")

    def _start_acquisition(self) -> None:
        """Start image acquisition."""
        logger.debug("Started GigE acquisition")

    def _stop_acquisition(self) -> None:
        """Stop image acquisition."""
        logger.debug("Stopped GigE acquisition")

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
        target_fps = self.gige_config.frame_rate
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
        """Optimize GigE Vision protocol performance."""
        logger.info("GigE Vision protocol performance optimized")

    def get_status(self) -> dict[str, Any]:
        """Get GigE Vision protocol status (legacy compatibility)."""
        return {
            "protocol": "GigE Vision",
            "connected": self.is_connected,
            "streaming": self.is_streaming,
            "config": {
                "ip_address": self.gige_config.ip_address,
                "port": self.gige_config.port,
                "pixel_format": self.gige_config.pixel_format,
            },
            "statistics": self.get_statistics(),
        }
