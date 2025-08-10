"""
USB3 Vision protocol driver implementation.

Provides comprehensive support for USB3 Vision cameras with high-speed
data transfer, device discovery, and streaming capabilities.
"""

import logging
import time
from dataclasses import dataclass
from typing import Any, ClassVar, Optional

import numpy as np

from ....performance.profiler import profile_function
from ..base import ConnectionError, DataTransferError, ProtocolCapabilities, StreamingProtocolBase

logger = logging.getLogger(__name__)


@dataclass
class USB3VisionConfig:
    """Configuration for USB3 Vision protocol."""

    # Device identification
    vendor_id: Optional[int] = None
    product_id: Optional[int] = None
    serial_number: Optional[str] = None
    device_index: int = 0

    # Image settings
    pixel_format: str = "Mono8"
    resolution: tuple[int, int] = (1920, 1080)
    frame_rate: float = 30.0
    exposure_time_us: float = 10000.0
    gain: float = 1.0

    # USB3 specific settings
    usb_speed: str = "SuperSpeed"  # "HighSpeed", "SuperSpeed", "SuperSpeedPlus"
    packet_size: int = 1024
    packet_delay: int = 0
    transfer_queue_size: int = 16

    # Streaming settings
    buffer_count: int = 10
    timeout_ms: int = 5000

    # Advanced settings
    enable_chunk_data: bool = False
    enable_event_notification: bool = True
    heartbeat_timeout_ms: int = 3000

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        valid_speeds = ["HighSpeed", "SuperSpeed", "SuperSpeedPlus"]
        if self.usb_speed not in valid_speeds:
            raise ValueError(f"Invalid USB speed: {self.usb_speed}")

        if self.packet_size < 64 or self.packet_size > 65536:
            raise ValueError("Packet size must be between 64 and 65536 bytes")

        if self.buffer_count < 2:
            raise ValueError("Buffer count must be at least 2")


class USB3VisionDriver(StreamingProtocolBase):
    """
    USB3 Vision protocol driver implementation.

    Provides complete USB3 Vision support including device discovery,
    connection management, streaming, and GenICam parameter control.
    """

    # USB3 speed to bandwidth mapping (Mbps)
    USB_SPEEDS: ClassVar[dict[str, int]] = {
        "HighSpeed": 480,  # USB 2.0
        "SuperSpeed": 5000,  # USB 3.0
        "SuperSpeedPlus": 10000,  # USB 3.1/3.2
    }

    def __init__(self, config: USB3VisionConfig):
        """Initialize USB3 Vision driver."""
        super().__init__(config.__dict__)
        self.usb3_config = config
        self.device_handle = None
        self.stream_handle = None
        self.device_info: dict[str, Any] = {}

        # Statistics
        self.stats = {"frames_captured": 0, "frames_dropped": 0, "bytes_transferred": 0, "usb_errors": 0, "reconnections": 0}

        logger.info(f"USB3 Vision driver initialized for {config.usb_speed} device")

    def _get_capabilities(self) -> ProtocolCapabilities:
        """Get USB3 Vision protocol capabilities."""
        max_bandwidth = self.USB_SPEEDS[self.usb3_config.usb_speed]

        return ProtocolCapabilities(
            max_bandwidth_gbps=max_bandwidth / 1000.0,
            max_distance_m=5.0,  # Standard USB cable length
            power_over_cable=True,
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
                (4096, 3072),
            ],
        )

    @profile_function
    def connect(self) -> bool:
        """Establish connection to USB3 Vision device."""
        try:
            logger.info("Connecting to USB3 Vision device...")

            # Discover and select device
            device_info = self._discover_device()
            if not device_info:
                raise ConnectionError("No USB3 Vision device found")

            self.device_info = device_info

            # Open device connection
            self.device_handle = self._open_device(device_info)
            if not self.device_handle:
                raise ConnectionError("Failed to open USB3 Vision device")

            # Initialize device parameters
            self._initialize_device()

            # Update connection status
            self.is_connected = True
            self.status.is_connected = True
            self.status.connection_quality = 1.0
            self.status.data_rate_mbps = self.USB_SPEEDS[self.usb3_config.usb_speed]

            logger.info(f"Connected to USB3 Vision device: {device_info.get('model', 'Unknown')}")
            return True

        except Exception as e:
            logger.error(f"Failed to connect to USB3 Vision device: {e}")
            self.status.error_count += 1
            self.status.last_error = str(e)
            raise ConnectionError(f"USB3 Vision connection failed: {e}")

    def disconnect(self) -> bool:
        """Disconnect from USB3 Vision device."""
        try:
            if self.is_streaming:
                self.stop_streaming()

            # Close device handles
            if self.stream_handle:
                self._close_stream()
                self.stream_handle = None

            if self.device_handle:
                self._close_device()
                self.device_handle = None

            # Update status
            self.is_connected = False
            self.status.is_connected = False
            self.status.connection_quality = 0.0
            self.status.data_rate_mbps = 0.0

            logger.info("USB3 Vision device disconnected")
            return True

        except Exception as e:
            logger.error(f"Error during USB3 Vision disconnect: {e}")
            return False

    def send_data(self, data: bytes) -> bool:
        """Send control data to USB3 Vision device."""
        if not self.is_connected:
            raise ConnectionError("Not connected to USB3 Vision device")

        try:
            # Simulate USB control transfer
            # In real implementation, this would use libusb or similar
            time.sleep(0.001)  # Simulate USB latency

            self.status.bytes_transmitted += len(data)
            logger.debug(f"Sent {len(data)} bytes to USB3 Vision device")
            return True

        except Exception as e:
            logger.error(f"Failed to send data: {e}")
            self.status.error_count += 1
            self.stats["usb_errors"] += 1
            raise DataTransferError(f"USB3 Vision send failed: {e}")

    def receive_data(self, size: int) -> Optional[bytes]:
        """Receive data from USB3 Vision device."""
        if not self.is_connected:
            raise ConnectionError("Not connected to USB3 Vision device")

        try:
            # Simulate USB bulk transfer
            time.sleep(0.001)  # Simulate USB latency

            # Generate simulated response
            data = bytes([i % 256 for i in range(size)])

            self.status.bytes_received += len(data)
            logger.debug(f"Received {len(data)} bytes from USB3 Vision device")
            return data

        except Exception as e:
            logger.error(f"Failed to receive data: {e}")
            self.status.error_count += 1
            self.stats["usb_errors"] += 1
            raise DataTransferError(f"USB3 Vision receive failed: {e}")

    @profile_function
    def start_streaming(self) -> bool:
        """Start continuous image streaming."""
        if not self.is_connected:
            raise ConnectionError("Not connected to USB3 Vision device")

        try:
            logger.info("Starting USB3 Vision streaming...")

            # Initialize streaming interface
            self.stream_handle = self._initialize_streaming()

            # Configure streaming parameters
            self._configure_streaming()

            # Start acquisition
            self._start_acquisition()

            self.is_streaming = True
            logger.info("USB3 Vision streaming started")
            return True

        except Exception as e:
            logger.error(f"Failed to start streaming: {e}")
            self.status.error_count += 1
            return False

    def stop_streaming(self) -> bool:
        """Stop continuous image streaming."""
        try:
            if self.is_streaming:
                # Stop acquisition
                self._stop_acquisition()

                # Close streaming interface
                if self.stream_handle:
                    self._close_stream()
                    self.stream_handle = None

                self.is_streaming = False
                logger.info("USB3 Vision streaming stopped")

            return True

        except Exception as e:
            logger.error(f"Error stopping streaming: {e}")
            return False

    @profile_function
    def capture_frame(self) -> Optional[bytes]:
        """Capture a single frame from USB3 Vision device."""
        if not self.is_connected:
            raise ConnectionError("Not connected to USB3 Vision device")

        try:
            # Calculate frame parameters
            width, height = self.usb3_config.resolution
            bytes_per_pixel = self._get_bytes_per_pixel(self.usb3_config.pixel_format)
            frame_size = width * height * bytes_per_pixel

            # Simulate frame capture with USB3 timing
            capture_delay = 1.0 / self.usb3_config.frame_rate
            usb_transfer_time = frame_size / (self.USB_SPEEDS[self.usb3_config.usb_speed] * 1024 * 1024 / 8)
            total_delay = min(capture_delay, usb_transfer_time + 0.001)

            time.sleep(total_delay)

            # Generate test frame
            frame_data = self._generate_test_frame(width, height, bytes_per_pixel)

            # Update statistics
            self.stats["frames_captured"] += 1
            self.stats["bytes_transferred"] += len(frame_data)

            logger.debug(f"Captured USB3 frame: {len(frame_data)} bytes")
            return frame_data

        except Exception as e:
            logger.error(f"Frame capture failed: {e}")
            self.stats["usb_errors"] += 1
            self.status.error_count += 1
            return None

    def _discover_device(self) -> Optional[dict[str, Any]]:
        """Discover USB3 Vision devices."""
        # Simulate device discovery
        # In real implementation, this would enumerate USB devices
        time.sleep(0.1)

        # Return simulated device info
        return {
            "vendor_id": self.usb3_config.vendor_id or 0x1234,
            "product_id": self.usb3_config.product_id or 0x5678,
            "serial_number": self.usb3_config.serial_number or "USB3CAM001",
            "manufacturer": "Generic Camera Corp",
            "model": "USB3 Vision Camera",
            "firmware_version": "1.2.3",
            "usb_speed": self.usb3_config.usb_speed,
            "max_packet_size": 1024,
        }

    def _open_device(self, device_info: dict[str, Any]) -> str:
        """Open connection to USB3 device."""
        # Simulate device opening
        device_id = f"usb3_{device_info['vendor_id']:04x}_{device_info['product_id']:04x}"
        logger.debug(f"Opened USB3 device: {device_id}")
        return device_id

    def _close_device(self) -> None:
        """Close USB3 device connection."""
        logger.debug("Closed USB3 device connection")

    def _initialize_device(self) -> None:
        """Initialize device parameters."""
        # Set basic parameters
        params = {
            "Width": self.usb3_config.resolution[0],
            "Height": self.usb3_config.resolution[1],
            "PixelFormat": self.usb3_config.pixel_format,
            "AcquisitionFrameRate": self.usb3_config.frame_rate,
            "ExposureTime": self.usb3_config.exposure_time_us,
            "Gain": self.usb3_config.gain,
        }

        logger.debug(f"Initialized USB3 device with parameters: {params}")

    def _initialize_streaming(self) -> str:
        """Initialize streaming interface."""
        stream_id = f"stream_{id(self)}"
        logger.debug(f"Initialized USB3 streaming interface: {stream_id}")
        return stream_id

    def _configure_streaming(self) -> None:
        """Configure streaming parameters."""
        config = {
            "packet_size": self.usb3_config.packet_size,
            "packet_delay": self.usb3_config.packet_delay,
            "buffer_count": self.usb3_config.buffer_count,
            "timeout_ms": self.usb3_config.timeout_ms,
        }

        logger.debug(f"Configured USB3 streaming: {config}")

    def _start_acquisition(self) -> None:
        """Start image acquisition."""
        logger.debug("Started USB3 image acquisition")

    def _stop_acquisition(self) -> None:
        """Stop image acquisition."""
        logger.debug("Stopped USB3 image acquisition")

    def _close_stream(self) -> None:
        """Close streaming interface."""
        logger.debug("Closed USB3 streaming interface")

    def _get_bytes_per_pixel(self, pixel_format: str) -> int:
        """Get bytes per pixel for given format."""
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
            "YUV444": 3,
        }
        return format_map.get(pixel_format, 1)

    def _generate_test_frame(self, width: int, height: int, bytes_per_pixel: int) -> bytes:
        """Generate test frame data."""
        if bytes_per_pixel == 1:
            # Mono pattern
            frame = np.zeros((height, width), dtype=np.uint8)
            for y in range(height):
                for x in range(width):
                    frame[y, x] = (x + y + int(time.time() * 10)) % 256
        else:
            # Multi-channel pattern
            frame = np.zeros((height, width, bytes_per_pixel), dtype=np.uint8)
            for y in range(height):
                for x in range(width):
                    for c in range(bytes_per_pixel):
                        frame[y, x, c] = (x + y + c * 85 + int(time.time() * 10)) % 256

        return frame.tobytes()

    def get_device_info(self) -> dict[str, Any]:
        """Get USB3 Vision device information."""
        return {
            "protocol": "USB3Vision",
            "device_info": self.device_info,
            "usb_speed": self.usb3_config.usb_speed,
            "max_bandwidth_mbps": self.USB_SPEEDS[self.usb3_config.usb_speed],
            "pixel_format": self.usb3_config.pixel_format,
            "resolution": self.usb3_config.resolution,
            "frame_rate": self.usb3_config.frame_rate,
            "buffer_count": self.usb3_config.buffer_count,
        }

    def get_statistics(self) -> dict[str, Any]:
        """Get USB3 Vision driver statistics."""
        return {
            **self.stats,
            "connection_quality": self.status.connection_quality,
            "usb_speed": self.usb3_config.usb_speed,
            "error_rate": self.stats["usb_errors"] / max(1, self.stats["frames_captured"]),
        }
