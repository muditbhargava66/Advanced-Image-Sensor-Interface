"""
Base classes and interfaces for camera protocols.

This module defines the abstract base classes that all protocol implementations
must inherit from, ensuring a consistent interface across different protocols.
"""

import logging
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional

logger = logging.getLogger(__name__)


class ProtocolError(Exception):
    """Base exception for protocol-related errors."""

    pass


class ConnectionError(ProtocolError):
    """Exception raised when connection to device fails."""

    pass


class DataTransferError(ProtocolError):
    """Exception raised when data transfer fails."""

    pass


class ConfigurationError(ProtocolError):
    """Exception raised when protocol configuration is invalid."""

    pass


@dataclass
class ProtocolCapabilities:
    """Capabilities supported by a protocol implementation."""

    max_bandwidth_gbps: float
    max_distance_m: float
    power_over_cable: bool
    hot_pluggable: bool
    multi_camera_support: bool
    hardware_trigger_support: bool
    software_trigger_support: bool
    supported_pixel_formats: list[str]
    supported_resolutions: list[tuple[int, int]]


@dataclass
class ProtocolStatus:
    """Current status of a protocol connection."""

    is_connected: bool
    connection_quality: float  # 0.0 to 1.0
    data_rate_mbps: float
    error_count: int
    last_error: Optional[str]
    uptime_seconds: float
    bytes_transmitted: int
    bytes_received: int


class ProtocolBase(ABC):
    """
    Abstract base class for all camera protocol implementations.

    This class defines the common interface that all protocol drivers
    must implement, ensuring consistency across different protocols.
    """

    def __init__(self, config: dict[str, Any]):
        """Initialize protocol with configuration."""
        self.config = config
        self.is_connected = False
        self._connected_at: Optional[float] = None
        self._connected_seconds = 0.0
        self.status = ProtocolStatus(
            is_connected=False,
            connection_quality=0.0,
            data_rate_mbps=0.0,
            error_count=0,
            last_error=None,
            uptime_seconds=0.0,
            bytes_transmitted=0,
            bytes_received=0,
        )
        self.capabilities = self._get_capabilities()
        logger.info(f"Initialized {self.__class__.__name__} protocol")

    @abstractmethod
    def _get_capabilities(self) -> ProtocolCapabilities:
        """Get the capabilities of this protocol implementation."""
        pass

    @abstractmethod
    def connect(self) -> bool:
        """
        Establish connection to the camera device.

        Returns:
            bool: True if connection successful, False otherwise

        Raises:
            ConnectionError: If connection fails due to hardware issues
            ConfigurationError: If configuration is invalid
        """
        pass

    @abstractmethod
    def disconnect(self) -> bool:
        """
        Disconnect from the camera device.

        Returns:
            bool: True if disconnection successful, False otherwise
        """
        pass

    @abstractmethod
    def send_data(self, data: bytes) -> bool:
        """
        Send data to the camera device.

        Args:
            data: Bytes to send to the device

        Returns:
            bool: True if data sent successfully, False otherwise

        Raises:
            DataTransferError: If data transfer fails
            ConnectionError: If not connected to device
        """
        pass

    @abstractmethod
    def receive_data(self, size: int) -> Optional[bytes]:
        """
        Receive data from the camera device.

        Args:
            size: Number of bytes to receive

        Returns:
            Optional[bytes]: Received data, or None if error occurred

        Raises:
            DataTransferError: If data transfer fails
            ConnectionError: If not connected to device
        """
        pass

    def get_status(self) -> ProtocolStatus | dict[str, Any]:
        """Get current protocol status."""
        self._refresh_status()
        return self.status

    def get_capabilities(self) -> ProtocolCapabilities:
        """Get protocol capabilities."""
        return self.capabilities

    def validate_configuration(self) -> bool:
        """
        Validate the current configuration.

        Returns:
            bool: True if configuration is valid

        Raises:
            ConfigurationError: If configuration is invalid
        """
        # Basic validation - subclasses should override for specific validation
        if not isinstance(self.config, dict):
            raise ConfigurationError("Configuration must be a dictionary")
        return True

    def reset(self) -> bool:
        """
        Reset the protocol connection.

        Returns:
            bool: True if reset successful
        """
        try:
            if self.is_connected:
                self.disconnect()
            return self.connect()
        except Exception as e:
            logger.error(f"Protocol reset failed: {e}")
            return False

    def _mark_connected(self) -> None:
        """Begin measuring connected time; link capacity is not measured throughput."""
        if self._connected_at is None:
            self._connected_at = time.monotonic()
        self.is_connected = self.status.is_connected = True
        self._refresh_status()

    def _mark_disconnected(self) -> None:
        self._refresh_status()
        self._connected_seconds = self.status.uptime_seconds
        self._connected_at = None
        self.is_connected = self.status.is_connected = False
        self.status.data_rate_mbps = 0.0

    def _refresh_status(self) -> None:
        elapsed = 0.0 if self._connected_at is None else max(0.0, time.monotonic() - self._connected_at)
        self.status.uptime_seconds = self._connected_seconds + elapsed
        self.status.is_connected = self.is_connected
        total_bytes = self.status.bytes_received + self.status.bytes_transmitted
        self.status.data_rate_mbps = (
            total_bytes * 8 / self.status.uptime_seconds / 1e6 if self.is_connected and self.status.uptime_seconds > 0 else 0.0
        )

    def get_performance_metrics(self) -> dict[str, Any]:
        """Measured traffic over accumulated connected time (not link capacity)."""
        self._refresh_status()
        return {
            "data_rate_mbps": self.status.data_rate_mbps,
            "connection_quality": self.status.connection_quality,
            "error_rate": self.status.error_count / max(1, self.status.uptime_seconds),
            "bytes_transmitted": self.status.bytes_transmitted,
            "bytes_received": self.status.bytes_received,
            "uptime_seconds": self.status.uptime_seconds,
        }

    def __enter__(self):
        """Context manager entry."""
        if not self.connect():
            raise ConnectionError("Failed to establish connection")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()


class StreamingProtocolBase(ProtocolBase):
    """
    Base class for streaming protocols that support continuous data flow.
    """

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self.is_streaming = False
        self._captured_frames = 0

    def _record_capture(self, size: int) -> None:
        self._captured_frames += 1
        self.status.bytes_received += size
        self._refresh_status()

    def _measured_frame_rate(self) -> float:
        self._refresh_status()
        return self._captured_frames / self.status.uptime_seconds if self.status.uptime_seconds > 0 else 0.0

    def get_performance_metrics(self) -> dict[str, Any]:
        return {**super().get_performance_metrics(), **self.get_streaming_status(), "frames_captured": self._captured_frames}

    @abstractmethod
    def start_streaming(self) -> bool:
        """
        Start continuous data streaming.

        Returns:
            bool: True if streaming started successfully
        """
        pass

    @abstractmethod
    def stop_streaming(self) -> bool:
        """
        Stop continuous data streaming.

        Returns:
            bool: True if streaming stopped successfully
        """
        pass

    @abstractmethod
    def capture_frame(self) -> Optional[bytes]:
        """
        Capture a single frame from the stream.

        Returns:
            Optional[bytes]: Frame data, or None if capture failed
        """
        pass

    def get_streaming_status(self) -> dict[str, Any]:
        """Get streaming-specific status information."""
        return {
            "is_streaming": self.is_streaming,
            "frame_rate": self._get_current_frame_rate(),
            "dropped_frames": self._get_dropped_frame_count(),
            "buffer_utilization": self._get_buffer_utilization(),
        }

    def _get_current_frame_rate(self) -> float:
        """Get current frame rate (to be implemented by subclasses)."""
        return 0.0

    def _get_dropped_frame_count(self) -> int:
        """Get number of dropped frames (to be implemented by subclasses)."""
        return 0

    def _get_buffer_utilization(self) -> float:
        """Get buffer utilization percentage (to be implemented by subclasses)."""
        return 0.0

    # -----------------------------------------------------------------
    # ARCH-2: Shared utility methods extracted from per-driver code.
    # Subclass drivers may delegate to these instead of duplicating
    # the logic.
    # -----------------------------------------------------------------

    @staticmethod
    def _pixel_storage(pixel_format: str) -> tuple[int, int, int]:
        """Return (significant bits, samples per pixel, bytes per sample).

        Storage is unpacked little-endian: <=8 bits use uint8, 10-16 use
        uint16, and 20 use uint32. RGB888 aliases RGB8. YUV422 uses Y/U,
        Y/V sample pairs; packed RAW suffixes and RGB565/666 are unsupported.
        """
        aliases = {"RGB888": "RGB8", "YUV422": "YUV422_8", "YUV444": "YUV444_8"}
        fmt = aliases.get(pixel_format, pixel_format)
        match = re.fullmatch(r"(RAW|Mono|Bayer(?:RG|GR|GB|BG)|RGB|BGR)(6|7|8|10|12|14|16|20)", fmt)
        if match:
            family, depth = match.groups()
            bits = int(depth)
            if bits < 8 and family != "RAW":
                raise ValueError(f"Unsupported pixel format: {pixel_format}")
            channels = 3 if family in ("RGB", "BGR") else 1
        else:
            match = re.fullmatch(r"YUV(422|444)_(8|10)", fmt)
            if not match:
                raise ValueError(f"Unsupported or packed pixel format: {pixel_format}")
            sampling, depth = match.groups()
            bits, channels = int(depth), 2 if sampling == "422" else 3
        return bits, channels, 1 if bits <= 8 else 2 if bits <= 16 else 4

    @staticmethod
    def _get_bytes_per_pixel_common(pixel_format: str) -> int:
        """Get exact unpacked storage size, rejecting unknown/packed formats."""
        _, channels, sample_bytes = StreamingProtocolBase._pixel_storage(pixel_format)
        return channels * sample_bytes

    @staticmethod
    def _generate_test_frame_vectorized(
        width: int, height: int, bytes_per_pixel: int, frame_count: int = 0, pixel_format: Optional[str] = None
    ) -> bytes:
        """Generate owned frame bytes with declared range and little-endian storage.

        The legacy size-only API remains available. Format-aware callers must
        provide a matching size; no wire-level bit packing is performed.
        """
        import numpy as np

        if width <= 0 or height <= 0 or bytes_per_pixel <= 0:
            raise ValueError("Frame dimensions and storage size must be positive")
        if pixel_format is None:
            bits, channels, sample_bytes = (16, 1, 2) if bytes_per_pixel == 2 else (8, bytes_per_pixel, 1)
        else:
            bits, channels, sample_bytes = StreamingProtocolBase._pixel_storage(pixel_format)
            if channels * sample_bytes != bytes_per_pixel:
                raise ValueError("Pixel format does not match bytes_per_pixel")
            if pixel_format.startswith("YUV422") and width % 2:
                raise ValueError("YUV422 requires an even width")
        gradient = np.arange(height, dtype=np.uint64)[:, None] + np.arange(width, dtype=np.uint64)[None, :]
        modulus = 1 << bits
        samples = [((gradient + (frame_count + c * 50) % modulus) % modulus) for c in range(channels)]
        frame = np.stack(samples, axis=-1).astype(np.dtype(f"<u{sample_bytes}"))
        return frame.tobytes()
