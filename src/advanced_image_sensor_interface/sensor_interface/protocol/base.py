"""
Base classes and interfaces for camera protocols.

This module defines the abstract base classes that all protocol implementations
must inherit from, ensuring a consistent interface across different protocols.
"""

import logging
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
    supported_resolutions: list[tuple]


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

    def get_status(self) -> ProtocolStatus:
        """Get current protocol status."""
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

    def get_performance_metrics(self) -> dict[str, Any]:
        """Get performance metrics for the protocol."""
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
