"""
Base Protocol Classes for Advanced Image Sensor Interface

This module provides abstract base classes for protocol implementations,
defining the common interface that all protocol drivers must implement.

Classes:
    ProtocolBase: Abstract base class for all protocol implementations.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional


class ProtocolBase(ABC):
    """
    Abstract base class for all protocol implementations.

    This class defines the common interface that all protocol drivers
    must implement, ensuring consistency across different protocols.
    """

    @abstractmethod
    def __init__(self, config: Any) -> None:
        """
        Initialize the protocol driver with configuration.

        Args:
            config: Protocol-specific configuration object.
        """
        pass

    @abstractmethod
    def connect(self) -> bool:
        """
        Establish connection to the device.

        Returns:
            bool: True if connection successful, False otherwise.
        """
        pass

    @abstractmethod
    def disconnect(self) -> bool:
        """
        Disconnect from the device.

        Returns:
            bool: True if disconnection successful, False otherwise.
        """
        pass

    @abstractmethod
    def send_data(self, data: bytes) -> bool:
        """
        Send data through the protocol interface.

        Args:
            data (bytes): Data to be sent.

        Returns:
            bool: True if data was sent successfully, False otherwise.
        """
        pass

    @abstractmethod
    def receive_data(self, size: int) -> Optional[bytes]:
        """
        Receive data from the protocol interface.

        Args:
            size (int): Number of bytes to receive.

        Returns:
            Optional[bytes]: Received data, or None if no data available.
        """
        pass

    @abstractmethod
    def get_status(self) -> dict[str, Any]:
        """
        Get the current status of the protocol driver.

        Returns:
            Dict[str, Any]: Dictionary containing status information.
        """
        pass

    @abstractmethod
    def optimize_performance(self) -> None:
        """
        Optimize protocol driver performance.
        """
        pass
