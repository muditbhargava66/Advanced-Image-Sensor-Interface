"""
Protocol implementations for camera interfaces.

This module provides support for multiple camera interface protocols:
- MIPI CSI-2: Mobile Industry Processor Interface (up to 4.5 Gbps/lane)
- CoaXPress: High-speed coaxial cable interface (up to 12.5 Gbps/lane)
- GigE Vision: Ethernet-based camera interface (1-10 Gbps with RoCE)
- USB3 Vision: USB 3.0-based camera interface (up to 10 Gbps)

Example:
    >>> from advanced_image_sensor_interface.sensor_interface.protocol import (
    ...     ProtocolBase,
    ...     ProtocolError,
    ...     ProtocolCapabilities,
    ... )
    >>> from advanced_image_sensor_interface.sensor_interface.protocol.mipi import MIPIProtocolDriver
    >>> from advanced_image_sensor_interface.sensor_interface.protocol.coaxpress import CoaXPressDriver
    >>> from advanced_image_sensor_interface.sensor_interface.protocol.gige import GigEProtocolDriver
    >>> from advanced_image_sensor_interface.sensor_interface.protocol.usb3 import USB3VisionDriver
"""

from .base import (
    ConfigurationError,
    ConnectionError,
    DataTransferError,
    ProtocolBase,
    ProtocolCapabilities,
    ProtocolError,
    ProtocolStatus,
    StreamingProtocolBase,
)

__all__ = [
    # Base classes
    "ProtocolBase",
    "StreamingProtocolBase",
    "ProtocolError",
    "ConnectionError",
    "DataTransferError",
    "ConfigurationError",
    "ProtocolCapabilities",
    "ProtocolStatus",
]
