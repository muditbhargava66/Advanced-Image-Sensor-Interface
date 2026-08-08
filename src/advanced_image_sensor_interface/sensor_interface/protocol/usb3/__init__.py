"""
USB3 Vision Protocol Package

This package provides USB3 Vision protocol implementations for
USB 3.0-based camera interfaces.

Modules:
    driver: Main USB3 Vision driver with streaming support.
    discovery: Device enumeration and hot-plug detection.
    streaming: Buffer management and stream control.

Example:
    >>> from advanced_image_sensor_interface.sensor_interface.protocol.usb3 import (
    ...     USB3VisionDriver,
    ...     USB3VisionConfig,
    ...     USB3DeviceDiscovery,
    ...     USB3StreamingManager,
    ... )
"""

from .discovery import (
    DeviceDescriptor,
    DeviceFilter,
    DeviceInfo,
    HotPlugEvent,
    HotPlugNotification,
    USB3DeviceDiscovery,
    USB3DeviceFactory,
    USBSpeed,
)
from .driver import USB3VisionConfig, USB3VisionDriver
from .streaming import (
    BufferPool,
    BufferState,
    FrameInfo,
    StreamBuffer,
    StreamConfig,
    StreamState,
    StreamStatistics,
    USB3StreamingManager,
)

__all__ = [
    # Main driver
    "USB3VisionDriver",
    "USB3VisionConfig",
    # Discovery
    "USB3DeviceDiscovery",
    "USB3DeviceFactory",
    "DeviceInfo",
    "DeviceDescriptor",
    "DeviceFilter",
    "USBSpeed",
    "HotPlugEvent",
    "HotPlugNotification",
    # Streaming
    "USB3StreamingManager",
    "StreamConfig",
    "StreamStatistics",
    "StreamState",
    "BufferPool",
    "BufferState",
    "StreamBuffer",
    "FrameInfo",
]
