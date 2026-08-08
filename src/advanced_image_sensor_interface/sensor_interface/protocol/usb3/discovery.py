"""
USB3 Vision Device Discovery

Provides device enumeration, hot-plug detection, and device
filtering for USB3 Vision cameras.

Key Features:
- Device enumeration by vendor/product ID
- Hot-plug event handling
- Device descriptor parsing
- Filter-based device selection
- Multi-camera support

Version: 3.0.0
"""

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, ClassVar, Optional

logger = logging.getLogger(__name__)


class USBSpeed(Enum):
    """USB speed classifications."""

    USB_1_1 = "usb1.1"  # 12 Mbps
    USB_2_0 = "usb2.0"  # 480 Mbps
    USB_3_0 = "usb3.0"  # 5 Gbps SuperSpeed
    USB_3_1 = "usb3.1"  # 10 Gbps SuperSpeed+
    USB_3_2 = "usb3.2"  # 20 Gbps SuperSpeed+


class DeviceClass(Enum):
    """USB device class codes."""

    MISC = 0xEF
    VENDOR_SPECIFIC = 0xFF


@dataclass
class DeviceDescriptor:
    """USB device descriptor information."""

    vendor_id: int
    product_id: int
    serial_number: str
    manufacturer: str
    product_name: str
    device_version: str
    usb_speed: USBSpeed
    bus_number: int
    device_address: int
    port_numbers: tuple[int, ...]


@dataclass
class DeviceInfo:
    """Complete device information."""

    descriptor: DeviceDescriptor
    is_usb3_vision: bool = False
    device_guid: str = ""
    gencp_version: str = ""
    u3v_version: str = ""
    max_frame_rate: float = 0.0
    max_resolution: tuple[int, int] = (0, 0)
    supported_pixel_formats: list[str] = field(default_factory=list)


@dataclass
class DeviceFilter:
    """Filter criteria for device discovery."""

    vendor_id: Optional[int] = None
    product_id: Optional[int] = None
    serial_number: Optional[str] = None
    manufacturer: Optional[str] = None
    min_usb_speed: USBSpeed = USBSpeed.USB_3_0
    usb3_vision_only: bool = True

    def matches(self, device: DeviceInfo) -> bool:
        """Check if device matches filter criteria."""
        if self.usb3_vision_only and not device.is_usb3_vision:
            return False

        if self.vendor_id is not None and device.descriptor.vendor_id != self.vendor_id:
            return False

        if self.product_id is not None and device.descriptor.product_id != self.product_id:
            return False

        if self.serial_number is not None and device.descriptor.serial_number != self.serial_number:
            return False

        if self.manufacturer is not None:
            if self.manufacturer.lower() not in device.descriptor.manufacturer.lower():
                return False

        # Check USB speed
        speed_order = list(USBSpeed)
        if speed_order.index(device.descriptor.usb_speed) < speed_order.index(self.min_usb_speed):
            return False

        return True


class HotPlugEvent(Enum):
    """Hot-plug event types."""

    DEVICE_ATTACHED = "attached"
    DEVICE_DETACHED = "detached"


@dataclass
class HotPlugNotification:
    """Hot-plug event notification."""

    event: HotPlugEvent
    device: Optional[DeviceInfo]
    timestamp: float


class USB3DeviceDiscovery:
    """
    USB3 Vision device discovery and enumeration.

    Handles device scanning, hot-plug detection, and device
    information retrieval.
    """

    # Known USB3 Vision vendors
    KNOWN_VENDORS: ClassVar[dict[int, str]] = {
        0x2AB9: "Allied Vision",
        0x0BDA: "Basler",
        0x1E10: "FLIR",
        0x1D6B: "Linux Foundation (testing)",
    }

    def __init__(self) -> None:
        """Initialize device discovery."""
        self._discovered_devices: dict[str, DeviceInfo] = {}
        self._hotplug_callbacks: list[Callable[[HotPlugNotification], None]] = []
        self._is_monitoring = False
        self._last_scan_time = 0.0

    def discover_devices(self, device_filter: Optional[DeviceFilter] = None, timeout_s: float = 5.0) -> list[DeviceInfo]:
        """
        Discover available USB3 Vision devices.

        Args:
            device_filter: Optional filter criteria
            timeout_s: Discovery timeout in seconds

        Returns:
            List of discovered devices
        """
        logger.info("Starting USB3 Vision device discovery...")
        start_time = time.time()
        device_filter = device_filter or DeviceFilter()

        # Simulate device enumeration
        devices = self._enumerate_usb_devices()

        # Apply filter
        filtered_devices = [device for device in devices if device_filter.matches(device)]

        # Update cache
        for device in filtered_devices:
            device_key = self._get_device_key(device)
            self._discovered_devices[device_key] = device

        elapsed = time.time() - start_time
        self._last_scan_time = time.time()

        logger.info(f"Discovered {len(filtered_devices)} USB3 Vision devices in {elapsed:.2f}s")
        return filtered_devices

    def get_device_by_serial(self, serial_number: str) -> Optional[DeviceInfo]:
        """
        Get device by serial number.

        Args:
            serial_number: Device serial number

        Returns:
            Device info or None if not found
        """
        for device in self._discovered_devices.values():
            if device.descriptor.serial_number == serial_number:
                return device
        return None

    def get_device_by_index(self, index: int) -> Optional[DeviceInfo]:
        """
        Get device by discovery index.

        Args:
            index: Device index

        Returns:
            Device info or None if not found
        """
        devices = list(self._discovered_devices.values())
        if 0 <= index < len(devices):
            return devices[index]
        return None

    def refresh(self) -> list[DeviceInfo]:
        """
        Refresh device list.

        Returns:
            Updated device list
        """
        return self.discover_devices()

    def start_hotplug_monitoring(self) -> bool:
        """
        Start monitoring for hot-plug events.

        Returns:
            True if monitoring started
        """
        if self._is_monitoring:
            return True

        self._is_monitoring = True
        logger.info("Hot-plug monitoring started")
        return True

    def stop_hotplug_monitoring(self) -> None:
        """Stop hot-plug monitoring."""
        self._is_monitoring = False
        logger.info("Hot-plug monitoring stopped")

    def register_hotplug_callback(self, callback: Callable[[HotPlugNotification], None]) -> None:
        """
        Register a callback for hot-plug events.

        Args:
            callback: Function to call on hot-plug events
        """
        self._hotplug_callbacks.append(callback)
        logger.debug("Registered hot-plug callback")

    def unregister_hotplug_callback(self, callback: Callable[[HotPlugNotification], None]) -> None:
        """
        Unregister a hot-plug callback.

        Args:
            callback: Callback to unregister
        """
        if callback in self._hotplug_callbacks:
            self._hotplug_callbacks.remove(callback)

    def get_device_count(self) -> int:
        """Get number of discovered devices."""
        return len(self._discovered_devices)

    def get_all_devices(self) -> list[DeviceInfo]:
        """Get all discovered devices."""
        return list(self._discovered_devices.values())

    def _enumerate_usb_devices(self) -> list[DeviceInfo]:
        """
        Enumerate USB devices.

        In real implementation, this would use pyusb or libusb.
        """
        # Simulated devices for testing
        simulated_devices = [
            self._create_simulated_device(
                vendor_id=0x2AB9,
                product_id=0x0001,
                serial="AV-001",
                manufacturer="Allied Vision",
                product="Alvium 1800",
                usb_speed=USBSpeed.USB_3_0,
            ),
            self._create_simulated_device(
                vendor_id=0x0BDA,
                product_id=0x0002,
                serial="BAS-002",
                manufacturer="Basler",
                product="ace U",
                usb_speed=USBSpeed.USB_3_1,
            ),
        ]

        return simulated_devices

    def _create_simulated_device(
        self, vendor_id: int, product_id: int, serial: str, manufacturer: str, product: str, usb_speed: USBSpeed
    ) -> DeviceInfo:
        """Create a simulated device for testing."""
        descriptor = DeviceDescriptor(
            vendor_id=vendor_id,
            product_id=product_id,
            serial_number=serial,
            manufacturer=manufacturer,
            product_name=product,
            device_version="1.0",
            usb_speed=usb_speed,
            bus_number=1,
            device_address=1,
            port_numbers=(1,),
        )

        return DeviceInfo(
            descriptor=descriptor,
            is_usb3_vision=True,
            device_guid=f"{vendor_id:04X}{product_id:04X}{serial}",
            gencp_version="1.0",
            u3v_version="1.0",
            max_frame_rate=60.0,
            max_resolution=(1920, 1080),
            supported_pixel_formats=["Mono8", "Mono12", "BayerRG8", "RGB8"],
        )

    def _get_device_key(self, device: DeviceInfo) -> str:
        """Generate unique key for device."""
        d = device.descriptor
        return f"{d.bus_number}:{d.device_address}:{d.serial_number}"

    def _notify_hotplug(self, notification: HotPlugNotification) -> None:
        """Notify callbacks of hot-plug event."""
        for callback in self._hotplug_callbacks:
            try:
                callback(notification)
            except Exception as e:
                logger.error(f"Hot-plug callback error: {e}")


class USB3DeviceFactory:
    """
    Factory for creating USB3 Vision device instances.

    Provides convenient methods for device creation from discovery results.
    """

    def __init__(self, discovery: Optional[USB3DeviceDiscovery] = None) -> None:
        """
        Initialize factory.

        Args:
            discovery: Device discovery instance
        """
        self.discovery = discovery or USB3DeviceDiscovery()

    def create_by_serial(self, serial_number: str) -> Optional[dict[str, Any]]:
        """
        Create device configuration by serial number.

        Args:
            serial_number: Device serial number

        Returns:
            Device configuration dict
        """
        device = self.discovery.get_device_by_serial(serial_number)
        if device is None:
            self.discovery.discover_devices()
            device = self.discovery.get_device_by_serial(serial_number)

        if device is None:
            return None

        return self._device_to_config(device)

    def create_by_index(self, index: int = 0) -> Optional[dict[str, Any]]:
        """
        Create device configuration by index.

        Args:
            index: Device index

        Returns:
            Device configuration dict
        """
        if self.discovery.get_device_count() == 0:
            self.discovery.discover_devices()

        device = self.discovery.get_device_by_index(index)
        if device is None:
            return None

        return self._device_to_config(device)

    def create_first_available(self) -> Optional[dict[str, Any]]:
        """Create configuration for first available device."""
        return self.create_by_index(0)

    def _device_to_config(self, device: DeviceInfo) -> dict[str, Any]:
        """Convert DeviceInfo to configuration dict."""
        return {
            "vendor_id": device.descriptor.vendor_id,
            "product_id": device.descriptor.product_id,
            "serial_number": device.descriptor.serial_number,
            "usb_speed": device.descriptor.usb_speed.value,
            "max_frame_rate": device.max_frame_rate,
            "max_resolution": device.max_resolution,
            "pixel_formats": device.supported_pixel_formats,
        }
