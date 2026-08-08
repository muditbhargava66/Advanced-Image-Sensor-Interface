"""
USB3 Vision Protocol Driver Implementation

Provides comprehensive USB3 Vision camera support with GenICam architecture,
transport layer abstractions, SFNC features, and proper device control.

Key Features:
- USB3 Vision 1.1 compliant
- GenICam/GenTL compatible
- SFNC standard features
- Bulk transfer simulation
- Device discovery and enumeration
- Full streaming support

Version: 3.0.0
"""

import logging
import struct
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Any, ClassVar, Optional


from ....performance.profiler import profile_function
from ..base import ConfigurationError, ConnectionError, DataTransferError, ProtocolCapabilities, StreamingProtocolBase

logger = logging.getLogger(__name__)


# =============================================================================
# USB3 Vision Constants (per USB3 Vision 1.1 specification)
# =============================================================================


class USB3VisionPrefix(IntEnum):
    """USB3 Vision command/acknowledge prefixes."""

    COMMAND = 0x43563355  # 'U3VC' little-endian
    ACKNOWLEDGE = 0x33563355  # 'U3V3' (simulated)
    PENDING_ACK = 0x50563355  # Pending acknowledge


class USB3VisionCommand(IntEnum):
    """USB3 Vision control commands."""

    READMEM_CMD = 0x0800
    READMEM_ACK = 0x0801
    WRITEMEM_CMD = 0x0802
    WRITEMEM_ACK = 0x0803
    PENDING_ACK = 0x0805
    EVENT_CMD = 0x0C00


class USB3RegisterAddress(IntEnum):
    """USB3 Vision bootstrap register addresses (per specification)."""

    # Bootstrap registers
    ABRM_GENCP_VERSION = 0x0000
    ABRM_MANUFACTURER_NAME = 0x0004
    ABRM_MODEL_NAME = 0x0044
    ABRM_FAMILY_NAME = 0x0084
    ABRM_DEVICE_VERSION = 0x00C4
    ABRM_MANUFACTURER_INFO = 0x0104
    ABRM_SERIAL_NUMBER = 0x0144
    ABRM_USER_DEFINED_NAME = 0x0184
    ABRM_DEVICE_CAPABILITY = 0x01C4
    ABRM_MAX_DEVICE_RESPONSE_TIME = 0x01CC
    ABRM_MANIFEST_TABLE_ADDRESS = 0x01D0
    ABRM_SBRM_ADDRESS = 0x01D8
    ABRM_DEVICE_CONFIGURATION = 0x01E0
    ABRM_HEARTBEAT_TIMEOUT = 0x01E8
    ABRM_MESSAGE_CHANNEL_ID = 0x01EC
    ABRM_TIMESTAMP = 0x01F0
    ABRM_TIMESTAMP_LATCH = 0x01F8
    ABRM_TIMESTAMP_INCREMENT = 0x01FC
    ABRM_ACCESS_PRIVILEGE = 0x0204
    ABRM_PROTOCOL_ENDIANNESS = 0x0208

    # Streaming bootstrap registers (SBRM)
    SBRM_SIRM_ADDRESS = 0x0000
    SBRM_SIRM_LENGTH = 0x0008
    SBRM_EIRM_ADDRESS = 0x000C
    SBRM_EIRM_LENGTH = 0x0014


class USBSpeed(Enum):
    """USB interface speed enumeration."""

    HIGH_SPEED = "HighSpeed"  # USB 2.0 - 480 Mbps
    SUPER_SPEED = "SuperSpeed"  # USB 3.0 - 5 Gbps
    SUPER_SPEED_PLUS = "SuperSpeedPlus"  # USB 3.1/3.2 - 10 Gbps


# =============================================================================
# GenICam SFNC Feature Nodes
# =============================================================================


class FeatureType(Enum):
    """GenICam feature node types."""

    INTEGER = "Integer"
    FLOAT = "Float"
    BOOLEAN = "Boolean"
    ENUMERATION = "Enumeration"
    STRING = "String"
    COMMAND = "Command"
    REGISTER = "Register"


@dataclass
class GenICamFeatureNode:
    """Represents a GenICam SFNC feature node."""

    name: str
    feature_type: FeatureType
    value: Any
    min_value: Any = None
    max_value: Any = None
    increment: Any = None
    unit: str = ""
    description: str = ""
    access_mode: str = "RW"  # RO, WO, RW
    visibility: str = "Beginner"  # Beginner, Expert, Guru, Invisible
    enum_entries: list[str] = field(default_factory=list)

    def set_value(self, value: Any) -> bool:
        """Set feature value with validation."""
        if self.access_mode == "RO":
            logger.warning(f"Cannot write to read-only feature: {self.name}")
            return False

        if self.feature_type == FeatureType.INTEGER:
            if self.min_value is not None and value < self.min_value:
                raise ValueError(f"{self.name}: value {value} below minimum {self.min_value}")
            if self.max_value is not None and value > self.max_value:
                raise ValueError(f"{self.name}: value {value} above maximum {self.max_value}")
            if self.increment and (value - self.min_value) % self.increment != 0:
                raise ValueError(f"{self.name}: value must be increment of {self.increment}")

        elif self.feature_type == FeatureType.FLOAT:
            if self.min_value is not None and value < self.min_value:
                raise ValueError(f"{self.name}: value {value} below minimum {self.min_value}")
            if self.max_value is not None and value > self.max_value:
                raise ValueError(f"{self.name}: value {value} above maximum {self.max_value}")

        elif self.feature_type == FeatureType.ENUMERATION:
            if value not in self.enum_entries:
                raise ValueError(f"{self.name}: invalid value '{value}'. Must be one of {self.enum_entries}")

        self.value = value
        return True


class GenICamNodeMap:
    """
    GenICam node map for SFNC standard features.

    Provides access to camera features through GenICam-compliant interface.
    """

    def __init__(self) -> None:
        """Initialize node map with SFNC standard features."""
        self._nodes: dict[str, GenICamFeatureNode] = {}
        self._callbacks: dict[str, list[Callable]] = {}
        self._initialize_sfnc_features()

    def _initialize_sfnc_features(self) -> None:
        """Initialize standard SFNC features."""
        # Device Information category
        self._add_node(
            GenICamFeatureNode(
                name="DeviceVendorName",
                feature_type=FeatureType.STRING,
                value="Advanced Image Sensor Interface",
                access_mode="RO",
                description="Name of the manufacturer",
            )
        )
        self._add_node(
            GenICamFeatureNode(
                name="DeviceModelName",
                feature_type=FeatureType.STRING,
                value="USB3 Vision Camera Simulator",
                access_mode="RO",
                description="Model name of the device",
            )
        )
        self._add_node(
            GenICamFeatureNode(
                name="DeviceSerialNumber",
                feature_type=FeatureType.STRING,
                value="USB3-SIM-001",
                access_mode="RO",
                description="Serial number",
            )
        )
        self._add_node(
            GenICamFeatureNode(
                name="DeviceFirmwareVersion",
                feature_type=FeatureType.STRING,
                value="3.0.0",
                access_mode="RO",
                description="Firmware version",
            )
        )

        # Image Format Control category
        self._add_node(
            GenICamFeatureNode(
                name="Width",
                feature_type=FeatureType.INTEGER,
                value=1920,
                min_value=1,
                max_value=4096,
                increment=1,
                unit="pixels",
                description="Width of the image",
            )
        )
        self._add_node(
            GenICamFeatureNode(
                name="Height",
                feature_type=FeatureType.INTEGER,
                value=1080,
                min_value=1,
                max_value=4096,
                increment=1,
                unit="pixels",
                description="Height of the image",
            )
        )
        self._add_node(
            GenICamFeatureNode(
                name="PixelFormat",
                feature_type=FeatureType.ENUMERATION,
                value="Mono8",
                enum_entries=[
                    "Mono8",
                    "Mono10",
                    "Mono12",
                    "Mono16",
                    "BayerGR8",
                    "BayerRG8",
                    "BayerGB8",
                    "BayerBG8",
                    "RGB8",
                    "BGR8",
                    "YUV422",
                ],
                description="Pixel format",
            )
        )
        self._add_node(
            GenICamFeatureNode(
                name="OffsetX",
                feature_type=FeatureType.INTEGER,
                value=0,
                min_value=0,
                max_value=4095,
                increment=1,
                unit="pixels",
                description="Horizontal offset from origin",
            )
        )
        self._add_node(
            GenICamFeatureNode(
                name="OffsetY",
                feature_type=FeatureType.INTEGER,
                value=0,
                min_value=0,
                max_value=4095,
                increment=1,
                unit="pixels",
                description="Vertical offset from origin",
            )
        )

        # Acquisition Control category
        self._add_node(
            GenICamFeatureNode(
                name="AcquisitionMode",
                feature_type=FeatureType.ENUMERATION,
                value="Continuous",
                enum_entries=["SingleFrame", "MultiFrame", "Continuous"],
                description="Acquisition mode",
            )
        )
        self._add_node(
            GenICamFeatureNode(
                name="AcquisitionFrameRate",
                feature_type=FeatureType.FLOAT,
                value=30.0,
                min_value=0.1,
                max_value=1000.0,
                unit="Hz",
                description="Frame rate",
            )
        )
        self._add_node(
            GenICamFeatureNode(
                name="AcquisitionFrameRateEnable",
                feature_type=FeatureType.BOOLEAN,
                value=True,
                description="Enable frame rate control",
            )
        )
        self._add_node(
            GenICamFeatureNode(
                name="ExposureTime",
                feature_type=FeatureType.FLOAT,
                value=10000.0,
                min_value=10.0,
                max_value=1000000.0,
                unit="us",
                description="Exposure time in microseconds",
            )
        )
        self._add_node(
            GenICamFeatureNode(
                name="ExposureAuto",
                feature_type=FeatureType.ENUMERATION,
                value="Off",
                enum_entries=["Off", "Once", "Continuous"],
                description="Auto exposure mode",
            )
        )

        # Analog Control category
        self._add_node(
            GenICamFeatureNode(
                name="Gain",
                feature_type=FeatureType.FLOAT,
                value=1.0,
                min_value=0.0,
                max_value=48.0,
                unit="dB",
                description="Analog gain",
            )
        )
        self._add_node(
            GenICamFeatureNode(
                name="GainAuto",
                feature_type=FeatureType.ENUMERATION,
                value="Off",
                enum_entries=["Off", "Once", "Continuous"],
                description="Auto gain mode",
            )
        )
        self._add_node(
            GenICamFeatureNode(
                name="BlackLevel",
                feature_type=FeatureType.FLOAT,
                value=0.0,
                min_value=0.0,
                max_value=255.0,
                description="Black level offset",
            )
        )
        self._add_node(
            GenICamFeatureNode(
                name="Gamma",
                feature_type=FeatureType.FLOAT,
                value=1.0,
                min_value=0.1,
                max_value=4.0,
                description="Gamma correction",
            )
        )

        # Trigger Control category
        self._add_node(
            GenICamFeatureNode(
                name="TriggerMode",
                feature_type=FeatureType.ENUMERATION,
                value="Off",
                enum_entries=["Off", "On"],
                description="Trigger mode",
            )
        )
        self._add_node(
            GenICamFeatureNode(
                name="TriggerSource",
                feature_type=FeatureType.ENUMERATION,
                value="Software",
                enum_entries=["Software", "Line0", "Line1", "Line2", "Action0"],
                description="Trigger source",
            )
        )
        self._add_node(
            GenICamFeatureNode(
                name="TriggerActivation",
                feature_type=FeatureType.ENUMERATION,
                value="RisingEdge",
                enum_entries=["RisingEdge", "FallingEdge", "AnyEdge", "LevelHigh", "LevelLow"],
                description="Trigger activation mode",
            )
        )
        self._add_node(
            GenICamFeatureNode(
                name="TriggerDelay",
                feature_type=FeatureType.FLOAT,
                value=0.0,
                min_value=0.0,
                max_value=1000000.0,
                unit="us",
                description="Trigger delay",
            )
        )

        # Transport Layer Control
        self._add_node(
            GenICamFeatureNode(
                name="PayloadSize",
                feature_type=FeatureType.INTEGER,
                value=2073600,  # 1920x1080
                access_mode="RO",
                unit="bytes",
                description="Payload size for each image",
            )
        )

    def _add_node(self, node: GenICamFeatureNode) -> None:
        """Add a feature node to the map."""
        self._nodes[node.name] = node
        self._callbacks[node.name] = []

    def get_node(self, name: str) -> Optional[GenICamFeatureNode]:
        """Get feature node by name."""
        return self._nodes.get(name)

    def set_value(self, name: str, value: Any) -> bool:
        """Set feature value by name."""
        node = self._nodes.get(name)
        if not node:
            raise ValueError(f"Unknown feature: {name}")

        if node.set_value(value):
            # Notify callbacks
            for callback in self._callbacks.get(name, []):
                callback(name, value)
            return True
        return False

    def get_value(self, name: str) -> Any:
        """Get feature value by name."""
        node = self._nodes.get(name)
        if not node:
            raise ValueError(f"Unknown feature: {name}")
        return node.value

    def register_callback(self, name: str, callback: Callable) -> None:
        """Register callback for feature changes."""
        if name in self._callbacks:
            self._callbacks[name].append(callback)

    def get_all_features(self) -> dict[str, Any]:
        """Get all feature values as dictionary."""
        return {name: node.value for name, node in self._nodes.items()}


# =============================================================================
# USB Transport Layer
# =============================================================================


@dataclass
class USBTransferResult:
    """Result of a USB transfer operation."""

    success: bool
    data: bytes = b""
    bytes_transferred: int = 0
    status: str = "OK"
    error_code: int = 0


class USBTransportLayer:
    """
    USB3 Vision transport layer abstraction.

    Implements USB bulk and control transfers for USB3 Vision protocol.
    """

    # USB error codes
    ERROR_NONE = 0
    ERROR_TIMEOUT = -1
    ERROR_STALL = -2
    ERROR_NO_DEVICE = -3
    ERROR_OVERFLOW = -4
    ERROR_BABBLE = -5

    def __init__(self, max_packet_size: int = 1024, transfer_timeout_ms: int = 5000) -> None:
        """Initialize USB transport layer."""
        self.max_packet_size = max_packet_size
        self.transfer_timeout_ms = transfer_timeout_ms
        self._is_open = False
        self._device_address: int = 0

        # Endpoint configuration (per USB3 Vision spec)
        self._control_endpoint = 0x00
        self._bulk_in_endpoint = 0x81  # Control channel read
        self._bulk_out_endpoint = 0x02  # Control channel write
        self._stream_endpoint = 0x83  # Stream channel

        # Transfer statistics
        self.stats = {
            "control_transfers": 0,
            "bulk_in_transfers": 0,
            "bulk_out_transfers": 0,
            "stream_transfers": 0,
            "bytes_received": 0,
            "bytes_sent": 0,
            "errors": 0,
            "timeouts": 0,
        }

    def open(self, device_address: int) -> bool:
        """Open USB device connection."""
        if self._is_open:
            logger.warning("USB transport already open")
            return True

        # Simulate USB device opening with enumeration delay
        time.sleep(0.05)
        self._device_address = device_address
        self._is_open = True
        logger.debug(f"USB transport opened for device {device_address}")
        return True

    def close(self) -> bool:
        """Close USB device connection."""
        if not self._is_open:
            return True

        self._is_open = False
        self._device_address = 0
        logger.debug("USB transport closed")
        return True

    def control_transfer(
        self, request_type: int, request: int, value: int, index: int, data: Optional[bytes] = None, length: int = 0
    ) -> USBTransferResult:
        """
        Perform USB control transfer.

        Args:
            request_type: bmRequestType (direction, type, recipient)
            request: bRequest
            value: wValue
            index: wIndex
            data: Data to send (for OUT) or None (for IN)
            length: Expected data length for IN transfers
        """
        if not self._is_open:
            return USBTransferResult(success=False, status="Device not open", error_code=self.ERROR_NO_DEVICE)

        self.stats["control_transfers"] += 1

        # Simulate control transfer timing
        time.sleep(0.001)

        # Direction is in bit 7 of request_type
        is_in = (request_type & 0x80) != 0

        if is_in:
            # Generate response data
            response = bytes([i % 256 for i in range(length)])
            self.stats["bytes_received"] += len(response)
            return USBTransferResult(success=True, data=response, bytes_transferred=len(response))
        else:
            # OUT transfer
            if data:
                self.stats["bytes_sent"] += len(data)
            return USBTransferResult(success=True, bytes_transferred=len(data) if data else 0)

    def bulk_write(self, endpoint: int, data: bytes) -> USBTransferResult:
        """Perform USB bulk OUT transfer."""
        if not self._is_open:
            return USBTransferResult(success=False, status="Device not open", error_code=self.ERROR_NO_DEVICE)

        self.stats["bulk_out_transfers"] += 1

        # Simulate bulk transfer with bandwidth-based timing
        transfer_time = len(data) / (5_000_000_000 / 8)  # 5 Gbps
        time.sleep(max(0.0001, transfer_time))

        self.stats["bytes_sent"] += len(data)
        return USBTransferResult(success=True, bytes_transferred=len(data))

    def bulk_read(self, endpoint: int, length: int) -> USBTransferResult:
        """Perform USB bulk IN transfer."""
        if not self._is_open:
            return USBTransferResult(success=False, status="Device not open", error_code=self.ERROR_NO_DEVICE)

        self.stats["bulk_in_transfers"] += 1

        # Simulate bulk transfer timing
        transfer_time = length / (5_000_000_000 / 8)
        time.sleep(max(0.0001, transfer_time))

        data = bytes([i % 256 for i in range(length)])
        self.stats["bytes_received"] += len(data)

        return USBTransferResult(success=True, data=data, bytes_transferred=len(data))

    def reset_statistics(self) -> None:
        """Reset transfer statistics."""
        for key in self.stats:
            self.stats[key] = 0


# =============================================================================
# USB3 Vision Device Register Map
# =============================================================================


class USB3DeviceRegisterMap:
    """
    USB3 Vision device register map.

    Simulates device bootstrap registers per USB3 Vision specification.
    """

    def __init__(self) -> None:
        """Initialize register map with default values."""
        self._registers: dict[int, bytes] = {}
        self._initialize_registers()

    def _initialize_registers(self) -> None:
        """Initialize bootstrap registers."""
        # GenCP version: 1.1
        self._set_string(USB3RegisterAddress.ABRM_GENCP_VERSION, "1.1")

        # Device identification
        self._set_string(USB3RegisterAddress.ABRM_MANUFACTURER_NAME, "Advanced Image Sensor Interface")
        self._set_string(USB3RegisterAddress.ABRM_MODEL_NAME, "USB3Vision-SIM")
        self._set_string(USB3RegisterAddress.ABRM_FAMILY_NAME, "Simulated Cameras")
        self._set_string(USB3RegisterAddress.ABRM_DEVICE_VERSION, "3.0.0")
        self._set_string(USB3RegisterAddress.ABRM_MANUFACTURER_INFO, "High-Performance Camera Framework")
        self._set_string(USB3RegisterAddress.ABRM_SERIAL_NUMBER, "USB3-SIM-001")
        self._set_string(USB3RegisterAddress.ABRM_USER_DEFINED_NAME, "SimCamera1")

        # Device capability flags
        self._set_uint32(USB3RegisterAddress.ABRM_DEVICE_CAPABILITY, 0x0000000F)

        # Timing parameters
        self._set_uint32(USB3RegisterAddress.ABRM_MAX_DEVICE_RESPONSE_TIME, 1000)
        self._set_uint32(USB3RegisterAddress.ABRM_HEARTBEAT_TIMEOUT, 3000)

        # Endianness (0 = little-endian)
        self._set_uint32(USB3RegisterAddress.ABRM_PROTOCOL_ENDIANNESS, 0)

    def _set_string(self, address: int, value: str, max_length: int = 64) -> None:
        """Set string register value."""
        data = value.encode("utf-8")[:max_length].ljust(max_length, b"\x00")
        self._registers[address] = data

    def _set_uint32(self, address: int, value: int) -> None:
        """Set 32-bit unsigned integer register."""
        self._registers[address] = struct.pack("<I", value)

    def _set_uint64(self, address: int, value: int) -> None:
        """Set 64-bit unsigned integer register."""
        self._registers[address] = struct.pack("<Q", value)

    def read(self, address: int, length: int) -> bytes:
        """Read from register map."""
        if address in self._registers:
            data = self._registers[address]
            return data[:length]
        return b"\x00" * length

    def write(self, address: int, data: bytes) -> bool:
        """Write to register map."""
        self._registers[address] = data
        return True

    def read_string(self, address: int, max_length: int = 64) -> str:
        """Read string from register."""
        data = self.read(address, max_length)
        return data.rstrip(b"\x00").decode("utf-8", errors="replace")

    def read_uint32(self, address: int) -> int:
        """Read 32-bit unsigned integer."""
        data = self.read(address, 4)
        return struct.unpack("<I", data)[0]


# =============================================================================
# USB3 Vision Configuration
# =============================================================================


@dataclass
class USB3VisionConfig:
    """
    Configuration for USB3 Vision protocol.

    All parameters are validated on initialization.
    """

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
    usb_speed: str = "SuperSpeed"
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
            raise ValueError(f"Invalid USB speed: {self.usb_speed}. Must be one of {valid_speeds}")

        if self.packet_size < 64 or self.packet_size > 65536:
            raise ValueError("Packet size must be between 64 and 65536 bytes")

        if self.buffer_count < 2:
            raise ValueError("Buffer count must be at least 2")

        if self.frame_rate <= 0:
            raise ValueError("Frame rate must be positive")

        if self.exposure_time_us < 0:
            raise ValueError("Exposure time cannot be negative")


# =============================================================================
# USB3 Vision Driver
# =============================================================================


class USB3VisionDriver(StreamingProtocolBase):
    """
    USB3 Vision protocol driver with GenICam architecture.

    Provides complete USB3 Vision 1.1 support including:
    - Device discovery and enumeration
    - GenICam/GenAPI feature control via SFNC
    - Bootstrap register access
    - Streaming with buffer management
    - Event notification support
    """

    # USB3 speed to bandwidth mapping (Mbps)
    USB_SPEEDS: ClassVar[dict[str, int]] = {"HighSpeed": 480, "SuperSpeed": 5000, "SuperSpeedPlus": 10000}

    def __init__(self, config: USB3VisionConfig) -> None:
        """
        Initialize USB3 Vision driver.

        Args:
            config: USB3 Vision configuration
        """
        self.usb3_config = config
        super().__init__(config.__dict__)

        # Transport and device components
        self._transport = USBTransportLayer(max_packet_size=config.packet_size, transfer_timeout_ms=config.timeout_ms)
        self._register_map = USB3DeviceRegisterMap()
        self._node_map = GenICamNodeMap()

        # Device state
        self.device_handle: Optional[str] = None
        self.stream_handle: Optional[str] = None
        self.device_info: dict[str, Any] = {}
        self._frame_count = 0
        self._start_time: Optional[float] = None

        # Statistics
        self.stats = {
            "frames_captured": 0,
            "frames_dropped": 0,
            "bytes_transferred": 0,
            "usb_errors": 0,
            "reconnections": 0,
            "incomplete_frames": 0,
        }

        # Apply configuration to node map
        self._apply_config_to_nodemap()

        logger.info(f"USB3 Vision driver initialized for {config.usb_speed} device")

    def _apply_config_to_nodemap(self) -> None:
        """Apply driver config to GenICam node map."""
        self._node_map.set_value("Width", self.usb3_config.resolution[0])
        self._node_map.set_value("Height", self.usb3_config.resolution[1])
        self._node_map.set_value("PixelFormat", self.usb3_config.pixel_format)
        self._node_map.set_value("AcquisitionFrameRate", self.usb3_config.frame_rate)
        self._node_map.set_value("ExposureTime", self.usb3_config.exposure_time_us)
        self._node_map.set_value("Gain", self.usb3_config.gain)

        # Update payload size
        width = self.usb3_config.resolution[0]
        height = self.usb3_config.resolution[1]
        bpp = self._get_bytes_per_pixel(self.usb3_config.pixel_format)
        self._node_map._nodes["PayloadSize"].value = width * height * bpp

    def _get_capabilities(self) -> ProtocolCapabilities:
        """Get USB3 Vision protocol capabilities."""
        max_bandwidth = self.USB_SPEEDS[self.usb3_config.usb_speed]

        return ProtocolCapabilities(
            max_bandwidth_gbps=max_bandwidth / 1000.0,
            max_distance_m=5.0,
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

            # Discover device
            device_info = self._discover_device()
            if not device_info:
                raise ConnectionError("No USB3 Vision device found")
            self.device_info = device_info

            # Open USB transport
            if not self._transport.open(device_info.get("device_address", 1)):
                raise ConnectionError("Failed to open USB transport")

            # Read bootstrap registers
            self._read_device_identification()

            # Open device handle
            self.device_handle = f"usb3_{device_info['vendor_id']:04x}_{device_info['product_id']:04x}"

            # Initialize device with config
            self._initialize_device()

            # Update status
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

            # Close transport
            self._transport.close()

            # Clear handles
            self.device_handle = None
            self.stream_handle = None

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
        """Send control data via USB bulk transfer."""
        if not self.is_connected:
            raise ConnectionError("Not connected to USB3 Vision device")

        try:
            result = self._transport.bulk_write(0x02, data)
            if not result.success:
                raise DataTransferError(f"USB bulk write failed: {result.status}")

            self.status.bytes_transmitted += len(data)
            logger.debug(f"Sent {len(data)} bytes to USB3 Vision device")
            return True

        except Exception as e:
            logger.error(f"Failed to send data: {e}")
            self.status.error_count += 1
            self.stats["usb_errors"] += 1
            raise DataTransferError(f"USB3 Vision send failed: {e}")

    def receive_data(self, size: int) -> Optional[bytes]:
        """Receive control data via USB bulk transfer."""
        if not self.is_connected:
            raise ConnectionError("Not connected to USB3 Vision device")

        try:
            result = self._transport.bulk_read(0x81, size)
            if not result.success:
                raise DataTransferError(f"USB bulk read failed: {result.status}")

            self.status.bytes_received += len(result.data)
            logger.debug(f"Received {len(result.data)} bytes from USB3 Vision device")
            return result.data

        except Exception as e:
            logger.error(f"Failed to receive data: {e}")
            self.status.error_count += 1
            self.stats["usb_errors"] += 1
            raise DataTransferError(f"USB3 Vision receive failed: {e}")

    def read_register(self, address: int, length: int = 4) -> bytes:
        """Read from device register."""
        if not self.is_connected:
            raise ConnectionError("Not connected")

        # Build READMEM command
        cmd = struct.pack(
            "<IIIHHI", USB3VisionPrefix.COMMAND, USB3VisionCommand.READMEM_CMD, 0, 0, length, address  # Request ID  # Flags
        )
        self._transport.bulk_write(0x02, cmd)

        # Read response from register map
        return self._register_map.read(address, length)

    def write_register(self, address: int, data: bytes) -> bool:
        """Write to device register."""
        if not self.is_connected:
            raise ConnectionError("Not connected")

        # Build WRITEMEM command
        cmd = (
            struct.pack(
                "<IIIHHI",
                USB3VisionPrefix.COMMAND,
                USB3VisionCommand.WRITEMEM_CMD,
                0,  # Request ID
                0,  # Flags
                len(data),
                address,
            )
            + data
        )

        self._transport.bulk_write(0x02, cmd)
        return self._register_map.write(address, data)

    @profile_function
    def start_streaming(self) -> bool:
        """Start image streaming."""
        if not self.is_connected:
            raise ConnectionError("Not connected to USB3 Vision device")

        try:
            logger.info("Starting USB3 Vision streaming...")

            # Configure stream parameters
            self._configure_streaming()

            # Allocate stream buffers
            self.stream_handle = f"stream_{id(self)}"

            # Start acquisition
            self._node_map.set_value("AcquisitionMode", "Continuous")
            self._start_acquisition()

            self.is_streaming = True
            self._start_time = time.time()
            self._frame_count = 0

            logger.info("USB3 Vision streaming started")
            return True

        except Exception as e:
            logger.error(f"Failed to start streaming: {e}")
            self.status.error_count += 1
            return False

    def stop_streaming(self) -> bool:
        """Stop image streaming."""
        try:
            if self.is_streaming:
                self._stop_acquisition()
                self.stream_handle = None
                self.is_streaming = False

                if self._start_time:
                    duration = time.time() - self._start_time
                    avg_fps = self._frame_count / duration if duration > 0 else 0
                    logger.info(f"Streaming stopped. Captured {self._frame_count} frames at {avg_fps:.2f} FPS")

            return True

        except Exception as e:
            logger.error(f"Error stopping streaming: {e}")
            return False

    @profile_function
    def capture_frame(self) -> Optional[bytes]:
        """Capture single frame."""
        if not self.is_connected:
            raise ConnectionError("Not connected to USB3 Vision device")

        try:
            width = self._node_map.get_value("Width")
            height = self._node_map.get_value("Height")
            pixel_format = self._node_map.get_value("PixelFormat")
            bpp = self._get_bytes_per_pixel(pixel_format)
            frame_size = width * height * bpp

            # Simulate frame capture timing
            frame_rate = self._node_map.get_value("AcquisitionFrameRate")
            capture_delay = 1.0 / frame_rate
            transfer_time = frame_size / (self.USB_SPEEDS[self.usb3_config.usb_speed] * 1024 * 1024 / 8)
            time.sleep(min(capture_delay, transfer_time + 0.001))

            # Generate frame
            frame_data = self._generate_frame(width, height, bpp)

            # Update statistics
            self._frame_count += 1
            self.stats["frames_captured"] += 1
            self.stats["bytes_transferred"] += len(frame_data)

            logger.debug(f"Captured USB3 frame {self._frame_count}: {len(frame_data)} bytes")
            return frame_data

        except Exception as e:
            logger.error(f"Frame capture failed: {e}")
            self.stats["usb_errors"] += 1
            self.status.error_count += 1
            return None

    # GenICam feature access
    def get_feature(self, name: str) -> Any:
        """Get GenICam feature value."""
        return self._node_map.get_value(name)

    def set_feature(self, name: str, value: Any) -> bool:
        """Set GenICam feature value."""
        try:
            return self._node_map.set_value(name, value)
        except ValueError as e:
            raise ConfigurationError(str(e))

    def get_node_map(self) -> GenICamNodeMap:
        """Get GenICam node map for direct access."""
        return self._node_map

    def _discover_device(self) -> Optional[dict[str, Any]]:
        """Discover USB3 Vision devices."""
        # Simulate USB enumeration delay
        time.sleep(0.1)

        return {
            "vendor_id": self.usb3_config.vendor_id or 0x1234,
            "product_id": self.usb3_config.product_id or 0x5678,
            "serial_number": self.usb3_config.serial_number or "USB3CAM001",
            "manufacturer": "Advanced Image Sensor Interface",
            "model": "USB3 Vision Camera Simulator",
            "firmware_version": "3.0.0",
            "usb_speed": self.usb3_config.usb_speed,
            "device_address": 1,
        }

    def _read_device_identification(self) -> None:
        """Read device identification from bootstrap registers."""
        self.device_info["manufacturer"] = self._register_map.read_string(USB3RegisterAddress.ABRM_MANUFACTURER_NAME)
        self.device_info["model"] = self._register_map.read_string(USB3RegisterAddress.ABRM_MODEL_NAME)
        self.device_info["serial_number"] = self._register_map.read_string(USB3RegisterAddress.ABRM_SERIAL_NUMBER)
        self.device_info["firmware_version"] = self._register_map.read_string(USB3RegisterAddress.ABRM_DEVICE_VERSION)

    def _initialize_device(self) -> None:
        """Initialize device with configuration."""
        params = {
            "Width": self.usb3_config.resolution[0],
            "Height": self.usb3_config.resolution[1],
            "PixelFormat": self.usb3_config.pixel_format,
            "AcquisitionFrameRate": self.usb3_config.frame_rate,
            "ExposureTime": self.usb3_config.exposure_time_us,
            "Gain": self.usb3_config.gain,
        }
        for name, value in params.items():
            try:
                self._node_map.set_value(name, value)
            except Exception as e:
                logger.warning(f"Could not set {name}: {e}")

        logger.debug(f"Initialized USB3 device with parameters: {params}")

    def _configure_streaming(self) -> None:
        """Configure streaming parameters."""
        config = {
            "packet_size": self.usb3_config.packet_size,
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

    def _get_bytes_per_pixel(self, pixel_format: str) -> int:
        """Get bytes per pixel using shared base implementation."""
        return self._get_bytes_per_pixel_common(pixel_format)

    def _generate_frame(self, width: int, height: int, bpp: int) -> bytes:
        """Generate frame data using shared vectorized implementation."""
        return self._generate_test_frame_vectorized(width, height, bpp, self._frame_count)

    def get_device_info(self) -> dict[str, Any]:
        """Get device information."""
        return {
            "protocol": "USB3Vision",
            "device_info": self.device_info,
            "usb_speed": self.usb3_config.usb_speed,
            "max_bandwidth_mbps": self.USB_SPEEDS[self.usb3_config.usb_speed],
            "pixel_format": self._node_map.get_value("PixelFormat"),
            "resolution": (self._node_map.get_value("Width"), self._node_map.get_value("Height")),
            "frame_rate": self._node_map.get_value("AcquisitionFrameRate"),
            "exposure_time_us": self._node_map.get_value("ExposureTime"),
            "gain_db": self._node_map.get_value("Gain"),
        }

    def get_statistics(self) -> dict[str, Any]:
        """Get driver statistics."""
        uptime = time.time() - self._start_time if self._start_time else 0
        return {
            **self.stats,
            **self._transport.stats,
            "connection_quality": self.status.connection_quality,
            "usb_speed": self.usb3_config.usb_speed,
            "uptime_seconds": uptime,
            "current_fps": self._frame_count / uptime if uptime > 0 else 0,
            "error_rate": self.stats["usb_errors"] / max(1, self.stats["frames_captured"]),
        }

    def _get_current_frame_rate(self) -> float:
        """Get current frame rate."""
        if self._start_time and self._frame_count > 0:
            elapsed = time.time() - self._start_time
            return self._frame_count / elapsed if elapsed > 0 else 0.0
        return 0.0

    def _get_dropped_frame_count(self) -> int:
        """Get number of dropped frames."""
        return self.stats["frames_dropped"]

    def _get_buffer_utilization(self) -> float:
        """Estimate buffer utilization from achieved frame rate."""
        target_fps = self.usb3_config.frame_rate
        actual_fps = self._get_current_frame_rate()
        if target_fps > 0:
            return min(100.0, (actual_fps / target_fps) * 100.0)
        return 0.0

    def get_status(self) -> dict[str, Any]:
        """Get USB3 Vision protocol status (legacy compatibility)."""
        return {
            "protocol": "USB3 Vision",
            "connected": self.is_connected,
            "streaming": self.is_streaming,
            "config": {
                "usb_speed": self.usb3_config.usb_speed,
                "packet_size": self.usb3_config.packet_size,
                "buffer_count": self.usb3_config.buffer_count,
                "pixel_format": self.usb3_config.pixel_format,
                "resolution": self.usb3_config.resolution,
                "frame_rate": self.usb3_config.frame_rate,
            },
            "statistics": self.get_statistics(),
        }
