"""
CoaXPress Protocol Driver Implementation

Provides comprehensive CoaXPress camera support with GenICam architecture,
CXP link layer, trigger control, and power-over-coax management.

Key Features:
- CoaXPress 2.1 compliant (CXP-1 to CXP-12)
- Multi-connection aggregation (up to 4x)
- GenICam/GenTL compatible
- SFNC standard features
- Hardware trigger support
- Power over Coax (PoCXP)

Version: 3.0.0
"""

import logging
import struct
import time
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Any, ClassVar, Optional


from ..base import ConfigurationError, ConnectionError, DataTransferError, ProtocolCapabilities, StreamingProtocolBase

logger = logging.getLogger(__name__)


# =============================================================================
# CoaXPress Protocol Constants
# =============================================================================


class CXPSpeedGrade(Enum):
    """CoaXPress speed grades per CXP 2.1 specification."""

    CXP_1 = "CXP-1"  # 1.25 Gbps
    CXP_2 = "CXP-2"  # 2.5 Gbps
    CXP_3 = "CXP-3"  # 3.125 Gbps
    CXP_5 = "CXP-5"  # 5.0 Gbps
    CXP_6 = "CXP-6"  # 6.25 Gbps
    CXP_10 = "CXP-10"  # 10.0 Gbps
    CXP_12 = "CXP-12"  # 12.5 Gbps


class CXPLinkState(Enum):
    """CoaXPress link state machine states."""

    DISCONNECTED = "Disconnected"
    DISCOVERY = "Discovery"
    TRAINING = "Training"
    SYNCHRONIZED = "Synchronized"
    CONNECTED = "Connected"
    ERROR = "Error"


class CXPMessageType(IntEnum):
    """CoaXPress message types (uplink/downlink)."""

    # Downlink (camera to host)
    DATA_STREAM = 0x01
    HEARTBEAT = 0x02
    ACKNOWLEDGE = 0x03
    EVENT = 0x04

    # Uplink (host to camera)
    CONTROL = 0x10
    TRIGGER = 0x11
    GPIO = 0x12


class CXPTriggerMode(Enum):
    """CoaXPress trigger modes."""

    DISABLED = "Disabled"
    SOFTWARE = "Software"
    HARDWARE_LINE0 = "Line0"
    HARDWARE_LINE1 = "Line1"
    HARDWARE_LINE2 = "Line2"
    GPIO = "GPIO"


class CXPRegisterAddress(IntEnum):
    """CoaXPress bootstrap register addresses."""

    # Standard registers
    STANDARD = 0x00000000
    REVISION = 0x00000004
    XML_MANIFEST = 0x00000008
    XML_MANIFEST_SIZE = 0x0000000C
    DEVICE_ID = 0x00000010
    MANUFACTURER_NAME = 0x00000020
    MODEL_NAME = 0x00000040
    MANUFACTURER_INFO = 0x00000060
    DEVICE_VERSION = 0x00000080
    SERIAL_NUMBER = 0x000000A0
    USER_DEFINED_NAME = 0x000000C0

    # Connection registers
    MASTER_LINK_SPEED = 0x00000100
    LINK_CONFIG = 0x00000104
    CONNECTION_CONFIG = 0x00000108
    STREAM_SIZE = 0x00000110

    # Power registers
    POCXP_STATUS = 0x00000200
    POCXP_CURRENT = 0x00000204
    POCXP_VOLTAGE = 0x00000208


# =============================================================================
# CXP Link Layer
# =============================================================================


@dataclass
class CXPLinkStatus:
    """Status of a single CoaXPress link."""

    link_id: int = 0
    state: CXPLinkState = CXPLinkState.DISCONNECTED
    speed_grade: str = "CXP-6"
    bandwidth_mbps: float = 6250.0
    error_count: int = 0
    last_error: str = ""
    link_up_time: float = 0.0
    crc_errors: int = 0
    disparity_errors: int = 0


class CXPLinkLayer:
    """
    CoaXPress link layer management.

    Handles link negotiation, speed grade detection, and connection management.
    """

    # Speed grade to bandwidth mapping (Mbps)
    SPEED_BANDWIDTH: ClassVar[dict[str, float]] = {
        "CXP-1": 1250.0,
        "CXP-2": 2500.0,
        "CXP-3": 3125.0,
        "CXP-5": 5000.0,
        "CXP-6": 6250.0,
        "CXP-10": 10000.0,
        "CXP-12": 12500.0,
    }

    def __init__(self, num_connections: int = 1) -> None:
        """
        Initialize CXP link layer.

        Args:
            num_connections: Number of coax connections (1-4)
        """
        if not 1 <= num_connections <= 4:
            raise ValueError("Number of connections must be between 1 and 4")

        self.num_connections = num_connections
        self._links: list[CXPLinkStatus] = [CXPLinkStatus(link_id=i) for i in range(num_connections)]
        self._negotiated_speed: str = "CXP-6"
        self._link_start_time: Optional[float] = None

    def negotiate_link(self, target_speed: str) -> bool:
        """
        Negotiate link speed with device.

        Implements CXP link training sequence.

        Args:
            target_speed: Target speed grade

        Returns:
            True if negotiation successful
        """
        if target_speed not in self.SPEED_BANDWIDTH:
            logger.error(f"Invalid speed grade: {target_speed}")
            return False

        logger.info(f"Starting CXP link negotiation for {target_speed}...")

        for link in self._links:
            # State machine transitions
            link.state = CXPLinkState.DISCOVERY
            time.sleep(0.01)  # Discovery phase

            link.state = CXPLinkState.TRAINING
            time.sleep(0.02)  # Training/equalization phase

            # Simulate successful training
            link.state = CXPLinkState.SYNCHRONIZED
            link.speed_grade = target_speed
            link.bandwidth_mbps = self.SPEED_BANDWIDTH[target_speed]

            time.sleep(0.01)  # Synchronization verification

            link.state = CXPLinkState.CONNECTED

        self._negotiated_speed = target_speed
        self._link_start_time = time.time()

        logger.info(f"CXP link negotiation complete: {target_speed} x {self.num_connections}")
        return True

    def get_link_status(self, link_id: int = 0) -> CXPLinkStatus:
        """Get status of specific link."""
        if 0 <= link_id < len(self._links):
            return self._links[link_id]
        raise ValueError(f"Invalid link ID: {link_id}")

    def get_total_bandwidth(self) -> float:
        """Get total aggregated bandwidth in Mbps."""
        return sum(link.bandwidth_mbps for link in self._links if link.state == CXPLinkState.CONNECTED)

    def is_connected(self) -> bool:
        """Check if all links are connected."""
        return all(link.state == CXPLinkState.CONNECTED for link in self._links)

    def disconnect(self) -> None:
        """Disconnect all links."""
        for link in self._links:
            link.state = CXPLinkState.DISCONNECTED
            link.bandwidth_mbps = 0.0
        self._link_start_time = None

    def get_uptime(self) -> float:
        """Get link uptime in seconds."""
        if self._link_start_time:
            return time.time() - self._link_start_time
        return 0.0


# =============================================================================
# CXP Control Channel
# =============================================================================


class CXPControlChannel:
    """
    CoaXPress control channel for uplink communication.

    Handles register access and camera control via GenCP-like protocol.
    """

    def __init__(self) -> None:
        """Initialize control channel."""
        self._registers: dict[int, bytes] = {}
        self._pending_acks: list[int] = []
        self._transaction_id: int = 0
        self._initialize_registers()

        # Statistics
        self.stats = {"read_commands": 0, "write_commands": 0, "bytes_read": 0, "bytes_written": 0, "errors": 0}

    def _initialize_registers(self) -> None:
        """Initialize device registers with default values."""
        # Device identification
        self._set_string(CXPRegisterAddress.MANUFACTURER_NAME, "Advanced Image Sensor Interface")
        self._set_string(CXPRegisterAddress.MODEL_NAME, "CoaXPress-SIM")
        self._set_string(CXPRegisterAddress.DEVICE_VERSION, "3.0.0")
        self._set_string(CXPRegisterAddress.SERIAL_NUMBER, "CXP-SIM-001")
        self._set_string(CXPRegisterAddress.USER_DEFINED_NAME, "SimCXPCamera")
        self._set_string(CXPRegisterAddress.MANUFACTURER_INFO, "High-Performance Camera Framework")

        # Standard register
        self._set_uint32(CXPRegisterAddress.STANDARD, 0x00020000)  # CXP 2.0
        self._set_uint32(CXPRegisterAddress.REVISION, 0x00000001)

        # PoCXP status (power available)
        self._set_uint32(CXPRegisterAddress.POCXP_STATUS, 0x00000001)
        self._set_uint32(CXPRegisterAddress.POCXP_VOLTAGE, 24000)  # 24V in mV
        self._set_uint32(CXPRegisterAddress.POCXP_CURRENT, 500)  # 500mA

    def _set_string(self, address: int, value: str, max_length: int = 32) -> None:
        """Set string register."""
        data = value.encode("utf-8")[:max_length].ljust(max_length, b"\x00")
        self._registers[address] = data

    def _set_uint32(self, address: int, value: int) -> None:
        """Set 32-bit register."""
        self._registers[address] = struct.pack(">I", value)  # CXP uses big-endian

    def read_register(self, address: int, length: int = 4) -> bytes:
        """
        Read from device register.

        Args:
            address: Register address
            length: Number of bytes to read

        Returns:
            Register data
        """
        self.stats["read_commands"] += 1
        self.stats["bytes_read"] += length

        # Simulate register read latency
        time.sleep(0.0005)

        if address in self._registers:
            data = self._registers[address]
            return data[:length]

        # Return zeros for uninitialized registers
        return b"\x00" * length

    def write_register(self, address: int, data: bytes) -> bool:
        """
        Write to device register.

        Args:
            address: Register address
            data: Data to write

        Returns:
            True if write successful
        """
        self.stats["write_commands"] += 1
        self.stats["bytes_written"] += len(data)

        # Simulate register write latency
        time.sleep(0.0005)

        self._registers[address] = data
        return True

    def read_string(self, address: int, max_length: int = 32) -> str:
        """Read string from register."""
        data = self.read_register(address, max_length)
        return data.rstrip(b"\x00").decode("utf-8", errors="replace")

    def read_uint32(self, address: int) -> int:
        """Read 32-bit unsigned integer."""
        data = self.read_register(address, 4)
        return struct.unpack(">I", data)[0]


# =============================================================================
# CXP Stream Channel
# =============================================================================


@dataclass
class CXPStreamPacket:
    """CoaXPress stream packet structure."""

    stream_id: int = 0
    packet_type: int = CXPMessageType.DATA_STREAM
    packet_id: int = 0
    payload: bytes = b""
    timestamp: float = 0.0


class CXPStreamChannel:
    """
    CoaXPress stream channel for downlink data transfer.

    Handles image data streaming from camera to host.
    """

    def __init__(self, packet_size: int = 8192) -> None:
        """
        Initialize stream channel.

        Args:
            packet_size: Maximum packet size in bytes
        """
        self.packet_size = packet_size
        self._stream_active = False
        self._packet_counter = 0
        self._frame_counter = 0

        # Statistics
        self.stats = {"packets_received": 0, "frames_received": 0, "bytes_received": 0, "crc_errors": 0, "dropped_packets": 0}

    def start_stream(self) -> bool:
        """Start streaming."""
        self._stream_active = True
        self._packet_counter = 0
        logger.debug("CXP stream channel started")
        return True

    def stop_stream(self) -> bool:
        """Stop streaming."""
        self._stream_active = False
        logger.debug("CXP stream channel stopped")
        return True

    def receive_packet(self) -> Optional[CXPStreamPacket]:
        """Receive stream packet (simulation)."""
        if not self._stream_active:
            return None

        self._packet_counter += 1
        self.stats["packets_received"] += 1

        packet = CXPStreamPacket(
            stream_id=0, packet_id=self._packet_counter, payload=bytes(self.packet_size), timestamp=time.time()
        )

        self.stats["bytes_received"] += len(packet.payload)
        return packet

    def receive_frame(self, frame_size: int) -> bytes:
        """
        Receive complete frame.

        Args:
            frame_size: Expected frame size in bytes

        Returns:
            Frame data
        """
        if not self._stream_active:
            raise DataTransferError("Stream not active")

        # Calculate packets needed
        num_packets = (frame_size + self.packet_size - 1) // self.packet_size

        # Receive all packets
        frame_data = bytearray(frame_size)
        offset = 0

        for _ in range(num_packets):
            packet = self.receive_packet()
            if packet:
                chunk_size = min(self.packet_size, frame_size - offset)
                frame_data[offset : offset + chunk_size] = packet.payload[:chunk_size]
                offset += chunk_size

        self._frame_counter += 1
        self.stats["frames_received"] += 1

        return bytes(frame_data)


# =============================================================================
# CXP Trigger Controller
# =============================================================================


@dataclass
class TriggerEvent:
    """Trigger event record."""

    timestamp: float
    source: str
    edge: str
    frame_id: int


class CXPTriggerController:
    """
    CoaXPress trigger controller.

    Handles hardware and software triggers with precise timing.
    """

    def __init__(self) -> None:
        """Initialize trigger controller."""
        self._mode = CXPTriggerMode.DISABLED
        self._source = "Software"
        self._activation = "RisingEdge"
        self._delay_us = 0.0
        self._divider = 1
        self._multiplier = 1

        self._trigger_count = 0
        self._last_trigger_time: Optional[float] = None
        self._trigger_events: list[TriggerEvent] = []

        # GPIO line states
        self._gpio_lines: dict[str, bool] = {"Line0": False, "Line1": False, "Line2": False}

    def configure(
        self, mode: CXPTriggerMode, source: str = "Software", activation: str = "RisingEdge", delay_us: float = 0.0
    ) -> None:
        """
        Configure trigger settings.

        Args:
            mode: Trigger mode
            source: Trigger source (Software, Line0, Line1, Line2)
            activation: Edge activation (RisingEdge, FallingEdge, AnyEdge)
            delay_us: Trigger delay in microseconds
        """
        self._mode = mode
        self._source = source
        self._activation = activation
        self._delay_us = delay_us

        logger.debug(f"Trigger configured: mode={mode.value}, source={source}, activation={activation}")

    def software_trigger(self) -> bool:
        """Execute software trigger."""
        if self._mode == CXPTriggerMode.DISABLED:
            logger.warning("Trigger disabled, ignoring software trigger")
            return False

        if self._mode != CXPTriggerMode.SOFTWARE:
            logger.warning("Not in software trigger mode")
            return False

        # Apply trigger delay
        if self._delay_us > 0:
            time.sleep(self._delay_us / 1_000_000)

        self._trigger_count += 1
        self._last_trigger_time = time.time()

        event = TriggerEvent(timestamp=self._last_trigger_time, source="Software", edge="Rising", frame_id=self._trigger_count)
        self._trigger_events.append(event)

        logger.debug(f"Software trigger executed: frame_id={self._trigger_count}")
        return True

    def set_gpio_line(self, line: str, state: bool) -> None:
        """Set GPIO line state."""
        if line in self._gpio_lines:
            old_state = self._gpio_lines[line]
            self._gpio_lines[line] = state

            # Check for edge trigger
            if self._source == line:
                if self._activation == "RisingEdge" and not old_state and state:
                    self._execute_hardware_trigger(line)
                elif self._activation == "FallingEdge" and old_state and not state:
                    self._execute_hardware_trigger(line)
                elif self._activation == "AnyEdge" and old_state != state:
                    self._execute_hardware_trigger(line)

    def _execute_hardware_trigger(self, source: str) -> None:
        """Execute hardware trigger from GPIO line."""
        if self._delay_us > 0:
            time.sleep(self._delay_us / 1_000_000)

        self._trigger_count += 1
        self._last_trigger_time = time.time()

        event = TriggerEvent(
            timestamp=self._last_trigger_time, source=source, edge=self._activation, frame_id=self._trigger_count
        )
        self._trigger_events.append(event)

        logger.debug(f"Hardware trigger from {source}: frame_id={self._trigger_count}")

    def get_trigger_count(self) -> int:
        """Get total trigger count."""
        return self._trigger_count

    def get_last_trigger_time(self) -> Optional[float]:
        """Get timestamp of last trigger."""
        return self._last_trigger_time

    def reset(self) -> None:
        """Reset trigger state."""
        self._trigger_count = 0
        self._last_trigger_time = None
        self._trigger_events.clear()


# =============================================================================
# Power over CoaXPress (PoCXP) Controller
# =============================================================================


@dataclass
class PoCXPStatus:
    """Power over CoaXPress status."""

    enabled: bool = True
    voltage_v: float = 24.0
    current_ma: float = 0.0
    max_current_ma: float = 3000.0
    power_w: float = 0.0
    status: str = "OK"
    temperature_c: float = 25.0


class PoCXPController:
    """
    Power over CoaXPress controller.

    Manages power delivery over coaxial connections.
    """

    def __init__(self, num_connections: int = 1) -> None:
        """Initialize PoCXP controller."""
        self.num_connections = num_connections
        self._enabled = False
        self._voltage = 24.0  # V
        self._current_per_link = 0.0  # mA
        self._max_current_total = 3000.0  # 3A max per CXP spec

        # Per-link current limits
        self._link_current_limits = [750.0] * num_connections  # 750mA per link

    def enable(self) -> bool:
        """Enable power delivery."""
        self._enabled = True
        self._current_per_link = 100.0  # Base current draw
        logger.info(f"PoCXP enabled: {self._voltage}V across {self.num_connections} links")
        return True

    def disable(self) -> None:
        """Disable power delivery."""
        self._enabled = False
        self._current_per_link = 0.0
        logger.info("PoCXP disabled")

    def get_status(self) -> PoCXPStatus:
        """Get power status."""
        total_current = self._current_per_link * self.num_connections
        power = (self._voltage * total_current) / 1000  # Convert to W

        return PoCXPStatus(
            enabled=self._enabled,
            voltage_v=self._voltage,
            current_ma=total_current,
            max_current_ma=self._max_current_total,
            power_w=power,
            status="OK" if self._enabled else "Disabled",
            temperature_c=25.0 + (power * 2),  # Simple thermal model
        )

    def set_current(self, current_ma: float) -> None:
        """Set current draw (simulation)."""
        self._current_per_link = min(current_ma / self.num_connections, *self._link_current_limits)


# =============================================================================
# GenICam Node Map for CoaXPress
# =============================================================================


@dataclass
class CXPFeatureNode:
    """CoaXPress GenICam feature node."""

    name: str
    value: Any
    min_value: Any = None
    max_value: Any = None
    access_mode: str = "RW"
    enum_entries: list[str] = field(default_factory=list)

    def set_value(self, value: Any) -> bool:
        """Set value with validation."""
        if self.access_mode == "RO":
            return False

        if self.min_value is not None and value < self.min_value:
            raise ValueError(f"{self.name}: value {value} < minimum {self.min_value}")
        if self.max_value is not None and value > self.max_value:
            raise ValueError(f"{self.name}: value {value} > maximum {self.max_value}")
        if self.enum_entries and value not in self.enum_entries:
            raise ValueError(f"{self.name}: invalid value '{value}'")

        self.value = value
        return True


class CXPNodeMap:
    """GenICam node map for CoaXPress cameras."""

    def __init__(self) -> None:
        """Initialize node map with SFNC features."""
        self._nodes: dict[str, CXPFeatureNode] = {}
        self._initialize_features()

    def _initialize_features(self) -> None:
        """Initialize standard features."""
        # Device info
        self._nodes["DeviceVendorName"] = CXPFeatureNode("DeviceVendorName", "Advanced Image Sensor Interface", access_mode="RO")
        self._nodes["DeviceModelName"] = CXPFeatureNode("DeviceModelName", "CoaXPress Camera Simulator", access_mode="RO")
        self._nodes["DeviceSerialNumber"] = CXPFeatureNode("DeviceSerialNumber", "CXP-SIM-001", access_mode="RO")

        # Image format
        self._nodes["Width"] = CXPFeatureNode("Width", 2048, min_value=1, max_value=16384)
        self._nodes["Height"] = CXPFeatureNode("Height", 2048, min_value=1, max_value=16384)
        self._nodes["PixelFormat"] = CXPFeatureNode(
            "PixelFormat",
            "Mono16",
            enum_entries=["Mono8", "Mono10", "Mono12", "Mono16", "BayerGR8", "BayerRG8", "BayerGB8", "BayerBG8", "RGB8"],
        )

        # Acquisition
        self._nodes["AcquisitionMode"] = CXPFeatureNode(
            "AcquisitionMode", "Continuous", enum_entries=["SingleFrame", "MultiFrame", "Continuous"]
        )
        self._nodes["AcquisitionFrameRate"] = CXPFeatureNode("AcquisitionFrameRate", 30.0, min_value=0.1, max_value=10000.0)
        self._nodes["ExposureTime"] = CXPFeatureNode("ExposureTime", 10000.0, min_value=1.0, max_value=10000000.0)

        # Analog
        self._nodes["Gain"] = CXPFeatureNode("Gain", 0.0, min_value=0.0, max_value=48.0)
        self._nodes["BlackLevel"] = CXPFeatureNode("BlackLevel", 0.0, min_value=0.0, max_value=65535.0)

        # Trigger
        self._nodes["TriggerMode"] = CXPFeatureNode("TriggerMode", "Off", enum_entries=["Off", "On"])
        self._nodes["TriggerSource"] = CXPFeatureNode(
            "TriggerSource", "Software", enum_entries=["Software", "Line0", "Line1", "Line2"]
        )
        self._nodes["TriggerActivation"] = CXPFeatureNode(
            "TriggerActivation", "RisingEdge", enum_entries=["RisingEdge", "FallingEdge", "AnyEdge"]
        )

        # CoaXPress specific
        self._nodes["CxpLinkConfiguration"] = CXPFeatureNode(
            "CxpLinkConfiguration",
            "CXP6_X1",
            enum_entries=[
                "CXP1_X1",
                "CXP2_X1",
                "CXP3_X1",
                "CXP5_X1",
                "CXP6_X1",
                "CXP6_X2",
                "CXP6_X4",
                "CXP10_X1",
                "CXP10_X2",
                "CXP12_X1",
            ],
        )
        self._nodes["CxpConnectionSelector"] = CXPFeatureNode("CxpConnectionSelector", 0, min_value=0, max_value=3)
        self._nodes["CxpConnectionSpeed"] = CXPFeatureNode("CxpConnectionSpeed", 6250.0, access_mode="RO")

    def get_value(self, name: str) -> Any:
        """Get feature value."""
        if name not in self._nodes:
            raise ValueError(f"Unknown feature: {name}")
        return self._nodes[name].value

    def set_value(self, name: str, value: Any) -> bool:
        """Set feature value."""
        if name not in self._nodes:
            raise ValueError(f"Unknown feature: {name}")
        return self._nodes[name].set_value(value)

    def get_all_features(self) -> dict[str, Any]:
        """Get all feature values."""
        return {name: node.value for name, node in self._nodes.items()}


# =============================================================================
# CoaXPress Configuration
# =============================================================================


@dataclass
class CoaXPressConfig:
    """
    Configuration for CoaXPress protocol.

    All parameters validated on initialization.
    """

    speed_grade: str = "CXP-6"
    connections: int = 1
    packet_size: int = 8192
    trigger_mode: str = "software"
    pixel_format: str = "Mono16"
    resolution: tuple[int, int] = (2048, 2048)
    frame_rate: float = 30.0
    power_over_coax: bool = True
    discovery_timeout: float = 5.0

    # Advanced settings
    master_host_connection: int = 0
    packet_delay: int = 0
    stream_packet_size: int = 0

    def __post_init__(self) -> None:
        """Validate configuration."""
        valid_speeds = ["CXP-1", "CXP-2", "CXP-3", "CXP-5", "CXP-6", "CXP-10", "CXP-12"]
        if self.speed_grade not in valid_speeds:
            raise ValueError(f"Invalid speed grade: {self.speed_grade}")

        if not 1 <= self.connections <= 4:
            raise ValueError("Connections must be between 1 and 4")

        if self.packet_size < 64 or self.packet_size > 65536:
            raise ValueError("Packet size must be between 64 and 65536 bytes")


# =============================================================================
# CoaXPress Driver
# =============================================================================


class CoaXPressDriver(StreamingProtocolBase):
    """
    CoaXPress protocol driver with complete implementation.

    Features:
    - CXP 2.1 compliant with CXP-1 to CXP-12 support
    - Multi-connection link aggregation
    - GenICam/SFNC feature control
    - Hardware and software trigger
    - Power over Coax management
    """

    SPEED_GRADES: ClassVar[dict[str, int]] = {
        "CXP-1": 1250,
        "CXP-2": 2500,
        "CXP-3": 3125,
        "CXP-5": 5000,
        "CXP-6": 6250,
        "CXP-10": 10000,
        "CXP-12": 12500,
    }

    def __init__(self, config: CoaXPressConfig) -> None:
        """Initialize CoaXPress driver."""
        self.cxp_config = config
        super().__init__(config.__dict__)

        # Protocol components
        self._link_layer = CXPLinkLayer(config.connections)
        self._control_channel = CXPControlChannel()
        self._stream_channel = CXPStreamChannel(config.packet_size)
        self._trigger_controller = CXPTriggerController()
        self._pocxp = PoCXPController(config.connections)
        self._node_map = CXPNodeMap()

        # State
        self.device_handle: Optional[str] = None
        self.stream_handle: Optional[str] = None
        self._frame_count = 0
        self._start_time: Optional[float] = None

        # Statistics
        self.stats = {
            "frames_captured": 0,
            "frames_dropped": 0,
            "bytes_transferred": 0,
            "errors": 0,
            "last_frame_time": 0.0,
            "trigger_count": 0,
        }

        # Apply config
        self._apply_config()

        logger.info(f"CoaXPress driver initialized with {config.speed_grade} at {config.connections} connections")

    def _apply_config(self) -> None:
        """Apply configuration to components."""
        self._node_map.set_value("Width", self.cxp_config.resolution[0])
        self._node_map.set_value("Height", self.cxp_config.resolution[1])
        self._node_map.set_value("PixelFormat", self.cxp_config.pixel_format)
        self._node_map.set_value("AcquisitionFrameRate", self.cxp_config.frame_rate)

        # Configure trigger
        if self.cxp_config.trigger_mode == "software":
            self._trigger_controller.configure(CXPTriggerMode.SOFTWARE)
            self._node_map.set_value("TriggerMode", "On")
            self._node_map.set_value("TriggerSource", "Software")
        elif self.cxp_config.trigger_mode == "hardware":
            self._trigger_controller.configure(CXPTriggerMode.HARDWARE_LINE0, "Line0")
            self._node_map.set_value("TriggerMode", "On")
            self._node_map.set_value("TriggerSource", "Line0")
        else:
            self._trigger_controller.configure(CXPTriggerMode.DISABLED)
            self._node_map.set_value("TriggerMode", "Off")

    def _get_capabilities(self) -> ProtocolCapabilities:
        """Get protocol capabilities."""
        max_bandwidth = self.SPEED_GRADES[self.cxp_config.speed_grade] * self.cxp_config.connections

        return ProtocolCapabilities(
            max_bandwidth_gbps=max_bandwidth / 1000.0,
            max_distance_m=100.0,
            power_over_cable=True,
            hot_pluggable=False,
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
                "RGB8",
                "BGR8",
                "YUV422",
            ],
            supported_resolutions=[
                (640, 480),
                (800, 600),
                (1024, 768),
                (1280, 1024),
                (1600, 1200),
                (2048, 1536),
                (2048, 2048),
                (4096, 4096),
                (8192, 8192),
            ],
        )

    def connect(self) -> bool:
        """Establish connection to CoaXPress device."""
        try:
            logger.info("Connecting to CoaXPress device...")

            # Enable power if configured
            if self.cxp_config.power_over_coax:
                self._pocxp.enable()
                time.sleep(0.1)  # Power stabilization

            # Negotiate link speed
            if not self._link_layer.negotiate_link(self.cxp_config.speed_grade):
                raise ConnectionError("Link negotiation failed")

            # Read device identification
            self._read_device_identification()

            # Create device handle
            self.device_handle = f"cxp_device_{id(self)}"

            # Configure device
            self._configure_device()

            # Update status
            self.is_connected = True
            self.status.is_connected = True
            self.status.connection_quality = 1.0
            self.status.data_rate_mbps = self._link_layer.get_total_bandwidth()

            logger.info("CoaXPress device connected successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to connect to CoaXPress device: {e}")
            self.status.error_count += 1
            self.status.last_error = str(e)
            raise ConnectionError(f"CoaXPress connection failed: {e}")

    def disconnect(self) -> bool:
        """Disconnect from CoaXPress device."""
        try:
            if self.is_streaming:
                self.stop_streaming()

            # Disconnect link layer
            self._link_layer.disconnect()

            # Disable power
            self._pocxp.disable()

            # Clear handles
            self.device_handle = None
            self.stream_handle = None

            # Update status
            self.is_connected = False
            self.status.is_connected = False
            self.status.connection_quality = 0.0
            self.status.data_rate_mbps = 0.0

            logger.info("CoaXPress device disconnected")
            return True

        except Exception as e:
            logger.error(f"Error during CoaXPress disconnect: {e}")
            return False

    def send_data(self, data: bytes) -> bool:
        """Send control data."""
        if not self.is_connected:
            raise ConnectionError("Not connected to CoaXPress device")

        try:
            # Write via control channel
            time.sleep(0.001)
            self.status.bytes_transmitted += len(data)
            logger.debug(f"Sent {len(data)} bytes to CoaXPress device")
            return True

        except Exception as e:
            logger.error(f"Failed to send data: {e}")
            self.status.error_count += 1
            self.stats["errors"] += 1
            raise DataTransferError(f"CoaXPress send failed: {e}")

    def receive_data(self, size: int) -> Optional[bytes]:
        """Receive control data."""
        if not self.is_connected:
            raise ConnectionError("Not connected to CoaXPress device")

        try:
            data = self._control_channel.read_register(0, size)
            self.status.bytes_received += len(data)
            logger.debug(f"Received {len(data)} bytes from CoaXPress device")
            return data

        except Exception as e:
            logger.error(f"Failed to receive data: {e}")
            self.status.error_count += 1
            raise DataTransferError(f"CoaXPress receive failed: {e}")

    def read_register(self, address: int, length: int = 4) -> bytes:
        """Read from device register."""
        if not self.is_connected:
            raise ConnectionError("Not connected")
        return self._control_channel.read_register(address, length)

    def write_register(self, address: int, data: bytes) -> bool:
        """Write to device register."""
        if not self.is_connected:
            raise ConnectionError("Not connected")
        return self._control_channel.write_register(address, data)

    def start_streaming(self) -> bool:
        """Start image streaming."""
        if not self.is_connected:
            raise ConnectionError("Not connected to CoaXPress device")

        try:
            logger.info("Starting CoaXPress streaming...")

            # Start stream channel
            self._stream_channel.start_stream()

            # Create stream handle
            self.stream_handle = f"cxp_stream_{id(self)}"

            self.is_streaming = True
            self._start_time = time.time()
            self._frame_count = 0

            logger.info("CoaXPress streaming started")
            return True

        except Exception as e:
            logger.error(f"Failed to start streaming: {e}")
            self.status.error_count += 1
            return False

    def stop_streaming(self) -> bool:
        """Stop image streaming."""
        try:
            if self.is_streaming:
                self._stream_channel.stop_stream()
                self.is_streaming = False
                self.stream_handle = None

                if self._start_time:
                    duration = time.time() - self._start_time
                    avg_fps = self._frame_count / duration if duration > 0 else 0
                    logger.info(f"Streaming stopped. Captured {self._frame_count} frames at {avg_fps:.2f} FPS")

            return True

        except Exception as e:
            logger.error(f"Error stopping streaming: {e}")
            return False

    def capture_frame(self) -> Optional[bytes]:
        """Capture single frame."""
        if not self.is_connected:
            raise ConnectionError("Not connected to CoaXPress device")

        try:
            width = self._node_map.get_value("Width")
            height = self._node_map.get_value("Height")
            bpp = self._get_bytes_per_pixel(self._node_map.get_value("PixelFormat"))
            _ = width * height * bpp  # Frame size for reference

            # Simulate capture timing
            frame_rate = self._node_map.get_value("AcquisitionFrameRate")
            capture_delay = 1.0 / frame_rate
            time.sleep(min(capture_delay, 0.1))

            # Generate frame
            frame_data = self._generate_frame(width, height, bpp)

            # Update statistics
            self._frame_count += 1
            self.stats["frames_captured"] += 1
            self.stats["bytes_transferred"] += len(frame_data)
            self.stats["last_frame_time"] = time.time()

            logger.debug(f"Captured frame {self._frame_count}, size: {len(frame_data)} bytes")
            return frame_data

        except Exception as e:
            logger.error(f"Frame capture failed: {e}")
            self.stats["errors"] += 1
            self.status.error_count += 1
            return None

    def software_trigger(self) -> bool:
        """Execute software trigger."""
        if self._trigger_controller.software_trigger():
            self.stats["trigger_count"] += 1
            return True
        return False

    def set_gpio_line(self, line: str, state: bool) -> None:
        """Set GPIO line for hardware trigger."""
        self._trigger_controller.set_gpio_line(line, state)

    def get_feature(self, name: str) -> Any:
        """Get GenICam feature value."""
        return self._node_map.get_value(name)

    def set_feature(self, name: str, value: Any) -> bool:
        """Set GenICam feature value."""
        try:
            return self._node_map.set_value(name, value)
        except ValueError as e:
            raise ConfigurationError(str(e))

    def get_pocxp_status(self) -> PoCXPStatus:
        """Get Power over CoaXPress status."""
        return self._pocxp.get_status()

    def get_link_status(self, link_id: int = 0) -> CXPLinkStatus:
        """Get link status."""
        return self._link_layer.get_link_status(link_id)

    def _read_device_identification(self) -> None:
        """Read device identification."""
        # Read from bootstrap registers
        manufacturer = self._control_channel.read_string(CXPRegisterAddress.MANUFACTURER_NAME)
        model = self._control_channel.read_string(CXPRegisterAddress.MODEL_NAME)
        serial = self._control_channel.read_string(CXPRegisterAddress.SERIAL_NUMBER)
        version = self._control_channel.read_string(CXPRegisterAddress.DEVICE_VERSION)

        logger.debug(f"Device: {manufacturer} {model} ({serial}), firmware: {version}")

    def _configure_device(self) -> None:
        """Configure device parameters."""
        config_params = {
            "Width": self.cxp_config.resolution[0],
            "Height": self.cxp_config.resolution[1],
            "PixelFormat": self.cxp_config.pixel_format,
            "AcquisitionFrameRate": self.cxp_config.frame_rate,
        }

        for name, value in config_params.items():
            try:
                self._node_map.set_value(name, value)
            except Exception as e:
                logger.warning(f"Could not set {name}: {e}")

        logger.debug(f"Configured CoaXPress device with parameters: {config_params}")

    def _get_bytes_per_pixel(self, pixel_format: str) -> int:
        """Get bytes per pixel using shared base implementation."""
        return self._get_bytes_per_pixel_common(pixel_format)

    def _generate_frame(self, width: int, height: int, bpp: int) -> bytes:
        """Generate frame data using shared vectorized implementation."""
        return self._generate_test_frame_vectorized(width, height, bpp, self._frame_count)

    def _get_current_frame_rate(self) -> float:
        """Get current frame rate."""
        if self._start_time and self._frame_count > 0:
            elapsed = time.time() - self._start_time
            return self._frame_count / elapsed if elapsed > 0 else 0.0
        return 0.0

    def _get_dropped_frame_count(self) -> int:
        """Get number of dropped frames."""
        return int(self.stats["frames_dropped"])

    def _get_buffer_utilization(self) -> float:
        """Estimate buffer utilization from achieved frame rate."""
        target_fps = self.cxp_config.frame_rate
        actual_fps = self._get_current_frame_rate()
        if target_fps > 0:
            return min(100.0, (actual_fps / target_fps) * 100.0)
        return 0.0

    def get_status(self) -> dict[str, Any]:
        """Get CoaXPress protocol status (legacy compatibility)."""
        return {
            "protocol": "CoaXPress",
            "connected": self.is_connected,
            "streaming": self.is_streaming,
            "config": {
                "speed_grade": self.cxp_config.speed_grade,
                "connections": self.cxp_config.connections,
                "packet_size": self.cxp_config.packet_size,
                "pixel_format": self.cxp_config.pixel_format,
                "resolution": self.cxp_config.resolution,
                "frame_rate": self.cxp_config.frame_rate,
            },
            "statistics": self.get_statistics(),
        }

    def get_device_info(self) -> dict[str, Any]:
        """Get device information."""
        pocxp = self._pocxp.get_status()
        return {
            "protocol": "CoaXPress",
            "speed_grade": self.cxp_config.speed_grade,
            "connections": self.cxp_config.connections,
            "max_bandwidth_mbps": self._link_layer.get_total_bandwidth(),
            "power_over_coax": pocxp.enabled,
            "pocxp_voltage_v": pocxp.voltage_v,
            "pocxp_current_ma": pocxp.current_ma,
            "pixel_format": self._node_map.get_value("PixelFormat"),
            "resolution": (self._node_map.get_value("Width"), self._node_map.get_value("Height")),
            "frame_rate": self._node_map.get_value("AcquisitionFrameRate"),
        }

    def get_statistics(self) -> dict[str, Any]:
        """Get driver statistics."""
        uptime = time.time() - self._start_time if self._start_time else 0

        return {
            **self.stats,
            **self._control_channel.stats,
            **self._stream_channel.stats,
            "uptime_seconds": uptime,
            "current_fps": self._frame_count / uptime if uptime > 0 else 0,
            "connection_quality": self.status.connection_quality,
            "link_bandwidth_mbps": self._link_layer.get_total_bandwidth(),
            "error_rate": self.stats["errors"] / max(1, uptime),
        }
