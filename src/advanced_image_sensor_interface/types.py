"""
Type definitions and annotations for the Advanced Image Sensor Interface.

This module provides shared type definitions to improve type safety across
maintained modules in the project.
"""

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Literal, Optional, Protocol, TypedDict, TypeVar, Union

import numpy as np
from numpy.typing import NDArray

# NumPy array type aliases for better readability
ImageArray = NDArray[np.uint8]
FloatArray = NDArray[np.floating[Any]]
IntArray = NDArray[np.integer[Any]]
GenericArray = NDArray[Any]

# Common type aliases
Milliseconds = float
Seconds = float
Bytes = int
Megabytes = float
Percentage = float
FrameRate = float
Voltage = float
Current = float
Power = float
Temperature = float

# Image processing types
Resolution = tuple[int, int]
ColorMatrix = NDArray[np.floating[Any]]
BayerPattern = Literal["RGGB", "BGGR", "GRBG", "GBRG"]
PixelFormat = Literal[
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
]

# Protocol types
ProtocolType = Literal["MIPI", "CoaXPress", "GigE", "USB3", "CameraLink"]
TriggerMode = Literal["software", "hardware", "continuous"]
StreamingState = Literal["stopped", "starting", "streaming", "stopping", "error"]

# Power management types
PowerRailName = str
PowerBackendType = Literal["simulation", "hardware", "i2c", "gpio", "sysfs"]

# Buffer management types
BufferSize = int
BufferId = str
PoolName = str


# Configuration dictionaries
@dataclass
class TimingConfig:
    """Timing configuration parameters."""

    frame_timeout_ms: Milliseconds
    exposure_time_us: float
    readout_time_us: float
    trigger_delay_us: float


@dataclass
class SecurityConfig:
    """Security configuration parameters."""

    max_image_size_mb: Megabytes
    max_buffer_size_mb: Megabytes
    operation_timeout_s: Seconds
    enable_validation: bool


# Protocol interfaces
class ProtocolInterface(Protocol):
    """Protocol interface for camera communication."""

    def connect(self) -> bool: ...
    def disconnect(self) -> bool: ...
    def send_data(self, data: bytes) -> bool: ...
    def receive_data(self, size: int) -> Optional[bytes]: ...
    def get_status(self) -> dict[str, Any]: ...


class StreamingProtocolInterface(ProtocolInterface, Protocol):
    """Extended protocol interface for streaming cameras."""

    def start_streaming(self) -> bool: ...
    def stop_streaming(self) -> bool: ...
    def capture_frame(self) -> Optional[bytes]: ...


# Buffer management interfaces
class BufferManagerInterface(Protocol):
    """Interface for buffer management systems."""

    def get_buffer(self, size: BufferSize) -> Optional[memoryview]: ...
    def return_buffer(self, buffer: memoryview) -> bool: ...
    def get_statistics(self) -> dict[str, Any]: ...


class AsyncBufferManagerInterface(Protocol):
    """Async interface for buffer management systems."""

    async def get_buffer_async(self, size: BufferSize) -> Optional[memoryview]: ...
    async def return_buffer_async(self, buffer: memoryview) -> bool: ...
    async def optimize_pools_async(self) -> None: ...


# Power management interfaces
class PowerBackendInterface(Protocol):
    """Interface for power management backends."""

    def initialize(self) -> bool: ...
    def shutdown(self) -> bool: ...
    def set_voltage(self, rail: PowerRailName, voltage: Voltage) -> bool: ...
    def get_voltage(self, rail: PowerRailName) -> Optional[Voltage]: ...
    def get_current(self, rail: PowerRailName) -> Optional[Current]: ...


# Security interfaces
class SecurityManagerInterface(Protocol):
    """Interface for security management."""

    def validate_image(self, image: ImageArray) -> bool: ...
    def validate_buffer_size(self, size: BufferSize) -> bool: ...
    def start_operation(self, operation_id: str) -> bool: ...
    def end_operation(self, operation_id: str) -> bool: ...


# Image processing types
class ImageProcessor(Protocol):
    """Protocol for image processing operations."""

    def process_frame(self, frame: ImageArray) -> ImageArray: ...
    def apply_noise_reduction(self, frame: ImageArray, strength: float) -> ImageArray: ...
    def apply_color_correction(self, frame: ImageArray, matrix: ColorMatrix) -> ImageArray: ...


# Callback types
FrameCallback = Callable[[ImageArray], None]
AsyncFrameCallback = Callable[[ImageArray], Awaitable[None]]
ErrorCallback = Callable[[Exception], None]
StatusCallback = Callable[[dict[str, Any]], None]

# Configuration types
ConfigDict = dict[str, Any]
ParameterDict = dict[str, Union[str, int, float, bool]]


# Statistics and metrics
class PerformanceMetrics(TypedDict):
    """Performance metrics structure."""

    frames_per_second: FrameRate
    dropped_frames: int
    buffer_utilization: Percentage
    memory_usage_mb: Megabytes
    cpu_usage_percent: Percentage
    temperature_c: Temperature


class PowerMetrics(TypedDict):
    """Power consumption metrics."""

    total_power_w: Power
    rail_voltages: dict[PowerRailName, Voltage]
    rail_currents: dict[PowerRailName, Current]
    efficiency_percent: Percentage
    temperature_c: Temperature


# Error types
class SensorError(Exception):
    """Base exception for sensor-related errors."""

    pass


class ProtocolError(SensorError):
    """Exception for protocol communication errors."""

    pass


class BufferError(SensorError):
    """Exception for buffer management errors."""

    pass


class PowerError(SensorError):
    """Exception for power management errors."""

    pass


class SecurityError(SensorError):
    """Exception for security validation errors."""

    pass


# Generic type variables
T = TypeVar("T")
P = TypeVar("P", bound=ProtocolInterface)
B = TypeVar("B", bound=BufferManagerInterface)
S = TypeVar("S", bound=SecurityManagerInterface)

# Factory types
ProtocolFactory = Callable[[ConfigDict], ProtocolInterface]
BufferManagerFactory = Callable[[ConfigDict], BufferManagerInterface]
PowerBackendFactory = Callable[[ConfigDict], PowerBackendInterface]

# Async types
AsyncTask = asyncio.Task[Any]
AsyncQueue = asyncio.Queue[Any]
AsyncEvent = asyncio.Event
AsyncLock = asyncio.Lock


# Hardware abstraction types
class HardwareInterface(Protocol):
    """Generic hardware interface protocol."""

    def initialize(self) -> bool: ...
    def shutdown(self) -> bool: ...
    def read_register(self, address: int) -> int: ...
    def write_register(self, address: int, value: int) -> bool: ...


# Validation types
ValidationResult = tuple[bool, Optional[str]]  # (is_valid, error_message)
ValidationCallback = Callable[[Any], ValidationResult]

# Logging types
LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
LogMessage = str
LogContext = dict[str, Any]


# Processing Result Types (v3.2.0+)
@dataclass
class ProcessingMetrics:
    """Metrics for processing operations."""

    processing_time_ms: float = 0.0
    memory_used_mb: float = 0.0
    algorithm_name: str = ""
    parameters: dict[str, Any] = None

    def __post_init__(self):
        if self.parameters is None:
            self.parameters = {}


@dataclass
class SignalProcessingResult:
    """Result of signal processing operation."""

    success: bool
    data: Optional[np.ndarray] = None
    error: Optional[str] = None
    warnings: list[str] = None
    metrics: ProcessingMetrics = None
    snr_improvement_db: float = 0.0
    noise_reduction_applied: bool = False
    dynamic_range_expanded: bool = False
    color_correction_applied: bool = False

    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []
        if self.metrics is None:
            self.metrics = ProcessingMetrics()


@dataclass
class HDRProcessingResult:
    """Result of HDR processing operation."""

    success: bool
    data: Optional[np.ndarray] = None
    error: Optional[str] = None
    warnings: list[str] = None
    metrics: ProcessingMetrics = None
    tone_mapping_algorithm: str = ""
    exposure_fusion_used: bool = False
    ghost_reduction_applied: bool = False
    alignment_quality: float = 0.0

    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []
        if self.metrics is None:
            self.metrics = ProcessingMetrics()


@dataclass
class RAWProcessingResult:
    """Result of RAW processing operation."""

    success: bool
    data: Optional[np.ndarray] = None
    error: Optional[str] = None
    warnings: list[str] = None
    metrics: ProcessingMetrics = None
    demosaicing_algorithm: str = ""
    white_balance_applied: bool = False
    bad_pixel_correction: int = 0
    vignetting_corrected: bool = False

    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []
        if self.metrics is None:
            self.metrics = ProcessingMetrics()


@dataclass
class LensCorrectionResult:
    """Result of lens correction operation."""

    success: bool
    data: Optional[np.ndarray] = None
    error: Optional[str] = None
    warnings: list[str] = None
    metrics: ProcessingMetrics = None
    pixels_corrected: int = 0
    max_displacement: float = 0.0
    distortion_type: str = ""
    radial_correction_applied: bool = False
    tangential_correction_applied: bool = False

    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []
        if self.metrics is None:
            self.metrics = ProcessingMetrics()


@dataclass
class DepthResult:
    """Result of stereo depth processing operation (v3.2.0)."""

    success: bool
    disparity_map: Optional[np.ndarray] = None
    depth_map: Optional[np.ndarray] = None
    point_cloud: Optional[np.ndarray] = None
    error: Optional[str] = None
    warnings: list[str] = None
    metrics: ProcessingMetrics = None
    algorithm_used: str = ""
    valid_pixel_ratio: float = 0.0

    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []
        if self.metrics is None:
            self.metrics = ProcessingMetrics()


# Union type for all processing results
ProcessingResult = Union[SignalProcessingResult, HDRProcessingResult, RAWProcessingResult, LensCorrectionResult, DepthResult]


# Simulation Delay Configuration (v3.2.0+)
@dataclass
class SimulationDelayConfig:
    """Configuration for simulation delays in protocol drivers.

    Allows configurable delays for various operations to simulate
    realistic hardware behavior without hardcoded sleep calls.
    """

    # Connection delays (seconds)
    device_discovery_delay: float = 0.1
    link_initialization_delay: float = 0.05
    link_training_delay: float = 0.02
    device_enumeration_delay: float = 0.03

    # Streaming delays (seconds)
    streaming_setup_delay: float = 0.01
    streaming_teardown_delay: float = 0.01
    frame_capture_delay: float = 0.001
    buffer_allocation_delay: float = 0.005

    # Control transfer delays (seconds)
    register_read_delay: float = 0.0001
    register_write_delay: float = 0.0002
    command_transfer_delay: float = 0.0005

    # Power management delays (seconds)
    power_state_transition_delay: float = 0.01
    power_on_delay: float = 0.1
    power_off_delay: float = 0.05

    # Security/authentication delays (seconds)
    authentication_delay: float = 0.01
    key_exchange_delay: float = 0.02
    session_creation_delay: float = 0.005

    # Error injection (for testing)
    enable_random_delays: bool = False
    random_delay_range_ms: tuple[float, float] = (0.0, 5.0)

    def __post_init__(self):
        """Validate configuration."""
        for field_name, value in self.__dict__.items():
            if field_name.endswith("_delay") and value < 0:
                raise ValueError(f"{field_name} must be non-negative")

        if self.random_delay_range_ms[0] < 0 or self.random_delay_range_ms[1] < self.random_delay_range_ms[0]:
            raise ValueError("Invalid random_delay_range_ms")

    def get_random_delay(self) -> float:
        """Get random delay if enabled."""
        if not self.enable_random_delays:
            return 0.0
        import random

        min_ms, max_ms = self.random_delay_range_ms
        return random.uniform(min_ms, max_ms) / 1000.0

    def apply_delay(self, base_delay: float) -> float:
        """Apply delay with optional random component.

        Args:
            base_delay: Base delay in seconds

        Returns:
            Total delay in seconds (base + random if enabled)
        """
        return base_delay + self.get_random_delay()
