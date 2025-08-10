"""
Type definitions and annotations for the Advanced Image Sensor Interface.

This module provides comprehensive type definitions to improve type safety
and mypy compliance across the entire project.
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
