"""
Comprehensive exception hierarchy for the Advanced Image Sensor Interface.

Provides specific exception types for different error conditions with
detailed error information and recovery hints.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional


class ErrorSeverity(Enum):
    """Error severity levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Error categories for classification."""

    HARDWARE = "hardware"
    SOFTWARE = "software"
    NETWORK = "network"
    CONFIGURATION = "configuration"
    RESOURCE = "resource"
    SECURITY = "security"
    USER = "user"


@dataclass
class ErrorContext:
    """Additional context information for errors."""

    timestamp: float
    component: str
    operation: str
    parameters: dict[str, Any]
    system_state: dict[str, Any]
    recovery_hints: list[str]


class SensorError(Exception):
    """
    Base exception for all sensor-related errors.

    Provides comprehensive error information including severity,
    category, context, and recovery suggestions.
    """

    def __init__(
        self,
        message: str,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        category: ErrorCategory = ErrorCategory.SOFTWARE,
        error_code: Optional[str] = None,
        context: Optional[ErrorContext] = None,
        cause: Optional[Exception] = None,
    ):
        """Initialize sensor error."""
        super().__init__(message)
        self.message = message
        self.severity = severity
        self.category = category
        self.error_code = error_code
        self.context = context
        self.cause = cause

    def __str__(self) -> str:
        """String representation of the error."""
        parts = [f"{self.__class__.__name__}: {self.message}"]

        if self.error_code:
            parts.append(f"Code: {self.error_code}")

        parts.append(f"Severity: {self.severity.value}")
        parts.append(f"Category: {self.category.value}")

        if self.context:
            parts.append(f"Component: {self.context.component}")
            parts.append(f"Operation: {self.context.operation}")

        if self.cause:
            parts.append(f"Caused by: {self.cause}")

        return " | ".join(parts)

    def get_recovery_hints(self) -> list[str]:
        """Get recovery hints for this error."""
        if self.context and self.context.recovery_hints:
            return self.context.recovery_hints
        return []


class ProtocolError(SensorError):
    """Exception for protocol communication errors."""

    def __init__(self, message: str, protocol: str, **kwargs):
        """Initialize protocol error."""
        super().__init__(message, category=ErrorCategory.NETWORK, **kwargs)
        self.protocol = protocol


class ConnectionError(ProtocolError):
    """Exception for connection-related errors."""

    def __init__(self, message: str, protocol: str = "unknown", **kwargs):
        """Initialize connection error."""
        super().__init__(message, protocol=protocol, severity=ErrorSeverity.HIGH, **kwargs)


class DataTransferError(ProtocolError):
    """Exception for data transfer errors."""

    def __init__(self, message: str, protocol: str = "unknown", bytes_transferred: int = 0, **kwargs):
        """Initialize data transfer error."""
        super().__init__(message, protocol=protocol, **kwargs)
        self.bytes_transferred = bytes_transferred


class BufferError(SensorError):
    """Exception for buffer management errors."""

    def __init__(self, message: str, buffer_size: Optional[int] = None, **kwargs):
        """Initialize buffer error."""
        super().__init__(message, category=ErrorCategory.RESOURCE, **kwargs)
        self.buffer_size = buffer_size


class PowerError(SensorError):
    """Exception for power management errors."""

    def __init__(self, message: str, rail: Optional[str] = None, voltage: Optional[float] = None, **kwargs):
        """Initialize power error."""
        super().__init__(message, category=ErrorCategory.HARDWARE, severity=ErrorSeverity.HIGH, **kwargs)
        self.rail = rail
        self.voltage = voltage


class SecurityError(SensorError):
    """Exception for security validation errors."""

    def __init__(self, message: str, validation_type: Optional[str] = None, **kwargs):
        """Initialize security error."""
        super().__init__(message, category=ErrorCategory.SECURITY, severity=ErrorSeverity.HIGH, **kwargs)
        self.validation_type = validation_type


class ConfigurationError(SensorError):
    """Exception for configuration-related errors."""

    def __init__(self, message: str, parameter: Optional[str] = None, value: Optional[Any] = None, **kwargs):
        """Initialize configuration error."""
        super().__init__(message, category=ErrorCategory.CONFIGURATION, **kwargs)
        self.parameter = parameter
        self.value = value


class TimeoutError(SensorError):
    """Exception for timeout-related errors."""

    def __init__(self, message: str, timeout_seconds: Optional[float] = None, operation: Optional[str] = None, **kwargs):
        """Initialize timeout error."""
        super().__init__(message, severity=ErrorSeverity.MEDIUM, **kwargs)
        self.timeout_seconds = timeout_seconds
        self.operation = operation


class HardwareError(SensorError):
    """Exception for hardware-related errors."""

    def __init__(self, message: str, component: Optional[str] = None, **kwargs):
        """Initialize hardware error."""
        super().__init__(message, category=ErrorCategory.HARDWARE, severity=ErrorSeverity.HIGH, **kwargs)
        self.component = component


class ValidationError(SensorError):
    """Exception for data validation errors."""

    def __init__(
        self, message: str, field: Optional[str] = None, expected: Optional[Any] = None, actual: Optional[Any] = None, **kwargs
    ):
        """Initialize validation error."""
        super().__init__(message, category=ErrorCategory.USER, **kwargs)
        self.field = field
        self.expected = expected
        self.actual = actual


class ResourceError(SensorError):
    """Exception for resource-related errors."""

    def __init__(
        self,
        message: str,
        resource_type: Optional[str] = None,
        requested: Optional[Any] = None,
        available: Optional[Any] = None,
        **kwargs,
    ):
        """Initialize resource error."""
        super().__init__(message, category=ErrorCategory.RESOURCE, severity=ErrorSeverity.MEDIUM, **kwargs)
        self.resource_type = resource_type
        self.requested = requested
        self.available = available


class CalibrationError(SensorError):
    """Exception for calibration-related errors."""

    def __init__(self, message: str, calibration_type: Optional[str] = None, **kwargs):
        """Initialize calibration error."""
        super().__init__(message, category=ErrorCategory.CONFIGURATION, **kwargs)
        self.calibration_type = calibration_type


class ProcessingError(SensorError):
    """Exception for image processing errors."""

    def __init__(self, message: str, processing_stage: Optional[str] = None, **kwargs):
        """Initialize processing error."""
        super().__init__(message, category=ErrorCategory.SOFTWARE, **kwargs)
        self.processing_stage = processing_stage


class SynchronizationError(SensorError):
    """Exception for multi-sensor synchronization errors."""

    def __init__(self, message: str, sensor_count: Optional[int] = None, sync_tolerance_us: Optional[float] = None, **kwargs):
        """Initialize synchronization error."""
        super().__init__(message, category=ErrorCategory.SOFTWARE, **kwargs)
        self.sensor_count = sensor_count
        self.sync_tolerance_us = sync_tolerance_us


class GPUError(SensorError):
    """Exception for GPU acceleration errors."""

    def __init__(self, message: str, gpu_device: Optional[str] = None, **kwargs):
        """Initialize GPU error."""
        super().__init__(message, category=ErrorCategory.HARDWARE, **kwargs)
        self.gpu_device = gpu_device


# Error factory functions for common error patterns
def create_connection_timeout_error(protocol: str, timeout_seconds: float) -> ConnectionError:
    """Create a connection timeout error."""
    return ConnectionError(
        f"Connection to {protocol} device timed out after {timeout_seconds}s",
        protocol=protocol,
        error_code="CONN_TIMEOUT",
        context=ErrorContext(
            timestamp=0.0,  # Will be set by error handler
            component=protocol,
            operation="connect",
            parameters={"timeout": timeout_seconds},
            system_state={},
            recovery_hints=[
                "Check device power and connections",
                "Verify network connectivity",
                "Increase connection timeout",
                "Restart device if necessary",
            ],
        ),
    )


def create_buffer_exhaustion_error(requested_size: int, available_size: int) -> BufferError:
    """Create a buffer exhaustion error."""
    return BufferError(
        f"Insufficient buffer space: requested {requested_size} bytes, only {available_size} available",
        buffer_size=requested_size,
        error_code="BUFFER_EXHAUSTED",
        severity=ErrorSeverity.HIGH,
        context=ErrorContext(
            timestamp=0.0,
            component="buffer_manager",
            operation="allocate",
            parameters={"requested": requested_size, "available": available_size},
            system_state={},
            recovery_hints=[
                "Reduce buffer allocation size",
                "Return unused buffers to pool",
                "Increase buffer pool size",
                "Enable buffer optimization",
            ],
        ),
    )


def create_power_rail_error(rail: str, expected_voltage: float, actual_voltage: float) -> PowerError:
    """Create a power rail error."""
    return PowerError(
        f"Power rail '{rail}' voltage out of range: expected {expected_voltage}V, got {actual_voltage}V",
        rail=rail,
        voltage=actual_voltage,
        error_code="POWER_RAIL_FAULT",
        severity=ErrorSeverity.CRITICAL,
        context=ErrorContext(
            timestamp=0.0,
            component="power_management",
            operation="monitor_voltage",
            parameters={"rail": rail, "expected": expected_voltage, "actual": actual_voltage},
            system_state={},
            recovery_hints=[
                "Check power supply connections",
                "Verify power supply capacity",
                "Check for short circuits",
                "Replace faulty power components",
            ],
        ),
    )


def create_validation_error(field: str, expected: Any, actual: Any) -> ValidationError:
    """Create a validation error."""
    return ValidationError(
        f"Validation failed for field '{field}': expected {expected}, got {actual}",
        field=field,
        expected=expected,
        actual=actual,
        error_code="VALIDATION_FAILED",
        context=ErrorContext(
            timestamp=0.0,
            component="validator",
            operation="validate",
            parameters={"field": field, "expected": expected, "actual": actual},
            system_state={},
            recovery_hints=[
                f"Correct the value for field '{field}'",
                "Check input data format",
                "Verify parameter constraints",
                "Update configuration if needed",
            ],
        ),
    )
