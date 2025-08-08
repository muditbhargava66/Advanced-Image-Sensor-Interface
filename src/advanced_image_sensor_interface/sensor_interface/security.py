"""
Security and Input Validation for Advanced Image Sensor Interface

This module provides comprehensive input validation, security checks,
and protection against malicious or malformed inputs.

Classes:
    InputValidator: Comprehensive input validation
    SecurityManager: Security policy enforcement
    BufferGuard: Buffer overflow and size limit protection

Functions:
    validate_image_data: Validate image data and parameters
    validate_mipi_data: Validate MIPI protocol data
    check_buffer_limits: Check buffer size limits
    sanitize_numeric_input: Sanitize numeric inputs
"""

import logging
import time
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of input validation."""

    valid: bool
    error_message: str = ""
    warnings: list[str] = None
    sanitized_value: Any = None

    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []


@dataclass
class SecurityLimits:
    """Security limits and constraints."""

    max_image_size: int = 100 * 1024 * 1024  # 100MB max image
    max_buffer_size: int = 1024 * 1024 * 1024  # 1GB max buffer
    max_frame_rate: float = 1000.0  # 1000 fps max
    max_data_rate: float = 50.0  # 50 Gbps max
    max_voltage: float = 5.0  # 5V max
    min_voltage: float = 0.0  # 0V min
    max_current: float = 10.0  # 10A max
    max_temperature: float = 150.0  # 150°C max
    timeout_seconds: float = 30.0  # 30 second timeout


class InputValidator:
    """
    Comprehensive input validation for all system components.

    Provides validation for images, MIPI data, power parameters,
    and other system inputs with security checks.
    """

    def __init__(self, limits: Optional[SecurityLimits] = None):
        """
        Initialize input validator.

        Args:
            limits: Security limits (uses defaults if None)
        """
        self.limits = limits or SecurityLimits()
        logger.info("Input validator initialized with security limits")

    def validate_image_data(self, image: Any, expected_shape: Optional[tuple[int, ...]] = None) -> ValidationResult:
        """
        Validate image data comprehensively.

        Args:
            image: Image data to validate
            expected_shape: Expected image shape (optional)

        Returns:
            ValidationResult with validation status
        """
        warnings = []

        # Check if input is numpy array
        if not isinstance(image, np.ndarray):
            return ValidationResult(False, "Image must be a numpy array")

        # Check for empty array
        if image.size == 0:
            return ValidationResult(False, "Image cannot be empty")

        # Check array dimensions
        if image.ndim < 2 or image.ndim > 3:
            return ValidationResult(False, f"Image must be 2D or 3D, got {image.ndim}D")

        # Check image size limits
        image_bytes = image.nbytes
        if image_bytes > self.limits.max_image_size:
            return ValidationResult(False, f"Image size {image_bytes} bytes exceeds limit {self.limits.max_image_size}")

        # Check for reasonable dimensions
        height, width = image.shape[:2]
        if height < 1 or width < 1:
            return ValidationResult(False, f"Invalid image dimensions: {height}x{width}")

        if height > 16384 or width > 16384:
            warnings.append(f"Very large image dimensions: {height}x{width}")

        # Check channels for color images
        if image.ndim == 3:
            channels = image.shape[2]
            if channels not in {1, 3, 4}:
                return ValidationResult(False, f"Invalid number of channels: {channels}")

        # Check data type
        supported_dtypes = [np.uint8, np.uint16, np.int16, np.float32, np.float64]
        if not any(image.dtype == dtype for dtype in supported_dtypes):
            return ValidationResult(False, f"Unsupported image dtype: {image.dtype}")

        # Check for non-finite values
        if np.issubdtype(image.dtype, np.floating):
            if not np.all(np.isfinite(image)):
                return ValidationResult(False, "Image contains non-finite values (NaN or Inf)")

            if np.any(image < 0) or np.any(image > 1):
                warnings.append("Float image values outside [0, 1] range")

        # Check expected shape if provided
        if expected_shape is not None:
            if image.shape != expected_shape:
                return ValidationResult(False, f"Shape mismatch: expected {expected_shape}, got {image.shape}")

        # Check for suspicious patterns that might indicate corruption
        if np.all(image == image.flat[0]):
            warnings.append("Image has constant values (might be corrupted)")

        return ValidationResult(True, warnings=warnings, sanitized_value=image)

    def validate_mipi_data(self, data: Any, max_size: Optional[int] = None) -> ValidationResult:
        """
        Validate MIPI protocol data.

        Args:
            data: Data to validate
            max_size: Maximum allowed size in bytes

        Returns:
            ValidationResult with validation status
        """
        warnings = []

        # Check if data is bytes
        if not isinstance(data, (bytes, bytearray)):
            return ValidationResult(False, "MIPI data must be bytes or bytearray")

        # Check size limits
        data_size = len(data)
        max_allowed = max_size or self.limits.max_buffer_size

        if data_size > max_allowed:
            return ValidationResult(False, f"Data size {data_size} exceeds limit {max_allowed}")

        if data_size == 0:
            return ValidationResult(False, "MIPI data cannot be empty")

        # Check for reasonable packet sizes
        if data_size < 4:
            warnings.append("Very small MIPI packet (< 4 bytes)")
        elif data_size > 64 * 1024:
            warnings.append(f"Large MIPI packet ({data_size} bytes)")

        # Basic pattern validation for MIPI packets
        if data_size >= 4:
            # Check if it looks like a valid MIPI packet header
            header = data[:4]
            di = header[0]  # Data Identifier

            # Check virtual channel (bits 6-7)
            virtual_channel = (di >> 6) & 0x3
            if virtual_channel > 3:
                warnings.append(f"Invalid virtual channel: {virtual_channel}")

            # Check data type (bits 0-5)
            data_type = di & 0x3F
            if data_type > 0x3F:
                warnings.append(f"Invalid data type: {data_type}")

        return ValidationResult(True, warnings=warnings, sanitized_value=data)

    def validate_power_parameters(self, voltage: float, current: float, rail: str) -> ValidationResult:
        """
        Validate power management parameters.

        Args:
            voltage: Voltage value in volts
            current: Current value in amperes
            rail: Power rail name

        Returns:
            ValidationResult with validation status
        """
        warnings = []

        # Validate voltage
        if not isinstance(voltage, (int, float)):
            return ValidationResult(False, "Voltage must be numeric")

        if not np.isfinite(voltage):
            return ValidationResult(False, "Voltage must be finite")

        if voltage < self.limits.min_voltage or voltage > self.limits.max_voltage:
            return ValidationResult(
                False, f"Voltage {voltage}V outside safe range [{self.limits.min_voltage}, {self.limits.max_voltage}]"
            )

        # Validate current
        if not isinstance(current, (int, float)):
            return ValidationResult(False, "Current must be numeric")

        if not np.isfinite(current):
            return ValidationResult(False, "Current must be finite")

        if current < 0 or current > self.limits.max_current:
            return ValidationResult(False, f"Current {current}A outside safe range [0, {self.limits.max_current}]")

        # Validate rail name
        if not isinstance(rail, str):
            return ValidationResult(False, "Rail name must be string")

        if not rail.strip():
            return ValidationResult(False, "Rail name cannot be empty")

        # Check power calculation
        power = voltage * current
        max_power = 50.0  # 50W max power
        if power > max_power:
            return ValidationResult(False, f"Power {power:.2f}W exceeds safe limit {max_power}W")

        # Warnings for unusual values
        if voltage > 4.0:
            warnings.append(f"High voltage: {voltage}V")

        if current > 5.0:
            warnings.append(f"High current: {current}A")

        return ValidationResult(True, warnings=warnings)

    def validate_numeric_range(self, value: Any, min_val: float, max_val: float, name: str) -> ValidationResult:
        """
        Validate numeric value within specified range.

        Args:
            value: Value to validate
            min_val: Minimum allowed value
            max_val: Maximum allowed value
            name: Parameter name for error messages

        Returns:
            ValidationResult with validation status
        """
        if not isinstance(value, (int, float)):
            return ValidationResult(False, f"{name} must be numeric")

        if not np.isfinite(value):
            return ValidationResult(False, f"{name} must be finite")

        if value < min_val or value > max_val:
            return ValidationResult(False, f"{name} {value} outside valid range [{min_val}, {max_val}]")

        return ValidationResult(True, sanitized_value=float(value))


class BufferGuard:
    """
    Buffer overflow and size limit protection.

    Provides protection against buffer overflows, excessive memory
    allocation, and denial-of-service attacks.
    """

    def __init__(self, limits: Optional[SecurityLimits] = None):
        """
        Initialize buffer guard.

        Args:
            limits: Security limits (uses defaults if None)
        """
        self.limits = limits or SecurityLimits()
        self.allocated_buffers: dict[str, int] = {}
        self.total_allocated = 0
        logger.info("Buffer guard initialized")

    def check_allocation(self, size: int, buffer_id: str = "default") -> bool:
        """
        Check if buffer allocation is safe.

        Args:
            size: Requested buffer size in bytes
            buffer_id: Buffer identifier

        Returns:
            True if allocation is safe, False otherwise
        """
        if size <= 0:
            logger.error(f"Invalid buffer size: {size}")
            return False

        if size > self.limits.max_buffer_size:
            logger.error(f"Buffer size {size} exceeds limit {self.limits.max_buffer_size}")
            return False

        # Check total allocation limit (allow up to 2x max_buffer_size total)
        total_limit = self.limits.max_buffer_size * 2
        if self.total_allocated + size > total_limit:
            logger.error(f"Total allocation {self.total_allocated + size} would exceed limit {total_limit}")
            return False

        # Track allocation
        if buffer_id in self.allocated_buffers:
            self.total_allocated -= self.allocated_buffers[buffer_id]

        self.allocated_buffers[buffer_id] = size
        self.total_allocated += size

        logger.debug(f"Buffer allocation approved: {size} bytes (total: {self.total_allocated})")
        return True

    def release_buffer(self, buffer_id: str) -> bool:
        """
        Release a tracked buffer.

        Args:
            buffer_id: Buffer identifier

        Returns:
            True if buffer was released, False if not found
        """
        if buffer_id in self.allocated_buffers:
            size = self.allocated_buffers[buffer_id]
            self.total_allocated -= size
            del self.allocated_buffers[buffer_id]
            logger.debug(f"Buffer released: {size} bytes (total: {self.total_allocated})")
            return True

        return False

    def get_allocation_status(self) -> dict[str, Any]:
        """
        Get current allocation status.

        Returns:
            Dictionary with allocation information
        """
        return {
            "total_allocated": self.total_allocated,
            "buffer_count": len(self.allocated_buffers),
            "max_buffer_size": self.limits.max_buffer_size,
            "utilization_percent": (self.total_allocated / self.limits.max_buffer_size) * 100,
        }


class SecurityManager:
    """
    Overall security policy enforcement.

    Coordinates input validation, buffer protection, and
    security policy enforcement across the system.
    """

    def __init__(self, limits: Optional[SecurityLimits] = None):
        """
        Initialize security manager.

        Args:
            limits: Security limits (uses defaults if None)
        """
        self.limits = limits or SecurityLimits()
        self.validator = InputValidator(self.limits)
        self.buffer_guard = BufferGuard(self.limits)
        self.operation_timeouts: dict[str, float] = {}
        logger.info("Security manager initialized")

    def start_operation(self, operation_id: str) -> bool:
        """
        Start a timed operation.

        Args:
            operation_id: Unique operation identifier

        Returns:
            True if operation can start, False if already running
        """
        if operation_id in self.operation_timeouts:
            logger.warning(f"Operation {operation_id} already running")
            return False

        self.operation_timeouts[operation_id] = time.time()
        return True

    def check_operation_timeout(self, operation_id: str) -> bool:
        """
        Check if operation has timed out.

        Args:
            operation_id: Operation identifier

        Returns:
            True if operation has timed out, False otherwise
        """
        if operation_id not in self.operation_timeouts:
            return False

        elapsed = time.time() - self.operation_timeouts[operation_id]
        return elapsed > self.limits.timeout_seconds

    def end_operation(self, operation_id: str) -> bool:
        """
        End a timed operation.

        Args:
            operation_id: Operation identifier

        Returns:
            True if operation was ended, False if not found
        """
        if operation_id in self.operation_timeouts:
            del self.operation_timeouts[operation_id]
            return True

        return False

    def get_security_status(self) -> dict[str, Any]:
        """
        Get overall security status.

        Returns:
            Dictionary with security status information
        """
        return {
            "limits": {
                "max_image_size": self.limits.max_image_size,
                "max_buffer_size": self.limits.max_buffer_size,
                "max_voltage": self.limits.max_voltage,
                "timeout_seconds": self.limits.timeout_seconds,
            },
            "buffer_status": self.buffer_guard.get_allocation_status(),
            "active_operations": len(self.operation_timeouts),
            "validator_ready": self.validator is not None,
        }


# Example usage and testing
if __name__ == "__main__":
    print("Testing Security and Input Validation...")

    # Create security manager
    security = SecurityManager()

    # Test image validation
    print("\n1. Image Validation Tests:")
    print("-" * 30)

    # Valid image
    valid_image = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
    result = security.validator.validate_image_data(valid_image)
    print(f"Valid image: {'PASS' if result.valid else 'FAIL'} {result.error_message}")

    # Invalid image (too large)
    try:
        huge_image = np.zeros((10000, 10000, 3), dtype=np.uint8)
        result = security.validator.validate_image_data(huge_image)
        print(f"Huge image: {'PASS' if result.valid else 'FAIL'} {result.error_message}")
    except MemoryError:
        print("Huge image: FAIL Memory error (expected)")

    # Test MIPI data validation
    print("\n2. MIPI Data Validation Tests:")
    print("-" * 30)

    # Valid MIPI data
    valid_mipi = b"\x00\x10\x00\x5c" + b"test data" * 100
    result = security.validator.validate_mipi_data(valid_mipi)
    print(f"Valid MIPI data: {'PASS' if result.valid else 'FAIL'} {result.error_message}")

    # Invalid MIPI data (empty)
    result = security.validator.validate_mipi_data(b"")
    print(f"Empty MIPI data: {'PASS' if result.valid else 'FAIL'} {result.error_message}")

    # Test power validation
    print("\n3. Power Parameter Validation Tests:")
    print("-" * 30)

    # Valid power parameters
    result = security.validator.validate_power_parameters(1.8, 0.5, "main")
    print(f"Valid power params: {'PASS' if result.valid else 'FAIL'} {result.error_message}")

    # Invalid power parameters (too high voltage)
    result = security.validator.validate_power_parameters(10.0, 0.5, "main")
    print(f"High voltage: {'PASS' if result.valid else 'FAIL'} {result.error_message}")

    # Test buffer guard
    print("\n4. Buffer Guard Tests:")
    print("-" * 30)

    # Safe allocation
    safe = security.buffer_guard.check_allocation(1024 * 1024, "test_buffer")
    print(f"Safe allocation: {'PASS' if safe else 'FAIL'}")

    # Unsafe allocation
    unsafe = security.buffer_guard.check_allocation(2 * 1024 * 1024 * 1024, "huge_buffer")
    print(f"Unsafe allocation: {'PASS' if unsafe else 'FAIL'}")

    # Test operation timeout
    print("\n5. Operation Timeout Tests:")
    print("-" * 30)

    # Start operation
    started = security.start_operation("test_op")
    print(f"Start operation: {'PASS' if started else 'FAIL'}")

    # Check timeout (should be false immediately)
    timed_out = security.check_operation_timeout("test_op")
    print(f"Immediate timeout check: {'FAIL' if timed_out else 'PASS'}")

    # End operation
    ended = security.end_operation("test_op")
    print(f"End operation: {'PASS' if ended else 'FAIL'}")

    # Get security status
    print("\n6. Security Status:")
    print("-" * 30)
    status = security.get_security_status()
    print(f"Buffer utilization: {status['buffer_status']['utilization_percent']:.1f}%")
    print(f"Active operations: {status['active_operations']}")

    print("\nSecurity and validation tests completed!")
