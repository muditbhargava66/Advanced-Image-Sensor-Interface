"""
Unit Tests for Security and Input Validation

This module contains unit tests for the security and input validation
functionality in the Advanced Image Sensor Interface project.

Classes:
    TestInputValidator: Test cases for input validation
    TestBufferGuard: Test cases for buffer protection
    TestSecurityManager: Test cases for security management

Usage:
    Run these tests using pytest:
    $ pytest tests/test_security.py
"""

import time

import numpy as np
import pytest
from advanced_image_sensor_interface.sensor_interface.security import BufferGuard, InputValidator, SecurityLimits, SecurityManager


class TestInputValidator:
    """Test cases for input validation."""

    def test_valid_image_validation(self):
        """Test validation of valid images."""
        validator = InputValidator()

        # Valid 8-bit RGB image
        image = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        result = validator.validate_image_data(image)
        assert result.valid
        assert result.error_message == ""

        # Valid 16-bit grayscale image
        image = np.random.randint(0, 65536, (480, 640), dtype=np.uint16)
        result = validator.validate_image_data(image)
        assert result.valid

    def test_invalid_image_validation(self):
        """Test validation of invalid images."""
        validator = InputValidator()

        # Non-array input
        result = validator.validate_image_data("not an array")
        assert not result.valid
        assert "numpy array" in result.error_message

        # Empty array
        result = validator.validate_image_data(np.array([]))
        assert not result.valid
        assert "empty" in result.error_message

        # Wrong dimensions
        result = validator.validate_image_data(np.random.rand(10, 10, 10, 10))
        assert not result.valid
        assert "2D or 3D" in result.error_message

    def test_image_size_limits(self):
        """Test image size limit enforcement."""
        # Create validator with small limits for testing
        limits = SecurityLimits(max_image_size=1024)  # 1KB limit
        validator = InputValidator(limits)

        # Small image should pass
        small_image = np.random.randint(0, 256, (10, 10), dtype=np.uint8)
        result = validator.validate_image_data(small_image)
        assert result.valid

        # Large image should fail
        large_image = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        result = validator.validate_image_data(large_image)
        assert not result.valid
        assert "exceeds limit" in result.error_message

    def test_mipi_data_validation(self):
        """Test MIPI data validation."""
        validator = InputValidator()

        # Valid MIPI data
        valid_data = b"\x00\x10\x00\x5c" + b"test data"
        result = validator.validate_mipi_data(valid_data)
        assert result.valid

        # Empty data
        result = validator.validate_mipi_data(b"")
        assert not result.valid
        assert "empty" in result.error_message

        # Non-bytes input
        result = validator.validate_mipi_data("not bytes")
        assert not result.valid
        assert "bytes" in result.error_message

    def test_power_parameter_validation(self):
        """Test power parameter validation."""
        validator = InputValidator()

        # Valid parameters
        result = validator.validate_power_parameters(1.8, 0.5, "main")
        assert result.valid

        # Invalid voltage (too high)
        result = validator.validate_power_parameters(10.0, 0.5, "main")
        assert not result.valid
        assert "outside safe range" in result.error_message

        # Invalid current (negative)
        result = validator.validate_power_parameters(1.8, -0.5, "main")
        assert not result.valid

        # Non-finite values
        result = validator.validate_power_parameters(float("inf"), 0.5, "main")
        assert not result.valid
        assert "finite" in result.error_message

    def test_numeric_range_validation(self):
        """Test numeric range validation."""
        validator = InputValidator()

        # Valid range
        result = validator.validate_numeric_range(5.0, 0.0, 10.0, "test_param")
        assert result.valid
        assert result.sanitized_value == 5.0

        # Out of range
        result = validator.validate_numeric_range(15.0, 0.0, 10.0, "test_param")
        assert not result.valid
        assert "outside valid range" in result.error_message

        # Non-numeric
        result = validator.validate_numeric_range("not a number", 0.0, 10.0, "test_param")
        assert not result.valid
        assert "must be numeric" in result.error_message


class TestBufferGuard:
    """Test cases for buffer protection."""

    def test_safe_allocation(self):
        """Test safe buffer allocation."""
        guard = BufferGuard()

        # Safe allocation
        assert guard.check_allocation(1024, "test_buffer")

        # Check status
        status = guard.get_allocation_status()
        assert status["total_allocated"] == 1024
        assert status["buffer_count"] == 1

    def test_allocation_limits(self):
        """Test allocation limit enforcement."""
        limits = SecurityLimits(max_buffer_size=2048)
        guard = BufferGuard(limits)

        # Within limit
        assert guard.check_allocation(1024, "buffer1")

        # Exceeds limit
        assert not guard.check_allocation(4096, "buffer2")

    def test_buffer_release(self):
        """Test buffer release."""
        guard = BufferGuard()

        # Allocate buffer
        assert guard.check_allocation(1024, "test_buffer")

        # Release buffer
        assert guard.release_buffer("test_buffer")

        # Check status
        status = guard.get_allocation_status()
        assert status["total_allocated"] == 0
        assert status["buffer_count"] == 0

        # Try to release non-existent buffer
        assert not guard.release_buffer("non_existent")

    def test_invalid_allocation_size(self):
        """Test invalid allocation sizes."""
        guard = BufferGuard()

        # Zero size
        assert not guard.check_allocation(0, "zero_buffer")

        # Negative size
        assert not guard.check_allocation(-1024, "negative_buffer")

    def test_total_allocation_limit(self):
        """Test total allocation limit."""
        limits = SecurityLimits(max_buffer_size=1024)
        guard = BufferGuard(limits)

        # First allocation
        assert guard.check_allocation(512, "buffer1")

        # Second allocation within total limit (total = 1024, limit = 2048)
        assert guard.check_allocation(512, "buffer2")

        # Third allocation would exceed total limit (total would be 2048, limit = 2048)
        assert guard.check_allocation(1024, "buffer3")

        # Fourth allocation would exceed total limit (total would be 3072, limit = 2048)
        assert not guard.check_allocation(1024, "buffer4")


class TestSecurityManager:
    """Test cases for security management."""

    def test_initialization(self):
        """Test security manager initialization."""
        manager = SecurityManager()

        assert manager.validator is not None
        assert manager.buffer_guard is not None
        assert len(manager.operation_timeouts) == 0

    def test_operation_management(self):
        """Test operation timeout management."""
        manager = SecurityManager()

        # Start operation
        assert manager.start_operation("test_op")

        # Try to start same operation again
        assert not manager.start_operation("test_op")

        # Check timeout (should be false immediately)
        assert not manager.check_operation_timeout("test_op")

        # End operation
        assert manager.end_operation("test_op")

        # Try to end non-existent operation
        assert not manager.end_operation("non_existent")

    def test_operation_timeout(self):
        """Test operation timeout detection."""
        # Create manager with very short timeout for testing
        limits = SecurityLimits(timeout_seconds=0.1)
        manager = SecurityManager(limits)

        # Start operation
        assert manager.start_operation("timeout_test")

        # Wait for timeout
        time.sleep(0.2)

        # Check timeout
        assert manager.check_operation_timeout("timeout_test")

        # Clean up
        manager.end_operation("timeout_test")

    def test_security_status(self):
        """Test security status reporting."""
        manager = SecurityManager()

        # Get initial status
        status = manager.get_security_status()

        assert "limits" in status
        assert "buffer_status" in status
        assert "active_operations" in status
        assert "validator_ready" in status

        assert status["validator_ready"] is True
        assert status["active_operations"] == 0

    def test_integrated_validation(self):
        """Test integrated validation workflow."""
        manager = SecurityManager()

        # Test image validation through security manager
        image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        result = manager.validator.validate_image_data(image)
        assert result.valid

        # Test buffer allocation
        assert manager.buffer_guard.check_allocation(1024, "test_buffer")

        # Test operation management
        assert manager.start_operation("integrated_test")
        assert not manager.check_operation_timeout("integrated_test")
        assert manager.end_operation("integrated_test")

    def test_custom_limits(self):
        """Test security manager with custom limits."""
        custom_limits = SecurityLimits(max_image_size=1024, max_buffer_size=2048, max_voltage=3.0, timeout_seconds=5.0)

        manager = SecurityManager(custom_limits)

        # Check that limits are applied
        status = manager.get_security_status()
        assert status["limits"]["max_image_size"] == 1024
        assert status["limits"]["max_buffer_size"] == 2048
        assert status["limits"]["max_voltage"] == 3.0
        assert status["limits"]["timeout_seconds"] == 5.0

    def test_concurrent_operations(self):
        """Test multiple concurrent operations."""
        manager = SecurityManager()

        # Start multiple operations
        assert manager.start_operation("op1")
        assert manager.start_operation("op2")
        assert manager.start_operation("op3")

        # Check status
        status = manager.get_security_status()
        assert status["active_operations"] == 3

        # End operations
        assert manager.end_operation("op1")
        assert manager.end_operation("op2")
        assert manager.end_operation("op3")

        # Check final status
        status = manager.get_security_status()
        assert status["active_operations"] == 0


class TestSecurityLimits:
    """Test cases for security limits configuration."""

    def test_default_limits(self):
        """Test default security limits."""
        limits = SecurityLimits()

        assert limits.max_image_size > 0
        assert limits.max_buffer_size > 0
        assert limits.max_voltage > 0
        assert limits.min_voltage >= 0
        assert limits.timeout_seconds > 0

    def test_custom_limits(self):
        """Test custom security limits."""
        limits = SecurityLimits(max_image_size=1024, max_buffer_size=2048, max_voltage=5.0, min_voltage=0.5, timeout_seconds=10.0)

        assert limits.max_image_size == 1024
        assert limits.max_buffer_size == 2048
        assert limits.max_voltage == 5.0
        assert limits.min_voltage == 0.5
        assert limits.timeout_seconds == 10.0


if __name__ == "__main__":
    pytest.main([__file__])
