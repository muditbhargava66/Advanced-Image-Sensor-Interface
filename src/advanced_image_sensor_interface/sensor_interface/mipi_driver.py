"""
MIPI CSI-2 Simulation for Advanced Image Sensor Interface

This module provides the legacy MIPI CSI-2 driver for backward compatibility.
For new code, use the protocol.mipi.driver.MIPIProtocolDriver instead.
"""

import asyncio
import logging
import threading
import time
import warnings
from abc import abstractmethod
from dataclasses import dataclass
from typing import Optional, Protocol, runtime_checkable

from ..config import get_processing_config, get_security_config, get_timing_config
from ..utils.buffer_manager import get_buffer_manager, ManagedBuffer
from .mipi_protocol import MIPIProtocolValidator
from .security import SecurityManager, SecurityLimits, ValidationResult

from .protocol.mipi.driver import MIPIConfig as NewMIPIConfig

logger = logging.getLogger(__name__)


# Protocol interfaces for dependency injection
@runtime_checkable
class SecurityManagerInterface(Protocol):
    """Interface for security manager dependency injection."""

    @abstractmethod
    def validate_mipi_data(self, data: bytes) -> "ValidationResult":
        """Validate MIPI data."""
        ...

    @property
    def validator(self):
        """Get the validator."""
        ...

    @property
    def buffer_guard(self):
        """Get the buffer guard."""
        ...

    def start_operation(self, operation_id: str) -> bool:
        """Start a timed operation."""
        ...

    def end_operation(self, operation_id: str) -> None:
        """End a timed operation."""
        ...

    def check_operation_timeout(self, operation_id: str) -> bool:
        """Check if operation timed out."""
        ...


@runtime_checkable
class ProtocolValidatorInterface(Protocol):
    """Interface for protocol validator dependency injection."""

    @abstractmethod
    def validate_packet(self, packet: bytes) -> bool:
        """Validate MIPI packet."""
        ...

    def get_statistics(self) -> dict:
        """Get validator statistics."""
        ...


logger = logging.getLogger(__name__)


# Backward compatibility alias with deprecation warning
class MIPIConfig:
    """Deprecated: Use sensor_interface.protocol.mipi.driver.MIPIConfig instead."""

    def __init__(self, lanes: int, data_rate: float, channel: int):
        warnings.warn(
            "MIPIConfig in mipi_driver is deprecated. " "Use sensor_interface.protocol.mipi.driver.MIPIConfig instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        # Convert to new MIPIConfig format
        self._new_config = NewMIPIConfig(
            lanes=lanes, data_rate_mbps=data_rate * 1000, virtual_channel=channel  # Convert Gbps to Mbps
        )

    @property
    def data_rate(self) -> float:
        """Data rate in Gbps (deprecated, use data_rate_mbps)."""
        return self._new_config.data_rate_mbps / 1000

    @data_rate.setter
    def data_rate(self, value: float):
        self._new_config.data_rate_mbps = value * 1000

    @property
    def channel(self) -> int:
        """Virtual channel ID (deprecated, use virtual_channel)."""
        return self._new_config.virtual_channel

    @channel.setter
    def channel(self, value: int):
        self._new_config.virtual_channel = value

    def __getattr__(self, name):
        return getattr(self._new_config, name)

    def __setattr__(self, name, value):
        if name in ("_new_config",):
            super().__setattr__(name, value)
        elif name == "data_rate":
            self._new_config.data_rate_mbps = value * 1000
        elif name == "channel":
            self._new_config.virtual_channel = value
        else:
            setattr(self._new_config, name, value)


logger = logging.getLogger(__name__)


# Get MIPI config constants
def get_mipi_config():
    """Get MIPI configuration constants."""
    from ..config.constants import get_config

    return get_config().mipi


@dataclass
class OperationResult:
    """Result of a MIPI operation with retry support."""

    success: bool
    data: Optional[bytes] = None
    error: str = ""
    retry_count: int = 0
    execution_time: float = 0.0


@dataclass
class ErrorStatistics:
    """Statistics for tracking MIPI driver errors."""

    total_errors: int = 0
    transmission_errors: int = 0
    timeout_errors: int = 0
    validation_errors: int = 0
    total_operations: int = 0
    successful_operations: int = 0

    def record_operation(self, success: bool):
        """Record an operation result."""
        self.total_operations += 1
        if success:
            self.successful_operations += 1
        else:
            self.total_errors += 1

    def record_error(self, error_type: str) -> None:
        """Record an error of the specified type."""
        self.total_errors += 1
        if error_type == "transmission":
            self.transmission_errors += 1
        elif error_type == "timeout":
            self.timeout_errors += 1
        elif error_type == "validation":
            self.validation_errors += 1

    def get_rate(self) -> float:
        """Get current error rate."""
        if self.total_operations == 0:
            return 0.0
        return self.total_errors / self.total_operations


# Legacy MIPIDriver class for backward compatibility with tests
# This implements the old interface but delegates to new driver where possible
class MIPIDriver:
    """
    High-performance MIPI CSI-2 driver for image sensor communication.

    Attributes:
        config (MIPIConfig): Configuration for MIPI communication.
    """

    def __init__(
        self,
        config: MIPIConfig,
        security_manager: Optional[SecurityManagerInterface] = None,
        protocol_validator: Optional[ProtocolValidatorInterface] = None,
    ):
        """
        Initialize the MIPIDriver with the given configuration.

        Args:
            config (MIPIConfig): Configuration for MIPI communication.
            security_manager: Security manager for validation (optional)
            protocol_validator: Protocol validator (optional)
        """
        self.config = config

        # Get configuration constants
        self._timing_config = get_timing_config()
        self._security_config = get_security_config()

        # Initialize buffer manager
        self._buffer_manager = get_buffer_manager(max_buffer_size=self._security_config.MAX_BUFFER_SIZE)

        # Use dependency injection or create defaults
        self._security_manager = security_manager or SecurityManager(
            SecurityLimits(
                max_buffer_size=self._security_config.MAX_BUFFER_SIZE, timeout_seconds=self._security_config.OPERATION_TIMEOUT
            )
        )
        self._protocol_validator = protocol_validator or MIPIProtocolValidator()

        # Thread safety
        self._lock = threading.RLock()

        # Performance tracking
        self._transfer_time = self._timing_config.TRANSFER_TIME_PER_MB
        self._error_stats = ErrorStatistics()

        # Storage for transmitted data (simulation)
        self._transmitted_data = None

        # Initialize driver
        self._initialize_driver()
        logger.info(f"MIPI Driver initialized with {self.config.lanes} lanes at {self.config.data_rate} Gbps")

    def _initialize_driver(self) -> None:
        """Initialize the MIPI driver system."""
        time.sleep(self._timing_config.DRIVER_INIT_DELAY)
        logger.info("MIPI driver system initialized successfully")

    def send_data(self, data: bytes) -> bool:
        """
        Send data through the MIPI interface with security validation.

        Args:
            data (bytes): Data to be sent.

        Returns:
            bool: True if data was sent successfully, False otherwise.

        Raises:
            ValueError: If data validation fails.
        """
        # Validate input data
        validation_result = self._security_manager.validator.validate_mipi_data(data)
        if not validation_result.valid:
            raise ValueError(f"Invalid MIPI data: {validation_result.error_message}")

        # Log warnings if any
        for warning in validation_result.warnings:
            logger.warning(f"MIPI data warning: {warning}")

        try:
            with self._lock:
                # Check buffer allocation
                if not self._security_manager.buffer_guard.check_allocation(len(data), f"mipi_tx_{id(data)}"):
                    raise ValueError("Buffer allocation denied - size limit exceeded")

                # Start timed operation
                operation_id = f"mipi_send_{time.time()}"
                if not self._security_manager.start_operation(operation_id):
                    raise ValueError("Cannot start operation - system busy")

                try:
                    # Simulate data transmission time based on data size and rate
                    transmission_time = len(data) / (self.config.data_rate * 1e9 / 8)  # Convert Gbps to bytes/s

                    # Check for timeout during transmission
                    if transmission_time > 1.0:  # Warn for long transmissions
                        logger.warning(f"Long transmission time: {transmission_time:.3f}s")

                    time.sleep(min(transmission_time, self._timing_config.MAX_SIMULATION_TIME))

                    # Check operation timeout
                    if self._security_manager.check_operation_timeout(operation_id):
                        raise TimeoutError("MIPI transmission timed out")

                    # Store transmitted data for simulation
                    self._transmitted_data = bytes(data)

                    # Use managed buffer for temporary storage
                    with ManagedBuffer(len(data), self._buffer_manager) as temp_buffer:
                        temp_buffer[:] = data
                        # Simulate processing/transmission
                        pass

                    # Record successful operation
                    self._error_stats.record_operation(True)
                    logger.info(f"Sent {len(data)} bytes through MIPI interface")
                    return True

                finally:
                    # Clean up operation and buffer tracking
                    self._security_manager.end_operation(operation_id)
                    self._security_manager.buffer_guard.release_buffer(f"mipi_tx_{id(data)}")

        except Exception as e:
            logger.error(f"Error sending data: {e}")
            self._error_stats.record_operation(False)
            if isinstance(e, (ValueError, TimeoutError)):
                raise  # Re-raise validation and timeout errors
            return False

    def receive_data(self, size: int) -> Optional[bytes]:
        """
        Receive data from the MIPI interface.

        Args:
            size (int): Number of bytes to receive.

        Returns:
            Optional[bytes]: Received data, or None if no data available.
        """
        try:
            with self._lock:
                # Return stored transmitted data if available and size matches
                if self._transmitted_data and len(self._transmitted_data) == size:
                    result = self._transmitted_data
                    logger.info(f"Received {len(result)} bytes from MIPI interface")
                    return result

                # Fallback to test data if no transmitted data available
                with ManagedBuffer(size, self._buffer_manager) as receive_buffer:
                    # Simulate receiving data
                    test_data = b"TestMIPIData" * (size // 12 + 1)
                    actual_data = test_data[:size]
                    receive_buffer[: len(actual_data)] = actual_data

                    # Return copy of the data
                    result = bytes(receive_buffer[: len(actual_data)])
                    logger.info(f"Received {len(result)} bytes from MIPI interface (fallback)")
                    return result

        except Exception as e:
            logger.error(f"Error receiving data: {e}")
            return None

    def get_status(self) -> dict:
        """
        Get the current status of the MIPI driver.

        Returns:
            dict: Dictionary containing status information.
        """
        protocol_stats = self._protocol_validator.get_statistics()
        buffer_stats = self.get_buffer_stats()

        return {
            "lanes": self.config.lanes,
            "data_rate": self.config.data_rate,
            "channel": self.config.channel,
            "error_rate": self._calculate_error_rate(),
            "throughput": self._calculate_throughput(),
            "protocol_stats": protocol_stats,
            "buffer_management": buffer_stats,
            "transfer_time_per_mb": self._transfer_time,
        }

    def _calculate_error_rate(self) -> float:
        """Calculate the current error rate based on actual metrics."""
        return self._error_stats.get_rate()

    def _calculate_throughput(self) -> float:
        """Calculate the current throughput in Gbps."""
        # Calculate theoretical throughput with deterministic efficiency
        theoretical_throughput = self.config.lanes * self.config.data_rate
        mipi_config = get_mipi_config()

        # Use error rate to determine efficiency
        error_rate = self._error_stats.get_rate()
        base_efficiency = (mipi_config.MIN_EFFICIENCY + mipi_config.MAX_EFFICIENCY) / 2
        efficiency = base_efficiency * (1.0 - error_rate * 10)  # Errors reduce efficiency
        efficiency = max(mipi_config.MIN_EFFICIENCY, min(mipi_config.MAX_EFFICIENCY, efficiency))

        return theoretical_throughput * efficiency

    def optimize_performance(self) -> None:
        """
        Optimize MIPI driver performance (SIMULATION ONLY).

        This method simulates performance optimization by adjusting internal timing
        parameters. In a real hardware implementation, this would involve:
        - PHY parameter tuning
        - Clock optimization
        - Buffer management improvements
        - Error correction optimization

        Simulated improvements based on configuration constants.
        """
        processing_config = get_processing_config()

        # Use configurable optimization factors
        self._transfer_time *= self._timing_config.OPTIMIZATION_FACTOR_PRODUCTION
        self.config.data_rate *= processing_config.DATA_RATE_IMPROVEMENT

        # Optimize buffer manager
        self._buffer_manager.optimize_pools()

        logger.info(f"Optimized performance (SIMULATED): Transfer time reduced to {self._transfer_time:.3f} seconds per MB")

    def send_data_with_retry(self, data: bytes, max_retries: int = 3) -> OperationResult:
        """
        Send data with automatic retry on failure.

        Args:
            data: Data to send
            max_retries: Maximum number of retry attempts

        Returns:
            OperationResult: Result with retry information
        """
        start_time = time.time()

        for attempt in range(max_retries):
            try:
                result = self.send_data(data)
                execution_time = time.time() - start_time

                if result:
                    return OperationResult(success=True, data=data, retry_count=attempt, execution_time=execution_time)

                # Exponential backoff for retries
                if attempt < max_retries - 1:
                    backoff_time = 2**attempt * 0.1  # 0.1, 0.2, 0.4 seconds
                    time.sleep(backoff_time)

            except Exception as e:
                if attempt == max_retries - 1:
                    execution_time = time.time() - start_time
                    return OperationResult(success=False, error=str(e), retry_count=attempt + 1, execution_time=execution_time)

                # Exponential backoff for exceptions too
                backoff_time = 2**attempt * 0.1
                time.sleep(backoff_time)
                continue

        execution_time = time.time() - start_time
        return OperationResult(
            success=False, error="Max retries exceeded", retry_count=max_retries, execution_time=execution_time
        )

    async def send_data_async(self, data: bytes) -> bool:
        """
        Async version of send_data for non-blocking operations.

        Args:
            data: Data to send

        Returns:
            bool: True if successful
        """
        # Validate input data
        validation_result = self._security_manager.validator.validate_mipi_data(data)
        if not validation_result.valid:
            raise ValueError(f"Invalid MIPI data: {validation_result.error_message}")

        # Log warnings if any
        for warning in validation_result.warnings:
            logger.warning(f"MIPI data warning: {warning}")

        try:
            # Check buffer allocation
            if not self._security_manager.buffer_guard.check_allocation(len(data), f"mipi_tx_async_{id(data)}"):
                raise ValueError("Buffer allocation denied - size limit exceeded")

            # Start timed operation
            operation_id = f"mipi_send_async_{time.time()}"
            if not self._security_manager.start_operation(operation_id):
                raise ValueError("Cannot start operation - system busy")

            try:
                # Simulate async data transmission
                transmission_time = len(data) / (self.config.data_rate * 1e9 / 8)

                if transmission_time > 1.0:
                    logger.warning(f"Long async transmission time: {transmission_time:.3f}s")

                await asyncio.sleep(min(transmission_time, self._timing_config.MAX_SIMULATION_TIME))

                # Check operation timeout
                if self._security_manager.check_operation_timeout(operation_id):
                    raise TimeoutError("Async MIPI transmission timed out")

                # Record successful operation
                self._error_stats.record_operation(True)
                logger.info(f"Async sent {len(data)} bytes through MIPI interface")
                return True

            finally:
                # Clean up operation and buffer tracking
                self._security_manager.end_operation(operation_id)
                self._security_manager.buffer_guard.release_buffer(f"mipi_tx_async_{id(data)}")

        except Exception as e:
            logger.error(f"Error in async send: {e}")
            self._error_stats.record_operation(False)
            if isinstance(e, (ValueError, TimeoutError)):
                raise
            return False

    def get_buffer_stats(self) -> dict:
        """Get buffer manager statistics."""
        return {
            "buffer_stats": self._buffer_manager.get_stats().__dict__ if self._buffer_manager.get_stats() else {},
            "memory_usage": self._buffer_manager.get_memory_usage(),
            "error_rate": self._error_stats.get_rate(),
        }


# Example usage demonstrating performance improvements
if __name__ == "__main__":
    config = MIPIConfig(lanes=4, data_rate=2.5, channel=0)
    driver = MIPIDriver(config)

    # Test data transmission
    test_data = b"Hello, MIPI!" * 1000

    # Measure initial performance
    start_time = time.time()
    driver.send_data(test_data)
    initial_time = time.time() - start_time

    # Optimize performance
    driver.optimize_performance()

    # Measure optimized performance
    start_time = time.time()
    driver.send_data(test_data)
    optimized_time = time.time() - start_time

    # Calculate improvement
    improvement = (initial_time - optimized_time) / initial_time * 100
    print(f"Performance improvement: {improvement:.2f}%")

    # Print status
    print("MIPI Driver Status:")
    print(driver.get_status())
