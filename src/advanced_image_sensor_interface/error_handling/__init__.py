"""
Enhanced error handling and recovery mechanisms for the Advanced Image Sensor Interface.

This module provides comprehensive error handling, recovery strategies,
and resilience mechanisms for robust sensor operation.
"""

from .circuit_breaker import CircuitBreaker, CircuitBreakerState
from .exceptions import *
from .monitoring import ErrorAnalyzer, ErrorMonitor
from .recovery import ErrorRecoveryManager, RecoveryStrategy
from .retry import ExponentialBackoff, RetryManager, RetryPolicy

__all__ = [
    # Exceptions
    "SensorError",
    "ProtocolError",
    "BufferError",
    "PowerError",
    "SecurityError",
    "ConnectionError",
    "DataTransferError",
    "ConfigurationError",
    "TimeoutError",
    "HardwareError",
    "ValidationError",
    "ResourceError",
    # Recovery
    "ErrorRecoveryManager",
    "RecoveryStrategy",
    # Circuit Breaker
    "CircuitBreaker",
    "CircuitBreakerState",
    # Retry
    "RetryManager",
    "RetryPolicy",
    "ExponentialBackoff",
    # Monitoring
    "ErrorMonitor",
    "ErrorAnalyzer",
]
