"""
Circuit breaker pattern implementation for fault tolerance.
"""

import logging
import time
from collections.abc import Callable
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class CircuitBreakerState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    """Circuit breaker for fault tolerance."""

    def __init__(self, failure_threshold: int = 5, timeout: float = 60.0):
        """Initialize circuit breaker."""
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = 0.0
        self.state = CircuitBreakerState.CLOSED

    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Call function through circuit breaker."""
        if self.state == CircuitBreakerState.OPEN:
            if time.time() - self.last_failure_time > self.timeout:
                self.state = CircuitBreakerState.HALF_OPEN
            else:
                raise Exception("Circuit breaker is OPEN")

        try:
            result = func(*args, **kwargs)
            if self.state == CircuitBreakerState.HALF_OPEN:
                self.state = CircuitBreakerState.CLOSED
                self.failure_count = 0
            return result

        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()

            if self.failure_count >= self.failure_threshold:
                self.state = CircuitBreakerState.OPEN

            raise e
