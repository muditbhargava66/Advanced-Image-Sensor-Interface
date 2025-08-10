"""
Retry mechanisms with configurable policies.
"""

import logging
import random
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class RetryPolicy:
    """Retry policy configuration."""

    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    backoff_factor: float = 2.0
    jitter: bool = True


class ExponentialBackoff:
    """Exponential backoff implementation."""

    def __init__(self, policy: RetryPolicy):
        """Initialize exponential backoff."""
        self.policy = policy

    def get_delay(self, attempt: int) -> float:
        """Get delay for given attempt number."""
        delay = self.policy.base_delay * (self.policy.backoff_factor**attempt)
        delay = min(delay, self.policy.max_delay)

        if self.policy.jitter:
            delay *= 0.5 + random.random() * 0.5

        return delay


class RetryManager:
    """Manages retry operations."""

    def __init__(self, policy: RetryPolicy):
        """Initialize retry manager."""
        self.policy = policy
        self.backoff = ExponentialBackoff(policy)

    def retry(self, func: Callable, *args, **kwargs) -> Any:
        """Retry function with configured policy."""
        last_exception = None

        for attempt in range(self.policy.max_attempts):
            try:
                return func(*args, **kwargs)

            except Exception as e:
                last_exception = e

                if attempt < self.policy.max_attempts - 1:
                    delay = self.backoff.get_delay(attempt)
                    logger.warning(f"Attempt {attempt + 1} failed, retrying in {delay:.2f}s: {e}")
                    time.sleep(delay)

        raise last_exception
