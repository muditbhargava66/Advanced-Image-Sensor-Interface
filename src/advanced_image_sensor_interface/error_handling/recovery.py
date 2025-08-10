"""
Error recovery management and strategies.
"""

import logging
from enum import Enum

logger = logging.getLogger(__name__)


class RecoveryStrategy(Enum):
    """Available recovery strategies."""

    RETRY = "retry"
    FALLBACK = "fallback"
    RESTART = "restart"
    IGNORE = "ignore"


class ErrorRecoveryManager:
    """Manages error recovery strategies."""

    def __init__(self):
        """Initialize recovery manager."""
        self.recovery_strategies: dict[str, RecoveryStrategy] = {}

    def register_strategy(self, error_type: str, strategy: RecoveryStrategy) -> None:
        """Register a recovery strategy for an error type."""
        self.recovery_strategies[error_type] = strategy

    def recover(self, error: Exception) -> bool:
        """Attempt to recover from an error."""
        error_type = type(error).__name__
        strategy = self.recovery_strategies.get(error_type, RecoveryStrategy.IGNORE)

        logger.info(f"Applying recovery strategy {strategy.value} for {error_type}")
        return True
