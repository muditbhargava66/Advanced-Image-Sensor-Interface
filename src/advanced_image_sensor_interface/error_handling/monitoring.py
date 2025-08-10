"""
Error monitoring and analysis utilities.
"""

import logging
import time
from collections import defaultdict, deque
from typing import Any

logger = logging.getLogger(__name__)


class ErrorMonitor:
    """Monitors and tracks errors."""

    def __init__(self, history_size: int = 1000):
        """Initialize error monitor."""
        self.history_size = history_size
        self.error_history: deque = deque(maxlen=history_size)
        self.error_counts: dict[str, int] = defaultdict(int)

    def record_error(self, error: Exception) -> None:
        """Record an error occurrence."""
        error_info = {
            "timestamp": time.time(),
            "type": type(error).__name__,
            "message": str(error),
            "severity": getattr(error, "severity", "unknown"),
        }

        self.error_history.append(error_info)
        self.error_counts[error_info["type"]] += 1

    def get_error_rate(self, window_seconds: float = 300.0) -> float:
        """Get error rate for time window."""
        cutoff_time = time.time() - window_seconds
        recent_errors = [e for e in self.error_history if e["timestamp"] >= cutoff_time]

        return len(recent_errors) / window_seconds if window_seconds > 0 else 0.0


class ErrorAnalyzer:
    """Analyzes error patterns and trends."""

    def __init__(self, monitor: ErrorMonitor):
        """Initialize error analyzer."""
        self.monitor = monitor

    def analyze_patterns(self) -> dict[str, Any]:
        """Analyze error patterns."""
        return {
            "total_errors": len(self.monitor.error_history),
            "error_types": dict(self.monitor.error_counts),
            "error_rate_5min": self.monitor.get_error_rate(300.0),
        }
