"""
Protocol selector for dynamic protocol switching and optimization.

This module provides intelligent protocol selection based on requirements,
performance metrics, and system capabilities.
"""

import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

from .protocol.base import ProtocolBase

logger = logging.getLogger(__name__)


class ProtocolType(Enum):
    """Supported protocol types."""

    MIPI = "mipi"
    COAXPRESS = "coaxpress"
    GIGE = "gige"
    USB3 = "usb3"


@dataclass
class ProtocolRequirements:
    """Requirements for protocol selection."""

    bandwidth_gbps: float = 1.0
    distance_m: float = 1.0
    power_over_cable: bool = False
    latency_ms: float = 10.0
    multi_camera: bool = False
    cost_priority: float = 0.5  # 0.0 = cost not important, 1.0 = cost critical
    reliability_priority: float = 0.7  # 0.0 = reliability not important, 1.0 = critical


@dataclass
class ProtocolScore:
    """Scoring result for a protocol."""

    protocol_type: ProtocolType
    total_score: float
    bandwidth_score: float
    distance_score: float
    power_score: float
    latency_score: float
    cost_score: float
    reliability_score: float
    compatibility_score: float

    @property
    def is_suitable(self) -> bool:
        """Check if protocol is suitable (score > 0.6)."""
        return self.total_score > 0.6


class ProtocolSelector:
    """
    Intelligent protocol selector for camera interfaces.

    This class analyzes requirements and system capabilities to recommend
    the optimal protocol for a given application.
    """

    def __init__(self):
        """Initialize protocol selector."""
        self.available_protocols: dict[ProtocolType, dict[str, Any]] = {}
        self.protocol_instances: dict[ProtocolType, ProtocolBase] = {}
        self.current_protocol: Optional[ProtocolType] = None
        self.performance_history: dict[ProtocolType, list[dict[str, float]]] = {}

        # Protocol characteristics for scoring
        self.protocol_characteristics = {
            ProtocolType.MIPI: {
                "max_bandwidth_gbps": 4.5,
                "max_distance_m": 1.0,
                "power_over_cable": False,
                "typical_latency_ms": 1.0,
                "cost_factor": 0.9,  # Low cost
                "reliability_factor": 0.8,
                "complexity_factor": 0.7,
            },
            ProtocolType.COAXPRESS: {
                "max_bandwidth_gbps": 12.5,
                "max_distance_m": 100.0,
                "power_over_cable": True,
                "typical_latency_ms": 5.0,
                "cost_factor": 0.3,  # High cost
                "reliability_factor": 0.95,
                "complexity_factor": 0.4,
            },
            ProtocolType.GIGE: {
                "max_bandwidth_gbps": 1.0,
                "max_distance_m": 100.0,
                "power_over_cable": True,
                "typical_latency_ms": 10.0,
                "cost_factor": 0.7,  # Medium cost
                "reliability_factor": 0.85,
                "complexity_factor": 0.8,
            },
            ProtocolType.USB3: {
                "max_bandwidth_gbps": 5.0,
                "max_distance_m": 5.0,
                "power_over_cable": True,
                "typical_latency_ms": 3.0,
                "cost_factor": 0.8,  # Low-medium cost
                "reliability_factor": 0.75,
                "complexity_factor": 0.9,
            },
        }

        logger.info("Protocol selector initialized")

    def register_protocol(self, protocol_type: ProtocolType, protocol_instance: ProtocolBase, config: dict[str, Any]) -> bool:
        """
        Register a protocol implementation.

        Args:
            protocol_type: Type of protocol
            protocol_instance: Protocol implementation instance
            config: Protocol configuration

        Returns:
            True if registration successful
        """
        try:
            self.available_protocols[protocol_type] = config
            self.protocol_instances[protocol_type] = protocol_instance
            self.performance_history[protocol_type] = []

            logger.info(f"Registered protocol: {protocol_type.value}")
            return True

        except Exception as e:
            logger.error(f"Failed to register protocol {protocol_type.value}: {e}")
            return False

    def unregister_protocol(self, protocol_type: ProtocolType) -> bool:
        """
        Unregister a protocol implementation.

        Args:
            protocol_type: Type of protocol to unregister

        Returns:
            True if unregistration successful
        """
        try:
            if protocol_type in self.available_protocols:
                del self.available_protocols[protocol_type]

            if protocol_type in self.protocol_instances:
                # Disconnect if currently active
                if self.current_protocol == protocol_type:
                    self.protocol_instances[protocol_type].disconnect()
                    self.current_protocol = None

                del self.protocol_instances[protocol_type]

            if protocol_type in self.performance_history:
                del self.performance_history[protocol_type]

            logger.info(f"Unregistered protocol: {protocol_type.value}")
            return True

        except Exception as e:
            logger.error(f"Failed to unregister protocol {protocol_type.value}: {e}")
            return False

    def select_optimal_protocol(self, requirements: ProtocolRequirements) -> Optional[ProtocolType]:
        """
        Select the optimal protocol based on requirements.

        Args:
            requirements: Protocol requirements

        Returns:
            Optimal protocol type, or None if no suitable protocol found
        """
        if not self.available_protocols:
            logger.warning("No protocols available for selection")
            return None

        # Score all available protocols
        protocol_scores = []

        for protocol_type in self.available_protocols:
            score = self._score_protocol(protocol_type, requirements)
            protocol_scores.append(score)

            logger.debug(f"Protocol {protocol_type.value} scored {score.total_score:.3f}")

        # Sort by total score (descending)
        protocol_scores.sort(key=lambda x: x.total_score, reverse=True)

        # Select the best suitable protocol
        for score in protocol_scores:
            if score.is_suitable:
                logger.info(f"Selected optimal protocol: {score.protocol_type.value} " f"(score: {score.total_score:.3f})")
                return score.protocol_type

        # If no protocol is suitable, return the best available
        if protocol_scores:
            best_protocol = protocol_scores[0].protocol_type
            logger.warning(
                f"No fully suitable protocol found. Using best available: "
                f"{best_protocol.value} (score: {protocol_scores[0].total_score:.3f})"
            )
            return best_protocol

        logger.error("No protocols available for selection")
        return None

    def _score_protocol(self, protocol_type: ProtocolType, requirements: ProtocolRequirements) -> ProtocolScore:
        """
        Score a protocol against requirements.

        Args:
            protocol_type: Protocol to score
            requirements: Requirements to score against

        Returns:
            Protocol score breakdown
        """
        characteristics = self.protocol_characteristics[protocol_type]

        # Bandwidth score
        bandwidth_ratio = min(1.0, characteristics["max_bandwidth_gbps"] / requirements.bandwidth_gbps)
        bandwidth_score = bandwidth_ratio

        # Distance score
        distance_ratio = min(1.0, characteristics["max_distance_m"] / requirements.distance_m)
        distance_score = distance_ratio

        # Power over cable score
        power_score = (
            1.0
            if (characteristics["power_over_cable"] == requirements.power_over_cable or not requirements.power_over_cable)
            else 0.5
        )

        # Latency score
        latency_ratio = min(1.0, requirements.latency_ms / characteristics["typical_latency_ms"])
        latency_score = latency_ratio

        # Cost score (higher is better for lower cost)
        cost_score = characteristics["cost_factor"]

        # Reliability score
        reliability_score = characteristics["reliability_factor"]

        # Compatibility score (based on complexity - simpler is more compatible)
        compatibility_score = characteristics["complexity_factor"]

        # Calculate weighted total score
        weights = {
            "bandwidth": 0.25,
            "distance": 0.15,
            "power": 0.10,
            "latency": 0.15,
            "cost": requirements.cost_priority * 0.15,
            "reliability": requirements.reliability_priority * 0.15,
            "compatibility": 0.05,
        }

        total_score = (
            weights["bandwidth"] * bandwidth_score
            + weights["distance"] * distance_score
            + weights["power"] * power_score
            + weights["latency"] * latency_score
            + weights["cost"] * cost_score
            + weights["reliability"] * reliability_score
            + weights["compatibility"] * compatibility_score
        )

        return ProtocolScore(
            protocol_type=protocol_type,
            total_score=total_score,
            bandwidth_score=bandwidth_score,
            distance_score=distance_score,
            power_score=power_score,
            latency_score=latency_score,
            cost_score=cost_score,
            reliability_score=reliability_score,
            compatibility_score=compatibility_score,
        )

    def activate_protocol(self, protocol_type: ProtocolType) -> bool:
        """
        Activate a specific protocol.

        Args:
            protocol_type: Protocol to activate

        Returns:
            True if activation successful
        """
        if protocol_type not in self.protocol_instances:
            logger.error(f"Protocol {protocol_type.value} not registered")
            return False

        try:
            # Disconnect current protocol if active
            if self.current_protocol and self.current_protocol != protocol_type:
                current_instance = self.protocol_instances[self.current_protocol]
                current_instance.disconnect()

            # Connect new protocol
            protocol_instance = self.protocol_instances[protocol_type]
            if protocol_instance.connect():
                self.current_protocol = protocol_type
                logger.info(f"Activated protocol: {protocol_type.value}")
                return True
            else:
                logger.error(f"Failed to connect protocol: {protocol_type.value}")
                return False

        except Exception as e:
            logger.error(f"Error activating protocol {protocol_type.value}: {e}")
            return False

    def get_current_protocol(self) -> Optional[ProtocolBase]:
        """
        Get the currently active protocol instance.

        Returns:
            Current protocol instance, or None if no protocol active
        """
        if self.current_protocol:
            return self.protocol_instances.get(self.current_protocol)
        return None

    def get_protocol_recommendations(self, requirements: ProtocolRequirements) -> list[ProtocolScore]:
        """
        Get ranked protocol recommendations.

        Args:
            requirements: Protocol requirements

        Returns:
            List of protocol scores, sorted by suitability
        """
        recommendations = []

        for protocol_type in self.available_protocols:
            score = self._score_protocol(protocol_type, requirements)
            recommendations.append(score)

        # Sort by total score (descending)
        recommendations.sort(key=lambda x: x.total_score, reverse=True)

        return recommendations

    def record_performance_metrics(self, protocol_type: ProtocolType, metrics: dict[str, float]) -> None:
        """
        Record performance metrics for a protocol.

        Args:
            protocol_type: Protocol type
            metrics: Performance metrics dictionary
        """
        if protocol_type not in self.performance_history:
            self.performance_history[protocol_type] = []

        # Add timestamp to metrics
        timestamped_metrics = {"timestamp": time.time(), **metrics}

        self.performance_history[protocol_type].append(timestamped_metrics)

        # Keep only recent history (last 100 entries)
        if len(self.performance_history[protocol_type]) > 100:
            self.performance_history[protocol_type] = self.performance_history[protocol_type][-100:]

        logger.debug(f"Recorded performance metrics for {protocol_type.value}")

    def get_performance_history(self, protocol_type: ProtocolType) -> list[dict[str, float]]:
        """
        Get performance history for a protocol.

        Args:
            protocol_type: Protocol type

        Returns:
            List of performance metrics
        """
        return self.performance_history.get(protocol_type, [])

    def analyze_protocol_performance(self, protocol_type: ProtocolType) -> dict[str, Any]:
        """
        Analyze performance statistics for a protocol.

        Args:
            protocol_type: Protocol type to analyze

        Returns:
            Performance analysis results
        """
        history = self.performance_history.get(protocol_type, [])

        if not history:
            return {"error": "No performance history available"}

        # Calculate statistics
        metrics_keys = set()
        for entry in history:
            metrics_keys.update(entry.keys())
        metrics_keys.discard("timestamp")

        analysis = {}

        for metric in metrics_keys:
            values = [entry[metric] for entry in history if metric in entry]

            if values:
                analysis[metric] = {
                    "mean": sum(values) / len(values),
                    "min": min(values),
                    "max": max(values),
                    "count": len(values),
                }

                # Calculate standard deviation
                mean_val = analysis[metric]["mean"]
                variance = sum((x - mean_val) ** 2 for x in values) / len(values)
                analysis[metric]["std"] = variance**0.5

        return analysis

    def get_selector_status(self) -> dict[str, Any]:
        """
        Get current status of the protocol selector.

        Returns:
            Status information dictionary
        """
        return {
            "available_protocols": list(self.available_protocols.keys()),
            "current_protocol": self.current_protocol.value if self.current_protocol else None,
            "total_registered": len(self.available_protocols),
            "performance_history_size": {protocol.value: len(history) for protocol, history in self.performance_history.items()},
        }

    def auto_optimize_protocol(self, requirements: ProtocolRequirements) -> bool:
        """
        Automatically optimize protocol selection based on current performance.

        Args:
            requirements: Current requirements

        Returns:
            True if optimization was performed
        """
        if not self.current_protocol:
            # No current protocol, select optimal one
            optimal = self.select_optimal_protocol(requirements)
            if optimal:
                return self.activate_protocol(optimal)
            return False

        # Analyze current protocol performance
        current_analysis = self.analyze_protocol_performance(self.current_protocol)

        if "error" in current_analysis:
            # No performance data, keep current protocol
            return False

        # Check if current protocol is underperforming
        # This is a simplified check - in practice, you'd have more sophisticated logic
        current_score = self._score_protocol(self.current_protocol, requirements)

        if current_score.total_score < 0.7:  # Performance threshold
            # Look for better alternative
            optimal = self.select_optimal_protocol(requirements)

            if optimal and optimal != self.current_protocol:
                optimal_score = self._score_protocol(optimal, requirements)

                # Switch if significantly better (>10% improvement)
                if optimal_score.total_score > current_score.total_score * 1.1:
                    logger.info(f"Auto-optimizing: switching from {self.current_protocol.value} " f"to {optimal.value}")
                    return self.activate_protocol(optimal)

        return False
