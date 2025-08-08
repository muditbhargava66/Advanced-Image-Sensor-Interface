"""
Protocol Selector for Advanced Image Sensor Interface

This module implements adaptive protocol switching for optimal performance
based on system conditions and requirements.

Classes:
    ProtocolSelector: Main class for protocol selection and switching.
"""

import logging
from enum import Enum
from typing import Any, Optional

from .protocol.base import ProtocolBase
from .protocol.coaxpress.driver import CoaXPressProtocolDriver
from .protocol.gige.driver import GigEProtocolDriver
from .protocol.mipi.driver import MIPIProtocolDriver

logger = logging.getLogger(__name__)


class ProtocolType(Enum):
    """Enumeration of supported protocol types."""

    MIPI = "mipi"
    GIGE = "gige"
    COAXPRESS = "coaxpress"


class ProtocolSelector:
    """
    Adaptive protocol selector for optimal performance.

    This class automatically selects and switches between different
    protocols based on system conditions and performance requirements.
    """

    def __init__(self):
        """Initialize the ProtocolSelector."""
        self.current_protocol: Optional[ProtocolBase] = None
        self.current_protocol_type: Optional[ProtocolType] = None
        self.protocol_configs: dict[ProtocolType, dict[str, Any]] = {}
        self._performance_metrics: dict[ProtocolType, dict[str, float]] = {}
        logger.info("Protocol Selector initialized")

    def register_protocol_config(self, protocol_type: ProtocolType, config: dict[str, Any]) -> None:
        """
        Register configuration for a protocol type.

        Args:
            protocol_type (ProtocolType): Type of protocol to configure.
            config (Dict[str, Any]): Configuration dictionary for the protocol.
        """
        self.protocol_configs[protocol_type] = config
        logger.info(f"Registered configuration for {protocol_type.value} protocol")

    def select_optimal_protocol(self, requirements: dict[str, Any]) -> ProtocolType:
        """
        Select the optimal protocol based on requirements.

        Args:
            requirements (Dict[str, Any]): Performance and feature requirements.

        Returns:
            ProtocolType: The optimal protocol type for the given requirements.
        """
        # Simple selection logic based on requirements
        bandwidth_req = requirements.get("bandwidth_gbps", 1.0)
        requirements.get("latency_ms", 10.0)
        distance_req = requirements.get("distance_m", 1.0)

        # MIPI is best for short distance, high bandwidth
        if distance_req <= 1.0 and bandwidth_req >= 10.0:
            selected = ProtocolType.MIPI
        # CoaXPress is good for medium distance, high bandwidth
        elif distance_req <= 100.0 and bandwidth_req >= 5.0:
            selected = ProtocolType.COAXPRESS
        # GigE is best for long distance, moderate bandwidth
        else:
            selected = ProtocolType.GIGE

        logger.info(f"Selected {selected.value} protocol based on requirements")
        return selected

    def switch_protocol(self, protocol_type: ProtocolType) -> bool:
        """
        Switch to a different protocol.

        Args:
            protocol_type (ProtocolType): Protocol type to switch to.

        Returns:
            bool: True if switch was successful, False otherwise.
        """
        try:
            # Disconnect current protocol if active
            if self.current_protocol:
                self.current_protocol.disconnect()

            # Get configuration for the new protocol
            if protocol_type not in self.protocol_configs:
                logger.error(f"No configuration found for {protocol_type.value} protocol")
                return False

            config = self.protocol_configs[protocol_type]

            # Create new protocol driver
            if protocol_type == ProtocolType.MIPI:
                self.current_protocol = MIPIProtocolDriver(config)
            elif protocol_type == ProtocolType.GIGE:
                self.current_protocol = GigEProtocolDriver(config)
            elif protocol_type == ProtocolType.COAXPRESS:
                self.current_protocol = CoaXPressProtocolDriver(config)
            else:
                logger.error(f"Unsupported protocol type: {protocol_type}")
                return False

            # Connect to the new protocol
            if self.current_protocol.connect():
                self.current_protocol_type = protocol_type
                logger.info(f"Successfully switched to {protocol_type.value} protocol")
                return True
            else:
                logger.error(f"Failed to connect to {protocol_type.value} protocol")
                return False

        except Exception as e:
            logger.error(f"Error switching to {protocol_type.value} protocol: {e}")
            return False

    def get_current_protocol(self) -> Optional[ProtocolBase]:
        """
        Get the currently active protocol driver.

        Returns:
            Optional[ProtocolBase]: Current protocol driver, or None if none active.
        """
        return self.current_protocol

    def get_protocol_status(self) -> dict[str, Any]:
        """
        Get the status of the current protocol.

        Returns:
            Dict[str, Any]: Dictionary containing protocol status information.
        """
        if self.current_protocol and self.current_protocol_type:
            status = self.current_protocol.get_status()
            status["protocol_type"] = self.current_protocol_type.value
            return status
        else:
            return {"protocol_type": None, "status": "No active protocol"}

    def optimize_current_protocol(self) -> bool:
        """
        Optimize the performance of the current protocol.

        Returns:
            bool: True if optimization was successful, False otherwise.
        """
        if self.current_protocol:
            try:
                self.current_protocol.optimize_performance()
                logger.info("Current protocol performance optimized")
                return True
            except Exception as e:
                logger.error(f"Failed to optimize current protocol: {e}")
                return False
        else:
            logger.warning("No active protocol to optimize")
            return False


# Example usage
if __name__ == "__main__":
    selector = ProtocolSelector()

    # Register protocol configurations
    selector.register_protocol_config(ProtocolType.MIPI, {"lanes": 4, "data_rate": 2.5, "channel": 0})

    selector.register_protocol_config(ProtocolType.GIGE, {"ip_address": "192.168.1.100", "port": 3956})

    selector.register_protocol_config(ProtocolType.COAXPRESS, {"connection_speed": "CXP-12", "link_count": 1})

    # Select optimal protocol based on requirements
    requirements = {"bandwidth_gbps": 12.0, "latency_ms": 5.0, "distance_m": 0.5}

    optimal_protocol = selector.select_optimal_protocol(requirements)
    print(f"Optimal protocol: {optimal_protocol.value}")

    # Switch to the optimal protocol
    if selector.switch_protocol(optimal_protocol):
        status = selector.get_protocol_status()
        print(f"Protocol status: {status}")

        # Optimize performance
        selector.optimize_current_protocol()
