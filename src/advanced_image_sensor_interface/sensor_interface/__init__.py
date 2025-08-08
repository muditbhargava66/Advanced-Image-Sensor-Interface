"""
Sensor Interface Simulation Module

This module provides simulation and modeling capabilities for CMOS image sensors,
including MIPI CSI-2 protocol simulation, power management modeling, and signal processing.

IMPORTANT: This is a simulation framework, not a hardware driver. It models the behavior
of image sensor interfaces for development, testing, and validation purposes.

Modules:
    mipi_driver: MIPI CSI-2 protocol simulation and interface modeling.
    power_management: Power delivery simulation and noise modeling.
    signal_processing: Image processing pipeline simulation.

For hardware integration guidance, see the documentation.
"""

from .mipi_driver import MIPIConfig, MIPIDriver
from .power_management import PowerConfig, PowerManager
from .signal_processing import AutomatedTestSuite, SignalConfig, SignalProcessor

__all__ = ["MIPIConfig", "MIPIDriver", "PowerConfig", "PowerManager", "SignalConfig", "SignalProcessor", "AutomatedTestSuite"]

__version__ = "1.1.0"
__author__ = "Mudit Bhargava"
__license__ = "MIT"
