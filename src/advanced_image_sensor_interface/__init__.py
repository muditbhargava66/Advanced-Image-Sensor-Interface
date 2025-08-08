"""
Advanced Image Sensor Interface

A high-performance simulation and interface model for next-generation camera modules.
This package provides MIPI CSI-2 simulation, signal processing, and power management
capabilities for image sensor development and testing.

Note: This is a simulation and modeling framework, not a hardware driver implementation.
For hardware integration, see the documentation on interfacing with actual sensor hardware.

Modules:
    sensor_interface: Core sensor interface components including MIPI simulation
    utils: Utility functions for performance metrics and noise reduction
    test_patterns: Test pattern generation for sensor validation

Example:
    >>> from advanced_image_sensor_interface.sensor_interface import MIPIDriver, MIPIConfig
    >>> config = MIPIConfig(lanes=4, data_rate=2.5, channel=0)
    >>> driver = MIPIDriver(config)
"""

from .sensor_interface import MIPIConfig, MIPIDriver, PowerConfig, PowerManager, SignalConfig, SignalProcessor, AutomatedTestSuite

__version__ = "2.0.0"
__author__ = "Mudit Bhargava"
__license__ = "MIT"

__all__ = ["MIPIConfig", "MIPIDriver", "PowerConfig", "PowerManager", "SignalConfig", "SignalProcessor", "AutomatedTestSuite"]
