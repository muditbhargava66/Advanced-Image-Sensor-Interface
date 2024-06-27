"""
Advanced Image Sensor Interface

This package provides a high-performance interface for CMOS image sensors,
including MIPI driver implementation, power management, and signal processing.

Modules:
    mipi_driver: Handles high-speed MIPI communication with the sensor.
    power_management: Manages low-noise power supply for the sensor.
    signal_processing: Processes and optimizes sensor signals.

For detailed usage instructions, please refer to the documentation.
"""

from .mipi_driver import MIPIDriver
from .power_management import PowerManager
from .signal_processing import SignalProcessor

__all__ = ['MIPIDriver', 'PowerManager', 'SignalProcessor']

__version__ = '1.0.0'
__author__ = 'Mudit Bhargava'
__license__ = 'MIT'