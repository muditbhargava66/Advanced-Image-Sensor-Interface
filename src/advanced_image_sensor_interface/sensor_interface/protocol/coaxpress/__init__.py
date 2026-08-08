"""
CoaXPress Protocol Package

This package provides CoaXPress protocol implementations for high-speed
industrial camera interfaces.

Modules:
    driver: Main CoaXPress driver (supports CXP-1 to CXP-12).
    cxp12: CoaXPress 2.0 / CXP-12 extension (12.5 Gbps/lane).

Example:
    >>> from advanced_image_sensor_interface.sensor_interface.protocol.coaxpress import (
    ...     CoaXPressDriver,
    ...     CoaXPressConfig,
    ...     CXP12Config,
    ...     CXP12Driver,
    ... )
"""

from .cxp12 import (
    CXP12Config,
    CXP12Driver,
    CXP12LinkManager,
    CXP12Statistics,
    CXP12TriggerController,
    CXPSpeed,
    CXPVersion,
    LinkMedium,
    TriggerMode,
)
from .driver import CoaXPressConfig, CoaXPressDriver

__all__ = [
    # Main driver
    "CoaXPressDriver",
    "CoaXPressConfig",
    # CXP-12 extension
    "CXP12Config",
    "CXP12Driver",
    "CXP12LinkManager",
    "CXP12TriggerController",
    "CXP12Statistics",
    "CXPSpeed",
    "CXPVersion",
    "LinkMedium",
    "TriggerMode",
]
