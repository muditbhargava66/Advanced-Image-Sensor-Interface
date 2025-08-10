"""
CoaXPress protocol implementation.

CoaXPress is a high-speed interface standard for industrial and scientific cameras,
using standard coaxial cables for both data transmission and power delivery.
"""

from .driver import CoaXPressConfig, CoaXPressDriver

__all__ = ["CoaXPressDriver", "CoaXPressConfig"]
