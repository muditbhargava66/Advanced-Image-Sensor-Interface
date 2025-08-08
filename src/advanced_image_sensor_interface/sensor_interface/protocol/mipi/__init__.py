"""
MIPI Protocol Package

This package provides MIPI CSI-2 protocol implementation for image sensor interfaces.

Modules:
    driver: Main MIPI driver implementation.
    v4_1: MIPI CSI-2 v4.1 specific implementation.
    security: Security framework for MIPI communication.
"""

from .driver import MIPIProtocolDriver

__all__ = ["MIPIProtocolDriver"]
