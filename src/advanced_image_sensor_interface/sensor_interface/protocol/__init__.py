"""
Protocol Package for Advanced Image Sensor Interface

This package provides protocol implementations for various image sensor interfaces
including MIPI CSI-2, GigE Vision, and CoaXPress.

Modules:
    base: Abstract base classes for protocol implementations.
    mipi: MIPI CSI-2 protocol implementation.
    gige: GigE Vision protocol implementation.
    coaxpress: CoaXPress protocol implementation.
"""

from .base import ProtocolBase

__all__ = ["ProtocolBase"]
