"""
Protocol implementations for camera interfaces.

This module provides support for multiple camera interface protocols:
- MIPI CSI-2: Mobile Industry Processor Interface
- CoaXPress: High-speed coaxial cable interface
- GigE Vision: Ethernet-based camera interface
- USB3 Vision: USB 3.0-based camera interface
"""

from .base import ProtocolBase, ProtocolError

__all__ = ["ProtocolBase", "ProtocolError"]
