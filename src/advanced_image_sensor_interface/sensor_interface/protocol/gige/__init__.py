"""
GigE Vision Protocol Package

This package provides GigE Vision protocol implementation for image sensor interfaces.

Modules:
    driver: Main GigE Vision driver implementation.
    roce: RDMA over Converged Ethernet implementation.
"""

from .driver import GigEProtocolDriver

__all__ = ["GigEProtocolDriver"]
