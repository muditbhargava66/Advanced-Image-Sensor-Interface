"""
CoaXPress Protocol Package

This package provides CoaXPress protocol implementation for image sensor interfaces.

Modules:
    driver: Main CoaXPress driver implementation.
    cxp12: CXP-12 specification implementation.
"""

from .driver import CoaXPressProtocolDriver

__all__ = ["CoaXPressProtocolDriver"]
