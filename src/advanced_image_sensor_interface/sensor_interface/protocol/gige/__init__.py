"""
GigE Vision Protocol Package

This package provides GigE Vision protocol implementations for
Ethernet-based camera interfaces.

Modules:
    driver: Main GigE Vision driver (1 Gbps with 10GigE option).
    roce: RDMA over Converged Ethernet for high-performance streaming.

Example:
    >>> from advanced_image_sensor_interface.sensor_interface.protocol.gige import (
    ...     GigEProtocolDriver,
    ...     GigEVisionConfig,
    ...     RoCETransport,
    ...     GigERoCEDriver,
    ... )
"""

from .driver import GigEProtocolDriver, GigESpeed, GigEVisionConfig
from .roce import (
    GigERoCEDriver,
    MemoryRegion,
    QueuePair,
    QueuePairType,
    RDMAOperation,
    RoCEConfig,
    RoCEStatistics,
    RoCETransport,
    RoCEVersion,
)

GigEDriver = GigEProtocolDriver
GigEVisionDriver = GigEProtocolDriver
GigEConfig = GigEVisionConfig

__all__ = [
    # Main driver
    "GigEProtocolDriver",
    "GigEDriver",
    "GigEVisionDriver",
    "GigEVisionConfig",
    "GigEConfig",
    "GigESpeed",
    # RoCE
    "RoCEConfig",
    "RoCETransport",
    "RoCEStatistics",
    "RoCEVersion",
    "QueuePairType",
    "RDMAOperation",
    "QueuePair",
    "MemoryRegion",
    "GigERoCEDriver",
]
