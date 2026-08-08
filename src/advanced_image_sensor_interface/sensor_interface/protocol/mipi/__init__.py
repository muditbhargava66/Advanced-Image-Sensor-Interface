"""
MIPI Protocol Package

This package provides MIPI CSI-2 protocol implementation for image sensor interfaces.

Modules:
    driver: Main MIPI driver implementation (protocol-level interface).
    v4_1: MIPI D-PHY v2.5 high-speed implementation (4.5 Gbps/lane).
    security: Security framework for MIPI communication.

Example:
    >>> from advanced_image_sensor_interface.sensor_interface.protocol.mipi import (
    ...     MIPIProtocolDriver,
    ...     MIPIConfig,
    ...     DPHY25Config,
    ...     DPHY25Driver,
    ...     MIPISecurityManager,
    ... )
"""

from .driver import MIPIConfig, MIPIProtocolDriver
from .security import (
    AuthenticationMethod,
    EncryptionAlgorithm,
    MIPISecurityManager,
    PrivilegeLevel,
    SecurityConfig,
    SecurityCredentials,
    SecurityLevel,
)
from .v4_1 import DeEmphasisLevel, DPHY25Config, DPHY25Driver, DPHYVersion, EqualizationMode

MIPIDriver = MIPIProtocolDriver

__all__ = [
    # Driver
    "MIPIProtocolDriver",
    "MIPIDriver",
    "MIPIConfig",
    # D-PHY v2.5
    "DPHY25Config",
    "DPHY25Driver",
    "DPHYVersion",
    "EqualizationMode",
    "DeEmphasisLevel",
    # Security
    "MIPISecurityManager",
    "SecurityConfig",
    "SecurityCredentials",
    "SecurityLevel",
    "EncryptionAlgorithm",
    "AuthenticationMethod",
    "PrivilegeLevel",
]
