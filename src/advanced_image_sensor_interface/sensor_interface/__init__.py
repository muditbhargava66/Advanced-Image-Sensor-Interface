"""
Advanced Image Sensor Interface - Sensor Interface Module

This module provides comprehensive sensor interface capabilities including:
- MIPI CSI-2 protocol implementation with D-PHY v2.5 and Security Framework
- CoaXPress CXP-12 support (50Gbps aggregate bandwidth)
- GigE Vision with RoCE transport (RDMA over Converged Ethernet)
- USB3 Vision with enhanced streaming and device discovery
- Advanced signal processing
- Power management with multi-system coordination
- Buffer management
- Image validation
- Security features

Version 3.0.0 Features:
- CoaXPress CXP-12, GigE RoCE, MIPI D-PHY v2.5, USB3 streaming
- Data integrity (CRC-32, Reed-Solomon FEC)
- Lens distortion correction pipeline
- Multi-system power management for sensor arrays
- MIPI Security Framework (AES-GCM encryption)

Previous Features (v2.0.0):
- Enhanced sensor interface (up to 8K resolution)
- HDR image processing pipeline
- RAW image format support
- Multi-sensor synchronization
- GPU acceleration support
- Advanced power states management

IMPORTANT: This is a simulation framework, not a hardware driver. It models the behavior
of image sensor interfaces for development, testing, and validation purposes.
"""

# isort: skip_file

# Legacy v1.x modules (maintained for compatibility)
from .mipi_driver import MIPIConfig, MIPIDriver
from .power_management import PowerConfig, PowerManager
from .signal_processing import AutomatedTestSuite, SignalConfig, SignalProcessor

# New v2.0.0 modules
try:
    from .advanced_power_management import (
        AdvancedPowerManager,
        PowerMode,
        PowerState,
        ThermalState,
        create_power_config_for_automotive,
        create_power_config_for_mobile,
    )
    from .advanced_power_management import PowerConfiguration as AdvancedPowerConfiguration
    from .enhanced_sensor import (
        EnhancedSensorInterface,
        HDRMode,
        RAWFormat,
        SensorConfiguration,
        SensorResolution,
        create_8k_sensor_config,
        create_multi_sensor_config,
    )
    from .gpu_acceleration import GPUAccelerator, GPUBackend, GPUConfiguration, ProcessingMode, create_gpu_config_for_automotive
    from .hdr_processing import (
        ExposureFusionMethod,
        HDRParameters,
        HDRProcessor,
        ToneMappingMethod,
        create_hdr_processor_for_automotive,
    )
    from .multi_sensor_sync import (
        MultiSensorSynchronizer,
        SyncConfiguration,
        SyncMode,
        SyncStatus,
        TriggerMode,
        create_multi_camera_sync_config,
        create_stereo_sync_config,
    )
    from .raw_processing import (
        BayerPattern,
        ColorSpace,
        DemosaicMethod,
        RAWParameters,
        RAWProcessor,
        create_raw_processor_for_automotive,
    )

    # v2.0.0 features available
    V2_FEATURES_AVAILABLE = True

except ImportError as e:
    # v2.0.0 features not available (missing dependencies)
    V2_FEATURES_AVAILABLE = False
    import logging

    logging.getLogger(__name__).warning(f"v2.0.0 features not available: {e}")

# Legacy exports (v1.x compatibility)
__all__ = ["MIPIConfig", "MIPIDriver", "PowerConfig", "PowerManager", "SignalConfig", "SignalProcessor", "AutomatedTestSuite"]

# Add v2.0.0 exports if available
if V2_FEATURES_AVAILABLE:
    __all__.extend(
        [
            # v2.0.0 Enhanced sensor interface
            "EnhancedSensorInterface",
            "SensorConfiguration",
            "SensorResolution",
            "HDRMode",
            "RAWFormat",
            "create_8k_sensor_config",
            "create_multi_sensor_config",
            # v2.0.0 HDR processing
            "HDRProcessor",
            "HDRParameters",
            "ToneMappingMethod",
            "ExposureFusionMethod",
            "create_hdr_processor_for_automotive",
            # v2.0.0 RAW processing
            "RAWProcessor",
            "RAWParameters",
            "BayerPattern",
            "DemosaicMethod",
            "ColorSpace",
            "create_raw_processor_for_automotive",
            # v2.0.0 Multi-sensor synchronization
            "MultiSensorSynchronizer",
            "SyncConfiguration",
            "SyncMode",
            "TriggerMode",
            "SyncStatus",
            "create_stereo_sync_config",
            "create_multi_camera_sync_config",
            # v2.0.0 GPU acceleration
            "GPUAccelerator",
            "GPUConfiguration",
            "GPUBackend",
            "ProcessingMode",
            "create_gpu_config_for_automotive",
            # v2.0.0 Advanced power management
            "AdvancedPowerManager",
            "AdvancedPowerConfiguration",
            "PowerState",
            "PowerMode",
            "ThermalState",
            "create_power_config_for_automotive",
            "create_power_config_for_mobile",
        ]
    )

__version__ = "3.0.0"
__author__ = "Mudit Bhargava"
__license__ = "MIT"
