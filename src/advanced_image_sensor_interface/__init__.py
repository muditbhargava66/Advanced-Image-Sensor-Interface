"""
Advanced Image Sensor Interface v2.0.0

A high-performance simulation and interface model for next-generation camera modules.
This package provides comprehensive sensor interface capabilities including MIPI CSI-2
simulation, advanced signal processing, HDR imaging, RAW processing, multi-sensor
synchronization, GPU acceleration, and advanced power management.

Version 2.0.0 Features:
- Enhanced sensor interface support (up to 8K resolution)
- HDR image processing pipeline with multiple tone mapping algorithms
- Comprehensive RAW image format support with advanced demosaicing
- Multi-sensor synchronization capabilities for stereo and multi-camera setups
- GPU acceleration support for high-performance image processing
- Advanced power states management with thermal monitoring
- Real-world scenario simulations for automotive, surveillance, and mobile applications

Note: This is a simulation and modeling framework, not a hardware driver implementation.
For hardware integration, see the documentation on interfacing with actual sensor hardware.

Modules:
    sensor_interface: Core sensor interface components with v2.0.0 enhancements
    utils: Utility functions for performance metrics and noise reduction
    test_patterns: Test pattern generation for sensor validation

Legacy Example (v1.x compatibility):
    >>> from advanced_image_sensor_interface.sensor_interface import MIPIDriver, MIPIConfig
    >>> config = MIPIConfig(lanes=4, data_rate=2.5, channel=0)
    >>> driver = MIPIDriver(config)

New v2.0.0 Example:
    >>> from advanced_image_sensor_interface.sensor_interface import EnhancedSensorInterface, create_8k_sensor_config
    >>> config = create_8k_sensor_config()
    >>> sensor = EnhancedSensorInterface(config)
    >>> sensor.start_streaming()
    >>> frame = sensor.capture_frame()
"""

# Import legacy v1.x components for backward compatibility
from .sensor_interface import AutomatedTestSuite, MIPIConfig, MIPIDriver, PowerConfig, PowerManager, SignalConfig, SignalProcessor

# Import v2.0.0 components (with graceful fallback)
try:
    from .sensor_interface import (  # Enhanced sensor interface; HDR processing; RAW processing; Multi-sensor synchronization; GPU acceleration; Advanced power management
        AdvancedPowerConfiguration,
        AdvancedPowerManager,
        BayerPattern,
        ColorSpace,
        DemosaicMethod,
        EnhancedSensorInterface,
        ExposureFusionMethod,
        GPUAccelerator,
        GPUBackend,
        GPUConfiguration,
        HDRMode,
        HDRParameters,
        HDRProcessor,
        MultiSensorSynchronizer,
        PowerMode,
        PowerState,
        ProcessingMode,
        RAWFormat,
        RAWParameters,
        RAWProcessor,
        SensorConfiguration,
        SensorResolution,
        SyncConfiguration,
        SyncMode,
        SyncStatus,
        ThermalState,
        ToneMappingMethod,
        TriggerMode,
        create_8k_sensor_config,
        create_gpu_config_for_automotive,
        create_hdr_processor_for_automotive,
        create_multi_camera_sync_config,
        create_multi_sensor_config,
        create_power_config_for_automotive,
        create_power_config_for_mobile,
        create_raw_processor_for_automotive,
        create_stereo_sync_config,
    )

    V2_FEATURES_AVAILABLE = True

    # Extended __all__ with v2.0.0 features
    __all__ = [
        # Legacy v1.x exports
        "MIPIConfig",
        "MIPIDriver",
        "PowerConfig",
        "PowerManager",
        "SignalConfig",
        "SignalProcessor",
        "AutomatedTestSuite",
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

except ImportError as e:
    # v2.0.0 features not available due to missing dependencies
    V2_FEATURES_AVAILABLE = False
    import logging

    logging.getLogger(__name__).warning(f"v2.0.0 features not available: {e}")

    # Fallback to legacy exports only
    __all__ = ["MIPIConfig", "MIPIDriver", "PowerConfig", "PowerManager", "SignalConfig", "SignalProcessor", "AutomatedTestSuite"]

# Version information
from ._version import __author__, __license__, __version__, get_release_info, get_version, get_version_info

# Backward compatibility
__version__ = __version__
__author__ = __author__
__license__ = __license__
