"""
Advanced Image Sensor Interface v3.2.0

A high-performance simulation and interface model for next-generation camera modules.
This package provides comprehensive sensor interface capabilities including MIPI CSI-2
simulation, advanced signal processing, HDR imaging, RAW processing, multi-sensor
synchronization, GPU acceleration, and advanced power management.

Version 3.2.0 Features:
- Typed ProcessingResult dataclasses (SignalProcessingResult, HDRProcessingResult,
  RAWProcessingResult, LensCorrectionResult) for explicit error handling
- SimulationDelayConfig for configurable per-driver simulated latencies
  (MIPI, GigE/RoCE, CoaXPress CXP-12, USB3)
- 3D/Depth module (StereoDepthProcessor): Block Matching and full 8-path
  semi-global matching (4 cardinal + 4 diagonal paths, optionally numba-accelerated
  with a bit-identical numpy fallback), disparity-to-depth conversion, point cloud
  generation, PLY export, and trimesh-backed mesh PLY export
- Native numpy/scipy photogrammetry calibration solver (Zhang's method + DLT),
  no OpenCV required, opt-in via SyncConfiguration.prefer_native_calibration
- Optional extras: PyWavelets and trimesh in the [full] extra (guarded imports)
- Python 3.11+ requirement with security fixes (keras, astropy)

Previous Features (v3.1.0):
- Neural Calibration Tuner with scikit-learn MLPRegressor
- AI/ML Enhancements: SceneClassifier, NoisePredictor, QualityAssessor
- Custom Extensions: AINoiseReducer, AdaptiveColorCorrector, HighSpeedMIPIDriver
- Multi-Sensor Synchronization: ORB+RANSAC feature alignment, Phase Correlation
- MIPI Security Framework: PRE_SHARED_KEY authentication

Previous Features (v3.0.0):
- CoaXPress CXP-12 support, GigE Vision RoCE transport, MIPI D-PHY v2.5
- USB3 Enhanced Streaming and Device Discovery, Data Integrity (CRC-32, RS-FEC)
- Lens Correction Pipeline, Multi-System Power Management

Note: This is a simulation and modeling framework, not a hardware driver implementation.
For hardware integration, see the documentation on interfacing with actual sensor hardware.

Modules:
    sensor_interface: Core sensor interface components with protocol drivers
    utils: Utilities including data integrity, lens correction, depth, and metrics
    test_patterns: Test pattern generation for sensor validation

v3.2.0 Typed Results Example:
    >>> from advanced_image_sensor_interface import SignalProcessor, SignalConfig
    >>> processor = SignalProcessor(SignalConfig())
    >>> result = processor.process_frame(frame)
    >>> if result.success:
    ...     processed = result.data

v3.2.0 Depth Example:
    >>> from advanced_image_sensor_interface import StereoDepthProcessor, DepthConfig
    >>> processor = StereoDepthProcessor(DepthConfig())
    >>> result = processor.compute_disparity(left_frame, right_frame)

v3.0.0 Protocol Example:
    >>> from advanced_image_sensor_interface.sensor_interface.protocol.coaxpress import CXP12Config, CXP12Driver, CXPSpeed
    >>> config = CXP12Config(speed=CXPSpeed.CXP_12, lanes=4)
    >>> driver = CXP12Driver(config)
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

    # Extended __all__ with v2.0.0+ features
    __all__ = [
        # Legacy v1.x exports
        "MIPIConfig",
        "MIPIDriver",
        "PowerConfig",
        "PowerManager",
        "SignalConfig",
        "SignalProcessor",
        "AutomatedTestSuite",
        # Enhanced sensor interface
        "EnhancedSensorInterface",
        "SensorConfiguration",
        "SensorResolution",
        "HDRMode",
        "RAWFormat",
        "create_8k_sensor_config",
        "create_multi_sensor_config",
        # HDR processing
        "HDRProcessor",
        "HDRParameters",
        "ToneMappingMethod",
        "ExposureFusionMethod",
        "create_hdr_processor_for_automotive",
        # RAW processing
        "RAWProcessor",
        "RAWParameters",
        "BayerPattern",
        "DemosaicMethod",
        "ColorSpace",
        "create_raw_processor_for_automotive",
        # Multi-sensor synchronization
        "MultiSensorSynchronizer",
        "SyncConfiguration",
        "SyncMode",
        "TriggerMode",
        "SyncStatus",
        "create_stereo_sync_config",
        "create_multi_camera_sync_config",
        # GPU acceleration
        "GPUAccelerator",
        "GPUConfiguration",
        "GPUBackend",
        "ProcessingMode",
        "create_gpu_config_for_automotive",
        # Advanced power management
        "AdvancedPowerManager",
        "AdvancedPowerConfiguration",
        "PowerState",
        "PowerMode",
        "ThermalState",
        "create_power_config_for_automotive",
        "create_power_config_for_mobile",
    ]

except ImportError as e:
    # v2.0.0+ features not available due to missing dependencies
    V2_FEATURES_AVAILABLE = False
    import logging

    logging.getLogger(__name__).error(f"v2.0.0+ features not available: {e}")

    # Fallback to legacy exports only
    __all__ = ["MIPIConfig", "MIPIDriver", "PowerConfig", "PowerManager", "SignalConfig", "SignalProcessor", "AutomatedTestSuite"]

# v3.2.0 typed results and depth module (core dependencies only)
from .types import (
    DepthResult,
    HDRProcessingResult,
    LensCorrectionResult,
    ProcessingMetrics,
    ProcessingResult,
    RAWProcessingResult,
    SignalProcessingResult,
    SimulationDelayConfig,
)
from .utils.depth import DepthConfig, DisparityAlgorithm, StereoDepthProcessor

__all__ += [
    # v3.2.0 typed processing results
    "ProcessingResult",
    "SignalProcessingResult",
    "HDRProcessingResult",
    "RAWProcessingResult",
    "LensCorrectionResult",
    "DepthResult",
    "ProcessingMetrics",
    # v3.2.0 simulation delays
    "SimulationDelayConfig",
    # v3.2.0 stereo depth
    "StereoDepthProcessor",
    "DepthConfig",
    "DisparityAlgorithm",
]

# Version information
from ._version import __author__, __license__, __version__, get_release_info, get_version, get_version_info

# Backward compatibility
__version__ = __version__
__author__ = __author__
__license__ = __license__
