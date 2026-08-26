# API Documentation

## 1. Introduction

This document provides comprehensive documentation for the API of the Advanced Image Sensor Interface project (v3.1.0). It covers all supported protocols, advanced image processing, multi-sensor synchronization, calibration, AI/ML enhancements, and buffer management interfaces.

## Table of Contents

1. [Protocol Drivers](#protocol-drivers)
2. [Enhanced Sensor Interface](#enhanced-sensor-interface)
3. [Multi-Sensor Synchronization](#multi-sensor-synchronization)
4. [Image Processing](#image-processing)
5. [Buffer Management](#buffer-management)
6. [Power Management](#power-management)
7. [Data Integrity](#data-integrity)
8. [Lens Correction](#lens-correction)
9. [Protocol Enhancements](#protocol-enhancements)
10. [Configuration Management](#configuration-management)

## Protocol Drivers

### MIPI CSI-2 Driver

#### Class: MIPIDriver

```python
from advanced_image_sensor_interface.sensor_interface.protocol.mipi import MIPIDriver, MIPIConfig

config = MIPIConfig(
    lanes=4,
    data_rate_mbps=2500,
    pixel_format="RAW12",
    resolution=(1920, 1080),
    frame_rate=60
)
driver = MIPIDriver(config)
```

**Methods:**
- `send_data(data: bytes) -> bool`: Send data over MIPI interface
- `receive_data(size: int) -> Optional[bytes]`: Receive data from MIPI interface
- `get_status() -> Dict[str, Any]`: Get driver status and statistics
- `reset() -> bool`: Reset the MIPI interface

### CoaXPress Driver

#### Class: CoaXPressDriver

```python
from advanced_image_sensor_interface.sensor_interface.protocol.coaxpress import (
    CoaXPressDriver, CoaXPressConfig
)

config = CoaXPressConfig(
    speed_grade="CXP-6",
    connections=2,
    packet_size=8192,
    power_over_coax=True
)
driver = CoaXPressDriver(config)
```

**Methods:**
- `connect() -> bool`: Establish CoaXPress connection
- `disconnect() -> bool`: Disconnect from device
- `send_command(command: bytes) -> bool`: Send control command
- `capture_frame() -> Optional[np.ndarray]`: Capture single frame
- `start_streaming() -> bool`: Start continuous streaming
- `stop_streaming() -> bool`: Stop streaming

### GigE Vision Driver

#### Class: GigEDriver

```python
from advanced_image_sensor_interface.sensor_interface.protocol.gige import (
    GigEDriver, GigEConfig
)

config = GigEConfig(
    ip_address="192.168.1.100",
    packet_size=1500,
    pixel_format="BayerRG8",
    resolution=(1920, 1200)
)
driver = GigEDriver(config)
```

**Methods:**
- `discover_devices() -> List[Dict]`: Discover GigE Vision devices
- `connect(device_info: Dict) -> bool`: Connect to specific device
- `set_parameter(name: str, value: Any) -> bool`: Set camera parameter
- `get_parameter(name: str) -> Any`: Get camera parameter
- `capture_frame() -> Optional[np.ndarray]`: Capture single frame

### USB3 Vision Driver

#### Class: USB3Driver

```python
from advanced_image_sensor_interface.sensor_interface.protocol.usb3 import (
    USB3Driver, USB3Config
)

config = USB3Config(
    device_id="USB3Vision_Device",
    transfer_size=1048576,
    pixel_format="BayerGR8"
)
driver = USB3Driver(config)
```

**Methods:**
- `enumerate_devices() -> List[Dict]`: Enumerate USB3 Vision devices
- `open_device(device_id: str) -> bool`: Open specific device
- `close_device() -> bool`: Close device connection
- `start_acquisition() -> bool`: Start image acquisition
- `stop_acquisition() -> bool`: Stop image acquisition

## Enhanced Sensor Interface

### Class: EnhancedSensorInterface

The main interface for advanced sensor operations with multi-protocol support.

```python
from advanced_image_sensor_interface.sensor_interface.enhanced_sensor import (
    EnhancedSensorInterface, SensorConfiguration
)

config = SensorConfiguration(
    resolution=SensorResolution.UHD_4K,
    frame_rate=30.0,
    bit_depth=12,
    enable_hdr=True,
    enable_gpu_acceleration=True
)

sensor = EnhancedSensorInterface(config)
```

**Methods:**
- `initialize() -> bool`: Initialize sensor interface
- `capture_frame(sensor_id: int = 0) -> Optional[np.ndarray]`: Capture single frame
- `capture_synchronized_frames() -> Optional[Dict[int, np.ndarray]]`: Capture from all sensors
- `start_streaming() -> bool`: Start continuous streaming
- `stop_streaming() -> bool`: Stop streaming
- `get_sensor_status(sensor_id: Optional[int] = None) -> Dict`: Get sensor status
- `set_exposure(exposure_us: float) -> bool`: Set exposure time
- `set_gain(gain_db: float) -> bool`: Set analog gain
- `enable_hdr(enable: bool) -> bool`: Enable/disable HDR processing

## Multi-Sensor Synchronization

### Class: MultiSensorSynchronizer

Provides hardware and software synchronization for multiple sensors.

```python
from advanced_image_sensor_interface.sensor_interface.multi_sensor_sync import (
    MultiSensorSynchronizer, SyncConfiguration, SyncMode, TriggerMode
)

config = SyncConfiguration(
    sync_mode=SyncMode.HARDWARE,
    master_sensor_id=0,
    sync_tolerance_us=100,
    hardware_sync_pin=18
)

sync_manager = MultiSensorSynchronizer(config)
```

**Methods:**
- `start_synchronization() -> bool`: Start synchronized operation
- `stop_synchronization() -> bool`: Stop synchronized operation
- `capture_synchronized_frames() -> Optional[Dict[int, Tuple[np.ndarray, float]]]`: Synchronized capture
- `get_synchronization_status() -> Dict`: Get synchronization state and statistics
- `set_frame_callback(...) -> None`: Register synchronized-frame callback
- `set_sync_error_callback(...) -> None`: Register synchronization error callback
- `calibrate_sensors() -> bool`: Apply built-in geometric calibration placeholders

## Image Processing

### HDR Processing

#### Class: HDRProcessor

```python
from advanced_image_sensor_interface.sensor_interface.hdr_processing import (
    HDRProcessor, HDRParameters
)

params = HDRParameters(
    tone_mapping_method=ToneMappingMethod.REINHARD,
    exposure_fusion_method=ExposureFusionMethod.MERTENS,
    gamma=2.2,
    key_value=0.18
)

hdr_processor = HDRProcessor(params)
```

**Methods:**
- `process_hdr_sequence(images: List[np.ndarray], exposures: List[float]) -> np.ndarray`: Process HDR sequence
- `tone_map(hdr_image: np.ndarray) -> np.ndarray`: Apply tone mapping
- `exposure_fusion(images: List[np.ndarray]) -> np.ndarray`: Fuse multiple exposures
- `estimate_camera_response(images: List[np.ndarray], exposures: List[float]) -> np.ndarray`: Estimate response curve

### RAW Processing

#### Class: RAWProcessor

```python
from advanced_image_sensor_interface.sensor_interface.raw_processing import (
    RAWProcessor, RAWParameters
)

params = RAWParameters(
    bayer_pattern=BayerPattern.RGGB,
    demosaic_method=DemosaicMethod.MALVAR,
    color_matrix=np.eye(3)
)

raw_processor = RAWProcessor(params)
```

**Methods:**
- `process_raw_image(raw_data: np.ndarray, metadata: Optional[Dict] = None) -> np.ndarray`: Process RAW to RGB
- `_demosaic(raw_data: np.ndarray) -> np.ndarray`: Demosaic Bayer pattern (internal)
- `_auto_white_balance(image: np.ndarray) -> np.ndarray`: Apply white balance (internal)
- `_apply_color_correction(image: np.ndarray) -> np.ndarray`: Apply color correction
- `_apply_gamma_correction(image: np.ndarray) -> np.ndarray`: Apply gamma correction

### GPU Acceleration

#### Class: GPUAccelerator

```python
from advanced_image_sensor_interface.sensor_interface.gpu_acceleration import (
    GPUAccelerator, GPUConfiguration
)

config = GPUConfiguration(
    backend=GPUBackend.CUDA,
    device_id=0,
    memory_pool_size_mb=512,
    enable_profiling=True
)

gpu_accel = GPUAccelerator(config)
```

**Methods:**
- `process_image_batch(images: List[np.ndarray], operation: str, **kwargs) -> List[np.ndarray]`: Process image batch
- `_cpu_gaussian_blur(image: np.ndarray, sigma: float = 1.0) -> np.ndarray`: CPU-based blur
- `_cpu_edge_detection(image: np.ndarray, **kwargs) -> np.ndarray`: Sobel edge detection
- `_cpu_histogram_equalization(image: np.ndarray, **kwargs) -> np.ndarray`: Histogram equalization
- `get_performance_stats() -> Dict`: Get GPU performance statistics
- `get_device_info() -> Dict`: Get GPU device information
- `cleanup() -> None`: Release GPU resources

## Buffer Management

### Class: BufferManager

Advanced buffer management with memory pooling and statistics.

```python
from advanced_image_sensor_interface.utils.buffer_manager import (
    BufferManager, get_buffer_manager
)

# Get global buffer manager
buffer_manager = get_buffer_manager(
    pool_size=100,
    max_buffer_size=8*1024*1024,  # 8MB
    enable_statistics=True
)
```

**Methods:**
- `allocate_buffer(size: int) -> Optional[bytearray]`: Allocate buffer from pool
- `deallocate_buffer(buffer: bytearray) -> bool`: Return buffer to pool
- `get_stats() -> Optional[BufferStats]`: Get buffer usage statistics
- `clear_pool() -> bool`: Clear all buffers from pool
- `resize_pool(new_size: int) -> bool`: Resize buffer pool

### Class: AsyncBufferManager

Asynchronous buffer operations for high-performance applications.

```python
from advanced_image_sensor_interface.utils.buffer_manager import AsyncBufferManager

async_manager = AsyncBufferManager(
    pool_size=50,
    max_buffer_size=4*1024*1024
)
```

**Methods:**
- `async allocate_buffer_async(size: int) -> Optional[bytearray]`: Async buffer allocation
- `async deallocate_buffer_async(buffer: bytearray) -> bool`: Async buffer deallocation
- `async get_stats_async() -> Optional[BufferStats]`: Async statistics retrieval

### Context Manager: ManagedBuffer

Automatic buffer lifecycle management.

```python
from advanced_image_sensor_interface.utils.buffer_manager import ManagedBuffer

# Automatic buffer management
with ManagedBuffer(size=1024*1024) as buffer:
    # Use buffer for operations
    buffer[:1000] = data
    # Buffer automatically returned to pool when exiting context
```

## Power Management

### Class: AdvancedPowerManager

Advanced power management with multiple power states and thermal monitoring.

```python
from advanced_image_sensor_interface.sensor_interface.advanced_power_management import (
    AdvancedPowerManager, PowerConfiguration
)

config = PowerConfiguration(
    enable_thermal_monitoring=True,
    thermal_threshold_c=75.0,
    enable_battery_monitoring=True,
    power_optimization_mode=PowerOptimizationMode.BALANCED
)

power_manager = AdvancedPowerManager(config)
```

**Methods:**
- `transition_to_state(state: PowerState) -> bool`: Transition to power state
- `get_power_metrics() -> PowerMetrics`: Get current power metrics
- `set_component_power(component: str, enabled: bool) -> bool`: Control component power
- `optimize_for_workload(workload: WorkloadType) -> bool`: Optimize for specific workload
- `get_thermal_status() -> Dict`: Get thermal monitoring status
- `get_battery_status() -> Dict`: Get battery status (if available)

## Data Integrity

### CRCValidator

```python
from advanced_image_sensor_interface.utils.data_integrity import CRCValidator

crc = CRCValidator()
data = b"critical_image_data"
protected = crc.append_crc(data)
is_valid, original = crc.verify_crc(protected)
```

### ForwardErrorCorrection

```python
from advanced_image_sensor_interface.utils.data_integrity import (
    ForwardErrorCorrection, ErrorCorrectionMode
)

fec = ForwardErrorCorrection(mode=ErrorCorrectionMode.REED_SOLOMON)
encoded = fec.encode(data)
decoded, errors_corrected = fec.decode(encoded)
```

### IntegrityChecker

```python
from advanced_image_sensor_interface.utils.data_integrity import IntegrityChecker

checker = IntegrityChecker()
protected = checker.protect(data)
result, recovered, errors = checker.verify(protected)
```

## Lens Correction

### LensProfile

```python
from advanced_image_sensor_interface.utils.lens_correction import (
    LensProfile, LensCorrectionPipeline, STANDARD_PROFILES
)

profile = STANDARD_PROFILES["gopro_wide"]
```

### LensCorrectionPipeline

```python
pipeline = LensCorrectionPipeline(profile)
result = pipeline.correct(distorted_image)
```

## Protocol Enhancements

### MIPI D-PHY v2.5

#### Class: DPHY25Driver

```python
from advanced_image_sensor_interface.sensor_interface.protocol.mipi import (
    DPHY25Driver, DPHY25Config, EqualizationMode
)

config = DPHY25Config(
    lanes=4,
    data_rate_gbps=4.5,
    equalization=EqualizationMode.ADAPTIVE
)
driver = DPHY25Driver(config)
```

**Methods:**
- `connect() -> bool`: Establish D-PHY v2.5 connection
- `disconnect() -> bool`: Disconnect from device
- `start_streaming() -> bool`: Start continuous streaming
- `stop_streaming() -> bool`: Stop streaming
- `capture_frame() -> Optional[bytes]`: Capture single frame
- `get_statistics() -> Dict`: Get link statistics and error counts

#### Class: DPHY25Config

```python
DPHY25Config(
    lanes: int = 4,
    data_rate_gbps: float = 2.5,
    equalization: EqualizationMode = EqualizationMode.ADAPTIVE,
    continuous_clock: bool = True
)
```

- `lanes`: Number of data lanes (1-4)
- `data_rate_gbps`: Data rate per lane in Gbps (up to 4.5 for D-PHY v2.5)
- `equalization`: Adaptive equalization mode (ADAPTIVE, MANUAL, DISABLED)
- `continuous_clock`: Enable continuous clock mode

### MIPI Security Framework

#### Class: MIPISecurityManager

```python
from advanced_image_sensor_interface.sensor_interface.protocol.mipi import (
    MIPISecurityManager, SecurityCredentials, PrivilegeLevel
)

manager = MIPISecurityManager()

# Register device credentials
creds = SecurityCredentials(
    identity="camera_01",
    privilege_level=PrivilegeLevel.ADMIN,
    pre_shared_key=b"secure_key_128bit"
)
manager.register_identity(creds)

# Create authenticated session
session = manager.create_session("camera_01", b"secure_key_128bit")
print(f"Session ID: {session.session_id}")
```

**Methods:**
- `register_identity(credentials: SecurityCredentials) -> bool`: Register device identity
- `create_session(identity: str, key: bytes) -> SecuritySession`: Create authenticated session
- `verify_session(session_id: str) -> bool`: Verify session validity
- `revoke_session(session_id: str) -> bool`: Revoke active session
- `encrypt_data(session_id: str, data: bytes) -> bytes`: Encrypt data with session key
- `decrypt_data(session_id: str, data: bytes) -> bytes`: Decrypt data with session key

#### Enum: PrivilegeLevel

```python
PrivilegeLevel.GUEST
PrivilegeLevel.OPERATOR
PrivilegeLevel.ADMIN
PrivilegeLevel.SERVICE
```

### CoaXPress CXP-12

#### Class: CXP12Driver

```python
from advanced_image_sensor_interface.sensor_interface.protocol.coaxpress import (
    CXP12Driver, CXP12Config, CXPSpeed
)

config = CXP12Config(
    speed=CXPSpeed.CXP_12,
    connections=4
)
print(f"Aggregate bandwidth: {config.aggregate_bandwidth_gbps}Gbps")  # 50.0

driver = CXP12Driver(config)
driver.connect()
driver.start_streaming()
frame = driver.capture_frame()
driver.stop_streaming()
driver.disconnect()
```

**Methods:**
- `connect() -> bool`: Establish CXP-12 connection with link negotiation
- `disconnect() -> bool`: Disconnect all links
- `start_streaming() -> bool`: Start streaming on all active links
- `stop_streaming() -> bool`: Stop streaming
- `capture_frame() -> Optional[bytes]`: Capture frame from aggregated links
- `get_pocxp_status() -> PoCXPStatus`: Get Power over CoaXPress status
- `get_link_status(link_id: int) -> CXPLinkStatus`: Get individual link status

#### Enum: CXPSpeed

```python
CXPSpeed.CXP_1   # 1.25 Gbps
CXPSpeed.CXP_2   # 2.5 Gbps
CXPSpeed.CXP_3   # 3.125 Gbps
CXPSpeed.CXP_5   # 5.0 Gbps
CXPSpeed.CXP_6   # 6.25 Gbps
CXPSpeed.CXP_10  # 10.0 Gbps
CXPSpeed.CXP_12  # 12.5 Gbps
```

### GigE Vision RoCE Transport

#### Class: RoCETransport

```python
from advanced_image_sensor_interface.sensor_interface.protocol.gige import (
    RoCETransport, RoCEConfig, RoCEVersion
)

config = RoCEConfig(version=RoCEVersion.ROCE_V2, mtu=4096)
transport = RoCETransport()
transport.initialize()

# Create queue pair for zero-copy transfers
qp_num = transport.create_queue_pair()
transport.connect_qp(qp_num, remote_qp=1, remote_gid=bytes(16))
transport.send(qp_num, frame_data)

stats = transport.get_statistics()
print(f"Bytes sent: {stats.bytes_sent}")
```

**Methods:**
- `initialize() -> bool`: Initialize RoCE transport
- `create_queue_pair() -> int`: Create RDMA queue pair
- `connect_qp(qp_num: int, remote_qp: int, remote_gid: bytes) -> bool`: Connect queue pair
- `send(qp_num: int, data: bytes) -> bool`: Send data via RDMA
- `post_receive(qp_num: int, buffer: bytearray) -> bool`: Post receive buffer
- `get_statistics() -> RoCEStatistics`: Get transport statistics

### USB3 Enhanced Streaming

#### Class: USB3StreamingManager

```python
from advanced_image_sensor_interface.sensor_interface.protocol.usb3 import (
    USB3StreamingManager, StreamConfig
)

config = StreamConfig(buffer_count=10, timeout_ms=5000)
manager = USB3StreamingManager(config)

manager.prepare(width=1920, height=1080, pixel_format="Mono8")
manager.start_streaming()

frame_data, frame_info = manager.get_frame()
print(f"Frame {frame_info.frame_id}: {len(frame_data)} bytes")

stats = manager.get_statistics()
print(f"Frames captured: {stats.frames_captured}")
```

**Methods:**
- `prepare(width: int, height: int, pixel_format: str) -> bool`: Prepare streaming buffers
- `start_streaming() -> bool`: Start image acquisition
- `stop_streaming() -> bool`: Stop image acquisition
- `get_frame() -> Tuple[bytes, FrameInfo]`: Get next frame
- `get_statistics() -> StreamingStatistics`: Get streaming statistics

### USB3 Device Discovery

#### Class: USB3DeviceDiscovery

```python
from advanced_image_sensor_interface.sensor_interface.protocol.usb3 import USB3DeviceDiscovery

discovery = USB3DeviceDiscovery()
devices = discovery.enumerate()
for device in devices:
    print(f"Found: {device.manufacturer} {device.model} ({device.serial_number})")

# Filter devices
cameras = discovery.filter_by_class("camera")
```

**Methods:**
- `enumerate() -> List[USB3DeviceInfo]`: Enumerate all USB3 Vision devices
- `filter_by_class(device_class: str) -> List[USB3DeviceInfo]`: Filter by device class
- `filter_by_vendor(vendor_id: int) -> List[USB3DeviceInfo]`: Filter by vendor ID
- `watch_hotplug(callback: Callable) -> None`: Register hot-plug callback

### Protocol Selector

#### Class: ProtocolSelector

```python
from advanced_image_sensor_interface.sensor_interface.protocol_selector import (
    ProtocolSelector, ProtocolType, ProtocolRequirements
)

selector = ProtocolSelector()

# Register protocol instances
selector.register_protocol(ProtocolType.MIPI, mipi_driver, mipi_config)
selector.register_protocol(ProtocolType.GIGE, gige_driver, gige_config)
selector.register_protocol(ProtocolType.COAXPRESS, cxp_driver, cxp_config)

# Select optimal protocol
requirements = ProtocolRequirements(
    bandwidth_gbps=2.0,
    distance_m=50,
    power_over_cable=True
)
optimal = selector.select_optimal_protocol(requirements)
selector.activate_protocol(optimal)
```

**Methods:**
- `register_protocol(protocol_type: ProtocolType, driver: ProtocolBase, config: Any) -> bool`: Register protocol
- `select_optimal_protocol(requirements: ProtocolRequirements) -> ProtocolType`: Select best protocol
- `activate_protocol(protocol_type: ProtocolType) -> bool`: Activate selected protocol
- `get_current_protocol() -> Optional[ProtocolBase]`: Get active protocol driver
- `get_protocol_capabilities(protocol_type: ProtocolType) -> ProtocolCapabilities`: Get capabilities

## Configuration Management

### ConfigManager

Centralized configuration management with environment support.

```python
from advanced_image_sensor_interface.config.constants import get_config

# Get configuration for current environment
config = get_config()

# Access configuration values
mipi_limits = config.mipi
processing = config.processing
security = config.security
```

**Configuration Sections:**
- `mipi` (`MIPISystemLimits`): MIPI CSI-2 system limits
- `processing` (`ProcessingConfig`): Image processing parameters
- `security` (`SecurityConfig`): Buffer limits, power limits, timeouts
- `timing` (`TimingConfig`): Initialization delays, optimization factors
- `testing` (`QAConfiguration`): Test frame count, timeouts

### Environment-Specific Configuration

```python
from advanced_image_sensor_interface.config.constants import get_config

# Get configuration for specific environment
dev_config = get_config(environment="development")
test_config = get_config(environment="testing")
prod_config = get_config(environment="production")
```

**Available Environments:**
- `development`: Development settings with debug enabled
- `testing`: Testing configuration with mock backends
- `production`: Production settings optimized for performance

## AI/ML Enhancements (v3.1.0)

### Neural Calibration Tuner

```python
from advanced_image_sensor_interface.sensor_interface.calibration.neural_tuner import NeuralCalibrationTuner

tuner = NeuralCalibrationTuner()

# Train on calibration sessions
tuner.train(sessions)

# Predict calibration quality
quality = tuner.predict_calibration_quality(images, image_points)

# Optimize calibration parameters
params = tuner.optimize_calibration_parameters({"num_images": 10, "calibration_flags": 0})
```

**Methods:**
- `train(sessions: List[Dict]) -> bool`: Train neural network on calibration data
- `predict_calibration_quality(images: List[np.ndarray], image_points: List[np.ndarray]) -> Dict`: Predict quality metrics
- `optimize_calibration_parameters(params: Dict) -> Dict`: Optimize calibration configuration

### Scene Classifier

```python
from advanced_image_sensor_interface.examples.ai_ml_enhancements import SceneClassifier

classifier = SceneClassifier()
scene = classifier.classify(image)
# Returns: 'portrait', 'landscape', 'night', 'sports', 'macro', 'document'
```

### Noise Predictor

```python
from advanced_image_sensor_interface.examples.ai_ml_enhancements import NoisePredictor

predictor = NoisePredictor()
noise_level = predictor.predict(image)
# Returns estimated noise standard deviation
```

### Quality Assessor

```python
from advanced_image_sensor_interface.examples.ai_ml_enhancements import QualityAssessor

assessor = QualityAssessor()
metrics = assessor.assess(image)
# Returns: sharpness, noise_level, color_accuracy, overall_quality
```

### Custom Extensions

```python
from advanced_image_sensor_interface.examples.custom_extension import AINoiseReducer, AdaptiveColorCorrector

# AI-based noise reduction with scikit-learn
noise_reducer = AINoiseReducer()
denoised = noise_reducer.process(noisy_image)

# Adaptive color correction
color_corrector = AdaptiveColorCorrector()
corrected = color_corrector.process(image)
```

## Error Handling

### Exception Hierarchy

```python
from advanced_image_sensor_interface.exceptions import (
    SensorInterfaceError,
    ProtocolError,
    CalibrationError,
    BufferError,
    PowerManagementError
)

try:
    sensor.capture_frame()
except ProtocolError as e:
    print(f"Protocol error: {e}")
except SensorInterfaceError as e:
    print(f"Sensor error: {e}")
```

### Validation Results

Many methods return validation results with detailed error information:

```python
from advanced_image_sensor_interface.sensor_interface.security import ValidationResult

result = validator.validate_image_data(image)
if not result.is_valid:
    print(f"Validation failed: {result.error_message}")
    for warning in result.warnings:
        print(f"Warning: {warning}")
```

## Performance Monitoring

### Performance Metrics

```python
from advanced_image_sensor_interface.utils.performance_metrics import (
    calculate_snr,
    calculate_dynamic_range,
    calculate_color_accuracy
)

# Calculate image quality metrics
snr = calculate_snr(image, noise_image)
dynamic_range = calculate_dynamic_range(image)
color_accuracy = calculate_color_accuracy(image, reference_image)
```

### Profiling and Benchmarking

```python
from advanced_image_sensor_interface.utils.profiling import ProfileManager

with ProfileManager("image_processing") as profiler:
    processed_image = processor.process_image(raw_image)
    
# Get profiling results
stats = profiler.get_statistics()
print(f"Processing time: {stats.total_time:.3f}s")
print(f"Memory usage: {stats.peak_memory_mb:.1f}MB")
```

This comprehensive API documentation covers all major components and features of the Advanced Image Sensor Interface v3.0.0.

Retrieves the current status of the MIPI driver.

- Returns: A dictionary containing status information:
  - `lanes`: Number of MIPI lanes
  - `data_rate`: Data rate in Gbps per lane
  - `channel`: MIPI channel number
  - `error_rate`: Current error rate
  - `throughput`: Current throughput
  - `total_data_sent`: Total bytes sent
  - `total_time`: Total time spent sending data

#### optimize_performance

```python
optimize_performance() -> None
```

Optimizes the driver performance for increased data transfer rates. This increases the data rate by 40% and reduces the error rate by 50%.

### 2.2 Class: MIPIConfig

#### Constructor

```python
MIPIConfig(lanes: int, data_rate: float, channel: int)
```

- `lanes`: Number of data lanes (1-4).
- `data_rate`: Data rate in Gbps per lane.
- `channel`: MIPI channel number.

## 3. Signal Processing API

### 3.1 Class: SignalProcessor

#### Constructor

```python
SignalProcessor(config: SignalConfig)
```

- `config`: An instance of `SignalConfig` containing the processor configuration.

#### Methods

##### process_frame

```python
process_frame(frame: np.ndarray) -> np.ndarray
```

Processes a single frame of image data.

- `frame`: Input frame as a numpy array.
- Returns: Processed frame as a numpy array.
- Raises `ValueError` if `frame` is not a numpy array or has an invalid shape.

##### _apply_noise_reduction

```python
_apply_noise_reduction(frame: np.ndarray) -> np.ndarray
```

Applies noise reduction to the frame using a Gaussian blur approach.

- `frame`: Input frame as a numpy array.
- Returns: Noise-reduced frame as a numpy array.

##### _apply_dynamic_range_expansion

```python
_apply_dynamic_range_expansion(frame: np.ndarray) -> np.ndarray
```

Applies dynamic range expansion to the frame.

- `frame`: Input frame as a numpy array.
- Returns: Frame with expanded dynamic range.

##### _apply_color_correction

```python
_apply_color_correction(frame: np.ndarray) -> np.ndarray
```

Applies color correction to the frame using the color correction matrix.

- `frame`: Input frame as a numpy array.
- Returns: Color-corrected frame as a numpy array.

##### optimize_performance

```python
optimize_performance() -> None
```

Optimizes the signal processing pipeline for increased speed by reducing processing time by 20% and improving noise reduction by 10%.

### 3.2 Class: SignalConfig

#### Constructor

```python
SignalConfig(bit_depth: int, noise_reduction_strength: float, color_correction_matrix: np.ndarray)
```

- `bit_depth`: Bit depth of the image data.
- `noise_reduction_strength`: Strength of noise reduction (0.0 - 1.0).
- `color_correction_matrix`: 3x3 color correction matrix.

## 4. Power Management API

### 4.1 Class: PowerManager

#### Constructor

```python
PowerManager(config: PowerConfig)
```

- `config`: An instance of `PowerConfig` containing the power management configuration.
- Raises `ValueError` if any configuration parameter is invalid (e.g., voltage_main ≤ 0, voltage_io ≤ 0, or current_limit ≤ 0).

#### Methods

##### set_voltage

```python
set_voltage(rail: str, voltage: float) -> bool
```

Sets the voltage for a specific power rail.

- `rail`: Power rail identifier ('main' or 'io').
- `voltage`: Desired voltage in volts.
- Returns: `True` if successful, `False` otherwise.
- Raises `ValueError` if rail is not 'main' or 'io'.
- Raises `Exception` if power consumption exceeds limits.

##### get_power_status

```python
get_power_status() -> Dict[str, Any]
```

Retrieves the current power status.

- Returns: A dictionary containing power status information:
  - `voltage_main`: Main voltage in volts
  - `voltage_io`: I/O voltage in volts
  - `current_main`: Main current in amperes
  - `current_io`: I/O current in amperes
  - `power_consumption`: Total power consumption in watts
  - `temperature`: System temperature in degrees Celsius
  - `noise_level`: Noise level in power delivery

##### optimize_noise_reduction

```python
optimize_noise_reduction() -> None
```

Optimizes power delivery for reduced signal noise. This reduces noise level by 30%.

### 4.2 Class: PowerConfig

#### Constructor

```python
PowerConfig(voltage_main: float, voltage_io: float, current_limit: float)
```

- `voltage_main`: Main voltage in volts.
- `voltage_io`: I/O voltage in volts.
- `current_limit`: Current limit in amperes.

## 5. Utility Functions

### 5.1 Performance Metrics

#### calculate_snr

```python
calculate_snr(signal: np.ndarray, noise: np.ndarray) -> float
```

Calculates the Signal-to-Noise Ratio.

- `signal`: Clean signal or reference image.
- `noise`: Noise component or difference between noisy and clean signal.
- Returns: SNR in decibels.
- Raises `ValueError` if the shapes of signal and noise do not match.

#### calculate_dynamic_range

```python
calculate_dynamic_range(image: np.ndarray) -> float
```

Calculates the dynamic range of an image.

- `image`: Input image.
- Returns: Dynamic range in decibels. Returns 0 if minimum value is 0.

#### calculate_color_accuracy

```python
calculate_color_accuracy(reference_colors: np.ndarray, measured_colors: np.ndarray) -> Tuple[float, np.ndarray]
```

Calculates color accuracy using a simplified Delta E formula.

- `reference_colors`: Array of reference RGB colors.
- `measured_colors`: Array of measured RGB colors.
- Returns: Tuple of mean Delta E value and array of Delta E values for each color.
- Raises `ValueError` if the shapes of reference_colors and measured_colors do not match.

## 6. Example Usage

### 6.1 Basic Usage with Error Handling

```python
import numpy as np
from advanced_image_sensor_interface import (
    MIPIDriver, MIPIConfig,
    SignalProcessor, SignalConfig,
    PowerManager, PowerConfig,
    calculate_snr, calculate_dynamic_range, calculate_color_accuracy
)

def main():
    try:
        # Define frame parameters explicitly
        frame_size = (3840, 2160, 3)  # 4K RGB frame
        frame_bytes = frame_size[0] * frame_size[1] * frame_size[2]
        
        # Initialize MIPI Driver with error handling
        mipi_config = MIPIConfig(lanes=4, data_rate=2.5, channel=0)
        mipi_driver = MIPIDriver(mipi_config)
        print("✓ MIPI Driver initialized successfully")

        # Initialize Signal Processor with proper color correction matrix
        color_correction_matrix = np.array([
            [1.2, -0.1, -0.1],
            [-0.1, 1.2, -0.1], 
            [-0.1, -0.1, 1.2]
        ])
        signal_config = SignalConfig(
            bit_depth=12, 
            noise_reduction_strength=0.5, 
            color_correction_matrix=color_correction_matrix
        )
        signal_processor = SignalProcessor(signal_config)
        print("✓ Signal Processor initialized successfully")

        # Initialize Power Manager with validation
        power_config = PowerConfig(voltage_main=1.8, voltage_io=3.3, current_limit=1.0)
        power_manager = PowerManager(power_config)
        print("✓ Power Manager initialized successfully")

        # Generate synthetic test frame (in real usage, this comes from sensor)
        test_frame = np.random.randint(0, 4095, frame_size, dtype=np.uint16)
        print(f"✓ Generated test frame: {test_frame.shape}")

        # Process the frame with error handling
        try:
            processed_frame = signal_processor.process_frame(test_frame)
            print(f"✓ Frame processed successfully: {processed_frame.shape}")
        except ValueError as e:
            print(f"✗ Frame processing failed: {e}")
            return

        # Optimize performance
        mipi_driver.optimize_performance()
        signal_processor.optimize_performance()
        power_manager.optimize_noise_reduction()
        print("✓ Performance optimization completed")

        # Get system status with error handling
        try:
            mipi_status = mipi_driver.get_status()
            power_status = power_manager.get_power_status()
            
            print(f"MIPI Status: {mipi_status}")
            print(f"Power Status: {power_status}")
        except Exception as e:
            print(f"✗ Status retrieval failed: {e}")

        # Calculate performance metrics with proper reference data
        try:
            # Create noise component for SNR calculation
            noise_component = test_frame.astype(np.float32) - processed_frame.astype(np.float32)
            snr = calculate_snr(processed_frame.astype(np.float32), noise_component)
            
            dynamic_range = calculate_dynamic_range(processed_frame)
            
            # Create reference colors for color accuracy test
            reference_colors = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]], dtype=np.uint8)
            measured_colors = np.array([[250, 5, 5], [5, 250, 5], [5, 5, 250]], dtype=np.uint8)
            color_accuracy, delta_e_values = calculate_color_accuracy(reference_colors, measured_colors)
            
            print(f"SNR: {snr:.2f} dB")
            print(f"Dynamic Range: {dynamic_range:.2f} dB")
            print(f"Color Accuracy (Mean Delta E): {color_accuracy:.2f}")
            
        except ValueError as e:
            print(f"✗ Performance metrics calculation failed: {e}")
        except Exception as e:
            print(f"✗ Unexpected error in metrics: {e}")

    except ValueError as e:
        print(f"✗ Configuration error: {e}")
    except Exception as e:
        print(f"✗ Unexpected error: {e}")

if __name__ == "__main__":
    main()
```

### 6.2 Advanced Usage with Custom Parameters

```python
def advanced_usage_example():
    """
    Advanced usage example showing custom configurations and batch processing
    """
    try:
        # High-performance 8K configuration
        frame_size = (7680, 4320, 3)  # 8K RGB frame
        
        # High-speed MIPI configuration
        mipi_config = MIPIConfig(lanes=8, data_rate=5.0, channel=0)
        mipi_driver = MIPIDriver(mipi_config)
        
        # Advanced signal processing configuration
        advanced_ccm = np.array([
            [1.5, -0.3, -0.2],
            [-0.2, 1.4, -0.2],
            [-0.1, -0.4, 1.5]
        ])
        signal_config = SignalConfig(
            bit_depth=16,
            noise_reduction_strength=0.8,
            color_correction_matrix=advanced_ccm
        )
        signal_processor = SignalProcessor(signal_config)
        
        # High-power configuration for 8K processing
        power_config = PowerConfig(voltage_main=2.5, voltage_io=3.3, current_limit=2.0)
        power_manager = PowerManager(power_config)
        
        # Batch process multiple frames
        num_frames = 10
        processed_frames = []
        
        for i in range(num_frames):
            # Generate test frame with varying characteristics
            test_frame = np.random.randint(0, 65535, frame_size, dtype=np.uint16)
            
            try:
                processed_frame = signal_processor.process_frame(test_frame)
                processed_frames.append(processed_frame)
                print(f"✓ Processed frame {i+1}/{num_frames}")
            except Exception as e:
                print(f"✗ Failed to process frame {i+1}: {e}")
                continue
        
        print(f"✓ Successfully processed {len(processed_frames)}/{num_frames} frames")
        
        # Performance analysis
        if processed_frames:
            avg_snr = np.mean([
                calculate_snr(frame.astype(np.float32), 
                            np.random.normal(0, 10, frame.shape).astype(np.float32))
                for frame in processed_frames[:3]  # Sample first 3 frames
            ])
            print(f"Average SNR: {avg_snr:.2f} dB")
        
    except Exception as e:
        print(f"✗ Advanced usage failed: {e}")

if __name__ == "__main__":
    advanced_usage_example()
```

## 7. Error Handling

All API functions use Python's built-in exception handling mechanism. Here are the common exceptions you might encounter:

- `ValueError`: Raised when an invalid argument is passed to a function, or for invalid configurations.
- `Exception`: Raised for general errors, such as power limit exceeded.
- `RuntimeError`: Raised when an operation fails due to an unexpected condition.
- `IOError`: Raised when a hardware-related operation fails.

Example of error handling:

```python
try:
    mipi_driver.send_data(frame_data)
except ValueError as e:
    print(f"Invalid data format: {e}")
except Exception as e:
    print(f"Error occurred during data transfer: {e}")
```

## 8. Best Practices

1. **Initialization**: Always initialize the MIPI Driver, Signal Processor, and Power Manager with appropriate configurations before use.

2. **Performance Optimization**: Call the `optimize_performance()` methods after the system has been running for a while to adapt to current conditions.

3. **Error Checking**: Always check the return values of methods like `send_data()` and `set_voltage()` to ensure operations were successful.

4. **Resource Management**: Properly clean up resources when they're no longer needed.

5. **Concurrent Access**: Implement proper synchronization mechanisms when accessing objects from multiple threads.

## 9. Version Compatibility

This API documentation is for version 3.0.0 of the Advanced Image Sensor Interface project. Future versions will maintain backwards compatibility for major version numbers.
