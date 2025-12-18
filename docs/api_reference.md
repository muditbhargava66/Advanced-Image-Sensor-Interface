# Advanced Image Sensor Interface - API Reference

## Overview

The Advanced Image Sensor Interface (AISI) v2.0.0 provides a comprehensive Python framework for interfacing with high-performance image sensors across multiple protocols including MIPI CSI-2, CoaXPress, and GigE Vision.

## Core Components

### Buffer Management

#### BufferManager

High-performance buffer management with memory pooling and optimization.

```python
from advanced_image_sensor_interface.utils.buffer_manager import BufferManager

# Initialize buffer manager
manager = BufferManager(
    max_pool_size=100,
    default_buffer_size=1024*1024,  # 1MB
    enable_optimization=True
)

# Get a buffer
buffer = manager.get_buffer(size=2048*2048*2)  # 8MP image buffer
if buffer:
    # Use buffer for image data
    # ... process image ...
    
    # Return buffer to pool
    manager.return_buffer(buffer)
```

**Key Features:**
- Memory pooling for reduced allocation overhead
- Automatic pool optimization
- Thread-safe operations
- Memory usage tracking
- Statistics and monitoring

#### AsyncBufferManager

Asynchronous buffer management for high-throughput applications.

```python
import asyncio
from advanced_image_sensor_interface.utils.buffer_manager import AsyncBufferManager

async def process_frames():
    manager = AsyncBufferManager(max_pool_size=200)
    
    # Get buffer asynchronously
    buffer = await manager.get_buffer_async(size=4096*4096*2)
    if buffer:
        # Process frame data
        await process_image_data(buffer)
        
        # Return buffer
        await manager.return_buffer_async(buffer)
```

### Protocol Support

#### MIPI CSI-2 Driver

```python
from advanced_image_sensor_interface.sensor_interface.mipi_driver import MIPIDriver, MIPIConfig

# Configure MIPI interface
config = MIPIConfig(
    lanes=4,           # Number of data lanes (1-4)
    data_rate=2.5,     # Data rate in Gbps per lane
    channel=0          # Virtual channel ID (0-3)
)

# Initialize driver
driver = MIPIDriver(config)

# Get driver status
status = driver.get_status()
print(f"Throughput: {status['throughput']:.2f} Gbps")

# Send data
test_data = b"Hello MIPI!" * 100
if driver.send_data(test_data):
    print(f"Sent {len(test_data)} bytes successfully")

# Receive data  
received = driver.receive_data(len(test_data))
if received:
    print(f"Received {len(received)} bytes")
```

#### CoaXPress Driver

```python
from advanced_image_sensor_interface.sensor_interface.protocol.coaxpress.driver import (
    CoaXPressDriver, CoaXPressConfig
)

# Configure CoaXPress
config = CoaXPressConfig(
    speed_grade="CXP-6",
    connections=2,
    pixel_format="Mono16",
    resolution=(2048, 2048),
    frame_rate=60.0,
    power_over_coax=True
)

# Initialize and use driver
driver = CoaXPressDriver(config)
driver.connect()
driver.start_streaming()

# High-speed frame capture
frame = driver.capture_frame()
```

### Image Processing

#### HDR Processing

```python
from advanced_image_sensor_interface.sensor_interface.hdr_processing import HDRProcessor
import numpy as np

# Initialize HDR processor
processor = HDRProcessor()

# Process exposure stack
exposures = [
    load_image("low_exposure.raw"),
    load_image("medium_exposure.raw"), 
    load_image("high_exposure.raw")
]

hdr_image = processor.process_exposure_stack(
    exposures=exposures,
    exposure_times=[1/1000, 1/250, 1/60],  # seconds
    tone_mapping="reinhard"
)
```

#### RAW Processing

```python
from advanced_image_sensor_interface.sensor_interface.raw_processing import RAWProcessor

processor = RAWProcessor()

# Process RAW Bayer image
rgb_image = processor.process_raw_to_rgb(
    raw_image=raw_data,
    bayer_pattern="RGGB",
    demosaic_method="bilinear",
    white_balance=[1.0, 1.2, 1.1],  # R, G, B gains
    color_correction_matrix=np.eye(3)
)
```

### Power Management

#### Hardware Power Backend

```python
from advanced_image_sensor_interface.sensor_interface.power_backends import (
    create_power_backend, PowerBackendType
)

# Create hardware power backend
backend = create_power_backend(
    PowerBackendType.HARDWARE,
    config={
        "interface": "i2c",
        "address": 0x48,
        "bus": 1
    }
)

# Initialize and control power rails
if backend.initialize():
    # Set voltages
    backend.set_voltage("main", 1.8)
    backend.set_voltage("io", 3.3)
    
    # Monitor power consumption
    current = backend.get_current("main")
    voltage = backend.get_voltage("main")
    power = voltage * current if voltage and current else 0
    
    print(f"Main rail: {voltage}V, {current}A, {power}W")
```

### Security and Validation

#### Input Validation

```python
from advanced_image_sensor_interface.sensor_interface.security import SecurityManager
import numpy as np

# Initialize security manager
security = SecurityManager()

# Validate image data
image = np.random.randint(0, 255, (1920, 1080, 3), dtype=np.uint8)
is_valid = security.validate_image(image)

if is_valid:
    # Process validated image
    process_image(image)
else:
    print("Image validation failed")
```

### Multi-Sensor Synchronization

```python
from advanced_image_sensor_interface.sensor_interface.multi_sensor_sync import MultiSensorSync

# Configure synchronized capture
sync_manager = MultiSensorSync()

# Add cameras to sync group
sync_manager.add_camera("cam1", driver1)
sync_manager.add_camera("cam2", driver2)
sync_manager.add_camera("cam3", driver3)

# Configure synchronization
sync_manager.configure_sync({
    "trigger_mode": "hardware",
    "sync_tolerance_us": 100,
    "frame_rate": 30.0
})

# Start synchronized capture
sync_manager.start_sync()

# Capture synchronized frames
frames = sync_manager.capture_synchronized_frames()
for camera_id, frame_data in frames.items():
    print(f"Camera {camera_id}: {len(frame_data)} bytes")
```

## Configuration

### Environment-Based Configuration

```python
import os
from advanced_image_sensor_interface.config.constants import get_config

# Set environment
os.environ['AISI_ENVIRONMENT'] = 'production'

# Get configuration
config = get_config()

# Access configuration values
mipi_config = config.mipi
buffer_size = config.processing.default_buffer_size_mb
security_enabled = config.security.enable_validation
```

### Custom Configuration

```python
from advanced_image_sensor_interface.config.constants import ConfigManager

# Create custom configuration
config_manager = ConfigManager(environment="custom")

# Override specific settings
config_manager.mipi.data_rate_mbps = 5000
config_manager.security.max_image_size_mb = 100
config_manager.processing.enable_gpu_acceleration = True

# Use custom configuration
config = config_manager.get_config_dict()
```

## Performance Optimization

### GPU Acceleration

```python
from advanced_image_sensor_interface.sensor_interface.gpu_acceleration import GPUAccelerator

# Initialize GPU accelerator
gpu = GPUAccelerator()

if gpu.is_available():
    # GPU-accelerated image processing
    processed = gpu.process_image_batch(
        images=image_batch,
        operations=["gaussian_blur", "edge_detection"],
        parameters={"sigma": 1.5, "threshold": 0.1}
    )
else:
    # Fallback to CPU processing
    processed = cpu_process_images(image_batch)
```

### Async Processing Pipeline

```python
import asyncio
from advanced_image_sensor_interface.sensor_interface.enhanced_sensor import EnhancedSensorInterface

async def high_throughput_pipeline():
    sensor = EnhancedSensorInterface()
    
    # Configure for high throughput
    await sensor.configure_async({
        "resolution": (4096, 3072),
        "frame_rate": 120.0,
        "buffer_count": 50,
        "processing_threads": 8
    })
    
    # Start async streaming
    await sensor.start_streaming_async()
    
    # Process frames asynchronously
    async for frame in sensor.frame_stream():
        # Non-blocking frame processing
        asyncio.create_task(process_frame_async(frame))
```

## Error Handling

### Exception Hierarchy

```python
from advanced_image_sensor_interface.types import (
    SensorError, ProtocolError, BufferError, PowerError, SecurityError
)

try:
    # Sensor operations
    driver.connect()
    driver.start_streaming()
    
except ProtocolError as e:
    print(f"Protocol communication error: {e}")
    
except BufferError as e:
    print(f"Buffer management error: {e}")
    
except PowerError as e:
    print(f"Power management error: {e}")
    
except SecurityError as e:
    print(f"Security validation error: {e}")
    
except SensorError as e:
    print(f"General sensor error: {e}")
```

## Testing and Validation

### Unit Testing

```python
import pytest
from advanced_image_sensor_interface.utils.buffer_manager import BufferManager

def test_buffer_allocation():
    manager = BufferManager(max_pool_size=10)
    
    # Test buffer allocation
    buffer = manager.get_buffer(1024)
    assert buffer is not None
    assert len(buffer) == 1024
    
    # Test buffer return
    result = manager.return_buffer(buffer)
    assert result is True
    
    # Test statistics
    stats = manager.get_statistics()
    assert stats['total_allocated'] == 1
    assert stats['total_returned'] == 1
```

### Integration Testing

```python
@pytest.mark.asyncio
async def test_async_pipeline():
    from advanced_image_sensor_interface.sensor_interface.enhanced_sensor import EnhancedSensorInterface
    
    sensor = EnhancedSensorInterface()
    
    # Test async configuration
    result = await sensor.configure_async({
        "resolution": (1920, 1080),
        "frame_rate": 30.0
    })
    assert result is True
    
    # Test async streaming
    await sensor.start_streaming_async()
    
    # Capture test frames
    frames = []
    async for frame in sensor.frame_stream():
        frames.append(frame)
        if len(frames) >= 10:
            break
    
    assert len(frames) == 10
    await sensor.stop_streaming_async()
```

## Best Practices

### Memory Management

1. **Use Buffer Pools**: Always use BufferManager for frequent allocations
2. **Return Buffers**: Ensure buffers are returned to pools after use
3. **Monitor Usage**: Track memory usage with statistics
4. **Optimize Pool Sizes**: Tune pool sizes based on workload

### Performance

1. **Async Operations**: Use async APIs for high-throughput applications
2. **GPU Acceleration**: Enable GPU processing when available
3. **Batch Processing**: Process multiple frames together when possible
4. **Pipeline Parallelism**: Use multiple processing stages

### Error Handling

1. **Specific Exceptions**: Catch specific exception types
2. **Resource Cleanup**: Use context managers or try/finally blocks
3. **Graceful Degradation**: Implement fallback mechanisms
4. **Logging**: Log errors with appropriate detail levels

### Security

1. **Input Validation**: Always validate external data
2. **Buffer Bounds**: Check buffer sizes and limits
3. **Timeout Operations**: Set timeouts for long-running operations
4. **Resource Limits**: Enforce memory and processing limits