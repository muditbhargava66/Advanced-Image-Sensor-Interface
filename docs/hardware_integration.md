# Hardware Integration Guide

## Overview

The Advanced Image Sensor Interface provides comprehensive protocol support and can be integrated with real hardware through various adapter patterns. This guide covers integration strategies for different camera protocols and hardware platforms.

## Protocol-Specific Integration

### MIPI CSI-2 Integration

MIPI CSI-2 is the most common interface for embedded and mobile camera systems.

#### Linux V4L2 Integration

```python
import v4l2
from advanced_image_sensor_interface.sensor_interface.protocol.mipi import MIPIDriver

class V4L2MIPIAdapter:
    """Adapter for V4L2 MIPI CSI-2 cameras."""
    
    def __init__(self, device_path="/dev/video0"):
        self.device_path = device_path
        self.device = None
        self.mipi_driver = None
    
    def initialize(self, mipi_config):
        """Initialize V4L2 device with MIPI configuration."""
        self.device = v4l2.open_device(self.device_path)
        
        # Configure V4L2 device based on MIPI config
        v4l2.set_format(self.device, {
            'width': mipi_config.resolution[0],
            'height': mipi_config.resolution[1],
            'pixelformat': self._convert_pixel_format(mipi_config.pixel_format),
            'field': v4l2.V4L2_FIELD_NONE
        })
        
        # Set frame rate
        v4l2.set_fps(self.device, mipi_config.frame_rate)
        
        # Initialize our MIPI driver for processing
        self.mipi_driver = MIPIDriver(mipi_config)
        
        return True
    
    def capture_frame(self):
        """Capture frame from V4L2 device."""
        if not self.device:
            return None
            
        # Capture from V4L2
        frame_data = v4l2.capture_frame(self.device)
        
        # Process through our MIPI pipeline
        processed_frame = self.mipi_driver.process_frame_data(frame_data)
        
        return processed_frame
    
    def _convert_pixel_format(self, mipi_format):
        """Convert MIPI pixel format to V4L2 format."""
        format_map = {
            "RAW8": v4l2.V4L2_PIX_FMT_SRGGB8,
            "RAW10": v4l2.V4L2_PIX_FMT_SRGGB10,
            "RAW12": v4l2.V4L2_PIX_FMT_SRGGB12,
            "YUV422": v4l2.V4L2_PIX_FMT_YUYV
        }
        return format_map.get(mipi_format, v4l2.V4L2_PIX_FMT_SRGGB8)
```

#### libcamera Integration

```python
import libcamera
from advanced_image_sensor_interface.sensor_interface.enhanced_sensor import EnhancedSensorInterface

class LibcameraMIPIAdapter:
    """Adapter for libcamera MIPI CSI-2 integration."""
    
    def __init__(self):
        self.camera_manager = libcamera.CameraManager.singleton()
        self.camera = None
        self.sensor_interface = None
    
    def initialize(self, sensor_config):
        """Initialize libcamera with sensor configuration."""
        self.camera_manager.start()
        
        # Get first available camera
        cameras = self.camera_manager.cameras
        if not cameras:
            raise RuntimeError("No cameras found")
        
        self.camera = cameras[0]
        self.camera.acquire()
        
        # Configure camera
        config = self.camera.generate_configuration([libcamera.StreamRole.Viewfinder])
        config.at(0).size = libcamera.Size(
            sensor_config.resolution[0], 
            sensor_config.resolution[1]
        )
        config.at(0).pixel_format = self._convert_pixel_format(sensor_config.pixel_format)
        
        self.camera.configure(config)
        
        # Initialize our enhanced sensor interface
        self.sensor_interface = EnhancedSensorInterface(sensor_config)
        
        return True
    
    def start_streaming(self):
        """Start camera streaming."""
        self.camera.start()
    
    def capture_frame(self):
        """Capture and process frame."""
        request = self.camera.create_request()
        
        # Capture frame
        self.camera.queue_request(request)
        completed_request = self.camera.get_completed_request()
        
        if completed_request:
            # Get frame buffer
            stream = list(completed_request.buffers.keys())[0]
            buffer = completed_request.buffers[stream]
            
            # Convert to numpy array and process
            frame_data = self._buffer_to_numpy(buffer)
            processed_frame = self.sensor_interface.process_frame(frame_data)
            
            return processed_frame
        
        return None
```

### CoaXPress Integration

CoaXPress is used in industrial and scientific applications with specialized frame grabbers.

#### EURESYS CoaXLink Integration

```python
from advanced_image_sensor_interface.sensor_interface.protocol.coaxpress import CoaXPressDriver

class EuresysCoaXPressAdapter:
    """Adapter for EURESYS CoaXLink frame grabbers."""
    
    def __init__(self, board_index=0):
        self.board_index = board_index
        self.gentl_producer = None
        self.device = None
        self.coaxpress_driver = None
    
    def initialize(self, coaxpress_config):
        """Initialize EURESYS frame grabber."""
        try:
            # Load EURESYS GenTL producer
            import EGrabber
            
            self.gentl_producer = EGrabber.EGenTL()
            
            # Open device
            self.device = self.gentl_producer.device_open(self.board_index)
            
            # Configure CoaXPress parameters
            self._configure_coaxpress_parameters(coaxpress_config)
            
            # Initialize our CoaXPress driver
            self.coaxpress_driver = CoaXPressDriver(coaxpress_config)
            
            return True
            
        except ImportError:
            print("EURESYS EGrabber not available - using simulation mode")
            self.coaxpress_driver = CoaXPressDriver(coaxpress_config)
            return True
    
    def _configure_coaxpress_parameters(self, config):
        """Configure CoaXPress-specific parameters."""
        # Set connection speed
        speed_map = {
            "CXP-1": 1250000000,  # 1.25 Gbps
            "CXP-2": 2500000000,  # 2.5 Gbps
            "CXP-3": 3125000000,  # 3.125 Gbps
            "CXP-5": 5000000000,  # 5.0 Gbps
            "CXP-6": 6250000000,  # 6.25 Gbps
            "CXP-10": 10000000000, # 10.0 Gbps
            "CXP-12": 12500000000  # 12.5 Gbps
        }
        
        connection_speed = speed_map.get(config.speed_grade, 6250000000)
        self.device.set_integer_feature("ConnectionSpeed", connection_speed)
        
        # Configure packet size
        self.device.set_integer_feature("PacketSize", config.packet_size)
        
        # Enable power over coax if supported
        if config.power_over_coax:
            self.device.set_boolean_feature("PoCXPEnable", True)
    
    def start_acquisition(self):
        """Start image acquisition."""
        if self.device:
            self.device.set_string_feature("AcquisitionMode", "Continuous")
            self.device.execute_command("AcquisitionStart")
    
    def capture_frame(self):
        """Capture frame from CoaXPress camera."""
        if self.device:
            # Capture from hardware
            buffer = self.device.pop_output_buffer()
            frame_data = buffer.get_data()
            
            # Process through our CoaXPress pipeline
            processed_frame = self.coaxpress_driver.process_frame_data(frame_data)
            
            # Return buffer to pool
            self.device.push_input_buffer(buffer)
            
            return processed_frame
        else:
            # Simulation mode
            return self.coaxpress_driver.capture_frame()
```

### GigE Vision Integration

GigE Vision cameras use standard Ethernet infrastructure.

#### Vimba SDK Integration

```python
from advanced_image_sensor_interface.sensor_interface.protocol.gige import GigEDriver

class VimbaGigEAdapter:
    """Adapter for Allied Vision Vimba SDK."""
    
    def __init__(self):
        self.vimba = None
        self.camera = None
        self.gige_driver = None
    
    def initialize(self, gige_config):
        """Initialize Vimba SDK."""
        try:
            from vimba import Vimba
            
            self.vimba = Vimba.get_instance()
            self.vimba.startup()
            
            # Find camera by IP address
            cameras = self.vimba.get_all_cameras()
            target_camera = None
            
            for camera in cameras:
                if camera.get_ip_address() == gige_config.ip_address:
                    target_camera = camera
                    break
            
            if not target_camera:
                raise RuntimeError(f"Camera not found at {gige_config.ip_address}")
            
            self.camera = target_camera
            self.camera.open_camera()
            
            # Configure camera parameters
            self._configure_gige_parameters(gige_config)
            
            # Initialize our GigE driver
            self.gige_driver = GigEDriver(gige_config)
            
            return True
            
        except ImportError:
            print("Vimba SDK not available - using simulation mode")
            self.gige_driver = GigEDriver(gige_config)
            return True
    
    def _configure_gige_parameters(self, config):
        """Configure GigE Vision parameters."""
        # Set pixel format
        self.camera.PixelFormat.set(config.pixel_format)
        
        # Set resolution
        self.camera.Width.set(config.resolution[0])
        self.camera.Height.set(config.resolution[1])
        
        # Set frame rate
        self.camera.AcquisitionFrameRateEnable.set(True)
        self.camera.AcquisitionFrameRate.set(config.frame_rate)
        
        # Configure packet size for optimal performance
        self.camera.GevSCPSPacketSize.set(config.packet_size)
        
        # Set trigger mode
        if config.trigger_mode == "software":
            self.camera.TriggerMode.set("On")
            self.camera.TriggerSource.set("Software")
        else:
            self.camera.TriggerMode.set("Off")
    
    def start_streaming(self):
        """Start continuous streaming."""
        if self.camera:
            self.camera.start_streaming(self._frame_handler)
    
    def _frame_handler(self, camera, frame):
        """Handle incoming frames."""
        try:
            # Convert frame to numpy array
            frame_data = frame.as_numpy_ndarray()
            
            # Process through our GigE pipeline
            processed_frame = self.gige_driver.process_frame_data(frame_data)
            
            # Store or forward processed frame
            self._store_processed_frame(processed_frame)
            
        except Exception as e:
            print(f"Frame processing error: {e}")
        finally:
            camera.queue_frame(frame)
    
    def capture_single_frame(self):
        """Capture single frame."""
        if self.camera:
            # Software trigger
            self.camera.TriggerSoftware.run()
            
            # Wait for frame
            frame = self.camera.get_frame()
            frame_data = frame.as_numpy_ndarray()
            
            # Process through our pipeline
            processed_frame = self.gige_driver.process_frame_data(frame_data)
            
            return processed_frame
        else:
            # Simulation mode
            return self.gige_driver.capture_frame()
```

### USB3 Vision Integration

USB3 Vision cameras provide high-speed USB connectivity.

#### Spinnaker SDK Integration

```python
from advanced_image_sensor_interface.sensor_interface.protocol.usb3 import USB3Driver

class SpinnakerUSB3Adapter:
    """Adapter for FLIR Spinnaker SDK."""
    
    def __init__(self):
        self.system = None
        self.camera = None
        self.usb3_driver = None
    
    def initialize(self, usb3_config):
        """Initialize Spinnaker SDK."""
        try:
            import PySpin
            
            self.system = PySpin.System.GetInstance()
            
            # Get camera list
            cam_list = self.system.GetCameras()
            
            if cam_list.GetSize() == 0:
                raise RuntimeError("No USB3 Vision cameras found")
            
            # Get first camera
            self.camera = cam_list.GetByIndex(0)
            self.camera.Init()
            
            # Configure camera
            self._configure_usb3_parameters(usb3_config)
            
            # Initialize our USB3 driver
            self.usb3_driver = USB3Driver(usb3_config)
            
            return True
            
        except ImportError:
            print("Spinnaker SDK not available - using simulation mode")
            self.usb3_driver = USB3Driver(usb3_config)
            return True
    
    def _configure_usb3_parameters(self, config):
        """Configure USB3 Vision parameters."""
        # Set pixel format
        self.camera.PixelFormat.SetValue(
            self._convert_pixel_format(config.pixel_format)
        )
        
        # Set resolution
        self.camera.Width.SetValue(config.resolution[0])
        self.camera.Height.SetValue(config.resolution[1])
        
        # Set frame rate
        self.camera.AcquisitionFrameRateEnable.SetValue(True)
        self.camera.AcquisitionFrameRate.SetValue(config.frame_rate)
        
        # Configure USB3 specific settings
        self.camera.DeviceLinkThroughputLimit.SetValue(config.transfer_size)
        
        # Set acquisition mode
        self.camera.AcquisitionMode.SetValue(PySpin.AcquisitionMode_Continuous)
    
    def start_acquisition(self):
        """Start image acquisition."""
        if self.camera:
            self.camera.BeginAcquisition()
    
    def capture_frame(self):
        """Capture frame from USB3 camera."""
        if self.camera:
            # Get next image
            image_result = self.camera.GetNextImage()
            
            if image_result.IsIncomplete():
                print(f"Image incomplete: {image_result.GetImageStatus()}")
                return None
            
            # Convert to numpy array
            frame_data = image_result.GetNDArray()
            
            # Process through our USB3 pipeline
            processed_frame = self.usb3_driver.process_frame_data(frame_data)
            
            # Release image
            image_result.Release()
            
            return processed_frame
        else:
            # Simulation mode
            return self.usb3_driver.capture_frame()
```

## Platform-Specific Integration

### Embedded Linux Platforms

#### Raspberry Pi Integration

```python
from advanced_image_sensor_interface.sensor_interface.enhanced_sensor import EnhancedSensorInterface

class RaspberryPiAdapter:
    """Raspberry Pi camera integration."""
    
    def __init__(self):
        self.sensor_interface = None
        self.pi_camera = None
    
    def initialize_pi_camera(self, sensor_config):
        """Initialize Raspberry Pi camera."""
        try:
            from picamera2 import Picamera2
            
            self.pi_camera = Picamera2()
            
            # Configure camera
            camera_config = self.pi_camera.create_still_configuration(
                main={"size": sensor_config.resolution},
                raw={"size": sensor_config.resolution}
            )
            
            self.pi_camera.configure(camera_config)
            self.pi_camera.start()
            
            # Initialize our sensor interface
            self.sensor_interface = EnhancedSensorInterface(sensor_config)
            
            return True
            
        except ImportError:
            print("picamera2 not available")
            return False
    
    def capture_and_process(self):
        """Capture and process frame."""
        if self.pi_camera:
            # Capture RAW and processed images
            raw_array = self.pi_camera.capture_array("raw")
            main_array = self.pi_camera.capture_array("main")
            
            # Process through our pipeline
            if self.sensor_interface:
                processed = self.sensor_interface.process_frame(raw_array)
                return processed
            else:
                return main_array
        
        return None
```

#### NVIDIA Jetson Integration

```python
from advanced_image_sensor_interface.sensor_interface.gpu_acceleration import GPUAccelerator

class JetsonAdapter:
    """NVIDIA Jetson platform integration."""
    
    def __init__(self):
        self.gpu_accelerator = None
        self.gstreamer_pipeline = None
    
    def initialize_jetson_camera(self, sensor_config):
        """Initialize Jetson CSI camera."""
        # GStreamer pipeline for Jetson CSI camera
        pipeline = (
            f"nvarguscamerasrc sensor-id=0 ! "
            f"video/x-raw(memory:NVMM), "
            f"width={sensor_config.resolution[0]}, "
            f"height={sensor_config.resolution[1]}, "
            f"framerate={int(sensor_config.frame_rate)}/1 ! "
            f"nvvidconv ! "
            f"video/x-raw, format=BGRx ! "
            f"videoconvert ! "
            f"video/x-raw, format=BGR ! "
            f"appsink"
        )
        
        self.gstreamer_pipeline = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        
        # Initialize GPU accelerator
        gpu_config = GPUConfiguration(
            backend=GPUBackend.CUDA,
            device_id=0,
            enable_memory_pool=True
        )
        self.gpu_accelerator = GPUAccelerator(gpu_config)
        
        return self.gstreamer_pipeline.isOpened()
    
    def capture_and_accelerate(self):
        """Capture frame and apply GPU acceleration."""
        if self.gstreamer_pipeline:
            ret, frame = self.gstreamer_pipeline.read()
            
            if ret and self.gpu_accelerator:
                # Apply GPU-accelerated processing
                processed_frame = self.gpu_accelerator.process_batch(
                    [frame], "gaussian_blur", sigma=1.0
                )[0]
                return processed_frame
            
            return frame
        
        return None
```

## Real-Time Considerations

### Latency Optimization

```python
import time
from advanced_image_sensor_interface.utils.performance_metrics import PerformanceMonitor

class RealTimeProcessor:
    """Real-time image processing with latency monitoring."""
    
    def __init__(self, max_latency_ms=33.0):  # ~30 FPS
        self.max_latency_ms = max_latency_ms
        self.performance_monitor = PerformanceMonitor()
        self.frame_buffer = []
        self.processing_thread = None
    
    def process_frame_realtime(self, frame):
        """Process frame with real-time constraints."""
        start_time = time.time()
        
        try:
            # Apply processing pipeline
            processed_frame = self._apply_processing_pipeline(frame)
            
            # Check latency constraint
            processing_time_ms = (time.time() - start_time) * 1000
            
            if processing_time_ms > self.max_latency_ms:
                print(f"Warning: Processing time {processing_time_ms:.1f}ms exceeds limit")
                # Consider reducing processing complexity
                processed_frame = self._apply_reduced_processing(frame)
            
            # Update performance metrics
            self.performance_monitor.record_frame_time(processing_time_ms)
            
            return processed_frame
            
        except Exception as e:
            print(f"Real-time processing error: {e}")
            return frame  # Return original frame on error
    
    def _apply_processing_pipeline(self, frame):
        """Apply full processing pipeline."""
        # Implement your processing pipeline here
        return frame
    
    def _apply_reduced_processing(self, frame):
        """Apply reduced processing for real-time constraints."""
        # Implement simplified processing for real-time
        return frame
```

### Memory Management

```python
from advanced_image_sensor_interface.utils.buffer_manager import get_buffer_manager

class HardwareBufferManager:
    """Hardware-optimized buffer management."""
    
    def __init__(self, num_buffers=8, buffer_size_mb=8):
        self.buffer_manager = get_buffer_manager(
            pool_size=num_buffers,
            max_buffer_size=buffer_size_mb * 1024 * 1024
        )
        self.dma_buffers = []
    
    def allocate_dma_buffers(self, count, size):
        """Allocate DMA-coherent buffers for hardware."""
        for i in range(count):
            buffer = self.buffer_manager.allocate_buffer(size)
            if buffer:
                self.dma_buffers.append(buffer)
        
        return len(self.dma_buffers) == count
    
    def get_next_buffer(self):
        """Get next available buffer."""
        if self.dma_buffers:
            return self.dma_buffers.pop(0)
        return None
    
    def return_buffer(self, buffer):
        """Return buffer to pool."""
        self.dma_buffers.append(buffer)
```

## Testing and Validation

### Hardware-in-the-Loop Testing

```python
class HardwareInTheLoopTester:
    """Test framework for hardware integration."""
    
    def __init__(self, hardware_adapter):
        self.hardware_adapter = hardware_adapter
        self.test_results = []
    
    def test_frame_rate_consistency(self, duration_seconds=30):
        """Test frame rate consistency."""
        start_time = time.time()
        frame_count = 0
        frame_times = []
        
        while time.time() - start_time < duration_seconds:
            frame_start = time.time()
            frame = self.hardware_adapter.capture_frame()
            
            if frame is not None:
                frame_count += 1
                frame_times.append(time.time() - frame_start)
        
        # Analyze results
        avg_frame_time = sum(frame_times) / len(frame_times)
        actual_fps = frame_count / duration_seconds
        
        result = {
            "test": "frame_rate_consistency",
            "duration_s": duration_seconds,
            "frame_count": frame_count,
            "actual_fps": actual_fps,
            "avg_frame_time_ms": avg_frame_time * 1000,
            "frame_time_std_ms": np.std(frame_times) * 1000
        }
        
        self.test_results.append(result)
        return result
    
    def test_synchronization_accuracy(self, num_cameras=2):
        """Test multi-camera synchronization."""
        sync_errors = []
        
        for i in range(100):  # Test 100 synchronized captures
            timestamps = []
            
            # Trigger synchronized capture
            for camera_id in range(num_cameras):
                timestamp = self.hardware_adapter.capture_synchronized_frame(camera_id)
                timestamps.append(timestamp)
            
            # Calculate synchronization error
            if len(timestamps) > 1:
                max_error = max(timestamps) - min(timestamps)
                sync_errors.append(max_error * 1000)  # Convert to ms
        
        result = {
            "test": "synchronization_accuracy",
            "num_cameras": num_cameras,
            "mean_sync_error_ms": np.mean(sync_errors),
            "max_sync_error_ms": np.max(sync_errors),
            "std_sync_error_ms": np.std(sync_errors)
        }
        
        self.test_results.append(result)
        return result
```

This comprehensive hardware integration guide provides the foundation for connecting the Advanced Image Sensor Interface with real camera hardware across multiple protocols and platforms.

## Hardware Integration Strategies

### 1. Embedded Linux Platforms

#### Raspberry Pi with Camera Module

```python
# Example integration with Raspberry Pi Camera
import cv2
from advanced_image_sensor_interface import RAWProcessor, HDRProcessor

class RaspberryPiCameraAdapter:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)  # Use libcamera backend
        self.raw_processor = RAWProcessor()
        self.hdr_processor = HDRProcessor()
    
    def capture_and_process(self):
        ret, frame = self.cap.read()
        if ret:
            # Apply our processing algorithms
            processed = self.hdr_processor.process_single_image(frame)
            return processed
        return None

# Usage
adapter = RaspberryPiCameraAdapter()
processed_frame = adapter.capture_and_process()
```

#### NVIDIA Jetson with CSI-2 Camera

```python
# Example integration with Jetson CSI-2 interface
import cv2
from advanced_image_sensor_interface import EnhancedSensorInterface, GPUAccelerator

class JetsonCSI2Adapter:
    def __init__(self):
        # GStreamer pipeline for CSI-2 camera
        self.pipeline = (
            "nvarguscamerasrc sensor-id=0 ! "
            "video/x-raw(memory:NVMM), width=1920, height=1080, framerate=30/1 ! "
            "nvvidconv ! video/x-raw, format=BGRx ! "
            "videoconvert ! video/x-raw, format=BGR ! appsink"
        )
        self.cap = cv2.VideoCapture(self.pipeline, cv2.CAP_GSTREAMER)
        self.gpu_accelerator = GPUAccelerator()
    
    def capture_batch_process(self, batch_size=4):
        frames = []
        for _ in range(batch_size):
            ret, frame = self.cap.read()
            if ret:
                frames.append(frame)
        
        if frames:
            # Use GPU acceleration for batch processing
            return self.gpu_accelerator.process_image_batch(frames, "gaussian_blur")
        return []
```

### 2. Industrial Camera Integration

#### GigE Vision Cameras

```python
# Example integration with GigE Vision cameras
from advanced_image_sensor_interface import RAWProcessor, MultiSensorSynchronizer

class GigEVisionAdapter:
    def __init__(self, camera_ips):
        self.camera_ips = camera_ips
        self.raw_processor = RAWProcessor()
        # Initialize cameras (pseudo-code)
        self.cameras = [self._init_camera(ip) for ip in camera_ips]
    
    def _init_camera(self, ip):
        # Initialize GigE camera (requires vendor SDK)
        # This is pseudo-code - actual implementation depends on camera vendor
        pass
    
    def synchronized_capture(self):
        # Trigger all cameras simultaneously
        raw_frames = []
        for camera in self.cameras:
            raw_frame = camera.capture_raw()  # Vendor-specific API
            rgb_frame = self.raw_processor.process_raw_image(raw_frame)
            raw_frames.append(rgb_frame)
        return raw_frames
```

### 3. Mobile Platform Integration

#### Android Camera2 API Integration

```python
# Example integration concept for Android (via Python-for-Android)
from advanced_image_sensor_interface import HDRProcessor, AdvancedPowerManager

class AndroidCameraAdapter:
    def __init__(self):
        self.hdr_processor = HDRProcessor()
        self.power_manager = AdvancedPowerManager()
        # Initialize Android Camera2 API (requires platform-specific code)
    
    def capture_hdr_burst(self):
        # Capture multiple exposures using Camera2 API
        exposures = self._capture_exposure_bracket()  # Platform-specific
        
        # Process using our HDR algorithms
        hdr_result = self.hdr_processor.process_exposure_stack(
            exposures, [-2.0, 0.0, 2.0]
        )
        return hdr_result
    
    def optimize_power_for_mobile(self):
        # Use our power management for mobile optimization
        self.power_manager.set_power_mode(PowerMode.POWER_SAVER)
        self.power_manager.optimize_for_workload("mobile_photography")
```

## Hardware Backend Architecture

### Abstract Backend Pattern

```python
from abc import ABC, abstractmethod
from advanced_image_sensor_interface.sensor_interface import PowerManager

class HardwarePowerBackend(ABC):
    """Abstract base class for hardware power management."""
    
    @abstractmethod
    def read_voltage(self, rail: str) -> float:
        """Read actual voltage from hardware."""
        pass
    
    @abstractmethod
    def read_current(self, rail: str) -> float:
        """Read actual current from hardware."""
        pass
    
    @abstractmethod
    def set_voltage(self, rail: str, voltage: float) -> bool:
        """Set hardware voltage."""
        pass

class I2CPowerBackend(HardwarePowerBackend):
    """I2C-based power management for embedded systems."""
    
    def __init__(self, i2c_bus=1):
        try:
            import smbus
            self.bus = smbus.SMBus(i2c_bus)
        except ImportError:
            raise ImportError("smbus library required for I2C power management")
    
    def read_voltage(self, rail: str) -> float:
        # Read from I2C PMIC (Power Management IC)
        # Implementation depends on specific PMIC
        address = self._get_pmic_address(rail)
        raw_value = self.bus.read_word_data(address, 0x02)  # Example register
        return self._convert_to_voltage(raw_value)
    
    def read_current(self, rail: str) -> float:
        # Similar implementation for current reading
        pass
    
    def set_voltage(self, rail: str, voltage: float) -> bool:
        # Set voltage via I2C
        pass

class SimulatedPowerBackend(HardwarePowerBackend):
    """Simulated power backend (default)."""
    
    def read_voltage(self, rail: str) -> float:
        # Return simulated values
        return 1.8 if rail == "main" else 3.3
    
    def read_current(self, rail: str) -> float:
        return 0.5  # Simulated current
    
    def set_voltage(self, rail: str, voltage: float) -> bool:
        return True  # Always succeeds in simulation

# Enhanced PowerManager with backend support
class HardwareAwarePowerManager(PowerManager):
    def __init__(self, config, backend=None):
        super().__init__(config)
        self.backend = backend or SimulatedPowerBackend()
    
    def measure_voltage(self, rail: str) -> float:
        """Override to use hardware backend."""
        return self.backend.read_voltage(rail)
    
    def measure_current(self, rail: str) -> float:
        """Override to use hardware backend."""
        return self.backend.read_current(rail)
```

## Platform-Specific Examples

### 1. Raspberry Pi Integration

```bash
# Install required packages
sudo apt-get update
sudo apt-get install python3-opencv libcamera-dev

# Enable camera interface
sudo raspi-config  # Enable camera in interface options

# Install Python dependencies
pip install opencv-python picamera2
```

```python
# Complete Raspberry Pi example
from picamera2 import Picamera2
from advanced_image_sensor_interface import RAWProcessor, HDRProcessor
import numpy as np

class RaspberryPiImageProcessor:
    def __init__(self):
        self.picam2 = Picamera2()
        self.raw_processor = RAWProcessor()
        self.hdr_processor = HDRProcessor()
        
        # Configure camera for RAW capture
        config = self.picam2.create_still_configuration(
            main={"size": (1920, 1080)},
            raw={"size": (1920, 1080)}
        )
        self.picam2.configure(config)
        self.picam2.start()
    
    def capture_and_process_raw(self):
        # Capture RAW image
        raw_array = self.picam2.capture_array("raw")
        
        # Process using our RAW pipeline
        rgb_result = self.raw_processor.process_raw_image(raw_array)
        return rgb_result
    
    def capture_hdr_sequence(self):
        # Capture multiple exposures
        exposures = [0.01, 0.05, 0.2]  # seconds
        images = []
        
        for exposure in exposures:
            self.picam2.set_controls({"ExposureTime": int(exposure * 1000000)})
            image = self.picam2.capture_array("main")
            images.append(image)
        
        # Process HDR
        exposure_values = [-2.0, 0.0, 2.0]
        hdr_result = self.hdr_processor.process_exposure_stack(images, exposure_values)
        return hdr_result
```

### 2. NVIDIA Jetson Integration

```bash
# Install JetPack SDK
sudo apt-get install nvidia-jetpack

# Install additional dependencies
pip install opencv-python jetson-stats
```

```python
# Jetson-optimized processing
import cv2
from advanced_image_sensor_interface import GPUAccelerator
import numpy as np

class JetsonOptimizedProcessor:
    def __init__(self):
        self.gpu_accelerator = GPUAccelerator()
        
        # Configure for Jetson's CUDA capabilities
        self.pipeline = self._create_gstreamer_pipeline()
        self.cap = cv2.VideoCapture(self.pipeline, cv2.CAP_GSTREAMER)
    
    def _create_gstreamer_pipeline(self):
        return (
            "nvarguscamerasrc sensor-id=0 ! "
            "video/x-raw(memory:NVMM), width=1920, height=1080, framerate=30/1 ! "
            "nvvidconv ! video/x-raw, format=BGRx ! "
            "videoconvert ! video/x-raw, format=BGR ! appsink"
        )
    
    def process_video_stream(self):
        batch = []
        batch_size = 4
        
        while True:
            ret, frame = self.cap.read()
            if ret:
                batch.append(frame)
                
                if len(batch) >= batch_size:
                    # Process batch with GPU acceleration
                    processed = self.gpu_accelerator.process_image_batch(
                        batch, "gaussian_blur", sigma=2.0
                    )
                    
                    # Display or save processed frames
                    for proc_frame in processed:
                        cv2.imshow("Processed", proc_frame)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            return
                    
                    batch = []
```

## Performance Expectations

### Simulation vs. Hardware Performance

| Component | Simulation (Python) | Embedded ARM | x86 + GPU | FPGA/ASIC |
|-----------|-------------------|--------------|-----------|-----------|
| **RAW Processing** | ~400ms (VGA) | ~100ms | ~10ms | ~1ms |
| **HDR Processing** | ~200ms (VGA) | ~50ms | ~5ms | <1ms |
| **Multi-Sensor Sync** | ~1ms tolerance | ~100μs | ~10μs | ~1μs |
| **Power Management** | Simulated | Real measurements | Real measurements | Hardware control |

### Optimization Strategies

1. **Algorithm Optimization**
   - Use NumPy vectorized operations
   - Implement critical paths in C/C++
   - Leverage SIMD instructions

2. **Hardware Acceleration**
   - GPU processing with CUDA/OpenCL
   - DSP acceleration where available
   - FPGA implementation for critical paths

3. **Memory Optimization**
   - Minimize memory allocations
   - Use memory pools
   - Optimize data layouts

4. **Real-time Considerations**
   - Implement proper buffering
   - Use dedicated processing threads
   - Consider interrupt-driven architectures

## Testing Hardware Integration

```python
# Hardware integration test framework
import unittest
from advanced_image_sensor_interface import EnhancedSensorInterface

class HardwareIntegrationTest(unittest.TestCase):
    def setUp(self):
        # Initialize with hardware backend
        self.sensor = EnhancedSensorInterface(config)
        self.hardware_available = self._check_hardware()
    
    def _check_hardware(self):
        # Check if actual hardware is available
        try:
            # Attempt to initialize hardware
            return True
        except:
            return False
    
    @unittest.skipUnless(hardware_available, "Hardware not available")
    def test_hardware_capture(self):
        # Test actual hardware capture
        pass
    
    def test_simulation_fallback(self):
        # Test that simulation works when hardware unavailable
        pass
```

## Deployment Checklist

- [ ] **Hardware Compatibility**: Verify target platform support
- [ ] **Driver Integration**: Implement platform-specific drivers
- [ ] **Performance Validation**: Benchmark on target hardware
- [ ] **Power Management**: Integrate with hardware power controls
- [ ] **Real-time Constraints**: Validate timing requirements
- [ ] **Error Handling**: Implement robust error recovery
- [ ] **Testing**: Comprehensive hardware-in-the-loop testing
- [ ] **Documentation**: Platform-specific deployment guides

## Support and Resources

- **MIPI Alliance**: [MIPI CSI-2 Specification](https://www.mipi.org/specifications/csi-2)
- **V4L2 Documentation**: [Video4Linux2 API](https://www.kernel.org/doc/html/latest/userspace-api/media/v4l/v4l2.html)
- **libcamera**: [Modern camera stack for Linux](https://libcamera.org/)
- **OpenCV**: [Hardware acceleration guide](https://docs.opencv.org/master/d2/de6/tutorial_py_setup_in_ubuntu.html)

---

**Remember**: This simulation framework provides the algorithms and processing pipelines. Hardware integration requires platform-specific adaptation and optimization.