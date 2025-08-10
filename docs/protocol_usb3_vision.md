# USB3 Vision Protocol - Simple Guide

## 🤔 What is USB3 Vision? (Explain Like I'm 5)

Imagine you want to connect a really good camera to your computer, just like plugging in a USB mouse or keyboard, but this camera can take AMAZING pictures super fast! USB3 Vision is like a special language that lets your computer talk to these professional cameras through a USB cable.

**Think of it like this:**
- Your computer is like your brain
- The camera is like your eyes, but MUCH better
- USB3 Vision is like the nerves that carry messages between your eyes and brain
- The USB cable is like a super-fast highway for pictures

### 🚀 Why is it Special?

1. **Plug and Play**: Just plug it in and it works (like a USB mouse)
2. **Super Fast**: Can send huge pictures very quickly
3. **Easy to Use**: No complicated setup needed
4. **Powers the Camera**: The USB cable can power small cameras
5. **Hot Pluggable**: Can unplug and plug back in while computer is running
6. **Standard USB**: Works with regular USB 3.0 ports

## 🔧 How We Use It in Our Code

### Basic Setup (Simple Example)

```python
# Think of this as connecting a professional camera to your computer
from advanced_image_sensor_interface.sensor_interface.protocol.usb3 import (
    USB3Driver, USB3Config
)

# Configure the USB camera
config = USB3Config(
    device_id="USB3Vision_Camera",  # Camera name (like "My Camera")
    vendor_id=0x1234,              # Camera manufacturer ID
    product_id=0x5678,             # Specific camera model ID
    endpoint_address=0x81,         # USB communication channel
    transfer_size=1048576,         # How much data to send at once (1MB)
    num_transfers=8,               # Number of data streams
    pixel_format="BayerGR8",       # Type of pictures (8-bit color)
    resolution=(1280, 1024),       # Picture size
    frame_rate=60,                 # 60 pictures per second
    trigger_mode="software",       # Computer tells camera when to take pictures
    exposure_auto=True,            # Camera adjusts brightness automatically
    gain_auto=True                 # Camera adjusts sensitivity automatically
)

# Connect to the USB camera
camera = USB3Driver(config)
```

### Taking USB Pictures

```python
# Start the camera
camera.start_streaming()

# Take a picture through USB
picture = camera.capture_frame()

# The picture came through the USB cable!
print(f"Got a USB picture: {picture.width} x {picture.height} pixels")
print(f"Camera model: {camera.get_device_info().model_name}")

# Stop the camera when done
camera.stop_streaming()
```

### Advanced Usage - Multiple USB Cameras

```python
# Connect multiple USB cameras to one computer
def setup_multiple_usb_cameras():
    """Set up multiple USB3 Vision cameras"""
    
    # Different cameras have different IDs
    camera_configs = [
        {
            "name": "Front Camera",
            "vendor_id": 0x1234,
            "product_id": 0x5678,
            "resolution": (1920, 1080),
            "frame_rate": 30
        },
        {
            "name": "Side Camera", 
            "vendor_id": 0x1234,
            "product_id": 0x5679,
            "resolution": (1280, 720),
            "frame_rate": 60
        },
        {
            "name": "Top Camera",
            "vendor_id": 0x1234, 
            "product_id": 0x567A,
            "resolution": (640, 480),
            "frame_rate": 120
        }
    ]
    
    cameras = {}
    
    # Set up each camera
    for cam_info in camera_configs:
        config = USB3Config(
            device_id=cam_info["name"],
            vendor_id=cam_info["vendor_id"],
            product_id=cam_info["product_id"],
            endpoint_address=0x81,
            transfer_size=1048576,
            num_transfers=4,  # Fewer transfers per camera to share USB bandwidth
            pixel_format="BayerGR8",
            resolution=cam_info["resolution"],
            frame_rate=cam_info["frame_rate"],
            trigger_mode="software",
            exposure_auto=True,
            gain_auto=True
        )
        
        camera = USB3Driver(config)
        cameras[cam_info["name"]] = camera
        
        print(f"✅ Connected {cam_info['name']}")
    
    return cameras

# Use multiple cameras
cameras = setup_multiple_usb_cameras()

# Start all cameras
for name, camera in cameras.items():
    camera.start_streaming()
    print(f"📹 {name} started")

# Take pictures from all cameras
all_pictures = {}
for name, camera in cameras.items():
    picture = camera.capture_frame()
    all_pictures[name] = picture
    print(f"📸 Got picture from {name}")

print(f"Captured {len(all_pictures)} pictures simultaneously!")
```

## 🔌 USB Configuration (Making It Work Better)

### USB Transfer Optimization
```python
# Configure USB transfers for best performance
usb_settings = {
    "bulk_transfer_size": 1048576,  # 1MB chunks (good for high resolution)
    "iso_transfer_size": 32768,     # 32KB chunks (good for real-time)
    "transfer_timeout": 1000,       # 1 second timeout
    "reset_on_error": True,         # Reset USB connection if error
    "power_management": False       # Don't let USB go to sleep
}

# Apply USB optimization
config = USB3Config(
    device_id="High_Performance_Camera",
    transfer_size=usb_settings["bulk_transfer_size"],
    num_transfers=8,  # Multiple transfers for smooth streaming
    **usb_settings
)

camera = USB3Driver(config)

# Check USB connection quality
usb_status = camera.get_usb_status()
print(f"USB connection speed: {usb_status.speed}")  # Should be "SuperSpeed" for USB 3.0
print(f"USB power available: {usb_status.power_available}mA")
```

### Automatic Camera Detection
```python
def find_usb3_cameras():
    """Automatically find all USB3 Vision cameras"""
    
    from advanced_image_sensor_interface.sensor_interface.protocol.usb3.discovery import discover_usb3_cameras
    
    print("🔍 Searching for USB3 Vision cameras...")
    
    # Scan all USB ports
    cameras = discover_usb3_cameras()
    
    if cameras:
        print(f"✅ Found {len(cameras)} USB3 Vision cameras:")
        
        for i, camera_info in enumerate(cameras):
            print(f"  Camera {i+1}:")
            print(f"    Name: {camera_info.device_name}")
            print(f"    Vendor: {camera_info.vendor_name}")
            print(f"    Model: {camera_info.model_name}")
            print(f"    Serial: {camera_info.serial_number}")
            print(f"    USB Port: {camera_info.usb_port}")
    else:
        print("❌ No USB3 Vision cameras found")
        print("Check:")
        print("  - Camera is plugged in")
        print("  - USB 3.0 port (blue connector)")
        print("  - Camera drivers installed")
        print("  - Camera is powered on")
    
    return cameras

# Find and connect to cameras automatically
available_cameras = find_usb3_cameras()

# Connect to first available camera
if available_cameras:
    first_camera = available_cameras[0]
    
    config = USB3Config(
        device_id=first_camera.device_name,
        vendor_id=first_camera.vendor_id,
        product_id=first_camera.product_id
    )
    
    camera = USB3Driver(config)
    print(f"✅ Connected to {first_camera.device_name}")
```

## 🎮 Real-World Examples

### Example 1: Desktop Microscope Camera
```python
class MicroscopeCamera:
    def __init__(self):
        self.config = USB3Config(
            device_id="Microscope_Camera",
            vendor_id=0x1234,
            product_id=0x5678,
            pixel_format="Mono12",          # 12-bit grayscale for scientific accuracy
            resolution=(2048, 1536),        # High resolution for detail
            frame_rate=15,                  # Slower for high quality
            trigger_mode="software",        # Manual control
            exposure_auto=False,            # Manual exposure for consistency
            gain_auto=False,                # Manual gain for accuracy
            exposure_time=50000,            # 50ms exposure for good light
            gain=1.0                        # No gain for accurate measurements
        )
        
        self.camera = USB3Driver(self.config)
        self.current_magnification = 1
    
    def start_live_view(self):
        """Start live microscope view"""
        self.camera.start_streaming()
        print("🔬 Microscope live view started")
        
        while True:
            # Get live image
            live_image = self.camera.capture_frame()
            
            # Display on screen (this would show on monitor)
            self.display_image(live_image)
            
            # Check if user wants to capture
            if self.user_pressed_capture():
                captured_image = self.capture_high_quality_image()
                self.save_image(captured_image)
                print("📸 High-quality image captured")
    
    def capture_high_quality_image(self):
        """Capture highest quality image for analysis"""
        # Switch to highest quality settings
        high_quality_config = USB3Config(
            device_id="Microscope_Camera",
            vendor_id=0x1234,
            product_id=0x5678,
            pixel_format="Mono16",          # 16-bit for maximum quality
            resolution=(4096, 3072),        # Maximum resolution
            frame_rate=1,                   # Very slow for maximum quality
            exposure_time=100000,           # Long exposure for best image
            gain=0.5                        # Low gain for low noise
        )
        
        # Temporarily reconfigure camera
        self.camera.reconfigure(high_quality_config)
        
        # Take high-quality picture
        hq_image = self.camera.capture_frame()
        
        # Switch back to live view settings
        self.camera.reconfigure(self.config)
        
        return hq_image
    
    def measure_specimen(self, image):
        """Measure specimen in microscope image"""
        # Use computer vision to measure objects
        measurements = self.analyze_image(image)
        
        # Convert pixels to real measurements based on magnification
        real_measurements = {}
        for feature, pixels in measurements.items():
            # Convert pixels to micrometers
            micrometers = pixels * self.get_pixel_size_um()
            real_measurements[feature] = micrometers
        
        return real_measurements
```

### Example 2: Quality Control Camera
```python
class QualityControlCamera:
    def __init__(self, station_name):
        self.station_name = station_name
        self.config = USB3Config(
            device_id=f"QC_Camera_{station_name}",
            vendor_id=0x1234,
            product_id=0x5678,
            pixel_format="BayerRG8",        # Color for defect detection
            resolution=(1920, 1080),        # Full HD for detail
            frame_rate=30,                  # Fast for production line
            trigger_mode="hardware",        # Triggered by conveyor belt
            exposure_auto=True,             # Auto adjust for lighting changes
            gain_auto=True,                 # Auto adjust for different products
            transfer_size=2097152,          # 2MB transfers for smooth operation
            num_transfers=6                 # Multiple transfers for no delays
        )
        
        self.camera = USB3Driver(self.config)
        self.defect_count = 0
        self.total_inspected = 0
    
    def start_quality_inspection(self):
        """Start automated quality inspection"""
        self.camera.start_streaming()
        print(f"🏭 Quality control started at {self.station_name}")
        
        while production_line_running():
            # Wait for product to arrive (hardware trigger)
            product_image = self.camera.capture_frame()
            self.total_inspected += 1
            
            # Inspect product for defects
            inspection_result = self.inspect_product(product_image)
            
            if inspection_result.has_defects:
                self.defect_count += 1
                self.reject_product(inspection_result.defects)
                print(f"❌ Product {self.total_inspected} rejected: {inspection_result.defects}")
                
                # Take detailed image of defect for analysis
                defect_image = self.capture_defect_detail(product_image, inspection_result.defect_locations)
                self.save_defect_image(defect_image)
            else:
                self.accept_product()
                print(f"✅ Product {self.total_inspected} passed inspection")
            
            # Report statistics every 100 products
            if self.total_inspected % 100 == 0:
                defect_rate = (self.defect_count / self.total_inspected) * 100
                print(f"📊 Defect rate: {defect_rate:.1f}% ({self.defect_count}/{self.total_inspected})")
    
    def inspect_product(self, image):
        """Inspect product for various defects"""
        defects = []
        
        # Check for scratches
        if self.detect_scratches(image):
            defects.append("scratches")
        
        # Check for dents
        if self.detect_dents(image):
            defects.append("dents")
        
        # Check for color variations
        if self.detect_color_defects(image):
            defects.append("color_variation")
        
        # Check for missing parts
        if self.detect_missing_parts(image):
            defects.append("missing_parts")
        
        return InspectionResult(
            has_defects=len(defects) > 0,
            defects=defects,
            defect_locations=self.find_defect_locations(image, defects)
        )
```

### Example 3: Sports Analysis Camera
```python
class SportsAnalysisCamera:
    def __init__(self, sport_type):
        self.sport_type = sport_type
        
        # Configure based on sport requirements
        if sport_type == "tennis":
            frame_rate = 120  # High speed for ball tracking
            resolution = (1280, 720)  # HD for speed
        elif sport_type == "golf":
            frame_rate = 240  # Very high speed for swing analysis
            resolution = (1024, 768)  # Lower resolution for extreme speed
        elif sport_type == "swimming":
            frame_rate = 60   # Standard speed
            resolution = (1920, 1080)  # Full HD for stroke analysis
        else:
            frame_rate = 60
            resolution = (1920, 1080)
        
        self.config = USB3Config(
            device_id=f"Sports_Camera_{sport_type}",
            vendor_id=0x1234,
            product_id=0x5678,
            pixel_format="BayerRG8",
            resolution=resolution,
            frame_rate=frame_rate,
            trigger_mode="software",
            exposure_auto=False,            # Manual for consistent lighting
            gain_auto=False,                # Manual for consistent quality
            exposure_time=2000,             # Fast exposure to freeze motion
            gain=1.5,                       # Boost for fast exposure
            transfer_size=1048576,          # 1MB for high-speed transfers
            num_transfers=12                # Many transfers for smooth high-speed
        )
        
        self.camera = USB3Driver(self.config)
    
    def analyze_tennis_serve(self):
        """Analyze tennis serve technique"""
        print("🎾 Starting tennis serve analysis")
        
        self.camera.start_streaming()
        
        # Wait for serve to start
        print("Waiting for serve...")
        self.wait_for_motion_start()
        
        # Capture high-speed sequence
        serve_sequence = []
        for i in range(60):  # 0.5 seconds at 120fps
            frame = self.camera.capture_frame()
            serve_sequence.append(frame)
        
        # Analyze the serve
        analysis = self.analyze_serve_sequence(serve_sequence)
        
        print(f"📊 Serve Analysis:")
        print(f"  Ball speed: {analysis.ball_speed_mph} mph")
        print(f"  Racket speed: {analysis.racket_speed_mph} mph")
        print(f"  Contact point height: {analysis.contact_height_ft} ft")
        print(f"  Spin rate: {analysis.spin_rpm} rpm")
        print(f"  Accuracy: {analysis.accuracy_score}/10")
        
        return analysis
    
    def analyze_golf_swing(self):
        """Analyze golf swing technique"""
        print("⛳ Starting golf swing analysis")
        
        self.camera.start_streaming()
        
        # Wait for swing to start
        print("Waiting for swing...")
        self.wait_for_motion_start()
        
        # Capture ultra-high-speed sequence
        swing_sequence = []
        for i in range(120):  # 0.5 seconds at 240fps
            frame = self.camera.capture_frame()
            swing_sequence.append(frame)
        
        # Analyze the swing
        analysis = self.analyze_swing_sequence(swing_sequence)
        
        print(f"📊 Swing Analysis:")
        print(f"  Club head speed: {analysis.club_speed_mph} mph")
        print(f"  Ball speed: {analysis.ball_speed_mph} mph")
        print(f"  Launch angle: {analysis.launch_angle_degrees}°")
        print(f"  Swing plane: {analysis.swing_plane_score}/10")
        print(f"  Tempo: {analysis.tempo_ratio}")
        
        return analysis
```

## 🔧 USB Troubleshooting

### Common USB Issues
```python
def diagnose_usb_issues():
    """Diagnose common USB3 Vision camera issues"""
    
    print("🔍 Diagnosing USB3 Vision camera issues...")
    
    # Check 1: USB port speed
    import usb.core
    import usb.util
    
    # Find USB devices
    devices = usb.core.find(find_all=True)
    usb3_devices = []
    
    for device in devices:
        try:
            if device.bcdUSB >= 0x0300:  # USB 3.0 or higher
                usb3_devices.append(device)
        except:
            pass
    
    if usb3_devices:
        print(f"✅ Found {len(usb3_devices)} USB 3.0+ devices")
    else:
        print("❌ No USB 3.0 devices found")
        print("Check: USB 3.0 port (blue connector), USB 3.0 cable")
        return
    
    # Check 2: Available bandwidth
    total_bandwidth = 0
    for device in usb3_devices:
        # Estimate bandwidth usage (this is simplified)
        bandwidth = 100  # Assume 100 Mbps per device
        total_bandwidth += bandwidth
    
    print(f"📊 Estimated USB bandwidth usage: {total_bandwidth} Mbps")
    
    if total_bandwidth > 4000:  # USB 3.0 limit is ~5000 Mbps
        print("⚠️ High USB bandwidth usage")
        print("Consider: Reduce resolution, frame rate, or number of cameras")
    
    # Check 3: Power consumption
    total_power = len(usb3_devices) * 500  # Assume 500mA per device
    
    print(f"🔋 Estimated USB power usage: {total_power} mA")
    
    if total_power > 900:  # USB 3.0 limit is 900mA
        print("⚠️ High USB power usage")
        print("Consider: External power supply, powered USB hub")
    
    # Check 4: Driver status
    try:
        # Try to connect to a camera
        test_config = USB3Config(
            device_id="Test_Camera",
            vendor_id=0x1234,
            product_id=0x5678
        )
        
        test_camera = USB3Driver(test_config)
        device_info = test_camera.get_device_info()
        
        print("✅ Camera driver working")
        print(f"  Camera: {device_info.model_name}")
        print(f"  Driver version: {device_info.driver_version}")
        
    except Exception as e:
        print(f"❌ Camera driver issue: {e}")
        print("Check: Camera drivers installed, camera permissions")

# Run diagnostics
diagnose_usb_issues()
```

### Performance Optimization
```python
def optimize_usb3_performance(cameras):
    """Optimize USB3 performance for multiple cameras"""
    
    num_cameras = len(cameras)
    print(f"🚀 Optimizing USB3 performance for {num_cameras} cameras")
    
    # Calculate optimal settings
    if num_cameras == 1:
        # Single camera - use maximum performance
        optimal_settings = {
            "transfer_size": 2097152,  # 2MB
            "num_transfers": 8,
            "frame_rate_multiplier": 1.0
        }
    elif num_cameras <= 3:
        # Few cameras - good performance
        optimal_settings = {
            "transfer_size": 1048576,  # 1MB
            "num_transfers": 6,
            "frame_rate_multiplier": 0.8
        }
    else:
        # Many cameras - share bandwidth
        optimal_settings = {
            "transfer_size": 524288,   # 512KB
            "num_transfers": 4,
            "frame_rate_multiplier": 0.6
        }
    
    # Apply optimizations to all cameras
    for camera_name, camera in cameras.items():
        original_config = camera.config
        
        # Create optimized configuration
        optimized_config = USB3Config(
            device_id=original_config.device_id,
            vendor_id=original_config.vendor_id,
            product_id=original_config.product_id,
            pixel_format=original_config.pixel_format,
            resolution=original_config.resolution,
            frame_rate=int(original_config.frame_rate * optimal_settings["frame_rate_multiplier"]),
            transfer_size=optimal_settings["transfer_size"],
            num_transfers=optimal_settings["num_transfers"],
            # Copy other settings
            trigger_mode=original_config.trigger_mode,
            exposure_auto=original_config.exposure_auto,
            gain_auto=original_config.gain_auto
        )
        
        # Apply optimization
        camera.reconfigure(optimized_config)
        
        print(f"✅ Optimized {camera_name}:")
        print(f"  Transfer size: {optimal_settings['transfer_size']} bytes")
        print(f"  Transfers: {optimal_settings['num_transfers']}")
        print(f"  Frame rate: {optimized_config.frame_rate} fps")
    
    return optimal_settings

# Use optimization
cameras = {
    "camera_1": camera1,
    "camera_2": camera2,
    "camera_3": camera3
}

optimization_results = optimize_usb3_performance(cameras)
```

## 📊 Performance Monitoring

```python
from advanced_image_sensor_interface.performance.monitor import PerformanceMonitor

class USB3PerformanceMonitor:
    def __init__(self, camera):
        self.camera = camera
        self.monitor = PerformanceMonitor()
    
    def monitor_usb_performance(self, duration_seconds=60):
        """Monitor USB camera performance"""
        
        print(f"📊 Monitoring USB3 camera performance for {duration_seconds} seconds...")
        
        self.monitor.start_monitoring()
        
        # Performance metrics
        frame_times = []
        usb_errors = []
        transfer_rates = []
        
        start_time = time.time()
        frame_count = 0
        
        while time.time() - start_time < duration_seconds:
            frame_start = self.monitor.get_timestamp()
            
            try:
                # Capture frame
                frame = self.camera.capture_frame()
                frame_end = self.monitor.get_timestamp()
                
                # Record metrics
                frame_time = frame_end - frame_start
                frame_times.append(frame_time)
                
                # Calculate transfer rate
                frame_size = frame.width * frame.height * (frame.bit_depth // 8)
                transfer_rate = frame_size / (frame_time / 1000)  # bytes per second
                transfer_rates.append(transfer_rate)
                
                usb_errors.append(0)  # No error
                frame_count += 1
                
            except Exception as e:
                # USB error occurred
                usb_errors.append(1)
                print(f"⚠️ USB error: {e}")
        
        # Calculate statistics
        if frame_times:
            avg_frame_time = sum(frame_times) / len(frame_times)
            actual_fps = 1000 / avg_frame_time
            avg_transfer_rate = sum(transfer_rates) / len(transfer_rates)
            error_rate = sum(usb_errors) / len(usb_errors)
            
            print(f"📈 USB3 Performance Results:")
            print(f"  Frames captured: {frame_count}")
            print(f"  Average frame time: {avg_frame_time:.1f}ms")
            print(f"  Actual FPS: {actual_fps:.1f}")
            print(f"  Average transfer rate: {avg_transfer_rate / 1_000_000:.1f} MB/s")
            print(f"  USB error rate: {error_rate:.1%}")
            
            # Performance warnings
            if avg_frame_time > 50:  # More than 50ms per frame
                print("⚠️ Slow frame capture - check USB bandwidth")
            
            if error_rate > 0.01:  # More than 1% errors
                print("⚠️ High USB error rate - check cable and connections")
            
            if actual_fps < self.camera.config.frame_rate * 0.9:
                print("⚠️ Low frame rate - reduce resolution or transfer size")
            
            return {
                "avg_frame_time_ms": avg_frame_time,
                "actual_fps": actual_fps,
                "transfer_rate_mbps": avg_transfer_rate / 1_000_000,
                "error_rate": error_rate
            }
        
        else:
            print("❌ No frames captured - check camera connection")
            return None

# Use performance monitoring
monitor = USB3PerformanceMonitor(camera)
performance_results = monitor.monitor_usb_performance(duration_seconds=30)
```

## 🎯 Best Practices

### 1. USB Port Selection
```python
def recommend_usb_setup(num_cameras):
    """Recommend optimal USB setup"""
    
    print(f"💡 USB Setup Recommendations for {num_cameras} cameras:")
    
    if num_cameras == 1:
        print("✅ Single camera setup:")
        print("  - Use any USB 3.0 port")
        print("  - Maximum performance available")
        print("  - No special considerations")
    
    elif num_cameras <= 2:
        print("✅ Dual camera setup:")
        print("  - Use separate USB controllers if possible")
        print("  - Check motherboard USB controller layout")
        print("  - Consider PCIe USB 3.0 card for more bandwidth")
    
    elif num_cameras <= 4:
        print("⚠️ Multi-camera setup:")
        print("  - Definitely need separate USB controllers")
        print("  - Use PCIe USB 3.0 cards")
        print("  - Reduce resolution/frame rate per camera")
        print("  - Consider powered USB hubs")
    
    else:
        print("❌ Many cameras setup:")
        print("  - USB 3.0 may not be sufficient")
        print("  - Consider GigE Vision instead")
        print("  - If using USB, need multiple PCIe cards")
        print("  - Significant performance compromises needed")
    
    # Bandwidth calculation
    estimated_bandwidth_per_camera = 200  # MB/s for typical camera
    total_bandwidth = estimated_bandwidth_per_camera * num_cameras
    usb3_limit = 500  # MB/s practical limit for USB 3.0
    
    print(f"📊 Bandwidth Analysis:")
    print(f"  Estimated total: {total_bandwidth} MB/s")
    print(f"  USB 3.0 limit: {usb3_limit} MB/s")
    
    if total_bandwidth > usb3_limit:
        print("⚠️ Bandwidth limit exceeded!")
        reduction_needed = total_bandwidth / usb3_limit
        print(f"  Need to reduce data rate by {reduction_needed:.1f}x")

# Get recommendations
recommend_usb_setup(num_cameras=3)
```

### 2. Camera Configuration Templates
```python
class USB3CameraTemplates:
    """Pre-configured templates for common use cases"""
    
    @staticmethod
    def high_quality_photography():
        """Template for high-quality photography"""
        return USB3Config(
            pixel_format="BayerRG12",       # 12-bit for maximum quality
            resolution=(4096, 3072),        # Very high resolution
            frame_rate=5,                   # Slow for maximum quality
            trigger_mode="software",        # Manual control
            exposure_auto=False,            # Manual exposure
            gain_auto=False,                # Manual gain
            transfer_size=4194304,          # 4MB transfers
            num_transfers=4                 # Fewer transfers for stability
        )
    
    @staticmethod
    def real_time_monitoring():
        """Template for real-time monitoring"""
        return USB3Config(
            pixel_format="BayerRG8",        # 8-bit for speed
            resolution=(1280, 720),         # HD for balance
            frame_rate=60,                  # High frame rate
            trigger_mode="continuous",      # Continuous capture
            exposure_auto=True,             # Auto exposure
            gain_auto=True,                 # Auto gain
            transfer_size=1048576,          # 1MB transfers
            num_transfers=8                 # Many transfers for smoothness
        )
    
    @staticmethod
    def high_speed_analysis():
        """Template for high-speed analysis"""
        return USB3Config(
            pixel_format="Mono8",           # Grayscale for speed
            resolution=(640, 480),          # Lower resolution for speed
            frame_rate=240,                 # Very high frame rate
            trigger_mode="hardware",        # Hardware trigger
            exposure_auto=False,            # Fixed exposure
            gain_auto=False,                # Fixed gain
            exposure_time=1000,             # Very fast exposure
            gain=2.0,                       # Higher gain for fast exposure
            transfer_size=524288,           # 512KB transfers
            num_transfers=12                # Many transfers for high speed
        )
    
    @staticmethod
    def multi_camera_setup():
        """Template for multiple cameras"""
        return USB3Config(
            pixel_format="BayerRG8",        # 8-bit to save bandwidth
            resolution=(1024, 768),         # Moderate resolution
            frame_rate=30,                  # Standard frame rate
            trigger_mode="software",        # Software trigger
            exposure_auto=True,             # Auto exposure
            gain_auto=True,                 # Auto gain
            transfer_size=786432,           # 768KB transfers
            num_transfers=6                 # Moderate transfers
        )

# Use templates
photography_config = USB3CameraTemplates.high_quality_photography()
monitoring_config = USB3CameraTemplates.real_time_monitoring()
analysis_config = USB3CameraTemplates.high_speed_analysis()
multi_config = USB3CameraTemplates.multi_camera_setup()

print("📋 Available camera templates:")
print("  - High Quality Photography")
print("  - Real-time Monitoring") 
print("  - High-speed Analysis")
print("  - Multi-camera Setup")
```

USB3 Vision is perfect for desktop applications, portable systems, and situations where you need the simplicity of USB with professional camera performance!