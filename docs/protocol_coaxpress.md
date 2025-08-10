# CoaXPress Protocol - Simple Guide

## 🤔 What is CoaXPress? (Explain Like I'm 5)

Imagine you have a really powerful camera that needs to send HUGE pictures very far away, like from a factory floor to a control room. CoaXPress is like a super-strong cable that can do two amazing things at once:

**Think of it like this:**
- Your camera is like a person taking GIANT, detailed photos
- The control room is like a photo studio far away
- CoaXPress is like a magical cable that can:
  1. Send enormous photos super fast (like a fire hose for data)
  2. Give power to the camera through the same cable (like an extension cord)

### 🏭 Why is it Special?

1. **Industrial Strength**: Built for factories, labs, and tough environments
2. **Super Long Distance**: Can work over 100 meters (longer than a football field!)
3. **Power + Data**: One cable does everything (like a Swiss Army knife)
4. **Ultra Fast**: Can send data faster than almost any other camera cable
5. **Very Reliable**: Won't break even in harsh conditions

## 🔧 How We Use It in Our Code

### Basic Setup (Simple Example)

```python
# Think of this as setting up your industrial camera system
from advanced_image_sensor_interface.sensor_interface.protocol.coaxpress import (
    CoaXPressDriver, CoaXPressConfig
)

# Configure the industrial camera
config = CoaXPressConfig(
    speed_grade="CXP-6",        # How fast the cable works (CXP-6 = very fast)
    connections=1,              # Number of cables (1 is usually enough)
    packet_size=8192,          # Size of data chunks (bigger = more efficient)
    trigger_mode="software",    # How to tell camera to take pictures
    pixel_format="Mono16",      # Type of pictures (16-bit grayscale)
    resolution=(2048, 2048),    # Picture size (very high resolution)
    frame_rate=30,              # 30 pictures per second
    power_over_coax=True,      # Send power through the cable
    discovery_timeout=5.0       # How long to wait to find the camera
)

# Start the industrial camera system
camera = CoaXPressDriver(config)
```

### Taking Industrial Pictures

```python
# Start the camera
camera.start_streaming()

# Take a high-quality industrial picture
picture = camera.capture_frame()

# This picture is HUGE and very detailed!
print(f"Got an industrial picture: {picture.width} x {picture.height} pixels")
print(f"Picture quality: {picture.bit_depth} bits per pixel")

# Stop the camera when inspection is done
camera.stop_streaming()
```

### Advanced Usage - High-Speed Inspection

```python
# Set up for high-speed factory inspection
high_speed_config = CoaXPressConfig(
    speed_grade="CXP-12",       # Fastest possible speed
    connections=2,              # Use 2 cables for maximum speed
    packet_size=16384,         # Large data chunks
    trigger_mode="hardware",    # Hardware trigger for precise timing
    pixel_format="Mono12",      # 12-bit grayscale (good balance)
    resolution=(4096, 4096),    # Ultra-high resolution
    frame_rate=60,              # 60 fps for fast-moving objects
    power_over_coax=True,      # Power the camera
    discovery_timeout=10.0      # More time for complex setup
)

camera = CoaXPressDriver(high_speed_config)

# Inspect products on assembly line
def inspect_products():
    camera.start_streaming()
    
    while factory_is_running():
        # Wait for product to arrive (hardware trigger)
        picture = camera.capture_frame()
        
        # Analyze the product
        quality_result = analyze_product_quality(picture)
        
        if quality_result.is_defective:
            reject_product()
            print("❌ Defective product rejected")
        else:
            accept_product()
            print("✅ Product passed inspection")
```

## 🚀 Speed Grades (How Fast Your Cable Can Go)

Think of speed grades like different types of highways:

```python
# Different speed grades for different needs
speed_grades = {
    "CXP-1": {
        "speed_gbps": 1.25,
        "description": "Local road - basic industrial cameras",
        "use_case": "Simple quality control"
    },
    "CXP-2": {
        "speed_gbps": 2.5,
        "description": "City street - standard industrial",
        "use_case": "Regular factory inspection"
    },
    "CXP-3": {
        "speed_gbps": 3.125,
        "description": "Highway - high-resolution cameras",
        "use_case": "Detailed part inspection"
    },
    "CXP-5": {
        "speed_gbps": 5.0,
        "description": "Interstate - high-speed imaging",
        "use_case": "Fast-moving assembly lines"
    },
    "CXP-6": {
        "speed_gbps": 6.25,
        "description": "Superhighway - professional systems",
        "use_case": "Professional inspection systems"
    },
    "CXP-10": {
        "speed_gbps": 10.0,
        "description": "Autobahn - scientific applications",
        "use_case": "Research and scientific imaging"
    },
    "CXP-12": {
        "speed_gbps": 12.5,
        "description": "Rocket ship - ultra-high-speed",
        "use_case": "Extreme high-speed applications"
    }
}

# Choose based on your needs
def choose_speed_grade(application):
    if application == "basic_inspection":
        return "CXP-2"
    elif application == "high_resolution":
        return "CXP-6"
    elif application == "scientific_research":
        return "CXP-10"
    elif application == "extreme_speed":
        return "CXP-12"
    else:
        return "CXP-3"  # Good default
```

## ⚡ Power Over Coax (One Cable for Everything)

```python
# Configure power delivery through the cable
power_config = {
    "power_class": "PoCXP+",    # Type of power delivery
    "max_power_w": 25,          # Maximum 25 watts
    "voltage_v": 24,            # 24 volts
    "current_limit_a": 1.0,     # 1 amp maximum
    "power_monitoring": True    # Monitor power usage
}

# Set up camera with power delivery
config = CoaXPressConfig(
    speed_grade="CXP-6",
    power_over_coax=True,
    power_settings=power_config,
    # ... other settings
)

# The camera gets both data connection AND power from one cable!
camera = CoaXPressDriver(config)

# Monitor power usage
def monitor_camera_power():
    power_status = camera.get_power_status()
    
    print(f"Camera power usage: {power_status.current_watts}W")
    print(f"Voltage: {power_status.voltage}V")
    print(f"Current: {power_status.current}A")
    
    if power_status.current_watts > 20:
        print("⚠️ High power usage detected")
```

## 🏭 Real-World Examples

### Example 1: Car Manufacturing Inspection
```python
class CarPartInspector:
    def __init__(self):
        self.config = CoaXPressConfig(
            speed_grade="CXP-6",        # Fast enough for car parts
            connections=1,              # One cable is sufficient
            pixel_format="Mono12",      # Good quality for defect detection
            resolution=(2048, 2048),    # High resolution for small defects
            frame_rate=30,              # 30 parts per minute
            trigger_mode="hardware",    # Triggered by conveyor belt
            power_over_coax=True       # Power the camera
        )
        self.camera = CoaXPressDriver(self.config)
    
    def inspect_car_parts(self):
        """Inspect car parts on assembly line"""
        self.camera.start_streaming()
        
        while assembly_line_running():
            # Wait for part to arrive
            part_image = self.camera.capture_frame()
            
            # Check for defects
            defects = self.find_defects(part_image)
            
            if defects:
                self.reject_part(defects)
                print(f"❌ Part rejected: {len(defects)} defects found")
            else:
                self.accept_part()
                print("✅ Part passed inspection")
    
    def find_defects(self, image):
        """Find defects in car part image"""
        # This would use computer vision algorithms
        # to find scratches, dents, missing parts, etc.
        pass
```

### Example 2: Medical Device Manufacturing
```python
class MedicalDeviceInspector:
    def __init__(self):
        self.config = CoaXPressConfig(
            speed_grade="CXP-10",       # Very high speed for precision
            connections=2,              # Two cables for maximum bandwidth
            pixel_format="Mono16",      # Maximum quality for medical devices
            resolution=(4096, 4096),    # Ultra-high resolution
            frame_rate=15,              # Slower but very precise
            trigger_mode="software",    # Controlled timing
            power_over_coax=True       # Power the camera
        )
        self.camera = CoaXPressDriver(self.config)
    
    def inspect_medical_devices(self):
        """Inspect medical devices with extreme precision"""
        self.camera.start_streaming()
        
        for device in medical_devices:
            # Take ultra-high-quality image
            device_image = self.camera.capture_frame()
            
            # Perform detailed analysis
            measurements = self.measure_device(device_image)
            surface_quality = self.check_surface(device_image)
            
            # Medical devices must be perfect
            if self.meets_medical_standards(measurements, surface_quality):
                self.approve_device(device)
                print("✅ Medical device approved")
            else:
                self.reject_device(device)
                print("❌ Medical device rejected - safety concern")
```

### Example 3: Scientific Research Camera
```python
class ScientificCamera:
    def __init__(self, experiment_type):
        self.experiment_type = experiment_type
        self.config = CoaXPressConfig(
            speed_grade="CXP-12",       # Maximum speed for research
            connections=4,              # Multiple cables for extreme bandwidth
            pixel_format="Mono16",      # Maximum bit depth
            resolution=(8192, 8192),    # Extremely high resolution
            frame_rate=1000,            # Ultra-high-speed capture
            trigger_mode="external",    # Synchronized with experiment
            power_over_coax=True       # Power the camera
        )
        self.camera = CoaXPressDriver(self.config)
    
    def capture_experiment(self):
        """Capture high-speed scientific phenomena"""
        self.camera.start_streaming()
        
        # Wait for experiment to start
        self.wait_for_experiment_trigger()
        
        # Capture ultra-high-speed sequence
        images = []
        for i in range(10000):  # 10,000 images in 10 seconds
            image = self.camera.capture_frame()
            images.append(image)
        
        # Analyze the scientific data
        results = self.analyze_experiment(images)
        
        print(f"🔬 Experiment complete: {len(images)} images captured")
        print(f"📊 Results: {results}")
        
        return results
```

## 🔧 Multiple Camera Setup

```python
# Set up multiple CoaXPress cameras for complete coverage
class MultiCameraSystem:
    def __init__(self, num_cameras):
        self.cameras = []
        
        for i in range(num_cameras):
            config = CoaXPressConfig(
                speed_grade="CXP-6",
                connections=1,
                pixel_format="Mono12",
                resolution=(2048, 2048),
                frame_rate=30,
                device_id=f"Camera_{i}",  # Unique ID for each camera
                power_over_coax=True
            )
            
            camera = CoaXPressDriver(config)
            self.cameras.append(camera)
    
    def start_all_cameras(self):
        """Start all cameras simultaneously"""
        for camera in self.cameras:
            camera.start_streaming()
    
    def capture_synchronized(self):
        """Capture images from all cameras at the same time"""
        images = []
        
        # Trigger all cameras simultaneously
        for camera in self.cameras:
            image = camera.capture_frame()
            images.append(image)
        
        return images
    
    def stop_all_cameras(self):
        """Stop all cameras"""
        for camera in self.cameras:
            camera.stop_streaming()

# Use the multi-camera system
system = MultiCameraSystem(num_cameras=4)
system.start_all_cameras()

# Get images from all 4 cameras at once
all_images = system.capture_synchronized()
print(f"Captured {len(all_images)} images simultaneously")
```

## 🛠️ Troubleshooting (Fixing Problems)

### Problem 1: Camera Not Found
```python
def find_coaxpress_cameras():
    """Find all CoaXPress cameras on the network"""
    from advanced_image_sensor_interface.sensor_interface.protocol.coaxpress.discovery import discover_devices
    
    print("🔍 Searching for CoaXPress cameras...")
    
    cameras = discover_devices(timeout=10.0)
    
    if cameras:
        print(f"✅ Found {len(cameras)} cameras:")
        for i, camera in enumerate(cameras):
            print(f"  Camera {i}: {camera.device_id} ({camera.speed_grade})")
    else:
        print("❌ No cameras found. Check:")
        print("  - Cable connections")
        print("  - Power supply")
        print("  - Camera compatibility")
    
    return cameras
```

### Problem 2: Slow Performance
```python
def optimize_performance():
    """Optimize CoaXPress performance"""
    
    # Use faster speed grade
    config = CoaXPressConfig(
        speed_grade="CXP-10",       # Upgrade to faster speed
        connections=2,              # Use multiple connections
        packet_size=16384,         # Larger packets
        # ... other settings
    )
    
    # Monitor performance
    camera = CoaXPressDriver(config)
    
    # Check actual performance
    performance = camera.get_performance_stats()
    
    print(f"Actual bandwidth: {performance.bandwidth_gbps} Gbps")
    print(f"Frame rate: {performance.actual_fps} fps")
    print(f"Dropped frames: {performance.dropped_frames}")
    
    if performance.bandwidth_gbps < config.expected_bandwidth:
        print("⚠️ Performance issue detected")
        print("Try:")
        print("  - Check cable quality")
        print("  - Reduce resolution")
        print("  - Increase packet size")
```

### Problem 3: Power Issues
```python
def diagnose_power_problems():
    """Diagnose power delivery problems"""
    
    camera = CoaXPressDriver(config)
    power_status = camera.get_power_status()
    
    print(f"Power delivery status:")
    print(f"  Voltage: {power_status.voltage}V")
    print(f"  Current: {power_status.current}A")
    print(f"  Power: {power_status.power}W")
    
    if power_status.voltage < 20:
        print("❌ Voltage too low - check power supply")
    elif power_status.current > 1.2:
        print("❌ Current too high - camera may be damaged")
    elif power_status.power > 25:
        print("❌ Power consumption too high")
    else:
        print("✅ Power delivery is normal")
```

## 📊 Performance Monitoring

```python
from advanced_image_sensor_interface.performance.monitor import PerformanceMonitor

class CoaXPressMonitor:
    def __init__(self, camera):
        self.camera = camera
        self.monitor = PerformanceMonitor()
    
    def start_monitoring(self):
        """Start monitoring camera performance"""
        self.monitor.start_monitoring()
        
        # Monitor for 1 minute
        for i in range(1800):  # 30 fps * 60 seconds
            start_time = self.monitor.get_timestamp()
            
            # Capture frame
            frame = self.camera.capture_frame()
            
            end_time = self.monitor.get_timestamp()
            
            # Record timing
            frame_time = end_time - start_time
            self.monitor.record_frame_time(frame_time)
            
            # Check for problems
            if frame_time > 50:  # More than 50ms is slow
                print(f"⚠️ Slow frame detected: {frame_time}ms")
        
        # Get final report
        report = self.monitor.get_performance_report()
        
        print(f"📊 Performance Report:")
        print(f"  Average frame time: {report.average_frame_time_ms}ms")
        print(f"  Actual FPS: {report.actual_fps}")
        print(f"  Dropped frames: {report.dropped_frames}")
        print(f"  Bandwidth used: {report.bandwidth_mbps}Mbps")
        
        return report

# Use the monitor
monitor = CoaXPressMonitor(camera)
performance_report = monitor.start_monitoring()
```

## 🎯 Best Practices

### 1. Choose the Right Speed Grade
```python
def select_optimal_speed(requirements):
    """Select optimal CoaXPress speed grade"""
    
    resolution = requirements.get('resolution', (1920, 1080))
    frame_rate = requirements.get('frame_rate', 30)
    bit_depth = requirements.get('bit_depth', 8)
    
    # Calculate required bandwidth
    pixels_per_frame = resolution[0] * resolution[1]
    bits_per_frame = pixels_per_frame * bit_depth
    bits_per_second = bits_per_frame * frame_rate
    gbps_required = bits_per_second / 1_000_000_000
    
    # Add 20% safety margin
    gbps_required *= 1.2
    
    # Select appropriate speed grade
    if gbps_required <= 1.25:
        return "CXP-1"
    elif gbps_required <= 2.5:
        return "CXP-2"
    elif gbps_required <= 3.125:
        return "CXP-3"
    elif gbps_required <= 5.0:
        return "CXP-5"
    elif gbps_required <= 6.25:
        return "CXP-6"
    elif gbps_required <= 10.0:
        return "CXP-10"
    else:
        return "CXP-12"

# Example usage
requirements = {
    'resolution': (4096, 4096),
    'frame_rate': 60,
    'bit_depth': 12
}

optimal_speed = select_optimal_speed(requirements)
print(f"Recommended speed grade: {optimal_speed}")
```

### 2. Cable Management
```python
def check_cable_quality():
    """Check CoaXPress cable quality"""
    
    # Test cable performance
    test_config = CoaXPressConfig(
        speed_grade="CXP-6",
        connections=1,
        test_mode=True  # Enable cable testing
    )
    
    camera = CoaXPressDriver(test_config)
    
    # Run cable test
    cable_test = camera.test_cable_quality()
    
    print(f"Cable test results:")
    print(f"  Signal quality: {cable_test.signal_quality}%")
    print(f"  Error rate: {cable_test.error_rate}")
    print(f"  Maximum distance: {cable_test.max_distance_m}m")
    
    if cable_test.signal_quality < 90:
        print("⚠️ Cable quality is poor - consider replacement")
    elif cable_test.error_rate > 0.001:
        print("⚠️ High error rate - check connections")
    else:
        print("✅ Cable quality is excellent")
```

CoaXPress is perfect for industrial applications where you need the highest quality images, longest distances, and most reliable operation!