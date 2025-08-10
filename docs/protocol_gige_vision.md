# GigE Vision Protocol - Simple Guide

## 🤔 What is GigE Vision? (Explain Like I'm 5)

Imagine you want to put cameras all around your house or office, and you want to watch them all from your computer. GigE Vision is like using your regular internet cables (the same ones for your WiFi router) to connect cameras!

**Think of it like this:**
- Your cameras are like security guards stationed around your building
- Your computer is like the security office where you watch everything
- GigE Vision is like using your building's internet cables to connect all the guards to the office
- You can have many cameras on the same network, just like having many computers on WiFi

### 🌐 Why is it Special?

1. **Uses Regular Network Cables**: Same cables as your internet (Ethernet)
2. **Long Distance**: Can work over 100 meters (longer than a football field!)
3. **Multiple Cameras**: Many cameras on one network (like a security system)
4. **Power Through Cable**: Can power cameras through the network cable
5. **Easy Setup**: Works with existing network equipment
6. **Remote Access**: Can view cameras from anywhere on the network

## 🔧 How We Use It in Our Code

### Basic Setup (Simple Example)

```python
# Think of this as setting up a network security camera
from advanced_image_sensor_interface.sensor_interface.protocol.gige import (
    GigEDriver, GigEConfig
)

# Configure the network camera
config = GigEConfig(
    ip_address="192.168.1.100",  # Camera's network address (like a house address)
    subnet_mask="255.255.255.0", # Network neighborhood boundaries
    gateway="192.168.1.1",       # Network router address
    port=3956,                   # Network port (like a specific door)
    packet_size=1500,           # Size of data chunks
    packet_delay=0,             # No delay between data chunks
    pixel_format="BayerRG8",    # Type of pictures (8-bit color)
    resolution=(1920, 1200),    # Picture size (Full HD+)
    frame_rate=25,              # 25 pictures per second
    trigger_mode="continuous",   # Keep taking pictures
    exposure_time=10000,        # How long to collect light (microseconds)
    gain=1.0                    # Signal amplification
)

# Start the network camera
camera = GigEDriver(config)
```

### Taking Network Pictures

```python
# Start the camera
camera.start_streaming()

# Take a picture over the network
picture = camera.capture_frame()

# The picture came through the network cable!
print(f"Got a network picture: {picture.width} x {picture.height} pixels")
print(f"Picture came from IP: {camera.get_camera_ip()}")

# Stop the camera when done
camera.stop_streaming()
```

### Advanced Usage - Multiple Network Cameras

```python
# Set up multiple cameras on the same network (like a security system)
camera_ips = ["192.168.1.100", "192.168.1.101", "192.168.1.102", "192.168.1.103"]
cameras = []

for i, ip in enumerate(camera_ips):
    config = GigEConfig(
        ip_address=ip,              # Each camera has unique IP
        subnet_mask="255.255.255.0",
        gateway="192.168.1.1",
        port=3956 + i,             # Each camera uses different port
        pixel_format="BayerRG8",
        resolution=(1280, 720),     # HD quality (saves network bandwidth)
        frame_rate=15,              # 15 fps (saves network bandwidth)
        trigger_mode="continuous",
        packet_size=1500           # Standard network packet size
    )
    
    camera = GigEDriver(config)
    cameras.append(camera)

# Start all cameras
for i, camera in enumerate(cameras):
    camera.start_streaming()
    print(f"✅ Camera {i+1} started at {camera_ips[i]}")

# Get pictures from all cameras
all_pictures = []
for i, camera in enumerate(cameras):
    picture = camera.capture_frame()
    all_pictures.append(picture)
    print(f"📸 Got picture from camera {i+1}")

print(f"Total pictures captured: {len(all_pictures)}")
```

## 🌐 Network Configuration (Setting Up Your Network)

### Basic Network Setup
```python
# Configure network settings for optimal performance
network_config = {
    "jumbo_frames": True,           # Use larger network packets (9000 bytes)
    "receive_buffer_size": 2097152, # 2MB buffer for receiving data
    "packet_resend": True,          # Resend lost packets
    "heartbeat_timeout": 3000,      # 3 seconds to check if camera is alive
    "command_timeout": 1000,        # 1 second for camera commands
    "multicast_enabled": False      # Don't broadcast to multiple receivers
}

# Apply network optimization
config = GigEConfig(
    ip_address="192.168.1.100",
    # ... other settings ...
    **network_config  # Apply all network optimizations
)

camera = GigEDriver(config)
```

### Power Over Ethernet (PoE) Setup
```python
# Configure Power over Ethernet (power camera through network cable)
poe_config = {
    "poe_enabled": True,        # Enable power over ethernet
    "poe_class": "Class 3",     # Power class (up to 15.4W)
    "voltage_monitoring": True,  # Monitor power voltage
    "power_budget_w": 12.0      # Expected power consumption
}

# Camera gets power AND data through one cable!
config = GigEConfig(
    ip_address="192.168.1.100",
    power_over_ethernet=poe_config,
    # ... other settings ...
)

camera = GigEDriver(config)

# Check power status
power_status = camera.get_power_status()
print(f"Camera power: {power_status.voltage}V, {power_status.current}A")
```

## 🏢 Real-World Examples

### Example 1: Office Security System
```python
class OfficeSecuritySystem:
    def __init__(self):
        # Set up cameras in different office locations
        self.camera_locations = {
            "entrance": "192.168.1.100",
            "lobby": "192.168.1.101", 
            "parking": "192.168.1.102",
            "warehouse": "192.168.1.103"
        }
        
        self.cameras = {}
        
        # Configure each camera
        for location, ip in self.camera_locations.items():
            config = GigEConfig(
                ip_address=ip,
                subnet_mask="255.255.255.0",
                gateway="192.168.1.1",
                pixel_format="BayerRG8",
                resolution=(1920, 1080),    # Full HD for security
                frame_rate=15,              # 15 fps saves bandwidth
                trigger_mode="continuous",
                exposure_time=20000,        # Good for indoor lighting
                gain=1.5                    # Boost for low light
            )
            
            self.cameras[location] = GigEDriver(config)
    
    def start_security_monitoring(self):
        """Start monitoring all office locations"""
        # Start all cameras
        for location, camera in self.cameras.items():
            camera.start_streaming()
            print(f"🔒 Security camera started at {location}")
        
        # Monitor continuously
        while office_is_open():
            for location, camera in self.cameras.items():
                # Get current frame
                frame = camera.capture_frame()
                
                # Check for motion or suspicious activity
                if self.detect_motion(frame):
                    self.alert_security(location, frame)
                    print(f"🚨 Motion detected at {location}!")
                
                # Save frame for later review
                self.save_security_footage(location, frame)
    
    def detect_motion(self, frame):
        """Detect motion in security footage"""
        # This would use computer vision to detect movement
        # For example, compare with previous frame
        pass
    
    def alert_security(self, location, frame):
        """Alert security personnel"""
        # Send email, text message, or sound alarm
        pass
```

### Example 2: Factory Quality Control
```python
class FactoryQualityControl:
    def __init__(self):
        # Set up cameras at different inspection stations
        self.inspection_stations = {
            "station_1": "192.168.10.100",  # Raw materials
            "station_2": "192.168.10.101",  # Assembly
            "station_3": "192.168.10.102",  # Final inspection
            "station_4": "192.168.10.103"   # Packaging
        }
        
        self.cameras = {}
        
        # Configure cameras for industrial use
        for station, ip in self.inspection_stations.items():
            config = GigEConfig(
                ip_address=ip,
                subnet_mask="255.255.255.0",
                gateway="192.168.10.1",
                pixel_format="Mono8",           # Grayscale for defect detection
                resolution=(2048, 1536),        # High resolution for details
                frame_rate=10,                  # 10 fps for inspection
                trigger_mode="hardware",        # Triggered by conveyor belt
                exposure_time=5000,             # Fast exposure for moving parts
                gain=1.0                        # No gain for accurate colors
            )
            
            self.cameras[station] = GigEDriver(config)
    
    def start_quality_inspection(self):
        """Start quality control inspection"""
        # Start all inspection cameras
        for station, camera in self.cameras.items():
            camera.start_streaming()
            print(f"🏭 Inspection camera started at {station}")
        
        # Inspect products continuously
        while factory_is_running():
            for station, camera in self.cameras.items():
                # Wait for product to arrive (hardware trigger)
                product_image = camera.capture_frame()
                
                # Inspect product quality
                quality_result = self.inspect_product(station, product_image)
                
                if quality_result.passed:
                    print(f"✅ Product passed inspection at {station}")
                else:
                    self.reject_product(station, quality_result.defects)
                    print(f"❌ Product rejected at {station}: {quality_result.defects}")
    
    def inspect_product(self, station, image):
        """Inspect product for defects"""
        # Use computer vision to find defects
        # Different inspection criteria for each station
        pass
```

### Example 3: Remote Monitoring System
```python
class RemoteMonitoringSystem:
    def __init__(self):
        # Set up cameras at remote locations
        self.remote_locations = {
            "solar_farm": "10.0.1.100",
            "wind_turbine": "10.0.2.100", 
            "water_treatment": "10.0.3.100",
            "power_substation": "10.0.4.100"
        }
        
        self.cameras = {}
        
        # Configure for outdoor/remote use
        for location, ip in self.remote_locations.items():
            config = GigEConfig(
                ip_address=ip,
                subnet_mask="255.255.0.0",      # Larger network
                gateway="10.0.0.1",
                pixel_format="BayerRG8",
                resolution=(1280, 720),         # HD saves bandwidth over long distance
                frame_rate=5,                   # Low frame rate for remote monitoring
                trigger_mode="continuous",
                exposure_time=15000,            # Good for outdoor lighting
                gain=1.2,                       # Slight boost for weather conditions
                packet_size=1500,              # Standard size for long distance
                packet_delay=100                # Small delay for network stability
            )
            
            self.cameras[location] = GigEDriver(config)
    
    def start_remote_monitoring(self):
        """Start monitoring remote locations"""
        # Start all remote cameras
        for location, camera in self.cameras.items():
            try:
                camera.start_streaming()
                print(f"🌐 Remote camera connected at {location}")
            except Exception as e:
                print(f"❌ Failed to connect to {location}: {e}")
        
        # Monitor remote locations
        while monitoring_active():
            for location, camera in self.cameras.items():
                try:
                    # Get status image
                    status_image = camera.capture_frame()
                    
                    # Check for issues
                    status = self.analyze_location_status(location, status_image)
                    
                    if status.has_issues:
                        self.alert_maintenance(location, status.issues)
                        print(f"⚠️ Issues detected at {location}: {status.issues}")
                    
                    # Log status for reports
                    self.log_status(location, status)
                    
                except Exception as e:
                    print(f"❌ Connection lost to {location}: {e}")
                    self.attempt_reconnection(location)
    
    def analyze_location_status(self, location, image):
        """Analyze status of remote location"""
        # Use computer vision to check:
        # - Equipment status
        # - Weather conditions
        # - Security issues
        # - Maintenance needs
        pass
```

## 🔧 Network Optimization

### Optimizing for Multiple Cameras
```python
def optimize_network_for_multiple_cameras(num_cameras):
    """Optimize network settings for multiple GigE cameras"""
    
    # Calculate total bandwidth needed
    bandwidth_per_camera = 1920 * 1080 * 8 * 25  # Full HD, 8-bit, 25fps
    total_bandwidth = bandwidth_per_camera * num_cameras
    
    print(f"Total bandwidth needed: {total_bandwidth / 1_000_000:.1f} Mbps")
    
    if total_bandwidth > 800_000_000:  # More than 800 Mbps
        print("⚠️ High bandwidth usage - optimizing...")
        
        # Reduce resolution or frame rate
        optimized_config = GigEConfig(
            resolution=(1280, 720),     # Reduce to HD
            frame_rate=15,              # Reduce frame rate
            pixel_format="BayerRG8",    # Use 8-bit instead of 12-bit
            packet_size=9000,          # Use jumbo frames
            packet_delay=0             # No delay
        )
        
        print("✅ Optimized for multiple cameras")
        return optimized_config
    
    else:
        # Standard configuration is fine
        return GigEConfig(
            resolution=(1920, 1080),
            frame_rate=25,
            pixel_format="BayerRG8"
        )

# Use optimization
config = optimize_network_for_multiple_cameras(num_cameras=8)
```

### Network Troubleshooting
```python
def diagnose_network_issues(camera_ip):
    """Diagnose network connectivity issues"""
    
    print(f"🔍 Diagnosing network issues for camera at {camera_ip}")
    
    # Test 1: Ping the camera
    import subprocess
    ping_result = subprocess.run(['ping', '-c', '4', camera_ip], 
                                capture_output=True, text=True)
    
    if ping_result.returncode == 0:
        print("✅ Camera responds to ping")
    else:
        print("❌ Camera does not respond to ping")
        print("Check: Cable connection, power, IP address")
        return
    
    # Test 2: Check GigE Vision port
    import socket
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)
        result = sock.connect_ex((camera_ip, 3956))
        sock.close()
        
        if result == 0:
            print("✅ GigE Vision port is open")
        else:
            print("❌ GigE Vision port is closed")
            print("Check: Camera firmware, firewall settings")
    except Exception as e:
        print(f"❌ Network error: {e}")
    
    # Test 3: Check bandwidth
    try:
        config = GigEConfig(ip_address=camera_ip)
        camera = GigEDriver(config)
        
        # Test data transfer
        start_time = time.time()
        test_frame = camera.capture_frame()
        end_time = time.time()
        
        transfer_time = end_time - start_time
        frame_size = test_frame.width * test_frame.height
        bandwidth = (frame_size * 8) / transfer_time  # bits per second
        
        print(f"✅ Bandwidth test: {bandwidth / 1_000_000:.1f} Mbps")
        
        if bandwidth < 100_000_000:  # Less than 100 Mbps
            print("⚠️ Low bandwidth detected")
            print("Check: Network switch, cable quality, network congestion")
    
    except Exception as e:
        print(f"❌ Camera connection failed: {e}")

# Use network diagnostics
diagnose_network_issues("192.168.1.100")
```

## 📊 Performance Monitoring

```python
from advanced_image_sensor_interface.performance.monitor import PerformanceMonitor

class GigEPerformanceMonitor:
    def __init__(self, cameras):
        self.cameras = cameras  # List of GigE cameras
        self.monitor = PerformanceMonitor()
    
    def monitor_network_performance(self):
        """Monitor network performance of all cameras"""
        
        self.monitor.start_monitoring()
        
        # Monitor each camera
        for camera_name, camera in self.cameras.items():
            print(f"📊 Monitoring {camera_name}...")
            
            # Test 100 frames
            frame_times = []
            packet_losses = []
            
            for i in range(100):
                start_time = self.monitor.get_timestamp()
                
                try:
                    frame = camera.capture_frame()
                    end_time = self.monitor.get_timestamp()
                    
                    frame_time = end_time - start_time
                    frame_times.append(frame_time)
                    packet_losses.append(0)  # No packet loss
                    
                except Exception as e:
                    # Frame capture failed (likely packet loss)
                    packet_losses.append(1)
                    print(f"⚠️ Frame {i} failed: {e}")
            
            # Calculate statistics
            avg_frame_time = sum(frame_times) / len(frame_times) if frame_times else 0
            packet_loss_rate = sum(packet_losses) / len(packet_losses)
            actual_fps = 1000 / avg_frame_time if avg_frame_time > 0 else 0
            
            print(f"📈 {camera_name} Performance:")
            print(f"  Average frame time: {avg_frame_time:.1f}ms")
            print(f"  Actual FPS: {actual_fps:.1f}")
            print(f"  Packet loss rate: {packet_loss_rate:.1%}")
            
            # Performance warnings
            if avg_frame_time > 100:  # More than 100ms per frame
                print(f"⚠️ {camera_name}: Slow frame capture")
            
            if packet_loss_rate > 0.01:  # More than 1% packet loss
                print(f"⚠️ {camera_name}: High packet loss")
            
            if actual_fps < camera.config.frame_rate * 0.9:  # Less than 90% of target
                print(f"⚠️ {camera_name}: Low frame rate")

# Use performance monitoring
cameras = {
    "entrance": entrance_camera,
    "lobby": lobby_camera,
    "parking": parking_camera
}

monitor = GigEPerformanceMonitor(cameras)
monitor.monitor_network_performance()
```

## 🎯 Best Practices

### 1. Network Design
```python
def design_gige_network(num_cameras, camera_requirements):
    """Design optimal network for GigE Vision cameras"""
    
    # Calculate bandwidth requirements
    total_bandwidth = 0
    for req in camera_requirements:
        camera_bandwidth = (req['width'] * req['height'] * 
                          req['bit_depth'] * req['fps'])
        total_bandwidth += camera_bandwidth
    
    # Convert to Mbps
    total_mbps = total_bandwidth / 1_000_000
    
    print(f"Network Design for {num_cameras} cameras:")
    print(f"Total bandwidth required: {total_mbps:.1f} Mbps")
    
    # Recommend network infrastructure
    if total_mbps < 100:
        print("✅ Fast Ethernet (100 Mbps) is sufficient")
        switch_type = "Fast Ethernet"
    elif total_mbps < 800:
        print("✅ Gigabit Ethernet (1000 Mbps) recommended")
        switch_type = "Gigabit Ethernet"
    else:
        print("⚠️ Multiple Gigabit links or 10 Gigabit Ethernet needed")
        switch_type = "10 Gigabit Ethernet"
    
    # Network recommendations
    recommendations = {
        "switch_type": switch_type,
        "cable_type": "Cat6 or better",
        "max_cable_length": "100 meters",
        "poe_required": any(req.get('poe', False) for req in camera_requirements),
        "managed_switch": num_cameras > 4,
        "vlan_isolation": num_cameras > 8
    }
    
    return recommendations

# Example usage
camera_specs = [
    {'width': 1920, 'height': 1080, 'bit_depth': 8, 'fps': 25, 'poe': True},
    {'width': 1920, 'height': 1080, 'bit_depth': 8, 'fps': 25, 'poe': True},
    {'width': 1280, 'height': 720, 'bit_depth': 8, 'fps': 30, 'poe': True},
    {'width': 1280, 'height': 720, 'bit_depth': 8, 'fps': 30, 'poe': True}
]

network_design = design_gige_network(4, camera_specs)
print(f"Recommended network: {network_design}")
```

### 2. IP Address Management
```python
class GigENetworkManager:
    def __init__(self, network_base="192.168.1"):
        self.network_base = network_base
        self.assigned_ips = set()
        self.cameras = {}
    
    def assign_ip_address(self, camera_name):
        """Assign unique IP address to camera"""
        
        # Start from .100 for cameras
        for i in range(100, 255):
            ip = f"{self.network_base}.{i}"
            
            if ip not in self.assigned_ips:
                self.assigned_ips.add(ip)
                self.cameras[camera_name] = ip
                print(f"📍 Assigned {ip} to {camera_name}")
                return ip
        
        raise Exception("No available IP addresses")
    
    def create_camera_config(self, camera_name, **kwargs):
        """Create camera configuration with assigned IP"""
        
        if camera_name not in self.cameras:
            self.assign_ip_address(camera_name)
        
        ip = self.cameras[camera_name]
        
        config = GigEConfig(
            ip_address=ip,
            subnet_mask="255.255.255.0",
            gateway=f"{self.network_base}.1",
            **kwargs
        )
        
        return config
    
    def list_cameras(self):
        """List all configured cameras"""
        print("📋 Configured cameras:")
        for name, ip in self.cameras.items():
            print(f"  {name}: {ip}")

# Use network manager
network = GigENetworkManager("192.168.10")

# Configure multiple cameras easily
entrance_config = network.create_camera_config("entrance", 
                                              resolution=(1920, 1080), 
                                              frame_rate=25)

lobby_config = network.create_camera_config("lobby", 
                                           resolution=(1280, 720), 
                                           frame_rate=30)

network.list_cameras()
```

GigE Vision is perfect for applications where you need multiple cameras connected over a network, like security systems, factory monitoring, or remote surveillance!