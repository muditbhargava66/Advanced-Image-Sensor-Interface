# Camera Protocols Documentation

## 📚 Complete Protocol Documentation

Welcome to the comprehensive camera protocol documentation for the Advanced Image Sensor Interface! This documentation is designed to help you understand and use all supported camera protocols, from beginner-friendly explanations to advanced implementation details.

## 🎯 Quick Navigation

### 🔍 **Start Here: Which Protocol Should I Use?**
- **[Protocol Comparison Guide](protocol_comparison_guide.md)** - Interactive guide to help you choose the right protocol for your application

### 📖 **Individual Protocol Guides (ELI5 + Technical)**

#### 🔵 MIPI CSI-2 Protocol
- **[MIPI CSI-2 Simple Guide](protocol_mipi_csi2.md)** - Complete guide from basics to advanced usage
- **Best for:** Mobile devices, embedded systems, drones, IoT devices
- **Key features:** Ultra-low power, very fast, compact, short distance

#### 🟠 CoaXPress Protocol  
- **[CoaXPress Simple Guide](protocol_coaxpress.md)** - Industrial-grade camera interface guide
- **Best for:** Factory inspection, scientific imaging, medical equipment
- **Key features:** Highest performance, long distance, industrial reliability, expensive

#### 🟢 GigE Vision Protocol
- **[GigE Vision Simple Guide](protocol_gige_vision.md)** - Network-based camera system guide  
- **Best for:** Security systems, multiple cameras, remote monitoring
- **Key features:** Network integration, multiple cameras, moderate cost, remote access

#### 🔴 USB3 Vision Protocol
- **[USB3 Vision Simple Guide](protocol_usb3_vision.md)** - Plug-and-play camera guide
- **Best for:** Desktop applications, laboratory equipment, development
- **Key features:** Plug-and-play, good performance, cost-effective, simple setup

### 📋 **Technical Reference**
- **[Complete Technical Protocols Guide](protocols.md)** - Comprehensive technical documentation for all protocols

## 🚀 Getting Started

### For Beginners
1. **Start with the [Protocol Comparison Guide](protocol_comparison_guide.md)** to understand which protocol fits your needs
2. **Read the specific protocol guide** for your chosen protocol (MIPI, CoaXPress, GigE, or USB3)
3. **Try the code examples** provided in each guide
4. **Refer to the technical guide** when you need detailed specifications

### For Experienced Developers
1. **Jump to the [Technical Protocols Guide](protocols.md)** for complete specifications
2. **Use the individual protocol guides** for implementation examples and best practices
3. **Check the comparison guide** when evaluating protocol alternatives

## 📊 Protocol Quick Reference

| Protocol | Speed | Distance | Cost | Complexity | Best Use Case |
|----------|-------|----------|------|------------|---------------|
| **MIPI CSI-2** | ⭐⭐⭐⭐⭐ | ⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ | Mobile/Embedded |
| **CoaXPress** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐ | ⭐⭐ | Industrial |
| **GigE Vision** | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | Security/Multi-camera |
| **USB3 Vision** | ⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Desktop/Lab |

## 🎓 Learning Path

### Beginner Path
```
1. Protocol Comparison Guide
   ↓
2. Choose your protocol
   ↓  
3. Read the ELI5 sections
   ↓
4. Try basic code examples
   ↓
5. Experiment with real-world examples
```

### Advanced Path
```
1. Technical Protocols Guide
   ↓
2. Individual protocol deep-dives
   ↓
3. Performance optimization sections
   ↓
4. Troubleshooting guides
   ↓
5. Best practices implementation
```

## 🛠️ Code Examples by Use Case

### Mobile/Embedded Applications
```python
# See: protocol_mipi_csi2.md
from advanced_image_sensor_interface.sensor_interface.protocol.mipi import MIPIDriver, MIPIConfig

config = MIPIConfig(
    lanes=4,
    data_rate_mbps=2500,
    pixel_format="RAW12",
    resolution=(1920, 1080),
    frame_rate=60
)
camera = MIPIDriver(config)
```

### Industrial Applications
```python
# See: protocol_coaxpress.md
from advanced_image_sensor_interface.sensor_interface.protocol.coaxpress import CoaXPressDriver, CoaXPressConfig

config = CoaXPressConfig(
    speed_grade="CXP-6",
    connections=1,
    pixel_format="Mono16",
    resolution=(2048, 2048),
    frame_rate=30,
    power_over_coax=True
)
camera = CoaXPressDriver(config)
```

### Security/Network Applications
```python
# See: protocol_gige_vision.md
from advanced_image_sensor_interface.sensor_interface.protocol.gige import GigEDriver, GigEConfig

config = GigEConfig(
    ip_address="192.168.1.100",
    pixel_format="BayerRG8",
    resolution=(1920, 1080),
    frame_rate=25,
    trigger_mode="continuous"
)
camera = GigEDriver(config)
```

### Desktop/Lab Applications
```python
# See: protocol_usb3_vision.md
from advanced_image_sensor_interface.sensor_interface.protocol.usb3 import USB3Driver, USB3Config

config = USB3Config(
    device_id="Lab_Camera",
    vendor_id=0x1234,
    product_id=0x5678,
    pixel_format="BayerGR8",
    resolution=(1280, 1024),
    frame_rate=60
)
camera = USB3Driver(config)
```

## 🔧 Implementation Support

### Protocol Selection Helper
```python
from advanced_image_sensor_interface.sensor_interface.protocol_selector import ProtocolSelector

selector = ProtocolSelector()

# Define your requirements
requirements = {
    "bandwidth_gbps": 2.0,
    "distance_m": 50,
    "power_over_cable": True,
    "latency_ms": 10,
    "num_cameras": 4
}

# Get recommendation
recommended_protocol = selector.select_optimal_protocol(requirements)
print(f"Recommended protocol: {recommended_protocol}")
```

### Multi-Protocol Support
```python
# The system supports dynamic protocol switching
from advanced_image_sensor_interface.sensor_interface.protocol_selector import ProtocolType

# Configure multiple protocols
selector.configure_protocol(ProtocolType.MIPI, mipi_config)
selector.configure_protocol(ProtocolType.GIGE, gige_config)
selector.configure_protocol(ProtocolType.USB3, usb3_config)
selector.configure_protocol(ProtocolType.COAXPRESS, coaxpress_config)

# Switch between protocols as needed
selector.activate_protocol(ProtocolType.GIGE)
```

## 📈 Performance Optimization

Each protocol guide includes specific optimization sections:

- **MIPI CSI-2**: Lane configuration, clock modes, buffer management
- **CoaXPress**: Speed grade selection, cable optimization, power management
- **GigE Vision**: Network tuning, bandwidth optimization, multi-camera setup
- **USB3 Vision**: Transfer optimization, bandwidth sharing, multi-camera considerations

## 🚨 Troubleshooting

Common issues and solutions are covered in each protocol guide:

- **Connection problems**: Cable, power, and configuration issues
- **Performance issues**: Bandwidth, latency, and throughput problems  
- **Compatibility issues**: Hardware, driver, and software conflicts
- **Quality issues**: Image artifacts, synchronization, and timing problems

## 🎯 Real-World Applications

### By Industry

**📱 Consumer Electronics**
- Smartphones: MIPI CSI-2
- Tablets: MIPI CSI-2  
- Laptops: USB3 Vision
- Gaming: USB3 Vision

**🏭 Industrial**
- Quality Control: CoaXPress
- Assembly Line Monitoring: GigE Vision
- Robotics: MIPI CSI-2 or GigE Vision
- Process Control: CoaXPress

**🔒 Security**
- Building Security: GigE Vision
- Traffic Monitoring: GigE Vision
- Perimeter Security: GigE Vision
- Access Control: USB3 Vision

**🔬 Scientific/Medical**
- Laboratory Equipment: USB3 Vision
- Medical Imaging: CoaXPress
- Research Cameras: CoaXPress
- Microscopy: USB3 Vision

**🚗 Automotive**
- ADAS Cameras: MIPI CSI-2
- Backup Cameras: MIPI CSI-2
- Dashboard Cameras: USB3 Vision
- Fleet Monitoring: GigE Vision

## 📞 Getting Help

### Documentation Structure
- **ELI5 Sections**: Simple explanations for beginners
- **Technical Sections**: Detailed specifications and parameters
- **Code Examples**: Working implementations for common scenarios
- **Real-World Examples**: Complete application implementations
- **Troubleshooting**: Solutions for common problems
- **Best Practices**: Optimization and performance tips

### Additional Resources
- **API Reference**: Complete function and class documentation
- **Hardware Integration Guide**: Connecting to real hardware
- **Performance Analysis**: Benchmarking and optimization
- **Testing Guide**: Validation and quality assurance

## 🎉 Quick Start Examples

### 30-Second Quick Start
```python
# 1. Choose your protocol (see comparison guide)
# 2. Import the driver
from advanced_image_sensor_interface.sensor_interface.protocol.usb3 import USB3Driver, USB3Config

# 3. Create configuration
config = USB3Config(device_id="My_Camera")

# 4. Create driver and capture
camera = USB3Driver(config)
camera.start_streaming()
image = camera.capture_frame()
camera.stop_streaming()

print(f"Captured {image.width}x{image.height} image!")
```

### 5-Minute Complete Setup
See the individual protocol guides for complete setup examples including:
- Hardware connection
- Software configuration  
- Error handling
- Performance optimization
- Real-world integration

---

**Ready to get started?** Choose your path:
- 🆕 **New to camera protocols?** → [Protocol Comparison Guide](protocol_comparison_guide.md)
- 📱 **Building mobile/embedded?** → [MIPI CSI-2 Guide](protocol_mipi_csi2.md)
- 🏭 **Industrial application?** → [CoaXPress Guide](protocol_coaxpress.md)
- 🔒 **Security/multi-camera?** → [GigE Vision Guide](protocol_gige_vision.md)
- 💻 **Desktop/lab equipment?** → [USB3 Vision Guide](protocol_usb3_vision.md)
- 🔧 **Need technical details?** → [Technical Protocols Guide](protocols.md)