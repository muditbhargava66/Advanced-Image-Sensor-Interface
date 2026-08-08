"""
Multi-Protocol Integration Example

Demonstrates all 4 camera protocols with consistent interfaces:
- MIPI CSI-2 (mobile/embedded)
- CoaXPress (industrial/scientific)
- GigE Vision (network cameras)
- USB3 Vision (desktop/portable)

Each protocol supports:
- Dataclass configuration with validation
- Streaming with start/stop/capture_frame()
- Statistics and device info
- Error handling

Version: 3.0.0
"""

import asyncio
import logging
import time
from typing import Any

# MIPI CSI-2 Protocol
from advanced_image_sensor_interface.sensor_interface.protocol.mipi import MIPIConfig, MIPIProtocolDriver

# CoaXPress Protocol
from advanced_image_sensor_interface.sensor_interface.protocol.coaxpress import CoaXPressConfig, CoaXPressDriver

# GigE Vision Protocol
from advanced_image_sensor_interface.sensor_interface.protocol.gige import GigEProtocolDriver, GigESpeed, GigEVisionConfig

# USB3 Vision Protocol
from advanced_image_sensor_interface.sensor_interface.protocol.usb3 import USB3VisionConfig, USB3VisionDriver

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def demo_mipi_protocol() -> dict[str, Any]:
    """Demonstrate MIPI CSI-2 protocol."""
    logger.info("=" * 60)
    logger.info("MIPI CSI-2 Protocol Demo")
    logger.info("=" * 60)

    # Configuration with validation
    config = MIPIConfig(
        lanes=4,
        data_rate_mbps=1500.0,  # 6 Gbps total bandwidth
        pixel_format="RAW10",
        resolution=(1920, 1080),
        frame_rate=60.0,
        virtual_channel=0,
        continuous_clock=True,
        enable_ecc=True,
        enable_crc=True,
    )

    driver = MIPIProtocolDriver(config)

    # Connect
    logger.info(f"Total bandwidth: {config.total_bandwidth_mbps} Mbps")
    driver.connect()
    logger.info(f"Connected: {driver.is_connected}")

    # Start streaming
    driver.start_streaming()
    logger.info(f"Streaming: {driver.is_streaming}")

    # Capture frames
    frames_captured = 0
    for _ in range(5):
        frame = driver.capture_frame()
        if frame:
            frames_captured += 1
            logger.info(f"  Frame {frames_captured}: {len(frame)} bytes")

    # Get statistics
    stats = driver.get_statistics()
    device_info = driver.get_device_info()

    # Cleanup
    driver.stop_streaming()
    driver.disconnect()

    return {
        "protocol": "MIPI CSI-2",
        "bandwidth_gbps": config.total_bandwidth_mbps / 1000,
        "frames_captured": frames_captured,
        "statistics": stats,
        "device_info": device_info,
    }


def demo_coaxpress_protocol() -> dict[str, Any]:
    """Demonstrate CoaXPress protocol."""
    logger.info("=" * 60)
    logger.info("CoaXPress Protocol Demo")
    logger.info("=" * 60)

    # Configuration - high-speed industrial camera
    config = CoaXPressConfig(
        speed_grade="CXP-12",  # 12.5 Gbps per connection
        connections=4,  # 50 Gbps total
        packet_size=8192,
        trigger_mode="software",
        pixel_format="Mono16",
        resolution=(4096, 4096),
        frame_rate=30.0,
        power_over_coax=True,
    )

    driver = CoaXPressDriver(config)

    # Connect
    driver.connect()
    logger.info(f"Connected: {driver.is_connected}")
    logger.info(f"Speed: {config.speed_grade} x {config.connections} connections")

    # Start streaming
    driver.start_streaming()
    logger.info(f"Streaming: {driver.is_streaming}")

    # Capture frames
    frames_captured = 0
    for _ in range(5):
        frame = driver.capture_frame()
        if frame:
            frames_captured += 1
            logger.info(f"  Frame {frames_captured}: {len(frame)} bytes")

    # Get statistics
    stats = driver.get_statistics()
    device_info = driver.get_device_info()

    # Cleanup
    driver.stop_streaming()
    driver.disconnect()

    return {
        "protocol": "CoaXPress",
        "speed_grade": config.speed_grade,
        "connections": config.connections,
        "frames_captured": frames_captured,
        "statistics": stats,
        "device_info": device_info,
    }


def demo_gige_protocol() -> dict[str, Any]:
    """Demonstrate GigE Vision protocol."""
    logger.info("=" * 60)
    logger.info("GigE Vision Protocol Demo")
    logger.info("=" * 60)

    # Configuration - network camera
    config = GigEVisionConfig(
        ip_address="192.168.1.100",
        speed=GigESpeed.GIGE_10G,  # 10 GigE
        packet_size=9000,  # Jumbo frames
        pixel_format="Mono8",
        resolution=(1920, 1080),
        frame_rate=30.0,
        heartbeat_timeout_ms=3000,
        enable_roce=False,  # RoCE optional
    )

    driver = GigEProtocolDriver(config)

    # Connect
    driver.connect()
    logger.info(f"Connected: {driver.is_connected}")
    logger.info(f"Camera IP: {config.ip_address}, Speed: {config.speed.name}")

    # GVCP operations
    driver.write_register(0x00100000, 1)  # Enable acquisition
    logger.info("Wrote acquisition enable register")

    # Start streaming
    driver.start_streaming()
    logger.info(f"Streaming: {driver.is_streaming}")

    # Capture frames
    frames_captured = 0
    for _ in range(5):
        frame = driver.capture_frame()
        if frame:
            frames_captured += 1
            logger.info(f"  Frame {frames_captured}: {len(frame)} bytes")

    # Get statistics
    stats = driver.get_statistics()
    device_info = driver.get_device_info()

    # Cleanup
    driver.stop_streaming()
    driver.disconnect()

    return {
        "protocol": "GigE Vision",
        "ip_address": config.ip_address,
        "speed": config.speed.name,
        "frames_captured": frames_captured,
        "statistics": stats,
        "device_info": device_info,
    }


def demo_usb3_protocol() -> dict[str, Any]:
    """Demonstrate USB3 Vision protocol."""
    logger.info("=" * 60)
    logger.info("USB3 Vision Protocol Demo")
    logger.info("=" * 60)

    # Configuration
    config = USB3VisionConfig(
        usb_speed="SuperSpeedPlus",  # USB 3.2 - 10 Gbps
        pixel_format="Mono16",
        resolution=(2048, 1536),
        frame_rate=60.0,
        buffer_count=20,
        packet_size=1024,
    )

    driver = USB3VisionDriver(config)

    # Connect (mock discovery for demo)
    driver._discover_device = lambda: {
        "vendor_id": 0x1234,
        "product_id": 0x5678,
        "model": "USB3 Demo Camera",
        "usb_speed": config.usb_speed,
    }
    driver._open_device = lambda x: f"handle_{x['vendor_id']:04x}"
    driver._initialize_device = lambda: None

    driver.connect()
    logger.info(f"Connected: {driver.is_connected}")
    logger.info(f"USB Speed: {config.usb_speed}")

    # Start streaming
    driver.start_streaming()
    logger.info(f"Streaming: {driver.is_streaming}")

    # Capture frames
    frames_captured = 0
    for _ in range(5):
        frame = driver.capture_frame()
        if frame:
            frames_captured += 1
            logger.info(f"  Frame {frames_captured}: {len(frame)} bytes")

    # Get statistics
    stats = driver.get_statistics()
    device_info = driver.get_device_info()

    # Cleanup
    driver.stop_streaming()
    driver.disconnect()

    return {
        "protocol": "USB3 Vision",
        "usb_speed": config.usb_speed,
        "frames_captured": frames_captured,
        "statistics": stats,
        "device_info": device_info,
    }


def print_comparison_table(results: list[dict[str, Any]]) -> None:
    """Print comparison of protocol results."""
    logger.info("")
    logger.info("=" * 80)
    logger.info("Protocol Comparison Summary")
    logger.info("=" * 80)

    print(f"\n{'Protocol':<15} {'Frames':<10} {'Bandwidth':<15} {'Key Feature':<25}")
    print("-" * 70)

    for r in results:
        protocol = r["protocol"]
        frames = r["frames_captured"]

        if protocol == "MIPI CSI-2":
            bandwidth = f"{r['bandwidth_gbps']:.1f} Gbps"
            feature = "Low power, mobile"
        elif protocol == "CoaXPress":
            bandwidth = f"{r['speed_grade']}"
            feature = f"{r['connections']}x connections, industrial"
        elif protocol == "GigE Vision":
            bandwidth = f"{r['speed']}"
            feature = "Network, long distance"
        else:
            bandwidth = f"{r['usb_speed']}"
            feature = "Plug & play, desktop"

        print(f"{protocol:<15} {frames:<10} {bandwidth:<15} {feature:<25}")


def main() -> None:
    """Run multi-protocol demonstration."""
    logger.info("Advanced Image Sensor Interface - Multi-Protocol Demo")
    logger.info("Demonstrating all 4 supported camera protocols\n")

    results = []

    # Demo each protocol
    try:
        results.append(demo_mipi_protocol())
    except Exception as e:
        logger.error(f"MIPI demo failed: {e}")

    try:
        results.append(demo_coaxpress_protocol())
    except Exception as e:
        logger.error(f"CoaXPress demo failed: {e}")

    try:
        results.append(demo_gige_protocol())
    except Exception as e:
        logger.error(f"GigE demo failed: {e}")

    try:
        results.append(demo_usb3_protocol())
    except Exception as e:
        logger.error(f"USB3 demo failed: {e}")

    # Print comparison
    print_comparison_table(results)

    logger.info("\nAll protocol demonstrations completed successfully!")


if __name__ == "__main__":
    main()
