#!/usr/bin/env python3
"""
Protocol Implementations Example

This example demonstrates the various camera interface protocols implemented
in the Advanced Image Sensor Interface, including MIPI CSI-2, CoaXPress,
GigE Vision, and USB3 Vision.

Author: Advanced Image Sensor Interface Team
Version: 2.0.0
"""

import argparse
import logging
import sys
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

try:
    from advanced_image_sensor_interface import MIPIConfig, MIPIDriver
except ImportError as e:
    logger.error(f"Failed to import required modules: {e}")
    logger.error("Please install the package with: pip install -e .")
    sys.exit(1)


class ProtocolType(Enum):
    """Supported camera interface protocols."""

    MIPI_CSI2 = "mipi_csi2"
    COAXPRESS = "coaxpress"
    GIGE_VISION = "gige_vision"
    USB3_VISION = "usb3_vision"


@dataclass
class ProtocolConfig:
    """Base configuration for camera protocols."""

    protocol_type: ProtocolType
    data_rate_mbps: float
    resolution: tuple
    frame_rate: float
    bit_depth: int


class MIPICSIInterface:
    """
    MIPI CSI-2 (Camera Serial Interface) implementation.

    MIPI CSI-2 is the most common mobile camera interface standard,
    supporting high-speed serial data transmission with low power consumption.
    """

    def __init__(self, config: ProtocolConfig):
        """Initialize MIPI CSI-2 interface."""
        self.config = config
        self.lanes = self._calculate_lanes(config.data_rate_mbps)
        self.mipi_driver = None
        self.is_streaming = False

        logger.info(f"MIPI CSI-2 interface initialized: {self.lanes} lanes @ {config.data_rate_mbps} Mbps")

    def _calculate_lanes(self, data_rate_mbps: float) -> int:
        """Calculate required number of MIPI lanes."""
        # Each lane supports up to 2.5 Gbps typically
        max_per_lane = 2500  # Mbps
        required_lanes = int(np.ceil(data_rate_mbps / max_per_lane))
        return min(max(required_lanes, 1), 8)  # 1-8 lanes typical

    def connect(self) -> bool:
        """Establish MIPI CSI-2 connection."""
        try:
            mipi_config = MIPIConfig(lanes=self.lanes, data_rate=self.config.data_rate_mbps / 1000, channel=0)  # Convert to Gbps
            self.mipi_driver = MIPIDriver(mipi_config)
            logger.info("✓ MIPI CSI-2 connection established")
            return True
        except Exception as e:
            logger.error(f"✗ MIPI CSI-2 connection failed: {e}")
            return False

    def start_streaming(self) -> bool:
        """Start MIPI CSI-2 streaming."""
        if not self.mipi_driver:
            logger.error("MIPI driver not initialized")
            return False

        try:
            # Configure streaming parameters
            self._configure_streaming()
            self.is_streaming = True
            logger.info("✓ MIPI CSI-2 streaming started")
            return True
        except Exception as e:
            logger.error(f"✗ MIPI CSI-2 streaming failed: {e}")
            return False

    def capture_frame(self) -> Optional[np.ndarray]:
        """Capture a frame via MIPI CSI-2."""
        if not self.is_streaming:
            logger.error("Streaming not active")
            return None

        try:
            # Calculate frame size
            width, height = self.config.resolution
            channels = 3 if self.config.bit_depth > 8 else 1
            width * height * channels * (self.config.bit_depth // 8)

            # Simulate frame capture
            frame_data = np.random.randint(
                0,
                2**self.config.bit_depth - 1,
                (height, width, channels),
                dtype=np.uint16 if self.config.bit_depth > 8 else np.uint8,
            )

            return frame_data
        except Exception as e:
            logger.error(f"✗ MIPI frame capture failed: {e}")
            return None

    def stop_streaming(self) -> bool:
        """Stop MIPI CSI-2 streaming."""
        self.is_streaming = False
        logger.info("✓ MIPI CSI-2 streaming stopped")
        return True

    def _configure_streaming(self):
        """Configure MIPI streaming parameters."""
        # Set up virtual channels, data types, etc.
        pass

    def get_status(self) -> dict[str, Any]:
        """Get MIPI CSI-2 interface status."""
        status = {
            "protocol": "MIPI CSI-2",
            "lanes": self.lanes,
            "data_rate_mbps": self.config.data_rate_mbps,
            "resolution": self.config.resolution,
            "frame_rate": self.config.frame_rate,
            "is_streaming": self.is_streaming,
            "bit_depth": self.config.bit_depth,
        }

        if self.mipi_driver:
            mipi_status = self.mipi_driver.get_status()
            status.update(mipi_status)

        return status


class CoaXPressInterface:
    """
    CoaXPress interface implementation.

    CoaXPress is a high-speed digital interface standard for machine vision
    applications, supporting data rates up to 12.5 Gbps over coaxial cables.
    """

    def __init__(self, config: ProtocolConfig):
        """Initialize CoaXPress interface."""
        self.config = config
        self.connections = self._calculate_connections(config.data_rate_mbps)
        self.is_connected = False
        self.is_streaming = False

        logger.info(f"CoaXPress interface initialized: {self.connections} connections @ {config.data_rate_mbps} Mbps")

    def _calculate_connections(self, data_rate_mbps: float) -> int:
        """Calculate required number of CoaXPress connections."""
        # CXP-12 supports up to 12.5 Gbps per connection
        max_per_connection = 12500  # Mbps
        required_connections = int(np.ceil(data_rate_mbps / max_per_connection))
        return min(max(required_connections, 1), 4)  # 1-4 connections typical

    def connect(self) -> bool:
        """Establish CoaXPress connection."""
        try:
            # Simulate CoaXPress connection establishment
            logger.info("Establishing CoaXPress connection...")
            time.sleep(0.1)  # Simulate connection time

            # Negotiate connection speed
            negotiated_speed = min(self.config.data_rate_mbps, 12500 * self.connections)
            logger.info(f"Negotiated speed: {negotiated_speed} Mbps")

            self.is_connected = True
            logger.info("✓ CoaXPress connection established")
            return True
        except Exception as e:
            logger.error(f"✗ CoaXPress connection failed: {e}")
            return False

    def start_streaming(self) -> bool:
        """Start CoaXPress streaming."""
        if not self.is_connected:
            logger.error("CoaXPress not connected")
            return False

        try:
            # Configure streaming parameters
            self._configure_streaming()
            self.is_streaming = True
            logger.info("✓ CoaXPress streaming started")
            return True
        except Exception as e:
            logger.error(f"✗ CoaXPress streaming failed: {e}")
            return False

    def capture_frame(self) -> Optional[np.ndarray]:
        """Capture a frame via CoaXPress."""
        if not self.is_streaming:
            logger.error("Streaming not active")
            return None

        try:
            # Generate high-quality frame data
            width, height = self.config.resolution

            # CoaXPress typically handles high bit-depth data
            frame_data = np.random.randint(
                0, 2**self.config.bit_depth - 1, (height, width), dtype=np.uint16 if self.config.bit_depth > 8 else np.uint8
            )

            return frame_data
        except Exception as e:
            logger.error(f"✗ CoaXPress frame capture failed: {e}")
            return None

    def stop_streaming(self) -> bool:
        """Stop CoaXPress streaming."""
        self.is_streaming = False
        logger.info("✓ CoaXPress streaming stopped")
        return True

    def _configure_streaming(self):
        """Configure CoaXPress streaming parameters."""
        # Set up trigger modes, packet size, etc.
        pass

    def get_status(self) -> dict[str, Any]:
        """Get CoaXPress interface status."""
        return {
            "protocol": "CoaXPress",
            "connections": self.connections,
            "data_rate_mbps": self.config.data_rate_mbps,
            "resolution": self.config.resolution,
            "frame_rate": self.config.frame_rate,
            "is_connected": self.is_connected,
            "is_streaming": self.is_streaming,
            "bit_depth": self.config.bit_depth,
        }


class GigEVisionInterface:
    """
    GigE Vision interface implementation.

    GigE Vision is an interface standard for industrial cameras based on
    Gigabit Ethernet, providing reliable data transmission over long distances.
    """

    def __init__(self, config: ProtocolConfig):
        """Initialize GigE Vision interface."""
        self.config = config
        self.ip_address = "192.168.1.100"  # Default camera IP
        self.is_connected = False
        self.is_streaming = False

        logger.info(f"GigE Vision interface initialized @ {config.data_rate_mbps} Mbps")

    def connect(self, ip_address: str | None = None) -> bool:
        """Establish GigE Vision connection."""
        if ip_address:
            self.ip_address = ip_address

        try:
            logger.info(f"Connecting to GigE camera at {self.ip_address}...")

            # Simulate network discovery and connection
            time.sleep(0.2)  # Simulate network latency

            # Validate network bandwidth
            max_gige_rate = 1000  # Mbps for Gigabit Ethernet
            if self.config.data_rate_mbps > max_gige_rate:
                logger.warning(f"Requested rate {self.config.data_rate_mbps} Mbps exceeds GigE limit")
                return False

            self.is_connected = True
            logger.info("✓ GigE Vision connection established")
            return True
        except Exception as e:
            logger.error(f"✗ GigE Vision connection failed: {e}")
            return False

    def start_streaming(self) -> bool:
        """Start GigE Vision streaming."""
        if not self.is_connected:
            logger.error("GigE Vision not connected")
            return False

        try:
            # Configure GVSP (GigE Vision Streaming Protocol)
            self._configure_gvsp()
            self.is_streaming = True
            logger.info("✓ GigE Vision streaming started")
            return True
        except Exception as e:
            logger.error(f"✗ GigE Vision streaming failed: {e}")
            return False

    def capture_frame(self) -> Optional[np.ndarray]:
        """Capture a frame via GigE Vision."""
        if not self.is_streaming:
            logger.error("Streaming not active")
            return None

        try:
            # Simulate network packet reception and frame reconstruction
            width, height = self.config.resolution

            frame_data = np.random.randint(
                0, 2**self.config.bit_depth - 1, (height, width, 3), dtype=np.uint16 if self.config.bit_depth > 8 else np.uint8
            )

            return frame_data
        except Exception as e:
            logger.error(f"✗ GigE Vision frame capture failed: {e}")
            return None

    def stop_streaming(self) -> bool:
        """Stop GigE Vision streaming."""
        self.is_streaming = False
        logger.info("✓ GigE Vision streaming stopped")
        return True

    def _configure_gvsp(self):
        """Configure GigE Vision Streaming Protocol."""
        # Set up packet size, inter-packet delay, etc.
        pass

    def get_status(self) -> dict[str, Any]:
        """Get GigE Vision interface status."""
        return {
            "protocol": "GigE Vision",
            "ip_address": self.ip_address,
            "data_rate_mbps": self.config.data_rate_mbps,
            "resolution": self.config.resolution,
            "frame_rate": self.config.frame_rate,
            "is_connected": self.is_connected,
            "is_streaming": self.is_streaming,
            "bit_depth": self.config.bit_depth,
        }


class USB3VisionInterface:
    """
    USB3 Vision interface implementation.

    USB3 Vision is a standard for industrial cameras using USB 3.0,
    providing plug-and-play connectivity with high data rates.
    """

    def __init__(self, config: ProtocolConfig):
        """Initialize USB3 Vision interface."""
        self.config = config
        self.device_id = None
        self.is_connected = False
        self.is_streaming = False

        logger.info(f"USB3 Vision interface initialized @ {config.data_rate_mbps} Mbps")

    def connect(self, device_id: str | None = None) -> bool:
        """Establish USB3 Vision connection."""
        try:
            logger.info("Scanning for USB3 Vision devices...")

            # Simulate device enumeration
            available_devices = ["USB3-CAM-001", "USB3-CAM-002"]

            if device_id and device_id in available_devices:
                self.device_id = device_id
            else:
                self.device_id = available_devices[0]  # Use first available

            logger.info(f"Connecting to device: {self.device_id}")

            # Validate USB3 bandwidth
            max_usb3_rate = 5000  # Mbps for USB 3.0
            if self.config.data_rate_mbps > max_usb3_rate:
                logger.warning(f"Requested rate {self.config.data_rate_mbps} Mbps exceeds USB3 limit")
                return False

            self.is_connected = True
            logger.info("✓ USB3 Vision connection established")
            return True
        except Exception as e:
            logger.error(f"✗ USB3 Vision connection failed: {e}")
            return False

    def start_streaming(self) -> bool:
        """Start USB3 Vision streaming."""
        if not self.is_connected:
            logger.error("USB3 Vision not connected")
            return False

        try:
            # Configure USB3 Vision streaming
            self._configure_streaming()
            self.is_streaming = True
            logger.info("✓ USB3 Vision streaming started")
            return True
        except Exception as e:
            logger.error(f"✗ USB3 Vision streaming failed: {e}")
            return False

    def capture_frame(self) -> Optional[np.ndarray]:
        """Capture a frame via USB3 Vision."""
        if not self.is_streaming:
            logger.error("Streaming not active")
            return None

        try:
            width, height = self.config.resolution

            frame_data = np.random.randint(
                0, 2**self.config.bit_depth - 1, (height, width, 3), dtype=np.uint16 if self.config.bit_depth > 8 else np.uint8
            )

            return frame_data
        except Exception as e:
            logger.error(f"✗ USB3 Vision frame capture failed: {e}")
            return None

    def stop_streaming(self) -> bool:
        """Stop USB3 Vision streaming."""
        self.is_streaming = False
        logger.info("✓ USB3 Vision streaming stopped")
        return True

    def _configure_streaming(self):
        """Configure USB3 Vision streaming parameters."""
        # Set up transfer parameters, buffer management, etc.
        pass

    def get_status(self) -> dict[str, Any]:
        """Get USB3 Vision interface status."""
        return {
            "protocol": "USB3 Vision",
            "device_id": self.device_id,
            "data_rate_mbps": self.config.data_rate_mbps,
            "resolution": self.config.resolution,
            "frame_rate": self.config.frame_rate,
            "is_connected": self.is_connected,
            "is_streaming": self.is_streaming,
            "bit_depth": self.config.bit_depth,
        }


class ProtocolFactory:
    """Factory class for creating protocol interfaces."""

    @staticmethod
    def create_interface(protocol_type: ProtocolType, config: ProtocolConfig):
        """Create appropriate protocol interface."""
        if protocol_type == ProtocolType.MIPI_CSI2:
            return MIPICSIInterface(config)
        elif protocol_type == ProtocolType.COAXPRESS:
            return CoaXPressInterface(config)
        elif protocol_type == ProtocolType.GIGE_VISION:
            return GigEVisionInterface(config)
        elif protocol_type == ProtocolType.USB3_VISION:
            return USB3VisionInterface(config)
        else:
            raise ValueError(f"Unsupported protocol: {protocol_type}")


def demo_protocol_comparison():
    """Demonstrate comparison of different protocols."""
    logger.info("=== Protocol Comparison Demo ===")

    # Common test configuration
    base_config = {"resolution": (1920, 1080), "frame_rate": 60.0, "bit_depth": 12}

    # Protocol-specific configurations
    protocols = [
        (ProtocolType.MIPI_CSI2, 2000),  # 2 Gbps
        (ProtocolType.COAXPRESS, 6250),  # 6.25 Gbps
        (ProtocolType.GIGE_VISION, 800),  # 800 Mbps
        (ProtocolType.USB3_VISION, 3000),  # 3 Gbps
    ]

    results = []

    for protocol_type, data_rate in protocols:
        logger.info(f"\nTesting {protocol_type.value}...")

        config = ProtocolConfig(protocol_type=protocol_type, data_rate_mbps=data_rate, **base_config)

        try:
            # Create interface
            interface = ProtocolFactory.create_interface(protocol_type, config)

            # Test connection
            start_time = time.time()
            connected = interface.connect()
            connection_time = time.time() - start_time

            if not connected:
                logger.error(f"Failed to connect {protocol_type.value}")
                continue

            # Test streaming
            start_time = time.time()
            streaming = interface.start_streaming()
            streaming_time = time.time() - start_time

            if not streaming:
                logger.error(f"Failed to start streaming {protocol_type.value}")
                continue

            # Capture test frames
            frames_captured = 0
            capture_times = []

            for i in range(5):
                start_time = time.time()
                frame = interface.capture_frame()
                capture_time = time.time() - start_time

                if frame is not None:
                    frames_captured += 1
                    capture_times.append(capture_time)

            # Stop streaming
            interface.stop_streaming()

            # Calculate performance metrics
            avg_capture_time = np.mean(capture_times) if capture_times else 0
            effective_fps = 1.0 / avg_capture_time if avg_capture_time > 0 else 0

            result = {
                "protocol": protocol_type.value,
                "data_rate_mbps": data_rate,
                "connection_time": connection_time,
                "streaming_time": streaming_time,
                "frames_captured": frames_captured,
                "avg_capture_time": avg_capture_time,
                "effective_fps": effective_fps,
                "status": interface.get_status(),
            }

            results.append(result)

            logger.info(f"  Connection time: {connection_time:.3f}s")
            logger.info(f"  Streaming setup: {streaming_time:.3f}s")
            logger.info(f"  Frames captured: {frames_captured}/5")
            logger.info(f"  Avg capture time: {avg_capture_time:.3f}s")
            logger.info(f"  Effective FPS: {effective_fps:.1f}")

        except Exception as e:
            logger.error(f"Error testing {protocol_type.value}: {e}")

    # Summary comparison
    logger.info("\n=== Protocol Comparison Summary ===")
    logger.info(f"{'Protocol':<15} {'Data Rate':<12} {'Conn Time':<10} {'Eff FPS':<10}")
    logger.info("-" * 50)

    for result in results:
        logger.info(
            f"{result['protocol']:<15} "
            f"{result['data_rate_mbps']:<12} "
            f"{result['connection_time']:.3f}s{'':<6} "
            f"{result['effective_fps']:.1f}"
        )


def demo_mipi_detailed():
    """Detailed MIPI CSI-2 demonstration."""
    logger.info("=== Detailed MIPI CSI-2 Demo ===")

    # Test different MIPI configurations
    test_configs = [
        {"resolution": (640, 480), "frame_rate": 120, "data_rate": 500},
        {"resolution": (1920, 1080), "frame_rate": 60, "data_rate": 2000},
        {"resolution": (3840, 2160), "frame_rate": 30, "data_rate": 6000},
    ]

    for i, test_config in enumerate(test_configs):
        logger.info(f"\nMIPI Test {i+1}: {test_config['resolution']} @ {test_config['frame_rate']} fps")

        config = ProtocolConfig(
            protocol_type=ProtocolType.MIPI_CSI2,
            resolution=test_config["resolution"],
            frame_rate=test_config["frame_rate"],
            data_rate_mbps=test_config["data_rate"],
            bit_depth=12,
        )

        interface = MIPICSIInterface(config)

        if interface.connect():
            if interface.start_streaming():
                # Capture frames and measure performance
                frame_times = []
                for j in range(10):
                    start_time = time.time()
                    frame = interface.capture_frame()
                    frame_time = time.time() - start_time

                    if frame is not None:
                        frame_times.append(frame_time)

                interface.stop_streaming()

                if frame_times:
                    avg_time = np.mean(frame_times)
                    fps = 1.0 / avg_time
                    logger.info(f"  Average frame time: {avg_time:.4f}s")
                    logger.info(f"  Achieved FPS: {fps:.1f}")
                    logger.info(f"  Target FPS: {test_config['frame_rate']}")
                    logger.info(f"  Performance: {(fps/test_config['frame_rate']*100):.1f}%")


def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Protocol Implementations Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --protocol mipi_csi2 --resolution 1920x1080
  %(prog)s --protocol coaxpress --data-rate 6250
  %(prog)s --demo comparison
  %(prog)s --demo mipi-detailed
        """,
    )

    parser.add_argument(
        "--protocol", choices=["mipi_csi2", "coaxpress", "gige_vision", "usb3_vision"], help="Protocol to demonstrate"
    )

    parser.add_argument("--resolution", type=str, default="1920x1080", help="Resolution in WIDTHxHEIGHT format")

    parser.add_argument("--frame-rate", type=float, default=60.0, help="Frame rate in fps")

    parser.add_argument("--data-rate", type=float, default=2000, help="Data rate in Mbps")

    parser.add_argument("--bit-depth", type=int, default=12, choices=[8, 10, 12, 14, 16], help="Bit depth")

    parser.add_argument("--demo", choices=["comparison", "mipi-detailed"], help="Run specific demo")

    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        logger.info("Protocol Implementations Demo")
        logger.info("=" * 40)

        if args.demo == "comparison":
            demo_protocol_comparison()
        elif args.demo == "mipi-detailed":
            demo_mipi_detailed()
        elif args.protocol:
            # Single protocol demo
            width_str, height_str = args.resolution.split("x")
            resolution = (int(width_str), int(height_str))

            protocol_type = ProtocolType(args.protocol)
            config = ProtocolConfig(
                protocol_type=protocol_type,
                data_rate_mbps=args.data_rate,
                resolution=resolution,
                frame_rate=args.frame_rate,
                bit_depth=args.bit_depth,
            )

            interface = ProtocolFactory.create_interface(protocol_type, config)

            logger.info(f"Testing {protocol_type.value}...")

            if interface.connect():
                if interface.start_streaming():
                    # Capture a few frames
                    for i in range(3):
                        frame = interface.capture_frame()
                        if frame is not None:
                            logger.info(f"Captured frame {i+1}: {frame.shape}")

                    interface.stop_streaming()

                    # Show status
                    status = interface.get_status()
                    logger.info(f"Final status: {status}")
        else:
            # Run all demos
            demo_protocol_comparison()
            demo_mipi_detailed()

        logger.info("=" * 40)
        logger.info("✓ Protocol demonstrations completed!")

    except Exception as e:
        logger.error(f"✗ Demo failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
