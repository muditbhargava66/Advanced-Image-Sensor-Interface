"""
Protocol Driver Coverage Tests

Comprehensive tests for all protocol drivers: MIPI, GigE, CoaXPress, USB3.
Tests connection, streaming, frame capture, and statistics functionality.
"""

from unittest.mock import MagicMock

import pytest

from advanced_image_sensor_interface.sensor_interface.protocol.base import ConnectionError
from advanced_image_sensor_interface.sensor_interface.protocol.coaxpress.driver import CoaXPressConfig, CoaXPressDriver
from advanced_image_sensor_interface.sensor_interface.protocol.gige import GigEConfig, GigEDriver, GigEVisionDriver
from advanced_image_sensor_interface.sensor_interface.protocol.gige.driver import GigEProtocolDriver, GigESpeed, GigEVisionConfig
from advanced_image_sensor_interface.sensor_interface.protocol.mipi.driver import MIPIConfig, MIPIProtocolDriver
from advanced_image_sensor_interface.sensor_interface.protocol.usb3.driver import USB3VisionConfig, USB3VisionDriver
from advanced_image_sensor_interface.sensor_interface.protocol_selector import ProtocolSelector, ProtocolType


class TestMIPIProtocol:
    """Tests for MIPI CSI-2 protocol driver."""

    @pytest.fixture
    def mipi_config(self):
        return MIPIConfig(lanes=4, data_rate_mbps=1500.0, pixel_format="RAW10", resolution=(1920, 1080), frame_rate=30.0)

    @pytest.fixture
    def mipi_dict_config(self):
        return {"lanes": 2, "data_rate_mbps": 1000.0, "pixel_format": "RAW8"}

    def test_initialization_with_dataclass(self, mipi_config):
        """Test initialization with dataclass config."""
        driver = MIPIProtocolDriver(mipi_config)
        assert driver.mipi_config == mipi_config
        assert driver.mipi_config.lanes == 4
        assert not driver.is_connected

    def test_initialization_with_dict(self, mipi_dict_config):
        """Test initialization with dict config for backward compatibility."""
        driver = MIPIProtocolDriver(mipi_dict_config)
        assert driver.mipi_config.lanes == 2
        assert not driver.is_connected

    def test_config_validation(self):
        """Test configuration validation."""
        with pytest.raises(ValueError):
            MIPIConfig(lanes=5)  # Invalid: max 4 lanes

        with pytest.raises(ValueError):
            MIPIConfig(data_rate_mbps=5000.0)  # Invalid: max 4500 Mbps

    def test_connect_disconnect(self, mipi_config):
        """Test connection lifecycle."""
        driver = MIPIProtocolDriver(mipi_config)

        assert driver.connect() is True
        assert driver.is_connected
        assert driver.connected  # Legacy property

        assert driver.disconnect() is True
        assert not driver.is_connected
        assert not driver.connected

    def test_streaming_lifecycle(self, mipi_config):
        """Test streaming start/stop."""
        driver = MIPIProtocolDriver(mipi_config)
        driver.connect()

        assert driver.start_streaming() is True
        assert driver.is_streaming

        assert driver.stop_streaming() is True
        assert not driver.is_streaming

        driver.disconnect()

    def test_frame_capture(self, mipi_config):
        """Test frame capture."""
        driver = MIPIProtocolDriver(mipi_config)
        driver.connect()
        driver.start_streaming()

        frame = driver.capture_frame()
        assert frame is not None
        assert len(frame) > 0

        # Verify frame size matches expected
        width, height = mipi_config.resolution
        expected_size = width * height * 2  # RAW10 = 2 bytes
        assert len(frame) == expected_size

        driver.stop_streaming()
        driver.disconnect()

    def test_data_transfer(self, mipi_config):
        """Test control data transfer."""
        driver = MIPIProtocolDriver(mipi_config)
        driver.connect()

        assert driver.send_data(b"test_command")
        data = driver.receive_data(10)
        assert data is not None
        assert len(data) == 10

        driver.disconnect()

    def test_error_without_connection(self, mipi_config):
        """Test error handling when not connected."""
        driver = MIPIProtocolDriver(mipi_config)

        with pytest.raises(ConnectionError):
            driver.send_data(b"fail")

        with pytest.raises(ConnectionError):
            driver.capture_frame()

    def test_device_info(self, mipi_config):
        """Test device info retrieval."""
        driver = MIPIProtocolDriver(mipi_config)
        info = driver.get_device_info()

        assert info["protocol"] == "MIPI CSI-2"
        assert info["lanes"] == 4
        assert info["resolution"] == (1920, 1080)

    def test_statistics(self, mipi_config):
        """Test statistics collection."""
        driver = MIPIProtocolDriver(mipi_config)
        driver.connect()
        driver.start_streaming()

        # Capture some frames
        for _ in range(5):
            driver.capture_frame()

        stats = driver.get_statistics()
        assert stats["frames_captured"] == 5
        assert stats["bytes_transferred"] > 0
        assert stats["current_fps"] > 0

        driver.stop_streaming()
        driver.disconnect()

    def test_capabilities(self, mipi_config):
        """Test protocol capabilities."""
        driver = MIPIProtocolDriver(mipi_config)
        caps = driver.get_capabilities()

        assert caps.max_bandwidth_gbps == 6.0  # 4 lanes * 1500 Mbps
        assert caps.hardware_trigger_support is True
        assert "RAW10" in caps.supported_pixel_formats

    def test_status(self, mipi_config):
        """Test legacy status access."""
        driver = MIPIProtocolDriver(mipi_config)
        status = driver.get_status()

        assert status["protocol"] == "MIPI CSI-2"
        assert status["config"]["data_rate_mbps"] == mipi_config.data_rate_mbps
        assert status["config"]["data_rate"] == pytest.approx(mipi_config.data_rate_mbps / 1000.0)


class TestGigEProtocol:
    """Tests for GigE Vision protocol driver."""

    @pytest.fixture
    def gige_config(self):
        return GigEVisionConfig(
            ip_address="192.168.1.100", speed=GigESpeed.GIGE_1G, packet_size=9000, pixel_format="Mono8", resolution=(1920, 1080)
        )

    @pytest.fixture
    def gige_dict_config(self):
        return {"ip_address": "192.168.1.100", "packet_size": 1500}

    def test_initialization_with_dataclass(self, gige_config):
        """Test initialization with dataclass config."""
        driver = GigEProtocolDriver(gige_config)
        assert driver.gige_config == gige_config
        assert not driver.is_connected

    def test_initialization_with_dict(self, gige_dict_config):
        """Test backward compatibility with dict config."""
        driver = GigEProtocolDriver(gige_dict_config)
        assert driver.gige_config.ip_address == "192.168.1.100"

    def test_legacy_alias_exports(self):
        """Test GitHub issue #1 compatibility aliases."""
        config = GigEConfig(ip_address="192.168.1.100", packet_size=1500)

        assert GigEConfig is GigEVisionConfig
        assert GigEDriver is GigEProtocolDriver
        assert GigEVisionDriver is GigEProtocolDriver
        assert isinstance(GigEDriver(config), GigEProtocolDriver)

    def test_connect_disconnect(self, gige_config):
        """Test connection lifecycle."""
        driver = GigEProtocolDriver(gige_config)

        assert driver.connect() is True
        assert driver.is_connected
        assert driver.connected  # Legacy property

        assert driver.disconnect() is True
        assert not driver.is_connected

    def test_streaming_lifecycle(self, gige_config):
        """Test streaming start/stop."""
        driver = GigEProtocolDriver(gige_config)
        driver.connect()

        assert driver.start_streaming() is True
        assert driver.is_streaming

        assert driver.stop_streaming() is True
        assert not driver.is_streaming

        driver.disconnect()

    def test_frame_capture(self, gige_config):
        """Test frame capture via GVSP."""
        driver = GigEProtocolDriver(gige_config)
        driver.connect()
        driver.start_streaming()

        frame = driver.capture_frame()
        assert frame is not None
        assert len(frame) > 0

        # Verify frame size
        width, height = gige_config.resolution
        expected_size = width * height  # Mono8 = 1 byte
        assert len(frame) == expected_size

        driver.stop_streaming()
        driver.disconnect()

    def test_gvcp_operations(self, gige_config):
        """Test GVCP register operations."""
        driver = GigEProtocolDriver(gige_config)
        driver.connect()

        assert driver.write_register(0x00000000, b"\x12\x34\x56\x78")
        value = driver.read_register(0x00000000)
        assert value is not None
        assert len(value) == 4

        driver.disconnect()

    def test_device_info(self, gige_config):
        """Test device info retrieval."""
        driver = GigEProtocolDriver(gige_config)
        driver.connect()

        info = driver.get_device_info()
        assert info["protocol"] == "GigE Vision"
        assert info["ip_address"] == "192.168.1.100"

        driver.disconnect()

    def test_statistics(self, gige_config):
        """Test statistics collection."""
        driver = GigEProtocolDriver(gige_config)
        driver.connect()
        driver.start_streaming()

        for _ in range(3):
            driver.capture_frame()

        stats = driver.get_statistics()
        assert stats["frames_captured"] == 3
        assert stats["packets_received"] > 0

        driver.stop_streaming()
        driver.disconnect()

    def test_status(self, gige_config):
        """Test legacy status access."""
        driver = GigEProtocolDriver(gige_config)
        status = driver.get_status()

        assert status["protocol"] == "GigE Vision"
        assert status["config"]["ip_address"] == gige_config.ip_address


class TestUSB3Protocol:
    """Tests for USB3 Vision protocol driver."""

    @pytest.fixture
    def usb3_config(self):
        return USB3VisionConfig(usb_speed="SuperSpeed", packet_size=1024)

    def test_initialization(self, usb3_config):
        driver = USB3VisionDriver(usb3_config)
        assert driver.usb3_config == usb3_config
        assert not driver.is_connected

    def test_connect_disconnect(self, usb3_config):
        """Test connection lifecycle with new GenICam architecture."""
        driver = USB3VisionDriver(usb3_config)

        # Connect using the real implementation (simulated)
        assert driver.connect() is True
        assert driver.is_connected

        # Verify GenICam node map is accessible
        assert driver.get_feature("Width") > 0
        assert driver.get_feature("Height") > 0

        assert driver.disconnect() is True
        assert not driver.is_connected

    def test_data_transfer(self, usb3_config):
        """Test data transfer via USB transport layer."""
        driver = USB3VisionDriver(usb3_config)
        driver.connect()

        data = b"test_data" * 100
        assert driver.send_data(data)
        received = driver.receive_data(len(data))
        assert received is not None
        assert len(received) == len(data)

        driver.disconnect()

    def test_genicam_features(self, usb3_config):
        """Test GenICam SFNC feature access."""
        driver = USB3VisionDriver(usb3_config)
        driver.connect()

        # Test getting features
        assert driver.get_feature("Gain") >= 0
        assert driver.get_feature("ExposureTime") > 0
        assert driver.get_feature("PixelFormat") == "Mono8"

        # Test setting features
        assert driver.set_feature("Gain", 5.0) is True
        assert driver.get_feature("Gain") == 5.0

        driver.disconnect()

    def test_streaming_and_capture(self, usb3_config):
        """Test streaming and frame capture."""
        driver = USB3VisionDriver(usb3_config)
        driver.connect()
        driver.start_streaming()

        frame = driver.capture_frame()
        assert frame is not None
        assert len(frame) > 0

        stats = driver.get_statistics()
        assert stats["frames_captured"] >= 1

        driver.stop_streaming()
        driver.disconnect()

    def test_status(self, usb3_config):
        """Test legacy status access."""
        driver = USB3VisionDriver(usb3_config)
        status = driver.get_status()

        assert status["protocol"] == "USB3 Vision"
        assert status["config"]["packet_size"] == usb3_config.packet_size

    def test_error_handling(self, usb3_config):
        driver = USB3VisionDriver(usb3_config)
        with pytest.raises(ConnectionError):
            driver.send_data(b"test")


class TestCoaXPressProtocol:
    """Tests for CoaXPress protocol driver."""

    @pytest.fixture
    def cxp_config(self):
        return CoaXPressConfig(speed_grade="CXP-6", connections=1, resolution=(1024, 1024))

    def test_lifecycle(self, cxp_config):
        """Test connection lifecycle with CXP link layer."""
        driver = CoaXPressDriver(cxp_config)
        assert not driver.is_connected

        driver.connect()
        assert driver.is_connected

        # Verify link layer is connected
        link_status = driver.get_link_status(0)
        assert link_status.bandwidth_mbps > 0

        driver.disconnect()
        assert not driver.is_connected

    def test_pocxp_power(self, cxp_config):
        """Test Power over CoaXPress functionality."""
        driver = CoaXPressDriver(cxp_config)
        driver.connect()

        pocxp = driver.get_pocxp_status()
        assert pocxp.enabled is True
        assert pocxp.voltage_v > 0

        driver.disconnect()

    def test_genicam_features(self, cxp_config):
        """Test GenICam feature access."""
        driver = CoaXPressDriver(cxp_config)
        driver.connect()

        # Test features
        assert driver.get_feature("Width") == 1024
        assert driver.get_feature("Height") == 1024

        # Set feature
        driver.set_feature("Gain", 10.0)
        assert driver.get_feature("Gain") == 10.0

        driver.disconnect()

    def test_trigger_system(self, cxp_config):
        """Test trigger controller."""
        driver = CoaXPressDriver(CoaXPressConfig(speed_grade="CXP-6", trigger_mode="software"))
        driver.connect()

        # Test software trigger
        initial_count = driver.stats["trigger_count"]
        driver.software_trigger()
        assert driver.stats["trigger_count"] == initial_count + 1

        driver.disconnect()

    def test_streaming_and_capture(self, cxp_config):
        """Test streaming and frame capture."""
        driver = CoaXPressDriver(cxp_config)
        driver.connect()
        driver.start_streaming()

        frame = driver.capture_frame()
        assert frame is not None

        stats = driver.get_statistics()
        assert stats["frames_captured"] >= 1

        driver.stop_streaming()
        driver.disconnect()

    def test_status(self, cxp_config):
        """Test legacy status access."""
        driver = CoaXPressDriver(cxp_config)
        status = driver.get_status()

        assert status["protocol"] == "CoaXPress"
        assert status["config"]["speed_grade"] == cxp_config.speed_grade


class TestProtocolSelector:
    """Tests for protocol selector."""

    def test_selection_logic(self):
        selector = ProtocolSelector()

        mipi_config = {"lane_count": 4}
        mipi_driver_mock = MagicMock()
        selector.register_protocol(ProtocolType.MIPI, mipi_driver_mock, mipi_config)

        assert selector is not None

    def test_configure_protocol_compatibility(self):
        """Verify the backward-compatible configure_protocol helper."""
        selector = ProtocolSelector()

        assert selector.configure_protocol(ProtocolType.MIPI, MIPIConfig())
        assert ProtocolType.MIPI in selector.protocol_instances


class TestProtocolConsistency:
    """Tests for consistent interface across all protocols."""

    def test_all_drivers_have_streaming_methods(self):
        """Verify all streaming drivers have consistent interface."""
        drivers = [
            MIPIProtocolDriver(MIPIConfig()),
            GigEProtocolDriver(GigEVisionConfig()),
            CoaXPressDriver(CoaXPressConfig()),
            USB3VisionDriver(USB3VisionConfig()),
        ]

        for driver in drivers:
            assert hasattr(driver, "start_streaming")
            assert hasattr(driver, "stop_streaming")
            assert hasattr(driver, "capture_frame")
            assert hasattr(driver, "get_statistics")
            assert hasattr(driver, "get_status")

    def test_all_drivers_have_connection_methods(self):
        """Verify all drivers have consistent connection interface."""
        drivers = [
            MIPIProtocolDriver(MIPIConfig()),
            GigEProtocolDriver(GigEVisionConfig()),
            CoaXPressDriver(CoaXPressConfig()),
            USB3VisionDriver(USB3VisionConfig()),
        ]

        for driver in drivers:
            assert hasattr(driver, "connect")
            assert hasattr(driver, "disconnect")
            assert hasattr(driver, "send_data")
            assert hasattr(driver, "receive_data")
            assert hasattr(driver, "is_connected")
