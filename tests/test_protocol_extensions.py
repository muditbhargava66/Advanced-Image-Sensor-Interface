"""
Tests for Protocol Extensions and Advanced Features

Tests for:
- MIPI Security Framework
- CoaXPress CXP-12 Extension
- GigE RoCE Transport
- USB3 Discovery and Streaming
"""

import numpy as np
import pytest


class TestMIPISecurity:
    """Tests for MIPI Security Framework."""

    def test_security_config_defaults(self):
        """Test security configuration defaults."""
        from advanced_image_sensor_interface.sensor_interface.protocol.mipi.security import (
            EncryptionAlgorithm,
            SecurityConfig,
            SecurityLevel,
        )

        config = SecurityConfig()
        assert config.security_level == SecurityLevel.STANDARD
        assert config.encryption == EncryptionAlgorithm.AES_128_GCM
        assert config.key_size == 128

    def test_key_manager(self):
        """Test key generation and derivation."""
        from advanced_image_sensor_interface.sensor_interface.protocol.mipi.security import KeyManager

        km = KeyManager(key_size=128)
        master_key = km.generate_master_key()
        assert len(master_key) == 16

        session_key = km.derive_session_key("session_1", b"salt1234567890ab")
        assert len(session_key) == 16

        km.revoke_session_key("session_1")
        assert km.get_session_key("session_1") is None

    def test_authenticator(self):
        """Test authentication flow."""
        from advanced_image_sensor_interface.sensor_interface.protocol.mipi.security import (
            AuthenticationMethod,
            Authenticator,
            PrivilegeLevel,
            SecurityCredentials,
        )

        auth = Authenticator(method=AuthenticationMethod.PRE_SHARED_KEY)

        # Register credentials
        creds = SecurityCredentials(identity="user1", privilege_level=PrivilegeLevel.ADMIN, pre_shared_key=b"secret_key_123")
        auth.register_credentials(creds)

        # Successful authentication
        success, level = auth.authenticate("user1", b"secret_key_123")
        assert success
        assert level == PrivilegeLevel.ADMIN

        # Failed authentication
        success, level = auth.authenticate("user1", b"wrong_key")
        assert not success
        assert level is None

    def test_security_manager_session(self):
        """Test security manager session lifecycle."""
        from advanced_image_sensor_interface.sensor_interface.protocol.mipi.security import (
            MIPISecurityManager,
            PrivilegeLevel,
            SecurityCredentials,
        )

        manager = MIPISecurityManager()

        # Register identity
        creds = SecurityCredentials(identity="device_1", privilege_level=PrivilegeLevel.USER, pre_shared_key=b"device_secret")
        manager.register_identity(creds)

        # Create challenge for HMAC auth
        manager.authenticator.method = manager.authenticator.method.PRE_SHARED_KEY

        # Create session with PSK
        session = manager.create_session("device_1", b"device_secret")
        assert session is not None
        assert session.identity == "device_1"

        # Validate session
        valid = manager.validate_session(session.session_id)
        assert valid is not None

        # Terminate session
        manager.terminate_session(session.session_id)
        assert manager.validate_session(session.session_id) is None


class TestCXP12:
    """Tests for CoaXPress CXP-12 Extension."""

    def test_config_validation(self):
        """Test CXP-12 configuration."""
        from advanced_image_sensor_interface.sensor_interface.protocol.coaxpress.cxp12 import CXP12Config, CXPSpeed

        config = CXP12Config(speed=CXPSpeed.CXP_12, lanes=4)
        assert config.aggregate_bandwidth_gbps == 50.0

    def test_invalid_lane_config(self):
        """Test invalid lane configuration."""
        from advanced_image_sensor_interface.sensor_interface.protocol.coaxpress.cxp12 import CXP12Config

        with pytest.raises(ValueError):
            CXP12Config(lanes=5)

    def test_link_manager(self):
        """Test link manager initialization."""
        from advanced_image_sensor_interface.sensor_interface.protocol.coaxpress.cxp12 import CXP12Config, CXP12LinkManager

        config = CXP12Config(lanes=4)
        lm = CXP12LinkManager(config)

        assert lm.initialize_links()
        assert lm.get_aggregate_bandwidth() == 50.0

        lm.shutdown_links()
        assert lm.get_aggregate_bandwidth() == 0.0

    def test_trigger_controller(self):
        """Test trigger controller."""
        from advanced_image_sensor_interface.sensor_interface.protocol.coaxpress.cxp12 import (
            CXP12Config,
            CXP12TriggerController,
            TriggerMode,
        )

        config = CXP12Config(trigger_mode=TriggerMode.SOFTWARE)
        tc = CXP12TriggerController(config)

        tc.arm()
        assert tc.software_trigger()
        assert tc.get_trigger_count() == 1

        tc.reset_trigger_count()
        assert tc.get_trigger_count() == 0

    def test_driver_lifecycle(self):
        """Test CXP-12 driver lifecycle."""
        from advanced_image_sensor_interface.sensor_interface.protocol.coaxpress.cxp12 import CXP12Config, CXP12Driver

        driver = CXP12Driver(CXP12Config())

        assert driver.connect()
        assert driver.is_connected

        assert driver.start_streaming()
        assert driver.is_streaming

        frame = driver.capture_frame()
        assert frame is not None
        assert frame.dtype == np.uint16

        assert driver.stop_streaming()
        assert driver.disconnect()


class TestRoCE:
    """Tests for GigE RoCE Transport."""

    def test_config_validation(self):
        """Test RoCE configuration."""
        from advanced_image_sensor_interface.sensor_interface.protocol.gige.roce import RoCEConfig, RoCEVersion

        config = RoCEConfig(version=RoCEVersion.ROCE_V2)
        assert config.version == RoCEVersion.ROCE_V2
        assert config.mtu == 4096

    def test_invalid_mtu(self):
        """Test invalid MTU configuration."""
        from advanced_image_sensor_interface.sensor_interface.protocol.gige.roce import RoCEConfig

        with pytest.raises(ValueError):
            RoCEConfig(mtu=1500)  # Not a standard RDMA MTU

    def test_memory_manager(self):
        """Test RDMA memory manager."""
        from advanced_image_sensor_interface.sensor_interface.protocol.gige.roce import MemoryManager

        mm = MemoryManager()
        buffer = b"test_data" * 1000

        region = mm.register_memory(buffer, "region_1")
        assert region.is_registered
        assert region.length == len(buffer)

        assert mm.validate_access(region.lkey, 0, 100)
        assert not mm.validate_access(region.lkey, 0, len(buffer) + 100)

        assert mm.deregister_memory("region_1")
        assert mm.get_region("region_1") is None

    def test_queue_pair(self):
        """Test RDMA queue pair."""
        from advanced_image_sensor_interface.sensor_interface.protocol.gige.roce import QueuePair, RDMAOperation, RoCEConfig

        qp = QueuePair(RoCEConfig(), qp_num=1)
        qp.connect(remote_qp_num=2, remote_gid=bytes(16))

        wr_id = qp.post_send(b"test_data")
        assert wr_id >= 0

        completions = qp.poll_completions()
        assert len(completions) > 0
        assert completions[0].operation == RDMAOperation.SEND

    def test_transport(self):
        """Test RoCE transport."""
        from advanced_image_sensor_interface.sensor_interface.protocol.gige.roce import RoCETransport

        transport = RoCETransport()
        assert transport.initialize()

        qp_num = transport.create_queue_pair()
        transport.connect_qp(qp_num, 1, bytes(16))

        assert transport.send(qp_num, b"test_data")

        stats = transport.get_statistics()
        assert stats.bytes_sent > 0

        transport.shutdown()


class TestUSB3Discovery:
    """Tests for USB3 Device Discovery."""

    def test_discovery(self):
        """Test device discovery."""
        from advanced_image_sensor_interface.sensor_interface.protocol.usb3.discovery import USB3DeviceDiscovery

        discovery = USB3DeviceDiscovery()
        devices = discovery.discover_devices()

        # Should find simulated devices
        assert len(devices) >= 1
        assert discovery.get_device_count() >= 1

    def test_device_filter(self):
        """Test device filtering."""
        from advanced_image_sensor_interface.sensor_interface.protocol.usb3.discovery import (
            DeviceFilter,
            USB3DeviceDiscovery,
            USBSpeed,
        )

        discovery = USB3DeviceDiscovery()

        # Filter by vendor
        filter = DeviceFilter(vendor_id=0x2AB9)
        devices = discovery.discover_devices(device_filter=filter)
        for device in devices:
            assert device.descriptor.vendor_id == 0x2AB9

        # Filter by speed
        filter = DeviceFilter(min_usb_speed=USBSpeed.USB_3_1)
        devices = discovery.discover_devices(device_filter=filter)
        for device in devices:
            assert device.descriptor.usb_speed in (USBSpeed.USB_3_1, USBSpeed.USB_3_2)

    def test_device_factory(self):
        """Test device factory."""
        from advanced_image_sensor_interface.sensor_interface.protocol.usb3.discovery import USB3DeviceFactory

        factory = USB3DeviceFactory()
        config = factory.create_first_available()

        assert config is not None
        assert "vendor_id" in config
        assert "pixel_formats" in config

    def test_hotplug_callbacks(self):
        """Test hot-plug callback registration."""
        from advanced_image_sensor_interface.sensor_interface.protocol.usb3.discovery import USB3DeviceDiscovery

        discovery = USB3DeviceDiscovery()
        callback_called = False

        def callback(notification):
            nonlocal callback_called
            callback_called = True

        discovery.register_hotplug_callback(callback)
        assert discovery.start_hotplug_monitoring()
        discovery.stop_hotplug_monitoring()


class TestUSB3Streaming:
    """Tests for USB3 Streaming."""

    def test_stream_config(self):
        """Test stream configuration."""
        from advanced_image_sensor_interface.sensor_interface.protocol.usb3.streaming import StreamConfig

        config = StreamConfig(buffer_count=10)
        assert config.buffer_count == 10
        assert config.timeout_ms == 5000

    def test_invalid_config(self):
        """Test invalid configuration."""
        from advanced_image_sensor_interface.sensor_interface.protocol.usb3.streaming import StreamConfig

        with pytest.raises(ValueError):
            StreamConfig(buffer_count=1)

    def test_buffer_pool(self):
        """Test buffer pool operations."""
        from advanced_image_sensor_interface.sensor_interface.protocol.usb3.streaming import BufferPool, StreamConfig

        config = StreamConfig(buffer_count=5)
        pool = BufferPool(config)

        assert pool.allocate(1920 * 1080)

        stats = pool.get_statistics()
        assert stats["total_buffers"] == 5
        assert stats["free_buffers"] == 5

        # Get and queue buffer
        buffer = pool.get_free_buffer()
        assert buffer is not None

        stats = pool.get_statistics()
        assert stats["free_buffers"] == 4
        assert stats["queued_buffers"] == 1

        pool.deallocate()

    def test_streaming_manager_lifecycle(self):
        """Test streaming manager lifecycle."""
        from advanced_image_sensor_interface.sensor_interface.protocol.usb3.streaming import StreamState, USB3StreamingManager

        manager = USB3StreamingManager()
        assert manager.state == StreamState.IDLE

        assert manager.prepare(1920, 1080, "Mono8")
        assert manager.start_streaming()
        assert manager.is_streaming

        # Get frame
        result = manager.get_frame()
        assert result is not None
        frame_data, frame_info = result
        assert frame_info.frame_id == 1

        assert manager.stop_streaming()
        assert manager.state == StreamState.IDLE

    def test_streaming_statistics(self):
        """Test streaming statistics."""
        from advanced_image_sensor_interface.sensor_interface.protocol.usb3.streaming import USB3StreamingManager

        manager = USB3StreamingManager()
        manager.prepare(640, 480, "Mono8")
        manager.start_streaming()

        # Capture some frames
        for _ in range(5):
            manager.get_frame()

        stats = manager.get_statistics()
        assert stats.frames_captured == 5
        assert stats.bytes_transferred > 0

        manager.reset_statistics()
        assert manager.get_statistics().frames_captured == 0

        manager.stop_streaming()


class TestProtocolImports:
    """Test that all protocol imports work correctly."""

    def test_mipi_imports(self):
        """Test MIPI package imports."""
        from advanced_image_sensor_interface.sensor_interface.protocol.mipi import (
            DPHY25Config,
            DPHY25Driver,
            MIPIProtocolDriver,
            MIPISecurityManager,
        )

        assert MIPIProtocolDriver is not None
        assert DPHY25Config is not None
        assert DPHY25Driver is not None
        assert MIPISecurityManager is not None

    def test_coaxpress_imports(self):
        """Test CoaXPress package imports."""
        from advanced_image_sensor_interface.sensor_interface.protocol.coaxpress import CoaXPressDriver, CXP12Config

        assert CoaXPressDriver is not None
        assert CXP12Config is not None

    def test_gige_imports(self):
        """Test GigE package imports."""
        from advanced_image_sensor_interface.sensor_interface.protocol.gige import GigEProtocolDriver, RoCETransport

        assert GigEProtocolDriver is not None
        assert RoCETransport is not None

    def test_usb3_imports(self):
        """Test USB3 package imports."""
        from advanced_image_sensor_interface.sensor_interface.protocol.usb3 import (
            USB3DeviceDiscovery,
            USB3StreamingManager,
            USB3VisionDriver,
        )

        assert USB3VisionDriver is not None
        assert USB3DeviceDiscovery is not None
        assert USB3StreamingManager is not None

    def test_base_imports(self):
        """Test base protocol imports."""
        from advanced_image_sensor_interface.sensor_interface.protocol import ProtocolBase, ProtocolError

        assert ProtocolBase is not None
        assert ProtocolError is not None
