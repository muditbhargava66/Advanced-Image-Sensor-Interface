"""Regression contracts for unpacked protocol frames and simulated lifecycles."""

from dataclasses import asdict, fields
from unittest.mock import Mock

import numpy as np
import pytest

from advanced_image_sensor_interface.sensor_interface.protocol.base import StreamingProtocolBase
from advanced_image_sensor_interface.sensor_interface.protocol.coaxpress.driver import CoaXPressConfig, CoaXPressDriver
from advanced_image_sensor_interface.sensor_interface.protocol.gige.driver import GigEProtocolDriver, GigEVisionConfig
from advanced_image_sensor_interface.sensor_interface.protocol.gige.roce import GigERoCEDriver, RoCEConfig
from advanced_image_sensor_interface.sensor_interface.protocol.mipi.driver import MIPIConfig, MIPIProtocolDriver
from advanced_image_sensor_interface.sensor_interface.protocol.usb3.driver import USB3VisionConfig, USB3VisionDriver
from advanced_image_sensor_interface.sensor_interface.protocol.usb3.streaming import (
    BufferPool,
    FrameInfo,
    StreamConfig,
    StreamState,
    USB3StreamingManager,
)
from advanced_image_sensor_interface.sensor_interface.protocol_selector import ProtocolSelector, ProtocolType
from advanced_image_sensor_interface.types import SimulationDelayConfig

DRIVERS = [
    (ProtocolType.MIPI, MIPIProtocolDriver, MIPIConfig),
    (ProtocolType.GIGE, GigEProtocolDriver, GigEVisionConfig),
    (ProtocolType.COAXPRESS, CoaXPressDriver, CoaXPressConfig),
    (ProtocolType.USB3, USB3VisionDriver, USB3VisionConfig),
]


def no_delays():
    return SimulationDelayConfig(**{f.name: 0.0 for f in fields(SimulationDelayConfig) if f.name.endswith("_delay")})


@pytest.fixture
def sleeps(monkeypatch):
    calls = []
    monkeypatch.setattr("time.sleep", calls.append)
    return calls


@pytest.mark.parametrize(
    "fmt,bpp,bits,channels",
    [
        ("RGB888", 3, 8, 3),
        ("RAW6", 1, 6, 1),
        ("RAW7", 1, 7, 1),
        ("RAW10", 2, 10, 1),
        ("RAW12", 2, 12, 1),
        ("RAW14", 2, 14, 1),
        ("RAW20", 4, 20, 1),
        ("Mono20", 4, 20, 1),
        ("RGB10", 6, 10, 3),
        ("RGB16", 6, 16, 3),
        ("BGR8", 3, 8, 3),
        ("BayerGB12", 2, 12, 1),
        ("BayerBG10", 2, 10, 1),
        ("YUV422_10", 4, 10, 2),
        ("YUV444", 3, 8, 3),
    ],
)
def test_unpacked_frame_size_range_and_endianness(fmt, bpp, bits, channels):
    assert StreamingProtocolBase._get_bytes_per_pixel_common(fmt) == bpp
    data = StreamingProtocolBase._generate_test_frame_vectorized(4, 4, bpp, 1021, pixel_format=fmt)
    assert len(data) == 16 * bpp
    dtype = "u1" if bits <= 8 else "<u2" if bits <= 16 else "<u4"
    values = np.frombuffer(data, dtype=dtype).reshape(4, 4, channels)
    assert values.max() <= (1 << bits) - 1
    assert np.unique(values).size > 1
    assert int(values[0, 0, 0]) == 1021 % (1 << bits)
    if bits > 8:
        assert data[: np.dtype(dtype).itemsize] == (1021 % (1 << bits)).to_bytes(np.dtype(dtype).itemsize, "little")


@pytest.mark.parametrize("fmt", ["unknown", "Mono12Packed", "RAW10p", "RGB565", "RGB666"])
def test_unknown_and_bit_packed_formats_fail_explicitly(fmt):
    with pytest.raises(ValueError):
        StreamingProtocolBase._get_bytes_per_pixel_common(fmt)


@pytest.mark.parametrize("protocol,driver_type,config_type", DRIVERS)
@pytest.mark.parametrize("as_mapping", [False, True])
def test_selector_normalizes_config_before_construction(protocol, driver_type, config_type, as_mapping, sleeps):
    config = config_type(resolution=(4, 4), pixel_format="RAW10" if protocol == ProtocolType.MIPI else "Mono10")
    selector = ProtocolSelector()
    assert selector.configure_protocol(protocol, asdict(config) if as_mapping else config)
    driver = selector.protocol_instances[protocol]
    assert isinstance(driver, driver_type)
    assert selector.activate_protocol(protocol)
    frame = driver.capture_frame()
    assert isinstance(frame, bytes) and len(frame) == 32
    assert driver.disconnect()


@pytest.mark.parametrize("raises", [False, True])
def test_selector_clears_disconnected_current_on_failed_connect(sleeps, monkeypatch, raises):
    selector = ProtocolSelector()
    assert selector.configure_protocol(ProtocolType.MIPI, MIPIConfig())
    assert selector.configure_protocol(ProtocolType.GIGE, GigEVisionConfig())
    assert selector.activate_protocol(ProtocolType.MIPI)
    replacement = selector.protocol_instances[ProtocolType.GIGE]
    monkeypatch.setattr(replacement, "connect", Mock(side_effect=RuntimeError("connect failed")) if raises else lambda: False)
    assert not selector.activate_protocol(ProtocolType.GIGE)
    assert selector.current_protocol is None
    assert selector.get_current_protocol() is None


@pytest.mark.parametrize("disconnects", [False, True])
def test_selector_does_not_switch_after_failed_disconnect(sleeps, monkeypatch, disconnects):
    selector = ProtocolSelector()
    for protocol, _, config_type in DRIVERS[:2]:
        assert selector.configure_protocol(protocol, config_type())
    assert selector.activate_protocol(ProtocolType.MIPI)
    previous = selector.get_current_protocol()

    def fail_disconnect():
        previous.is_connected = not disconnects
        return False

    monkeypatch.setattr(previous, "disconnect", fail_disconnect)
    assert not selector.activate_protocol(ProtocolType.GIGE)
    assert not selector.protocol_instances[ProtocolType.GIGE].is_connected
    assert selector.current_protocol == (None if disconnects else ProtocolType.MIPI)


@pytest.mark.parametrize("protocol,driver_type,config_type", DRIVERS)
def test_primary_operation_delays_can_be_disabled(protocol, driver_type, config_type, sleeps):
    config = config_type(resolution=(4, 4), simulation_delays=no_delays())
    driver = driver_type(config)
    assert driver.connect()
    assert driver.send_data(b"abc")
    assert driver.receive_data(4) is not None
    assert driver.start_streaming()
    assert driver.stop_streaming()
    assert driver.disconnect()
    assert all(delay == 0 for delay in sleeps)


@pytest.mark.parametrize("protocol,driver_type,config_type", DRIVERS[1:])
def test_capture_pacing_is_not_disabled_with_operation_delays(protocol, driver_type, config_type, sleeps):
    driver = driver_type(config_type(resolution=(4, 4), frame_rate=25, simulation_delays=no_delays()))
    assert driver.connect()
    sleeps.clear()
    assert driver.capture_frame() is not None
    assert any(delay > 0 for delay in sleeps)
    assert driver.disconnect()


@pytest.mark.parametrize("protocol,driver_type,config_type", DRIVERS)
def test_every_advertised_format_constructs_and_captures(protocol, driver_type, config_type, sleeps):
    formats = driver_type(config_type()).get_capabilities().supported_pixel_formats
    assert "RGB565" not in formats and "RGB666" not in formats
    for fmt in formats:
        driver = driver_type(config_type(resolution=(4, 4), pixel_format=fmt))
        assert driver.connect()
        frame = driver.capture_frame()
        assert isinstance(frame, bytes), fmt
        assert len(frame) == 16 * driver._get_bytes_per_pixel_common(fmt)
        assert driver.disconnect()


@pytest.mark.parametrize("protocol,driver_type,config_type", DRIVERS)
def test_primary_capture_counters_and_metrics_are_measured(protocol, driver_type, config_type, sleeps, monkeypatch):
    clock = [100.0]
    monkeypatch.setattr("time.monotonic", lambda: clock[0])
    driver = driver_type(config_type(resolution=(4, 4), pixel_format="RAW8" if protocol == ProtocolType.MIPI else "Mono8"))
    assert driver.connect()
    assert driver.get_performance_metrics()["data_rate_mbps"] == 0
    assert driver.start_streaming()
    clock[0] += 2
    frame = driver.capture_frame()
    assert len(frame) == 16
    status = driver.status
    assert status.bytes_received == 16
    assert status.uptime_seconds == 2
    metrics = driver.get_performance_metrics()
    stats = driver.get_statistics()
    assert metrics["bytes_received"] == stats["bytes_transferred"] == 16
    assert metrics["data_rate_mbps"] == pytest.approx(16 * 8 / 2 / 1e6)
    assert metrics["uptime_seconds"] == stats["uptime_seconds"] == 2
    assert driver.get_streaming_status()["frame_rate"] == stats["current_fps"] == 0.5
    assert driver.get_streaming_status()["buffer_utilization"] == 0
    assert driver.stop_streaming()
    assert driver.disconnect()
    clock[0] += 10
    assert driver.get_performance_metrics()["uptime_seconds"] == 2
    assert driver.status.data_rate_mbps == 0


@pytest.mark.parametrize("bits,dtype,size", [(8, "u1", 12), (10, "<u2", 24), (12, "<u2", 24), (16, "<u2", 24), (20, "<u4", 48)])
def test_roce_unpacked_frame_storage(bits, dtype, size, sleeps):
    driver = GigERoCEDriver(RoCEConfig(simulation_delays=no_delays()))
    assert driver.connect("192.168.1.100")
    assert driver.start_streaming()
    frame = driver.receive_frame(4, 3, bits)
    assert frame.shape == (3, 4)
    assert frame.dtype == np.dtype(dtype)
    assert frame.nbytes == size
    assert frame.max() < 1 << bits
    assert driver.transport.get_statistics().bytes_received == size
    assert driver.disconnect()
    assert driver.receive_frame(4, 3, bits) is None


def test_primary_gige_roce_capture_and_shutdown(sleeps):
    driver = GigEProtocolDriver(
        GigEVisionConfig(
            enable_roce=True, roce_priority=6, resolution=(4, 4), pixel_format="Mono12", simulation_delays=no_delays()
        )
    )
    assert driver.connect()
    transport = driver._roce_transport
    assert transport is not None
    assert transport.config.traffic_class == 6
    assert transport.config.simulation_delays == driver.gige_config.simulation_delays
    assert driver.start_streaming()
    frame = driver.capture_frame()
    assert isinstance(frame, bytes) and len(frame) == 32
    assert np.frombuffer(frame, dtype="<u2").max() <= 4095
    assert transport.get_statistics().bytes_received == 32
    assert driver.disconnect()
    assert not transport._is_initialized
    assert not transport._queue_pairs
    assert driver._roce_transport is None


def test_primary_gige_roce_initialization_failure_falls_back(sleeps, monkeypatch):
    from advanced_image_sensor_interface.sensor_interface.protocol.gige.roce import RoCETransport

    monkeypatch.setattr(RoCETransport, "initialize", lambda self: False)
    driver = GigEProtocolDriver(GigEVisionConfig(enable_roce=True, resolution=(4, 4)))
    assert driver.connect()
    assert driver._roce_transport is None
    assert len(driver.capture_frame()) == 16
    assert driver.disconnect()


def test_roce_short_frame_is_capture_failure_not_reshape_error(sleeps, monkeypatch):
    driver = GigERoCEDriver()
    assert driver.connect("192.168.1.100") and driver.start_streaming()
    monkeypatch.setattr(driver.transport, "receive", lambda *args, **kwargs: b"bad")
    assert driver.receive_frame(4, 3, 12) is None
    assert driver.disconnect()


@pytest.mark.parametrize(
    "width,height,fmt,size,dtype,maximum",
    [
        (4, 3, "Mono8", 12, "u1", 255),
        (6, 2, "Mono12", 24, "<u2", 4095),
        (4, 5, "RGB16", 120, "<u2", 65535),
        (2, 3, "RAW20", 24, "<u4", 1048575),
    ],
)
def test_two_buffer_stream_retains_format_data_and_owned_frames(*, width, height, fmt, size, dtype, maximum, sleeps):
    manager = USB3StreamingManager(StreamConfig(buffer_count=2, buffer_size=size + 20, simulation_delays=no_delays()))
    assert manager.prepare(width, height, fmt)
    assert manager.start_streaming()
    first, info = manager.get_frame()
    assert first.dtype == np.uint8 and first.ndim == 1 and first.nbytes == size
    assert (info.width, info.height, info.pixel_format, info.payload_size) == (width, height, fmt, size)
    assert np.frombuffer(first, dtype=dtype).max() <= maximum
    assert np.any(first)
    retained = first.copy()
    for _ in range(5):
        current, metadata = manager.get_frame()
        assert not np.shares_memory(first, current)
    np.testing.assert_array_equal(first, retained)
    assert metadata.frame_id == 6
    assert manager.pause_streaming()
    assert manager.get_frame() is None
    assert manager.start_streaming()  # PAUSED must resume, not reset IDs or queue ownership.
    _, info = manager.get_frame()
    assert info.frame_id == 7
    stats = manager.get_statistics()
    assert stats.frames_captured == 7 and stats.bytes_transferred == 7 * size
    assert stats.buffer_underruns == stats.frames_dropped == 0
    assert manager.stop_streaming()
    assert manager.state == StreamState.IDLE
    assert manager.buffer_pool.get_statistics()["total_buffers"] == 0


def test_stream_timeout_limits_capture_without_consuming_frame(sleeps):
    delays = no_delays()
    delays.frame_capture_delay = 0.02
    manager = USB3StreamingManager(StreamConfig(buffer_count=2, simulation_delays=delays))
    assert manager.prepare(4, 4, "Mono8") and manager.start_streaming()
    sleeps.clear()
    assert manager.get_frame(timeout_ms=1) is None
    assert sum(sleeps) <= 0.001
    assert manager.get_statistics().frames_captured == 0
    result = manager.get_frame(timeout_ms=100)
    assert result is not None and result[1].frame_id == 1
    assert manager.stop_streaming()


def test_prepared_stream_cleanup_and_invalid_format(sleeps):
    manager = USB3StreamingManager()
    assert not manager.prepare(4, 4, "unknown")
    assert manager.state == StreamState.IDLE
    assert manager.prepare(4, 4, "Mono8")
    assert manager.stop_streaming()
    assert manager.buffer_pool.get_statistics()["total_buffers"] == 0
    assert manager.state == StreamState.IDLE
    assert manager.prepare(6, 2, "Mono12") and manager.start_streaming()
    assert manager.get_frame()[1].payload_size == 24
    assert manager.stop_streaming()


def test_buffer_pool_rejects_invalid_transitions_and_preserves_fifo():
    pool = BufferPool(StreamConfig(buffer_count=2))
    assert pool.allocate(16)
    info = FrameInfo(1, 0, 0, 1, 4, 4, "Mono8", 16)
    assert not pool.queue_buffer(-1)
    assert not pool.mark_filled(-1, info)
    assert not pool.return_buffer(-1)
    assert not pool.mark_filled(0, info)
    assert not pool.return_buffer(0)
    first = pool.get_free_buffer()
    second = pool.get_free_buffer()
    assert (first.buffer_id, second.buffer_id) == (0, 1)
    assert pool.get_free_buffer() is None
    assert pool.mark_filled(0, info)
    assert not pool.mark_filled(0, info)
    assert pool.mark_filled(1, info)
    assert pool.get_filled_buffer().buffer_id == 0
    assert pool.return_buffer(0)
    assert not pool.return_buffer(0)
    assert pool.get_filled_buffer().buffer_id == 1
    assert pool.return_buffer(1)
    assert pool.get_statistics() == {"total_buffers": 2, "free_buffers": 2, "queued_buffers": 0, "filled_buffers": 0}
    pool.deallocate()


def test_roce_reports_active_transport_not_only_requested_flag(sleeps, monkeypatch):
    from advanced_image_sensor_interface.sensor_interface.protocol.gige.roce import RoCETransport

    driver = GigEProtocolDriver(GigEVisionConfig(enable_roce=True))
    assert not driver.get_device_info()["roce_enabled"]
    assert driver.connect()
    assert driver.get_device_info()["roce_enabled"]
    assert driver.disconnect()
    assert not driver.get_device_info()["roce_enabled"]
    monkeypatch.setattr(RoCETransport, "initialize", lambda self: False)
    assert driver.connect()
    assert not driver.get_device_info()["roce_enabled"]
    assert driver.disconnect()


@pytest.mark.parametrize("protocol,driver_type,config_type", DRIVERS)
def test_primary_capture_failure_counts_only_dropped_frames(protocol, driver_type, config_type, sleeps, monkeypatch):
    driver = driver_type(config_type(resolution=(4, 4)))
    assert driver.connect() and driver.start_streaming()
    method = "_generate_test_frame" if protocol in (ProtocolType.MIPI, ProtocolType.GIGE) else "_generate_frame"

    def fail(*args):
        raise RuntimeError("frame failed")

    monkeypatch.setattr(driver, method, fail)
    assert driver.capture_frame() is None
    assert driver.get_statistics()["frames_captured"] == 0
    assert driver.get_statistics()["frames_dropped"] == 1
    assert driver.status.bytes_received == 0
    assert driver.status.last_error == "frame failed"
    assert driver.get_streaming_status()["dropped_frames"] == 1
    assert driver.disconnect()


@pytest.mark.parametrize("protocol,driver_type,config_type", DRIVERS)
def test_primary_disconnect_does_not_hide_stop_failure(protocol, driver_type, config_type, sleeps, monkeypatch):
    driver = driver_type(config_type(resolution=(4, 4)))
    assert driver.connect() and driver.start_streaming()
    stop = driver.stop_streaming
    monkeypatch.setattr(driver, "stop_streaming", lambda: False)
    assert not driver.disconnect()
    assert driver.is_connected and driver.is_streaming
    monkeypatch.setattr(driver, "stop_streaming", stop)
    assert driver.disconnect()


@pytest.mark.parametrize("protocol,driver_type,config_type", DRIVERS)
def test_primary_lifecycle_uses_configured_operation_delays(protocol, driver_type, config_type, sleeps):
    delays = no_delays()
    delays.streaming_setup_delay = 0.123
    delays.streaming_teardown_delay = 0.234
    delays.frame_capture_delay = 0.345
    driver = driver_type(config_type(resolution=(4, 4), simulation_delays=delays))
    assert driver.connect()
    sleeps.clear()
    assert driver.start_streaming()
    assert 0.123 in sleeps
    sleeps.clear()
    assert driver.capture_frame() is not None
    assert 0.345 in sleeps
    sleeps.clear()
    assert driver.stop_streaming()
    assert 0.234 in sleeps
    assert driver.disconnect()


def test_primary_roce_repeated_connect_keeps_single_transport(sleeps):
    driver = GigEProtocolDriver(GigEVisionConfig(enable_roce=True))
    assert driver.connect()
    transport = driver._roce_transport
    assert driver.connect()
    assert driver._roce_transport is transport
    assert len(transport._queue_pairs) == 1
    assert driver.disconnect()
    assert not transport._is_initialized


def test_roce_failed_queue_connection_releases_resources(sleeps, monkeypatch):
    driver = GigERoCEDriver()
    monkeypatch.setattr(driver.transport, "connect_qp", lambda *args: False)
    assert not driver.connect("192.168.1.100")
    assert not driver.transport._is_initialized
    assert not driver.transport._queue_pairs
    assert not driver._is_connected


def test_roce_shutdown_releases_memory_and_receive_work(sleeps):
    driver = GigERoCEDriver()
    assert driver.connect("192.168.1.100") and driver.start_streaming()
    region = driver.transport.memory_manager.register_memory(b"buffer", "frame")
    for _ in range(4):
        assert driver.receive_frame(4, 4, 10) is not None
    qp = driver.transport._queue_pairs[driver._stream_qp]
    assert not qp._recv_queue
    assert driver.disconnect()
    assert driver._stream_qp is None
    assert not region.is_registered
    assert driver.transport.memory_manager.get_region("frame") is None


def test_primary_roce_capture_error_and_shutdown_retry(sleeps, monkeypatch):
    driver = GigEProtocolDriver(GigEVisionConfig(enable_roce=True, resolution=(4, 4)))
    assert driver.connect() and driver.start_streaming()
    transport = driver._roce_transport
    monkeypatch.setattr(transport, "receive", lambda *args: None)
    assert driver.capture_frame() is None
    assert driver.status.bytes_received == 0
    assert driver.get_statistics()["frames_dropped"] == 1
    shutdown = transport.shutdown

    def fail_shutdown():
        raise RuntimeError("shutdown failed")

    monkeypatch.setattr(transport, "shutdown", fail_shutdown)
    assert not driver.disconnect()
    assert driver._roce_transport is transport and driver.is_connected
    monkeypatch.setattr(transport, "shutdown", shutdown)
    assert driver.disconnect()


def test_stream_fifo_producer_reservation_and_callback_retention(sleeps):
    manager = USB3StreamingManager(StreamConfig(buffer_count=2, simulation_delays=no_delays()))
    frames = []
    manager.register_frame_callback(lambda data, info: frames.append((data, info)))
    assert manager.prepare(4, 4, "Mono10") and manager.start_streaming()
    first = manager.get_frame()
    retained = first[0].copy()
    for _ in range(5):
        assert manager.get_frame() is not None
    np.testing.assert_array_equal(frames[0][0], retained)
    assert [info.frame_id for _, info in frames] == list(range(1, 7))
    assert manager.pause_streaming() and manager.resume_streaming()
    assert manager.get_frame()[1].frame_id == 7
    assert manager.stop_streaming()


def test_zero_timeout_and_invalid_pool_allocation(sleeps):
    manager = USB3StreamingManager(StreamConfig(buffer_count=2, simulation_delays=no_delays()))
    assert manager.prepare(4, 4, "Mono8") and manager.start_streaming()
    assert manager.get_frame(timeout_ms=0) is None
    with pytest.raises(ValueError):
        manager.get_frame(timeout_ms=-1)
    assert manager.get_frame()[1].frame_id == 1
    assert manager.stop_streaming()
    pool = BufferPool(StreamConfig(buffer_count=2))
    assert not pool.allocate(0)
    assert pool.allocate(16)
    assert not pool.allocate(16)
    assert pool.queue_buffer(0)
    buffer = pool.get_queued_buffer()
    assert buffer.buffer_id == 0
    assert pool.get_queued_buffer() is None
    assert not pool.queue_buffer(0)
    info = FrameInfo(1, 0, 0, 1, 4, 4, "Mono8", 17)
    assert not pool.mark_filled(0, info)
    assert pool.return_buffer(0)
    pool.deallocate()
