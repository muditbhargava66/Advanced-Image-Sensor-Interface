"""Capture-to-processing contracts with real codecs, processors, and selectors."""

from importlib import import_module
from time import perf_counter

import numpy as np
import pytest

from advanced_image_sensor_interface.sensor_interface.image_validation import ImageFormat, SupportedDType
from advanced_image_sensor_interface.sensor_interface.protocol.base import (
    DataTransferError,
    ProtocolBase,
    ProtocolCapabilities,
    StreamingProtocolBase,
)
from advanced_image_sensor_interface.sensor_interface.protocol_selector import ProtocolSelector, ProtocolType
from advanced_image_sensor_interface.sensor_interface.raw_processing import DemosaicMethod, RAWParameters, RAWProcessor
from advanced_image_sensor_interface.sensor_interface.signal_processing import SignalConfig, SignalProcessor
from advanced_image_sensor_interface.types import DepthResult, ProcessingMetrics, SignalProcessingResult
from advanced_image_sensor_interface.utils.data_integrity import ErrorCorrectionMode, IntegrityChecker, IntegrityConfig
from advanced_image_sensor_interface.utils.depth import DepthConfig, StereoDepthProcessor


def pipeline_api():
    return import_module("advanced_image_sensor_interface.sensor_interface.pipeline")


def image_format(bits=12, *, height=2, width=2, channels=1):
    dtype = SupportedDType.UINT8 if bits == 8 else SupportedDType.UINT32 if bits == 20 else SupportedDType.UINT16
    return ImageFormat(height, width, channels, bits, dtype)


class WireTransport(StreamingProtocolBase):
    """Only the wire is doubled, to inject bytes unavailable from normal capture."""

    def __init__(self, payload):
        super().__init__({})
        self.payload = payload

    def _get_capabilities(self):
        return ProtocolCapabilities(1.0, 1.0, False, False, False, False, False, ["Mono12"], [(2, 2)])

    def connect(self):
        self.is_connected = True
        return True

    def disconnect(self):
        self.is_connected = self.is_streaming = False
        return True

    def send_data(self, data):
        self.payload = data
        return True

    def receive_data(self, size):
        return self.payload

    def start_streaming(self):
        self.is_streaming = True
        return True

    def stop_streaming(self):
        self.is_streaming = False
        return True

    def capture_frame(self):
        if isinstance(self.payload, Exception):
            raise self.payload
        return self.payload


class NonStreamingTransport(ProtocolBase):
    _get_capabilities = WireTransport._get_capabilities
    connect = WireTransport.connect
    disconnect = WireTransport.disconnect
    send_data = WireTransport.send_data
    receive_data = WireTransport.receive_data


def wire_pipeline(payload, fmt=None, integrity=None):
    api = pipeline_api()
    selector = ProtocolSelector()
    wire = WireTransport(payload)
    assert selector.register_protocol(ProtocolType.MIPI, wire, {})
    assert selector.activate_protocol(ProtocolType.MIPI)
    assert wire.start_streaming()
    return api.FramePipeline(selector, api.FrameCodec(fmt or image_format(), integrity)), wire


@pytest.mark.parametrize(
    "bits,values,wire",
    [
        (8, [0, 1, 128, 255], b"\x00\x01\x80\xff"),
        (10, [0, 1, 256, 1023], b"\x00\x00\x01\x00\x00\x01\xff\x03"),
        (12, [0, 1, 256, 4095], b"\x00\x00\x01\x00\x00\x01\xff\x0f"),
        (14, [0, 1, 256, 16383], b"\x00\x00\x01\x00\x00\x01\xff\x3f"),
        (16, [0, 1, 256, 65535], b"\x00\x00\x01\x00\x00\x01\xff\xff"),
        (20, [0, 1, 65536, 1048575], b"\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x01\x00\xff\xff\x0f\x00"),
    ],
)
def test_codec_literal_little_endian_unpacked_storage(bits, values, wire):
    fmt = image_format(bits)
    codec = pipeline_api().FrameCodec(fmt)
    image = np.array(values, dtype=fmt.numpy_dtype).reshape(2, 2)
    assert codec.encode(image) == wire
    decoded = codec.decode(wire)
    np.testing.assert_array_equal(decoded, image)
    assert decoded.dtype == fmt.numpy_dtype
    assert decoded.flags.owndata


@pytest.mark.parametrize("bits", [10, 12, 14, 16, 20])
def test_dark_frames_keep_explicit_storage_instead_of_inferred_8bit(bits):
    fmt = image_format(bits)
    codec = pipeline_api().FrameCodec(fmt)
    image = np.zeros(fmt.shape, dtype=fmt.numpy_dtype)
    decoded = codec.decode(codec.encode(image))
    assert decoded.dtype == fmt.numpy_dtype
    np.testing.assert_array_equal(decoded, image)


@pytest.mark.parametrize("channels", [3, 4])
def test_codec_color_and_noncontiguous_arrays(channels):
    fmt = image_format(12, channels=channels)
    codec = pipeline_api().FrameCodec(fmt)
    image = np.arange(4 * channels, dtype=np.uint16).reshape(fmt.shape)[:, ::-1]
    np.testing.assert_array_equal(codec.decode(codec.encode(image)), image)


@pytest.mark.parametrize("payload", [b"", b"\x00" * 7, b"\x00" * 9, b"\x00\x10" * 4, None, "not bytes", 8])
def test_invalid_payload_raises_transport_error(payload):
    codec = pipeline_api().FrameCodec(image_format())
    with pytest.raises(DataTransferError):
        codec.decode(payload)


@pytest.mark.parametrize(
    "image", [np.zeros((1, 4), dtype=np.uint16), np.zeros((2, 2), dtype=np.uint8), np.full((2, 2), 4096, dtype=np.uint16)]
)
def test_encode_validates_shape_dtype_and_significant_bits(image):
    codec = pipeline_api().FrameCodec(image_format())
    with pytest.raises(ValueError):
        codec.encode(image)


def test_float_format_is_not_an_unpacked_integer_wire_format():
    with pytest.raises(ValueError):
        pipeline_api().FrameCodec(ImageFormat(2, 2, 1, 12, SupportedDType.FLOAT32))


def test_decode_owns_memory_even_when_transport_reuses_buffer():
    payload = bytearray(b"\x01\x00" * 4)
    pipeline, wire = wire_pipeline(payload)
    first = pipeline.capture_frame()
    payload[:] = b"\x02\x00" * 4
    second = pipeline.capture_frame()
    np.testing.assert_array_equal(first, np.ones((2, 2), dtype=np.uint16))
    np.testing.assert_array_equal(second, np.full((2, 2), 2, dtype=np.uint16))
    assert first.flags.owndata and second.flags.owndata
    wire.disconnect()


@pytest.mark.parametrize("mask,accepted", [(0, True), (0x10, True), (3, False)])
def test_sender_envelope_survives_only_proven_hamming_recovery(mask, accepted):
    integrity = IntegrityChecker(IntegrityConfig(fec_mode=ErrorCorrectionMode.HAMMING))
    codec = pipeline_api().FrameCodec(image_format(), integrity)
    image = np.array([[1, 13], [300, 4095]], dtype=np.uint16)
    wire = bytearray(codec.encode(image))
    wire[0] ^= mask
    pipeline, _ = wire_pipeline(bytes(wire), integrity=integrity)
    if accepted:
        np.testing.assert_array_equal(pipeline.capture_frame(), image)
    else:
        with pytest.raises(DataTransferError):
            pipeline.capture_frame()


def test_integrity_is_sender_opt_in_not_checksum_of_received_bytes():
    integrity = IntegrityChecker(IntegrityConfig(enable_fec=False))
    codec = pipeline_api().FrameCodec(image_format(), integrity)
    with pytest.raises(DataTransferError):
        codec.decode(b"\x01\x00" * 4)
    # A valid envelope still cannot make a wrong-size frame valid.
    with pytest.raises(DataTransferError):
        codec.decode(integrity.protect(b"\x01\x00" * 3))


@pytest.mark.parametrize("state", ["absent", "nonstreaming", "disconnected", "stopped"])
def test_capture_requires_selected_connected_streaming_protocol(state):
    pipeline, wire = wire_pipeline(b"\x00" * 8)
    if state == "absent":
        pipeline.selector.current_protocol = None
    elif state == "nonstreaming":
        plain = NonStreamingTransport({})
        plain.connect()
        pipeline.selector.protocol_instances[ProtocolType.MIPI] = plain
    elif state == "disconnected":
        wire.disconnect()
    else:
        wire.stop_streaming()
    with pytest.raises(DataTransferError):
        pipeline.capture_frame()


def test_none_capture_is_not_a_guessed_array():
    pipeline, _ = wire_pipeline(None)
    with pytest.raises(DataTransferError):
        pipeline.capture_frame()


def test_capture_metrics_use_elapsed_time_and_actual_wire_size(monkeypatch):
    pipeline, _ = wire_pipeline(b"\x00" * 8)
    ticks = iter([10.0, 10.25])
    monkeypatch.setattr(pipeline_api(), "perf_counter", lambda: next(ticks))
    pipeline.capture_frame()
    history = pipeline.selector.get_performance_history(ProtocolType.MIPI)
    assert len(history) == 1
    assert history[0]["latency_ms"] == 250.0
    assert history[0]["payload_bytes"] == 8
    assert history[0]["throughput_mbps"] == pytest.approx(0.000256)


def test_failed_callback_result_and_measured_processing_time_are_preserved(monkeypatch):
    pipeline, _ = wire_pipeline(b"\x01\x00" * 4)
    failure = SignalProcessingResult(
        False, error="calibration failed", warnings=["keep me"], metrics=ProcessingMetrics(algorithm_name="caller")
    )
    ticks = iter([1.0, 1.25, 2.0, 2.5])
    monkeypatch.setattr(pipeline_api(), "perf_counter", lambda: next(ticks))

    def operation(frame):
        np.testing.assert_array_equal(frame, np.ones((2, 2), dtype=np.uint16))
        return failure

    result = pipeline.process_frame(operation)
    assert result is failure
    assert not result.success and result.error == "calibration failed"
    assert result.warnings == ["keep me"] and result.metrics.algorithm_name == "caller"
    assert result.metrics.processing_time_ms == 500.0
    assert pipeline.selector.get_performance_history(ProtocolType.MIPI)[-1]["processing_time_ms"] == 500.0


def test_callback_exception_is_not_hidden():
    pipeline, _ = wire_pipeline(b"\x00" * 8)

    def operation(frame):
        raise RuntimeError("callback failed")

    with pytest.raises(RuntimeError, match="callback failed"):
        pipeline.process_frame(operation)


@pytest.mark.parametrize("protocol", list(ProtocolType))
@pytest.mark.parametrize("operation_name", ["signal", "raw"])
def test_real_primary_driver_to_codec_to_typed_processing(protocol, operation_name):
    selector = ProtocolSelector()
    config = {"resolution": (16, 12), "pixel_format": "RAW12" if protocol is ProtocolType.MIPI else "Mono12", "frame_rate": 100.0}
    assert selector.configure_protocol(protocol, config)
    driver = selector.protocol_instances[protocol]
    try:
        assert selector.activate_protocol(protocol)
        assert driver.start_streaming()
        api = pipeline_api()
        pipeline = api.FramePipeline(selector, api.FrameCodec(image_format(height=12, width=16)))
        if operation_name == "signal":
            operation = SignalProcessor(SignalConfig(12, 0.0, np.eye(3))).process_frame
            expected_shape, expected_dtype = (12, 16), np.uint16
        else:
            operation = RAWProcessor(
                RAWParameters(
                    bit_depth=12, black_level=0, white_level=4095, demosaic_method=DemosaicMethod.BILINEAR, noise_reduction=False
                )
            ).process_raw_image
            expected_shape, expected_dtype = (12, 16, 3), np.uint8
        start = perf_counter()
        result = pipeline.process_frame(operation)
        elapsed_ms = (perf_counter() - start) * 1000
        assert result.success, result.error
        assert result.data.shape == expected_shape and result.data.dtype == expected_dtype
        capture = selector.get_performance_history(protocol)[0]
        assert capture["payload_bytes"] == 384
        assert 0 < capture["latency_ms"] <= elapsed_ms
        assert capture["throughput_mbps"] == pytest.approx(384 * 8 / (capture["latency_ms"] * 1000))
        assert 0 < result.metrics.processing_time_ms <= elapsed_ms
    finally:
        driver.disconnect()


def stereo_images():
    left = np.random.default_rng(17).integers(0, 256, (12, 24), dtype=np.uint8)
    right = np.zeros_like(left)
    right[:, :-3] = left[:, 3:]
    return left, right


def depth_processor():
    return StereoDepthProcessor(DepthConfig(window_size=3, num_disparities=6))


def test_stereo_end_to_end_exports_real_ply(tmp_path):
    left, right = stereo_images()
    fmt = image_format(8, height=12, width=24)
    left_pipeline, _ = wire_pipeline(left.tobytes(), fmt)
    right_pipeline, _ = wire_pipeline(right.tobytes(), fmt)
    path = tmp_path / "cloud.ply"
    result = left_pipeline.process_stereo_pair(right_pipeline, depth_processor(), 120.0, 0.1, export_path=path)
    assert isinstance(result, DepthResult) and result.success, result.error
    assert np.median(result.disparity_map[2:-2, 6:-2]) == 3.0
    np.testing.assert_allclose(np.median(result.depth_map[2:-2, 6:-2]), 4.0)
    assert result.point_cloud.shape[1] == 3 and result.point_cloud.shape[0] > 0
    assert path.read_bytes().startswith(b"ply\nformat binary_little_endian 1.0\n")


@pytest.mark.parametrize("payload", [None, b"short", RuntimeError("wire failed")])
def test_stereo_capture_failure_returns_typed_failure_without_arrays(payload):
    left, _ = wire_pipeline(b"\x00" * 8)
    right, _ = wire_pipeline(payload)
    result = left.process_stereo_pair(right, depth_processor(), 120.0, 0.1)
    assert isinstance(result, DepthResult) and not result.success
    assert result.error
    assert result.disparity_map is result.depth_map is result.point_cloud is None


def test_depth_export_invalid_destination_returns_typed_failure(tmp_path):
    left, right = stereo_images()
    result = depth_processor().process_stereo_pair(left, right, 120.0, 0.1, export_path=tmp_path)
    assert isinstance(result, DepthResult) and not result.success
    assert result.error


def test_depth_export_false_return_is_not_success(tmp_path):
    class RefusingExporter(StereoDepthProcessor):
        def export_ply(self, point_cloud, path, binary=True):
            return False

    left, right = stereo_images()
    result = RefusingExporter(DepthConfig(window_size=3, num_disparities=6)).process_stereo_pair(
        left, right, 120.0, 0.1, tmp_path / "refused.ply"
    )
    assert not result.success
    assert "export" in result.error.lower()
