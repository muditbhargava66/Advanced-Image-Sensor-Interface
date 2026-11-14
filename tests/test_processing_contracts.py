import numpy as np
import pytest

from advanced_image_sensor_interface.sensor_interface.enhanced_sensor import (
    EnhancedSensorInterface,
    RAWFormat,
    SensorConfiguration,
    SensorResolution,
)
from advanced_image_sensor_interface.sensor_interface.hdr_processing import HDRProcessor
from advanced_image_sensor_interface.sensor_interface.image_validation import BitDepthConverter
from advanced_image_sensor_interface.sensor_interface.raw_processing import RAWParameters, RAWProcessor
from advanced_image_sensor_interface.sensor_interface.signal_processing import SignalConfig, SignalProcessor


@pytest.mark.parametrize("value", [0, 100, 1000, 4095])
def test_signal_preserves_declared_depth_on_dark_frames(value):
    processor = SignalProcessor(SignalConfig(12, 0.0, np.eye(3)))
    frame = np.full((8, 8), value, dtype=np.uint16)
    result = processor.process_frame(frame)
    assert result.success, result.error
    assert result.data.dtype == np.uint16
    np.testing.assert_array_equal(result.data, frame)


def test_signal_preserves_twenty_bit_samples():
    processor = SignalProcessor(SignalConfig(20, 0.0, np.eye(3)))
    frame = np.full((8, 8), 524288, dtype=np.uint32)
    result = processor.process_frame(frame)
    assert result.success, result.error
    assert result.data.dtype == np.uint32
    np.testing.assert_array_equal(result.data, frame)


@pytest.mark.parametrize("depth", [9, 18])
def test_signal_rejects_unsupported_configured_depth(depth):
    with pytest.raises(ValueError):
        SignalProcessor(SignalConfig(depth, 0.0, np.eye(3)))


@pytest.mark.parametrize("strength", [float("nan"), float("inf"), -0.1])
def test_signal_rejects_invalid_noise_strength(strength):
    with pytest.raises(ValueError):
        SignalProcessor(SignalConfig(12, strength, np.eye(3)))


@pytest.mark.parametrize("frame", [np.ones((4, 4), dtype=np.int16), np.full((4, 4), np.nan, dtype=np.float32)])
def test_signal_invalid_samples_do_not_report_processing_success(frame):
    result = SignalProcessor(SignalConfig(12, 0.0, np.eye(3))).process_frame(frame)
    assert not result.success
    assert result.data is None
    assert result.error


def test_bit_depth_conversion_preserves_twenty_bit_maximum():
    result = BitDepthConverter.convert_bit_depth(np.array([0, 255], dtype=np.uint8), 8, 20)
    assert result.dtype == np.uint32
    np.testing.assert_array_equal(result, [0, 1048575])


def test_raw_twenty_bit_normalization():
    processor = RAWProcessor(RAWParameters(bit_depth=20, black_level=0, white_level=1048575))
    data = np.array([[0, 524288, 1048575]], dtype=np.uint32)
    normalized = processor._normalize_raw_data(data)
    np.testing.assert_allclose(normalized, [[0.0, 524288 / 1048575, 1.0]], rtol=1e-6)


def test_sensor_raw_twenty_bit_storage():
    config = SensorConfiguration(
        resolution=SensorResolution.CUSTOM,
        custom_resolution=(8, 8),
        raw_format=RAWFormat.RAW20,
        bit_depth=20,
        raw_processing=True,
        black_level=0,
        white_level=1048575,
    )
    sensor = EnhancedSensorInterface(config)
    try:
        assert sensor.start_streaming()
        frame = sensor.capture_frame()
        assert frame is not None
        assert frame.dtype == np.uint32
        assert np.max(frame) > 65535
        assert np.max(frame) <= 1048575
    finally:
        sensor.stop_streaming()


@pytest.mark.parametrize("frame", [b"raw bytes", np.empty((0, 0), dtype=np.uint16), np.full((4, 4), np.nan, dtype=np.float32)])
def test_raw_invalid_input_returns_typed_failure(frame):
    result = RAWProcessor().process_raw_image(frame)
    assert not result.success
    assert result.data is None
    assert result.error


def test_hdr_empty_stack_returns_typed_failure():
    result = HDRProcessor().process_exposure_stack([])
    assert not result.success
    assert result.data is None
    assert result.error
