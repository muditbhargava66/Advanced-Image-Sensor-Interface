# Calibration Guide

This guide documents the calibration-related APIs that are actually available in v3.0.0. The project provides calibration building blocks for lens distortion correction, synchronization, calibration result storage, and neural-tuner experimentation. It does not currently ship a full OpenCV-style `CameraCalibrator` workflow.

## Available Components

### Lens correction

Use `advanced_image_sensor_interface.utils.lens_correction` for radial and tangential distortion correction:

- `LensProfile`: distortion coefficients and optical parameters
- `LensCorrectionPipeline`: applies the combined correction maps
- `STANDARD_PROFILES`: ready-made profiles for common lens classes
- `InterpolationMethod`: nearest or bilinear resampling

```python
import numpy as np

from advanced_image_sensor_interface.utils.lens_correction import (
    InterpolationMethod,
    LensCorrectionPipeline,
    LensProfile,
)

frame = np.random.randint(0, 256, (1080, 1920, 3), dtype=np.uint8)

profile = LensProfile.barrel_distortion(strength=0.12, image_size=(1920, 1080))
pipeline = LensCorrectionPipeline(profile)

result = pipeline.correct(frame, interpolation=InterpolationMethod.BILINEAR)

print(result.image.shape)
print(result.processing_time_ms)
print(result.max_displacement)
```

### Standard lens profiles

```python
from advanced_image_sensor_interface.utils.lens_correction import LensCorrectionPipeline, STANDARD_PROFILES

pipeline = LensCorrectionPipeline(STANDARD_PROFILES["industrial_cs"])
result = pipeline.correct(frame)

print(pipeline.get_statistics())
```

### Multi-sensor synchronization and geometric calibration

Use `MultiSensorSynchronizer` when you need synchronized capture plus placeholder geometric-calibration state.

```python
from advanced_image_sensor_interface.sensor_interface.multi_sensor_sync import (
    MultiSensorSynchronizer,
    create_stereo_sync_config,
)

sync_config = create_stereo_sync_config()
sync = MultiSensorSynchronizer(sync_config)

sync.start_synchronization()
frames = sync.capture_synchronized_frames()
sync.calibrate_sensors()
status = sync.get_synchronization_status()
sync.stop_synchronization()

print(status["statistics"])
```

For larger rigs, `create_multi_camera_sync_config(num_cameras=4)` builds a multi-camera configuration with one master sensor and multiple slaves.

## Calibration Models

The `advanced_image_sensor_interface.sensor_interface.calibration` package contains structured result objects and storage helpers:

- `CalibrationResult`
- `StereoCalibrationResult`
- `CalibrationQualityMetrics`
- `CalibrationPattern`
- `CalibrationConfiguration`
- `CalibrationDatabase`

```python
import numpy as np

from advanced_image_sensor_interface.sensor_interface.calibration.models import (
    CalibrationDatabase,
    CalibrationPattern,
    CalibrationResult,
)

pattern = CalibrationPattern(pattern_type="checkerboard", pattern_size=(9, 6), square_size=25.0)
object_points = [pattern.generate_object_points()]
image_points = [np.zeros((54, 2), dtype=np.float32)]

result = CalibrationResult(
    camera_matrix=np.eye(3, dtype=np.float32),
    distortion_coefficients=np.zeros(5, dtype=np.float32),
    rotation_vectors=[np.zeros(3, dtype=np.float32)],
    translation_vectors=[np.zeros(3, dtype=np.float32)],
    rms_reprojection_error=0.42,
    image_size=(1920, 1080),
    calibration_flags=0,
    object_points=object_points,
    image_points=image_points,
)

database = CalibrationDatabase()
database.store_calibration("camera_0", result)

print(database.list_calibrations())
print(database.export_calibration("camera_0"))
```

## Neural Calibration Tuner

`NeuralCalibrationTuner` provides a simulation-oriented workflow for feature extraction, quality prediction, and parameter recommendation.

```python
import numpy as np

from advanced_image_sensor_interface.sensor_interface.calibration.models import CalibrationResult
from advanced_image_sensor_interface.sensor_interface.calibration.neural_tuner import NeuralCalibrationTuner

tuner = NeuralCalibrationTuner()

dummy_result = CalibrationResult(
    camera_matrix=np.eye(3, dtype=np.float32),
    distortion_coefficients=np.zeros(5, dtype=np.float32),
    rotation_vectors=[np.zeros(3, dtype=np.float32)],
    translation_vectors=[np.zeros(3, dtype=np.float32)],
    rms_reprojection_error=0.6,
    image_size=(1920, 1080),
    calibration_flags=0,
    object_points=[np.zeros((54, 3), dtype=np.float32)],
    image_points=[np.zeros((54, 2), dtype=np.float32)],
)

session = {
    "images": [np.random.randint(0, 256, (480, 640), dtype=np.uint8)],
    "image_points": [np.zeros((54, 2), dtype=np.float32)],
    "calibration_result": dummy_result,
}

tuner.train([session])
prediction = tuner.predict_calibration_quality(session["images"], session["image_points"])
optimized = tuner.optimize_calibration_parameters({"num_images": 10, "calibration_flags": 0})

print(prediction)
print(optimized)
```

## Color Calibration Hooks

Two existing processing pipelines expose calibration-friendly color transforms:

- `RAWParameters.color_matrix` in `raw_processing.py`
- `SignalConfig.color_correction_matrix` in `signal_processing.py`

```python
import numpy as np

from advanced_image_sensor_interface.sensor_interface.raw_processing import RAWParameters
from advanced_image_sensor_interface.sensor_interface.signal_processing import SignalConfig

raw_params = RAWParameters(color_matrix=np.eye(3, dtype=np.float32))
signal_config = SignalConfig(
    bit_depth=12,
    noise_reduction_strength=0.1,
    color_correction_matrix=np.eye(3, dtype=np.float32),
)
```

## Current Scope and Limitations

- Lens correction is fully implemented in NumPy and intended for simulation and algorithm work.
- `calibrate_sensors()` currently fills placeholder calibration matrices rather than solving a full photogrammetry problem.
- `NeuralCalibrationTuner` is a lightweight simulation tool, not a TensorFlow or PyTorch training pipeline.
- If you need full checkerboard detection, stereo rectification, or camera pose solving, you will need to integrate external computer-vision tooling on top of these data models.
