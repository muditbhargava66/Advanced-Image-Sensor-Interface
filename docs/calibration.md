# Calibration Guide

This guide documents the calibration-related APIs that are actually available in v3.2.0. The project provides calibration building blocks for lens distortion correction, synchronization, calibration result storage, and neural-tuner experimentation. It does not currently ship a full OpenCV-style `CameraCalibrator` workflow.

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

### Native photogrammetry solver (no OpenCV required)

The `calibration.photogrammetry` module solves camera calibration entirely with numpy/scipy, so it works in environments where OpenCV is not installed.

Two functions are provided:

- `calibrate_camera(object_points_per_view, image_points_per_view, image_size)` — Zhang-style planar-pattern calibration. Per-view homographies are estimated with a normalized DLT, intrinsics are solved in closed form from the homography constraints, per-view extrinsics follow from each homography, and everything is refined by minimizing reprojection error with Levenberg-Marquardt (`scipy.optimize.least_squares`). Requires at least 3 views of a planar pattern with at least 4 points each and returns a `CalibrationResult`.
- `solve_projection_matrix(object_points, image_points)` — general DLT projection-matrix estimation from at least six non-coplanar 3D-2D correspondences, decomposed into intrinsics and pose via RQ factorization.

```python
from advanced_image_sensor_interface.sensor_interface.calibration import (
    calibrate_camera,
    solve_projection_matrix,
)

result = calibrate_camera(object_points_per_view, image_points_per_view, image_size=(640, 480))
print(result.camera_matrix, result.rms_reprojection_error)

intrinsic, rotation, translation = solve_projection_matrix(object_points, image_points)
```

**Limitations (v1):** lens distortion is not estimated; `distortion_coefficients` are returned as zeros. For lenses with significant distortion, prefer the OpenCV calibration path. Degenerate view configurations (e.g. identical poses) are rejected with a `ValueError`.

`MultiSensorSynchronizer.calibrate_sensors()` uses OpenCV for checkerboard corner detection in both modes, but you can opt in to the native solver for the solve step with `SyncConfiguration(prefer_native_calibration=True)`; the default remains `cv2.calibrateCamera`.

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

`NeuralCalibrationTuner` provides a simulation-oriented workflow for feature extraction, quality prediction, and parameter recommendation. It uses scikit-learn's MLPRegressor for neural network-based calibration quality prediction and optimization.

### Features

- **Feature Extraction**: Edge detection (Sobel), corner detection (Harris), pattern regularity analysis, noise level estimation
- **MLPRegressor Training**: Multi-layer perceptron with early stopping and L2 regularization
- **Quality Prediction**: Predicts calibration quality from images and detected corner points
- **Parameter Optimization**: Gradient-based calibration parameter optimization
- **Complete scikit-learn Implementation**: No TensorFlow/PyTorch dependencies required

### Usage

```python
import numpy as np

from advanced_image_sensor_interface.sensor_interface.calibration.models import CalibrationResult
from advanced_image_sensor_interface.sensor_interface.calibration.neural_tuner import NeuralCalibrationTuner

tuner = NeuralCalibrationTuner()

# Prepare calibration sessions with images, detected corners, and calibration results
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

# Train the neural network on calibration sessions
tuner.train([session])

# Predict calibration quality for new images
prediction = tuner.predict_calibration_quality(session["images"], session["image_points"])
print(f"Quality prediction: {prediction}")

# Optimize calibration parameters
optimized = tuner.optimize_calibration_parameters({"num_images": 10, "calibration_flags": 0})
print(f"Optimized parameters: {optimized}")
```

### Implementation Details

The NeuralCalibrationTuner implements:

1. **Feature Extraction Pipeline**:
   - Sobel edge detection for edge strength analysis
   - Harris corner detection for corner response analysis
   - Pattern regularity analysis using 2D FFT on checkerboard patterns
   - Noise level estimation using Laplacian variance

2. **Neural Network Architecture**:
   - Input: Feature vector (edge_stats, corner_stats, pattern_stats, noise_stats)
   - Hidden layers: 2 layers with 64 and 32 neurons (configurable)
   - Activation: ReLU
   - Output: Single quality score (0-1)
   - Loss: MSE + L2 regularization
   - Optimizer: Adam with early stopping

3. **Training Process**:
   - Accepts list of calibration sessions with images, image_points, and CalibrationResult
   - Extracts features from each session
   - Trains MLPRegressor with validation split
   - Implements early stopping on validation loss

4. **Quality Prediction**:
   - Extracts features from new images
   - Runs forward pass through trained MLP
   - Returns normalized quality score

5. **Parameter Optimization**:
   - Uses gradient-based optimization on calibration parameters
   - Optimizes num_images, calibration_flags, and pattern_size
   - Returns optimized parameter dictionary

### API Reference

```python
class NeuralCalibrationTuner:
    def __init__(self):
        """Initialize the neural calibration tuner."""
        pass
    
    def train(self, sessions: list[dict]) -> bool:
        """
        Train the neural network on calibration sessions.
        
        Args:
            sessions: List of calibration sessions, each containing:
                - images: List of calibration images
                - image_points: List of detected corner points
                - calibration_result: CalibrationResult object
        
        Returns:
            True if training successful, False otherwise
        """
        pass
    
    def predict_calibration_quality(self, images: list[np.ndarray], 
                                    image_points: list[np.ndarray]) -> dict:
        """
        Predict calibration quality for given images and corner points.
        
        Args:
            images: List of calibration images
            image_points: List of detected corner points
        
        Returns:
            Dictionary with quality metrics
        """
        pass
    
    def optimize_calibration_parameters(self, params: dict) -> dict:
        """
        Optimize calibration parameters using gradient-based optimization.
        
        Args:
            params: Initial calibration parameters
        
        Returns:
            Optimized parameter dictionary
        """
        pass
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
- `calibrate_sensors()` solves geometric calibration with `cv2.calibrateCamera` by default, or with the native numpy/scipy solver when `prefer_native_calibration=True`; corner detection still requires OpenCV in both modes, and the method returns `False` with an explicit log message when OpenCV is unavailable.
- The native photogrammetry solver does not estimate lens distortion (coefficients are zeros); use the OpenCV path for lenses with significant distortion.
- `NeuralCalibrationTuner` is a lightweight simulation tool using scikit-learn MLPRegressor, not a TensorFlow or PyTorch training pipeline.
- If you need full checkerboard detection, stereo rectification, or camera pose solving, you will need to integrate external computer-vision tooling on top of these data models.
