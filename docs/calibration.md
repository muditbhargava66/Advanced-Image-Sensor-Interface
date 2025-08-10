# Camera Calibration Guide

This comprehensive guide covers camera calibration procedures for the Advanced Image Sensor Interface, including single camera calibration, multi-camera synchronization, and advanced calibration techniques.

## Overview

Camera calibration is essential for accurate image processing and computer vision applications. This guide covers:

- **Intrinsic Calibration**: Camera internal parameters
- **Extrinsic Calibration**: Camera pose and position
- **Multi-Camera Calibration**: Stereo and multi-camera systems
- **Distortion Correction**: Lens distortion compensation
- **Color Calibration**: Color accuracy and consistency
- **Temporal Calibration**: Frame timing and synchronization

## Intrinsic Camera Calibration

### Overview
Intrinsic calibration determines the internal camera parameters including focal length, principal point, and distortion coefficients.

### Camera Model
The pinhole camera model with distortion is used:

```
x' = x(1 + k1*r² + k2*r⁴ + k3*r⁶) + 2*p1*x*y + p2*(r² + 2*x²)
y' = y(1 + k1*r² + k2*r⁴ + k3*r⁶) + p1*(r² + 2*y²) + 2*p2*x*y
```

Where:
- `(x, y)` are normalized image coordinates
- `(x', y')` are distorted coordinates
- `k1, k2, k3` are radial distortion coefficients
- `p1, p2` are tangential distortion coefficients
- `r² = x² + y²`

### Calibration Implementation

```python
import numpy as np
from advanced_image_sensor_interface.sensor_interface.enhanced_sensor import EnhancedSensorInterface
from advanced_image_sensor_interface.utils.calibration import CameraCalibrator

# Initialize calibration system
calibrator = CameraCalibrator()
sensor = EnhancedSensorInterface()

# Calibration configuration
calibration_config = {
    "pattern_type": "checkerboard",    # Calibration pattern
    "pattern_size": (9, 6),           # Pattern dimensions
    "square_size": 25.0,              # Square size in mm
    "num_images": 20,                 # Number of calibration images
    "image_size": (1920, 1080),       # Image resolution
    "flags": [
        "CALIB_RATIONAL_MODEL",        # Use rational distortion model
        "CALIB_THIN_PRISM_MODEL",     # Include thin prism distortion
        "CALIB_TILTED_MODEL"          # Include sensor tilt
    ]
}

# Capture calibration images
calibration_images = []
for i in range(calibration_config["num_images"]):
    print(f"Capture calibration image {i+1}/{calibration_config['num_images']}")
    input("Position calibration pattern and press Enter...")
    
    image = sensor.capture_frame()
    if image is not None:
        calibration_images.append(image)
    else:
        print("Failed to capture image, retrying...")
        i -= 1

# Perform calibration
calibration_result = calibrator.calibrate_camera(
    images=calibration_images,
    config=calibration_config
)

# Extract calibration parameters
camera_matrix = calibration_result.camera_matrix
distortion_coeffs = calibration_result.distortion_coefficients
rvecs = calibration_result.rotation_vectors
tvecs = calibration_result.translation_vectors
rms_error = calibration_result.rms_reprojection_error

print(f"Calibration completed with RMS error: {rms_error:.3f} pixels")
print(f"Camera matrix:\n{camera_matrix}")
print(f"Distortion coefficients: {distortion_coeffs}")
```

### Calibration Quality Assessment

```python
# Assess calibration quality
quality_metrics = calibrator.assess_calibration_quality(calibration_result)

print("Calibration Quality Assessment:")
print(f"RMS Reprojection Error: {quality_metrics.rms_error:.3f} pixels")
print(f"Mean Error: {quality_metrics.mean_error:.3f} pixels")
print(f"Max Error: {quality_metrics.max_error:.3f} pixels")
print(f"Standard Deviation: {quality_metrics.std_error:.3f} pixels")

# Quality thresholds
if quality_metrics.rms_error < 0.5:
    print("✅ Excellent calibration quality")
elif quality_metrics.rms_error < 1.0:
    print("✅ Good calibration quality")
elif quality_metrics.rms_error < 2.0:
    print("⚠️ Acceptable calibration quality")
else:
    print("❌ Poor calibration quality - recalibration recommended")
```

### Distortion Correction

```python
# Apply distortion correction to images
def undistort_image(image, camera_matrix, distortion_coeffs):
    """Remove lens distortion from image."""
    h, w = image.shape[:2]
    
    # Generate optimal camera matrix
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix, distortion_coeffs, (w, h), 1, (w, h)
    )
    
    # Undistort image
    undistorted = cv2.undistort(
        image, camera_matrix, distortion_coeffs, None, new_camera_matrix
    )
    
    # Crop to region of interest
    x, y, w, h = roi
    undistorted = undistorted[y:y+h, x:x+w]
    
    return undistorted, new_camera_matrix

# Example usage
raw_image = sensor.capture_frame()
corrected_image, new_matrix = undistort_image(
    raw_image, camera_matrix, distortion_coeffs
)
```

## Multi-Camera Calibration

### Stereo Camera Calibration

```python
from advanced_image_sensor_interface.sensor_interface.multi_sensor_sync import MultiSensorSync

# Initialize stereo system
stereo_sync = MultiSensorSync()
stereo_calibrator = StereoCalibrator()

# Stereo calibration configuration
stereo_config = {
    "pattern_type": "checkerboard",
    "pattern_size": (9, 6),
    "square_size": 25.0,
    "num_image_pairs": 25,
    "flags": [
        "CALIB_RATIONAL_MODEL",
        "CALIB_SAME_FOCAL_LENGTH",    # Assume same focal length
        "CALIB_ZERO_TANGENT_DIST"     # Assume no tangential distortion
    ]
}

# Capture stereo image pairs
left_images = []
right_images = []

for i in range(stereo_config["num_image_pairs"]):
    print(f"Capture stereo pair {i+1}/{stereo_config['num_image_pairs']}")
    input("Position calibration pattern and press Enter...")
    
    # Capture synchronized frames
    frames = stereo_sync.capture_synchronized_frames()
    if len(frames) >= 2:
        left_images.append(frames[0][0])   # First camera frame
        right_images.append(frames[1][0])  # Second camera frame

# Perform stereo calibration
stereo_result = stereo_calibrator.calibrate_stereo(
    left_images=left_images,
    right_images=right_images,
    config=stereo_config
)

# Extract stereo parameters
R = stereo_result.rotation_matrix        # Rotation between cameras
T = stereo_result.translation_vector     # Translation between cameras
E = stereo_result.essential_matrix       # Essential matrix
F = stereo_result.fundamental_matrix     # Fundamental matrix

print(f"Stereo calibration RMS error: {stereo_result.rms_error:.3f} pixels")
print(f"Baseline distance: {np.linalg.norm(T):.2f} mm")
```

### Stereo Rectification

```python
# Compute rectification transforms
rectify_result = stereo_calibrator.compute_rectification(
    stereo_result, image_size=(1920, 1080)
)

R1 = rectify_result.rectification_transform_left
R2 = rectify_result.rectification_transform_right
P1 = rectify_result.projection_matrix_left
P2 = rectify_result.projection_matrix_right
Q = rectify_result.disparity_to_depth_matrix

# Create rectification maps
map1_left, map2_left = cv2.initUndistortRectifyMap(
    camera_matrix_left, distortion_left, R1, P1, image_size, cv2.CV_16SC2
)
map1_right, map2_right = cv2.initUndistortRectifyMap(
    camera_matrix_right, distortion_right, R2, P2, image_size, cv2.CV_16SC2
)

# Rectify stereo images
def rectify_stereo_pair(left_image, right_image):
    """Rectify stereo image pair."""
    left_rectified = cv2.remap(left_image, map1_left, map2_left, cv2.INTER_LINEAR)
    right_rectified = cv2.remap(right_image, map1_right, map2_right, cv2.INTER_LINEAR)
    return left_rectified, right_rectified
```

### Multi-Camera Array Calibration

```python
# Calibrate camera array (3+ cameras)
array_calibrator = MultiCameraCalibrator()

# Array configuration
array_config = {
    "num_cameras": 4,
    "reference_camera": 0,        # Reference camera index
    "pattern_type": "checkerboard",
    "pattern_size": (9, 6),
    "square_size": 25.0,
    "num_image_sets": 30
}

# Capture synchronized image sets
image_sets = []
for i in range(array_config["num_image_sets"]):
    print(f"Capture image set {i+1}/{array_config['num_image_sets']}")
    input("Position calibration pattern and press Enter...")
    
    frames = stereo_sync.capture_synchronized_frames()
    if len(frames) == array_config["num_cameras"]:
        image_set = [frame[0] for frame in frames.values()]
        image_sets.append(image_set)

# Perform multi-camera calibration
array_result = array_calibrator.calibrate_camera_array(
    image_sets=image_sets,
    config=array_config
)

# Extract relative poses
relative_poses = array_result.relative_poses
for i, pose in enumerate(relative_poses):
    if i == array_config["reference_camera"]:
        continue
    R, T = pose.rotation_matrix, pose.translation_vector
    print(f"Camera {i} relative to reference:")
    print(f"  Rotation: {R}")
    print(f"  Translation: {T} mm")
```

## Color Calibration

### Color Checker Calibration

```python
from advanced_image_sensor_interface.utils.color_calibration import ColorCalibrator

# Initialize color calibrator
color_calibrator = ColorCalibrator()

# Color calibration configuration
color_config = {
    "color_checker_type": "X-Rite ColorChecker Classic",
    "reference_illuminant": "D65",    # Standard daylight
    "white_balance_method": "gray_world",
    "color_space": "sRGB"
}

# Capture color checker image
print("Position X-Rite ColorChecker under standard illumination")
input("Press Enter to capture...")

color_checker_image = sensor.capture_frame()

# Detect color checker patches
patches = color_calibrator.detect_color_checker(
    color_checker_image, color_config["color_checker_type"]
)

if len(patches) == 24:  # Standard ColorChecker has 24 patches
    print("✅ Color checker detected successfully")
    
    # Compute color correction matrix
    color_correction_matrix = color_calibrator.compute_color_correction(
        patches, color_config
    )
    
    print(f"Color correction matrix:\n{color_correction_matrix}")
    
    # Apply color correction
    def apply_color_correction(image, ccm):
        """Apply color correction matrix to image."""
        # Reshape image for matrix multiplication
        h, w, c = image.shape
        image_flat = image.reshape(-1, c).astype(np.float32)
        
        # Apply color correction
        corrected_flat = np.dot(image_flat, ccm.T)
        
        # Clip values and reshape
        corrected_flat = np.clip(corrected_flat, 0, 255)
        corrected_image = corrected_flat.reshape(h, w, c).astype(np.uint8)
        
        return corrected_image
    
else:
    print("❌ Color checker detection failed")
```

### White Balance Calibration

```python
# White balance calibration
wb_calibrator = WhiteBalanceCalibrator()

# Capture white reference
print("Position white reference target")
input("Press Enter to capture white reference...")

white_reference = sensor.capture_frame()

# Compute white balance gains
wb_gains = wb_calibrator.compute_white_balance_gains(
    white_reference, method="gray_world"
)

print(f"White balance gains - R: {wb_gains[0]:.3f}, G: {wb_gains[1]:.3f}, B: {wb_gains[2]:.3f}")

# Apply white balance
def apply_white_balance(image, gains):
    """Apply white balance gains to image."""
    balanced = image.astype(np.float32)
    balanced[:, :, 0] *= gains[0]  # Red channel
    balanced[:, :, 1] *= gains[1]  # Green channel
    balanced[:, :, 2] *= gains[2]  # Blue channel
    
    return np.clip(balanced, 0, 255).astype(np.uint8)
```

## Temporal Calibration

### Frame Timing Calibration

```python
from advanced_image_sensor_interface.utils.timing_calibration import TimingCalibrator

# Initialize timing calibrator
timing_calibrator = TimingCalibrator()

# Timing calibration using LED flash
timing_config = {
    "flash_duration_ms": 1.0,     # LED flash duration
    "flash_frequency_hz": 10.0,   # Flash frequency
    "measurement_duration_s": 30.0, # Measurement duration
    "expected_frame_rate": 30.0   # Expected camera frame rate
}

print("Setup LED flash synchronized with external trigger")
input("Press Enter to start timing calibration...")

# Measure frame timing
timing_result = timing_calibrator.measure_frame_timing(
    sensor, timing_config
)

print("Frame Timing Results:")
print(f"Measured frame rate: {timing_result.actual_frame_rate:.3f} fps")
print(f"Frame rate error: {timing_result.frame_rate_error:.3f} fps")
print(f"Jitter (std dev): {timing_result.frame_jitter:.3f} ms")
print(f"Maximum latency: {timing_result.max_latency:.3f} ms")

# Timing quality assessment
if timing_result.frame_jitter < 1.0:
    print("✅ Excellent timing stability")
elif timing_result.frame_jitter < 5.0:
    print("✅ Good timing stability")
else:
    print("⚠️ Poor timing stability - check synchronization")
```

### Multi-Camera Synchronization Calibration

```python
# Synchronization calibration for multiple cameras
sync_calibrator = SynchronizationCalibrator()

# Synchronization test configuration
sync_config = {
    "num_cameras": 4,
    "test_duration_s": 60.0,
    "flash_frequency_hz": 5.0,
    "sync_tolerance_ms": 1.0
}

print("Setup synchronized LED flash visible to all cameras")
input("Press Enter to start synchronization test...")

# Measure synchronization accuracy
sync_result = sync_calibrator.measure_synchronization(
    stereo_sync, sync_config
)

print("Synchronization Results:")
for i, camera_sync in enumerate(sync_result.camera_synchronization):
    print(f"Camera {i}:")
    print(f"  Mean offset: {camera_sync.mean_offset:.3f} ms")
    print(f"  Std deviation: {camera_sync.std_offset:.3f} ms")
    print(f"  Max offset: {camera_sync.max_offset:.3f} ms")

# Overall synchronization quality
overall_sync = sync_result.overall_synchronization
print(f"\nOverall synchronization accuracy: {overall_sync:.3f} ms")

if overall_sync < 1.0:
    print("✅ Excellent synchronization")
elif overall_sync < 5.0:
    print("✅ Good synchronization")
else:
    print("⚠️ Poor synchronization - check hardware setup")
```

## Advanced Calibration Techniques

### Rolling Shutter Calibration

```python
# Rolling shutter distortion calibration
rs_calibrator = RollingShutterCalibrator()

# Rolling shutter test setup
rs_config = {
    "motion_type": "linear",      # Linear motion pattern
    "motion_speed_mps": 1.0,      # Motion speed in m/s
    "pattern_type": "vertical_lines", # Vertical line pattern
    "line_spacing": 10,           # Line spacing in pixels
    "exposure_time_us": 10000     # Exposure time in microseconds
}

print("Setup moving vertical line pattern")
input("Press Enter to capture rolling shutter test...")

# Capture test image with motion
rs_test_image = sensor.capture_frame()

# Analyze rolling shutter distortion
rs_result = rs_calibrator.analyze_rolling_shutter(
    rs_test_image, rs_config
)

print("Rolling Shutter Analysis:")
print(f"Readout time: {rs_result.readout_time_us:.1f} μs")
print(f"Line readout time: {rs_result.line_readout_time_us:.3f} μs")
print(f"Skew angle: {rs_result.skew_angle_deg:.3f}°")

# Rolling shutter correction
def correct_rolling_shutter(image, readout_time_us, motion_vector):
    """Correct rolling shutter distortion."""
    h, w = image.shape[:2]
    corrected = np.zeros_like(image)
    
    for row in range(h):
        # Calculate time offset for this row
        time_offset = (row / h) * readout_time_us * 1e-6
        
        # Calculate motion compensation
        motion_offset = motion_vector * time_offset
        
        # Apply motion compensation (simplified)
        offset_x = int(motion_offset[0])
        if 0 <= offset_x < w:
            corrected[row, :w-offset_x] = image[row, offset_x:]
    
    return corrected
```

### Geometric Distortion Calibration

```python
# Advanced geometric distortion calibration
geo_calibrator = GeometricCalibrator()

# Geometric calibration using grid pattern
geo_config = {
    "grid_type": "dot_grid",      # Dot grid pattern
    "grid_size": (15, 11),        # Grid dimensions
    "dot_spacing_mm": 10.0,       # Dot spacing in mm
    "detection_method": "blob",    # Blob detection
    "subpixel_accuracy": True     # Subpixel accuracy
}

print("Position dot grid calibration target")
input("Press Enter to capture geometric calibration image...")

geo_image = sensor.capture_frame()

# Detect grid points
grid_points = geo_calibrator.detect_grid_points(geo_image, geo_config)

if len(grid_points) >= geo_config["grid_size"][0] * geo_config["grid_size"][1] * 0.8:
    print("✅ Grid detection successful")
    
    # Compute geometric distortion model
    distortion_model = geo_calibrator.compute_geometric_distortion(
        grid_points, geo_config
    )
    
    print("Geometric Distortion Model:")
    print(f"Barrel distortion: {distortion_model.barrel_distortion:.6f}")
    print(f"Pincushion distortion: {distortion_model.pincushion_distortion:.6f}")
    print(f"Asymmetric distortion: {distortion_model.asymmetric_distortion}")
    
else:
    print("❌ Grid detection failed - check lighting and focus")
```

## Calibration Validation

### Cross-Validation

```python
# Cross-validation of calibration results
validator = CalibrationValidator()

# Validation configuration
validation_config = {
    "validation_method": "k_fold",
    "k_folds": 5,
    "metrics": ["reprojection_error", "3d_accuracy", "stereo_accuracy"],
    "test_patterns": ["checkerboard", "circles", "asymmetric_circles"]
}

# Perform cross-validation
validation_result = validator.cross_validate_calibration(
    calibration_images, calibration_config, validation_config
)

print("Cross-Validation Results:")
for metric, result in validation_result.metrics.items():
    print(f"{metric}:")
    print(f"  Mean: {result.mean:.3f}")
    print(f"  Std: {result.std:.3f}")
    print(f"  Min: {result.min:.3f}")
    print(f"  Max: {result.max:.3f}")
```

### Real-World Accuracy Test

```python
# Real-world accuracy validation
accuracy_tester = AccuracyTester()

# Setup known 3D reference points
reference_points_3d = np.array([
    [0, 0, 0],      # Origin
    [100, 0, 0],    # 100mm along X
    [0, 100, 0],    # 100mm along Y
    [0, 0, 100],    # 100mm along Z
    [100, 100, 0],  # Corner point
])

print("Position reference objects at known 3D coordinates")
input("Press Enter to capture validation image...")

validation_image = sensor.capture_frame()

# Detect reference points in image
detected_points_2d = accuracy_tester.detect_reference_points(
    validation_image, reference_points_3d
)

# Compute 3D accuracy
accuracy_result = accuracy_tester.compute_3d_accuracy(
    detected_points_2d, reference_points_3d, camera_matrix, distortion_coeffs
)

print("3D Accuracy Results:")
print(f"Mean 3D error: {accuracy_result.mean_error_mm:.2f} mm")
print(f"Max 3D error: {accuracy_result.max_error_mm:.2f} mm")
print(f"RMS 3D error: {accuracy_result.rms_error_mm:.2f} mm")

# Accuracy assessment
if accuracy_result.rms_error_mm < 1.0:
    print("✅ Excellent 3D accuracy")
elif accuracy_result.rms_error_mm < 5.0:
    print("✅ Good 3D accuracy")
else:
    print("⚠️ Poor 3D accuracy - recalibration recommended")
```

## Calibration Storage and Management

### Calibration Data Storage

```python
import json
import numpy as np
from datetime import datetime

class CalibrationManager:
    """Manage calibration data storage and retrieval."""
    
    def __init__(self, storage_path="calibration_data"):
        self.storage_path = storage_path
        os.makedirs(storage_path, exist_ok=True)
    
    def save_calibration(self, calibration_result, camera_id, metadata=None):
        """Save calibration data to file."""
        timestamp = datetime.now().isoformat()
        
        calibration_data = {
            "timestamp": timestamp,
            "camera_id": camera_id,
            "camera_matrix": calibration_result.camera_matrix.tolist(),
            "distortion_coefficients": calibration_result.distortion_coefficients.tolist(),
            "rms_error": float(calibration_result.rms_reprojection_error),
            "image_size": calibration_result.image_size,
            "metadata": metadata or {}
        }
        
        filename = f"{self.storage_path}/calibration_{camera_id}_{timestamp}.json"
        with open(filename, 'w') as f:
            json.dump(calibration_data, f, indent=2)
        
        print(f"Calibration saved to {filename}")
        return filename
    
    def load_calibration(self, camera_id, timestamp=None):
        """Load calibration data from file."""
        if timestamp is None:
            # Load most recent calibration
            pattern = f"{self.storage_path}/calibration_{camera_id}_*.json"
            files = glob.glob(pattern)
            if not files:
                raise FileNotFoundError(f"No calibration found for camera {camera_id}")
            filename = max(files)  # Most recent file
        else:
            filename = f"{self.storage_path}/calibration_{camera_id}_{timestamp}.json"
        
        with open(filename, 'r') as f:
            calibration_data = json.load(f)
        
        # Convert back to numpy arrays
        camera_matrix = np.array(calibration_data["camera_matrix"])
        distortion_coeffs = np.array(calibration_data["distortion_coefficients"])
        
        return {
            "camera_matrix": camera_matrix,
            "distortion_coefficients": distortion_coeffs,
            "rms_error": calibration_data["rms_error"],
            "image_size": calibration_data["image_size"],
            "timestamp": calibration_data["timestamp"],
            "metadata": calibration_data["metadata"]
        }

# Usage example
calibration_manager = CalibrationManager()

# Save calibration
calibration_manager.save_calibration(
    calibration_result, 
    camera_id="main_camera",
    metadata={
        "lens_model": "50mm f/1.8",
        "sensor_size": "APS-C",
        "calibration_environment": "laboratory",
        "operator": "calibration_technician"
    }
)

# Load calibration
loaded_calibration = calibration_manager.load_calibration("main_camera")
camera_matrix = loaded_calibration["camera_matrix"]
distortion_coeffs = loaded_calibration["distortion_coefficients"]
```

## Troubleshooting Common Issues

### Poor Calibration Quality

**Symptoms:**
- High RMS reprojection error (>2.0 pixels)
- Inconsistent results between calibration runs
- Poor undistortion results

**Solutions:**
1. **Improve calibration images:**
   - Use more images (30+ recommended)
   - Ensure good pattern coverage across image
   - Vary pattern distance and orientation
   - Check image sharpness and focus

2. **Check calibration pattern:**
   - Verify pattern dimensions
   - Ensure pattern is flat and rigid
   - Use high-contrast pattern
   - Check for pattern detection errors

3. **Optimize camera settings:**
   - Use manual focus
   - Disable auto-exposure during calibration
   - Use adequate lighting
   - Minimize motion blur

### Synchronization Issues

**Symptoms:**
- Large timing offsets between cameras
- Inconsistent frame timing
- Dropped frames during synchronized capture

**Solutions:**
1. **Hardware synchronization:**
   - Use external trigger signal
   - Check trigger signal quality
   - Verify cable connections
   - Use proper termination

2. **Software optimization:**
   - Increase buffer sizes
   - Use real-time scheduling
   - Minimize system load
   - Check USB/network bandwidth

### Distortion Correction Artifacts

**Symptoms:**
- Visible distortion after correction
- Image quality degradation
- Cropping issues

**Solutions:**
1. **Calibration improvement:**
   - Recalibrate with more images
   - Use appropriate distortion model
   - Check for rolling shutter effects
   - Validate calibration accuracy

2. **Correction optimization:**
   - Use optimal new camera matrix
   - Adjust alpha parameter
   - Use high-quality interpolation
   - Consider region of interest cropping

This comprehensive calibration guide ensures accurate and reliable camera calibration for all supported protocols and configurations in the Advanced Image Sensor Interface.