"""
Calibration data models and structures.

This module defines the data structures used for camera calibration results,
quality metrics, and calibration parameters.
"""

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np


@dataclass
class CalibrationResult:
    """Results from camera calibration."""

    camera_matrix: np.ndarray
    distortion_coefficients: np.ndarray
    rotation_vectors: list[np.ndarray]
    translation_vectors: list[np.ndarray]
    rms_reprojection_error: float
    image_size: tuple[int, int]
    calibration_flags: int
    object_points: list[np.ndarray]
    image_points: list[np.ndarray]

    def __post_init__(self):
        """Validate calibration result data."""
        if self.camera_matrix.shape != (3, 3):
            raise ValueError("Camera matrix must be 3x3")
        if len(self.distortion_coefficients) < 4:
            raise ValueError("Must have at least 4 distortion coefficients")


@dataclass
class StereoCalibrationResult:
    """Results from stereo camera calibration."""

    camera_matrix_left: np.ndarray
    camera_matrix_right: np.ndarray
    distortion_left: np.ndarray
    distortion_right: np.ndarray
    rotation_matrix: np.ndarray
    translation_vector: np.ndarray
    essential_matrix: np.ndarray
    fundamental_matrix: np.ndarray
    rms_error: float

    # Rectification results
    rectification_transform_left: Optional[np.ndarray] = None
    rectification_transform_right: Optional[np.ndarray] = None
    projection_matrix_left: Optional[np.ndarray] = None
    projection_matrix_right: Optional[np.ndarray] = None
    disparity_to_depth_matrix: Optional[np.ndarray] = None


@dataclass
class CalibrationQualityMetrics:
    """Quality assessment metrics for calibration."""

    rms_error: float
    mean_error: float
    max_error: float
    std_error: float
    per_image_errors: list[float]
    coverage_score: float  # How well the calibration pattern covers the image
    symmetry_score: float  # How symmetric the error distribution is

    @property
    def quality_grade(self) -> str:
        """Get quality grade based on RMS error."""
        if self.rms_error < 0.5:
            return "Excellent"
        elif self.rms_error < 1.0:
            return "Good"
        elif self.rms_error < 2.0:
            return "Acceptable"
        else:
            return "Poor"


@dataclass
class CalibrationPattern:
    """Calibration pattern configuration."""

    pattern_type: str  # "checkerboard", "circles", "asymmetric_circles"
    pattern_size: tuple[int, int]  # (width, height) in pattern units
    square_size: float  # Size of each square/circle in mm

    def generate_object_points(self) -> np.ndarray:
        """Generate 3D object points for the calibration pattern."""
        if self.pattern_type == "checkerboard":
            # Create checkerboard pattern points
            objp = np.zeros((self.pattern_size[0] * self.pattern_size[1], 3), np.float32)
            objp[:, :2] = np.mgrid[0 : self.pattern_size[0], 0 : self.pattern_size[1]].T.reshape(-1, 2)
            objp *= self.square_size
            return objp
        elif self.pattern_type == "circles":
            # Create circular grid pattern points
            objp = np.zeros((self.pattern_size[0] * self.pattern_size[1], 3), np.float32)
            objp[:, :2] = np.mgrid[0 : self.pattern_size[0], 0 : self.pattern_size[1]].T.reshape(-1, 2)
            objp *= self.square_size
            return objp
        else:
            raise ValueError(f"Unsupported pattern type: {self.pattern_type}")


@dataclass
class MultiCameraCalibrationResult:
    """Results from multi-camera array calibration."""

    camera_matrices: list[np.ndarray]
    distortion_coefficients: list[np.ndarray]
    relative_poses: list[dict[str, np.ndarray]]  # Rotation and translation relative to reference
    reference_camera_id: int
    rms_errors: list[float]
    overall_rms_error: float

    def get_camera_pose(self, camera_id: int) -> tuple[np.ndarray, np.ndarray]:
        """Get rotation and translation for a specific camera."""
        if camera_id == self.reference_camera_id:
            return np.eye(3), np.zeros(3)

        pose = self.relative_poses[camera_id]
        return pose["rotation"], pose["translation"]


@dataclass
class ColorCalibrationResult:
    """Results from color calibration."""

    color_correction_matrix: np.ndarray
    white_balance_gains: np.ndarray
    gamma_correction: float
    color_temperature: float
    illuminant_type: str
    delta_e_mean: float  # Mean color error
    delta_e_max: float  # Maximum color error

    def apply_color_correction(self, image: np.ndarray) -> np.ndarray:
        """Apply color correction to an image."""
        # Apply white balance
        corrected = image.astype(np.float32)
        for i in range(3):
            corrected[:, :, i] *= self.white_balance_gains[i]

        # Apply color correction matrix
        corrected = corrected.reshape(-1, 3)
        corrected = np.dot(corrected, self.color_correction_matrix.T)
        corrected = corrected.reshape(image.shape)

        # Apply gamma correction
        corrected = np.power(corrected / 255.0, 1.0 / self.gamma_correction) * 255.0

        return np.clip(corrected, 0, 255).astype(np.uint8)


@dataclass
class TemporalCalibrationResult:
    """Results from temporal/timing calibration."""

    frame_timing_offsets: list[float]  # Timing offset for each camera in milliseconds
    synchronization_accuracy: float  # RMS synchronization error in milliseconds
    frame_rate_accuracy: float  # Actual vs expected frame rate accuracy
    jitter_statistics: dict[str, float]  # Min, max, mean, std of timing jitter

    @property
    def is_synchronized(self) -> bool:
        """Check if cameras are well synchronized."""
        return self.synchronization_accuracy < 1.0  # Less than 1ms error


@dataclass
class CalibrationConfiguration:
    """Configuration for calibration process."""

    pattern: CalibrationPattern
    num_images: int = 20
    image_size: tuple[int, int] = (1920, 1080)
    calibration_flags: int = 0
    termination_criteria: tuple[int, int, float] = (3, 30, 0.001)  # cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER

    # Quality thresholds
    max_rms_error: float = 1.0
    min_coverage_score: float = 0.7

    def validate(self) -> bool:
        """Validate calibration configuration."""
        if self.num_images < 10:
            raise ValueError("Need at least 10 calibration images")
        if self.image_size[0] < 640 or self.image_size[1] < 480:
            raise ValueError("Image size too small for reliable calibration")
        return True


class CalibrationDatabase:
    """Database for storing and retrieving calibration results."""

    def __init__(self, storage_path: str = "calibration_data"):
        self.storage_path = storage_path
        self._calibrations: dict[str, CalibrationResult] = {}

    def store_calibration(self, camera_id: str, result: CalibrationResult) -> None:
        """Store calibration result for a camera."""
        self._calibrations[camera_id] = result

    def get_calibration(self, camera_id: str) -> Optional[CalibrationResult]:
        """Retrieve calibration result for a camera."""
        return self._calibrations.get(camera_id)

    def list_calibrations(self) -> list[str]:
        """List all stored calibration camera IDs."""
        return list(self._calibrations.keys())

    def export_calibration(self, camera_id: str) -> dict[str, Any]:
        """Export calibration data as dictionary."""
        result = self.get_calibration(camera_id)
        if result is None:
            raise ValueError(f"No calibration found for camera {camera_id}")

        return {
            "camera_matrix": result.camera_matrix.tolist(),
            "distortion_coefficients": result.distortion_coefficients.tolist(),
            "rms_error": result.rms_reprojection_error,
            "image_size": result.image_size,
            "calibration_flags": result.calibration_flags,
        }
