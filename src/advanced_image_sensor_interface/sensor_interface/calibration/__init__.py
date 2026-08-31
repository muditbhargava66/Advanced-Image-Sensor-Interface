"""
Camera calibration module for the Advanced Image Sensor Interface.

This module provides comprehensive camera calibration capabilities including:
- Intrinsic camera calibration
- Extrinsic camera calibration
- Multi-camera calibration
- Stereo calibration
- Color calibration
- Temporal calibration
- Native numpy/scipy photogrammetry solver (no OpenCV required)
"""

from .models import CalibrationQualityMetrics, CalibrationResult, StereoCalibrationResult
from .neural_tuner import NeuralCalibrationTuner
from .photogrammetry import calibrate_camera, solve_projection_matrix

__all__ = [
    "CalibrationResult",
    "StereoCalibrationResult",
    "CalibrationQualityMetrics",
    "NeuralCalibrationTuner",
    "calibrate_camera",
    "solve_projection_matrix",
]
