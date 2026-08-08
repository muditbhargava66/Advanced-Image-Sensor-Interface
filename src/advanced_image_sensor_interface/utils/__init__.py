"""
Utils Module

This module provides utility functions for image processing, performance evaluation,
data integrity, and lens correction for image sensor applications.

Modules:
    noise_reduction: Functions for reducing noise in image data.
    performance_metrics: Functions for calculating various performance metrics.
    buffer_manager: Advanced buffer management with memory pooling (v2.0.0).
    data_integrity: CRC-32 and Forward Error Correction (v3.0.0).
    lens_correction: Radial and tangential distortion correction (v3.0.0).

Usage:
    from advanced_image_sensor_interface.utils import reduce_noise, calculate_snr
"""

from .noise_reduction import (
    BilateralNoiseReducer,
    GaussianNoiseReducer,
    MedianNoiseReducer,
    NoiseReducer,
    NoiseReducerFactory,
    NoiseReductionConfig,
    NoiseType,
    adaptive_noise_reduction,
    reduce_noise,
)
from .performance_metrics import calculate_color_accuracy, calculate_dynamic_range, calculate_snr

__all__ = [
    # Noise reduction
    "reduce_noise",
    "adaptive_noise_reduction",
    "NoiseReducer",
    "NoiseReductionConfig",
    "NoiseType",
    "NoiseReducerFactory",
    "GaussianNoiseReducer",
    "BilateralNoiseReducer",
    "MedianNoiseReducer",
    # Performance metrics
    "calculate_snr",
    "calculate_dynamic_range",
    "calculate_color_accuracy",
]

__version__ = "3.0.0"
__author__ = "Mudit Bhargava"
__license__ = "MIT"
