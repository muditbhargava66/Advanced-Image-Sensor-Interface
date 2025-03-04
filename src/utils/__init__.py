"""
Utils Module

This module provides utility functions for image processing and performance evaluation,
specifically focused on noise reduction and metrics calculation for image sensor applications.

Modules:
    noise_reduction: Functions for reducing noise in image data.
    performance_metrics: Functions for calculating various performance metrics.

Usage:
    from utils import reduce_noise, calculate_snr
"""

from .noise_reduction import adaptive_noise_reduction, reduce_noise
from .performance_metrics import calculate_color_accuracy, calculate_dynamic_range, calculate_snr

__all__ = [
    'reduce_noise',
    'adaptive_noise_reduction',
    'calculate_snr',
    'calculate_dynamic_range',
    'calculate_color_accuracy'
]

__version__ = '1.0.1'
__author__ = 'Mudit Bhargava'
__license__ = 'MIT'
