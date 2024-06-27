"""
Performance Metrics Utilities

This module provides functions for calculating various performance metrics
for image sensor evaluation, including SNR, dynamic range, and color accuracy.

Functions:
    calculate_snr: Calculate Signal-to-Noise Ratio.
    calculate_dynamic_range: Calculate Dynamic Range.
    calculate_color_accuracy: Calculate Color Accuracy using Delta E.

Usage:
    from utils.performance_metrics import calculate_snr, calculate_dynamic_range, calculate_color_accuracy
    snr = calculate_snr(signal, noise)
"""

import numpy as np
from typing import Tuple, Union
import logging
from colormath.color_objects import sRGBColor, LabColor
from colormath.color_conversions import convert_color
from colormath.color_diff import delta_e_cie2000

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def calculate_snr(signal: np.ndarray, noise: np.ndarray) -> float:
    """
    Calculate the Signal-to-Noise Ratio (SNR) in decibels.

    Args:
        signal (np.ndarray): The clean signal or reference image.
        noise (np.ndarray): The noise component or the difference between the noisy and clean signal.

    Returns:
        float: The calculated SNR in decibels.

    Raises:
        ValueError: If the shapes of signal and noise do not match.
    """
    if signal.shape != noise.shape:
        raise ValueError("Signal and noise must have the same shape")

    signal_power = np.mean(signal**2)
    noise_power = np.mean(noise**2)
    
    if noise_power == 0:
        return float('inf')
    
    snr = 10 * np.log10(signal_power / noise_power)
    logger.info(f"Calculated SNR: {snr:.2f} dB")
    return snr

def calculate_dynamic_range(image: np.ndarray) -> float:
    """
    Calculate the dynamic range of an image in decibels.

    Args:
        image (np.ndarray): The input image.

    Returns:
        float: The calculated dynamic range in decibels.
    """
    min_val = np.min(image[image > 0])  # Minimum non-zero value
    max_val = np.max(image)
    
    if min_val == 0 or max_val == 0:
        return 0
    
    dynamic_range = 20 * np.log10(max_val / min_val)
    logger.info(f"Calculated Dynamic Range: {dynamic_range:.2f} dB")
    return dynamic_range

def calculate_color_accuracy(reference_colors: np.ndarray, measured_colors: np.ndarray) -> Tuple[float, np.ndarray]:
    """
    Calculate color accuracy using Delta E (CIEDE2000) color difference formula.

    Args:
        reference_colors (np.ndarray): Array of reference sRGB colors, shape (N, 3).
        measured_colors (np.ndarray): Array of measured sRGB colors, shape (N, 3).

    Returns:
        Tuple[float, np.ndarray]: Mean Delta E value and array of Delta E values for each color.

    Raises:
        ValueError: If the shapes of reference_colors and measured_colors do not match.
    """
    if reference_colors.shape != measured_colors.shape:
        raise ValueError("Reference and measured color arrays must have the same shape")

    delta_e_values = []

    # for ref, meas in zip(reference_colors, measured_colors):
    for ref, meas in zip(reference_colors.reshape(-1, 3), measured_colors.reshape(-1, 3)):
        ref_rgb = sRGBColor(ref[0], ref[1], ref[2], is_upscaled=True)
        meas_rgb = sRGBColor(meas[0], meas[1], meas[2], is_upscaled=True)
        
        ref_lab = convert_color(ref_rgb, LabColor)
        meas_lab = convert_color(meas_rgb, LabColor)
        
        delta_e = delta_e_cie2000(ref_lab, meas_lab)
        delta_e_values.append(delta_e)

    delta_e_array = np.array(delta_e_values)
    mean_delta_e = np.mean(delta_e_array)
    
    logger.info(f"Mean Color Accuracy (Delta E): {mean_delta_e:.2f}")
    return mean_delta_e, delta_e_array

# Example usage and testing
if __name__ == "__main__":
    # Test SNR calculation
    clean_signal = np.random.rand(100, 100)
    noise = 0.1 * np.random.randn(100, 100)
    noisy_signal = clean_signal + noise
    snr = calculate_snr(clean_signal, noisy_signal - clean_signal)
    logger.info(f"Test SNR: {snr:.2f} dB")

    # Test dynamic range calculation
    test_image = np.random.randint(1, 256, size=(100, 100)).astype(np.uint8)
    dynamic_range = calculate_dynamic_range(test_image)
    logger.info(f"Test Dynamic Range: {dynamic_range:.2f} dB")

    # Test color accuracy calculation
    reference_colors = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]], dtype=np.uint8)
    measured_colors = np.array([[250, 10, 10], [10, 250, 10], [10, 10, 250]], dtype=np.uint8)
    mean_delta_e, delta_e_values = calculate_color_accuracy(reference_colors, measured_colors)
    logger.info(f"Test Mean Delta E: {mean_delta_e:.2f}")
    logger.info(f"Test Delta E values: {delta_e_values}")

    logger.info("All tests completed successfully")