"""
Noise Analysis for Advanced Image Sensor Interface

This module contains benchmark tests for analyzing noise characteristics
and reduction efficacy in the Advanced Image Sensor Interface project.

Functions:
    generate_noisy_image: Generate a synthetic noisy image for testing.
    measure_noise_level: Measure the noise level in an image.
    benchmark_noise_reduction: Benchmark the noise reduction performance.
    analyze_snr_improvement: Analyze the improvement in Signal-to-Noise Ratio.
    run_noise_analysis: Run all noise analysis benchmarks and report results.

Usage:
    from benchmarks.noise_analysis import run_noise_analysis
    run_noise_analysis()
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import json
from typing import Any

import numpy as np

from src.sensor_interface.signal_processing import SignalConfig, SignalProcessor
from src.utils.performance_metrics import calculate_snr


def generate_noisy_image(size: tuple[int, int] = (1920, 1080), signal_level: int = 1000, noise_std: int = 50) -> np.ndarray:
    """
    Generate a synthetic noisy image for testing.

    Args:
    ----
        size (Tuple[int, int]): Size of the image to generate.
        signal_level (int): Mean signal level (0-4095 for 12-bit image).
        noise_std (int): Standard deviation of the noise.

    Returns:
    -------
        np.ndarray: Synthetic noisy image.

    """
    clean_signal = np.full(size, signal_level, dtype=np.uint16)
    noise = np.random.normal(0, noise_std, size).astype(np.int16)
    noisy_signal = np.clip(clean_signal + noise, 0, 4095).astype(np.uint16)
    return noisy_signal

def measure_noise_level(image: np.ndarray) -> float:
    """
    Measure the noise level in an image.

    Args:
    ----
        image (np.ndarray): Input image.

    Returns:
    -------
        float: Measured noise level (standard deviation).

    """
    return np.std(image)

def benchmark_noise_reduction(noise_levels: list[int] = [10, 50, 100]) -> dict[str, float]:
    """
    Benchmark the noise reduction performance.

    Args:
    ----
        noise_levels (list[int]): List of noise levels to test.

    Returns:
    -------
        Dict[str, float]: Dictionary of noise levels and their corresponding reduction percentages.

    """
    config = SignalConfig(bit_depth=12, noise_reduction_strength=0.1, color_correction_matrix=np.eye(3))
    processor = SignalProcessor(config)
    results = {}

    for noise_std in noise_levels:
        noisy_image = generate_noisy_image(noise_std=noise_std)
        initial_noise = measure_noise_level(noisy_image)

        processed_image = processor.process_frame(noisy_image)
        final_noise = measure_noise_level(processed_image)

        reduction_percentage = (initial_noise - final_noise) / initial_noise * 100
        results[f"noise_level_{noise_std}"] = reduction_percentage

    return results

def analyze_snr_improvement(signal_levels: list[int] = [500, 1000, 2000], noise_std: int = 50) -> dict[str, float]:
    """
    Analyze the improvement in Signal-to-Noise Ratio.

    Args:
    ----
        signal_levels (list[int]): List of signal levels to test.
        noise_std (int): Standard deviation of the noise.

    Returns:
    -------
        Dict[str, float]: Dictionary of signal levels and their corresponding SNR improvements in dB.

    """
    config = SignalConfig(bit_depth=12, noise_reduction_strength=0.1, color_correction_matrix=np.eye(3))
    processor = SignalProcessor(config)
    results = {}

    for signal_level in signal_levels:
        noisy_image = generate_noisy_image(signal_level=signal_level, noise_std=noise_std)
        initial_snr = calculate_snr(noisy_image, noisy_image - signal_level)

        processed_image = processor.process_frame(noisy_image)
        final_snr = calculate_snr(processed_image, processed_image - signal_level)

        snr_improvement = final_snr - initial_snr
        results[f"signal_level_{signal_level}"] = snr_improvement

    return results

def run_noise_analysis() -> dict[str, Any]:
    """
    Run all noise analysis benchmarks and compile results.

    Returns
    -------
        Dict[str, Any]: Dictionary containing all noise analysis results.

    """
    results = {}
    results['noise_reduction'] = benchmark_noise_reduction()
    results['snr_improvement'] = analyze_snr_improvement()
    return results

if __name__ == "__main__":
    noise_analysis_results = run_noise_analysis()
    print("Noise Analysis Results:")
    print(json.dumps(noise_analysis_results, indent=2))

    # Calculate and print overall noise reduction and SNR improvement
    avg_noise_reduction = np.mean(list(noise_analysis_results['noise_reduction'].values()))
    avg_snr_improvement = np.mean(list(noise_analysis_results['snr_improvement'].values()))

    print(f"\nAverage Noise Reduction: {avg_noise_reduction:.2f}%")
    print(f"Average SNR Improvement: {avg_snr_improvement:.2f} dB")
