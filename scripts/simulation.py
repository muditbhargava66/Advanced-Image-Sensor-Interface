"""
Image Sensor Pipeline Simulation

This script provides a comprehensive simulation of the Advanced Image Sensor Interface,
including MIPI data transfer, signal processing, and power management.

Usage:
    python simulation.py [options]

Options:
    --resolution RESOLUTION   Set the simulation resolution (default: 1920x1080)
    --frames FRAMES           Number of frames to simulate (default: 100)
    --noise NOISE             Noise level for simulation (default: 0.05)
    --output OUTPUT           Output file for simulation results (default: simulation_results.json)

Example:
-------
    python simulation.py --resolution 3840x2160 --frames 500 --noise 0.03 --output high_res_sim.json

"""

import argparse
import json
import os
import sys
import time
from typing import Any

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.sensor_interface.mipi_driver import MIPIConfig, MIPIDriver
from src.sensor_interface.power_management import PowerConfig, PowerManager
from src.sensor_interface.signal_processing import SignalConfig, SignalProcessor
from src.utils.performance_metrics import calculate_color_accuracy, calculate_dynamic_range, calculate_snr


def generate_synthetic_frame(width: int, height: int, noise_level: float) -> np.ndarray:
    """Generate a synthetic frame with realistic image characteristics and noise."""
    # Create a base image with some patterns
    x, y = np.meshgrid(np.linspace(0, 1, width), np.linspace(0, 1, height))
    base_image = np.sin(2 * np.pi * x) * np.sin(2 * np.pi * y) * 0.5 + 0.5

    # Add some circles
    for _ in range(5):
        cx, cy = np.random.rand(2)
        r = np.random.uniform(0.05, 0.2)
        mask = ((x - cx)**2 + (y - cy)**2) < r**2
        base_image[mask] = np.random.uniform(0.2, 0.8)

    # Convert to 12-bit range
    image = (base_image * 4095).astype(np.uint16)

    # Add noise
    noise = np.random.normal(0, noise_level * 4095, image.shape).astype(np.int16)
    noisy_image = np.clip(image + noise, 0, 4095).astype(np.uint16)

    return noisy_image

def simulate_pipeline(width: int, height: int, num_frames: int, noise_level: float) -> dict[str, Any]:
    """Simulate the entire image sensor pipeline and return performance metrics."""
    mipi_config = MIPIConfig(lanes=4, data_rate=2.5, channel=0)
    mipi_driver = MIPIDriver(mipi_config)

    signal_config = SignalConfig(bit_depth=12, noise_reduction_strength=0.1, color_correction_matrix=np.eye(3))
    signal_processor = SignalProcessor(signal_config)

    power_config = PowerConfig(voltage_main=1.8, voltage_io=3.3, current_limit=1.0)
    power_manager = PowerManager(power_config)

    metrics = {
        'snr': [],
        'dynamic_range': [],
        'color_accuracy': [],
        'power_consumption': [],
        'processing_time': [],
    }

    for _ in range(num_frames):
        # Generate synthetic frame
        raw_frame = generate_synthetic_frame(width, height, noise_level)

        # Simulate MIPI transfer
        start_time = time.time()
        mipi_driver.send_data(raw_frame.tobytes())
        mipi_transfer_time = time.time() - start_time

        # Process frame
        start_time = time.time()
        processed_frame = signal_processor.process_frame(raw_frame)
        processing_time = time.time() - start_time

        # Calculate metrics
        snr = calculate_snr(processed_frame, raw_frame - processed_frame)
        dr = calculate_dynamic_range(processed_frame)
        color_accuracy, _ = calculate_color_accuracy(raw_frame, processed_frame)
        power_status = power_manager.get_power_status()

        metrics['snr'].append(snr)
        metrics['dynamic_range'].append(dr)
        metrics['color_accuracy'].append(color_accuracy)
        metrics['power_consumption'].append(power_status['power_consumption'])
        metrics['processing_time'].append(processing_time + mipi_transfer_time)

    # Calculate average metrics
    for key, values in metrics.items():
        metrics[key] = np.mean(values)

    return metrics

def main():
    parser = argparse.ArgumentParser(description="Image Sensor Pipeline Simulation")
    parser.add_argument('--resolution', default='1920x1080', help='Simulation resolution (WxH)')
    parser.add_argument('--frames', type=int, default=100, help='Number of frames to simulate')
    parser.add_argument('--noise', type=float, default=0.05, help='Noise level for simulation')
    parser.add_argument('--output', default='simulation_results.json', help='Output file for results')

    args = parser.parse_args()

    width, height = map(int, args.resolution.split('x'))

    print(f"Starting simulation with {width}x{height} resolution, {args.frames} frames, and noise level {args.noise}")

    results = simulate_pipeline(width, height, args.frames, args.noise)

    print("Simulation complete. Results:")
    for key, value in results.items():
        print(f"{key}: {value}")

    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to {args.output}")

if __name__ == "__main__":
    main()
