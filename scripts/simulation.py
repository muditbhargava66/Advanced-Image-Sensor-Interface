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
    --output OUTPUT           Output file for simulation results

Example:
    python simulation.py --resolution 3840x2160 --frames 500 --noise 0.03
"""

import argparse
import json
import os
import sys
import time
from typing import Any

import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from advanced_image_sensor_interface.sensor_interface.protocol.mipi.driver import MIPIConfig, MIPIProtocolDriver
from advanced_image_sensor_interface.sensor_interface.power_management import PowerConfig, PowerManager
from advanced_image_sensor_interface.sensor_interface.signal_processing import SignalConfig, SignalProcessor
from advanced_image_sensor_interface.utils.performance_metrics import (
    calculate_color_accuracy,
    calculate_dynamic_range,
    calculate_snr,
)


def generate_synthetic_frame(width: int, height: int, noise_level: float) -> np.ndarray:
    """Generate a synthetic frame with realistic image characteristics and noise."""
    # Create a base image with gradient and pattern
    x, y = np.meshgrid(np.linspace(0, 1, width), np.linspace(0, 1, height))
    base_image = np.sin(2 * np.pi * x) * np.sin(2 * np.pi * y) * 0.5 + 0.5

    # Add circular features
    for _ in range(5):
        cx, cy = np.random.rand(2)
        r = np.random.uniform(0.05, 0.2)
        mask = ((x - cx) ** 2 + (y - cy) ** 2) < r**2
        base_image[mask] = np.random.uniform(0.2, 0.8)

    # Convert to 12-bit range
    image = (base_image * 4095).astype(np.uint16)

    # Add noise
    noise = np.random.normal(0, noise_level * 4095, image.shape).astype(np.int16)
    noisy_image = np.clip(image.astype(np.int32) + noise, 0, 4095).astype(np.uint16)

    return noisy_image


def simulate_pipeline(width: int, height: int, num_frames: int, noise_level: float) -> dict[str, Any]:
    """Simulate the entire image sensor pipeline and return performance metrics."""
    # Initialize components
    mipi_config = MIPIConfig(lanes=4, data_rate_mbps=2500.0, pixel_format="RAW12", resolution=(width, height))
    mipi_driver = MIPIProtocolDriver(mipi_config)

    signal_config = SignalConfig(bit_depth=12, noise_reduction_strength=0.1, color_correction_matrix=np.eye(3))
    signal_processor = SignalProcessor(signal_config)

    power_config = PowerConfig(voltage_main=1.8, voltage_io=3.3, current_limit=1.0)
    power_manager = PowerManager(power_config)

    # Connect driver
    mipi_driver.connect()
    mipi_driver.start_streaming()

    # Collect metrics
    metrics: dict[str, list[float]] = {
        "snr": [],
        "dynamic_range": [],
        "color_accuracy": [],
        "power_consumption": [],
        "processing_time": [],
    }

    print(f"Simulating {num_frames} frames at {width}x{height}...")

    for i in range(num_frames):
        # Generate synthetic frame
        raw_frame = generate_synthetic_frame(width, height, noise_level)

        # Process frame
        start_time = time.time()
        processed_frame = signal_processor.process_frame(raw_frame)
        processing_time = time.time() - start_time

        # Calculate metrics
        noise_estimate = raw_frame.astype(np.float32) - processed_frame.astype(np.float32)
        snr = calculate_snr(processed_frame, noise_estimate.astype(np.uint16))
        dr = calculate_dynamic_range(processed_frame)
        color_accuracy, _ = calculate_color_accuracy(raw_frame, processed_frame)
        power_status = power_manager.get_power_status()

        metrics["snr"].append(snr)
        metrics["dynamic_range"].append(dr)
        metrics["color_accuracy"].append(color_accuracy)
        metrics["power_consumption"].append(power_status["power_consumption"])
        metrics["processing_time"].append(processing_time)

        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{num_frames} frames")

    mipi_driver.stop_streaming()
    mipi_driver.disconnect()

    # Calculate average metrics
    result: dict[str, Any] = {}
    for key, values in metrics.items():
        result[key] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }

    # Add throughput calculations
    total_time = sum(metrics["processing_time"])
    result["throughput"] = {
        "fps": num_frames / total_time if total_time > 0 else 0,
        "total_frames": num_frames,
        "total_time_s": total_time,
    }

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Image Sensor Pipeline Simulation")
    parser.add_argument("--resolution", default="1920x1080", help="Simulation resolution (WxH)")
    parser.add_argument("--frames", type=int, default=100, help="Number of frames to simulate")
    parser.add_argument("--noise", type=float, default=0.05, help="Noise level for simulation (0.0-1.0)")
    parser.add_argument("--output", default="simulation_results.json", help="Output file for results")

    args = parser.parse_args()

    width, height = map(int, args.resolution.split("x"))

    print("Simulation Configuration:")
    print(f"  Resolution: {width}x{height}")
    print(f"  Frames: {args.frames}")
    print(f"  Noise Level: {args.noise}")
    print()

    results = simulate_pipeline(width, height, args.frames, args.noise)

    print("\nSimulation Results:")
    for key, value in results.items():
        if isinstance(value, dict) and "mean" in value:
            print(f"  {key}: {value['mean']:.3f} ± {value['std']:.3f}")
        elif isinstance(value, dict):
            print(f"  {key}: {value}")

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
