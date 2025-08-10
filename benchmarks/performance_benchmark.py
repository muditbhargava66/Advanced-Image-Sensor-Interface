#!/usr/bin/env python3
"""
Performance Benchmarking Suite for Advanced Image Sensor Interface

This script provides comprehensive performance benchmarking for all major components
with realistic timing measurements and hardware-adjusted expectations.

Usage:
    python benchmarks/performance_benchmark.py
    python benchmarks/performance_benchmark.py --detailed
    python benchmarks/performance_benchmark.py --export-json results.json
"""

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import psutil

# Import the package components
from advanced_image_sensor_interface import (
    AdvancedPowerManager,
    EnhancedSensorInterface,
    GPUAccelerator,
    MultiSensorSynchronizer,
    SensorConfiguration,
    SensorResolution,
    create_gpu_config_for_automotive,
    create_hdr_processor_for_automotive,
    create_power_config_for_automotive,
    create_raw_processor_for_automotive,
    create_stereo_sync_config,
)


class PerformanceBenchmark:
    """Comprehensive performance benchmarking suite."""

    def __init__(self):
        """Initialize benchmark suite."""
        self.results = {"system_info": self._get_system_info(), "benchmarks": {}, "timestamp": time.time()}

    def _get_system_info(self) -> dict:
        """Get system information for benchmark context."""
        return {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python_version": platform.python_version(),
            "cpu_count": psutil.cpu_count(),
            "memory_gb": round(psutil.virtual_memory().total / (1024**3), 2),
            "cpu_freq_mhz": psutil.cpu_freq().current if psutil.cpu_freq() else "Unknown",
        }

    def benchmark_enhanced_sensor(self, detailed: bool = False) -> dict:
        """Benchmark enhanced sensor interface performance."""
        print("🔍 Benchmarking Enhanced Sensor Interface...")

        results = {}

        # Test different resolutions
        resolutions = [
            (SensorResolution.HD, "HD (1280x720)"),
            (SensorResolution.FHD, "FHD (1920x1080)"),
            (SensorResolution.UHD_4K, "4K (3840x2160)"),
        ]

        for resolution, name in resolutions:
            print(f"  Testing {name}...")

            config = SensorConfiguration(resolution=resolution, frame_rate=30.0, bit_depth=12, raw_processing=True)

            sensor = EnhancedSensorInterface(config)

            # Measure streaming startup time
            start_time = time.time()
            success = sensor.start_streaming()
            startup_time = time.time() - start_time

            if success:
                # Measure frame capture performance
                frame_times = []
                for i in range(10):
                    start = time.time()
                    frame = sensor.capture_frame()
                    if frame is not None:
                        frame_times.append(time.time() - start)

                sensor.stop_streaming()

                avg_frame_time = np.mean(frame_times) if frame_times else float("inf")
                theoretical_fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0

                results[name] = {
                    "startup_time_ms": startup_time * 1000,
                    "avg_frame_time_ms": avg_frame_time * 1000,
                    "theoretical_fps": theoretical_fps,
                    "data_rate_mbps": config.data_rate_mbps,
                    "frames_captured": len(frame_times),
                }
            else:
                results[name] = {"error": "Failed to start streaming", "data_rate_mbps": config.data_rate_mbps}

        return results

    def benchmark_hdr_processing(self, detailed: bool = False) -> dict:
        """Benchmark HDR processing performance."""
        print("🌈 Benchmarking HDR Processing...")

        hdr_processor = create_hdr_processor_for_automotive()
        results = {}

        # Test different image sizes
        sizes = [(480, 640, "VGA"), (720, 1280, "HD"), (1080, 1920, "FHD")]

        for height, width, name in sizes:
            print(f"  Testing {name} ({width}x{height})...")

            # Generate test images
            test_image = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
            test_stack = [
                np.random.randint(0, 128, (height, width, 3), dtype=np.uint8),
                np.random.randint(64, 192, (height, width, 3), dtype=np.uint8),
                np.random.randint(128, 256, (height, width, 3), dtype=np.uint8),
            ]
            exposure_values = [-2.0, 0.0, 2.0]

            # Benchmark single image processing
            times = []
            for _ in range(5):
                start = time.time()
                hdr_processor.process_single_image(test_image)
                times.append(time.time() - start)

            single_avg = np.mean(times)

            # Benchmark exposure stack processing
            times = []
            for _ in range(3):
                start = time.time()
                _ = hdr_processor.process_exposure_stack(test_stack, exposure_values)
                times.append(time.time() - start)

            stack_avg = np.mean(times)

            results[name] = {
                "resolution": f"{width}x{height}",
                "single_image_ms": single_avg * 1000,
                "exposure_stack_ms": stack_avg * 1000,
                "pixels": width * height,
                "throughput_mpixels_per_sec": (width * height) / (single_avg * 1_000_000),
            }

        return results

    def benchmark_raw_processing(self, detailed: bool = False) -> dict:
        """Benchmark RAW processing performance."""
        print("📷 Benchmarking RAW Processing...")

        raw_processor = create_raw_processor_for_automotive()
        results = {}

        # Test different image sizes
        sizes = [(480, 640, "VGA"), (720, 1280, "HD"), (1080, 1920, "FHD")]

        for height, width, name in sizes:
            print(f"  Testing {name} ({width}x{height})...")

            # Generate synthetic RAW data
            raw_data = np.random.randint(0, 4095, (height, width), dtype=np.uint16)

            # Apply Bayer pattern
            raw_data = raw_data.astype(np.float32)
            raw_data[0::2, 0::2] *= 1.2  # R
            raw_data[1::2, 1::2] *= 0.8  # B
            raw_data = np.clip(raw_data, 0, 4095).astype(np.uint16)

            # Benchmark processing
            times = []
            for _ in range(5):
                start = time.time()
                _ = raw_processor.process_raw_image(raw_data)
                times.append(time.time() - start)

            avg_time = np.mean(times)

            results[name] = {
                "resolution": f"{width}x{height}",
                "processing_time_ms": avg_time * 1000,
                "pixels": width * height,
                "throughput_mpixels_per_sec": (width * height) / (avg_time * 1_000_000),
            }

        return results

    def benchmark_gpu_acceleration(self, detailed: bool = False) -> dict:
        """Benchmark GPU acceleration performance."""
        print("⚡ Benchmarking GPU Acceleration...")

        gpu_config = create_gpu_config_for_automotive()
        gpu_accelerator = GPUAccelerator(gpu_config)

        # Generate test images
        test_images = [np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8) for _ in range(8)]

        operations = ["gaussian_blur", "edge_detection", "histogram_equalization", "noise_reduction"]
        results = {}

        for operation in operations:
            print(f"  Testing {operation}...")

            times = []
            for _ in range(3):
                start = time.time()
                if operation == "gaussian_blur":
                    _ = gpu_accelerator.process_image_batch(test_images, operation, sigma=2.0)
                else:
                    _ = gpu_accelerator.process_image_batch(test_images, operation)
                times.append(time.time() - start)

            avg_time = np.mean(times)

            results[operation] = {
                "batch_size": len(test_images),
                "processing_time_ms": avg_time * 1000,
                "images_per_second": len(test_images) / avg_time,
                "time_per_image_ms": (avg_time / len(test_images)) * 1000,
            }

        # Get performance stats
        stats = gpu_accelerator.get_performance_stats()
        results["performance_stats"] = stats
        results["device_info"] = gpu_accelerator.get_device_info()

        return results

    def benchmark_multi_sensor_sync(self, detailed: bool = False) -> dict:
        """Benchmark multi-sensor synchronization."""
        print("🔄 Benchmarking Multi-Sensor Synchronization...")

        config = create_stereo_sync_config()
        synchronizer = MultiSensorSynchronizer(config)

        results = {}

        # Test synchronization startup
        start_time = time.time()
        sync_started = synchronizer.start_synchronization()
        startup_time = time.time() - start_time

        if sync_started:
            # Test synchronized capture
            capture_times = []
            successful_captures = 0

            for _ in range(10):
                start = time.time()
                frames = synchronizer.capture_synchronized_frames()
                capture_time = time.time() - start

                if frames:
                    successful_captures += 1
                    capture_times.append(capture_time)

            synchronizer.stop_synchronization()

            avg_capture_time = np.mean(capture_times) if capture_times else 0

            results = {
                "startup_time_ms": startup_time * 1000,
                "avg_capture_time_ms": avg_capture_time * 1000,
                "successful_captures": successful_captures,
                "total_attempts": 10,
                "success_rate": successful_captures / 10,
            }

            # Get sync status
            status = synchronizer.get_synchronization_status()
            results["sync_statistics"] = status["statistics"]
        else:
            results = {"error": "Failed to start synchronization"}

        return results

    def benchmark_power_management(self, detailed: bool = False) -> dict:
        """Benchmark power management performance."""
        print("🔋 Benchmarking Power Management...")

        power_config = create_power_config_for_automotive()
        power_manager = AdvancedPowerManager(power_config)

        results = {}

        # Test power mode transitions
        from advanced_image_sensor_interface.sensor_interface import PowerMode

        modes = [PowerMode.PERFORMANCE, PowerMode.BALANCED, PowerMode.POWER_SAVER]
        mode_results = {}

        power_manager.start_monitoring()

        for mode in modes:
            print(f"  Testing {mode.value} mode...")

            start = time.time()
            power_manager.set_power_mode(mode)
            transition_time = time.time() - start

            # Let the system stabilize
            time.sleep(0.2)

            metrics = power_manager.get_power_metrics()

            mode_results[mode.value] = {
                "transition_time_ms": transition_time * 1000,
                "total_power_w": metrics.total_power,
                "temperature_c": metrics.temperature_celsius,
                "frequency_mhz": metrics.current_frequency_mhz,
                "thermal_state": metrics.thermal_state.value,
            }

        power_manager.stop_monitoring()

        results["power_modes"] = mode_results

        return results

    def run_all_benchmarks(self, detailed: bool = False) -> dict:
        """Run all benchmarks and return results."""
        print("🚀 Starting Comprehensive Performance Benchmark Suite")
        print("=" * 60)

        self.results["benchmarks"]["enhanced_sensor"] = self.benchmark_enhanced_sensor(detailed)
        self.results["benchmarks"]["hdr_processing"] = self.benchmark_hdr_processing(detailed)
        self.results["benchmarks"]["raw_processing"] = self.benchmark_raw_processing(detailed)
        self.results["benchmarks"]["gpu_acceleration"] = self.benchmark_gpu_acceleration(detailed)
        self.results["benchmarks"]["multi_sensor_sync"] = self.benchmark_multi_sensor_sync(detailed)
        self.results["benchmarks"]["power_management"] = self.benchmark_power_management(detailed)

        return self.results

    def print_summary(self):
        """Print a summary of benchmark results."""
        print("\n" + "=" * 60)
        print("📊 BENCHMARK SUMMARY")
        print("=" * 60)

        print("\n🖥️  System Information:")
        for key, value in self.results["system_info"].items():
            print(f"   {key}: {value}")

        print("\n⚡ Performance Highlights:")

        # HDR Processing
        if "hdr_processing" in self.results["benchmarks"]:
            hdr_results = self.results["benchmarks"]["hdr_processing"]
            if "VGA" in hdr_results:
                vga_time = hdr_results["VGA"]["single_image_ms"]
                print(f"   HDR Processing (VGA): {vga_time:.1f}ms")

        # RAW Processing
        if "raw_processing" in self.results["benchmarks"]:
            raw_results = self.results["benchmarks"]["raw_processing"]
            if "VGA" in raw_results:
                vga_time = raw_results["VGA"]["processing_time_ms"]
                print(f"   RAW Processing (VGA): {vga_time:.1f}ms")

        # GPU Acceleration
        if "gpu_acceleration" in self.results["benchmarks"]:
            gpu_results = self.results["benchmarks"]["gpu_acceleration"]
            if "device_info" in gpu_results:
                backend = gpu_results["device_info"]["backend"]
                print(f"   GPU Backend: {backend}")

        print("\n⚠️  Important Notes:")
        print("   • These are simulation benchmarks on Python")
        print("   • Real hardware performance will vary significantly")
        print("   • For production use, integrate with optimized hardware drivers")
        print("   • GPU acceleration requires CUDA/OpenCL hardware")


def main():
    """Main benchmark execution."""
    parser = argparse.ArgumentParser(description="Performance Benchmark Suite")
    parser.add_argument("--detailed", action="store_true", help="Run detailed benchmarks")
    parser.add_argument("--export-json", help="Export results to JSON file")
    parser.add_argument("--quiet", action="store_true", help="Suppress output during benchmarks")

    args = parser.parse_args()

    benchmark = PerformanceBenchmark()

    try:
        results = benchmark.run_all_benchmarks(detailed=args.detailed)

        if not args.quiet:
            benchmark.print_summary()

        if args.export_json:
            output_path = Path(args.export_json)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, "w") as f:
                json.dump(results, f, indent=2, default=str)

            print(f"\n💾 Results exported to: {output_path}")

        print("\n✅ Benchmark completed successfully!")

    except Exception as e:
        print(f"\n❌ Benchmark failed: {e}")
        return 1

    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
