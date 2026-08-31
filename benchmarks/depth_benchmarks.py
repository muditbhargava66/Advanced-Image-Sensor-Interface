"""
Depth Module Benchmarks for Advanced Image Sensor Interface (v3.2.0)

This module benchmarks the stereo depth pipeline introduced in v3.2.0:

Functions:
    benchmark_disparity_algorithms: Block Matching vs full 8-path SGM throughput.
    benchmark_sgm_paths: 4-path cardinal vs 8-path full SGM aggregation.
    benchmark_sgm_backends: numba vs numpy kernels (timing + bit-identity check).
    run_depth_benchmarks: Run all depth benchmarks and report results.

Usage:
    from benchmarks.depth_benchmarks import run_depth_benchmarks
    run_depth_benchmarks()
"""

import json
import time
from typing import Any

import numpy as np

from advanced_image_sensor_interface.utils.depth import NUMBA_AVAILABLE, DepthConfig, DisparityAlgorithm, StereoDepthProcessor

FOCAL_LENGTH_PX = 800.0
BASELINE_M = 0.12


def make_stereo_pair(height: int, width: int, seed: int = 7) -> tuple[np.ndarray, np.ndarray]:
    """Generate a textured rectified stereo pair with three depth slabs.

    Args:
        height: Image height in pixels.
        width: Image width in pixels.
        seed: Random seed for reproducibility.

    Returns:
        (left, right) grayscale uint8 images with known per-pixel disparity.
    """
    from scipy.ndimage import uniform_filter

    rng = np.random.default_rng(seed)

    texture = uniform_filter(rng.random((height, width)).astype(np.float32), size=3)
    x, y = np.meshgrid(np.linspace(0, 1, width), np.linspace(0, 1, height))
    texture = texture + 0.2 * np.sin(8 * np.pi * x) * np.cos(6 * np.pi * y)
    texture = (np.clip(texture, 0, None) / texture.max() * 255).astype(np.uint8)

    disparity_true = np.zeros((height, width), dtype=np.float32)
    disparity_true[:, : width // 3] = 40.0
    disparity_true[:, width // 3 : 2 * width // 3] = 24.0
    disparity_true[:, 2 * width // 3 :] = 8.0

    xs = np.broadcast_to(np.arange(width)[None, :], (height, width)) + disparity_true
    xs = np.clip(xs, 0, width - 1).astype(np.int32)
    right = np.take_along_axis(texture, xs, axis=1)

    return texture, right


def _time_disparity(
    config: DepthConfig, left: np.ndarray, right: np.ndarray, warmup: int = 0, repeats: int = 3
) -> dict[str, Any]:
    """Time a disparity configuration and report per-run statistics.

    Args:
        config: Depth configuration under test.
        left: Left image of the rectified pair.
        right: Right image of the rectified pair.
        warmup: Untimed warm-up runs (useful for numba JIT compilation).
        repeats: Number of timed repetitions.

    Returns:
        Dict with mean/std run times, throughput, and validity metrics.
    """
    processor = StereoDepthProcessor(config)
    for _ in range(warmup):
        processor.compute_disparity(left, right)

    elapsed: list[float] = []
    result = None
    for _ in range(repeats):
        start = time.perf_counter()
        result = processor.compute_disparity(left, right)
        elapsed.append(time.perf_counter() - start)

    if result is None or not result.success:
        raise RuntimeError(f"Disparity benchmark failed: {result.error if result else 'no result'}")

    mean_ms = float(np.mean(elapsed)) * 1000.0
    return {
        "mean_ms": mean_ms,
        "std_ms": float(np.std(elapsed)) * 1000.0,
        "fps": 1000.0 / mean_ms if mean_ms > 0 else 0.0,
        "valid_pixel_ratio": result.valid_pixel_ratio,
        "backend_metrics_ms": result.metrics.processing_time_ms,
    }


def benchmark_disparity_algorithms(
    sizes: list[tuple[int, int]] = [(160, 120), (320, 240)], num_disparities: int = 64
) -> dict[str, Any]:
    """Benchmark Block Matching against full 8-path SGM.

    Args:
        sizes: List of (width, height) resolutions to test.
        num_disparities: Disparity search range.

    Returns:
        Per-size timing results for both algorithms.
    """
    results: dict[str, Any] = {}
    for width, height in sizes:
        left, right = make_stereo_pair(height, width)
        bm = _time_disparity(
            DepthConfig(algorithm=DisparityAlgorithm.BLOCK_MATCHING, num_disparities=num_disparities), left, right
        )
        sgm = _time_disparity(
            DepthConfig(algorithm=DisparityAlgorithm.SEMI_GLOBAL_MATCHING, num_disparities=num_disparities),
            left,
            right,
            warmup=1 if NUMBA_AVAILABLE else 0,
        )
        results[f"{width}x{height}"] = {"block_matching": bm, "sgm_8_paths": sgm}
    return results


def benchmark_sgm_paths(width: int = 320, height: int = 240, num_disparities: int = 64) -> dict[str, Any]:
    """Benchmark 4-path cardinal SGM against full 8-path SGM.

    Args:
        width: Image width in pixels.
        height: Image height in pixels.
        num_disparities: Disparity search range.

    Returns:
        Timing results for both path configurations.
    """
    left, right = make_stereo_pair(height, width)
    warmup = 1 if NUMBA_AVAILABLE else 0
    paths_4 = _time_disparity(
        DepthConfig(algorithm=DisparityAlgorithm.SEMI_GLOBAL_MATCHING, num_disparities=num_disparities, sgm_paths=4),
        left,
        right,
        warmup=warmup,
    )
    paths_8 = _time_disparity(
        DepthConfig(algorithm=DisparityAlgorithm.SEMI_GLOBAL_MATCHING, num_disparities=num_disparities, sgm_paths=8),
        left,
        right,
        warmup=warmup,
    )
    return {"4_paths_cardinal": paths_4, "8_paths_full": paths_8}


def benchmark_sgm_backends(width: int = 320, height: int = 240, num_disparities: int = 64) -> dict[str, Any]:
    """Benchmark numba vs numpy SGM kernels and verify bit-identical output.

    Args:
        width: Image width in pixels.
        height: Image height in pixels.
        num_disparities: Disparity search range.

    Returns:
        Timing results per backend plus the bit-identity verdict. When numba
        is unavailable, only the numpy backend is reported.
    """
    left, right = make_stereo_pair(height, width)
    sgm = DisparityAlgorithm.SEMI_GLOBAL_MATCHING

    numpy_stats = _time_disparity(DepthConfig(algorithm=sgm, num_disparities=num_disparities, use_numba=False), left, right)
    results: dict[str, Any] = {"numpy": numpy_stats, "numba_available": NUMBA_AVAILABLE}
    if not NUMBA_AVAILABLE:
        return results

    numba_stats = _time_disparity(
        DepthConfig(algorithm=sgm, num_disparities=num_disparities, use_numba=True), left, right, warmup=1
    )
    results["numba"] = numba_stats
    results["speedup"] = numpy_stats["mean_ms"] / numba_stats["mean_ms"] if numba_stats["mean_ms"] > 0 else 0.0

    processor_numba = StereoDepthProcessor(DepthConfig(algorithm=sgm, num_disparities=num_disparities, use_numba=True))
    processor_numpy = StereoDepthProcessor(DepthConfig(algorithm=sgm, num_disparities=num_disparities, use_numba=False))
    result_numba = processor_numba.compute_disparity(left, right)
    result_numpy = processor_numpy.compute_disparity(left, right)
    assert result_numba.disparity_map is not None and result_numpy.disparity_map is not None
    results["bit_identical"] = bool(np.array_equal(result_numba.disparity_map, result_numpy.disparity_map))
    return results


def run_depth_benchmarks(quick: bool = False) -> dict[str, Any]:
    """Run all depth benchmarks and compile results.

    Args:
        quick: Use a single small resolution for a fast smoke benchmark.

    Returns:
        Dictionary containing all depth benchmark results.
    """
    sizes = [(160, 120)] if quick else [(160, 120), (320, 240)]
    width, height = sizes[-1]
    return {
        "numba_available": NUMBA_AVAILABLE,
        "algorithms": benchmark_disparity_algorithms(sizes=sizes),
        "sgm_paths": benchmark_sgm_paths(width, height),
        "sgm_backends": benchmark_sgm_backends(width, height),
    }


if __name__ == "__main__":
    depth_results = run_depth_benchmarks()
    print("Depth Benchmark Results:")
    print(json.dumps(depth_results, indent=2))
