#!/usr/bin/env python3
"""
Stereo Depth Example for Advanced Image Sensor Interface (v3.2.0)

Demonstrates the 3D/Depth module end to end on a synthetic rectified stereo
pair with known ground-truth disparity:

- Block Matching vs full 8-path Semi-Global Matching disparity
- 4-path (cardinal) vs 8-path (full) SGM aggregation
- numba-accelerated vs pure-numpy SGM backends (bit-identical results)
- Disparity-to-depth conversion and point cloud generation
- PLY export (built-in writer) and trimesh-backed mesh PLY export
- Typed DepthResult error handling

Usage:
    python stereo_depth_example.py [options]

Options:
    --size WxH            Stereo pair size (default: 320x240)
    --output-dir DIR      Directory for PLY exports (default: depth_output)
    --demo NAME           One of algorithms, paths, backends, export, error, all
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

try:
    from advanced_image_sensor_interface import DepthConfig, DisparityAlgorithm, StereoDepthProcessor
    from advanced_image_sensor_interface.utils.depth import NUMBA_AVAILABLE, TRIMESH_AVAILABLE
except ImportError as e:
    logger.error(f"Failed to import required modules: {e}")
    logger.error("Please install the package with: pip install -e .")
    sys.exit(1)

FOCAL_LENGTH_PX = 800.0
BASELINE_M = 0.12


def make_stereo_pair(height: int, width: int, seed: int = 7) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate a rectified stereo pair with known ground-truth disparity.

    A textured scene is rendered at three depth slabs; the right image is
    created by shifting the left image according to the per-pixel disparity
    (inverse mapping, so every right-image pixel has a valid match).

    Returns:
        (left, right, ground_truth_disparity)
    """
    from scipy.ndimage import uniform_filter

    rng = np.random.default_rng(seed)

    # Rich texture: smoothed random noise plus low-frequency gradients
    texture = uniform_filter(rng.random((height, width)).astype(np.float32), size=3)
    x, y = np.meshgrid(np.linspace(0, 1, width), np.linspace(0, 1, height))
    texture = texture + 0.2 * np.sin(8 * np.pi * x) * np.cos(6 * np.pi * y)
    texture = (np.clip(texture, 0, None) / texture.max() * 255).astype(np.uint8)

    # Three depth slabs: near (large disparity), middle, far (small disparity)
    disparity_true = np.zeros((height, width), dtype=np.float32)
    disparity_true[:, : width // 3] = 40.0
    disparity_true[:, width // 3 : 2 * width // 3] = 24.0
    disparity_true[:, 2 * width // 3 :] = 8.0

    xs = np.broadcast_to(np.arange(width)[None, :], (height, width)) + disparity_true
    xs = np.clip(xs, 0, width - 1).astype(np.int32)
    right = np.take_along_axis(texture, xs, axis=1)

    return texture, right, disparity_true


def disparity_error(estimated: np.ndarray, truth: np.ndarray) -> float:
    """Mean absolute disparity error over pixels with valid estimates."""
    valid = estimated > 0
    if not valid.any():
        return float("inf")
    return float(np.abs(estimated[valid] - truth[valid]).mean())


def run_disparity(name: str, config: DepthConfig, left: np.ndarray, right: np.ndarray, truth: np.ndarray) -> float:
    """Run one disparity configuration and report accuracy/performance."""
    processor = StereoDepthProcessor(config)
    result = processor.compute_disparity(left, right)
    if not result.success or result.disparity_map is None:
        logger.error(f"{name}: failed ({result.error})")
        return float("inf")
    error = disparity_error(result.disparity_map, truth)
    logger.info(
        f"{name}: MAE {error:.3f} px | valid {result.valid_pixel_ratio:.1%} "
        f"| {result.metrics.processing_time_ms:.1f} ms ({result.metrics.algorithm_name})"
    )
    return error


def demo_algorithms(left: np.ndarray, right: np.ndarray, truth: np.ndarray) -> None:
    """Compare Block Matching against full 8-path SGM."""
    logger.info("=== Demo: Block Matching vs SGM ===")
    run_disparity("Block Matching ", DepthConfig(algorithm=DisparityAlgorithm.BLOCK_MATCHING), left, right, truth)
    run_disparity("SGM (8 paths)  ", DepthConfig(algorithm=DisparityAlgorithm.SEMI_GLOBAL_MATCHING), left, right, truth)


def demo_paths(left: np.ndarray, right: np.ndarray, truth: np.ndarray) -> None:
    """Compare 4-path cardinal SGM against full 8-path SGM."""
    logger.info("=== Demo: SGM path count (4 cardinal vs 8 full) ===")
    run_disparity(
        "SGM (4 paths)  ", DepthConfig(algorithm=DisparityAlgorithm.SEMI_GLOBAL_MATCHING, sgm_paths=4), left, right, truth
    )
    run_disparity(
        "SGM (8 paths)  ", DepthConfig(algorithm=DisparityAlgorithm.SEMI_GLOBAL_MATCHING, sgm_paths=8), left, right, truth
    )


def demo_backends(left: np.ndarray, right: np.ndarray, truth: np.ndarray) -> None:
    """Verify numba and numpy SGM backends produce bit-identical disparity."""
    logger.info("=== Demo: numba vs numpy SGM backends ===")
    if not NUMBA_AVAILABLE:
        logger.info("numba is not installed; the numpy fallback backend is used by default.")
        logger.info("Install it with `pip install numba` (part of the [full] extra) to compare.")
        return

    sgm = DisparityAlgorithm.SEMI_GLOBAL_MATCHING
    processor_numba = StereoDepthProcessor(DepthConfig(algorithm=sgm, use_numba=True))
    processor_numpy = StereoDepthProcessor(DepthConfig(algorithm=sgm, use_numba=False))

    result_numba = processor_numba.compute_disparity(left, right)
    result_numpy = processor_numpy.compute_disparity(left, right)
    if not result_numba.success or not result_numpy.success:
        logger.error("Backend comparison failed: one of the runs returned an error")
        return
    assert result_numba.disparity_map is not None and result_numpy.disparity_map is not None

    identical = np.array_equal(result_numba.disparity_map, result_numpy.disparity_map)
    logger.info(f"numba : {result_numba.metrics.processing_time_ms:.1f} ms")
    logger.info(f"numpy : {result_numpy.metrics.processing_time_ms:.1f} ms")
    logger.info(f"bit-identical results: {identical}")
    speedup = result_numpy.metrics.processing_time_ms / max(result_numba.metrics.processing_time_ms, 1e-9)
    logger.info(f"numba speedup: {speedup:.1f}x")


def demo_export(left: np.ndarray, right: np.ndarray, output_dir: Path) -> None:
    """Run the full pipeline: disparity -> depth -> point cloud -> PLY exports."""
    logger.info("=== Demo: depth conversion and PLY export ===")
    output_dir.mkdir(parents=True, exist_ok=True)

    config = DepthConfig(algorithm=DisparityAlgorithm.SEMI_GLOBAL_MATCHING)
    processor = StereoDepthProcessor(config)

    ply_path = output_dir / "stereo_point_cloud.ply"
    result = processor.process_stereo_pair(left, right, FOCAL_LENGTH_PX, BASELINE_M, export_path=ply_path)
    if not result.success or result.depth_map is None or result.point_cloud is None:
        logger.error(f"Full stereo pipeline failed: {result.error}")
        return

    depth = result.depth_map
    valid = depth > 0
    logger.info(f"Depth range: {depth[valid].min():.2f} m to {depth[valid].max():.2f} m (valid {valid.mean():.1%})")
    logger.info(f"Point cloud: {result.point_cloud.shape[0]} points exported to {ply_path}")

    mesh_path = output_dir / "stereo_mesh.ply"
    if TRIMESH_AVAILABLE:
        processor.export_mesh_ply(depth, FOCAL_LENGTH_PX, BASELINE_M, mesh_path)
        logger.info(f"trimesh-backed PLY export written to {mesh_path}")
    else:
        logger.info("trimesh is not installed; skipping export_mesh_ply (pip install trimesh or the [full] extra)")


def demo_error_handling() -> None:
    """Show typed DepthResult failure reporting on a mismatched stereo pair."""
    logger.info("=== Demo: typed error handling ===")
    processor = StereoDepthProcessor(DepthConfig())
    left = np.zeros((64, 128), dtype=np.uint8)
    right = np.zeros((48, 128), dtype=np.uint8)  # wrong shape on purpose

    result = processor.compute_disparity(left, right)
    logger.info(f"success={result.success}, error={result.error!r}")
    logger.info("No exception raised: failures are reported through the DepthResult object.")


def parse_size(size_str: str) -> tuple[int, int]:
    """Parse a WxH size string into (width, height)."""
    try:
        width_str, height_str = size_str.lower().split("x")
        return int(width_str), int(height_str)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid size format: {size_str!r} (expected WxH, e.g. 320x240)") from exc


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stereo Depth Example (v3.2.0)", formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--size", type=parse_size, default=(320, 240), help="Stereo pair size as WxH")
    parser.add_argument("--output-dir", type=Path, default=Path("depth_output"), help="Directory for PLY exports")
    parser.add_argument(
        "--demo", choices=["algorithms", "paths", "backends", "export", "error", "all"], default="all", help="Which demo to run"
    )
    args = parser.parse_args()

    width, height = args.size
    logger.info("Stereo Depth Example (v3.2.0)")
    logger.info("=" * 50)
    logger.info(f"Stereo pair: {width}x{height}, focal length {FOCAL_LENGTH_PX}px, baseline {BASELINE_M} m")
    logger.info(
        f"Backends: numba {'available' if NUMBA_AVAILABLE else 'not installed'}, trimesh {'available' if TRIMESH_AVAILABLE else 'not installed'}"
    )
    logger.info("=" * 50)

    left, right, truth = make_stereo_pair(height, width)

    if args.demo in ["algorithms", "all"]:
        demo_algorithms(left, right, truth)
    if args.demo in ["paths", "all"]:
        demo_paths(left, right, truth)
    if args.demo in ["backends", "all"]:
        demo_backends(left, right, truth)
    if args.demo in ["export", "all"]:
        demo_export(left, right, args.output_dir)
    if args.demo in ["error", "all"]:
        demo_error_handling()

    logger.info("=" * 50)
    logger.info("Stereo depth demos completed.")


if __name__ == "__main__":
    main()
