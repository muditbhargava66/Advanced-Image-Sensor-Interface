#!/usr/bin/env python3
"""
Native Calibration Solver Example for Advanced Image Sensor Interface (v3.2.0)

Demonstrates the OpenCV-free photogrammetry solver built on numpy/scipy:

- ``calibrate_camera``: Zhang-style planar-pattern calibration from multiple
  views (normalized DLT homographies, closed-form intrinsics, per-view
  extrinsics, Levenberg-Marquardt reprojection refinement)
- ``solve_projection_matrix``: general DLT projection-matrix estimation from
  3D-2D correspondences, decomposed into intrinsics and pose via RQ
- Typed error handling for degenerate inputs

The example recovers a known synthetic camera from noisy observations and
reports estimation errors. The same solver powers multi-sensor calibration
when ``SyncConfiguration.prefer_native_calibration`` is enabled.

Usage:
    python native_calibration_example.py [options]

Options:
    --views VIEWS         Number of calibration pattern views (default: 8)
    --noise SIGMA         Observation noise in pixels (default: 0.15)
    --seed SEED           Random seed for reproducibility
"""

import argparse
import logging
import sys

import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

try:
    from advanced_image_sensor_interface.sensor_interface.calibration import calibrate_camera, solve_projection_matrix
except ImportError as e:
    logger.error(f"Failed to import required modules: {e}")
    logger.error("Please install the package with: pip install -e .")
    sys.exit(1)

IMAGE_SIZE = (640, 480)  # (width, height)
TRUE_INTRINSIC = np.array([[800.0, 0.0, 320.0], [0.0, 800.0, 240.0], [0.0, 0.0, 1.0]])
PATTERN_SIZE = (7, 5)  # inner chessboard corners
SQUARE_MM = 25.0


def rotation_from_axis_angle(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    """Rodrigues rotation formula: axis-angle to rotation matrix."""
    k = axis / np.linalg.norm(axis)
    skew = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + np.sin(angle_rad) * skew + (1.0 - np.cos(angle_rad)) * (skew @ skew)


def project(points_3d: np.ndarray, rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    """Project 3D points to pixel coordinates with the demo camera intrinsics."""
    camera_coords = (rotation @ points_3d.T + translation[:, None]).T
    image_coords = (TRUE_INTRINSIC @ camera_coords.T).T
    return image_coords[:, :2] / image_coords[:, 2:3]


def pattern_object_points() -> np.ndarray:
    """(N, 3) object coordinates of the planar chessboard corners (Z = 0)."""
    grid = np.mgrid[0 : PATTERN_SIZE[0], 0 : PATTERN_SIZE[1]].T.reshape(-1, 2)
    object_points = np.zeros((grid.shape[0], 3))
    object_points[:, :2] = grid * SQUARE_MM
    return object_points


def generate_calibration_views(num_views: int, noise_sigma: float, seed: int) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Generate synthetic pattern views observed by the known camera.

    Views whose projected corners fall outside the sensor are rejected and
    regenerated, so every returned view is fully visible.
    """
    rng = np.random.default_rng(seed)
    object_points = pattern_object_points()
    margin = 40.0
    width, height = IMAGE_SIZE

    object_views: list[np.ndarray] = []
    image_views: list[np.ndarray] = []
    while len(object_views) < num_views:
        axis = rng.normal(size=3)
        angle = rng.uniform(0.1, 0.6)
        rotation = rotation_from_axis_angle(axis, angle)
        translation = np.array([rng.uniform(-80, 80), rng.uniform(-60, 60), rng.uniform(320, 480)])

        projected = project(object_points, rotation, translation)
        inside = (
            (projected[:, 0] > margin)
            & (projected[:, 0] < width - margin)
            & (projected[:, 1] > margin)
            & (projected[:, 1] < height - margin)
        )
        if not inside.all():
            continue

        noisy = projected + rng.normal(0.0, noise_sigma, projected.shape)
        object_views.append(object_points.copy())
        image_views.append(noisy)

    return object_views, image_views


def demo_zhang_calibration(num_views: int, noise_sigma: float, seed: int) -> None:
    """Recover camera intrinsics from multiple noisy pattern views."""
    logger.info("=== Demo: Zhang's method camera calibration ===")
    object_views, image_views = generate_calibration_views(num_views, noise_sigma, seed)
    logger.info(f"Generated {num_views} views of a {PATTERN_SIZE[0]}x{PATTERN_SIZE[1]} pattern (square {SQUARE_MM} mm)")

    result = calibrate_camera(object_views, image_views, IMAGE_SIZE)

    logger.info(f"RMS reprojection error: {result.rms_reprojection_error:.4f} px (observation noise {noise_sigma} px)")
    estimated = result.camera_matrix
    for name, row, col in [("fx", 0, 0), ("fy", 1, 1), ("cx", 0, 2), ("cy", 1, 2)]:
        true_value = TRUE_INTRINSIC[row, col]
        est_value = estimated[row, col]
        logger.info(
            f"  {name}: estimated {est_value:8.2f} | true {true_value:8.2f} | error {abs(est_value - true_value) / true_value:.2%}"
        )
    logger.info(f"  distortion coefficients: {result.distortion_coefficients} (not estimated in v1)")
    logger.info(f"  per-view extrinsics recovered for {len(result.rotation_vectors)} views")


def demo_projection_dlt(noise_sigma: float, seed: int) -> None:
    """Recover a projection matrix from general 3D-2D correspondences."""
    logger.info("=== Demo: DLT projection matrix + RQ decomposition ===")
    rng = np.random.default_rng(seed + 1)

    # Non-coplanar calibration object: points spread through a volume
    object_points = rng.uniform(-100.0, 100.0, (12, 3))
    object_points[:, 2] = rng.uniform(0.0, 150.0, 12) + 300.0  # ensure depth variation

    rotation = rotation_from_axis_angle(np.array([0.3, -0.8, 0.5]), 0.35)
    translation = np.array([10.0, -15.0, 120.0])

    image_points = project(object_points, rotation, translation)
    image_points = image_points + rng.normal(0.0, noise_sigma, image_points.shape)

    intrinsic, rotation_est, translation_est = solve_projection_matrix(object_points, image_points)

    # Normalize the scale/sign ambiguity inherent to the homogeneous DLT
    scale = intrinsic[2, 2]
    intrinsic = intrinsic / scale
    if intrinsic[0, 0] < 0:
        intrinsic = -intrinsic
        rotation_est = -rotation_est
        translation_est = -translation_est

    reproj = (intrinsic @ (rotation_est @ object_points.T + translation_est[:, None])).T
    reproj = reproj[:, :2] / reproj[:, 2:3]
    residual = float(np.sqrt(np.mean(np.sum((reproj - image_points) ** 2, axis=1))))
    logger.info(f"Reprojection RMS after RQ decomposition: {residual:.4f} px")

    for name, row, col in [("fx", 0, 0), ("fy", 1, 1), ("cx", 0, 2), ("cy", 1, 2)]:
        true_value = TRUE_INTRINSIC[row, col]
        est_value = intrinsic[row, col]
        logger.info(
            f"  {name}: estimated {est_value:8.2f} | true {true_value:8.2f} | error {abs(est_value - true_value) / true_value:.2%}"
        )

    # Pose comparison (translation compared up to scale; direction is what DLT fixes)
    rot_error = np.arccos(np.clip((np.trace(rotation_est.T @ rotation) - 1.0) / 2.0, -1.0, 1.0))
    logger.info(f"  rotation error: {np.degrees(rot_error):.3f} deg")
    true_dir = translation / np.linalg.norm(translation)
    est_dir = translation_est / np.linalg.norm(translation_est)
    logger.info(f"  translation direction error: {np.degrees(np.arccos(np.clip(true_dir @ est_dir, -1.0, 1.0))):.3f} deg")


def demo_error_handling() -> None:
    """Show explicit ValueError reporting for degenerate inputs."""
    logger.info("=== Demo: typed error handling ===")
    object_points = pattern_object_points()
    image_views = [np.zeros((object_points.shape[0], 2))] * 2  # only two views

    try:
        calibrate_camera([object_points] * 2, image_views, IMAGE_SIZE)
        logger.error("Expected ValueError was not raised")
    except ValueError as exc:
        logger.info(f"Too few views rejected: {exc}")

    try:
        solve_projection_matrix(np.zeros((4, 3)), np.zeros((4, 2)))
        logger.error("Expected ValueError was not raised")
    except ValueError as exc:
        logger.info(f"Too few correspondences rejected: {exc}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Native Calibration Solver Example (v3.2.0)", formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--views", type=int, default=8, help="Number of calibration pattern views")
    parser.add_argument("--noise", type=float, default=0.15, help="Observation noise in pixels")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    logger.info("Native Calibration Solver Example (v3.2.0)")
    logger.info("=" * 50)
    logger.info(f"Image size: {IMAGE_SIZE[0]}x{IMAGE_SIZE[1]}, true fx=fy=800, cx=320, cy=240")
    logger.info("=" * 50)

    demo_zhang_calibration(args.views, args.noise, args.seed)
    demo_projection_dlt(args.noise, args.seed)
    demo_error_handling()

    logger.info("=" * 50)
    logger.info("Native calibration demos completed.")


if __name__ == "__main__":
    main()
