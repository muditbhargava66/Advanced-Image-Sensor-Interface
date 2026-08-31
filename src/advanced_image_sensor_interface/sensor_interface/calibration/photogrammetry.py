"""
Native Photogrammetry Calibration Solver

Camera calibration implemented entirely with numpy/scipy so it works without
OpenCV. Provides:

- ``calibrate_camera``: Zhang-style planar-pattern calibration. Per-view
  homographies are estimated with the normalized DLT, intrinsics are solved
  in closed form from the homography constraints, per-view extrinsics follow
  from each homography, and everything is refined by minimizing reprojection
  error with Levenberg-Marquardt (``scipy.optimize.least_squares``).
- ``solve_projection_matrix``: general DLT projection-matrix estimation from
  at least six 3D-2D correspondences, decomposed into intrinsics and pose
  via RQ factorization.

Limitations (v1):
    Lens distortion is not estimated; distortion coefficients are returned
    as zeros. For lenses with significant distortion, prefer the OpenCV
    calibration path.

Author: Advanced Image Sensor Interface Team
Version: 3.2.0
"""

import logging
from collections.abc import Sequence

import numpy as np
from scipy.linalg import rq
from scipy.optimize import least_squares

from .models import CalibrationResult

logger = logging.getLogger(__name__)

_MIN_VIEWS = 3
_MIN_POINTS_PER_VIEW = 4
_MIN_POINTS_FOR_PROJECTION = 6


def calibrate_camera(
    object_points_per_view: Sequence[np.ndarray], image_points_per_view: Sequence[np.ndarray], image_size: tuple[int, int]
) -> CalibrationResult:
    """Calibrate a camera from multiple views of a planar pattern.

    Args:
        object_points_per_view: Per-view (N, 3) pattern coordinates with Z=0
            (cv2-style (N, 1, 3) arrays are accepted)
        image_points_per_view: Per-view (N, 2) detected image coordinates
            (cv2-style (N, 1, 2) arrays are accepted)
        image_size: Image dimensions as (width, height)

    Returns:
        CalibrationResult with estimated intrinsics, per-view rotation and
        translation vectors, and RMS reprojection error. Distortion
        coefficients are zeros (not estimated in v1).

    Raises:
        ValueError: Fewer than 3 views, fewer than 4 points in a view,
            mismatched view counts, a non-planar pattern, or a degenerate
            view configuration that prevents solving the intrinsics.
    """
    num_views = len(object_points_per_view)
    if num_views < _MIN_VIEWS:
        raise ValueError(f"calibration requires at least {_MIN_VIEWS} views, got {num_views}")
    if len(image_points_per_view) != num_views:
        raise ValueError("object_points_per_view and image_points_per_view must have the same length")

    object_views = [np.asarray(view, dtype=np.float64).reshape(-1, 3) for view in object_points_per_view]
    image_views = [np.asarray(view, dtype=np.float64).reshape(-1, 2) for view in image_points_per_view]
    for index, (obj_view, img_view) in enumerate(zip(object_views, image_views)):
        if len(obj_view) < _MIN_POINTS_PER_VIEW or len(img_view) < _MIN_POINTS_PER_VIEW:
            raise ValueError(f"view {index} has fewer than {_MIN_POINTS_PER_VIEW} points")
        if len(obj_view) != len(img_view):
            raise ValueError(f"view {index} has mismatched object/image point counts")
        if np.abs(obj_view[:, 2]).max() > 1e-6:
            raise ValueError(f"view {index}: the calibration pattern must be planar (Z = 0)")

    homographies = [_estimate_homography_dlt(obj_view[:, :2], img_view) for obj_view, img_view in zip(object_views, image_views)]
    intrinsic = _intrinsics_from_homographies(homographies)
    rotations, translations = _extrinsics_from_homographies(homographies, intrinsic)

    intrinsic, rotations, translations, rms_error = _refine_with_bundle_adjustment(
        intrinsic, rotations, translations, object_views, image_views
    )

    rotation_vectors = [_matrix_to_rodrigues(rotation) for rotation in rotations]
    return CalibrationResult(
        camera_matrix=intrinsic,
        distortion_coefficients=np.zeros(5),
        rotation_vectors=rotation_vectors,
        translation_vectors=[np.asarray(translation) for translation in translations],
        rms_reprojection_error=float(rms_error),
        image_size=(int(image_size[0]), int(image_size[1])),
        calibration_flags=0,
        object_points=[view.copy() for view in object_views],
        image_points=[view.copy() for view in image_views],
    )


def solve_projection_matrix(object_points: np.ndarray, image_points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Estimate a camera projection matrix from 3D-2D correspondences.

    Solves the 3x4 projection matrix P with the DLT (requires at least six
    non-coplanar points), then decomposes P = K [R | t] via RQ factorization.

    Args:
        object_points: (N, 3) world points
        image_points: (N, 2) image coordinates in pixels

    Returns:
        (K, R, t): 3x3 intrinsic matrix with unit bottom-right entry,
        orthonormal 3x3 rotation, and (3,) translation.

    Raises:
        ValueError: Fewer than 6 correspondences or a degenerate solution.
    """
    object_points = np.asarray(object_points, dtype=np.float64).reshape(-1, 3)
    image_points = np.asarray(image_points, dtype=np.float64).reshape(-1, 2)
    if len(object_points) < _MIN_POINTS_FOR_PROJECTION:
        raise ValueError(f"projection estimation requires at least {_MIN_POINTS_FOR_PROJECTION} points")
    if len(object_points) != len(image_points):
        raise ValueError("object_points and image_points must have the same length")

    equations = np.zeros((2 * len(object_points), 12))
    for i, ((x, y, z), (u, v)) in enumerate(zip(object_points, image_points)):
        equations[2 * i] = [x, y, z, 1.0, 0.0, 0.0, 0.0, 0.0, -u * x, -u * y, -u * z, -u]
        equations[2 * i + 1] = [0.0, 0.0, 0.0, 0.0, x, y, z, 1.0, -v * x, -v * y, -v * z, -v]

    _, _, vh = np.linalg.svd(equations)
    projection = vh[-1].reshape(3, 4)

    block = projection[:, :3]
    if np.linalg.det(block) < 0:
        projection = -projection
        block = -block

    upper, orthogonal = rq(block)
    sign_fix = np.diag(np.sign(np.diag(upper)))
    intrinsic = upper @ sign_fix
    rotation = sign_fix @ orthogonal
    if np.linalg.det(rotation) < 0:
        rotation = -rotation

    scale = intrinsic[2, 2]
    if not np.isfinite(scale) or abs(scale) < 1e-12:
        raise ValueError("degenerate projection matrix: cannot normalize intrinsics")
    translation = np.linalg.solve(intrinsic, projection[:, 3])
    intrinsic = intrinsic / scale
    return intrinsic, rotation, translation


# ----------------------------------------------------------------------
# Internal helpers
# ----------------------------------------------------------------------


def _normalize_2d(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Hartley normalization: centroid at origin, mean distance sqrt(2)."""
    centroid = points.mean(axis=0)
    mean_distance = np.sqrt(((points - centroid) ** 2).sum(axis=1)).mean()
    scale = np.sqrt(2.0) / max(mean_distance, 1e-12)
    transform = np.array([[scale, 0.0, -scale * centroid[0]], [0.0, scale, -scale * centroid[1]], [0.0, 0.0, 1.0]])
    normalized = (transform @ np.column_stack([points, np.ones(len(points))]).T).T
    return normalized[:, :2], transform


def _estimate_homography_dlt(pattern_xy: np.ndarray, image_points: np.ndarray) -> np.ndarray:
    """Normalized DLT homography from >= 4 planar correspondences."""
    src_norm, src_transform = _normalize_2d(pattern_xy)
    dst_norm, dst_transform = _normalize_2d(image_points)

    equations = np.zeros((2 * len(src_norm), 9))
    for i, ((x, y), (u, v)) in enumerate(zip(src_norm, dst_norm)):
        equations[2 * i] = [-x, -y, -1.0, 0.0, 0.0, 0.0, u * x, u * y, u]
        equations[2 * i + 1] = [0.0, 0.0, 0.0, -x, -y, -1.0, v * x, v * y, v]

    _, _, vh = np.linalg.svd(equations)
    homography = vh[-1].reshape(3, 3)
    return np.linalg.inv(dst_transform) @ homography @ src_transform


def _zhang_vector(h_i: np.ndarray, h_j: np.ndarray) -> np.ndarray:
    """One row of the Zhang constraint system for column pair (i, j)."""
    return np.array(
        [
            h_i[0] * h_j[0],
            h_i[0] * h_j[1] + h_i[1] * h_j[0],
            h_i[1] * h_j[1],
            h_i[2] * h_j[0] + h_i[0] * h_j[2],
            h_i[2] * h_j[1] + h_i[1] * h_j[2],
            h_i[2] * h_j[2],
        ]
    )


def _intrinsics_from_homographies(homographies: list[np.ndarray]) -> np.ndarray:
    """Closed-form intrinsic extraction from per-view homographies."""
    constraints = np.zeros((2 * len(homographies), 6))
    for i, homography in enumerate(homographies):
        h_1, h_2 = homography[:, 0], homography[:, 1]
        constraints[2 * i] = _zhang_vector(h_1, h_2)
        constraints[2 * i + 1] = _zhang_vector(h_1, h_1) - _zhang_vector(h_2, h_2)

    if np.linalg.matrix_rank(constraints) < 5:
        raise ValueError("degenerate calibration configuration: views do not constrain the intrinsics")

    _, _, vh = np.linalg.svd(constraints)
    b = vh[-1]
    if b[0] <= 0:  # null-vector direction is arbitrary; fix b_11 > 0
        b = -b
    b_11, b_12, b_22, b_13, b_23, b_33 = b

    weight = b_11 * b_22 - b_12**2
    if abs(weight) < 1e-12 or b_11 <= 0:
        raise ValueError("degenerate calibration configuration: cannot extract intrinsics")
    v0 = (b_12 * b_13 - b_11 * b_23) / weight
    scale = b_33 - (b_13**2 + v0 * (b_12 * b_13 - b_11 * b_23)) / b_11
    if scale <= 0:
        raise ValueError("degenerate calibration configuration: non-positive intrinsic scale")

    fx = float(np.sqrt(scale / b_11))
    fy = float(np.sqrt(scale * b_11 / weight))
    skew = float(-b_12 * fx**2 * fy / scale)
    cx = float(skew * v0 / fy - b_13 * fx**2 / scale)
    intrinsic = np.array([[fx, skew, cx], [0.0, fy, v0], [0.0, 0.0, 1.0]])
    if not np.all(np.isfinite(intrinsic)):
        raise ValueError("degenerate calibration configuration: non-finite intrinsics")
    return intrinsic


def _extrinsics_from_homographies(
    homographies: list[np.ndarray], intrinsic: np.ndarray
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Per-view rotation and translation from homographies and intrinsics."""
    intrinsic_inv = np.linalg.inv(intrinsic)
    rotations, translations = [], []
    for homography in homographies:
        h_1, h_2, h_3 = homography[:, 0], homography[:, 1], homography[:, 2]
        scale = 1.0 / np.linalg.norm(intrinsic_inv @ h_1)
        if float((scale * intrinsic_inv @ h_3)[2]) < 0:
            scale = -scale
        r_1 = scale * intrinsic_inv @ h_1
        r_2 = scale * intrinsic_inv @ h_2
        r_3 = np.cross(r_1, r_2)
        rotation = np.stack([r_1, r_2, r_3], axis=1)
        u_mat, _, v_t = np.linalg.svd(rotation)
        rotation = u_mat @ np.diag([1.0, 1.0, np.linalg.det(u_mat @ v_t)]) @ v_t
        rotations.append(rotation)
        translations.append(scale * intrinsic_inv @ h_3)
    return rotations, translations


def _rodrigues_to_matrix(rotation_vector: np.ndarray) -> np.ndarray:
    """Convert a Rodrigues rotation vector to a 3x3 rotation matrix."""
    theta = float(np.linalg.norm(rotation_vector))
    if theta < 1e-12:
        return np.eye(3)
    axis = rotation_vector / theta
    skew = np.array([[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]])
    return np.eye(3) + np.sin(theta) * skew + (1.0 - np.cos(theta)) * (skew @ skew)


def _matrix_to_rodrigues(rotation: np.ndarray) -> np.ndarray:
    """Convert a 3x3 rotation matrix to a Rodrigues rotation vector."""
    cos_theta = float(np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0))
    theta = float(np.arccos(cos_theta))
    if theta < 1e-9:
        return np.zeros(3)
    factor = theta / (2.0 * np.sin(theta))
    return factor * np.array([rotation[2, 1] - rotation[1, 2], rotation[0, 2] - rotation[2, 0], rotation[1, 0] - rotation[0, 1]])


def _project_points(pattern_xy: np.ndarray, rotation: np.ndarray, translation: np.ndarray, intrinsic: np.ndarray) -> np.ndarray:
    """Project planar pattern points through the pinhole model."""
    points_3d = np.column_stack([pattern_xy, np.zeros(len(pattern_xy))])
    camera_space = points_3d @ rotation.T + translation
    normalized = camera_space[:, :2] / camera_space[:, 2:3]
    return normalized @ intrinsic[:2, :2].T + intrinsic[:2, 2]


def _bundle_residuals(params: np.ndarray, object_views: list[np.ndarray], image_views: list[np.ndarray]) -> np.ndarray:
    """Stacked (u, v) reprojection residuals for all views."""
    intrinsic = np.array([[params[0], 0.0, params[2]], [0.0, params[1], params[3]], [0.0, 0.0, 1.0]])
    residuals = []
    offset = 4
    for obj_view, img_view in zip(object_views, image_views):
        rotation = _rodrigues_to_matrix(params[offset : offset + 3])
        translation = params[offset + 3 : offset + 6]
        offset += 6
        projected = _project_points(obj_view[:, :2], rotation, translation, intrinsic)
        residuals.append((projected - img_view).reshape(-1))
    return np.concatenate(residuals)


def _refine_with_bundle_adjustment(
    intrinsic: np.ndarray,
    rotations: list[np.ndarray],
    translations: list[np.ndarray],
    object_views: list[np.ndarray],
    image_views: list[np.ndarray],
) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray], float]:
    """Refine intrinsics and extrinsics by minimizing reprojection error."""
    params = np.concatenate(
        [[intrinsic[0, 0], intrinsic[1, 1], intrinsic[0, 2], intrinsic[1, 2]]]
        + [
            np.concatenate([_matrix_to_rodrigues(rotation), translation])
            for rotation, translation in zip(rotations, translations)
        ]
    )

    def residuals_fn(params_vec: np.ndarray) -> np.ndarray:
        return _bundle_residuals(params_vec, object_views, image_views)

    refined = least_squares(residuals_fn, params, method="lm")
    if not np.all(np.isfinite(refined.x)) or refined.x[0] <= 0 or refined.x[1] <= 0:
        logger.warning("Bundle adjustment diverged; keeping closed-form calibration estimate")
        refined_params = params
    else:
        refined_params = refined.x

    refined_intrinsic = np.array(
        [[refined_params[0], 0.0, refined_params[2]], [0.0, refined_params[1], refined_params[3]], [0.0, 0.0, 1.0]]
    )
    refined_rotations, refined_translations = [], []
    offset = 4
    for _ in object_views:
        refined_rotations.append(_rodrigues_to_matrix(refined_params[offset : offset + 3]))
        refined_translations.append(refined_params[offset + 3 : offset + 6])
        offset += 6

    final_residuals = _bundle_residuals(refined_params, object_views, image_views)
    num_points = len(final_residuals) // 2
    rms_error = float(np.sqrt(np.sum(final_residuals**2) / num_points))
    return refined_intrinsic, refined_rotations, refined_translations, rms_error
