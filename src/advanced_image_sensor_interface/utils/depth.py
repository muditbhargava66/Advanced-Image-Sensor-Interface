"""
3D/Depth Module

Provides stereo depth estimation utilities for image sensor applications.
Supports disparity estimation (Block Matching, full 8-path semi-global
matching, optional ORB-based alignment), disparity-to-depth conversion,
point cloud generation, and PLY export.

Key Features:
- Block Matching (BM) disparity via SAD cost volume with box-filter windowing
- Full 8-path Semi-Global Matching (4 cardinal + 4 diagonal scanline paths
  with P1/P2 smoothness penalties), optionally accelerated with numba when
  installed; a pure-numpy fallback keeps behavior identical without it
- Optional ORB feature alignment via OpenCV when available (graceful fallback)
- Disparity-to-depth conversion using the pinhole stereo model (Z = f * B / d)
- Point cloud generation and ASCII/binary PLY export (no external dependencies)

Note:
    This is a simulation-grade implementation. SGM aggregation runs the
    standard per-path recursion along eight directions but omits left-right
    consistency checks and sub-pixel refinement. ORB alignment uses a global
    shift derived from matched keypoints, not a full homography.

Author: Advanced Image Sensor Interface Team
Version: 3.2.0
"""

import logging
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional, Union

import numpy as np
from scipy.ndimage import uniform_filter

from ..types import DepthResult, ProcessingMetrics

try:
    import cv2

    CV2_AVAILABLE = True
except ImportError:
    cv2 = None
    CV2_AVAILABLE = False

try:
    from numba import njit

    NUMBA_AVAILABLE = True
except ImportError:
    njit = None
    NUMBA_AVAILABLE = False

logger = logging.getLogger(__name__)


class DisparityAlgorithm(Enum):
    """Stereo disparity estimation algorithms."""

    BLOCK_MATCHING = "block_matching"
    SEMI_GLOBAL_MATCHING = "semi_global_matching"
    ORB_ALIGNMENT = "orb_alignment"


@dataclass
class DepthConfig:
    """Configuration for stereo depth processing.

    Attributes:
        algorithm: Disparity estimation algorithm to use
        window_size: Odd-sized matching window (pixels) for cost aggregation
        num_disparities: Disparity search range size (pixels)
        min_disparity: Minimum disparity to search (usually 0)
        sgm_p1: SGM penalty for +/-1 disparity change along a path
        sgm_p2: SGM penalty for larger disparity changes along a path
        sgm_paths: Number of SGM aggregation paths (4 cardinal or 8 full)
        use_cv2_orb: Allow OpenCV ORB features when available
        use_numba: Prefer numba-accelerated SGM kernels when numba is installed
    """

    algorithm: DisparityAlgorithm = DisparityAlgorithm.BLOCK_MATCHING
    window_size: int = 15
    num_disparities: int = 64
    min_disparity: int = 0
    sgm_p1: float = 8.0
    sgm_p2: float = 32.0
    sgm_paths: int = 8
    use_cv2_orb: bool = True
    use_numba: bool = True

    def __post_init__(self):
        """Validate configuration."""
        if self.window_size < 3 or self.window_size % 2 == 0:
            raise ValueError("window_size must be an odd integer >= 3")
        if self.num_disparities < 1:
            raise ValueError("num_disparities must be >= 1")
        if self.min_disparity < 0:
            raise ValueError("min_disparity must be non-negative")
        if self.sgm_p1 <= 0 or self.sgm_p2 <= self.sgm_p1:
            raise ValueError("require 0 < sgm_p1 < sgm_p2")
        if self.sgm_paths not in (4, 8):
            raise ValueError("sgm_paths must be 4 (cardinal) or 8 (full)")


def _sgm_scan_numpy(  # noqa: PLR0917
    cost_volume: np.ndarray, p1: float, p2: float, reverse: bool, shift: int, aggregated: np.ndarray, scan_rows: bool
) -> None:
    """Aggregate one SGM path over the cost volume using pure numpy.

    Iterates slice-by-slice along the scan axis while carrying the previous
    slice's aggregated costs as state. ``shift`` displaces the predecessor
    along the batch axis (diagonal paths); predecessors outside the image
    start a new path. Accumulates the per-path costs into ``aggregated``.
    """
    num_disp, height, width = cost_volume.shape
    n_steps = height if scan_rows else width
    prev: Optional[np.ndarray] = None
    for step_idx in range(n_steps):
        s = (n_steps - 1 - step_idx) if reverse else step_idx
        current = (cost_volume[:, s, :] if scan_rows else cost_volume[:, :, s]).transpose().astype(np.float64)
        if prev is None:
            l_path = current
        else:
            if shift > 0:
                shifted = np.concatenate([np.full_like(prev[:1], np.inf), prev[:-1]], axis=0)
            elif shift < 0:
                shifted = np.concatenate([prev[1:], np.full_like(prev[-1:], np.inf)], axis=0)
            else:
                shifted = prev
            prev_min = shifted.min(axis=1, keepdims=True)
            finite = np.isfinite(prev_min[:, 0])
            l_path = current.copy()
            if finite.any():
                fin_shifted = shifted[finite]
                fin_prev_min = prev_min[finite]
                d_left = np.concatenate([np.full_like(fin_shifted[:, :1], np.inf), fin_shifted[:, :-1]], axis=1)
                d_right = np.concatenate([fin_shifted[:, 1:], np.full_like(fin_shifted[:, -1:], np.inf)], axis=1)
                candidates = np.stack(
                    [fin_shifted, d_left + p1, d_right + p1, np.broadcast_to(fin_prev_min + p2, fin_shifted.shape)]
                )
                l_path[finite] = current[finite] + candidates.min(axis=0) - fin_prev_min
        if scan_rows:
            aggregated[s, :, :] += l_path
        else:
            aggregated[:, s, :] += l_path
        prev = l_path


def _sgm_scan_rows_numpy(  # noqa: PLR0917
    cost_volume: np.ndarray, p1: float, p2: float, reverse: bool, shift: int, aggregated: np.ndarray
) -> None:
    """Row-sequential (vertical and diagonal) SGM path, numpy backend."""
    _sgm_scan_numpy(cost_volume, p1, p2, reverse, shift, aggregated, scan_rows=True)


def _sgm_scan_cols_numpy(  # noqa: PLR0917
    cost_volume: np.ndarray, p1: float, p2: float, reverse: bool, shift: int, aggregated: np.ndarray
) -> None:
    """Column-sequential (horizontal) SGM path, numpy backend."""
    _sgm_scan_numpy(cost_volume, p1, p2, reverse, shift, aggregated, scan_rows=False)


if NUMBA_AVAILABLE:

    @njit(cache=False)  # type: ignore[misc]
    def _sgm_scan_rows_numba(cost_volume, p1, p2, reverse, shift, aggregated):  # noqa: PLR0917
        """Row-sequential SGM path, numba backend (mirrors the numpy recursion)."""
        num_disp, height, width = cost_volume.shape
        prev = np.empty((width, num_disp), dtype=np.float64)
        new_l = np.empty((width, num_disp), dtype=np.float64)
        for step_idx in range(height):
            y = (height - 1 - step_idx) if reverse else step_idx
            for x in range(width):
                src = x - shift
                if step_idx == 0 or src < 0 or src >= width:
                    for d in range(num_disp):
                        new_l[x, d] = cost_volume[d, y, x]
                else:
                    prev_min = prev[src, 0]
                    for d in range(1, num_disp):
                        if prev[src, d] < prev_min:
                            prev_min = prev[src, d]
                    for d in range(num_disp):
                        best = prev[src, d]
                        if d > 0:
                            cand = prev[src, d - 1] + p1
                            if cand < best:
                                best = cand
                        if d < num_disp - 1:
                            cand = prev[src, d + 1] + p1
                            if cand < best:
                                best = cand
                        cand = prev_min + p2
                        if cand < best:
                            best = cand
                        new_l[x, d] = cost_volume[d, y, x] + best - prev_min
            for x in range(width):
                for d in range(num_disp):
                    aggregated[y, x, d] += new_l[x, d]
            prev, new_l = new_l, prev

    @njit(cache=False)  # type: ignore[misc]
    def _sgm_scan_cols_numba(cost_volume, p1, p2, reverse, shift, aggregated):  # noqa: PLR0917
        """Column-sequential SGM path, numba backend (mirrors the numpy recursion)."""
        num_disp, height, width = cost_volume.shape
        prev = np.empty((height, num_disp), dtype=np.float64)
        new_l = np.empty((height, num_disp), dtype=np.float64)
        for step_idx in range(width):
            x = (width - 1 - step_idx) if reverse else step_idx
            for y in range(height):
                src = y - shift
                if step_idx == 0 or src < 0 or src >= height:
                    for d in range(num_disp):
                        new_l[y, d] = cost_volume[d, y, x]
                else:
                    prev_min = prev[src, 0]
                    for d in range(1, num_disp):
                        if prev[src, d] < prev_min:
                            prev_min = prev[src, d]
                    for d in range(num_disp):
                        best = prev[src, d]
                        if d > 0:
                            cand = prev[src, d - 1] + p1
                            if cand < best:
                                best = cand
                        if d < num_disp - 1:
                            cand = prev[src, d + 1] + p1
                            if cand < best:
                                best = cand
                        cand = prev_min + p2
                        if cand < best:
                            best = cand
                        new_l[y, d] = cost_volume[d, y, x] + best - prev_min
            for y in range(height):
                for d in range(num_disp):
                    aggregated[y, x, d] += new_l[y, d]
            prev, new_l = new_l, prev

else:
    _sgm_scan_rows_numba = None
    _sgm_scan_cols_numba = None


class StereoDepthProcessor:
    """Stereo depth processor for disparity, depth, and point cloud generation.

    Example:
        >>> processor = StereoDepthProcessor(DepthConfig())
        >>> result = processor.compute_disparity(left_frame, right_frame)
        >>> if result.success:
        ...     depth = processor.disparity_to_depth(result.disparity_map, 800.0, 0.12)
    """

    def __init__(self, config: Optional[DepthConfig] = None):
        """Initialize the stereo depth processor.

        Args:
            config: Depth configuration; defaults to DepthConfig()
        """
        self.config = config or DepthConfig()

    def compute_disparity(self, left: np.ndarray, right: np.ndarray) -> DepthResult:
        """Compute a disparity map from a rectified stereo pair.

        Args:
            left: Left image (H, W) grayscale or (H, W, C) color
            right: Right image with the same shape as left

        Returns:
            DepthResult with disparity_map set on success. Pixels without a
            valid match hold disparity 0.
        """
        start = time.perf_counter()
        warnings: list[str] = []

        try:
            left_gray, right_gray = self._validate_stereo_pair(left, right)
        except (TypeError, ValueError) as exc:
            return DepthResult(success=False, error=str(exc), warnings=warnings)

        width = left_gray.shape[1]
        max_search = min(self.config.num_disparities, width - 1)
        if max_search < self.config.num_disparities:
            warnings.append(f"num_disparities clamped to image width: {max_search}")

        algorithm = self.config.algorithm
        orb_shift = 0
        try:
            if algorithm is DisparityAlgorithm.SEMI_GLOBAL_MATCHING:
                cost_volume = self._build_cost_volume(left_gray, right_gray, max_search)
                disparity = self._sgm_aggregate(cost_volume) + self.config.min_disparity
            elif algorithm is DisparityAlgorithm.ORB_ALIGNMENT:
                aligned, shift, orb_warning = self._orb_align(left_gray, right_gray)
                if orb_warning:
                    warnings.append(orb_warning)
                    algorithm = DisparityAlgorithm.BLOCK_MATCHING
                    disparity = self._block_matching(left_gray, right_gray, max_search)
                else:
                    orb_shift = shift
                    disparity = self._block_matching(left_gray, aligned, max_search) + shift
            else:
                disparity = self._block_matching(left_gray, right_gray, max_search)
        except Exception as exc:
            logger.exception("Disparity computation failed")
            return DepthResult(success=False, error=f"Disparity computation failed: {exc}", warnings=warnings)

        disparity = disparity.astype(np.float32)
        # Winner-take-all matches landing on min_disparity are treated as invalid.
        disparity[disparity <= self.config.min_disparity] = 0.0

        valid = disparity > 0
        valid_ratio = float(valid.mean()) if disparity.size else 0.0
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        metrics = ProcessingMetrics(
            processing_time_ms=elapsed_ms,
            algorithm_name=algorithm.value,
            parameters={
                "window_size": self.config.window_size,
                "num_disparities": max_search,
                "min_disparity": self.config.min_disparity,
                "orb_shift_px": orb_shift,
            },
        )
        return DepthResult(
            success=True,
            disparity_map=disparity,
            warnings=warnings,
            metrics=metrics,
            algorithm_used=algorithm.value,
            valid_pixel_ratio=valid_ratio,
        )

    def disparity_to_depth(self, disparity: np.ndarray, focal_length_px: float, baseline_m: float) -> np.ndarray:
        """Convert a disparity map to depth using the pinhole stereo model.

        Depth Z = focal_length_px * baseline_m / disparity. Invalid or
        non-positive disparities map to depth 0.

        Args:
            disparity: Disparity map in pixels
            focal_length_px: Focal length in pixels
            baseline_m: Stereo baseline in meters

        Returns:
            Depth map in meters (float64), 0 where disparity is invalid.
        """
        if focal_length_px <= 0:
            raise ValueError("focal_length_px must be positive")
        if baseline_m <= 0:
            raise ValueError("baseline_m must be positive")
        disparity = np.asarray(disparity, dtype=np.float64)
        depth = np.zeros_like(disparity)
        valid = disparity > 0
        depth[valid] = focal_length_px * baseline_m / disparity[valid]
        return depth

    def generate_point_cloud(
        self, depth: np.ndarray, focal_length_px: float, baseline_m: float, cx: Optional[float] = None, cy: Optional[float] = None
    ) -> np.ndarray:
        """Generate a 3D point cloud from a depth map.

        Args:
            depth: Depth map in meters (0 marks invalid pixels)
            focal_length_px: Focal length in pixels
            baseline_m: Stereo baseline in meters (kept for API symmetry and validation)
            cx: Principal point x (defaults to image center)
            cy: Principal point y (defaults to image center)

        Returns:
            (N, 3) float32 array of valid 3D points [X, Y, Z] in meters.
        """
        if focal_length_px <= 0:
            raise ValueError("focal_length_px must be positive")
        if baseline_m <= 0:
            raise ValueError("baseline_m must be positive")
        depth = np.asarray(depth, dtype=np.float64)
        if depth.ndim != 2:
            raise ValueError("depth must be a 2D array")

        height, width = depth.shape
        cx = (width - 1) / 2.0 if cx is None else cx
        cy = (height - 1) / 2.0 if cy is None else cy

        ys, xs = np.mgrid[0:height, 0:width]
        valid = depth > 0
        z = depth[valid]
        x = (xs[valid] - cx) * z / focal_length_px
        y = (ys[valid] - cy) * z / focal_length_px
        return np.column_stack([x, y, z]).astype(np.float32)

    def export_ply(self, point_cloud: np.ndarray, path: Union[str, Path], binary: bool = True) -> bool:
        """Export a point cloud to a PLY file.

        Args:
            point_cloud: (N, 3) array of points
            path: Output file path
            binary: Write binary little-endian PLY (True) or ASCII PLY (False)

        Returns:
            True when the file was written.
        """
        points = np.asarray(point_cloud, dtype=np.float32)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("point_cloud must have shape (N, 3)")
        if points.shape[0] == 0:
            raise ValueError("point_cloud must contain at least one point")

        fmt = "binary_little_endian 1.0" if binary else "ascii 1.0"
        header = (
            "ply\n"
            f"format {fmt}\n"
            f"element vertex {points.shape[0]}\n"
            "property float x\n"
            "property float y\n"
            "property float z\n"
            "end_header\n"
        )

        path = Path(path)
        if binary:
            with open(path, "wb") as f:
                f.write(header.encode("ascii"))
                f.write(points.astype("<f4").tobytes())
        else:
            with open(path, "w", encoding="ascii") as f:
                f.write(header)
                for x, y, z in points:
                    f.write(f"{x:.6f} {y:.6f} {z:.6f}\n")
        logger.debug("Exported %d points to %s", points.shape[0], path)
        return True

    def process_stereo_pair(  # noqa: PLR0917
        self,
        left: np.ndarray,
        right: np.ndarray,
        focal_length_px: float,
        baseline_m: float,
        export_path: Optional[Union[str, Path]] = None,
        binary_ply: bool = True,
    ) -> DepthResult:
        """Run the full stereo pipeline: disparity, depth, point cloud, optional PLY export.

        Args:
            left: Left image of the rectified stereo pair
            right: Right image of the rectified stereo pair
            focal_length_px: Focal length in pixels
            baseline_m: Stereo baseline in meters
            export_path: Optional PLY output path for the point cloud
            binary_ply: Write binary PLY when exporting

        Returns:
            DepthResult with disparity_map, depth_map, and point_cloud populated.
        """
        result = self.compute_disparity(left, right)
        if not result.success:
            return result

        try:
            depth = self.disparity_to_depth(result.disparity_map, focal_length_px, baseline_m)
            point_cloud = self.generate_point_cloud(depth, focal_length_px, baseline_m)
            if export_path is not None:
                self.export_ply(point_cloud, export_path, binary=binary_ply)
        except ValueError as exc:
            return DepthResult(success=False, error=str(exc), warnings=result.warnings)

        result.depth_map = depth
        result.point_cloud = point_cloud
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_stereo_pair(left: np.ndarray, right: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Validate and convert a stereo pair to float32 grayscale."""
        if not isinstance(left, np.ndarray) or not isinstance(right, np.ndarray):
            raise TypeError("left and right must be numpy arrays")
        if left.shape != right.shape:
            raise ValueError(f"left/right shape mismatch: {left.shape} != {right.shape}")
        if left.ndim not in (2, 3):
            raise ValueError("images must be 2D grayscale or 3D color arrays")

        def to_gray(img: np.ndarray) -> np.ndarray:
            if img.ndim == 3:
                return img.mean(axis=2).astype(np.float32)
            return img.astype(np.float32)

        return to_gray(left), to_gray(right)

    def _build_cost_volume(self, left: np.ndarray, right: np.ndarray, num_disp: int) -> np.ndarray:
        """Build the SAD cost volume, shape (num_disp, H, W)."""
        height, width = left.shape
        costs = np.empty((num_disp, height, width), dtype=np.float32)
        window = self.config.window_size
        for i, d in enumerate(range(self.config.min_disparity, self.config.min_disparity + num_disp)):
            if d == 0:
                sad = np.abs(left - right)
            else:
                sad = np.zeros((height, width), dtype=np.float32)
                sad[:, d:] = np.abs(left[:, d:] - right[:, : width - d])
            costs[i] = uniform_filter(sad, size=window, mode="constant") * (window * window)
        return costs

    def _block_matching(self, left: np.ndarray, right: np.ndarray, num_disp: int) -> np.ndarray:
        """Winner-take-all block matching on the SAD cost volume."""
        cost_volume = self._build_cost_volume(left, right, num_disp)
        return np.argmin(cost_volume, axis=0).astype(np.float32) + self.config.min_disparity

    def _sgm_aggregate(self, cost_volume: np.ndarray) -> np.ndarray:
        """Aggregate the cost volume along scanline paths (full SGM).

        Runs the standard SGM recursion with P1/P2 smoothness penalties along
        the four cardinal paths, plus the four diagonal paths when
        ``config.sgm_paths`` is 8. Uses numba kernels when installed and
        enabled; the pure-numpy fallback produces identical results.
        """
        num_disp, height, width = cost_volume.shape
        p1, p2 = self.config.sgm_p1, self.config.sgm_p2
        aggregated = np.zeros((height, width, num_disp), dtype=np.float64)

        use_numba = self.config.use_numba and NUMBA_AVAILABLE
        if use_numba:
            scan_rows, scan_cols = _sgm_scan_rows_numba, _sgm_scan_cols_numba
        else:
            logger.debug("SGM aggregation using numpy backend (numba %s)", "disabled" if NUMBA_AVAILABLE else "unavailable")
            scan_rows, scan_cols = _sgm_scan_rows_numpy, _sgm_scan_cols_numpy

        path_specs = [
            (scan_rows, False, 0),  # top -> bottom
            (scan_rows, True, 0),  # bottom -> top
            (scan_cols, False, 0),  # left -> right
            (scan_cols, True, 0),  # right -> left
        ]
        if self.config.sgm_paths == 8:
            path_specs.extend(
                [
                    (scan_rows, False, 1),  # top-left -> bottom-right
                    (scan_rows, False, -1),  # top-right -> bottom-left
                    (scan_rows, True, 1),  # bottom-left -> top-right
                    (scan_rows, True, -1),  # bottom-right -> top-left
                ]
            )
        for scan, reverse, shift in path_specs:
            scan(cost_volume, p1, p2, reverse, shift, aggregated)

        return np.argmin(aggregated, axis=2).astype(np.float32)

    def _orb_align(self, left: np.ndarray, right: np.ndarray) -> tuple[Optional[np.ndarray], int, Optional[str]]:
        """Align the right image to the left using ORB feature matching.

        Returns:
            (aligned_right, global_shift_px, warning). When alignment fails the
            warning explains why and the caller falls back to block matching.
        """
        if not CV2_AVAILABLE or not self.config.use_cv2_orb:
            return None, 0, "OpenCV unavailable; ORB alignment fell back to block matching"

        try:
            orb = cv2.ORB_create(nfeatures=500)
            kp_left, des_left = orb.detectAndCompute(left, None)
            kp_right, des_right = orb.detectAndCompute(right, None)
            if des_left is None or des_right is None or len(kp_left) < 8 or len(kp_right) < 8:
                return None, 0, "Insufficient ORB features; fell back to block matching"

            matches = cv2.BFMatcher(cv2.NORM_HAMMING).match(des_left, des_right)
            if len(matches) < 8:
                return None, 0, "Too few ORB matches; fell back to block matching"
            matches = sorted(matches, key=lambda m: m.distance)[: max(16, len(matches) // 2)]

            deltas = np.array([kp_left[m.queryIdx].pt[0] - kp_right[m.trainIdx].pt[0] for m in matches], dtype=np.float64)
            shift = round(float(np.median(deltas)))
            aligned = np.roll(right, shift, axis=1)
            return aligned, shift, None
        except Exception as exc:
            logger.debug("ORB alignment failed: %s", exc)
            return None, 0, f"ORB alignment failed ({exc}); fell back to block matching"
