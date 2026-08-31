"""
Unit Tests for the Native Photogrammetry Calibration Solver

Covers Zhang-style camera calibration without OpenCV (intrinsic recovery
from synthetic views, validation failures, degenerate configurations), the
standalone DLT projection-matrix solver, and the multi-sensor sync dispatch
via SyncConfiguration.prefer_native_calibration.

Usage:
    Run these tests using pytest:
    $ pytest tests/test_calibration_solver.py
"""

import numpy as np
import pytest

import advanced_image_sensor_interface.sensor_interface.multi_sensor_sync as sync_module
from advanced_image_sensor_interface.sensor_interface.calibration.models import CalibrationResult
from advanced_image_sensor_interface.sensor_interface.calibration.photogrammetry import calibrate_camera, solve_projection_matrix
from advanced_image_sensor_interface.sensor_interface.multi_sensor_sync import MultiSensorSynchronizer, SyncConfiguration

IMAGE_SIZE = (640, 480)  # (width, height)
TRUE_K = np.array([[800.0, 0.0, 320.0], [0.0, 800.0, 240.0], [0.0, 0.0, 1.0]])
PATTERN = (9, 6)
SQUARE_MM = 25.0


def _pattern_object_points() -> np.ndarray:
    """Centered planar checkerboard corners in millimeters, shape (N, 3)."""
    points = np.zeros((PATTERN[0] * PATTERN[1], 3), dtype=np.float64)
    points[:, :2] = np.mgrid[0 : PATTERN[0], 0 : PATTERN[1]].T.reshape(-1, 2) * SQUARE_MM
    points[:, :2] -= points[:, :2].mean(axis=0)
    return points


def _rotation_matrix(rx: float, ry: float, rz: float) -> np.ndarray:
    """Compose small rotations about X, Y, and Z axes."""
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    rot_x = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]])
    rot_y = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
    rot_z = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])
    return rot_z @ rot_y @ rot_x


def _project(points: np.ndarray, rotation: np.ndarray, translation: np.ndarray, noise_std: float, rng) -> np.ndarray:
    """Project 3D points through the true pinhole camera with optional noise."""
    camera_space = points @ rotation.T + translation
    normalized = camera_space[:, :2] / camera_space[:, 2:3]
    image = normalized @ TRUE_K[:2, :2].T + TRUE_K[:2, 2]
    if noise_std > 0:
        image = image + rng.normal(0.0, noise_std, image.shape)
    return image


def _random_pose(rng) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic random pose keeping the pattern in front of the camera."""
    rotation = _rotation_matrix(*rng.uniform(-0.25, 0.25, size=3))
    translation = np.array([rng.uniform(-40.0, 40.0), rng.uniform(-30.0, 30.0), rng.uniform(350.0, 550.0)])
    return rotation, translation


def make_views(noise_std: float = 0.5, n_views: int = 12, seed: int = 123) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Synthetic calibration views of the pattern under distinct poses."""
    rng = np.random.default_rng(seed)
    obj_points = _pattern_object_points()
    object_views, image_views = [], []
    for _ in range(n_views):
        rotation, translation = _random_pose(rng)
        image_points = _project(obj_points, rotation, translation, noise_std, rng)
        object_views.append(obj_points.copy())
        image_views.append(image_points.astype(np.float32))
    return object_views, image_views


class TestCalibrateCamera:
    """Zhang-style intrinsic calibration from synthetic views."""

    def test_recovers_intrinsics(self):
        object_views, image_views = make_views(noise_std=0.5)

        result = calibrate_camera(object_views, image_views, IMAGE_SIZE)

        assert isinstance(result, CalibrationResult)
        intrinsic = result.camera_matrix
        assert intrinsic.shape == (3, 3)
        assert intrinsic[0, 0] == pytest.approx(800.0, rel=0.02)
        assert intrinsic[1, 1] == pytest.approx(800.0, rel=0.02)
        assert abs(intrinsic[0, 2] - 320.0) <= 5.0
        assert abs(intrinsic[1, 2] - 240.0) <= 5.0
        assert result.rms_reprojection_error < 1.0
        assert len(result.rotation_vectors) == 12
        assert len(result.translation_vectors) == 12
        assert result.image_size == IMAGE_SIZE

    def test_clean_data_has_near_zero_reprojection_error(self):
        object_views, image_views = make_views(noise_std=0.0)

        result = calibrate_camera(object_views, image_views, IMAGE_SIZE)

        assert result.rms_reprojection_error < 1e-4

    def test_distortion_coefficients_are_zeros(self):
        object_views, image_views = make_views()

        result = calibrate_camera(object_views, image_views, IMAGE_SIZE)

        assert len(result.distortion_coefficients) >= 4
        assert np.all(result.distortion_coefficients == 0.0)

    def test_requires_at_least_three_views(self):
        object_views, image_views = make_views(n_views=2)

        with pytest.raises(ValueError, match="3 views"):
            calibrate_camera(object_views, image_views, IMAGE_SIZE)

    def test_requires_at_least_four_points_per_view(self):
        object_views, image_views = make_views()
        object_views[0] = object_views[0][:3]
        image_views[0] = image_views[0][:3]

        with pytest.raises(ValueError, match="4 points"):
            calibrate_camera(object_views, image_views, IMAGE_SIZE)

    def test_view_lists_must_match(self):
        object_views, image_views = make_views()

        with pytest.raises(ValueError):
            calibrate_camera(object_views[:-1], image_views, IMAGE_SIZE)

    def test_degenerate_identical_poses_rejected(self):
        rng = np.random.default_rng(5)
        obj_points = _pattern_object_points()
        rotation, translation = _random_pose(rng)
        image_points = _project(obj_points, rotation, translation, 0.0, rng).astype(np.float32)

        object_views = [obj_points.copy() for _ in range(5)]
        image_views = [image_points.copy() for _ in range(5)]

        with pytest.raises(ValueError, match="degenerate"):
            calibrate_camera(object_views, image_views, IMAGE_SIZE)


class TestSolveProjectionMatrix:
    """DLT projection-matrix estimation and decomposition."""

    def test_round_trip_recovers_camera(self):
        rng = np.random.default_rng(21)
        points = rng.uniform(-50.0, 50.0, size=(12, 3))
        points[:, 2] += 400.0
        rotation = _rotation_matrix(0.1, -0.15, 0.05)
        translation = np.array([5.0, -8.0, 30.0])
        image_points = _project(points, rotation, translation, 0.0, rng)

        intrinsic, recovered_rotation, recovered_translation = solve_projection_matrix(points, image_points)

        np.testing.assert_allclose(intrinsic, TRUE_K, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(recovered_rotation, rotation, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(recovered_translation, translation, rtol=1e-6, atol=1e-6)

    def test_requires_at_least_six_points(self):
        rng = np.random.default_rng(21)
        points = rng.uniform(-50.0, 50.0, size=(5, 3))
        image_points = _project(points, np.eye(3), np.array([0.0, 0.0, 300.0]), 0.0, rng)

        with pytest.raises(ValueError, match="6"):
            solve_projection_matrix(points, image_points)


class TestMultiSensorSyncDispatch:
    """calibrate_sensors uses the native solver when prefer_native_calibration is set."""

    @staticmethod
    def _stub_cv2(corners_per_frame):
        class _StubCv2:
            TERM_CRITERIA_EPS = 1
            TERM_CRITERIA_MAX_ITER = 2
            COLOR_BGR2GRAY = 0

            def __init__(self):
                self._index = 0
                self.calibrate_camera_calls = 0

            def cvtColor(self, frame, code):
                return frame.mean(axis=2).astype(np.uint8)

            def cornerSubPix(self, gray, corners, win_size, zero_zone, criteria):
                return corners

            def findChessboardCorners(self, gray, pattern_size, flags):
                corners = corners_per_frame[self._index % len(corners_per_frame)]
                self._index += 1
                return True, corners

            def calibrateCamera(self, *_args):
                self.calibrate_camera_calls += 1
                raise AssertionError("cv2.calibrateCamera must not run when prefer_native_calibration is set")

        return _StubCv2()

    def test_native_solver_dispatched(self, monkeypatch):
        rng = np.random.default_rng(7)
        obj_points = _pattern_object_points()
        corners_per_frame = []
        for _ in range(5):
            rotation, translation = _random_pose(rng)
            image_points = _project(obj_points, rotation, translation, 0.2, rng)
            corners_per_frame.append(image_points.reshape(-1, 1, 2).astype(np.float32))

        stub = self._stub_cv2(corners_per_frame)
        monkeypatch.setattr(sync_module, "cv2", stub)

        config = SyncConfiguration(enable_geometric_calibration=True, prefer_native_calibration=True)
        synchronizer = MultiSensorSynchronizer(config)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        monkeypatch.setattr(synchronizer, "_capture_from_sensor", lambda sensor_id, trigger_time=None: (frame, 0.0))

        assert synchronizer.calibrate_sensors() is True

        intrinsic = synchronizer.sensors[0].calibration_matrix
        assert intrinsic is not None
        assert intrinsic[0, 0] == pytest.approx(800.0, rel=0.02)
        assert intrinsic[1, 1] == pytest.approx(800.0, rel=0.02)
        assert stub.calibrate_camera_calls == 0
