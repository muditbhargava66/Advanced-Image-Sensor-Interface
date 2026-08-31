"""
Unit Tests for the v3.2.0 3D/Depth Module

Covers StereoDepthProcessor disparity estimation (Block Matching, SGM-lite,
ORB alignment with graceful fallback), disparity-to-depth conversion, point
cloud generation, PLY export round-trips, and input validation.

Usage:
    Run these tests using pytest:
    $ pytest tests/test_depth_module.py
"""

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

import advanced_image_sensor_interface.utils.depth as depth_module
from advanced_image_sensor_interface import (
    DepthConfig,
    DisparityAlgorithm,
    StereoDepthProcessor,
)

TRUE_DISPARITY = 7
FOCAL_LENGTH_PX = 800.0
BASELINE_M = 0.12


@pytest.fixture
def stereo_pair() -> tuple[np.ndarray, np.ndarray]:
    """Smooth-texture stereo pair where right is left shifted left by TRUE_DISPARITY."""
    rng = np.random.default_rng(42)
    height, width = 96, 160
    left = gaussian_filter(rng.random((height, width)) * 255.0, sigma=2.0)
    right = np.roll(left, -TRUE_DISPARITY, axis=1)
    return left.astype(np.float32), right.astype(np.float32)


def interior_mask(shape: tuple[int, int], disparity: int, margin: int) -> np.ndarray:
    """Mask of pixels far from the image border and the wrapped roll columns."""
    height, width = shape
    mask = np.zeros((height, width), dtype=bool)
    mask[margin : height - margin, disparity + margin : width - margin] = True
    return mask


class TestDepthConfig:
    """DepthConfig validation behavior."""

    def test_defaults_are_valid(self):
        config = DepthConfig()
        assert config.window_size == 15
        assert config.num_disparities == 64
        assert config.algorithm is DisparityAlgorithm.BLOCK_MATCHING

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"window_size": 4},  # even
            {"window_size": 1},  # too small
            {"num_disparities": 0},
            {"min_disparity": -1},
            {"sgm_p1": 0.0},
            {"sgm_p1": 40.0, "sgm_p2": 32.0},  # p2 must exceed p1
        ],
    )
    def test_invalid_config_rejected(self, kwargs):
        with pytest.raises(ValueError):
            DepthConfig(**kwargs)


class TestBlockMatching:
    """Block matching disparity on synthetic stereo pairs."""

    def test_recovers_known_disparity(self, stereo_pair):
        processor = StereoDepthProcessor(DepthConfig(algorithm=DisparityAlgorithm.BLOCK_MATCHING))
        result = processor.compute_disparity(*stereo_pair)

        assert result.success
        assert result.disparity_map is not None
        assert result.algorithm_used == DisparityAlgorithm.BLOCK_MATCHING.value
        assert result.metrics.processing_time_ms > 0

        mask = interior_mask(stereo_pair[0].shape, TRUE_DISPARITY, margin=16)
        error = np.abs(result.disparity_map[mask] - TRUE_DISPARITY)
        assert np.median(error) <= 1.0
        assert (error <= 2).mean() >= 0.95

    def test_color_input_supported(self, stereo_pair):
        left, right = stereo_pair
        left_color = np.stack([left] * 3, axis=-1).astype(np.uint8)
        right_color = np.stack([right] * 3, axis=-1).astype(np.uint8)

        processor = StereoDepthProcessor(DepthConfig(window_size=9))
        result = processor.compute_disparity(left_color, right_color)
        assert result.success
        assert result.disparity_map.shape == left.shape


class TestSgmLite:
    """SGM-lite (4-path aggregation) accuracy."""

    def test_at_least_as_accurate_as_block_matching(self, stereo_pair):
        mask = interior_mask(stereo_pair[0].shape, TRUE_DISPARITY, margin=16)

        bm_processor = StereoDepthProcessor(DepthConfig(algorithm=DisparityAlgorithm.BLOCK_MATCHING))
        bm_error = np.abs(bm_processor.compute_disparity(*stereo_pair).disparity_map[mask] - TRUE_DISPARITY)

        sgm_processor = StereoDepthProcessor(DepthConfig(algorithm=DisparityAlgorithm.SEMI_GLOBAL_MATCHING))
        sgm_result = sgm_processor.compute_disparity(*stereo_pair)
        assert sgm_result.success
        sgm_error = np.abs(sgm_result.disparity_map[mask] - TRUE_DISPARITY)

        assert np.median(sgm_error) <= np.median(bm_error) + 0.5
        assert (sgm_error <= 2).mean() >= (bm_error <= 2).mean() - 0.05


class TestOrbAlignment:
    """ORB alignment path and graceful fallback."""

    def test_falls_back_without_cv2(self, monkeypatch, stereo_pair):
        monkeypatch.setattr(depth_module, "CV2_AVAILABLE", False)
        processor = StereoDepthProcessor(DepthConfig(algorithm=DisparityAlgorithm.ORB_ALIGNMENT))

        result = processor.compute_disparity(*stereo_pair)

        assert result.success
        assert any("block matching" in warning for warning in result.warnings)
        assert result.algorithm_used == DisparityAlgorithm.BLOCK_MATCHING.value

    @pytest.mark.skipif(not depth_module.CV2_AVAILABLE, reason="OpenCV not installed")
    def test_orb_path_with_cv2(self, stereo_pair):
        processor = StereoDepthProcessor(DepthConfig(algorithm=DisparityAlgorithm.ORB_ALIGNMENT))
        result = processor.compute_disparity(*stereo_pair)
        assert result.success
        assert result.disparity_map is not None


class TestDisparityToDepth:
    """Pinhole stereo model Z = f * B / d."""

    def test_matches_pinhole_model(self):
        processor = StereoDepthProcessor()
        disparity = np.full((8, 8), TRUE_DISPARITY, dtype=np.float32)
        disparity[0, 0] = 0.0  # invalid pixel

        depth = processor.disparity_to_depth(disparity, FOCAL_LENGTH_PX, BASELINE_M)

        expected = FOCAL_LENGTH_PX * BASELINE_M / TRUE_DISPARITY
        assert depth[4, 4] == pytest.approx(expected)
        assert depth[0, 0] == 0.0

    def test_rejects_non_positive_geometry(self):
        processor = StereoDepthProcessor()
        disparity = np.ones((4, 4), dtype=np.float32)

        with pytest.raises(ValueError):
            processor.disparity_to_depth(disparity, focal_length_px=0.0, baseline_m=BASELINE_M)
        with pytest.raises(ValueError):
            processor.disparity_to_depth(disparity, focal_length_px=FOCAL_LENGTH_PX, baseline_m=-0.1)


class TestPointCloudAndExport:
    """Point cloud generation and PLY export round-trips."""

    def test_generate_point_cloud(self):
        processor = StereoDepthProcessor()
        depth = np.zeros((4, 6), dtype=np.float64)
        depth[1:3, 2:4] = 2.0

        point_cloud = processor.generate_point_cloud(depth, focal_length_px=100.0, baseline_m=0.1)

        assert point_cloud.shape == (4, 3)
        assert point_cloud.dtype == np.float32
        np.testing.assert_allclose(point_cloud[:, 2], 2.0, atol=1e-6)
        # Pixel (row=1, col=2) with default principal point (2.5, 1.5):
        # X = (2 - 2.5) * 2 / 100 = -0.01, Y = (1 - 1.5) * 2 / 100 = -0.01
        assert point_cloud[0, 0] == pytest.approx(-0.01, abs=1e-6)
        assert point_cloud[0, 1] == pytest.approx(-0.01, abs=1e-6)

    def test_generate_point_cloud_requires_2d(self):
        processor = StereoDepthProcessor()
        with pytest.raises(ValueError):
            processor.generate_point_cloud(np.ones((2, 2, 2)), focal_length_px=1.0, baseline_m=0.1)

    def test_export_ply_binary_round_trip(self, tmp_path):
        processor = StereoDepthProcessor()
        points = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
        path = tmp_path / "cloud.ply"

        assert processor.export_ply(points, path, binary=True)

        raw = path.read_bytes()
        header, _, payload = raw.partition(b"end_header\n")
        assert b"format binary_little_endian 1.0" in header
        assert b"element vertex 2" in header
        loaded = np.frombuffer(payload, dtype="<f4").reshape(-1, 3)
        np.testing.assert_allclose(loaded, points, atol=1e-6)

    def test_export_ply_ascii_round_trip(self, tmp_path):
        processor = StereoDepthProcessor()
        points = np.array([[0.5, -1.5, 2.25]], dtype=np.float32)
        path = tmp_path / "cloud_ascii.ply"

        assert processor.export_ply(points, path, binary=False)

        text = path.read_text(encoding="ascii")
        assert "format ascii 1.0" in text
        assert "element vertex 1" in text
        vertex_line = text.split("end_header\n", 1)[1].strip()
        x, y, z = (float(value) for value in vertex_line.split())
        assert (x, y, z) == pytest.approx((0.5, -1.5, 2.25), abs=1e-5)

    def test_export_ply_rejects_bad_shapes(self, tmp_path):
        processor = StereoDepthProcessor()
        with pytest.raises(ValueError):
            processor.export_ply(np.ones((4, 2), dtype=np.float32), tmp_path / "bad.ply")
        with pytest.raises(ValueError):
            processor.export_ply(np.zeros((0, 3), dtype=np.float32), tmp_path / "empty.ply")


class TestFullPipeline:
    """process_stereo_pair end-to-end."""

    def test_pipeline_with_export(self, stereo_pair, tmp_path):
        processor = StereoDepthProcessor(DepthConfig())
        export_path = tmp_path / "scene.ply"

        result = processor.process_stereo_pair(
            *stereo_pair,
            focal_length_px=FOCAL_LENGTH_PX,
            baseline_m=BASELINE_M,
            export_path=export_path,
        )

        assert result.success
        assert result.depth_map is not None
        assert result.point_cloud is not None
        assert export_path.exists()
        assert export_path.stat().st_size > 0

        valid_depth = result.depth_map[result.depth_map > 0]
        expected = FOCAL_LENGTH_PX * BASELINE_M / TRUE_DISPARITY
        assert np.median(valid_depth) == pytest.approx(expected, rel=0.05)

    def test_pipeline_propagates_failure(self):
        processor = StereoDepthProcessor(DepthConfig())
        result = processor.process_stereo_pair(
            np.zeros((8, 8), dtype=np.uint8),
            np.zeros((8, 10), dtype=np.uint8),
            focal_length_px=FOCAL_LENGTH_PX,
            baseline_m=BASELINE_M,
        )
        assert not result.success
        assert result.error


class TestInvalidInputs:
    """Invalid stereo inputs produce failed results, not exceptions."""

    def test_shape_mismatch(self):
        processor = StereoDepthProcessor(DepthConfig())
        result = processor.compute_disparity(np.zeros((10, 10)), np.zeros((10, 12)))
        assert not result.success
        assert result.error
        assert result.disparity_map is None

    def test_non_array_inputs(self):
        processor = StereoDepthProcessor(DepthConfig())
        result = processor.compute_disparity("left", "right")
        assert not result.success
        assert result.error

    def test_1d_input_rejected(self):
        processor = StereoDepthProcessor(DepthConfig())
        result = processor.compute_disparity(np.zeros(10), np.zeros(10))
        assert not result.success
        assert result.error
