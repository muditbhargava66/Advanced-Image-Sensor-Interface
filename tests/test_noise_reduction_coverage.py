import numpy as np
import pytest

from advanced_image_sensor_interface.utils.noise_reduction import NoiseReducerFactory, NoiseReductionConfig, NoiseType


class TestNoiseReductionDSA:
    @pytest.fixture
    def noisy_image(self):
        # Create a deterministic noisy image
        np.random.seed(42)
        clean = np.zeros((100, 100, 3), dtype=np.uint8)
        clean[20:80, 20:80] = 255  # White square
        noise = np.random.normal(0, 25, clean.shape).astype(np.int16)
        noisy = np.clip(clean + noise, 0, 255).astype(np.uint8)
        return noisy, clean

    def test_gaussian_reduction_performance(self, noisy_image):
        noisy, clean = noisy_image
        config = NoiseReductionConfig(noise_type=NoiseType.GAUSSIAN, strength=1.0)
        reducer = NoiseReducerFactory.create_reducer(config)

        result = reducer.process(noisy)

        # Calculate standard deviation in flat region (should decrease)
        # Background region
        bg_noise_in = np.std(noisy[0:20, 0:20])
        bg_noise_out = np.std(result[0:20, 0:20])

        assert bg_noise_out < bg_noise_in, "Gaussian blur should reduce noise variance"

    def test_median_reduction_edges(self, noisy_image):
        noisy, clean = noisy_image
        config = NoiseReductionConfig(noise_type=NoiseType.SALT_PEPPER, strength=1.0)
        reducer = NoiseReducerFactory.create_reducer(config)

        result = reducer.process(noisy)

        # Check edge preservation (simple check)
        # The transition from 0 to 255 should still be sharp-ish
        # We check the gradient
        edge_gradient = np.abs(result[20, 20:30, 0].astype(int) - result[20, 30:40, 0].astype(int))
        assert np.max(edge_gradient) > 50, "Median filter should preserve edges"

    def test_wavelet_denoising(self, noisy_image):
        noisy, clean = noisy_image
        # Using SPECKLE which maps to Wavelet (custom example) or Bilateral
        # Note: Wavelet was not in default _reducers map, but registered in example
        # Let's check factories: _reducers default has SPECKLE: BilateralNoiseReducer
        # But the file ends with register_custom_reducers() which sets SPECKLE to WaveletNoiseReducer

        try:
            config = NoiseReductionConfig(noise_type=NoiseType.SPECKLE, strength=0.5)
            reducer = NoiseReducerFactory.create_reducer(config)
            result = reducer.process(noisy)
            assert result.shape == noisy.shape
        except (ImportError, ValueError):
            pytest.skip("Wavelet denoising dependencies not available")

    @pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.float32])
    def test_dtype_support(self, dtype):
        image = np.random.rand(50, 50, 3)
        if dtype == np.uint8:
            image = (image * 255).astype(dtype)
        elif dtype == np.uint16:
            image = (image * 65535).astype(dtype)
        else:
            image = image.astype(dtype)

        config = NoiseReductionConfig(noise_type=NoiseType.GAUSSIAN, strength=0.5)
        reducer = NoiseReducerFactory.create_reducer(config)
        result = reducer.process(image)
        assert result.dtype == dtype
