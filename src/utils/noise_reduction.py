"""
Noise Reduction Utilities

This module provides advanced noise reduction algorithms for image processing,
specifically tailored for CMOS image sensor data.

Functions:
    reduce_noise: Apply noise reduction using a bilateral filter.
    adaptive_noise_reduction: Apply adaptive noise reduction based on local variance.

Usage:
    from utils.noise_reduction import reduce_noise, adaptive_noise_reduction
    denoised_image = reduce_noise(noisy_image)
"""

import logging

import numpy as np
from scipy import ndimage

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def reduce_noise(image: np.ndarray, sigma_spatial: float = 1.0, sigma_range: float = 0.1) -> np.ndarray:
    """
    Apply noise reduction to an image using a bilateral filter.

    Args:
    ----
        image (np.ndarray): Input image (2D or 3D numpy array).
        sigma_spatial (float): Standard deviation for spatial kernel. Default is 1.0.
        sigma_range (float): Standard deviation for range kernel. Default is 0.1.

    Returns:
    -------
        np.ndarray: Denoised image.

    Raises:
    ------
        ValueError: If the input image is not 2D or 3D.

    """
    if image.ndim not in (2, 3):
        raise ValueError("Input image must be 2D or 3D")

    logger.info(f"Applying noise reduction with sigma_spatial={sigma_spatial}, sigma_range={sigma_range}")

    # Normalize image to [0, 1] range
    image_norm = image.astype(float) / np.max(image)

    # Apply bilateral filter
    if image.ndim == 2:
        denoised = _bilateral_filter(image_norm, sigma_spatial, sigma_range)
    else:
        denoised = np.dstack([_bilateral_filter(image_norm[..., i], sigma_spatial, sigma_range)
                              for i in range(image.shape[2])])

    # Rescale back to original range
    denoised = (denoised * np.max(image)).astype(image.dtype)

    logger.info("Noise reduction completed successfully")
    return denoised

def _bilateral_filter(image: np.ndarray, sigma_spatial: float, sigma_range: float) -> np.ndarray:
    """
    Apply bilateral filter to a 2D image.

    Args:
    ----
        image (np.ndarray): Input 2D image.
        sigma_spatial (float): Standard deviation for spatial kernel.
        sigma_range (float): Standard deviation for range kernel.

    Returns:
    -------
        np.ndarray: Filtered image.

    """
    # Create spatial kernel
    kernel_size = int(4 * sigma_spatial + 1)
    x, y = np.mgrid[-kernel_size:kernel_size+1, -kernel_size:kernel_size+1]
    spatial_kernel = np.exp(-(x**2 + y**2) / (2 * sigma_spatial**2))

    # Apply filter
    output = np.zeros_like(image)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            window = image[max(i-kernel_size,0):min(i+kernel_size+1, image.shape[0]),
                           max(j-kernel_size,0):min(j+kernel_size+1, image.shape[1])]
            range_kernel = np.exp(-((window - image[i, j])**2) / (2 * sigma_range**2))
            kernel = spatial_kernel[:window.shape[0], :window.shape[1]] * range_kernel
            output[i, j] = np.sum(window * kernel) / np.sum(kernel)

    return output

def adaptive_noise_reduction(image: np.ndarray, window_size: int = 5, k: float = 0.1) -> np.ndarray:
    """
    Apply adaptive noise reduction based on local variance.

    Args:
    ----
        image (np.ndarray): Input image (2D or 3D numpy array).
        window_size (int): Size of the local window for variance calculation. Default is 5.
        k (float): Adaptation parameter. Default is 0.1.

    Returns:
    -------
        np.ndarray: Denoised image.

    Raises:
    ------
        ValueError: If the input image is not 2D or 3D.

    """
    if image.ndim not in (2, 3):
        raise ValueError("Input image must be 2D or 3D")

    logger.info(f"Applying adaptive noise reduction with window_size={window_size}, k={k}")

    # Normalize image to [0, 1] range
    image_norm = image.astype(float) / np.max(image)

    # Calculate local mean and variance
    local_mean = ndimage.uniform_filter(image_norm, size=window_size)
    local_var = ndimage.uniform_filter(image_norm**2, size=window_size) - local_mean**2

    # Estimate noise variance (assumes noise is Gaussian)
    noise_var = np.mean(local_var)

    # Calculate adaptive factor
    adaptive_factor = (local_var - noise_var) / (local_var + k*noise_var)
    adaptive_factor = np.clip(adaptive_factor, 0, 1)

    # Apply adaptive filtering
    denoised = local_mean + adaptive_factor * (image_norm - local_mean)

    # Rescale back to original range
    denoised = (denoised * np.max(image)).astype(image.dtype)

    logger.info("Adaptive noise reduction completed successfully")
    return denoised

# Example usage and testing
if __name__ == "__main__":
    # Generate a noisy test image
    np.random.seed(0)
    test_image = np.random.rand(100, 100)
    noisy_image = test_image + 0.1 * np.random.randn(100, 100)

    # Apply noise reduction
    denoised_image = reduce_noise(noisy_image)
    adaptive_denoised_image = adaptive_noise_reduction(noisy_image)

    # Calculate and print MSE for both methods
    mse_bilateral = np.mean((test_image - denoised_image)**2)
    mse_adaptive = np.mean((test_image - adaptive_denoised_image)**2)
    logger.info(f"Mean Squared Error (Bilateral): {mse_bilateral:.6f}")
    logger.info(f"Mean Squared Error (Adaptive): {mse_adaptive:.6f}")

    # Test with 3D image
    test_image_3d = np.random.rand(100, 100, 3)
    noisy_image_3d = test_image_3d + 0.1 * np.random.randn(100, 100, 3)
    denoised_image_3d = reduce_noise(noisy_image_3d)
    adaptive_denoised_image_3d = adaptive_noise_reduction(noisy_image_3d)

    logger.info("All tests completed successfully")
