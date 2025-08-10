"""GPU Acceleration Support for v2.0.0.

This module provides GPU acceleration capabilities for image processing including:
- CUDA-based processing (when available)
- OpenCL support
- Memory management for GPU operations
- Parallel processing pipelines
- Performance optimization
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Try to import GPU acceleration libraries
try:
    import numba
    from numba import cuda, jit

    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    logger.info("Numba not available - GPU acceleration limited")

try:
    import cupy as cp

    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    logger.info("CuPy not available - CUDA acceleration disabled")


class GPUBackend(Enum):
    """Available GPU backends."""

    NONE = "none"
    NUMBA_CUDA = "numba_cuda"
    CUPY = "cupy"
    OPENCL = "opencl"
    AUTO = "auto"


class ProcessingMode(Enum):
    """GPU processing modes."""

    CPU_ONLY = "cpu_only"
    GPU_ONLY = "gpu_only"
    HYBRID = "hybrid"
    AUTO = "auto"


@dataclass
class GPUConfiguration:
    """GPU acceleration configuration."""

    # Backend selection
    preferred_backend: GPUBackend = GPUBackend.AUTO
    fallback_to_cpu: bool = True

    # Memory management
    gpu_memory_limit_mb: Optional[int] = None
    enable_memory_pool: bool = True
    memory_pool_size_mb: int = 512

    # Processing configuration
    processing_mode: ProcessingMode = ProcessingMode.AUTO
    batch_size: int = 4
    num_streams: int = 2

    # Performance tuning
    enable_async_processing: bool = True
    prefetch_data: bool = True
    optimize_memory_access: bool = True

    def __post_init__(self):
        """Validate configuration."""
        if self.batch_size < 1:
            raise ValueError("Batch size must be at least 1")

        if self.num_streams < 1:
            raise ValueError("Number of streams must be at least 1")

        if self.memory_pool_size_mb < 64:
            raise ValueError("Memory pool size must be at least 64 MB")


class GPUAccelerator:
    """GPU acceleration manager."""

    def __init__(self, config: Optional[GPUConfiguration] = None):
        """Initialize GPU accelerator.

        Args:
            config: GPU configuration
        """
        self.config = config or GPUConfiguration()
        self.backend = GPUBackend.NONE
        self.device_info: dict[str, Any] = {}
        self.memory_pool = None
        self.is_initialized = False

        # Performance statistics
        self.stats = {
            "operations_processed": 0,
            "gpu_operations": 0,
            "cpu_operations": 0,
            "total_gpu_time": 0.0,
            "total_cpu_time": 0.0,
            "memory_transfers": 0,
            "memory_transfer_time": 0.0,
        }

        self._initialize_gpu()

    def _initialize_gpu(self) -> None:
        """Initialize GPU backend."""
        try:
            # Auto-detect best available backend
            if self.config.preferred_backend == GPUBackend.AUTO:
                self.backend = self._detect_best_backend()
            else:
                self.backend = self.config.preferred_backend

            # Initialize selected backend
            if self.backend == GPUBackend.CUPY and CUPY_AVAILABLE:
                self._initialize_cupy()
            elif self.backend == GPUBackend.NUMBA_CUDA and NUMBA_AVAILABLE:
                self._initialize_numba_cuda()
            else:
                self.backend = GPUBackend.NONE
                logger.info("GPU acceleration not available, using CPU only")

            self.is_initialized = True
            logger.info(f"GPU accelerator initialized with backend: {self.backend.value}")

        except Exception as e:
            logger.error(f"GPU initialization failed: {e}")
            self.backend = GPUBackend.NONE
            if not self.config.fallback_to_cpu:
                raise

    def _detect_best_backend(self) -> GPUBackend:
        """Detect the best available GPU backend."""
        # Check for CUDA availability
        if CUPY_AVAILABLE:
            try:
                cp.cuda.runtime.getDeviceCount()
                return GPUBackend.CUPY
            except Exception:
                pass

        if NUMBA_AVAILABLE:
            try:
                if cuda.is_available():
                    return GPUBackend.NUMBA_CUDA
            except Exception:
                pass

        return GPUBackend.NONE

    def _initialize_cupy(self) -> None:
        """Initialize CuPy backend."""
        try:
            # Get device information
            device_id = cp.cuda.Device().id
            device_props = cp.cuda.runtime.getDeviceProperties(device_id)

            self.device_info = {
                "name": device_props["name"].decode("utf-8"),
                "compute_capability": f"{device_props['major']}.{device_props['minor']}",
                "total_memory": device_props["totalGlobalMem"],
                "multiprocessors": device_props["multiProcessorCount"],
                "max_threads_per_block": device_props["maxThreadsPerBlock"],
            }

            # Initialize memory pool if enabled
            if self.config.enable_memory_pool:
                mempool = cp.get_default_memory_pool()
                mempool.set_limit(size=self.config.memory_pool_size_mb * 1024 * 1024)
                self.memory_pool = mempool

            logger.info(f"CuPy initialized on device: {self.device_info['name']}")

        except Exception as e:
            logger.error(f"CuPy initialization failed: {e}")
            raise

    def _initialize_numba_cuda(self) -> None:
        """Initialize Numba CUDA backend."""
        try:
            # Get device information
            device = cuda.get_current_device()

            self.device_info = {
                "name": device.name.decode("utf-8"),
                "compute_capability": f"{device.compute_capability[0]}.{device.compute_capability[1]}",
                "total_memory": device.memory_info.total,
                "multiprocessors": device.MULTIPROCESSOR_COUNT,
                "max_threads_per_block": device.MAX_THREADS_PER_BLOCK,
            }

            logger.info(f"Numba CUDA initialized on device: {self.device_info['name']}")

        except Exception as e:
            logger.error(f"Numba CUDA initialization failed: {e}")
            raise

    def process_image_batch(self, images: list[np.ndarray], operation: str, **kwargs) -> list[np.ndarray]:
        """Process a batch of images using GPU acceleration.

        Args:
            images: List of input images
            operation: Processing operation to perform
            **kwargs: Additional parameters for the operation

        Returns:
            List of processed images
        """
        import time

        start_time = time.time()

        try:
            if not self.is_initialized or self.backend == GPUBackend.NONE:
                result = self._process_batch_cpu_internal(images, operation, **kwargs)
                backend_used = GPUBackend.NONE
            elif self.backend == GPUBackend.CUPY:
                result = self._process_batch_cupy(images, operation, **kwargs)
                backend_used = GPUBackend.CUPY
            elif self.backend == GPUBackend.NUMBA_CUDA:
                result = self._process_batch_numba(images, operation, **kwargs)
                backend_used = GPUBackend.NUMBA_CUDA
            else:
                result = self._process_batch_cpu_internal(images, operation, **kwargs)
                backend_used = GPUBackend.NONE

            # Update statistics
            processing_time = time.time() - start_time
            self.stats["operations_processed"] += len(images)

            if backend_used != GPUBackend.NONE:
                self.stats["gpu_operations"] += len(images)
                self.stats["total_gpu_time"] += processing_time
            else:
                self.stats["cpu_operations"] += len(images)
                self.stats["total_cpu_time"] += processing_time

            return result

        except Exception as e:
            logger.error(f"GPU batch processing failed: {e}")
            if self.config.fallback_to_cpu:
                # Fallback to CPU without double-counting stats
                result = self._process_batch_cpu_internal(images, operation, **kwargs)
                processing_time = time.time() - start_time
                self.stats["operations_processed"] += len(images)
                self.stats["cpu_operations"] += len(images)
                self.stats["total_cpu_time"] += processing_time
                return result
            else:
                raise

    def _process_batch_cupy(self, images: list[np.ndarray], operation: str, **kwargs) -> list[np.ndarray]:
        """Process batch using CuPy."""
        import time

        transfer_start = time.time()

        # Transfer images to GPU
        gpu_images = [cp.asarray(img) for img in images]

        self.stats["memory_transfers"] += len(images)
        self.stats["memory_transfer_time"] += time.time() - transfer_start

        # Process on GPU
        processed_gpu = []
        for gpu_img in gpu_images:
            if operation == "gaussian_blur":
                processed = self._gpu_gaussian_blur_cupy(gpu_img, **kwargs)
            elif operation == "edge_detection":
                processed = self._gpu_edge_detection_cupy(gpu_img, **kwargs)
            elif operation == "histogram_equalization":
                processed = self._gpu_histogram_equalization_cupy(gpu_img, **kwargs)
            elif operation == "noise_reduction":
                processed = self._gpu_noise_reduction_cupy(gpu_img, **kwargs)
            else:
                logger.warning(f"Unknown GPU operation: {operation}")
                processed = gpu_img

            processed_gpu.append(processed)

        # Transfer results back to CPU
        transfer_start = time.time()
        result = [cp.asnumpy(gpu_img) for gpu_img in processed_gpu]
        self.stats["memory_transfer_time"] += time.time() - transfer_start

        return result

    def _process_batch_numba(self, images: list[np.ndarray], operation: str, **kwargs) -> list[np.ndarray]:
        """Process batch using Numba CUDA."""
        # For Numba CUDA, we'll use CPU processing with JIT compilation
        return self._process_batch_cpu_jit(images, operation, **kwargs)

    def _process_batch_cpu(self, images: list[np.ndarray], operation: str, **kwargs) -> list[np.ndarray]:
        """Process batch using CPU (with stats update for legacy compatibility)."""
        import time

        start_time = time.time()

        result = self._process_batch_cpu_internal(images, operation, **kwargs)

        processing_time = time.time() - start_time
        self.stats["cpu_operations"] += len(images)
        self.stats["total_cpu_time"] += processing_time

        return result

    def _process_batch_cpu_internal(self, images: list[np.ndarray], operation: str, **kwargs) -> list[np.ndarray]:
        """Process batch using CPU (internal method without stats update)."""
        result = []
        for img in images:
            if operation == "gaussian_blur":
                processed = self._cpu_gaussian_blur(img, **kwargs)
            elif operation == "edge_detection":
                processed = self._cpu_edge_detection(img, **kwargs)
            elif operation == "histogram_equalization":
                processed = self._cpu_histogram_equalization(img, **kwargs)
            elif operation == "noise_reduction":
                processed = self._cpu_noise_reduction(img, **kwargs)
            else:
                logger.warning(f"Unknown operation: {operation}")
                processed = img

            result.append(processed)

        return result

    def _process_batch_cpu_jit(self, images: list[np.ndarray], operation: str, **kwargs) -> list[np.ndarray]:
        """Process batch using CPU with JIT compilation."""
        if not NUMBA_AVAILABLE:
            return self._process_batch_cpu(images, operation, **kwargs)

        # Use Numba JIT for CPU acceleration
        result = []
        for img in images:
            if operation == "gaussian_blur":
                processed = self._jit_gaussian_blur(img, **kwargs)
            else:
                processed = self._process_batch_cpu([img], operation, **kwargs)[0]

            result.append(processed)

        return result

    # GPU processing functions (CuPy)
    def _gpu_gaussian_blur_cupy(self, gpu_img: "cp.ndarray", sigma: float = 1.0) -> "cp.ndarray":
        """Apply Gaussian blur using CuPy."""
        from cupyx.scipy import ndimage

        return ndimage.gaussian_filter(gpu_img, sigma=sigma)

    def _gpu_edge_detection_cupy(self, gpu_img: "cp.ndarray", **kwargs) -> "cp.ndarray":
        """Apply edge detection using CuPy."""
        from cupyx.scipy import ndimage

        # Convert to grayscale if needed
        if len(gpu_img.shape) == 3:
            gray = cp.mean(gpu_img, axis=2)
        else:
            gray = gpu_img

        # Sobel edge detection
        sobel_x = ndimage.sobel(gray, axis=1)
        sobel_y = ndimage.sobel(gray, axis=0)
        edges = cp.sqrt(sobel_x**2 + sobel_y**2)

        return edges

    def _gpu_histogram_equalization_cupy(self, gpu_img: "cp.ndarray", **kwargs) -> "cp.ndarray":
        """Apply histogram equalization using CuPy."""
        # Simplified histogram equalization
        if len(gpu_img.shape) == 3:
            # Process each channel separately
            result = cp.zeros_like(gpu_img)
            for c in range(gpu_img.shape[2]):
                result[:, :, c] = self._equalize_channel_cupy(gpu_img[:, :, c])
            return result
        else:
            return self._equalize_channel_cupy(gpu_img)

    def _equalize_channel_cupy(self, channel: "cp.ndarray") -> "cp.ndarray":
        """Equalize single channel using CuPy."""
        # Compute histogram
        hist, bins = cp.histogram(channel.flatten(), bins=256, range=(0, 256))

        # Compute cumulative distribution function
        cdf = hist.cumsum()
        cdf_normalized = cdf * 255 / cdf[-1]

        # Apply equalization
        equalized = cp.interp(channel.flatten(), bins[:-1], cdf_normalized)
        return equalized.reshape(channel.shape).astype(cp.uint8)

    def _gpu_noise_reduction_cupy(self, gpu_img: "cp.ndarray", strength: float = 0.5) -> "cp.ndarray":
        """Apply noise reduction using CuPy."""
        from cupyx.scipy import ndimage

        # Simple Gaussian denoising
        sigma = strength * 2.0
        return ndimage.gaussian_filter(gpu_img, sigma=sigma)

    # CPU processing functions
    def _cpu_gaussian_blur(self, img: np.ndarray, sigma: float = 1.0) -> np.ndarray:
        """Apply Gaussian blur using CPU."""
        from scipy import ndimage

        return ndimage.gaussian_filter(img, sigma=sigma)

    def _cpu_edge_detection(self, img: np.ndarray, **kwargs) -> np.ndarray:
        """Apply edge detection using CPU."""
        from scipy import ndimage

        # Convert to grayscale if needed
        if len(img.shape) == 3:
            gray = np.mean(img, axis=2)
        else:
            gray = img

        # Sobel edge detection
        sobel_x = ndimage.sobel(gray, axis=1)
        sobel_y = ndimage.sobel(gray, axis=0)
        edges = np.sqrt(sobel_x**2 + sobel_y**2)

        return edges

    def _cpu_histogram_equalization(self, img: np.ndarray, **kwargs) -> np.ndarray:
        """Apply histogram equalization using CPU."""
        from skimage import exposure

        if len(img.shape) == 3:
            # Process each channel separately
            result = np.zeros_like(img)
            for c in range(img.shape[2]):
                result[:, :, c] = exposure.equalize_hist(img[:, :, c]) * 255
            return result.astype(np.uint8)
        else:
            return (exposure.equalize_hist(img) * 255).astype(np.uint8)

    def _cpu_noise_reduction(self, img: np.ndarray, strength: float = 0.5) -> np.ndarray:
        """Apply noise reduction using CPU."""
        from scipy import ndimage

        # Simple Gaussian denoising
        sigma = strength * 2.0
        return ndimage.gaussian_filter(img, sigma=sigma)

    # JIT-compiled functions
    if NUMBA_AVAILABLE:

        @staticmethod
        @jit(nopython=True, parallel=True)
        def _jit_gaussian_blur(img: np.ndarray, sigma: float = 1.0) -> np.ndarray:
            """JIT-compiled Gaussian blur (simplified)."""
            # Simplified Gaussian blur implementation
            kernel_size = int(6 * sigma + 1)
            if kernel_size % 2 == 0:
                kernel_size += 1

            # Create Gaussian kernel
            kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
            center = kernel_size // 2

            for i in range(kernel_size):
                for j in range(kernel_size):
                    x, y = i - center, j - center
                    kernel[i, j] = np.exp(-(x * x + y * y) / (2 * sigma * sigma))

            kernel = kernel / np.sum(kernel)

            # Apply convolution
            if len(img.shape) == 3:
                result = np.zeros_like(img)
                for c in range(img.shape[2]):
                    result[:, :, c] = GPUAccelerator._convolve_2d_jit(img[:, :, c], kernel)
                return result
            else:
                return GPUAccelerator._convolve_2d_jit(img, kernel)

        @staticmethod
        @jit(nopython=True, parallel=True)
        def _convolve_2d_jit(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
            """JIT-compiled 2D convolution."""
            h, w = img.shape
            kh, kw = kernel.shape
            pad_h, pad_w = kh // 2, kw // 2

            result = np.zeros_like(img)

            for i in range(pad_h, h - pad_h):
                for j in range(pad_w, w - pad_w):
                    value = 0.0
                    for ki in range(kh):
                        for kj in range(kw):
                            value += img[i - pad_h + ki, j - pad_w + kj] * kernel[ki, kj]
                    result[i, j] = value

            return result

    def get_device_info(self) -> dict[str, Any]:
        """Get GPU device information.

        Returns:
            Dictionary containing device information
        """
        return {
            "backend": self.backend.value,
            "device_info": self.device_info,
            "is_initialized": self.is_initialized,
            "memory_pool_enabled": self.config.enable_memory_pool,
        }

    def get_performance_stats(self) -> dict[str, Any]:
        """Get performance statistics.

        Returns:
            Dictionary containing performance statistics
        """
        total_ops = self.stats["operations_processed"]
        if total_ops == 0:
            return self.stats

        stats = self.stats.copy()
        stats.update(
            {
                "gpu_percentage": (self.stats["gpu_operations"] / total_ops) * 100,
                "cpu_percentage": (self.stats["cpu_operations"] / total_ops) * 100,
                "avg_gpu_time_per_op": self.stats["total_gpu_time"] / max(self.stats["gpu_operations"], 1),
                "avg_cpu_time_per_op": self.stats["total_cpu_time"] / max(self.stats["cpu_operations"], 1),
                "avg_transfer_time": self.stats["memory_transfer_time"] / max(self.stats["memory_transfers"], 1),
            }
        )

        return stats

    def cleanup(self) -> None:
        """Cleanup GPU resources."""
        try:
            if self.backend == GPUBackend.CUPY and self.memory_pool:
                self.memory_pool.free_all_blocks()

            logger.info("GPU resources cleaned up")

        except Exception as e:
            logger.error(f"GPU cleanup failed: {e}")


def create_gpu_config_for_automotive() -> GPUConfiguration:
    """Create GPU configuration optimized for automotive applications.

    Returns:
        GPUConfiguration for automotive use
    """
    return GPUConfiguration(
        preferred_backend=GPUBackend.CUPY,
        processing_mode=ProcessingMode.HYBRID,
        batch_size=8,
        num_streams=4,
        memory_pool_size_mb=1024,
        enable_async_processing=True,
        prefetch_data=True,
    )


def create_gpu_config_for_surveillance() -> GPUConfiguration:
    """Create GPU configuration optimized for surveillance applications.

    Returns:
        GPUConfiguration for surveillance use
    """
    return GPUConfiguration(
        preferred_backend=GPUBackend.AUTO,
        processing_mode=ProcessingMode.GPU_ONLY,
        batch_size=16,
        num_streams=2,
        memory_pool_size_mb=2048,
        enable_async_processing=True,
        optimize_memory_access=True,
    )
