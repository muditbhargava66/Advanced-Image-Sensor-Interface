"""
Advanced buffer management for image sensor interfaces.

This module provides sophisticated buffer management capabilities including
DMA buffer pools, zero-copy operations, and memory-mapped I/O for high-performance
image data handling.
"""

import logging
import threading
import time
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class DMABuffer:
    """DMA buffer representation."""

    buffer_id: int
    size: int
    data: Optional[np.ndarray] = None
    in_use: bool = False
    timestamp: float = 0.0
    metadata: dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class BufferPoolStats:
    """Statistics for buffer pool usage."""

    total_buffers: int = 0
    available_buffers: int = 0
    buffers_in_use: int = 0
    total_allocations: int = 0
    total_deallocations: int = 0
    peak_usage: int = 0
    memory_usage_bytes: int = 0
    allocation_failures: int = 0


class DMABufferPool:
    """
    DMA buffer pool for high-performance image data handling.

    This class manages a pool of DMA-compatible buffers for efficient
    image data transfer between hardware and software components.
    """

    def __init__(self, buffer_count: int = 16, buffer_size: int = 8 * 1024 * 1024):
        """
        Initialize DMA buffer pool.

        Args:
            buffer_count: Number of buffers in the pool
            buffer_size: Size of each buffer in bytes
        """
        self.buffer_count = buffer_count
        self.buffer_size = buffer_size
        self.buffers: list[DMABuffer] = []
        self.available_buffers: list[DMABuffer] = []
        self.lock = threading.RLock()

        # Statistics
        self.stats = BufferPoolStats()

        # Initialize buffer pool
        self._initialize_buffers()

        logger.info(f"DMA buffer pool initialized: {buffer_count} buffers of {buffer_size} bytes each")

    def _initialize_buffers(self):
        """Initialize the buffer pool."""
        with self.lock:
            for i in range(self.buffer_count):
                # Create DMA-compatible buffer (simulation)
                # In real implementation, this would allocate DMA-coherent memory
                buffer_data = np.zeros(self.buffer_size, dtype=np.uint8)

                buffer = DMABuffer(buffer_id=i, size=self.buffer_size, data=buffer_data, in_use=False, timestamp=time.time())

                self.buffers.append(buffer)
                self.available_buffers.append(buffer)

            # Update statistics
            self.stats.total_buffers = self.buffer_count
            self.stats.available_buffers = self.buffer_count
            self.stats.memory_usage_bytes = self.buffer_count * self.buffer_size

        logger.info("DMA buffer pool initialized")

    def acquire_buffer(self) -> Optional[DMABuffer]:
        """
        Acquire an available buffer from the pool.

        Returns:
            DMABuffer if available, None if pool is exhausted
        """
        with self.lock:
            if not self.available_buffers:
                self.stats.allocation_failures += 1
                logger.warning("No available buffers in pool")
                return None

            # Get buffer from available list
            buffer = self.available_buffers.pop(0)
            buffer.in_use = True
            buffer.timestamp = time.time()

            # Update statistics
            self.stats.available_buffers -= 1
            self.stats.buffers_in_use += 1
            self.stats.total_allocations += 1

            if self.stats.buffers_in_use > self.stats.peak_usage:
                self.stats.peak_usage = self.stats.buffers_in_use

            logger.debug(f"Buffer {buffer.buffer_id} acquired")
            return buffer

    def release_buffer(self, buffer: DMABuffer) -> bool:
        """
        Release a buffer back to the pool.

        Args:
            buffer: Buffer to release

        Returns:
            True if successfully released, False otherwise
        """
        with self.lock:
            if not buffer.in_use:
                logger.warning(f"Attempting to release buffer {buffer.buffer_id} that is not in use")
                return False

            # Reset buffer state
            buffer.in_use = False
            buffer.timestamp = time.time()
            buffer.metadata.clear()

            # Add back to available list
            self.available_buffers.append(buffer)

            # Update statistics
            self.stats.available_buffers += 1
            self.stats.buffers_in_use -= 1
            self.stats.total_deallocations += 1

            logger.debug(f"Buffer {buffer.buffer_id} released")
            return True

    def get_buffer_by_id(self, buffer_id: int) -> Optional[DMABuffer]:
        """Get buffer by ID."""
        with self.lock:
            for buffer in self.buffers:
                if buffer.buffer_id == buffer_id:
                    return buffer
            return None

    def get_statistics(self) -> BufferPoolStats:
        """Get buffer pool statistics."""
        with self.lock:
            return BufferPoolStats(
                total_buffers=self.stats.total_buffers,
                available_buffers=self.stats.available_buffers,
                buffers_in_use=self.stats.buffers_in_use,
                total_allocations=self.stats.total_allocations,
                total_deallocations=self.stats.total_deallocations,
                peak_usage=self.stats.peak_usage,
                memory_usage_bytes=self.stats.memory_usage_bytes,
                allocation_failures=self.stats.allocation_failures,
            )

    def reset_statistics(self):
        """Reset buffer pool statistics."""
        with self.lock:
            self.stats.total_allocations = 0
            self.stats.total_deallocations = 0
            self.stats.peak_usage = 0
            self.stats.allocation_failures = 0

    def cleanup(self):
        """Cleanup buffer pool resources."""
        with self.lock:
            # In real implementation, this would free DMA memory
            for buffer in self.buffers:
                buffer.data = None

            self.buffers.clear()
            self.available_buffers.clear()

            logger.info("DMA buffer pool cleaned up")


class MemoryMappedBuffer:
    """
    Memory-mapped buffer for efficient file I/O operations.

    This class provides memory-mapped access to files for high-performance
    image data storage and retrieval.
    """

    def __init__(self, filename: str, size: int, mode: str = "r+b"):
        """
        Initialize memory-mapped buffer.

        Args:
            filename: Path to the file to map
            size: Size of the mapping in bytes
            mode: File access mode
        """
        self.filename = filename
        self.size = size
        self.mode = mode
        self.file_handle = None
        self.mmap_handle = None
        self.is_mapped = False

        logger.info(f"Memory-mapped buffer created for {filename}")

    def map(self) -> bool:
        """
        Map the file into memory.

        Returns:
            True if mapping successful, False otherwise
        """
        try:
            import mmap

            # Open file
            self.file_handle = open(self.filename, self.mode)

            # Create memory mapping
            if "w" in self.mode or "+" in self.mode:
                access = mmap.ACCESS_WRITE
            else:
                access = mmap.ACCESS_READ

            self.mmap_handle = mmap.mmap(self.file_handle.fileno(), self.size, access=access)

            self.is_mapped = True
            logger.info(f"File {self.filename} mapped to memory")
            return True

        except Exception as e:
            logger.error(f"Failed to map file {self.filename}: {e}")
            return False

    def unmap(self):
        """Unmap the file from memory."""
        try:
            if self.mmap_handle:
                self.mmap_handle.close()
                self.mmap_handle = None

            if self.file_handle:
                self.file_handle.close()
                self.file_handle = None

            self.is_mapped = False
            logger.info(f"File {self.filename} unmapped from memory")

        except Exception as e:
            logger.error(f"Error unmapping file {self.filename}: {e}")

    def read(self, offset: int, size: int) -> Optional[bytes]:
        """
        Read data from memory-mapped buffer.

        Args:
            offset: Offset in bytes from start of mapping
            size: Number of bytes to read

        Returns:
            Data bytes if successful, None otherwise
        """
        if not self.is_mapped:
            logger.error("Buffer is not mapped")
            return None

        try:
            self.mmap_handle.seek(offset)
            return self.mmap_handle.read(size)

        except Exception as e:
            logger.error(f"Error reading from memory-mapped buffer: {e}")
            return None

    def write(self, offset: int, data: bytes) -> bool:
        """
        Write data to memory-mapped buffer.

        Args:
            offset: Offset in bytes from start of mapping
            data: Data to write

        Returns:
            True if successful, False otherwise
        """
        if not self.is_mapped:
            logger.error("Buffer is not mapped")
            return False

        if "w" not in self.mode and "+" not in self.mode:
            logger.error("Buffer is not writable")
            return False

        try:
            self.mmap_handle.seek(offset)
            self.mmap_handle.write(data)
            return True

        except Exception as e:
            logger.error(f"Error writing to memory-mapped buffer: {e}")
            return False

    def flush(self) -> bool:
        """
        Flush changes to disk.

        Returns:
            True if successful, False otherwise
        """
        if not self.is_mapped:
            return False

        try:
            self.mmap_handle.flush()
            return True

        except Exception as e:
            logger.error(f"Error flushing memory-mapped buffer: {e}")
            return False

    def __enter__(self):
        """Context manager entry."""
        if self.map():
            return self
        else:
            raise RuntimeError("Failed to map buffer")

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.unmap()


class ZeroCopyBuffer:
    """
    Zero-copy buffer for efficient data transfer.

    This class provides zero-copy data transfer capabilities for high-performance
    image processing pipelines.
    """

    def __init__(self, data: np.ndarray):
        """
        Initialize zero-copy buffer.

        Args:
            data: NumPy array containing the data
        """
        self.data = data
        self.views: list[np.ndarray] = []
        self.lock = threading.RLock()

        logger.debug(f"Zero-copy buffer created with shape {data.shape}")

    def get_view(self, offset: int = 0, size: Optional[int] = None) -> Optional[np.ndarray]:
        """
        Get a view of the buffer data without copying.

        Args:
            offset: Offset in elements from start of buffer
            size: Number of elements in view (None for all remaining)

        Returns:
            NumPy array view if successful, None otherwise
        """
        with self.lock:
            try:
                if size is None:
                    view = self.data[offset:]
                else:
                    view = self.data[offset : offset + size]

                self.views.append(view)
                logger.debug(f"Created view with offset {offset}, size {size}")
                return view

            except Exception as e:
                logger.error(f"Error creating buffer view: {e}")
                return None

    def get_slice(self, start_row: int, end_row: int, start_col: int = 0, end_col: Optional[int] = None) -> Optional[np.ndarray]:
        """
        Get a 2D slice of the buffer without copying.

        Args:
            start_row: Starting row index
            end_row: Ending row index
            start_col: Starting column index
            end_col: Ending column index (None for all columns)

        Returns:
            NumPy array slice if successful, None otherwise
        """
        with self.lock:
            try:
                if len(self.data.shape) < 2:
                    logger.error("Buffer must be at least 2D for slice operation")
                    return None

                if end_col is None:
                    slice_view = self.data[start_row:end_row, start_col:]
                else:
                    slice_view = self.data[start_row:end_row, start_col:end_col]

                self.views.append(slice_view)
                logger.debug(f"Created slice [{start_row}:{end_row}, {start_col}:{end_col}]")
                return slice_view

            except Exception as e:
                logger.error(f"Error creating buffer slice: {e}")
                return None

    def reshape_view(self, new_shape: tuple) -> Optional[np.ndarray]:
        """
        Get a reshaped view of the buffer without copying.

        Args:
            new_shape: New shape for the view

        Returns:
            Reshaped NumPy array view if successful, None otherwise
        """
        with self.lock:
            try:
                reshaped = self.data.reshape(new_shape)
                self.views.append(reshaped)
                logger.debug(f"Created reshaped view with shape {new_shape}")
                return reshaped

            except Exception as e:
                logger.error(f"Error reshaping buffer: {e}")
                return None

    def get_statistics(self) -> dict[str, Any]:
        """Get buffer statistics."""
        with self.lock:
            return {
                "data_shape": self.data.shape,
                "data_dtype": str(self.data.dtype),
                "data_size_bytes": self.data.nbytes,
                "active_views": len(self.views),
                "is_contiguous": self.data.flags.c_contiguous,
                "is_writeable": self.data.flags.writeable,
            }
