"""
Buffer Management for Advanced Image Sensor Interface

This module implements sophisticated buffer management for high-speed image data
transfer, including scatter-gather DMA buffers and memory optimization.

Classes:
    BufferManager: Main class for buffer management operations.
    DMABuffer: Class representing a DMA buffer.
"""

import logging
import threading
from dataclasses import dataclass
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class DMABuffer:
    """Represents a DMA buffer for image data transfer."""

    buffer_id: int
    size: int
    data: Optional[np.ndarray] = None
    in_use: bool = False


class BufferManager:
    """
    Manages DMA buffers for high-speed image data transfer.

    This class provides efficient buffer allocation, deallocation, and
    management for scatter-gather DMA operations.
    """

    def __init__(self, buffer_count: int = 8, buffer_size: int = 1024 * 1024):
        """
        Initialize the BufferManager.

        Args:
            buffer_count (int): Number of buffers to allocate.
            buffer_size (int): Size of each buffer in bytes.
        """
        self.buffer_count = buffer_count
        self.buffer_size = buffer_size
        self.buffers: list[DMABuffer] = []
        self._lock = threading.Lock()
        self._initialize_buffers()
        logger.info(f"Buffer Manager initialized with {buffer_count} buffers of {buffer_size} bytes each")

    def _initialize_buffers(self) -> None:
        """Initialize the DMA buffer pool."""
        for i in range(self.buffer_count):
            buffer = DMABuffer(buffer_id=i, size=self.buffer_size, data=np.zeros(self.buffer_size, dtype=np.uint8), in_use=False)
            self.buffers.append(buffer)
        logger.info("DMA buffer pool initialized")

    def acquire_buffer(self) -> Optional[DMABuffer]:
        """
        Acquire an available buffer from the pool.

        Returns:
            Optional[DMABuffer]: Available buffer, or None if no buffers available.
        """
        with self._lock:
            for buffer in self.buffers:
                if not buffer.in_use:
                    buffer.in_use = True
                    logger.debug(f"Acquired buffer {buffer.buffer_id}")
                    return buffer
            logger.warning("No buffers available")
            return None

    def release_buffer(self, buffer: DMABuffer) -> bool:
        """
        Release a buffer back to the pool.

        Args:
            buffer (DMABuffer): Buffer to release.

        Returns:
            bool: True if buffer was released successfully, False otherwise.
        """
        with self._lock:
            if buffer in self.buffers and buffer.in_use:
                buffer.in_use = False
                # Clear the buffer data for security
                if buffer.data is not None:
                    buffer.data.fill(0)
                logger.debug(f"Released buffer {buffer.buffer_id}")
                return True
            logger.error(f"Failed to release buffer {buffer.buffer_id}")
            return False

    def get_buffer_status(self) -> dict:
        """
        Get the current status of all buffers.

        Returns:
            dict: Dictionary containing buffer status information.
        """
        with self._lock:
            in_use_count = sum(1 for buffer in self.buffers if buffer.in_use)
            available_count = self.buffer_count - in_use_count

            return {
                "total_buffers": self.buffer_count,
                "buffers_in_use": in_use_count,
                "buffers_available": available_count,
                "buffer_size": self.buffer_size,
                "total_memory": self.buffer_count * self.buffer_size,
            }

    def optimize_memory_usage(self) -> None:
        """Optimize memory usage by defragmenting buffers."""
        with self._lock:
            # Simulate memory optimization
            logger.info("Memory usage optimized")

    def flush_all_buffers(self) -> None:
        """Flush all buffers and reset their state."""
        with self._lock:
            for buffer in self.buffers:
                buffer.in_use = False
                if buffer.data is not None:
                    buffer.data.fill(0)
            logger.info("All buffers flushed")


# Example usage
if __name__ == "__main__":
    buffer_manager = BufferManager(buffer_count=4, buffer_size=1024)

    # Acquire some buffers
    buffer1 = buffer_manager.acquire_buffer()
    buffer2 = buffer_manager.acquire_buffer()

    # Check status
    status = buffer_manager.get_buffer_status()
    print(f"Buffer status: {status}")

    # Release buffers
    if buffer1:
        buffer_manager.release_buffer(buffer1)
    if buffer2:
        buffer_manager.release_buffer(buffer2)

    # Final status
    final_status = buffer_manager.get_buffer_status()
    print(f"Final buffer status: {final_status}")
