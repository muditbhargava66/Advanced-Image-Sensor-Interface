"""
USB3 Vision Streaming Management

Provides buffer management, frame handling, and stream control
for USB3 Vision cameras.

Key Features:
- Buffer pool with configurable size
- Frame completion callbacks
- Stream statistics and monitoring
- Error recovery and retry logic
- Multi-stream support

Version: 3.0.0
"""

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class StreamState(Enum):
    """Stream state machine states."""

    IDLE = "idle"
    PREPARING = "preparing"
    STREAMING = "streaming"
    PAUSED = "paused"
    STOPPING = "stopping"
    ERROR = "error"


class BufferState(Enum):
    """Buffer state in the pool."""

    FREE = "free"
    QUEUED = "queued"
    FILLED = "filled"
    PROCESSING = "processing"


@dataclass
class FrameInfo:
    """Metadata for a captured frame."""

    frame_id: int
    timestamp_ns: int
    exposure_time_us: float
    gain: float
    width: int
    height: int
    pixel_format: str
    payload_size: int
    is_incomplete: bool = False


@dataclass
class StreamBuffer:
    """Buffer for frame data."""

    buffer_id: int
    capacity: int
    data: Optional[np.ndarray] = None
    frame_info: Optional[FrameInfo] = None
    state: BufferState = BufferState.FREE
    queue_time: float = 0.0
    completion_time: float = 0.0


@dataclass
class StreamConfig:
    """Configuration for streaming."""

    buffer_count: int = 10
    buffer_size: int = 0  # 0 = auto-calculate
    timeout_ms: int = 5000
    payload_type: str = "image"
    enable_chunk_data: bool = False
    enable_timestamps: bool = True
    max_retries: int = 3

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.buffer_count < 2:
            raise ValueError("Buffer count must be at least 2")

        if self.timeout_ms < 100:
            raise ValueError("Timeout must be at least 100ms")


@dataclass
class StreamStatistics:
    """Statistics for stream performance."""

    frames_captured: int = 0
    frames_dropped: int = 0
    frames_incomplete: int = 0
    bytes_transferred: int = 0
    buffer_underruns: int = 0
    retries: int = 0
    average_frame_rate: float = 0.0
    average_latency_ms: float = 0.0
    max_latency_ms: float = 0.0
    uptime_seconds: float = 0.0


class BufferPool:
    """
    Manages a pool of buffers for streaming.

    Handles buffer allocation, queuing, and recycling.
    """

    def __init__(self, config: StreamConfig) -> None:
        """
        Initialize buffer pool.

        Args:
            config: Stream configuration
        """
        self.config = config
        self._buffers: list[StreamBuffer] = []
        self._lock = threading.Lock()
        self._free_buffers: list[int] = []
        self._queued_buffers: list[int] = []
        self._filled_buffers: list[int] = []

    def allocate(self, buffer_size: int) -> bool:
        """
        Allocate buffers in the pool.

        Args:
            buffer_size: Size of each buffer in bytes

        Returns:
            True if allocation successful
        """
        try:
            with self._lock:
                for i in range(self.config.buffer_count):
                    buffer = StreamBuffer(
                        buffer_id=i, capacity=buffer_size, data=np.zeros(buffer_size, dtype=np.uint8), state=BufferState.FREE
                    )
                    self._buffers.append(buffer)
                    self._free_buffers.append(i)

            logger.info(f"Allocated {self.config.buffer_count} buffers, {buffer_size} bytes each")
            return True

        except Exception as e:
            logger.error(f"Buffer allocation failed: {e}")
            return False

    def deallocate(self) -> None:
        """Deallocate all buffers."""
        with self._lock:
            self._buffers.clear()
            self._free_buffers.clear()
            self._queued_buffers.clear()
            self._filled_buffers.clear()

        logger.info("Buffer pool deallocated")

    def get_free_buffer(self) -> Optional[StreamBuffer]:
        """
        Get a free buffer from the pool.

        Returns:
            Free buffer or None if none available
        """
        with self._lock:
            if not self._free_buffers:
                return None

            buffer_id = self._free_buffers.pop(0)
            buffer = self._buffers[buffer_id]
            buffer.state = BufferState.QUEUED
            buffer.queue_time = time.time()
            self._queued_buffers.append(buffer_id)
            return buffer

    def queue_buffer(self, buffer_id: int) -> bool:
        """
        Queue a buffer for frame capture.

        Args:
            buffer_id: Buffer to queue

        Returns:
            True if queued successfully
        """
        with self._lock:
            if buffer_id >= len(self._buffers):
                return False

            buffer = self._buffers[buffer_id]
            if buffer.state != BufferState.FREE:
                return False

            buffer.state = BufferState.QUEUED
            buffer.queue_time = time.time()
            self._free_buffers.remove(buffer_id)
            self._queued_buffers.append(buffer_id)
            return True

    def mark_filled(self, buffer_id: int, frame_info: FrameInfo) -> bool:
        """
        Mark a buffer as filled with frame data.

        Args:
            buffer_id: Buffer that was filled
            frame_info: Information about the captured frame

        Returns:
            True if marked successfully
        """
        with self._lock:
            if buffer_id >= len(self._buffers):
                return False

            buffer = self._buffers[buffer_id]
            buffer.state = BufferState.FILLED
            buffer.frame_info = frame_info
            buffer.completion_time = time.time()

            if buffer_id in self._queued_buffers:
                self._queued_buffers.remove(buffer_id)
            self._filled_buffers.append(buffer_id)
            return True

    def get_filled_buffer(self) -> Optional[StreamBuffer]:
        """
        Get the oldest filled buffer.

        Returns:
            Filled buffer or None if none available
        """
        with self._lock:
            if not self._filled_buffers:
                return None

            buffer_id = self._filled_buffers.pop(0)
            buffer = self._buffers[buffer_id]
            buffer.state = BufferState.PROCESSING
            return buffer

    def return_buffer(self, buffer_id: int) -> bool:
        """
        Return a buffer to the pool.

        Args:
            buffer_id: Buffer to return

        Returns:
            True if returned successfully
        """
        with self._lock:
            if buffer_id >= len(self._buffers):
                return False

            buffer = self._buffers[buffer_id]
            buffer.state = BufferState.FREE
            buffer.frame_info = None
            self._free_buffers.append(buffer_id)
            return True

    def get_statistics(self) -> dict[str, int]:
        """Get buffer pool statistics."""
        with self._lock:
            return {
                "total_buffers": len(self._buffers),
                "free_buffers": len(self._free_buffers),
                "queued_buffers": len(self._queued_buffers),
                "filled_buffers": len(self._filled_buffers),
            }


class USB3StreamingManager:
    """
    USB3 Vision streaming management.

    Provides high-level streaming control with buffer management,
    callbacks, and statistics.
    """

    FrameCallback = Callable[[np.ndarray, FrameInfo], None]

    def __init__(self, config: Optional[StreamConfig] = None) -> None:
        """
        Initialize streaming manager.

        Args:
            config: Stream configuration
        """
        self.config = config or StreamConfig()
        self.buffer_pool = BufferPool(self.config)
        self._state = StreamState.IDLE
        self._statistics = StreamStatistics()
        self._frame_callbacks: list[USB3StreamingManager.FrameCallback] = []
        self._lock = threading.Lock()
        self._frame_counter = 0
        self._start_time: Optional[float] = None
        self._last_frame_time: Optional[float] = None

        logger.info("USB3 Streaming Manager initialized")

    @property
    def state(self) -> StreamState:
        """Get current stream state."""
        return self._state

    @property
    def is_streaming(self) -> bool:
        """Check if currently streaming."""
        return self._state == StreamState.STREAMING

    def prepare(self, width: int, height: int, pixel_format: str) -> bool:
        """
        Prepare stream for capture.

        Args:
            width: Image width
            height: Image height
            pixel_format: Pixel format string

        Returns:
            True if preparation successful
        """
        if self._state != StreamState.IDLE:
            logger.warning("Stream not in IDLE state")
            return False

        self._state = StreamState.PREPARING

        # Calculate buffer size
        bytes_per_pixel = self._get_bytes_per_pixel(pixel_format)
        buffer_size = width * height * bytes_per_pixel

        if self.config.buffer_size > 0:
            buffer_size = max(buffer_size, self.config.buffer_size)

        # Allocate buffers
        if not self.buffer_pool.allocate(buffer_size):
            self._state = StreamState.ERROR
            return False

        logger.info(f"Stream prepared: {width}x{height} {pixel_format}")
        return True

    def start_streaming(self) -> bool:
        """
        Start streaming.

        Returns:
            True if started successfully
        """
        if self._state not in (StreamState.PREPARING, StreamState.PAUSED):
            logger.warning(f"Cannot start from state {self._state.value}")
            return False

        self._state = StreamState.STREAMING
        self._start_time = time.time()
        self._frame_counter = 0

        # Queue initial buffers
        for _ in range(min(5, self.config.buffer_count)):
            buffer = self.buffer_pool.get_free_buffer()
            if buffer is None:
                break

        logger.info("Streaming started")
        return True

    def stop_streaming(self) -> bool:
        """
        Stop streaming.

        Returns:
            True if stopped successfully
        """
        if self._state not in (StreamState.STREAMING, StreamState.PAUSED):
            return True

        self._state = StreamState.STOPPING

        # Update statistics
        if self._start_time:
            self._statistics.uptime_seconds = time.time() - self._start_time

        self.buffer_pool.deallocate()
        self._state = StreamState.IDLE

        logger.info("Streaming stopped")
        return True

    def pause_streaming(self) -> bool:
        """Pause streaming."""
        if self._state != StreamState.STREAMING:
            return False

        self._state = StreamState.PAUSED
        logger.info("Streaming paused")
        return True

    def resume_streaming(self) -> bool:
        """Resume paused streaming."""
        if self._state != StreamState.PAUSED:
            return False

        self._state = StreamState.STREAMING
        logger.info("Streaming resumed")
        return True

    def register_frame_callback(self, callback: FrameCallback) -> None:
        """
        Register a callback for frame events.

        Args:
            callback: Function to call when frame is ready
        """
        self._frame_callbacks.append(callback)
        logger.debug("Registered frame callback")

    def unregister_frame_callback(self, callback: FrameCallback) -> None:
        """
        Unregister a frame callback.

        Args:
            callback: Callback to remove
        """
        if callback in self._frame_callbacks:
            self._frame_callbacks.remove(callback)

    def get_frame(self, timeout_ms: Optional[int] = None) -> Optional[tuple[np.ndarray, FrameInfo]]:
        """
        Get the next available frame.

        Args:
            timeout_ms: Timeout in milliseconds

        Returns:
            Tuple of (frame_data, frame_info) or None
        """
        if self._state != StreamState.STREAMING:
            return None

        _timeout = timeout_ms or self.config.timeout_ms

        # Simulate frame capture
        buffer = self._simulate_frame_capture()
        if buffer is None:
            return None

        # Get frame data
        frame_data = buffer.data
        frame_info = buffer.frame_info

        # Return buffer to pool
        self.buffer_pool.return_buffer(buffer.buffer_id)

        if frame_data is None or frame_info is None:
            self._statistics.frames_dropped += 1
            logger.warning("Captured buffer was missing frame data or metadata")
            return None

        # Update statistics
        self._update_statistics(frame_info)

        # Notify callbacks
        self._notify_callbacks(frame_data, frame_info)

        return frame_data, frame_info

    def get_statistics(self) -> StreamStatistics:
        """Get stream statistics."""
        return self._statistics

    def reset_statistics(self) -> None:
        """Reset stream statistics."""
        self._statistics = StreamStatistics()
        logger.debug("Statistics reset")

    def _simulate_frame_capture(self) -> Optional[StreamBuffer]:
        """Simulate frame capture for testing."""
        buffer = self.buffer_pool.get_free_buffer()
        if buffer is None:
            self._statistics.buffer_underruns += 1
            return None

        # Generate simulated frame
        self._frame_counter += 1
        frame_info = FrameInfo(
            frame_id=self._frame_counter,
            timestamp_ns=int(time.time() * 1e9),
            exposure_time_us=10000.0,
            gain=1.0,
            width=1920,
            height=1080,
            pixel_format="Mono8",
            payload_size=1920 * 1080,
        )

        self.buffer_pool.mark_filled(buffer.buffer_id, frame_info)
        return self.buffer_pool.get_filled_buffer()

    def _update_statistics(self, frame_info: Optional[FrameInfo]) -> None:
        """Update stream statistics."""
        self._statistics.frames_captured += 1

        if frame_info is not None:
            self._statistics.bytes_transferred += frame_info.payload_size

            if frame_info.is_incomplete:
                self._statistics.frames_incomplete += 1

        # Calculate frame rate
        current_time = time.time()
        if self._last_frame_time is not None:
            interval = current_time - self._last_frame_time
            if interval > 0:
                instant_fps = 1.0 / interval
                # Exponential moving average
                alpha = 0.1
                self._statistics.average_frame_rate = alpha * instant_fps + (1 - alpha) * self._statistics.average_frame_rate

        self._last_frame_time = current_time

    def _notify_callbacks(self, frame_data: np.ndarray, frame_info: FrameInfo) -> None:
        """Notify registered callbacks."""
        for callback in self._frame_callbacks:
            try:
                callback(frame_data, frame_info)
            except Exception as e:
                logger.error(f"Frame callback error: {e}")

    def _get_bytes_per_pixel(self, pixel_format: str) -> int:
        """Get bytes per pixel for a format."""
        format_sizes = {
            "Mono8": 1,
            "Mono10": 2,
            "Mono12": 2,
            "Mono16": 2,
            "BayerRG8": 1,
            "BayerRG10": 2,
            "BayerRG12": 2,
            "RGB8": 3,
            "BGR8": 3,
            "YUV422": 2,
        }
        return format_sizes.get(pixel_format, 1)
