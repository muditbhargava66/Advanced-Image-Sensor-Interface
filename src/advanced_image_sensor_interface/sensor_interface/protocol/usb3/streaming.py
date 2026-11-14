"""USB3 Vision simulation with locked FIFO buffers and owned frame payloads.

Frames use unpacked little-endian storage, exposed as flat uint8 arrays plus
FrameInfo metadata. This is a synchronous simulator, not lock-free USB I/O.
"""

import logging
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np

from advanced_image_sensor_interface.types import SimulationDelayConfig

from ..base import StreamingProtocolBase

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
    """Buffer ownership states."""

    FREE = "free"
    QUEUED = "queued"
    FILLING = "filling"
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
    """Pool-owned storage; consumers must copy before returning it."""

    buffer_id: int
    capacity: int
    data: Optional[np.ndarray] = None
    frame_info: Optional[FrameInfo] = None
    state: BufferState = BufferState.FREE
    queue_time: float = 0.0
    completion_time: float = 0.0


@dataclass
class StreamConfig:
    """Configuration for streaming and simulated operation delays."""

    buffer_count: int = 10
    buffer_size: int = 0
    timeout_ms: int = 5000
    payload_type: str = "image"
    enable_chunk_data: bool = False
    enable_timestamps: bool = True
    max_retries: int = 3
    simulation_delays: SimulationDelayConfig = field(default_factory=SimulationDelayConfig)

    def __post_init__(self) -> None:
        if self.buffer_count < 2:
            raise ValueError("Buffer count must be at least 2")
        if self.buffer_size < 0:
            raise ValueError("Buffer size cannot be negative")
        if self.timeout_ms < 100:
            raise ValueError("Timeout must be at least 100ms")
        if self.simulation_delays is None:
            self.simulation_delays = SimulationDelayConfig()
        elif isinstance(self.simulation_delays, dict):
            self.simulation_delays = SimulationDelayConfig(**self.simulation_delays)


@dataclass
class StreamStatistics:
    """Measured statistics for the current capture session."""

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
    """Locked buffer pool with O(1) FIFO acquisition and completion pops."""

    def __init__(self, config: StreamConfig) -> None:
        self.config = config
        self._buffers: list[StreamBuffer] = []
        self._lock = threading.Lock()
        self._free_buffers: deque[int] = deque()
        self._queued_buffers: deque[int] = deque()
        self._filled_buffers: deque[int] = deque()

    def allocate(self, buffer_size: int) -> bool:
        with self._lock:
            if buffer_size <= 0 or self._buffers:
                return False
            try:
                buffers = [
                    StreamBuffer(i, buffer_size, np.zeros(buffer_size, dtype=np.uint8)) for i in range(self.config.buffer_count)
                ]
            except (MemoryError, ValueError) as e:
                logger.error(f"Buffer allocation failed: {e}")
                return False
            self._buffers = buffers
            self._free_buffers.extend(range(len(buffers)))
            return True

    def deallocate(self) -> None:
        with self._lock:
            self._buffers.clear()
            self._free_buffers.clear()
            self._queued_buffers.clear()
            self._filled_buffers.clear()

    def _valid_id(self, buffer_id: int) -> bool:
        return 0 <= buffer_id < len(self._buffers)

    def get_free_buffer(self) -> Optional[StreamBuffer]:
        """Reserve the oldest free buffer and queue it for capture."""
        with self._lock:
            if not self._free_buffers:
                return None
            buffer = self._buffers[self._free_buffers.popleft()]
            buffer.state = BufferState.QUEUED
            buffer.queue_time = time.monotonic()
            self._queued_buffers.append(buffer.buffer_id)
            return buffer

    def queue_buffer(self, buffer_id: int) -> bool:
        with self._lock:
            if not self._valid_id(buffer_id):
                return False
            buffer = self._buffers[buffer_id]
            if buffer.state != BufferState.FREE:
                return False
            self._free_buffers.remove(buffer_id)
            buffer.state = BufferState.QUEUED
            buffer.queue_time = time.monotonic()
            self._queued_buffers.append(buffer_id)
            return True

    def get_queued_buffer(self) -> Optional[StreamBuffer]:
        """Transfer the oldest queued buffer to the producer."""
        with self._lock:
            if not self._queued_buffers:
                return None
            buffer = self._buffers[self._queued_buffers.popleft()]
            buffer.state = BufferState.FILLING
            return buffer

    def mark_filled(self, buffer_id: int, frame_info: FrameInfo) -> bool:
        with self._lock:
            if not self._valid_id(buffer_id):
                return False
            buffer = self._buffers[buffer_id]
            if buffer.state not in (BufferState.QUEUED, BufferState.FILLING):
                return False
            if not 0 <= frame_info.payload_size <= buffer.capacity:
                return False
            if buffer.state == BufferState.QUEUED:
                self._queued_buffers.remove(buffer_id)
            buffer.state = BufferState.FILLED
            buffer.frame_info = frame_info
            buffer.completion_time = time.monotonic()
            self._filled_buffers.append(buffer_id)
            return True

    def get_filled_buffer(self) -> Optional[StreamBuffer]:
        with self._lock:
            if not self._filled_buffers:
                return None
            buffer = self._buffers[self._filled_buffers.popleft()]
            buffer.state = BufferState.PROCESSING
            return buffer

    def return_buffer(self, buffer_id: int) -> bool:
        """Release consumer ownership (or cancel a producer/queued request)."""
        with self._lock:
            if not self._valid_id(buffer_id):
                return False
            buffer = self._buffers[buffer_id]
            if buffer.state not in (BufferState.PROCESSING, BufferState.FILLING, BufferState.QUEUED):
                return False
            if buffer.state == BufferState.QUEUED:
                self._queued_buffers.remove(buffer_id)
            buffer.state = BufferState.FREE
            buffer.frame_info = None
            self._free_buffers.append(buffer_id)
            return True

    def get_statistics(self) -> dict[str, int]:
        with self._lock:
            return {
                "total_buffers": len(self._buffers),
                "free_buffers": len(self._free_buffers),
                "queued_buffers": len(self._queued_buffers),
                "filled_buffers": len(self._filled_buffers),
            }


class USB3StreamingManager:
    """Synchronous streaming simulator with bounded waits and owned results."""

    FrameCallback = Callable[[np.ndarray, FrameInfo], None]

    def __init__(self, config: Optional[StreamConfig] = None) -> None:
        self.config = config or StreamConfig()
        self.buffer_pool = BufferPool(self.config)
        self._state = StreamState.IDLE
        self._statistics = StreamStatistics()
        self._frame_callbacks: list[USB3StreamingManager.FrameCallback] = []
        self._lock = threading.Lock()
        self._frame_counter = 0
        self._start_time: Optional[float] = None
        self._last_frame_time: Optional[float] = None
        self._prepared: Optional[tuple[int, int, str, int]] = None

    @property
    def state(self) -> StreamState:
        return self._state

    @property
    def is_streaming(self) -> bool:
        return self._state == StreamState.STREAMING

    def prepare(self, width: int, height: int, pixel_format: str) -> bool:
        with self._lock:
            if self._state != StreamState.IDLE:
                return False
            try:
                bpp = self._get_bytes_per_pixel(pixel_format)
                if width <= 0 or height <= 0 or (pixel_format.startswith("YUV422") and width % 2):
                    raise ValueError("Invalid frame dimensions")
                payload_size = width * height * bpp
                delays = self.config.simulation_delays
                time.sleep(delays.apply_delay(delays.buffer_allocation_delay))
                if not self.buffer_pool.allocate(max(payload_size, self.config.buffer_size)):
                    self._state = StreamState.ERROR
                    return False
            except (ValueError, TypeError) as e:
                logger.error(f"Stream preparation failed: {e}")
                return False
            self._prepared = (width, height, pixel_format, payload_size)
            self._state = StreamState.PREPARING
            return True

    def start_streaming(self) -> bool:
        with self._lock:
            if self._state == StreamState.PAUSED:
                self._state = StreamState.STREAMING
                return True
            if self._state != StreamState.PREPARING or self._prepared is None:
                return False
            delays = self.config.simulation_delays
            time.sleep(delays.apply_delay(delays.streaming_setup_delay))
            while self.buffer_pool.get_free_buffer() is not None:
                pass
            self._frame_counter = 0
            self._statistics = StreamStatistics()
            self._start_time = time.monotonic()
            self._last_frame_time = None
            self._state = StreamState.STREAMING
            return True

    def stop_streaming(self) -> bool:
        with self._lock:
            if self._state == StreamState.IDLE:
                return True
            self._state = StreamState.STOPPING
            delays = self.config.simulation_delays
            time.sleep(delays.apply_delay(delays.streaming_teardown_delay))
            self._refresh_statistics()
            self._start_time = None
            self.buffer_pool.deallocate()
            self._prepared = None
            self._state = StreamState.IDLE
            return True

    def pause_streaming(self) -> bool:
        with self._lock:
            if self._state != StreamState.STREAMING:
                return False
            self._state = StreamState.PAUSED
            return True

    def resume_streaming(self) -> bool:
        with self._lock:
            if self._state != StreamState.PAUSED:
                return False
            self._state = StreamState.STREAMING
            return True

    def register_frame_callback(self, callback: FrameCallback) -> None:
        with self._lock:
            self._frame_callbacks.append(callback)

    def unregister_frame_callback(self, callback: FrameCallback) -> None:
        with self._lock:
            if callback in self._frame_callbacks:
                self._frame_callbacks.remove(callback)

    def get_frame(self, timeout_ms: Optional[int] = None) -> Optional[tuple[np.ndarray, FrameInfo]]:
        """Return an owned flat-byte array and metadata, or None on timeout.

        The timeout bounds lock acquisition and simulated capture latency. Zero
        is a non-blocking request; CPU generation is checked against the deadline.
        """
        timeout = self.config.timeout_ms if timeout_ms is None else timeout_ms
        if timeout < 0:
            raise ValueError("Timeout cannot be negative")
        started = time.monotonic()
        deadline = started + timeout / 1000
        if not self._lock.acquire(timeout=timeout / 1000):
            return None
        try:
            if self._state != StreamState.STREAMING:
                return None
            buffer = self._simulate_frame_capture(deadline)
            if buffer is None:
                return None
            try:
                info = buffer.frame_info
                if buffer.data is None or info is None:
                    self._statistics.frames_dropped += 1
                    return None
                data = buffer.data[: info.payload_size].copy()
            finally:
                self.buffer_pool.return_buffer(buffer.buffer_id)
                # FIFO acquisition keeps the normal recycle path O(1).
                self.buffer_pool.get_free_buffer()
            self._update_statistics(info)
            latency = (time.monotonic() - started) * 1000
            count = self._statistics.frames_captured
            self._statistics.average_latency_ms += (latency - self._statistics.average_latency_ms) / count
            self._statistics.max_latency_ms = max(self._statistics.max_latency_ms, latency)
            callbacks = tuple(self._frame_callbacks)
        finally:
            self._lock.release()
        # Call user code outside locks. No subsequent capture can overwrite data.
        for callback in callbacks:
            try:
                callback(data, info)
            except Exception as e:
                logger.error(f"Frame callback error: {e}")
        return data, info

    def _simulate_frame_capture(self, deadline: Optional[float] = None) -> Optional[StreamBuffer]:
        if self._prepared is None:
            return None
        delays = self.config.simulation_delays
        delay = delays.apply_delay(delays.frame_capture_delay)
        if deadline is not None:
            remaining = max(0.0, deadline - time.monotonic())
            if remaining <= 0 or delay > remaining:
                if remaining > 0:
                    time.sleep(remaining)
                return None
        time.sleep(delay)
        width, height, fmt, size = self._prepared
        payload = StreamingProtocolBase._generate_test_frame_vectorized(
            width, height, self._get_bytes_per_pixel(fmt), self._frame_counter + 1, pixel_format=fmt
        )
        if deadline is not None and time.monotonic() > deadline:
            return None
        buffer = self.buffer_pool.get_queued_buffer()
        if buffer is None:
            self._statistics.buffer_underruns += 1
            self._statistics.frames_dropped += 1
            return None
        try:
            if buffer.data is None:
                raise ValueError("Buffer is missing storage")
            buffer.data[:size] = np.frombuffer(payload, dtype=np.uint8)
            info = FrameInfo(
                frame_id=self._frame_counter + 1,
                timestamp_ns=time.time_ns() if self.config.enable_timestamps else 0,
                exposure_time_us=10000.0,
                gain=1.0,
                width=width,
                height=height,
                pixel_format=fmt,
                payload_size=size,
            )
            if not self.buffer_pool.mark_filled(buffer.buffer_id, info):
                raise ValueError("Invalid buffer completion")
            self._frame_counter += 1
            return self.buffer_pool.get_filled_buffer()
        except Exception as e:
            logger.error(f"Simulated capture failed: {e}")
            self.buffer_pool.return_buffer(buffer.buffer_id)
            self.buffer_pool.get_free_buffer()
            self._statistics.frames_dropped += 1
            return None

    def _refresh_statistics(self) -> None:
        if self._start_time is not None:
            self._statistics.uptime_seconds = max(0.0, time.monotonic() - self._start_time)
        elapsed = self._statistics.uptime_seconds
        self._statistics.average_frame_rate = self._statistics.frames_captured / elapsed if elapsed > 0 else 0.0

    def _update_statistics(self, frame_info: Optional[FrameInfo]) -> None:
        if frame_info is None:
            return
        self._statistics.frames_captured += 1
        self._statistics.bytes_transferred += frame_info.payload_size
        self._statistics.frames_incomplete += int(frame_info.is_incomplete)
        self._last_frame_time = time.monotonic()
        self._refresh_statistics()

    def get_statistics(self) -> StreamStatistics:
        with self._lock:
            self._refresh_statistics()
            return self._statistics

    def reset_statistics(self) -> None:
        with self._lock:
            self._statistics = StreamStatistics()
            self._last_frame_time = None
            if self._start_time is not None:
                self._start_time = time.monotonic()

    def _get_bytes_per_pixel(self, pixel_format: str) -> int:
        return StreamingProtocolBase._get_bytes_per_pixel_common(pixel_format)
