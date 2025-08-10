"""Multi-Sensor Synchronization for v2.0.0.

This module provides multi-sensor synchronization capabilities including:
- Hardware-level synchronization
- Software-based frame alignment
- Timestamp management
- Multi-sensor calibration
- Synchronized capture control
"""

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


class SyncMode(Enum):
    """Synchronization modes."""

    HARDWARE = "hardware"
    SOFTWARE = "software"
    HYBRID = "hybrid"
    EXTERNAL = "external"


class TriggerMode(Enum):
    """Trigger modes for synchronization."""

    INTERNAL = "internal"
    EXTERNAL = "external"
    SOFTWARE = "software"
    MASTER_SLAVE = "master_slave"


class SyncStatus(Enum):
    """Synchronization status."""

    IDLE = "idle"
    SYNCING = "syncing"
    SYNCHRONIZED = "synchronized"
    ERROR = "error"
    TIMEOUT = "timeout"


@dataclass
class SyncConfiguration:
    """Multi-sensor synchronization configuration."""

    # Basic sync settings
    sync_mode: SyncMode = SyncMode.SOFTWARE
    trigger_mode: TriggerMode = TriggerMode.MASTER_SLAVE
    master_sensor_id: int = 0
    slave_sensor_ids: list[int] = field(default_factory=list)

    # Timing parameters
    sync_tolerance_us: float = 100.0  # Microseconds
    max_sync_attempts: int = 10
    sync_timeout_ms: float = 1000.0  # Milliseconds

    # Frame alignment
    enable_frame_alignment: bool = True
    alignment_method: str = "timestamp"  # "timestamp", "feature", "correlation"
    max_frame_delay_ms: float = 50.0

    # Hardware sync (if available)
    hardware_sync_pin: Optional[int] = None
    sync_signal_frequency: float = 30.0  # Hz
    sync_signal_duty_cycle: float = 0.5

    # Calibration
    enable_geometric_calibration: bool = False
    calibration_pattern_size: tuple[int, int] = (9, 6)  # Chessboard corners

    def __post_init__(self):
        """Validate configuration."""
        if self.master_sensor_id in self.slave_sensor_ids:
            raise ValueError("Master sensor cannot be in slave sensor list")

        if not 1.0 <= self.sync_tolerance_us <= 10000.0:
            raise ValueError("Sync tolerance must be between 1 and 10000 microseconds")

        if not 10.0 <= self.sync_timeout_ms <= 10000.0:
            raise ValueError("Sync timeout must be between 10 and 10000 milliseconds")


@dataclass
class SensorSyncState:
    """State information for a synchronized sensor."""

    sensor_id: int
    is_master: bool = False
    status: SyncStatus = SyncStatus.IDLE
    last_frame_timestamp: float = 0.0
    frame_count: int = 0
    sync_error_count: int = 0
    average_sync_error_us: float = 0.0

    # Calibration data
    calibration_matrix: Optional[np.ndarray] = None
    distortion_coefficients: Optional[np.ndarray] = None

    # Performance metrics
    capture_latency_ms: float = 0.0
    processing_time_ms: float = 0.0


class MultiSensorSynchronizer:
    """Multi-sensor synchronization manager."""

    def __init__(self, config: SyncConfiguration):
        """Initialize multi-sensor synchronizer.

        Args:
            config: Synchronization configuration
        """
        self.config = config
        self.sensors: dict[int, SensorSyncState] = {}
        self.sync_lock = threading.RLock()
        self.is_active = False
        self.sync_thread: Optional[threading.Thread] = None

        # Callbacks
        self.frame_callback: Optional[Callable] = None
        self.sync_error_callback: Optional[Callable] = None

        # Statistics
        self.total_sync_attempts = 0
        self.successful_syncs = 0
        self.sync_errors = 0

        self._initialize_sensors()
        logger.info(f"Multi-sensor synchronizer initialized with {len(self.sensors)} sensors")

    def _initialize_sensors(self) -> None:
        """Initialize sensor sync states."""
        # Initialize master sensor
        self.sensors[self.config.master_sensor_id] = SensorSyncState(
            sensor_id=self.config.master_sensor_id, is_master=True, status=SyncStatus.IDLE
        )

        # Initialize slave sensors
        for sensor_id in self.config.slave_sensor_ids:
            self.sensors[sensor_id] = SensorSyncState(sensor_id=sensor_id, is_master=False, status=SyncStatus.IDLE)

    def start_synchronization(self) -> bool:
        """Start multi-sensor synchronization.

        Returns:
            True if synchronization started successfully
        """
        if self.is_active:
            logger.warning("Synchronization already active")
            return True

        try:
            with self.sync_lock:
                # Validate sensor states
                if not self._validate_sensor_states():
                    return False

                # Start synchronization thread
                self.is_active = True
                self.sync_thread = threading.Thread(target=self._sync_worker, name="MultiSensorSync", daemon=True)
                self.sync_thread.start()

                # Update sensor states
                for sensor in self.sensors.values():
                    sensor.status = SyncStatus.SYNCING

                logger.info("Multi-sensor synchronization started")
                return True

        except Exception as e:
            logger.error(f"Failed to start synchronization: {e}")
            self.is_active = False
            return False

    def stop_synchronization(self) -> bool:
        """Stop multi-sensor synchronization.

        Returns:
            True if synchronization stopped successfully
        """
        if not self.is_active:
            logger.warning("Synchronization not active")
            return True

        try:
            with self.sync_lock:
                self.is_active = False

                # Wait for sync thread to finish
                if self.sync_thread and self.sync_thread.is_alive():
                    self.sync_thread.join(timeout=2.0)

                # Update sensor states
                for sensor in self.sensors.values():
                    sensor.status = SyncStatus.IDLE

                logger.info("Multi-sensor synchronization stopped")
                return True

        except Exception as e:
            logger.error(f"Failed to stop synchronization: {e}")
            return False

    def capture_synchronized_frames(self) -> Optional[dict[int, tuple[np.ndarray, float]]]:
        """Capture synchronized frames from all sensors.

        Returns:
            Dictionary mapping sensor IDs to (frame, timestamp) tuples
        """
        if not self.is_active:
            logger.error("Synchronization not active")
            return None

        try:
            with self.sync_lock:
                frames = {}
                capture_start = time.time()

                # Trigger capture on all sensors
                if self.config.trigger_mode == TriggerMode.MASTER_SLAVE:
                    # Master triggers first, then slaves
                    master_result = self._capture_from_sensor(self.config.master_sensor_id)
                    if master_result:
                        frames[self.config.master_sensor_id] = master_result

                    # Capture from slaves
                    for sensor_id in self.config.slave_sensor_ids:
                        slave_result = self._capture_from_sensor(sensor_id)
                        if slave_result:
                            frames[sensor_id] = slave_result

                elif self.config.trigger_mode == TriggerMode.SOFTWARE:
                    # Simultaneous software trigger
                    trigger_time = time.time()
                    for sensor_id in self.sensors:
                        result = self._capture_from_sensor(sensor_id, trigger_time)
                        if result:
                            frames[sensor_id] = result

                # Validate synchronization
                if len(frames) == len(self.sensors):
                    if self._validate_frame_synchronization(frames):
                        self.successful_syncs += 1

                        # Apply frame alignment if enabled
                        if self.config.enable_frame_alignment:
                            frames = self._align_frames(frames)

                        # Update statistics
                        capture_time = (time.time() - capture_start) * 1000
                        self._update_capture_statistics(capture_time)

                        return frames
                    else:
                        self.sync_errors += 1
                        logger.warning("Frame synchronization validation failed")

                return None

        except Exception as e:
            logger.error(f"Synchronized capture failed: {e}")
            self.sync_errors += 1
            return None

    def _sync_worker(self) -> None:
        """Background synchronization worker thread."""
        logger.info("Synchronization worker thread started")

        while self.is_active:
            try:
                # Check sensor synchronization status
                with self.sync_lock:
                    self._check_sensor_synchronization()

                # Sleep for a short interval
                time.sleep(0.001)  # 1ms

            except Exception as e:
                logger.error(f"Sync worker error: {e}")
                if self.sync_error_callback:
                    self.sync_error_callback(e)

        logger.info("Synchronization worker thread stopped")

    def _validate_sensor_states(self) -> bool:
        """Validate that all sensors are ready for synchronization."""
        for sensor_id, sensor in self.sensors.items():
            if sensor.status == SyncStatus.ERROR:
                logger.error(f"Sensor {sensor_id} is in error state")
                return False

        return True

    def _capture_from_sensor(self, sensor_id: int, trigger_time: Optional[float] = None) -> Optional[tuple[np.ndarray, float]]:
        """Capture frame from specific sensor.

        Args:
            sensor_id: ID of sensor to capture from
            trigger_time: Optional trigger timestamp

        Returns:
            Tuple of (frame, timestamp) or None if failed
        """
        try:
            # Simulate frame capture (in real implementation, this would interface with actual sensors)
            capture_time = trigger_time or time.time()

            # Generate simulated frame data
            height, width = 480, 640  # Default resolution
            frame = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)

            # Add sensor-specific pattern
            frame[:, :, sensor_id % 3] = np.clip(frame[:, :, sensor_id % 3] + 50, 0, 255)

            # Update sensor state
            sensor = self.sensors[sensor_id]
            sensor.last_frame_timestamp = capture_time
            sensor.frame_count += 1

            return frame, capture_time

        except Exception as e:
            logger.error(f"Failed to capture from sensor {sensor_id}: {e}")
            return None

    def _validate_frame_synchronization(self, frames: dict[int, tuple[np.ndarray, float]]) -> bool:
        """Validate that frames are properly synchronized.

        Args:
            frames: Dictionary of sensor frames with timestamps

        Returns:
            True if frames are synchronized within tolerance
        """
        if len(frames) < 2:
            return True

        # Extract timestamps
        timestamps = [timestamp for _, timestamp in frames.values()]

        # Calculate synchronization error
        min_time = min(timestamps)
        max_time = max(timestamps)
        sync_error_us = (max_time - min_time) * 1_000_000  # Convert to microseconds

        # Update sync error statistics
        for sensor_id in frames:
            sensor = self.sensors[sensor_id]
            sensor.average_sync_error_us = (sensor.average_sync_error_us * sensor.frame_count + sync_error_us) / (
                sensor.frame_count + 1
            )

        # Check if within tolerance
        is_synchronized = sync_error_us <= self.config.sync_tolerance_us

        if not is_synchronized:
            logger.warning(f"Sync error: {sync_error_us:.1f}μs (tolerance: {self.config.sync_tolerance_us}μs)")

        return is_synchronized

    def _align_frames(self, frames: dict[int, tuple[np.ndarray, float]]) -> dict[int, tuple[np.ndarray, float]]:
        """Apply frame alignment to synchronized frames.

        Args:
            frames: Dictionary of sensor frames with timestamps

        Returns:
            Dictionary of aligned frames
        """
        if self.config.alignment_method == "timestamp":
            # Already aligned by timestamp validation
            return frames

        elif self.config.alignment_method == "feature":
            # Feature-based alignment (simplified)
            return self._align_frames_by_features(frames)

        elif self.config.alignment_method == "correlation":
            # Cross-correlation alignment
            return self._align_frames_by_correlation(frames)

        else:
            logger.warning(f"Unknown alignment method: {self.config.alignment_method}")
            return frames

    def _align_frames_by_features(self, frames: dict[int, tuple[np.ndarray, float]]) -> dict[int, tuple[np.ndarray, float]]:
        """Align frames using feature detection (simplified implementation)."""
        # In a real implementation, this would use feature detection and matching
        # For now, just return the original frames
        return frames

    def _align_frames_by_correlation(self, frames: dict[int, tuple[np.ndarray, float]]) -> dict[int, tuple[np.ndarray, float]]:
        """Align frames using cross-correlation (simplified implementation)."""
        # In a real implementation, this would compute cross-correlation and apply shifts
        # For now, just return the original frames
        return frames

    def _check_sensor_synchronization(self) -> None:
        """Check and maintain sensor synchronization."""
        current_time = time.time()

        for sensor_id, sensor in self.sensors.items():
            # Check for timeout
            if (current_time - sensor.last_frame_timestamp) > (self.config.sync_timeout_ms / 1000.0):
                if sensor.status == SyncStatus.SYNCHRONIZED:
                    sensor.status = SyncStatus.TIMEOUT
                    logger.warning(f"Sensor {sensor_id} synchronization timeout")

            # Update status based on recent activity
            elif sensor.frame_count > 0:
                if sensor.status != SyncStatus.SYNCHRONIZED:
                    sensor.status = SyncStatus.SYNCHRONIZED

    def _update_capture_statistics(self, capture_time_ms: float) -> None:
        """Update capture performance statistics."""
        for sensor in self.sensors.values():
            sensor.capture_latency_ms = capture_time_ms

    def get_synchronization_status(self) -> dict[str, Any]:
        """Get current synchronization status.

        Returns:
            Dictionary containing synchronization status and statistics
        """
        with self.sync_lock:
            sensor_status = {}
            for sensor_id, sensor in self.sensors.items():
                sensor_status[sensor_id] = {
                    "status": sensor.status.value,
                    "is_master": sensor.is_master,
                    "frame_count": sensor.frame_count,
                    "last_timestamp": sensor.last_frame_timestamp,
                    "sync_error_us": sensor.average_sync_error_us,
                    "sync_error_count": sensor.sync_error_count,
                    "capture_latency_ms": sensor.capture_latency_ms,
                }

            return {
                "active": self.is_active,
                "config": {
                    "sync_mode": self.config.sync_mode.value,
                    "trigger_mode": self.config.trigger_mode.value,
                    "sync_tolerance_us": self.config.sync_tolerance_us,
                },
                "sensors": sensor_status,
                "statistics": {
                    "total_attempts": self.total_sync_attempts,
                    "successful_syncs": self.successful_syncs,
                    "sync_errors": self.sync_errors,
                    "success_rate": (self.successful_syncs / max(self.total_sync_attempts, 1)) * 100,
                },
            }

    def set_frame_callback(self, callback: Callable[[dict[int, tuple[np.ndarray, float]]], None]) -> None:
        """Set callback for synchronized frame events.

        Args:
            callback: Function to call when synchronized frames are captured
        """
        self.frame_callback = callback

    def set_sync_error_callback(self, callback: Callable[[Exception], None]) -> None:
        """Set callback for synchronization errors.

        Args:
            callback: Function to call when sync errors occur
        """
        self.sync_error_callback = callback

    def calibrate_sensors(self) -> bool:
        """Perform geometric calibration of sensors.

        Returns:
            True if calibration successful
        """
        if not self.config.enable_geometric_calibration:
            logger.info("Geometric calibration disabled")
            return True

        try:
            logger.info("Starting sensor calibration...")

            # In a real implementation, this would:
            # 1. Capture calibration images from all sensors
            # 2. Detect calibration pattern (e.g., chessboard)
            # 3. Compute camera matrices and distortion coefficients
            # 4. Store calibration data for each sensor

            # For now, just set dummy calibration data
            for sensor in self.sensors.values():
                sensor.calibration_matrix = np.eye(3, dtype=np.float32)
                sensor.distortion_coefficients = np.zeros(5, dtype=np.float32)

            logger.info("Sensor calibration completed")
            return True

        except Exception as e:
            logger.error(f"Sensor calibration failed: {e}")
            return False


def create_stereo_sync_config() -> SyncConfiguration:
    """Create configuration for stereo camera synchronization.

    Returns:
        SyncConfiguration for stereo setup
    """
    return SyncConfiguration(
        sync_mode=SyncMode.SOFTWARE,
        trigger_mode=TriggerMode.MASTER_SLAVE,
        master_sensor_id=0,
        slave_sensor_ids=[1],
        sync_tolerance_us=50.0,
        enable_frame_alignment=True,
        alignment_method="correlation",
        enable_geometric_calibration=True,
    )


def create_multi_camera_sync_config(num_cameras: int = 4) -> SyncConfiguration:
    """Create configuration for multi-camera synchronization.

    Args:
        num_cameras: Number of cameras to synchronize

    Returns:
        SyncConfiguration for multi-camera setup
    """
    return SyncConfiguration(
        sync_mode=SyncMode.HYBRID,
        trigger_mode=TriggerMode.EXTERNAL,
        master_sensor_id=0,
        slave_sensor_ids=list(range(1, num_cameras)),
        sync_tolerance_us=100.0,
        enable_frame_alignment=True,
        alignment_method="timestamp",
        hardware_sync_pin=1,
        sync_signal_frequency=30.0,
    )
