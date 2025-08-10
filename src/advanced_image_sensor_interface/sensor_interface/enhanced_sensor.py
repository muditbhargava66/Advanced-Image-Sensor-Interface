"""Enhanced Sensor Interface for v2.0.0.

This module provides enhanced sensor interface capabilities including:
- Support for resolutions up to 8K (7680x4320)
- HDR image processing
- RAW format support
- Multi-sensor synchronization
- Advanced timing controls
"""

import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class SensorResolution(Enum):
    """Supported sensor resolutions."""

    # Standard resolutions
    VGA = (640, 480)
    HD = (1280, 720)
    FHD = (1920, 1080)
    QHD = (2560, 1440)
    UHD_4K = (3840, 2160)

    # High-end resolutions (v2.0.0 feature)
    UHD_5K = (5120, 2880)
    UHD_6K = (6144, 3456)
    UHD_8K = (7680, 4320)

    # Custom resolutions
    CUSTOM = (0, 0)  # Will be set dynamically


class HDRMode(Enum):
    """HDR processing modes."""

    DISABLED = "disabled"
    HDR10 = "hdr10"
    HDR10_PLUS = "hdr10_plus"
    DOLBY_VISION = "dolby_vision"
    CUSTOM = "custom"


class RAWFormat(Enum):
    """Supported RAW image formats."""

    RAW8 = 8
    RAW10 = 10
    RAW12 = 12
    RAW14 = 14
    RAW16 = 16
    RAW20 = 20


@dataclass
class SensorConfiguration:
    """Enhanced sensor configuration for v2.0.0."""

    # Basic configuration
    resolution: SensorResolution = SensorResolution.FHD
    custom_resolution: Optional[tuple[int, int]] = None
    frame_rate: float = 30.0
    bit_depth: int = 12

    # HDR configuration
    hdr_mode: HDRMode = HDRMode.DISABLED
    hdr_exposure_ratio: float = 4.0
    hdr_tone_mapping: bool = True

    # RAW format support
    raw_format: RAWFormat = RAWFormat.RAW12
    raw_processing: bool = False

    # Multi-sensor configuration
    sensor_count: int = 1
    synchronization_enabled: bool = False
    master_sensor_id: int = 0

    # Advanced timing
    exposure_time: float = 1.0 / 60.0  # seconds
    gain: float = 1.0
    black_level: int = 64
    white_level: int = 1023

    # Performance settings
    gpu_acceleration: bool = False
    parallel_processing: bool = True
    buffer_count: int = 4

    def __post_init__(self):
        """Validate configuration parameters."""
        if self.resolution == SensorResolution.CUSTOM and not self.custom_resolution:
            raise ValueError("Custom resolution must be specified when using CUSTOM resolution mode")

        if self.frame_rate <= 0 or self.frame_rate > 240:
            raise ValueError("Frame rate must be between 0 and 240 fps")

        if self.sensor_count < 1 or self.sensor_count > 8:
            raise ValueError("Sensor count must be between 1 and 8")

        if self.synchronization_enabled and self.sensor_count == 1:
            logger.warning("Synchronization enabled but only one sensor configured")

    @property
    def effective_resolution(self) -> tuple[int, int]:
        """Get the effective resolution based on configuration."""
        if self.resolution == SensorResolution.CUSTOM:
            return self.custom_resolution or (1920, 1080)
        return self.resolution.value

    @property
    def pixel_count(self) -> int:
        """Get total pixel count."""
        width, height = self.effective_resolution
        return width * height

    @property
    def data_rate_mbps(self) -> float:
        """Calculate data rate in Mbps."""
        pixels_per_second = self.pixel_count * self.frame_rate
        bits_per_second = pixels_per_second * self.bit_depth
        return bits_per_second / 1_000_000  # Convert to Mbps


class EnhancedSensorInterface:
    """Enhanced sensor interface with v2.0.0 capabilities."""

    def __init__(self, config: SensorConfiguration):
        """Initialize enhanced sensor interface.

        Args:
            config: Sensor configuration
        """
        self.config = config
        self.sensors: dict[int, dict] = {}
        self.is_streaming = False
        self.frame_count = 0
        self.start_time: Optional[float] = None

        # Initialize sensors
        self._initialize_sensors()

        logger.info(f"Enhanced sensor interface initialized with {config.sensor_count} sensors")
        logger.info(f"Resolution: {config.effective_resolution}, Frame rate: {config.frame_rate} fps")
        logger.info(f"Data rate: {config.data_rate_mbps:.2f} Mbps")

    def _initialize_sensors(self) -> None:
        """Initialize all configured sensors."""
        for sensor_id in range(self.config.sensor_count):
            self.sensors[sensor_id] = {
                "id": sensor_id,
                "is_master": sensor_id == self.config.master_sensor_id,
                "status": "initialized",
                "frame_count": 0,
                "last_frame_time": 0.0,
                "exposure_time": self.config.exposure_time,
                "gain": self.config.gain,
                "temperature": 25.0,  # Celsius
            }

    def start_streaming(self) -> bool:
        """Start sensor streaming.

        Returns:
            True if streaming started successfully
        """
        if self.is_streaming:
            logger.warning("Streaming already active")
            return True

        try:
            # Validate configuration
            self._validate_streaming_config()

            # Start all sensors
            for sensor_id in self.sensors:
                self.sensors[sensor_id]["status"] = "streaming"

            self.is_streaming = True
            self.start_time = time.time()
            self.frame_count = 0

            logger.info(f"Started streaming on {len(self.sensors)} sensors")
            return True

        except Exception as e:
            logger.error(f"Failed to start streaming: {e}")
            return False

    def stop_streaming(self) -> bool:
        """Stop sensor streaming.

        Returns:
            True if streaming stopped successfully
        """
        if not self.is_streaming:
            logger.warning("Streaming not active")
            return True

        try:
            # Stop all sensors
            for sensor_id in self.sensors:
                self.sensors[sensor_id]["status"] = "stopped"

            self.is_streaming = False

            # Calculate statistics
            if self.start_time:
                duration = time.time() - self.start_time
                avg_fps = self.frame_count / duration if duration > 0 else 0
                logger.info(f"Streaming stopped. Duration: {duration:.2f}s, Avg FPS: {avg_fps:.2f}")

            return True

        except Exception as e:
            logger.error(f"Failed to stop streaming: {e}")
            return False

    def capture_frame(self, sensor_id: int = 0) -> Optional[np.ndarray]:
        """Capture a frame from specified sensor.

        Args:
            sensor_id: ID of sensor to capture from

        Returns:
            Captured frame as numpy array, or None if failed
        """
        if not self.is_streaming:
            logger.error("Cannot capture frame: streaming not active")
            return None

        if sensor_id not in self.sensors:
            logger.error(f"Invalid sensor ID: {sensor_id}")
            return None

        try:
            # Generate simulated frame data
            width, height = self.config.effective_resolution

            if self.config.raw_processing:
                # Generate RAW data
                frame = self._generate_raw_frame(width, height, sensor_id)
            else:
                # Generate RGB data
                frame = self._generate_rgb_frame(width, height, sensor_id)

            # Update sensor statistics
            self.sensors[sensor_id]["frame_count"] += 1
            self.sensors[sensor_id]["last_frame_time"] = time.time()
            self.frame_count += 1

            # Apply HDR processing if enabled
            if self.config.hdr_mode != HDRMode.DISABLED:
                frame = self._apply_hdr_processing(frame, sensor_id)

            return frame

        except Exception as e:
            logger.error(f"Failed to capture frame from sensor {sensor_id}: {e}")
            return None

    def capture_synchronized_frames(self) -> Optional[dict[int, np.ndarray]]:
        """Capture synchronized frames from all sensors.

        Returns:
            Dictionary mapping sensor IDs to captured frames
        """
        if not self.config.synchronization_enabled:
            logger.error("Synchronization not enabled")
            return None

        if not self.is_streaming:
            logger.error("Cannot capture frames: streaming not active")
            return None

        try:
            frames = {}
            sync_time = time.time()

            # Capture from all sensors simultaneously
            for sensor_id in self.sensors:
                frame = self.capture_frame(sensor_id)
                if frame is not None:
                    frames[sensor_id] = frame
                    # Mark synchronization timestamp
                    self.sensors[sensor_id]["sync_time"] = sync_time

            if len(frames) == len(self.sensors):
                logger.debug(f"Captured synchronized frames from {len(frames)} sensors")
                return frames
            else:
                logger.warning(f"Only captured {len(frames)}/{len(self.sensors)} synchronized frames")
                return frames if frames else None

        except Exception as e:
            logger.error(f"Failed to capture synchronized frames: {e}")
            return None

    def _validate_streaming_config(self) -> None:
        """Validate streaming configuration."""
        width, height = self.config.effective_resolution

        # Check resolution limits
        if width > 7680 or height > 4320:
            raise ValueError(f"Resolution {width}x{height} exceeds maximum supported (7680x4320)")

        # Check data rate limits
        if self.config.data_rate_mbps > 10000:  # 10 Gbps limit
            raise ValueError(f"Data rate {self.config.data_rate_mbps:.2f} Mbps exceeds limit")

        # Check frame rate for high resolutions
        if width >= 3840 and self.config.frame_rate > 60:
            logger.warning(f"High frame rate ({self.config.frame_rate} fps) with 4K+ resolution may not be achievable")

    def _generate_raw_frame(self, width: int, height: int, sensor_id: int) -> np.ndarray:
        """Generate simulated RAW frame data."""
        # Generate Bayer pattern data
        np.random.seed(int(time.time() * 1000) % 2**32 + sensor_id)

        # Create base noise pattern
        noise_level = 0.02
        base_level = self.config.black_level
        max_level = self.config.white_level

        # Generate RAW data with Bayer pattern
        raw_data = np.random.normal(
            base_level + (max_level - base_level) * 0.5, (max_level - base_level) * noise_level, (height, width)
        )

        # Apply Bayer pattern (RGGB)
        raw_data[0::2, 0::2] *= 1.2  # R
        raw_data[0::2, 1::2] *= 1.0  # G
        raw_data[1::2, 0::2] *= 1.0  # G
        raw_data[1::2, 1::2] *= 0.8  # B

        # Clip to valid range
        raw_data = np.clip(raw_data, base_level, max_level)

        # Convert to appropriate bit depth
        if self.config.raw_format == RAWFormat.RAW8:
            return raw_data.astype(np.uint8)
        elif self.config.raw_format == RAWFormat.RAW16:
            return raw_data.astype(np.uint16)
        else:
            return raw_data.astype(np.uint16)  # Default to 16-bit

    def _generate_rgb_frame(self, width: int, height: int, sensor_id: int) -> np.ndarray:
        """Generate simulated RGB frame data."""
        np.random.seed(int(time.time() * 1000) % 2**32 + sensor_id)

        # Generate test pattern with some variation
        frame = np.zeros((height, width, 3), dtype=np.uint8)

        # Create gradient pattern
        x_grad = np.linspace(0, 255, width)
        y_grad = np.linspace(0, 255, height)

        frame[:, :, 0] = np.outer(y_grad, np.ones(width))  # Red gradient
        frame[:, :, 1] = np.outer(np.ones(height), x_grad)  # Green gradient
        frame[:, :, 2] = 128 + 64 * np.sin(np.outer(y_grad, x_grad) / 1000)  # Blue pattern

        # Add some noise
        noise = np.random.normal(0, 5, frame.shape)
        frame = np.clip(frame + noise, 0, 255).astype(np.uint8)

        return frame

    def _apply_hdr_processing(self, frame: np.ndarray, sensor_id: int) -> np.ndarray:
        """Apply HDR processing to frame."""
        if self.config.hdr_mode == HDRMode.DISABLED:
            return frame

        try:
            # Simple HDR tone mapping simulation
            if frame.dtype == np.uint8:
                frame_float = frame.astype(np.float32) / 255.0
            else:
                frame_float = frame.astype(np.float32) / np.max(frame)

            # Apply tone mapping curve
            if self.config.hdr_tone_mapping:
                # Reinhard tone mapping
                frame_float = frame_float / (1.0 + frame_float)

            # Apply exposure adjustment
            frame_float *= self.config.hdr_exposure_ratio
            frame_float = np.clip(frame_float, 0.0, 1.0)

            # Convert back to original format
            if frame.dtype == np.uint8:
                return (frame_float * 255).astype(np.uint8)
            else:
                return (frame_float * np.max(frame)).astype(frame.dtype)

        except Exception as e:
            logger.error(f"HDR processing failed: {e}")
            return frame

    def get_sensor_status(self, sensor_id: Optional[int] = None) -> dict:
        """Get status of specified sensor or all sensors.

        Args:
            sensor_id: Specific sensor ID, or None for all sensors

        Returns:
            Dictionary containing sensor status information
        """
        if sensor_id is not None:
            if sensor_id not in self.sensors:
                raise ValueError(f"Invalid sensor ID: {sensor_id}")
            return self.sensors[sensor_id].copy()

        return {
            "sensors": {sid: sensor.copy() for sid, sensor in self.sensors.items()},
            "streaming": self.is_streaming,
            "total_frames": self.frame_count,
            "configuration": {
                "resolution": self.config.effective_resolution,
                "frame_rate": self.config.frame_rate,
                "hdr_mode": self.config.hdr_mode.value,
                "raw_processing": self.config.raw_processing,
                "synchronization": self.config.synchronization_enabled,
            },
        }


def create_8k_sensor_config() -> SensorConfiguration:
    """Create a configuration for 8K sensor operation.

    Returns:
        SensorConfiguration optimized for 8K operation
    """
    return SensorConfiguration(
        resolution=SensorResolution.UHD_8K,
        frame_rate=30.0,
        bit_depth=12,
        hdr_mode=HDRMode.HDR10,
        raw_format=RAWFormat.RAW12,
        raw_processing=True,
        gpu_acceleration=True,
        parallel_processing=True,
        buffer_count=8,
    )


def create_multi_sensor_config(sensor_count: int = 4) -> SensorConfiguration:
    """Create a configuration for multi-sensor operation.

    Args:
        sensor_count: Number of sensors to configure

    Returns:
        SensorConfiguration for multi-sensor setup
    """
    return SensorConfiguration(
        resolution=SensorResolution.UHD_4K,
        frame_rate=60.0,
        bit_depth=12,
        sensor_count=sensor_count,
        synchronization_enabled=True,
        master_sensor_id=0,
        gpu_acceleration=True,
        parallel_processing=True,
    )
