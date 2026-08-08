"""
MIPI D-PHY v2.5 Protocol Support

This module provides support for MIPI D-PHY v2.5 specification with data rates
up to 4.5 Gbps per lane. Includes high-speed signaling improvements with
equalization and de-emphasis for reliable high-frequency operation.

Key Features:
- Data rates: 2.5, 3.0, 3.5, 4.0, 4.5 Gbps per lane
- Up to 4 data lanes (18 Gbps aggregate)
- Improved signal integrity for long traces
- Backward compatible with D-PHY v1.x and v2.0

Version: 3.0.0
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


class DPHYVersion(Enum):
    """Supported D-PHY versions."""

    V1_0 = "1.0"  # Up to 1.0 Gbps
    V1_1 = "1.1"  # Up to 1.5 Gbps
    V1_2 = "1.2"  # Up to 2.0 Gbps
    V2_0 = "2.0"  # Up to 2.5 Gbps
    V2_1 = "2.1"  # Up to 3.0 Gbps
    V2_5 = "2.5"  # Up to 4.5 Gbps


class EqualizationMode(Enum):
    """Equalization modes for high-speed signaling."""

    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    ADAPTIVE = "adaptive"


class DeEmphasisLevel(Enum):
    """De-emphasis levels for transmitter signal conditioning."""

    DISABLED = 0.0
    LEVEL_1 = 3.5  # dB
    LEVEL_2 = 6.0  # dB
    LEVEL_3 = 9.5  # dB


@dataclass
class DPHY25Config:
    """
    Configuration for MIPI D-PHY v2.5 interface.

    Attributes:
        lanes: Number of data lanes (1-4)
        data_rate_gbps: Data rate per lane in Gbps (max 4.5)
        clock_mode: Continuous or non-continuous clock
        equalization: Equalization mode for receiver
        de_emphasis: Transmitter de-emphasis level
        settle_time_ns: Time for receiver to settle after HS transition
        hs_prepare_time_ns: High-speed prepare time
        hs_zero_time_ns: High-speed zero time
    """

    lanes: int = 4
    data_rate_gbps: float = 4.5
    clock_mode: str = "continuous"
    equalization: EqualizationMode = EqualizationMode.ADAPTIVE
    de_emphasis: DeEmphasisLevel = DeEmphasisLevel.LEVEL_1
    settle_time_ns: float = 85.0  # Minimum for v2.5
    hs_prepare_time_ns: float = 40.0
    hs_zero_time_ns: float = 100.0
    enable_scrambling: bool = True
    lane_mapping: list[int] = field(default_factory=lambda: [0, 1, 2, 3])

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if not 1 <= self.lanes <= 4:
            raise ValueError(f"Lane count must be 1-4, got {self.lanes}")

        if not 0.5 <= self.data_rate_gbps <= 4.5:
            raise ValueError(f"Data rate must be 0.5-4.5 Gbps for D-PHY v2.5, got {self.data_rate_gbps}")

        if self.clock_mode not in ("continuous", "non-continuous"):
            raise ValueError(f"Invalid clock mode: {self.clock_mode}")

        # Validate lane mapping
        if len(self.lane_mapping) != self.lanes:
            self.lane_mapping = list(range(self.lanes))

    @property
    def aggregate_bandwidth_gbps(self) -> float:
        """Calculate total bandwidth across all lanes."""
        return self.lanes * self.data_rate_gbps

    @property
    def dphy_version(self) -> DPHYVersion:
        """Determine D-PHY version based on data rate."""
        if self.data_rate_gbps <= 1.0:
            return DPHYVersion.V1_0
        elif self.data_rate_gbps <= 1.5:
            return DPHYVersion.V1_1
        elif self.data_rate_gbps <= 2.0:
            return DPHYVersion.V1_2
        elif self.data_rate_gbps <= 2.5:
            return DPHYVersion.V2_0
        elif self.data_rate_gbps <= 3.0:
            return DPHYVersion.V2_1
        else:
            return DPHYVersion.V2_5


@dataclass
class TransferStatistics:
    """Statistics for D-PHY v2.5 data transfers."""

    bytes_transferred: int = 0
    packets_sent: int = 0
    packets_received: int = 0
    crc_errors: int = 0
    sync_errors: int = 0
    lane_errors: list[int] = field(default_factory=lambda: [0, 0, 0, 0])
    equalization_adjustments: int = 0
    average_latency_ns: float = 0.0


class DPHY25Driver:
    """
    MIPI D-PHY v2.5 Driver implementation.

    Provides high-speed data transfer capabilities with advanced signal
    conditioning for reliable operation at up to 4.5 Gbps per lane.
    """

    # Timing constants (nanoseconds)
    MIN_HS_PREPARE = 40.0
    MIN_HS_ZERO = 100.0
    MIN_HS_TRAIL = 60.0
    MIN_CLK_PREPARE = 38.0
    MIN_CLK_ZERO = 100.0

    def __init__(self, config: DPHY25Config) -> None:
        """
        Initialize D-PHY v2.5 driver.

        Args:
            config: D-PHY v2.5 configuration
        """
        self.config = config
        self.is_connected = False
        self.is_streaming = False
        self._statistics = TransferStatistics()
        self._equalization_state: dict[int, float] = {}
        self._lane_status: dict[int, str] = {}

        # Initialize lane states
        for lane in range(config.lanes):
            self._equalization_state[lane] = 0.0
            self._lane_status[lane] = "idle"

        logger.info(
            f"D-PHY v2.5 driver initialized: {config.lanes} lanes @ "
            f"{config.data_rate_gbps} Gbps ({config.aggregate_bandwidth_gbps} Gbps aggregate)"
        )

    def connect(self) -> bool:
        """
        Establish D-PHY v2.5 connection.

        Performs lane training and equalization calibration.

        Returns:
            True if connection successful
        """
        try:
            logger.info("Starting D-PHY v2.5 connection sequence...")

            # Lane training sequence
            for lane in range(self.config.lanes):
                if not self._train_lane(lane):
                    logger.error(f"Lane {lane} training failed")
                    return False
                self._lane_status[lane] = "ready"

            # Equalization calibration
            if self.config.equalization != EqualizationMode.NONE:
                self._calibrate_equalization()

            self.is_connected = True
            logger.info("D-PHY v2.5 connection established successfully")
            return True

        except Exception as e:
            logger.error(f"D-PHY v2.5 connection failed: {e}")
            return False

    def disconnect(self) -> bool:
        """
        Disconnect D-PHY v2.5 interface.

        Returns:
            True if disconnection successful
        """
        if self.is_streaming:
            self.stop_streaming()

        self.is_connected = False
        for lane in range(self.config.lanes):
            self._lane_status[lane] = "disconnected"

        logger.info("D-PHY v2.5 disconnected")
        return True

    def start_streaming(self) -> bool:
        """
        Start high-speed data streaming.

        Returns:
            True if streaming started successfully
        """
        if not self.is_connected:
            logger.error("Cannot start streaming: not connected")
            return False

        for lane in range(self.config.lanes):
            self._lane_status[lane] = "streaming"

        self.is_streaming = True
        logger.info(f"D-PHY v2.5 streaming started at {self.config.aggregate_bandwidth_gbps} Gbps")
        return True

    def stop_streaming(self) -> bool:
        """
        Stop data streaming.

        Returns:
            True if streaming stopped successfully
        """
        self.is_streaming = False
        for lane in range(self.config.lanes):
            if self._lane_status[lane] == "streaming":
                self._lane_status[lane] = "ready"

        logger.info("D-PHY v2.5 streaming stopped")
        return True

    def send_packet(self, data: bytes, lane: Optional[int] = None) -> bool:
        """
        Send data packet through D-PHY interface.

        Args:
            data: Data to send
            lane: Specific lane to use (None for automatic distribution)

        Returns:
            True if packet sent successfully
        """
        if not self.is_connected:
            logger.error("Cannot send: not connected")
            return False

        try:
            # Apply scrambling if enabled
            if self.config.enable_scrambling:
                data = self._scramble_data(data)

            # Distribute across lanes if lane not specified
            if lane is None:
                lane = self._statistics.packets_sent % self.config.lanes

            # Simulate transmission with timing based on data rate
            transfer_time_ns = (len(data) * 8) / self.config.data_rate_gbps

            self._statistics.bytes_transferred += len(data)
            self._statistics.packets_sent += 1
            self._update_latency(transfer_time_ns)

            return True

        except Exception as e:
            lane_index = 0 if lane is None else lane
            self._statistics.lane_errors[lane_index] += 1
            logger.error(f"Send error on lane {lane_index}: {e}")
            return False

    def receive_packet(self, size: int) -> Optional[bytes]:
        """
        Receive data packet from D-PHY interface.

        Args:
            size: Number of bytes to receive

        Returns:
            Received data or None if unavailable
        """
        if not self.is_connected:
            return None

        try:
            # Simulate received data
            data = np.random.bytes(size)
            self._statistics.packets_received += 1
            self._statistics.bytes_transferred += size

            # Apply descrambling if enabled
            if self.config.enable_scrambling:
                data = self._descramble_data(data)

            return data

        except Exception as e:
            logger.error(f"Receive error: {e}")
            return None

    def get_statistics(self) -> TransferStatistics:
        """Get current transfer statistics."""
        return self._statistics

    def get_lane_status(self) -> dict[int, str]:
        """Get status of all lanes."""
        return self._lane_status.copy()

    def get_capabilities(self) -> dict[str, Any]:
        """
        Get D-PHY v2.5 capabilities.

        Returns:
            Dictionary of capability information
        """
        return {
            "version": "D-PHY v2.5",
            "max_data_rate_gbps": 4.5,
            "max_lanes": 4,
            "max_aggregate_bandwidth_gbps": 18.0,
            "supports_scrambling": True,
            "supports_equalization": True,
            "supports_de_emphasis": True,
            "clock_modes": ["continuous", "non-continuous"],
            "equalization_modes": [e.value for e in EqualizationMode],
        }

    def _train_lane(self, lane: int) -> bool:
        """
        Perform lane training sequence.

        Args:
            lane: Lane index to train

        Returns:
            True if training successful
        """
        logger.debug(f"Training lane {lane}...")

        # Simulate training sequence
        # In real hardware this would involve:
        # 1. LP-to-HS transition detection
        # 2. Bit synchronization
        # 3. Word alignment
        # 4. Deskew calibration

        training_success = True  # Simulated result

        if training_success:
            logger.debug(f"Lane {lane} training complete")

        return training_success

    def _calibrate_equalization(self) -> None:
        """Perform equalization calibration across all lanes."""
        logger.debug(f"Calibrating equalization (mode: {self.config.equalization.value})")

        for lane in range(self.config.lanes):
            if self.config.equalization == EqualizationMode.ADAPTIVE:
                # Simulate adaptive equalization
                self._equalization_state[lane] = self._calculate_optimal_eq(lane)
            else:
                # Fixed equalization level
                eq_levels = {
                    EqualizationMode.NONE: 0.0,
                    EqualizationMode.LOW: 2.0,
                    EqualizationMode.MEDIUM: 4.0,
                    EqualizationMode.HIGH: 6.0,
                }
                self._equalization_state[lane] = eq_levels.get(self.config.equalization, 0.0)

        logger.debug("Equalization calibration complete")

    def _calculate_optimal_eq(self, lane: int) -> float:
        """
        Calculate optimal equalization for a lane.

        Args:
            lane: Lane index

        Returns:
            Optimal equalization value in dB
        """
        # Equalization needs increase with data rate
        base_eq = (self.config.data_rate_gbps / 4.5) * 6.0
        # Add some lane-specific variation
        lane_variation = (lane % 2) * 0.5
        return base_eq + lane_variation

    def _scramble_data(self, data: bytes) -> bytes:
        """
        Apply scrambling to data.

        Args:
            data: Input data

        Returns:
            Scrambled data
        """
        # Simple XOR-based scrambling simulation
        # Real implementation would use LFSR-based polynomial scrambler
        scramble_key = 0xA5
        return bytes(b ^ scramble_key for b in data)

    def _descramble_data(self, data: bytes) -> bytes:
        """
        Remove scrambling from data.

        Args:
            data: Scrambled data

        Returns:
            Original data
        """
        # Descrambling is same as scrambling for XOR
        return self._scramble_data(data)

    def _update_latency(self, transfer_time_ns: float) -> None:
        """Update average latency statistics."""
        total_packets = self._statistics.packets_sent + self._statistics.packets_received
        if total_packets > 0:
            current_avg = self._statistics.average_latency_ns
            self._statistics.average_latency_ns = (current_avg * (total_packets - 1) + transfer_time_ns) / total_packets
