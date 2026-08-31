"""
Tests for Advanced Imaging Features

Tests for:
- MIPI D-PHY v2.5 protocol support
- Data Integrity (CRC, FEC)
- Lens Correction (radial, tangential)
- Multi-System Power Management
"""

import numpy as np
import pytest


class TestMIPIDPHY25:
    """Tests for MIPI D-PHY v2.5 protocol support."""

    def test_config_validation(self):
        """Test configuration validation."""
        from advanced_image_sensor_interface.sensor_interface.protocol.mipi.v4_1 import DPHY25Config, EqualizationMode

        # Valid configuration
        config = DPHY25Config(lanes=4, data_rate_gbps=4.5)
        assert config.lanes == 4
        assert config.data_rate_gbps == 4.5
        assert config.aggregate_bandwidth_gbps == 18.0

        # Test equalization modes
        config = DPHY25Config(equalization=EqualizationMode.ADAPTIVE)
        assert config.equalization == EqualizationMode.ADAPTIVE

    def test_invalid_config(self):
        """Test invalid configuration raises errors."""
        from advanced_image_sensor_interface.sensor_interface.protocol.mipi.v4_1 import DPHY25Config

        with pytest.raises(ValueError):
            DPHY25Config(lanes=5)  # Max 4 lanes

        with pytest.raises(ValueError):
            DPHY25Config(data_rate_gbps=5.0)  # Max 4.5 Gbps

    def test_driver_lifecycle(self):
        """Test driver connect/disconnect."""
        from advanced_image_sensor_interface.sensor_interface.protocol.mipi.v4_1 import DPHY25Config, DPHY25Driver

        config = DPHY25Config(lanes=2, data_rate_gbps=2.5)
        driver = DPHY25Driver(config)

        assert not driver.is_connected
        assert driver.connect()
        assert driver.is_connected

        assert driver.start_streaming()
        assert driver.is_streaming

        assert driver.stop_streaming()
        assert not driver.is_streaming

        assert driver.disconnect()
        assert not driver.is_connected

    def test_data_transfer(self):
        """Test data send/receive."""
        from advanced_image_sensor_interface.sensor_interface.protocol.mipi.v4_1 import DPHY25Config, DPHY25Driver

        config = DPHY25Config(lanes=4, data_rate_gbps=4.5)
        driver = DPHY25Driver(config)
        driver.connect()

        # Send data
        data = b"test_data" * 100
        assert driver.send_packet(data)

        # Receive data
        received = driver.receive_packet(100)
        assert received is not None
        assert len(received) == 100

        # Check statistics
        stats = driver.get_statistics()
        assert stats.packets_sent >= 1
        assert stats.bytes_transferred > 0

        driver.disconnect()

    def test_dphy_version_detection(self):
        """Test D-PHY version detection based on data rate."""
        from advanced_image_sensor_interface.sensor_interface.protocol.mipi.v4_1 import DPHY25Config, DPHYVersion

        config_1 = DPHY25Config(data_rate_gbps=1.0)
        assert config_1.dphy_version == DPHYVersion.V1_0

        config_2 = DPHY25Config(data_rate_gbps=2.5)
        assert config_2.dphy_version == DPHYVersion.V2_0

        config_3 = DPHY25Config(data_rate_gbps=4.5)
        assert config_3.dphy_version == DPHYVersion.V2_5


class TestDataIntegrity:
    """Tests for Data Integrity module."""

    def test_crc_calculation(self):
        """Test CRC-32 calculation."""
        from advanced_image_sensor_interface.utils.data_integrity import CRCValidator

        validator = CRCValidator()
        data = b"Hello, World!"
        crc = validator.calculate_crc(data)

        assert isinstance(crc, int)
        assert crc == validator.calculate_crc(data)  # Deterministic

    def test_crc_append_verify(self):
        """Test CRC append and verify."""
        from advanced_image_sensor_interface.utils.data_integrity import CRCValidator

        validator = CRCValidator()
        original_data = b"Test data for CRC verification"

        # Append CRC
        data_with_crc = validator.append_crc(original_data)
        assert len(data_with_crc) == len(original_data) + 4

        # Verify CRC
        is_valid, extracted = validator.verify_crc(data_with_crc)
        assert is_valid
        assert extracted == original_data

    def test_crc_corruption_detection(self):
        """Test CRC detects corruption."""
        from advanced_image_sensor_interface.utils.data_integrity import CRCValidator

        validator = CRCValidator()
        data = b"Original data"
        data_with_crc = validator.append_crc(data)

        # Corrupt the data
        corrupted = bytearray(data_with_crc)
        corrupted[5] ^= 0xFF  # Flip bits

        is_valid, _ = validator.verify_crc(bytes(corrupted))
        assert not is_valid

    def test_fec_parity(self):
        """Test parity-based FEC."""
        from advanced_image_sensor_interface.utils.data_integrity import ErrorCorrectionMode, ForwardErrorCorrection

        fec = ForwardErrorCorrection(mode=ErrorCorrectionMode.PARITY)
        data = b"Test data"

        encoded = fec.encode(data)
        assert len(encoded) == len(data) + 1

        decoded, errors = fec.decode(encoded)
        assert decoded == data
        assert errors == 0

    def test_fec_reed_solomon(self):
        """Test Reed-Solomon FEC."""
        from advanced_image_sensor_interface.utils.data_integrity import ErrorCorrectionMode, ForwardErrorCorrection

        fec = ForwardErrorCorrection(mode=ErrorCorrectionMode.REED_SOLOMON)
        data = b"Test data for Reed-Solomon encoding"

        encoded = fec.encode(data)
        assert len(encoded) > len(data)

        decoded, errors = fec.decode(encoded)
        assert decoded == data

    def test_integrity_checker(self):
        """Test IntegrityChecker facade."""
        from advanced_image_sensor_interface.utils.data_integrity import IntegrityChecker, IntegrityCheckResult

        checker = IntegrityChecker()
        data = b"Important data that needs protection"

        # Protect data
        protected = checker.protect(data)
        assert len(protected) > len(data)

        # Verify data
        result, recovered, errors = checker.verify(protected)
        assert result == IntegrityCheckResult.VALID
        assert recovered == data


class TestLensCorrection:
    """Tests for Lens Correction module."""

    def test_lens_profile_defaults(self):
        """Test lens profile default values."""
        from advanced_image_sensor_interface.utils.lens_correction import LensProfile

        profile = LensProfile()
        assert profile.image_width == 1920
        assert profile.image_height == 1080
        assert profile.cx == 960.0
        assert profile.cy == 540.0

    def test_lens_profile_barrel(self):
        """Test barrel distortion profile."""
        from advanced_image_sensor_interface.utils.lens_correction import DistortionType, LensProfile

        profile = LensProfile.barrel_distortion(strength=0.2)
        assert profile.distortion_type == DistortionType.BARREL
        assert profile.k1 < 0

    def test_lens_profile_pincushion(self):
        """Test pincushion distortion profile."""
        from advanced_image_sensor_interface.utils.lens_correction import DistortionType, LensProfile

        profile = LensProfile.pincushion_distortion(strength=0.15)
        assert profile.distortion_type == DistortionType.PINCUSHION
        assert profile.k1 > 0

    def test_radial_correction(self):
        """Test radial distortion correction."""
        from advanced_image_sensor_interface.utils.lens_correction import LensProfile, RadialDistortionCorrector

        # Create test image
        test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

        profile = LensProfile(image_width=100, image_height=100, k1=-0.1)
        corrector = RadialDistortionCorrector(profile)

        result = corrector.correct(test_image)
        assert result.shape == test_image.shape
        assert result.dtype == test_image.dtype

    def test_lens_correction_pipeline(self):
        """Test complete lens correction pipeline."""
        from advanced_image_sensor_interface.utils.lens_correction import LensCorrectionPipeline, LensProfile

        # Create test image
        test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        test_image[40:60, 40:60] = 255  # White square in center

        profile = LensProfile(image_width=100, image_height=100, k1=-0.05, p1=0.001)
        pipeline = LensCorrectionPipeline(profile)

        result = pipeline.correct(test_image)
        assert result.success
        assert result.data.shape == test_image.shape
        assert result.metrics.processing_time_ms > 0
        assert result.pixels_corrected == 10000
        assert result.max_displacement >= 0
        assert result.radial_correction_applied
        assert result.tangential_correction_applied

    def test_standard_profiles(self):
        """Test pre-defined standard profiles."""
        from advanced_image_sensor_interface.utils.lens_correction import STANDARD_PROFILES

        assert "gopro_wide" in STANDARD_PROFILES
        assert "smartphone_wide" in STANDARD_PROFILES
        assert "dslr_kit" in STANDARD_PROFILES

        gopro = STANDARD_PROFILES["gopro_wide"]
        assert gopro.k1 < 0  # Wide angle has barrel distortion


class TestMultiSystemPower:
    """Tests for Multi-System Power Management."""

    def test_power_budget_allocator(self):
        """Test power budget allocation."""
        from advanced_image_sensor_interface.sensor_interface.advanced_power_management import (
            PowerBudget,
            PowerBudgetAllocator,
            SensorPowerProfile,
        )

        budget = PowerBudget(total_budget_watts=10.0, reserve_watts=2.0)
        allocator = PowerBudgetAllocator(budget)

        # Register sensors
        allocator.register_sensor(SensorPowerProfile(sensor_id="sensor_1", max_power_watts=3.0, priority=1))
        allocator.register_sensor(SensorPowerProfile(sensor_id="sensor_2", max_power_watts=3.0, priority=2))

        # Allocate power
        allocations = allocator.allocate_power()

        assert "sensor_1" in allocations
        assert "sensor_2" in allocations
        assert allocator.get_total_allocated() <= budget.total_budget_watts - budget.reserve_watts

    def test_priority_allocation(self):
        """Test priority-based power allocation."""
        from advanced_image_sensor_interface.sensor_interface.advanced_power_management import (
            PowerBudget,
            PowerBudgetAllocator,
            SensorPowerProfile,
        )

        budget = PowerBudget(total_budget_watts=5.0, reserve_watts=1.0, priority_based_allocation=True)
        allocator = PowerBudgetAllocator(budget)

        # High priority sensor
        allocator.register_sensor(SensorPowerProfile(sensor_id="primary", max_power_watts=3.0, priority=1))
        # Low priority sensor
        allocator.register_sensor(SensorPowerProfile(sensor_id="secondary", max_power_watts=3.0, priority=5))

        allocations = allocator.allocate_power()

        # Primary should get more power due to higher priority
        assert allocations["primary"] >= allocations["secondary"]

    def test_sensor_array_manager(self):
        """Test sensor array power manager."""
        from advanced_image_sensor_interface.sensor_interface.advanced_power_management import (
            PowerBudget,
            SensorArrayPowerManager,
            SensorPowerProfile,
        )

        budget = PowerBudget(total_budget_watts=15.0)
        manager = SensorArrayPowerManager(budget)

        # Add sensors
        manager.add_sensor("front_camera", SensorPowerProfile(sensor_id="front_camera", priority=1))
        manager.add_sensor("rear_camera", SensorPowerProfile(sensor_id="rear_camera", priority=2))

        # Get metrics
        metrics = manager.get_array_metrics()
        assert metrics["total_sensors"] == 2

    def test_synchronized_transition(self):
        """Test synchronized power state transition."""
        from advanced_image_sensor_interface.sensor_interface.advanced_power_management import (
            PowerState,
            SensorArrayPowerManager,
            SensorPowerProfile,
        )

        manager = SensorArrayPowerManager()

        manager.add_sensor("sensor_1", SensorPowerProfile(sensor_id="sensor_1"))
        manager.add_sensor("sensor_2", SensorPowerProfile(sensor_id="sensor_2"))

        # Synchronized transition
        results = manager.synchronized_transition(PowerState.IDLE)

        assert all(results.values())

    def test_cascading_power_down(self):
        """Test cascading power down based on priority."""
        from advanced_image_sensor_interface.sensor_interface.advanced_power_management import (
            SensorArrayPowerManager,
            SensorPowerProfile,
        )

        manager = SensorArrayPowerManager()

        # Add sensors with different priorities
        manager.add_sensor("critical", SensorPowerProfile(sensor_id="critical", priority=1, can_be_disabled=False))
        manager.add_sensor("important", SensorPowerProfile(sensor_id="important", priority=2))
        manager.add_sensor("optional", SensorPowerProfile(sensor_id="optional", priority=5))

        # Power down low priority sensors
        powered_down = manager.cascading_power_down(preserve_priority=2)

        assert "optional" in powered_down
        assert "critical" not in powered_down
        assert "important" not in powered_down

    def test_multi_sensor_config_factory(self):
        """Test multi-sensor configuration factory."""
        from advanced_image_sensor_interface.sensor_interface.advanced_power_management import (
            create_power_config_for_multi_sensor,
        )

        budget, config = create_power_config_for_multi_sensor()

        assert budget.total_budget_watts == 15.0
        assert budget.priority_based_allocation is True
        assert config.enable_thermal_monitoring is True
