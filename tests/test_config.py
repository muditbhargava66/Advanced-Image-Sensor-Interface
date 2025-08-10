"""
Tests for configuration management system.
"""

import os
from unittest.mock import patch

from advanced_image_sensor_interface.config import (
    ConfigManager,
    MIPIConfig,
    ProcessingConfig,
    SecurityConfig,
    TestingConfiguration,
    TimingConfig,
    get_config,
    get_mipi_config,
    get_processing_config,
    get_security_config,
    get_test_config,
    get_timing_config,
)


class TestTimingConfig:
    """Test timing configuration."""

    def test_default_values(self):
        """Test default timing configuration values."""
        config = TimingConfig()

        assert config.INIT_DELAY == 0.1
        assert config.DRIVER_INIT_DELAY == 0.1
        assert config.PIPELINE_INIT_DELAY == 0.1
        assert config.MAX_SIMULATION_TIME == 0.1
        assert config.MAX_TRANSMISSION_TIME == 0.1
        assert config.OPTIMIZATION_FACTOR_PRODUCTION == 0.8
        assert config.OPTIMIZATION_FACTOR_TESTING == 0.5
        assert config.DEFAULT_PROCESSING_TIME == 0.1
        assert config.TRANSFER_TIME_PER_MB == 0.1


class TestSecurityConfig:
    """Test security configuration."""

    def test_default_values(self):
        """Test default security configuration values."""
        config = SecurityConfig()

        assert config.MAX_IMAGE_SIZE == 100 * 1024 * 1024
        assert config.MAX_BUFFER_SIZE == 1024 * 1024 * 1024
        assert config.MAX_CONCURRENT_OPERATIONS == 10
        assert config.MAX_VOLTAGE == 5.0
        assert config.MIN_VOLTAGE == 0.0
        assert config.MAX_CURRENT == 10.0
        assert config.MAX_POWER == 50.0
        assert config.MAX_TEMPERATURE == 150.0
        assert config.MAX_FRAME_RATE == 1000.0
        assert config.MAX_DATA_RATE == 50.0
        assert config.OPERATION_TIMEOUT == 30.0
        assert config.TRANSMISSION_TIMEOUT == 10.0

    def test_dynamic_memory_limit(self):
        """Test dynamic memory limit calculation."""
        # Test that MAX_MEMORY_USAGE is set to a reasonable value
        config = SecurityConfig()

        # Should be either the default buffer size or a calculated value
        assert config.MAX_MEMORY_USAGE > 0
        assert config.MAX_MEMORY_USAGE <= config.MAX_BUFFER_SIZE * 2  # Reasonable upper bound


class TestProcessingConfig:
    """Test processing configuration."""

    def test_default_values(self):
        """Test default processing configuration values."""
        config = ProcessingConfig()

        assert config.NOISE_REDUCTION_SIGMA_MULTIPLIER == 2.0
        assert config.MIN_KERNEL_SIZE == 3
        assert config.KERNEL_SIZE_MULTIPLIER == 2
        assert config.NOISE_REDUCTION_IMPROVEMENT == 0.9
        assert config.DATA_RATE_IMPROVEMENT == 1.4
        assert config.MAX_IMAGE_DIMENSION == 16384
        assert config.MIN_IMAGE_DIMENSION == 1
        assert config.SUPPORTED_CHANNELS == (1, 3, 4)
        assert config.SUPPORTED_BIT_DEPTHS == (8, 10, 12, 14, 16)
        assert config.DEFAULT_BIT_DEPTH == 12


class TestMIPIConfig:
    """Test MIPI configuration."""

    def test_default_values(self):
        """Test default MIPI configuration values."""
        config = MIPIConfig()

        assert config.MIN_LANES == 1
        assert config.MAX_LANES == 4
        assert config.MIN_CHANNEL == 0
        assert config.MAX_CHANNEL == 3
        assert config.MIN_DATA_RATE == 0.1
        assert config.MAX_DATA_RATE == 50.0
        assert config.MIN_PACKET_SIZE == 4
        assert config.SMALL_PACKET_THRESHOLD == 4
        assert config.LARGE_PACKET_THRESHOLD == 64 * 1024
        assert config.MIN_ERROR_RATE == 0.0001
        assert config.MAX_ERROR_RATE == 0.001
        assert config.MIN_EFFICIENCY == 0.85
        assert config.MAX_EFFICIENCY == 0.95


class TestTestingConfiguration:
    """Test testing configuration."""

    def test_default_values(self):
        """Test default test configuration values."""
        config = TestingConfiguration()

        assert config.TEST_FRAME_COUNT == 100
        assert config.DEFAULT_TEST_WIDTH == 1920
        assert config.DEFAULT_TEST_HEIGHT == 1080
        assert config.DEFAULT_TEST_CHANNELS == 3
        assert config.PERFORMANCE_IMPROVEMENT_THRESHOLD == 0.8
        assert config.PERFORMANCE_DEGRADATION_THRESHOLD == 1.2
        assert config.TEST_TIMEOUT == 60.0
        assert config.OPERATION_TEST_TIMEOUT == 5.0


class TestConfigManager:
    """Test configuration manager."""

    def test_default_initialization(self):
        """Test default configuration manager initialization."""
        manager = ConfigManager()

        assert manager.environment == "production"
        assert isinstance(manager.timing, TimingConfig)
        assert isinstance(manager.security, SecurityConfig)
        assert isinstance(manager.processing, ProcessingConfig)
        assert isinstance(manager.mipi, MIPIConfig)
        assert isinstance(manager.testing, TestingConfiguration)

    def test_testing_environment_overrides(self):
        """Test testing environment configuration overrides."""
        manager = ConfigManager("testing")

        assert manager.environment == "testing"
        # Testing should use more aggressive optimization
        assert manager.timing.OPTIMIZATION_FACTOR_PRODUCTION == manager.timing.OPTIMIZATION_FACTOR_TESTING
        # Shorter timeout for tests
        assert manager.security.OPERATION_TIMEOUT == 5.0
        # Smaller image size limit for tests
        assert manager.security.MAX_IMAGE_SIZE == 10 * 1024 * 1024

    def test_development_environment_overrides(self):
        """Test development environment configuration overrides."""
        manager = ConfigManager("development")

        assert manager.environment == "development"
        # Longer timeout for debugging
        assert manager.security.OPERATION_TIMEOUT == 60.0
        # Larger image size for development
        assert manager.security.MAX_IMAGE_SIZE == 500 * 1024 * 1024

    def test_get_config_dict(self):
        """Test configuration dictionary export."""
        manager = ConfigManager()
        config_dict = manager.get_config_dict()

        assert "environment" in config_dict
        assert "timing" in config_dict
        assert "security" in config_dict
        assert "processing" in config_dict
        assert "mipi" in config_dict
        assert "testing" in config_dict

        assert config_dict["environment"] == "production"
        assert isinstance(config_dict["timing"], dict)

    def test_update_from_env(self):
        """Test configuration updates from environment variables."""
        with patch.dict(os.environ, {
            'AISI_INIT_DELAY': '0.2',
            'AISI_MAX_SIM_TIME': '0.5',
            'AISI_MAX_IMAGE_SIZE': '50000000',
            'AISI_OPERATION_TIMEOUT': '45.0',
            'AISI_NOISE_SIGMA_MULT': '3.0'
        }):
            manager = ConfigManager()
            manager.update_from_env()

            assert manager.timing.INIT_DELAY == 0.2
            assert manager.timing.MAX_SIMULATION_TIME == 0.5
            assert manager.security.MAX_IMAGE_SIZE == 50000000
            assert manager.security.OPERATION_TIMEOUT == 45.0
            assert manager.processing.NOISE_REDUCTION_SIGMA_MULTIPLIER == 3.0


class TestGlobalConfigFunctions:
    """Test global configuration access functions."""

    def test_get_config_singleton(self):
        """Test that get_config returns singleton instance."""
        # Clear any existing global instance
        import advanced_image_sensor_interface.config.constants as config_module
        config_module._config_manager = None

        config1 = get_config()
        config2 = get_config()

        assert config1 is config2
        assert config1.environment == "production"

    def test_get_config_with_environment(self):
        """Test get_config with specific environment."""
        # Clear any existing global instance
        import advanced_image_sensor_interface.config.constants as config_module
        config_module._config_manager = None

        config = get_config("testing")
        assert config.environment == "testing"

    @patch.dict(os.environ, {'AISI_ENVIRONMENT': 'development'})
    def test_get_config_from_env_var(self):
        """Test get_config reads environment from env var."""
        # Clear any existing global instance
        import advanced_image_sensor_interface.config.constants as config_module
        config_module._config_manager = None

        config = get_config()
        assert config.environment == "development"

    def test_convenience_functions(self):
        """Test convenience configuration access functions."""
        # Clear any existing global instance
        import advanced_image_sensor_interface.config.constants as config_module
        config_module._config_manager = None

        timing = get_timing_config()
        security = get_security_config()
        processing = get_processing_config()
        mipi = get_mipi_config()
        testing = get_test_config()

        assert isinstance(timing, TimingConfig)
        assert isinstance(security, SecurityConfig)
        assert isinstance(processing, ProcessingConfig)
        assert isinstance(mipi, MIPIConfig)
        assert isinstance(testing, TestingConfiguration)


class TestConfigIntegration:
    """Test configuration integration with other components."""

    def test_config_consistency(self):
        """Test that configuration values are consistent across components."""
        manager = ConfigManager()

        # MIPI config should be consistent
        assert manager.mipi.MIN_LANES >= 1
        assert manager.mipi.MAX_LANES <= 4
        assert manager.mipi.MIN_CHANNEL >= 0
        assert manager.mipi.MAX_CHANNEL <= 3

        # Security limits should be reasonable
        assert manager.security.MAX_IMAGE_SIZE > 0
        assert manager.security.MAX_BUFFER_SIZE >= manager.security.MAX_IMAGE_SIZE
        assert manager.security.MAX_VOLTAGE > manager.security.MIN_VOLTAGE
        assert manager.security.MAX_CURRENT > 0
        assert manager.security.MAX_POWER > 0

        # Processing config should be valid
        assert manager.processing.MIN_KERNEL_SIZE >= 1
        assert manager.processing.MIN_IMAGE_DIMENSION >= 1
        assert manager.processing.MAX_IMAGE_DIMENSION > manager.processing.MIN_IMAGE_DIMENSION
        assert len(manager.processing.SUPPORTED_CHANNELS) > 0
        assert len(manager.processing.SUPPORTED_BIT_DEPTHS) > 0

    def test_environment_specific_consistency(self):
        """Test that environment-specific overrides maintain consistency."""
        for env in ["production", "testing", "development"]:
            manager = ConfigManager(env)

            # All environments should have positive timeouts
            assert manager.security.OPERATION_TIMEOUT > 0
            assert manager.timing.INIT_DELAY >= 0
            assert manager.timing.MAX_SIMULATION_TIME > 0

            # Optimization factors should be between 0 and 1
            assert 0 < manager.timing.OPTIMIZATION_FACTOR_PRODUCTION <= 1
            assert 0 < manager.timing.OPTIMIZATION_FACTOR_TESTING <= 1
