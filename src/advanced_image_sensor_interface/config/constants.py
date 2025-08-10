"""
Configuration Constants for Advanced Image Sensor Interface

This module contains all configurable constants used throughout the system,
replacing hardcoded magic numbers with named constants for better maintainability.
"""

import os
from dataclasses import dataclass
from typing import Any


@dataclass
class TimingConfig:
    """Timing configuration constants."""

    # Initialization delays
    INIT_DELAY: float = 0.1
    DRIVER_INIT_DELAY: float = 0.1
    PIPELINE_INIT_DELAY: float = 0.1

    # Simulation time limits
    MAX_SIMULATION_TIME: float = 0.1
    MAX_TRANSMISSION_TIME: float = 0.1

    # Performance optimization factors
    OPTIMIZATION_FACTOR_PRODUCTION: float = 0.8
    OPTIMIZATION_FACTOR_TESTING: float = 0.5
    OPTIMIZATION_FACTOR_AGGRESSIVE: float = 0.5

    # Processing time defaults
    DEFAULT_PROCESSING_TIME: float = 0.1
    TRANSFER_TIME_PER_MB: float = 0.1


class SecurityConfig:
    """Security and validation configuration constants."""

    def __init__(self):
        """Initialize security configuration with dynamic limits."""
        # Buffer limits
        self.MAX_IMAGE_SIZE: int = 100 * 1024 * 1024  # 100MB
        self.MAX_BUFFER_SIZE: int = 1024 * 1024 * 1024  # 1GB
        self.MAX_CONCURRENT_OPERATIONS: int = 10

        # Dynamic limits based on system resources
        try:
            import psutil

            system_memory = psutil.virtual_memory().total
            # Use at most 25% of system memory
            self.MAX_MEMORY_USAGE = min(self.MAX_BUFFER_SIZE, system_memory // 4)
        except ImportError:
            self.MAX_MEMORY_USAGE = self.MAX_BUFFER_SIZE

        # Power limits
        self.MAX_VOLTAGE: float = 5.0
        self.MIN_VOLTAGE: float = 0.0
        self.MAX_CURRENT: float = 10.0
        self.MAX_POWER: float = 50.0  # 50W
        self.MAX_TEMPERATURE: float = 150.0  # 150°C

        # Communication limits
        self.MAX_FRAME_RATE: float = 1000.0  # 1000 fps
        self.MAX_DATA_RATE: float = 50.0  # 50 Gbps

        # Timeout settings
        self.OPERATION_TIMEOUT: float = 30.0  # 30 seconds
        self.TRANSMISSION_TIMEOUT: float = 10.0  # 10 seconds


@dataclass
class ProcessingConfig:
    """Signal processing configuration constants."""

    # Noise reduction parameters
    NOISE_REDUCTION_SIGMA_MULTIPLIER: float = 2.0  # Conservative approach
    MIN_KERNEL_SIZE: int = 3
    KERNEL_SIZE_MULTIPLIER: int = 2

    # Performance improvement factors
    NOISE_REDUCTION_IMPROVEMENT: float = 0.9  # 10% improvement
    DATA_RATE_IMPROVEMENT: float = 1.4  # 40% increase

    # Image processing limits
    MAX_IMAGE_DIMENSION: int = 16384
    MIN_IMAGE_DIMENSION: int = 1
    SUPPORTED_CHANNELS: tuple = (1, 3, 4)

    # Bit depth configurations
    SUPPORTED_BIT_DEPTHS: tuple = (8, 10, 12, 14, 16)
    DEFAULT_BIT_DEPTH: int = 12


@dataclass
class MIPIConfig:
    """MIPI protocol configuration constants."""

    # Lane configuration
    MIN_LANES: int = 1
    MAX_LANES: int = 4

    # Channel configuration
    MIN_CHANNEL: int = 0
    MAX_CHANNEL: int = 3

    # Data rate limits
    MIN_DATA_RATE: float = 0.1  # 0.1 Gbps
    MAX_DATA_RATE: float = 50.0  # 50 Gbps

    # Packet size limits
    MIN_PACKET_SIZE: int = 4
    SMALL_PACKET_THRESHOLD: int = 4
    LARGE_PACKET_THRESHOLD: int = 64 * 1024

    # Error rate simulation
    MIN_ERROR_RATE: float = 0.0001  # 0.01%
    MAX_ERROR_RATE: float = 0.001  # 0.1%

    # Efficiency factors
    MIN_EFFICIENCY: float = 0.85  # 85%
    MAX_EFFICIENCY: float = 0.95  # 95%


@dataclass
class TestingConfiguration:
    """Testing configuration constants."""

    # Test data generation
    TEST_FRAME_COUNT: int = 100
    DEFAULT_TEST_WIDTH: int = 1920
    DEFAULT_TEST_HEIGHT: int = 1080
    DEFAULT_TEST_CHANNELS: int = 3

    # Performance test thresholds
    PERFORMANCE_IMPROVEMENT_THRESHOLD: float = 0.8  # 20% improvement expected
    PERFORMANCE_DEGRADATION_THRESHOLD: float = 1.2  # 20% degradation allowed

    # Test timeouts
    TEST_TIMEOUT: float = 60.0  # 60 seconds
    OPERATION_TEST_TIMEOUT: float = 5.0  # 5 seconds


class ConfigManager:
    """Centralized configuration management."""

    def __init__(self, environment: str = "production"):
        """
        Initialize configuration manager.

        Args:
            environment: Environment type ("production", "testing", "development")
        """
        self.environment = environment
        self.timing = TimingConfig()
        self.security = SecurityConfig()
        self.processing = ProcessingConfig()
        self.mipi = MIPIConfig()
        self.testing = TestingConfiguration()

        # Apply environment-specific overrides
        self._apply_environment_overrides()

    def _apply_environment_overrides(self):
        """Apply environment-specific configuration overrides."""
        if self.environment == "testing":
            # More aggressive optimization for testing
            self.timing.OPTIMIZATION_FACTOR_PRODUCTION = self.timing.OPTIMIZATION_FACTOR_TESTING
            # Shorter timeouts for faster tests
            self.security.OPERATION_TIMEOUT = 5.0
            # Smaller limits for testing
            self.security.MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10MB for tests

        elif self.environment == "development":
            # More verbose logging and relaxed limits
            self.security.OPERATION_TIMEOUT = 60.0  # Longer timeout for debugging
            # Allow larger images for development
            self.security.MAX_IMAGE_SIZE = 500 * 1024 * 1024  # 500MB

    def get_config_dict(self) -> dict[str, Any]:
        """Get all configuration as a dictionary."""
        return {
            "environment": self.environment,
            "timing": self.timing.__dict__,
            "security": self.security.__dict__,
            "processing": self.processing.__dict__,
            "mipi": self.mipi.__dict__,
            "testing": self.testing.__dict__,
        }

    def update_from_env(self):
        """Update configuration from environment variables."""
        # Timing overrides
        if "AISI_INIT_DELAY" in os.environ:
            self.timing.INIT_DELAY = float(os.environ["AISI_INIT_DELAY"])

        if "AISI_MAX_SIM_TIME" in os.environ:
            self.timing.MAX_SIMULATION_TIME = float(os.environ["AISI_MAX_SIM_TIME"])

        # Security overrides
        if "AISI_MAX_IMAGE_SIZE" in os.environ:
            self.security.MAX_IMAGE_SIZE = int(os.environ["AISI_MAX_IMAGE_SIZE"])

        if "AISI_OPERATION_TIMEOUT" in os.environ:
            self.security.OPERATION_TIMEOUT = float(os.environ["AISI_OPERATION_TIMEOUT"])

        # Processing overrides
        if "AISI_NOISE_SIGMA_MULT" in os.environ:
            self.processing.NOISE_REDUCTION_SIGMA_MULTIPLIER = float(os.environ["AISI_NOISE_SIGMA_MULT"])


# Global configuration instance
_config_manager = None


def get_config(environment: str | None = None) -> ConfigManager:
    """
    Get the global configuration manager instance.

    Args:
        environment: Environment type (only used on first call)

    Returns:
        ConfigManager instance
    """
    global _config_manager
    if _config_manager is None:
        env = environment or os.environ.get("AISI_ENVIRONMENT", "production")
        _config_manager = ConfigManager(env)
        _config_manager.update_from_env()
    return _config_manager


# Convenience functions for common configurations
def get_timing_config() -> TimingConfig:
    """Get timing configuration."""
    return get_config().timing


def get_security_config() -> SecurityConfig:
    """Get security configuration."""
    return get_config().security


def get_processing_config() -> ProcessingConfig:
    """Get processing configuration."""
    return get_config().processing


def get_mipi_config() -> MIPIConfig:
    """Get MIPI configuration."""
    return get_config().mipi


def get_test_config() -> TestingConfiguration:
    """Get test configuration."""
    return get_config().testing
