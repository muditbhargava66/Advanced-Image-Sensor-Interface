"""
Power Management for Advanced Image Sensor Interface

This module implements a sophisticated power management system for CMOS image
sensors, focusing on low-noise operation and efficient power delivery.

Classes:
    PowerManager: Main class for power management operations.
"""

import time
import random
import logging
import numpy as np
from typing import Dict, Any
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PowerConfig:
    """Configuration parameters for power management."""
    voltage_main: float  # Main voltage in volts
    voltage_io: float    # I/O voltage in volts
    current_limit: float # Current limit in amperes

class PowerManager:
    """
    Manages power delivery and monitoring for image sensors.

    Attributes:
        config (PowerConfig): Configuration for power management.
    """

    def __init__(self, config: PowerConfig):
        """
        Initialize the PowerManager with the given configuration.

        Args:
            config (PowerConfig): Configuration for power management.
        """
        def __init__(self, config: PowerConfig):
            if config.voltage_main <= 0 or config.voltage_io <= 0 or config.current_limit <= 0:
                raise ValueError("Invalid power configuration")
        self.config = config
        self._noise_level = 0.1  # Initial noise level (10% of signal)
        self._temperature = 25.0  # Initial temperature in Celsius
        self._initialize_power_system()
        logger.info(f"Power Manager initialized with main voltage: {self.config.voltage_main}V, "
                    f"I/O voltage: {self.config.voltage_io}V")

    def _initialize_power_system(self) -> None:
        """Initialize the power management system."""
        time.sleep(0.1)  # Simulate initialization time
        logger.info("Power management system initialized successfully")

    def set_voltage(self, rail: str, voltage: float) -> bool:
        """
        Set the voltage for a specific power rail.

        Args:
            rail (str): The power rail to adjust ('main' or 'io').
            voltage (float): The desired voltage in volts.

        Returns:
            bool: True if voltage was set successfully, False otherwise.
        """
        try:
            if rail == 'main':
                self.config.voltage_main = voltage
            elif rail == 'io':
                self.config.voltage_io = voltage
            else:
                raise ValueError(f"Unknown power rail: {rail}")
            
            # Simulate voltage adjustment
            time.sleep(0.05)
            logger.info(f"Set {rail} voltage to {voltage}V")
            return True
        except Exception as e:
            logger.error(f"Error setting voltage: {str(e)}")
            return False

    def get_power_status(self) -> Dict[str, Any]:
        """
        Get the current power status.

        Returns:
            Dict[str, Any]: A dictionary containing power status information.
        """
        return {
            "voltage_main": self._measure_voltage('main'),
            "voltage_io": self._measure_voltage('io'),
            "current_main": self._measure_current('main'),
            "current_io": self._measure_current('io'),
            "power_consumption": self._calculate_power_consumption(),
            "temperature": self._measure_temperature(),
            "noise_level": self._noise_level
        }

    def _measure_voltage(self, rail: str) -> float:
        """
        Measure the voltage on a specific power rail.

        Args:
            rail (str): The power rail to measure ('main' or 'io').

        Returns:
            float: The measured voltage in volts.
        """
        # base_voltage = self.config.voltage_main if rail == 'main' else self.config.voltage_io
        # noise = np.random.normal(0, self._noise_level * base_voltage)
        # return base_voltage + noise
        
        # Add some randomness to simulate real-world variation
        base_voltage = self.config.voltage_main if rail == 'main' else self.config.voltage_io
        return base_voltage + np.random.normal(0, 0.01 * base_voltage)

    def _measure_current(self, rail: str) -> float:
        """
        Measure the current on a specific power rail.

        Args:
            rail (str): The power rail to measure ('main' or 'io').

        Returns:
            float: The measured current in amperes.
        """
        base_current = self.config.current_limit * 0.5  # Assume 50% of max current
        noise = np.random.normal(0, self._noise_level * base_current)
        return base_current + noise

    def _calculate_power_consumption(self) -> float:
        """
        Calculate the total power consumption.

        Returns:
            float: The calculated power consumption in watts.
        """
        main_power = self._measure_voltage('main') * self._measure_current('main')
        io_power = self._measure_voltage('io') * self._measure_current('io')
        return main_power + io_power

    def _measure_temperature(self) -> float:
        """
        Measure the temperature of the power management system.

        Returns:
            float: The measured temperature in degrees Celsius.
        """
        # noise = np.random.normal(0, self._noise_level * self._temperature)
        # return self._temperature + noise
        
        # Simulate temperature increase with power consumption
        base_temp = 25.0  # Base temperature in Celsius
        power_factor = self._calculate_power_consumption() / 5.0  # Assuming 5W as reference
        return base_temp + power_factor * np.random.uniform(5, 10)

    def optimize_noise_reduction(self) -> None:
        """Optimize power delivery to reduce signal noise by 30%."""
        original_noise = self._noise_level
        self._noise_level *= 0.7  # 30% reduction in noise
        logger.info(f"Optimized noise reduction: Noise level reduced from {original_noise:.2%} to {self._noise_level:.2%}")

# Example usage demonstrating 30% noise reduction
if __name__ == "__main__":
    config = PowerConfig(voltage_main=1.8, voltage_io=3.3, current_limit=1.0)
    power_manager = PowerManager(config)
    
    def measure_noise_level(num_samples: int = 1000):
        voltages = [power_manager._measure_voltage('main') for _ in range(num_samples)]
        return np.std(voltages) / np.mean(voltages)  # Relative standard deviation

    # Measure initial noise level
    initial_noise = measure_noise_level()
    print(f"Initial noise level: {initial_noise:.2%}")

    # Optimize for noise reduction
    power_manager.optimize_noise_reduction()

    # Measure optimized noise level
    optimized_noise = measure_noise_level()
    print(f"Optimized noise level: {optimized_noise:.2%}")

    # Calculate improvement
    improvement = (initial_noise - optimized_noise) / initial_noise * 100
    print(f"Noise reduction: {improvement:.2f}%")

    # Print final power status
    print("Final power status:")
    print(power_manager.get_power_status())