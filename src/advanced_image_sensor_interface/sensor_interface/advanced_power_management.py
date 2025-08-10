"""Advanced Power Management for v2.0.0.

This module provides advanced power management capabilities including:
- Dynamic power state management
- Thermal monitoring and control
- Power consumption optimization
- Battery management (for mobile applications)
- Sleep/wake scheduling
- Performance vs power trade-offs
"""

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class PowerState(Enum):
    """System power states."""

    ACTIVE = "active"
    IDLE = "idle"
    STANDBY = "standby"
    SLEEP = "sleep"
    DEEP_SLEEP = "deep_sleep"
    HIBERNATE = "hibernate"
    OFF = "off"


class ThermalState(Enum):
    """Thermal management states."""

    NORMAL = "normal"
    WARM = "warm"
    HOT = "hot"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class PowerMode(Enum):
    """Power management modes."""

    PERFORMANCE = "performance"
    BALANCED = "balanced"
    POWER_SAVER = "power_saver"
    ULTRA_LOW_POWER = "ultra_low_power"
    CUSTOM = "custom"


@dataclass
class PowerConfiguration:
    """Advanced power management configuration."""

    # Power mode settings
    power_mode: PowerMode = PowerMode.BALANCED
    auto_power_management: bool = True

    # Thermal management
    enable_thermal_monitoring: bool = True
    thermal_update_interval_ms: float = 1000.0
    temperature_thresholds: dict[str, float] = field(
        default_factory=lambda: {
            "normal_max": 45.0,  # Celsius
            "warm_max": 60.0,
            "hot_max": 75.0,
            "critical_max": 85.0,
            "emergency_max": 95.0,
        }
    )

    # Power state transitions
    idle_timeout_ms: float = 5000.0
    standby_timeout_ms: float = 30000.0
    sleep_timeout_ms: float = 300000.0  # 5 minutes

    # Performance scaling
    enable_dynamic_frequency_scaling: bool = True
    min_frequency_mhz: float = 100.0
    max_frequency_mhz: float = 1000.0
    frequency_step_mhz: float = 50.0

    # Voltage scaling
    enable_dynamic_voltage_scaling: bool = True
    min_voltage_v: float = 0.8
    max_voltage_v: float = 1.2
    voltage_step_v: float = 0.05

    # Battery management (for mobile applications)
    enable_battery_monitoring: bool = False
    battery_low_threshold: float = 20.0  # Percentage
    battery_critical_threshold: float = 5.0

    # Component power control
    sensor_power_control: bool = True
    processing_unit_control: bool = True
    memory_power_control: bool = True
    io_power_control: bool = True

    def __post_init__(self):
        """Validate configuration."""
        if self.thermal_update_interval_ms < 100.0:
            raise ValueError("Thermal update interval must be at least 100ms")

        if self.min_frequency_mhz >= self.max_frequency_mhz:
            raise ValueError("Min frequency must be less than max frequency")

        if self.min_voltage_v >= self.max_voltage_v:
            raise ValueError("Min voltage must be less than max voltage")


@dataclass
class PowerMetrics:
    """Power consumption and performance metrics."""

    # Power consumption (Watts)
    total_power: float = 0.0
    sensor_power: float = 0.0
    processing_power: float = 0.0
    memory_power: float = 0.0
    io_power: float = 0.0

    # Thermal metrics
    temperature_celsius: float = 25.0
    thermal_state: ThermalState = ThermalState.NORMAL

    # Performance metrics
    current_frequency_mhz: float = 500.0
    current_voltage_v: float = 1.0
    cpu_utilization: float = 0.0
    memory_utilization: float = 0.0

    # Battery metrics (if applicable)
    battery_level: Optional[float] = None
    battery_voltage: Optional[float] = None
    battery_current: Optional[float] = None
    estimated_runtime_hours: Optional[float] = None

    # Efficiency metrics
    performance_per_watt: float = 0.0
    thermal_efficiency: float = 1.0


class AdvancedPowerManager:
    """Advanced power management system."""

    def __init__(self, config: Optional[PowerConfiguration] = None):
        """Initialize advanced power manager.

        Args:
            config: Power management configuration
        """
        self.config = config or PowerConfiguration()
        self.current_state = PowerState.ACTIVE
        self.current_mode = self.config.power_mode
        self.metrics = PowerMetrics()

        # Threading
        self.power_lock = threading.RLock()
        self.monitoring_active = False
        self.monitoring_thread: Optional[threading.Thread] = None

        # State tracking
        self.last_activity_time = time.time()
        self.state_transition_history: list[tuple[float, PowerState]] = []

        # Callbacks
        self.power_state_callback: Optional[Callable] = None
        self.thermal_alert_callback: Optional[Callable] = None
        self.battery_alert_callback: Optional[Callable] = None

        # Component controllers
        self.component_states: dict[str, bool] = {"sensors": True, "processing_unit": True, "memory": True, "io": True}

        logger.info(f"Advanced power manager initialized in {self.current_mode.value} mode")

    def start_monitoring(self) -> bool:
        """Start power monitoring and management.

        Returns:
            True if monitoring started successfully
        """
        if self.monitoring_active:
            logger.warning("Power monitoring already active")
            return True

        try:
            with self.power_lock:
                self.monitoring_active = True
                self.monitoring_thread = threading.Thread(target=self._monitoring_worker, name="PowerMonitoring", daemon=True)
                self.monitoring_thread.start()

                logger.info("Power monitoring started")
                return True

        except Exception as e:
            logger.error(f"Failed to start power monitoring: {e}")
            self.monitoring_active = False
            return False

    def stop_monitoring(self) -> bool:
        """Stop power monitoring and management.

        Returns:
            True if monitoring stopped successfully
        """
        if not self.monitoring_active:
            logger.warning("Power monitoring not active")
            return True

        try:
            with self.power_lock:
                self.monitoring_active = False

                # Wait for monitoring thread to finish
                if self.monitoring_thread and self.monitoring_thread.is_alive():
                    self.monitoring_thread.join(timeout=2.0)

                logger.info("Power monitoring stopped")
                return True

        except Exception as e:
            logger.error(f"Failed to stop power monitoring: {e}")
            return False

    def set_power_mode(self, mode: PowerMode) -> bool:
        """Set power management mode.

        Args:
            mode: Power mode to set

        Returns:
            True if mode set successfully
        """
        try:
            with self.power_lock:
                old_mode = self.current_mode
                self.current_mode = mode

                # Apply mode-specific settings
                self._apply_power_mode_settings(mode)

                logger.info(f"Power mode changed from {old_mode.value} to {mode.value}")
                return True

        except Exception as e:
            logger.error(f"Failed to set power mode: {e}")
            return False

    def transition_to_state(self, target_state: PowerState, force: bool = False) -> bool:
        """Transition to a specific power state.

        Args:
            target_state: Target power state
            force: Force transition even if not recommended

        Returns:
            True if transition successful
        """
        try:
            with self.power_lock:
                if not force and not self._can_transition_to_state(target_state):
                    logger.warning(f"Cannot transition to {target_state.value} from {self.current_state.value}")
                    return False

                old_state = self.current_state

                # Perform state transition
                if self._perform_state_transition(target_state):
                    self.current_state = target_state
                    self.state_transition_history.append((time.time(), target_state))

                    # Trigger callback if set
                    if self.power_state_callback:
                        self.power_state_callback(old_state, target_state)

                    logger.info(f"Power state transitioned from {old_state.value} to {target_state.value}")
                    return True
                else:
                    logger.error(f"Failed to perform state transition to {target_state.value}")
                    return False

        except Exception as e:
            logger.error(f"Power state transition failed: {e}")
            return False

    def register_activity(self) -> None:
        """Register system activity to prevent power state transitions."""
        with self.power_lock:
            self.last_activity_time = time.time()

            # Wake up from low power states if needed
            if self.current_state in [PowerState.STANDBY, PowerState.SLEEP]:
                self.transition_to_state(PowerState.ACTIVE)

    def get_power_metrics(self) -> PowerMetrics:
        """Get current power metrics.

        Returns:
            Current power metrics
        """
        with self.power_lock:
            # Update metrics
            self._update_power_metrics()
            return self.metrics

    def set_component_power(self, component: str, enabled: bool) -> bool:
        """Control power to specific components.

        Args:
            component: Component name ('sensors', 'processing_unit', 'memory', 'io')
            enabled: True to enable, False to disable

        Returns:
            True if power control successful
        """
        if component not in self.component_states:
            logger.error(f"Unknown component: {component}")
            return False

        try:
            with self.power_lock:
                old_state = self.component_states[component]
                self.component_states[component] = enabled

                # Apply component power control
                self._apply_component_power_control(component, enabled)

                logger.info(f"Component '{component}' power: {old_state} -> {enabled}")
                return True

        except Exception as e:
            logger.error(f"Failed to control component power: {e}")
            return False

    def optimize_for_workload(self, workload_type: str, **kwargs) -> bool:
        """Optimize power settings for specific workload.

        Args:
            workload_type: Type of workload ('streaming', 'processing', 'idle', etc.)
            **kwargs: Additional workload parameters

        Returns:
            True if optimization applied successfully
        """
        try:
            with self.power_lock:
                if workload_type == "streaming":
                    return self._optimize_for_streaming(**kwargs)
                elif workload_type == "processing":
                    return self._optimize_for_processing(**kwargs)
                elif workload_type == "idle":
                    return self._optimize_for_idle(**kwargs)
                elif workload_type == "burst":
                    return self._optimize_for_burst(**kwargs)
                else:
                    logger.warning(f"Unknown workload type: {workload_type}")
                    return False

        except Exception as e:
            logger.error(f"Workload optimization failed: {e}")
            return False

    def _monitoring_worker(self) -> None:
        """Background monitoring worker thread."""
        logger.info("Power monitoring worker started")

        while self.monitoring_active:
            try:
                with self.power_lock:
                    # Update metrics
                    self._update_power_metrics()

                    # Check thermal state
                    self._check_thermal_state()

                    # Check battery state (if enabled)
                    if self.config.enable_battery_monitoring:
                        self._check_battery_state()

                    # Auto power management
                    if self.config.auto_power_management:
                        self._auto_power_management()

                # Sleep for update interval
                time.sleep(self.config.thermal_update_interval_ms / 1000.0)

            except Exception as e:
                logger.error(f"Power monitoring error: {e}")
                time.sleep(1.0)  # Prevent tight error loop

        logger.info("Power monitoring worker stopped")

    def _apply_power_mode_settings(self, mode: PowerMode) -> None:
        """Apply settings for specific power mode."""
        if mode == PowerMode.PERFORMANCE:
            self._set_frequency_voltage(self.config.max_frequency_mhz, self.config.max_voltage_v)
            self._enable_all_components()

        elif mode == PowerMode.BALANCED:
            mid_freq = (self.config.min_frequency_mhz + self.config.max_frequency_mhz) / 2
            mid_voltage = (self.config.min_voltage_v + self.config.max_voltage_v) / 2
            self._set_frequency_voltage(mid_freq, mid_voltage)

        elif mode == PowerMode.POWER_SAVER:
            self._set_frequency_voltage(self.config.min_frequency_mhz, self.config.min_voltage_v)
            self._optimize_component_power()

        elif mode == PowerMode.ULTRA_LOW_POWER:
            self._set_frequency_voltage(self.config.min_frequency_mhz, self.config.min_voltage_v)
            self._disable_non_essential_components()

    def _can_transition_to_state(self, target_state: PowerState) -> bool:
        """Check if transition to target state is allowed."""
        # Define valid transitions
        valid_transitions = {
            PowerState.ACTIVE: [PowerState.IDLE, PowerState.STANDBY, PowerState.OFF],
            PowerState.IDLE: [PowerState.ACTIVE, PowerState.STANDBY, PowerState.SLEEP],
            PowerState.STANDBY: [PowerState.ACTIVE, PowerState.IDLE, PowerState.SLEEP],
            PowerState.SLEEP: [PowerState.ACTIVE, PowerState.DEEP_SLEEP, PowerState.HIBERNATE],
            PowerState.DEEP_SLEEP: [PowerState.ACTIVE, PowerState.HIBERNATE],
            PowerState.HIBERNATE: [PowerState.ACTIVE],
            PowerState.OFF: [PowerState.ACTIVE],
        }

        return target_state in valid_transitions.get(self.current_state, [])

    def _perform_state_transition(self, target_state: PowerState) -> bool:
        """Perform the actual state transition."""
        try:
            if target_state == PowerState.ACTIVE:
                return self._transition_to_active()
            elif target_state == PowerState.IDLE:
                return self._transition_to_idle()
            elif target_state == PowerState.STANDBY:
                return self._transition_to_standby()
            elif target_state == PowerState.SLEEP:
                return self._transition_to_sleep()
            elif target_state == PowerState.DEEP_SLEEP:
                return self._transition_to_deep_sleep()
            elif target_state == PowerState.HIBERNATE:
                return self._transition_to_hibernate()
            elif target_state == PowerState.OFF:
                return self._transition_to_off()
            else:
                return False

        except Exception as e:
            logger.error(f"State transition implementation failed: {e}")
            return False

    def _transition_to_active(self) -> bool:
        """Transition to active state."""
        # Enable all necessary components
        self._enable_all_components()

        # Set appropriate frequency/voltage for current mode
        self._apply_power_mode_settings(self.current_mode)

        return True

    def _transition_to_idle(self) -> bool:
        """Transition to idle state."""
        # Reduce frequency slightly
        current_freq = self.metrics.current_frequency_mhz
        new_freq = max(current_freq * 0.8, self.config.min_frequency_mhz)
        self._set_frequency_voltage(new_freq, None)

        return True

    def _transition_to_standby(self) -> bool:
        """Transition to standby state."""
        # Reduce frequency significantly
        self._set_frequency_voltage(self.config.min_frequency_mhz, self.config.min_voltage_v)

        # Disable non-essential components
        self.set_component_power("io", False)

        return True

    def _transition_to_sleep(self) -> bool:
        """Transition to sleep state."""
        # Minimal frequency/voltage
        self._set_frequency_voltage(self.config.min_frequency_mhz, self.config.min_voltage_v)

        # Disable most components
        self.set_component_power("processing_unit", False)
        self.set_component_power("io", False)

        return True

    def _transition_to_deep_sleep(self) -> bool:
        """Transition to deep sleep state."""
        # Disable all non-essential components
        self.set_component_power("sensors", False)
        self.set_component_power("processing_unit", False)
        self.set_component_power("io", False)

        return True

    def _transition_to_hibernate(self) -> bool:
        """Transition to hibernate state."""
        # Save state and disable almost everything
        self._save_system_state()

        # Disable all components except minimal memory
        for component in self.component_states:
            if component != "memory":
                self.set_component_power(component, False)

        return True

    def _transition_to_off(self) -> bool:
        """Transition to off state."""
        # Save state and disable everything
        self._save_system_state()

        # Disable all components
        for component in self.component_states:
            self.set_component_power(component, False)

        return True

    def _update_power_metrics(self) -> None:
        """Update current power consumption metrics."""
        # Simulate power measurements (in real implementation, read from hardware)
        base_power = 1.0  # Base system power in Watts

        # Calculate component power based on state
        sensor_power = 0.5 if self.component_states["sensors"] else 0.0
        processing_power = (
            (self.metrics.current_frequency_mhz / 1000.0) * 2.0 if self.component_states["processing_unit"] else 0.0
        )
        memory_power = 0.3 if self.component_states["memory"] else 0.0
        io_power = 0.2 if self.component_states["io"] else 0.0

        # Update metrics
        self.metrics.sensor_power = sensor_power
        self.metrics.processing_power = processing_power
        self.metrics.memory_power = memory_power
        self.metrics.io_power = io_power
        self.metrics.total_power = base_power + sensor_power + processing_power + memory_power + io_power

        # Simulate temperature based on power consumption
        self.metrics.temperature_celsius = 25.0 + (self.metrics.total_power * 10.0)

        # Calculate efficiency metrics
        if self.metrics.total_power > 0:
            self.metrics.performance_per_watt = self.metrics.current_frequency_mhz / self.metrics.total_power

        # Simulate utilization
        self.metrics.cpu_utilization = min(100.0, self.metrics.current_frequency_mhz / 10.0)
        self.metrics.memory_utilization = 30.0 + np.random.normal(0, 5)

    def _check_thermal_state(self) -> None:
        """Check and update thermal state."""
        temp = self.metrics.temperature_celsius
        old_state = self.metrics.thermal_state

        # Determine thermal state
        if temp <= self.config.temperature_thresholds["normal_max"]:
            self.metrics.thermal_state = ThermalState.NORMAL
        elif temp <= self.config.temperature_thresholds["warm_max"]:
            self.metrics.thermal_state = ThermalState.WARM
        elif temp <= self.config.temperature_thresholds["hot_max"]:
            self.metrics.thermal_state = ThermalState.HOT
        elif temp <= self.config.temperature_thresholds["critical_max"]:
            self.metrics.thermal_state = ThermalState.CRITICAL
        else:
            self.metrics.thermal_state = ThermalState.EMERGENCY

        # Handle thermal state changes
        if self.metrics.thermal_state != old_state:
            self._handle_thermal_state_change(old_state, self.metrics.thermal_state)

    def _handle_thermal_state_change(self, old_state: ThermalState, new_state: ThermalState) -> None:
        """Handle thermal state changes."""
        logger.info(f"Thermal state changed from {old_state.value} to {new_state.value}")

        # Apply thermal mitigation
        if new_state == ThermalState.HOT:
            # Reduce frequency by 20%
            new_freq = self.metrics.current_frequency_mhz * 0.8
            self._set_frequency_voltage(new_freq, None)

        elif new_state == ThermalState.CRITICAL:
            # Reduce frequency by 50%
            new_freq = self.metrics.current_frequency_mhz * 0.5
            self._set_frequency_voltage(new_freq, None)

        elif new_state == ThermalState.EMERGENCY:
            # Emergency shutdown
            logger.critical("Emergency thermal shutdown initiated")
            self.transition_to_state(PowerState.OFF, force=True)

        # Trigger callback
        if self.thermal_alert_callback:
            self.thermal_alert_callback(old_state, new_state)

    def _check_battery_state(self) -> None:
        """Check battery state (if applicable)."""
        if not self.config.enable_battery_monitoring:
            return

        # Simulate battery level (in real implementation, read from battery management system)
        if self.metrics.battery_level is None:
            self.metrics.battery_level = 100.0

        # Simulate battery drain
        power_drain = self.metrics.total_power / 10.0  # Simplified drain calculation
        self.metrics.battery_level = max(0.0, self.metrics.battery_level - power_drain * 0.01)

        # Check thresholds
        if self.metrics.battery_level <= self.config.battery_critical_threshold:
            logger.critical(f"Battery critical: {self.metrics.battery_level:.1f}%")
            if self.battery_alert_callback:
                self.battery_alert_callback("critical", self.metrics.battery_level)

        elif self.metrics.battery_level <= self.config.battery_low_threshold:
            logger.warning(f"Battery low: {self.metrics.battery_level:.1f}%")
            if self.battery_alert_callback:
                self.battery_alert_callback("low", self.metrics.battery_level)

    def _auto_power_management(self) -> None:
        """Automatic power management based on activity and conditions."""
        current_time = time.time()
        time_since_activity = (current_time - self.last_activity_time) * 1000  # Convert to ms

        # Auto state transitions based on inactivity
        if self.current_state == PowerState.ACTIVE:
            if time_since_activity > self.config.idle_timeout_ms:
                self.transition_to_state(PowerState.IDLE)

        elif self.current_state == PowerState.IDLE:
            if time_since_activity > self.config.standby_timeout_ms:
                self.transition_to_state(PowerState.STANDBY)

        elif self.current_state == PowerState.STANDBY:
            if time_since_activity > self.config.sleep_timeout_ms:
                self.transition_to_state(PowerState.SLEEP)

    def _set_frequency_voltage(self, frequency_mhz: Optional[float], voltage_v: Optional[float]) -> None:
        """Set system frequency and voltage."""
        if frequency_mhz is not None:
            self.metrics.current_frequency_mhz = np.clip(
                frequency_mhz, self.config.min_frequency_mhz, self.config.max_frequency_mhz
            )

        if voltage_v is not None:
            self.metrics.current_voltage_v = np.clip(voltage_v, self.config.min_voltage_v, self.config.max_voltage_v)

    def _enable_all_components(self) -> None:
        """Enable all system components."""
        for component in self.component_states:
            self.component_states[component] = True

    def _disable_non_essential_components(self) -> None:
        """Disable non-essential components."""
        self.component_states["io"] = False

    def _optimize_component_power(self) -> None:
        """Optimize component power for current workload."""
        # This would contain workload-specific optimizations
        pass

    def _apply_component_power_control(self, component: str, enabled: bool) -> None:
        """Apply power control to specific component."""
        # In real implementation, this would interface with hardware power controllers
        logger.debug(f"Component '{component}' power control: {enabled}")

    def _save_system_state(self) -> None:
        """Save current system state for hibernation/shutdown."""
        # In real implementation, this would save state to non-volatile memory
        logger.debug("System state saved")

    def _optimize_for_streaming(self, **kwargs) -> bool:
        """Optimize for streaming workload."""
        # Enable all sensors and processing
        self.set_component_power("sensors", True)
        self.set_component_power("processing_unit", True)
        self.set_component_power("io", True)

        # Set moderate frequency for sustained performance
        target_freq = self.config.max_frequency_mhz * 0.7
        self._set_frequency_voltage(target_freq, None)

        return True

    def _optimize_for_processing(self, **kwargs) -> bool:
        """Optimize for intensive processing workload."""
        # Maximum performance mode
        self._set_frequency_voltage(self.config.max_frequency_mhz, self.config.max_voltage_v)
        self._enable_all_components()

        return True

    def _optimize_for_idle(self, **kwargs) -> bool:
        """Optimize for idle workload."""
        # Minimum power consumption
        self._set_frequency_voltage(self.config.min_frequency_mhz, self.config.min_voltage_v)
        self._disable_non_essential_components()

        return True

    def _optimize_for_burst(self, **kwargs) -> bool:
        """Optimize for burst workload."""
        # High performance for short duration
        self._set_frequency_voltage(self.config.max_frequency_mhz, self.config.max_voltage_v)
        self._enable_all_components()

        return True


def create_power_config_for_automotive() -> PowerConfiguration:
    """Create power configuration optimized for automotive applications.

    Returns:
        PowerConfiguration for automotive use
    """
    return PowerConfiguration(
        power_mode=PowerMode.BALANCED,
        auto_power_management=True,
        enable_thermal_monitoring=True,
        thermal_update_interval_ms=500.0,
        temperature_thresholds={
            "normal_max": 50.0,
            "warm_max": 70.0,
            "hot_max": 85.0,
            "critical_max": 95.0,
            "emergency_max": 105.0,
        },
        enable_dynamic_frequency_scaling=True,
        max_frequency_mhz=1200.0,
        enable_battery_monitoring=False,  # Automotive typically has stable power
    )


def create_power_config_for_mobile() -> PowerConfiguration:
    """Create power configuration optimized for mobile applications.

    Returns:
        PowerConfiguration for mobile use
    """
    return PowerConfiguration(
        power_mode=PowerMode.POWER_SAVER,
        auto_power_management=True,
        enable_thermal_monitoring=True,
        thermal_update_interval_ms=2000.0,
        idle_timeout_ms=2000.0,
        standby_timeout_ms=10000.0,
        sleep_timeout_ms=60000.0,
        enable_battery_monitoring=True,
        battery_low_threshold=25.0,
        battery_critical_threshold=10.0,
        max_frequency_mhz=800.0,
    )
