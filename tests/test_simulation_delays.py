"""
Unit Tests for v3.2.0 Simulation Delay Configuration

Covers SimulationDelayConfig validation and verifies that the MIPI, GigE,
CoaXPress-12, and USB3 protocol drivers route simulated hardware operations
through the configured delays (time.sleep is recorded, never actually slept).

Usage:
    Run these tests using pytest:
    $ pytest tests/test_simulation_delays.py
"""

import pytest

from advanced_image_sensor_interface import SimulationDelayConfig
from advanced_image_sensor_interface.sensor_interface.protocol.coaxpress.cxp12 import CXP12Config, CXP12Driver
from advanced_image_sensor_interface.sensor_interface.protocol.gige.roce import GigERoCEDriver, RoCEConfig
from advanced_image_sensor_interface.sensor_interface.protocol.mipi.driver import MIPIConfig, MIPIProtocolDriver
from advanced_image_sensor_interface.sensor_interface.protocol.usb3.streaming import StreamConfig, USB3StreamingManager


@pytest.fixture
def sleep_calls(monkeypatch) -> list[float]:
    """Replace time.sleep with a recorder so tests never actually sleep."""
    calls: list[float] = []
    monkeypatch.setattr("time.sleep", calls.append)
    return calls


class TestSimulationDelayConfigValidation:
    """SimulationDelayConfig construction and arithmetic."""

    def test_defaults_are_non_negative(self):
        config = SimulationDelayConfig()
        for field_name, value in config.__dict__.items():
            if field_name.endswith("_delay"):
                assert value >= 0, field_name

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"device_discovery_delay": -0.1},
            {"frame_capture_delay": -0.001},
            {"register_write_delay": -1.0},
        ],
    )
    def test_negative_delay_rejected(self, kwargs):
        with pytest.raises(ValueError):
            SimulationDelayConfig(**kwargs)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"random_delay_range_ms": (-1.0, 5.0)},
            {"random_delay_range_ms": (5.0, 1.0)},
        ],
    )
    def test_invalid_random_range_rejected(self, kwargs):
        with pytest.raises(ValueError):
            SimulationDelayConfig(**kwargs)

    def test_get_random_delay_disabled(self):
        config = SimulationDelayConfig(enable_random_delays=False)
        assert config.get_random_delay() == 0.0

    def test_apply_delay_without_random_component(self):
        config = SimulationDelayConfig()
        assert config.apply_delay(0.5) == 0.5

    def test_random_delay_within_configured_range(self):
        config = SimulationDelayConfig(enable_random_delays=True, random_delay_range_ms=(1.0, 2.0))
        for _ in range(50):
            delay = config.get_random_delay()
            assert 0.001 <= delay <= 0.002

    def test_apply_delay_bounds_with_random_component(self):
        config = SimulationDelayConfig(enable_random_delays=True, random_delay_range_ms=(1.0, 2.0))
        for _ in range(50):
            total = config.apply_delay(0.1)
            assert 0.101 <= total <= 0.102


class TestDefaultDelayInjection:
    """Drivers without explicit delay configs get a default SimulationDelayConfig."""

    def test_mipi_default(self):
        assert isinstance(MIPIConfig().simulation_delays, SimulationDelayConfig)

    def test_gige_default(self):
        assert isinstance(RoCEConfig().simulation_delays, SimulationDelayConfig)

    def test_cxp12_default(self):
        assert isinstance(CXP12Config().simulation_delays, SimulationDelayConfig)

    def test_usb3_default(self):
        assert isinstance(StreamConfig().simulation_delays, SimulationDelayConfig)


class TestMIPIDelayWiring:
    """MIPI driver routes control transfers through configured delays."""

    def test_control_transfer_delays(self, sleep_calls):
        delays = SimulationDelayConfig(
            link_initialization_delay=0.00011,
            command_transfer_delay=0.00042,
            register_read_delay=0.00021,
        )
        driver = MIPIProtocolDriver(MIPIConfig(simulation_delays=delays))

        assert driver.connect()
        assert 0.00011 in sleep_calls  # lane/link initialization

        sleep_calls.clear()
        assert driver.send_data(b"register write")
        assert 0.00042 in sleep_calls

        sleep_calls.clear()
        assert driver.receive_data(8) is not None
        assert 0.00021 in sleep_calls


class TestGigEDelayWiring:
    """GigE RoCE driver routes transport operations through configured delays."""

    def test_transport_initialization_delay(self, sleep_calls):
        delays = SimulationDelayConfig(link_initialization_delay=0.00033)
        driver = GigERoCEDriver(RoCEConfig(simulation_delays=delays))

        assert driver.transport.initialize()
        assert 0.00033 in sleep_calls

    def test_connection_discovery_delay(self, sleep_calls):
        delays = SimulationDelayConfig(device_discovery_delay=0.00044)
        driver = GigERoCEDriver(RoCEConfig(simulation_delays=delays))

        assert driver.connect("192.168.1.100")
        assert 0.00044 in sleep_calls


class TestCXP12DelayWiring:
    """CoaXPress-12 driver routes discovery through configured delays."""

    def test_connect_discovery_delay(self, sleep_calls):
        delays = SimulationDelayConfig(device_discovery_delay=0.00031)
        driver = CXP12Driver(CXP12Config(simulation_delays=delays))

        assert driver.connect()
        assert 0.00031 in sleep_calls


class TestUSB3DelayWiring:
    """USB3 streaming manager routes buffer/stream operations through delays."""

    def test_streaming_lifecycle_delays(self, sleep_calls):
        delays = SimulationDelayConfig(
            buffer_allocation_delay=0.00051,
            streaming_setup_delay=0.00061,
            streaming_teardown_delay=0.00071,
        )
        manager = USB3StreamingManager(StreamConfig(simulation_delays=delays))

        assert manager.prepare(width=640, height=480, pixel_format="Mono8")
        assert 0.00051 in sleep_calls

        sleep_calls.clear()
        assert manager.start_streaming()
        assert 0.00061 in sleep_calls

        sleep_calls.clear()
        assert manager.stop_streaming()
        assert 0.00071 in sleep_calls
