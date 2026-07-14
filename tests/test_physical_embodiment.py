"""
Integration tests for Physical Embodiment Architecture.

Tests the complete hardware interface layer with mocked GPIO, servos,
LED matrix, battery monitor, and privacy switch. All hardware dependencies
are mocked so tests run on any platform.

Requirements: 18.2
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch, MagicMock, PropertyMock
from pathlib import Path
import tempfile

from core.physical_embodiment import (
    EmbeddedConfig,
    load_embedded_config,
    SensorHub,
    ActuatorHub,
    ProximityManager,
    GestureController,
    LEDExpressionDisplay,
    PowerManager,
    PrivacySwitchMonitor,
    PhysicalEmbodiment,
    EmbodimentState,
    PowerMode,
    PowerSource,
    LED_PATTERNS,
)
from core.event_bus import EventBus


# ─── Fixtures ────────────────────────────────────────────────────────────

@pytest.fixture
def config():
    """Create a default embedded config for testing."""
    return EmbeddedConfig()


@pytest.fixture
def event_bus():
    """Create an event bus with embodiment events registered."""
    bus = EventBus()
    events = [
        "proximity.detected",
        "touch.detected",
        "embodiment.started",
        "embodiment.stopped",
        "privacy.switch_activated",
        "privacy.switch_deactivated",
        "power.low_battery",
        "power.charging",
        "power.mode_changed",
        "vital_received",
    ]
    for event_type in events:
        bus.register_event_type(event_type)
    return bus


@pytest.fixture
def mock_gpio():
    """Mock RPi.GPIO module."""
    with patch("core.physical_embodiment.GPIO_AVAILABLE", True), \
         patch("core.physical_embodiment.GPIO") as mock_gpio_module:
        mock_gpio_module.BCM = 11
        mock_gpio_module.IN = 1
        mock_gpio_module.OUT = 0
        mock_gpio_module.HIGH = 1
        mock_gpio_module.LOW = 0
        mock_gpio_module.PUD_DOWN = 21
        mock_gpio_module.setmode = Mock()
        mock_gpio_module.setwarnings = Mock()
        mock_gpio_module.setup = Mock()
        mock_gpio_module.input = Mock(return_value=0)
        mock_gpio_module.output = Mock()
        mock_gpio_module.cleanup = Mock()
        yield mock_gpio_module


# ─── Embedded Config Tests (Task 18.1) ──────────────────────────────────

def test_embedded_config_defaults():
    """Test that EmbeddedConfig has sensible defaults."""
    config = EmbeddedConfig()
    assert config.target == "jetson_nano"
    assert config.quantization_mode == "INT8"
    assert config.max_concurrent_tasks == 4
    assert config.pir_sensor_pin == 17
    assert config.low_battery_threshold == 20.0
    assert config.deactivation_timeout_seconds == 300
    assert config.led_brightness == 0.5


def test_load_embedded_config_from_file():
    """Test loading embedded config from YAML file."""
    config = load_embedded_config(
        Path(__file__).resolve().parent.parent / "data" / "embedded.yaml"
    )
    assert config.target == "jetson_nano"
    assert config.quantization_mode == "INT8"
    assert config.pir_sensor_pin == 17
    assert config.max_concurrent_tasks == 4


def test_load_embedded_config_missing_file():
    """Test loading config from non-existent file returns defaults."""
    config = load_embedded_config(Path("/nonexistent/path/embedded.yaml"))
    assert config.target == "jetson_nano"
    assert config.quantization_mode == "INT8"


def test_load_embedded_config_empty_file():
    """Test loading config from empty YAML file returns defaults."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write("")
        f.flush()
        config = load_embedded_config(Path(f.name))
    assert config.target == "jetson_nano"


# ─── Sensor Hub Tests (Task 18.2) ───────────────────────────────────────

def test_sensor_hub_init_with_gpio(mock_gpio, config):
    """Test SensorHub initializes correctly with GPIO available."""
    hub = SensorHub(config)
    assert hub._initialized is True
    mock_gpio.setmode.assert_called_once()
    mock_gpio.setup.assert_called()


def test_sensor_hub_init_without_gpio(config):
    """Test SensorHub degrades gracefully without GPIO."""
    with patch("core.physical_embodiment.GPIO_AVAILABLE", False):
        hub = SensorHub(config)
        assert hub._initialized is False
        assert hub.read_pir() is False
        assert hub.read_touch() is False


def test_sensor_hub_read_pir_motion_detected(mock_gpio, config):
    """Test PIR sensor reads motion detected."""
    mock_gpio.input.return_value = 1
    hub = SensorHub(config)
    assert hub.read_pir() is True


def test_sensor_hub_read_pir_no_motion(mock_gpio, config):
    """Test PIR sensor reads no motion."""
    mock_gpio.input.return_value = 0
    hub = SensorHub(config)
    assert hub.read_pir() is False


def test_sensor_hub_read_touch_detected(mock_gpio, config):
    """Test touch sensor reads touch detected."""
    hub = SensorHub(config)
    # Override for touch pin specifically
    mock_gpio.input.return_value = 1
    assert hub.read_touch() is True


def test_sensor_hub_read_touch_not_detected(mock_gpio, config):
    """Test touch sensor reads no touch."""
    hub = SensorHub(config)
    mock_gpio.input.return_value = 0
    assert hub.read_touch() is False


def test_sensor_hub_gpio_error_handling(mock_gpio, config):
    """Test SensorHub handles GPIO read errors gracefully."""
    hub = SensorHub(config)
    mock_gpio.input.side_effect = RuntimeError("GPIO error")
    assert hub.read_pir() is False
    assert hub.read_touch() is False


def test_sensor_hub_cleanup(mock_gpio, config):
    """Test SensorHub cleanup releases GPIO."""
    hub = SensorHub(config)
    hub.cleanup()
    mock_gpio.cleanup.assert_called_once()


# ─── Actuator Hub Tests (Task 18.2) ─────────────────────────────────────

def test_actuator_hub_servo_simulation(config):
    """Test ActuatorHub servo works in simulation mode (no hardware)."""
    with patch("core.physical_embodiment.SERVO_AVAILABLE", False), \
         patch("core.physical_embodiment.LED_MATRIX_AVAILABLE", False), \
         patch("core.physical_embodiment.GPIO_AVAILABLE", False):
        hub = ActuatorHub(config)
        # Should succeed in simulation
        assert hub.set_servo(0, 45.0) is True


def test_actuator_hub_led_matrix_simulation(config):
    """Test ActuatorHub LED matrix works in simulation mode."""
    with patch("core.physical_embodiment.SERVO_AVAILABLE", False), \
         patch("core.physical_embodiment.LED_MATRIX_AVAILABLE", False), \
         patch("core.physical_embodiment.GPIO_AVAILABLE", False):
        hub = ActuatorHub(config)
        pattern = [[0] * 8 for _ in range(8)]
        assert hub.set_led_matrix(pattern) is True
        assert hub.clear_led_matrix() is True
        assert hub.set_led_brightness(0.5) is True


def test_actuator_hub_status_led(mock_gpio, config):
    """Test ActuatorHub can control status LEDs."""
    with patch("core.physical_embodiment.SERVO_AVAILABLE", False), \
         patch("core.physical_embodiment.LED_MATRIX_AVAILABLE", False):
        hub = ActuatorHub(config)
        assert hub.set_status_led(23, True) is True
        mock_gpio.output.assert_called()


def test_actuator_hub_play_sound_unavailable(config):
    """Test ActuatorHub handles missing aplay gracefully."""
    with patch("core.physical_embodiment.SERVO_AVAILABLE", False), \
         patch("core.physical_embodiment.LED_MATRIX_AVAILABLE", False), \
         patch("core.physical_embodiment.GPIO_AVAILABLE", False):
        hub = ActuatorHub(config)
        # aplay won't be found on Windows dev machines
        result = hub.play_sound("nonexistent.wav")
        assert result is False


# ─── Proximity Manager Tests (Task 18.3) ────────────────────────────────

@pytest.mark.asyncio
async def test_proximity_activation_on_motion(event_bus, config):
    """Test robot activates when motion detected."""
    mock_sensor = Mock(spec=SensorHub)
    mock_sensor.read_pir.return_value = True

    manager = ProximityManager(mock_sensor, config)
    assert manager.state == EmbodimentState.IDLE

    # Simulate enough consecutive detections
    manager.consecutive_detections = config.consecutive_detections_required
    await manager.activate()
    assert manager.state == EmbodimentState.ACTIVE


@pytest.mark.asyncio
async def test_proximity_deactivation_after_timeout(event_bus, config):
    """Test robot deactivates after idle timeout."""
    mock_sensor = Mock(spec=SensorHub)

    manager = ProximityManager(mock_sensor, config)
    manager.state = EmbodimentState.ACTIVE

    await manager.deactivate()
    assert manager.state == EmbodimentState.IDLE
    assert manager.consecutive_detections == 0


@pytest.mark.asyncio
async def test_proximity_state_transitions(event_bus, config):
    """Test complete state transition cycle: IDLE → ACTIVE → IDLE."""
    mock_sensor = Mock(spec=SensorHub)

    manager = ProximityManager(mock_sensor, config)

    # Start in IDLE
    assert manager.state == EmbodimentState.IDLE
    assert manager.is_active is False

    # Activate
    await manager.activate()
    assert manager.state == EmbodimentState.ACTIVE
    assert manager.is_active is True

    # Deactivate
    await manager.deactivate()
    assert manager.state == EmbodimentState.IDLE
    assert manager.is_active is False


@pytest.mark.asyncio
async def test_proximity_event_emission(event_bus, config):
    """Test proximity manager emits events on state changes."""
    mock_sensor = Mock(spec=SensorHub)
    events_received = []
    event_bus.on("proximity.detected", lambda e: events_received.append(e))

    manager = ProximityManager(mock_sensor, config)
    await manager.start_monitoring(event_bus)

    # Manually activate (bypasses polling loop)
    await manager.activate()

    assert len(events_received) == 1
    assert events_received[0].data["state"] == "active"

    # Deactivate
    await manager.deactivate()
    assert len(events_received) == 2
    assert events_received[1].data["state"] == "idle"

    await manager.stop_monitoring()


# ─── Gesture Controller Tests (Task 18.4) ───────────────────────────────

@pytest.mark.asyncio
async def test_gesture_nod(config):
    """Test nod gesture executes servo sequence."""
    with patch("core.physical_embodiment.SERVO_AVAILABLE", False), \
         patch("core.physical_embodiment.LED_MATRIX_AVAILABLE", False), \
         patch("core.physical_embodiment.GPIO_AVAILABLE", False):
        actuator = ActuatorHub(config)
        gesture = GestureController(actuator, config)
        result = await gesture.nod(count=1, amplitude=15.0)
        assert result is True


@pytest.mark.asyncio
async def test_gesture_tilt(config):
    """Test head tilt gesture."""
    with patch("core.physical_embodiment.SERVO_AVAILABLE", False), \
         patch("core.physical_embodiment.LED_MATRIX_AVAILABLE", False), \
         patch("core.physical_embodiment.GPIO_AVAILABLE", False):
        actuator = ActuatorHub(config)
        gesture = GestureController(actuator, config)

        result = await gesture.tilt("right", 15.0)
        assert result is True
        assert gesture.current_roll == 15.0

        result = await gesture.tilt("left", 15.0)
        assert result is True
        assert gesture.current_roll == -15.0


@pytest.mark.asyncio
async def test_gesture_look_at(config):
    """Test look_at positions head correctly."""
    with patch("core.physical_embodiment.SERVO_AVAILABLE", False), \
         patch("core.physical_embodiment.LED_MATRIX_AVAILABLE", False), \
         patch("core.physical_embodiment.GPIO_AVAILABLE", False):
        actuator = ActuatorHub(config)
        gesture = GestureController(actuator, config)

        result = await gesture.look_at(pan_angle=45.0, tilt_angle=10.0)
        assert result is True
        assert gesture.current_pan == 45.0
        assert gesture.current_tilt == 10.0


@pytest.mark.asyncio
async def test_gesture_look_at_clamps_to_limits(config):
    """Test look_at clamps angles to configured limits."""
    with patch("core.physical_embodiment.SERVO_AVAILABLE", False), \
         patch("core.physical_embodiment.LED_MATRIX_AVAILABLE", False), \
         patch("core.physical_embodiment.GPIO_AVAILABLE", False):
        actuator = ActuatorHub(config)
        gesture = GestureController(actuator, config)

        # Try to exceed pan limit
        result = await gesture.look_at(pan_angle=180.0, tilt_angle=-60.0)
        assert result is True
        assert gesture.current_pan == config.pan_max  # clamped to 90
        assert gesture.current_tilt == config.tilt_min  # clamped to -30


@pytest.mark.asyncio
async def test_gesture_shake_head(config):
    """Test head shake gesture."""
    with patch("core.physical_embodiment.SERVO_AVAILABLE", False), \
         patch("core.physical_embodiment.LED_MATRIX_AVAILABLE", False), \
         patch("core.physical_embodiment.GPIO_AVAILABLE", False):
        actuator = ActuatorHub(config)
        gesture = GestureController(actuator, config)
        result = await gesture.shake_head(count=1, amplitude=20.0)
        assert result is True


@pytest.mark.asyncio
async def test_gesture_reset_position(config):
    """Test reset_position returns servos to neutral."""
    with patch("core.physical_embodiment.SERVO_AVAILABLE", False), \
         patch("core.physical_embodiment.LED_MATRIX_AVAILABLE", False), \
         patch("core.physical_embodiment.GPIO_AVAILABLE", False):
        actuator = ActuatorHub(config)
        gesture = GestureController(actuator, config)

        # Move to some position
        await gesture.look_at(45.0, 15.0)
        assert gesture.current_pan == 45.0

        # Reset
        result = await gesture.reset_position()
        assert result is True
        assert gesture.current_pan == 0.0
        assert gesture.current_tilt == 0.0
        assert gesture.current_roll == 0.0


# ─── LED Expression Display Tests (Task 18.5) ───────────────────────────

def test_led_patterns_exist():
    """Test all expected LED patterns are defined."""
    expected = ["happy", "sad", "thinking", "listening", "speaking",
                "error", "idle", "greeting"]
    for name in expected:
        assert name in LED_PATTERNS
        pattern = LED_PATTERNS[name]
        assert len(pattern) == 8
        for row in pattern:
            assert len(row) == 8


def test_led_show_expression(config):
    """Test showing an expression on the LED matrix."""
    with patch("core.physical_embodiment.SERVO_AVAILABLE", False), \
         patch("core.physical_embodiment.LED_MATRIX_AVAILABLE", False), \
         patch("core.physical_embodiment.GPIO_AVAILABLE", False):
        actuator = ActuatorHub(config)
        display = LEDExpressionDisplay(actuator, config)

        result = display.show_expression("happy")
        assert result is True
        assert display.current_expression == "happy"


def test_led_show_expression_unknown(config):
    """Test showing an unknown expression returns False."""
    with patch("core.physical_embodiment.SERVO_AVAILABLE", False), \
         patch("core.physical_embodiment.LED_MATRIX_AVAILABLE", False), \
         patch("core.physical_embodiment.GPIO_AVAILABLE", False):
        actuator = ActuatorHub(config)
        display = LEDExpressionDisplay(actuator, config)

        result = display.show_expression("nonexistent_emotion")
        assert result is False


def test_led_show_status(config):
    """Test showing status indicators."""
    with patch("core.physical_embodiment.SERVO_AVAILABLE", False), \
         patch("core.physical_embodiment.LED_MATRIX_AVAILABLE", False), \
         patch("core.physical_embodiment.GPIO_AVAILABLE", False):
        actuator = ActuatorHub(config)
        display = LEDExpressionDisplay(actuator, config)

        for status in ["listening", "thinking", "speaking", "error", "idle"]:
            result = display.show_status(status)
            assert result is True


@pytest.mark.asyncio
async def test_led_animate(config):
    """Test LED animation sequence."""
    with patch("core.physical_embodiment.SERVO_AVAILABLE", False), \
         patch("core.physical_embodiment.LED_MATRIX_AVAILABLE", False), \
         patch("core.physical_embodiment.GPIO_AVAILABLE", False):
        actuator = ActuatorHub(config)
        display = LEDExpressionDisplay(actuator, config)

        result = await display.animate(
            ["happy", "thinking", "listening"],
            frame_duration=0.01  # fast for testing
        )
        assert result is True


def test_led_brightness(config):
    """Test LED brightness control."""
    with patch("core.physical_embodiment.SERVO_AVAILABLE", False), \
         patch("core.physical_embodiment.LED_MATRIX_AVAILABLE", False), \
         patch("core.physical_embodiment.GPIO_AVAILABLE", False):
        actuator = ActuatorHub(config)
        display = LEDExpressionDisplay(actuator, config)

        display.set_brightness(0.8)
        assert display.current_brightness == 0.8

        # Clamp to bounds
        display.set_brightness(1.5)
        assert display.current_brightness == 1.0

        display.set_brightness(-0.3)
        assert display.current_brightness == 0.0


def test_led_clear(config):
    """Test clearing the LED display."""
    with patch("core.physical_embodiment.SERVO_AVAILABLE", False), \
         patch("core.physical_embodiment.LED_MATRIX_AVAILABLE", False), \
         patch("core.physical_embodiment.GPIO_AVAILABLE", False):
        actuator = ActuatorHub(config)
        display = LEDExpressionDisplay(actuator, config)

        display.show_expression("happy")
        assert display.current_expression == "happy"

        result = display.clear()
        assert result is True
        assert display.current_expression is None


# ─── Power Manager Tests (Task 18.6) ────────────────────────────────────

def test_power_manager_default_state(config):
    """Test PowerManager initializes with full power mode."""
    with patch("core.physical_embodiment.INA219_AVAILABLE", False):
        pm = PowerManager(config)
        assert pm.mode == PowerMode.FULL
        assert pm.get_battery_level() == 100.0


@pytest.mark.asyncio
async def test_power_manager_low_power_mode(config):
    """Test entering and exiting low-power mode."""
    with patch("core.physical_embodiment.INA219_AVAILABLE", False):
        pm = PowerManager(config)

        result = await pm.enter_low_power_mode()
        assert result is True
        assert pm.mode == PowerMode.LOW_POWER

        result = await pm.exit_low_power_mode()
        assert result is True
        assert pm.mode == PowerMode.FULL


@pytest.mark.asyncio
async def test_power_manager_idempotent_mode_changes(config):
    """Test that entering the same mode twice is idempotent."""
    with patch("core.physical_embodiment.INA219_AVAILABLE", False):
        pm = PowerManager(config)

        # Already in FULL mode
        result = await pm.exit_low_power_mode()
        assert result is True
        assert pm.mode == PowerMode.FULL

        # Enter low power twice
        await pm.enter_low_power_mode()
        result = await pm.enter_low_power_mode()
        assert result is True
        assert pm.mode == PowerMode.LOW_POWER


def test_power_manager_charger_detection(config):
    """Test charger detection without hardware."""
    with patch("core.physical_embodiment.INA219_AVAILABLE", False):
        pm = PowerManager(config)
        # Without INA219, should return False
        assert pm.is_on_charger() is False


def test_power_manager_get_power_source(config):
    """Test getting power source without hardware."""
    with patch("core.physical_embodiment.INA219_AVAILABLE", False):
        pm = PowerManager(config)
        source = pm.get_power_source()
        assert source == "unknown"


@pytest.mark.asyncio
async def test_power_manager_low_battery_event(event_bus, config):
    """Test low battery event emission."""
    with patch("core.physical_embodiment.INA219_AVAILABLE", False):
        pm = PowerManager(config)
        # Simulate low battery
        pm._battery_level = 15.0

        events_received = []
        event_bus.on("power.low_battery", lambda e: events_received.append(e))

        # Manually invoke what the monitor loop does
        pm._event_bus = event_bus
        pm._running = True

        level = pm.get_battery_level()
        assert level == 15.0
        assert level <= config.low_battery_threshold

        # Emit manually (since we're not running the loop)
        await event_bus.emit("power.low_battery", {
            "level": level,
            "critical": False,
            "timestamp": datetime.now()
        })

        assert len(events_received) == 1
        assert events_received[0].data["level"] == 15.0


# ─── Privacy Switch Tests (Task 18.7) ───────────────────────────────────

@pytest.mark.asyncio
async def test_privacy_switch_activation(event_bus, config, mock_gpio):
    """Test privacy mode activation via switch."""
    with patch("core.physical_embodiment.SERVO_AVAILABLE", False), \
         patch("core.physical_embodiment.LED_MATRIX_AVAILABLE", False):
        sensor = SensorHub(config)
        actuator = ActuatorHub(config)
        monitor = PrivacySwitchMonitor(sensor, actuator, config)

        events_received = []
        event_bus.on("privacy.switch_activated", lambda e: events_received.append(e))

        await monitor.start_monitoring(event_bus)

        # Manually activate privacy
        await monitor._activate_privacy()

        assert monitor.is_privacy_mode() is True
        assert len(events_received) == 1
        assert events_received[0].data["camera_disabled"] is True
        assert events_received[0].data["microphone_disabled"] is True

        await monitor.stop_monitoring()


@pytest.mark.asyncio
async def test_privacy_switch_deactivation(event_bus, config, mock_gpio):
    """Test privacy mode deactivation via switch."""
    with patch("core.physical_embodiment.SERVO_AVAILABLE", False), \
         patch("core.physical_embodiment.LED_MATRIX_AVAILABLE", False):
        sensor = SensorHub(config)
        actuator = ActuatorHub(config)
        monitor = PrivacySwitchMonitor(sensor, actuator, config)

        events_received = []
        event_bus.on("privacy.switch_deactivated", lambda e: events_received.append(e))

        await monitor.start_monitoring(event_bus)

        # First activate
        await monitor._activate_privacy()
        assert monitor.is_privacy_mode() is True

        # Then deactivate
        await monitor._deactivate_privacy()
        assert monitor.is_privacy_mode() is False
        assert len(events_received) == 1
        assert events_received[0].data["camera_disabled"] is False

        await monitor.stop_monitoring()


@pytest.mark.asyncio
async def test_privacy_switch_led_control(event_bus, config, mock_gpio):
    """Test privacy LED is controlled on switch toggle."""
    with patch("core.physical_embodiment.SERVO_AVAILABLE", False), \
         patch("core.physical_embodiment.LED_MATRIX_AVAILABLE", False):
        sensor = SensorHub(config)
        actuator = ActuatorHub(config)
        monitor = PrivacySwitchMonitor(sensor, actuator, config)

        await monitor.start_monitoring(event_bus)

        # Activate privacy — should turn on privacy LED
        await monitor._activate_privacy()
        # Check GPIO output was called for privacy LED pin (pin 24)
        mock_gpio.output.assert_any_call(config.privacy_led_pin, mock_gpio.HIGH)

        await monitor.stop_monitoring()


# ─── PhysicalEmbodiment Integration Tests ────────────────────────────────

@pytest.mark.asyncio
async def test_physical_embodiment_lifecycle(event_bus, config):
    """Test full start/stop lifecycle of PhysicalEmbodiment."""
    with patch("core.physical_embodiment.GPIO_AVAILABLE", False), \
         patch("core.physical_embodiment.SERVO_AVAILABLE", False), \
         patch("core.physical_embodiment.LED_MATRIX_AVAILABLE", False), \
         patch("core.physical_embodiment.INA219_AVAILABLE", False):

        embodiment = PhysicalEmbodiment(event_bus, config)

        events_received = []
        event_bus.on("embodiment.started", lambda e: events_received.append(e))
        event_bus.on("embodiment.stopped", lambda e: events_received.append(e))

        # Start
        await embodiment.start()
        assert embodiment.is_running is True

        # Verify start event
        assert len(events_received) == 1
        assert events_received[0].data["target"] == "jetson_nano"

        # Stop
        await embodiment.stop()
        assert embodiment.is_running is False

        # Verify stop event
        assert len(events_received) == 2


@pytest.mark.asyncio
async def test_physical_embodiment_graceful_degradation(event_bus, config):
    """Test PhysicalEmbodiment works without any hardware libraries."""
    with patch("core.physical_embodiment.GPIO_AVAILABLE", False), \
         patch("core.physical_embodiment.SERVO_AVAILABLE", False), \
         patch("core.physical_embodiment.LED_MATRIX_AVAILABLE", False), \
         patch("core.physical_embodiment.INA219_AVAILABLE", False):

        embodiment = PhysicalEmbodiment(event_bus, config)

        # Should start without errors
        await embodiment.start()
        assert embodiment.is_running is True

        # Gestures should work in simulation
        result = await embodiment.gesture_controller.nod()
        assert result is True

        # LED display should work in simulation
        result = embodiment.led_display.show_expression("happy")
        assert result is True

        # Power should report defaults
        assert embodiment.power_manager.get_battery_level() == 100.0

        # Privacy should report not active
        assert embodiment.privacy_switch.is_privacy_mode() is False

        await embodiment.stop()


@pytest.mark.asyncio
async def test_physical_embodiment_event_bus_integration(event_bus, config):
    """Test PhysicalEmbodiment correctly integrates with event bus."""
    with patch("core.physical_embodiment.GPIO_AVAILABLE", False), \
         patch("core.physical_embodiment.SERVO_AVAILABLE", False), \
         patch("core.physical_embodiment.LED_MATRIX_AVAILABLE", False), \
         patch("core.physical_embodiment.INA219_AVAILABLE", False):

        embodiment = PhysicalEmbodiment(event_bus, config)

        # Collect all events
        all_events = []
        for event_type in ["embodiment.started", "embodiment.stopped",
                           "proximity.detected"]:
            event_bus.on(event_type, lambda e: all_events.append(e))

        await embodiment.start()

        # Trigger proximity activation
        await embodiment.proximity_manager.activate()

        # Verify events
        event_types = [e.event_type for e in all_events]
        assert "embodiment.started" in event_types
        assert "proximity.detected" in event_types

        await embodiment.stop()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
