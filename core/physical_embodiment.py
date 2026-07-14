"""
Physical Embodiment Architecture Module

This module provides the hardware interface layer for deploying Aegis on
embedded platforms (Jetson Nano, Raspberry Pi). It includes sensor input
(PIR, touch), actuator control (servos, LED matrix, speaker), proximity-based
activation, physical gestures, LED expression display, power management,
and a physical privacy switch.

All hardware dependencies use graceful degradation — if the required
libraries (RPi.GPIO, adafruit-servokit, etc.) are not installed, the
module can still be imported and used with mock/simulated hardware.

Requirements: 14.1, 14.2, 14.3, 14.4, 14.5, 14.6, 14.7, 14.8, 14.9,
              14.10, 14.11, 14.14, 14.15, 14.16
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, List, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum

import yaml

from core.event_bus import EventBus

logger = logging.getLogger(__name__)

# ─── Graceful Degradation Imports ────────────────────────────────────────────
# GPIO library (Raspberry Pi)
try:
    import RPi.GPIO as GPIO
    GPIO_AVAILABLE = True
except ImportError:
    GPIO_AVAILABLE = False
    GPIO = None

# Servo control library (PCA9685 via adafruit-servokit)
try:
    from adafruit_servokit import ServoKit
    SERVO_AVAILABLE = True
except ImportError:
    SERVO_AVAILABLE = False
    ServoKit = None

# LED matrix library (HT16K33 via adafruit)
try:
    from adafruit_ht16k33.matrix import Matrix8x8
    import board
    import busio
    LED_MATRIX_AVAILABLE = True
except ImportError:
    LED_MATRIX_AVAILABLE = False
    Matrix8x8 = None

# INA219 battery monitor
try:
    from ina219 import INA219
    INA219_AVAILABLE = True
except ImportError:
    INA219_AVAILABLE = False
    INA219 = None


# ─── Embedded Configuration (Task 18.1) ─────────────────────────────────────

@dataclass
class EmbeddedConfig:
    """Configuration for embedded deployment.

    Requirements: 14.1
    """
    # Platform
    target: str = "jetson_nano"
    architecture: str = "aarch64"
    cpu_cores: int = 4
    gpu_enabled: bool = True

    # Quantization
    quantization_mode: str = "INT8"
    tensorrt_enabled: bool = True

    # Resources
    max_concurrent_tasks: int = 4
    audio_buffer_size: int = 4096
    frame_buffer_size: int = 5
    max_memory_mb: int = 1024

    # GPIO Pins
    pir_sensor_pin: int = 17
    touch_sensor_pin: int = 27
    privacy_switch_pin: int = 22
    status_led_pin: int = 23
    privacy_led_pin: int = 24
    battery_led_pin: int = 25

    # Servos
    pca9685_address: int = 0x40
    servo_pan_channel: int = 0
    servo_tilt_channel: int = 1
    servo_roll_channel: int = 2
    pan_min: int = -90
    pan_max: int = 90
    tilt_min: int = -30
    tilt_max: int = 30
    roll_min: int = -20
    roll_max: int = 20
    servo_speed: int = 60
    servo_smoothing: float = 0.3

    # LED Matrix
    led_i2c_address: int = 0x70
    led_brightness: float = 0.5
    animation_fps: int = 10
    auto_dim_seconds: int = 300
    dim_brightness: float = 0.1

    # Power
    power_monitor_method: str = "ina219"
    ina219_address: int = 0x41
    low_battery_threshold: float = 20.0
    critical_battery_threshold: float = 5.0
    battery_check_interval: int = 60
    idle_timeout_seconds: int = 300

    # Proximity
    detection_range_meters: float = 2.0
    activation_delay_seconds: float = 1.0
    deactivation_timeout_seconds: int = 300
    proximity_poll_interval: float = 0.5
    consecutive_detections_required: int = 2

    # Speaker
    alsa_device: str = "default"
    default_volume: float = 0.7


def load_embedded_config(config_path: Optional[Path] = None) -> EmbeddedConfig:
    """
    Load embedded deployment configuration from YAML file.

    Args:
        config_path: Path to embedded.yaml. Defaults to data/embedded.yaml

    Returns:
        EmbeddedConfig populated from the YAML file

    Requirements: 14.1
    """
    if config_path is None:
        config_path = Path(__file__).resolve().parent.parent / "data" / "embedded.yaml"

    config = EmbeddedConfig()

    if not config_path.exists():
        logger.warning(f"Embedded config not found at {config_path}, using defaults")
        return config

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            raw = yaml.safe_load(f)

        if raw is None:
            logger.warning("Empty embedded config file, using defaults")
            return config

        # Map YAML sections to dataclass fields
        platform = raw.get("platform", {})
        config.target = platform.get("target", config.target)
        config.architecture = platform.get("architecture", config.architecture)
        config.cpu_cores = platform.get("cpu_cores", config.cpu_cores)
        config.gpu_enabled = platform.get("gpu_enabled", config.gpu_enabled)

        quant = raw.get("quantization", {})
        config.quantization_mode = quant.get("mode", config.quantization_mode)
        config.tensorrt_enabled = quant.get("tensorrt_enabled", config.tensorrt_enabled)

        resources = raw.get("resources", {})
        config.max_concurrent_tasks = resources.get("max_concurrent_tasks", config.max_concurrent_tasks)
        config.audio_buffer_size = resources.get("audio_buffer_size", config.audio_buffer_size)
        config.frame_buffer_size = resources.get("frame_buffer_size", config.frame_buffer_size)
        config.max_memory_mb = resources.get("max_memory_mb", config.max_memory_mb)

        gpio = raw.get("gpio", {})
        config.pir_sensor_pin = gpio.get("pir_sensor_pin", config.pir_sensor_pin)
        config.touch_sensor_pin = gpio.get("touch_sensor_pin", config.touch_sensor_pin)
        config.privacy_switch_pin = gpio.get("privacy_switch_pin", config.privacy_switch_pin)
        config.status_led_pin = gpio.get("status_led_pin", config.status_led_pin)
        config.privacy_led_pin = gpio.get("privacy_led_pin", config.privacy_led_pin)
        config.battery_led_pin = gpio.get("battery_led_pin", config.battery_led_pin)

        servos = raw.get("servos", {})
        config.pca9685_address = servos.get("pca9685_address", config.pca9685_address)
        channels = servos.get("channels", {})
        config.servo_pan_channel = channels.get("pan", config.servo_pan_channel)
        config.servo_tilt_channel = channels.get("tilt", config.servo_tilt_channel)
        config.servo_roll_channel = channels.get("roll", config.servo_roll_channel)
        limits = servos.get("limits", {})
        config.pan_min = limits.get("pan_min", config.pan_min)
        config.pan_max = limits.get("pan_max", config.pan_max)
        config.tilt_min = limits.get("tilt_min", config.tilt_min)
        config.tilt_max = limits.get("tilt_max", config.tilt_max)
        config.roll_min = limits.get("roll_min", config.roll_min)
        config.roll_max = limits.get("roll_max", config.roll_max)
        config.servo_speed = servos.get("speed", config.servo_speed)
        config.servo_smoothing = servos.get("smoothing", config.servo_smoothing)

        led = raw.get("led_matrix", {})
        config.led_i2c_address = led.get("i2c_address", config.led_i2c_address)
        config.led_brightness = led.get("brightness", config.led_brightness)
        config.animation_fps = led.get("animation_fps", config.animation_fps)
        config.auto_dim_seconds = led.get("auto_dim_seconds", config.auto_dim_seconds)
        config.dim_brightness = led.get("dim_brightness", config.dim_brightness)

        power = raw.get("power", {})
        config.power_monitor_method = power.get("monitor_method", config.power_monitor_method)
        config.ina219_address = power.get("ina219_address", config.ina219_address)
        config.low_battery_threshold = power.get("low_battery_threshold", config.low_battery_threshold)
        config.critical_battery_threshold = power.get("critical_battery_threshold", config.critical_battery_threshold)
        config.battery_check_interval = power.get("check_interval_seconds", config.battery_check_interval)
        config.idle_timeout_seconds = power.get("idle_timeout_seconds", config.idle_timeout_seconds)

        proximity = raw.get("proximity", {})
        config.detection_range_meters = proximity.get("detection_range_meters", config.detection_range_meters)
        config.activation_delay_seconds = proximity.get("activation_delay_seconds", config.activation_delay_seconds)
        config.deactivation_timeout_seconds = proximity.get("deactivation_timeout_seconds", config.deactivation_timeout_seconds)
        config.proximity_poll_interval = proximity.get("poll_interval_seconds", config.proximity_poll_interval)
        config.consecutive_detections_required = proximity.get("consecutive_detections_required", config.consecutive_detections_required)

        speaker = raw.get("speaker", {})
        config.alsa_device = speaker.get("alsa_device", config.alsa_device)
        config.default_volume = speaker.get("default_volume", config.default_volume)

        logger.info(f"Loaded embedded config: target={config.target}, quant={config.quantization_mode}")
        return config

    except Exception as e:
        logger.error(f"Error loading embedded config: {e}")
        return EmbeddedConfig()


# ─── Sensor Hub (Task 18.2) ──────────────────────────────────────────────────

class SensorHub:
    """
    Hardware sensor hub for reading PIR and touch sensors.

    Provides a unified interface for GPIO-based sensors with graceful
    degradation when GPIO libraries are unavailable.

    Requirements: 14.2, 14.3, 14.4
    """

    def __init__(self, config: EmbeddedConfig):
        """
        Initialize the sensor hub.

        Args:
            config: Embedded deployment configuration
        """
        self.config = config
        self._initialized = False
        self._gpio_available = GPIO_AVAILABLE

        if self._gpio_available:
            try:
                GPIO.setmode(GPIO.BCM)
                GPIO.setwarnings(False)

                # Set up PIR sensor pin as input
                GPIO.setup(config.pir_sensor_pin, GPIO.IN)
                # Set up touch sensor pin as input with pull-down
                GPIO.setup(config.touch_sensor_pin, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)

                self._initialized = True
                logger.info("SensorHub initialized with GPIO")
            except Exception as e:
                logger.error(f"Failed to initialize GPIO sensors: {e}")
                self._initialized = False
        else:
            logger.warning("GPIO not available — SensorHub running in simulation mode")

    def read_pir(self) -> bool:
        """
        Read the PIR motion sensor.

        Returns:
            True if motion detected, False otherwise

        Requirements: 14.3
        """
        if not self._initialized:
            return False

        try:
            return bool(GPIO.input(self.config.pir_sensor_pin))
        except Exception as e:
            logger.error(f"Error reading PIR sensor: {e}")
            return False

    def read_touch(self) -> bool:
        """
        Read the touch sensor.

        Returns:
            True if touch detected, False otherwise

        Requirements: 14.4
        """
        if not self._initialized:
            return False

        try:
            return bool(GPIO.input(self.config.touch_sensor_pin))
        except Exception as e:
            logger.error(f"Error reading touch sensor: {e}")
            return False

    def read_ambient_light(self) -> float:
        """
        Read ambient light level.

        Returns:
            Light level in lux (0.0 if unavailable)
        """
        # Ambient light sensor requires I2C ADC — placeholder
        return 0.0

    def cleanup(self):
        """Release GPIO resources."""
        if self._initialized and self._gpio_available:
            try:
                GPIO.cleanup()
                logger.info("SensorHub GPIO cleaned up")
            except Exception as e:
                logger.error(f"Error cleaning up GPIO: {e}")


# ─── Actuator Hub (Task 18.2) ────────────────────────────────────────────────

class ActuatorHub:
    """
    Hardware actuator hub for controlling servos, LED matrix, and speaker.

    Provides a unified interface for output devices with graceful degradation
    when hardware libraries are unavailable.

    Requirements: 14.5, 14.6, 14.7
    """

    def __init__(self, config: EmbeddedConfig):
        """
        Initialize the actuator hub.

        Args:
            config: Embedded deployment configuration
        """
        self.config = config
        self._servo_kit = None
        self._led_matrix = None
        self._gpio_available = GPIO_AVAILABLE

        # Initialize servos
        if SERVO_AVAILABLE:
            try:
                self._servo_kit = ServoKit(
                    channels=16,
                    address=config.pca9685_address
                )
                logger.info("ServoKit initialized")
            except Exception as e:
                logger.error(f"Failed to initialize ServoKit: {e}")
        else:
            logger.warning("ServoKit not available — servos disabled")

        # Initialize LED matrix
        if LED_MATRIX_AVAILABLE:
            try:
                i2c = busio.I2C(board.SCL, board.SDA)
                self._led_matrix = Matrix8x8(i2c, address=config.led_i2c_address)
                self._led_matrix.brightness = config.led_brightness
                logger.info("LED matrix initialized")
            except Exception as e:
                logger.error(f"Failed to initialize LED matrix: {e}")
        else:
            logger.warning("LED matrix library not available — LED display disabled")

        # Initialize status LEDs via GPIO
        if self._gpio_available:
            try:
                GPIO.setmode(GPIO.BCM)
                GPIO.setwarnings(False)
                GPIO.setup(config.status_led_pin, GPIO.OUT, initial=GPIO.LOW)
                GPIO.setup(config.privacy_led_pin, GPIO.OUT, initial=GPIO.LOW)
                GPIO.setup(config.battery_led_pin, GPIO.OUT, initial=GPIO.LOW)
                logger.info("Status LEDs initialized")
            except Exception as e:
                logger.error(f"Failed to initialize status LEDs: {e}")

    def set_servo(self, channel: int, angle: float) -> bool:
        """
        Set a servo to a specific angle.

        Args:
            channel: Servo channel on PCA9685
            angle: Target angle in degrees (mapped to 0-180 for the servo)

        Returns:
            True if successful, False otherwise

        Requirements: 14.5
        """
        if self._servo_kit is None:
            logger.debug(f"Servo not available — simulating channel={channel} angle={angle}")
            return True  # Succeed silently in simulation mode

        try:
            # Map signed angle (e.g. -90..+90) to servo range (0..180)
            servo_angle = angle + 90.0
            servo_angle = max(0.0, min(180.0, servo_angle))
            self._servo_kit.servo[channel].angle = servo_angle
            return True
        except Exception as e:
            logger.error(f"Error setting servo channel {channel} to {angle}°: {e}")
            return False

    def set_led_matrix(self, pattern: List[List[int]]) -> bool:
        """
        Display a pattern on the 8x8 LED matrix.

        Args:
            pattern: 8x8 list of 0/1 values (row-major)

        Returns:
            True if successful, False otherwise

        Requirements: 14.6
        """
        if self._led_matrix is None:
            logger.debug("LED matrix not available — simulating pattern display")
            return True

        try:
            self._led_matrix.fill(0)
            for row_idx, row in enumerate(pattern):
                for col_idx, pixel in enumerate(row):
                    if row_idx < 8 and col_idx < 8:
                        self._led_matrix[col_idx, row_idx] = 1 if pixel else 0
            self._led_matrix.show()
            return True
        except Exception as e:
            logger.error(f"Error setting LED matrix: {e}")
            return False

    def set_led_brightness(self, brightness: float) -> bool:
        """
        Set LED matrix brightness.

        Args:
            brightness: Brightness level 0.0 to 1.0

        Returns:
            True if successful, False otherwise
        """
        if self._led_matrix is None:
            return True

        try:
            self._led_matrix.brightness = max(0.0, min(1.0, brightness))
            return True
        except Exception as e:
            logger.error(f"Error setting LED brightness: {e}")
            return False

    def clear_led_matrix(self) -> bool:
        """
        Turn off all LEDs on the matrix.

        Returns:
            True if successful, False otherwise
        """
        if self._led_matrix is None:
            return True
        try:
            self._led_matrix.fill(0)
            self._led_matrix.show()
            return True
        except Exception as e:
            logger.error(f"Error clearing LED matrix: {e}")
            return False

    def set_status_led(self, pin: int, state: bool) -> bool:
        """
        Set a GPIO status LED on or off.

        Args:
            pin: GPIO pin number (BCM)
            state: True for on, False for off

        Returns:
            True if successful, False otherwise
        """
        if not self._gpio_available:
            return True

        try:
            GPIO.output(pin, GPIO.HIGH if state else GPIO.LOW)
            return True
        except Exception as e:
            logger.error(f"Error setting LED pin {pin}: {e}")
            return False

    def play_sound(self, filepath: str) -> bool:
        """
        Play a sound file through the speaker.

        Args:
            filepath: Path to the WAV audio file

        Returns:
            True if successful, False otherwise

        Requirements: 14.7
        """
        try:
            import subprocess
            result = subprocess.run(
                ["aplay", "-D", self.config.alsa_device, filepath],
                capture_output=True,
                timeout=30
            )
            return result.returncode == 0
        except FileNotFoundError:
            logger.warning("aplay not found — speaker playback unavailable")
            return False
        except Exception as e:
            logger.error(f"Error playing sound: {e}")
            return False

    def cleanup(self):
        """Release actuator resources."""
        if self._led_matrix is not None:
            try:
                self._led_matrix.fill(0)
                self._led_matrix.show()
            except Exception:
                pass

        if self._servo_kit is not None:
            # Move servos to neutral (90°) before release
            try:
                for ch in [self.config.servo_pan_channel,
                           self.config.servo_tilt_channel,
                           self.config.servo_roll_channel]:
                    self._servo_kit.servo[ch].angle = 90
            except Exception:
                pass


# ─── Proximity Manager (Task 18.3) ──────────────────────────────────────────

class EmbodimentState(Enum):
    """State machine for physical embodiment."""
    IDLE = "idle"
    ACTIVE = "active"
    LOW_POWER = "low_power"
    PRIVACY = "privacy"


class ProximityManager:
    """
    Proximity-based activation manager.

    Polls the PIR sensor to detect user presence and manages
    activation/deactivation state transitions.

    Requirements: 14.8, 14.9
    """

    def __init__(self, sensor_hub: SensorHub, config: EmbeddedConfig):
        """
        Initialize the proximity manager.

        Args:
            sensor_hub: SensorHub instance for PIR sensor access
            config: Embedded configuration
        """
        self.sensor_hub = sensor_hub
        self.config = config

        self.state = EmbodimentState.IDLE
        self.last_motion_time: Optional[datetime] = None
        self.consecutive_detections = 0
        self._running = False
        self._poll_task: Optional[asyncio.Task] = None

    @property
    def is_active(self) -> bool:
        """Check if the robot is in active state."""
        return self.state == EmbodimentState.ACTIVE

    async def start_monitoring(self, event_bus: EventBus):
        """
        Start proximity monitoring loop.

        Args:
            event_bus: EventBus for emitting proximity events
        """
        self._running = True
        self._event_bus = event_bus
        self._poll_task = asyncio.create_task(self._monitor_loop())
        logger.info("Proximity monitoring started")

    async def stop_monitoring(self):
        """Stop proximity monitoring."""
        self._running = False
        if self._poll_task:
            self._poll_task.cancel()
            try:
                await self._poll_task
            except asyncio.CancelledError:
                pass
        logger.info("Proximity monitoring stopped")

    async def _monitor_loop(self):
        """
        Main proximity monitoring loop.

        Polls the PIR sensor and manages state transitions:
        - IDLE → ACTIVE when motion detected (after consecutive_detections_required)
        - ACTIVE → IDLE when no motion for deactivation_timeout_seconds

        Requirements: 14.8, 14.9
        """
        while self._running:
            try:
                motion_detected = self.sensor_hub.read_pir()

                if motion_detected:
                    self.last_motion_time = datetime.now()
                    self.consecutive_detections += 1

                    if (self.state == EmbodimentState.IDLE and
                            self.consecutive_detections >= self.config.consecutive_detections_required):
                        await self.activate()
                else:
                    self.consecutive_detections = 0

                    # Check for deactivation timeout
                    if (self.state == EmbodimentState.ACTIVE and
                            self.last_motion_time is not None):
                        elapsed = (datetime.now() - self.last_motion_time).total_seconds()
                        if elapsed >= self.config.deactivation_timeout_seconds:
                            await self.deactivate()

                await asyncio.sleep(self.config.proximity_poll_interval)

            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.error(f"Error in proximity monitor: {e}")
                await asyncio.sleep(1)

    async def activate(self):
        """
        Activate the robot — transition from IDLE to ACTIVE.

        Requirements: 14.8
        """
        previous_state = self.state
        self.state = EmbodimentState.ACTIVE
        logger.info(f"Robot activated (was {previous_state.value})")

        if hasattr(self, '_event_bus'):
            await self._event_bus.emit("proximity.detected", {
                "state": "active",
                "previous_state": previous_state.value,
                "timestamp": datetime.now()
            })

    async def deactivate(self):
        """
        Deactivate the robot — transition from ACTIVE to IDLE.

        Requirements: 14.9
        """
        previous_state = self.state
        self.state = EmbodimentState.IDLE
        self.consecutive_detections = 0
        logger.info(f"Robot deactivated — entering idle (was {previous_state.value})")

        if hasattr(self, '_event_bus'):
            await self._event_bus.emit("proximity.detected", {
                "state": "idle",
                "previous_state": previous_state.value,
                "timestamp": datetime.now()
            })


# ─── Gesture Controller (Task 18.4) ─────────────────────────────────────────

class GestureController:
    """
    Physical gesture controller for head movements via servo motors.

    Provides high-level gesture methods (nod, tilt, shake, look_at) that
    translate into smooth servo motor sequences.

    Requirements: 14.10
    """

    def __init__(self, actuator_hub: ActuatorHub, config: EmbeddedConfig):
        """
        Initialize the gesture controller.

        Args:
            actuator_hub: ActuatorHub for servo control
            config: Embedded configuration
        """
        self.actuator_hub = actuator_hub
        self.config = config

        # Current positions
        self.current_pan = 0.0
        self.current_tilt = 0.0
        self.current_roll = 0.0

    async def _smooth_move(self, channel: int, start_angle: float, end_angle: float,
                           duration: float = 0.5) -> bool:
        """
        Smoothly move a servo from start to end angle over a duration.

        Args:
            channel: Servo channel
            start_angle: Starting angle in degrees
            end_angle: Ending angle in degrees
            duration: Movement duration in seconds

        Returns:
            True if successful
        """
        steps = max(1, int(duration * 20))  # 20 steps per second
        step_delay = duration / steps
        step_size = (end_angle - start_angle) / steps

        for i in range(steps + 1):
            angle = start_angle + step_size * i
            self.actuator_hub.set_servo(channel, angle)
            await asyncio.sleep(step_delay)

        return True

    async def nod(self, count: int = 1, amplitude: float = 15.0) -> bool:
        """
        Perform a nodding gesture (head moves up and down).

        Args:
            count: Number of nods
            amplitude: Nod angle in degrees

        Returns:
            True if successful

        Requirements: 14.10
        """
        tilt_channel = self.config.servo_tilt_channel

        for _ in range(count):
            # Tilt down
            await self._smooth_move(tilt_channel, self.current_tilt,
                                    -amplitude, 0.3)
            # Tilt back up
            await self._smooth_move(tilt_channel, -amplitude,
                                    self.current_tilt, 0.3)

        return True

    async def tilt(self, direction: str = "right", angle: float = 15.0) -> bool:
        """
        Tilt the head to one side.

        Args:
            direction: "left" or "right"
            angle: Tilt angle in degrees

        Returns:
            True if successful

        Requirements: 14.10
        """
        roll_channel = self.config.servo_roll_channel
        target = angle if direction == "right" else -angle

        # Clamp to limits
        target = max(self.config.roll_min, min(self.config.roll_max, target))

        await self._smooth_move(roll_channel, self.current_roll, target, 0.4)
        self.current_roll = target
        return True

    async def look_at(self, pan_angle: float, tilt_angle: float = 0.0) -> bool:
        """
        Move head to look at a specific direction.

        Args:
            pan_angle: Horizontal angle (-90 to +90, 0 = center)
            tilt_angle: Vertical angle (-30 to +30, 0 = center)

        Returns:
            True if successful

        Requirements: 14.10
        """
        pan_channel = self.config.servo_pan_channel
        tilt_channel = self.config.servo_tilt_channel

        # Clamp to limits
        pan_angle = max(self.config.pan_min, min(self.config.pan_max, pan_angle))
        tilt_angle = max(self.config.tilt_min, min(self.config.tilt_max, tilt_angle))

        # Move both axes simultaneously
        pan_task = self._smooth_move(pan_channel, self.current_pan, pan_angle, 0.5)
        tilt_task = self._smooth_move(tilt_channel, self.current_tilt, tilt_angle, 0.5)

        await asyncio.gather(pan_task, tilt_task)

        self.current_pan = pan_angle
        self.current_tilt = tilt_angle
        return True

    async def shake_head(self, count: int = 2, amplitude: float = 20.0) -> bool:
        """
        Shake the head (left-right motion for "no" gesture).

        Args:
            count: Number of shakes
            amplitude: Shake angle in degrees

        Returns:
            True if successful

        Requirements: 14.10
        """
        pan_channel = self.config.servo_pan_channel

        for _ in range(count):
            # Pan left
            await self._smooth_move(pan_channel, self.current_pan,
                                    -amplitude, 0.2)
            # Pan right
            await self._smooth_move(pan_channel, -amplitude,
                                    amplitude, 0.4)
            # Return to center
            await self._smooth_move(pan_channel, amplitude,
                                    self.current_pan, 0.2)

        return True

    async def reset_position(self) -> bool:
        """
        Return all servos to neutral position.

        Returns:
            True if successful
        """
        pan_ch = self.config.servo_pan_channel
        tilt_ch = self.config.servo_tilt_channel
        roll_ch = self.config.servo_roll_channel

        tasks = [
            self._smooth_move(pan_ch, self.current_pan, 0.0, 0.5),
            self._smooth_move(tilt_ch, self.current_tilt, 0.0, 0.5),
            self._smooth_move(roll_ch, self.current_roll, 0.0, 0.5),
        ]
        await asyncio.gather(*tasks)

        self.current_pan = 0.0
        self.current_tilt = 0.0
        self.current_roll = 0.0
        return True


# ─── LED Expression Display (Task 18.5) ─────────────────────────────────────

# Pre-defined 8x8 LED patterns for emotions and status indicators
LED_PATTERNS: Dict[str, List[List[int]]] = {
    "happy": [
        [0, 0, 1, 1, 1, 1, 0, 0],
        [0, 1, 0, 0, 0, 0, 1, 0],
        [1, 0, 1, 0, 0, 1, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 1, 0, 0, 1, 0, 1],
        [1, 0, 0, 1, 1, 0, 0, 1],
        [0, 1, 0, 0, 0, 0, 1, 0],
        [0, 0, 1, 1, 1, 1, 0, 0],
    ],
    "sad": [
        [0, 0, 1, 1, 1, 1, 0, 0],
        [0, 1, 0, 0, 0, 0, 1, 0],
        [1, 0, 1, 0, 0, 1, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 1, 1, 0, 0, 1],
        [1, 0, 1, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 0, 1, 0],
        [0, 0, 1, 1, 1, 1, 0, 0],
    ],
    "thinking": [
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 0, 0, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
    ],
    "listening": [
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 0, 0, 1, 1, 0],
        [0, 1, 1, 0, 0, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 1, 1, 0, 0],
        [0, 0, 1, 1, 1, 1, 0, 0],
        [0, 0, 0, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
    ],
    "speaking": [
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 0, 0, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 1, 0],
        [1, 0, 0, 0, 0, 0, 0, 1],
        [0, 1, 1, 1, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
    ],
    "error": [
        [1, 0, 0, 0, 0, 0, 0, 1],
        [0, 1, 0, 0, 0, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0, 0],
        [0, 0, 0, 1, 1, 0, 0, 0],
        [0, 0, 0, 1, 1, 0, 0, 0],
        [0, 0, 1, 0, 0, 1, 0, 0],
        [0, 1, 0, 0, 0, 0, 1, 0],
        [1, 0, 0, 0, 0, 0, 0, 1],
    ],
    "idle": [
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 1, 1, 0, 0],
        [0, 0, 1, 1, 1, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 1, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
    ],
    "greeting": [
        [0, 0, 1, 1, 1, 1, 0, 0],
        [0, 1, 0, 0, 0, 0, 1, 0],
        [1, 0, 1, 0, 0, 1, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 1, 1, 1, 1, 0, 1],
        [0, 1, 0, 0, 0, 0, 1, 0],
        [0, 0, 1, 1, 1, 1, 0, 0],
    ],
}


class LEDExpressionDisplay:
    """
    LED expression display for showing emotions and status on an 8x8 matrix.

    Requirements: 14.11, 14.5
    """

    def __init__(self, actuator_hub: ActuatorHub, config: EmbeddedConfig):
        """
        Initialize the LED expression display.

        Args:
            actuator_hub: ActuatorHub for LED matrix control
            config: Embedded configuration
        """
        self.actuator_hub = actuator_hub
        self.config = config
        self.current_expression: Optional[str] = None
        self.current_brightness = config.led_brightness

    def show_expression(self, emotion: str) -> bool:
        """
        Display a facial expression pattern for the given emotion.

        Args:
            emotion: Emotion name (happy, sad, thinking, listening, etc.)

        Returns:
            True if successful, False if emotion not recognized

        Requirements: 14.11
        """
        pattern = LED_PATTERNS.get(emotion)
        if pattern is None:
            logger.warning(f"Unknown emotion for LED display: {emotion}")
            return False

        self.current_expression = emotion
        return self.actuator_hub.set_led_matrix(pattern)

    def show_status(self, status: str) -> bool:
        """
        Display a status indicator.

        Args:
            status: Status name (listening, thinking, speaking, error)

        Returns:
            True if successful

        Requirements: 14.5
        """
        # Map status names to LED patterns
        status_map = {
            "listening": "listening",
            "thinking": "thinking",
            "speaking": "speaking",
            "error": "error",
            "idle": "idle",
            "greeting": "greeting",
        }

        pattern_name = status_map.get(status, "idle")
        return self.show_expression(pattern_name)

    async def animate(self, pattern_sequence: List[str], frame_duration: float = 0.5) -> bool:
        """
        Animate through a sequence of patterns.

        Args:
            pattern_sequence: List of emotion/status names to display
            frame_duration: Duration of each frame in seconds

        Returns:
            True if successful

        Requirements: 14.11
        """
        for pattern_name in pattern_sequence:
            success = self.show_expression(pattern_name)
            if not success:
                return False
            await asyncio.sleep(frame_duration)

        return True

    def set_brightness(self, brightness: float) -> bool:
        """
        Set LED matrix brightness.

        Args:
            brightness: Brightness level 0.0 to 1.0

        Returns:
            True if successful
        """
        self.current_brightness = max(0.0, min(1.0, brightness))
        return self.actuator_hub.set_led_brightness(self.current_brightness)

    def clear(self) -> bool:
        """
        Turn off all LEDs.

        Returns:
            True if successful
        """
        self.current_expression = None
        return self.actuator_hub.clear_led_matrix()


# ─── Power Manager (Task 18.6) ──────────────────────────────────────────────

class PowerSource(Enum):
    """Power source type."""
    BATTERY = "battery"
    CHARGER = "charger"
    DOCK = "dock"
    UNKNOWN = "unknown"


class PowerMode(Enum):
    """Power mode."""
    FULL = "full"
    LOW_POWER = "low_power"


class PowerManager:
    """
    Power management for embedded deployment.

    Monitors battery level, manages low-power mode, and detects
    charging dock connection.

    Requirements: 14.14, 14.15
    """

    def __init__(self, config: EmbeddedConfig):
        """
        Initialize the power manager.

        Args:
            config: Embedded configuration
        """
        self.config = config
        self._ina219 = None
        self._mode = PowerMode.FULL
        self._running = False
        self._monitor_task: Optional[asyncio.Task] = None
        self._battery_level = 100.0
        self._power_source = PowerSource.UNKNOWN

        # Initialize battery monitor
        if INA219_AVAILABLE and config.power_monitor_method == "ina219":
            try:
                self._ina219 = INA219(
                    shunt_ohms=0.1,
                    address=config.ina219_address
                )
                self._ina219.configure()
                logger.info("INA219 battery monitor initialized")
            except Exception as e:
                logger.error(f"Failed to initialize INA219: {e}")
        else:
            logger.warning("Battery monitor not available — using simulated values")

    @property
    def mode(self) -> PowerMode:
        """Current power mode."""
        return self._mode

    def get_battery_level(self) -> float:
        """
        Get current battery level.

        Returns:
            Battery level as percentage (0-100)

        Requirements: 14.14
        """
        if self._ina219 is not None:
            try:
                # Read voltage and estimate battery percentage
                # Typical Li-Po: 3.0V (empty) to 4.2V (full) per cell
                voltage = self._ina219.voltage()
                # Assuming single-cell Li-Po (3.0-4.2V range)
                level = ((voltage - 3.0) / (4.2 - 3.0)) * 100.0
                self._battery_level = max(0.0, min(100.0, level))
            except Exception as e:
                logger.error(f"Error reading battery level: {e}")

        return self._battery_level

    def get_power_source(self) -> str:
        """
        Get current power source.

        Returns:
            Power source string: "battery", "charger", or "dock"

        Requirements: 14.15
        """
        if self._ina219 is not None:
            try:
                current = self._ina219.current()
                # If current is negative, we're charging
                if current < -50:  # mA threshold for charging detection
                    self._power_source = PowerSource.CHARGER
                else:
                    self._power_source = PowerSource.BATTERY
            except Exception:
                pass

        return self._power_source.value

    def is_on_charger(self) -> bool:
        """
        Check if device is on charger/dock.

        Returns:
            True if charging, False otherwise

        Requirements: 14.15
        """
        source = self.get_power_source()
        return source in ("charger", "dock")

    async def enter_low_power_mode(self) -> bool:
        """
        Enter low-power mode to conserve battery.

        Reduces CPU frequency, disables camera, and dims LEDs.

        Returns:
            True if successful

        Requirements: 14.14
        """
        if self._mode == PowerMode.LOW_POWER:
            return True

        self._mode = PowerMode.LOW_POWER
        logger.info("Entering low-power mode")

        # Reduce CPU frequency via sysfs (Linux only)
        try:
            cpu_freq_path = Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq")
            if cpu_freq_path.exists():
                target_khz = self.config.idle_timeout_seconds * 1000  # Not ideal, uses dedicated field
                # Use the configured target MHz
                target_khz = 600 * 1000  # 600 MHz default low-power
                cpu_freq_path.write_text(str(target_khz))
        except Exception as e:
            logger.debug(f"Could not set CPU frequency: {e}")

        return True

    async def exit_low_power_mode(self) -> bool:
        """
        Exit low-power mode and restore full operation.

        Returns:
            True if successful

        Requirements: 14.14
        """
        if self._mode == PowerMode.FULL:
            return True

        self._mode = PowerMode.FULL
        logger.info("Exiting low-power mode — full power restored")

        # Restore CPU frequency
        try:
            cpu_freq_path = Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq")
            if cpu_freq_path.exists():
                # Read max available frequency
                max_freq_path = Path("/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq")
                if max_freq_path.exists():
                    max_freq = max_freq_path.read_text().strip()
                    cpu_freq_path.write_text(max_freq)
        except Exception as e:
            logger.debug(f"Could not restore CPU frequency: {e}")

        return True

    async def start_monitoring(self, event_bus: EventBus):
        """
        Start battery monitoring background task.

        Args:
            event_bus: EventBus for emitting power events
        """
        self._running = True
        self._event_bus = event_bus
        self._monitor_task = asyncio.create_task(self._monitor_loop())
        logger.info("Battery monitoring started")

    async def stop_monitoring(self):
        """Stop battery monitoring."""
        self._running = False
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass

    async def _monitor_loop(self):
        """Background battery monitoring loop."""
        while self._running:
            try:
                level = self.get_battery_level()

                # Check for low battery
                if level <= self.config.critical_battery_threshold:
                    await self._event_bus.emit("power.low_battery", {
                        "level": level,
                        "critical": True,
                        "timestamp": datetime.now()
                    })
                elif level <= self.config.low_battery_threshold:
                    await self._event_bus.emit("power.low_battery", {
                        "level": level,
                        "critical": False,
                        "timestamp": datetime.now()
                    })

                # Check charging state
                if self.is_on_charger():
                    await self._event_bus.emit("power.charging", {
                        "level": level,
                        "source": self.get_power_source(),
                        "timestamp": datetime.now()
                    })

                await asyncio.sleep(self.config.battery_check_interval)

            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.error(f"Error in power monitor: {e}")
                await asyncio.sleep(5)


# ─── Privacy Switch Monitor (Task 18.7) ─────────────────────────────────────

class PrivacySwitchMonitor:
    """
    Monitor for the physical privacy switch.

    When the privacy switch is activated, disables camera and microphone
    and lights the privacy LED. When deactivated, re-enables them.

    Requirements: 14.16
    """

    def __init__(self, sensor_hub: SensorHub, actuator_hub: ActuatorHub,
                 config: EmbeddedConfig):
        """
        Initialize the privacy switch monitor.

        Args:
            sensor_hub: SensorHub for reading privacy switch GPIO
            actuator_hub: ActuatorHub for controlling privacy LED
            config: Embedded configuration
        """
        self.sensor_hub = sensor_hub
        self.actuator_hub = actuator_hub
        self.config = config

        self._privacy_mode = False
        self._running = False
        self._monitor_task: Optional[asyncio.Task] = None
        self._gpio_available = GPIO_AVAILABLE

    @property
    def privacy_mode(self) -> bool:
        """Whether privacy mode is currently active."""
        return self._privacy_mode

    def is_privacy_mode(self) -> bool:
        """
        Check if privacy mode is active.

        Returns:
            True if privacy switch is engaged

        Requirements: 14.16
        """
        return self._privacy_mode

    def _read_switch(self) -> bool:
        """
        Read the physical privacy switch state.

        Returns:
            True if switch is in "privacy on" position
        """
        if not self._gpio_available:
            return False

        try:
            return bool(GPIO.input(self.config.privacy_switch_pin))
        except Exception as e:
            logger.error(f"Error reading privacy switch: {e}")
            return False

    async def start_monitoring(self, event_bus: EventBus):
        """
        Start privacy switch monitoring.

        Args:
            event_bus: EventBus for emitting privacy events
        """
        self._running = True
        self._event_bus = event_bus
        self._monitor_task = asyncio.create_task(self._monitor_loop())
        logger.info("Privacy switch monitoring started")

    async def stop_monitoring(self):
        """Stop privacy switch monitoring."""
        self._running = False
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass

    async def _monitor_loop(self):
        """
        Background privacy switch monitoring loop.

        Polls the GPIO pin and emits events on state changes.

        Requirements: 14.16
        """
        while self._running:
            try:
                switch_state = self._read_switch()

                if switch_state and not self._privacy_mode:
                    # Privacy switch activated
                    await self._activate_privacy()
                elif not switch_state and self._privacy_mode:
                    # Privacy switch deactivated
                    await self._deactivate_privacy()

                await asyncio.sleep(0.5)

            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.error(f"Error in privacy switch monitor: {e}")
                await asyncio.sleep(1)

    async def _activate_privacy(self):
        """
        Activate privacy mode.

        Disables camera, microphone, and lights privacy LED.

        Requirements: 14.16
        """
        self._privacy_mode = True
        logger.info("Privacy mode ACTIVATED — camera and microphone disabled")

        # Light the privacy LED
        self.actuator_hub.set_status_led(self.config.privacy_led_pin, True)

        # Turn off status LED (camera/mic no longer active)
        self.actuator_hub.set_status_led(self.config.status_led_pin, False)

        if hasattr(self, '_event_bus'):
            await self._event_bus.emit("privacy.switch_activated", {
                "privacy_mode": True,
                "camera_disabled": True,
                "microphone_disabled": True,
                "timestamp": datetime.now()
            })

    async def _deactivate_privacy(self):
        """
        Deactivate privacy mode.

        Re-enables camera and microphone, turns off privacy LED.

        Requirements: 14.16
        """
        self._privacy_mode = False
        logger.info("Privacy mode DEACTIVATED — camera and microphone re-enabled")

        # Turn off privacy LED
        self.actuator_hub.set_status_led(self.config.privacy_led_pin, False)

        if hasattr(self, '_event_bus'):
            await self._event_bus.emit("privacy.switch_deactivated", {
                "privacy_mode": False,
                "camera_disabled": False,
                "microphone_disabled": False,
                "timestamp": datetime.now()
            })


# ─── Physical Embodiment Orchestrator (Task 18.2) ───────────────────────────

class PhysicalEmbodiment:
    """
    Main orchestrator for physical embodiment.

    Owns and coordinates all hardware subsystems: sensors, actuators,
    proximity manager, gesture controller, LED display, power manager,
    and privacy switch.

    Requirements: 14.2, 14.3, 14.4, 14.5, 14.6, 14.7
    """

    def __init__(self, event_bus: EventBus, config: Optional[EmbeddedConfig] = None):
        """
        Initialize the physical embodiment system.

        Args:
            event_bus: EventBus for inter-component communication
            config: Embedded configuration (auto-loaded if None)
        """
        self.event_bus = event_bus
        self.config = config or load_embedded_config()

        # Register embodiment event types
        embodiment_events = [
            "proximity.detected",
            "touch.detected",
            "embodiment.started",
            "embodiment.stopped",
            "privacy.switch_activated",
            "privacy.switch_deactivated",
            "power.low_battery",
            "power.charging",
            "power.mode_changed",
        ]
        for event_type in embodiment_events:
            try:
                self.event_bus.register_event_type(event_type)
            except ValueError:
                pass  # Already registered

        # Initialize subsystems
        self.sensor_hub = SensorHub(self.config)
        self.actuator_hub = ActuatorHub(self.config)
        self.proximity_manager = ProximityManager(self.sensor_hub, self.config)
        self.gesture_controller = GestureController(self.actuator_hub, self.config)
        self.led_display = LEDExpressionDisplay(self.actuator_hub, self.config)
        self.power_manager = PowerManager(self.config)
        self.privacy_switch = PrivacySwitchMonitor(
            self.sensor_hub, self.actuator_hub, self.config
        )

        self._running = False

        logger.info(f"PhysicalEmbodiment initialized (target={self.config.target})")

    async def start(self):
        """
        Start all physical embodiment subsystems.

        Requirements: 14.2
        """
        self._running = True

        # Start proximity monitoring
        await self.proximity_manager.start_monitoring(self.event_bus)

        # Start power monitoring
        await self.power_manager.start_monitoring(self.event_bus)

        # Start privacy switch monitoring
        await self.privacy_switch.start_monitoring(self.event_bus)

        # Show greeting expression
        self.led_display.show_expression("greeting")

        # Emit start event
        await self.event_bus.emit("embodiment.started", {
            "target": self.config.target,
            "timestamp": datetime.now()
        })

        logger.info("Physical embodiment started")

    async def stop(self):
        """
        Stop all physical embodiment subsystems and release resources.

        Requirements: 14.2
        """
        self._running = False

        # Stop monitoring tasks
        await self.proximity_manager.stop_monitoring()
        await self.power_manager.stop_monitoring()
        await self.privacy_switch.stop_monitoring()

        # Return servos to neutral
        await self.gesture_controller.reset_position()

        # Clear LED display
        self.led_display.clear()

        # Cleanup hardware
        self.sensor_hub.cleanup()
        self.actuator_hub.cleanup()

        # Emit stop event
        await self.event_bus.emit("embodiment.stopped", {
            "timestamp": datetime.now()
        })

        logger.info("Physical embodiment stopped")

    @property
    def is_running(self) -> bool:
        """Check if the embodiment system is running."""
        return self._running
