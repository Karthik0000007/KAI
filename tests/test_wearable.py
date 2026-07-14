"""
Unit tests for wearable device integration.

Tests cover:
- BLE device discovery
- Heart rate data parsing
- Data validation
- Event emission
- Auto-reconnection logic

Requirements: 18.1
"""

import pytest
import asyncio
import struct
from datetime import datetime
from unittest.mock import Mock, AsyncMock, patch, MagicMock

from core.wearable import (
    WearableInterface,
    VitalReading,
    HEART_RATE_MIN,
    HEART_RATE_MAX,
    SPO2_MIN,
    SPO2_MAX,
    BLEAK_AVAILABLE
)
from core.event_bus import EventBus


# ─── Fixtures ────────────────────────────────────────────────────────────

@pytest.fixture
def event_bus():
    """Create an event bus for testing."""
    return EventBus()


@pytest.fixture
def mock_db():
    """Create a mock database connection."""
    db = Mock()
    db.cursor = Mock(return_value=Mock())
    db.commit = Mock()
    return db


@pytest.fixture
def wearable_interface(event_bus, mock_db):
    """Create a wearable interface for testing."""
    return WearableInterface(event_bus, mock_db)


# ─── Vital Validation Tests ──────────────────────────────────────────────

def test_validate_heart_rate_valid(wearable_interface):
    """Test heart rate validation with valid values."""
    assert wearable_interface.validate_vital("heart_rate", 60) is True
    assert wearable_interface.validate_vital("heart_rate", 100) is True
    assert wearable_interface.validate_vital("heart_rate", HEART_RATE_MIN) is True
    assert wearable_interface.validate_vital("heart_rate", HEART_RATE_MAX) is True


def test_validate_heart_rate_invalid(wearable_interface):
    """Test heart rate validation with invalid values."""
    assert wearable_interface.validate_vital("heart_rate", 29) is False
    assert wearable_interface.validate_vital("heart_rate", 221) is False
    assert wearable_interface.validate_vital("heart_rate", 0) is False
    assert wearable_interface.validate_vital("heart_rate", 300) is False


def test_validate_spo2_valid(wearable_interface):
    """Test SpO2 validation with valid values."""
    assert wearable_interface.validate_vital("spo2", 95) is True
    assert wearable_interface.validate_vital("spo2", 100) is True
    assert wearable_interface.validate_vital("spo2", 70) is True


def test_validate_spo2_invalid(wearable_interface):
    """Test SpO2 validation with invalid values."""
    assert wearable_interface.validate_vital("spo2", 69) is False
    assert wearable_interface.validate_vital("spo2", 101) is False


def test_validate_temperature_valid(wearable_interface):
    """Test temperature validation with valid values."""
    assert wearable_interface.validate_vital("temperature", 36.5) is True
    assert wearable_interface.validate_vital("temperature", 37.0) is True
    assert wearable_interface.validate_vital("temperature", 38.5) is True


def test_validate_temperature_invalid(wearable_interface):
    """Test temperature validation with invalid values."""
    assert wearable_interface.validate_vital("temperature", 34.9) is False
    assert wearable_interface.validate_vital("temperature", 42.1) is False


@pytest.mark.asyncio
async def test_handle_temperature_celsius(wearable_interface, event_bus):
    """Test parsing temperature data in Celsius."""
    # Create mock data: flags=0x00 (Celsius), temperature=36.5°C (IEEE 754 float)
    temperature_value = 36.5
    data = bytearray([0x00]) + bytearray(struct.pack('<f', temperature_value))
    
    # Set up event listener
    events_received = []
    event_bus.on("vital_received", lambda e: events_received.append(e))
    
    # Handle the temperature data
    await wearable_interface._handle_temperature(data, "AA:BB:CC:DD:EE:FF", "Kinsa Thermometer")
    
    # Verify event was emitted
    assert len(events_received) == 1
    event = events_received[0]
    assert event.data["vital_type"] == "temperature"
    assert abs(event.data["value"] - 36.5) < 0.1
    assert event.data["unit"] == "°C"
    assert event.data["device_name"] == "Kinsa Thermometer"


@pytest.mark.asyncio
async def test_handle_temperature_fahrenheit(wearable_interface, event_bus):
    """Test parsing temperature data in Fahrenheit with conversion to Celsius."""
    # Create mock data: flags=0x01 (Fahrenheit), temperature=98.6°F (should convert to ~37.0°C)
    temperature_fahrenheit = 98.6
    data = bytearray([0x01]) + bytearray(struct.pack('<f', temperature_fahrenheit))
    
    # Set up event listener
    events_received = []
    event_bus.on("vital_received", lambda e: events_received.append(e))
    
    # Handle the temperature data
    await wearable_interface._handle_temperature(data, "AA:BB:CC:DD:EE:FF", "Withings Thermo")
    
    # Verify event was emitted with converted value
    assert len(events_received) == 1
    event = events_received[0]
    assert event.data["vital_type"] == "temperature"
    # 98.6°F = 37.0°C
    assert abs(event.data["value"] - 37.0) < 0.2
    assert event.data["unit"] == "°C"
    assert event.data["device_name"] == "Withings Thermo"


@pytest.mark.asyncio
async def test_handle_temperature_invalid_value(wearable_interface, event_bus):
    """Test handling invalid temperature value (out of range)."""
    # Create mock data with invalid temperature (45.0°C - too high)
    temperature_value = 45.0
    data = bytearray([0x00]) + bytearray(struct.pack('<f', temperature_value))
    
    # Set up event listener
    events_received = []
    event_bus.on("vital_received", lambda e: events_received.append(e))
    
    # Handle the temperature data
    await wearable_interface._handle_temperature(data, "AA:BB:CC:DD:EE:FF", "Test Device")
    
    # Verify no event was emitted (invalid value)
    assert len(events_received) == 0


@pytest.mark.asyncio
async def test_handle_temperature_stores_in_database(wearable_interface, mock_db):
    """Test that valid temperature readings are stored in database."""
    # Create mock VitalRecord save method
    saved_vitals = []
    def mock_save_vital(vital_record):
        saved_vitals.append(vital_record)
    
    mock_db.save_vital = mock_save_vital
    wearable_interface.db = mock_db
    
    # Create mock data: temperature=37.2°C
    temperature_value = 37.2
    data = bytearray([0x00]) + bytearray(struct.pack('<f', temperature_value))
    
    # Handle the temperature data
    await wearable_interface._handle_temperature(data, "AA:BB:CC:DD:EE:FF", "Kinsa Thermometer")
    
    # Verify database save was called
    assert len(saved_vitals) == 1
    vital = saved_vitals[0]
    assert abs(vital.temperature - 37.2) < 0.1


@pytest.mark.asyncio
async def test_handle_temperature_insufficient_data(wearable_interface, event_bus):
    """Test handling insufficient temperature data."""
    # Create malformed data (too short - only 3 bytes instead of 5)
    data = bytearray([0x00, 0x01, 0x02])
    
    # Set up event listener
    events_received = []
    event_bus.on("vital_received", lambda e: events_received.append(e))
    
    # Handle the malformed data
    await wearable_interface._handle_temperature(data, "AA:BB:CC:DD:EE:FF", "Test Device")
    
    # Verify no event was emitted (insufficient data)
    assert len(events_received) == 0


@pytest.mark.asyncio
async def test_temperature_boundary_values(wearable_interface, event_bus):
    """Test temperature validation at boundary values."""
    # Set up event listener
    events_received = []
    event_bus.on("vital_received", lambda e: events_received.append(e))
    
    # Test minimum valid temperature (35.0°C)
    data_min = bytearray([0x00]) + bytearray(struct.pack('<f', 35.0))
    await wearable_interface._handle_temperature(data_min, "AA:BB:CC:DD:EE:FF", "Test Device")
    assert len(events_received) == 1
    
    # Test maximum valid temperature (42.0°C)
    events_received.clear()
    data_max = bytearray([0x00]) + bytearray(struct.pack('<f', 42.0))
    await wearable_interface._handle_temperature(data_max, "AA:BB:CC:DD:EE:FF", "Test Device")
    assert len(events_received) == 1
    
    # Test just below minimum (34.9°C) - should be rejected
    events_received.clear()
    data_below = bytearray([0x00]) + bytearray(struct.pack('<f', 34.9))
    await wearable_interface._handle_temperature(data_below, "AA:BB:CC:DD:EE:FF", "Test Device")
    assert len(events_received) == 0
    
    # Test just above maximum (42.1°C) - should be rejected
    events_received.clear()
    data_above = bytearray([0x00]) + bytearray(struct.pack('<f', 42.1))
    await wearable_interface._handle_temperature(data_above, "AA:BB:CC:DD:EE:FF", "Test Device")
    assert len(events_received) == 0


def test_validate_steps_valid(wearable_interface):
    """Test steps validation with valid values."""
    assert wearable_interface.validate_vital("steps", 0) is True
    assert wearable_interface.validate_vital("steps", 10000) is True
    assert wearable_interface.validate_vital("steps", 50000) is True


def test_validate_steps_invalid(wearable_interface):
    """Test steps validation with invalid values."""
    assert wearable_interface.validate_vital("steps", -1) is False
    assert wearable_interface.validate_vital("steps", 100001) is False


def test_validate_unknown_vital_type(wearable_interface):
    """Test validation with unknown vital type."""
    assert wearable_interface.validate_vital("unknown_type", 100) is False


# ─── Heart Rate Parsing Tests ────────────────────────────────────────────

@pytest.mark.asyncio
async def test_handle_heart_rate_uint8_format(wearable_interface, event_bus):
    """Test parsing heart rate data in uint8 format."""
    # Create mock data: flags=0x00 (uint8), heart_rate=75
    data = bytearray([0x00, 75])
    
    # Set up event listener
    events_received = []
    event_bus.on("vital_received", lambda e: events_received.append(e))
    
    # Handle the data
    await wearable_interface._handle_heart_rate(data, "test_device")
    
    # Verify event was emitted
    assert len(events_received) == 1
    event = events_received[0]
    assert event.data["vital_type"] == "heart_rate"
    assert event.data["value"] == 75.0
    assert event.data["unit"] == "bpm"
    assert event.data["device_id"] == "test_device"


@pytest.mark.asyncio
async def test_handle_heart_rate_uint16_format(wearable_interface, event_bus):
    """Test parsing heart rate data in uint16 format."""
    # Create mock data: flags=0x01 (uint16), heart_rate=180 (little-endian)
    data = bytearray([0x01]) + struct.pack('<H', 180)
    
    # Set up event listener
    events_received = []
    event_bus.on("vital_received", lambda e: events_received.append(e))
    
    # Handle the data
    await wearable_interface._handle_heart_rate(data, "test_device")
    
    # Verify event was emitted
    assert len(events_received) == 1
    event = events_received[0]
    assert event.data["vital_type"] == "heart_rate"
    assert event.data["value"] == 180.0
    assert event.data["unit"] == "bpm"


@pytest.mark.asyncio
async def test_handle_heart_rate_invalid_value(wearable_interface, event_bus):
    """Test handling invalid heart rate value (out of range)."""
    # Create mock data with invalid heart rate (250 bpm)
    data = bytearray([0x00, 250])
    
    # Set up event listener
    events_received = []
    event_bus.on("vital_received", lambda e: events_received.append(e))
    
    # Handle the data
    await wearable_interface._handle_heart_rate(data, "test_device")
    
    # Verify no event was emitted (invalid data)
    assert len(events_received) == 0


@pytest.mark.asyncio
async def test_handle_heart_rate_stores_in_database(wearable_interface, mock_db):
    """Test that valid heart rate readings are stored in database."""
    # Create mock VitalRecord save method
    mock_db.save_vital = Mock()
    
    # Create mock data
    data = bytearray([0x00, 75])
    
    # Handle the data
    await wearable_interface._handle_heart_rate(data, "test_device")
    
    # Verify database save was called
    mock_db.save_vital.assert_called_once()
    
    # Verify the VitalRecord has correct heart_rate
    call_args = mock_db.save_vital.call_args
    vital_record = call_args[0][0]
    assert vital_record.heart_rate == 75


# ─── SpO2 Parsing Tests ──────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_handle_nonin_spo2_basic(wearable_interface, event_bus):
    """Test parsing Nonin 3230 SpO2 data with basic format."""
    # Create mock data: status=0x00, spo2=95
    data = bytearray([0x00, 95])
    
    # Set up event listener
    events_received = []
    event_bus.on("vital_received", lambda e: events_received.append(e))
    
    # Handle the data
    await wearable_interface._handle_nonin_spo2(data, "test_device", "Nonin 3230")
    
    # Verify event was emitted
    assert len(events_received) == 1
    event = events_received[0]
    assert event.data["vital_type"] == "spo2"
    assert event.data["value"] == 95.0
    assert event.data["unit"] == "%"
    assert event.data["device_id"] == "test_device"
    assert event.data["device_name"] == "Nonin 3230"


@pytest.mark.asyncio
async def test_handle_nonin_spo2_with_pulse_rate(wearable_interface, event_bus):
    """Test parsing Nonin SpO2 data with pulse rate."""
    # Create mock data: status=0x00, spo2=98, pulse_rate=72 (little-endian)
    data = bytearray([0x00, 98]) + struct.pack('<H', 72)
    
    # Set up event listener
    events_received = []
    event_bus.on("vital_received", lambda e: events_received.append(e))
    
    # Handle the data
    await wearable_interface._handle_nonin_spo2(data, "test_device", "Nonin 3230")
    
    # Verify event was emitted with correct SpO2
    assert len(events_received) == 1
    event = events_received[0]
    assert event.data["value"] == 98.0


@pytest.mark.asyncio
async def test_handle_nonin_spo2_with_signal_quality(wearable_interface, event_bus):
    """Test parsing Nonin SpO2 data with signal quality."""
    # Create mock data: status=0x00, spo2=97, pulse_rate=75, signal_quality=3
    data = bytearray([0x00, 97]) + struct.pack('<H', 75) + bytearray([3])
    
    # Set up event listener
    events_received = []
    event_bus.on("vital_received", lambda e: events_received.append(e))
    
    # Handle the data
    await wearable_interface._handle_nonin_spo2(data, "test_device", "Nonin 3230")
    
    # Verify event was emitted with confidence based on signal quality
    assert len(events_received) == 1
    event = events_received[0]
    assert event.data["value"] == 97.0
    assert event.data["confidence"] == 0.75  # 3/4


@pytest.mark.asyncio
async def test_handle_nonin_spo2_invalid_value(wearable_interface, event_bus):
    """Test handling invalid Nonin SpO2 value (out of range)."""
    # Create mock data with invalid SpO2 (65%)
    data = bytearray([0x00, 65])
    
    # Set up event listener
    events_received = []
    event_bus.on("vital_received", lambda e: events_received.append(e))
    
    # Handle the data
    await wearable_interface._handle_nonin_spo2(data, "test_device", "Nonin 3230")
    
    # Verify no event was emitted (invalid data)
    assert len(events_received) == 0


@pytest.mark.asyncio
async def test_handle_nonin_spo2_insufficient_data(wearable_interface, event_bus):
    """Test handling insufficient Nonin SpO2 data."""
    # Create malformed data (too short)
    data = bytearray([0x00])
    
    # Set up event listener
    events_received = []
    event_bus.on("vital_received", lambda e: events_received.append(e))
    
    # Handle the data (should not crash)
    await wearable_interface._handle_nonin_spo2(data, "test_device", "Nonin 3230")
    
    # Verify no event was emitted
    assert len(events_received) == 0


@pytest.mark.asyncio
async def test_handle_masimo_spo2_basic(wearable_interface, event_bus):
    """Test parsing Masimo MightySat SpO2 data with basic format."""
    # Create mock data: message_type=0x01, spo2=96
    data = bytearray([0x01, 96])
    
    # Set up event listener
    events_received = []
    event_bus.on("vital_received", lambda e: events_received.append(e))
    
    # Handle the data
    await wearable_interface._handle_masimo_spo2(data, "test_device", "Masimo MightySat")
    
    # Verify event was emitted
    assert len(events_received) == 1
    event = events_received[0]
    assert event.data["vital_type"] == "spo2"
    assert event.data["value"] == 96.0
    assert event.data["unit"] == "%"
    assert event.data["device_id"] == "test_device"
    assert event.data["device_name"] == "Masimo MightySat"


@pytest.mark.asyncio
async def test_handle_masimo_spo2_with_pulse_rate(wearable_interface, event_bus):
    """Test parsing Masimo SpO2 data with pulse rate."""
    # Create mock data: message_type=0x01, spo2=99, pulse_rate=68 (big-endian)
    data = bytearray([0x01, 99]) + struct.pack('>H', 68)
    
    # Set up event listener
    events_received = []
    event_bus.on("vital_received", lambda e: events_received.append(e))
    
    # Handle the data
    await wearable_interface._handle_masimo_spo2(data, "test_device", "Masimo MightySat")
    
    # Verify event was emitted with correct SpO2
    assert len(events_received) == 1
    event = events_received[0]
    assert event.data["value"] == 99.0


@pytest.mark.asyncio
async def test_handle_masimo_spo2_with_signal_quality(wearable_interface, event_bus):
    """Test parsing Masimo SpO2 data with signal quality."""
    # Create mock data: message_type=0x01, spo2=97, pulse_rate=70, perfusion=0x00, signal_quality=85
    data = bytearray([0x01, 97]) + struct.pack('>H', 70) + bytearray([0x00, 85])
    
    # Set up event listener
    events_received = []
    event_bus.on("vital_received", lambda e: events_received.append(e))
    
    # Handle the data
    await wearable_interface._handle_masimo_spo2(data, "test_device", "Masimo MightySat")
    
    # Verify event was emitted with confidence based on signal quality
    assert len(events_received) == 1
    event = events_received[0]
    assert event.data["value"] == 97.0
    assert event.data["confidence"] == 0.85  # 85/100


@pytest.mark.asyncio
async def test_handle_masimo_spo2_invalid_value(wearable_interface, event_bus):
    """Test handling invalid Masimo SpO2 value (out of range)."""
    # Create mock data with invalid SpO2 (105%)
    data = bytearray([0x01, 105])
    
    # Set up event listener
    events_received = []
    event_bus.on("vital_received", lambda e: events_received.append(e))
    
    # Handle the data
    await wearable_interface._handle_masimo_spo2(data, "test_device", "Masimo MightySat")
    
    # Verify no event was emitted (invalid data)
    assert len(events_received) == 0


@pytest.mark.asyncio
async def test_handle_masimo_spo2_insufficient_data(wearable_interface, event_bus):
    """Test handling insufficient Masimo SpO2 data."""
    # Create malformed data (too short)
    data = bytearray([0x01])
    
    # Set up event listener
    events_received = []
    event_bus.on("vital_received", lambda e: events_received.append(e))
    
    # Handle the data (should not crash)
    await wearable_interface._handle_masimo_spo2(data, "test_device", "Masimo MightySat")
    
    # Verify no event was emitted
    assert len(events_received) == 0


@pytest.mark.asyncio
async def test_spo2_stores_in_database(wearable_interface, mock_db):
    """Test that valid SpO2 readings are stored in database."""
    # Create mock VitalRecord save method
    mock_db.save_vital = Mock()
    
    # Create mock data
    data = bytearray([0x00, 95])
    
    # Handle the data
    await wearable_interface._handle_nonin_spo2(data, "test_device", "Nonin 3230")
    
    # Verify database save was called
    mock_db.save_vital.assert_called_once()
    
    # Verify the VitalRecord has correct spo2
    call_args = mock_db.save_vital.call_args
    vital_record = call_args[0][0]
    assert vital_record.spo2 == 95.0


# ─── VitalReading Tests ──────────────────────────────────────────────────

def test_vital_reading_creation():
    """Test creating a VitalReading object."""
    reading = VitalReading(
        vital_type="heart_rate",
        value=75.0,
        unit="bpm",
        timestamp=datetime.now(),
        device_id="test_device",
        confidence=1.0
    )
    
    assert reading.vital_type == "heart_rate"
    assert reading.value == 75.0
    assert reading.unit == "bpm"
    assert reading.device_id == "test_device"
    assert reading.confidence == 1.0


def test_vital_reading_default_confidence():
    """Test VitalReading with default confidence value."""
    reading = VitalReading(
        vital_type="heart_rate",
        value=75.0,
        unit="bpm",
        timestamp=datetime.now(),
        device_id="test_device"
    )
    
    assert reading.confidence == 1.0


# ─── BLE Integration Tests (with mocking) ────────────────────────────────

@pytest.mark.skipif(not BLEAK_AVAILABLE, reason="bleak not installed")
@pytest.mark.asyncio
async def test_discover_devices_finds_heart_rate_monitor():
    """Test discovering BLE devices with Heart Rate Service."""
    from core.wearable import HEART_RATE_SERVICE_UUID
    
    # Create mock device and advertisement data
    mock_device = Mock()
    mock_device.name = "Heart Rate Monitor"
    mock_device.address = "AA:BB:CC:DD:EE:FF"
    
    mock_adv = Mock()
    mock_adv.service_uuids = [HEART_RATE_SERVICE_UUID]
    mock_adv.rssi = -50
    
    # Mock BleakScanner.discover
    with patch('core.wearable.BleakScanner.discover') as mock_discover:
        mock_discover.return_value = {
            "device1": (mock_device, mock_adv)
        }
        
        event_bus = EventBus()
        wearable = WearableInterface(event_bus, None)
        
        devices = await wearable.discover_devices(timeout=1.0)
        
        assert len(devices) == 1
        assert devices[0]["name"] == "Heart Rate Monitor"
        assert devices[0]["address"] == "AA:BB:CC:DD:EE:FF"
        assert devices[0]["services"] == ["heart_rate"]


@pytest.mark.skipif(not BLEAK_AVAILABLE, reason="bleak not installed")
@pytest.mark.asyncio
async def test_discover_devices_finds_nonin_spo2_sensor():
    """Test discovering BLE devices with Nonin SpO2 Service."""
    from core.wearable import NONIN_SPO2_SERVICE_UUID
    
    # Create mock device and advertisement data
    mock_device = Mock()
    mock_device.name = "Nonin 3230"
    mock_device.address = "11:22:33:44:55:66"
    
    mock_adv = Mock()
    mock_adv.service_uuids = [NONIN_SPO2_SERVICE_UUID]
    mock_adv.rssi = -55
    
    # Mock BleakScanner.discover
    with patch('core.wearable.BleakScanner.discover') as mock_discover:
        mock_discover.return_value = {
            "device1": (mock_device, mock_adv)
        }
        
        event_bus = EventBus()
        wearable = WearableInterface(event_bus, None)
        
        devices = await wearable.discover_devices(timeout=1.0)
        
        assert len(devices) == 1
        assert devices[0]["name"] == "Nonin 3230"
        assert devices[0]["address"] == "11:22:33:44:55:66"
        assert devices[0]["services"] == ["spo2_nonin"]


@pytest.mark.skipif(not BLEAK_AVAILABLE, reason="bleak not installed")
@pytest.mark.asyncio
async def test_discover_devices_finds_masimo_spo2_sensor():
    """Test discovering BLE devices with Masimo SpO2 Service."""
    from core.wearable import MASIMO_SPO2_SERVICE_UUID
    
    # Create mock device and advertisement data
    mock_device = Mock()
    mock_device.name = "Masimo MightySat"
    mock_device.address = "AA:11:BB:22:CC:33"
    
    mock_adv = Mock()
    mock_adv.service_uuids = [MASIMO_SPO2_SERVICE_UUID]
    mock_adv.rssi = -60
    
    # Mock BleakScanner.discover
    with patch('core.wearable.BleakScanner.discover') as mock_discover:
        mock_discover.return_value = {
            "device1": (mock_device, mock_adv)
        }
        
        event_bus = EventBus()
        wearable = WearableInterface(event_bus, None)
        
        devices = await wearable.discover_devices(timeout=1.0)
        
        assert len(devices) == 1
        assert devices[0]["name"] == "Masimo MightySat"
        assert devices[0]["address"] == "AA:11:BB:22:CC:33"
        assert devices[0]["services"] == ["spo2_masimo"]


@pytest.mark.skipif(not BLEAK_AVAILABLE, reason="bleak not installed")
@pytest.mark.asyncio
async def test_discover_devices_finds_multiple_device_types():
    """Test discovering multiple device types simultaneously."""
    from core.wearable import HEART_RATE_SERVICE_UUID, NONIN_SPO2_SERVICE_UUID
    
    # Create mock HR device
    mock_hr_device = Mock()
    mock_hr_device.name = "HR Monitor"
    mock_hr_device.address = "AA:BB:CC:DD:EE:FF"
    
    mock_hr_adv = Mock()
    mock_hr_adv.service_uuids = [HEART_RATE_SERVICE_UUID]
    mock_hr_adv.rssi = -50
    
    # Create mock SpO2 device
    mock_spo2_device = Mock()
    mock_spo2_device.name = "Nonin 3230"
    mock_spo2_device.address = "11:22:33:44:55:66"
    
    mock_spo2_adv = Mock()
    mock_spo2_adv.service_uuids = [NONIN_SPO2_SERVICE_UUID]
    mock_spo2_adv.rssi = -55
    
    # Mock BleakScanner.discover
    with patch('core.wearable.BleakScanner.discover') as mock_discover:
        mock_discover.return_value = {
            "device1": (mock_hr_device, mock_hr_adv),
            "device2": (mock_spo2_device, mock_spo2_adv)
        }
        
        event_bus = EventBus()
        wearable = WearableInterface(event_bus, None)
        
        devices = await wearable.discover_devices(timeout=1.0)
        
        assert len(devices) == 2
        
        # Check HR device
        hr_device = next(d for d in devices if "heart_rate" in d["services"])
        assert hr_device["name"] == "HR Monitor"
        
        # Check SpO2 device
        spo2_device = next(d for d in devices if "spo2_nonin" in d["services"])
        assert spo2_device["name"] == "Nonin 3230"


@pytest.mark.skipif(not BLEAK_AVAILABLE, reason="bleak not installed")
@pytest.mark.asyncio
async def test_discover_devices_filters_non_hr_devices():
    """Test that device discovery filters out non-heart-rate devices."""
    # Create mock device without Heart Rate Service
    mock_device = Mock()
    mock_device.name = "Other Device"
    mock_device.address = "11:22:33:44:55:66"
    
    mock_adv = Mock()
    mock_adv.service_uuids = ["some-other-uuid"]
    mock_adv.rssi = -60
    
    # Mock BleakScanner.discover
    with patch('core.wearable.BleakScanner.discover') as mock_discover:
        mock_discover.return_value = {
            "device1": (mock_device, mock_adv)
        }
        
        event_bus = EventBus()
        wearable = WearableInterface(event_bus, None)
        
        devices = await wearable.discover_devices(timeout=1.0)
        
        # Should find no devices (filtered out)
        assert len(devices) == 0


@pytest.mark.skipif(not BLEAK_AVAILABLE, reason="bleak not installed")
@pytest.mark.asyncio
async def test_connect_device_emits_event(event_bus):
    """Test that connecting to a device emits a connection event."""
    # Set up event listener
    events_received = []
    event_bus.on("wearable_connected", lambda e: events_received.append(e))
    
    # Create mock device
    mock_device = Mock()
    mock_device.name = "Test HR Monitor"
    mock_device.address = "AA:BB:CC:DD:EE:FF"
    
    # Create wearable interface
    wearable = WearableInterface(event_bus, None)
    wearable.known_devices["AA:BB:CC:DD:EE:FF"] = mock_device
    
    # Mock BleakClient
    with patch('core.wearable.BleakClient') as mock_client_class:
        mock_client = AsyncMock()
        mock_client.is_connected = True
        mock_client.connect = AsyncMock()
        mock_client.start_notify = AsyncMock()
        mock_client_class.return_value = mock_client
        
        # Connect to device
        success = await wearable.connect_device("AA:BB:CC:DD:EE:FF")
        
        assert success is True
        assert len(events_received) == 1
        assert events_received[0]["device_address"] == "AA:BB:CC:DD:EE:FF"
        assert events_received[0]["device_name"] == "Test HR Monitor"


@pytest.mark.skipif(not BLEAK_AVAILABLE, reason="bleak not installed")
@pytest.mark.asyncio
async def test_disconnect_device_emits_event(event_bus):
    """Test that disconnecting from a device emits a disconnection event."""
    # Set up event listener
    events_received = []
    event_bus.on("wearable_disconnected", lambda e: events_received.append(e))
    
    # Create wearable interface with mock connected device
    wearable = WearableInterface(event_bus, None)
    
    mock_client = AsyncMock()
    mock_client.disconnect = AsyncMock()
    wearable.connected_devices["AA:BB:CC:DD:EE:FF"] = mock_client
    
    # Disconnect from device
    success = await wearable.disconnect_device("AA:BB:CC:DD:EE:FF")
    
    assert success is True
    assert len(events_received) == 1
    assert events_received[0]["device_address"] == "AA:BB:CC:DD:EE:FF"


# ─── Error Handling Tests ────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_handle_heart_rate_with_malformed_data(wearable_interface, event_bus):
    """Test handling malformed heart rate data."""
    # Create malformed data (too short)
    data = bytearray([0x00])
    
    # Set up event listener
    events_received = []
    event_bus.on("vital_received", lambda e: events_received.append(e))
    
    # Handle the data (should not crash)
    await wearable_interface._handle_heart_rate(data, "test_device")
    
    # Verify no event was emitted
    assert len(events_received) == 0


def test_wearable_interface_without_bleak():
    """Test that WearableInterface handles missing bleak gracefully."""
    with patch('core.wearable.BLEAK_AVAILABLE', False):
        event_bus = EventBus()
        wearable = WearableInterface(event_bus, None)
        
        # Should not crash, just log warning
        assert wearable is not None


@pytest.mark.asyncio
async def test_discover_devices_without_bleak():
    """Test device discovery when bleak is not available."""
    with patch('core.wearable.BLEAK_AVAILABLE', False):
        event_bus = EventBus()
        wearable = WearableInterface(event_bus, None)
        
        devices = await wearable.discover_devices()
        
        # Should return empty list
        assert devices == []


@pytest.mark.asyncio
async def test_connect_device_without_bleak():
    """Test device connection when bleak is not available."""
    with patch('core.wearable.BLEAK_AVAILABLE', False):
        event_bus = EventBus()
        wearable = WearableInterface(event_bus, None)
        
        success = await wearable.connect_device("AA:BB:CC:DD:EE:FF")
        
        # Should return False
        assert success is False


# ─── Start/Stop Tests ────────────────────────────────────────────────────

@pytest.mark.skipif(not BLEAK_AVAILABLE, reason="bleak not installed")
@pytest.mark.asyncio
async def test_start_wearable_interface():
    """Test starting the wearable interface."""
    event_bus = EventBus()
    wearable = WearableInterface(event_bus, None)
    
    await wearable.start()
    
    assert wearable.running is True
    assert wearable._reconnect_task is not None
    
    # Clean up
    await wearable.stop()


@pytest.mark.skipif(not BLEAK_AVAILABLE, reason="bleak not installed")
@pytest.mark.asyncio
async def test_stop_wearable_interface():
    """Test stopping the wearable interface."""
    event_bus = EventBus()
    wearable = WearableInterface(event_bus, None)
    
    await wearable.start()
    await wearable.stop()
    
    assert wearable.running is False


@pytest.mark.asyncio
async def test_start_without_bleak():
    """Test starting wearable interface when bleak is not available."""
    with patch('core.wearable.BLEAK_AVAILABLE', False):
        event_bus = EventBus()
        wearable = WearableInterface(event_bus, None)
        
        await wearable.start()
        
        # Should not start
        assert wearable.running is False


# ─── Manual Vital Entry Tests ────────────────────────────────────────────

@pytest.mark.asyncio
async def test_enter_vital_manually_heart_rate(wearable_interface, event_bus):
    """Test manually entering a heart rate value."""
    # Set up event listener
    events_received = []
    event_bus.on("vital_received", lambda e: events_received.append(e))
    
    # Enter heart rate manually
    success = await wearable_interface.enter_vital_manually("heart_rate", 72)
    
    # Verify success
    assert success is True
    
    # Verify event was emitted
    assert len(events_received) == 1
    event = events_received[0]
    assert event.data["vital_type"] == "heart_rate"
    assert event.data["value"] == 72.0
    assert event.data["unit"] == "bpm"
    assert event.data["device_id"] == "manual"
    assert event.data["device_name"] == "Manual Entry"


@pytest.mark.asyncio
async def test_enter_vital_manually_spo2(wearable_interface, event_bus):
    """Test manually entering an SpO2 value."""
    # Set up event listener
    events_received = []
    event_bus.on("vital_received", lambda e: events_received.append(e))
    
    # Enter SpO2 manually
    success = await wearable_interface.enter_vital_manually("spo2", 98)
    
    # Verify success
    assert success is True
    
    # Verify event was emitted
    assert len(events_received) == 1
    event = events_received[0]
    assert event.data["vital_type"] == "spo2"
    assert event.data["value"] == 98.0
    assert event.data["unit"] == "%"
    assert event.data["device_id"] == "manual"


@pytest.mark.asyncio
async def test_enter_vital_manually_temperature(wearable_interface, event_bus):
    """Test manually entering a temperature value."""
    # Set up event listener
    events_received = []
    event_bus.on("vital_received", lambda e: events_received.append(e))
    
    # Enter temperature manually
    success = await wearable_interface.enter_vital_manually("temperature", 37.2)
    
    # Verify success
    assert success is True
    
    # Verify event was emitted
    assert len(events_received) == 1
    event = events_received[0]
    assert event.data["vital_type"] == "temperature"
    assert event.data["value"] == 37.2
    assert event.data["unit"] == "°C"
    assert event.data["device_id"] == "manual"


@pytest.mark.asyncio
async def test_enter_vital_manually_steps(wearable_interface, event_bus):
    """Test manually entering a step count."""
    # Set up event listener
    events_received = []
    event_bus.on("vital_received", lambda e: events_received.append(e))
    
    # Enter steps manually
    success = await wearable_interface.enter_vital_manually("steps", 8500)
    
    # Verify success
    assert success is True
    
    # Verify event was emitted
    assert len(events_received) == 1
    event = events_received[0]
    assert event.data["vital_type"] == "steps"
    assert event.data["value"] == 8500.0
    assert event.data["unit"] == "steps"
    assert event.data["device_id"] == "manual"


@pytest.mark.asyncio
async def test_enter_vital_manually_calories(wearable_interface, event_bus):
    """Test manually entering calories burned."""
    # Set up event listener
    events_received = []
    event_bus.on("vital_received", lambda e: events_received.append(e))
    
    # Enter calories manually
    success = await wearable_interface.enter_vital_manually("calories", 2200)
    
    # Verify success
    assert success is True
    
    # Verify event was emitted
    assert len(events_received) == 1
    event = events_received[0]
    assert event.data["vital_type"] == "calories"
    assert event.data["value"] == 2200.0
    assert event.data["unit"] == "kcal"
    assert event.data["device_id"] == "manual"


@pytest.mark.asyncio
async def test_enter_vital_manually_active_minutes(wearable_interface, event_bus):
    """Test manually entering active minutes."""
    # Set up event listener
    events_received = []
    event_bus.on("vital_received", lambda e: events_received.append(e))
    
    # Enter active minutes manually
    success = await wearable_interface.enter_vital_manually("active_minutes", 45)
    
    # Verify success
    assert success is True
    
    # Verify event was emitted
    assert len(events_received) == 1
    event = events_received[0]
    assert event.data["vital_type"] == "active_minutes"
    assert event.data["value"] == 45.0
    assert event.data["unit"] == "minutes"
    assert event.data["device_id"] == "manual"


@pytest.mark.asyncio
async def test_enter_vital_manually_invalid_heart_rate(wearable_interface, event_bus):
    """Test manually entering an invalid heart rate value."""
    # Set up event listener
    events_received = []
    event_bus.on("vital_received", lambda e: events_received.append(e))
    
    # Try to enter invalid heart rate (too high)
    success = await wearable_interface.enter_vital_manually("heart_rate", 250)
    
    # Verify failure
    assert success is False
    
    # Verify no event was emitted
    assert len(events_received) == 0


@pytest.mark.asyncio
async def test_enter_vital_manually_invalid_spo2(wearable_interface, event_bus):
    """Test manually entering an invalid SpO2 value."""
    # Set up event listener
    events_received = []
    event_bus.on("vital_received", lambda e: events_received.append(e))
    
    # Try to enter invalid SpO2 (too low)
    success = await wearable_interface.enter_vital_manually("spo2", 65)
    
    # Verify failure
    assert success is False
    
    # Verify no event was emitted
    assert len(events_received) == 0


@pytest.mark.asyncio
async def test_enter_vital_manually_invalid_temperature(wearable_interface, event_bus):
    """Test manually entering an invalid temperature value."""
    # Set up event listener
    events_received = []
    event_bus.on("vital_received", lambda e: events_received.append(e))
    
    # Try to enter invalid temperature (too high)
    success = await wearable_interface.enter_vital_manually("temperature", 45.0)
    
    # Verify failure
    assert success is False
    
    # Verify no event was emitted
    assert len(events_received) == 0


@pytest.mark.asyncio
async def test_enter_vital_manually_stores_in_database(wearable_interface, mock_db):
    """Test that manually entered vitals are stored in database."""
    # Create mock VitalRecord save method
    saved_vitals = []
    def mock_save_vital(vital_record):
        saved_vitals.append(vital_record)
    
    mock_db.save_vital = mock_save_vital
    wearable_interface.db = mock_db
    
    # Enter heart rate manually
    success = await wearable_interface.enter_vital_manually("heart_rate", 75)
    
    # Verify success
    assert success is True
    
    # Verify database save was called
    assert len(saved_vitals) == 1
    vital = saved_vitals[0]
    assert vital.heart_rate == 75


@pytest.mark.asyncio
async def test_enter_vital_manually_custom_unit(wearable_interface, event_bus):
    """Test manually entering a vital with custom unit."""
    # Set up event listener
    events_received = []
    event_bus.on("vital_received", lambda e: events_received.append(e))
    
    # Enter temperature with custom unit
    success = await wearable_interface.enter_vital_manually("temperature", 37.0, unit="Celsius")
    
    # Verify success
    assert success is True
    
    # Verify event was emitted with custom unit
    assert len(events_received) == 1
    event = events_received[0]
    assert event.data["unit"] == "Celsius"


@pytest.mark.asyncio
async def test_enter_vital_manually_boundary_values(wearable_interface, event_bus):
    """Test manually entering vitals at boundary values."""
    # Set up event listener
    events_received = []
    event_bus.on("vital_received", lambda e: events_received.append(e))
    
    # Test minimum valid heart rate
    success = await wearable_interface.enter_vital_manually("heart_rate", HEART_RATE_MIN)
    assert success is True
    assert len(events_received) == 1
    
    # Test maximum valid heart rate
    events_received.clear()
    success = await wearable_interface.enter_vital_manually("heart_rate", HEART_RATE_MAX)
    assert success is True
    assert len(events_received) == 1
    
    # Test just below minimum (should fail)
    events_received.clear()
    success = await wearable_interface.enter_vital_manually("heart_rate", HEART_RATE_MIN - 1)
    assert success is False
    assert len(events_received) == 0
    
    # Test just above maximum (should fail)
    events_received.clear()
    success = await wearable_interface.enter_vital_manually("heart_rate", HEART_RATE_MAX + 1)
    assert success is False
    assert len(events_received) == 0


@pytest.mark.asyncio
async def test_enter_vital_manually_multiple_vitals(wearable_interface, event_bus):
    """Test manually entering multiple vitals in sequence."""
    # Set up event listener
    events_received = []
    event_bus.on("vital_received", lambda e: events_received.append(e))
    
    # Enter multiple vitals
    vitals = [
        ("heart_rate", 72),
        ("spo2", 98),
        ("temperature", 37.0),
        ("steps", 8000),
        ("calories", 2000),
        ("active_minutes", 30)
    ]
    
    for vital_type, value in vitals:
        success = await wearable_interface.enter_vital_manually(vital_type, value)
        assert success is True
    
    # Verify all events were emitted
    assert len(events_received) == 6
    
    # Verify each vital type
    vital_types = [e.data["vital_type"] for e in events_received]
    assert "heart_rate" in vital_types
    assert "spo2" in vital_types
    assert "temperature" in vital_types
    assert "steps" in vital_types
    assert "calories" in vital_types
    assert "active_minutes" in vital_types


@pytest.mark.asyncio
async def test_enter_vital_manually_without_database(event_bus):
    """Test manually entering vitals when database is not available."""
    # Create wearable interface without database
    wearable = WearableInterface(event_bus, None)
    
    # Set up event listener
    events_received = []
    event_bus.on("vital_received", lambda e: events_received.append(e))
    
    # Enter heart rate manually (should still work, just not stored)
    success = await wearable.enter_vital_manually("heart_rate", 75)
    
    # Verify success (event emitted even without database)
    assert success is True
    assert len(events_received) == 1


@pytest.mark.asyncio
async def test_enter_vital_manually_zero_values(wearable_interface, event_bus):
    """Test manually entering zero values where valid."""
    # Set up event listener
    events_received = []
    event_bus.on("vital_received", lambda e: events_received.append(e))
    
    # Zero steps is valid
    success = await wearable_interface.enter_vital_manually("steps", 0)
    assert success is True
    assert len(events_received) == 1
    
    # Zero calories is valid
    events_received.clear()
    success = await wearable_interface.enter_vital_manually("calories", 0)
    assert success is True
    assert len(events_received) == 1
    
    # Zero active minutes is valid
    events_received.clear()
    success = await wearable_interface.enter_vital_manually("active_minutes", 0)
    assert success is True
    assert len(events_received) == 1
    
    # Zero heart rate is invalid
    events_received.clear()
    success = await wearable_interface.enter_vital_manually("heart_rate", 0)
    assert success is False
    assert len(events_received) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
