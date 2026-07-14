"""
Integration tests for complete wearable support.

Tests the complete data flow from device to database to event bus for:
- SpO2 sensor integration (Nonin 3230, Masimo MightySat)
- Temperature sensor integration (BLE Health Thermometer Service)
- Fitness tracker integration (steps, calories, active minutes)

These tests verify end-to-end functionality including:
- Device data parsing
- Data validation
- Database storage
- Event bus emission
- Error handling

Requirements: 18.2
"""

import pytest
import asyncio
import struct
import sqlite3
from datetime import datetime
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from pathlib import Path
import tempfile

from core.wearable import (
    WearableInterface,
    VitalReading,
    HEART_RATE_MIN,
    HEART_RATE_MAX,
    SPO2_MIN,
    SPO2_MAX,
    TEMPERATURE_MIN,
    TEMPERATURE_MAX,
    STEPS_MIN,
    STEPS_MAX,
    BLEAK_AVAILABLE
)
from core.event_bus import EventBus
from core.health_db import HealthDatabase
from core.models import VitalRecord


# ─── Fixtures ────────────────────────────────────────────────────────────

@pytest.fixture
def event_bus():
    """Create an event bus for testing."""
    return EventBus()


@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_health.db"
        db = HealthDatabase(db_path=db_path)
        yield db
        db.close()


@pytest.fixture
def wearable_interface(event_bus, temp_db):
    """Create a wearable interface with real database for integration testing."""
    return WearableInterface(event_bus, temp_db)


# ─── SpO2 Sensor Integration Tests ───────────────────────────────────────

@pytest.mark.asyncio
async def test_nonin_spo2_complete_integration(wearable_interface, event_bus, temp_db):
    """
    Test complete integration of Nonin 3230 SpO2 sensor.
    
    Validates: Requirements 18.2
    
    Tests the complete data flow:
    1. Device sends SpO2 data
    2. Data is parsed and validated
    3. Data is stored in database
    4. Event is emitted to event bus
    """
    # Set up event listener to capture emitted events
    events_received = []
    event_bus.on("vital_received", lambda e: events_received.append(e))
    
    # Create mock Nonin SpO2 data: status=0x00, spo2=96%, pulse_rate=72, signal_quality=3
    spo2_value = 96
    pulse_rate = 72
    signal_quality = 3
    data = bytearray([0x00, spo2_value]) + struct.pack('<H', pulse_rate) + bytearray([signal_quality])
    
    device_address = "AA:BB:CC:DD:EE:FF"
    device_name = "Nonin 3230"
    
    # Handle the SpO2 data
    await wearable_interface._handle_nonin_spo2(data, device_address, device_name)
    
    # Verify event was emitted
    assert len(events_received) == 1
    event = events_received[0]
    assert event.data["vital_type"] == "spo2"
    assert event.data["value"] == 96.0
    assert event.data["unit"] == "%"
    assert event.data["device_id"] == device_address
    assert event.data["device_name"] == device_name
    assert event.data["confidence"] == 0.75  # 3/4
    
    # Verify data was stored in database
    vitals = temp_db.get_recent_vitals(days=1)
    assert len(vitals) == 1
    vital = vitals[0]
    assert vital["spo2"] == 96.0
    assert vital["timestamp"] is not None


@pytest.mark.asyncio
async def test_masimo_spo2_complete_integration(wearable_interface, event_bus, temp_db):
    """
    Test complete integration of Masimo MightySat SpO2 sensor.
    
    Validates: Requirements 18.2
    
    Tests the complete data flow:
    1. Device sends SpO2 data
    2. Data is parsed and validated
    3. Data is stored in database
    4. Event is emitted to event bus
    """
    # Set up event listener
    events_received = []
    event_bus.on("vital_received", lambda e: events_received.append(e))
    
    # Create mock Masimo SpO2 data: message_type=0x01, spo2=98%, pulse_rate=68, perfusion=0x00, signal_quality=90
    spo2_value = 98
    pulse_rate = 68
    signal_quality = 90
    data = bytearray([0x01, spo2_value]) + struct.pack('>H', pulse_rate) + bytearray([0x00, signal_quality])
    
    device_address = "11:22:33:44:55:66"
    device_name = "Masimo MightySat"
    
    # Handle the SpO2 data
    await wearable_interface._handle_masimo_spo2(data, device_address, device_name)
    
    # Verify event was emitted
    assert len(events_received) == 1
    event = events_received[0]
    assert event.data["vital_type"] == "spo2"
    assert event.data["value"] == 98.0
    assert event.data["unit"] == "%"
    assert event.data["device_id"] == device_address
    assert event.data["device_name"] == device_name
    assert event.data["confidence"] == 0.90  # 90/100
    
    # Verify data was stored in database
    vitals = temp_db.get_recent_vitals(days=1)
    assert len(vitals) == 1
    vital = vitals[0]
    assert vital["spo2"] == 98.0


@pytest.mark.asyncio
async def test_spo2_invalid_value_not_stored(wearable_interface, event_bus, temp_db):
    """
    Test that invalid SpO2 values are rejected and not stored.
    
    Validates: Requirements 18.2
    
    Tests error handling:
    1. Device sends invalid SpO2 data (out of range)
    2. Data is rejected during validation
    3. No event is emitted
    4. No data is stored in database
    """
    # Set up event listener
    events_received = []
    event_bus.on("vital_received", lambda e: events_received.append(e))
    
    # Create mock data with invalid SpO2 (65% - below minimum)
    invalid_spo2 = 65
    data = bytearray([0x00, invalid_spo2])
    
    # Handle the invalid data
    await wearable_interface._handle_nonin_spo2(data, "test_device", "Test Device")
    
    # Verify no event was emitted
    assert len(events_received) == 0
    
    # Verify no data was stored in database
    vitals = temp_db.get_recent_vitals(days=1)
    assert len(vitals) == 0


# ─── Temperature Sensor Integration Tests ────────────────────────────────

@pytest.mark.asyncio
async def test_temperature_celsius_complete_integration(wearable_interface, event_bus, temp_db):
    """
    Test complete integration of temperature sensor with Celsius data.
    
    Validates: Requirements 18.2
    
    Tests the complete data flow:
    1. Device sends temperature data in Celsius
    2. Data is parsed and validated
    3. Data is stored in database
    4. Event is emitted to event bus
    """
    # Set up event listener
    events_received = []
    event_bus.on("vital_received", lambda e: events_received.append(e))
    
    # Create mock temperature data: flags=0x00 (Celsius), temperature=37.2°C
    temperature_value = 37.2
    data = bytearray([0x00]) + bytearray(struct.pack('<f', temperature_value))
    
    device_address = "AA:11:BB:22:CC:33"
    device_name = "Kinsa Thermometer"
    
    # Handle the temperature data
    await wearable_interface._handle_temperature(data, device_address, device_name)
    
    # Verify event was emitted
    assert len(events_received) == 1
    event = events_received[0]
    assert event.data["vital_type"] == "temperature"
    assert abs(event.data["value"] - 37.2) < 0.1
    assert event.data["unit"] == "°C"
    assert event.data["device_id"] == device_address
    assert event.data["device_name"] == device_name
    
    # Verify data was stored in database
    vitals = temp_db.get_recent_vitals(days=1)
    assert len(vitals) == 1
    vital = vitals[0]
    assert abs(vital["temperature"] - 37.2) < 0.1


@pytest.mark.asyncio
async def test_temperature_fahrenheit_conversion_integration(wearable_interface, event_bus, temp_db):
    """
    Test complete integration of temperature sensor with Fahrenheit to Celsius conversion.
    
    Validates: Requirements 18.2
    
    Tests the complete data flow with unit conversion:
    1. Device sends temperature data in Fahrenheit
    2. Data is converted to Celsius
    3. Converted data is validated
    4. Data is stored in database
    5. Event is emitted to event bus
    """
    # Set up event listener
    events_received = []
    event_bus.on("vital_received", lambda e: events_received.append(e))
    
    # Create mock temperature data: flags=0x01 (Fahrenheit), temperature=98.6°F (should convert to ~37.0°C)
    temperature_fahrenheit = 98.6
    data = bytearray([0x01]) + bytearray(struct.pack('<f', temperature_fahrenheit))
    
    device_address = "BB:22:CC:33:DD:44"
    device_name = "Withings Thermo"
    
    # Handle the temperature data
    await wearable_interface._handle_temperature(data, device_address, device_name)
    
    # Verify event was emitted with converted value
    assert len(events_received) == 1
    event = events_received[0]
    assert event.data["vital_type"] == "temperature"
    # 98.6°F = 37.0°C
    assert abs(event.data["value"] - 37.0) < 0.2
    assert event.data["unit"] == "°C"
    
    # Verify converted data was stored in database
    vitals = temp_db.get_recent_vitals(days=1)
    assert len(vitals) == 1
    vital = vitals[0]
    assert abs(vital["temperature"] - 37.0) < 0.2


@pytest.mark.asyncio
async def test_temperature_boundary_values_integration(wearable_interface, event_bus, temp_db):
    """
    Test temperature sensor integration at boundary values.
    
    Validates: Requirements 18.2
    
    Tests validation at boundaries:
    1. Minimum valid temperature (35.0°C) - should be accepted
    2. Maximum valid temperature (42.0°C) - should be accepted
    3. Below minimum (34.9°C) - should be rejected
    4. Above maximum (42.1°C) - should be rejected
    """
    # Set up event listener
    events_received = []
    event_bus.on("vital_received", lambda e: events_received.append(e))
    
    device_address = "test_device"
    device_name = "Test Thermometer"
    
    # Test minimum valid temperature (35.0°C)
    data_min = bytearray([0x00]) + bytearray(struct.pack('<f', 35.0))
    await wearable_interface._handle_temperature(data_min, device_address, device_name)
    assert len(events_received) == 1
    
    # Test maximum valid temperature (42.0°C)
    events_received.clear()
    data_max = bytearray([0x00]) + bytearray(struct.pack('<f', 42.0))
    await wearable_interface._handle_temperature(data_max, device_address, device_name)
    assert len(events_received) == 1
    
    # Test just below minimum (34.9°C) - should be rejected
    events_received.clear()
    data_below = bytearray([0x00]) + bytearray(struct.pack('<f', 34.9))
    await wearable_interface._handle_temperature(data_below, device_address, device_name)
    assert len(events_received) == 0
    
    # Test just above maximum (42.1°C) - should be rejected
    events_received.clear()
    data_above = bytearray([0x00]) + bytearray(struct.pack('<f', 42.1))
    await wearable_interface._handle_temperature(data_above, device_address, device_name)
    assert len(events_received) == 0
    
    # Verify only 2 valid readings were stored (min and max)
    vitals = temp_db.get_recent_vitals(days=1)
    assert len(vitals) == 2


# ─── Fitness Tracker Integration Tests ───────────────────────────────────

@pytest.mark.asyncio
async def test_fitness_tracker_complete_integration(wearable_interface, event_bus, temp_db):
    """
    Test complete integration of fitness tracker data ingestion.
    
    Validates: Requirements 18.2
    
    Tests the complete data flow for all fitness metrics:
    1. Fitness tracker provides steps, calories, and active minutes
    2. All data is validated
    3. All data is stored in database
    4. Events are emitted for each metric
    """
    # Set up event listener
    events_received = []
    event_bus.on("vital_received", lambda e: events_received.append(e))
    
    device_id = "fitbit_user_123"
    device_name = "Fitbit Charge 5"
    steps = 8500
    calories = 2200.5
    active_minutes = 45
    
    # Ingest fitness tracker data
    success = await wearable_interface.ingest_activity_data(
        device_id=device_id,
        device_name=device_name,
        steps=steps,
        calories=calories,
        active_minutes=active_minutes
    )
    
    # Verify ingestion was successful
    assert success is True
    
    # Verify 3 events were emitted (one for each metric)
    assert len(events_received) == 3
    
    # Verify steps event
    steps_event = next(e for e in events_received if e.data["vital_type"] == "steps")
    assert steps_event.data["value"] == 8500.0
    assert steps_event.data["unit"] == "steps"
    assert steps_event.data["device_id"] == device_id
    assert steps_event.data["device_name"] == device_name
    
    # Verify calories event
    calories_event = next(e for e in events_received if e.data["vital_type"] == "calories")
    assert calories_event.data["value"] == 2200.5
    assert calories_event.data["unit"] == "kcal"
    assert calories_event.data["device_id"] == device_id
    
    # Verify active minutes event
    active_event = next(e for e in events_received if e.data["vital_type"] == "active_minutes")
    assert active_event.data["value"] == 45.0
    assert active_event.data["unit"] == "minutes"
    assert active_event.data["device_id"] == device_id
    
    # Verify all data was stored in database
    vitals = temp_db.get_recent_vitals(days=1)
    assert len(vitals) == 3
    
    # Verify each vital type was stored
    vital_types = set()
    for v in vitals:
        if v.get("steps") is not None:
            vital_types.add(v["steps"])
        if v.get("calories") is not None:
            vital_types.add(v["calories"])
        if v.get("active_minutes") is not None:
            vital_types.add(v["active_minutes"])
    
    assert 8500 in vital_types
    assert 2200.5 in vital_types
    assert 45 in vital_types


@pytest.mark.asyncio
async def test_fitness_tracker_partial_data_integration(wearable_interface, event_bus, temp_db):
    """
    Test fitness tracker integration with partial data (only some metrics provided).
    
    Validates: Requirements 18.2
    
    Tests handling of partial data:
    1. Only steps provided (no calories or active minutes)
    2. Steps data is validated and stored
    3. Event is emitted only for steps
    """
    # Set up event listener
    events_received = []
    event_bus.on("vital_received", lambda e: events_received.append(e))
    
    device_id = "garmin_device_456"
    device_name = "Garmin Vivosmart"
    steps = 12000
    
    # Ingest only steps data
    success = await wearable_interface.ingest_activity_data(
        device_id=device_id,
        device_name=device_name,
        steps=steps,
        calories=None,
        active_minutes=None
    )
    
    # Verify ingestion was successful
    assert success is True
    
    # Verify only 1 event was emitted (for steps)
    assert len(events_received) == 1
    event = events_received[0]
    assert event.data["vital_type"] == "steps"
    assert event.data["value"] == 12000.0
    
    # Verify only steps data was stored
    vitals = temp_db.get_recent_vitals(days=1)
    assert len(vitals) == 1
    vital = vitals[0]
    assert vital["steps"] == 12000


@pytest.mark.asyncio
async def test_fitness_tracker_invalid_data_rejection(wearable_interface, event_bus, temp_db):
    """
    Test fitness tracker integration with invalid data.
    
    Validates: Requirements 18.2
    
    Tests error handling:
    1. Invalid steps value (negative)
    2. Invalid calories value (too high)
    3. Invalid active minutes (exceeds 24 hours)
    4. No data is stored
    5. No events are emitted
    """
    # Set up event listener
    events_received = []
    event_bus.on("vital_received", lambda e: events_received.append(e))
    
    device_id = "test_device"
    device_name = "Test Tracker"
    
    # Try to ingest invalid data
    success = await wearable_interface.ingest_activity_data(
        device_id=device_id,
        device_name=device_name,
        steps=-100,  # Invalid: negative
        calories=15000,  # Invalid: too high
        active_minutes=2000  # Invalid: exceeds 24 hours
    )
    
    # Verify ingestion failed
    assert success is False
    
    # Verify no events were emitted
    assert len(events_received) == 0
    
    # Verify no data was stored
    vitals = temp_db.get_recent_vitals(days=1)
    assert len(vitals) == 0


@pytest.mark.asyncio
async def test_fitness_tracker_boundary_values_integration(wearable_interface, event_bus, temp_db):
    """
    Test fitness tracker integration at boundary values.
    
    Validates: Requirements 18.2
    
    Tests validation at boundaries:
    1. Zero steps (valid minimum)
    2. Maximum steps (100,000)
    3. Zero calories (valid minimum)
    4. Maximum active minutes (1440 = 24 hours)
    """
    # Set up event listener
    events_received = []
    event_bus.on("vital_received", lambda e: events_received.append(e))
    
    device_id = "test_device"
    device_name = "Test Tracker"
    
    # Test minimum values (all zeros)
    success = await wearable_interface.ingest_activity_data(
        device_id=device_id,
        device_name=device_name,
        steps=0,
        calories=0.0,
        active_minutes=0
    )
    assert success is True
    assert len(events_received) == 3
    
    # Test maximum values
    events_received.clear()
    success = await wearable_interface.ingest_activity_data(
        device_id=device_id,
        device_name=device_name,
        steps=100000,
        calories=10000.0,
        active_minutes=1440
    )
    assert success is True
    assert len(events_received) == 3
    
    # Verify all data was stored (6 vitals total: 3 min + 3 max)
    vitals = temp_db.get_recent_vitals(days=1)
    assert len(vitals) == 6


# ─── Multi-Device Integration Tests ──────────────────────────────────────

@pytest.mark.asyncio
async def test_multiple_devices_concurrent_integration(wearable_interface, event_bus, temp_db):
    """
    Test integration of multiple wearable devices sending data concurrently.
    
    Validates: Requirements 18.2
    
    Tests concurrent data handling:
    1. Multiple devices send data simultaneously
    2. All data is correctly parsed and stored
    3. Events are emitted for all devices
    4. Data from different devices is properly isolated
    """
    # Set up event listener
    events_received = []
    event_bus.on("vital_received", lambda e: events_received.append(e))
    
    # Prepare data from multiple devices
    # Device 1: Heart rate monitor
    hr_data = bytearray([0x00, 75])
    hr_address = "HR:11:22:33:44:55"
    hr_name = "Polar H10"
    
    # Device 2: SpO2 sensor
    spo2_data = bytearray([0x00, 97])
    spo2_address = "SP:66:77:88:99:AA"
    spo2_name = "Nonin 3230"
    
    # Device 3: Temperature sensor
    temp_data = bytearray([0x00]) + bytearray(struct.pack('<f', 36.8))
    temp_address = "TM:BB:CC:DD:EE:FF"
    temp_name = "Kinsa Thermometer"
    
    # Handle all devices concurrently
    await asyncio.gather(
        wearable_interface._handle_heart_rate(hr_data, hr_address, hr_name),
        wearable_interface._handle_nonin_spo2(spo2_data, spo2_address, spo2_name),
        wearable_interface._handle_temperature(temp_data, temp_address, temp_name)
    )
    
    # Verify 3 events were emitted (one from each device)
    assert len(events_received) == 3
    
    # Verify each device's data
    hr_event = next(e for e in events_received if e.data["vital_type"] == "heart_rate")
    assert hr_event.data["device_id"] == hr_address
    assert hr_event.data["value"] == 75.0
    
    spo2_event = next(e for e in events_received if e.data["vital_type"] == "spo2")
    assert spo2_event.data["device_id"] == spo2_address
    assert spo2_event.data["value"] == 97.0
    
    temp_event = next(e for e in events_received if e.data["vital_type"] == "temperature")
    assert temp_event.data["device_id"] == temp_address
    assert abs(temp_event.data["value"] - 36.8) < 0.1
    
    # Verify all data was stored in database
    vitals = temp_db.get_recent_vitals(days=1)
    assert len(vitals) == 3


@pytest.mark.asyncio
async def test_complete_wearable_ecosystem_integration(wearable_interface, event_bus, temp_db):
    """
    Test complete wearable ecosystem with all device types.
    
    Validates: Requirements 18.2
    
    Tests a realistic scenario with multiple device types:
    1. Heart rate monitor provides continuous HR data
    2. SpO2 sensor provides periodic SpO2 readings
    3. Temperature sensor provides temperature readings
    4. Fitness tracker provides daily activity summary
    5. All data flows correctly through the system
    6. Database contains complete health picture
    7. Events are emitted for all vitals
    """
    # Set up event listener
    events_received = []
    event_bus.on("vital_received", lambda e: events_received.append(e))
    
    # Simulate a complete health monitoring session
    
    # 1. Heart rate reading
    hr_data = bytearray([0x00, 72])
    await wearable_interface._handle_heart_rate(hr_data, "hr_device", "Polar H10")
    
    # 2. SpO2 reading (Nonin)
    spo2_data = bytearray([0x00, 98])
    await wearable_interface._handle_nonin_spo2(spo2_data, "spo2_device", "Nonin 3230")
    
    # 3. Temperature reading
    temp_data = bytearray([0x00]) + bytearray(struct.pack('<f', 37.0))
    await wearable_interface._handle_temperature(temp_data, "temp_device", "Kinsa Thermometer")
    
    # 4. Fitness tracker data
    await wearable_interface.ingest_activity_data(
        device_id="fitbit_device",
        device_name="Fitbit",
        steps=10000,
        calories=2500.0,
        active_minutes=60
    )
    
    # Verify all events were emitted (7 total: HR + SpO2 + Temp + Steps + Calories + Active)
    assert len(events_received) == 6
    
    # Verify all vital types are present
    vital_types = {e.data["vital_type"] for e in events_received}
    assert "heart_rate" in vital_types
    assert "spo2" in vital_types
    assert "temperature" in vital_types
    assert "steps" in vital_types
    assert "calories" in vital_types
    assert "active_minutes" in vital_types
    
    # Verify all data was stored in database
    vitals = temp_db.get_recent_vitals(days=1)
    assert len(vitals) == 6
    
    # Verify database contains all vital types
    db_vital_types = set()
    for vital in vitals:
        if vital.get("heart_rate") is not None:
            db_vital_types.add("heart_rate")
        if vital.get("spo2") is not None:
            db_vital_types.add("spo2")
        if vital.get("temperature") is not None:
            db_vital_types.add("temperature")
        if vital.get("steps") is not None:
            db_vital_types.add("steps")
        if vital.get("calories") is not None:
            db_vital_types.add("calories")
        if vital.get("active_minutes") is not None:
            db_vital_types.add("active_minutes")
    
    assert "heart_rate" in db_vital_types
    assert "spo2" in db_vital_types
    assert "temperature" in db_vital_types
    assert "steps" in db_vital_types
    assert "calories" in db_vital_types
    assert "active_minutes" in db_vital_types


# ─── Error Recovery Integration Tests ────────────────────────────────────

@pytest.mark.asyncio
async def test_database_error_recovery_integration(wearable_interface, event_bus):
    """
    Test error recovery when database is unavailable.
    
    Validates: Requirements 18.2
    
    Tests graceful degradation:
    1. Database connection fails
    2. Data is still validated
    3. Events are still emitted
    4. System continues to operate
    """
    # Remove database connection to simulate failure
    wearable_interface.db = None
    
    # Set up event listener
    events_received = []
    event_bus.on("vital_received", lambda e: events_received.append(e))
    
    # Send heart rate data
    hr_data = bytearray([0x00, 80])
    await wearable_interface._handle_heart_rate(hr_data, "test_device", "Test Device")
    
    # Verify event was still emitted despite database failure
    assert len(events_received) == 1
    event = events_received[0]
    assert event.data["vital_type"] == "heart_rate"
    assert event.data["value"] == 80.0


@pytest.mark.asyncio
async def test_malformed_data_recovery_integration(wearable_interface, event_bus, temp_db):
    """
    Test error recovery with malformed device data.
    
    Validates: Requirements 18.2
    
    Tests error handling:
    1. Device sends malformed data
    2. Error is caught and logged
    3. System continues to operate
    4. Subsequent valid data is processed correctly
    """
    # Set up event listener
    events_received = []
    event_bus.on("vital_received", lambda e: events_received.append(e))
    
    # Send malformed data (too short)
    malformed_data = bytearray([0x00])
    await wearable_interface._handle_heart_rate(malformed_data, "test_device", "Test Device")
    
    # Verify no event was emitted for malformed data
    assert len(events_received) == 0
    
    # Send valid data
    valid_data = bytearray([0x00, 75])
    await wearable_interface._handle_heart_rate(valid_data, "test_device", "Test Device")
    
    # Verify valid data was processed correctly
    assert len(events_received) == 1
    event = events_received[0]
    assert event.data["value"] == 75.0
    
    # Verify only valid data was stored
    vitals = temp_db.get_recent_vitals(days=1)
    assert len(vitals) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
