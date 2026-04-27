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
    # Create mock data
    data = bytearray([0x00, 75])
    
    # Handle the data
    await wearable_interface._handle_heart_rate(data, "test_device")
    
    # Verify database insert was called
    cursor = mock_db.cursor.return_value
    cursor.execute.assert_called_once()
    
    # Verify the SQL query
    call_args = cursor.execute.call_args
    assert "INSERT INTO vital_records" in call_args[0][0]
    assert call_args[0][1][1] == "heart_rate"  # vital_type
    assert call_args[0][1][2] == 75.0  # value
    assert call_args[0][1][3] == "bpm"  # unit
    assert call_args[0][1][4] == "test_device"  # device_id


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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
