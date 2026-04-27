"""
Integration tests for wearable device integration.

Tests cover:
- End-to-end BLE device discovery and connection
- Heart rate data flow from device to database
- Event bus integration
- Proactive engine integration with wearable data

Requirements: 18.2
"""

import pytest
import asyncio
import sqlite3
import struct
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch

from core.wearable import WearableInterface, BLEAK_AVAILABLE
from core.event_bus import EventBus
from core.health_db import HealthDatabase
from core.proactive import ProactiveEngine


# ─── Fixtures ────────────────────────────────────────────────────────────

@pytest.fixture
def temp_db(tmp_path):
    """Create a temporary test database."""
    db_path = tmp_path / "test_wearable.db"
    db = HealthDatabase(db_path)
    yield db
    db.close()


@pytest.fixture
def event_bus():
    """Create an event bus for testing."""
    return EventBus()


@pytest.fixture
def wearable_interface(event_bus, temp_db):
    """Create a wearable interface with real database."""
    return WearableInterface(event_bus, temp_db._get_conn())


# ─── Integration Tests ───────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_heart_rate_data_flow_to_database(wearable_interface, temp_db, event_bus):
    """
    Test complete data flow: heart rate data → parsing → validation → database storage.
    
    Requirements: 12.5, 12.12, 12.13
    """
    # Create mock heart rate data (75 bpm)
    data = bytearray([0x00, 75])
    
    # Set up event listener to verify event emission
    events_received = []
    event_bus.on("vital_received", lambda e: events_received.append(e))
    
    # Handle the heart rate data
    await wearable_interface._handle_heart_rate(data, "test_device_123")
    
    # Verify event was emitted
    assert len(events_received) == 1
    event = events_received[0]
    assert event["vital_type"] == "heart_rate"
    assert event["value"] == 75.0
    assert event["device_id"] == "test_device_123"
    
    # Verify data was stored in database
    vitals = temp_db.get_recent_vitals(days=1)
    assert len(vitals) > 0
    
    # Find our vital record
    hr_vitals = [v for v in vitals if v.get("vital_type") == "heart_rate"]
    assert len(hr_vitals) == 1
    
    vital = hr_vitals[0]
    assert vital["value"] == 75.0
    assert vital["unit"] == "bpm"
    assert vital["device_id"] == "test_device_123"


@pytest.mark.asyncio
async def test_multiple_heart_rate_readings(wearable_interface, temp_db):
    """
    Test storing multiple heart rate readings over time.
    
    Requirements: 12.5, 12.13
    """
    # Simulate multiple readings
    readings = [60, 65, 70, 75, 80]
    
    for hr in readings:
        data = bytearray([0x00, hr])
        await wearable_interface._handle_heart_rate(data, "test_device")
        await asyncio.sleep(0.01)  # Small delay between readings
    
    # Verify all readings were stored
    vitals = temp_db.get_recent_vitals(days=1)
    hr_vitals = [v for v in vitals if v.get("vital_type") == "heart_rate"]
    
    assert len(hr_vitals) == len(readings)
    
    # Verify values match
    stored_values = sorted([v["value"] for v in hr_vitals])
    assert stored_values == sorted([float(r) for r in readings])


@pytest.mark.asyncio
async def test_invalid_heart_rate_not_stored(wearable_interface, temp_db):
    """
    Test that invalid heart rate values are rejected and not stored.
    
    Requirements: 12.12
    """
    # Create invalid heart rate data (250 bpm - out of range)
    data = bytearray([0x00, 250])
    
    # Handle the data
    await wearable_interface._handle_heart_rate(data, "test_device")
    
    # Verify no data was stored
    vitals = temp_db.get_recent_vitals(days=1)
    hr_vitals = [v for v in vitals if v.get("vital_type") == "heart_rate"]
    
    assert len(hr_vitals) == 0


@pytest.mark.asyncio
async def test_proactive_engine_detects_elevated_heart_rate(temp_db):
    """
    Test that proactive engine detects elevated heart rate patterns.
    
    Requirements: 12.14
    """
    # Create proactive engine
    engine = ProactiveEngine(temp_db)
    
    # Manually insert elevated heart rate readings
    conn = temp_db._get_conn()
    cursor = conn.cursor()
    
    # Insert 3 readings over 10 minutes, all > 100 bpm
    base_time = datetime.now()
    for i in range(3):
        timestamp = base_time + timedelta(minutes=i * 5)
        cursor.execute("""
            INSERT INTO vital_records 
            (id, timestamp, vital_type, value, unit, device_id, confidence)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            f"vital_{i}",
            timestamp.isoformat(),
            "heart_rate",
            110.0 + i * 5,  # 110, 115, 120 bpm
            "bpm",
            "test_device",
            1.0
        ))
    conn.commit()
    
    # Run proactive analysis
    alerts = engine.run_analysis()
    
    # Verify sustained elevated heart rate alert was generated
    elevated_hr_alerts = [
        a for a in alerts 
        if a.alert_type in ("sustained_elevated_hr", "elevated_hr")
    ]
    
    assert len(elevated_hr_alerts) > 0
    alert = elevated_hr_alerts[0]
    assert "heart rate" in alert.message.lower()


@pytest.mark.asyncio
async def test_event_bus_integration(event_bus, wearable_interface):
    """
    Test that wearable interface properly integrates with event bus.
    
    Requirements: 12.13
    """
    # Set up multiple event listeners
    vital_events = []
    connection_events = []
    
    event_bus.on("vital_received", lambda e: vital_events.append(e))
    event_bus.on("wearable_connected", lambda e: connection_events.append(e))
    event_bus.on("wearable_disconnected", lambda e: connection_events.append(e))
    
    # Simulate heart rate data
    data = bytearray([0x00, 80])
    await wearable_interface._handle_heart_rate(data, "test_device")
    
    # Verify vital event was emitted
    assert len(vital_events) == 1
    assert vital_events[0]["vital_type"] == "heart_rate"
    assert vital_events[0]["value"] == 80.0


@pytest.mark.skipif(not BLEAK_AVAILABLE, reason="bleak not installed")
@pytest.mark.asyncio
async def test_device_discovery_and_connection_flow():
    """
    Test the complete flow: discover devices → connect → receive data.
    
    This is a mock-based test since we don't have real BLE devices in CI.
    
    Requirements: 12.1, 12.11
    """
    from core.wearable import HEART_RATE_SERVICE_UUID
    
    event_bus = EventBus()
    wearable = WearableInterface(event_bus, None)
    
    # Mock device discovery
    mock_device = Mock()
    mock_device.name = "Test HR Monitor"
    mock_device.address = "AA:BB:CC:DD:EE:FF"
    
    mock_adv = Mock()
    mock_adv.service_uuids = [HEART_RATE_SERVICE_UUID]
    mock_adv.rssi = -50
    
    with patch('core.wearable.BleakScanner.discover') as mock_discover:
        mock_discover.return_value = {
            "device1": (mock_device, mock_adv)
        }
        
        # Discover devices
        devices = await wearable.discover_devices(timeout=1.0)
        
        assert len(devices) == 1
        assert devices[0]["address"] == "AA:BB:CC:DD:EE:FF"
        
        # Mock connection
        with patch('core.wearable.BleakClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client.is_connected = True
            mock_client.connect = AsyncMock()
            mock_client.start_notify = AsyncMock()
            mock_client_class.return_value = mock_client
            
            # Connect to device
            success = await wearable.connect_device("AA:BB:CC:DD:EE:FF")
            
            assert success is True
            assert "AA:BB:CC:DD:EE:FF" in wearable.connected_devices


@pytest.mark.asyncio
async def test_concurrent_heart_rate_processing(wearable_interface, temp_db):
    """
    Test processing multiple heart rate readings concurrently.
    
    Requirements: 12.5, 12.13
    """
    # Create multiple concurrent heart rate readings
    readings = [60, 65, 70, 75, 80, 85, 90, 95]
    
    # Process all readings concurrently
    tasks = []
    for i, hr in enumerate(readings):
        data = bytearray([0x00, hr])
        task = wearable_interface._handle_heart_rate(data, f"device_{i % 2}")
        tasks.append(task)
    
    await asyncio.gather(*tasks)
    
    # Verify all readings were stored
    vitals = temp_db.get_recent_vitals(days=1)
    hr_vitals = [v for v in vitals if v.get("vital_type") == "heart_rate"]
    
    assert len(hr_vitals) == len(readings)


@pytest.mark.asyncio
async def test_heart_rate_with_stressed_emotion_correlation(temp_db):
    """
    Test proactive engine correlates elevated heart rate with stressed emotion.
    
    Requirements: 12.14
    """
    from core.models import HealthCheckIn
    
    # Insert elevated heart rate
    conn = temp_db._get_conn()
    cursor = conn.cursor()
    
    timestamp = datetime.now()
    cursor.execute("""
        INSERT INTO vital_records 
        (id, timestamp, vital_type, value, unit, device_id, confidence)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        "vital_1",
        timestamp.isoformat(),
        "heart_rate",
        120.0,
        "bpm",
        "test_device",
        1.0
    ))
    
    # Insert stressed emotion check-in
    checkin = HealthCheckIn(
        mood_score=4.0,
        detected_emotion="stressed",
        emotion_confidence=0.8
    )
    temp_db.save_checkin(checkin)
    
    conn.commit()
    
    # Run proactive analysis
    engine = ProactiveEngine(temp_db)
    alerts = engine.run_analysis()
    
    # Verify elevated HR + stress alert was generated
    stress_alerts = [
        a for a in alerts 
        if "elevated_hr_stress" in a.alert_type
    ]
    
    assert len(stress_alerts) > 0
    alert = stress_alerts[0]
    assert "stressed" in alert.message.lower() or "breathing" in alert.message.lower()


@pytest.mark.asyncio
async def test_database_schema_supports_wearable_data(temp_db):
    """
    Test that database schema properly supports wearable vital records.
    
    Requirements: 12.13
    """
    conn = temp_db._get_conn()
    cursor = conn.cursor()
    
    # Verify vital_records table has required columns
    cursor.execute("PRAGMA table_info(vital_records)")
    columns = {row[1] for row in cursor.fetchall()}
    
    required_columns = {
        "id", "timestamp", "vital_type", "value", 
        "unit", "device_id", "confidence"
    }
    
    assert required_columns.issubset(columns), \
        f"Missing columns: {required_columns - columns}"
    
    # Test inserting a vital record
    cursor.execute("""
        INSERT INTO vital_records 
        (id, timestamp, vital_type, value, unit, device_id, confidence)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        "test_vital",
        datetime.now().isoformat(),
        "heart_rate",
        75.0,
        "bpm",
        "test_device",
        1.0
    ))
    conn.commit()
    
    # Verify it was inserted
    cursor.execute("SELECT * FROM vital_records WHERE id = ?", ("test_vital",))
    row = cursor.fetchone()
    
    assert row is not None
    assert dict(row)["vital_type"] == "heart_rate"
    assert dict(row)["value"] == 75.0


@pytest.mark.skipif(not BLEAK_AVAILABLE, reason="bleak not installed")
@pytest.mark.asyncio
async def test_auto_reconnection_on_disconnect():
    """
    Test that wearable interface attempts to reconnect when device disconnects.
    
    Requirements: 12.11
    """
    event_bus = EventBus()
    wearable = WearableInterface(event_bus, None)
    
    # Set up mock connected device
    mock_client = AsyncMock()
    mock_client.is_connected = False  # Simulate disconnection
    mock_client.disconnect = AsyncMock()
    
    mock_device = Mock()
    mock_device.name = "Test Device"
    mock_device.address = "AA:BB:CC:DD:EE:FF"
    
    wearable.connected_devices["AA:BB:CC:DD:EE:FF"] = mock_client
    wearable.known_devices["AA:BB:CC:DD:EE:FF"] = mock_device
    
    # Track disconnection events
    disconnect_events = []
    event_bus.on("wearable_disconnected", lambda e: disconnect_events.append(e))
    
    # Mock reconnection
    with patch.object(wearable, 'connect_device', new_callable=AsyncMock) as mock_connect:
        mock_connect.return_value = True
        
        # Start monitoring (will detect disconnection)
        wearable.running = True
        
        # Run one iteration of connection monitoring
        await wearable._monitor_connections()
        
        # Give it time to process
        await asyncio.sleep(0.1)
        
        # Verify disconnection was detected
        assert len(disconnect_events) > 0
        assert disconnect_events[0]["device_address"] == "AA:BB:CC:DD:EE:FF"
        assert disconnect_events[0].get("unexpected") is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
