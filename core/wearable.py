"""
Wearable Device Integration Module

This module provides BLE (Bluetooth Low Energy) integration for wearable health devices,
focusing on heart rate monitoring with support for standard BLE Heart Rate Service.

Requirements: 12.1, 12.5, 12.11, 12.12, 12.13, 12.14
"""

import asyncio
import logging
import struct
from datetime import datetime
from typing import Optional, Dict, List, Callable
from dataclasses import dataclass

try:
    from bleak import BleakScanner, BleakClient
    from bleak.backends.device import BLEDevice
    from bleak.exc import BleakError
    BLEAK_AVAILABLE = True
except ImportError:
    BLEAK_AVAILABLE = False
    BLEDevice = None
    BleakClient = None

from core.event_bus import EventBus

logger = logging.getLogger(__name__)

# BLE Standard UUIDs
HEART_RATE_SERVICE_UUID = "0000180d-0000-1000-8000-00805f9b34fb"
HEART_RATE_MEASUREMENT_UUID = "00002a37-0000-1000-8000-00805f9b34fb"

# Temperature Service (BLE Health Thermometer Service)
TEMPERATURE_SERVICE_UUID = "00001809-0000-1000-8000-00805f9b34fb"
TEMPERATURE_MEASUREMENT_UUID = "00002a1c-0000-1000-8000-00805f9b34fb"

# SpO2 Device-Specific UUIDs
# Nonin 3230 Pulse Oximeter
NONIN_SPO2_SERVICE_UUID = "46a970e0-0d5f-11e2-8b5e-0002a5d5c51b"
NONIN_SPO2_MEASUREMENT_UUID = "0aad7ea0-0d60-11e2-8e3c-0002a5d5c51b"

# Masimo MightySat Pulse Oximeter
MASIMO_SPO2_SERVICE_UUID = "74696d65-4465-6c61-7920-53657276696e"
MASIMO_SPO2_MEASUREMENT_UUID = "74696d65-4465-6c61-7920-436861726163"

# Vital validation ranges
HEART_RATE_MIN = 30  # bpm
HEART_RATE_MAX = 220  # bpm
SPO2_MIN = 70  # percentage
SPO2_MAX = 100  # percentage
TEMPERATURE_MIN = 35.0  # Celsius
TEMPERATURE_MAX = 42.0  # Celsius
STEPS_MIN = 0  # daily steps
STEPS_MAX = 100000  # daily steps
CALORIES_MIN = 0  # kcal
CALORIES_MAX = 10000  # kcal per day
ACTIVE_MINUTES_MIN = 0  # minutes
ACTIVE_MINUTES_MAX = 1440  # minutes per day (24 hours)


@dataclass
class VitalReading:
    """Represents a vital sign reading from a wearable device."""
    vital_type: str  # "heart_rate", "spo2", "temperature", "steps", "calories", "active_minutes"
    value: float
    unit: str
    timestamp: datetime
    device_id: str
    device_name: str = "Unknown"
    confidence: float = 1.0


class WearableInterface:
    """
    Interface for connecting to and receiving data from BLE wearable devices.
    
    Supports:
    - Heart rate monitors (BLE Heart Rate Service)
    - SpO2 sensors (Nonin 3230, Masimo MightySat)
    - Temperature sensors (BLE Health Thermometer Service)
    - Auto-reconnection to known devices
    - Data validation and event emission
    
    Requirements: 12.1, 12.2, 12.3, 12.5, 12.6, 12.7, 12.11, 12.12, 12.13
    """
    
    def __init__(self, event_bus: EventBus, db_connection=None):
        """
        Initialize the wearable interface.
        
        Args:
            event_bus: EventBus instance for emitting vital events
            db_connection: Database connection for storing vitals
        """
        if not BLEAK_AVAILABLE:
            logger.warning("bleak library not available - wearable integration disabled")
            logger.warning("Install with: pip install bleak")
        
        self.event_bus = event_bus
        self.db = db_connection
        self.connected_devices: Dict[str, BleakClient] = {}
        self.known_devices: Dict[str, BLEDevice] = {}
        self.running = False
        self._reconnect_task: Optional[asyncio.Task] = None
        
        # Register event types
        self.event_bus.register_event_type("vital_received")
        self.event_bus.register_event_type("wearable_connected")
        self.event_bus.register_event_type("wearable_disconnected")
        
    async def discover_devices(self, timeout: float = 10.0) -> List[Dict[str, str]]:
        """
        Discover nearby BLE devices with Heart Rate Service, SpO2 sensors, or Temperature Service.
        
        Args:
            timeout: Scan duration in seconds
            
        Returns:
            List of discovered devices with name, address, and services
            
        Requirements: 12.1, 12.2, 12.3
        """
        if not BLEAK_AVAILABLE:
            logger.error("Cannot discover devices - bleak not available")
            return []
        
        logger.info(f"Scanning for BLE devices for {timeout} seconds...")
        
        try:
            devices = await BleakScanner.discover(timeout=timeout, return_adv=True)
            discovered_devices = []
            
            for device, adv_data in devices.values():
                device_services = []
                
                # Check for Heart Rate Service
                if HEART_RATE_SERVICE_UUID in adv_data.service_uuids:
                    device_services.append("heart_rate")
                
                # Check for Temperature Service
                if TEMPERATURE_SERVICE_UUID in adv_data.service_uuids:
                    device_services.append("temperature")
                
                # Check for Nonin SpO2 Service
                if NONIN_SPO2_SERVICE_UUID in adv_data.service_uuids:
                    device_services.append("spo2_nonin")
                
                # Check for Masimo SpO2 Service
                if MASIMO_SPO2_SERVICE_UUID in adv_data.service_uuids:
                    device_services.append("spo2_masimo")
                
                # If device has any supported services, add it
                if device_services:
                    device_info = {
                        "name": device.name or "Unknown",
                        "address": device.address,
                        "rssi": adv_data.rssi,
                        "services": device_services
                    }
                    discovered_devices.append(device_info)
                    self.known_devices[device.address] = device
                    logger.info(f"Found device: {device_info['name']} ({device.address}) - Services: {device_services}")
            
            return discovered_devices
            
        except Exception as e:
            logger.error(f"Error during device discovery: {e}")
            return []
    
    async def connect_device(self, address: str) -> bool:
        """
        Connect to a BLE device by address.
        
        Args:
            address: BLE device address (MAC address)
            
        Returns:
            True if connection successful, False otherwise
            
        Requirements: 12.1, 12.2, 12.11
        """
        if not BLEAK_AVAILABLE:
            logger.error("Cannot connect - bleak not available")
            return False
        
        if address in self.connected_devices:
            logger.info(f"Already connected to {address}")
            return True
        
        try:
            device = self.known_devices.get(address)
            if not device:
                logger.error(f"Device {address} not in known devices - run discover_devices first")
                return False
            
            logger.info(f"Connecting to {device.name} ({address})...")
            client = BleakClient(device)
            await client.connect()
            
            if client.is_connected:
                self.connected_devices[address] = client
                logger.info(f"Successfully connected to {device.name}")
                
                # Subscribe to appropriate services based on device capabilities
                services = await client.get_services()
                
                # Subscribe to heart rate if available
                if any(s.uuid == HEART_RATE_SERVICE_UUID for s in services):
                    await self._subscribe_heart_rate(client, address)
                
                # Subscribe to temperature if available
                if any(s.uuid == TEMPERATURE_SERVICE_UUID for s in services):
                    await self._subscribe_temperature(client, address, device.name)
                
                # Subscribe to Nonin SpO2 if available
                if any(s.uuid == NONIN_SPO2_SERVICE_UUID for s in services):
                    await self._subscribe_nonin_spo2(client, address, device.name)
                
                # Subscribe to Masimo SpO2 if available
                if any(s.uuid == MASIMO_SPO2_SERVICE_UUID for s in services):
                    await self._subscribe_masimo_spo2(client, address, device.name)
                
                # Emit connection event
                await self.event_bus.emit("wearable_connected", {
                    "device_address": address,
                    "device_name": device.name,
                    "timestamp": datetime.now()
                })
                
                return True
            else:
                logger.error(f"Failed to connect to {address}")
                return False
                
        except Exception as e:
            logger.error(f"Error connecting to device {address}: {e}")
            return False
    
    async def disconnect_device(self, address: str) -> bool:
        """
        Disconnect from a BLE device.
        
        Args:
            address: BLE device address
            
        Returns:
            True if disconnection successful, False otherwise
        """
        if address not in self.connected_devices:
            logger.warning(f"Device {address} not connected")
            return False
        
        try:
            client = self.connected_devices[address]
            await client.disconnect()
            del self.connected_devices[address]
            
            logger.info(f"Disconnected from {address}")
            
            # Emit disconnection event
            await self.event_bus.emit("wearable_disconnected", {
                "device_address": address,
                "timestamp": datetime.now()
            })
            
            return True
            
        except Exception as e:
            logger.error(f"Error disconnecting from {address}: {e}")
            return False
    
    async def _subscribe_heart_rate(self, client: BleakClient, address: str):
        """
        Subscribe to heart rate measurement notifications.
        
        Args:
            client: Connected BleakClient
            address: Device address for callback context
            
        Requirements: 12.5
        """
        try:
            # Get device name from known_devices
            device = self.known_devices.get(address)
            device_name = device.name if device else "Unknown"
            
            # Create callback with device address and name bound
            def callback(sender, data):
                asyncio.create_task(self._handle_heart_rate(data, address, device_name))
            
            await client.start_notify(HEART_RATE_MEASUREMENT_UUID, callback)
            logger.info(f"Subscribed to heart rate notifications from {address}")
            
        except Exception as e:
            logger.error(f"Error subscribing to heart rate: {e}")
    
    async def _handle_heart_rate(self, data: bytearray, device_address: str, device_name: str = "Unknown"):
        """
        Parse and handle heart rate measurement data.
        
        BLE Heart Rate Measurement format (per Bluetooth spec):
        - Byte 0: Flags
          - Bit 0: 0 = uint8 bpm, 1 = uint16 bpm
          - Bit 1-2: Sensor contact status
          - Bit 3: Energy expended present
          - Bit 4: RR-Interval present
        - Byte 1+: Heart rate value (uint8 or uint16)
        
        Args:
            data: Raw BLE heart rate measurement data
            device_address: Address of device that sent the data
            device_name: Name of device that sent the data
            
        Requirements: 12.5, 12.12
        """
        try:
            flags = data[0]
            hr_format = flags & 0x01  # 0 = uint8, 1 = uint16
            
            if hr_format == 0:
                # uint8 format
                heart_rate = data[1]
            else:
                # uint16 format (little-endian)
                heart_rate = struct.unpack('<H', data[1:3])[0]
            
            logger.debug(f"Received heart rate: {heart_rate} bpm from {device_address}")
            
            # Validate the reading
            if self.validate_vital("heart_rate", heart_rate):
                reading = VitalReading(
                    vital_type="heart_rate",
                    value=float(heart_rate),
                    unit="bpm",
                    timestamp=datetime.now(),
                    device_id=device_address,
                    device_name=device_name,
                    confidence=1.0
                )
                
                # Store in database
                if self.db:
                    await self._store_vital(reading)
                
                # Emit event
                await self.event_bus.emit("vital_received", {
                    "vital_type": reading.vital_type,
                    "value": reading.value,
                    "unit": reading.unit,
                    "timestamp": reading.timestamp,
                    "device_id": reading.device_id,
                    "device_name": reading.device_name
                })
                
                logger.info(f"Heart rate: {heart_rate} bpm")
            else:
                logger.warning(f"Invalid heart rate reading: {heart_rate} bpm (out of range {HEART_RATE_MIN}-{HEART_RATE_MAX})")
                
        except Exception as e:
            logger.error(f"Error parsing heart rate data: {e}")
    
    async def _subscribe_temperature(self, client: BleakClient, address: str, device_name: str):
        """
        Subscribe to temperature measurement notifications.
        
        Args:
            client: Connected BleakClient
            address: Device address for callback context
            device_name: Device name for logging
            
        Requirements: 12.3, 12.7
        """
        try:
            # Create callback with device address and name bound
            def callback(sender, data):
                asyncio.create_task(self._handle_temperature(data, address, device_name))
            
            await client.start_notify(TEMPERATURE_MEASUREMENT_UUID, callback)
            logger.info(f"Subscribed to temperature notifications from {device_name} ({address})")
            
        except Exception as e:
            logger.error(f"Error subscribing to temperature: {e}")
    
    async def _handle_temperature(self, data: bytearray, device_address: str, device_name: str = "Unknown"):
        """
        Parse and handle temperature measurement data.
        
        BLE Temperature Measurement format (per Bluetooth spec - IEEE 11073):
        - Byte 0: Flags
          - Bit 0: 0 = Celsius, 1 = Fahrenheit
          - Bit 1: Time stamp present
          - Bit 2: Temperature type present
        - Bytes 1-4: Temperature value (IEEE-11073 32-bit float)
        - Optional: Time stamp (7 bytes if present)
        - Optional: Temperature type (1 byte if present)
        
        Args:
            data: Raw BLE temperature measurement data
            device_address: Address of device that sent the data
            device_name: Name of device that sent the data
            
        Requirements: 12.3, 12.7, 12.12
        """
        try:
            if len(data) < 5:
                logger.warning(f"Insufficient temperature data: {len(data)} bytes")
                return
            
            flags = data[0]
            unit_fahrenheit = flags & 0x01  # 0 = Celsius, 1 = Fahrenheit
            
            # Parse IEEE-11073 32-bit SFLOAT (simplified - treating as IEEE 754 float for common devices)
            # Many modern devices use standard IEEE 754 float instead of IEEE 11073 SFLOAT
            # For full IEEE 11073 support, we'd need to parse mantissa and exponent separately
            temperature_raw = struct.unpack('<f', data[1:5])[0]
            
            # Convert Fahrenheit to Celsius if needed
            if unit_fahrenheit:
                temperature_celsius = (temperature_raw - 32.0) * 5.0 / 9.0
                logger.debug(f"Converted temperature from {temperature_raw}°F to {temperature_celsius}°C")
            else:
                temperature_celsius = temperature_raw
            
            logger.debug(f"Received temperature: {temperature_celsius}°C from {device_name}")
            
            # Validate the reading
            if self.validate_vital("temperature", temperature_celsius):
                reading = VitalReading(
                    vital_type="temperature",
                    value=float(temperature_celsius),
                    unit="°C",
                    timestamp=datetime.now(),
                    device_id=device_address,
                    device_name=device_name,
                    confidence=1.0
                )
                
                # Store in database
                if self.db:
                    await self._store_vital(reading)
                
                # Emit event
                await self.event_bus.emit("vital_received", {
                    "vital_type": reading.vital_type,
                    "value": reading.value,
                    "unit": reading.unit,
                    "timestamp": reading.timestamp,
                    "device_id": reading.device_id,
                    "device_name": reading.device_name
                })
                
                logger.info(f"Temperature: {temperature_celsius:.1f}°C from {device_name}")
            else:
                logger.warning(f"Invalid temperature reading: {temperature_celsius}°C (out of range {TEMPERATURE_MIN}-{TEMPERATURE_MAX})")
                
        except Exception as e:
            logger.error(f"Error parsing temperature data: {e}")
    
    async def _subscribe_nonin_spo2(self, client: BleakClient, address: str, device_name: str):
        """
        Subscribe to Nonin 3230 SpO2 measurement notifications.
        
        Args:
            client: Connected BleakClient
            address: Device address for callback context
            device_name: Device name for logging
            
        Requirements: 12.2, 12.6
        """
        try:
            # Create callback with device address and name bound
            def callback(sender, data):
                asyncio.create_task(self._handle_nonin_spo2(data, address, device_name))
            
            await client.start_notify(NONIN_SPO2_MEASUREMENT_UUID, callback)
            logger.info(f"Subscribed to Nonin SpO2 notifications from {device_name} ({address})")
            
        except Exception as e:
            logger.error(f"Error subscribing to Nonin SpO2: {e}")
    
    async def _handle_nonin_spo2(self, data: bytearray, device_address: str, device_name: str):
        """
        Parse and handle Nonin 3230 SpO2 measurement data.
        
        Nonin 3230 data format (proprietary):
        - Byte 0: Status flags
        - Byte 1: SpO2 value (percentage)
        - Byte 2-3: Pulse rate (uint16, little-endian)
        - Byte 4: Signal quality (0-4)
        
        Args:
            data: Raw BLE SpO2 measurement data
            device_address: Address of device that sent the data
            device_name: Name of device that sent the data
            
        Requirements: 12.2, 12.6, 12.12
        """
        try:
            if len(data) < 2:
                logger.warning(f"Insufficient Nonin SpO2 data: {len(data)} bytes")
                return
            
            # Parse SpO2 value from byte 1
            spo2_value = data[1]
            
            # Parse pulse rate if available (bytes 2-3)
            pulse_rate = None
            if len(data) >= 4:
                pulse_rate = struct.unpack('<H', data[2:4])[0]
            
            # Parse signal quality if available (byte 4)
            signal_quality = None
            confidence = 1.0
            if len(data) >= 5:
                signal_quality = data[4]
                # Convert signal quality (0-4) to confidence (0.0-1.0)
                confidence = signal_quality / 4.0 if signal_quality <= 4 else 1.0
            
            logger.debug(f"Received Nonin SpO2: {spo2_value}% (pulse: {pulse_rate}, quality: {signal_quality}) from {device_name}")
            
            # Validate the reading
            if self.validate_vital("spo2", spo2_value):
                reading = VitalReading(
                    vital_type="spo2",
                    value=float(spo2_value),
                    unit="%",
                    timestamp=datetime.now(),
                    device_id=device_address,
                    device_name=device_name,
                    confidence=confidence
                )
                
                # Store in database
                if self.db:
                    await self._store_vital(reading)
                
                # Emit event
                await self.event_bus.emit("vital_received", {
                    "vital_type": reading.vital_type,
                    "value": reading.value,
                    "unit": reading.unit,
                    "timestamp": reading.timestamp,
                    "device_id": reading.device_id,
                    "device_name": reading.device_name,
                    "confidence": reading.confidence
                })
                
                logger.info(f"SpO2: {spo2_value}% from {device_name}")
            else:
                logger.warning(f"Invalid SpO2 reading: {spo2_value}% (out of range {SPO2_MIN}-{SPO2_MAX})")
                
        except Exception as e:
            logger.error(f"Error parsing Nonin SpO2 data: {e}")
    
    async def _subscribe_masimo_spo2(self, client: BleakClient, address: str, device_name: str):
        """
        Subscribe to Masimo MightySat SpO2 measurement notifications.
        
        Args:
            client: Connected BleakClient
            address: Device address for callback context
            device_name: Device name for logging
            
        Requirements: 12.2, 12.6
        """
        try:
            # Create callback with device address and name bound
            def callback(sender, data):
                asyncio.create_task(self._handle_masimo_spo2(data, address, device_name))
            
            await client.start_notify(MASIMO_SPO2_MEASUREMENT_UUID, callback)
            logger.info(f"Subscribed to Masimo SpO2 notifications from {device_name} ({address})")
            
        except Exception as e:
            logger.error(f"Error subscribing to Masimo SpO2: {e}")
    
    async def _handle_masimo_spo2(self, data: bytearray, device_address: str, device_name: str):
        """
        Parse and handle Masimo MightySat SpO2 measurement data.
        
        Masimo MightySat data format (proprietary):
        - Byte 0: Message type
        - Byte 1: SpO2 value (percentage)
        - Byte 2-3: Pulse rate (uint16, big-endian)
        - Byte 4: Perfusion index
        - Byte 5: Signal quality indicator
        
        Args:
            data: Raw BLE SpO2 measurement data
            device_address: Address of device that sent the data
            device_name: Name of device that sent the data
            
        Requirements: 12.2, 12.6, 12.12
        """
        try:
            if len(data) < 2:
                logger.warning(f"Insufficient Masimo SpO2 data: {len(data)} bytes")
                return
            
            # Parse SpO2 value from byte 1
            spo2_value = data[1]
            
            # Parse pulse rate if available (bytes 2-3, big-endian)
            pulse_rate = None
            if len(data) >= 4:
                pulse_rate = struct.unpack('>H', data[2:4])[0]
            
            # Parse signal quality if available (byte 5)
            signal_quality = None
            confidence = 1.0
            if len(data) >= 6:
                signal_quality = data[5]
                # Masimo signal quality is typically 0-100
                confidence = min(signal_quality / 100.0, 1.0) if signal_quality <= 100 else 1.0
            
            logger.debug(f"Received Masimo SpO2: {spo2_value}% (pulse: {pulse_rate}, quality: {signal_quality}) from {device_name}")
            
            # Validate the reading
            if self.validate_vital("spo2", spo2_value):
                reading = VitalReading(
                    vital_type="spo2",
                    value=float(spo2_value),
                    unit="%",
                    timestamp=datetime.now(),
                    device_id=device_address,
                    device_name=device_name,
                    confidence=confidence
                )
                
                # Store in database
                if self.db:
                    await self._store_vital(reading)
                
                # Emit event
                await self.event_bus.emit("vital_received", {
                    "vital_type": reading.vital_type,
                    "value": reading.value,
                    "unit": reading.unit,
                    "timestamp": reading.timestamp,
                    "device_id": reading.device_id,
                    "device_name": reading.device_name,
                    "confidence": reading.confidence
                })
                
                logger.info(f"SpO2: {spo2_value}% from {device_name}")
            else:
                logger.warning(f"Invalid SpO2 reading: {spo2_value}% (out of range {SPO2_MIN}-{SPO2_MAX})")
                
        except Exception as e:
            logger.error(f"Error parsing Masimo SpO2 data: {e}")
    
    def validate_vital(self, vital_type: str, value: float) -> bool:
        """
        Validate a vital sign reading for plausibility.
        
        Args:
            vital_type: Type of vital ("heart_rate", "spo2", "temperature", "steps", "calories", "active_minutes")
            value: Vital value to validate
            
        Returns:
            True if valid, False otherwise
            
        Requirements: 12.12
        """
        if vital_type == "heart_rate":
            return HEART_RATE_MIN <= value <= HEART_RATE_MAX
        elif vital_type == "spo2":
            return SPO2_MIN <= value <= SPO2_MAX
        elif vital_type == "temperature":
            return TEMPERATURE_MIN <= value <= TEMPERATURE_MAX
        elif vital_type == "steps":
            return STEPS_MIN <= value <= STEPS_MAX
        elif vital_type == "calories":
            return CALORIES_MIN <= value <= CALORIES_MAX
        elif vital_type == "active_minutes":
            return ACTIVE_MINUTES_MIN <= value <= ACTIVE_MINUTES_MAX
        else:
            logger.warning(f"Unknown vital type: {vital_type}")
            return False
    
    async def _store_vital(self, reading: VitalReading):
        """
        Store a vital reading in the database.
        
        Args:
            reading: VitalReading to store
            
        Requirements: 12.13
        """
        if not self.db:
            logger.warning("No database connection - vital not stored")
            return
        
        try:
            from core.models import VitalRecord
            
            # Create VitalRecord with appropriate field based on vital type
            vital_record = VitalRecord(
                timestamp=reading.timestamp.isoformat()
            )
            
            # Set the appropriate field based on vital type
            if reading.vital_type == "heart_rate":
                vital_record.heart_rate = int(reading.value)
            elif reading.vital_type == "spo2":
                vital_record.spo2 = reading.value
            elif reading.vital_type == "temperature":
                vital_record.temperature = reading.value
            elif reading.vital_type == "steps":
                vital_record.steps = int(reading.value)
            elif reading.vital_type == "calories":
                vital_record.calories = reading.value
            elif reading.vital_type == "active_minutes":
                vital_record.active_minutes = int(reading.value)
            
            # Save to database
            self.db.save_vital(vital_record)
            logger.debug(f"Stored {reading.vital_type} reading in database")
            
        except Exception as e:
            logger.error(f"Error storing vital in database: {e}")
    
    async def _monitor_connections(self):
        """
        Monitor device connections and attempt reconnection if disconnected.
        
        Requirements: 12.11
        """
        while self.running:
            try:
                # Check each connected device
                for address in list(self.connected_devices.keys()):
                    client = self.connected_devices[address]
                    
                    if not client.is_connected:
                        logger.warning(f"Device {address} disconnected - attempting reconnection")
                        
                        # Emit disconnection event
                        await self.event_bus.emit("wearable_disconnected", {
                            "device_address": address,
                            "timestamp": datetime.now(),
                            "unexpected": True
                        })
                        
                        # Remove from connected devices
                        del self.connected_devices[address]
                        
                        # Attempt reconnection
                        await asyncio.sleep(5)  # Wait before reconnecting
                        success = await self.connect_device(address)
                        
                        if success:
                            logger.info(f"Successfully reconnected to {address}")
                        else:
                            logger.error(f"Failed to reconnect to {address}")
                
                # Wait before next check
                await asyncio.sleep(10)
                
            except Exception as e:
                logger.error(f"Error in connection monitor: {e}")
                await asyncio.sleep(10)
    
    # ─── Fitness Tracker Integration ─────────────────────────────────────
    
    async def ingest_activity_data(self, device_id: str, device_name: str, 
                                   steps: Optional[int] = None,
                                   calories: Optional[float] = None,
                                   active_minutes: Optional[int] = None) -> bool:
        """
        Ingest activity data from a fitness tracker.
        
        This method provides a generic interface for ingesting activity data
        from any fitness tracker source. It validates the data, stores it in
        the database, and emits events.
        
        Args:
            device_id: Unique identifier for the device
            device_name: Human-readable device name
            steps: Step count (0-100000)
            calories: Calories burned in kcal (0-10000)
            active_minutes: Minutes of activity (0-1440)
            
        Returns:
            True if at least one valid reading was processed, False otherwise
            
        Requirements: 12.4, 12.8
        """
        timestamp = datetime.now()
        valid_count = 0
        
        # Process steps
        if steps is not None:
            if self.validate_vital("steps", steps):
                reading = VitalReading(
                    vital_type="steps",
                    value=float(steps),
                    unit="steps",
                    timestamp=timestamp,
                    device_id=device_id,
                    device_name=device_name,
                    confidence=1.0
                )
                
                # Store in database
                if self.db:
                    await self._store_vital(reading)
                
                # Emit event
                await self.event_bus.emit("vital_received", {
                    "vital_type": reading.vital_type,
                    "value": reading.value,
                    "unit": reading.unit,
                    "timestamp": reading.timestamp,
                    "device_id": reading.device_id,
                    "device_name": reading.device_name
                })
                
                valid_count += 1
                logger.info(f"Steps: {steps} from {device_name}")
            else:
                logger.warning(f"Invalid steps value: {steps} (out of range {STEPS_MIN}-{STEPS_MAX})")
        
        # Process calories
        if calories is not None:
            if self.validate_vital("calories", calories):
                reading = VitalReading(
                    vital_type="calories",
                    value=float(calories),
                    unit="kcal",
                    timestamp=timestamp,
                    device_id=device_id,
                    device_name=device_name,
                    confidence=1.0
                )
                
                # Store in database
                if self.db:
                    await self._store_vital(reading)
                
                # Emit event
                await self.event_bus.emit("vital_received", {
                    "vital_type": reading.vital_type,
                    "value": reading.value,
                    "unit": reading.unit,
                    "timestamp": reading.timestamp,
                    "device_id": reading.device_id,
                    "device_name": reading.device_name
                })
                
                valid_count += 1
                logger.info(f"Calories: {calories} kcal from {device_name}")
            else:
                logger.warning(f"Invalid calories value: {calories} (out of range {CALORIES_MIN}-{CALORIES_MAX})")
        
        # Process active minutes
        if active_minutes is not None:
            if self.validate_vital("active_minutes", active_minutes):
                reading = VitalReading(
                    vital_type="active_minutes",
                    value=float(active_minutes),
                    unit="minutes",
                    timestamp=timestamp,
                    device_id=device_id,
                    device_name=device_name,
                    confidence=1.0
                )
                
                # Store in database
                if self.db:
                    await self._store_vital(reading)
                
                # Emit event
                await self.event_bus.emit("vital_received", {
                    "vital_type": reading.vital_type,
                    "value": reading.value,
                    "unit": reading.unit,
                    "timestamp": reading.timestamp,
                    "device_id": reading.device_id,
                    "device_name": reading.device_name
                })
                
                valid_count += 1
                logger.info(f"Active minutes: {active_minutes} from {device_name}")
            else:
                logger.warning(f"Invalid active_minutes value: {active_minutes} (out of range {ACTIVE_MINUTES_MIN}-{ACTIVE_MINUTES_MAX})")
        
        return valid_count > 0
    
    # ─── Fitness Tracker Stubs (Placeholder Methods) ─────────────────────
    
    async def connect_fitbit(self, device_id: str) -> bool:
        """
        Connect to a Fitbit fitness tracker.
        
        PLACEHOLDER METHOD: This is a stub for future Fitbit integration.
        
        Fitbit devices use proprietary protocols and require OAuth authentication
        with the Fitbit Web API. To implement this:
        
        1. Register your application at https://dev.fitbit.com/
        2. Implement OAuth 2.0 flow to obtain access tokens
        3. Use the Fitbit Web API to fetch activity data:
           - GET /1/user/-/activities/date/{date}.json for daily summary
           - GET /1/user/-/activities/steps/date/{date}/1d.json for steps
        4. Poll the API hourly to fetch new data (Requirement 12.8)
        5. Call ingest_activity_data() with the fetched data
        
        Example implementation:
        ```python
        import requests
        
        # Fetch activity summary
        response = requests.get(
            f"https://api.fitbit.com/1/user/-/activities/date/{date}.json",
            headers={"Authorization": f"Bearer {access_token}"}
        )
        data = response.json()
        
        # Ingest the data
        await self.ingest_activity_data(
            device_id="fitbit_user_id",
            device_name="Fitbit",
            steps=data["summary"]["steps"],
            calories=data["summary"]["caloriesOut"],
            active_minutes=data["summary"]["fairlyActiveMinutes"] + 
                          data["summary"]["veryActiveMinutes"]
        )
        ```
        
        Args:
            device_id: Fitbit user ID or device identifier
            
        Returns:
            True if connection successful, False otherwise
            
        Requirements: 12.4, 12.8
        """
        logger.warning("Fitbit integration not yet implemented - this is a placeholder")
        logger.info("To implement Fitbit support, see the docstring for connect_fitbit()")
        return False
    
    async def connect_garmin(self, device_id: str) -> bool:
        """
        Connect to a Garmin fitness tracker.
        
        PLACEHOLDER METHOD: This is a stub for future Garmin integration.
        
        Garmin devices can be accessed via:
        1. Garmin Connect API (unofficial, requires reverse engineering)
        2. USB connection with FIT file parsing
        3. Bluetooth connection (device-specific protocols)
        
        Recommended approach - FIT file parsing:
        
        1. Install the fitparse library: pip install fitparse
        2. Connect to the device via USB and locate FIT files
        3. Parse activity files to extract steps, calories, and active time
        4. Call ingest_activity_data() with the parsed data
        
        Example implementation:
        ```python
        from fitparse import FitFile
        
        # Parse FIT file
        fitfile = FitFile('/path/to/activity.fit')
        
        steps = 0
        calories = 0
        active_seconds = 0
        
        for record in fitfile.get_messages('record'):
            for field in record:
                if field.name == 'total_steps':
                    steps = field.value
                elif field.name == 'total_calories':
                    calories = field.value
                elif field.name == 'total_timer_time':
                    active_seconds = field.value
        
        # Ingest the data
        await self.ingest_activity_data(
            device_id="garmin_device_id",
            device_name="Garmin",
            steps=steps,
            calories=calories,
            active_minutes=active_seconds // 60
        )
        ```
        
        Args:
            device_id: Garmin device identifier
            
        Returns:
            True if connection successful, False otherwise
            
        Requirements: 12.4, 12.8
        """
        logger.warning("Garmin integration not yet implemented - this is a placeholder")
        logger.info("To implement Garmin support, see the docstring for connect_garmin()")
        return False
    
    async def connect_mi_band(self, device_address: str) -> bool:
        """
        Connect to a Xiaomi Mi Band fitness tracker.
        
        PLACEHOLDER METHOD: This is a stub for future Mi Band integration.
        
        Xiaomi Mi Band devices use Bluetooth Low Energy (BLE) with custom
        protocols. To implement this:
        
        1. Install the bluepy library: pip install bluepy (Linux only)
           Or use bleak for cross-platform support (already available)
        2. Discover Mi Band devices by scanning for specific service UUIDs
        3. Authenticate with the device using the Mi Fit app pairing key
        4. Subscribe to activity data characteristics
        5. Parse the proprietary data format
        6. Call ingest_activity_data() with the parsed data
        
        Example implementation using bleak:
        ```python
        from bleak import BleakClient
        
        MI_BAND_SERVICE_UUID = "0000fee0-0000-1000-8000-00805f9b34fb"
        ACTIVITY_DATA_UUID = "00000007-0000-3512-2118-0009af100700"
        
        async def handle_activity_data(sender, data):
            # Parse Mi Band activity data format
            steps = int.from_bytes(data[1:5], byteorder='little')
            calories = int.from_bytes(data[5:9], byteorder='little') / 1000
            active_minutes = int.from_bytes(data[9:11], byteorder='little')
            
            await self.ingest_activity_data(
                device_id=device_address,
                device_name="Mi Band",
                steps=steps,
                calories=calories,
                active_minutes=active_minutes
            )
        
        client = BleakClient(device_address)
        await client.connect()
        await client.start_notify(ACTIVITY_DATA_UUID, handle_activity_data)
        ```
        
        Args:
            device_address: Mi Band BLE MAC address
            
        Returns:
            True if connection successful, False otherwise
            
        Requirements: 12.4, 12.8
        """
        logger.warning("Mi Band integration not yet implemented - this is a placeholder")
        logger.info("To implement Mi Band support, see the docstring for connect_mi_band()")
        return False
    
    async def start(self):
        """
        Start the wearable interface and connection monitoring.
        
        Requirements: 12.11
        """
        if not BLEAK_AVAILABLE:
            logger.error("Cannot start wearable interface - bleak not available")
            return
        
        self.running = True
        self._reconnect_task = asyncio.create_task(self._monitor_connections())
        logger.info("Wearable interface started")
    
    async def stop(self):
        """
        Stop the wearable interface and disconnect all devices.
        """
        self.running = False
        
        if self._reconnect_task:
            self._reconnect_task.cancel()
            try:
                await self._reconnect_task
            except asyncio.CancelledError:
                pass
        
        # Disconnect all devices
        for address in list(self.connected_devices.keys()):
            await self.disconnect_device(address)
        
        logger.info("Wearable interface stopped")
    
    # ─── Manual Vital Entry ──────────────────────────────────────────────
    
    async def enter_vital_manually(self, vital_type: str, value: float, 
                                   unit: Optional[str] = None) -> bool:
        """
        Manually enter a vital sign when wearables are unavailable.
        
        This method allows users to manually input vital signs when they don't
        have access to wearable devices. All manually entered vitals are:
        - Validated using the same validation logic as wearable data
        - Stored in the database with a "manual" device_id
        - Emitted to the event bus for proactive monitoring
        
        Args:
            vital_type: Type of vital ("heart_rate", "spo2", "temperature", 
                       "steps", "calories", "active_minutes")
            value: Vital value to record
            unit: Optional unit override (uses default if not provided)
            
        Returns:
            True if vital was valid and recorded, False otherwise
            
        Requirements: 12.15
        
        Example:
            >>> await wearable.enter_vital_manually("heart_rate", 72)
            True
            >>> await wearable.enter_vital_manually("temperature", 37.2)
            True
            >>> await wearable.enter_vital_manually("heart_rate", 300)
            False  # Invalid value
        """
        # Validate the vital
        if not self.validate_vital(vital_type, value):
            logger.warning(f"Invalid {vital_type} value: {value}")
            return False
        
        # Determine unit if not provided
        if unit is None:
            unit_map = {
                "heart_rate": "bpm",
                "spo2": "%",
                "temperature": "°C",
                "steps": "steps",
                "calories": "kcal",
                "active_minutes": "minutes"
            }
            unit = unit_map.get(vital_type, "")
        
        # Create reading
        reading = VitalReading(
            vital_type=vital_type,
            value=float(value),
            unit=unit,
            timestamp=datetime.now(),
            device_id="manual",
            device_name="Manual Entry",
            confidence=1.0
        )
        
        # Store in database
        if self.db:
            await self._store_vital(reading)
        
        # Emit event
        await self.event_bus.emit("vital_received", {
            "vital_type": reading.vital_type,
            "value": reading.value,
            "unit": reading.unit,
            "timestamp": reading.timestamp,
            "device_id": reading.device_id,
            "device_name": reading.device_name
        })
        
        logger.info(f"Manually entered {vital_type}: {value} {unit}")
        return True
