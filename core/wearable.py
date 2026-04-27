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

# Vital validation ranges
HEART_RATE_MIN = 30  # bpm
HEART_RATE_MAX = 220  # bpm


@dataclass
class VitalReading:
    """Represents a vital sign reading from a wearable device."""
    vital_type: str  # "heart_rate", "spo2", "temperature", "steps"
    value: float
    unit: str
    timestamp: datetime
    device_id: str
    confidence: float = 1.0


class WearableInterface:
    """
    Interface for connecting to and receiving data from BLE wearable devices.
    
    Supports:
    - Heart rate monitors (BLE Heart Rate Service)
    - Auto-reconnection to known devices
    - Data validation and event emission
    
    Requirements: 12.1, 12.5, 12.11, 12.12, 12.13
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
        Discover nearby BLE devices with Heart Rate Service.
        
        Args:
            timeout: Scan duration in seconds
            
        Returns:
            List of discovered devices with name, address, and services
            
        Requirements: 12.1
        """
        if not BLEAK_AVAILABLE:
            logger.error("Cannot discover devices - bleak not available")
            return []
        
        logger.info(f"Scanning for BLE devices for {timeout} seconds...")
        
        try:
            devices = await BleakScanner.discover(timeout=timeout, return_adv=True)
            heart_rate_devices = []
            
            for device, adv_data in devices.values():
                # Check if device advertises Heart Rate Service
                if HEART_RATE_SERVICE_UUID in adv_data.service_uuids:
                    device_info = {
                        "name": device.name or "Unknown",
                        "address": device.address,
                        "rssi": adv_data.rssi,
                        "services": ["heart_rate"]
                    }
                    heart_rate_devices.append(device_info)
                    self.known_devices[device.address] = device
                    logger.info(f"Found heart rate device: {device_info['name']} ({device.address})")
            
            return heart_rate_devices
            
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
            
        Requirements: 12.1, 12.11
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
                
                # Subscribe to heart rate notifications
                await self._subscribe_heart_rate(client, address)
                
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
            # Create callback with device address bound
            def callback(sender, data):
                asyncio.create_task(self._handle_heart_rate(data, address))
            
            await client.start_notify(HEART_RATE_MEASUREMENT_UUID, callback)
            logger.info(f"Subscribed to heart rate notifications from {address}")
            
        except Exception as e:
            logger.error(f"Error subscribing to heart rate: {e}")
    
    async def _handle_heart_rate(self, data: bytearray, device_address: str):
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
                    "device_id": reading.device_id
                })
                
                logger.info(f"Heart rate: {heart_rate} bpm")
            else:
                logger.warning(f"Invalid heart rate reading: {heart_rate} bpm (out of range {HEART_RATE_MIN}-{HEART_RATE_MAX})")
                
        except Exception as e:
            logger.error(f"Error parsing heart rate data: {e}")
    
    def validate_vital(self, vital_type: str, value: float) -> bool:
        """
        Validate a vital sign reading for plausibility.
        
        Args:
            vital_type: Type of vital ("heart_rate", "spo2", "temperature", "steps")
            value: Vital value to validate
            
        Returns:
            True if valid, False otherwise
            
        Requirements: 12.12
        """
        if vital_type == "heart_rate":
            return HEART_RATE_MIN <= value <= HEART_RATE_MAX
        elif vital_type == "spo2":
            return 70 <= value <= 100
        elif vital_type == "temperature":
            return 35.0 <= value <= 42.0
        elif vital_type == "steps":
            return 0 <= value <= 100000  # Reasonable daily step limit
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
            cursor = self.db.cursor()
            cursor.execute("""
                INSERT INTO vital_records 
                (timestamp, vital_type, value, unit, device_id, confidence)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                reading.timestamp.isoformat(),
                reading.vital_type,
                reading.value,
                reading.unit,
                reading.device_id,
                reading.confidence
            ))
            self.db.commit()
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
