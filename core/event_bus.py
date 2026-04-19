"""
Event Bus Module - Central message broker for inter-component communication

This module provides an event-driven architecture for async pipeline coordination.
Supports both callback-based handlers and async iterator subscriptions.

**Validates: Requirements 3.6, 3.7, 3.8**
"""

import asyncio
import logging
from typing import Dict, Any, List, Callable, AsyncIterator, Set, Optional
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict
import threading


logger = logging.getLogger(__name__)


@dataclass
class Event:
    """Represents an event in the system"""
    event_type: str
    data: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    event_id: Optional[str] = None
    
    def __post_init__(self):
        if self.event_id is None:
            self.event_id = f"{self.event_type}_{self.timestamp.timestamp()}"


class EventBus:
    """
    Central message broker for inter-component communication.
    
    Provides three patterns for event handling:
    1. Callback-based: Register handlers with on()
    2. Async iterator: Subscribe to event streams with subscribe()
    3. Direct emission: Emit events to all subscribers with emit()
    
    Thread-safe for concurrent access from multiple async tasks.
    
    Example:
        >>> bus = EventBus()
        >>> bus.register_event_type("stt.completed")
        >>> 
        >>> # Callback pattern
        >>> def handler(event):
        >>>     print(f"Received: {event.data}")
        >>> bus.on("stt.completed", handler)
        >>> 
        >>> # Async iterator pattern
        >>> async for event in bus.subscribe(["stt.completed"]):
        >>>     print(f"Received: {event.data}")
        >>> 
        >>> # Emit event
        >>> await bus.emit("stt.completed", {"text": "Hello", "language": "en"})
    """
    
    def __init__(self):
        """Initialize the event bus"""
        # Event type registry for validation
        self._registered_types: Set[str] = set()
        
        # Callback handlers: event_type -> list of handlers
        self._handlers: Dict[str, List[Callable]] = defaultdict(list)
        
        # Async subscribers: event_type -> list of queues
        self._subscribers: Dict[str, List[asyncio.Queue]] = defaultdict(list)
        
        # Lock for thread-safe access
        self._lock = threading.Lock()
        
        # Statistics
        self._event_count: Dict[str, int] = defaultdict(int)
        
        logger.info("EventBus initialized")
    
    def register_event_type(self, event_type: str) -> None:
        """
        Register a valid event type.
        
        Args:
            event_type: The event type to register (e.g., "stt.completed")
        
        Raises:
            ValueError: If event_type is empty or invalid
        """
        if not event_type or not isinstance(event_type, str):
            raise ValueError(f"Invalid event type: {event_type}")
        
        with self._lock:
            self._registered_types.add(event_type)
            logger.debug(f"Registered event type: {event_type}")
    
    def register_event_types(self, event_types: List[str]) -> None:
        """
        Register multiple event types at once.
        
        Args:
            event_types: List of event types to register
        """
        for event_type in event_types:
            self.register_event_type(event_type)
    
    def is_registered(self, event_type: str) -> bool:
        """
        Check if an event type is registered.
        
        Args:
            event_type: The event type to check
        
        Returns:
            True if registered, False otherwise
        """
        with self._lock:
            return event_type in self._registered_types
    
    def _validate_event_type(self, event_type: str) -> None:
        """
        Validate that an event type is registered.
        
        Args:
            event_type: The event type to validate
        
        Raises:
            ValueError: If event type is not registered
        """
        if not self.is_registered(event_type):
            raise ValueError(
                f"Event type '{event_type}' is not registered. "
                f"Registered types: {sorted(self._registered_types)}"
            )
    
    async def emit(self, event_type: str, data: Dict[str, Any]) -> None:
        """
        Emit an event to all subscribers and handlers.
        
        This method:
        1. Validates the event type
        2. Creates an Event object
        3. Calls all registered callback handlers
        4. Puts the event in all subscriber queues
        
        Args:
            event_type: The type of event to emit
            data: Event data dictionary
        
        Raises:
            ValueError: If event_type is not registered
        
        Example:
            >>> await bus.emit("stt.completed", {
            >>>     "text": "Hello world",
            >>>     "language": "en",
            >>>     "confidence": 0.95
            >>> })
        """
        # Validate event type
        self._validate_event_type(event_type)
        
        # Create event object
        event = Event(event_type=event_type, data=data)
        
        # Update statistics
        with self._lock:
            self._event_count[event_type] += 1
        
        logger.debug(f"Emitting event: {event_type} (id={event.event_id})")
        
        # Call callback handlers
        handlers = self._handlers.get(event_type, [])
        for handler in handlers:
            try:
                # Support both sync and async handlers
                if asyncio.iscoroutinefunction(handler):
                    await handler(event)
                else:
                    handler(event)
            except Exception as e:
                logger.error(
                    f"Error in handler for {event_type}: {e}",
                    exc_info=True
                )
        
        # Put event in subscriber queues
        subscribers = self._subscribers.get(event_type, [])
        for queue in subscribers:
            try:
                await queue.put(event)
            except Exception as e:
                logger.error(
                    f"Error putting event in subscriber queue: {e}",
                    exc_info=True
                )
    
    def on(self, event_type: str, handler: Callable) -> None:
        """
        Register a callback handler for an event type.
        
        The handler will be called whenever an event of this type is emitted.
        Handlers can be either sync or async functions.
        
        Args:
            event_type: The event type to listen for
            handler: Callback function that takes an Event object
        
        Raises:
            ValueError: If event_type is not registered
        
        Example:
            >>> def on_stt_complete(event: Event):
            >>>     print(f"Transcribed: {event.data['text']}")
            >>> 
            >>> bus.on("stt.completed", on_stt_complete)
        """
        # Validate event type
        self._validate_event_type(event_type)
        
        # Register handler
        with self._lock:
            self._handlers[event_type].append(handler)
        
        logger.debug(f"Registered handler for {event_type}: {handler.__name__}")
    
    def off(self, event_type: str, handler: Callable) -> bool:
        """
        Unregister a callback handler.
        
        Args:
            event_type: The event type
            handler: The handler to remove
        
        Returns:
            True if handler was removed, False if not found
        """
        with self._lock:
            handlers = self._handlers.get(event_type, [])
            if handler in handlers:
                handlers.remove(handler)
                logger.debug(f"Unregistered handler for {event_type}: {handler.__name__}")
                return True
        return False
    
    async def subscribe(self, event_types: List[str]) -> AsyncIterator[Event]:
        """
        Subscribe to specific event types using an async iterator.
        
        This method returns an async iterator that yields events as they occur.
        The iterator will continue indefinitely until the task is cancelled.
        
        Args:
            event_types: List of event types to subscribe to
        
        Yields:
            Event objects as they are emitted
        
        Raises:
            ValueError: If any event_type is not registered
        
        Example:
            >>> async for event in bus.subscribe(["stt.completed", "emotion.analyzed"]):
            >>>     print(f"Event: {event.event_type}")
            >>>     print(f"Data: {event.data}")
            >>>     if event.event_type == "stt.completed":
            >>>         text = event.data["text"]
        """
        # Validate all event types
        for event_type in event_types:
            self._validate_event_type(event_type)
        
        # Create a queue for this subscriber
        queue: asyncio.Queue = asyncio.Queue()
        
        # Register queue for all requested event types
        with self._lock:
            for event_type in event_types:
                self._subscribers[event_type].append(queue)
        
        logger.debug(f"New subscriber for event types: {event_types}")
        
        try:
            # Yield events from the queue
            while True:
                event = await queue.get()
                yield event
        finally:
            # Cleanup: remove queue from all event types
            with self._lock:
                for event_type in event_types:
                    if queue in self._subscribers[event_type]:
                        self._subscribers[event_type].remove(queue)
            
            logger.debug(f"Subscriber unsubscribed from: {event_types}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get event bus statistics.
        
        Returns:
            Dictionary with statistics including:
            - registered_types: List of registered event types
            - event_counts: Count of events emitted per type
            - handler_counts: Count of handlers per type
            - subscriber_counts: Count of active subscribers per type
        """
        with self._lock:
            return {
                "registered_types": sorted(self._registered_types),
                "event_counts": dict(self._event_count),
                "handler_counts": {
                    event_type: len(handlers)
                    for event_type, handlers in self._handlers.items()
                },
                "subscriber_counts": {
                    event_type: len(subscribers)
                    for event_type, subscribers in self._subscribers.items()
                }
            }
    
    def clear_handlers(self, event_type: Optional[str] = None) -> None:
        """
        Clear all handlers for a specific event type or all handlers.
        
        Args:
            event_type: Event type to clear handlers for, or None to clear all
        """
        with self._lock:
            if event_type is None:
                self._handlers.clear()
                logger.info("Cleared all event handlers")
            else:
                self._handlers[event_type].clear()
                logger.info(f"Cleared handlers for {event_type}")
    
    def reset(self) -> None:
        """
        Reset the event bus to initial state.
        
        Clears all handlers, subscribers, and statistics.
        Does NOT clear registered event types.
        """
        with self._lock:
            self._handlers.clear()
            self._subscribers.clear()
            self._event_count.clear()
        
        logger.info("EventBus reset")


# Predefined event types for the Aegis system
AEGIS_EVENT_TYPES = [
    # Audio pipeline events
    "audio.recorded",
    "audio.vad_detected",
    
    # STT events
    "stt.started",
    "stt.completed",
    "stt.failed",
    
    # Emotion analysis events
    "emotion.started",
    "emotion.analyzed",
    "emotion.failed",
    
    # Vision events
    "vision.captured",
    "vision.expression_detected",
    "vision.failed",
    
    # Wearable events
    "wearable.connected",
    "wearable.disconnected",
    "wearable.vital_received",
    
    # Health signal extraction events
    "health.extraction_started",
    "health.signals_extracted",
    "health.extraction_failed",
    
    # LLM events
    "llm.started",
    "llm.response_generated",
    "llm.failed",
    
    # TTS events
    "tts.started",
    "tts.completed",
    "tts.failed",
    
    # Database events
    "db.checkin_saved",
    "db.vital_saved",
    "db.conversation_saved",
    
    # Proactive engine events
    "proactive.alert_generated",
    "proactive.analysis_completed",
    
    # Pipeline events
    "pipeline.turn_started",
    "pipeline.turn_completed",
    "pipeline.turn_failed",
    
    # System events
    "system.startup",
    "system.shutdown",
    "system.error",
]


def create_aegis_event_bus() -> EventBus:
    """
    Create an EventBus pre-configured with Aegis event types.
    
    Returns:
        EventBus instance with all Aegis event types registered
    """
    bus = EventBus()
    bus.register_event_types(AEGIS_EVENT_TYPES)
    logger.info(f"Created Aegis EventBus with {len(AEGIS_EVENT_TYPES)} event types")
    return bus
