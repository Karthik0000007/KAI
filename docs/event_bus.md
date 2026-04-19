# Event Bus Documentation

## Overview

The Event Bus is a central message broker for inter-component communication in the Aegis async pipeline architecture. It enables decoupled, event-driven coordination between pipeline stages.

**Validates Requirements:** 3.6, 3.7, 3.8

## Features

- **Async Event Handling**: Built on asyncio for non-blocking event processing
- **Multiple Patterns**: Supports both callback-based and async iterator patterns
- **Event Type Registry**: Validates events against registered types
- **Thread-Safe**: Safe for concurrent access from multiple async tasks
- **Error Handling**: Gracefully handles handler errors without breaking emission
- **Statistics**: Built-in monitoring and statistics collection

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         Event Bus                            │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐ │
│  │   Callback   │    │    Async     │    │   Event      │ │
│  │   Handlers   │    │ Subscribers  │    │   Registry   │ │
│  └──────────────┘    └──────────────┘    └──────────────┘ │
│                                                              │
│  emit() ──────────────────────────────────────────────────> │
│         │                                                    │
│         ├──> Call handlers                                  │
│         └──> Put in subscriber queues                       │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Usage Patterns

### 1. Callback Pattern

Register handlers that are called when events are emitted:

```python
from core.event_bus import create_aegis_event_bus

bus = create_aegis_event_bus()

def on_stt_complete(event):
    print(f"Transcribed: {event.data['text']}")

bus.on("stt.completed", on_stt_complete)

await bus.emit("stt.completed", {
    "text": "Hello world",
    "language": "en"
})
```

### 2. Async Iterator Pattern

Subscribe to event streams using async iterators:

```python
async def process_events():
    async for event in bus.subscribe(["stt.completed", "emotion.analyzed"]):
        print(f"Event: {event.event_type}")
        print(f"Data: {event.data}")
        
        if some_condition:
            break  # Stop subscribing
```

### 3. Pipeline Coordination

Coordinate async pipeline stages with dependencies:

```python
async def pipeline():
    bus = create_aegis_event_bus()
    
    # Start independent stages in parallel
    await asyncio.gather(
        stt_stage(bus),      # Emits stt.completed
        emotion_stage(bus),  # Emits emotion.analyzed
        health_stage(bus),   # Waits for stt.completed, emits health.signals_extracted
        llm_stage(bus),      # Waits for all context, emits llm.response_generated
        tts_stage(bus)       # Waits for llm.response_generated
    )
```

## API Reference

### EventBus Class

#### `__init__()`
Initialize a new EventBus instance.

#### `register_event_type(event_type: str)`
Register a valid event type. Events must be registered before use.

#### `register_event_types(event_types: List[str])`
Register multiple event types at once.

#### `async emit(event_type: str, data: Dict[str, Any])`
Emit an event to all subscribers and handlers.

**Parameters:**
- `event_type`: The type of event (must be registered)
- `data`: Event data dictionary

**Raises:**
- `ValueError`: If event_type is not registered

#### `on(event_type: str, handler: Callable)`
Register a callback handler for an event type.

**Parameters:**
- `event_type`: The event type to listen for
- `handler`: Callback function (sync or async) that takes an Event object

#### `off(event_type: str, handler: Callable) -> bool`
Unregister a callback handler.

**Returns:** True if handler was removed, False if not found

#### `async subscribe(event_types: List[str]) -> AsyncIterator[Event]`
Subscribe to specific event types using an async iterator.

**Parameters:**
- `event_types`: List of event types to subscribe to

**Yields:** Event objects as they are emitted

**Usage:**
```python
async for event in bus.subscribe(["event.type"]):
    process(event)
```

#### `get_statistics() -> Dict[str, Any]`
Get event bus statistics including:
- `registered_types`: List of registered event types
- `event_counts`: Count of events emitted per type
- `handler_counts`: Count of handlers per type
- `subscriber_counts`: Count of active subscribers per type

#### `clear_handlers(event_type: Optional[str] = None)`
Clear handlers for a specific event type or all handlers.

#### `reset()`
Reset the event bus to initial state (clears handlers, subscribers, statistics).

### Event Class

Dataclass representing an event:

```python
@dataclass
class Event:
    event_type: str              # Type of event
    data: Dict[str, Any]         # Event data
    timestamp: datetime          # When event was created
    event_id: Optional[str]      # Unique event ID
```

### Helper Functions

#### `create_aegis_event_bus() -> EventBus`
Create an EventBus pre-configured with all Aegis event types.

## Predefined Event Types

The following event types are pre-registered in Aegis:

### Audio Pipeline
- `audio.recorded`
- `audio.vad_detected`

### STT (Speech-to-Text)
- `stt.started`
- `stt.completed`
- `stt.failed`

### Emotion Analysis
- `emotion.started`
- `emotion.analyzed`
- `emotion.failed`

### Vision
- `vision.captured`
- `vision.expression_detected`
- `vision.failed`

### Wearable Devices
- `wearable.connected`
- `wearable.disconnected`
- `wearable.vital_received`

### Health Signal Extraction
- `health.extraction_started`
- `health.signals_extracted`
- `health.extraction_failed`

### LLM
- `llm.started`
- `llm.response_generated`
- `llm.failed`

### TTS (Text-to-Speech)
- `tts.started`
- `tts.completed`
- `tts.failed`

### Database
- `db.checkin_saved`
- `db.vital_saved`
- `db.conversation_saved`

### Proactive Engine
- `proactive.alert_generated`
- `proactive.analysis_completed`

### Pipeline
- `pipeline.turn_started`
- `pipeline.turn_completed`
- `pipeline.turn_failed`

### System
- `system.startup`
- `system.shutdown`
- `system.error`

## Best Practices

### 1. Always Register Event Types
```python
bus = EventBus()
bus.register_event_type("my.custom.event")
```

### 2. Use Descriptive Event Names
Follow the pattern: `component.action`
- Good: `stt.completed`, `emotion.analyzed`
- Bad: `done`, `finished`, `event1`

### 3. Include Relevant Data
```python
await bus.emit("stt.completed", {
    "text": "transcribed text",
    "language": "en",
    "confidence": 0.95,
    "duration": 2.3
})
```

### 4. Handle Errors in Handlers
```python
def safe_handler(event):
    try:
        process(event)
    except Exception as e:
        logger.error(f"Handler error: {e}")
```

### 5. Clean Up Subscribers
```python
async def subscriber():
    try:
        async for event in bus.subscribe(["event.type"]):
            process(event)
    finally:
        # Cleanup happens automatically when iterator exits
        pass
```

### 6. Use Context Managers for Temporary Handlers
```python
class HandlerContext:
    def __init__(self, bus, event_type, handler):
        self.bus = bus
        self.event_type = event_type
        self.handler = handler
    
    def __enter__(self):
        self.bus.on(self.event_type, self.handler)
        return self
    
    def __exit__(self, *args):
        self.bus.off(self.event_type, self.handler)
```

## Performance Considerations

### Concurrent Emissions
The Event Bus is thread-safe and supports concurrent emissions:

```python
await asyncio.gather(
    bus.emit("event.one", {}),
    bus.emit("event.two", {}),
    bus.emit("event.three", {})
)
```

### Handler Execution
- Sync handlers are called directly
- Async handlers are awaited
- Handler errors don't block other handlers
- Handlers execute in registration order

### Subscriber Queues
- Each subscriber gets its own queue
- Queues are unbounded (be careful with slow consumers)
- Cleanup happens automatically when iterator exits

## Testing

The Event Bus includes comprehensive unit tests covering:
- Basic functionality (registration, emission)
- Callback handlers (sync and async)
- Async subscriptions
- Concurrency and thread-safety
- Error handling
- Statistics and monitoring
- Integration scenarios

Run tests:
```bash
pytest core/test_event_bus.py -v
```

## Examples

See `examples/event_bus_demo.py` for complete working examples including:
- Pipeline coordination
- Callback pattern
- Subscription pattern
- Monitoring and statistics

Run demo:
```bash
python examples/event_bus_demo.py
```

## Requirements Validation

This implementation validates the following requirements:

- **3.6**: WHERE parallel execution is possible, THE Aegis_System SHALL execute independent operations concurrently
  - ✓ Event Bus enables concurrent pipeline stages
  - ✓ Multiple stages can emit/subscribe simultaneously

- **3.7**: THE Aegis_System SHALL provide a configuration option to enable asynchronous pipeline execution
  - ✓ Event Bus provides async/await API
  - ✓ Supports both sync and async handlers

- **3.8**: WHEN asynchronous mode is enabled, THE Voice_Pipeline SHALL allow recording the next turn while TTS is playing
  - ✓ Event-driven architecture decouples stages
  - ✓ Stages can run independently based on events
