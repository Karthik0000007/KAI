"""
Unit tests for Event Bus module

**Validates: Requirements 3.6, 3.7, 3.8**
"""

import pytest
import asyncio
from datetime import datetime
from core.event_bus import EventBus, Event, create_aegis_event_bus, AEGIS_EVENT_TYPES


class TestEventBusBasics:
    """Test basic Event Bus functionality"""
    
    def test_event_bus_initialization(self):
        """Test that EventBus initializes correctly"""
        bus = EventBus()
        assert bus is not None
        stats = bus.get_statistics()
        assert stats["registered_types"] == []
        assert stats["event_counts"] == {}
    
    def test_event_creation(self):
        """Test Event dataclass creation"""
        event = Event(
            event_type="test.event",
            data={"key": "value"}
        )
        assert event.event_type == "test.event"
        assert event.data == {"key": "value"}
        assert isinstance(event.timestamp, datetime)
        assert event.event_id is not None
    
    def test_register_event_type(self):
        """Test registering event types"""
        bus = EventBus()
        bus.register_event_type("test.event")
        assert bus.is_registered("test.event")
        assert not bus.is_registered("other.event")
    
    def test_register_multiple_event_types(self):
        """Test registering multiple event types at once"""
        bus = EventBus()
        event_types = ["event.one", "event.two", "event.three"]
        bus.register_event_types(event_types)
        
        for event_type in event_types:
            assert bus.is_registered(event_type)
    
    def test_register_invalid_event_type(self):
        """Test that invalid event types raise ValueError"""
        bus = EventBus()
        
        with pytest.raises(ValueError):
            bus.register_event_type("")
        
        with pytest.raises(ValueError):
            bus.register_event_type(None)


class TestEventEmission:
    """Test event emission functionality"""
    
    @pytest.mark.asyncio
    async def test_emit_unregistered_event_type(self):
        """Test that emitting unregistered event type raises ValueError"""
        bus = EventBus()
        
        with pytest.raises(ValueError, match="not registered"):
            await bus.emit("unregistered.event", {"data": "value"})
    
    @pytest.mark.asyncio
    async def test_emit_registered_event(self):
        """Test emitting a registered event"""
        bus = EventBus()
        bus.register_event_type("test.event")
        
        # Should not raise
        await bus.emit("test.event", {"key": "value"})
        
        # Check statistics
        stats = bus.get_statistics()
        assert stats["event_counts"]["test.event"] == 1
    
    @pytest.mark.asyncio
    async def test_emit_multiple_events(self):
        """Test emitting multiple events updates statistics"""
        bus = EventBus()
        bus.register_event_type("test.event")
        
        for i in range(5):
            await bus.emit("test.event", {"count": i})
        
        stats = bus.get_statistics()
        assert stats["event_counts"]["test.event"] == 5


class TestCallbackHandlers:
    """Test callback-based event handlers"""
    
    @pytest.mark.asyncio
    async def test_register_handler(self):
        """Test registering a callback handler"""
        bus = EventBus()
        bus.register_event_type("test.event")
        
        received_events = []
        
        def handler(event: Event):
            received_events.append(event)
        
        bus.on("test.event", handler)
        
        # Emit event
        await bus.emit("test.event", {"key": "value"})
        
        # Check handler was called
        assert len(received_events) == 1
        assert received_events[0].event_type == "test.event"
        assert received_events[0].data == {"key": "value"}
    
    @pytest.mark.asyncio
    async def test_multiple_handlers(self):
        """Test multiple handlers for same event type"""
        bus = EventBus()
        bus.register_event_type("test.event")
        
        handler1_calls = []
        handler2_calls = []
        
        def handler1(event: Event):
            handler1_calls.append(event)
        
        def handler2(event: Event):
            handler2_calls.append(event)
        
        bus.on("test.event", handler1)
        bus.on("test.event", handler2)
        
        # Emit event
        await bus.emit("test.event", {"key": "value"})
        
        # Both handlers should be called
        assert len(handler1_calls) == 1
        assert len(handler2_calls) == 1
    
    @pytest.mark.asyncio
    async def test_async_handler(self):
        """Test async callback handlers"""
        bus = EventBus()
        bus.register_event_type("test.event")
        
        received_events = []
        
        async def async_handler(event: Event):
            await asyncio.sleep(0.01)  # Simulate async work
            received_events.append(event)
        
        bus.on("test.event", async_handler)
        
        # Emit event
        await bus.emit("test.event", {"key": "value"})
        
        # Check handler was called
        assert len(received_events) == 1
    
    @pytest.mark.asyncio
    async def test_handler_error_handling(self):
        """Test that handler errors don't break event emission"""
        bus = EventBus()
        bus.register_event_type("test.event")
        
        good_handler_calls = []
        
        def bad_handler(event: Event):
            raise RuntimeError("Handler error")
        
        def good_handler(event: Event):
            good_handler_calls.append(event)
        
        bus.on("test.event", bad_handler)
        bus.on("test.event", good_handler)
        
        # Emit event - should not raise despite bad_handler error
        await bus.emit("test.event", {"key": "value"})
        
        # Good handler should still be called
        assert len(good_handler_calls) == 1
    
    @pytest.mark.asyncio
    async def test_unregister_handler(self):
        """Test unregistering a handler"""
        bus = EventBus()
        bus.register_event_type("test.event")
        
        handler_calls = []
        
        def handler(event: Event):
            handler_calls.append(event)
        
        bus.on("test.event", handler)
        
        # Emit event
        await bus.emit("test.event", {"key": "value"})
        assert len(handler_calls) == 1
        
        # Unregister handler
        removed = bus.off("test.event", handler)
        assert removed is True
        
        # Emit again - handler should not be called
        await bus.emit("test.event", {"key": "value2"})
        assert len(handler_calls) == 1  # Still 1, not 2
    
    def test_register_handler_for_unregistered_event(self):
        """Test that registering handler for unregistered event raises ValueError"""
        bus = EventBus()
        
        def handler(event: Event):
            pass
        
        with pytest.raises(ValueError, match="not registered"):
            bus.on("unregistered.event", handler)


class TestAsyncSubscription:
    """Test async iterator subscription pattern"""
    
    @pytest.mark.asyncio
    async def test_subscribe_to_events(self):
        """Test subscribing to events with async iterator"""
        bus = EventBus()
        bus.register_event_type("test.event")
        
        received_events = []
        
        async def subscriber():
            async for event in bus.subscribe(["test.event"]):
                received_events.append(event)
                if len(received_events) >= 3:
                    break
        
        # Start subscriber task
        subscriber_task = asyncio.create_task(subscriber())
        
        # Give subscriber time to start
        await asyncio.sleep(0.01)
        
        # Emit events
        await bus.emit("test.event", {"count": 1})
        await bus.emit("test.event", {"count": 2})
        await bus.emit("test.event", {"count": 3})
        
        # Wait for subscriber to finish
        await subscriber_task
        
        # Check received events
        assert len(received_events) == 3
        assert received_events[0].data["count"] == 1
        assert received_events[1].data["count"] == 2
        assert received_events[2].data["count"] == 3
    
    @pytest.mark.asyncio
    async def test_subscribe_to_multiple_event_types(self):
        """Test subscribing to multiple event types"""
        bus = EventBus()
        bus.register_event_type("event.one")
        bus.register_event_type("event.two")
        
        received_events = []
        
        async def subscriber():
            async for event in bus.subscribe(["event.one", "event.two"]):
                received_events.append(event)
                if len(received_events) >= 4:
                    break
        
        # Start subscriber task
        subscriber_task = asyncio.create_task(subscriber())
        
        # Give subscriber time to start
        await asyncio.sleep(0.01)
        
        # Emit events of different types
        await bus.emit("event.one", {"type": "one", "count": 1})
        await bus.emit("event.two", {"type": "two", "count": 2})
        await bus.emit("event.one", {"type": "one", "count": 3})
        await bus.emit("event.two", {"type": "two", "count": 4})
        
        # Wait for subscriber to finish
        await subscriber_task
        
        # Check received events
        assert len(received_events) == 4
        assert received_events[0].event_type == "event.one"
        assert received_events[1].event_type == "event.two"
    
    @pytest.mark.asyncio
    async def test_multiple_subscribers(self):
        """Test multiple subscribers receive same events"""
        bus = EventBus()
        bus.register_event_type("test.event")
        
        subscriber1_events = []
        subscriber2_events = []
        
        async def subscriber1():
            async for event in bus.subscribe(["test.event"]):
                subscriber1_events.append(event)
                if len(subscriber1_events) >= 2:
                    break
        
        async def subscriber2():
            async for event in bus.subscribe(["test.event"]):
                subscriber2_events.append(event)
                if len(subscriber2_events) >= 2:
                    break
        
        # Start both subscribers
        task1 = asyncio.create_task(subscriber1())
        task2 = asyncio.create_task(subscriber2())
        
        # Give subscribers time to start
        await asyncio.sleep(0.01)
        
        # Emit events
        await bus.emit("test.event", {"count": 1})
        await bus.emit("test.event", {"count": 2})
        
        # Wait for both subscribers
        await asyncio.gather(task1, task2)
        
        # Both should receive both events
        assert len(subscriber1_events) == 2
        assert len(subscriber2_events) == 2
    
    @pytest.mark.asyncio
    async def test_subscribe_to_unregistered_event(self):
        """Test that subscribing to unregistered event raises ValueError"""
        bus = EventBus()
        
        with pytest.raises(ValueError, match="not registered"):
            async for event in bus.subscribe(["unregistered.event"]):
                pass
    
    @pytest.mark.asyncio
    async def test_subscriber_cleanup(self):
        """Test that cancelled subscribers are cleaned up"""
        bus = EventBus()
        bus.register_event_type("test.event")
        
        async def subscriber():
            async for event in bus.subscribe(["test.event"]):
                pass
        
        # Start subscriber
        task = asyncio.create_task(subscriber())
        await asyncio.sleep(0.01)
        
        # Check subscriber is registered
        stats = bus.get_statistics()
        assert stats["subscriber_counts"]["test.event"] == 1
        
        # Cancel subscriber
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        
        # Give cleanup time to run
        await asyncio.sleep(0.01)
        
        # Check subscriber is cleaned up
        stats = bus.get_statistics()
        assert stats["subscriber_counts"].get("test.event", 0) == 0


class TestConcurrency:
    """Test thread-safety and concurrent access"""
    
    @pytest.mark.asyncio
    async def test_concurrent_emissions(self):
        """Test concurrent event emissions"""
        bus = EventBus()
        bus.register_event_type("test.event")
        
        received_events = []
        
        def handler(event: Event):
            received_events.append(event)
        
        bus.on("test.event", handler)
        
        # Emit events concurrently
        tasks = [
            bus.emit("test.event", {"count": i})
            for i in range(10)
        ]
        await asyncio.gather(*tasks)
        
        # All events should be received
        assert len(received_events) == 10
    
    @pytest.mark.asyncio
    async def test_concurrent_subscriptions(self):
        """Test concurrent subscriptions"""
        bus = EventBus()
        bus.register_event_type("test.event")
        
        subscriber_counts = []
        
        async def subscriber(subscriber_id: int):
            count = 0
            async for event in bus.subscribe(["test.event"]):
                count += 1
                if count >= 5:
                    break
            subscriber_counts.append((subscriber_id, count))
        
        # Start multiple subscribers concurrently
        tasks = [
            asyncio.create_task(subscriber(i))
            for i in range(5)
        ]
        
        # Give subscribers time to start
        await asyncio.sleep(0.01)
        
        # Emit events
        for i in range(5):
            await bus.emit("test.event", {"count": i})
        
        # Wait for all subscribers
        await asyncio.gather(*tasks)
        
        # All subscribers should receive all events
        assert len(subscriber_counts) == 5
        for subscriber_id, count in subscriber_counts:
            assert count == 5


class TestStatistics:
    """Test statistics and monitoring"""
    
    @pytest.mark.asyncio
    async def test_get_statistics(self):
        """Test getting event bus statistics"""
        bus = EventBus()
        bus.register_event_types(["event.one", "event.two"])
        
        def handler(event: Event):
            pass
        
        bus.on("event.one", handler)
        bus.on("event.one", handler)
        bus.on("event.two", handler)
        
        await bus.emit("event.one", {})
        await bus.emit("event.one", {})
        await bus.emit("event.two", {})
        
        stats = bus.get_statistics()
        
        assert "event.one" in stats["registered_types"]
        assert "event.two" in stats["registered_types"]
        assert stats["event_counts"]["event.one"] == 2
        assert stats["event_counts"]["event.two"] == 1
        assert stats["handler_counts"]["event.one"] == 2
        assert stats["handler_counts"]["event.two"] == 1
    
    def test_clear_handlers(self):
        """Test clearing handlers"""
        bus = EventBus()
        bus.register_event_types(["event.one", "event.two"])
        
        def handler(event: Event):
            pass
        
        bus.on("event.one", handler)
        bus.on("event.two", handler)
        
        # Clear specific event type
        bus.clear_handlers("event.one")
        stats = bus.get_statistics()
        assert stats["handler_counts"].get("event.one", 0) == 0
        assert stats["handler_counts"]["event.two"] == 1
        
        # Clear all handlers
        bus.clear_handlers()
        stats = bus.get_statistics()
        assert stats["handler_counts"] == {}
    
    @pytest.mark.asyncio
    async def test_reset(self):
        """Test resetting the event bus"""
        bus = EventBus()
        bus.register_event_type("test.event")
        
        def handler(event: Event):
            pass
        
        bus.on("test.event", handler)
        await bus.emit("test.event", {})
        
        # Reset
        bus.reset()
        
        stats = bus.get_statistics()
        assert stats["handler_counts"] == {}
        assert stats["event_counts"] == {}
        # Registered types should remain
        assert "test.event" in stats["registered_types"]


class TestAegisEventBus:
    """Test Aegis-specific event bus creation"""
    
    def test_create_aegis_event_bus(self):
        """Test creating Aegis event bus with predefined types"""
        bus = create_aegis_event_bus()
        
        # Check all Aegis event types are registered
        for event_type in AEGIS_EVENT_TYPES:
            assert bus.is_registered(event_type)
        
        stats = bus.get_statistics()
        assert len(stats["registered_types"]) == len(AEGIS_EVENT_TYPES)
    
    @pytest.mark.asyncio
    async def test_aegis_event_types_usable(self):
        """Test that Aegis event types can be used"""
        bus = create_aegis_event_bus()
        
        received_events = []
        
        def handler(event: Event):
            received_events.append(event)
        
        # Register handlers for some Aegis events
        bus.on("stt.completed", handler)
        bus.on("emotion.analyzed", handler)
        bus.on("llm.response_generated", handler)
        
        # Emit events
        await bus.emit("stt.completed", {"text": "Hello", "language": "en"})
        await bus.emit("emotion.analyzed", {"emotion": "happy", "confidence": 0.8})
        await bus.emit("llm.response_generated", {"response": "Hi there!"})
        
        # Check all events received
        assert len(received_events) == 3


class TestIntegrationScenarios:
    """Test realistic integration scenarios"""
    
    @pytest.mark.asyncio
    async def test_pipeline_coordination(self):
        """Test coordinating async pipeline with event bus"""
        bus = create_aegis_event_bus()
        
        pipeline_events = []
        
        # Simulate pipeline stages
        async def stt_stage():
            await asyncio.sleep(0.01)
            await bus.emit("stt.completed", {"text": "Hello", "language": "en"})
        
        async def emotion_stage():
            await asyncio.sleep(0.01)
            await bus.emit("emotion.analyzed", {"emotion": "happy", "confidence": 0.8})
        
        async def llm_stage():
            # Wait for STT to complete
            async for event in bus.subscribe(["stt.completed"]):
                text = event.data["text"]
                await asyncio.sleep(0.01)
                await bus.emit("llm.response_generated", {"response": f"Response to: {text}"})
                break
        
        # Track all events
        def track_event(event: Event):
            pipeline_events.append(event.event_type)
        
        bus.on("stt.completed", track_event)
        bus.on("emotion.analyzed", track_event)
        bus.on("llm.response_generated", track_event)
        
        # Run pipeline stages concurrently
        await asyncio.gather(
            stt_stage(),
            emotion_stage(),
            llm_stage()
        )
        
        # Check all stages completed
        assert "stt.completed" in pipeline_events
        assert "emotion.analyzed" in pipeline_events
        assert "llm.response_generated" in pipeline_events
    
    @pytest.mark.asyncio
    async def test_mixed_handlers_and_subscribers(self):
        """Test mixing callback handlers and async subscribers"""
        bus = EventBus()
        bus.register_event_type("test.event")
        
        handler_events = []
        subscriber_events = []
        
        def handler(event: Event):
            handler_events.append(event)
        
        async def subscriber():
            async for event in bus.subscribe(["test.event"]):
                subscriber_events.append(event)
                if len(subscriber_events) >= 3:
                    break
        
        # Register handler and start subscriber
        bus.on("test.event", handler)
        subscriber_task = asyncio.create_task(subscriber())
        
        await asyncio.sleep(0.01)
        
        # Emit events
        await bus.emit("test.event", {"count": 1})
        await bus.emit("test.event", {"count": 2})
        await bus.emit("test.event", {"count": 3})
        
        await subscriber_task
        
        # Both handler and subscriber should receive all events
        assert len(handler_events) == 3
        assert len(subscriber_events) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
