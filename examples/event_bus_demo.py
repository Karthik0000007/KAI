"""
Event Bus Demo - Demonstrates the Event Bus usage patterns

This example shows how to use the Event Bus for coordinating
async pipeline operations in the Aegis system.
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.event_bus import create_aegis_event_bus, Event

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


async def simulate_stt_stage(bus):
    """Simulate Speech-to-Text processing"""
    logger.info("STT: Starting transcription...")
    await asyncio.sleep(0.5)  # Simulate processing time
    
    await bus.emit("stt.completed", {
        "text": "I slept well last night",
        "language": "en",
        "confidence": 0.95
    })
    logger.info("STT: Transcription completed")


async def simulate_emotion_stage(bus):
    """Simulate emotion analysis"""
    logger.info("Emotion: Starting analysis...")
    await asyncio.sleep(0.3)  # Simulate processing time
    
    await bus.emit("emotion.analyzed", {
        "emotion": "happy",
        "confidence": 0.82,
        "features": {"pitch": 0.6, "energy": 0.7}
    })
    logger.info("Emotion: Analysis completed")


async def simulate_health_extraction(bus):
    """Simulate health signal extraction - waits for STT"""
    logger.info("Health: Waiting for transcription...")
    
    async for event in bus.subscribe(["stt.completed"]):
        text = event.data["text"]
        logger.info(f"Health: Extracting signals from: '{text}'")
        
        await asyncio.sleep(0.2)  # Simulate extraction
        
        await bus.emit("health.signals_extracted", {
            "sleep_hours": 8,
            "sleep_quality": "good",
            "mood": "positive"
        })
        logger.info("Health: Signals extracted")
        break


async def simulate_llm_stage(bus):
    """Simulate LLM response generation - waits for multiple inputs"""
    logger.info("LLM: Waiting for context...")
    
    context = {}
    required_events = {"stt.completed", "emotion.analyzed", "health.signals_extracted"}
    received_events = set()
    
    async for event in bus.subscribe(list(required_events)):
        received_events.add(event.event_type)
        context[event.event_type] = event.data
        
        logger.info(f"LLM: Received {event.event_type}")
        
        if received_events == required_events:
            logger.info("LLM: All context received, generating response...")
            await asyncio.sleep(0.4)  # Simulate LLM processing
            
            await bus.emit("llm.response_generated", {
                "response": "That's wonderful! Getting good sleep is so important for your health.",
                "context_used": list(context.keys())
            })
            logger.info("LLM: Response generated")
            break


async def simulate_tts_stage(bus):
    """Simulate Text-to-Speech - waits for LLM"""
    logger.info("TTS: Waiting for response...")
    
    async for event in bus.subscribe(["llm.response_generated"]):
        response = event.data["response"]
        logger.info(f"TTS: Synthesizing: '{response[:50]}...'")
        
        await asyncio.sleep(0.3)  # Simulate TTS
        
        await bus.emit("tts.completed", {
            "audio_duration": 3.5,
            "success": True
        })
        logger.info("TTS: Speech synthesis completed")
        break


def setup_monitoring(bus):
    """Setup monitoring handlers for all pipeline events"""
    
    def log_pipeline_event(event: Event):
        logger.info(f"Monitor: {event.event_type} at {event.timestamp.strftime('%H:%M:%S.%f')[:-3]}")
    
    # Register handlers for key events
    pipeline_events = [
        "stt.completed",
        "emotion.analyzed",
        "health.signals_extracted",
        "llm.response_generated",
        "tts.completed"
    ]
    
    for event_type in pipeline_events:
        bus.on(event_type, log_pipeline_event)


async def run_pipeline_demo():
    """Run a complete pipeline demonstration"""
    logger.info("=" * 60)
    logger.info("Event Bus Demo - Async Pipeline Coordination")
    logger.info("=" * 60)
    
    # Create event bus with Aegis event types
    bus = create_aegis_event_bus()
    
    # Setup monitoring
    setup_monitoring(bus)
    
    # Emit pipeline start event
    await bus.emit("pipeline.turn_started", {"turn_id": "demo_001"})
    
    # Run pipeline stages concurrently
    # STT and Emotion can run in parallel
    # Health extraction waits for STT
    # LLM waits for all context
    # TTS waits for LLM
    await asyncio.gather(
        simulate_stt_stage(bus),
        simulate_emotion_stage(bus),
        simulate_health_extraction(bus),
        simulate_llm_stage(bus),
        simulate_tts_stage(bus)
    )
    
    # Emit pipeline complete event
    await bus.emit("pipeline.turn_completed", {
        "turn_id": "demo_001",
        "duration": 1.5
    })
    
    # Show statistics
    logger.info("=" * 60)
    logger.info("Pipeline Statistics:")
    stats = bus.get_statistics()
    logger.info(f"Total events emitted: {sum(stats['event_counts'].values())}")
    logger.info(f"Event breakdown: {stats['event_counts']}")
    logger.info("=" * 60)


async def run_callback_demo():
    """Demonstrate callback-based event handling"""
    logger.info("\n" + "=" * 60)
    logger.info("Callback Pattern Demo")
    logger.info("=" * 60)
    
    bus = create_aegis_event_bus()
    
    # Define handlers
    def on_stt_complete(event: Event):
        logger.info(f"Handler: Transcribed '{event.data['text']}'")
    
    async def on_emotion_analyzed(event: Event):
        logger.info(f"Handler: Detected emotion '{event.data['emotion']}' "
                   f"with confidence {event.data['confidence']}")
    
    # Register handlers
    bus.on("stt.completed", on_stt_complete)
    bus.on("emotion.analyzed", on_emotion_analyzed)
    
    # Emit events
    await bus.emit("stt.completed", {"text": "Hello world", "language": "en"})
    await bus.emit("emotion.analyzed", {"emotion": "neutral", "confidence": 0.75})
    
    logger.info("=" * 60)


async def run_subscription_demo():
    """Demonstrate async iterator subscription"""
    logger.info("\n" + "=" * 60)
    logger.info("Subscription Pattern Demo")
    logger.info("=" * 60)
    
    bus = create_aegis_event_bus()
    
    # Create subscriber task
    async def subscriber():
        logger.info("Subscriber: Waiting for events...")
        event_count = 0
        
        async for event in bus.subscribe(["stt.completed", "emotion.analyzed"]):
            logger.info(f"Subscriber: Received {event.event_type}")
            event_count += 1
            
            if event_count >= 4:
                logger.info("Subscriber: Received all events, stopping")
                break
    
    # Start subscriber
    subscriber_task = asyncio.create_task(subscriber())
    
    # Give subscriber time to start
    await asyncio.sleep(0.1)
    
    # Emit events
    await bus.emit("stt.completed", {"text": "Event 1", "language": "en"})
    await bus.emit("emotion.analyzed", {"emotion": "happy", "confidence": 0.8})
    await bus.emit("stt.completed", {"text": "Event 2", "language": "en"})
    await bus.emit("emotion.analyzed", {"emotion": "sad", "confidence": 0.7})
    
    # Wait for subscriber to finish
    await subscriber_task
    
    logger.info("=" * 60)


async def main():
    """Run all demos"""
    await run_pipeline_demo()
    await run_callback_demo()
    await run_subscription_demo()
    
    logger.info("\n✓ All demos completed successfully!")


if __name__ == "__main__":
    asyncio.run(main())
