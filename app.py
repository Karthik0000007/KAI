"""
╔═══════════════════════════════════════════════════════════════════════╗
║   AEGIS — Offline Voice-First Personal Health AI System              ║
║   Privacy-preserving · Emotion-aware · Proactive                     ║
╚═══════════════════════════════════════════════════════════════════════╝

Main orchestrator: ties together voice input, emotion analysis,
health-aware LLM, proactive engine, encrypted storage, and TTS output.
"""

import asyncio
import sys
import signal
import logging
from datetime import datetime

from core.config import RECORD_DURATION_DEFAULT
from core.stt import listen_and_analyze_async
from core.llm import get_response_async, extract_health_signals_async
from core.tts import speak_text_async
from core.emotion import emotion_to_tone_mode
from core.health_db import HealthDatabase
from core.proactive import ProactiveEngine
from core.models import HealthCheckIn, Session, ConversationTurn
from core.event_bus import create_aegis_event_bus
from core.startup_validator import StartupValidator, ValidationError
from core.logger import setup_logging, get_logger, log_turn_metrics
from pathlib import Path

# ─── Logging ─────────────────────────────────────────────────────────────────
# Initialize structured logging
log_file = Path("data/logs/aegis.log")
structured_logger = setup_logging(
    log_file=log_file,
    log_level="INFO",
    max_size_mb=10,
    backup_count=5,
    log_format="json",
    console_enabled=False
)

# Keep basic logging for console output
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("aegis")


# ─── Banner ──────────────────────────────────────────────────────────────────

BANNER = r"""
    ___    _____ __________
   /   |  / ___// ____/  _/ ___
  / /| | / __/ / / __ / /  __ \
 / ___ |/ /___/ /_/ // /  ___/ /
/_/  |_/_____/\____/___/ /____/

  Offline Voice-First Health AI
  Privacy-first · Emotion-aware
"""


def print_status(label: str, value: str, icon: str = ""):
    """Pretty-print a status line."""
    prefix = f"  {icon} " if icon else "  "
    print(f"{prefix}[{label}] {value}")


# ─── Alert Callback ─────────────────────────────────────────────────────────

def on_proactive_alert(alert):
    """Called by the proactive engine when a new alert is generated."""
    severity_icons = {"info": "i", "warning": "!", "urgent": "!!"}
    icon = severity_icons.get(alert.severity, "?")
    print(f"\n  [{icon}] AEGIS ALERT ({alert.alert_type}):")
    print(f"      {alert.message}\n")


# ─── Main Loop ───────────────────────────────────────────────────────────────

async def main():
    print(BANNER)
    print("  Initializing Aegis...\n")

    # Validate all dependencies before starting
    print("  Validating system dependencies...\n")
    validator = StartupValidator()
    try:
        validator.validate_all()
    except ValidationError as e:
        logger.error(f"Startup validation failed: {e}")
        sys.exit(1)

    # Initialize subsystems
    db = HealthDatabase()
    session = Session()
    event_bus = create_aegis_event_bus()
    proactive = ProactiveEngine(db, on_alert=on_proactive_alert)
    proactive.start()

    # Graceful shutdown
    def shutdown(sig=None, frame=None):
        print("\n  Shutting down Aegis...")
        proactive.stop()
        db.close()
        print("  Goodbye. Take care of yourself.\n")
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)

    # Emit system startup event
    await event_bus.emit("system.startup", {"timestamp": datetime.now().isoformat()})

    # Run initial proactive check for pending alerts
    pending_alerts = db.get_unacknowledged_alerts()
    if pending_alerts:
        print(f"  You have {len(pending_alerts)} pending health alert(s):\n")
        for alert in pending_alerts:
            print(f"    - {alert.get('message', '')}")
        print()

    # ─── Conversation Loop ───────────────────────────────────────────────
    print("  Ready. Speak to Aegis (Ctrl+C to exit).\n")

    turn_count = 0
    while True:
        try:
            turn_count += 1
            turn_start_time = datetime.now()
            print(f"  --- Turn {turn_count} ---")

            # Initialize stage timing dictionary
            stage_durations = {}

            # Emit turn started event
            await event_bus.emit("pipeline.turn_started", {
                "turn_number": turn_count,
                "timestamp": turn_start_time.isoformat()
            })

            # 1) Listen + Transcribe + Emotion (parallel STT and emotion analysis)
            print_status("LISTEN", f"Recording {RECORD_DURATION_DEFAULT}s...", "mic")
            
            # Emit audio recording event
            await event_bus.emit("audio.recorded", {
                "duration": RECORD_DURATION_DEFAULT,
                "timestamp": datetime.now().isoformat()
            })
            
            # Requirement 3.1: Log start time of STT stage
            stt_start_time = datetime.now()
            text, lang, emotion = await listen_and_analyze_async(duration=RECORD_DURATION_DEFAULT)
            stt_duration = (datetime.now() - stt_start_time).total_seconds()
            stage_durations["stt"] = stt_duration

            if not text.strip():
                print_status("STT", "No speech detected. Try again.", "?")
                await event_bus.emit("stt.failed", {
                    "reason": "empty_transcription",
                    "timestamp": datetime.now().isoformat()
                })
                continue

            # Emit STT completed event
            await event_bus.emit("stt.completed", {
                "text": text,
                "language": lang,
                "timestamp": datetime.now().isoformat()
            })

            print_status("STT", f'"{text}"', "speech")
            print_status("LANG", lang.upper(), "globe")
            print_status("EMOTION", f"{emotion.label} ({emotion.confidence:.0%})", "heart")

            # Note: Emotion analysis duration is included in STT duration since they run in parallel
            # Emit emotion analyzed event
            await event_bus.emit("emotion.analyzed", {
                "label": emotion.label,
                "confidence": emotion.confidence,
                "timestamp": datetime.now().isoformat()
            })

            # 2) Determine response tone
            tone_mode = emotion_to_tone_mode(emotion.label)
            print_status("TONE", tone_mode, "note")

            # 3) Extract health signals from text (async)
            health_signals = await extract_health_signals_async(text)
            if health_signals:
                print_status("HEALTH", str(health_signals), "vitals")
                await event_bus.emit("health.signals_extracted", {
                    "signals": health_signals,
                    "timestamp": datetime.now().isoformat()
                })

            # 4) Save check-in if health data was found (run in thread to not block)
            async def save_checkin_async():
                if health_signals:
                    checkin = HealthCheckIn(
                        user_text=text,
                        mood_score=health_signals.get("mood_score"),
                        sleep_hours=health_signals.get("sleep_hours"),
                        energy_level=health_signals.get("energy_level"),
                        medication_taken=health_signals.get("medication_taken"),
                        pain_notes=text if health_signals.get("pain_mentioned") else None,
                        detected_emotion=emotion.label,
                        emotion_confidence=emotion.confidence,
                    )
                    await asyncio.to_thread(db.save_checkin, checkin)
                    print_status("DB", "Health check-in saved", "disk")
                    await event_bus.emit("db.checkin_saved", {
                        "timestamp": datetime.now().isoformat()
                    })

            # 5) Gather context for LLM (run in thread)
            async def gather_context_async():
                health_stats = await asyncio.to_thread(db.get_checkin_stats, days=7)
                active_alerts = await asyncio.to_thread(db.get_unacknowledged_alerts)
                return health_stats, active_alerts

            # Run save_checkin and gather_context in parallel
            save_task = asyncio.create_task(save_checkin_async())
            context_task = asyncio.create_task(gather_context_async())
            
            health_stats, active_alerts = await context_task
            await save_task  # Ensure save completes

            conv_history = [t.to_dict() for t in session.turns[-8:]]

            # 6) Record user turn
            session.add_turn("user", text, emotion=emotion.label)
            await asyncio.to_thread(
                db.save_conversation_turn,
                session.id,
                ConversationTurn(role="user", content=text, emotion=emotion.label),
            )

            # 7) Get LLM response (async)
            # Requirement 3.1: Log start time of LLM stage
            print_status("LLM", "Thinking...", "brain")
            await event_bus.emit("llm.started", {
                "timestamp": datetime.now().isoformat()
            })
            
            llm_start_time = datetime.now()
            reply = await get_response_async(
                user_input=text,
                emotion_label=emotion.label,
                tone_mode=tone_mode,
                health_stats=health_stats,
                active_alerts=active_alerts,
                conversation_history=conv_history,
                language=lang,
            )
            llm_duration = (datetime.now() - llm_start_time).total_seconds()
            stage_durations["llm"] = llm_duration
            
            await event_bus.emit("llm.response_generated", {
                "response_length": len(reply),
                "timestamp": datetime.now().isoformat()
            })
            
            print_status("AEGIS", reply, "robot")

            # 8) Record assistant turn
            session.add_turn("assistant", reply, tone_mode=tone_mode)
            
            # 9) Speak response and save conversation turn in parallel
            async def speak_and_save():
                # Start TTS
                # Requirement 3.1: Log start time of TTS stage
                print_status("TTS", f"Speaking [{lang.upper()}]...", "speaker")
                await event_bus.emit("tts.started", {
                    "language": lang,
                    "timestamp": datetime.now().isoformat()
                })
                
                tts_start_time = datetime.now()
                speak_task = speak_text_async(reply, language=lang, tone_mode=tone_mode)
                
                # Save conversation turn
                save_task = asyncio.to_thread(
                    db.save_conversation_turn,
                    session.id,
                    ConversationTurn(role="assistant", content=reply, tone_mode=tone_mode),
                )
                
                # Run both in parallel
                await asyncio.gather(speak_task, save_task)
                
                tts_duration = (datetime.now() - tts_start_time).total_seconds()
                stage_durations["tts"] = tts_duration
                
                await event_bus.emit("tts.completed", {
                    "timestamp": datetime.now().isoformat()
                })
                
                return tts_duration

            tts_duration = await speak_and_save()

            # 10) Acknowledge any alerts that were addressed
            for alert in active_alerts:
                await asyncio.to_thread(db.acknowledge_alert, alert["id"])

            # Emit turn completed event
            turn_duration = (datetime.now() - turn_start_time).total_seconds()
            await event_bus.emit("pipeline.turn_completed", {
                "turn_number": turn_count,
                "duration_seconds": turn_duration,
                "timestamp": datetime.now().isoformat()
            })
            
            # Requirement 3.2: Log total turn duration and per-stage durations
            # Requirement 5.5: Log structured JSON record with turn_id, duration, emotion, health_signals, response_length
            log_turn_metrics(
                logger=structured_logger,
                turn_id=f"turn_{turn_count}",
                duration=turn_duration,
                emotion=emotion.label,
                health_signals=health_signals,
                response_length=len(reply),
                stage_durations=stage_durations
            )
            
            logger.info(f"Turn {turn_count} completed in {turn_duration:.2f}s")
            print()

        except RuntimeError as e:
            logger.error(f"Runtime error: {e}")
            print(f"\n  Error: {e}")
            print("  Retrying...\n")
            await event_bus.emit("pipeline.turn_failed", {
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            })
            continue

        except KeyboardInterrupt:
            await event_bus.emit("system.shutdown", {
                "timestamp": datetime.now().isoformat()
            })
            shutdown()

        except Exception as e:
            logger.error(f"Unexpected error: {e}", exc_info=True)
            print(f"\n  Unexpected error: {e}")
            print("  Recovering...\n")
            await event_bus.emit("system.error", {
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            })
            continue


if __name__ == "__main__":
    asyncio.run(main())
