"""
╔═══════════════════════════════════════════════════════════════════════╗
║   AEGIS — Offline Voice-First Personal Health AI System              ║
║   Privacy-preserving · Emotion-aware · Proactive                     ║
╚═══════════════════════════════════════════════════════════════════════╝

Main orchestrator: ties together voice input, emotion analysis,
health-aware LLM, proactive engine, encrypted storage, and TTS output.
"""

import sys
import signal
import logging
from datetime import datetime

from core.config import RECORD_DURATION_DEFAULT
from core.stt import listen_and_analyze
from core.llm import get_response, extract_health_signals
from core.tts import speak_text
from core.emotion import emotion_to_tone_mode
from core.health_db import HealthDatabase
from core.proactive import ProactiveEngine
from core.models import HealthCheckIn, Session, ConversationTurn

# ─── Logging ─────────────────────────────────────────────────────────────────
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

def main():
    print(BANNER)
    print("  Initializing Aegis...\n")

    # Initialize subsystems
    db = HealthDatabase()
    session = Session()
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
            print(f"  --- Turn {turn_count} ---")

            # 1) Listen + Transcribe + Emotion
            print_status("LISTEN", f"Recording {RECORD_DURATION_DEFAULT}s...", "mic")
            text, lang, emotion = listen_and_analyze(duration=RECORD_DURATION_DEFAULT)

            if not text.strip():
                print_status("STT", "No speech detected. Try again.", "?")
                continue

            print_status("STT", f'"{text}"', "speech")
            print_status("LANG", lang.upper(), "globe")
            print_status("EMOTION", f"{emotion.label} ({emotion.confidence:.0%})", "heart")

            # 2) Determine response tone
            tone_mode = emotion_to_tone_mode(emotion.label)
            print_status("TONE", tone_mode, "note")

            # 3) Extract health signals from text
            health_signals = extract_health_signals(text)
            if health_signals:
                print_status("HEALTH", str(health_signals), "vitals")

            # 4) Save check-in if health data was found
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
                db.save_checkin(checkin)
                print_status("DB", "Health check-in saved", "disk")

            # 5) Gather context for LLM
            health_stats = db.get_checkin_stats(days=7)
            active_alerts = db.get_unacknowledged_alerts()
            conv_history = [t.to_dict() for t in session.turns[-8:]]

            # 6) Record user turn
            session.add_turn("user", text, emotion=emotion.label)
            db.save_conversation_turn(
                session.id,
                ConversationTurn(role="user", content=text, emotion=emotion.label),
            )

            # 7) Get LLM response
            print_status("LLM", "Thinking...", "brain")
            reply = get_response(
                user_input=text,
                emotion_label=emotion.label,
                tone_mode=tone_mode,
                health_stats=health_stats,
                active_alerts=active_alerts,
                conversation_history=conv_history,
                language=lang,
            )
            print_status("AEGIS", reply, "robot")

            # 8) Record assistant turn
            session.add_turn("assistant", reply, tone_mode=tone_mode)
            db.save_conversation_turn(
                session.id,
                ConversationTurn(role="assistant", content=reply, tone_mode=tone_mode),
            )

            # 9) Speak response
            print_status("TTS", f"Speaking [{lang.upper()}]...", "speaker")
            speak_text(reply, language=lang, tone_mode=tone_mode)

            # 10) Acknowledge any alerts that were addressed
            for alert in active_alerts:
                db.acknowledge_alert(alert["id"])

            print()

        except RuntimeError as e:
            logger.error(f"Runtime error: {e}")
            print(f"\n  Error: {e}")
            print("  Retrying...\n")
            continue

        except KeyboardInterrupt:
            shutdown()

        except Exception as e:
            logger.error(f"Unexpected error: {e}", exc_info=True)
            print(f"\n  Unexpected error: {e}")
            print("  Recovering...\n")
            continue


if __name__ == "__main__":
    main()
