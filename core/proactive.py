"""
Aegis Proactive Health Engine
Runs periodic background analysis on local health data to generate
proactive interventions — mimicking "care" rather than passive response.
"""

import logging
import threading
import time
from datetime import datetime, timedelta
from typing import List, Optional, Callable

from core.config import PROACTIVE_CONFIG
from core.models import ProactiveAlert, HealthCheckIn
from core.health_db import HealthDatabase

logger = logging.getLogger("aegis.proactive")


class ProactiveEngine:
    """
    Background engine that periodically analyzes health patterns and
    generates alerts/interventions when concerning trends are detected.
    
    Patterns detected:
        - Consecutive days of low mood
        - Sleep deficit
        - Missed medications
        - Elevated heart rate + stressed emotion
        - Declining energy trend
        - Emotional distress pattern
    """

    def __init__(self, db: HealthDatabase,
                 on_alert: Optional[Callable[[ProactiveAlert], None]] = None):
        self.db = db
        self.on_alert = on_alert  # callback when a new alert is generated
        self.cfg = PROACTIVE_CONFIG
        self._running = False
        self._thread: Optional[threading.Thread] = None

    # ─── Background Loop ─────────────────────────────────────────────────

    def start(self):
        """Start the proactive engine in a background thread."""
        if self._running:
            logger.warning("Proactive engine already running")
            return
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        logger.info("Proactive engine started")

    def stop(self):
        """Stop the background engine."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("Proactive engine stopped")

    def _run_loop(self):
        while self._running:
            try:
                self.run_analysis()
            except Exception as e:
                logger.error(f"Proactive analysis error: {e}")
            time.sleep(self.cfg.check_interval_minutes * 60)

    # ─── Main Analysis ───────────────────────────────────────────────────

    def run_analysis(self) -> List[ProactiveAlert]:
        """
        Run all proactive checks and return generated alerts.
        """
        logger.info("Running proactive health analysis...")
        alerts: List[ProactiveAlert] = []

        alerts.extend(self._check_mood_pattern())
        alerts.extend(self._check_sleep_deficit())
        alerts.extend(self._check_medication_compliance())
        alerts.extend(self._check_vital_signs())
        alerts.extend(self._check_emotion_pattern())
        alerts.extend(self._check_energy_trend())

        # Save and notify
        for alert in alerts:
            self.db.save_alert(alert)
            if self.on_alert:
                try:
                    self.on_alert(alert)
                except Exception as e:
                    logger.error(f"Alert callback error: {e}")

        if alerts:
            logger.info(f"Generated {len(alerts)} proactive alert(s)")
        else:
            logger.info("No proactive alerts at this time")

        return alerts

    # ─── Pattern Checks ──────────────────────────────────────────────────

    def _check_mood_pattern(self) -> List[ProactiveAlert]:
        """Detect consecutive days of low mood."""
        stats = self.db.get_checkin_stats(days=7)
        alerts = []

        low_mood_days = stats.get("low_mood_days", 0)
        if low_mood_days >= self.cfg.low_mood_days_threshold:
            avg_mood = stats.get("avg_mood")
            alerts.append(ProactiveAlert(
                alert_type="low_mood_pattern",
                severity="warning",
                message=(
                    f"I've noticed your mood has been low for {low_mood_days} days "
                    f"(average: {avg_mood}/10). Would you like to talk about what's "
                    f"on your mind? Sometimes sharing helps."
                ),
                context={"low_mood_days": low_mood_days, "avg_mood": avg_mood},
            ))

        return alerts

    def _check_sleep_deficit(self) -> List[ProactiveAlert]:
        """Detect sustained low sleep."""
        stats = self.db.get_checkin_stats(days=7)
        alerts = []

        avg_sleep = stats.get("avg_sleep")
        low_sleep_days = stats.get("low_sleep_days", 0)

        if avg_sleep is not None and avg_sleep < self.cfg.low_sleep_hours:
            alerts.append(ProactiveAlert(
                alert_type="sleep_deficit",
                severity="warning",
                message=(
                    f"Your average sleep has been {avg_sleep} hours over the past week. "
                    f"That's below the recommended amount. Try winding down earlier tonight — "
                    f"perhaps some light reading or breathing exercises before bed?"
                ),
                context={"avg_sleep": avg_sleep, "low_sleep_days": low_sleep_days},
            ))

        return alerts

    def _check_medication_compliance(self) -> List[ProactiveAlert]:
        """Check for missed medications based on today's check-ins."""
        alerts = []
        meds = self.db.get_active_medications()
        if not meds:
            return alerts

        # Check today's check-ins for medication_taken
        today_checkins = self.db.get_recent_checkins(days=1)
        any_taken = any(c.get("medication_taken") for c in today_checkins)

        now = datetime.now()
        for med in meds:
            schedule = med.get("schedule_time", "")
            if not schedule:
                continue
            try:
                hour, minute = map(int, schedule.split(":"))
                scheduled_time = now.replace(hour=hour, minute=minute, second=0)
                # If past scheduled time + delay and no confirmation
                delay = timedelta(minutes=self.cfg.missed_medication_reminder_delay)
                if now > scheduled_time + delay and not any_taken:
                    alerts.append(ProactiveAlert(
                        alert_type="medication_missed",
                        severity="info",
                        message=(
                            f"Gentle reminder: it looks like you may have missed your "
                            f"{med['name']} ({med.get('dosage', '')}) scheduled for "
                            f"{schedule}. Have you taken it yet?"
                        ),
                        context={"medication": med["name"], "schedule": schedule},
                    ))
            except (ValueError, AttributeError):
                continue

        return alerts

    def _check_vital_signs(self) -> List[ProactiveAlert]:
        """Check for elevated heart rate combined with stressed emotion."""
        alerts = []
        vitals = self.db.get_recent_vitals(days=1)
        if not vitals:
            return alerts

        recent_hr = [v["heart_rate"] for v in vitals if v.get("heart_rate")]
        if not recent_hr:
            return alerts

        max_hr = max(recent_hr)
        if max_hr >= self.cfg.elevated_hr_threshold:
            # Check if emotion was also stressed
            stats = self.db.get_checkin_stats(days=1)
            recent_emotions = stats.get("recent_emotions", [])
            is_stressed = any(e in ("stressed", "anxious") for e in recent_emotions)

            if is_stressed:
                alerts.append(ProactiveAlert(
                    alert_type="elevated_hr_stress",
                    severity="warning",
                    message=(
                        f"Your heart rate reached {max_hr} bpm and you seem stressed. "
                        f"Let's try a quick breathing exercise: breathe in for 4 counts, "
                        f"hold for 4, and exhale for 6. Repeat 3 times."
                    ),
                    context={"max_hr": max_hr, "emotions": recent_emotions},
                ))
            else:
                alerts.append(ProactiveAlert(
                    alert_type="elevated_hr",
                    severity="info",
                    message=(
                        f"Your heart rate reached {max_hr} bpm recently. "
                        f"If you haven't been exercising, consider resting for a bit."
                    ),
                    context={"max_hr": max_hr},
                ))

        return alerts

    def _check_emotion_pattern(self) -> List[ProactiveAlert]:
        """Detect repeated negative emotions over several check-ins."""
        alerts = []
        stats = self.db.get_checkin_stats(days=5)
        recent_emotions = stats.get("recent_emotions", [])

        if not recent_emotions:
            return alerts

        negative_count = sum(
            1 for e in recent_emotions if e in ("stressed", "anxious", "fatigued")
        )

        if negative_count >= 3:
            alerts.append(ProactiveAlert(
                alert_type="emotional_distress_pattern",
                severity="warning",
                message=(
                    "I've been sensing some tension and fatigue in our recent "
                    "conversations. You don't have to carry everything alone — "
                    "would you like to talk about what's been weighing on you?"
                ),
                context={"negative_emotions": negative_count, "total": len(recent_emotions)},
            ))

        return alerts

    def _check_energy_trend(self) -> List[ProactiveAlert]:
        """Detect declining energy levels over several days."""
        alerts = []
        checkins = self.db.get_recent_checkins(days=5)
        energies = [
            c["energy_level"] for c in checkins
            if c.get("energy_level") is not None
        ]

        if len(energies) < 3:
            return alerts

        # Check if consistently declining
        declining = all(energies[i] >= energies[i + 1] for i in range(len(energies) - 1))
        avg_energy = sum(energies) / len(energies)

        if declining and avg_energy < 4.0:
            alerts.append(ProactiveAlert(
                alert_type="energy_decline",
                severity="info",
                message=(
                    "Your energy levels have been trending downward. This might be a "
                    "sign to slow down a bit. Make sure you're eating well, staying "
                    "hydrated, and getting some gentle movement during the day."
                ),
                context={"energies": energies, "avg": round(avg_energy, 1)},
            ))

        return alerts


# ─── One-shot analysis (for non-background use) ─────────────────────────────

def run_proactive_check(db: HealthDatabase) -> List[ProactiveAlert]:
    """Run a single proactive analysis pass and return alerts."""
    engine = ProactiveEngine(db)
    return engine.run_analysis()
