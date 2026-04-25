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
        
        Applies prioritization and deduplication:
        - Prioritizes by severity and recency
        - Deduplicates similar alerts within 24 hours
        - Limits to 3 alerts per day
        
        Requirements: 10.7, 10.8
        """
        logger.info("Running proactive health analysis...")
        alerts: List[ProactiveAlert] = []

        alerts.extend(self._check_mood_pattern())
        alerts.extend(self._check_sleep_deficit())
        alerts.extend(self._check_medication_compliance())
        alerts.extend(self._check_vital_signs())
        alerts.extend(self._check_emotion_pattern())
        alerts.extend(self._check_energy_trend())
        alerts.extend(self._check_sleep_pattern_disruption())
        alerts.extend(self._check_activity_level_changes())
        alerts.extend(self._check_pain_trends())

        # Apply deduplication (Requirement 10.8)
        alerts = self._deduplicate_alerts(alerts)
        
        # Apply prioritization (Requirement 10.7)
        alerts = self._prioritize_alerts(alerts)
        
        # Limit to 3 alerts per day (Requirement 10.8)
        alerts = self._limit_alerts_per_day(alerts, max_alerts=3)

        # Save and notify
        for alert in alerts:
            self.db.save_alert(alert)
            if self.on_alert:
                try:
                    self.on_alert(alert)
                except Exception as e:
                    logger.error(f"Alert callback error: {e}")

        if alerts:
            logger.info(f"Generated {len(alerts)} proactive alert(s) after prioritization and deduplication")
        else:
            logger.info("No proactive alerts at this time")

        return alerts

    # ─── Alert Prioritization and Deduplication ─────────────────────────

    def _prioritize_alerts(self, alerts: List[ProactiveAlert]) -> List[ProactiveAlert]:
        """
        Prioritize alerts by severity and recency.
        
        Severity order: urgent > warning > info
        Within same severity, more recent alerts are prioritized.
        
        Requirement 10.7: Prioritize alerts by severity and recency
        
        Args:
            alerts: List of alerts to prioritize
            
        Returns:
            Sorted list of alerts (highest priority first)
        """
        if not alerts:
            return alerts
        
        # Define severity weights (higher = more important)
        severity_weights = {
            "urgent": 3,
            "warning": 2,
            "info": 1
        }
        
        # Calculate priority score for each alert
        # Score = severity_weight * 1000 + recency_score
        # Recency score: newer alerts get higher scores
        now = datetime.now()
        
        def calculate_priority(alert: ProactiveAlert) -> float:
            severity_score = severity_weights.get(alert.severity, 1) * 1000
            
            # Parse timestamp and calculate recency (seconds ago)
            try:
                alert_time = datetime.fromisoformat(alert.timestamp)
                seconds_ago = (now - alert_time).total_seconds()
                # Recency score: inverse of age (newer = higher score)
                # Cap at 24 hours (86400 seconds)
                recency_score = max(0, 86400 - seconds_ago) / 86400 * 100
            except (ValueError, AttributeError):
                recency_score = 0
            
            return severity_score + recency_score
        
        # Sort by priority (highest first)
        prioritized = sorted(alerts, key=calculate_priority, reverse=True)
        
        logger.debug(f"Prioritized {len(prioritized)} alerts by severity and recency")
        return prioritized
    
    def _deduplicate_alerts(self, alerts: List[ProactiveAlert]) -> List[ProactiveAlert]:
        """
        Deduplicate similar alerts within 24 hours.
        
        Two alerts are considered duplicates if they have the same alert_type
        and were generated within 24 hours of each other.
        
        Requirement 10.8: Deduplicate similar alerts within 24 hours
        
        Args:
            alerts: List of alerts to deduplicate
            
        Returns:
            Deduplicated list of alerts
        """
        if not alerts:
            return alerts
        
        # Get recent alerts from database (last 24 hours)
        recent_alerts = self.db.get_recent_alerts(hours=24)
        
        # Build set of recent alert types
        recent_alert_types = {alert.get("alert_type") for alert in recent_alerts}
        
        # Filter out duplicate alerts
        deduplicated = []
        for alert in alerts:
            if alert.alert_type not in recent_alert_types:
                deduplicated.append(alert)
            else:
                logger.debug(
                    f"Deduplicating alert: {alert.alert_type} "
                    f"(already alerted within 24 hours)"
                )
        
        if len(deduplicated) < len(alerts):
            logger.info(
                f"Deduplicated {len(alerts) - len(deduplicated)} alerts "
                f"({len(deduplicated)} remaining)"
            )
        
        return deduplicated
    
    def _limit_alerts_per_day(
        self, 
        alerts: List[ProactiveAlert], 
        max_alerts: int = 3
    ) -> List[ProactiveAlert]:
        """
        Limit alerts to a maximum number per day to avoid alert fatigue.
        
        Assumes alerts are already prioritized (highest priority first).
        Checks how many alerts have been generated today and limits accordingly.
        
        Requirement 10.8: Avoid alert fatigue by limiting alerts to 3 per day
        
        Args:
            alerts: List of prioritized alerts
            max_alerts: Maximum number of alerts per day (default: 3)
            
        Returns:
            Limited list of alerts
        """
        if not alerts:
            return alerts
        
        # Get alerts generated today
        today_alerts = self.db.get_alerts_today()
        alerts_today_count = len(today_alerts)
        
        # Calculate how many more alerts we can send today
        remaining_slots = max(0, max_alerts - alerts_today_count)
        
        if remaining_slots == 0:
            logger.info(
                f"Alert limit reached: {alerts_today_count}/{max_alerts} alerts "
                f"already sent today. Suppressing {len(alerts)} new alerts."
            )
            return []
        
        if len(alerts) > remaining_slots:
            logger.info(
                f"Limiting alerts: {len(alerts)} generated, "
                f"but only {remaining_slots} slots remaining today "
                f"({alerts_today_count}/{max_alerts} already sent)"
            )
            return alerts[:remaining_slots]
        
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
                explanation=(
                    f"This alert was generated because your mood scores have been "
                    f"consistently low for {low_mood_days} consecutive days, which exceeds "
                    f"the threshold of {self.cfg.low_mood_days_threshold} days. "
                    f"Your average mood score during this period was {avg_mood}/10. "
                    f"Sustained low mood can indicate emotional distress that may benefit "
                    f"from attention or support."
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
                explanation=(
                    f"This alert was triggered because your average sleep duration over the past "
                    f"7 days was {avg_sleep:.1f} hours, which is below the healthy threshold of "
                    f"{self.cfg.low_sleep_hours} hours. You had {low_sleep_days} days with "
                    f"insufficient sleep. Chronic sleep deprivation can affect mood, energy, "
                    f"cognitive function, and overall health."
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
                        explanation=(
                            f"This reminder was generated because your medication "
                            f"{med['name']} was scheduled for {schedule}, and it's now "
                            f"{self.cfg.missed_medication_reminder_delay} minutes past that time "
                            f"without confirmation of taking it. No check-ins today indicated "
                            f"medication was taken. Consistent medication adherence is important "
                            f"for treatment effectiveness."
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
                    explanation=(
                        f"This alert was generated because your heart rate reached {max_hr} bpm, "
                        f"which exceeds the elevated threshold of {self.cfg.elevated_hr_threshold} bpm, "
                        f"and your recent emotional state included stress or anxiety indicators "
                        f"({', '.join(recent_emotions)}). The combination of elevated heart rate "
                        f"and emotional stress suggests your body may be in a heightened state of arousal "
                        f"that could benefit from calming techniques."
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
                    explanation=(
                        f"This alert was triggered because your heart rate reached {max_hr} bpm, "
                        f"which is above the elevated threshold of {self.cfg.elevated_hr_threshold} bpm. "
                        f"Your recent emotional state appeared neutral or calm. Elevated heart rate "
                        f"without stress indicators may be due to physical activity, caffeine, or other "
                        f"factors. If you haven't been active, it may be worth monitoring."
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
                explanation=(
                    f"This alert was generated because {negative_count} out of your last "
                    f"{len(recent_emotions)} emotional assessments over the past 5 days showed "
                    f"negative emotions (stressed, anxious, or fatigued). This pattern of sustained "
                    f"emotional distress may indicate you're experiencing ongoing challenges that "
                    f"could benefit from discussion or additional support."
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
                explanation=(
                    f"This alert was triggered because your energy levels have shown a consistent "
                    f"declining trend over the past {len(energies)} check-ins (from {energies[0]:.1f} "
                    f"to {energies[-1]:.1f}), with an average of {avg_energy:.1f}/10. "
                    f"Sustained declining energy can indicate insufficient rest, poor nutrition, "
                    f"dehydration, or accumulating stress. Early intervention can help prevent "
                    f"further decline."
                ),
                context={"energies": energies, "avg": round(avg_energy, 1)},
            ))

        return alerts

    def _check_sleep_pattern_disruption(self) -> List[ProactiveAlert]:
        """Detect unusual sleep times based on check-in timestamps."""
        alerts = []
        checkins = self.db.get_recent_checkins(days=7)
        
        # Extract check-ins with sleep data
        sleep_checkins = [
            c for c in checkins
            if c.get("sleep_hours") is not None
        ]
        
        if len(sleep_checkins) < 3:
            return alerts
        
        # Analyze check-in times as proxy for sleep schedule
        # Unusual times: check-ins between 2 AM - 6 AM or very late (after midnight)
        unusual_times = []
        for checkin in sleep_checkins:
            try:
                timestamp = datetime.fromisoformat(checkin["timestamp"])
                hour = timestamp.hour
                
                # Flag unusual times: 2 AM - 6 AM (likely disrupted sleep)
                if 2 <= hour < 6:
                    unusual_times.append({
                        "timestamp": checkin["timestamp"],
                        "hour": hour,
                        "sleep_hours": checkin.get("sleep_hours")
                    })
            except (ValueError, KeyError):
                continue
        
        # Alert if multiple unusual sleep times detected
        if len(unusual_times) >= 2:
            hours_list = [u["hour"] for u in unusual_times]
            alerts.append(ProactiveAlert(
                alert_type="sleep_pattern_disruption",
                severity="warning",
                message=(
                    f"I've noticed you've been checking in at unusual times "
                    f"({len(unusual_times)} times in the early morning hours). "
                    f"This might indicate disrupted sleep patterns. Try to maintain "
                    f"a consistent sleep schedule — going to bed and waking up at "
                    f"the same time each day can really help."
                ),
                explanation=(
                    f"This alert was generated because you had {len(unusual_times)} check-ins "
                    f"during early morning hours (2 AM - 6 AM) over the past 7 days, specifically "
                    f"at hours: {', '.join(f'{h}:00' for h in hours_list)}. Check-ins during these "
                    f"hours suggest disrupted sleep patterns or irregular sleep schedules. "
                    f"Consistent sleep timing is crucial for sleep quality and overall health."
                ),
                context={
                    "unusual_checkins": len(unusual_times),
                    "times": hours_list
                },
            ))
        
        return alerts

    def _check_activity_level_changes(self) -> List[ProactiveAlert]:
        """Detect sudden decrease in energy reports."""
        alerts = []
        checkins = self.db.get_recent_checkins(days=7)
        
        # Extract energy levels with timestamps
        energy_data = [
            {
                "energy": c["energy_level"],
                "timestamp": c["timestamp"]
            }
            for c in checkins
            if c.get("energy_level") is not None
        ]
        
        if len(energy_data) < 4:
            return alerts
        
        # Split into recent (last 2 days) and baseline (previous days)
        recent_cutoff = (datetime.now() - timedelta(days=2)).isoformat()
        recent_energies = [
            e["energy"] for e in energy_data
            if e["timestamp"] >= recent_cutoff
        ]
        baseline_energies = [
            e["energy"] for e in energy_data
            if e["timestamp"] < recent_cutoff
        ]
        
        if not recent_energies or not baseline_energies:
            return alerts
        
        # Calculate averages
        avg_recent = sum(recent_energies) / len(recent_energies)
        avg_baseline = sum(baseline_energies) / len(baseline_energies)
        
        # Detect sudden drop (>2 points decrease)
        energy_drop = avg_baseline - avg_recent
        
        if energy_drop >= 2.0 and avg_recent < 5.0:
            alerts.append(ProactiveAlert(
                alert_type="activity_level_change",
                severity="warning",
                message=(
                    f"Your energy levels have dropped significantly in the past couple days "
                    f"(from {avg_baseline:.1f} to {avg_recent:.1f}). This sudden change "
                    f"might indicate you need more rest, or it could be worth checking if "
                    f"something else is affecting your energy. How are you feeling overall?"
                ),
                explanation=(
                    f"This alert was triggered because your energy levels showed a sudden significant "
                    f"decrease. Your baseline energy over the past week (excluding the last 2 days) "
                    f"averaged {avg_baseline:.1f}/10 across {len(baseline_energies)} check-ins, "
                    f"but in the last 2 days it dropped to {avg_recent:.1f}/10 across "
                    f"{len(recent_energies)} check-ins — a drop of {energy_drop:.1f} points. "
                    f"Sudden energy changes can indicate illness onset, increased stress, or other "
                    f"factors requiring attention."
                ),
                context={
                    "baseline_avg": round(avg_baseline, 1),
                    "recent_avg": round(avg_recent, 1),
                    "drop": round(energy_drop, 1)
                },
            ))
        
        return alerts

    def _check_pain_trends(self) -> List[ProactiveAlert]:
        """Detect worsening pain trends based on pain notes frequency."""
        alerts = []
        checkins = self.db.get_recent_checkins(days=14)
        
        # Extract check-ins with pain notes
        pain_checkins = [
            {
                "timestamp": c["timestamp"],
                "pain_notes": c.get("pain_notes", "")
            }
            for c in checkins
            if c.get("pain_notes")
        ]
        
        if len(pain_checkins) < 3:
            return alerts
        
        # Split into recent (last 7 days) and previous (7-14 days ago)
        recent_cutoff = (datetime.now() - timedelta(days=7)).isoformat()
        recent_pain = [
            p for p in pain_checkins
            if p["timestamp"] >= recent_cutoff
        ]
        previous_pain = [
            p for p in pain_checkins
            if p["timestamp"] < recent_cutoff
        ]
        
        # Detect increasing frequency of pain reports
        recent_count = len(recent_pain)
        previous_count = len(previous_pain)
        
        # Alert if pain reports increased significantly
        if recent_count >= 3 and recent_count > previous_count * 1.5:
            increase_pct = ((recent_count - previous_count) / max(previous_count, 1)) * 100
            alerts.append(ProactiveAlert(
                alert_type="pain_trend_worsening",
                severity="warning",
                message=(
                    f"I've noticed you've been reporting pain more frequently lately "
                    f"({recent_count} times in the past week vs {previous_count} the week before). "
                    f"Increasing pain frequency can be concerning. If this continues or gets worse, "
                    f"it might be worth discussing with your healthcare provider."
                ),
                explanation=(
                    f"This alert was generated because your pain reports increased significantly. "
                    f"In the past 7 days, you reported pain {recent_count} times, compared to "
                    f"{previous_count} times in the previous 7 days (days 8-14) — an increase of "
                    f"{increase_pct:.0f}%. This exceeds the 50% increase threshold. "
                    f"Increasing pain frequency can indicate worsening conditions, inadequate pain "
                    f"management, or new health issues that may require medical attention."
                ),
                context={
                    "recent_count": recent_count,
                    "previous_count": previous_count,
                    "trend": "increasing"
                },
            ))
        
        return alerts


# ─── One-shot analysis (for non-background use) ─────────────────────────────

def run_proactive_check(db: HealthDatabase) -> List[ProactiveAlert]:
    """Run a single proactive analysis pass and return alerts."""
    engine = ProactiveEngine(db)
    return engine.run_analysis()
