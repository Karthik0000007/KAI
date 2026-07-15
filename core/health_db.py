"""
Aegis Health Memory Store
Encrypted SQLite database for local health data persistence.
Crash-safe, privacy-preserving, fully offline.
"""

import json
import sqlite3
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Dict, Any

from core.config import DB_PATH
from core.encryption import encrypt_string, decrypt_string, get_fernet, sanitize_for_storage
from core.models import (
    HealthCheckIn, MedicationReminder, VitalRecord,
    ProactiveAlert, ConversationTurn,
)

logger = logging.getLogger("aegis.health_db")


class HealthDatabase:
    """
    Encrypted local health data store backed by SQLite.
    
    Features:
        - Encrypted text fields (notes, user_text, pain_notes)
        - Differential privacy noise on numeric health metrics
        - WAL mode for crash safety
        - Simple query API for the proactive engine
    """

    SENSITIVE_TEXT_FIELDS = {"user_text", "pain_notes", "notes", "message"}
    NOISY_NUMERIC_FIELDS = ["mood_score", "sleep_hours", "energy_level"]

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = str(db_path or DB_PATH)
        self.fernet = get_fernet()
        self._conn: Optional[sqlite3.Connection] = None
        self._init_db()

    # ─── Connection ──────────────────────────────────────────────────────

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_path)
            self._conn.execute("PRAGMA journal_mode=WAL;")
            self._conn.execute("PRAGMA foreign_keys=ON;")
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def close(self):
        if self._conn:
            self._conn.close()
            self._conn = None

    # ─── Schema ──────────────────────────────────────────────────────────

    def _init_db(self):
        conn = self._get_conn()
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS health_checkins (
                id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                mood_score REAL,
                sleep_hours REAL,
                energy_level REAL,
                pain_notes TEXT,
                medication_taken INTEGER,
                user_text TEXT,
                detected_emotion TEXT,
                emotion_confidence REAL,
                notes TEXT
            );

            CREATE TABLE IF NOT EXISTS medication_reminders (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                dosage TEXT,
                schedule_time TEXT,
                active INTEGER DEFAULT 1,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS vital_records (
                id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                vital_type TEXT NOT NULL,
                value REAL NOT NULL,
                unit TEXT NOT NULL,
                device_id TEXT,
                confidence REAL DEFAULT 1.0,
                heart_rate INTEGER,
                blood_pressure_sys INTEGER,
                blood_pressure_dia INTEGER,
                spo2 REAL,
                temperature REAL,
                steps INTEGER,
                calories REAL,
                active_minutes INTEGER
            );

            CREATE TABLE IF NOT EXISTS proactive_alerts (
                id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                alert_type TEXT NOT NULL,
                severity TEXT DEFAULT 'info',
                message TEXT,
                acknowledged INTEGER DEFAULT 0,
                context TEXT
            );

            CREATE TABLE IF NOT EXISTS conversation_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                emotion TEXT,
                tone_mode TEXT
            );

            CREATE TABLE IF NOT EXISTS emotion_transitions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                from_emotion TEXT NOT NULL,
                to_emotion TEXT NOT NULL,
                from_confidence REAL NOT NULL,
                to_confidence REAL NOT NULL,
                timestamp TEXT NOT NULL,
                transition_type TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_checkin_ts ON health_checkins(timestamp);
            CREATE INDEX IF NOT EXISTS idx_vital_ts ON vital_records(timestamp);
            CREATE INDEX IF NOT EXISTS idx_alert_ts ON proactive_alerts(timestamp);
            CREATE INDEX IF NOT EXISTS idx_conv_session ON conversation_history(session_id);
            CREATE INDEX IF NOT EXISTS idx_emotion_transition_session ON emotion_transitions(session_id);
            CREATE INDEX IF NOT EXISTS idx_emotion_transition_ts ON emotion_transitions(timestamp);
        """)
        conn.commit()
        logger.info(f"Health database initialized at {self.db_path}")

    # ─── Encryption helpers ──────────────────────────────────────────────

    def _encrypt_sensitive(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Encrypt sensitive text fields before storage."""
        encrypted = data.copy()
        for field in self.SENSITIVE_TEXT_FIELDS:
            if field in encrypted and encrypted[field] is not None:
                encrypted[field] = encrypt_string(str(encrypted[field]), self.fernet)
        return encrypted

    def _decrypt_sensitive(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Decrypt sensitive text fields after retrieval."""
        decrypted = data.copy()
        for field in self.SENSITIVE_TEXT_FIELDS:
            if field in decrypted and decrypted[field] is not None:
                try:
                    decrypted[field] = decrypt_string(str(decrypted[field]), self.fernet)
                except Exception:
                    pass  # If decryption fails, leave as-is (might be plaintext)
        return decrypted

    # ─── Health Check-ins ────────────────────────────────────────────────

    def save_checkin(self, checkin: HealthCheckIn) -> str:
        """
        Save a health check-in with encryption and DP noise.
        
        Implements graceful degradation:
        - Retries on transient database errors (locked, busy)
        - Logs error and continues on permanent failure
        """
        try:
            data = checkin.to_dict()
            data = sanitize_for_storage(data, self.NOISY_NUMERIC_FIELDS)
            data = self._encrypt_sensitive(data)

            conn = self._get_conn()
            
            # Retry logic for database locks
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    conn.execute("""
                        INSERT OR REPLACE INTO health_checkins
                        (id, timestamp, mood_score, sleep_hours, energy_level, pain_notes,
                         medication_taken, user_text, detected_emotion, emotion_confidence, notes)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        data["id"], data["timestamp"], data.get("mood_score"),
                        data.get("sleep_hours"), data.get("energy_level"),
                        data.get("pain_notes"), 1 if data.get("medication_taken") else 0,
                        data.get("user_text"), data.get("detected_emotion"),
                        data.get("emotion_confidence"), data.get("notes"),
                    ))
                    conn.commit()
                    logger.info(f"Saved check-in {checkin.id}")
                    return checkin.id
                except sqlite3.OperationalError as e:
                    if "locked" in str(e).lower() or "busy" in str(e).lower():
                        if attempt < max_retries - 1:
                            import time
                            wait_time = 0.1 * (2 ** attempt)
                            logger.warning(f"Database locked, retrying in {wait_time}s...")
                            time.sleep(wait_time)
                            continue
                    raise
        except Exception as e:
            logger.error(f"Failed to save check-in {checkin.id}: {e}. Data may be lost.")
            # Continue execution - don't crash on database errors
            return checkin.id

    def get_recent_checkins(self, days: int = 7) -> List[Dict[str, Any]]:
        """
        Get check-ins from the last N days.
        
        Implements graceful degradation:
        - Returns empty list on database errors
        """
        try:
            cutoff = (datetime.now() - timedelta(days=days)).isoformat()
            conn = self._get_conn()
            rows = conn.execute(
                "SELECT * FROM health_checkins WHERE timestamp >= ? ORDER BY timestamp DESC",
                (cutoff,)
            ).fetchall()
            return [self._decrypt_sensitive(dict(r)) for r in rows]
        except Exception as e:
            logger.error(f"Failed to retrieve recent check-ins: {e}")
            return []

    def get_checkin_stats(self, days: int = 7) -> Dict[str, Any]:
        """
        Aggregate stats for proactive analysis.
        
        Implements graceful degradation:
        - Returns default empty stats on errors
        """
        try:
            checkins = self.get_recent_checkins(days)
            if not checkins:
                return {"count": 0, "avg_mood": None, "avg_sleep": None, "avg_energy": None}

            moods = [c["mood_score"] for c in checkins if c.get("mood_score") is not None]
            sleeps = [c["sleep_hours"] for c in checkins if c.get("sleep_hours") is not None]
            energies = [c["energy_level"] for c in checkins if c.get("energy_level") is not None]
            emotions = [c["detected_emotion"] for c in checkins if c.get("detected_emotion")]

            return {
                "count": len(checkins),
                "avg_mood": round(sum(moods) / len(moods), 1) if moods else None,
                "avg_sleep": round(sum(sleeps) / len(sleeps), 1) if sleeps else None,
                "avg_energy": round(sum(energies) / len(energies), 1) if energies else None,
                "recent_emotions": emotions[:5],
                "low_mood_days": sum(1 for m in moods if m <= 3),
                "low_sleep_days": sum(1 for s in sleeps if s < 5.0),
            }
        except Exception as e:
            logger.error(f"Failed to compute check-in stats: {e}")
            return {"count": 0, "avg_mood": None, "avg_sleep": None, "avg_energy": None}

    # ─── Medication Reminders ────────────────────────────────────────────

    def save_medication(self, med: MedicationReminder) -> str:
        conn = self._get_conn()
        conn.execute("""
            INSERT OR REPLACE INTO medication_reminders
            (id, name, dosage, schedule_time, active, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (med.id, med.name, med.dosage, med.schedule_time,
              1 if med.active else 0, med.created_at))
        conn.commit()
        return med.id

    def get_active_medications(self) -> List[Dict[str, Any]]:
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM medication_reminders WHERE active = 1"
        ).fetchall()
        return [dict(r) for r in rows]

    # ─── Vital Records ──────────────────────────────────────────────────

    def save_vital(self, vital: VitalRecord) -> str:
        data = vital.to_dict()
        conn = self._get_conn()
        
        # Determine vital_type, value, and unit from the VitalRecord
        vital_type = None
        value = None
        unit = None
        
        if data.get("heart_rate") is not None:
            vital_type = "heart_rate"
            value = float(data["heart_rate"])
            unit = "bpm"
        elif data.get("spo2") is not None:
            vital_type = "spo2"
            value = float(data["spo2"])
            unit = "%"
        elif data.get("temperature") is not None:
            vital_type = "temperature"
            value = float(data["temperature"])
            unit = "°C"
        elif data.get("steps") is not None:
            vital_type = "steps"
            value = float(data["steps"])
            unit = "steps"
        elif data.get("calories") is not None:
            vital_type = "calories"
            value = float(data["calories"])
            unit = "kcal"
        elif data.get("active_minutes") is not None:
            vital_type = "active_minutes"
            value = float(data["active_minutes"])
            unit = "minutes"
        
        # If no vital type determined, use defaults
        if vital_type is None:
            vital_type = "unknown"
            value = 0.0
            unit = ""
        
        conn.execute("""
            INSERT OR REPLACE INTO vital_records
            (id, timestamp, vital_type, value, unit, device_id, confidence,
             heart_rate, blood_pressure_sys, blood_pressure_dia,
             spo2, temperature, steps, calories, active_minutes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (data["id"], data["timestamp"], vital_type, value, unit,
              "unknown", 1.0,  # device_id and confidence defaults
              data.get("heart_rate"),
              data.get("blood_pressure_sys"), data.get("blood_pressure_dia"),
              data.get("spo2"), data.get("temperature"), data.get("steps"),
              data.get("calories"), data.get("active_minutes")))
        conn.commit()
        return vital.id

    def get_recent_vitals(self, days: int = 7) -> List[Dict[str, Any]]:
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM vital_records WHERE timestamp >= ? ORDER BY timestamp DESC",
            (cutoff,)
        ).fetchall()
        return [dict(r) for r in rows]

    # ─── Proactive Alerts ────────────────────────────────────────────────

    def save_alert(self, alert: ProactiveAlert) -> str:
        data = alert.to_dict()
        data = self._encrypt_sensitive(data)
        conn = self._get_conn()
        conn.execute("""
            INSERT OR REPLACE INTO proactive_alerts
            (id, timestamp, alert_type, severity, message, acknowledged, context)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (data["id"], data["timestamp"], data["alert_type"], data["severity"],
              data.get("message", ""), 1 if data.get("acknowledged") else 0,
              json.dumps(data.get("context", {}))))
        conn.commit()
        return alert.id

    def get_unacknowledged_alerts(self) -> List[Dict[str, Any]]:
        """
        Get unacknowledged proactive alerts.
        
        Implements graceful degradation:
        - Returns empty list on database errors
        """
        try:
            conn = self._get_conn()
            rows = conn.execute(
                "SELECT * FROM proactive_alerts WHERE acknowledged = 0 ORDER BY timestamp DESC"
            ).fetchall()
            results = []
            for r in rows:
                d = dict(r)
                d = self._decrypt_sensitive(d)
                if d.get("context"):
                    try:
                        d["context"] = json.loads(d["context"])
                    except (json.JSONDecodeError, TypeError):
                        d["context"] = {}
                results.append(d)
            return results
        except Exception as e:
            logger.error(f"Failed to retrieve unacknowledged alerts: {e}")
            return []

    def acknowledge_alert(self, alert_id: str):
        conn = self._get_conn()
        conn.execute(
            "UPDATE proactive_alerts SET acknowledged = 1 WHERE id = ?", (alert_id,)
        )
        conn.commit()

    def get_recent_alerts(self, hours: int = 24) -> List[Dict[str, Any]]:
        """
        Get alerts from the last N hours.
        
        Used for deduplication to avoid sending similar alerts too frequently.
        
        Args:
            hours: Number of hours to look back (default: 24)
            
        Returns:
            List of alert dictionaries
            
        Requirement 10.8: Deduplicate similar alerts within 24 hours
        """
        try:
            cutoff = (datetime.now() - timedelta(hours=hours)).isoformat()
            conn = self._get_conn()
            rows = conn.execute(
                "SELECT * FROM proactive_alerts WHERE timestamp >= ? ORDER BY timestamp DESC",
                (cutoff,)
            ).fetchall()
            results = []
            for r in rows:
                d = dict(r)
                d = self._decrypt_sensitive(d)
                if d.get("context"):
                    try:
                        d["context"] = json.loads(d["context"])
                    except (json.JSONDecodeError, TypeError):
                        d["context"] = {}
                results.append(d)
            return results
        except Exception as e:
            logger.error(f"Failed to retrieve recent alerts: {e}")
            return []
    
    def get_alerts_today(self) -> List[Dict[str, Any]]:
        """
        Get all alerts generated today.
        
        Used for limiting the number of alerts per day to avoid alert fatigue.
        
        Returns:
            List of alert dictionaries from today
            
        Requirement 10.8: Avoid alert fatigue by limiting alerts to 3 per day
        """
        try:
            # Get start of today (midnight)
            today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0).isoformat()
            conn = self._get_conn()
            rows = conn.execute(
                "SELECT * FROM proactive_alerts WHERE timestamp >= ? ORDER BY timestamp DESC",
                (today_start,)
            ).fetchall()
            results = []
            for r in rows:
                d = dict(r)
                d = self._decrypt_sensitive(d)
                if d.get("context"):
                    try:
                        d["context"] = json.loads(d["context"])
                    except (json.JSONDecodeError, TypeError):
                        d["context"] = {}
                results.append(d)
            return results
        except Exception as e:
            logger.error(f"Failed to retrieve today's alerts: {e}")
            return []

    # ─── Conversation History ────────────────────────────────────────────

    def save_conversation_turn(self, session_id: str, turn: ConversationTurn):
        """
        Save a conversation turn to history.
        
        Implements graceful degradation:
        - Retries on database locks
        - Logs error and continues on failure
        """
        try:
            data = turn.to_dict()
            data = self._encrypt_sensitive(data)
            conn = self._get_conn()
            
            # Retry logic for database locks
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    conn.execute("""
                        INSERT INTO conversation_history
                        (session_id, role, content, timestamp, emotion, tone_mode)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """, (session_id, data["role"],
                          encrypt_string(data["content"], self.fernet),
                          data["timestamp"], data.get("emotion"), data.get("tone_mode")))
                    conn.commit()
                    return
                except sqlite3.OperationalError as e:
                    if "locked" in str(e).lower() or "busy" in str(e).lower():
                        if attempt < max_retries - 1:
                            import time
                            wait_time = 0.1 * (2 ** attempt)
                            logger.warning(f"Database locked, retrying in {wait_time}s...")
                            time.sleep(wait_time)
                            continue
                    raise
        except Exception as e:
            logger.error(f"Failed to save conversation turn: {e}. Turn may be lost.")

    def get_session_history(self, session_id: str, limit: int = 20) -> List[Dict]:
        conn = self._get_conn()
        rows = conn.execute("""
            SELECT * FROM conversation_history 
            WHERE session_id = ? ORDER BY id DESC LIMIT ?
        """, (session_id, limit)).fetchall()
        results = []
        for r in reversed(list(rows)):
            d = dict(r)
            try:
                d["content"] = decrypt_string(d["content"], self.fernet)
            except Exception:
                pass
            results.append(d)
        return results

    # ─── Emotion Transitions ─────────────────────────────────────────────

    def save_emotion_transition(self, session_id: str, transition: Dict[str, Any]):
        """
        Save an emotion transition to the database.
        
        Args:
            session_id: The session ID
            transition: Dictionary with transition information
        
        Implements graceful degradation:
        - Retries on database locks
        - Logs error and continues on failure
        """
        try:
            conn = self._get_conn()
            
            # Retry logic for database locks
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    conn.execute("""
                        INSERT INTO emotion_transitions
                        (session_id, from_emotion, to_emotion, from_confidence, 
                         to_confidence, timestamp, transition_type)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (
                        session_id,
                        transition["from_emotion"],
                        transition["to_emotion"],
                        transition["from_confidence"],
                        transition["to_confidence"],
                        transition["timestamp"],
                        transition["transition_type"]
                    ))
                    conn.commit()
                    logger.info(f"Saved emotion transition: {transition['transition_type']}")
                    return
                except sqlite3.OperationalError as e:
                    if "locked" in str(e).lower() or "busy" in str(e).lower():
                        if attempt < max_retries - 1:
                            import time
                            wait_time = 0.1 * (2 ** attempt)
                            logger.warning(f"Database locked, retrying in {wait_time}s...")
                            time.sleep(wait_time)
                            continue
                    raise
        except Exception as e:
            logger.error(f"Failed to save emotion transition: {e}. Transition may be lost.")

    def get_emotion_transitions(self, session_id: Optional[str] = None, 
                               days: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get emotion transitions from the database.
        
        Args:
            session_id: Optional session ID to filter by
            days: Optional number of days to look back
        
        Returns:
            List of emotion transition dictionaries
        
        Implements graceful degradation:
        - Returns empty list on database errors
        """
        try:
            conn = self._get_conn()
            
            if session_id and days:
                cutoff = (datetime.now() - timedelta(days=days)).isoformat()
                rows = conn.execute("""
                    SELECT * FROM emotion_transitions 
                    WHERE session_id = ? AND timestamp >= ?
                    ORDER BY timestamp DESC
                """, (session_id, cutoff)).fetchall()
            elif session_id:
                rows = conn.execute("""
                    SELECT * FROM emotion_transitions 
                    WHERE session_id = ?
                    ORDER BY timestamp DESC
                """, (session_id,)).fetchall()
            elif days:
                cutoff = (datetime.now() - timedelta(days=days)).isoformat()
                rows = conn.execute("""
                    SELECT * FROM emotion_transitions 
                    WHERE timestamp >= ?
                    ORDER BY timestamp DESC
                """, (cutoff,)).fetchall()
            else:
                rows = conn.execute("""
                    SELECT * FROM emotion_transitions 
                    ORDER BY timestamp DESC
                """).fetchall()
            
            return [dict(r) for r in rows]
        except Exception as e:
            logger.error(f"Failed to retrieve emotion transitions: {e}")
            return []

