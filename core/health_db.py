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
                heart_rate INTEGER,
                blood_pressure_sys INTEGER,
                blood_pressure_dia INTEGER,
                spo2 REAL,
                temperature REAL,
                steps INTEGER
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

            CREATE INDEX IF NOT EXISTS idx_checkin_ts ON health_checkins(timestamp);
            CREATE INDEX IF NOT EXISTS idx_vital_ts ON vital_records(timestamp);
            CREATE INDEX IF NOT EXISTS idx_alert_ts ON proactive_alerts(timestamp);
            CREATE INDEX IF NOT EXISTS idx_conv_session ON conversation_history(session_id);
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
        """Save a health check-in with encryption and DP noise."""
        data = checkin.to_dict()
        data = sanitize_for_storage(data, self.NOISY_NUMERIC_FIELDS)
        data = self._encrypt_sensitive(data)

        conn = self._get_conn()
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

    def get_recent_checkins(self, days: int = 7) -> List[Dict[str, Any]]:
        """Get check-ins from the last N days."""
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM health_checkins WHERE timestamp >= ? ORDER BY timestamp DESC",
            (cutoff,)
        ).fetchall()
        return [self._decrypt_sensitive(dict(r)) for r in rows]

    def get_checkin_stats(self, days: int = 7) -> Dict[str, Any]:
        """Aggregate stats for proactive analysis."""
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
        conn.execute("""
            INSERT OR REPLACE INTO vital_records
            (id, timestamp, heart_rate, blood_pressure_sys, blood_pressure_dia,
             spo2, temperature, steps)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (data["id"], data["timestamp"], data.get("heart_rate"),
              data.get("blood_pressure_sys"), data.get("blood_pressure_dia"),
              data.get("spo2"), data.get("temperature"), data.get("steps")))
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

    def acknowledge_alert(self, alert_id: str):
        conn = self._get_conn()
        conn.execute(
            "UPDATE proactive_alerts SET acknowledged = 1 WHERE id = ?", (alert_id,)
        )
        conn.commit()

    # ─── Conversation History ────────────────────────────────────────────

    def save_conversation_turn(self, session_id: str, turn: ConversationTurn):
        data = turn.to_dict()
        data = self._encrypt_sensitive(data)
        conn = self._get_conn()
        conn.execute("""
            INSERT INTO conversation_history
            (session_id, role, content, timestamp, emotion, tone_mode)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (session_id, data["role"],
              encrypt_string(data["content"], self.fernet),
              data["timestamp"], data.get("emotion"), data.get("tone_mode")))
        conn.commit()

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
