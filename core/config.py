"""
Aegis Configuration Module
Central configuration for the offline health AI system.
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional


# ─── Paths ───────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DB_DIR = DATA_DIR / "db"
AUDIO_DIR = DATA_DIR / "audio"
LOGS_DIR = DATA_DIR / "logs"
MODELS_DIR = DATA_DIR / "models"

# Ensure directories exist
for d in [DATA_DIR, DB_DIR, AUDIO_DIR, LOGS_DIR, MODELS_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# ─── Database ────────────────────────────────────────────────────────────────
DB_PATH = DB_DIR / "aegis_health.db"
ENCRYPTION_KEY_FILE = DB_DIR / ".aegis_key"


# ─── Audio ───────────────────────────────────────────────────────────────────
SAMPLE_RATE = 16000
RECORD_DURATION_DEFAULT = 5  # seconds
INPUT_AUDIO_FILE = AUDIO_DIR / "input.wav"
OUTPUT_AUDIO_FILE = AUDIO_DIR / "aegis_response.wav"


# ─── Whisper STT ─────────────────────────────────────────────────────────────
WHISPER_MODEL_SIZE = "base"  # "tiny", "base", "small", "medium"


# ─── LLM (Ollama) ───────────────────────────────────────────────────────────
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "gemma:2b"
LLM_CONTEXT_WINDOW = 4096


# ─── TTS ─────────────────────────────────────────────────────────────────────
COQUI_MODEL_EN = "tts_models/en/jenny/jenny"
VOICEVOX_URL = "http://127.0.0.1:50021"
VOICEVOX_SPEAKER_ID = 3  # Tsumugi


# ─── Emotion Detection ──────────────────────────────────────────────────────
@dataclass
class EmotionConfig:
    """Thresholds for emotion classification from audio features."""
    # Pitch thresholds (Hz)
    pitch_low: float = 100.0
    pitch_high: float = 250.0
    # Energy thresholds (RMS)
    energy_low: float = 0.01
    energy_high: float = 0.08
    # Speech rate thresholds (syllables/sec approx via word count)
    rate_slow: float = 1.5
    rate_fast: float = 4.0
    # Confidence threshold for emotion classification
    confidence_threshold: float = 0.4


EMOTION_CONFIG = EmotionConfig()


# ─── Emotion Labels ─────────────────────────────────────────────────────────
EMOTION_LABELS = ["calm", "stressed", "anxious", "fatigued", "neutral"]


# ─── Proactive Engine ───────────────────────────────────────────────────────
@dataclass
class ProactiveConfig:
    """Thresholds for proactive health interventions."""
    low_mood_days_threshold: int = 3          # days of consecutive low mood
    low_sleep_hours: float = 5.0              # hours considered low
    missed_medication_reminder_delay: int = 30  # minutes
    elevated_hr_threshold: int = 100          # bpm
    check_interval_minutes: int = 60          # background check interval
    mood_low_threshold: float = 3.0           # mood score 1-10


PROACTIVE_CONFIG = ProactiveConfig()


# ─── Health Check-in ────────────────────────────────────────────────────────
DAILY_CHECKIN_QUESTIONS = [
    "How are you feeling today?",
    "How many hours did you sleep last night?",
    "Have you taken your medications today?",
    "Any pain or discomfort you'd like to mention?",
    "On a scale of 1 to 10, how would you rate your energy level?",
]


# ─── Response Tone Modes ────────────────────────────────────────────────────
TONE_MODES = {
    "calm": {
        "system_modifier": "Respond in a calm, soothing, and reassuring tone. Use gentle language.",
        "speech_rate_factor": 0.9,
    },
    "encouraging": {
        "system_modifier": "Respond in an upbeat, encouraging, and motivating tone. Be positive.",
        "speech_rate_factor": 1.0,
    },
    "gentle_support": {
        "system_modifier": "Respond with gentle empathy and support. Be understanding and caring.",
        "speech_rate_factor": 0.85,
    },
    "neutral": {
        "system_modifier": "Respond in a clear, neutral, and informative tone.",
        "speech_rate_factor": 1.0,
    },
}


# ─── System Prompt ───────────────────────────────────────────────────────────
AEGIS_SYSTEM_PROMPT = """You are Aegis, an offline personal health AI companion. You are caring, \
empathetic, and privacy-conscious. You NEVER suggest calling emergency services unless explicitly \
asked about emergencies. Your role is to:

1. Listen attentively to the user's health concerns
2. Provide gentle, evidence-based wellness suggestions
3. Track patterns in mood, sleep, and daily habits
4. Offer proactive interventions when you detect concerning patterns
5. Adapt your communication tone to the user's emotional state
6. Remind users about medications and healthy routines

You are NOT a replacement for medical professionals. Always clarify this when discussing \
serious symptoms. Keep responses concise (2-4 sentences) unless the user asks for detail.

You speak like a caring friend who happens to know about health and wellness."""


# ─── Privacy ─────────────────────────────────────────────────────────────────
DIFFERENTIAL_PRIVACY_EPSILON = 1.0  # Privacy budget for noise injection
ENABLE_DIFFERENTIAL_PRIVACY = True
