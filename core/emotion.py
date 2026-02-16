"""
Aegis Emotion Detection Module
Classifies emotions from audio features: pitch, energy, speech rate, and linguistic cues.
Fully offline — uses librosa for audio analysis, no cloud APIs.
"""

import math
import logging
from typing import Optional

import numpy as np
import librosa
import soundfile as sf

from core.config import EMOTION_CONFIG, EMOTION_LABELS, SAMPLE_RATE
from core.models import EmotionResult

logger = logging.getLogger("aegis.emotion")


# ─── Feature Extraction ─────────────────────────────────────────────────────

def extract_audio_features(audio_path: str, sr: int = SAMPLE_RATE) -> dict:
    """
    Extract prosodic / acoustic features from an audio file.
    
    Returns dict with:
        - pitch_mean, pitch_std (Hz)
        - energy_rms (root mean square amplitude)
        - speech_rate (estimated from onset detection)
        - duration (seconds)
        - zcr_mean (zero crossing rate, correlates with noisiness)
        - spectral_centroid_mean (brightness of sound)
    """
    try:
        y, sr_loaded = librosa.load(audio_path, sr=sr, mono=True)
    except Exception as e:
        logger.error(f"Failed to load audio: {e}")
        return _empty_features()

    duration = librosa.get_duration(y=y, sr=sr_loaded)
    if duration < 0.5:
        logger.warning("Audio too short for reliable analysis")
        return _empty_features()

    # ── Pitch (F0) via pyin ──
    f0, voiced_flag, _ = librosa.pyin(
        y, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7"), sr=sr_loaded
    )
    f0_valid = f0[~np.isnan(f0)] if f0 is not None else np.array([])
    pitch_mean = float(np.mean(f0_valid)) if len(f0_valid) > 0 else 0.0
    pitch_std = float(np.std(f0_valid)) if len(f0_valid) > 0 else 0.0

    # ── Energy (RMS) ──
    rms = librosa.feature.rms(y=y)[0]
    energy_rms = float(np.mean(rms))

    # ── Speech rate estimate (onset detection → events/second) ──
    onset_env = librosa.onset.onset_strength(y=y, sr=sr_loaded)
    onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr_loaded)
    speech_rate = len(onsets) / duration if duration > 0 else 0.0

    # ── Zero crossing rate ──
    zcr = librosa.feature.zero_crossing_rate(y=y)[0]
    zcr_mean = float(np.mean(zcr))

    # ── Spectral centroid ──
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr_loaded)[0]
    centroid_mean = float(np.mean(centroid))

    return {
        "pitch_mean": round(pitch_mean, 2),
        "pitch_std": round(pitch_std, 2),
        "energy_rms": round(energy_rms, 6),
        "speech_rate": round(speech_rate, 2),
        "duration": round(duration, 2),
        "zcr_mean": round(zcr_mean, 6),
        "spectral_centroid_mean": round(centroid_mean, 2),
    }


def _empty_features() -> dict:
    return {
        "pitch_mean": 0.0, "pitch_std": 0.0, "energy_rms": 0.0,
        "speech_rate": 0.0, "duration": 0.0, "zcr_mean": 0.0,
        "spectral_centroid_mean": 0.0,
    }


# ─── Emotion Classification (Rule-based + Heuristic) ────────────────────────

def classify_emotion(features: dict, transcript: Optional[str] = None) -> EmotionResult:
    """
    Classify emotion from extracted audio features and optional transcript.
    
    Uses a weighted rule-based approach combining:
      - Pitch analysis (high pitch → stressed/anxious, low → fatigued/calm)
      - Energy levels (low RMS → fatigued, high → stressed)
      - Speech rate (fast → anxious, slow → fatigued/calm)
      - Linguistic cues from transcript
    
    Returns an EmotionResult with label and confidence.
    """
    cfg = EMOTION_CONFIG
    scores = {label: 0.0 for label in EMOTION_LABELS}

    pitch = features.get("pitch_mean", 0.0)
    pitch_std = features.get("pitch_std", 0.0)
    energy = features.get("energy_rms", 0.0)
    rate = features.get("speech_rate", 0.0)

    # ── Pitch-based signals ──
    if pitch > 0:
        if pitch > cfg.pitch_high:
            scores["stressed"] += 0.25
            scores["anxious"] += 0.20
        elif pitch < cfg.pitch_low:
            scores["fatigued"] += 0.20
            scores["calm"] += 0.15
        else:
            scores["neutral"] += 0.15
            scores["calm"] += 0.10

        # High pitch variance → emotional instability → anxious/stressed
        if pitch_std > 40:
            scores["anxious"] += 0.15
            scores["stressed"] += 0.10

    # ── Energy-based signals ──
    if energy > 0:
        if energy < cfg.energy_low:
            scores["fatigued"] += 0.25
            scores["calm"] += 0.10
        elif energy > cfg.energy_high:
            scores["stressed"] += 0.20
            scores["anxious"] += 0.10
        else:
            scores["neutral"] += 0.10

    # ── Speech rate signals ──
    if rate > 0:
        if rate > cfg.rate_fast:
            scores["anxious"] += 0.20
            scores["stressed"] += 0.15
        elif rate < cfg.rate_slow:
            scores["fatigued"] += 0.15
            scores["calm"] += 0.15
        else:
            scores["neutral"] += 0.10

    # ── Linguistic cues (keyword spotting in transcript) ──
    if transcript:
        text_lower = transcript.lower()
        linguistic_signals = {
            "stressed": ["stressed", "overwhelmed", "too much", "can't handle",
                         "pressure", "deadline", "worried about work",
                         "ストレス", "大変", "辛い", "きつい"],
            "anxious": ["anxious", "nervous", "panic", "scared", "afraid",
                        "heart racing", "can't breathe", "worry",
                        "不安", "怖い", "心配", "緊張"],
            "fatigued": ["tired", "exhausted", "sleepy", "no energy", "drained",
                         "worn out", "can't sleep", "insomnia",
                         "疲れ", "眠い", "だるい", "しんどい"],
            "calm": ["relaxed", "peaceful", "fine", "good", "great", "wonderful",
                     "happy", "content",
                     "元気", "良い", "楽", "気持ちいい"],
        }
        for emotion, keywords in linguistic_signals.items():
            for kw in keywords:
                if kw in text_lower:
                    scores[emotion] += 0.20
                    break  # one match per category is enough

    # ── Normalize and pick winner ──
    total = sum(scores.values())
    if total > 0:
        scores = {k: v / total for k, v in scores.items()}

    best_label = max(scores, key=scores.get)
    confidence = scores[best_label]

    # If confidence too low, default to neutral
    if confidence < cfg.confidence_threshold:
        best_label = "neutral"
        confidence = scores.get("neutral", 0.5)

    return EmotionResult(
        label=best_label,
        confidence=round(confidence, 3),
        pitch_mean=features.get("pitch_mean", 0.0),
        pitch_std=features.get("pitch_std", 0.0),
        energy_rms=features.get("energy_rms", 0.0),
        speech_rate=features.get("speech_rate", 0.0),
    )


# ─── Convenience ─────────────────────────────────────────────────────────────

def analyze_emotion(audio_path: str, transcript: Optional[str] = None) -> EmotionResult:
    """
    Full pipeline: extract features from audio file → classify emotion.
    """
    logger.info(f"Analyzing emotion from: {audio_path}")
    features = extract_audio_features(audio_path)
    result = classify_emotion(features, transcript)
    logger.info(f"Emotion: {result.label} (confidence={result.confidence:.2f})")
    return result


def emotion_to_tone_mode(emotion_label: str) -> str:
    """
    Map detected emotion to a response tone mode for the LLM/TTS.
    """
    mapping = {
        "calm": "neutral",
        "stressed": "calm",              # respond calmly to stressed users
        "anxious": "gentle_support",     # gentle support for anxiety
        "fatigued": "encouraging",       # encourage fatigued users
        "neutral": "neutral",
    }
    return mapping.get(emotion_label, "neutral")
