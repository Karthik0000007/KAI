"""
Aegis Emotion Detection Module
Classifies emotions from audio features: pitch, energy, speech rate, and linguistic cues.
Fully offline — uses librosa for audio analysis, no cloud APIs.
"""

import math
import logging
import json
from pathlib import Path
from typing import Optional, Dict, List
from datetime import datetime

import numpy as np
import librosa
import soundfile as sf

from core.config import EMOTION_CONFIG, EMOTION_LABELS, SAMPLE_RATE, DATA_DIR
from core.models import EmotionResult

logger = logging.getLogger("aegis.emotion")

# Path to calibration data file
CALIBRATION_FILE = DATA_DIR / "emotion_calibration.json"


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

def classify_emotion(features: dict, transcript: Optional[str] = None, user_id: str = "default") -> EmotionResult:
    """
    Classify emotion from extracted audio features and optional transcript.
    
    Uses a weighted rule-based approach combining:
      - Pitch analysis (high pitch → stressed/anxious, low → fatigued/calm)
      - Energy levels (low RMS → fatigued, high → stressed)
      - Speech rate (fast → anxious, slow → fatigued/calm)
      - Linguistic cues from transcript
    
    Detects mixed emotions when the top 2 emotions have confidence difference < 0.2.
    
    If calibration data exists for the user, uses personalized thresholds.
    Otherwise, falls back to default thresholds.
    
    Args:
        features: Dictionary of audio features from extract_audio_features()
        transcript: Optional transcript text for linguistic analysis
        user_id: User identifier for personalized thresholds (default: "default")
    
    Returns an EmotionResult with label, confidence, and optional secondary emotion.
    """
    # Get appropriate config (calibrated or default)
    cfg = get_emotion_config(user_id)
    
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
            "stressed": [
                # English keywords
                "stressed", "overwhelmed", "too much", "can't handle",
                "pressure", "deadline", "worried about work",
                # Japanese keywords - expanded coverage
                "ストレス", "大変", "辛い", "きつい", "プレッシャー", 
                "締め切り", "仕事が心配", "無理", "限界", "追い詰め",
                "やばい", "間に合わ", "手に負え"
            ],
            "anxious": [
                # English keywords
                "anxious", "nervous", "panic", "scared", "afraid",
                "heart racing", "can't breathe", "worry",
                # Japanese keywords - expanded coverage
                "不安", "怖い", "心配", "緊張", "パニック", "恐れ",
                "ドキドキ", "息苦し", "落ち着か", "そわそわ",
                "気になる", "気になって", "びくびく", "おびえ"
            ],
            "fatigued": [
                # English keywords
                "tired", "exhausted", "sleepy", "no energy", "drained",
                "worn out", "can't sleep", "insomnia",
                # Japanese keywords - expanded coverage
                "疲れ", "眠い", "だるい", "しんどい", "疲労", "へとへと",
                "くたくた", "ぐったり", "眠れ", "不眠", "力が出",
                "やる気が出", "消耗", "バテ"
            ],
            "calm": [
                # English keywords
                "relaxed", "peaceful", "fine", "good", "great", "wonderful",
                "happy", "content",
                # Japanese keywords - expanded coverage
                "元気", "良い", "楽", "気持ちいい", "気持ち良", "リラックス", "穏やか",
                "平和", "幸せ", "満足", "快適", "安心", "落ち着い",
                "すっきり", "爽やか", "最高"
            ],
        }
        for emotion, keywords in linguistic_signals.items():
            for kw in keywords:
                if kw in text_lower:
                    scores[emotion] += 0.20
                    break  # one match per category is enough

    # ── Normalize and pick top 2 emotions ──
    total = sum(scores.values())
    if total > 0:
        scores = {k: v / total for k, v in scores.items()}

    # Sort emotions by confidence (descending)
    sorted_emotions = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    
    best_label = sorted_emotions[0][0]
    best_confidence = sorted_emotions[0][1]
    
    second_label = sorted_emotions[1][0] if len(sorted_emotions) > 1 else None
    second_confidence = sorted_emotions[1][1] if len(sorted_emotions) > 1 else 0.0

    # Detect mixed emotions when confidence difference < 0.2
    is_mixed = False
    if second_label and (best_confidence - second_confidence) < 0.2:
        is_mixed = True

    # If confidence too low, default to neutral
    if best_confidence < cfg.confidence_threshold:
        best_label = "neutral"
        best_confidence = scores.get("neutral", 0.5)
        is_mixed = False
        second_label = None
        second_confidence = None

    return EmotionResult(
        label=best_label,
        confidence=round(best_confidence, 3),
        pitch_mean=features.get("pitch_mean", 0.0),
        pitch_std=features.get("pitch_std", 0.0),
        energy_rms=features.get("energy_rms", 0.0),
        speech_rate=features.get("speech_rate", 0.0),
        secondary_label=second_label if is_mixed else None,
        secondary_confidence=round(second_confidence, 3) if is_mixed else None,
        is_mixed=is_mixed,
    )


# ─── Convenience ─────────────────────────────────────────────────────────────

def analyze_emotion(audio_path: str, transcript: Optional[str] = None, user_id: str = "default") -> EmotionResult:
    """
    Full pipeline: extract features from audio file → classify emotion.
    
    Args:
        audio_path: Path to audio file
        transcript: Optional transcript text for linguistic analysis
        user_id: User identifier for personalized thresholds (default: "default")
    """
    logger.info(f"Analyzing emotion from: {audio_path}")
    features = extract_audio_features(audio_path)
    result = classify_emotion(features, transcript, user_id)
    
    if result.is_mixed:
        logger.info(f"Emotion: {result.label} and {result.secondary_label} "
                   f"(confidence={result.confidence:.2f}/{result.secondary_confidence:.2f})")
    else:
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


# ─── Async Wrappers ──────────────────────────────────────────────────────────

async def extract_audio_features_async(audio_path: str, sr: int = SAMPLE_RATE) -> dict:
    """
    Async wrapper for extract_audio_features using asyncio.to_thread.
    
    Returns dict with audio features.
    """
    import asyncio
    return await asyncio.to_thread(extract_audio_features, audio_path, sr)


async def classify_emotion_async(features: dict, transcript: Optional[str] = None, user_id: str = "default") -> EmotionResult:
    """
    Async wrapper for classify_emotion using asyncio.to_thread.
    
    Args:
        features: Dictionary of audio features
        transcript: Optional transcript text for linguistic analysis
        user_id: User identifier for personalized thresholds (default: "default")
    
    Returns an EmotionResult with label and confidence.
    """
    import asyncio
    return await asyncio.to_thread(classify_emotion, features, transcript, user_id)


async def analyze_emotion_async(audio_path: str, transcript: Optional[str] = None, user_id: str = "default") -> EmotionResult:
    """
    Async version of full emotion analysis pipeline with graceful degradation:
      1. Extract features from audio file
      2. Classify emotion
    
    Implements:
    - Retry logic (2 attempts) for transient failures
    - Timeout handling (10s per attempt)
    - Fallback to neutral emotion on failure
    
    This function runs the blocking operations in a thread pool to avoid
    blocking the async event loop.
    
    Returns:
        EmotionResult with emotion label, confidence, and audio features.
    """
    import asyncio
    from core.error_handling import with_retry_and_timeout, FallbackStrategies
    
    logger.info(f"Analyzing emotion from: {audio_path} (async)")
    
    try:
        # Run with retry and timeout
        features = await with_retry_and_timeout(
            extract_audio_features,
            audio_path,
            max_retries=2,
            timeout=10.0,
            initial_delay=0.5
        )
        result = await asyncio.to_thread(classify_emotion, features, transcript, user_id)
        
        logger.info(f"Emotion: {result.label} (confidence={result.confidence:.2f})")
        return result
    
    except Exception as e:
        logger.error(f"Emotion analysis failed after retries: {e}")
        # Use fallback strategy
        return await FallbackStrategies.emotion_fallback(audio_path)


# ─── Emotion Calibration ─────────────────────────────────────────────────────

def calibrate_emotion(
    emotion_state: str,
    audio_samples: List[str],
    user_id: str = "default"
) -> Dict[str, float]:
    """
    Calibrate emotion detection thresholds for a specific user and emotional state.
    
    Records multiple audio samples in a given emotional state and computes
    user-specific baseline thresholds for pitch, energy, and speech rate.
    
    Args:
        emotion_state: The emotional state being calibrated (calm, stressed, anxious, fatigued)
        audio_samples: List of paths to audio files recorded in this emotional state
        user_id: User identifier for multi-user support (default: "default")
    
    Returns:
        Dictionary containing computed statistics for this emotional state:
        - pitch_mean, pitch_std
        - energy_mean, energy_std
        - rate_mean, rate_std
        - sample_count
    
    Requirements: 9.7
    """
    if emotion_state not in ["calm", "stressed", "anxious", "fatigued"]:
        raise ValueError(f"Invalid emotion state: {emotion_state}. Must be one of: calm, stressed, anxious, fatigued")
    
    if len(audio_samples) < 3:
        raise ValueError(f"At least 3 audio samples required for calibration, got {len(audio_samples)}")
    
    logger.info(f"Calibrating emotion '{emotion_state}' with {len(audio_samples)} samples for user '{user_id}'")
    
    # Extract features from all samples
    pitch_values = []
    energy_values = []
    rate_values = []
    
    for audio_path in audio_samples:
        try:
            features = extract_audio_features(audio_path)
            
            # Only include valid samples (non-zero features)
            if features["pitch_mean"] > 0:
                pitch_values.append(features["pitch_mean"])
                energy_values.append(features["energy_rms"])
                rate_values.append(features["speech_rate"])
            else:
                logger.warning(f"Skipping invalid sample: {audio_path}")
        
        except Exception as e:
            logger.error(f"Failed to extract features from {audio_path}: {e}")
            continue
    
    if len(pitch_values) < 3:
        raise ValueError(f"Not enough valid samples for calibration. Got {len(pitch_values)}, need at least 3")
    
    # Compute statistics
    stats = {
        "pitch_mean": float(np.mean(pitch_values)),
        "pitch_std": float(np.std(pitch_values)),
        "energy_mean": float(np.mean(energy_values)),
        "energy_std": float(np.std(energy_values)),
        "rate_mean": float(np.mean(rate_values)),
        "rate_std": float(np.std(rate_values)),
        "sample_count": len(pitch_values),
        "timestamp": datetime.now().isoformat()
    }
    
    logger.info(f"Calibration stats for '{emotion_state}': pitch={stats['pitch_mean']:.1f}±{stats['pitch_std']:.1f}, "
                f"energy={stats['energy_mean']:.4f}±{stats['energy_std']:.4f}, "
                f"rate={stats['rate_mean']:.2f}±{stats['rate_std']:.2f}")
    
    # Load existing calibration data
    calibration_data = load_calibration_data()
    
    # Update calibration data for this user and emotion state
    if user_id not in calibration_data:
        calibration_data[user_id] = {}
    
    calibration_data[user_id][emotion_state] = stats
    
    # Save updated calibration data
    save_calibration_data(calibration_data)
    
    logger.info(f"Calibration data saved for user '{user_id}', emotion '{emotion_state}'")
    
    return stats


def load_calibration_data() -> Dict:
    """
    Load emotion calibration data from JSON file.
    
    Returns:
        Dictionary containing calibration data for all users and emotion states.
        Structure: {user_id: {emotion_state: {stats}}}
    """
    if not CALIBRATION_FILE.exists():
        logger.debug("No calibration file found, returning empty data")
        return {}
    
    try:
        with open(CALIBRATION_FILE, 'r') as f:
            data = json.load(f)
        logger.debug(f"Loaded calibration data for {len(data)} users")
        return data
    
    except Exception as e:
        logger.error(f"Failed to load calibration data: {e}")
        return {}


def save_calibration_data(calibration_data: Dict) -> None:
    """
    Save emotion calibration data to JSON file.
    
    Args:
        calibration_data: Dictionary containing calibration data for all users
    """
    try:
        with open(CALIBRATION_FILE, 'w') as f:
            json.dump(calibration_data, f, indent=2)
        logger.debug(f"Saved calibration data to {CALIBRATION_FILE}")
    
    except Exception as e:
        logger.error(f"Failed to save calibration data: {e}")
        raise


def compute_calibrated_thresholds(user_id: str = "default") -> Optional[Dict[str, float]]:
    """
    Compute personalized emotion detection thresholds from calibration data.
    
    Uses calibrated baseline statistics to compute adaptive thresholds:
    - pitch_low: mean of calm samples
    - pitch_high: mean of stressed/anxious samples
    - energy_low: mean of fatigued samples
    - energy_high: mean of stressed samples
    - rate_slow: mean of fatigued/calm samples
    - rate_fast: mean of anxious samples
    
    Args:
        user_id: User identifier (default: "default")
    
    Returns:
        Dictionary of calibrated thresholds, or None if insufficient calibration data
    """
    calibration_data = load_calibration_data()
    
    if user_id not in calibration_data:
        logger.debug(f"No calibration data found for user '{user_id}'")
        return None
    
    user_data = calibration_data[user_id]
    
    # Check if we have enough calibration data
    required_states = ["calm", "stressed"]  # Minimum required states
    if not all(state in user_data for state in required_states):
        logger.debug(f"Insufficient calibration data for user '{user_id}'. Need at least: {required_states}")
        return None
    
    thresholds = {}
    
    # Compute pitch thresholds
    if "calm" in user_data:
        thresholds["pitch_low"] = user_data["calm"]["pitch_mean"]
    
    if "stressed" in user_data and "anxious" in user_data:
        # Average of stressed and anxious
        thresholds["pitch_high"] = (user_data["stressed"]["pitch_mean"] + 
                                    user_data["anxious"]["pitch_mean"]) / 2
    elif "stressed" in user_data:
        thresholds["pitch_high"] = user_data["stressed"]["pitch_mean"]
    elif "anxious" in user_data:
        thresholds["pitch_high"] = user_data["anxious"]["pitch_mean"]
    
    # Compute energy thresholds
    if "fatigued" in user_data:
        thresholds["energy_low"] = user_data["fatigued"]["energy_mean"]
    
    if "stressed" in user_data:
        thresholds["energy_high"] = user_data["stressed"]["energy_mean"]
    
    # Compute speech rate thresholds
    if "fatigued" in user_data and "calm" in user_data:
        # Average of fatigued and calm
        thresholds["rate_slow"] = (user_data["fatigued"]["rate_mean"] + 
                                   user_data["calm"]["rate_mean"]) / 2
    elif "fatigued" in user_data:
        thresholds["rate_slow"] = user_data["fatigued"]["rate_mean"]
    elif "calm" in user_data:
        thresholds["rate_slow"] = user_data["calm"]["rate_mean"]
    
    if "anxious" in user_data:
        thresholds["rate_fast"] = user_data["anxious"]["rate_mean"]
    
    logger.info(f"Computed calibrated thresholds for user '{user_id}': {thresholds}")
    
    return thresholds


def get_emotion_config(user_id: str = "default"):
    """
    Get emotion configuration with calibrated thresholds if available.
    
    Returns calibrated thresholds for the specified user, or falls back to
    default configuration if no calibration data exists.
    
    Args:
        user_id: User identifier (default: "default")
    
    Returns:
        EmotionConfig object with appropriate thresholds
    """
    calibrated_thresholds = compute_calibrated_thresholds(user_id)
    
    if calibrated_thresholds is None:
        logger.debug(f"Using default emotion config for user '{user_id}'")
        return EMOTION_CONFIG
    
    # Create a copy of the default config and update with calibrated values
    from dataclasses import replace
    
    config = replace(EMOTION_CONFIG)
    
    # Update thresholds with calibrated values
    for key, value in calibrated_thresholds.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    logger.debug(f"Using calibrated emotion config for user '{user_id}'")
    return config


def clear_calibration_data(user_id: Optional[str] = None) -> None:
    """
    Clear calibration data for a specific user or all users.
    
    Args:
        user_id: User identifier to clear, or None to clear all users
    """
    if user_id is None:
        # Clear all calibration data
        if CALIBRATION_FILE.exists():
            CALIBRATION_FILE.unlink()
            logger.info("Cleared all calibration data")
    else:
        # Clear data for specific user
        calibration_data = load_calibration_data()
        if user_id in calibration_data:
            del calibration_data[user_id]
            save_calibration_data(calibration_data)
            logger.info(f"Cleared calibration data for user '{user_id}'")
        else:
            logger.warning(f"No calibration data found for user '{user_id}'")
