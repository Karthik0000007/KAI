"""
Aegis Speech-to-Text Module
Local ASR via Whisper with integrated emotion feature extraction.
No cloud dependency.
"""

import logging
from typing import Tuple, Optional

import whisper
import sounddevice as sd
import soundfile as sf
import numpy as np

from core.config import WHISPER_MODEL_SIZE, SAMPLE_RATE, RECORD_DURATION_DEFAULT, INPUT_AUDIO_FILE
from core.emotion import analyze_emotion
from core.models import EmotionResult

logger = logging.getLogger("aegis.stt")

# ─── Whisper Model (lazy load) ───────────────────────────────────────────────
_whisper_model = None


def _get_model():
    global _whisper_model
    if _whisper_model is None:
        logger.info(f"Loading Whisper model: {WHISPER_MODEL_SIZE}")
        _whisper_model = whisper.load_model(WHISPER_MODEL_SIZE)
        logger.info("Whisper model loaded")
    return _whisper_model


# ─── Audio Recording ─────────────────────────────────────────────────────────

def record_audio(
    filename: Optional[str] = None,
    duration: int = RECORD_DURATION_DEFAULT,
    samplerate: int = SAMPLE_RATE,
) -> str:
    """
    Record audio from microphone with graceful error handling.
    
    Returns:
        Path to the saved WAV file.
    """
    filepath = str(filename or INPUT_AUDIO_FILE)
    logger.info(f"Recording {duration}s of audio...")
    print(f"  [mic] Recording for {duration} seconds...")

    try:
        audio = sd.rec(
            int(duration * samplerate),
            samplerate=samplerate,
            channels=1,
            dtype="float32",
        )
        sd.wait()

        # Check if we got meaningful audio (not silence)
        rms = float(np.sqrt(np.mean(audio ** 2)))
        if rms < 0.001:
            logger.warning("Audio appears to be silence")
            print("  [mic] Warning: Very quiet or no audio detected")

        sf.write(filepath, audio, samplerate)
        logger.info(f"Audio saved to {filepath} (RMS: {rms:.4f})")
        print(f"  [mic] Audio saved")
        return filepath

    except sd.PortAudioError as e:
        logger.error(f"Audio device error: {e}")
        print(f"  [mic] Error: No audio input device found. Check your microphone.")
        raise RuntimeError(f"Microphone not available: {e}")
    except Exception as e:
        logger.error(f"Recording error: {e}")
        raise


# ─── Transcription ───────────────────────────────────────────────────────────

def transcribe_audio(
    filename: Optional[str] = None,
) -> Tuple[str, str]:
    """
    Transcribe audio file with language detection.
    
    Returns:
        Tuple of (transcribed_text, language_code).
    """
    filepath = str(filename or INPUT_AUDIO_FILE)
    model = _get_model()

    logger.info(f"Transcribing: {filepath}")

    try:
        # Detect language
        audio = whisper.load_audio(filepath)
        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio).to(model.device)

        _, probs = model.detect_language(mel)
        detected_lang = max(probs, key=probs.get)
        logger.info(f"Detected language: {detected_lang}")

        # Transcribe
        result = model.transcribe(filepath, language=detected_lang)
        text = result["text"].strip()

        if not text:
            logger.warning("Empty transcription")
            text = ""

        logger.info(f"Transcription: {text[:80]}...")
        return text, detected_lang

    except Exception as e:
        logger.error(f"Transcription error: {e}")
        # Graceful degradation: return empty text rather than crash
        return "", "en"


# ─── Combined: Record + Transcribe + Emotion ────────────────────────────────

def listen_and_analyze(
    duration: int = RECORD_DURATION_DEFAULT,
) -> Tuple[str, str, EmotionResult]:
    """
    Full voice input pipeline:
      1. Record audio from microphone
      2. Transcribe with Whisper
      3. Analyze emotion from audio features + transcript
    
    Returns:
        Tuple of (transcribed_text, language_code, emotion_result).
    """
    # Step 1: Record
    audio_path = record_audio(duration=duration)

    # Step 2: Transcribe
    text, lang = transcribe_audio(audio_path)

    # Step 3: Emotion analysis
    try:
        emotion = analyze_emotion(audio_path, transcript=text)
    except Exception as e:
        logger.error(f"Emotion analysis failed (non-fatal): {e}")
        emotion = EmotionResult(
            label="neutral", confidence=0.5,
            pitch_mean=0.0, pitch_std=0.0,
            energy_rms=0.0, speech_rate=0.0,
        )

    return text, lang, emotion

