"""
Aegis Text-to-Speech Module
Emotion-adaptive local TTS with support for English (Coqui) and Japanese (VOICEVOX).
No cloud dependency.
"""

import asyncio
import os
import json
import shutil
import subprocess
import time
import logging
import platform
from typing import Optional

import torch
import requests
from TTS.api import TTS
from TTS.utils.radam import RAdam
from collections import defaultdict
from langdetect import detect
from fugashi import Tagger

from core.config import (
    COQUI_MODEL_EN, VOICEVOX_URL, VOICEVOX_SPEAKER_ID,
    OUTPUT_AUDIO_FILE, TONE_MODES,
)

logger = logging.getLogger("aegis.tts")

# Lazy-loaded MeCab tagger (avoids crash if dictionary not installed)
_tagger = None


def _get_tagger():
    global _tagger
    if _tagger is None:
        try:
            # Try using unidic-lite dictionary path explicitly
            try:
                import unidic_lite
                _tagger = Tagger(f'-d "{unidic_lite.DICDIR}"')
            except ImportError:
                _tagger = Tagger()
        except RuntimeError as e:
            logger.warning(f"MeCab/fugashi init failed: {e}. "
                           "Install 'unidic-lite' for Japanese tokenization.")
            return None
    return _tagger


def clean_japanese_text(text: str) -> str:
    """Tokenize and clean Japanese text for better TTS clarity."""
    t = _get_tagger()
    if t is None:
        return text  # fallback: return text as-is
    tokens = [word.surface for word in t(text)]
    return "".join(tokens)


# ─── Coqui TTS (lazy load) ──────────────────────────────────────────────────
_coqui_tts = None


def _get_coqui():
    global _coqui_tts
    if _coqui_tts is None:
        logger.info(f"Loading Coqui TTS model: {COQUI_MODEL_EN}")
        # add_safe_globals is only available in PyTorch 2.6+
        if hasattr(torch.serialization, "add_safe_globals"):
            torch.serialization.add_safe_globals([RAdam, defaultdict, dict])
        _coqui_tts = TTS(model_name=COQUI_MODEL_EN)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _coqui_tts.to(torch.device(device))
        logger.info(f"Coqui TTS loaded on {device}")
    return _coqui_tts


# ─── Language Detection ─────────────────────────────────────────────────────

def detect_language(text: str) -> str:
    """Detect language of text, returning 'en' or 'ja'."""
    try:
        lang = detect(text)
        if lang.startswith("ja"):
            return "ja"
        return "en"
    except Exception:
        return "en"


# ─── Emotion-Adaptive Speech ────────────────────────────────────────────────

def _adapt_text_for_emotion(text: str, tone_mode: Optional[str] = None) -> str:
    """
    Optionally prepend a brief spoken preamble or adjust phrasing
    based on the desired tone mode. This helps the TTS convey the right feel.
    """
    if not tone_mode or tone_mode == "neutral":
        return text

    # For TTS, slight pauses (commas) and softer openers help convey tone
    preambles = {
        "calm": "",          # calm = no extra words, steady delivery
        "encouraging": "",   # the LLM already adapts content
        "gentle_support": "",
    }

    return preambles.get(tone_mode, "") + text


# ─── Main TTS Function ──────────────────────────────────────────────────────

def speak_text(
    text: str,
    language: Optional[str] = None,
    tone_mode: Optional[str] = None,
    filename: Optional[str] = None,
    play_audio: bool = True,
) -> Optional[str]:
    """
    Synthesize speech from text with emotion-adaptive delivery.
    
    Args:
        text: Text to speak.
        language: 'en' or 'ja'. Auto-detected if None.
        tone_mode: Response tone (calm/encouraging/gentle_support/neutral).
        filename: Output WAV path.
        play_audio: Whether to play the audio after synthesis.
    
    Returns:
        Path to the generated audio file, or None on failure.
    """
    if not text or not text.strip():
        logger.warning("Empty text, skipping TTS")
        return None

    filepath = str(filename or OUTPUT_AUDIO_FILE)
    if language is None:
        language = detect_language(text)

    # Adapt text for emotional delivery
    adapted_text = _adapt_text_for_emotion(text, tone_mode)

    logger.info(f"TTS [{language.upper()}] tone={tone_mode or 'neutral'}: {adapted_text[:60]}...")

    if language == "ja":
        success = _synthesize_japanese(adapted_text, filepath)
    elif language == "en":
        success = _synthesize_english(adapted_text, filepath)
    else:
        logger.warning(f"Unsupported language: {language}, falling back to English")
        success = _synthesize_english(adapted_text, filepath)

    if success and play_audio:
        _play_audio(filepath)

    return filepath if success else None


# ─── Synthesis Backends ──────────────────────────────────────────────────────

def _synthesize_english(text: str, filepath: str) -> bool:
    """Synthesize English speech using Coqui TTS."""
    try:
        tts = _get_coqui()
        tts.tts_to_file(text=text, file_path=filepath)
        logger.info(f"English TTS saved to {filepath}")
        return True
    except Exception as e:
        logger.error(f"Coqui TTS error: {e}")
        print(f"  [tts] English synthesis failed: {e}")
        return False


# ─── VOICEVOX auto-start ─────────────────────────────────────────────────────
_voicevox_process = None


def _ensure_voicevox_running() -> bool:
    """Check if VOICEVOX is reachable; if not, try to auto-start it from PATH."""
    global _voicevox_process

    # Already alive?
    try:
        r = requests.get(f"{VOICEVOX_URL}/version", timeout=2)
        if r.status_code == 200:
            return True
    except Exception:
        pass

    # Try to find and launch the engine
    exe_names = ["VOICEVOX", "run", "engine", "voicevox_engine"]
    exe_path = None

    # Check PATH
    for name in exe_names:
        found = shutil.which(name)
        if found:
            exe_path = found
            break

    # Check common install locations (Windows, macOS, Linux)
    if exe_path is None:
        common_paths = []
        
        # Windows common locations
        if os.name == "nt":
            common_paths.extend([
                os.path.expandvars(r"%LOCALAPPDATA%\Programs\VOICEVOX\run.exe"),
                os.path.expandvars(r"%LOCALAPPDATA%\Programs\VOICEVOX\VOICEVOX.exe"),
                r"C:\Program Files\VOICEVOX\run.exe",
                r"C:\VOICEVOX\run.exe",
            ])
            # Also search PATH directories for any voicevox-related exe
            path_dirs = os.environ.get("PATH", "").split(os.pathsep)
            for d in path_dirs:
                for name in ["run.exe", "VOICEVOX.exe", "voicevox_engine.exe"]:
                    candidate = os.path.join(d, name)
                    if os.path.isfile(candidate):
                        common_paths.insert(0, candidate)
        
        # macOS common locations
        elif os.name == "posix" and platform.system() == "Darwin":
            common_paths.extend([
                "/Applications/VOICEVOX.app/Contents/MacOS/run",
                "/Applications/VOICEVOX.app/Contents/MacOS/VOICEVOX",
                os.path.expanduser("~/Applications/VOICEVOX.app/Contents/MacOS/run"),
                os.path.expanduser("~/Applications/VOICEVOX.app/Contents/MacOS/VOICEVOX"),
                "/usr/local/bin/voicevox",
                "/opt/homebrew/bin/voicevox",
            ])
        
        # Linux common locations
        elif os.name == "posix":
            common_paths.extend([
                "/opt/voicevox/run",
                "/opt/voicevox/voicevox",
                "/usr/local/bin/voicevox",
                "/usr/bin/voicevox",
                os.path.expanduser("~/.local/share/voicevox/run"),
                os.path.expanduser("~/.local/share/voicevox/voicevox"),
                os.path.expanduser("~/voicevox/run"),
            ])

        for p in common_paths:
            if os.path.isfile(p):
                exe_path = p
                break

    if exe_path is None:
        logger.warning("VOICEVOX executable not found in PATH or common locations")
        return False

    logger.info(f"Starting VOICEVOX engine: {exe_path}")
    try:
        # Platform-specific subprocess flags
        if os.name == "nt":
            # Windows: hide console window
            creation_flags = subprocess.CREATE_NO_WINDOW
        else:
            # Unix-like: no special flags needed
            creation_flags = 0
        
        _voicevox_process = subprocess.Popen(
            [exe_path, "--host", "127.0.0.1", "--port", "50021"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=creation_flags,
        )
        
        # Wait for it to become ready
        logger.info("Waiting for VOICEVOX engine to start...")
        for attempt in range(20):  # up to ~10 seconds
            time.sleep(0.5)
            try:
                r = requests.get(f"{VOICEVOX_URL}/version", timeout=1)
                if r.status_code == 200:
                    version = r.text.strip()
                    logger.info(f"VOICEVOX engine started successfully (version {version})")
                    return True
            except requests.RequestException:
                continue
        
        logger.warning("VOICEVOX started but not responding in time")
        return False
    except FileNotFoundError as e:
        logger.error(f"Failed to start VOICEVOX: executable not found - {e}")
        return False
    except PermissionError as e:
        logger.error(f"Failed to start VOICEVOX: permission denied - {e}")
        return False
    except Exception as e:
        logger.error(f"Failed to start VOICEVOX: {e}")
        return False


def _synthesize_japanese(text: str, filepath: str) -> bool:
    """Synthesize Japanese speech using VOICEVOX (auto-started if needed)."""
    # Ensure VOICEVOX is running
    if not _ensure_voicevox_running():
        logger.warning("VOICEVOX unavailable — falling back to pyttsx3")
        return _synthesize_pyttsx3(text, filepath, lang="ja")

    try:
        cleaned = clean_japanese_text(text)
        query = requests.post(
            f"{VOICEVOX_URL}/audio_query",
            params={"text": cleaned, "speaker": VOICEVOX_SPEAKER_ID},
            timeout=10,
        )
        query.raise_for_status()

        synthesis = requests.post(
            f"{VOICEVOX_URL}/synthesis",
            headers={"Content-Type": "application/json"},
            params={"speaker": VOICEVOX_SPEAKER_ID},
            data=json.dumps(query.json()),
            timeout=30,
        )
        synthesis.raise_for_status()

        with open(filepath, "wb") as f:
            f.write(synthesis.content)

        logger.info(f"Japanese TTS saved to {filepath}")
        return True
    except requests.ConnectionError:
        logger.warning("VOICEVOX not running — falling back to pyttsx3 for Japanese")
        return _synthesize_pyttsx3(text, filepath, lang="ja")
    except Exception as e:
        logger.error(f"VOICEVOX TTS error: {e}")
        return _synthesize_pyttsx3(text, filepath, lang="ja")


def _synthesize_pyttsx3(text: str, filepath: str, lang: str = "ja") -> bool:
    """Fallback TTS using pyttsx3 (Windows SAPI5 / espeak)."""
    try:
        import pyttsx3
        engine = pyttsx3.init()
        voices = engine.getProperty('voices')
        # Try to find a matching voice for the language
        target_lang = "Japanese" if lang == "ja" else lang
        for voice in voices:
            if target_lang.lower() in voice.name.lower() or lang in voice.id.lower():
                engine.setProperty('voice', voice.id)
                break
        engine.setProperty('rate', 150)
        engine.save_to_file(text, filepath)
        engine.runAndWait()
        logger.info(f"pyttsx3 fallback TTS saved to {filepath}")
        return True
    except Exception as e:
        logger.error(f"pyttsx3 fallback TTS failed: {e}")
        print(f"  [tts] All Japanese TTS engines failed. Install VOICEVOX or pyttsx3.")
        return False


# ─── Audio Playback ─────────────────────────────────────────────────────────

def _play_audio(filepath: str):
    """Play an audio file using the system's default player."""
    try:
        if os.name == "nt":
            # Windows: use start command (non-blocking)
            os.system(f'start "" "{filepath}"')
        elif os.name == "posix":
            # Linux/Mac
            if os.path.exists("/usr/bin/aplay"):
                os.system(f"aplay -q {filepath}")
            elif os.path.exists("/usr/bin/afplay"):
                os.system(f"afplay {filepath}")  # macOS
            else:
                logger.info(f"Audio saved to {filepath} (no player found)")
        else:
            logger.info(f"Audio saved to {filepath}")
    except Exception as e:
        logger.error(f"Audio playback error: {e}")


# ─── Async Wrappers ──────────────────────────────────────────────────────────

async def speak_text_async(
    text: str,
    language: Optional[str] = None,
    tone_mode: Optional[str] = None,
    filename: Optional[str] = None,
    play_audio: bool = True,
) -> Optional[str]:
    """
    Async wrapper for speak_text using asyncio.to_thread.
    
    Implements graceful degradation:
    - Retry logic (2 attempts) for transient TTS failures
    - Timeout handling (30s per attempt)
    - Fallback to silent mode on failure
    
    Synthesize speech from text with emotion-adaptive delivery.
    
    Args:
        text: Text to speak.
        language: 'en' or 'ja'. Auto-detected if None.
        tone_mode: Response tone (calm/encouraging/gentle_support/neutral).
        filename: Output WAV path.
        play_audio: Whether to play the audio after synthesis.
    
    Returns:
        Path to the generated audio file, or None on failure.
    """
    from core.error_handling import with_retry_and_timeout, FallbackStrategies
    
    try:
        return await with_retry_and_timeout(
            speak_text,
            text,
            language,
            tone_mode,
            filename,
            play_audio,
            max_retries=2,
            timeout=30.0,
            initial_delay=1.0
        )
    except Exception as e:
        logger.error(f"TTS failed after all retries: {e}")
        # Use fallback strategy (silent mode)
        return await FallbackStrategies.tts_fallback(text, language or "en")
