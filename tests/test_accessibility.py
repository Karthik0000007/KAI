import pytest
import numpy as np
import os
import wave
from pathlib import Path

from core.config import ACCESSIBILITY_CONFIG
from core.tts import _apply_audio_effects, detect_language
from core.llm import extract_health_signals
from core.emotion import classify_emotion

# Create a temporary dummy wav file for testing
@pytest.fixture
def dummy_wav(tmp_path):
    path = tmp_path / "test.wav"
    # Create 1 second of silence at 16kHz
    sr = 16000
    n_samples = sr
    audio_data = np.zeros(n_samples, dtype=np.int16)
    
    with wave.open(str(path), 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(audio_data.tobytes())
        
    return str(path)

# ─── TTS and Audio Effect Tests ───
def test_audio_effects(dummy_wav):
    # Test applying rate and volume
    success = _apply_audio_effects(dummy_wav, rate_factor=1.5, volume_factor=0.5)
    assert success is True
    
    # We just ensure it doesn't crash and returns True
    # Soundfile and Librosa handle the actual math

# ─── Language Detection Tests ───
def test_detect_language():
    assert detect_language("Hello how are you?") == "en"
    assert detect_language("こんにちは、元気ですか？") == "ja"
    assert detect_language("Hola, ¿cómo estás?") == "es"
    assert detect_language("Bonjour, comment ça va?") == "fr"
    assert detect_language("Guten Tag, wie geht es dir?") == "de"

# ─── Multi-language LLM Extraction (Regex Fallback) Tests ───
def test_es_extraction():
    text = "Dormí unas 7.5 horas pero me duele la cabeza y tomé mi pastilla."
    signals = extract_health_signals(text, language="es")
    assert signals.get("sleep_hours") == 7.5
    assert signals.get("pain_mentioned") is True
    assert signals.get("medication_taken") is True

def test_fr_extraction():
    text = "J'ai dormi environ 6 heures, pas pris mon médicament."
    signals = extract_health_signals(text, language="fr")
    assert signals.get("sleep_hours") == 6.0
    assert signals.get("medication_taken") is False

def test_de_extraction():
    text = "Ich habe 8 stunden geschlafen. Keine tablette genommen."
    signals = extract_health_signals(text, language="de")
    assert signals.get("sleep_hours") == 8.0
    assert signals.get("medication_taken") is False

# ─── Multi-language Emotion Keywords Tests ───
def test_es_emotion_keywords():
    features = {"pitch_mean": 150} # Base features
    # Check Spanish 'stressed' keyword
    result = classify_emotion(features, transcript="Estoy muy estresado por el trabajo")
    # Because 'estresado' is matched, 'stressed' score should increase
    # It might not be the absolute highest if default values pull it, but let's check secondary label too
    assert result.label == "stressed" or result.secondary_label == "stressed"

def test_fr_emotion_keywords():
    features = {"pitch_mean": 150} 
    result = classify_emotion(features, transcript="Je suis tellement fatigué")
    assert result.label == "fatigued" or result.secondary_label == "fatigued"

def test_de_emotion_keywords():
    features = {"pitch_mean": 150} 
    result = classify_emotion(features, transcript="Ich bin sehr ängstlich")
    assert result.label == "anxious" or result.secondary_label == "anxious"
