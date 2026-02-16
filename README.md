# Aegis — Offline Voice-First Personal Health AI

A fully offline, privacy-preserving, emotion-aware, proactive health assistant designed to run locally on edge devices. Blends the gentle care of Baymax, the responsiveness of JARVIS, and the privacy-conscious, localized power of Gemma.

---

## Features

| Capability | Description |
|---|---|
| **Fully Offline** | All inference runs locally — no cloud, no telemetry |
| **Voice-First** | Mic → Whisper ASR → LLM → TTS → Speaker |
| **Emotion-Aware** | Detects calm / stressed / anxious / fatigued / neutral from pitch, energy, speech rate & linguistic cues |
| **Proactive Engine** | Background analysis detects low-mood streaks, sleep deficits, missed meds, elevated HR patterns |
| **Privacy by Design** | Encrypted SQLite, differential privacy noise injection, no raw data leaves the device |
| **Health Memory** | Tracks mood, sleep, energy, medications, vitals, and conversation history |
| **Emotion-Adaptive TTS** | Response tone adapts (calm / encouraging / gentle support) based on detected emotion |
| **Bilingual (WIP)** | English (Coqui TTS) working · Japanese (VOICEVOX) in progress |

### Status

| Feature | Status |
|---|---|
| English STT (Whisper) | **Working** |
| English TTS (Coqui Jenny) | **Working** |
| Emotion Detection (librosa) | **Working** |
| Health-Aware LLM (Ollama + Gemma 2B) | **Working** |
| Encrypted Health DB (SQLite + Fernet) | **Working** |
| Proactive Engine (background analysis) | **Working** |
| Japanese STT (Whisper) | Partial — transcription works, MeCab tokenization WIP |
| Japanese TTS (VOICEVOX) | WIP — pyttsx3 fallback available |

---

## Architecture

```
Microphone
   │
   ▼
Local ASR (Whisper)  ──►  Language Detection
   │
   ▼
Emotion Classifier  ──►  pitch · energy · rate · keywords
   │
   ▼
Health Signal Extractor  ──►  sleep · mood · meds · pain
   │
   ▼
LLM Reasoning Engine (Ollama / Gemma)
   │  ├── conversation history
   │  ├── health stats context
   │  ├── proactive alerts
   │  └── emotion-adaptive tone
   ▼
TTS Engine (Coqui / VOICEVOX)
   │
   ▼
Speaker
   │
   ╰─── Encrypted Health DB (SQLite + Fernet AES)
         ├── check-ins
         ├── medications
         ├── vitals
         ├── conversation history
         └── proactive alerts
```

---

## Project Structure

```
KAI/
├── app.py                    # Main orchestrator
├── requirements.txt
├── README.md
├── core/
│   ├── __init__.py
│   ├── config.py             # Central configuration
│   ├── models.py             # Data models (dataclasses)
│   ├── encryption.py         # Fernet encryption + differential privacy
│   ├── emotion.py            # Audio emotion detection (librosa)
│   ├── health_db.py          # Encrypted SQLite health store
│   ├── proactive.py          # Background proactive health engine
│   ├── llm.py                # Health-aware LLM (Ollama)
│   ├── stt.py                # Whisper STT + emotion pipeline
│   └── tts.py                # Emotion-adaptive TTS
└── data/
    ├── db/                   # Encrypted database + keys
    ├── audio/                # Temp audio files
    ├── logs/
    └── models/
```

---

## Quick Start

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.ai/) running locally with `gemma:2b` pulled
- CUDA GPU recommended (CPU fallback supported)
- VOICEVOX (optional, for Japanese TTS)

### Install

```bash
pip install -r requirements.txt
```

### Pull the LLM model

```bash
ollama pull gemma:2b
```

### Run

```bash
python app.py
```

Speak to Aegis. It will:
1. Record your voice
2. Transcribe (Whisper)
3. Detect your emotional state from audio
4. Extract health signals (sleep, mood, meds, pain)
5. Query the LLM with full health context
6. Respond in an emotion-adaptive tone
7. Store encrypted health data locally
8. Run proactive checks in the background

---

## Proactive Interventions

The background engine detects patterns and alerts you:

| Pattern | Trigger | Response |
|---|---|---|
| Low mood streak | 3+ days mood ≤ 3/10 | Suggests talking / rest |
| Sleep deficit | Avg sleep < 5h | Recommends sleep hygiene |
| Missed medication | Past schedule + no confirmation | Gentle reminder |
| Elevated HR + stress | HR ≥ 100 bpm + stressed emotion | Breathing exercise |
| Emotional distress | 3+ negative emotions in 5 check-ins | Empathetic outreach |
| Energy decline | Declining trend + avg < 4/10 | Lifestyle suggestions |

---

## Privacy & Security

- **No cloud calls** — everything runs on localhost
- **Fernet (AES-128-CBC)** encryption for sensitive text fields at rest
- **Differential privacy** — Laplace noise injected on numeric health metrics before storage
- **WAL mode SQLite** — crash-safe database writes
- **Key stored locally** with restricted file permissions
- **No telemetry, no analytics, no external requests**

---

## Tech Stack

| Component | Technology |
|---|---|
| STT | OpenAI Whisper (local) |
| LLM | Ollama + Gemma 2B (quantized) |
| TTS (EN) | Coqui TTS (Jenny) |
| TTS (JA) | VOICEVOX |
| Emotion | librosa (pitch/energy/ZCR/spectral) |
| Database | SQLite + Fernet encryption |
| Privacy | Differential privacy (Laplace mechanism) |
| Language | Python 3.10+ |

---

## Roadmap

- [ ] Japanese TTS via VOICEVOX (auto-start + MeCab fix)
- [ ] Vision module (camera-based health observation)
- [ ] Wearable integration (real heart rate, SpO2, temperature)
- [ ] Web dashboard for health trend visualization
- [ ] Physical embodiment — Baymax-inspired companion robot

> _"I want to build Baymax. I need to add vision, then maybe build the actual robot."_

---

## License

Private project — all rights reserved.
