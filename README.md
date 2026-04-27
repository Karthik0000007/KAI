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
| **Health Dashboard (React + FastAPI)** | **✅ COMPLETE** |
| **Wearable Integration (BLE Heart Rate)** | **✅ COMPLETE** |
| **Vision Module (Facial Expression)** | **✅ COMPLETE** |
| **Backup & Recovery System** | **✅ COMPLETE** |
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
├── start_dashboard_server.py # Dashboard server startup
├── requirements.txt
├── README.md
├── core/
│   ├── __init__.py
│   ├── config.py             # Central configuration
│   ├── models.py             # Data models (dataclasses)
│   ├── encryption.py         # AES-256-GCM encryption + differential privacy
│   ├── emotion.py            # Audio emotion detection (librosa)
│   ├── health_db.py          # Encrypted SQLite health store
│   ├── proactive.py          # Background proactive health engine
│   ├── llm.py                # Health-aware LLM (Ollama)
│   ├── stt.py                # Whisper STT + emotion pipeline
│   ├── tts.py                # Emotion-adaptive TTS
│   ├── wearable.py           # BLE wearable integration (heart rate)
│   ├── vision.py             # Facial expression recognition
│   ├── backup_manager.py     # Automated backup & recovery
│   ├── audit_logger.py       # Tamper-evident audit logging
│   ├── key_manager.py        # Encryption key management
│   └── dashboard_api.py      # FastAPI backend for dashboard
├── dashboard/                # React frontend (NEW)
│   ├── package.json
│   ├── public/
│   ├── src/
│   │   ├── App.js
│   │   ├── components/
│   │   │   ├── Dashboard.js
│   │   │   ├── Login.js
│   │   │   ├── CalendarView.js
│   │   │   ├── AlertsPanel.js
│   │   │   ├── StatisticsPanel.js
│   │   │   └── charts/
│   │   │       ├── TrendChart.js
│   │   │       ├── EmotionPieChart.js
│   │   │       ├── VitalSignsChart.js
│   │   │       └── CorrelationChart.js
│   │   └── services/
│   │       └── api.js        # API + WebSocket client
│   ├── README.md
│   └── QUICKSTART.md
├── data/
│   ├── db/                   # Encrypted database + keys
│   ├── audio/                # Temp audio files
│   ├── logs/
│   └── models/
└── tests/                    # Comprehensive test suite
    ├── test_dashboard_api.py
    ├── test_backup_manager.py
    ├── test_wearable.py
    ├── test_vision.py
    └── ...
```

---

## Quick Start

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.ai/) running locally with `gemma:2b` pulled
- Node.js 16+ and npm (for dashboard)
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

### Run Voice Assistant

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

### Run Health Dashboard (NEW)

**Start Backend:**
```bash
python start_dashboard_server.py
```

**Start Frontend (in new terminal):**
```bash
cd dashboard
npm install  # First time only
npm start
```

**Access Dashboard:**
- Frontend: http://localhost:3000
- Backend API: http://127.0.0.1:8000
- API Docs: http://127.0.0.1:8000/docs

**Login:**
- Username: `admin`
- Password: `admin`

**Features:**
- 📊 Interactive trend charts (mood, sleep, energy)
- 🥧 Emotion distribution analysis
- 💓 Vital signs tracking (heart rate, SpO2, temperature)
- 📈 Correlation analysis (sleep vs mood, energy vs sleep)
- 📅 Calendar view with daily health summaries
- 🔔 Proactive alerts with acknowledgment
- 📥 Export data (PNG charts, CSV data)
- 🔄 Real-time WebSocket updates
- ♿ Full accessibility support (WCAG 2.1 AA)

See `dashboard/QUICKSTART.md` for detailed instructions.

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
- **AES-256-GCM encryption** for sensitive data at rest (upgraded from AES-128)
- **Keyring integration** — encryption keys stored in OS keyring (Windows Credential Manager, macOS Keychain, Linux Secret Service)
- **Differential privacy** — Laplace noise injected on numeric health metrics before storage
- **Tamper-evident audit logging** — HMAC chain for integrity verification
- **Automated backups** — Daily backups with 30-day retention and integrity verification
- **WAL mode SQLite** — crash-safe database writes
- **Session-based authentication** — 30-minute timeout for dashboard access
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
| Vision | OpenCV + Haar Cascade (facial expression) |
| Wearables | Bleak (BLE) for heart rate monitors |
| Database | SQLite + AES-256-GCM encryption |
| Privacy | Differential privacy (Laplace mechanism) |
| Backup | SQLite backup API + integrity verification |
| Dashboard Backend | FastAPI + WebSocket |
| Dashboard Frontend | React + Chart.js + react-chartjs-2 |
| Language | Python 3.10+ / JavaScript (React) |

---

## Roadmap

### ✅ Completed (Week 1-3)
- [x] Async pipeline with event bus
- [x] Error handling and graceful degradation
- [x] YAML configuration management
- [x] Enhanced logging and metrics
- [x] Japanese language support (partial)
- [x] LLM-based health signal extraction
- [x] Advanced emotion detection with calibration
- [x] Proactive engine with 12 pattern detectors
- [x] Property-based testing infrastructure
- [x] AES-256-GCM encryption upgrade
- [x] Keyring integration for key storage
- [x] Audit logging with tamper detection
- [x] Automated backup and recovery system
- [x] Vision module (facial expression recognition)
- [x] Wearable integration (BLE heart rate)
- [x] **Health Dashboard (React + FastAPI) - COMPLETE**

### 🚧 In Progress (Week 4)
- [ ] Multi-user support with voice biometrics
- [ ] Advanced vision features (eye fatigue, posture)
- [ ] Complete wearable support (SpO2, temperature, fitness trackers)
- [ ] Physical embodiment architecture (robot hardware)
- [ ] Plugin system and extensibility
- [ ] Accessibility features (voice-only, screen reader)
- [ ] Setup wizard and user onboarding
- [ ] Complete documentation
- [ ] End-to-end testing
- [ ] Performance optimization

### 🔮 Future
- [ ] Japanese TTS via VOICEVOX (auto-start + MeCab fix)
- [ ] Physical embodiment — Baymax-inspired companion robot
- [ ] Multi-language support (Spanish, French, German)
- [ ] Mobile app version

> _"I want to build Baymax. I need to add vision, then maybe build the actual robot."_

---

## License

Private project — all rights reserved.
