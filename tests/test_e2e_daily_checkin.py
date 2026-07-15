"""
E2E Test: Daily Check-in Scenario
Simulates a complete conversation turn from STT (mocked) to Database to TTS (mocked).
"""

import pytest
import asyncio
from unittest.mock import patch, Mock
import requests
from datetime import datetime
import json

from core.health_db import HealthDatabase
from core.models import Session, EmotionResult
from core.event_bus import create_aegis_event_bus
from core.llm import extract_health_signals_async, get_response_async

@pytest.fixture
def temp_db(tmp_path):
    """Provide a temporary database for testing."""
    db_file = tmp_path / "test_aegis.db"
    db = HealthDatabase(db_path=str(db_file))
    yield db
    db.close()


@pytest.mark.asyncio
async def test_daily_checkin_e2e(temp_db):
    """
    E2E Test: Daily check-in scenario.
    User says "I slept for 7 hours and feel pretty good today."
    We verify the pipeline extracts the signals and saves them to the database.
    """
    event_bus = create_aegis_event_bus()
    
    # 1. Mock STT output
    stt_text = "I slept 7.0 hours and feel amazing today."
    lang = "en"
    emotion = EmotionResult(
        label="calm", 
        confidence=0.92,
        pitch_mean=120.0,
        pitch_std=10.0,
        energy_rms=0.05,
        speech_rate=3.0
    )
    
    # 2. Mock LLM Response for health extraction and reply
    with patch('core.llm.requests.post') as mock_post:
        # We need two mock responses: one for extraction, one for the conversation reply.
        
        # Extraction Response
        extract_response = Mock()
        extract_response.status_code = 200
        # The prompt asks for JSON back.
        extract_response.json.return_value = {
            "response": '```json\n{"sleep_hours": 7.0, "energy_level": 8, "mood_score": 8, "pain_level": 0}\n```'
        }
        
        # Conversation Response
        reply_response = Mock()
        reply_response.status_code = 200
        reply_response.json.return_value = {
            "response": "I'm glad to hear you got good sleep and are feeling well."
        }
        
        mock_post.side_effect = [reply_response]
        
        # --- PIPELINE SIMULATION ---
        
        # 1. STT Phase (Mocked)
        await event_bus.emit("stt.completed", {"text": stt_text, "language": lang})
        
        # 2. Extract Health Signals
        signals = await extract_health_signals_async(stt_text)
        
        # Assert signals were extracted successfully
        assert signals is not None
        assert signals.get("sleep_hours") == 7.0
        assert signals.get("mood_score") >= 7.0
        
        # 3. Database phase
        # Save checkin to DB
        from core.models import HealthCheckIn, ConversationTurn
        
        temp_db.save_checkin(HealthCheckIn(
            sleep_hours=signals.get("sleep_hours"),
            energy_level=signals.get("energy_level"),
            mood_score=signals.get("mood_score"),
            user_text=stt_text,
            detected_emotion=emotion.label,
            emotion_confidence=emotion.confidence,
            notes=f"Extracted from: {stt_text}"
        ))
        
        # Save conversation turn
        temp_db.save_conversation_turn("test_session", ConversationTurn(
            role="user",
            content=stt_text,
            emotion=emotion.label,
            tone_mode="calm"
        ))
        
        # Query stats
        stats = temp_db.get_checkin_stats(days=7)
        assert stats["count"] == 1
        assert "avg_sleep" in stats
        
        # 4. LLM Generation
        reply = await get_response_async(
            user_input=stt_text,
            emotion_label=emotion.label,
            tone_mode="calm",
            health_stats=stats,
            active_alerts=[],
            conversation_history=[],
            language=lang
        )
        
        # Update DB with final reply
        temp_db.save_conversation_turn("test_session", ConversationTurn(
            role="assistant",
            content=reply,
            tone_mode="calm"
        ))
        
        # 5. Verify Output
        assert reply == "I'm glad to hear you got good sleep and are feeling well."
        
        # Verify event bus emits
        await event_bus.emit("tts.completed", {"text": reply, "language": lang})
        
        print("\n[E2E Daily Check-in Success]")
        print(f"User: {stt_text}")
        print(f"Extracted Signals: {signals}")
        print(f"Aegis: {reply}")
