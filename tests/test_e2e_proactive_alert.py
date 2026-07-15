"""
E2E Test: Proactive Alert Scenario
Simulates the system identifying a health alert and emitting a notification.
"""

import pytest
import asyncio
from unittest.mock import patch, Mock
import json
from datetime import datetime, timedelta

from core.health_db import HealthDatabase
from core.models import ProactiveAlert, HealthCheckIn
from core.proactive import ProactiveEngine
from core.event_bus import create_aegis_event_bus

@pytest.fixture
def temp_db(tmp_path):
    """Provide a temporary database for testing."""
    db_file = tmp_path / "test_aegis.db"
    db = HealthDatabase(db_path=str(db_file))
    yield db
    db.close()

@pytest.mark.asyncio
async def test_proactive_alert_e2e(temp_db):
    """
    E2E Test: Proactive Alert.
    We add multiple check-ins with low mood.
    The proactive engine should identify this trend and emit an alert.
    """
    event_bus = create_aegis_event_bus()
    # Track emitted alerts
    emitted_alerts = []
    
    def handle_alert(alert):
        emitted_alerts.append(alert.to_dict())
        
    engine = ProactiveEngine(db=temp_db, on_alert=handle_alert)
    
    # 1. Simulate a history of low mood (e.g. 3 days in a row)
    now = datetime.now()
    for i in range(4):
        ts = (now - timedelta(days=i)).isoformat()
        temp_db.save_checkin(HealthCheckIn(
            id=f"checkin_{i}",
            timestamp=ts,
            mood_score=1.0,  # Very low mood to survive DP noise
            sleep_hours=7.0,
            energy_level=5.0,
            user_text="I'm feeling really down today.",
            detected_emotion="sad",
            emotion_confidence=0.9
        ))
    
    # 2. Run proactive engine check
    engine.run_analysis()
    
    # Yield to let async event handlers run
    await asyncio.sleep(0.1)
    
    # 3. Verify alert was generated and emitted
    assert len(emitted_alerts) == 1
    
    alert = emitted_alerts[0]
    assert alert["alert_type"] == "low_mood_pattern"
    assert alert["severity"] == "warning"
    
    print("\n[E2E Proactive Alert Success]")
    print(f"Alert generated: {alert['message']}")
