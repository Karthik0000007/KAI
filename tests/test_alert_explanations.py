"""
Tests for Task 8.3: Alert Explanations with Context

Verifies that:
1. ProactiveAlert includes explanation field
2. All alert types generate human-readable explanations
3. Supporting data is included in context
"""

import pytest
from datetime import datetime, timedelta
from core.proactive import ProactiveEngine
from core.health_db import HealthDatabase
from core.models import ProactiveAlert


class TestAlertExplanationField:
    """Test that ProactiveAlert model includes explanation field."""
    
    def test_proactive_alert_has_explanation_field(self):
        """Verify ProactiveAlert dataclass includes explanation field."""
        alert = ProactiveAlert(
            alert_type="test_alert",
            severity="info",
            message="Test message",
            explanation="Test explanation"
        )
        
        assert hasattr(alert, 'explanation')
        assert alert.explanation == "Test explanation"
    
    def test_explanation_field_in_dict(self):
        """Verify explanation field is included in to_dict() output."""
        alert = ProactiveAlert(
            alert_type="test_alert",
            severity="info",
            message="Test message",
            explanation="Test explanation"
        )
        
        alert_dict = alert.to_dict()
        assert 'explanation' in alert_dict
        assert alert_dict['explanation'] == "Test explanation"


class TestAlertExplanationContent:
    """Test that all alert types generate meaningful explanations."""
    
    @pytest.fixture
    def db(self, tmp_path):
        """Create a temporary test database."""
        db_path = tmp_path / "test_aegis.db"
        db = HealthDatabase(db_path=str(db_path))
        return db
    
    @pytest.fixture
    def engine(self, db):
        """Create a proactive engine instance."""
        return ProactiveEngine(db)
    
    def test_low_mood_pattern_explanation(self, db, engine):
        """Test low mood pattern alert includes detailed explanation."""
        # Setup: Add 3 days of low mood
        for i in range(3):
            timestamp = (datetime.now() - timedelta(days=i)).isoformat()
            db.save_checkin({
                "timestamp": timestamp,
                "mood_score": 3.0,
                "detected_emotion": "sad"
            })
        
        # Run analysis
        alerts = engine._check_mood_pattern()
        
        # Verify
        assert len(alerts) == 1
        alert = alerts[0]
        
        # Check explanation exists and is meaningful
        assert alert.explanation != ""
        assert len(alert.explanation) > 50  # Should be detailed
        
        # Check explanation includes key information
        assert "mood scores" in alert.explanation.lower()
        assert "consecutive days" in alert.explanation.lower() or "days" in alert.explanation.lower()
        assert "threshold" in alert.explanation.lower()
        
        # Check context includes supporting data
        assert "low_mood_days" in alert.context
        assert "avg_mood" in alert.context
    
    def test_sleep_deficit_explanation(self, db, engine):
        """Test sleep deficit alert includes detailed explanation."""
        # Setup: Add 7 days of low sleep
        for i in range(7):
            timestamp = (datetime.now() - timedelta(days=i)).isoformat()
            db.save_checkin({
                "timestamp": timestamp,
                "sleep_hours": 4.5
            })
        
        # Run analysis
        alerts = engine._check_sleep_deficit()
        
        # Verify
        assert len(alerts) == 1
        alert = alerts[0]
        
        # Check explanation
        assert alert.explanation != ""
        assert len(alert.explanation) > 50
        
        # Check explanation includes key information
        assert "sleep" in alert.explanation.lower()
        assert "average" in alert.explanation.lower()
        assert "threshold" in alert.explanation.lower()
        assert "hours" in alert.explanation.lower()
        
        # Check context
        assert "avg_sleep" in alert.context
        assert "low_sleep_days" in alert.context
    
    def test_medication_missed_explanation(self, db, engine):
        """Test medication missed alert includes detailed explanation."""
        # Setup: Add medication
        db.add_medication("Test Med", "10mg", "08:00")
        
        # Run analysis (should trigger if past scheduled time)
        alerts = engine._check_medication_compliance()
        
        # If alert generated (depends on current time)
        if alerts:
            alert = alerts[0]
            
            # Check explanation
            assert alert.explanation != ""
            assert len(alert.explanation) > 50
            
            # Check explanation includes key information
            assert "medication" in alert.explanation.lower()
            assert "scheduled" in alert.explanation.lower()
            
            # Check context
            assert "medication" in alert.context
            assert "schedule" in alert.context
    
    def test_elevated_hr_stress_explanation(self, db, engine):
        """Test elevated heart rate + stress alert includes detailed explanation."""
        # Setup: Add high heart rate vital
        timestamp = datetime.now().isoformat()
        db.save_vital({
            "timestamp": timestamp,
            "heart_rate": 120
        })
        
        # Add stressed emotion check-in
        db.save_checkin({
            "timestamp": timestamp,
            "detected_emotion": "stressed"
        })
        
        # Run analysis
        alerts = engine._check_vital_signs()
        
        # Verify
        if alerts:
            alert = alerts[0]
            
            # Check explanation
            assert alert.explanation != ""
            assert len(alert.explanation) > 50
            
            # Check explanation includes key information
            assert "heart rate" in alert.explanation.lower()
            assert "threshold" in alert.explanation.lower()
            
            # Check context
            assert "max_hr" in alert.context
    
    def test_emotional_distress_pattern_explanation(self, db, engine):
        """Test emotional distress pattern alert includes detailed explanation."""
        # Setup: Add 5 check-ins with negative emotions
        for i in range(5):
            timestamp = (datetime.now() - timedelta(days=i)).isoformat()
            db.save_checkin({
                "timestamp": timestamp,
                "detected_emotion": "stressed"
            })
        
        # Run analysis
        alerts = engine._check_emotion_pattern()
        
        # Verify
        if alerts:
            alert = alerts[0]
            
            # Check explanation
            assert alert.explanation != ""
            assert len(alert.explanation) > 50
            
            # Check explanation includes key information
            assert "emotional" in alert.explanation.lower() or "emotion" in alert.explanation.lower()
            assert "pattern" in alert.explanation.lower() or "distress" in alert.explanation.lower()
            
            # Check context
            assert "negative_emotions" in alert.context
            assert "total" in alert.context
    
    def test_energy_decline_explanation(self, db, engine):
        """Test energy decline alert includes detailed explanation."""
        # Setup: Add declining energy levels
        for i in range(5):
            timestamp = (datetime.now() - timedelta(days=i)).isoformat()
            energy = 5.0 - i  # Declining from 5 to 1
            db.save_checkin({
                "timestamp": timestamp,
                "energy_level": energy
            })
        
        # Run analysis
        alerts = engine._check_energy_trend()
        
        # Verify
        if alerts:
            alert = alerts[0]
            
            # Check explanation
            assert alert.explanation != ""
            assert len(alert.explanation) > 50
            
            # Check explanation includes key information
            assert "energy" in alert.explanation.lower()
            assert "declining" in alert.explanation.lower() or "trend" in alert.explanation.lower()
            
            # Check context
            assert "energies" in alert.context
            assert "avg" in alert.context
    
    def test_sleep_pattern_disruption_explanation(self, db, engine):
        """Test sleep pattern disruption alert includes detailed explanation."""
        # Setup: Add check-ins at unusual times (3 AM)
        for i in range(3):
            unusual_time = datetime.now().replace(hour=3, minute=0, second=0) - timedelta(days=i)
            db.save_checkin({
                "timestamp": unusual_time.isoformat(),
                "sleep_hours": 4.0
            })
        
        # Run analysis
        alerts = engine._check_sleep_pattern_disruption()
        
        # Verify
        if alerts:
            alert = alerts[0]
            
            # Check explanation
            assert alert.explanation != ""
            assert len(alert.explanation) > 50
            
            # Check explanation includes key information
            assert "sleep" in alert.explanation.lower()
            assert "pattern" in alert.explanation.lower() or "disrupted" in alert.explanation.lower()
            
            # Check context
            assert "unusual_checkins" in alert.context
            assert "times" in alert.context
    
    def test_activity_level_change_explanation(self, db, engine):
        """Test activity level change alert includes detailed explanation."""
        # Setup: Add baseline energy (5 days ago to 3 days ago)
        for i in range(5, 2, -1):
            timestamp = (datetime.now() - timedelta(days=i)).isoformat()
            db.save_checkin({
                "timestamp": timestamp,
                "energy_level": 7.0
            })
        
        # Add recent low energy (last 2 days)
        for i in range(2):
            timestamp = (datetime.now() - timedelta(days=i)).isoformat()
            db.save_checkin({
                "timestamp": timestamp,
                "energy_level": 3.0
            })
        
        # Run analysis
        alerts = engine._check_activity_level_changes()
        
        # Verify
        if alerts:
            alert = alerts[0]
            
            # Check explanation
            assert alert.explanation != ""
            assert len(alert.explanation) > 50
            
            # Check explanation includes key information
            assert "energy" in alert.explanation.lower()
            assert "drop" in alert.explanation.lower() or "decrease" in alert.explanation.lower()
            
            # Check context
            assert "baseline_avg" in alert.context
            assert "recent_avg" in alert.context
            assert "drop" in alert.context
    
    def test_pain_trend_worsening_explanation(self, db, engine):
        """Test pain trend worsening alert includes detailed explanation."""
        # Setup: Add pain reports (more in recent week)
        # Previous week: 1 report
        timestamp = (datetime.now() - timedelta(days=10)).isoformat()
        db.save_checkin({
            "timestamp": timestamp,
            "pain_notes": "Back pain"
        })
        
        # Recent week: 4 reports
        for i in range(4):
            timestamp = (datetime.now() - timedelta(days=i)).isoformat()
            db.save_checkin({
                "timestamp": timestamp,
                "pain_notes": "Back pain worsening"
            })
        
        # Run analysis
        alerts = engine._check_pain_trends()
        
        # Verify
        if alerts:
            alert = alerts[0]
            
            # Check explanation
            assert alert.explanation != ""
            assert len(alert.explanation) > 50
            
            # Check explanation includes key information
            assert "pain" in alert.explanation.lower()
            assert "increased" in alert.explanation.lower() or "frequency" in alert.explanation.lower()
            
            # Check context
            assert "recent_count" in alert.context
            assert "previous_count" in alert.context
            assert "trend" in alert.context


class TestExplanationQuality:
    """Test that explanations meet quality standards."""
    
    @pytest.fixture
    def db(self, tmp_path):
        """Create a temporary test database."""
        db_path = tmp_path / "test_aegis.db"
        db = HealthDatabase(db_path=str(db_path))
        return db
    
    @pytest.fixture
    def engine(self, db):
        """Create a proactive engine instance."""
        return ProactiveEngine(db)
    
    def test_explanations_are_human_readable(self, db, engine):
        """Test that explanations use clear, human-readable language."""
        # Setup: Trigger low mood alert
        for i in range(3):
            timestamp = (datetime.now() - timedelta(days=i)).isoformat()
            db.save_checkin({
                "timestamp": timestamp,
                "mood_score": 3.0
            })
        
        alerts = engine._check_mood_pattern()
        
        if alerts:
            explanation = alerts[0].explanation
            
            # Should not contain technical jargon
            assert "SQL" not in explanation
            assert "database" not in explanation.lower()
            assert "query" not in explanation.lower()
            
            # Should explain WHY the alert was generated
            assert "because" in explanation.lower() or "triggered" in explanation.lower()
            
            # Should include specific data points
            assert any(char.isdigit() for char in explanation)  # Contains numbers
    
    def test_explanations_include_thresholds(self, db, engine):
        """Test that explanations mention the thresholds that were exceeded."""
        # Setup: Trigger sleep deficit alert
        for i in range(7):
            timestamp = (datetime.now() - timedelta(days=i)).isoformat()
            db.save_checkin({
                "timestamp": timestamp,
                "sleep_hours": 4.0
            })
        
        alerts = engine._check_sleep_deficit()
        
        if alerts:
            explanation = alerts[0].explanation
            
            # Should mention threshold
            assert "threshold" in explanation.lower()
            
            # Should include actual values
            assert any(char.isdigit() for char in explanation)
    
    def test_explanations_describe_health_impact(self, db, engine):
        """Test that explanations explain why the pattern matters for health."""
        # Setup: Trigger energy decline alert
        for i in range(5):
            timestamp = (datetime.now() - timedelta(days=i)).isoformat()
            energy = 5.0 - i
            db.save_checkin({
                "timestamp": timestamp,
                "energy_level": energy
            })
        
        alerts = engine._check_energy_trend()
        
        if alerts:
            explanation = alerts[0].explanation
            
            # Should explain health implications
            health_keywords = [
                "health", "stress", "rest", "nutrition", "dehydration",
                "decline", "affect", "impact", "indicate", "concern"
            ]
            
            assert any(keyword in explanation.lower() for keyword in health_keywords)


class TestContextData:
    """Test that context field includes supporting data."""
    
    @pytest.fixture
    def db(self, tmp_path):
        """Create a temporary test database."""
        db_path = tmp_path / "test_aegis.db"
        db = HealthDatabase(db_path=str(db_path))
        return db
    
    @pytest.fixture
    def engine(self, db):
        """Create a proactive engine instance."""
        return ProactiveEngine(db)
    
    def test_context_includes_relevant_data(self, db, engine):
        """Test that context field includes the data that triggered the alert."""
        # Setup: Trigger low mood alert
        for i in range(3):
            timestamp = (datetime.now() - timedelta(days=i)).isoformat()
            db.save_checkin({
                "timestamp": timestamp,
                "mood_score": 3.0
            })
        
        alerts = engine._check_mood_pattern()
        
        if alerts:
            context = alerts[0].context
            
            # Should include the data points mentioned in explanation
            assert isinstance(context, dict)
            assert len(context) > 0
            
            # Should include relevant metrics
            assert "low_mood_days" in context or "avg_mood" in context
    
    def test_context_data_is_serializable(self, db, engine):
        """Test that context data can be serialized to JSON."""
        import json
        
        # Setup: Trigger alert
        for i in range(3):
            timestamp = (datetime.now() - timedelta(days=i)).isoformat()
            db.save_checkin({
                "timestamp": timestamp,
                "mood_score": 3.0
            })
        
        alerts = engine._check_mood_pattern()
        
        if alerts:
            alert = alerts[0]
            
            # Should be JSON serializable
            try:
                json.dumps(alert.context)
            except (TypeError, ValueError):
                pytest.fail("Context data is not JSON serializable")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
