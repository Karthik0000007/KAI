"""
Tests for new proactive pattern detectors (Task 8.1)
Tests the three new pattern detection methods:
- _check_sleep_pattern_disruption()
- _check_activity_level_changes()
- _check_pain_trends()
"""

import pytest
from datetime import datetime, timedelta
from pathlib import Path
import tempfile

from core.proactive import ProactiveEngine
from core.health_db import HealthDatabase
from core.models import HealthCheckIn


@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_health.db"
        db = HealthDatabase(db_path=db_path)
        yield db
        db.close()


@pytest.fixture
def proactive_engine(temp_db):
    """Create a proactive engine with temporary database."""
    return ProactiveEngine(db=temp_db)


class TestSleepPatternDisruption:
    """Test _check_sleep_pattern_disruption() method."""
    
    def test_detects_unusual_sleep_times(self, temp_db, proactive_engine):
        """Test detection of check-ins at unusual times (2 AM - 6 AM)."""
        # Create check-ins at unusual times (3 AM and 4 AM)
        base_time = datetime.now().replace(hour=3, minute=0, second=0, microsecond=0)
        
        for i in range(3):
            checkin_time = base_time - timedelta(days=i)
            checkin = HealthCheckIn(
                timestamp=checkin_time.isoformat(),
                sleep_hours=4.0,
                mood_score=5.0
            )
            temp_db.save_checkin(checkin)
        
        # Run the check
        alerts = proactive_engine._check_sleep_pattern_disruption()
        
        # Should generate an alert
        assert len(alerts) == 1
        assert alerts[0].alert_type == "sleep_pattern_disruption"
        assert alerts[0].severity == "warning"
        assert "unusual times" in alerts[0].message.lower()
    
    def test_no_alert_for_normal_times(self, temp_db, proactive_engine):
        """Test no alert for check-ins at normal times."""
        # Create check-ins at normal times (10 AM)
        base_time = datetime.now().replace(hour=10, minute=0, second=0, microsecond=0)
        
        for i in range(5):
            checkin_time = base_time - timedelta(days=i)
            checkin = HealthCheckIn(
                timestamp=checkin_time.isoformat(),
                sleep_hours=7.0,
                mood_score=7.0
            )
            temp_db.save_checkin(checkin)
        
        # Run the check
        alerts = proactive_engine._check_sleep_pattern_disruption()
        
        # Should not generate an alert
        assert len(alerts) == 0
    
    def test_insufficient_data(self, temp_db, proactive_engine):
        """Test no alert when insufficient data (< 3 check-ins)."""
        # Create only 2 check-ins
        base_time = datetime.now().replace(hour=3, minute=0, second=0, microsecond=0)
        
        for i in range(2):
            checkin_time = base_time - timedelta(days=i)
            checkin = HealthCheckIn(
                timestamp=checkin_time.isoformat(),
                sleep_hours=5.0
            )
            temp_db.save_checkin(checkin)
        
        # Run the check
        alerts = proactive_engine._check_sleep_pattern_disruption()
        
        # Should not generate an alert
        assert len(alerts) == 0


class TestActivityLevelChanges:
    """Test _check_activity_level_changes() method."""
    
    def test_detects_sudden_energy_drop(self, temp_db, proactive_engine):
        """Test detection of sudden energy level decrease."""
        now = datetime.now()
        
        # Create baseline energy levels (days 3-7 ago, high energy)
        for i in range(3, 8):
            checkin_time = now - timedelta(days=i)
            checkin = HealthCheckIn(
                timestamp=checkin_time.isoformat(),
                energy_level=7.0,
                mood_score=7.0
            )
            temp_db.save_checkin(checkin)
        
        # Create recent low energy levels (last 2 days)
        for i in range(2):
            checkin_time = now - timedelta(days=i)
            checkin = HealthCheckIn(
                timestamp=checkin_time.isoformat(),
                energy_level=3.0,
                mood_score=5.0
            )
            temp_db.save_checkin(checkin)
        
        # Run the check
        alerts = proactive_engine._check_activity_level_changes()
        
        # Should generate an alert
        assert len(alerts) == 1
        assert alerts[0].alert_type == "activity_level_change"
        assert alerts[0].severity == "warning"
        assert "dropped significantly" in alerts[0].message.lower()
        assert alerts[0].context["drop"] >= 2.0
    
    def test_no_alert_for_stable_energy(self, temp_db, proactive_engine):
        """Test no alert when energy levels are stable."""
        now = datetime.now()
        
        # Create stable energy levels
        for i in range(7):
            checkin_time = now - timedelta(days=i)
            checkin = HealthCheckIn(
                timestamp=checkin_time.isoformat(),
                energy_level=6.0,
                mood_score=6.0
            )
            temp_db.save_checkin(checkin)
        
        # Run the check
        alerts = proactive_engine._check_activity_level_changes()
        
        # Should not generate an alert
        assert len(alerts) == 0
    
    def test_insufficient_data(self, temp_db, proactive_engine):
        """Test no alert when insufficient data (< 4 check-ins)."""
        now = datetime.now()
        
        # Create only 3 check-ins
        for i in range(3):
            checkin_time = now - timedelta(days=i)
            checkin = HealthCheckIn(
                timestamp=checkin_time.isoformat(),
                energy_level=5.0
            )
            temp_db.save_checkin(checkin)
        
        # Run the check
        alerts = proactive_engine._check_activity_level_changes()
        
        # Should not generate an alert
        assert len(alerts) == 0


class TestPainTrends:
    """Test _check_pain_trends() method."""
    
    def test_detects_increasing_pain_frequency(self, temp_db, proactive_engine):
        """Test detection of increasing pain report frequency."""
        now = datetime.now()
        
        # Create 2 pain reports in previous week (7-14 days ago)
        for i in range(8, 10):
            checkin_time = now - timedelta(days=i)
            checkin = HealthCheckIn(
                timestamp=checkin_time.isoformat(),
                pain_notes="Back pain",
                mood_score=6.0
            )
            temp_db.save_checkin(checkin)
        
        # Create 4 pain reports in recent week (last 7 days)
        for i in range(7):
            if i % 2 == 0:  # Every other day
                checkin_time = now - timedelta(days=i)
                checkin = HealthCheckIn(
                    timestamp=checkin_time.isoformat(),
                    pain_notes="Back pain getting worse",
                    mood_score=5.0
                )
                temp_db.save_checkin(checkin)
        
        # Run the check
        alerts = proactive_engine._check_pain_trends()
        
        # Should generate an alert
        assert len(alerts) == 1
        assert alerts[0].alert_type == "pain_trend_worsening"
        assert alerts[0].severity == "warning"
        assert "more frequently" in alerts[0].message.lower()
        assert alerts[0].context["recent_count"] >= 3
    
    def test_no_alert_for_stable_pain(self, temp_db, proactive_engine):
        """Test no alert when pain frequency is stable."""
        now = datetime.now()
        
        # Create consistent pain reports (2 per week)
        for i in [1, 4, 8, 11]:
            checkin_time = now - timedelta(days=i)
            checkin = HealthCheckIn(
                timestamp=checkin_time.isoformat(),
                pain_notes="Mild headache",
                mood_score=6.0
            )
            temp_db.save_checkin(checkin)
        
        # Run the check
        alerts = proactive_engine._check_pain_trends()
        
        # Should not generate an alert
        assert len(alerts) == 0
    
    def test_insufficient_data(self, temp_db, proactive_engine):
        """Test no alert when insufficient pain data (< 3 reports)."""
        now = datetime.now()
        
        # Create only 2 pain reports
        for i in range(2):
            checkin_time = now - timedelta(days=i)
            checkin = HealthCheckIn(
                timestamp=checkin_time.isoformat(),
                pain_notes="Minor pain"
            )
            temp_db.save_checkin(checkin)
        
        # Run the check
        alerts = proactive_engine._check_pain_trends()
        
        # Should not generate an alert
        assert len(alerts) == 0


class TestIntegration:
    """Integration tests for all new pattern detectors."""
    
    def test_run_analysis_includes_new_patterns(self, temp_db, proactive_engine):
        """Test that run_analysis() calls all new pattern detectors."""
        now = datetime.now()
        
        # Create data that should trigger all three new patterns
        
        # 1. Unusual sleep times
        for i in range(3):
            checkin_time = now.replace(hour=3, minute=0) - timedelta(days=i)
            checkin = HealthCheckIn(
                timestamp=checkin_time.isoformat(),
                sleep_hours=4.0,
                energy_level=7.0,
                mood_score=7.0
            )
            temp_db.save_checkin(checkin)
        
        # 2. Energy drop
        for i in range(3, 8):
            checkin_time = now - timedelta(days=i)
            checkin = HealthCheckIn(
                timestamp=checkin_time.isoformat(),
                energy_level=7.0,
                mood_score=7.0
            )
            temp_db.save_checkin(checkin)
        
        for i in range(2):
            checkin_time = now - timedelta(days=i, hours=12)
            checkin = HealthCheckIn(
                timestamp=checkin_time.isoformat(),
                energy_level=3.0,
                mood_score=5.0
            )
            temp_db.save_checkin(checkin)
        
        # 3. Increasing pain
        for i in range(8, 10):
            checkin_time = now - timedelta(days=i)
            checkin = HealthCheckIn(
                timestamp=checkin_time.isoformat(),
                pain_notes="Pain"
            )
            temp_db.save_checkin(checkin)
        
        for i in range(7):
            if i % 2 == 0:
                checkin_time = now - timedelta(days=i, hours=6)
                checkin = HealthCheckIn(
                    timestamp=checkin_time.isoformat(),
                    pain_notes="Pain worsening"
                )
                temp_db.save_checkin(checkin)
        
        # Run full analysis
        alerts = proactive_engine.run_analysis()
        
        # Should have alerts from new pattern detectors
        alert_types = [a.alert_type for a in alerts]
        
        # Check that at least some of the new patterns were detected
        new_pattern_types = [
            "sleep_pattern_disruption",
            "activity_level_change",
            "pain_trend_worsening"
        ]
        
        detected_new_patterns = [t for t in alert_types if t in new_pattern_types]
        assert len(detected_new_patterns) > 0, f"Expected new pattern alerts, got: {alert_types}"
