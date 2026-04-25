"""
Tests for alert prioritization and deduplication (Task 8.2)

Tests the three new methods:
- _prioritize_alerts()
- _deduplicate_alerts()
- _limit_alerts_per_day()

Requirements: 10.7, 10.8
"""

import pytest
import tempfile
from datetime import datetime, timedelta

from core.proactive import ProactiveEngine
from core.health_db import HealthDatabase
from core.models import ProactiveAlert


@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db = HealthDatabase(db_path=f.name)
        yield db
        db.close()


@pytest.fixture
def proactive_engine(temp_db):
    """Create a proactive engine with temporary database."""
    return ProactiveEngine(db=temp_db)


class TestAlertPrioritization:
    """Test _prioritize_alerts() method."""
    
    def test_prioritizes_by_severity(self, proactive_engine):
        """Test that alerts are prioritized by severity (urgent > warning > info)."""
        # Create alerts with different severities
        alerts = [
            ProactiveAlert(
                alert_type="test_info",
                severity="info",
                message="Info alert"
            ),
            ProactiveAlert(
                alert_type="test_urgent",
                severity="urgent",
                message="Urgent alert"
            ),
            ProactiveAlert(
                alert_type="test_warning",
                severity="warning",
                message="Warning alert"
            ),
        ]
        
        # Prioritize
        prioritized = proactive_engine._prioritize_alerts(alerts)
        
        # Should be ordered: urgent, warning, info
        assert len(prioritized) == 3
        assert prioritized[0].severity == "urgent"
        assert prioritized[1].severity == "warning"
        assert prioritized[2].severity == "info"
    
    def test_prioritizes_by_recency_within_same_severity(self, proactive_engine):
        """Test that within same severity, more recent alerts are prioritized."""
        now = datetime.now()
        
        # Create alerts with same severity but different timestamps
        alerts = [
            ProactiveAlert(
                alert_type="test_old",
                severity="warning",
                message="Old alert",
                timestamp=(now - timedelta(hours=12)).isoformat()
            ),
            ProactiveAlert(
                alert_type="test_new",
                severity="warning",
                message="New alert",
                timestamp=now.isoformat()
            ),
            ProactiveAlert(
                alert_type="test_medium",
                severity="warning",
                message="Medium alert",
                timestamp=(now - timedelta(hours=6)).isoformat()
            ),
        ]
        
        # Prioritize
        prioritized = proactive_engine._prioritize_alerts(alerts)
        
        # Should be ordered by recency (newest first)
        assert len(prioritized) == 3
        assert prioritized[0].alert_type == "test_new"
        assert prioritized[1].alert_type == "test_medium"
        assert prioritized[2].alert_type == "test_old"
    
    def test_combined_severity_and_recency(self, proactive_engine):
        """Test prioritization with both severity and recency factors."""
        now = datetime.now()
        
        # Create alerts with mixed severity and timestamps
        alerts = [
            ProactiveAlert(
                alert_type="old_urgent",
                severity="urgent",
                message="Old urgent",
                timestamp=(now - timedelta(hours=20)).isoformat()
            ),
            ProactiveAlert(
                alert_type="new_info",
                severity="info",
                message="New info",
                timestamp=now.isoformat()
            ),
            ProactiveAlert(
                alert_type="new_warning",
                severity="warning",
                message="New warning",
                timestamp=now.isoformat()
            ),
        ]
        
        # Prioritize
        prioritized = proactive_engine._prioritize_alerts(alerts)
        
        # Urgent should come first despite being older
        # Then warning, then info
        assert len(prioritized) == 3
        assert prioritized[0].alert_type == "old_urgent"
        assert prioritized[1].alert_type == "new_warning"
        assert prioritized[2].alert_type == "new_info"
    
    def test_empty_list(self, proactive_engine):
        """Test prioritization with empty list."""
        prioritized = proactive_engine._prioritize_alerts([])
        assert prioritized == []


class TestAlertDeduplication:
    """Test _deduplicate_alerts() method."""
    
    def test_deduplicates_recent_alerts(self, temp_db, proactive_engine):
        """Test that alerts of same type within 24 hours are deduplicated."""
        # Save an alert to the database (simulating a recent alert)
        recent_alert = ProactiveAlert(
            alert_type="low_mood_pattern",
            severity="warning",
            message="Recent low mood alert",
            timestamp=(datetime.now() - timedelta(hours=12)).isoformat()
        )
        temp_db.save_alert(recent_alert)
        
        # Create new alerts including a duplicate type
        new_alerts = [
            ProactiveAlert(
                alert_type="low_mood_pattern",
                severity="warning",
                message="Duplicate low mood alert"
            ),
            ProactiveAlert(
                alert_type="sleep_deficit",
                severity="warning",
                message="New sleep alert"
            ),
        ]
        
        # Deduplicate
        deduplicated = proactive_engine._deduplicate_alerts(new_alerts)
        
        # Should only have the sleep_deficit alert (low_mood_pattern is duplicate)
        assert len(deduplicated) == 1
        assert deduplicated[0].alert_type == "sleep_deficit"
    
    def test_allows_alerts_after_24_hours(self, temp_db, proactive_engine):
        """Test that alerts of same type after 24 hours are allowed."""
        # Save an old alert (more than 24 hours ago)
        old_alert = ProactiveAlert(
            alert_type="low_mood_pattern",
            severity="warning",
            message="Old low mood alert",
            timestamp=(datetime.now() - timedelta(hours=25)).isoformat()
        )
        temp_db.save_alert(old_alert)
        
        # Create new alert of same type
        new_alerts = [
            ProactiveAlert(
                alert_type="low_mood_pattern",
                severity="warning",
                message="New low mood alert"
            ),
        ]
        
        # Deduplicate
        deduplicated = proactive_engine._deduplicate_alerts(new_alerts)
        
        # Should allow the new alert (old one is beyond 24 hour window)
        assert len(deduplicated) == 1
        assert deduplicated[0].alert_type == "low_mood_pattern"
    
    def test_no_deduplication_for_different_types(self, temp_db, proactive_engine):
        """Test that different alert types are not deduplicated."""
        # Save alerts of different types
        alert1 = ProactiveAlert(
            alert_type="low_mood_pattern",
            severity="warning",
            message="Low mood alert"
        )
        temp_db.save_alert(alert1)
        
        # Create new alerts of different types
        new_alerts = [
            ProactiveAlert(
                alert_type="sleep_deficit",
                severity="warning",
                message="Sleep alert"
            ),
            ProactiveAlert(
                alert_type="energy_decline",
                severity="info",
                message="Energy alert"
            ),
        ]
        
        # Deduplicate
        deduplicated = proactive_engine._deduplicate_alerts(new_alerts)
        
        # Should keep all alerts (different types)
        assert len(deduplicated) == 2
    
    def test_empty_list(self, proactive_engine):
        """Test deduplication with empty list."""
        deduplicated = proactive_engine._deduplicate_alerts([])
        assert deduplicated == []


class TestAlertLimiting:
    """Test _limit_alerts_per_day() method."""
    
    def test_limits_to_3_alerts_per_day(self, temp_db, proactive_engine):
        """Test that alerts are limited to 3 per day."""
        # Create 5 alerts
        alerts = [
            ProactiveAlert(
                alert_type=f"test_alert_{i}",
                severity="warning",
                message=f"Alert {i}"
            )
            for i in range(5)
        ]
        
        # Limit to 3
        limited = proactive_engine._limit_alerts_per_day(alerts, max_alerts=3)
        
        # Should only have 3 alerts
        assert len(limited) == 3
    
    def test_respects_existing_alerts_today(self, temp_db, proactive_engine):
        """Test that existing alerts today are counted toward the limit."""
        # Save 2 alerts today
        for i in range(2):
            alert = ProactiveAlert(
                alert_type=f"existing_alert_{i}",
                severity="info",
                message=f"Existing alert {i}"
            )
            temp_db.save_alert(alert)
        
        # Create 3 new alerts
        new_alerts = [
            ProactiveAlert(
                alert_type=f"new_alert_{i}",
                severity="warning",
                message=f"New alert {i}"
            )
            for i in range(3)
        ]
        
        # Limit to 3 total per day
        limited = proactive_engine._limit_alerts_per_day(new_alerts, max_alerts=3)
        
        # Should only allow 1 more alert (2 existing + 1 new = 3 total)
        assert len(limited) == 1
    
    def test_blocks_all_when_limit_reached(self, temp_db, proactive_engine):
        """Test that all alerts are blocked when daily limit is reached."""
        # Save 3 alerts today (reaching the limit)
        for i in range(3):
            alert = ProactiveAlert(
                alert_type=f"existing_alert_{i}",
                severity="info",
                message=f"Existing alert {i}"
            )
            temp_db.save_alert(alert)
        
        # Create new alerts
        new_alerts = [
            ProactiveAlert(
                alert_type="new_alert",
                severity="urgent",
                message="New urgent alert"
            ),
        ]
        
        # Limit to 3 per day
        limited = proactive_engine._limit_alerts_per_day(new_alerts, max_alerts=3)
        
        # Should block all new alerts
        assert len(limited) == 0
    
    def test_allows_all_when_under_limit(self, temp_db, proactive_engine):
        """Test that all alerts are allowed when under the daily limit."""
        # Save 1 alert today
        alert = ProactiveAlert(
            alert_type="existing_alert",
            severity="info",
            message="Existing alert"
        )
        temp_db.save_alert(alert)
        
        # Create 2 new alerts
        new_alerts = [
            ProactiveAlert(
                alert_type=f"new_alert_{i}",
                severity="warning",
                message=f"New alert {i}"
            )
            for i in range(2)
        ]
        
        # Limit to 3 per day
        limited = proactive_engine._limit_alerts_per_day(new_alerts, max_alerts=3)
        
        # Should allow both new alerts (1 existing + 2 new = 3 total)
        assert len(limited) == 2
    
    def test_empty_list(self, proactive_engine):
        """Test limiting with empty list."""
        limited = proactive_engine._limit_alerts_per_day([], max_alerts=3)
        assert limited == []
    
    def test_custom_max_alerts(self, temp_db, proactive_engine):
        """Test limiting with custom max_alerts parameter."""
        # Create 10 alerts
        alerts = [
            ProactiveAlert(
                alert_type=f"test_alert_{i}",
                severity="warning",
                message=f"Alert {i}"
            )
            for i in range(10)
        ]
        
        # Limit to 5
        limited = proactive_engine._limit_alerts_per_day(alerts, max_alerts=5)
        
        # Should only have 5 alerts
        assert len(limited) == 5


class TestIntegration:
    """Integration tests for the complete prioritization and deduplication flow."""
    
    def test_full_pipeline(self, temp_db, proactive_engine):
        """Test the complete pipeline: deduplicate -> prioritize -> limit."""
        # Save a recent alert to trigger deduplication
        # Use 12 hours ago - within 24h window for deduplication
        # But we need to ensure it's from yesterday for the daily limit test
        # Solution: Use yesterday at 23:00 if current time is before 23:00,
        # otherwise use today at 01:00
        now = datetime.now()
        if now.hour < 23:
            # Use yesterday at 23:00 (within 24h, but yesterday)
            yesterday_23 = now.replace(hour=23, minute=0, second=0, microsecond=0) - timedelta(days=1)
            recent_timestamp = yesterday_23
        else:
            # Use today at 01:00 (within 24h, but early today so we can add 2 more alerts)
            # Actually, this won't work for the daily limit test
            # Let's just use 12 hours ago and adjust the test expectations
            recent_timestamp = now - timedelta(hours=12)
        
        recent_alert = ProactiveAlert(
            alert_type="low_mood_pattern",
            severity="warning",
            message="Recent alert",
            timestamp=recent_timestamp.isoformat()
        )
        temp_db.save_alert(recent_alert)
        
        # Save 1 more alert today to test limiting (total 2 including recent_alert if it's today)
        # Check if recent_alert is today
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        alerts_today_count = 1 if recent_timestamp >= today_start else 0
        
        # Add enough alerts to reach 2 total today
        for i in range(2 - alerts_today_count):
            alert = ProactiveAlert(
                alert_type=f"today_alert_{i}",
                severity="info",
                message=f"Today alert {i}"
            )
            temp_db.save_alert(alert)
        
        # Create a mix of new alerts
        new_alerts = [
            ProactiveAlert(
                alert_type="low_mood_pattern",  # Duplicate
                severity="warning",
                message="Duplicate mood alert"
            ),
            ProactiveAlert(
                alert_type="urgent_alert",
                severity="urgent",
                message="Urgent alert"
            ),
            ProactiveAlert(
                alert_type="info_alert",
                severity="info",
                message="Info alert"
            ),
            ProactiveAlert(
                alert_type="warning_alert",
                severity="warning",
                message="Warning alert"
            ),
        ]
        
        # Apply deduplication
        deduplicated = proactive_engine._deduplicate_alerts(new_alerts)
        
        # Should remove the duplicate low_mood_pattern
        assert len(deduplicated) == 3
        assert not any(a.alert_type == "low_mood_pattern" for a in deduplicated)
        
        # Apply prioritization
        prioritized = proactive_engine._prioritize_alerts(deduplicated)
        
        # Should be ordered: urgent, warning, info
        assert prioritized[0].severity == "urgent"
        assert prioritized[1].severity == "warning"
        assert prioritized[2].severity == "info"
        
        # Apply limiting (2 already today, max 3, so only 1 more allowed)
        limited = proactive_engine._limit_alerts_per_day(prioritized, max_alerts=3)
        
        # Should only have 1 alert (the highest priority one)
        assert len(limited) == 1
        assert limited[0].severity == "urgent"
    
    def test_run_analysis_applies_all_filters(self, temp_db, proactive_engine):
        """Test that run_analysis() applies all filters correctly."""
        # This is a smoke test to ensure the integration works
        # We won't create specific conditions to trigger alerts,
        # just verify the method runs without errors
        
        alerts = proactive_engine.run_analysis()
        
        # Should return a list (may be empty)
        assert isinstance(alerts, list)
        
        # All returned alerts should be ProactiveAlert instances
        for alert in alerts:
            assert isinstance(alert, ProactiveAlert)
