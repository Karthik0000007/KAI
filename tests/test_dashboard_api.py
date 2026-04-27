"""
Tests for Health Dashboard API.

Tests cover:
- Authentication and session management
- Health data endpoints (mood, sleep, energy)
- Emotion distribution
- Vital signs data
- Proactive alerts
- Data export (CSV)
- WebSocket live updates

Requirements: 18.2
"""

import pytest
import json
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
from fastapi.testclient import TestClient

from core.dashboard_api import (
    create_app,
    SessionManager,
    Session,
    LoginRequest,
    HealthDataPoint,
    HealthTrendResponse,
    EmotionDistribution,
    VitalSignsData,
    ProactiveAlertData,
)
from core.health_db import HealthDatabase
from core.models import HealthCheckIn


# ─── Fixtures ────────────────────────────────────────────────────────────

@pytest.fixture
def mock_db():
    """Create a mock database."""
    db = Mock(spec=HealthDatabase)
    
    # Mock get_recent_checkins
    db.get_recent_checkins.return_value = [
        {
            "id": "1",
            "timestamp": datetime.now().isoformat(),
            "mood_score": 7.0,
            "sleep_hours": 8.0,
            "energy_level": 7.0,
            "detected_emotion": "happy",
            "notes": "Feeling good"
        }
    ]
    
    # Mock get_checkin_stats
    db.get_checkin_stats.return_value = {
        "count": 1,
        "avg_mood": 7.0,
        "avg_sleep": 8.0,
        "avg_energy": 7.0,
        "low_mood_days": 0,
        "low_sleep_days": 0,
    }
    
    # Mock get_recent_vitals
    db.get_recent_vitals.return_value = [
        {
            "timestamp": datetime.now().isoformat(),
            "heart_rate": 75,
            "spo2": 98,
            "temperature": 37.0
        }
    ]
    
    # Mock get_unacknowledged_alerts
    db.get_unacknowledged_alerts.return_value = []
    
    return db


@pytest.fixture
def client(mock_db):
    """Create a test client."""
    app = create_app(mock_db)
    return TestClient(app)


@pytest.fixture
def session_manager():
    """Create a session manager."""
    return SessionManager(timeout_minutes=30)


# ─── Session Management Tests ───────────────────────────────────────────

def test_session_creation(session_manager):
    """Test creating a new session."""
    token = session_manager.create_session(user_id="test_user")
    
    assert token is not None
    assert len(token) > 0
    assert token in session_manager.sessions


def test_session_validation(session_manager):
    """Test validating a session."""
    token = session_manager.create_session()
    
    assert session_manager.validate_session(token) is True


def test_session_expiration(session_manager):
    """Test session expiration."""
    # Create session with very short timeout
    session_manager.timeout = timedelta(seconds=0)
    token = session_manager.create_session()
    
    # Session should be expired
    assert session_manager.validate_session(token) is False


def test_session_invalidation(session_manager):
    """Test invalidating a session."""
    token = session_manager.create_session()
    
    assert session_manager.invalidate_session(token) is True
    assert session_manager.validate_session(token) is False


def test_session_cleanup(session_manager):
    """Test cleaning up expired sessions."""
    # Create session with very short timeout
    session_manager.timeout = timedelta(seconds=0)
    token = session_manager.create_session()
    
    # Cleanup
    session_manager.cleanup_expired()
    
    # Session should be removed
    assert token not in session_manager.sessions


# ─── Authentication Endpoint Tests ──────────────────────────────────────

def test_login_endpoint(client):
    """Test login endpoint."""
    response = client.post(
        "/api/auth/login",
        json={"username": "test_user", "password": "password"}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "token" in data
    assert "expires_in" in data
    assert data["expires_in"] == 30 * 60


def test_logout_endpoint(client):
    """Test logout endpoint."""
    # First login
    login_response = client.post(
        "/api/auth/login",
        json={"username": "test_user", "password": "password"}
    )
    token = login_response.json()["token"]
    
    # Then logout
    logout_response = client.post(
        f"/api/auth/logout?token={token}"
    )
    
    assert logout_response.status_code == 200


# ─── Health Data Endpoint Tests ─────────────────────────────────────────

def test_get_mood_trend(client):
    """Test getting mood trend data."""
    # First login
    login_response = client.post(
        "/api/auth/login",
        json={"username": "test_user", "password": "password"}
    )
    token = login_response.json()["token"]
    
    # Get mood trend
    response = client.get(
        f"/api/health/mood?days=7&token={token}"
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "data" in data
    assert "average" in data
    assert "trend" in data


def test_get_sleep_trend(client):
    """Test getting sleep trend data."""
    # First login
    login_response = client.post(
        "/api/auth/login",
        json={"username": "test_user", "password": "password"}
    )
    token = login_response.json()["token"]
    
    # Get sleep trend
    response = client.get(
        f"/api/health/sleep?days=7&token={token}"
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "data" in data
    assert "average" in data
    assert "trend" in data


def test_get_energy_trend(client):
    """Test getting energy trend data."""
    # First login
    login_response = client.post(
        "/api/auth/login",
        json={"username": "test_user", "password": "password"}
    )
    token = login_response.json()["token"]
    
    # Get energy trend
    response = client.get(
        f"/api/health/energy?days=7&token={token}"
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "data" in data
    assert "average" in data
    assert "trend" in data


def test_get_emotion_distribution(client):
    """Test getting emotion distribution."""
    # First login
    login_response = client.post(
        "/api/auth/login",
        json={"username": "test_user", "password": "password"}
    )
    token = login_response.json()["token"]
    
    # Get emotion distribution
    response = client.get(
        f"/api/health/emotions?days=7&token={token}"
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "emotions" in data


def test_get_vital_signs(client):
    """Test getting vital signs data."""
    # First login
    login_response = client.post(
        "/api/auth/login",
        json={"username": "test_user", "password": "password"}
    )
    token = login_response.json()["token"]
    
    # Get vital signs
    response = client.get(
        f"/api/health/vitals?days=7&token={token}"
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "vitals" in data


def test_get_proactive_alerts(client):
    """Test getting proactive alerts."""
    # First login
    login_response = client.post(
        "/api/auth/login",
        json={"username": "test_user", "password": "password"}
    )
    token = login_response.json()["token"]
    
    # Get alerts
    response = client.get(
        f"/api/health/alerts?token={token}"
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "alerts" in data


def test_acknowledge_alert(client, mock_db):
    """Test acknowledging an alert."""
    # First login
    login_response = client.post(
        "/api/auth/login",
        json={"username": "test_user", "password": "password"}
    )
    token = login_response.json()["token"]
    
    # Acknowledge alert
    response = client.post(
        f"/api/health/alerts/alert_123/acknowledge?token={token}"
    )
    
    assert response.status_code == 200
    mock_db.acknowledge_alert.assert_called_once_with("alert_123")


def test_get_health_statistics(client):
    """Test getting health statistics."""
    # First login
    login_response = client.post(
        "/api/auth/login",
        json={"username": "test_user", "password": "password"}
    )
    token = login_response.json()["token"]
    
    # Get statistics
    response = client.get(
        f"/api/health/stats?days=7&token={token}"
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "period_days" in data
    assert "check_ins" in data
    assert "avg_mood" in data


def test_export_data_csv(client):
    """Test exporting data as CSV."""
    # First login
    login_response = client.post(
        "/api/auth/login",
        json={"username": "test_user", "password": "password"}
    )
    token = login_response.json()["token"]
    
    # Export CSV
    response = client.get(
        f"/api/health/export/csv?days=7&token={token}"
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "format" in data
    assert data["format"] == "csv"
    assert "data" in data
    assert "filename" in data


# ─── Authentication Tests ───────────────────────────────────────────────

def test_unauthorized_access(client):
    """Test accessing endpoint without token."""
    response = client.get("/api/health/mood?days=7")
    
    assert response.status_code == 422  # Missing required parameter


def test_invalid_token(client):
    """Test accessing endpoint with invalid token."""
    response = client.get(
        "/api/health/mood?days=7&token=invalid_token"
    )
    
    assert response.status_code == 401


# ─── API Status Tests ───────────────────────────────────────────────────

def test_api_status(client):
    """Test API status endpoint."""
    response = client.get("/api/health/status")
    
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "version" in data
    assert "timestamp" in data


# ─── Data Model Tests ───────────────────────────────────────────────────

def test_health_data_point_model():
    """Test HealthDataPoint model."""
    point = HealthDataPoint(
        timestamp="2026-04-26T10:00:00",
        value=7.5,
        label="happy"
    )
    
    assert point.timestamp == "2026-04-26T10:00:00"
    assert point.value == 7.5
    assert point.label == "happy"


def test_emotion_distribution_model():
    """Test EmotionDistribution model."""
    dist = EmotionDistribution(
        emotion="happy",
        count=10,
        percentage=50.0
    )
    
    assert dist.emotion == "happy"
    assert dist.count == 10
    assert dist.percentage == 50.0


def test_vital_signs_data_model():
    """Test VitalSignsData model."""
    vital = VitalSignsData(
        timestamp="2026-04-26T10:00:00",
        heart_rate=75.0,
        spo2=98.0,
        temperature=37.0
    )
    
    assert vital.timestamp == "2026-04-26T10:00:00"
    assert vital.heart_rate == 75.0
    assert vital.spo2 == 98.0
    assert vital.temperature == 37.0


def test_proactive_alert_data_model():
    """Test ProactiveAlertData model."""
    alert = ProactiveAlertData(
        id="alert_1",
        timestamp="2026-04-26T10:00:00",
        alert_type="low_mood",
        severity="warning",
        message="Your mood has been low",
        acknowledged=False
    )
    
    assert alert.id == "alert_1"
    assert alert.alert_type == "low_mood"
    assert alert.severity == "warning"
    assert alert.acknowledged is False


# ─── Query Parameter Tests ──────────────────────────────────────────────

def test_mood_trend_with_different_days(client):
    """Test mood trend with different day ranges."""
    # First login
    login_response = client.post(
        "/api/auth/login",
        json={"username": "test_user", "password": "password"}
    )
    token = login_response.json()["token"]
    
    # Test with 7 days
    response_7 = client.get(f"/api/health/mood?days=7&token={token}")
    assert response_7.status_code == 200
    
    # Test with 30 days
    response_30 = client.get(f"/api/health/mood?days=30&token={token}")
    assert response_30.status_code == 200
    
    # Test with 90 days
    response_90 = client.get(f"/api/health/mood?days=90&token={token}")
    assert response_90.status_code == 200


def test_invalid_day_range(client):
    """Test with invalid day range."""
    # First login
    login_response = client.post(
        "/api/auth/login",
        json={"username": "test_user", "password": "password"}
    )
    token = login_response.json()["token"]
    
    # Test with 0 days (invalid)
    response = client.get(f"/api/health/mood?days=0&token={token}")
    assert response.status_code == 422
    
    # Test with 100 days (invalid)
    response = client.get(f"/api/health/mood?days=100&token={token}")
    assert response.status_code == 422


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
