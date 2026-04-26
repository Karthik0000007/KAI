"""
Custom Hypothesis strategies for Aegis domain objects.

This module provides reusable Hypothesis strategies for generating valid
instances of Aegis domain objects for property-based testing.

Strategies are provided for:
- HealthCheckIn: Health check-in records
- EmotionResult: Emotion analysis results
- ProactiveAlert: Proactive health alerts
- VitalRecord: Vital sign measurements
- MedicationReminder: Medication reminders
- ConversationTurn: Conversation turns
- Session: Conversation sessions

Usage:
    from hypothesis import given
    from tests.hypothesis_strategies import health_checkins, emotion_results
    
    @given(health_checkins())
    def test_health_checkin_serialization(checkin):
        # Test implementation
        pass
"""

from datetime import datetime, timedelta
from typing import Optional
import uuid

from hypothesis import strategies as st
from hypothesis.strategies import SearchStrategy

from core.models import (
    HealthCheckIn,
    EmotionResult,
    ProactiveAlert,
    VitalRecord,
    MedicationReminder,
    ConversationTurn,
    Session,
)


# ─── Primitive Strategies ───────────────────────────────────────────────────

def valid_timestamps() -> SearchStrategy[str]:
    """Generate valid ISO 8601 timestamp strings."""
    # Generate timestamps within the last year
    now = datetime.now()
    one_year_ago = now - timedelta(days=365)
    
    return st.datetimes(
        min_value=one_year_ago,
        max_value=now,
    ).map(lambda dt: dt.isoformat())


def valid_ids() -> SearchStrategy[str]:
    """Generate valid ID strings (12-character hex)."""
    return st.text(
        alphabet='0123456789abcdef',
        min_size=12,
        max_size=12,
    )


def optional_text(min_size: int = 0, max_size: int = 500) -> SearchStrategy[Optional[str]]:
    """Generate optional text strings."""
    return st.none() | st.text(
        min_size=min_size,
        max_size=max_size,
        alphabet=st.characters(
            whitelist_categories=('Lu', 'Ll', 'Nd', 'P', 'Z'),
        ),
    )


# ─── EmotionResult Strategies ───────────────────────────────────────────────

def emotion_labels() -> SearchStrategy[str]:
    """Generate valid emotion labels."""
    return st.sampled_from([
        'calm',
        'stressed',
        'anxious',
        'fatigued',
        'neutral',
        'happy',
        'sad',
        'angry',
    ])


def emotion_results(
    include_mixed: bool = True,
    include_secondary: bool = True,
) -> SearchStrategy[EmotionResult]:
    """
    Generate valid EmotionResult instances.
    
    Args:
        include_mixed: Whether to generate mixed emotions
        include_secondary: Whether to include secondary emotion labels
    
    Returns:
        Strategy for generating EmotionResult instances
    """
    # Base emotion result without mixed emotions
    base_strategy = st.builds(
        EmotionResult,
        label=emotion_labels(),
        confidence=st.floats(min_value=0.0, max_value=1.0),
        pitch_mean=st.floats(min_value=50.0, max_value=400.0),
        pitch_std=st.floats(min_value=0.0, max_value=100.0),
        energy_rms=st.floats(min_value=0.0, max_value=1.0),
        speech_rate=st.floats(min_value=0.5, max_value=10.0),
        secondary_label=st.none(),
        secondary_confidence=st.none(),
        is_mixed=st.just(False),
        timestamp=valid_timestamps(),
    )
    
    if not include_mixed:
        return base_strategy
    
    # Mixed emotion result with secondary emotion
    mixed_strategy = st.builds(
        EmotionResult,
        label=emotion_labels(),
        confidence=st.floats(min_value=0.4, max_value=1.0),
        pitch_mean=st.floats(min_value=50.0, max_value=400.0),
        pitch_std=st.floats(min_value=0.0, max_value=100.0),
        energy_rms=st.floats(min_value=0.0, max_value=1.0),
        speech_rate=st.floats(min_value=0.5, max_value=10.0),
        secondary_label=emotion_labels() if include_secondary else st.none(),
        secondary_confidence=st.floats(min_value=0.2, max_value=0.8) if include_secondary else st.none(),
        is_mixed=st.just(True),
        timestamp=valid_timestamps(),
    )
    
    return st.one_of(base_strategy, mixed_strategy)


# ─── HealthCheckIn Strategies ───────────────────────────────────────────────

def health_checkins(
    require_all_fields: bool = False,
) -> SearchStrategy[HealthCheckIn]:
    """
    Generate valid HealthCheckIn instances.
    
    Args:
        require_all_fields: If True, all optional fields will be populated
    
    Returns:
        Strategy for generating HealthCheckIn instances
    """
    if require_all_fields:
        # All fields populated
        return st.builds(
            HealthCheckIn,
            id=valid_ids(),
            timestamp=valid_timestamps(),
            mood_score=st.floats(min_value=1.0, max_value=10.0),
            sleep_hours=st.floats(min_value=0.0, max_value=24.0),
            energy_level=st.floats(min_value=1.0, max_value=10.0),
            pain_notes=st.text(min_size=1, max_size=200),
            medication_taken=st.booleans(),
            user_text=st.text(min_size=1, max_size=500),
            detected_emotion=emotion_labels(),
            emotion_confidence=st.floats(min_value=0.0, max_value=1.0),
            notes=st.text(min_size=0, max_size=500),
        )
    else:
        # Optional fields may be None
        return st.builds(
            HealthCheckIn,
            id=valid_ids(),
            timestamp=valid_timestamps(),
            mood_score=st.none() | st.floats(min_value=1.0, max_value=10.0),
            sleep_hours=st.none() | st.floats(min_value=0.0, max_value=24.0),
            energy_level=st.none() | st.floats(min_value=1.0, max_value=10.0),
            pain_notes=optional_text(max_size=200),
            medication_taken=st.none() | st.booleans(),
            user_text=optional_text(max_size=500),
            detected_emotion=st.none() | emotion_labels(),
            emotion_confidence=st.none() | st.floats(min_value=0.0, max_value=1.0),
            notes=optional_text(max_size=500),
        )


# ─── ProactiveAlert Strategies ──────────────────────────────────────────────

def alert_types() -> SearchStrategy[str]:
    """Generate valid alert types."""
    return st.sampled_from([
        'low_mood',
        'sleep_deficit',
        'medication_missed',
        'elevated_heart_rate',
        'low_energy',
        'pain_trend',
        'sleep_pattern_disruption',
        'activity_level_change',
    ])


def alert_severities() -> SearchStrategy[str]:
    """Generate valid alert severity levels."""
    return st.sampled_from(['info', 'warning', 'urgent'])


def proactive_alerts() -> SearchStrategy[ProactiveAlert]:
    """
    Generate valid ProactiveAlert instances.
    
    Returns:
        Strategy for generating ProactiveAlert instances
    """
    return st.builds(
        ProactiveAlert,
        id=valid_ids(),
        timestamp=valid_timestamps(),
        alert_type=alert_types(),
        severity=alert_severities(),
        message=st.text(min_size=10, max_size=200),
        explanation=st.text(min_size=20, max_size=500),
        acknowledged=st.booleans(),
        context=st.dictionaries(
            keys=st.text(min_size=1, max_size=50, alphabet=st.characters(
                whitelist_categories=('Lu', 'Ll', 'Nd'),
                whitelist_characters='_',
            )),
            values=st.one_of(
                st.integers(),
                st.floats(allow_nan=False, allow_infinity=False),
                st.text(max_size=100),
                st.booleans(),
            ),
            min_size=0,
            max_size=10,
        ),
    )


# ─── VitalRecord Strategies ─────────────────────────────────────────────────

def vital_records(
    require_all_vitals: bool = False,
) -> SearchStrategy[VitalRecord]:
    """
    Generate valid VitalRecord instances.
    
    Args:
        require_all_vitals: If True, all vital fields will be populated
    
    Returns:
        Strategy for generating VitalRecord instances
    """
    if require_all_vitals:
        # All vitals populated
        return st.builds(
            VitalRecord,
            id=valid_ids(),
            timestamp=valid_timestamps(),
            heart_rate=st.integers(min_value=30, max_value=220),
            blood_pressure_sys=st.integers(min_value=70, max_value=200),
            blood_pressure_dia=st.integers(min_value=40, max_value=130),
            spo2=st.floats(min_value=70.0, max_value=100.0),
            temperature=st.floats(min_value=35.0, max_value=42.0),
            steps=st.integers(min_value=0, max_value=100000),
        )
    else:
        # Optional vitals may be None
        return st.builds(
            VitalRecord,
            id=valid_ids(),
            timestamp=valid_timestamps(),
            heart_rate=st.none() | st.integers(min_value=30, max_value=220),
            blood_pressure_sys=st.none() | st.integers(min_value=70, max_value=200),
            blood_pressure_dia=st.none() | st.integers(min_value=40, max_value=130),
            spo2=st.none() | st.floats(min_value=70.0, max_value=100.0),
            temperature=st.none() | st.floats(min_value=35.0, max_value=42.0),
            steps=st.none() | st.integers(min_value=0, max_value=100000),
        )


# ─── MedicationReminder Strategies ──────────────────────────────────────────

def medication_reminders() -> SearchStrategy[MedicationReminder]:
    """
    Generate valid MedicationReminder instances.
    
    Returns:
        Strategy for generating MedicationReminder instances
    """
    # Generate valid time strings in HH:MM format
    time_strategy = st.builds(
        lambda h, m: f"{h:02d}:{m:02d}",
        h=st.integers(min_value=0, max_value=23),
        m=st.integers(min_value=0, max_value=59),
    )
    
    return st.builds(
        MedicationReminder,
        id=valid_ids(),
        name=st.text(min_size=1, max_size=100),
        dosage=st.text(min_size=1, max_size=50),
        schedule_time=time_strategy,
        active=st.booleans(),
        created_at=valid_timestamps(),
    )


# ─── ConversationTurn Strategies ────────────────────────────────────────────

def conversation_roles() -> SearchStrategy[str]:
    """Generate valid conversation roles."""
    return st.sampled_from(['user', 'assistant'])


def conversation_turns() -> SearchStrategy[ConversationTurn]:
    """
    Generate valid ConversationTurn instances.
    
    Returns:
        Strategy for generating ConversationTurn instances
    """
    return st.builds(
        ConversationTurn,
        role=conversation_roles(),
        content=st.text(min_size=1, max_size=500),
        timestamp=valid_timestamps(),
        emotion=st.none() | emotion_labels(),
        tone_mode=st.none() | st.sampled_from(['empathetic', 'informative', 'encouraging']),
    )


# ─── Session Strategies ─────────────────────────────────────────────────────

def sessions(
    min_turns: int = 0,
    max_turns: int = 20,
    min_emotions: int = 0,
    max_emotions: int = 10,
) -> SearchStrategy[Session]:
    """
    Generate valid Session instances.
    
    Args:
        min_turns: Minimum number of conversation turns
        max_turns: Maximum number of conversation turns
        min_emotions: Minimum number of emotion results
        max_emotions: Maximum number of emotion results
    
    Returns:
        Strategy for generating Session instances
    """
    return st.builds(
        Session,
        id=valid_ids(),
        started_at=valid_timestamps(),
        turns=st.lists(
            conversation_turns(),
            min_size=min_turns,
            max_size=max_turns,
        ),
        emotion_history=st.lists(
            emotion_results(),
            min_size=min_emotions,
            max_size=max_emotions,
        ),
        active=st.booleans(),
    )


# ─── Composite Strategies ───────────────────────────────────────────────────

def complete_health_records() -> SearchStrategy[dict]:
    """
    Generate complete health records with all related data.
    
    Returns a dictionary containing:
    - checkin: HealthCheckIn instance
    - emotion: EmotionResult instance
    - vitals: VitalRecord instance (optional)
    - alerts: List of ProactiveAlert instances
    
    Returns:
        Strategy for generating complete health records
    """
    return st.fixed_dictionaries({
        'checkin': health_checkins(require_all_fields=True),
        'emotion': emotion_results(),
        'vitals': st.none() | vital_records(require_all_vitals=True),
        'alerts': st.lists(
            proactive_alerts(),
            min_size=0,
            max_size=5,
        ),
    })


# ─── Export All Strategies ──────────────────────────────────────────────────

__all__ = [
    # Primitive strategies
    'valid_timestamps',
    'valid_ids',
    'optional_text',
    
    # Emotion strategies
    'emotion_labels',
    'emotion_results',
    
    # Health data strategies
    'health_checkins',
    'vital_records',
    'medication_reminders',
    
    # Alert strategies
    'alert_types',
    'alert_severities',
    'proactive_alerts',
    
    # Conversation strategies
    'conversation_roles',
    'conversation_turns',
    'sessions',
    
    # Composite strategies
    'complete_health_records',
]
