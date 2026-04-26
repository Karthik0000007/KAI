"""
Property-based tests for JSON serialization round-trip preservation.

**Validates: Requirements 22.1, 22.5, 22.6**

This module tests that JSON serialization and deserialization operations preserve
data integrity for HealthCheckIn objects across a wide range of inputs, including
edge cases like optional fields, null values, and various field combinations.

Property 4: JSON Serialization Round-Trip Preservation
- Any HealthCheckIn object can be serialized to JSON and parsed back
- The parsed object is equivalent to the original
- All field combinations work (some null, some present)
- Edge cases work (empty strings, unicode, special characters)
"""

import json
import pytest
from hypothesis import given, strategies as st, assume

from core.models import HealthCheckIn
from tests.hypothesis_strategies import health_checkins


# ─── Property Tests ─────────────────────────────────────────────────────────

@given(checkin=health_checkins())
def test_json_serialization_roundtrip_preserves_data(checkin: HealthCheckIn):
    """
    **Property 4: JSON Serialization Round-Trip Preservation**
    **Validates: Requirements 22.1, 22.5, 22.6**
    
    Property: For any HealthCheckIn H, from_json(to_json(H)) == H
    
    This test verifies that:
    1. Any HealthCheckIn object can be serialized to JSON
    2. The JSON string is valid JSON
    3. The JSON can be parsed back to a HealthCheckIn object
    4. The parsed object equals the original
    
    This property must hold for all inputs including:
    - All fields populated
    - Some fields None
    - Empty strings
    - Unicode characters
    - Special characters in text fields
    """
    # Serialize to JSON
    json_str = checkin.to_json()
    
    # Verify valid JSON
    json_obj = json.loads(json_str)
    assert isinstance(json_obj, dict), "JSON should parse to a dictionary"
    
    # Parse back
    parsed_checkin = HealthCheckIn.from_json(json_str)
    
    # Verify equivalence - compare all fields
    assert parsed_checkin.id == checkin.id, "ID should be preserved"
    assert parsed_checkin.timestamp == checkin.timestamp, "Timestamp should be preserved"
    assert parsed_checkin.mood_score == checkin.mood_score, "Mood score should be preserved"
    assert parsed_checkin.sleep_hours == checkin.sleep_hours, "Sleep hours should be preserved"
    assert parsed_checkin.energy_level == checkin.energy_level, "Energy level should be preserved"
    assert parsed_checkin.pain_notes == checkin.pain_notes, "Pain notes should be preserved"
    assert parsed_checkin.medication_taken == checkin.medication_taken, "Medication taken should be preserved"
    assert parsed_checkin.user_text == checkin.user_text, "User text should be preserved"
    assert parsed_checkin.detected_emotion == checkin.detected_emotion, "Detected emotion should be preserved"
    assert parsed_checkin.emotion_confidence == checkin.emotion_confidence, "Emotion confidence should be preserved"
    assert parsed_checkin.notes == checkin.notes, "Notes should be preserved"
    
    # Verify overall equality
    assert parsed_checkin == checkin, (
        f"Round-trip failed: parsed object does not match original.\n"
        f"Original: {checkin}\n"
        f"Parsed: {parsed_checkin}"
    )


@given(checkin=health_checkins(require_all_fields=True))
def test_json_serialization_with_all_fields_populated(checkin: HealthCheckIn):
    """
    **Property 4: JSON Serialization Round-Trip Preservation**
    **Validates: Requirements 22.1, 22.5, 22.6**
    
    Property: For any HealthCheckIn H with all fields populated,
    from_json(to_json(H)) == H
    
    This test specifically verifies that serialization works correctly
    when all optional fields are populated.
    """
    # Serialize to JSON
    json_str = checkin.to_json()
    
    # Parse back
    parsed_checkin = HealthCheckIn.from_json(json_str)
    
    # Verify all fields are preserved
    assert parsed_checkin == checkin
    
    # Verify no fields are None (all should be populated)
    assert parsed_checkin.mood_score is not None
    assert parsed_checkin.sleep_hours is not None
    assert parsed_checkin.energy_level is not None
    assert parsed_checkin.pain_notes is not None
    assert parsed_checkin.medication_taken is not None
    assert parsed_checkin.user_text is not None
    assert parsed_checkin.detected_emotion is not None
    assert parsed_checkin.emotion_confidence is not None


@given(checkin=health_checkins())
def test_json_output_is_valid_json(checkin: HealthCheckIn):
    """
    **Property 4: JSON Serialization Round-Trip Preservation**
    **Validates: Requirements 22.3**
    
    Property: For any HealthCheckIn H, to_json(H) produces valid JSON
    
    This test verifies that the JSON output:
    1. Is valid JSON (can be parsed)
    2. Contains all expected fields
    3. Has correct types for all fields
    """
    # Serialize to JSON
    json_str = checkin.to_json()
    
    # Parse JSON
    json_obj = json.loads(json_str)
    
    # Verify structure
    assert isinstance(json_obj, dict)
    assert "id" in json_obj
    assert "timestamp" in json_obj
    
    # Verify types
    assert isinstance(json_obj["id"], str)
    assert isinstance(json_obj["timestamp"], str)
    
    # Verify optional fields are either None or correct type
    if json_obj.get("mood_score") is not None:
        assert isinstance(json_obj["mood_score"], (int, float))
    
    if json_obj.get("sleep_hours") is not None:
        assert isinstance(json_obj["sleep_hours"], (int, float))
    
    if json_obj.get("energy_level") is not None:
        assert isinstance(json_obj["energy_level"], (int, float))
    
    if json_obj.get("pain_notes") is not None:
        assert isinstance(json_obj["pain_notes"], str)
    
    if json_obj.get("medication_taken") is not None:
        assert isinstance(json_obj["medication_taken"], bool)
    
    if json_obj.get("user_text") is not None:
        assert isinstance(json_obj["user_text"], str)
    
    if json_obj.get("detected_emotion") is not None:
        assert isinstance(json_obj["detected_emotion"], str)
    
    if json_obj.get("emotion_confidence") is not None:
        assert isinstance(json_obj["emotion_confidence"], (int, float))
    
    if json_obj.get("notes") is not None:
        assert isinstance(json_obj["notes"], str)


@given(checkin=health_checkins())
def test_json_serialization_is_idempotent(checkin: HealthCheckIn):
    """
    **Property 4: JSON Serialization Round-Trip Preservation**
    **Validates: Requirements 22.1, 22.5, 22.6**
    
    Property: For any HealthCheckIn H,
    from_json(to_json(from_json(to_json(H)))) == H
    
    This test verifies that multiple serialization/deserialization cycles
    produce the same result (idempotency).
    """
    # First round-trip
    json_str1 = checkin.to_json()
    parsed1 = HealthCheckIn.from_json(json_str1)
    
    # Second round-trip
    json_str2 = parsed1.to_json()
    parsed2 = HealthCheckIn.from_json(json_str2)
    
    # Third round-trip
    json_str3 = parsed2.to_json()
    parsed3 = HealthCheckIn.from_json(json_str3)
    
    # All should be equal to original
    assert parsed1 == checkin
    assert parsed2 == checkin
    assert parsed3 == checkin
    
    # All JSON strings should be equivalent (may differ in whitespace/order)
    assert json.loads(json_str1) == json.loads(json_str2)
    assert json.loads(json_str2) == json.loads(json_str3)


@given(checkin=health_checkins())
def test_json_handles_none_values_correctly(checkin: HealthCheckIn):
    """
    **Property 4: JSON Serialization Round-Trip Preservation**
    **Validates: Requirements 22.1, 22.5, 22.6**
    
    Property: For any HealthCheckIn H with None values,
    from_json(to_json(H)) preserves None values correctly
    
    This test verifies that None values are correctly serialized
    and deserialized (not converted to empty strings or other values).
    """
    # Serialize to JSON
    json_str = checkin.to_json()
    json_obj = json.loads(json_str)
    
    # Parse back
    parsed_checkin = HealthCheckIn.from_json(json_str)
    
    # Verify None values are preserved
    if checkin.mood_score is None:
        assert json_obj["mood_score"] is None
        assert parsed_checkin.mood_score is None
    
    if checkin.sleep_hours is None:
        assert json_obj["sleep_hours"] is None
        assert parsed_checkin.sleep_hours is None
    
    if checkin.energy_level is None:
        assert json_obj["energy_level"] is None
        assert parsed_checkin.energy_level is None
    
    if checkin.pain_notes is None:
        assert json_obj["pain_notes"] is None
        assert parsed_checkin.pain_notes is None
    
    if checkin.medication_taken is None:
        assert json_obj["medication_taken"] is None
        assert parsed_checkin.medication_taken is None
    
    if checkin.user_text is None:
        assert json_obj["user_text"] is None
        assert parsed_checkin.user_text is None
    
    if checkin.detected_emotion is None:
        assert json_obj["detected_emotion"] is None
        assert parsed_checkin.detected_emotion is None
    
    if checkin.emotion_confidence is None:
        assert json_obj["emotion_confidence"] is None
        assert parsed_checkin.emotion_confidence is None
    
    if checkin.notes is None:
        assert json_obj["notes"] is None
        assert parsed_checkin.notes is None


# ─── Edge Case Tests ────────────────────────────────────────────────────────

def test_empty_strings_roundtrip():
    """
    Test that empty strings in text fields are preserved.
    
    This is an important edge case that should be explicitly tested.
    """
    checkin = HealthCheckIn(
        id="test12345678",
        timestamp="2024-01-01T00:00:00",
        pain_notes="",
        user_text="",
        detected_emotion="",
        notes="",
    )
    
    json_str = checkin.to_json()
    parsed = HealthCheckIn.from_json(json_str)
    
    assert parsed == checkin
    assert parsed.pain_notes == ""
    assert parsed.user_text == ""
    assert parsed.detected_emotion == ""
    assert parsed.notes == ""


def test_unicode_text_roundtrip():
    """
    Test that unicode characters in text fields are preserved.
    
    This tests handling of multi-byte UTF-8 characters.
    """
    checkin = HealthCheckIn(
        id="test12345678",
        timestamp="2024-01-01T00:00:00",
        pain_notes="頭痛がします 😢",
        user_text="今日は気分が良いです 🌸",
        detected_emotion="悲しい",
        notes="メモ: 薬を飲みました 💊",
    )
    
    json_str = checkin.to_json()
    parsed = HealthCheckIn.from_json(json_str)
    
    assert parsed == checkin
    assert parsed.pain_notes == "頭痛がします 😢"
    assert parsed.user_text == "今日は気分が良いです 🌸"
    assert parsed.detected_emotion == "悲しい"
    assert parsed.notes == "メモ: 薬を飲みました 💊"


def test_special_characters_roundtrip():
    """
    Test that special characters in text fields are preserved.
    
    This tests handling of quotes, newlines, and other special characters.
    """
    checkin = HealthCheckIn(
        id="test12345678",
        timestamp="2024-01-01T00:00:00",
        pain_notes='Pain in "lower back"\nSharp sensation',
        user_text="I said: \"I'm feeling better\"\tToday is good!",
        notes="Line 1\nLine 2\r\nLine 3\tTabbed",
    )
    
    json_str = checkin.to_json()
    parsed = HealthCheckIn.from_json(json_str)
    
    assert parsed == checkin
    assert parsed.pain_notes == 'Pain in "lower back"\nSharp sensation'
    assert parsed.user_text == "I said: \"I'm feeling better\"\tToday is good!"
    assert parsed.notes == "Line 1\nLine 2\r\nLine 3\tTabbed"


def test_all_none_values_roundtrip():
    """
    Test that a HealthCheckIn with all optional fields as None is preserved.
    
    This tests the minimal valid HealthCheckIn.
    """
    checkin = HealthCheckIn(
        id="test12345678",
        timestamp="2024-01-01T00:00:00",
        mood_score=None,
        sleep_hours=None,
        energy_level=None,
        pain_notes=None,
        medication_taken=None,
        user_text=None,
        detected_emotion=None,
        emotion_confidence=None,
        notes=None,
    )
    
    json_str = checkin.to_json()
    parsed = HealthCheckIn.from_json(json_str)
    
    assert parsed == checkin
    assert parsed.mood_score is None
    assert parsed.sleep_hours is None
    assert parsed.energy_level is None
    assert parsed.pain_notes is None
    assert parsed.medication_taken is None
    assert parsed.user_text is None
    assert parsed.detected_emotion is None
    assert parsed.emotion_confidence is None
    assert parsed.notes is None


def test_boundary_values_roundtrip():
    """
    Test that boundary values for numeric fields are preserved.
    
    This tests edge cases like 0, 1, 10, 24 for various fields.
    """
    checkin = HealthCheckIn(
        id="test12345678",
        timestamp="2024-01-01T00:00:00",
        mood_score=1.0,  # minimum
        sleep_hours=0.0,  # minimum
        energy_level=10.0,  # maximum
        emotion_confidence=0.0,  # minimum
    )
    
    json_str = checkin.to_json()
    parsed = HealthCheckIn.from_json(json_str)
    
    assert parsed == checkin
    assert parsed.mood_score == 1.0
    assert parsed.sleep_hours == 0.0
    assert parsed.energy_level == 10.0
    assert parsed.emotion_confidence == 0.0
    
    # Test maximum values
    checkin2 = HealthCheckIn(
        id="test87654321",
        timestamp="2024-01-02T00:00:00",
        mood_score=10.0,  # maximum
        sleep_hours=24.0,  # maximum
        energy_level=10.0,  # maximum
        emotion_confidence=1.0,  # maximum
    )
    
    json_str2 = checkin2.to_json()
    parsed2 = HealthCheckIn.from_json(json_str2)
    
    assert parsed2 == checkin2
    assert parsed2.mood_score == 10.0
    assert parsed2.sleep_hours == 24.0
    assert parsed2.energy_level == 10.0
    assert parsed2.emotion_confidence == 1.0


def test_very_long_text_roundtrip():
    """
    Test that very long text fields are preserved.
    
    This tests handling of large text content.
    """
    long_text = "A" * 10000  # 10KB of text
    
    checkin = HealthCheckIn(
        id="test12345678",
        timestamp="2024-01-01T00:00:00",
        pain_notes=long_text,
        user_text=long_text,
        notes=long_text,
    )
    
    json_str = checkin.to_json()
    parsed = HealthCheckIn.from_json(json_str)
    
    assert parsed == checkin
    assert len(parsed.pain_notes) == 10000
    assert len(parsed.user_text) == 10000
    assert len(parsed.notes) == 10000


def test_float_precision_roundtrip():
    """
    Test that float precision is preserved for numeric fields.
    
    This tests that decimal values are not truncated or rounded incorrectly.
    """
    checkin = HealthCheckIn(
        id="test12345678",
        timestamp="2024-01-01T00:00:00",
        mood_score=7.123456789,
        sleep_hours=6.789012345,
        energy_level=8.456789012,
        emotion_confidence=0.987654321,
    )
    
    json_str = checkin.to_json()
    parsed = HealthCheckIn.from_json(json_str)
    
    # JSON may have some floating point precision loss, but should be very close
    assert abs(parsed.mood_score - checkin.mood_score) < 1e-9
    assert abs(parsed.sleep_hours - checkin.sleep_hours) < 1e-9
    assert abs(parsed.energy_level - checkin.energy_level) < 1e-9
    assert abs(parsed.emotion_confidence - checkin.emotion_confidence) < 1e-9


def test_boolean_values_roundtrip():
    """
    Test that boolean values are preserved correctly.
    
    This tests that True/False are not converted to 1/0 or strings.
    """
    # Test True
    checkin_true = HealthCheckIn(
        id="test12345678",
        timestamp="2024-01-01T00:00:00",
        medication_taken=True,
    )
    
    json_str_true = checkin_true.to_json()
    parsed_true = HealthCheckIn.from_json(json_str_true)
    
    assert parsed_true == checkin_true
    assert parsed_true.medication_taken is True
    assert isinstance(parsed_true.medication_taken, bool)
    
    # Test False
    checkin_false = HealthCheckIn(
        id="test87654321",
        timestamp="2024-01-02T00:00:00",
        medication_taken=False,
    )
    
    json_str_false = checkin_false.to_json()
    parsed_false = HealthCheckIn.from_json(json_str_false)
    
    assert parsed_false == checkin_false
    assert parsed_false.medication_taken is False
    assert isinstance(parsed_false.medication_taken, bool)
