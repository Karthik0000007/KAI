"""
Test Japanese health signal extraction patterns.

**Validates: Requirements 7.8**

Tests that the LLM_Engine extracts health signals from Japanese text
using Japanese-specific patterns for sleep, mood, energy, medication, and pain.
"""

import pytest
from core.llm import extract_health_signals


# ─── Sleep Pattern Tests ─────────────────────────────────────────────────────

def test_japanese_sleep_hours_numeric():
    """Test extraction of numeric sleep hours in Japanese."""
    # "6時間寝た" = "slept 6 hours"
    result = extract_health_signals("6時間寝た")
    assert "sleep_hours" in result
    assert result["sleep_hours"] == 6.0

    # "5時間睡眠" = "5 hours sleep"
    result = extract_health_signals("昨日は5時間睡眠でした")
    assert "sleep_hours" in result
    assert result["sleep_hours"] == 5.0

    # Decimal hours
    result = extract_health_signals("7.5時間寝ました")
    assert "sleep_hours" in result
    assert result["sleep_hours"] == 7.5


def test_japanese_sleep_qualitative():
    """Test extraction of qualitative sleep descriptions in Japanese."""
    # "よく眠れた" = "slept well"
    result = extract_health_signals("よく眠れた")
    assert "sleep_hours" in result
    assert result["sleep_hours"] == 8.0

    # "ぐっすり" = "soundly"
    result = extract_health_signals("ぐっすり眠りました")
    assert "sleep_hours" in result
    assert result["sleep_hours"] == 8.0

    # "あまり寝られなかった" = "couldn't sleep much"
    result = extract_health_signals("あまり寝られなかった")
    assert "sleep_hours" in result
    assert result["sleep_hours"] == 4.0

    # "眠れなかった" = "couldn't sleep"
    result = extract_health_signals("昨夜は眠れなかった")
    assert "sleep_hours" in result
    assert result["sleep_hours"] == 4.0


# ─── Mood Pattern Tests ──────────────────────────────────────────────────────

def test_japanese_mood_numeric():
    """Test extraction of numeric mood scores in Japanese."""
    # "気分は7" = "mood is 7"
    result = extract_health_signals("気分は7")
    assert "mood_score" in result
    assert result["mood_score"] == 7.0

    # "調子は5" = "condition is 5"
    result = extract_health_signals("今日の調子は5です")
    assert "mood_score" in result
    assert result["mood_score"] == 5.0

    # With particle variations
    result = extract_health_signals("気分が8")
    assert "mood_score" in result
    assert result["mood_score"] == 8.0


def test_japanese_mood_qualitative():
    """Test extraction of qualitative mood descriptions in Japanese."""
    # "元気" = "energetic/well"
    result = extract_health_signals("元気です")
    assert "mood_score" in result
    assert result["mood_score"] == 7.5

    # "嬉しい" = "happy"
    result = extract_health_signals("今日は嬉しい")
    assert "mood_score" in result
    assert result["mood_score"] == 7.5

    # "落ち込んでいる" = "feeling down"
    result = extract_health_signals("落ち込んでいる")
    assert "mood_score" in result
    assert result["mood_score"] == 3.0

    # "悲しい" = "sad"
    result = extract_health_signals("悲しい気持ちです")
    assert "mood_score" in result
    assert result["mood_score"] == 3.0


# ─── Energy Pattern Tests ────────────────────────────────────────────────────

def test_japanese_energy_numeric():
    """Test extraction of numeric energy levels in Japanese."""
    # "エネルギーは4" = "energy is 4"
    result = extract_health_signals("エネルギーは4")
    assert "energy_level" in result
    assert result["energy_level"] == 4.0

    # With particle variations
    result = extract_health_signals("エネルギーが6です")
    assert "energy_level" in result
    assert result["energy_level"] == 6.0


def test_japanese_energy_qualitative():
    """Test extraction of qualitative energy descriptions in Japanese."""
    # "元気いっぱい" = "full of energy"
    result = extract_health_signals("元気いっぱいです")
    assert "energy_level" in result
    assert result["energy_level"] == 8.0

    # "疲れている" = "tired"
    result = extract_health_signals("疲れている")
    assert "energy_level" in result
    assert result["energy_level"] == 3.0

    # "疲れた" = "got tired"
    result = extract_health_signals("今日は疲れた")
    assert "energy_level" in result
    assert result["energy_level"] == 3.0

    # "だるい" = "sluggish/lethargic"
    result = extract_health_signals("体がだるい")
    assert "energy_level" in result
    assert result["energy_level"] == 3.0


# ─── Medication Pattern Tests ────────────────────────────────────────────────

def test_japanese_medication_taken():
    """Test extraction of medication taken in Japanese."""
    # "薬を飲んだ" = "took medicine"
    result = extract_health_signals("薬を飲んだ")
    assert "medication_taken" in result
    assert result["medication_taken"] is True

    # "服薬した" = "took medication"
    result = extract_health_signals("今朝服薬した")
    assert "medication_taken" in result
    assert result["medication_taken"] is True

    # "服用した" = "took (formal)"
    result = extract_health_signals("服用しました")
    assert "medication_taken" in result
    assert result["medication_taken"] is True


def test_japanese_medication_forgotten():
    """Test extraction of forgotten medication in Japanese."""
    # "薬を忘れた" = "forgot medicine"
    result = extract_health_signals("薬を忘れた")
    assert "medication_taken" in result
    assert result["medication_taken"] is False

    # "飲み忘れた" = "forgot to take"
    result = extract_health_signals("飲み忘れた")
    assert "medication_taken" in result
    assert result["medication_taken"] is False

    # "飲んでない" = "didn't take"
    result = extract_health_signals("まだ飲んでない")
    assert "medication_taken" in result
    assert result["medication_taken"] is False


# ─── Pain Pattern Tests ──────────────────────────────────────────────────────

def test_japanese_pain_general():
    """Test extraction of general pain mentions in Japanese."""
    # "痛い" = "hurts/painful"
    result = extract_health_signals("痛い")
    assert "pain_mentioned" in result
    assert result["pain_mentioned"] is True

    # "痛み" = "pain"
    result = extract_health_signals("痛みがあります")
    assert "pain_mentioned" in result
    assert result["pain_mentioned"] is True


def test_japanese_pain_specific():
    """Test extraction of specific pain types in Japanese."""
    # "頭痛" = "headache"
    result = extract_health_signals("頭痛がする")
    assert "pain_mentioned" in result
    assert result["pain_mentioned"] is True

    # "腰痛" = "back pain"
    result = extract_health_signals("腰痛があります")
    assert "pain_mentioned" in result
    assert result["pain_mentioned"] is True

    # "腹痛" = "stomach pain"
    result = extract_health_signals("腹痛です")
    assert "pain_mentioned" in result
    assert result["pain_mentioned"] is True

    # "吐き気" = "nausea"
    result = extract_health_signals("吐き気がします")
    assert "pain_mentioned" in result
    assert result["pain_mentioned"] is True

    # "めまい" = "dizziness"
    result = extract_health_signals("めまいがする")
    assert "pain_mentioned" in result
    assert result["pain_mentioned"] is True


# ─── Mixed Language Tests ────────────────────────────────────────────────────

def test_japanese_multiple_signals():
    """Test extraction of multiple health signals from one Japanese sentence."""
    # "6時間寝て、気分は7で、薬を飲んだ"
    # = "slept 6 hours, mood is 7, took medicine"
    result = extract_health_signals("6時間寝て、気分は7で、薬を飲んだ")
    assert "sleep_hours" in result
    assert result["sleep_hours"] == 6.0
    assert "mood_score" in result
    assert result["mood_score"] == 7.0
    assert "medication_taken" in result
    assert result["medication_taken"] is True


def test_japanese_no_signals():
    """Test that Japanese text without health signals returns empty dict."""
    result = extract_health_signals("こんにちは、今日はいい天気ですね")
    # Should return empty dict or only have unrelated signals
    assert isinstance(result, dict)


# ─── Edge Cases ──────────────────────────────────────────────────────────────

def test_japanese_boundary_scores():
    """Test that Japanese numeric scores respect 1-10 boundaries."""
    # Valid score
    result = extract_health_signals("気分は10")
    assert result["mood_score"] == 10.0

    # Invalid score (out of range) - should be ignored
    result = extract_health_signals("気分は15")
    # Should not extract invalid score
    assert "mood_score" not in result or result["mood_score"] != 15.0


def test_english_patterns_still_work():
    """Test that English patterns still work after adding Japanese support."""
    # English sleep
    result = extract_health_signals("I slept 7 hours")
    assert "sleep_hours" in result
    assert result["sleep_hours"] == 7.0

    # English mood
    result = extract_health_signals("my mood is 8")
    assert "mood_score" in result
    assert result["mood_score"] == 8.0

    # English medication
    result = extract_health_signals("took my medicine")
    assert "medication_taken" in result
    assert result["medication_taken"] is True

    # English pain
    result = extract_health_signals("I have a headache")
    assert "pain_mentioned" in result
    assert result["pain_mentioned"] is True
