"""
Integration test for mixed emotion detection.

Verifies that the mixed emotion detection works end-to-end with
the emotion classification pipeline.

Requirements: 9.3
"""

import pytest
from core.emotion import classify_emotion
from core.models import EmotionResult


def test_mixed_emotion_end_to_end():
    """Test mixed emotion detection end-to-end."""
    # Create features that could indicate mixed emotions
    features = {
        "pitch_mean": 220.0,  # Moderately high
        "pitch_std": 35.0,
        "energy_rms": 0.05,   # Moderate
        "speech_rate": 3.5,   # Moderately fast
        "duration": 5.0,
        "zcr_mean": 0.06,
        "spectral_centroid_mean": 1800.0,
    }
    
    # Transcript with conflicting emotions
    transcript = "I'm stressed about work but trying to stay calm"
    
    result = classify_emotion(features, transcript=transcript)
    
    # Verify result structure
    assert isinstance(result, EmotionResult)
    assert hasattr(result, 'label')
    assert hasattr(result, 'confidence')
    assert hasattr(result, 'is_mixed')
    assert hasattr(result, 'secondary_label')
    assert hasattr(result, 'secondary_confidence')
    
    # Verify result values
    assert result.label is not None
    assert result.confidence > 0.0
    assert isinstance(result.is_mixed, bool)
    
    # If mixed, verify secondary emotion
    if result.is_mixed:
        assert result.secondary_label is not None
        assert result.secondary_confidence is not None
        assert result.secondary_label != result.label
        assert result.confidence >= result.secondary_confidence
        assert (result.confidence - result.secondary_confidence) < 0.2


def test_emotion_result_to_dict_with_mixed():
    """Test that EmotionResult.to_dict() includes mixed emotion fields."""
    features = {
        "pitch_mean": 220.0,
        "pitch_std": 35.0,
        "energy_rms": 0.05,
        "speech_rate": 3.5,
        "duration": 5.0,
        "zcr_mean": 0.06,
        "spectral_centroid_mean": 1800.0,
    }
    
    transcript = "I'm anxious but hopeful"
    result = classify_emotion(features, transcript=transcript)
    
    result_dict = result.to_dict()
    
    # Verify all fields are in dict
    assert 'label' in result_dict
    assert 'confidence' in result_dict
    assert 'is_mixed' in result_dict
    assert 'secondary_label' in result_dict
    assert 'secondary_confidence' in result_dict
    assert 'pitch_mean' in result_dict
    assert 'pitch_std' in result_dict
    assert 'energy_rms' in result_dict
    assert 'speech_rate' in result_dict
    assert 'timestamp' in result_dict


def test_get_emotion_description():
    """Test get_emotion_description() method."""
    features = {
        "pitch_mean": 220.0,
        "pitch_std": 35.0,
        "energy_rms": 0.05,
        "speech_rate": 3.5,
        "duration": 5.0,
        "zcr_mean": 0.06,
        "spectral_centroid_mean": 1800.0,
    }
    
    # Test single emotion
    result_single = classify_emotion(features, transcript=None)
    desc_single = result_single.get_emotion_description()
    assert isinstance(desc_single, str)
    assert result_single.label in desc_single
    
    # Test mixed emotion
    transcript = "I'm stressed and tired"
    result_mixed = classify_emotion(features, transcript=transcript)
    desc_mixed = result_mixed.get_emotion_description()
    assert isinstance(desc_mixed, str)
    
    if result_mixed.is_mixed:
        assert result_mixed.label in desc_mixed
        assert result_mixed.secondary_label in desc_mixed
        assert "and" in desc_mixed


def test_backward_compatibility():
    """Test that existing code still works with new EmotionResult fields."""
    features = {
        "pitch_mean": 280.0,
        "pitch_std": 45.0,
        "energy_rms": 0.09,
        "speech_rate": 4.5,
        "duration": 5.0,
        "zcr_mean": 0.08,
        "spectral_centroid_mean": 2000.0,
    }
    
    result = classify_emotion(features, transcript="I'm stressed")
    
    # Old code should still work
    assert result.label in ["stressed", "anxious", "neutral", "calm", "fatigued"]
    assert 0.0 <= result.confidence <= 1.0
    assert result.pitch_mean > 0
    assert result.energy_rms > 0
    
    # New fields should exist but may be None
    assert hasattr(result, 'is_mixed')
    assert hasattr(result, 'secondary_label')
    assert hasattr(result, 'secondary_confidence')


def test_mixed_emotion_with_user_id():
    """Test mixed emotion detection with user_id parameter."""
    features = {
        "pitch_mean": 220.0,
        "pitch_std": 35.0,
        "energy_rms": 0.05,
        "speech_rate": 3.5,
        "duration": 5.0,
        "zcr_mean": 0.06,
        "spectral_centroid_mean": 1800.0,
    }
    
    transcript = "I'm anxious but calm"
    
    # Test with default user
    result_default = classify_emotion(features, transcript=transcript, user_id="default")
    assert isinstance(result_default, EmotionResult)
    
    # Test with custom user
    result_custom = classify_emotion(features, transcript=transcript, user_id="test_user")
    assert isinstance(result_custom, EmotionResult)
    
    # Both should have mixed emotion fields
    assert hasattr(result_default, 'is_mixed')
    assert hasattr(result_custom, 'is_mixed')


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
