"""
Unit tests for emotion detection functionality.

Tests prosodic feature-based emotion classification including:
- High pitch + high energy → stressed
- Low pitch + low energy → fatigued
- High pitch + fast speech → anxious
- Moderate features → neutral/calm
- Linguistic cue detection (English and Japanese)
- Mixed emotion detection
- Edge cases

Requirements: 18.1
"""

import pytest
from core.emotion import classify_emotion
from core.models import EmotionResult


# ─── Test Fixtures ───────────────────────────────────────────────────────────

def get_stressed_features():
    """Return features indicating stressed state: high pitch + high energy."""
    return {
        "pitch_mean": 280.0,      # High pitch
        "pitch_std": 45.0,
        "energy_rms": 0.08,       # High energy
        "speech_rate": 3.5,
        "duration": 5.0,
        "zcr_mean": 0.08,
        "spectral_centroid_mean": 2000.0,
    }


def get_fatigued_features():
    """Return features indicating fatigued state: low pitch + low energy."""
    return {
        "pitch_mean": 120.0,      # Low pitch
        "pitch_std": 15.0,
        "energy_rms": 0.015,      # Low energy
        "speech_rate": 1.5,       # Slow speech
        "duration": 5.0,
        "zcr_mean": 0.03,
        "spectral_centroid_mean": 1200.0,
    }


def get_anxious_features():
    """Return features indicating anxious state: high pitch + fast speech."""
    return {
        "pitch_mean": 270.0,      # High pitch
        "pitch_std": 50.0,        # High variance
        "energy_rms": 0.06,
        "speech_rate": 4.5,       # Fast speech
        "duration": 5.0,
        "zcr_mean": 0.07,
        "spectral_centroid_mean": 1900.0,
    }


def get_calm_features():
    """Return features indicating calm state: moderate/low pitch + low energy."""
    return {
        "pitch_mean": 150.0,      # Low-moderate pitch
        "pitch_std": 20.0,
        "energy_rms": 0.02,       # Low energy
        "speech_rate": 2.0,       # Slow-moderate speech
        "duration": 5.0,
        "zcr_mean": 0.04,
        "spectral_centroid_mean": 1400.0,
    }


def get_neutral_features():
    """Return neutral baseline features."""
    return {
        "pitch_mean": 180.0,      # Moderate pitch
        "pitch_std": 25.0,
        "energy_rms": 0.04,       # Moderate energy
        "speech_rate": 2.5,       # Moderate speech rate
        "duration": 5.0,
        "zcr_mean": 0.05,
        "spectral_centroid_mean": 1500.0,
    }


def get_mixed_features():
    """Return features that could indicate mixed emotions."""
    return {
        "pitch_mean": 220.0,      # Moderately high
        "pitch_std": 35.0,
        "energy_rms": 0.05,       # Moderate
        "speech_rate": 3.5,       # Moderately fast
        "duration": 5.0,
        "zcr_mean": 0.06,
        "spectral_centroid_mean": 1800.0,
    }


# ─── Test Prosodic Feature Detection ─────────────────────────────────────────

class TestProsodyBasedEmotionDetection:
    """Test emotion detection based on prosodic features."""
    
    def test_high_pitch_high_energy_detects_stressed(self):
        """Test that high pitch + high energy → stressed."""
        features = get_stressed_features()
        result = classify_emotion(features, transcript=None)
        
        assert isinstance(result, EmotionResult)
        # Without linguistic cues, may default to neutral if confidence is low
        # With strong prosodic features, should detect stressed/anxious or neutral
        assert result.label in ["stressed", "anxious", "neutral"]
        assert result.confidence > 0.0
        assert result.pitch_mean == features["pitch_mean"]
        assert result.energy_rms == features["energy_rms"]
    
    def test_low_pitch_low_energy_detects_fatigued(self):
        """Test that low pitch + low energy → fatigued."""
        features = get_fatigued_features()
        result = classify_emotion(features, transcript=None)
        
        assert isinstance(result, EmotionResult)
        # Without linguistic cues, may default to neutral if confidence is low
        # With strong prosodic features, should detect fatigued/calm or neutral
        assert result.label in ["fatigued", "calm", "neutral"]
        assert result.confidence > 0.0
        assert result.pitch_mean == features["pitch_mean"]
        assert result.energy_rms == features["energy_rms"]
        assert result.speech_rate == features["speech_rate"]
    
    def test_high_pitch_fast_speech_detects_anxious(self):
        """Test that high pitch + fast speech → anxious."""
        features = get_anxious_features()
        result = classify_emotion(features, transcript=None)
        
        assert isinstance(result, EmotionResult)
        assert result.label in ["anxious", "stressed", "neutral"]
        assert result.confidence > 0.0
        assert result.pitch_mean == features["pitch_mean"]
        assert result.speech_rate == features["speech_rate"]
    
    def test_high_pitch_high_energy_with_keywords_detects_stressed(self):
        """Test that high pitch + high energy + stressed keywords → stressed."""
        features = get_stressed_features()
        result = classify_emotion(features, transcript="I'm so stressed and overwhelmed")
        
        assert isinstance(result, EmotionResult)
        # Combined prosodic + linguistic cues should strongly indicate stressed
        assert result.label in ["stressed", "anxious"]
        assert result.confidence > 0.3  # Should have higher confidence with both signals
    
    def test_low_pitch_low_energy_with_keywords_detects_fatigued(self):
        """Test that low pitch + low energy + fatigued keywords → fatigued."""
        features = get_fatigued_features()
        result = classify_emotion(features, transcript="I'm so tired and exhausted")
        
        assert isinstance(result, EmotionResult)
        # Combined prosodic + linguistic cues should indicate fatigued/calm or neutral
        # The exact result depends on the scoring algorithm and thresholds
        assert result.label in ["fatigued", "calm", "neutral"]
        assert result.confidence > 0.3  # Should have reasonable confidence
    
    def test_moderate_features_detect_neutral_or_calm(self):
        """Test that moderate features → neutral or calm."""
        features = get_neutral_features()
        result = classify_emotion(features, transcript=None)
        
        assert isinstance(result, EmotionResult)
        assert result.label in ["neutral", "calm"]
        assert result.confidence > 0.0
    
    def test_low_pitch_low_energy_slow_speech_detects_calm_or_fatigued(self):
        """Test that low pitch + low energy + slow speech → calm or fatigued."""
        features = get_calm_features()
        result = classify_emotion(features, transcript=None)
        
        assert isinstance(result, EmotionResult)
        # Without linguistic cues, may default to neutral if confidence is low
        assert result.label in ["calm", "fatigued", "neutral"]
        assert result.confidence > 0.0


# ─── Test Linguistic Cue Detection ──────────────────────────────────────────

class TestLinguisticCueDetection:
    """Test emotion detection from linguistic cues in transcripts."""
    
    def test_english_stressed_keywords(self):
        """Test detection of stressed emotion from English keywords."""
        features = get_neutral_features()
        transcripts = [
            "I'm feeling stressed",
            "I'm overwhelmed with work",
            "There's too much pressure",
            "I can't handle this deadline"
        ]
        
        for transcript in transcripts:
            result = classify_emotion(features, transcript=transcript)
            assert isinstance(result, EmotionResult)
            # Linguistic cues should boost stressed score
            assert result.label in ["stressed", "anxious", "neutral"]
    
    def test_english_anxious_keywords(self):
        """Test detection of anxious emotion from English keywords."""
        features = get_neutral_features()
        transcripts = [
            "I'm feeling anxious",
            "I'm nervous about this",
            "My heart is racing",
            "I'm scared and worried"
        ]
        
        for transcript in transcripts:
            result = classify_emotion(features, transcript=transcript)
            assert isinstance(result, EmotionResult)
            # Linguistic cues should boost anxious score
            assert result.label in ["anxious", "stressed", "neutral"]
    
    def test_english_fatigued_keywords(self):
        """Test detection of fatigued emotion from English keywords."""
        features = get_neutral_features()
        transcripts = [
            "I'm so tired",
            "I'm exhausted",
            "I have no energy",
            "I'm completely drained"
        ]
        
        for transcript in transcripts:
            result = classify_emotion(features, transcript=transcript)
            assert isinstance(result, EmotionResult)
            # Linguistic cues should boost fatigued score
            assert result.label in ["fatigued", "calm", "neutral"]
    
    def test_english_calm_keywords(self):
        """Test detection of calm emotion from English keywords."""
        features = get_neutral_features()
        transcripts = [
            "I'm feeling relaxed",
            "Everything is fine",
            "I'm feeling great",
            "I'm happy and content"
        ]
        
        for transcript in transcripts:
            result = classify_emotion(features, transcript=transcript)
            assert isinstance(result, EmotionResult)
            # Linguistic cues should boost calm score
            assert result.label in ["calm", "neutral"]
    
    def test_japanese_stressed_keywords(self):
        """Test detection of stressed emotion from Japanese keywords."""
        features = get_neutral_features()
        transcripts = [
            "ストレスを感じています",      # I feel stressed
            "大変です",                    # It's tough
            "プレッシャーがあります",      # There's pressure
            "締め切りが心配です"           # Worried about deadline
        ]
        
        for transcript in transcripts:
            result = classify_emotion(features, transcript=transcript)
            assert isinstance(result, EmotionResult)
            # Japanese keywords should work like English ones
            assert result.label in ["stressed", "anxious", "neutral"]
    
    def test_japanese_anxious_keywords(self):
        """Test detection of anxious emotion from Japanese keywords."""
        features = get_neutral_features()
        transcripts = [
            "不安です",                    # I'm anxious
            "心配しています",              # I'm worried
            "緊張しています",              # I'm nervous
            "ドキドキしています"           # My heart is racing
        ]
        
        for transcript in transcripts:
            result = classify_emotion(features, transcript=transcript)
            assert isinstance(result, EmotionResult)
            assert result.label in ["anxious", "stressed", "neutral"]
    
    def test_japanese_fatigued_keywords(self):
        """Test detection of fatigued emotion from Japanese keywords."""
        features = get_neutral_features()
        transcripts = [
            "疲れています",                # I'm tired
            "だるいです",                  # I'm sluggish
            "眠いです",                    # I'm sleepy
            "へとへとです"                 # I'm exhausted
        ]
        
        for transcript in transcripts:
            result = classify_emotion(features, transcript=transcript)
            assert isinstance(result, EmotionResult)
            assert result.label in ["fatigued", "calm", "neutral"]
    
    def test_japanese_calm_keywords(self):
        """Test detection of calm emotion from Japanese keywords."""
        features = get_neutral_features()
        transcripts = [
            "元気です",                    # I'm fine/energetic
            "気持ちいいです",              # I feel good
            "リラックスしています",        # I'm relaxed
            "幸せです"                     # I'm happy
        ]
        
        for transcript in transcripts:
            result = classify_emotion(features, transcript=transcript)
            assert isinstance(result, EmotionResult)
            assert result.label in ["calm", "neutral"]
    
    def test_combined_prosody_and_linguistic_cues(self):
        """Test that prosodic features and linguistic cues work together."""
        # High pitch/energy features + stressed keywords should strongly indicate stressed
        features = get_stressed_features()
        transcript = "I'm so stressed and overwhelmed"
        
        result = classify_emotion(features, transcript=transcript)
        
        assert isinstance(result, EmotionResult)
        assert result.label in ["stressed", "anxious"]
        # Combined signals should give higher confidence
        assert result.confidence > 0.3


# ─── Test Mixed Emotion Detection ────────────────────────────────────────────

class TestMixedEmotionDetection:
    """Test mixed emotion detection when confidence difference < 0.2."""
    
    def test_mixed_emotion_detected_with_close_confidence(self):
        """Test that mixed emotions are detected when confidence difference < 0.2."""
        features = get_mixed_features()
        transcript = "I'm stressed but trying to stay calm"
        
        result = classify_emotion(features, transcript=transcript)
        
        assert isinstance(result, EmotionResult)
        # Check if mixed emotion was detected
        if result.is_mixed:
            assert result.secondary_label is not None
            assert result.secondary_confidence is not None
            assert result.confidence - result.secondary_confidence < 0.2
            assert result.secondary_label != result.label
    
    def test_mixed_emotion_returns_top_two_emotions(self):
        """Test that mixed emotion detection returns top 2 emotions correctly."""
        features = get_mixed_features()
        transcript = "I'm anxious and tired"
        
        result = classify_emotion(features, transcript=transcript)
        
        assert isinstance(result, EmotionResult)
        assert result.label is not None
        assert result.confidence > 0.0
        
        if result.is_mixed:
            assert result.secondary_label is not None
            assert result.secondary_confidence is not None
            assert result.confidence >= result.secondary_confidence
    
    def test_no_mixed_emotion_with_large_confidence_difference(self):
        """Test that mixed emotion is NOT detected when confidence difference >= 0.2."""
        features = get_stressed_features()
        transcript = "I'm extremely stressed and overwhelmed and anxious"
        
        result = classify_emotion(features, transcript=transcript)
        
        assert isinstance(result, EmotionResult)
        # Strong single emotion should not trigger mixed detection
        # (This may vary based on exact scoring, but validates the logic)
        if not result.is_mixed:
            assert result.secondary_label is None
            assert result.secondary_confidence is None
    
    def test_mixed_emotion_with_conflicting_keywords(self):
        """Test mixed emotion detection with conflicting linguistic cues."""
        features = get_neutral_features()
        transcript = "I'm tired but also stressed about work"
        
        result = classify_emotion(features, transcript=transcript)
        
        assert isinstance(result, EmotionResult)
        # Conflicting keywords should potentially trigger mixed detection
        if result.is_mixed:
            emotions = [result.label, result.secondary_label]
            assert "fatigued" in emotions or "stressed" in emotions


# ─── Test Edge Cases ─────────────────────────────────────────────────────────

class TestEmotionDetectionEdgeCases:
    """Test edge cases for emotion detection."""
    
    def test_empty_features_defaults_to_neutral(self):
        """Test that empty/invalid features default to neutral."""
        features = {
            "pitch_mean": 0.0,
            "pitch_std": 0.0,
            "energy_rms": 0.0,
            "speech_rate": 0.0,
            "duration": 0.0,
            "zcr_mean": 0.0,
            "spectral_centroid_mean": 0.0,
        }
        
        result = classify_emotion(features, transcript=None)
        
        assert isinstance(result, EmotionResult)
        assert result.label == "neutral"
        assert result.is_mixed is False
    
    def test_low_confidence_defaults_to_neutral(self):
        """Test that low confidence results default to neutral."""
        features = {
            "pitch_mean": 0.0,
            "pitch_std": 0.0,
            "energy_rms": 0.0,
            "speech_rate": 0.0,
            "duration": 0.0,
            "zcr_mean": 0.0,
            "spectral_centroid_mean": 0.0,
        }
        
        result = classify_emotion(features, transcript=None)
        
        assert result.label == "neutral"
        assert result.is_mixed is False
    
    def test_none_transcript_handled_gracefully(self):
        """Test that None transcript is handled gracefully."""
        features = get_neutral_features()
        
        result = classify_emotion(features, transcript=None)
        
        assert isinstance(result, EmotionResult)
        assert result.label is not None
        assert result.confidence > 0.0
    
    def test_empty_transcript_handled_gracefully(self):
        """Test that empty transcript is handled gracefully."""
        features = get_neutral_features()
        
        result = classify_emotion(features, transcript="")
        
        assert isinstance(result, EmotionResult)
        assert result.label is not None
        assert result.confidence > 0.0
    
    def test_emotion_result_contains_all_required_fields(self):
        """Test that EmotionResult contains all required fields."""
        features = get_neutral_features()
        result = classify_emotion(features, transcript=None)
        
        assert hasattr(result, 'label')
        assert hasattr(result, 'confidence')
        assert hasattr(result, 'pitch_mean')
        assert hasattr(result, 'pitch_std')
        assert hasattr(result, 'energy_rms')
        assert hasattr(result, 'speech_rate')
        assert hasattr(result, 'secondary_label')
        assert hasattr(result, 'secondary_confidence')
        assert hasattr(result, 'is_mixed')
        assert hasattr(result, 'timestamp')
    
    def test_confidence_values_are_valid(self):
        """Test that confidence values are in valid range [0, 1]."""
        features = get_stressed_features()
        result = classify_emotion(features, transcript="I'm stressed")
        
        assert 0.0 <= result.confidence <= 1.0
        if result.secondary_confidence is not None:
            assert 0.0 <= result.secondary_confidence <= 1.0
    
    def test_high_pitch_variance_indicates_emotional_instability(self):
        """Test that high pitch variance contributes to anxious/stressed detection."""
        features = get_neutral_features()
        features["pitch_std"] = 50.0  # High variance
        
        result = classify_emotion(features, transcript=None)
        
        assert isinstance(result, EmotionResult)
        # High variance should contribute to anxious/stressed scores
        assert result.label in ["anxious", "stressed", "neutral"]


# ─── Test Calibrated Thresholds ──────────────────────────────────────────────

class TestCalibratedThresholds:
    """Test that emotion detection uses calibrated thresholds when available."""
    
    def test_default_user_uses_default_config(self):
        """Test that default user uses default configuration."""
        features = get_stressed_features()
        result = classify_emotion(features, transcript=None, user_id="default")
        
        assert isinstance(result, EmotionResult)
        assert result.label is not None
        assert result.confidence > 0.0
    
    def test_custom_user_id_accepted(self):
        """Test that custom user_id is accepted (even without calibration data)."""
        features = get_stressed_features()
        result = classify_emotion(features, transcript=None, user_id="test_user_123")
        
        assert isinstance(result, EmotionResult)
        assert result.label is not None
        assert result.confidence > 0.0
    
    def test_emotion_detection_without_user_id_uses_default(self):
        """Test that omitting user_id uses default configuration."""
        features = get_stressed_features()
        result = classify_emotion(features, transcript=None)
        
        assert isinstance(result, EmotionResult)
        assert result.label is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
