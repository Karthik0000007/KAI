"""
Unit tests for emotion calibration functionality.

Tests the calibration mode that allows users to record samples in different
emotional states to compute personalized baseline thresholds.

Requirements: 9.7
"""

import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

from core.emotion import (
    calibrate_emotion,
    load_calibration_data,
    save_calibration_data,
    compute_calibrated_thresholds,
    get_emotion_config,
    clear_calibration_data,
    CALIBRATION_FILE
)


@pytest.fixture
def temp_calibration_file(monkeypatch):
    """Create a temporary calibration file for testing."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        temp_path = Path(f.name)
    
    # Patch the CALIBRATION_FILE constant
    monkeypatch.setattr('core.emotion.CALIBRATION_FILE', temp_path)
    
    yield temp_path
    
    # Cleanup
    if temp_path.exists():
        temp_path.unlink()


@pytest.fixture
def mock_audio_samples(tmp_path):
    """Create mock audio sample files."""
    samples = []
    for i in range(5):
        sample_path = tmp_path / f"sample_{i}.wav"
        sample_path.touch()  # Create empty file
        samples.append(str(sample_path))
    return samples


@pytest.fixture
def mock_extract_features():
    """Mock extract_audio_features to return consistent test data."""
    def mock_features(audio_path):
        # Return different features based on filename
        if "calm" in audio_path:
            return {
                "pitch_mean": 150.0,
                "pitch_std": 20.0,
                "energy_rms": 0.03,
                "speech_rate": 2.0,
                "duration": 5.0,
                "zcr_mean": 0.05,
                "spectral_centroid_mean": 1500.0
            }
        elif "stressed" in audio_path:
            return {
                "pitch_mean": 280.0,
                "pitch_std": 45.0,
                "energy_rms": 0.09,
                "speech_rate": 4.5,
                "duration": 5.0,
                "zcr_mean": 0.08,
                "spectral_centroid_mean": 2000.0
            }
        elif "anxious" in audio_path:
            return {
                "pitch_mean": 260.0,
                "pitch_std": 50.0,
                "energy_rms": 0.06,
                "speech_rate": 5.0,
                "duration": 5.0,
                "zcr_mean": 0.09,
                "spectral_centroid_mean": 2200.0
            }
        elif "fatigued" in audio_path:
            return {
                "pitch_mean": 120.0,
                "pitch_std": 15.0,
                "energy_rms": 0.015,
                "speech_rate": 1.2,
                "duration": 5.0,
                "zcr_mean": 0.03,
                "spectral_centroid_mean": 1200.0
            }
        else:
            # Default features
            return {
                "pitch_mean": 200.0,
                "pitch_std": 30.0,
                "energy_rms": 0.05,
                "speech_rate": 3.0,
                "duration": 5.0,
                "zcr_mean": 0.06,
                "spectral_centroid_mean": 1800.0
            }
    
    return mock_features


class TestEmotionCalibration:
    """Test emotion calibration functionality."""
    
    def test_calibrate_emotion_valid_input(self, temp_calibration_file, mock_extract_features):
        """Test calibration with valid input."""
        with patch('core.emotion.extract_audio_features', side_effect=mock_extract_features):
            # Create mock audio samples with "calm" in filename
            samples = [f"calm_sample_{i}.wav" for i in range(5)]
            
            stats = calibrate_emotion("calm", samples, user_id="test_user")
            
            # Verify stats structure
            assert "pitch_mean" in stats
            assert "pitch_std" in stats
            assert "energy_mean" in stats
            assert "energy_std" in stats
            assert "rate_mean" in stats
            assert "rate_std" in stats
            assert "sample_count" in stats
            assert "timestamp" in stats
            
            # Verify sample count
            assert stats["sample_count"] == 5
            
            # Verify stats are reasonable for calm emotion
            assert 100 < stats["pitch_mean"] < 200
            assert stats["energy_mean"] < 0.05
            assert stats["rate_mean"] < 3.0
    
    def test_calibrate_emotion_invalid_state(self, temp_calibration_file):
        """Test calibration with invalid emotion state."""
        with pytest.raises(ValueError, match="Invalid emotion state"):
            calibrate_emotion("happy", ["sample1.wav"], user_id="test_user")
    
    def test_calibrate_emotion_insufficient_samples(self, temp_calibration_file):
        """Test calibration with too few samples."""
        with pytest.raises(ValueError, match="At least 3 audio samples required"):
            calibrate_emotion("calm", ["sample1.wav", "sample2.wav"], user_id="test_user")
    
    def test_calibrate_emotion_saves_data(self, temp_calibration_file, mock_extract_features):
        """Test that calibration data is saved to file."""
        with patch('core.emotion.extract_audio_features', side_effect=mock_extract_features):
            samples = [f"calm_sample_{i}.wav" for i in range(5)]
            
            calibrate_emotion("calm", samples, user_id="test_user")
            
            # Verify file was created
            assert temp_calibration_file.exists()
            
            # Verify data structure
            with open(temp_calibration_file, 'r') as f:
                data = json.load(f)
            
            assert "test_user" in data
            assert "calm" in data["test_user"]
            assert data["test_user"]["calm"]["sample_count"] == 5
    
    def test_calibrate_multiple_emotions(self, temp_calibration_file, mock_extract_features):
        """Test calibrating multiple emotion states for same user."""
        with patch('core.emotion.extract_audio_features', side_effect=mock_extract_features):
            # Calibrate calm
            calm_samples = [f"calm_sample_{i}.wav" for i in range(5)]
            calibrate_emotion("calm", calm_samples, user_id="test_user")
            
            # Calibrate stressed
            stressed_samples = [f"stressed_sample_{i}.wav" for i in range(5)]
            calibrate_emotion("stressed", stressed_samples, user_id="test_user")
            
            # Verify both are saved
            data = load_calibration_data()
            assert "test_user" in data
            assert "calm" in data["test_user"]
            assert "stressed" in data["test_user"]
    
    def test_calibrate_multiple_users(self, temp_calibration_file, mock_extract_features):
        """Test calibrating for multiple users."""
        with patch('core.emotion.extract_audio_features', side_effect=mock_extract_features):
            # Calibrate user 1
            samples1 = [f"calm_sample_{i}.wav" for i in range(5)]
            calibrate_emotion("calm", samples1, user_id="user1")
            
            # Calibrate user 2
            samples2 = [f"calm_sample_{i}.wav" for i in range(5)]
            calibrate_emotion("calm", samples2, user_id="user2")
            
            # Verify both users are saved
            data = load_calibration_data()
            assert "user1" in data
            assert "user2" in data


class TestCalibrationDataManagement:
    """Test calibration data loading, saving, and clearing."""
    
    def test_load_calibration_data_empty(self, temp_calibration_file):
        """Test loading when no calibration file exists."""
        if temp_calibration_file.exists():
            temp_calibration_file.unlink()
        
        data = load_calibration_data()
        assert data == {}
    
    def test_save_and_load_calibration_data(self, temp_calibration_file):
        """Test saving and loading calibration data."""
        test_data = {
            "user1": {
                "calm": {
                    "pitch_mean": 150.0,
                    "pitch_std": 20.0,
                    "energy_mean": 0.03,
                    "energy_std": 0.01,
                    "rate_mean": 2.0,
                    "rate_std": 0.5,
                    "sample_count": 5
                }
            }
        }
        
        save_calibration_data(test_data)
        loaded_data = load_calibration_data()
        
        assert loaded_data == test_data
    
    def test_clear_calibration_data_specific_user(self, temp_calibration_file):
        """Test clearing calibration data for specific user."""
        # Create data for multiple users
        test_data = {
            "user1": {"calm": {"pitch_mean": 150.0}},
            "user2": {"calm": {"pitch_mean": 160.0}}
        }
        save_calibration_data(test_data)
        
        # Clear user1
        clear_calibration_data("user1")
        
        # Verify user1 is gone but user2 remains
        data = load_calibration_data()
        assert "user1" not in data
        assert "user2" in data
    
    def test_clear_calibration_data_all_users(self, temp_calibration_file):
        """Test clearing all calibration data."""
        # Create data
        test_data = {
            "user1": {"calm": {"pitch_mean": 150.0}},
            "user2": {"calm": {"pitch_mean": 160.0}}
        }
        save_calibration_data(test_data)
        
        # Clear all
        clear_calibration_data(None)
        
        # Verify file is deleted
        assert not temp_calibration_file.exists()


class TestCalibratedThresholds:
    """Test computation and usage of calibrated thresholds."""
    
    def test_compute_calibrated_thresholds_no_data(self, temp_calibration_file):
        """Test computing thresholds when no calibration data exists."""
        thresholds = compute_calibrated_thresholds("nonexistent_user")
        assert thresholds is None
    
    def test_compute_calibrated_thresholds_insufficient_data(self, temp_calibration_file):
        """Test computing thresholds with insufficient calibration data."""
        # Only calm, missing stressed
        test_data = {
            "user1": {
                "calm": {
                    "pitch_mean": 150.0,
                    "energy_mean": 0.03,
                    "rate_mean": 2.0
                }
            }
        }
        save_calibration_data(test_data)
        
        thresholds = compute_calibrated_thresholds("user1")
        assert thresholds is None
    
    def test_compute_calibrated_thresholds_minimal_data(self, temp_calibration_file):
        """Test computing thresholds with minimal required data (calm + stressed)."""
        test_data = {
            "user1": {
                "calm": {
                    "pitch_mean": 150.0,
                    "energy_mean": 0.03,
                    "rate_mean": 2.0
                },
                "stressed": {
                    "pitch_mean": 280.0,
                    "energy_mean": 0.09,
                    "rate_mean": 4.5
                }
            }
        }
        save_calibration_data(test_data)
        
        thresholds = compute_calibrated_thresholds("user1")
        
        assert thresholds is not None
        assert "pitch_low" in thresholds
        assert "pitch_high" in thresholds
        assert thresholds["pitch_low"] == 150.0
        assert thresholds["pitch_high"] == 280.0
    
    def test_compute_calibrated_thresholds_full_data(self, temp_calibration_file):
        """Test computing thresholds with all emotion states calibrated."""
        test_data = {
            "user1": {
                "calm": {
                    "pitch_mean": 150.0,
                    "energy_mean": 0.03,
                    "rate_mean": 2.0
                },
                "stressed": {
                    "pitch_mean": 280.0,
                    "energy_mean": 0.09,
                    "rate_mean": 4.5
                },
                "anxious": {
                    "pitch_mean": 260.0,
                    "energy_mean": 0.06,
                    "rate_mean": 5.0
                },
                "fatigued": {
                    "pitch_mean": 120.0,
                    "energy_mean": 0.015,
                    "rate_mean": 1.2
                }
            }
        }
        save_calibration_data(test_data)
        
        thresholds = compute_calibrated_thresholds("user1")
        
        assert thresholds is not None
        assert "pitch_low" in thresholds
        assert "pitch_high" in thresholds
        assert "energy_low" in thresholds
        assert "energy_high" in thresholds
        assert "rate_slow" in thresholds
        assert "rate_fast" in thresholds
        
        # Verify threshold values are reasonable
        assert thresholds["pitch_low"] == 150.0  # calm
        assert thresholds["pitch_high"] == 270.0  # average of stressed and anxious
        assert thresholds["energy_low"] == 0.015  # fatigued
        assert thresholds["energy_high"] == 0.09  # stressed
        assert thresholds["rate_slow"] == 1.6  # average of fatigued and calm
        assert thresholds["rate_fast"] == 5.0  # anxious
    
    def test_get_emotion_config_default(self, temp_calibration_file):
        """Test getting emotion config when no calibration exists."""
        from core.config import EMOTION_CONFIG
        
        config = get_emotion_config("nonexistent_user")
        
        # Should return default config
        assert config == EMOTION_CONFIG
    
    def test_get_emotion_config_calibrated(self, temp_calibration_file):
        """Test getting emotion config with calibration data."""
        from core.config import EMOTION_CONFIG
        
        # Create calibration data
        test_data = {
            "user1": {
                "calm": {
                    "pitch_mean": 150.0,
                    "energy_mean": 0.03,
                    "rate_mean": 2.0
                },
                "stressed": {
                    "pitch_mean": 280.0,
                    "energy_mean": 0.09,
                    "rate_mean": 4.5
                }
            }
        }
        save_calibration_data(test_data)
        
        config = get_emotion_config("user1")
        
        # Should have calibrated thresholds
        assert config.pitch_low == 150.0
        assert config.pitch_high == 280.0
        
        # Other values should remain from default
        assert config.confidence_threshold == EMOTION_CONFIG.confidence_threshold


class TestCalibrationIntegration:
    """Integration tests for calibration with emotion classification."""
    
    def test_classify_emotion_with_calibration(self, temp_calibration_file, mock_extract_features):
        """Test that emotion classification uses calibrated thresholds."""
        from core.emotion import classify_emotion
        
        # Create calibration data with custom thresholds
        test_data = {
            "user1": {
                "calm": {
                    "pitch_mean": 140.0,  # Lower than default
                    "energy_mean": 0.025,
                    "rate_mean": 1.8
                },
                "stressed": {
                    "pitch_mean": 300.0,  # Higher than default
                    "energy_mean": 0.10,
                    "rate_mean": 5.0
                }
            }
        }
        save_calibration_data(test_data)
        
        # Test features that would be "neutral" with default thresholds
        # but should be "stressed" with calibrated thresholds
        features = {
            "pitch_mean": 270.0,  # Between default high (250) and calibrated high (300)
            "pitch_std": 30.0,
            "energy_rms": 0.07,
            "speech_rate": 4.2,
            "duration": 5.0,
            "zcr_mean": 0.06,
            "spectral_centroid_mean": 1800.0
        }
        
        # Classify with calibration
        result = classify_emotion(features, transcript=None, user_id="user1")
        
        # Should detect some emotion (not just neutral)
        assert result.label in ["calm", "stressed", "anxious", "fatigued", "neutral"]
        assert 0.0 <= result.confidence <= 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
