"""
Tests for Voice Biometrics Module

Requirements: 16.2, 16.3, 16.4, 16.5, 18.1
"""

import pytest
import numpy as np
from pathlib import Path
from datetime import datetime
import tempfile
import shutil

from core.voice_biometrics import (
    VoiceBiometrics,
    VoiceEmbedding,
    IdentificationResult,
    SPEECHBRAIN_AVAILABLE
)


@pytest.fixture
def temp_embeddings_dir():
    """Create temporary directory for embeddings."""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def voice_bio(temp_embeddings_dir):
    """Create VoiceBiometrics instance."""
    return VoiceBiometrics(embeddings_dir=temp_embeddings_dir)


# ─── Voice Embedding Tests ──────────────────────────────────────────────

def test_voice_embedding_creation():
    """Test VoiceEmbedding creation."""
    embedding = VoiceEmbedding(
        user_id="test_user",
        embedding=np.array([0.1, 0.2, 0.3]),
        sample_count=5,
        created_at="2024-01-01T00:00:00"
    )
    
    assert embedding.user_id == "test_user"
    assert len(embedding.embedding) == 3
    assert embedding.sample_count == 5


def test_voice_embedding_serialization():
    """Test VoiceEmbedding to/from dict."""
    embedding = VoiceEmbedding(
        user_id="test_user",
        embedding=np.array([0.1, 0.2, 0.3]),
        sample_count=5,
        created_at="2024-01-01T00:00:00"
    )
    
    # To dict
    data = embedding.to_dict()
    assert data['user_id'] == "test_user"
    assert isinstance(data['embedding'], list)
    
    # From dict
    restored = VoiceEmbedding.from_dict(data)
    assert restored.user_id == embedding.user_id
    assert np.allclose(restored.embedding, embedding.embedding)


# ─── VoiceBiometrics Initialization Tests ───────────────────────────────

def test_voice_biometrics_initialization(voice_bio):
    """Test VoiceBiometrics initialization."""
    assert voice_bio is not None
    assert voice_bio.embeddings_dir.exists()
    assert voice_bio.similarity_threshold == 0.25
    assert voice_bio.min_enrollment_samples == 5


def test_voice_biometrics_availability(voice_bio):
    """Test checking if voice biometrics is available."""
    is_available = voice_bio.is_available()
    assert isinstance(is_available, bool)
    
    if SPEECHBRAIN_AVAILABLE:
        # Should be available if SpeechBrain is installed
        assert is_available or voice_bio.model is None
    else:
        assert not is_available


# ─── Cosine Distance Tests ──────────────────────────────────────────────

def test_cosine_distance_identical(voice_bio):
    """Test cosine distance for identical embeddings."""
    emb = np.array([1.0, 2.0, 3.0])
    distance = voice_bio._cosine_distance(emb, emb)
    assert distance < 0.01  # Should be very close to 0


def test_cosine_distance_opposite(voice_bio):
    """Test cosine distance for opposite embeddings."""
    emb1 = np.array([1.0, 2.0, 3.0])
    emb2 = np.array([-1.0, -2.0, -3.0])
    distance = voice_bio._cosine_distance(emb1, emb2)
    assert distance > 1.9  # Should be close to 2


def test_cosine_distance_orthogonal(voice_bio):
    """Test cosine distance for orthogonal embeddings."""
    emb1 = np.array([1.0, 0.0, 0.0])
    emb2 = np.array([0.0, 1.0, 0.0])
    distance = voice_bio._cosine_distance(emb1, emb2)
    assert 0.9 < distance < 1.1  # Should be close to 1


# ─── Enrollment Tests (Mock) ────────────────────────────────────────────

def test_enroll_user_insufficient_samples(voice_bio):
    """Test enrollment with insufficient samples."""
    # Should fail with less than min_enrollment_samples
    success = voice_bio.enroll_user(
        user_id="test_user",
        audio_samples=["sample1.wav", "sample2.wav"],  # Only 2 samples
        created_at=datetime.now().isoformat()
    )
    
    # Will fail because we can't extract embeddings from non-existent files
    assert not success


def test_get_enrolled_users_empty(voice_bio):
    """Test getting enrolled users when none exist."""
    users = voice_bio.get_enrolled_users()
    assert users == []


def test_remove_user_not_enrolled(voice_bio):
    """Test removing user that doesn't exist."""
    success = voice_bio.remove_user("nonexistent_user")
    assert not success


# ─── Identification Tests (Mock) ────────────────────────────────────────

def test_identify_user_no_enrollments(voice_bio):
    """Test identification when no users are enrolled."""
    result = voice_bio.identify_user("test_audio.wav")
    
    assert isinstance(result, IdentificationResult)
    assert result.user_id is None
    assert not result.is_known_user
    assert result.confidence == 0.0


def test_verify_user_not_enrolled(voice_bio):
    """Test verification for non-enrolled user."""
    is_verified, confidence = voice_bio.verify_user("test_user", "test_audio.wav")
    
    assert not is_verified
    assert confidence == 0.0


# ─── Embedding Storage Tests ────────────────────────────────────────────

def test_save_and_load_embeddings(voice_bio):
    """Test saving and loading embeddings."""
    # Manually add an embedding
    embedding = VoiceEmbedding(
        user_id="test_user",
        embedding=np.array([0.1, 0.2, 0.3]),
        sample_count=5,
        created_at=datetime.now().isoformat()
    )
    voice_bio.user_embeddings["test_user"] = embedding
    
    # Save
    voice_bio._save_embeddings()
    
    # Create new instance and load
    new_voice_bio = VoiceBiometrics(embeddings_dir=voice_bio.embeddings_dir)
    
    # Check if loaded
    assert "test_user" in new_voice_bio.user_embeddings
    loaded_emb = new_voice_bio.user_embeddings["test_user"]
    assert loaded_emb.user_id == "test_user"
    assert np.allclose(loaded_emb.embedding, embedding.embedding)


def test_remove_user_enrolled(voice_bio):
    """Test removing enrolled user."""
    # Add user
    embedding = VoiceEmbedding(
        user_id="test_user",
        embedding=np.array([0.1, 0.2, 0.3]),
        sample_count=5,
        created_at=datetime.now().isoformat()
    )
    voice_bio.user_embeddings["test_user"] = embedding
    
    # Remove
    success = voice_bio.remove_user("test_user")
    assert success
    assert "test_user" not in voice_bio.user_embeddings


def test_get_enrolled_users_with_users(voice_bio):
    """Test getting enrolled users."""
    # Add users
    for i in range(3):
        embedding = VoiceEmbedding(
            user_id=f"user_{i}",
            embedding=np.random.rand(192),
            sample_count=5,
            created_at=datetime.now().isoformat()
        )
        voice_bio.user_embeddings[f"user_{i}"] = embedding
    
    users = voice_bio.get_enrolled_users()
    assert len(users) == 3
    assert "user_0" in users
    assert "user_1" in users
    assert "user_2" in users


# ─── Integration Tests (Conditional) ────────────────────────────────────

@pytest.mark.skipif(not SPEECHBRAIN_AVAILABLE, reason="SpeechBrain not available")
def test_model_loading(voice_bio):
    """Test that model loads successfully."""
    if voice_bio.model is not None:
        assert voice_bio.is_available()


# ─── Edge Cases ─────────────────────────────────────────────────────────

def test_identification_result_creation():
    """Test IdentificationResult creation."""
    result = IdentificationResult(
        user_id="test_user",
        confidence=0.95,
        is_known_user=True,
        similarity_scores={"test_user": 0.1, "other_user": 0.5}
    )
    
    assert result.user_id == "test_user"
    assert result.confidence == 0.95
    assert result.is_known_user
    assert len(result.similarity_scores) == 2


def test_cosine_distance_zero_vector(voice_bio):
    """Test cosine distance with zero vector."""
    emb1 = np.array([1.0, 2.0, 3.0])
    emb2 = np.array([0.0, 0.0, 0.0])
    
    # Should handle zero vector gracefully (due to epsilon)
    distance = voice_bio._cosine_distance(emb1, emb2)
    assert isinstance(distance, float)


def test_embedding_persistence_across_instances(temp_embeddings_dir):
    """Test that embeddings persist across VoiceBiometrics instances."""
    # Create first instance and add embedding
    voice_bio1 = VoiceBiometrics(embeddings_dir=temp_embeddings_dir)
    embedding = VoiceEmbedding(
        user_id="persistent_user",
        embedding=np.array([0.5, 0.6, 0.7]),
        sample_count=5,
        created_at=datetime.now().isoformat()
    )
    voice_bio1.user_embeddings["persistent_user"] = embedding
    voice_bio1._save_embeddings()
    
    # Create second instance
    voice_bio2 = VoiceBiometrics(embeddings_dir=temp_embeddings_dir)
    
    # Check if embedding persisted
    assert "persistent_user" in voice_bio2.user_embeddings
    loaded = voice_bio2.user_embeddings["persistent_user"]
    assert np.allclose(loaded.embedding, embedding.embedding)
