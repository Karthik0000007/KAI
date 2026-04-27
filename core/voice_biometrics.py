"""
Voice Biometrics Module - Speaker Recognition for Multi-User Support

Uses SpeechBrain's ECAPA-TDNN model for speaker verification and identification.
Enables user identification from voice samples with high accuracy.

Requirements: 16.2, 16.3, 16.4, 16.5
"""

import logging
import numpy as np
from pathlib import Path
from typing import Optional, List, Tuple, Dict
from dataclasses import dataclass
import json
import hashlib

logger = logging.getLogger(__name__)

# Try to import SpeechBrain
try:
    import torch
    import torchaudio
    from speechbrain.pretrained import EncoderClassifier
    SPEECHBRAIN_AVAILABLE = True
except ImportError:
    SPEECHBRAIN_AVAILABLE = False
    logger.warning("SpeechBrain not available - voice biometrics disabled")
    logger.warning("Install with: pip install speechbrain torch torchaudio")


@dataclass
class VoiceEmbedding:
    """Represents a voice embedding for a user."""
    user_id: str
    embedding: np.ndarray
    sample_count: int
    created_at: str
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'user_id': self.user_id,
            'embedding': self.embedding.tolist(),
            'sample_count': self.sample_count,
            'created_at': self.created_at
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'VoiceEmbedding':
        """Create from dictionary."""
        return cls(
            user_id=data['user_id'],
            embedding=np.array(data['embedding']),
            sample_count=data['sample_count'],
            created_at=data['created_at']
        )


@dataclass
class IdentificationResult:
    """Result of speaker identification."""
    user_id: Optional[str]
    confidence: float
    is_known_user: bool
    similarity_scores: Dict[str, float]


class VoiceBiometrics:
    """
    Voice biometrics system for speaker recognition.
    
    Uses SpeechBrain's ECAPA-TDNN model for extracting speaker embeddings
    and cosine similarity for verification/identification.
    
    Requirements: 16.2, 16.3
    """
    
    def __init__(
        self,
        model_name: str = "speechbrain/spkrec-ecapa-voxceleb",
        embeddings_dir: Path = Path("data/voice_embeddings"),
        similarity_threshold: float = 0.25,  # Cosine distance threshold
        min_enrollment_samples: int = 5
    ):
        """
        Initialize voice biometrics system.
        
        Args:
            model_name: SpeechBrain model identifier
            embeddings_dir: Directory to store user embeddings
            similarity_threshold: Threshold for user identification (lower = more similar)
            min_enrollment_samples: Minimum samples required for enrollment
        """
        self.model_name = model_name
        self.embeddings_dir = Path(embeddings_dir)
        self.embeddings_dir.mkdir(parents=True, exist_ok=True)
        self.similarity_threshold = similarity_threshold
        self.min_enrollment_samples = min_enrollment_samples
        
        self.model = None
        self.user_embeddings: Dict[str, VoiceEmbedding] = {}
        
        if SPEECHBRAIN_AVAILABLE:
            self._load_model()
            self._load_embeddings()
        else:
            logger.warning("Voice biometrics disabled - SpeechBrain not available")
    
    def _load_model(self):
        """Load SpeechBrain speaker recognition model."""
        try:
            logger.info(f"Loading SpeechBrain model: {self.model_name}")
            self.model = EncoderClassifier.from_hparams(
                source=self.model_name,
                savedir=Path("data/models/speechbrain")
            )
            logger.info("SpeechBrain model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading SpeechBrain model: {e}")
            self.model = None
    
    def _load_embeddings(self):
        """Load stored user embeddings from disk."""
        embeddings_file = self.embeddings_dir / "user_embeddings.json"
        if embeddings_file.exists():
            try:
                with open(embeddings_file, 'r') as f:
                    data = json.load(f)
                    for user_data in data:
                        embedding = VoiceEmbedding.from_dict(user_data)
                        self.user_embeddings[embedding.user_id] = embedding
                logger.info(f"Loaded {len(self.user_embeddings)} user embeddings")
            except Exception as e:
                logger.error(f"Error loading embeddings: {e}")
    
    def _save_embeddings(self):
        """Save user embeddings to disk."""
        embeddings_file = self.embeddings_dir / "user_embeddings.json"
        try:
            data = [emb.to_dict() for emb in self.user_embeddings.values()]
            with open(embeddings_file, 'w') as f:
                json.dump(data, f, indent=2)
            logger.info(f"Saved {len(self.user_embeddings)} user embeddings")
        except Exception as e:
            logger.error(f"Error saving embeddings: {e}")
    
    def extract_embedding(self, audio_path: str) -> Optional[np.ndarray]:
        """
        Extract speaker embedding from audio file.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Speaker embedding as numpy array, or None if extraction fails
            
        Requirements: 16.2
        """
        if not SPEECHBRAIN_AVAILABLE or self.model is None:
            logger.warning("Cannot extract embedding - SpeechBrain not available")
            return None
        
        try:
            # Load audio
            signal, sr = torchaudio.load(audio_path)
            
            # Resample to 16kHz if needed
            if sr != 16000:
                resampler = torchaudio.transforms.Resample(sr, 16000)
                signal = resampler(signal)
            
            # Extract embedding
            with torch.no_grad():
                embedding = self.model.encode_batch(signal)
                embedding = embedding.squeeze().cpu().numpy()
            
            return embedding
            
        except Exception as e:
            logger.error(f"Error extracting embedding: {e}")
            return None
    
    def enroll_user(
        self,
        user_id: str,
        audio_samples: List[str],
        created_at: str
    ) -> bool:
        """
        Enroll a new user with voice samples.
        
        Args:
            user_id: Unique user identifier
            audio_samples: List of paths to audio files (5-10 samples recommended)
            created_at: Timestamp of enrollment
            
        Returns:
            True if enrollment successful, False otherwise
            
        Requirements: 16.5
        """
        if not SPEECHBRAIN_AVAILABLE or self.model is None:
            logger.error("Cannot enroll user - SpeechBrain not available")
            return False
        
        if len(audio_samples) < self.min_enrollment_samples:
            logger.error(
                f"Insufficient samples for enrollment: {len(audio_samples)} "
                f"(minimum: {self.min_enrollment_samples})"
            )
            return False
        
        # Extract embeddings from all samples
        embeddings = []
        for audio_path in audio_samples:
            embedding = self.extract_embedding(audio_path)
            if embedding is not None:
                embeddings.append(embedding)
        
        if len(embeddings) < self.min_enrollment_samples:
            logger.error(
                f"Failed to extract sufficient embeddings: {len(embeddings)} "
                f"(minimum: {self.min_enrollment_samples})"
            )
            return False
        
        # Average embeddings to create user template
        avg_embedding = np.mean(embeddings, axis=0)
        
        # Store user embedding
        voice_embedding = VoiceEmbedding(
            user_id=user_id,
            embedding=avg_embedding,
            sample_count=len(embeddings),
            created_at=created_at
        )
        
        self.user_embeddings[user_id] = voice_embedding
        self._save_embeddings()
        
        logger.info(f"User {user_id} enrolled with {len(embeddings)} samples")
        return True
    
    def identify_user(
        self,
        audio_path: str,
        top_k: int = 1
    ) -> IdentificationResult:
        """
        Identify user from audio sample.
        
        Args:
            audio_path: Path to audio file
            top_k: Number of top matches to consider
            
        Returns:
            IdentificationResult with user_id and confidence
            
        Requirements: 16.3, 16.4
        """
        if not SPEECHBRAIN_AVAILABLE or self.model is None:
            logger.warning("Cannot identify user - SpeechBrain not available")
            return IdentificationResult(
                user_id=None,
                confidence=0.0,
                is_known_user=False,
                similarity_scores={}
            )
        
        if not self.user_embeddings:
            logger.warning("No enrolled users - cannot identify")
            return IdentificationResult(
                user_id=None,
                confidence=0.0,
                is_known_user=False,
                similarity_scores={}
            )
        
        # Extract embedding from audio
        query_embedding = self.extract_embedding(audio_path)
        if query_embedding is None:
            logger.error("Failed to extract embedding from audio")
            return IdentificationResult(
                user_id=None,
                confidence=0.0,
                is_known_user=False,
                similarity_scores={}
            )
        
        # Compute similarity with all enrolled users
        similarities = {}
        for user_id, voice_emb in self.user_embeddings.items():
            # Cosine distance (lower = more similar)
            distance = self._cosine_distance(query_embedding, voice_emb.embedding)
            similarities[user_id] = distance
        
        # Find best match
        best_user = min(similarities, key=similarities.get)
        best_distance = similarities[best_user]
        
        # Check if match is above threshold
        is_known = best_distance < self.similarity_threshold
        confidence = 1.0 - best_distance  # Convert distance to confidence
        
        result = IdentificationResult(
            user_id=best_user if is_known else None,
            confidence=confidence,
            is_known_user=is_known,
            similarity_scores=similarities
        )
        
        if is_known:
            logger.info(
                f"User identified: {best_user} "
                f"(confidence: {confidence:.3f}, distance: {best_distance:.3f})"
            )
        else:
            logger.info(
                f"Unknown user (best match: {best_user}, "
                f"distance: {best_distance:.3f} > threshold: {self.similarity_threshold})"
            )
        
        return result
    
    def verify_user(
        self,
        user_id: str,
        audio_path: str
    ) -> Tuple[bool, float]:
        """
        Verify if audio matches claimed user identity.
        
        Args:
            user_id: Claimed user identifier
            audio_path: Path to audio file
            
        Returns:
            Tuple of (is_verified, confidence)
        """
        if user_id not in self.user_embeddings:
            logger.warning(f"User {user_id} not enrolled")
            return False, 0.0
        
        result = self.identify_user(audio_path)
        is_verified = result.user_id == user_id
        confidence = result.confidence if is_verified else 0.0
        
        return is_verified, confidence
    
    def _cosine_distance(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """
        Compute cosine distance between two embeddings.
        
        Returns value in [0, 2] where 0 = identical, 2 = opposite
        """
        # Normalize embeddings
        emb1_norm = emb1 / (np.linalg.norm(emb1) + 1e-8)
        emb2_norm = emb2 / (np.linalg.norm(emb2) + 1e-8)
        
        # Cosine similarity
        similarity = np.dot(emb1_norm, emb2_norm)
        
        # Convert to distance (0 = identical, 2 = opposite)
        distance = 1.0 - similarity
        
        return float(distance)
    
    def remove_user(self, user_id: str) -> bool:
        """Remove user enrollment."""
        if user_id in self.user_embeddings:
            del self.user_embeddings[user_id]
            self._save_embeddings()
            logger.info(f"Removed user {user_id}")
            return True
        return False
    
    def get_enrolled_users(self) -> List[str]:
        """Get list of enrolled user IDs."""
        return list(self.user_embeddings.keys())
    
    def is_available(self) -> bool:
        """Check if voice biometrics is available."""
        return SPEECHBRAIN_AVAILABLE and self.model is not None
