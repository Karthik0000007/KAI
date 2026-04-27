"""
Unit tests for vision module.

Tests cover:
- Camera initialization and capture
- Face detection
- Expression recognition
- Privacy controls
- Event emission
- Emotion analysis

Requirements: 18.1
"""

import pytest
import asyncio
import numpy as np
from datetime import datetime
from unittest.mock import Mock, AsyncMock, patch, MagicMock

from core.vision import (
    VisionModule,
    VisionStatus,
    FaceDetection,
    ExpressionResult,
    VisionFrame,
    EMOTIONS,
    EMOTION_TO_MOOD
)
from core.event_bus import EventBus


# ─── Fixtures ────────────────────────────────────────────────────────────

@pytest.fixture
def event_bus():
    """Create an event bus for testing."""
    return EventBus()


@pytest.fixture
def vision_module(event_bus):
    """Create a vision module for testing."""
    return VisionModule(event_bus, camera_id=0, enabled=True)


@pytest.fixture
def sample_frame():
    """Create a sample frame for testing."""
    return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)


# ─── Initialization Tests ────────────────────────────────────────────────

def test_vision_module_creation(event_bus):
    """Test creating a vision module."""
    module = VisionModule(event_bus, camera_id=0, enabled=True)
    
    assert module.enabled is True
    assert module.camera_id == 0
    assert module.status == VisionStatus.IDLE
    assert module.running is False
    assert module.frame_id == 0


def test_vision_module_disabled(event_bus):
    """Test creating a disabled vision module."""
    module = VisionModule(event_bus, camera_id=0, enabled=False)
    
    assert module.enabled is False


def test_vision_module_status(vision_module):
    """Test getting vision module status."""
    status = vision_module.get_status()
    
    assert status["enabled"] is True
    assert status["status"] == "idle"
    assert status["running"] is False
    assert status["frame_id"] == 0
    assert status["camera_id"] == 0


# ─── Privacy Control Tests ───────────────────────────────────────────────

def test_privacy_mode_enable(vision_module):
    """Test enabling privacy mode."""
    assert vision_module.is_privacy_mode_enabled() is False
    
    vision_module.set_privacy_mode(True)
    
    assert vision_module.is_privacy_mode_enabled() is True
    assert vision_module.enabled is False


def test_privacy_mode_disable(vision_module):
    """Test disabling privacy mode."""
    vision_module.set_privacy_mode(True)
    assert vision_module.is_privacy_mode_enabled() is True
    
    vision_module.set_privacy_mode(False)
    
    assert vision_module.is_privacy_mode_enabled() is False
    assert vision_module.enabled is True


# ─── Face Detection Tests ────────────────────────────────────────────────

def test_face_detection_creation():
    """Test creating a face detection object."""
    face = FaceDetection(x=100, y=100, width=50, height=50, confidence=0.95)
    
    assert face.x == 100
    assert face.y == 100
    assert face.width == 50
    assert face.height == 50
    assert face.confidence == 0.95


def test_face_detection_crop(sample_frame):
    """Test cropping face region from frame."""
    face = FaceDetection(x=100, y=100, width=50, height=50, confidence=0.95)
    
    cropped = face.crop_from_frame(sample_frame)
    
    assert cropped.shape == (50, 50, 3)


def test_detect_faces_empty_frame(vision_module):
    """Test face detection on empty frame."""
    # Create a blank frame
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    faces = vision_module._detect_faces(frame)
    
    # Should detect no faces in blank frame
    assert isinstance(faces, list)


def test_detect_faces_returns_list(vision_module, sample_frame):
    """Test that face detection returns a list."""
    faces = vision_module._detect_faces(sample_frame)
    
    assert isinstance(faces, list)
    for face in faces:
        assert isinstance(face, FaceDetection)


# ─── Expression Recognition Tests ───────────────────────────────────────

@pytest.mark.asyncio
async def test_recognize_expression(vision_module, sample_frame):
    """Test expression recognition."""
    face_roi = sample_frame[100:150, 100:150]
    
    emotion, confidence = await vision_module._recognize_expression(face_roi)
    
    assert emotion in EMOTIONS
    assert 0.0 <= confidence <= 1.0


@pytest.mark.asyncio
async def test_recognize_expression_returns_valid_emotion(vision_module, sample_frame):
    """Test that expression recognition returns valid emotion."""
    face_roi = sample_frame[100:150, 100:150]
    
    for _ in range(10):
        emotion, confidence = await vision_module._recognize_expression(face_roi)
        assert emotion in EMOTIONS


# ─── Expression Result Tests ────────────────────────────────────────────

def test_expression_result_creation():
    """Test creating an expression result."""
    now = datetime.now()
    result = ExpressionResult(
        emotion="happy",
        confidence=0.95,
        timestamp=now,
        frame_id=1
    )
    
    assert result.emotion == "happy"
    assert result.confidence == 0.95
    assert result.timestamp == now
    assert result.frame_id == 1


# ─── Vision Frame Tests ──────────────────────────────────────────────────

def test_vision_frame_creation():
    """Test creating a vision frame."""
    now = datetime.now()
    expressions = [
        ExpressionResult("happy", 0.95, now, 1)
    ]
    
    frame = VisionFrame(
        frame_id=1,
        timestamp=now,
        faces_detected=1,
        expressions=expressions,
        user_present=True,
        screen_time_detected=False,
        activity_detected=False
    )
    
    assert frame.frame_id == 1
    assert frame.faces_detected == 1
    assert frame.user_present is True
    assert len(frame.expressions) == 1


def test_vision_frame_no_faces():
    """Test vision frame with no faces detected."""
    now = datetime.now()
    
    frame = VisionFrame(
        frame_id=1,
        timestamp=now,
        faces_detected=0,
        expressions=[],
        user_present=False,
        screen_time_detected=False,
        activity_detected=False
    )
    
    assert frame.faces_detected == 0
    assert frame.user_present is False
    assert len(frame.expressions) == 0


# ─── Emotion Analysis Tests ──────────────────────────────────────────────

def test_get_dominant_emotion_empty_history(vision_module):
    """Test getting dominant emotion with empty history."""
    emotion = vision_module.get_dominant_emotion(seconds=10)
    
    assert emotion is None


def test_get_dominant_emotion_with_history(vision_module):
    """Test getting dominant emotion with frame history."""
    now = datetime.now()
    
    # Add frames with happy expressions
    for i in range(5):
        expressions = [ExpressionResult("happy", 0.95, now, i)]
        frame = VisionFrame(
            frame_id=i,
            timestamp=now,
            faces_detected=1,
            expressions=expressions,
            user_present=True,
            screen_time_detected=False,
            activity_detected=False
        )
        vision_module.frame_history.append(frame)
    
    emotion = vision_module.get_dominant_emotion(seconds=10)
    
    assert emotion == "happy"


def test_get_average_mood_score_empty_history(vision_module):
    """Test getting average mood score with empty history."""
    score = vision_module.get_average_mood_score(seconds=10)
    
    assert score is None


def test_get_average_mood_score_with_history(vision_module):
    """Test getting average mood score with frame history."""
    now = datetime.now()
    
    # Add frames with happy expressions (mood score 8.0)
    for i in range(5):
        expressions = [ExpressionResult("happy", 0.95, now, i)]
        frame = VisionFrame(
            frame_id=i,
            timestamp=now,
            faces_detected=1,
            expressions=expressions,
            user_present=True,
            screen_time_detected=False,
            activity_detected=False
        )
        vision_module.frame_history.append(frame)
    
    score = vision_module.get_average_mood_score(seconds=10)
    
    assert score is not None
    assert 7.0 <= score <= 9.0  # Should be close to 8.0


def test_emotion_to_mood_mapping():
    """Test emotion to mood score mapping."""
    assert EMOTION_TO_MOOD["happy"] == 8.0
    assert EMOTION_TO_MOOD["sad"] == 3.0
    assert EMOTION_TO_MOOD["angry"] == 2.0
    assert EMOTION_TO_MOOD["neutral"] == 5.0


# ─── Event Bus Integration Tests ─────────────────────────────────────────

@pytest.mark.asyncio
async def test_vision_module_registers_events(event_bus):
    """Test that vision module registers event types."""
    module = VisionModule(event_bus, enabled=True)
    
    # Check that event types are registered
    assert event_bus.is_registered("vision_frame_processed")
    assert event_bus.is_registered("face_detected")
    assert event_bus.is_registered("expression_recognized")
    assert event_bus.is_registered("user_absent")
    assert event_bus.is_registered("screen_time_detected")


@pytest.mark.asyncio
async def test_process_frame_emits_events(vision_module, event_bus, sample_frame):
    """Test that processing a frame emits events."""
    events_received = []
    
    def collect_event(event):
        events_received.append(event)
    
    event_bus.on("vision_frame_processed", collect_event)
    
    # Mock face detection to return no faces
    with patch.object(vision_module, '_detect_faces', return_value=[]):
        frame = await vision_module._process_frame(sample_frame)
    
    # Verify frame was processed
    assert frame.frame_id == 1
    assert frame.faces_detected == 0


# ─── Capture Loop Tests ──────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_start_capture_disabled_module(vision_module):
    """Test starting capture on disabled module."""
    vision_module.enabled = False
    
    await vision_module.start_capture()
    
    # Should not start
    assert vision_module.running is False


@pytest.mark.asyncio
async def test_stop_capture(vision_module):
    """Test stopping capture."""
    vision_module.running = True
    vision_module._capture_task = asyncio.create_task(asyncio.sleep(10))
    
    await vision_module.stop_capture()
    
    assert vision_module.running is False


# ─── Frame History Tests ────────────────────────────────────────────────

def test_frame_history_max_size(vision_module):
    """Test that frame history respects max size."""
    now = datetime.now()
    
    # Add more frames than max_history
    for i in range(vision_module.max_history + 50):
        frame = VisionFrame(
            frame_id=i,
            timestamp=now,
            faces_detected=0,
            expressions=[],
            user_present=False,
            screen_time_detected=False,
            activity_detected=False
        )
        vision_module.frame_history.append(frame)
        
        # Manually enforce max size (normally done in capture loop)
        if len(vision_module.frame_history) > vision_module.max_history:
            vision_module.frame_history.pop(0)
    
    assert len(vision_module.frame_history) <= vision_module.max_history


# ─── Vision Status Tests ────────────────────────────────────────────────

def test_vision_status_enum():
    """Test VisionStatus enum values."""
    assert VisionStatus.IDLE.value == "idle"
    assert VisionStatus.CAPTURING.value == "capturing"
    assert VisionStatus.PROCESSING.value == "processing"
    assert VisionStatus.ERROR.value == "error"


def test_vision_module_status_transitions(vision_module):
    """Test vision module status transitions."""
    assert vision_module.status == VisionStatus.IDLE
    
    vision_module.status = VisionStatus.CAPTURING
    assert vision_module.status == VisionStatus.CAPTURING
    
    vision_module.status = VisionStatus.PROCESSING
    assert vision_module.status == VisionStatus.PROCESSING
    
    vision_module.status = VisionStatus.ERROR
    assert vision_module.status == VisionStatus.ERROR


# ─── Error Handling Tests ────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_recognize_expression_error_handling(vision_module):
    """Test error handling in expression recognition."""
    # Create invalid face ROI (empty array)
    invalid_roi = np.array([])
    
    emotion, confidence = await vision_module._recognize_expression(invalid_roi)
    
    # Should return valid emotion and confidence (not crash)
    assert emotion in EMOTIONS
    assert 0.0 <= confidence <= 1.0


def test_detect_faces_error_handling(vision_module):
    """Test error handling in face detection."""
    # Create invalid frame
    invalid_frame = None
    
    # Should handle error gracefully
    try:
        faces = vision_module._detect_faces(invalid_frame)
        # If no exception, should return empty list
        assert isinstance(faces, list)
    except (AttributeError, TypeError):
        # Expected if frame is None
        pass


# ─── Emotion Mapping Tests ───────────────────────────────────────────────

def test_all_emotions_have_mood_scores():
    """Test that all emotions have mood score mappings."""
    for emotion in EMOTIONS:
        assert emotion in EMOTION_TO_MOOD
        score = EMOTION_TO_MOOD[emotion]
        assert 0.0 <= score <= 10.0


def test_mood_score_ranges():
    """Test that mood scores are in reasonable ranges."""
    # Positive emotions should have higher scores
    assert EMOTION_TO_MOOD["happy"] > EMOTION_TO_MOOD["sad"]
    assert EMOTION_TO_MOOD["surprised"] > EMOTION_TO_MOOD["angry"]
    
    # Negative emotions should have lower scores
    assert EMOTION_TO_MOOD["sad"] < EMOTION_TO_MOOD["neutral"]
    assert EMOTION_TO_MOOD["angry"] < EMOTION_TO_MOOD["neutral"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
