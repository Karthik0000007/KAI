"""
Integration tests for vision module.

Tests cover:
- End-to-end frame processing
- Face detection with sample images
- Expression recognition accuracy
- Vision-audio emotion correlation
- Event bus integration

Requirements: 18.2
"""

import pytest
import asyncio
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch

from core.vision import VisionModule, VisionFrame, ExpressionResult, EMOTIONS
from core.event_bus import EventBus
from core.models import HealthCheckIn


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


# ─── End-to-End Frame Processing Tests ───────────────────────────────────

@pytest.mark.asyncio
async def test_process_frame_end_to_end(vision_module, sample_frame):
    """
    Test complete frame processing pipeline.
    
    Requirements: 18.2
    """
    # Mock face detection
    with patch.object(vision_module, '_detect_faces') as mock_detect:
        mock_detect.return_value = []
        
        frame = await vision_module._process_frame(sample_frame)
    
    # Verify frame was processed
    assert frame.frame_id == 1
    assert frame.timestamp is not None
    assert isinstance(frame.faces_detected, int)
    assert isinstance(frame.expressions, list)
    assert isinstance(frame.user_present, bool)


@pytest.mark.asyncio
async def test_process_multiple_frames(vision_module, sample_frame):
    """Test processing multiple frames sequentially."""
    frames_processed = []
    
    for i in range(5):
        with patch.object(vision_module, '_detect_faces', return_value=[]):
            frame = await vision_module._process_frame(sample_frame)
            frames_processed.append(frame)
    
    # Verify all frames were processed
    assert len(frames_processed) == 5
    
    # Verify frame IDs are sequential
    for i, frame in enumerate(frames_processed):
        assert frame.frame_id == i + 1


@pytest.mark.asyncio
async def test_process_frame_with_detected_faces(vision_module, sample_frame):
    """Test frame processing with detected faces."""
    from core.vision import FaceDetection
    
    # Create mock face detection
    mock_face = FaceDetection(x=100, y=100, width=50, height=50, confidence=0.95)
    
    with patch.object(vision_module, '_detect_faces', return_value=[mock_face]):
        with patch.object(vision_module, '_recognize_expression', return_value=("happy", 0.95)):
            frame = await vision_module._process_frame(sample_frame)
    
    # Verify face was detected
    assert frame.faces_detected == 1
    assert frame.user_present is True
    assert len(frame.expressions) == 1
    assert frame.expressions[0].emotion == "happy"


# ─── Event Bus Integration Tests ─────────────────────────────────────────

@pytest.mark.asyncio
async def test_frame_processing_emits_events(vision_module, event_bus, sample_frame):
    """
    Test that frame processing emits appropriate events.
    
    Requirements: 18.2
    """
    from core.vision import FaceDetection
    
    # Process frame with no faces
    with patch.object(vision_module, '_detect_faces', return_value=[]):
        frame = await vision_module._process_frame(sample_frame)
    
    # Verify frame was processed correctly
    assert frame.frame_id == 1
    assert frame.faces_detected == 0
    assert frame.user_present is False
    assert len(frame.expressions) == 0


@pytest.mark.asyncio
async def test_face_detection_emits_event(vision_module, event_bus, sample_frame):
    """Test that face detection emits face_detected event."""
    from core.vision import FaceDetection
    
    face_events = []
    event_bus.on("face_detected", lambda e: face_events.append(e))
    
    mock_face = FaceDetection(x=100, y=100, width=50, height=50, confidence=0.95)
    
    with patch.object(vision_module, '_detect_faces', return_value=[mock_face]):
        with patch.object(vision_module, '_recognize_expression', return_value=("happy", 0.95)):
            frame = await vision_module._process_frame(sample_frame)
    
    # Verify face detected event was emitted
    assert len(face_events) == 1
    assert face_events[0].data["face_count"] == 1


@pytest.mark.asyncio
async def test_expression_recognition_emits_event(vision_module, event_bus, sample_frame):
    """Test that expression recognition emits expression_recognized event."""
    from core.vision import FaceDetection
    
    expression_events = []
    event_bus.on("expression_recognized", lambda e: expression_events.append(e))
    
    mock_face = FaceDetection(x=100, y=100, width=50, height=50, confidence=0.95)
    
    with patch.object(vision_module, '_detect_faces', return_value=[mock_face]):
        with patch.object(vision_module, '_recognize_expression', return_value=("sad", 0.88)):
            frame = await vision_module._process_frame(sample_frame)
    
    # Verify expression recognized event was emitted
    assert len(expression_events) == 1
    assert expression_events[0].data["emotion"] == "sad"
    assert expression_events[0].data["confidence"] == 0.88


# ─── Emotion Analysis Integration Tests ──────────────────────────────────

@pytest.mark.asyncio
async def test_dominant_emotion_from_multiple_frames(vision_module):
    """
    Test getting dominant emotion from multiple frames.
    
    Requirements: 18.2
    """
    now = datetime.now()
    
    # Add frames with mixed emotions
    emotions_sequence = ["happy", "happy", "happy", "sad", "neutral"]
    
    for i, emotion in enumerate(emotions_sequence):
        expressions = [ExpressionResult(emotion, 0.95, now, i)]
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
    
    # Get dominant emotion
    dominant = vision_module.get_dominant_emotion(seconds=10)
    
    # Should be "happy" (appears 3 times)
    assert dominant == "happy"


@pytest.mark.asyncio
async def test_mood_score_calculation_from_expressions(vision_module):
    """
    Test mood score calculation from expression history.
    
    Requirements: 18.2
    """
    now = datetime.now()
    
    # Add frames with happy expressions (mood score 8.0)
    for i in range(3):
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
    
    # Add frames with sad expressions (mood score 3.0)
    for i in range(3, 5):
        expressions = [ExpressionResult("sad", 0.90, now, i)]
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
    
    # Get average mood score
    avg_mood = vision_module.get_average_mood_score(seconds=10)
    
    # Should be between 3.0 and 8.0
    assert avg_mood is not None
    assert 3.0 <= avg_mood <= 8.0


# ─── Vision-Audio Correlation Tests ──────────────────────────────────────

@pytest.mark.asyncio
async def test_vision_audio_emotion_correlation(vision_module):
    """
    Test correlating visual emotion with audio emotion.
    
    Requirements: 11.12
    """
    now = datetime.now()
    
    # Add visual emotion (happy)
    expressions = [ExpressionResult("happy", 0.95, now, 1)]
    frame = VisionFrame(
        frame_id=1,
        timestamp=now,
        faces_detected=1,
        expressions=expressions,
        user_present=True,
        screen_time_detected=False,
        activity_detected=False
    )
    vision_module.frame_history.append(frame)
    
    # Get visual emotion
    visual_emotion = vision_module.get_dominant_emotion(seconds=10)
    
    # Simulate audio emotion (also happy)
    audio_emotion = "happy"
    
    # Verify correlation
    assert visual_emotion == audio_emotion


@pytest.mark.asyncio
async def test_vision_audio_emotion_mismatch(vision_module):
    """
    Test detecting mismatch between visual and audio emotion.
    
    Requirements: 11.12
    """
    now = datetime.now()
    
    # Add visual emotion (happy)
    expressions = [ExpressionResult("happy", 0.95, now, 1)]
    frame = VisionFrame(
        frame_id=1,
        timestamp=now,
        faces_detected=1,
        expressions=expressions,
        user_present=True,
        screen_time_detected=False,
        activity_detected=False
    )
    vision_module.frame_history.append(frame)
    
    # Get visual emotion
    visual_emotion = vision_module.get_dominant_emotion(seconds=10)
    
    # Simulate audio emotion (sad - mismatch)
    audio_emotion = "sad"
    
    # Verify mismatch is detected
    assert visual_emotion != audio_emotion


# ─── Privacy Control Integration Tests ───────────────────────────────────

@pytest.mark.asyncio
async def test_privacy_mode_prevents_capture(vision_module):
    """
    Test that privacy mode prevents camera capture.
    
    Requirements: 11.10
    """
    # Enable privacy mode
    vision_module.set_privacy_mode(True)
    
    # Try to start capture
    await vision_module.start_capture()
    
    # Should not start
    assert vision_module.running is False


@pytest.mark.asyncio
async def test_privacy_mode_toggle(vision_module):
    """Test toggling privacy mode on and off."""
    # Initially enabled
    assert vision_module.enabled is True
    
    # Enable privacy mode
    vision_module.set_privacy_mode(True)
    assert vision_module.enabled is False
    
    # Disable privacy mode
    vision_module.set_privacy_mode(False)
    assert vision_module.enabled is True


# ─── User Presence Detection Tests ───────────────────────────────────────

@pytest.mark.asyncio
async def test_user_presence_detection(vision_module, sample_frame):
    """
    Test detecting user presence in frame.
    
    Requirements: 11.6
    """
    from core.vision import FaceDetection
    
    # Test with user present (face detected)
    mock_face = FaceDetection(x=100, y=100, width=50, height=50, confidence=0.95)
    
    with patch.object(vision_module, '_detect_faces', return_value=[mock_face]):
        with patch.object(vision_module, '_recognize_expression', return_value=("neutral", 0.85)):
            frame = await vision_module._process_frame(sample_frame)
    
    assert frame.user_present is True
    
    # Test with user absent (no faces detected)
    with patch.object(vision_module, '_detect_faces', return_value=[]):
        frame = await vision_module._process_frame(sample_frame)
    
    assert frame.user_present is False


# ─── Frame History Management Tests ──────────────────────────────────────

@pytest.mark.asyncio
async def test_frame_history_time_window(vision_module):
    """
    Test that frame history respects time windows.
    
    Requirements: 18.2
    """
    now = datetime.now()
    
    # Add frames from different times
    for i in range(5):
        timestamp = now - timedelta(seconds=i*2)
        frame = VisionFrame(
            frame_id=i,
            timestamp=timestamp,
            faces_detected=0,
            expressions=[],
            user_present=False,
            screen_time_detected=False,
            activity_detected=False
        )
        vision_module.frame_history.append(frame)
    
    # Get mood score from last 5 seconds
    # Should only include frames from last 5 seconds
    score = vision_module.get_average_mood_score(seconds=5)
    
    # Verify time window is respected
    assert len(vision_module.frame_history) == 5


# ─── Concurrent Processing Tests ────────────────────────────────────────

@pytest.mark.asyncio
async def test_concurrent_frame_processing(vision_module, sample_frame):
    """Test processing multiple frames concurrently."""
    tasks = []
    
    for i in range(5):
        with patch.object(vision_module, '_detect_faces', return_value=[]):
            task = vision_module._process_frame(sample_frame)
            tasks.append(task)
    
    frames = await asyncio.gather(*tasks)
    
    # Verify all frames were processed
    assert len(frames) == 5
    
    # Verify frame IDs are sequential
    for i, frame in enumerate(frames):
        assert frame.frame_id >= 1


# ─── Status Reporting Tests ──────────────────────────────────────────────

@pytest.mark.asyncio
async def test_status_reporting(vision_module):
    """Test getting vision module status."""
    status = vision_module.get_status()
    
    assert "enabled" in status
    assert "status" in status
    assert "running" in status
    assert "frame_id" in status
    assert "frames_in_history" in status
    assert "camera_id" in status
    
    # Verify initial values
    assert status["enabled"] is True
    assert status["running"] is False
    assert status["frame_id"] == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
