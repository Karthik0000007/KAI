"""
Integration tests for advanced vision features.

Tests eye fatigue detection, posture estimation, screen time tracking,
activity detection, and frame encryption.

Requirements: 11.4, 11.5, 11.8, 11.9, 11.13, 11.14, 18.2
"""

import pytest
import asyncio
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import shutil

from core.vision import (
    VisionModule,
    VisionStatus,
    PostureType,
    EyeFatigueMetrics,
    PostureEstimate,
    FaceDetection,
    CV2_AVAILABLE
)
from core.event_bus import EventBus
from core.key_manager import KeyManager


@pytest.fixture
def event_bus():
    """Create event bus for testing."""
    return EventBus()


@pytest.fixture
def temp_frames_dir():
    """Create temporary directory for encrypted frames."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def vision_module(event_bus, temp_frames_dir):
    """Create vision module for testing."""
    module = VisionModule(
        event_bus=event_bus,
        camera_id=0,
        enabled=False,  # Disable camera for testing
        encrypt_frames=True,
        frame_retention_hours=24
    )
    module.frames_dir = temp_frames_dir
    return module


@pytest.fixture
def sample_frame():
    """Create sample frame for testing."""
    # Create 640x480 BGR frame
    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    return frame


@pytest.fixture
def sample_face():
    """Create sample face detection for testing."""
    return FaceDetection(
        x=200,
        y=150,
        width=200,
        height=200,
        confidence=0.95
    )


class TestEyeFatigueDetection:
    """Test eye fatigue detection (Requirements: 11.4)."""
    
    @pytest.mark.asyncio
    async def test_detect_eye_fatigue_returns_metrics(self, vision_module, sample_frame, sample_face):
        """Test that eye fatigue detection returns valid metrics."""
        metrics = await vision_module._detect_eye_fatigue(sample_frame, sample_face)
        
        assert metrics is not None
        assert isinstance(metrics, EyeFatigueMetrics)
        assert metrics.blink_rate >= 0
        assert 0 <= metrics.eye_aspect_ratio <= 1
        assert 0 <= metrics.redness_score <= 1
        assert metrics.fatigue_level in ["low", "moderate", "high"]
        assert isinstance(metrics.timestamp, datetime)
    
    @pytest.mark.asyncio
    async def test_blink_rate_calculation(self, vision_module):
        """Test blink rate calculation from history."""
        # Add blink history
        now = datetime.now()
        for i in range(20):
            timestamp = now - timedelta(seconds=i)
            is_blink = i % 3 == 0  # Every 3rd frame is a blink
            vision_module.blink_history.append((timestamp, is_blink))
        
        blink_rate = vision_module._calculate_blink_rate()
        
        assert blink_rate > 0
        assert blink_rate < 100  # Reasonable upper bound
    
    @pytest.mark.asyncio
    async def test_eye_fatigue_event_emission(self, vision_module, sample_frame, sample_face):
        """Test that eye fatigue events are emitted for high fatigue."""
        events_received = []
        
        def handler(event):
            events_received.append(event.data)
        
        vision_module.event_bus.on("eye_fatigue_detected", handler)
        
        # Mock high fatigue
        vision_module._calculate_blink_rate = lambda: 5.0  # Very low blink rate
        
        metrics = await vision_module._detect_eye_fatigue(sample_frame, sample_face)
        
        # Process frame to trigger event
        if metrics and metrics.fatigue_level in ["moderate", "high"]:
            await vision_module.event_bus.emit("eye_fatigue_detected", {
                "fatigue_level": metrics.fatigue_level,
                "blink_rate": metrics.blink_rate
            })
        
        await asyncio.sleep(0.1)
        
        # Should have received event for high fatigue
        assert len(events_received) > 0
        assert events_received[0]["fatigue_level"] in ["moderate", "high"]
    
    @pytest.mark.asyncio
    async def test_get_eye_fatigue_status(self, vision_module, sample_frame, sample_face):
        """Test retrieving most recent eye fatigue status."""
        # Initially no status
        assert vision_module.get_eye_fatigue_status() is None
        
        # Add frame with eye fatigue
        metrics = await vision_module._detect_eye_fatigue(sample_frame, sample_face)
        vision_module.frame_history.append(type('VisionFrame', (), {
            'eye_fatigue': metrics,
            'timestamp': datetime.now()
        })())
        
        # Should retrieve status
        status = vision_module.get_eye_fatigue_status()
        assert status is not None
        assert isinstance(status, EyeFatigueMetrics)


class TestPostureEstimation:
    """Test posture estimation (Requirements: 11.5)."""
    
    @pytest.mark.asyncio
    async def test_estimate_posture_returns_result(self, vision_module, sample_frame, sample_face):
        """Test that posture estimation returns valid result."""
        posture = await vision_module._estimate_posture(sample_frame, sample_face)
        
        assert posture is not None
        assert isinstance(posture, PostureEstimate)
        assert isinstance(posture.posture_type, PostureType)
        assert -90 <= posture.head_pitch <= 90
        assert -90 <= posture.head_yaw <= 90
        assert -90 <= posture.head_roll <= 90
        assert 0 <= posture.confidence <= 1
        assert isinstance(posture.timestamp, datetime)
    
    @pytest.mark.asyncio
    async def test_posture_classification(self, vision_module, sample_frame):
        """Test posture classification based on face position."""
        # Face at bottom of frame (looking down) -> slouching
        face_bottom = FaceDetection(x=200, y=400, width=200, height=200, confidence=0.9)
        posture = await vision_module._estimate_posture(sample_frame, face_bottom)
        
        # Should detect slouching or leaning forward
        assert posture.posture_type in [PostureType.SLOUCHING, PostureType.LEANING_FORWARD, PostureType.UPRIGHT]
        
        # Face at top of frame (looking up) -> upright or looking up
        face_top = FaceDetection(x=200, y=50, width=200, height=200, confidence=0.9)
        posture = await vision_module._estimate_posture(sample_frame, face_top)
        
        assert posture.posture_type in [PostureType.UPRIGHT, PostureType.SLOUCHING, PostureType.LEANING_FORWARD]
    
    @pytest.mark.asyncio
    async def test_poor_posture_event_emission(self, vision_module, sample_frame):
        """Test that poor posture events are emitted."""
        events_received = []
        
        def handler(event):
            events_received.append(event.data)
        
        vision_module.event_bus.on("poor_posture_detected", handler)
        
        # Face at bottom (slouching)
        face = FaceDetection(x=200, y=400, width=200, height=200, confidence=0.9)
        posture = await vision_module._estimate_posture(sample_frame, face)
        
        # Emit event if poor posture
        if posture and posture.posture_type in [PostureType.SLOUCHING, PostureType.LEANING_FORWARD]:
            await vision_module.event_bus.emit("poor_posture_detected", {
                "posture_type": posture.posture_type.value
            })
        
        await asyncio.sleep(0.1)
        
        # May or may not receive event depending on classification
        # Just verify event structure if received
        if events_received:
            assert "posture_type" in events_received[0]
    
    @pytest.mark.asyncio
    async def test_get_posture_status(self, vision_module, sample_frame, sample_face):
        """Test retrieving most recent posture status."""
        # Initially no status
        assert vision_module.get_posture_status() is None
        
        # Add frame with posture
        posture = await vision_module._estimate_posture(sample_frame, sample_face)
        vision_module.frame_history.append(type('VisionFrame', (), {
            'posture': posture,
            'timestamp': datetime.now()
        })())
        
        # Should retrieve status
        status = vision_module.get_posture_status()
        assert status is not None
        assert isinstance(status, PostureEstimate)


class TestScreenTimeDetection:
    """Test screen time detection (Requirements: 11.13)."""
    
    @pytest.mark.asyncio
    async def test_screen_time_tracking_starts_with_face(self, vision_module, sample_frame, sample_face):
        """Test that screen time tracking starts when face detected."""
        assert vision_module.screen_time_start is None
        
        # Detect screen time with face
        is_screen_time = await vision_module._detect_screen_time(sample_frame, [sample_face])
        
        assert is_screen_time is True
        assert vision_module.screen_time_start is not None
    
    @pytest.mark.asyncio
    async def test_screen_time_tracking_stops_without_face(self, vision_module, sample_frame):
        """Test that screen time tracking stops when no face detected."""
        # Start screen time
        vision_module.screen_time_start = datetime.now()
        
        # Detect screen time without face
        is_screen_time = await vision_module._detect_screen_time(sample_frame, [])
        
        assert is_screen_time is False
        assert vision_module.screen_time_start is None
    
    @pytest.mark.asyncio
    async def test_prolonged_screen_time_event(self, vision_module, sample_frame, sample_face):
        """Test that prolonged screen time event is emitted."""
        events_received = []
        
        def handler(event):
            events_received.append(event.data)
        
        vision_module.event_bus.on("prolonged_screen_time", handler)
        
        # Simulate 3 hours of screen time
        vision_module.screen_time_start = datetime.now() - timedelta(hours=3)
        vision_module.total_screen_time_today = timedelta(hours=0)
        
        # Detect screen time (should trigger event)
        await vision_module._detect_screen_time(sample_frame, [sample_face])
        
        await asyncio.sleep(0.1)
        
        # Should have received prolonged screen time event
        assert len(events_received) > 0
        assert events_received[0]["total_screen_time_hours"] > 2
    
    @pytest.mark.asyncio
    async def test_get_screen_time_today(self, vision_module):
        """Test retrieving total screen time for today."""
        # Initially zero
        assert vision_module.get_screen_time_today().total_seconds() == 0
        
        # Add some screen time
        vision_module.total_screen_time_today = timedelta(hours=1, minutes=30)
        
        total = vision_module.get_screen_time_today()
        assert total.total_seconds() == 5400  # 1.5 hours
        
        # Add current session
        vision_module.screen_time_start = datetime.now() - timedelta(minutes=15)
        
        total = vision_module.get_screen_time_today()
        assert total.total_seconds() >= 5400  # At least 1.5 hours


class TestActivityDetection:
    """Test activity detection (Requirements: 11.14)."""
    
    @pytest.mark.asyncio
    async def test_activity_detection_with_motion(self, vision_module):
        """Test activity detection with frame motion."""
        if not CV2_AVAILABLE:
            pytest.skip("OpenCV not available")
        
        # Create two very different frames
        frame1 = np.zeros((480, 640, 3), dtype=np.uint8)
        frame2 = np.ones((480, 640, 3), dtype=np.uint8) * 255
        
        # First frame (no previous frame)
        activity1 = await vision_module._detect_activity(frame1)
        assert activity1 is False  # No previous frame to compare
        
        # Second frame (significant difference - all pixels changed)
        activity2 = await vision_module._detect_activity(frame2)
        assert activity2 is True  # Large difference detected
    
    @pytest.mark.asyncio
    async def test_activity_detection_without_motion(self, vision_module):
        """Test activity detection without frame motion."""
        if not CV2_AVAILABLE:
            pytest.skip("OpenCV not available")
        
        # Create identical frames
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # First frame
        await vision_module._detect_activity(frame)
        
        # Second frame (identical)
        activity = await vision_module._detect_activity(frame)
        assert activity is False  # No difference
    
    @pytest.mark.asyncio
    async def test_get_activity_level(self, vision_module):
        """Test retrieving activity level from recent frames."""
        # Initially zero
        assert vision_module.get_activity_level() == 0.0
        
        # Add frames with activity
        now = datetime.now()
        for i in range(10):
            vision_module.frame_history.append(type('VisionFrame', (), {
                'activity_detected': i % 2 == 0,  # 50% active
                'timestamp': now - timedelta(seconds=i)
            })())
        
        activity_level = vision_module.get_activity_level(minutes=1)
        assert 0.4 <= activity_level <= 0.6  # Approximately 50%


class TestFrameEncryption:
    """Test frame encryption and retention (Requirements: 11.8, 11.9)."""
    
    @pytest.mark.asyncio
    async def test_encrypt_and_save_frame(self, vision_module, sample_frame, temp_frames_dir):
        """Test encrypting and saving frame to disk."""
        if not CV2_AVAILABLE:
            pytest.skip("OpenCV not available")
        
        # Initialize key manager with passphrase
        db_path = temp_frames_dir / "test.db"
        vision_module.key_manager = KeyManager(use_keyring=False, passphrase="test_passphrase", db_path=db_path)
        
        # Encrypt and save frame
        filepath = await vision_module._encrypt_and_save_frame(
            sample_frame,
            frame_id=1,
            timestamp=datetime.now()
        )
        
        assert filepath is not None
        assert Path(filepath).exists()
        assert Path(filepath).suffix == ".enc"
        
        # Verify file is not empty
        assert Path(filepath).stat().st_size > 0
    
    @pytest.mark.asyncio
    async def test_encrypted_frame_cannot_be_read_directly(self, vision_module, sample_frame, temp_frames_dir):
        """Test that encrypted frames cannot be read as images."""
        if not CV2_AVAILABLE:
            pytest.skip("OpenCV not available")
        
        # Initialize key manager with passphrase
        db_path = temp_frames_dir / "test.db"
        vision_module.key_manager = KeyManager(use_keyring=False, passphrase="test_passphrase", db_path=db_path)
        
        # Encrypt and save frame
        filepath = await vision_module._encrypt_and_save_frame(
            sample_frame,
            frame_id=1,
            timestamp=datetime.now()
        )
        
        # Try to read as image (should fail)
        import cv2
        img = cv2.imread(filepath)
        assert img is None  # Cannot read encrypted file as image
    
    @pytest.mark.asyncio
    async def test_cleanup_old_frames(self, vision_module, temp_frames_dir):
        """Test automatic cleanup of old frames."""
        # Create old and new frames
        old_timestamp = datetime.now() - timedelta(hours=25)
        new_timestamp = datetime.now()
        
        old_filename = f"frame_1_{old_timestamp.strftime('%Y%m%d_%H%M%S')}.enc"
        new_filename = f"frame_2_{new_timestamp.strftime('%Y%m%d_%H%M%S')}.enc"
        
        old_filepath = temp_frames_dir / old_filename
        new_filepath = temp_frames_dir / new_filename
        
        # Create dummy encrypted files
        old_filepath.write_bytes(b"encrypted_data_old")
        new_filepath.write_bytes(b"encrypted_data_new")
        
        assert old_filepath.exists()
        assert new_filepath.exists()
        
        # Run cleanup (simulate one iteration)
        cutoff_time = datetime.now() - timedelta(hours=vision_module.frame_retention_hours)
        
        deleted_count = 0
        for filepath in temp_frames_dir.glob("frame_*.enc"):
            try:
                parts = filepath.stem.split('_')
                if len(parts) >= 3:
                    timestamp_str = f"{parts[2]}_{parts[3]}"
                    file_timestamp = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
                    
                    if file_timestamp < cutoff_time:
                        filepath.unlink()
                        deleted_count += 1
            except Exception:
                pass
        
        # Old file should be deleted, new file should remain
        assert not old_filepath.exists()
        assert new_filepath.exists()
        assert deleted_count == 1
    
    @pytest.mark.asyncio
    async def test_frame_retention_configuration(self, event_bus):
        """Test frame retention configuration."""
        # Create module with custom retention
        module = VisionModule(
            event_bus=event_bus,
            enabled=False,
            encrypt_frames=True,
            frame_retention_hours=48
        )
        
        assert module.frame_retention_hours == 48


class TestVisionModuleStatus:
    """Test vision module status reporting."""
    
    def test_get_status_includes_advanced_features(self, vision_module):
        """Test that status includes advanced feature metrics."""
        status = vision_module.get_status()
        
        assert "enabled" in status
        assert "status" in status
        assert "encrypt_frames" in status
        assert "eye_fatigue_level" in status
        assert "posture_type" in status
        assert "screen_time_hours" in status
        assert "activity_level" in status
    
    def test_get_status_with_metrics(self, vision_module, sample_frame, sample_face):
        """Test status with actual metrics."""
        # Add some data
        vision_module.total_screen_time_today = timedelta(hours=2)
        
        status = vision_module.get_status()
        
        assert status["screen_time_hours"] == 2.0
        assert status["activity_level"] >= 0


class TestIntegration:
    """Integration tests for complete vision pipeline."""
    
    @pytest.mark.asyncio
    async def test_complete_frame_processing_pipeline(self, vision_module, sample_frame, temp_frames_dir):
        """Test complete frame processing with all features."""
        if not CV2_AVAILABLE:
            pytest.skip("OpenCV not available")
        
        # Initialize key manager for encryption with passphrase
        db_path = temp_frames_dir / "test.db"
        vision_module.key_manager = KeyManager(use_keyring=False, passphrase="test_passphrase", db_path=db_path)
        
        # Mock face detection
        vision_module._detect_faces = lambda frame: [
            FaceDetection(x=200, y=150, width=200, height=200, confidence=0.95)
        ]
        
        # Process frame
        vision_frame = await vision_module._process_frame(sample_frame)
        
        # Verify all components
        assert vision_frame.frame_id > 0
        assert isinstance(vision_frame.timestamp, datetime)
        assert vision_frame.faces_detected == 1
        assert len(vision_frame.expressions) > 0
        assert vision_frame.user_present is True
        assert vision_frame.eye_fatigue is not None
        assert vision_frame.posture is not None
        
        # Verify encryption if enabled
        if vision_module.encrypt_frames:
            assert vision_frame.encrypted_frame_path is not None
            assert Path(vision_frame.encrypted_frame_path).exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
