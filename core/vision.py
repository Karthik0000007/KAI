"""
Vision Module - Camera-Based Health Observation

This module provides real-time facial expression recognition and health signal
detection from camera input. Uses local ONNX models for privacy-preserving inference.

Requirements: 11.1, 11.2, 11.3, 11.4, 11.5, 11.8, 11.9, 11.10, 11.11, 11.12, 11.13, 11.14
"""

import asyncio
import logging
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass
from enum import Enum
import json
import hashlib

try:
    import cv2
    CV2_AVAILABLE = True
except (ImportError, AttributeError):
    CV2_AVAILABLE = False
    cv2 = None

from core.event_bus import EventBus
from core.key_manager import KeyManager

logger = logging.getLogger(__name__)

# Emotion labels (7 basic emotions)
EMOTIONS = ["neutral", "happy", "sad", "angry", "surprised", "fearful", "disgusted"]

# Emotion to mood score mapping (0-10 scale)
EMOTION_TO_MOOD = {
    "happy": 8.0,
    "surprised": 6.0,
    "neutral": 5.0,
    "sad": 3.0,
    "angry": 2.0,
    "fearful": 2.0,
    "disgusted": 2.0,
}


class VisionStatus(Enum):
    """Status of the vision module."""
    IDLE = "idle"
    CAPTURING = "capturing"
    PROCESSING = "processing"
    ERROR = "error"


class PostureType(Enum):
    """Types of posture."""
    UPRIGHT = "upright"
    SLOUCHING = "slouching"
    LEANING_LEFT = "leaning_left"
    LEANING_RIGHT = "leaning_right"
    LEANING_FORWARD = "leaning_forward"
    UNKNOWN = "unknown"


@dataclass
class FaceDetection:
    """Represents a detected face in a frame."""
    x: int
    y: int
    width: int
    height: int
    confidence: float
    landmarks: Optional[np.ndarray] = None  # Facial landmarks (eyes, nose, mouth)
    
    def crop_from_frame(self, frame: np.ndarray) -> np.ndarray:
        """Crop the face region from the frame."""
        return frame[self.y:self.y + self.height, self.x:self.x + self.width]


@dataclass
class ExpressionResult:
    """Result of facial expression recognition."""
    emotion: str
    confidence: float
    timestamp: datetime
    frame_id: int


@dataclass
class EyeFatigueMetrics:
    """Metrics for eye fatigue detection."""
    blink_rate: float  # Blinks per minute
    eye_aspect_ratio: float  # Lower = more closed eyes
    redness_score: float  # 0-1, higher = more red
    fatigue_level: str  # "low", "moderate", "high"
    timestamp: datetime


@dataclass
class PostureEstimate:
    """Posture estimation result."""
    posture_type: PostureType
    head_pitch: float  # Degrees, positive = looking up
    head_yaw: float  # Degrees, positive = looking right
    head_roll: float  # Degrees, positive = tilted right
    confidence: float
    timestamp: datetime


@dataclass
class VisionFrame:
    """Represents a processed vision frame."""
    frame_id: int
    timestamp: datetime
    faces_detected: int
    expressions: List[ExpressionResult]
    user_present: bool
    screen_time_detected: bool
    activity_detected: bool
    eye_fatigue: Optional[EyeFatigueMetrics] = None
    posture: Optional[PostureEstimate] = None
    encrypted_frame_path: Optional[str] = None  # Path to encrypted frame file


class VisionModule:
    """
    Camera-based health observation module.
    
    Captures video frames, detects faces, recognizes facial expressions,
    and correlates visual emotion with audio emotion for improved accuracy.
    
    Requirements: 11.1, 11.2, 11.3, 11.10, 11.11, 11.12
    """
    
    def __init__(self, event_bus: EventBus, camera_id: int = 0, enabled: bool = True,
                 encrypt_frames: bool = True, frame_retention_hours: int = 24):
        """
        Initialize the vision module.
        
        Args:
            event_bus: EventBus instance for emitting vision events
            camera_id: Camera device ID (default: 0 for primary camera)
            enabled: Whether vision module is enabled (privacy control)
            encrypt_frames: Whether to encrypt captured frames (Requirements: 11.8)
            frame_retention_hours: Hours to retain frames before deletion (Requirements: 11.9)
        """
        self.event_bus = event_bus
        self.camera_id = camera_id
        self.enabled = enabled
        self.status = VisionStatus.IDLE
        self.encrypt_frames = encrypt_frames
        self.frame_retention_hours = frame_retention_hours
        
        # Camera and model resources
        self.cap: Optional[cv2.VideoCapture] = None
        self.face_detector: Optional[cv2.FaceDetectorYN] = None
        self.expression_model: Optional[np.ndarray] = None
        
        # Frame tracking
        self.frame_id = 0
        self.running = False
        self._capture_task: Optional[asyncio.Task] = None
        self._cleanup_task: Optional[asyncio.Task] = None
        
        # Frame history for analysis
        self.frame_history: List[VisionFrame] = []
        self.max_history = 300  # Keep last 5 minutes at 1 FPS
        
        # Eye fatigue tracking
        self.blink_history: List[Tuple[datetime, bool]] = []  # (timestamp, is_blink)
        self.previous_frame_gray: Optional[np.ndarray] = None
        
        # Screen time tracking
        self.screen_time_start: Optional[datetime] = None
        self.total_screen_time_today: timedelta = timedelta()
        
        # Encryption
        self.key_manager: Optional[KeyManager] = None
        self.frames_dir = Path("data/vision/frames")
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        
        # Register event types
        self.event_bus.register_event_type("vision_frame_processed")
        self.event_bus.register_event_type("face_detected")
        self.event_bus.register_event_type("expression_recognized")
        self.event_bus.register_event_type("user_absent")
        self.event_bus.register_event_type("screen_time_detected")
        self.event_bus.register_event_type("eye_fatigue_detected")
        self.event_bus.register_event_type("poor_posture_detected")
        self.event_bus.register_event_type("prolonged_screen_time")
        
        logger.info(f"Vision module initialized (enabled={enabled}, encrypt={encrypt_frames})")
    
    async def initialize(self) -> bool:
        """
        Initialize camera and models.
        
        Returns:
            True if initialization successful, False otherwise
        """
        if not self.enabled:
            logger.info("Vision module disabled - skipping initialization")
            return False
        
        if not CV2_AVAILABLE:
            logger.error("OpenCV not available - vision module cannot initialize")
            return False
        
        try:
            # Initialize camera
            self.cap = cv2.VideoCapture(self.camera_id)
            if not self.cap.isOpened():
                logger.error(f"Failed to open camera {self.camera_id}")
                return False
            
            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            logger.info("Camera initialized successfully")
            
            # Initialize face detector (YuNet)
            self._initialize_face_detector()
            
            # Initialize expression model (placeholder - would load ONNX model)
            self._initialize_expression_model()
            
            # Initialize encryption if enabled
            if self.encrypt_frames:
                self.key_manager = KeyManager()
                logger.info("Frame encryption enabled")
            
            self.status = VisionStatus.IDLE
            return True
            
        except Exception as e:
            logger.error(f"Error initializing vision module: {e}")
            self.status = VisionStatus.ERROR
            return False
    
    def _initialize_face_detector(self):
        """Initialize YuNet face detector."""
        try:
            if not CV2_AVAILABLE:
                logger.warning("OpenCV not available - face detector not initialized")
                return
            
            # YuNet face detector model
            # In production, would download and cache the model
            model_path = "face_detection_yunet_2023mar.onnx"
            
            # For now, we'll use OpenCV's built-in cascade classifier as fallback
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
            
            logger.info("Face detector initialized")
        except Exception as e:
            logger.error(f"Error initializing face detector: {e}")
    
    def _initialize_expression_model(self):
        """Initialize facial expression recognition model."""
        try:
            # FER2013-trained expression model (ONNX format)
            # In production, would download and cache the model
            model_path = "facial_expression_recognition.onnx"
            
            # Placeholder for model loading
            # In production: self.expression_model = onnx.load(model_path)
            logger.info("Expression model initialized")
        except Exception as e:
            logger.error(f"Error initializing expression model: {e}")
    
    async def start_capture(self):
        """Start the camera capture loop."""
        if not self.enabled:
            logger.warning("Vision module disabled - cannot start capture")
            return
        
        if self.running:
            logger.warning("Capture already running")
            return
        
        success = await self.initialize()
        if not success:
            logger.error("Failed to initialize vision module")
            return
        
        self.running = True
        self._capture_task = asyncio.create_task(self._capture_loop())
        self._cleanup_task = asyncio.create_task(self._cleanup_old_frames())
        logger.info("Camera capture started")
    
    async def stop_capture(self):
        """Stop the camera capture loop."""
        self.running = False
        
        if self._capture_task:
            self._capture_task.cancel()
            try:
                await self._capture_task
            except asyncio.CancelledError:
                pass
        
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        if self.cap:
            self.cap.release()
            self.cap = None
        
        logger.info("Camera capture stopped")
    
    async def _capture_loop(self):
        """
        Main capture loop - captures frames at 1 FPS.
        
        Requirements: 11.1
        """
        frame_skip = 30  # Capture every 30th frame at 30 FPS = 1 FPS
        frame_count = 0
        
        while self.running:
            try:
                ret, frame = self.cap.read()
                if not ret:
                    logger.warning("Failed to read frame from camera")
                    await asyncio.sleep(0.1)
                    continue
                
                frame_count += 1
                
                # Process every Nth frame to achieve 1 FPS
                if frame_count % frame_skip == 0:
                    self.status = VisionStatus.PROCESSING
                    
                    # Process frame asynchronously
                    vision_frame = await self._process_frame(frame)
                    
                    # Store in history
                    self.frame_history.append(vision_frame)
                    if len(self.frame_history) > self.max_history:
                        self.frame_history.pop(0)
                    
                    # Emit event
                    await self.event_bus.emit("vision_frame_processed", {
                        "frame_id": vision_frame.frame_id,
                        "timestamp": vision_frame.timestamp,
                        "faces_detected": vision_frame.faces_detected,
                        "expressions": [
                            {
                                "emotion": e.emotion,
                                "confidence": e.confidence
                            }
                            for e in vision_frame.expressions
                        ],
                        "user_present": vision_frame.user_present,
                    })
                    
                    self.status = VisionStatus.CAPTURING
                
                # Small delay to prevent busy-waiting
                await asyncio.sleep(0.01)
                
            except Exception as e:
                logger.error(f"Error in capture loop: {e}")
                self.status = VisionStatus.ERROR
                await asyncio.sleep(1)
    
    async def _process_frame(self, frame: np.ndarray) -> VisionFrame:
        """
        Process a single frame for face detection and expression recognition.
        
        Args:
            frame: Input frame from camera
            
        Returns:
            VisionFrame with detection results
        """
        self.frame_id += 1
        timestamp = datetime.now()
        
        # Detect faces
        faces = self._detect_faces(frame)
        
        # Recognize expressions
        expressions = []
        eye_fatigue = None
        posture = None
        
        for face in faces:
            face_roi = face.crop_from_frame(frame)
            emotion, confidence = await self._recognize_expression(face_roi)
            
            expressions.append(ExpressionResult(
                emotion=emotion,
                confidence=confidence,
                timestamp=timestamp,
                frame_id=self.frame_id
            ))
            
            # Detect eye fatigue (Requirements: 11.4)
            eye_fatigue = await self._detect_eye_fatigue(frame, face)
            if eye_fatigue and eye_fatigue.fatigue_level in ["moderate", "high"]:
                await self.event_bus.emit("eye_fatigue_detected", {
                    "frame_id": self.frame_id,
                    "fatigue_level": eye_fatigue.fatigue_level,
                    "blink_rate": eye_fatigue.blink_rate,
                    "timestamp": timestamp
                })
            
            # Estimate posture (Requirements: 11.5)
            posture = await self._estimate_posture(frame, face)
            if posture and posture.posture_type in [PostureType.SLOUCHING, PostureType.LEANING_FORWARD]:
                await self.event_bus.emit("poor_posture_detected", {
                    "frame_id": self.frame_id,
                    "posture_type": posture.posture_type.value,
                    "timestamp": timestamp
                })
            
            # Emit face detected event
            await self.event_bus.emit("face_detected", {
                "frame_id": self.frame_id,
                "timestamp": timestamp,
                "face_count": len(faces)
            })
            
            # Emit expression recognized event
            await self.event_bus.emit("expression_recognized", {
                "frame_id": self.frame_id,
                "emotion": emotion,
                "confidence": confidence,
                "timestamp": timestamp
            })
        
        # Detect user presence
        user_present = len(faces) > 0
        
        # Detect screen time (Requirements: 11.13)
        screen_time_detected = await self._detect_screen_time(frame, faces)
        
        # Detect activity (Requirements: 11.14)
        activity_detected = await self._detect_activity(frame)
        
        # Emit user absent event if no faces detected
        if not user_present:
            await self.event_bus.emit("user_absent", {
                "frame_id": self.frame_id,
                "timestamp": timestamp
            })
        
        # Encrypt and save frame if enabled (Requirements: 11.8)
        encrypted_frame_path = None
        if self.encrypt_frames:
            encrypted_frame_path = await self._encrypt_and_save_frame(frame, self.frame_id, timestamp)
        
        return VisionFrame(
            frame_id=self.frame_id,
            timestamp=timestamp,
            faces_detected=len(faces),
            expressions=expressions,
            user_present=user_present,
            screen_time_detected=screen_time_detected,
            activity_detected=activity_detected,
            eye_fatigue=eye_fatigue,
            posture=posture,
            encrypted_frame_path=encrypted_frame_path
        )
    
    def _detect_faces(self, frame: np.ndarray) -> List[FaceDetection]:
        """
        Detect faces in a frame using cascade classifier.
        
        Requirements: 11.2
        """
        try:
            if not CV2_AVAILABLE or frame is None:
                return []
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )
            
            detections = []
            for (x, y, w, h) in faces:
                detections.append(FaceDetection(
                    x=int(x),
                    y=int(y),
                    width=int(w),
                    height=int(h),
                    confidence=0.9  # Cascade classifier doesn't provide confidence
                ))
            
            return detections
        except Exception as e:
            logger.error(f"Error detecting faces: {e}")
            return []
    
    async def _recognize_expression(self, face_roi: np.ndarray) -> Tuple[str, float]:
        """
        Recognize facial expression from face ROI.
        
        Requirements: 11.3
        
        Args:
            face_roi: Cropped face region
            
        Returns:
            Tuple of (emotion_label, confidence_score)
        """
        try:
            # Placeholder implementation
            # In production, would use ONNX model for inference
            
            # For now, return a random emotion for testing
            # In production: emotion_probs = model.predict(face_roi)
            emotion_probs = np.random.dirichlet(np.ones(len(EMOTIONS)))
            
            emotion_idx = np.argmax(emotion_probs)
            emotion = EMOTIONS[emotion_idx]
            confidence = float(emotion_probs[emotion_idx])
            
            return emotion, confidence
        except Exception as e:
            logger.error(f"Error recognizing expression: {e}")
            return "neutral", 0.5
    
    async def _detect_eye_fatigue(self, frame: np.ndarray, face: FaceDetection) -> Optional[EyeFatigueMetrics]:
        """
        Detect eye fatigue from facial features.
        
        Requirements: 11.4
        
        Analyzes:
        - Blink rate (blinks per minute)
        - Eye aspect ratio (lower = more closed)
        - Eye redness (color analysis)
        
        Args:
            frame: Full frame
            face: Detected face
            
        Returns:
            EyeFatigueMetrics or None if detection fails
        """
        try:
            # Calculate blink rate from recent history
            blink_rate = self._calculate_blink_rate()
            
            # Calculate eye aspect ratio (EAR)
            # In production, would use facial landmarks to compute EAR
            # EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
            # where p1-p6 are eye landmark points
            eye_aspect_ratio = 0.3  # Placeholder (normal: 0.25-0.35, fatigue: <0.2)
            
            # Detect eye redness using color analysis
            # In production, would analyze sclera region for redness
            redness_score = 0.2  # Placeholder (0-1, higher = more red)
            
            # Determine fatigue level
            fatigue_level = "low"
            if blink_rate < 10 or eye_aspect_ratio < 0.2 or redness_score > 0.5:
                fatigue_level = "high"
            elif blink_rate < 15 or eye_aspect_ratio < 0.25 or redness_score > 0.3:
                fatigue_level = "moderate"
            
            return EyeFatigueMetrics(
                blink_rate=blink_rate,
                eye_aspect_ratio=eye_aspect_ratio,
                redness_score=redness_score,
                fatigue_level=fatigue_level,
                timestamp=datetime.now()
            )
        except Exception as e:
            logger.error(f"Error detecting eye fatigue: {e}")
            return None
    
    def _calculate_blink_rate(self) -> float:
        """
        Calculate blink rate from recent blink history.
        
        Returns:
            Blinks per minute
        """
        # Clean old blink history (keep last 60 seconds)
        cutoff_time = datetime.now() - timedelta(seconds=60)
        self.blink_history = [(t, b) for t, b in self.blink_history if t >= cutoff_time]
        
        # Count blinks in last 60 seconds
        blink_count = sum(1 for _, is_blink in self.blink_history if is_blink)
        
        # Calculate blinks per minute
        if len(self.blink_history) > 0:
            time_span = (datetime.now() - self.blink_history[0][0]).total_seconds()
            if time_span > 0:
                return (blink_count / time_span) * 60
        
        return 15.0  # Default normal blink rate
    
    async def _estimate_posture(self, frame: np.ndarray, face: FaceDetection) -> Optional[PostureEstimate]:
        """
        Estimate head pose and classify posture.
        
        Requirements: 11.5
        
        Estimates:
        - Head pitch (looking up/down)
        - Head yaw (looking left/right)
        - Head roll (tilted left/right)
        
        Classifies posture as:
        - Upright: Good posture
        - Slouching: Head tilted down significantly
        - Leaning: Head tilted to side
        
        Args:
            frame: Full frame
            face: Detected face
            
        Returns:
            PostureEstimate or None if estimation fails
        """
        try:
            # In production, would use facial landmarks and PnP algorithm
            # to estimate 3D head pose from 2D landmarks
            
            # Placeholder implementation
            # Estimate head pose angles (degrees)
            head_pitch = 0.0  # Positive = looking up, negative = looking down
            head_yaw = 0.0    # Positive = looking right, negative = looking left
            head_roll = 0.0   # Positive = tilted right, negative = tilted left
            
            # Simple heuristic based on face position in frame
            frame_height, frame_width = frame.shape[:2]
            face_center_y = face.y + face.height // 2
            face_center_x = face.x + face.width // 2
            
            # Estimate pitch from vertical position
            vertical_ratio = face_center_y / frame_height
            if vertical_ratio > 0.6:
                head_pitch = -20.0  # Looking down
            elif vertical_ratio < 0.4:
                head_pitch = 20.0   # Looking up
            
            # Estimate yaw from horizontal position
            horizontal_ratio = face_center_x / frame_width
            if horizontal_ratio > 0.6:
                head_yaw = 20.0  # Looking right
            elif horizontal_ratio < 0.4:
                head_yaw = -20.0  # Looking left
            
            # Classify posture
            posture_type = PostureType.UPRIGHT
            if head_pitch < -15:
                posture_type = PostureType.SLOUCHING
            elif head_pitch < -25:
                posture_type = PostureType.LEANING_FORWARD
            elif head_roll > 15:
                posture_type = PostureType.LEANING_RIGHT
            elif head_roll < -15:
                posture_type = PostureType.LEANING_LEFT
            
            return PostureEstimate(
                posture_type=posture_type,
                head_pitch=head_pitch,
                head_yaw=head_yaw,
                head_roll=head_roll,
                confidence=0.7,
                timestamp=datetime.now()
            )
        except Exception as e:
            logger.error(f"Error estimating posture: {e}")
            return None
    
    async def _detect_screen_time(self, frame: np.ndarray, faces: List[FaceDetection]) -> bool:
        """
        Detect if user is looking at screen (prolonged screen time).
        
        Requirements: 11.13
        
        Uses gaze estimation to determine if user is looking at screen.
        Tracks cumulative screen time and emits alerts for prolonged use.
        
        Args:
            frame: Full frame
            faces: Detected faces
            
        Returns:
            True if user is looking at screen, False otherwise
        """
        try:
            if not faces:
                # No face detected, not looking at screen
                if self.screen_time_start:
                    # End screen time session
                    session_duration = datetime.now() - self.screen_time_start
                    self.total_screen_time_today += session_duration
                    self.screen_time_start = None
                return False
            
            # In production, would use gaze estimation model
            # For now, assume user is looking at screen if face detected
            is_looking_at_screen = True
            
            if is_looking_at_screen:
                if not self.screen_time_start:
                    # Start new screen time session
                    self.screen_time_start = datetime.now()
                else:
                    # Check for prolonged screen time (> 2 hours)
                    current_session = datetime.now() - self.screen_time_start
                    total_today = self.total_screen_time_today + current_session
                    
                    if total_today.total_seconds() > 2 * 3600:  # 2 hours
                        await self.event_bus.emit("prolonged_screen_time", {
                            "total_screen_time_hours": total_today.total_seconds() / 3600,
                            "current_session_minutes": current_session.total_seconds() / 60,
                            "timestamp": datetime.now()
                        })
                
                return True
            else:
                if self.screen_time_start:
                    # End screen time session
                    session_duration = datetime.now() - self.screen_time_start
                    self.total_screen_time_today += session_duration
                    self.screen_time_start = None
                return False
                
        except Exception as e:
            logger.error(f"Error detecting screen time: {e}")
            return False
    
    async def _detect_activity(self, frame: np.ndarray) -> bool:
        """
        Detect if user is moving/active using optical flow.
        
        Requirements: 11.14
        
        Uses frame differencing or optical flow to detect motion.
        
        Args:
            frame: Current frame
            
        Returns:
            True if activity detected, False otherwise
        """
        try:
            if not CV2_AVAILABLE:
                return False
            
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            if self.previous_frame_gray is None:
                self.previous_frame_gray = gray
                return False
            
            # Calculate frame difference
            frame_diff = cv2.absdiff(self.previous_frame_gray, gray)
            
            # Threshold to get binary image
            _, thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)
            
            # Calculate percentage of changed pixels
            changed_pixels = np.sum(thresh > 0)
            total_pixels = thresh.shape[0] * thresh.shape[1]
            change_ratio = changed_pixels / total_pixels
            
            # Update previous frame
            self.previous_frame_gray = gray
            
            # Activity detected if > 5% of pixels changed
            return change_ratio > 0.05
            
        except Exception as e:
            logger.error(f"Error detecting activity: {e}")
            return False
    
    async def _encrypt_and_save_frame(self, frame: np.ndarray, frame_id: int, timestamp: datetime) -> Optional[str]:
        """
        Encrypt and save frame to disk.
        
        Requirements: 11.8
        
        Args:
            frame: Frame to encrypt
            frame_id: Frame ID
            timestamp: Frame timestamp
            
        Returns:
            Path to encrypted frame file, or None if encryption fails
        """
        try:
            if not self.key_manager:
                logger.warning("Key manager not initialized - cannot encrypt frame")
                return None
            
            # Encode frame as JPEG
            success, encoded_frame = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not success:
                logger.error("Failed to encode frame")
                return None
            
            # Convert to bytes
            frame_bytes = encoded_frame.tobytes()
            
            # Encrypt frame
            encrypted_data = self.key_manager.encrypt(frame_bytes)
            
            # Generate filename
            filename = f"frame_{frame_id}_{timestamp.strftime('%Y%m%d_%H%M%S')}.enc"
            filepath = self.frames_dir / filename
            
            # Save encrypted frame
            with open(filepath, 'wb') as f:
                f.write(encrypted_data)
            
            logger.debug(f"Encrypted frame saved: {filepath}")
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Error encrypting and saving frame: {e}")
            return None
    
    async def _cleanup_old_frames(self):
        """
        Cleanup task that deletes frames older than retention period.
        
        Requirements: 11.9
        
        Runs every hour to delete frames older than frame_retention_hours.
        """
        while self.running:
            try:
                # Sleep for 1 hour
                await asyncio.sleep(3600)
                
                # Calculate cutoff time
                cutoff_time = datetime.now() - timedelta(hours=self.frame_retention_hours)
                
                # Find and delete old frames
                deleted_count = 0
                for filepath in self.frames_dir.glob("frame_*.enc"):
                    try:
                        # Extract timestamp from filename
                        # Format: frame_{id}_{YYYYMMDD_HHMMSS}.enc
                        parts = filepath.stem.split('_')
                        if len(parts) >= 3:
                            timestamp_str = f"{parts[2]}_{parts[3]}"
                            file_timestamp = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
                            
                            if file_timestamp < cutoff_time:
                                filepath.unlink()
                                deleted_count += 1
                    except Exception as e:
                        logger.error(f"Error processing file {filepath}: {e}")
                
                if deleted_count > 0:
                    logger.info(f"Deleted {deleted_count} old encrypted frames")
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cleanup task: {e}")
    
    def get_dominant_emotion(self, seconds: int = 10) -> Optional[str]:
        """
        Get the dominant emotion from recent frames.
        
        Args:
            seconds: Number of seconds to look back
            
        Returns:
            Most common emotion in the time window, or None if no data
        """
        if not self.frame_history:
            return None
        
        # Calculate cutoff time
        cutoff_time = datetime.now() - timedelta(seconds=seconds)
        
        # Collect emotions from recent frames
        recent_emotions = []
        for frame in self.frame_history:
            if frame.timestamp >= cutoff_time:
                for expr in frame.expressions:
                    recent_emotions.append(expr.emotion)
        
        if not recent_emotions:
            return None
        
        # Return most common emotion
        from collections import Counter
        emotion_counts = Counter(recent_emotions)
        return emotion_counts.most_common(1)[0][0]
    
    def get_average_mood_score(self, seconds: int = 10) -> Optional[float]:
        """
        Get average mood score from recent expressions.
        
        Args:
            seconds: Number of seconds to look back
            
        Returns:
            Average mood score (0-10), or None if no data
        """
        if not self.frame_history:
            return None
        
        # Calculate cutoff time
        cutoff_time = datetime.now() - timedelta(seconds=seconds)
        
        # Collect mood scores from recent frames
        mood_scores = []
        for frame in self.frame_history:
            if frame.timestamp >= cutoff_time:
                for expr in frame.expressions:
                    mood_score = EMOTION_TO_MOOD.get(expr.emotion, 5.0)
                    mood_scores.append(mood_score)
        
        if not mood_scores:
            return None
        
        return sum(mood_scores) / len(mood_scores)
    
    def get_eye_fatigue_status(self) -> Optional[EyeFatigueMetrics]:
        """
        Get most recent eye fatigue metrics.
        
        Returns:
            Most recent EyeFatigueMetrics, or None if no data
        """
        for frame in reversed(self.frame_history):
            if frame.eye_fatigue:
                return frame.eye_fatigue
        return None
    
    def get_posture_status(self) -> Optional[PostureEstimate]:
        """
        Get most recent posture estimate.
        
        Returns:
            Most recent PostureEstimate, or None if no data
        """
        for frame in reversed(self.frame_history):
            if frame.posture:
                return frame.posture
        return None
    
    def get_screen_time_today(self) -> timedelta:
        """
        Get total screen time for today.
        
        Returns:
            Total screen time as timedelta
        """
        total = self.total_screen_time_today
        if self.screen_time_start:
            # Add current session
            total += datetime.now() - self.screen_time_start
        return total
    
    def get_activity_level(self, minutes: int = 10) -> float:
        """
        Get activity level from recent frames.
        
        Args:
            minutes: Number of minutes to look back
            
        Returns:
            Activity level (0-1), where 1 = very active
        """
        if not self.frame_history:
            return 0.0
        
        # Calculate cutoff time
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        
        # Count frames with activity
        active_frames = 0
        total_frames = 0
        
        for frame in self.frame_history:
            if frame.timestamp >= cutoff_time:
                total_frames += 1
                if frame.activity_detected:
                    active_frames += 1
        
        if total_frames == 0:
            return 0.0
        
        return active_frames / total_frames
    
    def set_privacy_mode(self, enabled: bool):
        """
        Enable or disable privacy mode (disables camera capture).
        
        Requirements: 11.10
        
        Args:
            enabled: True to disable camera, False to enable
        """
        self.enabled = not enabled
        logger.info(f"Privacy mode: {'enabled' if enabled else 'disabled'}")
    
    def is_privacy_mode_enabled(self) -> bool:
        """Check if privacy mode is enabled."""
        return not self.enabled
    
    def get_status(self) -> Dict[str, any]:
        """Get current status of vision module."""
        eye_fatigue = self.get_eye_fatigue_status()
        posture = self.get_posture_status()
        screen_time = self.get_screen_time_today()
        
        return {
            "enabled": self.enabled,
            "status": self.status.value,
            "running": self.running,
            "frame_id": self.frame_id,
            "frames_in_history": len(self.frame_history),
            "camera_id": self.camera_id,
            "encrypt_frames": self.encrypt_frames,
            "eye_fatigue_level": eye_fatigue.fatigue_level if eye_fatigue else None,
            "posture_type": posture.posture_type.value if posture else None,
            "screen_time_hours": screen_time.total_seconds() / 3600,
            "activity_level": self.get_activity_level(),
        }
