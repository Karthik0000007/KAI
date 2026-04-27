"""
Health Dashboard API - FastAPI Backend

Provides REST endpoints for health data visualization and analysis.
Implements session-based authentication and real-time WebSocket updates.

Requirements: 13.1, 13.15, 13.16, 13.17
"""

import logging
import asyncio
import json
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any
from dataclasses import dataclass, asdict
import secrets
import hashlib

from fastapi import FastAPI, HTTPException, WebSocket, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from core.health_db import HealthDatabase

logger = logging.getLogger(__name__)

# ─── Session Management ──────────────────────────────────────────────────

@dataclass
class Session:
    """Represents a user session."""
    token: str
    created_at: datetime
    last_accessed: datetime
    expires_at: datetime
    user_id: str = "default"


class SessionManager:
    """Manages user sessions with timeout."""
    
    def __init__(self, timeout_minutes: int = 30):
        self.timeout = timedelta(minutes=timeout_minutes)
        self.sessions: Dict[str, Session] = {}
    
    def create_session(self, user_id: str = "default") -> str:
        """Create a new session and return token."""
        token = secrets.token_urlsafe(32)
        now = datetime.now()
        
        session = Session(
            token=token,
            created_at=now,
            last_accessed=now,
            expires_at=now + self.timeout,
            user_id=user_id
        )
        
        self.sessions[token] = session
        logger.info(f"Created session for user {user_id}")
        return token
    
    def validate_session(self, token: str) -> bool:
        """Validate session token and update last accessed time."""
        if token not in self.sessions:
            return False
        
        session = self.sessions[token]
        now = datetime.now()
        
        # Check if expired
        if now >= session.expires_at:
            del self.sessions[token]
            logger.warning(f"Session {token} expired")
            return False
        
        # Update last accessed
        session.last_accessed = now
        session.expires_at = now + self.timeout
        
        return True
    
    def invalidate_session(self, token: str) -> bool:
        """Invalidate a session."""
        if token in self.sessions:
            del self.sessions[token]
            logger.info(f"Invalidated session {token}")
            return True
        return False
    
    def cleanup_expired(self):
        """Remove expired sessions."""
        now = datetime.now()
        expired = [
            token for token, session in self.sessions.items()
            if now >= session.expires_at
        ]
        
        for token in expired:
            del self.sessions[token]
        
        if expired:
            logger.info(f"Cleaned up {len(expired)} expired sessions")


# ─── Pydantic Models ────────────────────────────────────────────────────

class LoginRequest(BaseModel):
    """Login request model."""
    username: str = "default"
    password: str = "default"


class LoginResponse(BaseModel):
    """Login response model."""
    token: str
    expires_in: int


class HealthDataPoint(BaseModel):
    """Single health data point."""
    timestamp: str
    value: float
    label: Optional[str] = None


class HealthTrendResponse(BaseModel):
    """Health trend data response."""
    data: List[HealthDataPoint]
    average: float
    min: float
    max: float
    trend: str  # "improving", "stable", "declining"


class EmotionDistribution(BaseModel):
    """Emotion distribution data."""
    emotion: str
    count: int
    percentage: float


class MedicationComplianceDay(BaseModel):
    """Medication compliance for a single day."""
    date: str
    taken: bool
    percentage: float


class VitalSignsData(BaseModel):
    """Vital signs data point."""
    timestamp: str
    heart_rate: Optional[float] = None
    spo2: Optional[float] = None
    temperature: Optional[float] = None


class ProactiveAlertData(BaseModel):
    """Proactive alert data."""
    id: str
    timestamp: str
    alert_type: str
    severity: str
    message: str
    acknowledged: bool


class CorrelationData(BaseModel):
    """Correlation analysis data."""
    x_values: List[float]
    y_values: List[float]
    correlation_coefficient: float
    trend_line: Optional[List[float]] = None


# ─── FastAPI Application ────────────────────────────────────────────────

def create_app(db: HealthDatabase) -> FastAPI:
    """Create and configure FastAPI application."""
    app = FastAPI(title="Aegis Health Dashboard API", version="1.0.0")
    
    # Add CORS middleware (localhost only)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000", "http://localhost:8080"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Session manager
    session_manager = SessionManager(timeout_minutes=30)
    
    # ─── Authentication Endpoints ────────────────────────────────────
    
    @app.post("/api/auth/login", response_model=LoginResponse)
    async def login(request: LoginRequest):
        """
        Login endpoint - creates a session token.
        
        Requirements: 13.15, 13.16
        """
        # Simple authentication (in production, would verify credentials)
        token = session_manager.create_session(user_id=request.username)
        
        return LoginResponse(
            token=token,
            expires_in=30 * 60  # 30 minutes in seconds
        )
    
    @app.post("/api/auth/logout")
    async def logout(token: str = Query(...)):
        """Logout endpoint - invalidates session token."""
        if session_manager.invalidate_session(token):
            return {"message": "Logged out successfully"}
        else:
            raise HTTPException(status_code=401, detail="Invalid token")
    
    # ─── Health Data Endpoints ──────────────────────────────────────
    
    def verify_token(token: str = Query(...)):
        """Dependency to verify session token."""
        if not session_manager.validate_session(token):
            raise HTTPException(status_code=401, detail="Invalid or expired token")
        return token
    
    @app.get("/api/health/mood", response_model=HealthTrendResponse)
    async def get_mood_trend(
        days: int = Query(7, ge=1, le=90),
        token: str = Depends(verify_token)
    ):
        """
        Get mood trend data.
        
        Requirements: 13.2
        """
        try:
            checkins = db.get_recent_checkins(days=days)
            
            mood_data = [
                HealthDataPoint(
                    timestamp=c["timestamp"],
                    value=c.get("mood_score", 0),
                    label=c.get("detected_emotion")
                )
                for c in checkins
                if c.get("mood_score") is not None
            ]
            
            if not mood_data:
                return HealthTrendResponse(
                    data=[],
                    average=0,
                    min=0,
                    max=0,
                    trend="stable"
                )
            
            values = [d.value for d in mood_data]
            avg = sum(values) / len(values)
            
            # Calculate trend
            if len(values) >= 2:
                recent_avg = sum(values[-len(values)//2:]) / (len(values)//2)
                older_avg = sum(values[:len(values)//2]) / (len(values)//2)
                
                if recent_avg > older_avg + 0.5:
                    trend = "improving"
                elif recent_avg < older_avg - 0.5:
                    trend = "declining"
                else:
                    trend = "stable"
            else:
                trend = "stable"
            
            return HealthTrendResponse(
                data=mood_data,
                average=round(avg, 1),
                min=min(values),
                max=max(values),
                trend=trend
            )
        except Exception as e:
            logger.error(f"Error getting mood trend: {e}")
            raise HTTPException(status_code=500, detail="Error retrieving mood data")
    
    @app.get("/api/health/sleep", response_model=HealthTrendResponse)
    async def get_sleep_trend(
        days: int = Query(7, ge=1, le=90),
        token: str = Depends(verify_token)
    ):
        """
        Get sleep trend data.
        
        Requirements: 13.3
        """
        try:
            checkins = db.get_recent_checkins(days=days)
            
            sleep_data = [
                HealthDataPoint(
                    timestamp=c["timestamp"],
                    value=c.get("sleep_hours", 0)
                )
                for c in checkins
                if c.get("sleep_hours") is not None
            ]
            
            if not sleep_data:
                return HealthTrendResponse(
                    data=[],
                    average=0,
                    min=0,
                    max=0,
                    trend="stable"
                )
            
            values = [d.value for d in sleep_data]
            avg = sum(values) / len(values)
            
            # Calculate trend
            if len(values) >= 2:
                recent_avg = sum(values[-len(values)//2:]) / (len(values)//2)
                older_avg = sum(values[:len(values)//2]) / (len(values)//2)
                
                if recent_avg > older_avg + 0.5:
                    trend = "improving"
                elif recent_avg < older_avg - 0.5:
                    trend = "declining"
                else:
                    trend = "stable"
            else:
                trend = "stable"
            
            return HealthTrendResponse(
                data=sleep_data,
                average=round(avg, 1),
                min=min(values),
                max=max(values),
                trend=trend
            )
        except Exception as e:
            logger.error(f"Error getting sleep trend: {e}")
            raise HTTPException(status_code=500, detail="Error retrieving sleep data")
    
    @app.get("/api/health/energy", response_model=HealthTrendResponse)
    async def get_energy_trend(
        days: int = Query(7, ge=1, le=90),
        token: str = Depends(verify_token)
    ):
        """
        Get energy trend data.
        
        Requirements: 13.4
        """
        try:
            checkins = db.get_recent_checkins(days=days)
            
            energy_data = [
                HealthDataPoint(
                    timestamp=c["timestamp"],
                    value=c.get("energy_level", 0)
                )
                for c in checkins
                if c.get("energy_level") is not None
            ]
            
            if not energy_data:
                return HealthTrendResponse(
                    data=[],
                    average=0,
                    min=0,
                    max=0,
                    trend="stable"
                )
            
            values = [d.value for d in energy_data]
            avg = sum(values) / len(values)
            
            # Calculate trend
            if len(values) >= 2:
                recent_avg = sum(values[-len(values)//2:]) / (len(values)//2)
                older_avg = sum(values[:len(values)//2]) / (len(values)//2)
                
                if recent_avg > older_avg + 0.5:
                    trend = "improving"
                elif recent_avg < older_avg - 0.5:
                    trend = "declining"
                else:
                    trend = "stable"
            else:
                trend = "stable"
            
            return HealthTrendResponse(
                data=energy_data,
                average=round(avg, 1),
                min=min(values),
                max=max(values),
                trend=trend
            )
        except Exception as e:
            logger.error(f"Error getting energy trend: {e}")
            raise HTTPException(status_code=500, detail="Error retrieving energy data")
    
    @app.get("/api/health/emotions")
    async def get_emotion_distribution(
        days: int = Query(7, ge=1, le=90),
        token: str = Depends(verify_token)
    ):
        """
        Get emotion distribution data.
        
        Requirements: 13.5
        """
        try:
            checkins = db.get_recent_checkins(days=days)
            
            emotion_counts: Dict[str, int] = {}
            for c in checkins:
                emotion = c.get("detected_emotion")
                if emotion:
                    emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
            
            total = sum(emotion_counts.values())
            
            distribution = [
                EmotionDistribution(
                    emotion=emotion,
                    count=count,
                    percentage=round((count / total) * 100, 1) if total > 0 else 0
                )
                for emotion, count in sorted(
                    emotion_counts.items(),
                    key=lambda x: x[1],
                    reverse=True
                )
            ]
            
            return {"emotions": distribution}
        except Exception as e:
            logger.error(f"Error getting emotion distribution: {e}")
            raise HTTPException(status_code=500, detail="Error retrieving emotion data")
    
    @app.get("/api/health/vitals")
    async def get_vital_signs(
        days: int = Query(7, ge=1, le=90),
        token: str = Depends(verify_token)
    ):
        """
        Get vital signs data (heart rate, SpO2, temperature).
        
        Requirements: 13.7
        """
        try:
            vitals = db.get_recent_vitals(days=days)
            
            vital_data = [
                VitalSignsData(
                    timestamp=v.get("timestamp", ""),
                    heart_rate=v.get("heart_rate"),
                    spo2=v.get("spo2"),
                    temperature=v.get("temperature")
                )
                for v in vitals
            ]
            
            return {"vitals": vital_data}
        except Exception as e:
            logger.error(f"Error getting vital signs: {e}")
            raise HTTPException(status_code=500, detail="Error retrieving vital signs")
    
    @app.get("/api/health/alerts")
    async def get_proactive_alerts(
        token: str = Depends(verify_token)
    ):
        """
        Get proactive alerts history.
        
        Requirements: 13.8, 13.9
        """
        try:
            alerts = db.get_unacknowledged_alerts()
            
            alert_data = [
                ProactiveAlertData(
                    id=a.get("id", ""),
                    timestamp=a.get("timestamp", ""),
                    alert_type=a.get("alert_type", ""),
                    severity=a.get("severity", "info"),
                    message=a.get("message", ""),
                    acknowledged=bool(a.get("acknowledged", False))
                )
                for a in alerts
            ]
            
            return {"alerts": alert_data}
        except Exception as e:
            logger.error(f"Error getting alerts: {e}")
            raise HTTPException(status_code=500, detail="Error retrieving alerts")
    
    @app.post("/api/health/alerts/{alert_id}/acknowledge")
    async def acknowledge_alert(
        alert_id: str,
        token: str = Depends(verify_token)
    ):
        """Acknowledge a proactive alert."""
        try:
            db.acknowledge_alert(alert_id)
            return {"message": "Alert acknowledged"}
        except Exception as e:
            logger.error(f"Error acknowledging alert: {e}")
            raise HTTPException(status_code=500, detail="Error acknowledging alert")
    
    @app.get("/api/health/stats")
    async def get_health_statistics(
        days: int = Query(7, ge=1, le=90),
        token: str = Depends(verify_token)
    ):
        """
        Get overall health statistics.
        
        Requirements: 13.14
        """
        try:
            stats = db.get_checkin_stats(days=days)
            
            return {
                "period_days": days,
                "check_ins": stats.get("count", 0),
                "avg_mood": stats.get("avg_mood"),
                "avg_sleep": stats.get("avg_sleep"),
                "avg_energy": stats.get("avg_energy"),
                "low_mood_days": stats.get("low_mood_days", 0),
                "low_sleep_days": stats.get("low_sleep_days", 0),
            }
        except Exception as e:
            logger.error(f"Error getting health statistics: {e}")
            raise HTTPException(status_code=500, detail="Error retrieving statistics")
    
    @app.get("/api/health/export/csv")
    async def export_data_csv(
        days: int = Query(7, ge=1, le=90),
        token: str = Depends(verify_token)
    ):
        """
        Export health data as CSV.
        
        Requirements: 13.11, 13.12
        """
        try:
            checkins = db.get_recent_checkins(days=days)
            
            # Build CSV content
            csv_lines = [
                "timestamp,mood_score,sleep_hours,energy_level,emotion,notes"
            ]
            
            for c in checkins:
                csv_lines.append(
                    f"{c.get('timestamp', '')},"
                    f"{c.get('mood_score', '')},"
                    f"{c.get('sleep_hours', '')},"
                    f"{c.get('energy_level', '')},"
                    f"{c.get('detected_emotion', '')},"
                    f"\"{c.get('notes', '')}\""
                )
            
            csv_content = "\n".join(csv_lines)
            
            return {
                "format": "csv",
                "data": csv_content,
                "filename": f"aegis_health_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            }
        except Exception as e:
            logger.error(f"Error exporting CSV: {e}")
            raise HTTPException(status_code=500, detail="Error exporting data")
    
    @app.websocket("/ws/health/live")
    async def websocket_live_updates(websocket: WebSocket):
        """
        WebSocket endpoint for live health data updates.
        
        Requirements: 13.17
        """
        await websocket.accept()
        logger.info("WebSocket client connected")
        
        try:
            while True:
                # Receive token from client
                data = await websocket.receive_text()
                message = json.loads(data)
                
                if message.get("type") == "subscribe":
                    token = message.get("token")
                    
                    if not session_manager.validate_session(token):
                        await websocket.send_json({
                            "type": "error",
                            "message": "Invalid token"
                        })
                        break
                    
                    # Send initial data
                    stats = db.get_checkin_stats(days=7)
                    await websocket.send_json({
                        "type": "health_update",
                        "data": {
                            "avg_mood": stats.get("avg_mood"),
                            "avg_sleep": stats.get("avg_sleep"),
                            "avg_energy": stats.get("avg_energy"),
                            "timestamp": datetime.now().isoformat()
                        }
                    })
                    
                    # Keep connection alive
                    await asyncio.sleep(30)
                
        except Exception as e:
            logger.error(f"WebSocket error: {e}")
        finally:
            await websocket.close()
            logger.info("WebSocket client disconnected")
    
    @app.get("/api/health/status")
    async def get_api_status():
        """Get API status."""
        return {
            "status": "healthy",
            "version": "1.0.0",
            "timestamp": datetime.now().isoformat()
        }
    
    return app


# ─── Standalone Server ──────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    
    # Initialize database
    db = HealthDatabase()
    
    # Create app
    app = create_app(db)
    
    # Run server
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8080,
        log_level="info"
    )
