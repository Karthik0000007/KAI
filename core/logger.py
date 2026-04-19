"""
Aegis Logging Module
Structured logging with JSON output and rotating file handler.

Requirements: 5.1, 5.2, 5.3, 5.4
"""

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Any, Dict, Optional
from datetime import datetime

import structlog


def setup_logging(
    log_file: Path,
    log_level: str = "INFO",
    max_size_mb: int = 10,
    backup_count: int = 5,
    log_format: str = "json",
    console_enabled: bool = True
) -> structlog.BoundLogger:
    """
    Configure structured logging with JSON output and rotating file handler.
    
    Args:
        log_file: Path to log file
        log_level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        max_size_mb: Maximum log file size in MB before rotation
        backup_count: Number of backup log files to keep
        log_format: Log format ("json" or "text")
        console_enabled: Whether to enable console logging
        
    Returns:
        Configured structlog logger
        
    Requirements:
        - 5.1: Write logs to rotating file at data/logs/aegis.log
        - 5.2: Rotate log files when they exceed 10 MB
        - 5.3: Retain the last 5 log files
        - 5.4: Support configurable log levels
    """
    # Ensure log directory exists
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert log level string to logging constant
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        level=numeric_level,
        handlers=[],
        force=True  # Force reconfiguration
    )
    
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    
    # Clear any existing handlers
    root_logger.handlers.clear()
    
    # Create rotating file handler
    # Requirement 5.1: Write logs to rotating file
    # Requirement 5.2: Rotate when exceeds 10 MB (configurable)
    # Requirement 5.3: Retain last 5 log files (configurable)
    max_bytes = max_size_mb * 1024 * 1024  # Convert MB to bytes
    file_handler = logging.handlers.RotatingFileHandler(
        filename=str(log_file),
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding='utf-8'
    )
    file_handler.setLevel(numeric_level)
    file_handler.setFormatter(logging.Formatter('%(message)s'))
    
    # Add file handler to root logger
    root_logger.addHandler(file_handler)
    
    # Add console handler if enabled
    if console_enabled:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(numeric_level)
        console_handler.setFormatter(logging.Formatter('%(message)s'))
        root_logger.addHandler(console_handler)
    
    # Configure structlog processors based on format
    if log_format == "json":
        # JSON format for production
        processors = [
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer()
        ]
    else:
        # Text format for development
        processors = [
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.dev.ConsoleRenderer()
        ]
    
    # Configure structlog
    structlog.configure(
        processors=processors,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    # Return configured logger
    logger = structlog.get_logger()
    logger.info(
        "logging_initialized",
        log_file=str(log_file),
        log_level=log_level,
        max_size_mb=max_size_mb,
        backup_count=backup_count,
        log_format=log_format,
        console_enabled=console_enabled
    )
    
    return logger


def get_logger(name: Optional[str] = None) -> structlog.BoundLogger:
    """
    Get a structlog logger instance.
    
    Args:
        name: Logger name (optional, defaults to calling module)
        
    Returns:
        Configured structlog logger
    """
    if name:
        return structlog.get_logger(name)
    else:
        return structlog.get_logger()


def shutdown_logging() -> None:
    """
    Shutdown logging and close all handlers.
    
    This should be called before exiting the application or in test cleanup
    to ensure all log files are properly closed.
    """
    logging.shutdown()


class LoggerMixin:
    """
    Mixin class that provides a logger property to any class.
    
    Usage:
        class MyClass(LoggerMixin):
            def my_method(self):
                self.logger.info("doing_something", param=value)
    """
    
    @property
    def logger(self) -> structlog.BoundLogger:
        """Get logger for this class."""
        return structlog.get_logger(self.__class__.__name__)


def log_turn_metrics(
    logger: structlog.BoundLogger,
    turn_id: str,
    duration: float,
    emotion: Optional[str] = None,
    health_signals: Optional[Dict[str, Any]] = None,
    response_length: Optional[int] = None,
    stage_durations: Optional[Dict[str, float]] = None
) -> None:
    """
    Log structured metrics for a conversation turn.
    
    Args:
        logger: Structlog logger instance
        turn_id: Unique identifier for this turn
        duration: Total turn duration in seconds
        emotion: Detected emotion (optional)
        health_signals: Extracted health signals (optional)
        response_length: Length of response in characters (optional)
        stage_durations: Per-stage durations in seconds (optional)
        
    Requirement 5.5: Log turn metrics with structured data
    """
    log_data = {
        "event": "turn_completed",
        "turn_id": turn_id,
        "duration_seconds": round(duration, 3),
    }
    
    if emotion:
        log_data["emotion"] = emotion
    
    if health_signals:
        log_data["health_signals"] = health_signals
    
    if response_length is not None:
        log_data["response_length"] = response_length
    
    if stage_durations:
        log_data["stage_durations"] = {
            stage: round(dur, 3) for stage, dur in stage_durations.items()
        }
    
    logger.info(**log_data)


def log_proactive_alert(
    logger: structlog.BoundLogger,
    alert_type: str,
    severity: str,
    context: Dict[str, Any],
    message: str
) -> None:
    """
    Log a proactive health alert.
    
    Args:
        logger: Structlog logger instance
        alert_type: Type of alert (e.g., "low_mood_pattern")
        severity: Alert severity (e.g., "low", "medium", "high")
        context: Context data for the alert
        message: Human-readable alert message
        
    Requirement 5.8: Log all proactive alerts with timestamp, type, severity, and context
    """
    logger.warning(
        "proactive_alert_generated",
        alert_type=alert_type,
        severity=severity,
        context=context,
        message=message
    )


def log_error_with_context(
    logger: structlog.BoundLogger,
    error: Exception,
    context: Dict[str, Any],
    message: str = "error_occurred"
) -> None:
    """
    Log an error with additional context.
    
    Args:
        logger: Structlog logger instance
        error: Exception that occurred
        context: Additional context data
        message: Error message
    """
    logger.error(
        message,
        error_type=type(error).__name__,
        error_message=str(error),
        **context,
        exc_info=True
    )


def log_startup_validation(
    logger: structlog.BoundLogger,
    component: str,
    status: str,
    details: Optional[Dict[str, Any]] = None
) -> None:
    """
    Log startup validation check results.
    
    Args:
        logger: Structlog logger instance
        component: Component being validated (e.g., "ollama", "microphone")
        status: Validation status ("success", "warning", "failure")
        details: Additional details about the validation
    """
    log_data = {
        "event": "startup_validation",
        "component": component,
        "status": status
    }
    
    if details:
        log_data.update(details)
    
    if status == "success":
        logger.info(**log_data)
    elif status == "warning":
        logger.warning(**log_data)
    else:
        logger.error(**log_data)


def log_performance_metric(
    logger: structlog.BoundLogger,
    metric_name: str,
    value: float,
    unit: str,
    context: Optional[Dict[str, Any]] = None
) -> None:
    """
    Log a performance metric.
    
    Args:
        logger: Structlog logger instance
        metric_name: Name of the metric (e.g., "stt_latency")
        value: Metric value
        unit: Unit of measurement (e.g., "seconds", "ms")
        context: Additional context data
    """
    log_data = {
        "event": "performance_metric",
        "metric": metric_name,
        "value": round(value, 3),
        "unit": unit
    }
    
    if context:
        log_data.update(context)
    
    logger.debug(**log_data)


# Example usage and testing
if __name__ == "__main__":
    # Test logging setup
    from pathlib import Path
    
    log_file = Path("data/logs/test_aegis.log")
    
    # Test JSON format
    print("Testing JSON format...")
    logger = setup_logging(
        log_file=log_file,
        log_level="DEBUG",
        max_size_mb=1,
        backup_count=3,
        log_format="json",
        console_enabled=True
    )
    
    logger.debug("debug_message", detail="This is a debug message")
    logger.info("info_message", detail="This is an info message")
    logger.warning("warning_message", detail="This is a warning")
    logger.error("error_message", detail="This is an error")
    
    # Test turn metrics logging
    log_turn_metrics(
        logger=logger,
        turn_id="turn_001",
        duration=8.5,
        emotion="calm",
        health_signals={"mood": 7, "sleep_hours": 8.0},
        response_length=150,
        stage_durations={
            "stt": 3.2,
            "emotion": 0.5,
            "llm": 2.8,
            "tts": 2.0
        }
    )
    
    # Test proactive alert logging
    log_proactive_alert(
        logger=logger,
        alert_type="low_mood_pattern",
        severity="medium",
        context={"days": 3, "avg_mood": 3.5},
        message="Low mood detected for 3 consecutive days"
    )
    
    # Test error logging
    try:
        raise ValueError("Test error")
    except ValueError as e:
        log_error_with_context(
            logger=logger,
            error=e,
            context={"component": "test", "operation": "example"},
            message="test_error_occurred"
        )
    
    # Test startup validation logging
    log_startup_validation(
        logger=logger,
        component="ollama",
        status="success",
        details={"url": "http://localhost:11434", "model": "gemma:2b"}
    )
    
    # Test performance metric logging
    log_performance_metric(
        logger=logger,
        metric_name="stt_latency",
        value=3.245,
        unit="seconds",
        context={"model": "whisper-base", "audio_duration": 5.0}
    )
    
    print(f"\nTest logs written to: {log_file}")
    print("Check the file to verify JSON format and rotation settings.")
