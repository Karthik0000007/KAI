"""
Aegis Error Handling Module
Provides graceful degradation utilities for all components.
Implements retry logic, timeout handling, and fallback chains.
"""

import asyncio
import logging
from typing import Callable, TypeVar, Optional, Any
from functools import wraps

logger = logging.getLogger("aegis.error_handling")

T = TypeVar('T')


# ─── Retry with Exponential Backoff ─────────────────────────────────────────

async def with_retry(
    func: Callable[..., T],
    *args,
    max_retries: int = 3,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0,
    exceptions: tuple = (Exception,),
    **kwargs
) -> T:
    """
    Execute an async function with exponential backoff retry logic.
    
    Args:
        func: Async function to execute
        *args: Positional arguments for func
        max_retries: Maximum number of retry attempts (default: 3)
        initial_delay: Initial delay in seconds before first retry (default: 1.0)
        backoff_factor: Multiplier for delay between retries (default: 2.0)
        exceptions: Tuple of exception types to catch and retry (default: all exceptions)
        **kwargs: Keyword arguments for func
    
    Returns:
        Result of successful function execution
    
    Raises:
        Last exception if all retries exhausted
    
    Example:
        result = await with_retry(transcribe_audio, audio_path, max_retries=2)
    """
    last_exception = None
    delay = initial_delay
    
    # Handle edge case: max_retries=0 means no attempts
    if max_retries <= 0:
        raise ValueError("max_retries must be at least 1")
    
    for attempt in range(max_retries):
        try:
            if asyncio.iscoroutinefunction(func):
                return await func(*args, **kwargs)
            else:
                return await asyncio.to_thread(func, *args, **kwargs)
        except exceptions as e:
            last_exception = e
            
            if attempt < max_retries - 1:
                logger.warning(
                    f"Attempt {attempt + 1}/{max_retries} failed for {func.__name__}: {e}. "
                    f"Retrying in {delay:.1f}s..."
                )
                await asyncio.sleep(delay)
                delay *= backoff_factor
            else:
                logger.error(
                    f"All {max_retries} attempts failed for {func.__name__}: {e}"
                )
    
    # All retries exhausted
    raise last_exception


# ─── Timeout Wrapper ────────────────────────────────────────────────────────

async def with_timeout(
    func: Callable[..., T],
    *args,
    timeout: float,
    timeout_message: Optional[str] = None,
    **kwargs
) -> T:
    """
    Execute an async function with a timeout.
    
    Args:
        func: Async function to execute
        *args: Positional arguments for func
        timeout: Timeout in seconds
        timeout_message: Custom timeout error message
        **kwargs: Keyword arguments for func
    
    Returns:
        Result of successful function execution
    
    Raises:
        asyncio.TimeoutError: If operation exceeds timeout
    
    Example:
        result = await with_timeout(get_llm_response, prompt, timeout=120.0)
    """
    try:
        if asyncio.iscoroutinefunction(func):
            return await asyncio.wait_for(func(*args, **kwargs), timeout=timeout)
        else:
            return await asyncio.wait_for(
                asyncio.to_thread(func, *args, **kwargs),
                timeout=timeout
            )
    except asyncio.TimeoutError:
        msg = timeout_message or f"{func.__name__} timed out after {timeout}s"
        logger.error(msg)
        raise asyncio.TimeoutError(msg)


# ─── Fallback Chain ─────────────────────────────────────────────────────────

async def with_fallback(
    primary: Callable[..., T],
    fallback: Callable[..., T],
    *args,
    fallback_exceptions: tuple = (Exception,),
    **kwargs
) -> T:
    """
    Try primary function, fall back to secondary on failure.
    
    Args:
        primary: Primary async function to try first
        fallback: Fallback async function to use if primary fails
        *args: Positional arguments for both functions
        fallback_exceptions: Tuple of exception types that trigger fallback
        **kwargs: Keyword arguments for both functions
    
    Returns:
        Result from primary or fallback function
    
    Example:
        result = await with_fallback(
            voicevox_tts,
            pyttsx3_tts,
            text,
            language="ja"
        )
    """
    try:
        if asyncio.iscoroutinefunction(primary):
            return await primary(*args, **kwargs)
        else:
            return await asyncio.to_thread(primary, *args, **kwargs)
    except fallback_exceptions as e:
        logger.warning(
            f"Primary function {primary.__name__} failed: {e}. "
            f"Using fallback {fallback.__name__}"
        )
        
        if asyncio.iscoroutinefunction(fallback):
            return await fallback(*args, **kwargs)
        else:
            return await asyncio.to_thread(fallback, *args, **kwargs)


# ─── Combined Retry + Timeout ───────────────────────────────────────────────

async def with_retry_and_timeout(
    func: Callable[..., T],
    *args,
    max_retries: int = 3,
    timeout: float = 30.0,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0,
    **kwargs
) -> T:
    """
    Execute function with both retry logic and timeout.
    
    Args:
        func: Async function to execute
        *args: Positional arguments for func
        max_retries: Maximum number of retry attempts
        timeout: Timeout in seconds for each attempt
        initial_delay: Initial delay before first retry
        backoff_factor: Multiplier for delay between retries
        **kwargs: Keyword arguments for func
    
    Returns:
        Result of successful function execution
    
    Raises:
        Last exception if all retries exhausted
    
    Example:
        result = await with_retry_and_timeout(
            ollama_request,
            prompt,
            max_retries=2,
            timeout=120.0
        )
    """
    async def wrapped():
        return await with_timeout(func, *args, timeout=timeout, **kwargs)
    
    return await with_retry(
        wrapped,
        max_retries=max_retries,
        initial_delay=initial_delay,
        backoff_factor=backoff_factor
    )


# ─── Graceful Degradation Decorator ────────────────────────────────────────

def graceful_degradation(
    fallback_value: Any = None,
    log_error: bool = True,
    exceptions: tuple = (Exception,)
):
    """
    Decorator for graceful degradation - return fallback value on error.
    
    Args:
        fallback_value: Value to return on error (default: None)
        log_error: Whether to log the error (default: True)
        exceptions: Tuple of exception types to catch
    
    Example:
        @graceful_degradation(fallback_value={"emotion": "neutral"})
        async def analyze_emotion(audio_path):
            # ... emotion analysis logic
            pass
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                if asyncio.iscoroutinefunction(func):
                    return await func(*args, **kwargs)
                else:
                    return await asyncio.to_thread(func, *args, **kwargs)
            except exceptions as e:
                if log_error:
                    logger.error(
                        f"Error in {func.__name__}: {e}. "
                        f"Returning fallback value: {fallback_value}"
                    )
                return fallback_value
        return wrapper
    return decorator


# ─── Safe Execution Context ─────────────────────────────────────────────────

class SafeExecutionContext:
    """
    Context manager for safe execution with automatic error handling.
    
    Example:
        async with SafeExecutionContext("STT transcription") as ctx:
            text = await transcribe_audio(audio_path)
            ctx.result = text
        
        if ctx.success:
            print(f"Transcription: {ctx.result}")
        else:
            print(f"Transcription failed: {ctx.error}")
    """
    
    def __init__(self, operation_name: str, log_errors: bool = True):
        self.operation_name = operation_name
        self.log_errors = log_errors
        self.success = False
        self.error = None
        self.result = None
    
    async def __aenter__(self):
        logger.debug(f"Starting {self.operation_name}")
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            self.success = True
            logger.debug(f"{self.operation_name} completed successfully")
        else:
            self.success = False
            self.error = exc_val
            if self.log_errors:
                logger.error(
                    f"{self.operation_name} failed: {exc_val}",
                    exc_info=(exc_type, exc_val, exc_tb)
                )
        
        # Suppress exception (don't propagate)
        return True


# ─── Component-Specific Fallback Strategies ────────────────────────────────

class FallbackStrategies:
    """
    Predefined fallback strategies for each component.
    """
    
    @staticmethod
    async def stt_fallback(audio_path: str) -> tuple[str, str]:
        """
        STT fallback: Return empty transcript with warning.
        
        Returns:
            Tuple of (empty_text, default_language)
        """
        logger.warning("STT failed - returning empty transcript")
        return "", "en"
    
    @staticmethod
    async def emotion_fallback(audio_path: str) -> dict:
        """
        Emotion analysis fallback: Return neutral emotion.
        
        Returns:
            Dict with neutral emotion result
        """
        logger.warning("Emotion analysis failed - defaulting to neutral")
        from core.models import EmotionResult
        return EmotionResult(
            label="neutral",
            confidence=0.5,
            pitch_mean=0.0,
            pitch_std=0.0,
            energy_rms=0.0,
            speech_rate=0.0,
        )
    
    @staticmethod
    async def health_extraction_fallback(text: str) -> dict:
        """
        Health extraction fallback: Return empty signals.
        
        Returns:
            Empty dict (no health signals detected)
        """
        logger.warning("Health extraction failed - returning empty signals")
        return {}
    
    @staticmethod
    async def tts_fallback(text: str, language: str = "en") -> Optional[str]:
        """
        TTS fallback: Silent mode (skip audio output).
        
        Returns:
            None (no audio file generated)
        """
        logger.warning("TTS failed - entering silent mode (no audio output)")
        print(f"  [tts] Silent mode: {text}")
        return None
    
    @staticmethod
    async def database_fallback(operation: str, data: Any = None) -> Any:
        """
        Database fallback: Log warning and continue.
        
        For read operations: return empty/default values
        For write operations: log data loss warning
        
        Returns:
            Default value based on operation type
        """
        logger.error(f"Database operation '{operation}' failed - data may be lost")
        
        # Return appropriate defaults for read operations
        if "get" in operation or "query" in operation:
            if "stats" in operation:
                return {"count": 0, "avg_mood": None, "avg_sleep": None, "avg_energy": None}
            elif "alerts" in operation:
                return []
            elif "history" in operation:
                return []
            else:
                return None
        
        # For write operations, just log and return None
        return None


# ─── Utility Functions ──────────────────────────────────────────────────────

def is_critical_error(exception: Exception) -> bool:
    """
    Determine if an error is critical (should stop execution).
    
    Args:
        exception: The exception to check
    
    Returns:
        True if error is critical, False otherwise
    """
    critical_errors = (
        KeyboardInterrupt,
        SystemExit,
        MemoryError,
    )
    return isinstance(exception, critical_errors)


async def log_and_continue(
    func: Callable[..., T],
    *args,
    default_value: Any = None,
    **kwargs
) -> Any:
    """
    Execute function and return default value on error (don't raise).
    
    Args:
        func: Function to execute
        *args: Positional arguments
        default_value: Value to return on error
        **kwargs: Keyword arguments
    
    Returns:
        Function result or default_value on error
    """
    try:
        if asyncio.iscoroutinefunction(func):
            return await func(*args, **kwargs)
        else:
            return await asyncio.to_thread(func, *args, **kwargs)
    except Exception as e:
        logger.error(f"Error in {func.__name__}: {e}. Continuing with default value.")
        return default_value
