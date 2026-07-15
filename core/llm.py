"""
Aegis LLM Module
Health-aware, emotion-adaptive local inference via Ollama.
Fully offline — connects to a local Ollama instance.
"""

import asyncio
import json
import logging
import random
from typing import Optional, List, Dict, Any

import requests

from core.config import (
    OLLAMA_URL, OLLAMA_MODEL, AEGIS_SYSTEM_PROMPT,
    TONE_MODES, LLM_CONTEXT_WINDOW,
)

logger = logging.getLogger("aegis.llm")


# ─── Fallback Response Templates ────────────────────────────────────────────

FALLBACK_TEMPLATES = {
    "general_conversation": {
        "calm": [
            "I'm here with you. Could you tell me more about how you're feeling?",
            "I'm listening. Take your time and share what's on your mind.",
            "I want to understand better. What's been happening with you?",
        ],
        "stressed": [
            "I can hear that things feel overwhelming right now. Let's take this one step at a time.",
            "It sounds like you're dealing with a lot. I'm here to support you through this.",
            "Take a deep breath with me. We'll work through this together.",
        ],
        "anxious": [
            "I'm here for you. Let's focus on what we can control right now.",
            "Your feelings are valid. What would help you feel more grounded?",
            "I'm listening without judgment. You're safe to share whatever you need.",
        ],
        "fatigued": [
            "It sounds like you're really tired. Have you been able to rest?",
            "Your body might be telling you it needs more rest. How have you been sleeping?",
            "Fatigue can be tough. Let's talk about what might help you feel more energized.",
        ],
        "neutral": [
            "I'm here to listen. What would you like to talk about?",
            "How can I support you today?",
            "Tell me what's on your mind.",
        ],
    },
    "health_checkin": {
        "calm": [
            "Thank you for sharing that with me. How else are you feeling today?",
            "I've noted that. Is there anything else about your health you'd like to mention?",
            "I appreciate you keeping track of your health. What else should I know?",
        ],
        "stressed": [
            "I've recorded that information. Remember, tracking your health is a positive step.",
            "Thank you for sharing. Taking care of yourself matters, even when things are hard.",
            "I've got that noted. You're doing well by staying aware of your health.",
        ],
        "anxious": [
            "I've saved that information. You're taking good care of yourself by tracking this.",
            "Thank you for telling me. Monitoring your health is important, and you're doing great.",
            "I've recorded that. Remember, I'm here to help you stay on top of your wellbeing.",
        ],
        "fatigued": [
            "I've noted that. Rest is important - are you getting enough sleep?",
            "Thank you for sharing. Your body needs care, especially when you're tired.",
            "I've recorded that. Make sure you're giving yourself time to recover.",
        ],
        "neutral": [
            "I've recorded that information. What else would you like to share?",
            "Thank you for the update. Is there anything else?",
            "I've noted that. How else can I help you today?",
        ],
    },
    "emotional_support": {
        "calm": [
            "I'm glad you're sharing this with me. Your feelings matter.",
            "Thank you for trusting me with this. I'm here for you.",
            "I hear you. Let's work through this together.",
        ],
        "stressed": [
            "I can hear how difficult this is for you. You're not alone in this.",
            "What you're feeling is completely understandable. I'm here to support you.",
            "This sounds really challenging. Let's take it one moment at a time.",
        ],
        "anxious": [
            "Your concerns are valid. Let's talk through what's worrying you.",
            "I'm here with you. Anxiety can be overwhelming, but we'll face it together.",
            "It's okay to feel this way. What would help you feel safer right now?",
        ],
        "fatigued": [
            "Being tired can make everything feel harder. You deserve rest and care.",
            "Exhaustion affects everything. Let's talk about how to help you recover.",
            "Your tiredness is real. What would help you feel more rested?",
        ],
        "neutral": [
            "I'm listening. Tell me more about what you're experiencing.",
            "I'm here to support you. What do you need right now?",
            "Your wellbeing matters to me. How can I help?",
        ],
    },
    "proactive_alert": {
        "calm": [
            "I've noticed some patterns in your health data. Can we talk about them?",
            "I want to check in with you about something I've observed. Do you have a moment?",
            "There's something I'd like to discuss with you about your recent health patterns.",
        ],
        "stressed": [
            "I've noticed you've been having a tough time lately. I'm concerned and want to help.",
            "Your recent health patterns show you might be struggling. Let's talk about it.",
            "I'm here because I care about you. I've seen some concerning patterns we should discuss.",
        ],
        "anxious": [
            "I've been monitoring your health, and I want to make sure you're okay. Can we talk?",
            "I've noticed some changes that concern me. You're not alone - let's work through this.",
            "I'm reaching out because I care. There are some patterns we should address together.",
        ],
        "fatigued": [
            "I've noticed you've been really tired lately. Let's talk about what might help.",
            "Your energy levels have been low. I'm concerned and want to support you.",
            "I can see you're exhausted. Let's discuss ways to help you feel better.",
        ],
        "neutral": [
            "I've observed some patterns in your health data. Let's review them together.",
            "There are some health trends I'd like to discuss with you.",
            "I have some observations about your recent health that we should talk about.",
        ],
    },
}


def get_fallback_response(
    emotion_label: Optional[str] = None,
    health_stats: Optional[Dict[str, Any]] = None,
    active_alerts: Optional[List[Dict]] = None,
    conversation_history: Optional[List[Dict]] = None,
) -> str:
    """
    Generate a contextually appropriate fallback response when LLM is unavailable.
    
    Uses emotion, health signals, conversation history, and alerts to select
    the most appropriate template category and tone.
    
    Args:
        emotion_label: Detected emotion from voice analysis.
        health_stats: Aggregated health stats from the database.
        active_alerts: Unacknowledged proactive alerts.
        conversation_history: Recent conversation turns.
    
    Returns:
        Contextually appropriate fallback response string.
    """
    # Determine emotion tone (default to neutral)
    tone = emotion_label if emotion_label in FALLBACK_TEMPLATES["general_conversation"] else "neutral"
    
    # Determine scenario category based on context
    category = "general_conversation"  # default
    
    # Check if there are active alerts
    if active_alerts and len(active_alerts) > 0:
        category = "proactive_alert"
    
    # Check if user mentioned health signals in recent conversation
    elif conversation_history and len(conversation_history) > 0:
        last_user_message = None
        for turn in reversed(conversation_history):
            if turn.get("role") == "user":
                last_user_message = turn.get("content", "").lower()
                break
        
        if last_user_message:
            # Check for health-related keywords
            health_keywords = ["sleep", "mood", "energy", "medication", "pain", "tired", "feeling"]
            if any(keyword in last_user_message for keyword in health_keywords):
                category = "health_checkin"
            
            # Check for emotional keywords
            emotional_keywords = ["worried", "scared", "anxious", "stressed", "sad", "depressed", 
                                "upset", "angry", "frustrated", "overwhelmed"]
            if any(keyword in last_user_message for keyword in emotional_keywords):
                category = "emotional_support"
    
    # Check if health stats show concerning patterns
    elif health_stats and health_stats.get("count", 0) > 0:
        if health_stats.get("low_mood_days", 0) >= 2:
            category = "emotional_support"
        elif health_stats.get("avg_sleep", 0) < 6.0:
            category = "emotional_support"
    
    # Select random template from appropriate category and tone
    templates = FALLBACK_TEMPLATES.get(category, {}).get(tone, FALLBACK_TEMPLATES["general_conversation"]["neutral"])
    response = random.choice(templates)
    
    logger.info(f"Generated fallback response: category={category}, tone={tone}")
    
    return response


def build_health_context(
    emotion_label: Optional[str] = None,
    tone_mode: Optional[str] = None,
    health_stats: Optional[Dict[str, Any]] = None,
    active_alerts: Optional[List[Dict]] = None,
    conversation_history: Optional[List[Dict]] = None,
) -> str:
    """
    Build a rich context prompt incorporating:
      - System personality
      - Current emotion state
      - Tone adaptation
      - Recent health stats
      - Pending proactive alerts
      - Conversation history
    """
    parts = [AEGIS_SYSTEM_PROMPT]

    # Tone adaptation
    if tone_mode and tone_mode in TONE_MODES:
        parts.append(f"\n[TONE]: {TONE_MODES[tone_mode]['system_modifier']}")

    # Emotion context
    if emotion_label:
        parts.append(
            f"\n[EMOTION DETECTED]: The user currently sounds {emotion_label}. "
            f"Adapt your response accordingly."
        )

    # Health stats summary
    if health_stats and health_stats.get("count", 0) > 0:
        stats_lines = ["\n[RECENT HEALTH CONTEXT]:"]
        if health_stats.get("avg_mood") is not None:
            stats_lines.append(f"  - Average mood (7 days): {health_stats['avg_mood']}/10")
        if health_stats.get("avg_sleep") is not None:
            stats_lines.append(f"  - Average sleep (7 days): {health_stats['avg_sleep']} hrs")
        if health_stats.get("avg_energy") is not None:
            stats_lines.append(f"  - Average energy (7 days): {health_stats['avg_energy']}/10")
        if health_stats.get("low_mood_days", 0) > 0:
            stats_lines.append(f"  - Low mood days: {health_stats['low_mood_days']}")
        if health_stats.get("recent_emotions"):
            stats_lines.append(
                f"  - Recent detected emotions: {', '.join(health_stats['recent_emotions'])}"
            )
        parts.append("\n".join(stats_lines))

    # Active alerts
    if active_alerts:
        alert_lines = ["\n[PENDING ALERTS — mention gently if relevant]:"]
        for alert in active_alerts[:3]:
            alert_lines.append(f"  - [{alert.get('severity', 'info')}] {alert.get('message', '')}")
        parts.append("\n".join(alert_lines))

    # Conversation history
    if conversation_history:
        history_lines = ["\n[CONVERSATION HISTORY]:"]
        for turn in conversation_history[-8:]:
            role = turn.get("role", "user")
            content = turn.get("content", "")
            history_lines.append(f"  {role}: {content}")
        parts.append("\n".join(history_lines))

    return "\n".join(parts)


def get_response(
    user_input: str,
    emotion_label: Optional[str] = None,
    tone_mode: Optional[str] = None,
    health_stats: Optional[Dict[str, Any]] = None,
    active_alerts: Optional[List[Dict]] = None,
    conversation_history: Optional[List[Dict]] = None,
    language: Optional[str] = None,
) -> str:
    """
    Get a health-aware, emotion-adaptive response from the local LLM.
    
    Implements fallback logic when Ollama is unavailable:
    - Timeout handling (120s)
    - Retry logic with exponential backoff
    - Contextual fallback responses
    
    Args:
        user_input: Transcribed user speech.
        emotion_label: Detected emotion from voice analysis.
        tone_mode: Response tone mode (calm/encouraging/gentle_support/neutral).
        health_stats: Aggregated health stats from the database.
        active_alerts: Unacknowledged proactive alerts.
        conversation_history: Recent conversation turns.
        language: Detected language code (e.g. 'en', 'ja').
    
    Returns:
        LLM response string or fallback response if LLM unavailable.
    """
    context = build_health_context(
        emotion_label=emotion_label,
        tone_mode=tone_mode,
        health_stats=health_stats,
        active_alerts=active_alerts,
        conversation_history=conversation_history,
    )

    # Instruct LLM to respond in the user's language with equal detail
    lang_instruction = ""
    if language and language != "en":
        lang_names = {"ja": "Japanese", "zh": "Chinese", "ko": "Korean",
                      "es": "Spanish", "fr": "French", "de": "German"}
        lang_name = lang_names.get(language, language)
        lang_instruction = (
            f"\n[LANGUAGE]: The user is speaking {lang_name}. "
            f"You MUST respond entirely in {lang_name}. "
            f"Do NOT respond in English. Do NOT mix languages.\n"
            f"Your response MUST be detailed and caring — at least 3-5 sentences "
            f"in {lang_name}. Show the same warmth, empathy, and depth as you would "
            f"in English. Ask follow-up questions, offer suggestions, and show concern."
        )

    full_prompt = f"{context}{lang_instruction}\n\nUser: {user_input}\nAegis:"

    logger.info(f"Sending prompt to Ollama ({OLLAMA_MODEL})...")
    logger.debug(f"Prompt length: {len(full_prompt)} chars")

    # Retry configuration
    max_retries = 2
    timeout_seconds = 120
    
    for attempt in range(max_retries):
        try:
            response = requests.post(
                OLLAMA_URL,
                json={
                    "model": OLLAMA_MODEL,
                    "prompt": full_prompt,
                    "stream": False,
                    "options": {
                        "num_ctx": LLM_CONTEXT_WINDOW,
                        "temperature": 0.7,
                        "top_p": 0.9,
                    }
                },
                timeout=timeout_seconds,
            )
            response.raise_for_status()
            reply = response.json().get("response", "").strip()

            if not reply:
                logger.warning("Empty response from LLM")
                # Use fallback for empty response
                return get_fallback_response(
                    emotion_label=emotion_label,
                    health_stats=health_stats,
                    active_alerts=active_alerts,
                    conversation_history=conversation_history,
                )

            logger.info("LLM response received successfully")
            return reply

        except requests.ConnectionError as e:
            logger.error(f"Cannot connect to Ollama (attempt {attempt + 1}/{max_retries}): {e}")
            
            if attempt < max_retries - 1:
                # Exponential backoff: 2^attempt seconds
                wait_time = 2 ** attempt
                logger.info(f"Retrying in {wait_time} seconds...")
                import time
                time.sleep(wait_time)
                continue
            else:
                # Final attempt failed, use fallback
                logger.error("All retry attempts exhausted. Using fallback response.")
                fallback = get_fallback_response(
                    emotion_label=emotion_label,
                    health_stats=health_stats,
                    active_alerts=active_alerts,
                    conversation_history=conversation_history,
                )
                logger.info("Fallback response: LLM unavailable - please ensure Ollama is running")
                return fallback
        
        except requests.Timeout as e:
            logger.error(f"Ollama request timed out after {timeout_seconds}s (attempt {attempt + 1}/{max_retries}): {e}")
            
            if attempt < max_retries - 1:
                # Retry once on timeout
                wait_time = 2 ** attempt
                logger.info(f"Retrying in {wait_time} seconds...")
                import time
                time.sleep(wait_time)
                continue
            else:
                # Timeout on final attempt, use fallback
                logger.error("Request timed out on final attempt. Using fallback response.")
                fallback = get_fallback_response(
                    emotion_label=emotion_label,
                    health_stats=health_stats,
                    active_alerts=active_alerts,
                    conversation_history=conversation_history,
                )
                logger.info("Fallback response: LLM timeout - the model may be overloaded")
                return fallback
        
        except requests.RequestException as e:
            logger.error(f"Ollama request failed (attempt {attempt + 1}/{max_retries}): {e}")
            
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                logger.info(f"Retrying in {wait_time} seconds...")
                import time
                time.sleep(wait_time)
                continue
            else:
                # Request failed on final attempt, use fallback
                logger.error("Request failed on final attempt. Using fallback response.")
                fallback = get_fallback_response(
                    emotion_label=emotion_label,
                    health_stats=health_stats,
                    active_alerts=active_alerts,
                    conversation_history=conversation_history,
                )
                logger.info("Fallback response: LLM error occurred")
                return fallback
        
        except Exception as e:
            logger.error(f"Unexpected LLM error (attempt {attempt + 1}/{max_retries}): {e}")
            
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                logger.info(f"Retrying in {wait_time} seconds...")
                import time
                time.sleep(wait_time)
                continue
            else:
                # Unexpected error on final attempt, use fallback
                logger.error("Unexpected error on final attempt. Using fallback response.")
                fallback = get_fallback_response(
                    emotion_label=emotion_label,
                    health_stats=health_stats,
                    active_alerts=active_alerts,
                    conversation_history=conversation_history,
                )
                logger.info("Fallback response: Unexpected error occurred")
                return fallback
    
    # Should never reach here, but just in case
    return get_fallback_response(
        emotion_label=emotion_label,
        health_stats=health_stats,
        active_alerts=active_alerts,
        conversation_history=conversation_history,
    )


# ─── Health Signal Validation ────────────────────────────────────────────────

# Valid ranges for health signals
HEALTH_SIGNAL_RANGES = {
    'sleep_hours': (0, 24),           # hours
    'mood_score': (1, 10),            # 1-10 scale
    'energy_level': (1, 10),          # 1-10 scale
    'pain_intensity': (1, 10),        # 1-10 scale
    'heart_rate': (30, 220),          # bpm (for wearables)
    'spo2': (70, 100),                # percentage
    'temperature': (35.0, 42.0),      # Celsius
    'steps': (0, 100000),             # daily steps
}


def validate_vital(vital_type: str, value: float) -> bool:
    """
    Validate a health signal value for plausibility.
    
    Checks if the value falls within the expected range for the given signal type.
    This prevents invalid data from corrupting the health database.
    
    Args:
        vital_type: Type of health signal (e.g., 'sleep_hours', 'mood_score', 'heart_rate')
        value: The value to validate
    
    Returns:
        True if the value is within the valid range, False otherwise
    
    Examples:
        >>> validate_vital('sleep_hours', 8.0)
        True
        >>> validate_vital('sleep_hours', 25.0)
        False
        >>> validate_vital('mood_score', 5.0)
        True
        >>> validate_vital('mood_score', 11.0)
        False
        >>> validate_vital('heart_rate', 75)
        True
        >>> validate_vital('heart_rate', 250)
        False
    """
    if vital_type not in HEALTH_SIGNAL_RANGES:
        logger.warning(f"Unknown vital type: {vital_type}")
        return False
    
    min_val, max_val = HEALTH_SIGNAL_RANGES[vital_type]
    is_valid = min_val <= value <= max_val
    
    if not is_valid:
        logger.warning(f"Invalid {vital_type} value: {value} (valid range: {min_val}-{max_val})")
    
    return is_valid


def calculate_extraction_confidence(signals: Dict[str, Any], extraction_method: str = "regex") -> float:
    """
    Calculate confidence score for extracted health signals.
    
    Confidence is based on:
    - Number of signals extracted (more signals = higher confidence)
    - Extraction method (regex = high confidence, LLM = variable confidence)
    - Signal validity (all valid = higher confidence)
    
    Args:
        signals: Dict of extracted health signals
        extraction_method: "regex" or "llm"
    
    Returns:
        Confidence score between 0.0 and 1.0
    """
    if not signals or all(v is None for v in signals.values()):
        return 0.0
    
    # Count valid signals (excluding metadata fields)
    metadata_fields = {'confidence', 'pain_mentioned'}
    signal_fields = [k for k in signals.keys() if k not in metadata_fields and signals[k] is not None]
    num_signals = len(signal_fields)
    
    if num_signals == 0:
        return 0.0
    
    # Base confidence by extraction method
    if extraction_method == "regex":
        # Regex extraction is high confidence when it finds explicit patterns
        base_confidence = 0.9
    elif extraction_method == "llm":
        # LLM extraction confidence is provided by the LLM itself
        base_confidence = signals.get('confidence', 0.5)
    else:
        base_confidence = 0.5
    
    # Validate all extracted signals
    validation_scores = []
    for field in signal_fields:
        value = signals[field]
        if field in HEALTH_SIGNAL_RANGES:
            is_valid = validate_vital(field, value)
            validation_scores.append(1.0 if is_valid else 0.0)
    
    # If we have validation scores, factor them in
    if validation_scores:
        validation_factor = sum(validation_scores) / len(validation_scores)
        # Invalid signals reduce confidence
        confidence = base_confidence * validation_factor
    else:
        # No validatable signals (e.g., medication_taken, pain_location)
        confidence = base_confidence
    
    # Boost confidence slightly for multiple signals (up to 3 signals)
    signal_boost = min(num_signals / 3.0, 1.0) * 0.1
    confidence = min(confidence + signal_boost, 1.0)
    
    return round(confidence, 2)


# ─── LLM-Based Health Signal Extraction ─────────────────────────────────────

# JSON Schema for LLM extraction output
HEALTH_SIGNAL_SCHEMA = {
    "type": "object",
    "properties": {
        "sleep_hours": {
            "type": ["number", "null"],
            "minimum": 0,
            "maximum": 24,
            "description": "Hours of sleep (0-24), null if not mentioned"
        },
        "mood_score": {
            "type": ["number", "null"],
            "minimum": 1,
            "maximum": 10,
            "description": "Mood score (1-10), null if not mentioned"
        },
        "energy_level": {
            "type": ["number", "null"],
            "minimum": 1,
            "maximum": 10,
            "description": "Energy level (1-10), null if not mentioned"
        },
        "medication_taken": {
            "type": ["boolean", "null"],
            "description": "Whether medication was taken, null if not mentioned"
        },
        "pain_location": {
            "type": ["string", "null"],
            "description": "Location of pain (e.g., 'head', 'back', 'chest'), null if not mentioned"
        },
        "pain_intensity": {
            "type": ["number", "null"],
            "minimum": 1,
            "maximum": 10,
            "description": "Pain intensity (1-10), null if not mentioned"
        },
        "confidence": {
            "type": "number",
            "minimum": 0,
            "maximum": 1,
            "description": "Confidence score for the extraction (0-1)"
        }
    },
    "required": ["confidence"]
}

# Few-shot examples for LLM extraction
FEW_SHOT_EXAMPLES = [
    {
        "text": "I slept really well last night, feeling great today!",
        "output": {
            "sleep_hours": 8.0,
            "mood_score": 8.0,
            "energy_level": 8.0,
            "medication_taken": None,
            "pain_location": None,
            "pain_intensity": None,
            "confidence": 0.85
        }
    },
    {
        "text": "Barely got any rest, maybe 4 hours. Feeling exhausted and my head is killing me.",
        "output": {
            "sleep_hours": 4.0,
            "mood_score": 3.0,
            "energy_level": 2.0,
            "medication_taken": None,
            "pain_location": "head",
            "pain_intensity": 8.0,
            "confidence": 0.9
        }
    },
    {
        "text": "Forgot to take my medication this morning. Not feeling great.",
        "output": {
            "sleep_hours": None,
            "mood_score": 4.0,
            "energy_level": 4.0,
            "medication_taken": False,
            "pain_location": None,
            "pain_intensity": None,
            "confidence": 0.8
        }
    },
    {
        "text": "よく眠れた。今日は元気いっぱい！",
        "output": {
            "sleep_hours": 8.0,
            "mood_score": 8.0,
            "energy_level": 9.0,
            "medication_taken": None,
            "pain_location": None,
            "pain_intensity": None,
            "confidence": 0.85
        }
    },
    {
        "text": "Dormí 6 horas. Me duele un poco la cabeza.",
        "output": {
            "sleep_hours": 6.0,
            "mood_score": 5.0,
            "energy_level": 5.0,
            "medication_taken": None,
            "pain_location": "head",
            "pain_intensity": 4.0,
            "confidence": 0.85
        }
    },
    {
        "text": "J'ai bien pris mes médicaments. Je me sens super bien !",
        "output": {
            "sleep_hours": None,
            "mood_score": 9.0,
            "energy_level": 9.0,
            "medication_taken": True,
            "pain_location": None,
            "pain_intensity": None,
            "confidence": 0.9
        }
    },
    {
        "text": "Ich habe schlecht geschlafen, nur 4 Stunden. Ich bin sehr müde.",
        "output": {
            "sleep_hours": 4.0,
            "mood_score": 3.0,
            "energy_level": 2.0,
            "medication_taken": None,
            "pain_location": None,
            "pain_intensity": None,
            "confidence": 0.85
        }
    }
]


def build_extraction_prompt(text: str, language: Optional[str] = None) -> str:
    """
    Build a structured prompt for LLM-based health signal extraction.
    
    Uses few-shot learning with examples to guide the LLM to extract
    health signals in a consistent JSON format.
    
    Args:
        text: User text to extract health signals from
        language: Detected language code (e.g., 'en', 'ja')
    
    Returns:
        Formatted prompt string for LLM
    """
    # Language-specific instructions
    lang_instruction = ""
    if language == "ja":
        lang_instruction = """
Note: The text is in Japanese. Extract health signals appropriately:
- 睡眠 (sleep) → sleep_hours
- 気分/調子 (mood) → mood_score
- エネルギー/元気 (energy) → energy_level
- 薬 (medication) → medication_taken
- 痛み (pain) → pain_location, pain_intensity
"""
    elif language == "es":
        lang_instruction = """
Note: The text is in Spanish. Extract health signals appropriately:
- sueño/dormir (sleep) → sleep_hours
- humor/estado de ánimo (mood) → mood_score
- energía (energy) → energy_level
- medicación/pastillas (medication) → medication_taken
- dolor (pain) → pain_location, pain_intensity
"""
    elif language == "fr":
        lang_instruction = """
Note: The text is in French. Extract health signals appropriately:
- sommeil/dormi (sleep) → sleep_hours
- humeur (mood) → mood_score
- énergie (energy) → energy_level
- médicaments (medication) → medication_taken
- douleur/mal (pain) → pain_location, pain_intensity
"""
    elif language == "de":
        lang_instruction = """
Note: The text is in German. Extract health signals appropriately:
- schlaf/geschlafen (sleep) → sleep_hours
- stimmung (mood) → mood_score
- energie (energy) → energy_level
- medikamente (medication) → medication_taken
- schmerzen (pain) → pain_location, pain_intensity
"""
    
    # Build few-shot examples section
    examples_section = "Examples:\n\n"
    for i, example in enumerate(FEW_SHOT_EXAMPLES, 1):
        examples_section += f"Example {i}:\n"
        examples_section += f"Text: \"{example['text']}\"\n"
        examples_section += f"JSON: {json.dumps(example['output'], ensure_ascii=False)}\n\n"
    
    # Build complete prompt
    prompt = f"""You are a health signal extraction system. Extract structured health information from user text.

Return ONLY valid JSON with these fields (use null if not mentioned):
- sleep_hours: number (0-24) - hours of sleep
- mood_score: number (1-10) - mood/feeling score
- energy_level: number (1-10) - energy level
- medication_taken: boolean - whether medication was taken
- pain_location: string - location of pain (e.g., "head", "back", "chest")
- pain_intensity: number (1-10) - pain intensity
- confidence: number (0-1) - your confidence in the extraction

{lang_instruction}

{examples_section}

Now extract from this text:
Text: "{text}"

JSON:"""
    
    return prompt


def extract_health_signals(text: str, language: Optional[str] = None) -> Dict[str, Any]:
    """
    Extract structured health signals from user text using two-tier extraction.
    
    **Two-Tier Architecture**:
    - **Tier 1 (Regex)**: Fast pattern matching for explicit mentions (fast path)
    - **Tier 2 (LLM)**: Structured extraction for paraphrased/contextual mentions (fallback)
    
    The function first attempts regex-based extraction. If no signals are found,
    it falls back to LLM-based extraction for more robust natural language understanding.
    
    Supports both English and Japanese text with language-specific patterns.
    
    Args:
        text: User text to extract health signals from
        language: Detected language code (e.g., 'en', 'ja') - used for LLM fallback
    
    Returns:
        Dict with any detected values:
            - mood_score: float (1-10) if mentioned
            - sleep_hours: float if mentioned
            - energy_level: float (1-10) if mentioned
            - medication_taken: bool if mentioned
            - pain_mentioned: bool (regex tier)
            - pain_location: str (LLM tier)
            - pain_intensity: float (1-10, LLM tier)
            - confidence: float (0-1, LLM tier only)
    """
    import re
    
    # ─── TIER 1: REGEX-BASED EXTRACTION (FAST PATH) ─────────────────────────
    
    logger.debug("Starting Tier 1 (regex) extraction")
    signals: Dict[str, Any] = {}
    text_lower = text.lower()

    # ─── English Patterns ────────────────────────────────────────────────────

    # Sleep hours: "slept 6 hours", "got 5 hours of sleep", etc.
    sleep_match = re.search(
        r'(?:slept|sleep|got)\s+(?:about\s+)?(\d+(?:\.\d+)?)\s*(?:hours?|hrs?)', text_lower
    )
    if sleep_match:
        signals["sleep_hours"] = float(sleep_match.group(1))

    # Mood/feeling score: "my mood is 7", "feeling like a 5", etc.
    mood_match = re.search(
        r'(?:mood|feeling)\s+(?:is\s+)?(?:like\s+)?(?:a\s+)?(\d+)(?:\s*(?:out of|\/)\s*10)?',
        text_lower
    )
    if mood_match:
        score = int(mood_match.group(1))
        if 1 <= score <= 10:
            signals["mood_score"] = float(score)

    # Energy: "energy is 4", "energy level 6", etc.
    energy_match = re.search(
        r'energy\s+(?:level\s+)?(?:is\s+)?(\d+)(?:\s*(?:out of|\/)\s*10)?', text_lower
    )
    if energy_match:
        score = int(energy_match.group(1))
        if 1 <= score <= 10:
            signals["energy_level"] = float(score)

    # Medication
    med_positive = any(kw in text_lower for kw in [
        "took my med", "taken my med", "took medicine", "had my pill",
        "yes i took", "already taken"
    ])
    med_negative = any(kw in text_lower for kw in [
        "forgot my med", "didn't take", "haven't taken", "missed my",
        "no i didn't", "forgot to take"
    ])
    if med_positive:
        signals["medication_taken"] = True
    elif med_negative:
        signals["medication_taken"] = False

    # Pain
    pain_keywords = ["pain", "hurt", "ache", "sore", "cramp", "headache",
                     "migraine", "backache", "nausea", "dizzy"]
    if any(kw in text_lower for kw in pain_keywords):
        signals["pain_mentioned"] = True

    # Qualitative mood inference (English)
    if "mood_score" not in signals:
        positive = ["great", "wonderful", "amazing", "fantastic", "good",
                     "happy", "cheerful", "excellent", "on top of the world"]
        negative = ["terrible", "awful", "bad", "horrible", "depressed",
                     "miserable", "sad", "down", "low"]
        if any(w in text_lower for w in positive):
            signals["mood_score"] = 7.5
        elif any(w in text_lower for w in negative):
            signals["mood_score"] = 3.0
    
    # Qualitative sleep patterns (English)
    if "sleep_hours" not in signals:
        poor_sleep = ["barely slept", "hardly slept", "couldn't sleep", "no sleep"]
        if any(pattern in text_lower for pattern in poor_sleep):
            signals["sleep_hours"] = 3.0  # Poor sleep

    # Pain
    pain_keywords = ["pain", "hurt", "ache", "sore", "cramp", "headache",
                     "migraine", "backache", "nausea", "dizzy", "killing me"]
    if any(kw in text_lower for kw in pain_keywords):
        signals["pain_mentioned"] = True
    # ─── Japanese Patterns ────────────────────────────────────────────────────
    ja_sleep_match = re.search(r'(\d+(?:\.\d+)?)\s*時間(?:寝|睡眠)', text)
    if ja_sleep_match:
        signals["sleep_hours"] = float(ja_sleep_match.group(1))
        
    if "薬" in text and ("飲んだ" in text or "のみました" in text):
        signals["medication_taken"] = True
    elif "薬" in text and ("忘れた" in text or "のんでない" in text):
        signals["medication_taken"] = False
        
    if "痛" in text:
        signals["pain_mentioned"] = True

    # ─── Spanish Patterns ─────────────────────────────────────────────────────
    es_sleep_match = re.search(r'(?:dormí|dormido)\s+(?:unas\s+)?(\d+(?:\.\d+)?)\s*horas?', text_lower)
    if es_sleep_match:
        signals["sleep_hours"] = float(es_sleep_match.group(1))
        
    if "pastilla" in text_lower or "medicación" in text_lower or "medicamento" in text_lower:
        if "tomé" in text_lower or "tomado" in text_lower:
            signals["medication_taken"] = True
        elif "olvidé" in text_lower or "no tomé" in text_lower:
            signals["medication_taken"] = False
            
    if "dolor" in text_lower or "duele" in text_lower:
        signals["pain_mentioned"] = True

    # ─── French Patterns ──────────────────────────────────────────────────────
    fr_sleep_match = re.search(r'(?:dormi)\s+(?:environ\s+)?(\d+(?:\.\d+)?)\s*heures?', text_lower)
    if fr_sleep_match:
        signals["sleep_hours"] = float(fr_sleep_match.group(1))
        
    if "médicament" in text_lower or "pilule" in text_lower:
        if "oublié" in text_lower or "pas pris" in text_lower:
            signals["medication_taken"] = False
        elif "pris" in text_lower:
            signals["medication_taken"] = True
            
    if "douleur" in text_lower or "mal à" in text_lower:
        signals["pain_mentioned"] = True

    # ─── German Patterns ──────────────────────────────────────────────────────
    de_sleep_match = re.search(r'(\d+(?:\.\d+)?)\s*stunden?\s*(?:geschlafen)', text_lower)
    if de_sleep_match:
        signals["sleep_hours"] = float(de_sleep_match.group(1))
        
    if "medikament" in text_lower or "tablette" in text_lower or "pille" in text_lower:
        if "vergessen" in text_lower or "nicht genommen" in text_lower or "keine" in text_lower:
            signals["medication_taken"] = False
        elif "genommen" in text_lower:
            signals["medication_taken"] = True
            
    if "schmerz" in text_lower or "wehtun" in text_lower:
        signals["pain_mentioned"] = True
    if ja_sleep_match:
        signals["sleep_hours"] = float(ja_sleep_match.group(1))
    
    # Qualitative Japanese sleep patterns
    if "sleep_hours" not in signals:
        if any(kw in text for kw in ["よく眠れた", "ぐっすり", "熟睡"]):
            signals["sleep_hours"] = 8.0  # Good sleep
        elif any(kw in text for kw in ["あまり寝られなかった", "眠れなかった", "不眠"]):
            signals["sleep_hours"] = 4.0  # Poor sleep

    # Japanese mood patterns: "気分は7", "調子は5", "元気", "落ち込んでいる"
    ja_mood_match = re.search(r'(?:気分|調子)(?:は|が)?(\d+)', text)
    if ja_mood_match:
        score = int(ja_mood_match.group(1))
        if 1 <= score <= 10:
            signals["mood_score"] = float(score)

    # Japanese energy patterns: "エネルギーは4", "疲れている", "元気いっぱい"
    ja_energy_match = re.search(r'エネルギー(?:は|が)?(\d+)', text)
    if ja_energy_match:
        score = int(ja_energy_match.group(1))
        if 1 <= score <= 10:
            signals["energy_level"] = float(score)

    # Qualitative Japanese energy patterns
    if "energy_level" not in signals:
        if any(kw in text for kw in ["元気いっぱい", "元気", "活力"]):
            signals["energy_level"] = 8.0  # High energy
        elif any(kw in text for kw in ["疲れている", "疲れた", "だるい", "倦怠感"]):
            signals["energy_level"] = 3.0  # Low energy

    # Japanese medication patterns: "薬を飲んだ", "薬を忘れた", "服薬した"
    ja_med_positive = any(kw in text for kw in [
        "薬を飲んだ", "薬飲んだ", "服薬した", "服用した", "飲みました", "服用しました"
    ])
    ja_med_negative = any(kw in text for kw in [
        "薬を忘れた", "薬忘れた", "飲み忘れた", "飲んでない", "飲まなかった"
    ])
    if ja_med_positive:
        signals["medication_taken"] = True
    elif ja_med_negative:
        signals["medication_taken"] = False

    # Japanese pain patterns: "痛い", "頭痛", "腰痛", "体が痛い"
    ja_pain_keywords = ["痛い", "痛み", "頭痛", "腰痛", "背中痛", "関節痛", 
                        "筋肉痛", "腹痛", "胸痛", "吐き気", "めまい"]
    if any(kw in text for kw in ja_pain_keywords):
        signals["pain_mentioned"] = True

    # Qualitative Japanese mood inference (if not already set)
    if "mood_score" not in signals:
        ja_positive = ["元気", "嬉しい", "楽しい", "幸せ", "最高", "良い", "いい感じ"]
        ja_negative = ["落ち込んでいる", "悲しい", "辛い", "苦しい", "憂鬱", "気分が悪い", "調子悪い"]
        if any(w in text for w in ja_positive):
            signals["mood_score"] = 7.5
        elif any(w in text for w in ja_negative):
            signals["mood_score"] = 3.0

    # ─── CHECK IF TIER 1 FOUND ANY SIGNALS ──────────────────────────────────
    
    # Check if regex extraction found any meaningful signals
    has_signals = any(
        signals.get(key) is not None 
        for key in ["sleep_hours", "mood_score", "energy_level", "medication_taken", "pain_mentioned"]
    )
    
    if has_signals:
        # Validate extracted signals
        validated_signals = {}
        for key, value in signals.items():
            if key in HEALTH_SIGNAL_RANGES and value is not None:
                if validate_vital(key, value):
                    validated_signals[key] = value
                else:
                    logger.warning(f"Tier 1 extracted invalid {key}: {value}, discarding")
            else:
                # Non-validatable fields (medication_taken, pain_mentioned)
                validated_signals[key] = value
        
        # Calculate confidence score for regex extraction
        confidence = calculate_extraction_confidence(validated_signals, extraction_method="regex")
        validated_signals['confidence'] = confidence
        
        logger.info(f"Tier 1 (regex) extraction successful: {len(validated_signals)} signals found, confidence={confidence}")
        return validated_signals
    
    # ─── TIER 2: LLM-BASED EXTRACTION (FALLBACK) ────────────────────────────
    
    logger.info("Tier 1 (regex) found no signals, falling back to Tier 2 (LLM)")
    
    try:
        llm_signals = extract_health_signals_llm(text, language=language, timeout=30)
        
        # Check if LLM extraction was successful (confidence > 0)
        if llm_signals.get("confidence", 0.0) > 0.0:
            logger.info(f"Tier 2 (LLM) extraction successful: confidence={llm_signals.get('confidence')}")
            
            # Convert LLM signals to match regex format where needed
            # LLM uses pain_location/pain_intensity, regex uses pain_mentioned
            if llm_signals.get("pain_location") or llm_signals.get("pain_intensity"):
                llm_signals["pain_mentioned"] = True
            
            return llm_signals
        else:
            logger.warning("Tier 2 (LLM) extraction failed or returned low confidence")
            return {}
    
    except Exception as e:
        logger.error(f"Tier 2 (LLM) extraction error: {e}")
        return {}


def extract_health_signals_llm(
    text: str,
    language: Optional[str] = None,
    timeout: int = 30
) -> Dict[str, Any]:
    """
    Extract health signals using LLM-based structured extraction.
    
    This is Tier 2 extraction - use when regex-based extraction (Tier 1)
    returns no signals. Uses few-shot learning to guide the LLM to extract
    health signals in a consistent JSON format.
    
    Args:
        text: User text to extract health signals from
        language: Detected language code (e.g., 'en', 'ja')
        timeout: Timeout in seconds for LLM request
    
    Returns:
        Dict with extracted health signals:
            - sleep_hours: float (0-24) or None
            - mood_score: float (1-10) or None
            - energy_level: float (1-10) or None
            - medication_taken: bool or None
            - pain_location: str or None
            - pain_intensity: float (1-10) or None
            - confidence: float (0-1)
    """
    logger.info("Using LLM-based health signal extraction")
    
    # Build extraction prompt with few-shot examples
    prompt = build_extraction_prompt(text, language)
    
    try:
        # Call Ollama with extraction prompt
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.3,  # Lower temperature for more consistent extraction
                    "top_p": 0.9,
                }
            },
            timeout=timeout,
        )
        response.raise_for_status()
        
        # Extract JSON from response
        llm_output = response.json().get("response", "").strip()
        
        # Try to parse JSON from response
        # LLM might wrap JSON in markdown code blocks
        if "```json" in llm_output:
            # Extract JSON from markdown code block
            json_start = llm_output.find("```json") + 7
            json_end = llm_output.find("```", json_start)
            json_str = llm_output[json_start:json_end].strip()
        elif "```" in llm_output:
            # Extract from generic code block
            json_start = llm_output.find("```") + 3
            json_end = llm_output.find("```", json_start)
            json_str = llm_output[json_start:json_end].strip()
        else:
            # Assume entire response is JSON
            json_str = llm_output
        
        # Parse JSON
        signals = json.loads(json_str)
        
        # Validate extracted signals
        validated_signals = validate_extracted_signals(signals)
        
        logger.info(f"LLM extraction successful: {validated_signals}")
        return validated_signals
        
    except requests.RequestException as e:
        logger.error(f"LLM extraction failed due to request error: {e}")
        return {"confidence": 0.0}
    
    except json.JSONDecodeError as e:
        logger.error(f"LLM extraction failed due to JSON parse error: {e}")
        logger.debug(f"LLM output was: {llm_output}")
        return {"confidence": 0.0}
    
    except Exception as e:
        logger.error(f"LLM extraction failed: {e}")
        return {"confidence": 0.0}


def validate_extracted_signals(signals: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate and sanitize extracted health signals.
    
    Ensures all values are within valid ranges and converts to appropriate types.
    Uses the validate_vital() function for plausibility checks.
    
    Args:
        signals: Raw extracted signals from LLM
    
    Returns:
        Validated signals dict with sanitized values
    """
    validated = {}
    
    # Sleep hours: 0-24
    if "sleep_hours" in signals and signals["sleep_hours"] is not None:
        try:
            sleep = float(signals["sleep_hours"])
            if validate_vital('sleep_hours', sleep):
                validated["sleep_hours"] = sleep
            else:
                validated["sleep_hours"] = None
        except (ValueError, TypeError):
            validated["sleep_hours"] = None
    else:
        validated["sleep_hours"] = None
    
    # Mood score: 1-10
    if "mood_score" in signals and signals["mood_score"] is not None:
        try:
            mood = float(signals["mood_score"])
            if validate_vital('mood_score', mood):
                validated["mood_score"] = mood
            else:
                validated["mood_score"] = None
        except (ValueError, TypeError):
            validated["mood_score"] = None
    else:
        validated["mood_score"] = None
    
    # Energy level: 1-10
    if "energy_level" in signals and signals["energy_level"] is not None:
        try:
            energy = float(signals["energy_level"])
            if validate_vital('energy_level', energy):
                validated["energy_level"] = energy
            else:
                validated["energy_level"] = None
        except (ValueError, TypeError):
            validated["energy_level"] = None
    else:
        validated["energy_level"] = None
    
    # Medication taken: boolean
    if "medication_taken" in signals and signals["medication_taken"] is not None:
        validated["medication_taken"] = bool(signals["medication_taken"])
    else:
        validated["medication_taken"] = None
    
    # Pain location: string
    if "pain_location" in signals and signals["pain_location"]:
        validated["pain_location"] = str(signals["pain_location"]).lower()
    else:
        validated["pain_location"] = None
    
    # Pain intensity: 1-10
    if "pain_intensity" in signals and signals["pain_intensity"] is not None:
        try:
            pain = float(signals["pain_intensity"])
            if validate_vital('pain_intensity', pain):
                validated["pain_intensity"] = pain
            else:
                validated["pain_intensity"] = None
        except (ValueError, TypeError):
            validated["pain_intensity"] = None
    else:
        validated["pain_intensity"] = None
    
    # Confidence: 0-1
    if "confidence" in signals:
        try:
            confidence = float(signals["confidence"])
            validated["confidence"] = max(0.0, min(1.0, confidence))
        except (ValueError, TypeError):
            validated["confidence"] = 0.5  # Default medium confidence
    else:
        validated["confidence"] = 0.5
    
    # Recalculate confidence based on validation results
    # This ensures confidence reflects actual data quality
    validated["confidence"] = calculate_extraction_confidence(validated, extraction_method="llm")
    
    return validated


# ─── Clarification Questions for Low Confidence ─────────────────────────────

# Clarification question templates by signal type and language
CLARIFICATION_TEMPLATES = {
    "en": {
        "sleep_hours": [
            "I want to make sure I understand correctly - how many hours did you sleep last night?",
            "Could you tell me more specifically how long you slept?",
            "To track your sleep accurately, can you share how many hours you got?",
        ],
        "mood_score": [
            "I'd like to understand your mood better - on a scale of 1 to 10, how would you rate how you're feeling?",
            "Could you help me understand your mood more clearly? What number from 1 to 10 describes it best?",
            "To support you better, can you rate your current mood from 1 (very low) to 10 (excellent)?",
        ],
        "energy_level": [
            "I want to check in on your energy - on a scale of 1 to 10, how energetic do you feel?",
            "Could you describe your energy level more specifically? What number from 1 to 10 fits best?",
            "To understand how you're doing, can you rate your energy from 1 (exhausted) to 10 (very energetic)?",
        ],
        "medication_taken": [
            "Just to confirm - did you take your medication today?",
            "I want to make sure I have this right - have you taken your medication?",
            "Could you clarify whether you've taken your medication today?",
        ],
        "pain_location": [
            "I heard you mention pain - where exactly are you feeling it?",
            "Could you tell me more about where the pain is located?",
            "To help you better, can you describe where you're experiencing pain?",
        ],
        "pain_intensity": [
            "I want to understand your pain better - on a scale of 1 to 10, how intense is it?",
            "Could you rate your pain level from 1 (mild) to 10 (severe)?",
            "To track this properly, can you tell me how strong the pain is on a scale of 1 to 10?",
        ],
        "general": [
            "I want to make sure I understand your health status correctly. Could you share more details?",
            "I'd like to get a clearer picture of how you're doing. Can you tell me more?",
            "To support you better, could you provide more specific information about how you're feeling?",
        ],
    },
    "ja": {
        "sleep_hours": [
            "正確に把握したいのですが、昨夜は何時間眠りましたか？",
            "もう少し具体的に、どのくらい眠ったか教えていただけますか？",
            "睡眠を正確に記録するため、何時間寝たか教えてください。",
        ],
        "mood_score": [
            "気分をもっとよく理解したいのですが、1から10のスケールで、今の気分を評価していただけますか？",
            "気分をもっと明確に理解したいです。1から10で、どの数字が一番合っていますか？",
            "より良いサポートのため、現在の気分を1（とても低い）から10（素晴らしい）で評価してください。",
        ],
        "energy_level": [
            "エネルギーレベルを確認したいのですが、1から10のスケールで、どのくらい元気ですか？",
            "もう少し具体的に、エネルギーレベルを教えていただけますか？1から10でどれが合いますか？",
            "状態を理解するため、エネルギーを1（疲れ切っている）から10（とても元気）で評価してください。",
        ],
        "medication_taken": [
            "確認させてください - 今日、薬を飲みましたか？",
            "正確に把握したいのですが、薬を服用しましたか？",
            "今日、薬を飲んだかどうか教えていただけますか？",
        ],
        "pain_location": [
            "痛みがあるとおっしゃいましたが、具体的にどこが痛いですか？",
            "痛みの場所について、もう少し詳しく教えていただけますか？",
            "より良いサポートのため、どこに痛みを感じているか教えてください。",
        ],
        "pain_intensity": [
            "痛みをもっとよく理解したいのですが、1から10のスケールで、どのくらい強いですか？",
            "痛みのレベルを1（軽い）から10（激しい）で評価していただけますか？",
            "正確に記録するため、痛みの強さを1から10で教えてください。",
        ],
        "general": [
            "健康状態を正確に理解したいです。もう少し詳しく教えていただけますか？",
            "状態をもっと明確に把握したいです。詳しく教えてください。",
            "より良いサポートのため、具体的な情報を教えていただけますか？",
        ],
    },
}


def needs_clarification(signals: Dict[str, Any], confidence_threshold: float = 0.6) -> bool:
    """
    Determine if extracted health signals need clarification.
    
    Returns True if:
    - Confidence score is below threshold (< 0.6)
    - OR no signals were extracted at all
    
    Args:
        signals: Extracted health signals dict
        confidence_threshold: Minimum confidence threshold (default 0.6)
    
    Returns:
        True if clarification is needed, False otherwise
    """
    # No signals extracted
    if not signals or all(v is None for k, v in signals.items() if k != 'confidence'):
        return True
    
    # Low confidence
    confidence = signals.get('confidence', 0.0)
    if confidence < confidence_threshold:
        return True
    
    return False


def generate_clarification_question(
    signals: Dict[str, Any],
    language: str = "en"
) -> str:
    """
    Generate a clarifying question based on extracted signals and confidence.
    
    Selects the most appropriate question based on:
    - Which signals were partially extracted
    - Which signals are missing
    - The language of the conversation
    
    Args:
        signals: Extracted health signals dict (may be incomplete or low confidence)
        language: Language code ('en' or 'ja')
    
    Returns:
        Clarifying question string in the appropriate language
    """
    import random
    
    # Normalize language code
    lang = "ja" if language == "ja" else "en"
    templates = CLARIFICATION_TEMPLATES.get(lang, CLARIFICATION_TEMPLATES["en"])
    
    # Determine which signal to ask about
    # Priority: ask about signals that were mentioned but unclear
    
    # Check for partial/unclear signals (non-None but low confidence)
    unclear_signals = []
    for signal_type in ["sleep_hours", "mood_score", "energy_level", "medication_taken", 
                        "pain_location", "pain_intensity"]:
        if signal_type in signals and signals[signal_type] is not None:
            unclear_signals.append(signal_type)
    
    # If we have unclear signals, ask about the first one
    if unclear_signals:
        signal_type = unclear_signals[0]
        if signal_type in templates:
            return random.choice(templates[signal_type])
    
    # Otherwise, ask a general clarification question
    return random.choice(templates["general"])


def parse_clarification_response(
    response_text: str,
    original_signals: Dict[str, Any],
    language: str = "en"
) -> Dict[str, Any]:
    """
    Parse a clarification response and extract additional health signals.
    
    This function re-runs health signal extraction on the clarification response
    and merges it with the original signals, giving priority to the new extraction.
    
    Args:
        response_text: User's response to the clarification question
        original_signals: Original extracted signals (may be incomplete)
        language: Language code ('en' or 'ja')
    
    Returns:
        Updated signals dict with clarified information
    """
    # Extract signals from clarification response
    new_signals = extract_health_signals(response_text, language=language)
    
    # Merge with original signals, prioritizing new extraction
    merged_signals = original_signals.copy()
    
    # Update with new signals (overwrite if present)
    for key, value in new_signals.items():
        if value is not None:
            merged_signals[key] = value
    
    # Update confidence to reflect clarification
    # If new extraction has higher confidence, use it
    new_confidence = new_signals.get('confidence', 0.0)
    old_confidence = original_signals.get('confidence', 0.0)
    merged_signals['confidence'] = max(new_confidence, old_confidence)
    
    # If we got new signals, boost confidence slightly
    new_signal_count = sum(1 for k, v in new_signals.items() 
                          if k != 'confidence' and v is not None)
    if new_signal_count > 0:
        merged_signals['confidence'] = min(merged_signals['confidence'] + 0.1, 1.0)
    
    return merged_signals


# ─── Async Wrappers ──────────────────────────────────────────────────────────

async def get_response_async(
    user_input: str,
    emotion_label: Optional[str] = None,
    tone_mode: Optional[str] = None,
    health_stats: Optional[Dict[str, Any]] = None,
    active_alerts: Optional[List[Dict]] = None,
    conversation_history: Optional[List[Dict]] = None,
    language: Optional[str] = None,
) -> str:
    """
    Async wrapper for get_response using asyncio.to_thread.
    
    Get a health-aware, emotion-adaptive response from the local LLM.
    
    Args:
        user_input: Transcribed user speech.
        emotion_label: Detected emotion from voice analysis.
        tone_mode: Response tone mode (calm/encouraging/gentle_support/neutral).
        health_stats: Aggregated health stats from the database.
        active_alerts: Unacknowledged proactive alerts.
        conversation_history: Recent conversation turns.
        language: Detected language code (e.g. 'en', 'ja').
    
    Returns:
        LLM response string.
    """
    return await asyncio.to_thread(
        get_response,
        user_input,
        emotion_label,
        tone_mode,
        health_stats,
        active_alerts,
        conversation_history,
        language,
    )


async def extract_health_signals_async(text: str, language: Optional[str] = None) -> Dict[str, Any]:
    """
    Async wrapper for extract_health_signals using asyncio.to_thread.
    
    Implements two-tier extraction with graceful degradation:
    - Tier 1: Regex-based extraction (fast path)
    - Tier 2: LLM-based extraction (fallback when regex finds nothing)
    - Retry logic (2 attempts) for transient failures
    - Timeout handling (5s per attempt)
    - Fallback to empty signals on failure
    
    Args:
        text: User text to extract health signals from
        language: Detected language code (e.g., 'en', 'ja') - used for LLM fallback
    
    Returns:
        Dict with any detected values:
            - mood_score: float (1-10) if mentioned
            - sleep_hours: float if mentioned
            - energy_level: float (1-10) if mentioned
            - medication_taken: bool if mentioned
            - pain_mentioned: bool (regex tier)
            - pain_location: str (LLM tier)
            - pain_intensity: float (1-10, LLM tier)
            - confidence: float (0-1, LLM tier only)
    """
    from core.error_handling import with_retry_and_timeout, FallbackStrategies
    
    try:
        return await with_retry_and_timeout(
            extract_health_signals,
            text,
            language,
            max_retries=2,
            timeout=5.0,
            initial_delay=0.5
        )
    except Exception as e:
        logger.error(f"Health signal extraction failed after retries: {e}")
        return await FallbackStrategies.health_extraction_fallback(text)
