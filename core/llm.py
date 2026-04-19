"""
Aegis LLM Module
Health-aware, emotion-adaptive local inference via Ollama.
Fully offline — connects to a local Ollama instance.
"""

import asyncio
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


def extract_health_signals(text: str) -> Dict[str, Any]:
    """
    Extract structured health signals from user text using keyword parsing.
    This is a lightweight local alternative to full NLU.
    
    Supports both English and Japanese text with language-specific patterns.
    
    Returns dict with any detected values:
        - mood_score: float (1-10) if mentioned
        - sleep_hours: float if mentioned
        - energy_level: float (1-10) if mentioned
        - medication_taken: bool if mentioned
        - pain_mentioned: bool
    """
    import re
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
                     "happy", "cheerful", "excellent"]
        negative = ["terrible", "awful", "bad", "horrible", "depressed",
                     "miserable", "sad", "down", "low"]
        if any(w in text_lower for w in positive):
            signals["mood_score"] = 7.5
        elif any(w in text_lower for w in negative):
            signals["mood_score"] = 3.0

    # ─── Japanese Patterns ───────────────────────────────────────────────────

    # Japanese sleep patterns: "6時間寝た", "5時間睡眠", "よく眠れた", "あまり寝られなかった"
    ja_sleep_match = re.search(r'(\d+(?:\.\d+)?)\s*時間(?:寝|睡眠)', text)
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

    return signals


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


async def extract_health_signals_async(text: str) -> Dict[str, Any]:
    """
    Async wrapper for extract_health_signals using asyncio.to_thread.
    
    Implements graceful degradation:
    - Retry logic (2 attempts) for transient failures
    - Timeout handling (5s per attempt)
    - Fallback to empty signals on failure
    
    Extract structured health signals from user text using keyword parsing.
    
    Returns dict with any detected values:
        - mood_score: float (1-10) if mentioned
        - sleep_hours: float if mentioned
        - energy_level: float (1-10) if mentioned
        - medication_taken: bool if mentioned
        - pain_mentioned: bool
    """
    from core.error_handling import with_retry_and_timeout, FallbackStrategies
    
    try:
        return await with_retry_and_timeout(
            extract_health_signals,
            text,
            max_retries=2,
            timeout=5.0,
            initial_delay=0.5
        )
    except Exception as e:
        logger.error(f"Health signal extraction failed after retries: {e}")
        return await FallbackStrategies.health_extraction_fallback(text)
