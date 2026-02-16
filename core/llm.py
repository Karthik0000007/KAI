"""
Aegis LLM Module
Health-aware, emotion-adaptive local inference via Ollama.
Fully offline — connects to a local Ollama instance.
"""

import logging
from typing import Optional, List, Dict, Any

import requests

from core.config import (
    OLLAMA_URL, OLLAMA_MODEL, AEGIS_SYSTEM_PROMPT,
    TONE_MODES, LLM_CONTEXT_WINDOW,
)

logger = logging.getLogger("aegis.llm")


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
            timeout=120,
        )
        response.raise_for_status()
        reply = response.json().get("response", "").strip()

        if not reply:
            logger.warning("Empty response from LLM")
            return "I'm here for you. Could you say that again?"

        return reply

    except requests.ConnectionError:
        logger.error("Cannot connect to Ollama. Is it running?")
        return (
            "I'm having trouble connecting to my language model. "
            "Please make sure Ollama is running locally."
        )
    except requests.Timeout:
        logger.error("Ollama request timed out")
        return "I need a moment to think. Could you try again?"
    except Exception as e:
        logger.error(f"LLM error: {e}")
        return "Something went wrong on my end. Let me try again."


def extract_health_signals(text: str) -> Dict[str, Any]:
    """
    Extract structured health signals from user text using keyword parsing.
    This is a lightweight local alternative to full NLU.
    
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

    # Qualitative mood inference
    if "mood_score" not in signals:
        positive = ["great", "wonderful", "amazing", "fantastic", "good",
                     "happy", "cheerful", "excellent"]
        negative = ["terrible", "awful", "bad", "horrible", "depressed",
                     "miserable", "sad", "down", "low"]
        if any(w in text_lower for w in positive):
            signals["mood_score"] = 7.5
        elif any(w in text_lower for w in negative):
            signals["mood_score"] = 3.0

    return signals
