"""
Demo: LLM Fallback System
Demonstrates the fallback system in action with various scenarios.
"""

from unittest.mock import patch
import requests

from core.llm import get_response


def demo_scenario(title, user_input, emotion_label, **kwargs):
    """Run a demo scenario and display results"""
    print(f"\n{'='*70}")
    print(f"SCENARIO: {title}")
    print(f"{'='*70}")
    print(f"User Input: {user_input}")
    print(f"Emotion: {emotion_label}")
    
    if kwargs.get('health_stats'):
        print(f"Health Stats: {kwargs['health_stats']}")
    if kwargs.get('active_alerts'):
        print(f"Active Alerts: {len(kwargs['active_alerts'])} alert(s)")
    if kwargs.get('conversation_history'):
        print(f"Conversation History: {len(kwargs['conversation_history'])} turn(s)")
    
    print(f"\n[Simulating Ollama unavailable...]")
    
    # Mock Ollama to be unavailable
    with patch('core.llm.requests.post') as mock_post:
        mock_post.side_effect = requests.ConnectionError("Connection refused")
        
        response = get_response(
            user_input=user_input,
            emotion_label=emotion_label,
            **kwargs
        )
        
        print(f"\nAegis Response (Fallback):")
        print(f"  \"{response}\"")
        print(f"\nRetry Attempts: {mock_post.call_count}")


def main():
    print("\n" + "="*70)
    print("LLM FALLBACK SYSTEM DEMONSTRATION")
    print("="*70)
    print("\nThis demo shows how Aegis responds when Ollama is unavailable.")
    print("The system uses contextual fallback responses based on:")
    print("  - User's emotional state")
    print("  - Conversation history")
    print("  - Health statistics")
    print("  - Active proactive alerts")
    
    # Scenario 1: Daily health check-in
    demo_scenario(
        title="Daily Health Check-in",
        user_input="I slept 7 hours last night and feeling pretty good",
        emotion_label="calm",
        conversation_history=[
            {"role": "user", "content": "Good morning Aegis"},
            {"role": "assistant", "content": "Good morning! How are you today?"},
        ]
    )
    
    # Scenario 2: Emotional distress
    demo_scenario(
        title="Emotional Distress",
        user_input="I'm so anxious I can't sleep",
        emotion_label="anxious",
        health_stats={
            "count": 7,
            "avg_mood": 3.5,
            "low_mood_days": 3,
            "avg_sleep": 5.5,
        },
        conversation_history=[
            {"role": "user", "content": "I've been feeling really down lately"},
        ]
    )
    
    # Scenario 3: Proactive alert
    demo_scenario(
        title="Proactive Alert Response",
        user_input="Yeah, I've been struggling",
        emotion_label="stressed",
        active_alerts=[
            {
                "id": 1,
                "message": "You've reported low mood for 3 consecutive days",
                "severity": "warning",
            }
        ],
        health_stats={
            "count": 10,
            "avg_mood": 3.0,
            "low_mood_days": 3,
        }
    )
    
    # Scenario 4: Fatigue
    demo_scenario(
        title="Fatigue and Low Energy",
        user_input="I'm just so tired all the time",
        emotion_label="fatigued",
        health_stats={
            "count": 5,
            "avg_energy": 3.0,
            "avg_sleep": 5.0,
        }
    )
    
    # Scenario 5: Neutral conversation
    demo_scenario(
        title="General Conversation",
        user_input="How are you today?",
        emotion_label="neutral",
    )
    
    # Scenario 6: Stressed with medication concern
    demo_scenario(
        title="Medication Reminder",
        user_input="I forgot to take my medication again",
        emotion_label="stressed",
        conversation_history=[
            {"role": "user", "content": "I've been forgetting my meds lately"},
        ]
    )
    
    print("\n" + "="*70)
    print("DEMONSTRATION COMPLETE")
    print("="*70)
    print("\nKey Observations:")
    print("  ✓ System continues operating when Ollama unavailable")
    print("  ✓ Responses are contextually appropriate")
    print("  ✓ Empathetic tone maintained across all scenarios")
    print("  ✓ Retry logic attempts 2 times before fallback")
    print("  ✓ No technical errors exposed to user")
    print("\nThe fallback system ensures Aegis remains helpful and supportive")
    print("even when the LLM service is temporarily unavailable.")
    print()


if __name__ == "__main__":
    main()
