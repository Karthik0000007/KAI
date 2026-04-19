"""
Demonstration of ConfigManager usage in the Aegis application.

This shows how the ConfigManager would be used in practice.
"""

from core.config import ConfigManager


def main():
    """Demonstrate ConfigManager usage."""
    print("=" * 80)
    print("ConfigManager Usage Demonstration")
    print("=" * 80)
    
    # Initialize ConfigManager (loads and validates config.yaml)
    print("\n1. Initializing ConfigManager...")
    config = ConfigManager()
    print("   ✓ Configuration loaded and validated")
    
    # Example 1: Audio configuration for recording
    print("\n2. Audio Configuration:")
    print(f"   Sample Rate: {config.get('audio.sample_rate')} Hz")
    print(f"   Record Duration: {config.get('audio.record_duration')} seconds")
    print(f"   VAD Enabled: {config.get('audio.vad_enabled')}")
    
    # Example 2: Model configuration for AI components
    print("\n3. Model Configuration:")
    print(f"   Whisper Model: {config.get('models.whisper_size')}")
    print(f"   Ollama Model: {config.get('models.ollama_model')}")
    print(f"   Ollama URL: {config.get('models.ollama_url')}")
    print(f"   Ollama Timeout: {config.get('models.ollama_timeout')} seconds")
    
    # Example 3: LLM configuration for response generation
    print("\n4. LLM Configuration:")
    print(f"   Temperature: {config.get('llm.temperature')}")
    print(f"   Max Tokens: {config.get('llm.max_response_tokens')}")
    print(f"   Context Window: {config.get('llm.context_window')}")
    
    # Example 4: Emotion detection thresholds
    print("\n5. Emotion Detection Configuration:")
    print(f"   Pitch Range: {config.get('emotion.pitch_low')} - {config.get('emotion.pitch_high')} Hz")
    print(f"   Energy Range: {config.get('emotion.energy_low')} - {config.get('emotion.energy_high')}")
    print(f"   Confidence Threshold: {config.get('emotion.confidence_threshold')}")
    
    # Example 5: Proactive monitoring settings
    print("\n6. Proactive Monitoring Configuration:")
    print(f"   Check Interval: {config.get('proactive.check_interval_minutes')} minutes")
    print(f"   Low Mood Threshold: {config.get('proactive.low_mood_days_threshold')} days")
    print(f"   Low Sleep Hours: {config.get('proactive.low_sleep_hours')} hours")
    
    # Example 6: Privacy and security settings
    print("\n7. Privacy Configuration:")
    print(f"   Encryption: {config.get('privacy.encryption_algorithm')}")
    print(f"   Use Keyring: {config.get('privacy.use_keyring')}")
    print(f"   Differential Privacy: {config.get('privacy.enable_differential_privacy')}")
    print(f"   Privacy Epsilon: {config.get('privacy.differential_privacy_epsilon')}")
    
    # Example 7: Using default values for optional settings
    print("\n8. Using Default Values:")
    vision_enabled = config.get('vision.enabled', False)
    wearables_enabled = config.get('wearables.enabled', False)
    dashboard_enabled = config.get('dashboard.enabled', False)
    print(f"   Vision Module: {vision_enabled}")
    print(f"   Wearables: {wearables_enabled}")
    print(f"   Dashboard: {dashboard_enabled}")
    
    print("\n" + "=" * 80)
    print("✓ ConfigManager demonstration complete")
    print("=" * 80)
    
    # Show how to access the full config dictionary if needed
    print("\n9. Direct Access to Config Dictionary:")
    print(f"   Available sections: {list(config.config.keys())}")
    
    print("\n" + "=" * 80)
    print("ConfigManager Features:")
    print("=" * 80)
    print("✓ Loads configuration from YAML file (Requirement 4.1)")
    print("✓ Creates default config if missing (Requirement 4.2)")
    print("✓ Validates all configuration values (Requirement 4.6)")
    print("✓ Type checking for all values")
    print("✓ Range validation for numeric values")
    print("✓ Enum validation for string values")
    print("✓ Clear error messages on validation failure (Requirement 4.4)")
    print("✓ Dot-notation access to nested values")
    print("✓ Default value support for optional settings")
    print("=" * 80)


if __name__ == "__main__":
    main()
