"""
Property-Based Test for Configuration Round-Trip - Task 3.4

Property 1: Configuration Round-Trip Preservation
Validates: Requirements 21.1, 21.4, 21.5

For any valid Configuration object, serializing to YAML then parsing back
SHALL produce an equivalent Configuration object with all fields preserved.

This test uses Hypothesis to generate random valid configurations and verifies
that the round-trip through YAML serialization preserves all data.
"""

import sys
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
import yaml

from hypothesis import given, strategies as st, settings, assume, HealthCheck
from hypothesis.strategies import SearchStrategy

from core.config import ConfigManager


# ─── Hypothesis Strategies for Configuration Generation ─────────────────────


def valid_audio_config() -> SearchStrategy[Dict[str, Any]]:
    """Generate valid audio configuration."""
    return st.fixed_dictionaries({
        'sample_rate': st.integers(min_value=8000, max_value=48000),
        'record_duration': st.integers(min_value=1, max_value=30),
        'silence_threshold': st.floats(min_value=0.0001, max_value=0.1),
        'vad_enabled': st.booleans(),
        'vad_silence_duration': st.floats(min_value=0.5, max_value=5.0),
        'input_device': st.none() | st.integers(min_value=0, max_value=10),
        'output_device': st.none() | st.integers(min_value=0, max_value=10),
    })


def valid_models_config() -> SearchStrategy[Dict[str, Any]]:
    """Generate valid models configuration."""
    return st.fixed_dictionaries({
        'whisper_size': st.sampled_from(['tiny', 'base', 'small', 'medium', 'large']),
        'ollama_model': st.text(min_size=1, max_size=50, alphabet=st.characters(
            whitelist_categories=('Lu', 'Ll', 'Nd'), whitelist_characters=':.-_'
        )),
        'ollama_url': st.just('http://localhost:11434'),
        'ollama_timeout': st.integers(min_value=30, max_value=300),
        'coqui_model': st.text(min_size=1, max_size=100, alphabet=st.characters(
            min_codepoint=32, max_codepoint=126  # Printable ASCII only
        )),
        'voicevox_url': st.just('http://127.0.0.1:50021'),
        'voicevox_speaker_id': st.integers(min_value=0, max_value=50),
        'voicevox_auto_start': st.booleans(),
        'voicevox_path': st.none() | st.text(min_size=1, max_size=100, alphabet=st.characters(
            min_codepoint=32, max_codepoint=126  # Printable ASCII only
        )),
    })


def valid_llm_config() -> SearchStrategy[Dict[str, Any]]:
    """Generate valid LLM configuration."""
    return st.fixed_dictionaries({
        'context_window': st.integers(min_value=512, max_value=8192),
        'max_response_tokens': st.integers(min_value=50, max_value=500),
        'temperature': st.floats(min_value=0.0, max_value=2.0),
        'top_p': st.floats(min_value=0.0, max_value=1.0),
        'llm_extraction_enabled': st.booleans(),
        'extraction_confidence_threshold': st.floats(min_value=0.0, max_value=1.0),
        'context_aware': st.booleans(),
        'context_turns': st.integers(min_value=0, max_value=10),
    })


def valid_tts_config() -> SearchStrategy[Dict[str, Any]]:
    """Generate valid TTS configuration."""
    return st.fixed_dictionaries({
        'speech_rate': st.floats(min_value=0.5, max_value=2.0),
        'volume': st.floats(min_value=0.0, max_value=1.0),
        'adaptive_rate': st.booleans(),
        'fallback_engine': st.sampled_from(['pyttsx3', 'none']),
    })


def valid_emotion_config() -> SearchStrategy[Dict[str, Any]]:
    """Generate valid emotion configuration with proper constraints."""
    # Generate pitch_low and pitch_high such that pitch_high > pitch_low
    pitch_low = st.floats(min_value=50.0, max_value=200.0)
    pitch_high = st.floats(min_value=150.0, max_value=400.0)
    
    # Generate energy_low and energy_high such that energy_high > energy_low
    energy_low = st.floats(min_value=0.001, max_value=0.1)
    energy_high = st.floats(min_value=0.01, max_value=0.2)
    
    # Generate rate_slow and rate_fast
    rate_slow = st.floats(min_value=1.0, max_value=3.0)
    rate_fast = st.floats(min_value=3.5, max_value=6.0)
    
    return st.builds(
        lambda pl, ph, el, eh, rs, rf, ct, ce, me, vc, aw: {
            'pitch_low': pl,
            'pitch_high': max(ph, pl + 10.0),  # Ensure pitch_high > pitch_low
            'energy_low': el,
            'energy_high': max(eh, el + 0.001),  # Ensure energy_high > energy_low
            'rate_slow': rs,
            'rate_fast': rf,
            'confidence_threshold': ct,
            'calibration_enabled': ce,
            'mixed_emotions_enabled': me,
            'visual_correlation_enabled': vc,
            'audio_weight': aw,
        },
        pitch_low,
        pitch_high,
        energy_low,
        energy_high,
        rate_slow,
        rate_fast,
        st.floats(min_value=0.0, max_value=1.0),
        st.booleans(),
        st.booleans(),
        st.booleans(),
        st.floats(min_value=0.0, max_value=1.0),
    )


def valid_proactive_config() -> SearchStrategy[Dict[str, Any]]:
    """Generate valid proactive configuration."""
    return st.fixed_dictionaries({
        'enabled': st.booleans(),
        'check_interval_minutes': st.integers(min_value=15, max_value=1440),
        'low_mood_days_threshold': st.integers(min_value=1, max_value=14),
        'mood_low_threshold': st.floats(min_value=1.0, max_value=10.0),
        'low_sleep_hours': st.floats(min_value=0.0, max_value=12.0),
        'sleep_deficit_days': st.integers(min_value=1, max_value=14),
        'elevated_hr_threshold': st.integers(min_value=80, max_value=180),
        'elevated_hr_duration': st.integers(min_value=5, max_value=60),
        'missed_medication_reminder_delay': st.integers(min_value=15, max_value=180),
        'max_alerts_per_day': st.integers(min_value=1, max_value=10),
        'deduplication_window_hours': st.integers(min_value=1, max_value=72),
        'prioritize_alerts': st.booleans(),
        'include_explanations': st.booleans(),
    })


def valid_privacy_config() -> SearchStrategy[Dict[str, Any]]:
    """Generate valid privacy configuration."""
    return st.fixed_dictionaries({
        'encryption_algorithm': st.sampled_from(['AES-256-GCM', 'AES-128-GCM']),
        'use_keyring': st.booleans(),
        'require_passphrase': st.booleans(),
        'pbkdf2_iterations': st.integers(min_value=100000, max_value=1000000),
        'enable_differential_privacy': st.booleans(),
        'differential_privacy_epsilon': st.floats(min_value=0.1, max_value=10.0),
        'auto_lock_minutes': st.integers(min_value=0, max_value=1440),
        'audit_logging_enabled': st.booleans(),
        'secure_delete': st.booleans(),
        'retention_days': st.integers(min_value=0, max_value=3650),
    })


def valid_logging_config() -> SearchStrategy[Dict[str, Any]]:
    """Generate valid logging configuration."""
    return st.fixed_dictionaries({
        'level': st.sampled_from(['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']),
        'file': st.just('data/logs/aegis.log'),
        'max_size_mb': st.integers(min_value=1, max_value=100),
        'backup_count': st.integers(min_value=1, max_value=20),
        'format': st.sampled_from(['json', 'text']),
        'console_enabled': st.booleans(),
        'metrics_enabled': st.booleans(),
        'debug_mode': st.booleans(),
        'debug_dir': st.just('data/debug'),
    })


def valid_backup_config() -> SearchStrategy[Dict[str, Any]]:
    """Generate valid backup configuration."""
    return st.fixed_dictionaries({
        'enabled': st.booleans(),
        'interval_hours': st.integers(min_value=1, max_value=168),
        'retention_days': st.integers(min_value=7, max_value=365),
        'backup_dir': st.just('data/db/backups'),
        'verify_integrity': st.booleans(),
        'compress': st.booleans(),
        'backup_on_shutdown': st.booleans(),
    })


def valid_configuration() -> SearchStrategy[Dict[str, Any]]:
    """
    Generate a complete valid configuration dictionary.
    
    This strategy generates random but valid configurations that should pass
    all validation checks in ConfigManager.
    """
    return st.fixed_dictionaries({
        'audio': valid_audio_config(),
        'models': valid_models_config(),
        'llm': valid_llm_config(),
        'tts': valid_tts_config(),
        'emotion': valid_emotion_config(),
        'proactive': valid_proactive_config(),
        'privacy': valid_privacy_config(),
        'logging': valid_logging_config(),
        'backup': valid_backup_config(),
    })


# ─── Helper Functions ────────────────────────────────────────────────────────


def write_config_to_yaml(config: Dict[str, Any], path: Path) -> None:
    """
    Write configuration dictionary to YAML file.
    
    This implements the "Configuration_Pretty_Printer" functionality
    mentioned in Requirement 21.4.
    """
    with open(path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)


def configs_are_equivalent(config1: Dict[str, Any], config2: Dict[str, Any]) -> bool:
    """
    Check if two configuration dictionaries are equivalent.
    
    Handles floating point comparison with tolerance and None values.
    """
    if set(config1.keys()) != set(config2.keys()):
        return False
    
    for key in config1.keys():
        val1 = config1[key]
        val2 = config2[key]
        
        # Recursively check nested dictionaries
        if isinstance(val1, dict) and isinstance(val2, dict):
            if not configs_are_equivalent(val1, val2):
                return False
        # Handle None values
        elif val1 is None and val2 is None:
            pass
        elif val1 is None or val2 is None:
            return False
        # Handle floating point comparison with tolerance
        elif isinstance(val1, float) and isinstance(val2, float):
            if abs(val1 - val2) > 1e-9:
                return False
        # Handle lists
        elif isinstance(val1, list) and isinstance(val2, list):
            if len(val1) != len(val2):
                return False
            for item1, item2 in zip(val1, val2):
                if isinstance(item1, dict) and isinstance(item2, dict):
                    if not configs_are_equivalent(item1, item2):
                        return False
                elif item1 != item2:
                    return False
        # Direct comparison for other types
        elif val1 != val2:
            return False
    
    return True


# ─── Property-Based Tests ────────────────────────────────────────────────────


@given(config=valid_configuration())
@settings(
    max_examples=100,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow]
)
def test_config_roundtrip_property(config: Dict[str, Any]):
    """
    Property 1: Configuration Round-Trip Preservation
    
    Validates: Requirements 21.1, 21.4, 21.5
    
    For any valid Configuration object, serializing to YAML then parsing back
    SHALL produce an equivalent Configuration object with all fields preserved.
    
    Test steps:
    1. Generate random valid configuration (Hypothesis)
    2. Write configuration to YAML file (Requirement 21.4 - Pretty Printer)
    3. Load configuration using ConfigManager (Requirement 21.1 - Parser)
    4. Write loaded configuration back to YAML
    5. Load again using ConfigManager
    6. Verify final configuration equals original (Requirement 21.5 - Round-trip)
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        
        # Step 1: Write original config to YAML (serialization)
        config_path_1 = tmpdir_path / "config1.yaml"
        write_config_to_yaml(config, config_path_1)
        
        # Step 2: Parse YAML into Configuration object (Requirement 21.1)
        config_manager_1 = ConfigManager(config_path=config_path_1)
        loaded_config_1 = config_manager_1.config
        
        # Step 3: Serialize loaded config back to YAML (Requirement 21.4)
        config_path_2 = tmpdir_path / "config2.yaml"
        write_config_to_yaml(loaded_config_1, config_path_2)
        
        # Step 4: Parse YAML again (second round-trip)
        config_manager_2 = ConfigManager(config_path=config_path_2)
        loaded_config_2 = config_manager_2.config
        
        # Step 5: Verify round-trip preservation (Requirement 21.5)
        assert configs_are_equivalent(loaded_config_1, loaded_config_2), \
            "Configuration changed after round-trip through YAML serialization"
        
        # Additional verification: Check that original config matches loaded config
        # (accounting for any normalization that ConfigManager might do)
        assert configs_are_equivalent(config, loaded_config_1), \
            "Loaded configuration differs from original"


@given(
    config=valid_configuration(),
    unicode_text=st.text(min_size=1, max_size=50, alphabet=st.characters(
        whitelist_categories=('Lu', 'Ll', 'Nd', 'Lo'),  # Include CJK characters
        min_codepoint=0x0020,
        max_codepoint=0x9FFF
    ))
)
@settings(max_examples=50, deadline=None)
def test_config_roundtrip_with_unicode(config: Dict[str, Any], unicode_text: str):
    """
    Test configuration round-trip with unicode strings.
    
    Edge case: Verify that unicode characters (including CJK) survive round-trip.
    """
    # Inject unicode text into a string field
    config['models']['ollama_model'] = unicode_text
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        
        # Write and load config
        config_path = tmpdir_path / "config_unicode.yaml"
        write_config_to_yaml(config, config_path)
        
        config_manager = ConfigManager(config_path=config_path)
        loaded_config = config_manager.config
        
        # Verify unicode text preserved
        assert loaded_config['models']['ollama_model'] == unicode_text, \
            f"Unicode text not preserved: expected '{unicode_text}', got '{loaded_config['models']['ollama_model']}'"


@given(config=valid_configuration())
@settings(max_examples=50, deadline=None)
def test_config_roundtrip_with_null_values(config: Dict[str, Any]):
    """
    Test configuration round-trip with null/None values.
    
    Edge case: Verify that null values in optional fields survive round-trip.
    """
    # Set some optional fields to None
    config['audio']['input_device'] = None
    config['audio']['output_device'] = None
    config['models']['voicevox_path'] = None
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        
        # Write and load config
        config_path = tmpdir_path / "config_null.yaml"
        write_config_to_yaml(config, config_path)
        
        config_manager = ConfigManager(config_path=config_path)
        loaded_config = config_manager.config
        
        # Verify null values preserved
        assert loaded_config['audio']['input_device'] is None
        assert loaded_config['audio']['output_device'] is None
        assert loaded_config['models']['voicevox_path'] is None


@given(config=valid_configuration())
@settings(max_examples=50, deadline=None)
def test_config_roundtrip_with_boundary_values(config: Dict[str, Any]):
    """
    Test configuration round-trip with boundary values.
    
    Edge case: Verify that min/max values for ranges survive round-trip.
    """
    # Set fields to boundary values
    config['audio']['sample_rate'] = 8000  # min
    config['audio']['record_duration'] = 30  # max
    config['llm']['temperature'] = 0.0  # min
    config['llm']['top_p'] = 1.0  # max
    config['emotion']['confidence_threshold'] = 0.0  # min
    config['proactive']['retention_days'] = 3650  # max
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        
        # Write and load config
        config_path = tmpdir_path / "config_boundary.yaml"
        write_config_to_yaml(config, config_path)
        
        config_manager = ConfigManager(config_path=config_path)
        loaded_config = config_manager.config
        
        # Verify boundary values preserved
        assert loaded_config['audio']['sample_rate'] == 8000
        assert loaded_config['audio']['record_duration'] == 30
        assert abs(loaded_config['llm']['temperature'] - 0.0) < 1e-9
        assert abs(loaded_config['llm']['top_p'] - 1.0) < 1e-9
        assert abs(loaded_config['emotion']['confidence_threshold'] - 0.0) < 1e-9
        assert loaded_config['proactive']['retention_days'] == 3650


# ─── Manual Test Runner ──────────────────────────────────────────────────────


def run_manual_tests():
    """
    Run manual tests to verify property-based tests work correctly.
    
    This is useful for debugging and understanding test behavior.
    """
    print("=" * 80)
    print("Configuration Round-Trip Property Test - Task 3.4")
    print("=" * 80)
    print()
    print("Running property-based tests with Hypothesis...")
    print()
    
    # Import pytest to run tests
    try:
        import pytest
        
        # Run tests with pytest
        exit_code = pytest.main([
            __file__,
            '-v',
            '--tb=short',
            '-k', 'test_config_roundtrip'
        ])
        
        return exit_code
        
    except ImportError:
        print("pytest not installed. Running basic manual test...")
        print()
        
        # Run a simple manual test
        from hypothesis import find
        
        try:
            # Generate one example and test it
            example_config = valid_configuration().example()
            
            print("Generated example configuration:")
            print(f"  - Audio sample rate: {example_config['audio']['sample_rate']}")
            print(f"  - Whisper model: {example_config['models']['whisper_size']}")
            print(f"  - LLM temperature: {example_config['llm']['temperature']}")
            print()
            
            # Test round-trip
            with tempfile.TemporaryDirectory() as tmpdir:
                tmpdir_path = Path(tmpdir)
                config_path = tmpdir_path / "test_config.yaml"
                
                # Write config
                write_config_to_yaml(example_config, config_path)
                print(f"✓ Wrote configuration to {config_path}")
                
                # Load config
                config_manager = ConfigManager(config_path=config_path)
                loaded_config = config_manager.config
                print("✓ Loaded configuration successfully")
                
                # Verify equivalence
                if configs_are_equivalent(example_config, loaded_config):
                    print("✓ Configuration round-trip successful!")
                    print()
                    print("=" * 80)
                    print("Manual test PASSED")
                    print("=" * 80)
                    return 0
                else:
                    print("✗ Configuration changed after round-trip")
                    return 1
                    
        except Exception as e:
            print(f"✗ Test failed: {e}")
            import traceback
            traceback.print_exc()
            return 1


if __name__ == "__main__":
    sys.exit(run_manual_tests())
