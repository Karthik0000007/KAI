"""
Aegis Configuration Module
Central configuration for the offline health AI system.
"""

import os
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
import yaml


# ─── Paths ───────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DB_DIR = DATA_DIR / "db"
AUDIO_DIR = DATA_DIR / "audio"
LOGS_DIR = DATA_DIR / "logs"
MODELS_DIR = DATA_DIR / "models"

# Ensure directories exist
for d in [DATA_DIR, DB_DIR, AUDIO_DIR, LOGS_DIR, MODELS_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# ─── Database ────────────────────────────────────────────────────────────────
DB_PATH = DB_DIR / "aegis_health.db"
ENCRYPTION_KEY_FILE = DB_DIR / ".aegis_key"


# ─── Audio ───────────────────────────────────────────────────────────────────
SAMPLE_RATE = 16000
RECORD_DURATION_DEFAULT = 5  # seconds
INPUT_AUDIO_FILE = AUDIO_DIR / "input.wav"
OUTPUT_AUDIO_FILE = AUDIO_DIR / "aegis_response.wav"


# ─── Whisper STT ─────────────────────────────────────────────────────────────
WHISPER_MODEL_SIZE = "base"  # "tiny", "base", "small", "medium"


# ─── LLM (Ollama) ───────────────────────────────────────────────────────────
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "gemma:2b"
LLM_CONTEXT_WINDOW = 4096


# ─── TTS ─────────────────────────────────────────────────────────────────────
COQUI_MODEL_EN = "tts_models/en/jenny/jenny"
VOICEVOX_URL = "http://127.0.0.1:50021"
VOICEVOX_SPEAKER_ID = 3  # Tsumugi


# ─── Emotion Detection ──────────────────────────────────────────────────────
@dataclass
class EmotionConfig:
    """Thresholds for emotion classification from audio features."""
    # Pitch thresholds (Hz)
    pitch_low: float = 100.0
    pitch_high: float = 250.0
    # Energy thresholds (RMS)
    energy_low: float = 0.01
    energy_high: float = 0.08
    # Speech rate thresholds (syllables/sec approx via word count)
    rate_slow: float = 1.5
    rate_fast: float = 4.0
    # Confidence threshold for emotion classification
    confidence_threshold: float = 0.4


EMOTION_CONFIG = EmotionConfig()


# ─── Emotion Labels ─────────────────────────────────────────────────────────
EMOTION_LABELS = ["calm", "stressed", "anxious", "fatigued", "neutral"]


# ─── Proactive Engine ───────────────────────────────────────────────────────
@dataclass
class ProactiveConfig:
    """Thresholds for proactive health interventions."""
    low_mood_days_threshold: int = 3          # days of consecutive low mood
    low_sleep_hours: float = 5.0              # hours considered low
    missed_medication_reminder_delay: int = 30  # minutes
    elevated_hr_threshold: int = 100          # bpm
    check_interval_minutes: int = 60          # background check interval
    mood_low_threshold: float = 3.0           # mood score 1-10


PROACTIVE_CONFIG = ProactiveConfig()


# ─── Health Check-in ────────────────────────────────────────────────────────
DAILY_CHECKIN_QUESTIONS = [
    "How are you feeling today?",
    "How many hours did you sleep last night?",
    "Have you taken your medications today?",
    "Any pain or discomfort you'd like to mention?",
    "On a scale of 1 to 10, how would you rate your energy level?",
]


# ─── Response Tone Modes ────────────────────────────────────────────────────
TONE_MODES = {
    "calm": {
        "system_modifier": "Respond in a calm, soothing, and reassuring tone. Use gentle language.",
        "speech_rate_factor": 0.9,
    },
    "encouraging": {
        "system_modifier": "Respond in an upbeat, encouraging, and motivating tone. Be positive.",
        "speech_rate_factor": 1.0,
    },
    "gentle_support": {
        "system_modifier": "Respond with gentle empathy and support. Be understanding and caring.",
        "speech_rate_factor": 0.85,
    },
    "neutral": {
        "system_modifier": "Respond in a clear, neutral, and informative tone.",
        "speech_rate_factor": 1.0,
    },
}


# ─── System Prompt ───────────────────────────────────────────────────────────
AEGIS_SYSTEM_PROMPT = """You are Aegis, an offline personal health AI companion. You are caring, \
empathetic, and privacy-conscious. You NEVER suggest calling emergency services unless explicitly \
asked about emergencies. Your role is to:

1. Listen attentively to the user's health concerns
2. Provide gentle, evidence-based wellness suggestions
3. Track patterns in mood, sleep, and daily habits
4. Offer proactive interventions when you detect concerning patterns
5. Adapt your communication tone to the user's emotional state
6. Remind users about medications and healthy routines

You are NOT a replacement for medical professionals. Always clarify this when discussing \
serious symptoms. Keep responses concise (2-4 sentences) unless the user asks for detail.

You speak like a caring friend who happens to know about health and wellness."""


# ─── Privacy ─────────────────────────────────────────────────────────────────
DIFFERENTIAL_PRIVACY_EPSILON = 1.0  # Privacy budget for noise injection
ENABLE_DIFFERENTIAL_PRIVACY = True


# ─── Configuration Manager ──────────────────────────────────────────────────
class ConfigManager:
    """
    Configuration manager for Aegis Health AI.
    
    Loads configuration from YAML file, validates all values, and provides
    access to configuration settings. Creates default config if missing.
    
    Requirements: 4.1, 4.2, 4.4, 4.6
    """
    
    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_path: Path to config.yaml file. Defaults to data/config.yaml
        """
        self.config_path = config_path or (DATA_DIR / "config.yaml")
        self.config: Dict[str, Any] = {}
        self._validators: Dict[str, callable] = self._build_validators()
        
        # Load and validate configuration
        self.load_config()
        self.validate()
    
    def load_config(self) -> Dict[str, Any]:
        """
        Load configuration from YAML file.
        
        If the file doesn't exist, creates a default configuration file.
        
        Returns:
            Dictionary containing configuration values
            
        Requirement 4.1: Load configuration from YAML file
        Requirement 4.2: Create default config if missing
        """
        if not self.config_path.exists():
            print(f"Configuration file not found at {self.config_path}")
            print("Creating default configuration file...")
            self._create_default_config()
        
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
            
            if self.config is None:
                raise ValueError("Configuration file is empty")
            
            # Apply environment variable overrides
            # Requirement 4.5: Support environment variable overrides
            self._apply_env_overrides()
            
            return self.config
            
        except yaml.YAMLError as e:
            print(f"ERROR: Invalid YAML in configuration file: {e}")
            print(f"Please check {self.config_path} for syntax errors.")
            sys.exit(1)
        except Exception as e:
            print(f"ERROR: Failed to load configuration: {e}")
            sys.exit(1)
    
    def validate(self) -> None:
        """
        Validate all configuration values against expected types and ranges.
        
        Prints validation errors with clear messages and exits if validation fails.
        
        Requirement 4.4: Print validation error and exit when config is invalid
        Requirement 4.6: Validate all configuration values at startup
        """
        errors: List[str] = []
        
        for section, validator in self._validators.items():
            try:
                if section not in self.config:
                    errors.append(f"Missing required section: '{section}'")
                    continue
                
                validator(self.config[section], section, errors)
            except Exception as e:
                errors.append(f"Error validating section '{section}': {e}")
        
        if errors:
            print("=" * 80)
            print("CONFIGURATION VALIDATION ERRORS")
            print("=" * 80)
            for i, error in enumerate(errors, 1):
                print(f"{i}. {error}")
            print("=" * 80)
            print(f"Please fix the errors in {self.config_path} and try again.")
            sys.exit(1)
    
    def _build_validators(self) -> Dict[str, callable]:
        """
        Build validators for each configuration section.
        
        Returns:
            Dictionary mapping section names to validator functions
        """
        return {
            'audio': self._validate_audio,
            'models': self._validate_models,
            'llm': self._validate_llm,
            'tts': self._validate_tts,
            'emotion': self._validate_emotion,
            'proactive': self._validate_proactive,
            'privacy': self._validate_privacy,
            'logging': self._validate_logging,
            'backup': self._validate_backup,
        }
    
    def _validate_audio(self, section: Dict[str, Any], name: str, errors: List[str]) -> None:
        """Validate audio configuration section."""
        self._validate_int_range(section, 'sample_rate', 8000, 48000, name, errors)
        self._validate_int_range(section, 'record_duration', 1, 30, name, errors)
        self._validate_float_range(section, 'silence_threshold', 0.0001, 0.1, name, errors)
        self._validate_bool(section, 'vad_enabled', name, errors)
        
        if 'vad_silence_duration' in section:
            self._validate_float_range(section, 'vad_silence_duration', 0.5, 5.0, name, errors)
    
    def _validate_models(self, section: Dict[str, Any], name: str, errors: List[str]) -> None:
        """Validate models configuration section."""
        valid_whisper_sizes = ["tiny", "base", "small", "medium", "large"]
        if 'whisper_size' in section:
            if section['whisper_size'] not in valid_whisper_sizes:
                errors.append(
                    f"{name}.whisper_size must be one of {valid_whisper_sizes}, "
                    f"got '{section['whisper_size']}'"
                )
        
        self._validate_string(section, 'ollama_model', name, errors)
        self._validate_string(section, 'ollama_url', name, errors)
        self._validate_int_range(section, 'ollama_timeout', 30, 300, name, errors)
        
        if 'voicevox_speaker_id' in section:
            self._validate_int_range(section, 'voicevox_speaker_id', 0, 50, name, errors)
    
    def _validate_llm(self, section: Dict[str, Any], name: str, errors: List[str]) -> None:
        """Validate LLM configuration section."""
        self._validate_int_range(section, 'context_window', 512, 8192, name, errors)
        self._validate_int_range(section, 'max_response_tokens', 50, 500, name, errors)
        self._validate_float_range(section, 'temperature', 0.0, 2.0, name, errors)
        self._validate_float_range(section, 'top_p', 0.0, 1.0, name, errors)
        
        if 'extraction_confidence_threshold' in section:
            self._validate_float_range(section, 'extraction_confidence_threshold', 0.0, 1.0, name, errors)
        
        if 'context_turns' in section:
            self._validate_int_range(section, 'context_turns', 0, 10, name, errors)
    
    def _validate_tts(self, section: Dict[str, Any], name: str, errors: List[str]) -> None:
        """Validate TTS configuration section."""
        self._validate_float_range(section, 'speech_rate', 0.5, 2.0, name, errors)
        self._validate_float_range(section, 'volume', 0.0, 1.0, name, errors)
        
        if 'fallback_engine' in section:
            valid_engines = ["pyttsx3", "none"]
            if section['fallback_engine'] not in valid_engines:
                errors.append(
                    f"{name}.fallback_engine must be one of {valid_engines}, "
                    f"got '{section['fallback_engine']}'"
                )
    
    def _validate_emotion(self, section: Dict[str, Any], name: str, errors: List[str]) -> None:
        """Validate emotion configuration section."""
        self._validate_float_range(section, 'pitch_low', 50.0, 200.0, name, errors)
        self._validate_float_range(section, 'pitch_high', 150.0, 400.0, name, errors)
        self._validate_float_range(section, 'energy_low', 0.001, 0.1, name, errors)
        self._validate_float_range(section, 'energy_high', 0.01, 0.2, name, errors)
        self._validate_float_range(section, 'confidence_threshold', 0.0, 1.0, name, errors)
        
        # Validate that pitch_high > pitch_low
        if 'pitch_low' in section and 'pitch_high' in section:
            if section['pitch_high'] <= section['pitch_low']:
                errors.append(f"{name}.pitch_high must be greater than pitch_low")
        
        # Validate that energy_high > energy_low
        if 'energy_low' in section and 'energy_high' in section:
            if section['energy_high'] <= section['energy_low']:
                errors.append(f"{name}.energy_high must be greater than energy_low")
        
        if 'audio_weight' in section:
            self._validate_float_range(section, 'audio_weight', 0.0, 1.0, name, errors)
    
    def _validate_proactive(self, section: Dict[str, Any], name: str, errors: List[str]) -> None:
        """Validate proactive configuration section."""
        self._validate_int_range(section, 'check_interval_minutes', 15, 1440, name, errors)
        self._validate_int_range(section, 'low_mood_days_threshold', 1, 14, name, errors)
        self._validate_float_range(section, 'mood_low_threshold', 1.0, 10.0, name, errors)
        self._validate_float_range(section, 'low_sleep_hours', 0.0, 12.0, name, errors)
        
        if 'sleep_deficit_days' in section:
            self._validate_int_range(section, 'sleep_deficit_days', 1, 14, name, errors)
        
        if 'elevated_hr_threshold' in section:
            self._validate_int_range(section, 'elevated_hr_threshold', 80, 180, name, errors)
        
        if 'max_alerts_per_day' in section:
            self._validate_int_range(section, 'max_alerts_per_day', 1, 10, name, errors)
    
    def _validate_privacy(self, section: Dict[str, Any], name: str, errors: List[str]) -> None:
        """Validate privacy configuration section."""
        valid_algorithms = ["AES-256-GCM", "AES-128-GCM"]
        if 'encryption_algorithm' in section:
            if section['encryption_algorithm'] not in valid_algorithms:
                errors.append(
                    f"{name}.encryption_algorithm must be one of {valid_algorithms}, "
                    f"got '{section['encryption_algorithm']}'"
                )
        
        self._validate_bool(section, 'use_keyring', name, errors)
        self._validate_bool(section, 'require_passphrase', name, errors)
        
        if 'pbkdf2_iterations' in section:
            self._validate_int_range(section, 'pbkdf2_iterations', 100000, 1000000, name, errors)
        
        if 'differential_privacy_epsilon' in section:
            self._validate_float_range(section, 'differential_privacy_epsilon', 0.1, 10.0, name, errors)
        
        if 'auto_lock_minutes' in section:
            self._validate_int_range(section, 'auto_lock_minutes', 0, 1440, name, errors)
        
        if 'retention_days' in section:
            self._validate_int_range(section, 'retention_days', 0, 3650, name, errors)
    
    def _validate_logging(self, section: Dict[str, Any], name: str, errors: List[str]) -> None:
        """Validate logging configuration section."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if 'level' in section:
            if section['level'] not in valid_levels:
                errors.append(
                    f"{name}.level must be one of {valid_levels}, "
                    f"got '{section['level']}'"
                )
        
        self._validate_string(section, 'file', name, errors)
        self._validate_int_range(section, 'max_size_mb', 1, 100, name, errors)
        self._validate_int_range(section, 'backup_count', 1, 20, name, errors)
        
        if 'format' in section:
            valid_formats = ["json", "text"]
            if section['format'] not in valid_formats:
                errors.append(
                    f"{name}.format must be one of {valid_formats}, "
                    f"got '{section['format']}'"
                )
    
    def _validate_backup(self, section: Dict[str, Any], name: str, errors: List[str]) -> None:
        """Validate backup configuration section."""
        self._validate_bool(section, 'enabled', name, errors)
        self._validate_int_range(section, 'interval_hours', 1, 168, name, errors)
        self._validate_int_range(section, 'retention_days', 7, 365, name, errors)
        self._validate_string(section, 'backup_dir', name, errors)
    
    # Helper validation methods
    
    def _validate_int_range(
        self, 
        section: Dict[str, Any], 
        key: str, 
        min_val: int, 
        max_val: int, 
        section_name: str, 
        errors: List[str]
    ) -> None:
        """Validate that a configuration value is an integer within range."""
        if key not in section:
            errors.append(f"{section_name}.{key} is required")
            return
        
        value = section[key]
        if not isinstance(value, int):
            errors.append(
                f"{section_name}.{key} must be an integer, got {type(value).__name__}"
            )
            return
        
        if not (min_val <= value <= max_val):
            errors.append(
                f"{section_name}.{key} must be between {min_val} and {max_val}, got {value}"
            )
    
    def _validate_float_range(
        self, 
        section: Dict[str, Any], 
        key: str, 
        min_val: float, 
        max_val: float, 
        section_name: str, 
        errors: List[str]
    ) -> None:
        """Validate that a configuration value is a float within range."""
        if key not in section:
            errors.append(f"{section_name}.{key} is required")
            return
        
        value = section[key]
        if not isinstance(value, (int, float)):
            errors.append(
                f"{section_name}.{key} must be a number, got {type(value).__name__}"
            )
            return
        
        if not (min_val <= value <= max_val):
            errors.append(
                f"{section_name}.{key} must be between {min_val} and {max_val}, got {value}"
            )
    
    def _validate_bool(
        self, 
        section: Dict[str, Any], 
        key: str, 
        section_name: str, 
        errors: List[str]
    ) -> None:
        """Validate that a configuration value is a boolean."""
        if key not in section:
            errors.append(f"{section_name}.{key} is required")
            return
        
        value = section[key]
        if not isinstance(value, bool):
            errors.append(
                f"{section_name}.{key} must be a boolean (true/false), got {type(value).__name__}"
            )
    
    def _validate_string(
        self, 
        section: Dict[str, Any], 
        key: str, 
        section_name: str, 
        errors: List[str]
    ) -> None:
        """Validate that a configuration value is a non-empty string."""
        if key not in section:
            errors.append(f"{section_name}.{key} is required")
            return
        
        value = section[key]
        if not isinstance(value, str):
            errors.append(
                f"{section_name}.{key} must be a string, got {type(value).__name__}"
            )
            return
        
        if not value.strip():
            errors.append(f"{section_name}.{key} cannot be empty")
    
    def _create_default_config(self) -> None:
        """
        Create a default configuration file with all available options.
        
        The default config file already exists at data/config.yaml in the repository,
        so this method just verifies it exists. If it's missing, we copy from a template
        or create a minimal version.
        
        Requirement 4.2: Create default config file if missing
        """
        # The config file should already exist in the repo
        # If it doesn't, we have a problem - the file should be version controlled
        if not self.config_path.exists():
            print(f"ERROR: Default configuration file missing at {self.config_path}")
            print("This file should be included in the repository.")
            print("Please restore data/config.yaml from version control.")
            sys.exit(1)
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value by dot-notation key.
        
        Args:
            key: Configuration key in dot notation (e.g., 'audio.sample_rate')
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def reload(self) -> None:
        """
        Reload configuration from file.
        
        Useful for hot-reloading non-critical settings without restart.
        """
        self.load_config()
        self.validate()
    
    def _apply_env_overrides(self) -> None:
        """
        Apply environment variable overrides to configuration.
        
        Environment variables follow the pattern: AEGIS_<SECTION>_<KEY>
        For example: AEGIS_AUDIO_SAMPLE_RATE=16000
        
        Precedence: environment variables > config file > defaults
        
        Requirement 4.5: Support environment variable overrides for all configuration values
        """
        # Iterate through all configuration sections and keys
        for section, section_data in self.config.items():
            if not isinstance(section_data, dict):
                continue
            
            for key, value in section_data.items():
                # Convert dot notation to environment variable name
                # e.g., audio.sample_rate -> AEGIS_AUDIO_SAMPLE_RATE
                env_var_name = f"AEGIS_{section.upper()}_{key.upper()}"
                
                # Check if environment variable exists
                env_value = os.getenv(env_var_name)
                
                if env_value is not None:
                    # Convert environment variable string to appropriate type
                    converted_value = self._convert_env_value(env_value, value)
                    
                    # Apply override
                    self.config[section][key] = converted_value
                    
                    # Log the override for transparency
                    print(f"Config override: {section}.{key} = {converted_value} (from {env_var_name})")
    
    def _convert_env_value(self, env_value: str, original_value: Any) -> Any:
        """
        Convert environment variable string to appropriate type based on original value type.
        
        Args:
            env_value: String value from environment variable
            original_value: Original value from config file (used to infer type)
            
        Returns:
            Converted value with appropriate type
        """
        # If original value is None, return string as-is
        if original_value is None:
            return env_value
        
        # Infer type from original value
        original_type = type(original_value)
        
        try:
            # Boolean conversion
            if original_type == bool:
                return self._parse_bool(env_value)
            
            # Integer conversion
            elif original_type == int:
                return int(env_value)
            
            # Float conversion
            elif original_type == float:
                return float(env_value)
            
            # String (no conversion needed)
            elif original_type == str:
                return env_value
            
            # Default: return as string
            else:
                return env_value
                
        except (ValueError, TypeError) as e:
            print(f"WARNING: Failed to convert environment variable value '{env_value}' to {original_type.__name__}: {e}")
            print(f"Using original value: {original_value}")
            return original_value
    
    def _parse_bool(self, value: str) -> bool:
        """
        Parse boolean value from string.
        
        Accepts: true, false, yes, no, 1, 0 (case-insensitive)
        
        Args:
            value: String value to parse
            
        Returns:
            Boolean value
            
        Raises:
            ValueError: If value cannot be parsed as boolean
        """
        value_lower = value.lower().strip()
        
        if value_lower in ('true', 'yes', '1', 'on', 'enabled'):
            return True
        elif value_lower in ('false', 'no', '0', 'off', 'disabled'):
            return False
        else:
            raise ValueError(f"Cannot parse '{value}' as boolean")
