# Environment Variable Configuration

## Overview

Aegis supports environment variable overrides for all configuration values. This allows you to customize configuration without modifying the `config.yaml` file, which is especially useful for:

- **Different deployment environments** (development, staging, production)
- **CI/CD pipelines** where configuration needs to be injected
- **Docker containers** where environment variables are the standard configuration method
- **Temporary overrides** for testing or debugging

## Naming Convention

Environment variables follow this pattern:

```
AEGIS_<SECTION>_<KEY>
```

Where:
- `AEGIS_` is the required prefix
- `<SECTION>` is the configuration section name in uppercase
- `<KEY>` is the configuration key in uppercase
- Underscores (`_`) separate the parts

### Examples

| Config Path | Environment Variable |
|------------|---------------------|
| `audio.sample_rate` | `AEGIS_AUDIO_SAMPLE_RATE` |
| `models.whisper_size` | `AEGIS_MODELS_WHISPER_SIZE` |
| `llm.temperature` | `AEGIS_LLM_TEMPERATURE` |
| `proactive.enabled` | `AEGIS_PROACTIVE_ENABLED` |
| `privacy.use_keyring` | `AEGIS_PRIVACY_USE_KEYRING` |

## Precedence

Configuration values are resolved in this order (highest to lowest priority):

1. **Environment variables** (highest priority)
2. **Config file** (`data/config.yaml`)
3. **Default values** (hardcoded in `core/config.py`)

This means environment variables will always override values from the config file.

## Type Conversion

Environment variables are strings, but Aegis automatically converts them to the appropriate type based on the original config value:

### Integers

```bash
export AEGIS_AUDIO_SAMPLE_RATE=22050
export AEGIS_LLM_MAX_RESPONSE_TOKENS=300
```

### Floats

```bash
export AEGIS_LLM_TEMPERATURE=0.8
export AEGIS_TTS_SPEECH_RATE=1.2
```

### Booleans

Aegis accepts multiple boolean formats (case-insensitive):

**True values:** `true`, `yes`, `1`, `on`, `enabled`
**False values:** `false`, `no`, `0`, `off`, `disabled`

```bash
export AEGIS_AUDIO_VAD_ENABLED=true
export AEGIS_PROACTIVE_ENABLED=false
export AEGIS_PRIVACY_USE_KEYRING=yes
```

### Strings

```bash
export AEGIS_MODELS_WHISPER_SIZE=small
export AEGIS_MODELS_OLLAMA_MODEL=llama2:7b
export AEGIS_LOGGING_LEVEL=DEBUG
```

## Usage Examples

### Example 1: Development Environment

Override settings for local development:

```bash
# Use smaller models for faster iteration
export AEGIS_MODELS_WHISPER_SIZE=tiny
export AEGIS_MODELS_OLLAMA_MODEL=gemma:2b

# Enable debug logging
export AEGIS_LOGGING_LEVEL=DEBUG
export AEGIS_LOGGING_DEBUG_MODE=true

# Disable proactive monitoring during development
export AEGIS_PROACTIVE_ENABLED=false

# Run Aegis
python app.py
```

### Example 2: Production Environment

Configure for production deployment:

```bash
# Use larger, more accurate models
export AEGIS_MODELS_WHISPER_SIZE=small
export AEGIS_MODELS_OLLAMA_MODEL=gemma:7b

# Conservative LLM settings
export AEGIS_LLM_TEMPERATURE=0.6
export AEGIS_LLM_MAX_RESPONSE_TOKENS=200

# Production logging
export AEGIS_LOGGING_LEVEL=WARNING
export AEGIS_LOGGING_DEBUG_MODE=false

# Enhanced security
export AEGIS_PRIVACY_USE_KEYRING=true
export AEGIS_PRIVACY_AUDIT_LOGGING_ENABLED=true

# More frequent backups
export AEGIS_BACKUP_INTERVAL_HOURS=12

# Run Aegis
python app.py
```

### Example 3: Docker Container

In a `docker-compose.yml` file:

```yaml
version: '3.8'
services:
  aegis:
    image: aegis-health-ai:latest
    environment:
      - AEGIS_MODELS_OLLAMA_URL=http://ollama:11434
      - AEGIS_MODELS_WHISPER_SIZE=base
      - AEGIS_AUDIO_SAMPLE_RATE=16000
      - AEGIS_LOGGING_LEVEL=INFO
      - AEGIS_PRIVACY_USE_KEYRING=false
      - AEGIS_PRIVACY_REQUIRE_PASSPHRASE=true
    volumes:
      - ./data:/app/data
```

### Example 4: CI/CD Testing

Override settings for automated testing:

```bash
# Fast models for quick tests
export AEGIS_MODELS_WHISPER_SIZE=tiny
export AEGIS_MODELS_OLLAMA_TIMEOUT=30

# Disable features not needed for tests
export AEGIS_PROACTIVE_ENABLED=false
export AEGIS_BACKUP_ENABLED=false
export AEGIS_DASHBOARD_ENABLED=false

# Run tests
pytest
```

### Example 5: Temporary Override

Override a single setting temporarily:

```bash
# Test with a different sample rate
AEGIS_AUDIO_SAMPLE_RATE=44100 python app.py
```

## Validation

Environment variable overrides are validated just like config file values:

- **Type checking**: Values must be convertible to the expected type
- **Range validation**: Numeric values must be within valid ranges
- **Enum validation**: String values must be from allowed options

If an environment variable provides an invalid value, Aegis will print a validation error and exit.

### Example: Invalid Override

```bash
# This will fail validation (out of range)
export AEGIS_AUDIO_SAMPLE_RATE=99999
python app.py
# Output: ERROR: audio.sample_rate must be between 8000 and 48000, got 99999
```

## Viewing Active Overrides

When Aegis starts, it prints all active environment variable overrides:

```
Config override: audio.sample_rate = 22050 (from AEGIS_AUDIO_SAMPLE_RATE)
Config override: llm.temperature = 0.8 (from AEGIS_LLM_TEMPERATURE)
Config override: logging.level = DEBUG (from AEGIS_LOGGING_LEVEL)
```

This helps you verify which overrides are active.

## Complete List of Environment Variables

### Audio Configuration

```bash
AEGIS_AUDIO_SAMPLE_RATE=16000
AEGIS_AUDIO_RECORD_DURATION=5
AEGIS_AUDIO_SILENCE_THRESHOLD=0.001
AEGIS_AUDIO_VAD_ENABLED=true
AEGIS_AUDIO_VAD_SILENCE_DURATION=1.5
AEGIS_AUDIO_INPUT_DEVICE=null
AEGIS_AUDIO_OUTPUT_DEVICE=null
```

### Model Configuration

```bash
AEGIS_MODELS_WHISPER_SIZE=base
AEGIS_MODELS_OLLAMA_MODEL=gemma:2b
AEGIS_MODELS_OLLAMA_URL=http://localhost:11434
AEGIS_MODELS_OLLAMA_TIMEOUT=120
AEGIS_MODELS_COQUI_MODEL=tts_models/en/jenny/jenny
AEGIS_MODELS_VOICEVOX_URL=http://127.0.0.1:50021
AEGIS_MODELS_VOICEVOX_SPEAKER_ID=3
AEGIS_MODELS_VOICEVOX_AUTO_START=true
AEGIS_MODELS_VOICEVOX_PATH=null
```

### LLM Configuration

```bash
AEGIS_LLM_CONTEXT_WINDOW=4096
AEGIS_LLM_MAX_RESPONSE_TOKENS=200
AEGIS_LLM_TEMPERATURE=0.7
AEGIS_LLM_TOP_P=0.9
AEGIS_LLM_LLM_EXTRACTION_ENABLED=true
AEGIS_LLM_EXTRACTION_CONFIDENCE_THRESHOLD=0.6
AEGIS_LLM_CONTEXT_AWARE=true
AEGIS_LLM_CONTEXT_TURNS=3
```

### TTS Configuration

```bash
AEGIS_TTS_SPEECH_RATE=1.0
AEGIS_TTS_VOLUME=0.8
AEGIS_TTS_ADAPTIVE_RATE=true
AEGIS_TTS_FALLBACK_ENGINE=pyttsx3
```

### Emotion Detection Configuration

```bash
AEGIS_EMOTION_PITCH_LOW=100.0
AEGIS_EMOTION_PITCH_HIGH=250.0
AEGIS_EMOTION_ENERGY_LOW=0.01
AEGIS_EMOTION_ENERGY_HIGH=0.08
AEGIS_EMOTION_RATE_SLOW=1.5
AEGIS_EMOTION_RATE_FAST=4.0
AEGIS_EMOTION_CONFIDENCE_THRESHOLD=0.4
AEGIS_EMOTION_CALIBRATION_ENABLED=false
AEGIS_EMOTION_MIXED_EMOTIONS_ENABLED=true
AEGIS_EMOTION_VISUAL_CORRELATION_ENABLED=false
AEGIS_EMOTION_AUDIO_WEIGHT=0.6
```

### Proactive Monitoring Configuration

```bash
AEGIS_PROACTIVE_ENABLED=true
AEGIS_PROACTIVE_CHECK_INTERVAL_MINUTES=60
AEGIS_PROACTIVE_LOW_MOOD_DAYS_THRESHOLD=3
AEGIS_PROACTIVE_MOOD_LOW_THRESHOLD=4.0
AEGIS_PROACTIVE_LOW_SLEEP_HOURS=5.0
AEGIS_PROACTIVE_SLEEP_DEFICIT_DAYS=3
AEGIS_PROACTIVE_ELEVATED_HR_THRESHOLD=100
AEGIS_PROACTIVE_ELEVATED_HR_DURATION=10
AEGIS_PROACTIVE_MISSED_MEDICATION_REMINDER_DELAY=30
AEGIS_PROACTIVE_MAX_ALERTS_PER_DAY=3
AEGIS_PROACTIVE_DEDUPLICATION_WINDOW_HOURS=24
AEGIS_PROACTIVE_PRIORITIZE_ALERTS=true
AEGIS_PROACTIVE_INCLUDE_EXPLANATIONS=true
```

### Privacy and Security Configuration

```bash
AEGIS_PRIVACY_ENCRYPTION_ALGORITHM=AES-256-GCM
AEGIS_PRIVACY_USE_KEYRING=true
AEGIS_PRIVACY_REQUIRE_PASSPHRASE=false
AEGIS_PRIVACY_PBKDF2_ITERATIONS=100000
AEGIS_PRIVACY_ENABLE_DIFFERENTIAL_PRIVACY=true
AEGIS_PRIVACY_DIFFERENTIAL_PRIVACY_EPSILON=1.0
AEGIS_PRIVACY_AUTO_LOCK_MINUTES=30
AEGIS_PRIVACY_AUDIT_LOGGING_ENABLED=true
AEGIS_PRIVACY_SECURE_DELETE=true
AEGIS_PRIVACY_RETENTION_DAYS=365
```

### Logging Configuration

```bash
AEGIS_LOGGING_LEVEL=INFO
AEGIS_LOGGING_FILE=data/logs/aegis.log
AEGIS_LOGGING_MAX_SIZE_MB=10
AEGIS_LOGGING_BACKUP_COUNT=5
AEGIS_LOGGING_FORMAT=json
AEGIS_LOGGING_CONSOLE_ENABLED=true
AEGIS_LOGGING_METRICS_ENABLED=true
AEGIS_LOGGING_DEBUG_MODE=false
AEGIS_LOGGING_DEBUG_DIR=data/debug
```

### Backup Configuration

```bash
AEGIS_BACKUP_ENABLED=true
AEGIS_BACKUP_INTERVAL_HOURS=24
AEGIS_BACKUP_RETENTION_DAYS=30
AEGIS_BACKUP_BACKUP_DIR=data/db/backups
AEGIS_BACKUP_VERIFY_INTEGRITY=true
AEGIS_BACKUP_COMPRESS=true
AEGIS_BACKUP_BACKUP_ON_SHUTDOWN=true
```

## Best Practices

1. **Use environment variables for deployment-specific settings** (URLs, timeouts, resource limits)
2. **Keep sensitive settings in environment variables** (API keys, passphrases) rather than config files
3. **Document your environment variables** in deployment documentation or `.env.example` files
4. **Use `.env` files** with tools like `python-dotenv` for local development
5. **Validate environment variables** before deployment to catch configuration errors early
6. **Use consistent naming** across environments to avoid confusion

## Troubleshooting

### Override Not Applied

**Problem:** Environment variable set but config value unchanged

**Solution:** Check that:
- Variable name follows the `AEGIS_<SECTION>_<KEY>` pattern exactly
- Variable is exported: `export AEGIS_AUDIO_SAMPLE_RATE=22050`
- Variable is set in the same shell session where you run Aegis
- No typos in the variable name (case-sensitive)

### Type Conversion Error

**Problem:** Warning about failed type conversion

**Solution:** Ensure the environment variable value matches the expected type:
- Integers: `AEGIS_AUDIO_SAMPLE_RATE=16000` (no quotes, no decimals)
- Floats: `AEGIS_LLM_TEMPERATURE=0.7` (use decimal point)
- Booleans: `AEGIS_PROACTIVE_ENABLED=true` (use true/false, yes/no, 1/0)
- Strings: `AEGIS_MODELS_WHISPER_SIZE=base` (any text)

### Validation Error

**Problem:** Configuration validation fails with environment override

**Solution:** Check that the override value is within the valid range:
- Review the error message for the valid range
- Consult `data/config.yaml` comments for valid values
- Ensure the value makes sense for the setting

## Related Documentation

- [Configuration File Reference](../data/config.yaml) - Complete configuration options
- [Deployment Guide](deployment.md) - Deployment-specific configuration
- [Docker Guide](docker.md) - Using environment variables with Docker
