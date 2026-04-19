"""
Aegis Startup Validator Module
Validates all system dependencies at startup before attempting to use them.
"""

import logging
import shutil
import subprocess
from typing import List, Dict, Optional, Tuple

import requests
import sounddevice as sd

from core.config import OLLAMA_URL, OLLAMA_MODEL

logger = logging.getLogger("aegis.startup_validator")


class ValidationError(Exception):
    """Raised when a critical dependency validation fails."""
    pass


class StartupValidator:
    """
    Validates all system dependencies at startup.
    
    Checks:
    - Ollama server reachability
    - Ollama model availability
    - ffmpeg availability
    - Microphone availability
    - Speaker availability
    
    Provides clear error messages with remediation steps for each failure.
    """
    
    def __init__(self):
        self.validation_results: List[Dict[str, any]] = []
    
    def validate_all(self) -> bool:
        """
        Run all validation checks.
        
        Returns:
            True if all validations pass, False otherwise.
        
        Raises:
            ValidationError: If any critical dependency is missing.
        """
        logger.info("Starting dependency validation...")
        
        checks = [
            ("Ollama Server", self._validate_ollama_server),
            ("Ollama Model", self._validate_ollama_model),
            ("ffmpeg", self._validate_ffmpeg),
            ("Microphone", self._validate_microphone),
            ("Speaker", self._validate_speaker),
        ]
        
        all_passed = True
        failed_checks = []
        
        for check_name, check_func in checks:
            try:
                passed, message = check_func()
                self.validation_results.append({
                    "check": check_name,
                    "passed": passed,
                    "message": message
                })
                
                if passed:
                    logger.info(f"✓ {check_name}: {message}")
                    print(f"  ✓ {check_name}: {message}")
                else:
                    logger.error(f"✗ {check_name}: {message}")
                    print(f"  ✗ {check_name}: {message}")
                    all_passed = False
                    failed_checks.append((check_name, message))
                    
            except Exception as e:
                logger.error(f"✗ {check_name}: Validation failed with error: {e}")
                print(f"  ✗ {check_name}: Validation failed with error: {e}")
                all_passed = False
                failed_checks.append((check_name, str(e)))
        
        if not all_passed:
            print("\n" + "="*70)
            print("STARTUP VALIDATION FAILED")
            print("="*70)
            print("\nThe following dependencies are missing or misconfigured:\n")
            
            for check_name, message in failed_checks:
                print(f"  • {check_name}")
                print(f"    {message}\n")
            
            print("Please resolve these issues before starting Aegis.")
            print("="*70 + "\n")
            
            raise ValidationError(
                f"Startup validation failed: {len(failed_checks)} check(s) failed"
            )
        
        logger.info("All dependency validations passed")
        print("\n  All dependency checks passed! ✓\n")
        return True
    
    def _validate_ollama_server(self) -> Tuple[bool, str]:
        """
        Validate that Ollama server is reachable.
        
        Returns:
            Tuple of (success, message)
        """
        try:
            response = requests.get(
                f"{OLLAMA_URL.replace('/api/generate', '')}/api/tags",
                timeout=5
            )
            
            if response.status_code == 200:
                return True, "Ollama server is running"
            else:
                return False, (
                    f"Ollama server returned status {response.status_code}\n"
                    f"    Remediation: Check Ollama server logs for errors"
                )
                
        except requests.ConnectionError:
            return False, (
                f"Cannot connect to Ollama at {OLLAMA_URL}\n"
                f"    Remediation: Start Ollama with 'ollama serve' or ensure it's running"
            )
        except requests.Timeout:
            return False, (
                f"Ollama server at {OLLAMA_URL} timed out\n"
                f"    Remediation: Check if Ollama is responding or restart it"
            )
        except Exception as e:
            return False, (
                f"Unexpected error connecting to Ollama: {e}\n"
                f"    Remediation: Verify OLLAMA_URL in config.py is correct"
            )
    
    def _validate_ollama_model(self) -> Tuple[bool, str]:
        """
        Validate that the configured Ollama model is available.
        
        Returns:
            Tuple of (success, message)
        """
        try:
            response = requests.get(
                f"{OLLAMA_URL.replace('/api/generate', '')}/api/tags",
                timeout=5
            )
            
            if response.status_code != 200:
                return False, (
                    f"Cannot query Ollama models (status {response.status_code})\n"
                    f"    Remediation: Ensure Ollama server is running properly"
                )
            
            data = response.json()
            models = data.get("models", [])
            model_names = [m.get("name", "") for m in models]
            
            # Check if our configured model is in the list
            if OLLAMA_MODEL in model_names:
                return True, f"Model '{OLLAMA_MODEL}' is available"
            
            # Check if model name without tag matches
            model_base = OLLAMA_MODEL.split(":")[0]
            matching_models = [m for m in model_names if m.startswith(model_base)]
            
            if matching_models:
                return True, f"Model '{OLLAMA_MODEL}' is available (found: {matching_models[0]})"
            
            return False, (
                f"Model '{OLLAMA_MODEL}' is not pulled\n"
                f"    Available models: {', '.join(model_names) if model_names else 'none'}\n"
                f"    Remediation: Pull the model with 'ollama pull {OLLAMA_MODEL}'"
            )
            
        except requests.ConnectionError:
            return False, (
                f"Cannot connect to Ollama to check models\n"
                f"    Remediation: Ensure Ollama server is running first"
            )
        except Exception as e:
            return False, (
                f"Error checking Ollama models: {e}\n"
                f"    Remediation: Verify Ollama is running and accessible"
            )
    
    def _validate_ffmpeg(self) -> Tuple[bool, str]:
        """
        Validate that ffmpeg is available in PATH.
        
        Returns:
            Tuple of (success, message)
        """
        ffmpeg_path = shutil.which("ffmpeg")
        
        if ffmpeg_path:
            try:
                # Try to get version to confirm it works
                result = subprocess.run(
                    ["ffmpeg", "-version"],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                
                if result.returncode == 0:
                    # Extract version from first line
                    version_line = result.stdout.split('\n')[0]
                    return True, f"ffmpeg is available ({version_line})"
                else:
                    return False, (
                        f"ffmpeg found but not working properly\n"
                        f"    Remediation: Reinstall ffmpeg"
                    )
                    
            except subprocess.TimeoutExpired:
                return False, (
                    f"ffmpeg command timed out\n"
                    f"    Remediation: Check ffmpeg installation"
                )
            except Exception as e:
                return False, (
                    f"Error running ffmpeg: {e}\n"
                    f"    Remediation: Verify ffmpeg installation"
                )
        else:
            return False, (
                f"ffmpeg not found in PATH\n"
                f"    Remediation: Install ffmpeg:\n"
                f"      - Windows: Download from https://ffmpeg.org/download.html\n"
                f"      - macOS: brew install ffmpeg\n"
                f"      - Linux: sudo apt install ffmpeg (Ubuntu/Debian) or sudo yum install ffmpeg (RHEL/CentOS)"
            )
    
    def _validate_microphone(self) -> Tuple[bool, str]:
        """
        Validate that a microphone is available.
        
        Returns:
            Tuple of (success, message)
        """
        try:
            # Query available input devices
            devices = sd.query_devices()
            
            # Find input devices
            input_devices = [
                d for d in devices
                if isinstance(d, dict) and d.get('max_input_channels', 0) > 0
            ]
            
            if not input_devices:
                return False, (
                    f"No microphone devices found\n"
                    f"    Remediation: Connect a microphone and ensure it's enabled in system settings"
                )
            
            # Get default input device
            try:
                default_input = sd.query_devices(kind='input')
                device_name = default_input.get('name', 'Unknown')
                return True, f"Microphone available: {device_name}"
            except Exception:
                # If no default, but we have input devices, that's still okay
                device_name = input_devices[0].get('name', 'Unknown')
                return True, f"Microphone available: {device_name}"
                
        except sd.PortAudioError as e:
            return False, (
                f"Audio system error: {e}\n"
                f"    Remediation: Check audio drivers and system audio settings"
            )
        except Exception as e:
            return False, (
                f"Error checking microphone: {e}\n"
                f"    Remediation: Verify audio system is working properly"
            )
    
    def _validate_speaker(self) -> Tuple[bool, str]:
        """
        Validate that a speaker is available.
        
        Returns:
            Tuple of (success, message)
        """
        try:
            # Query available output devices
            devices = sd.query_devices()
            
            # Find output devices
            output_devices = [
                d for d in devices
                if isinstance(d, dict) and d.get('max_output_channels', 0) > 0
            ]
            
            if not output_devices:
                return False, (
                    f"No speaker devices found\n"
                    f"    Remediation: Connect speakers/headphones and ensure they're enabled in system settings"
                )
            
            # Get default output device
            try:
                default_output = sd.query_devices(kind='output')
                device_name = default_output.get('name', 'Unknown')
                return True, f"Speaker available: {device_name}"
            except Exception:
                # If no default, but we have output devices, that's still okay
                device_name = output_devices[0].get('name', 'Unknown')
                return True, f"Speaker available: {device_name}"
                
        except sd.PortAudioError as e:
            return False, (
                f"Audio system error: {e}\n"
                f"    Remediation: Check audio drivers and system audio settings"
            )
        except Exception as e:
            return False, (
                f"Error checking speaker: {e}\n"
                f"    Remediation: Verify audio system is working properly"
            )
    
    def get_results(self) -> List[Dict[str, any]]:
        """
        Get the results of all validation checks.
        
        Returns:
            List of validation results with check name, pass/fail status, and message.
        """
        return self.validation_results
