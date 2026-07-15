#!/usr/bin/env python3
"""
Aegis Setup Wizard
Guides new users through configuring language, TTS voice, privacy settings,
and testing basic dependencies.
"""

import sys
import time
import requests
import yaml
from pathlib import Path

# Add core to path so we can import config
sys.path.append(str(Path(__file__).resolve().parent))

from core.config import ConfigManager, DATA_DIR

def print_header(title):
    print("\n" + "="*50)
    print(f"  {title}")
    print("="*50 + "\n")

def check_dependencies():
    print_header("Step 1: Checking Dependencies")
    print("Checking for Ollama (LLM)...")
    try:
        response = requests.get("http://localhost:11434/")
        if response.status_code == 200:
            print("[OK] Ollama is running.")
        else:
            print("[WARN] Ollama is running but returned unexpected status.")
    except requests.ConnectionError:
        print("[ERROR] Ollama is not running. Please start Ollama before using Aegis.")
        
    print("\nChecking for VOICEVOX (Japanese TTS)...")
    try:
        response = requests.get("http://127.0.0.1:50021/version")
        if response.status_code == 200:
            print("[OK] VOICEVOX is running.")
        else:
            print("[WARN] VOICEVOX is running but returned unexpected status.")
    except requests.ConnectionError:
        print("[WARN] VOICEVOX is not running. Japanese TTS will use fallback if selected.")

def test_audio():
    print_header("Step 2: Audio Test")
    print("Testing audio output...")
    try:
        import numpy as np
        import sounddevice as sd
        # Generate a 1-second 440Hz beep
        fs = 44100
        t = np.linspace(0, 1, fs, False)
        beep = np.sin(440 * 2 * np.pi * t)
        print("Playing a test beep. Please listen...")
        sd.play(beep, fs)
        sd.wait()
        print("[OK] Audio output tested.")
    except ImportError:
        print("[WARN] sounddevice or numpy not installed. Skipping audio test.")
    except Exception as e:
        print(f"[ERROR] Failed to play audio: {e}")

def configure_preferences():
    print_header("Step 3: Personalization & Privacy")
    
    # Load existing config
    cm = ConfigManager()
    config = cm.config
    config_path = cm.config_path
    
    # Language
    print("Select your preferred language:")
    print("1. English (en)")
    print("2. Japanese (ja)")
    print("3. Spanish (es)")
    print("4. French (fr)")
    print("5. German (de)")
    lang_choice = input("Enter choice (1-5) [default 1]: ").strip()
    lang_map = {'1': 'en', '2': 'ja', '3': 'es', '4': 'fr', '5': 'de'}
    
    if 'global' not in config:
        config['global'] = {}
    config['global']['language'] = lang_map.get(lang_choice, 'en')
    
    # TTS Voice (just update the config if we had it, but let's skip tts voice string for now if it's too engine specific)
    
    # Privacy
    print("\nPrivacy Configuration:")
    print("Aegis uses Differential Privacy to protect your health data.")
    print("A lower epsilon value provides more privacy but adds more noise.")
    
    if 'privacy' not in config:
        config['privacy'] = {}
        
    current_epsilon = config['privacy'].get('differential_privacy_epsilon', 1.0)
    epsilon_str = input(f"Enter DP Epsilon [current: {current_epsilon}]: ").strip()
    if epsilon_str:
        try:
            config['privacy']['differential_privacy_epsilon'] = float(epsilon_str)
        except ValueError:
            print("Invalid input, keeping default.")
            
    current_retention = config['privacy'].get('retention_days', 365)
    retention_str = input(f"Enter data retention period in days [current: {current_retention}]: ").strip()
    if retention_str:
        try:
            config['privacy']['retention_days'] = int(retention_str)
        except ValueError:
            print("Invalid input, keeping default.")

    # Save to yaml
    print("\nSaving preferences...")
    try:
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False)
        print(f"[OK] Preferences saved to {config_path}")
    except Exception as e:
        print(f"[ERROR] Failed to save config: {e}")

def main():
    print_header("Welcome to Aegis Setup Wizard")
    print("This wizard will help you configure your offline health AI.")
    time.sleep(1)
    
    check_dependencies()
    time.sleep(1)
    
    test_audio()
    time.sleep(1)
    
    configure_preferences()
    time.sleep(1)
    
    print_header("Setup Complete!")
    print("You can now run 'python app.py' to start Aegis.")

if __name__ == "__main__":
    main()
