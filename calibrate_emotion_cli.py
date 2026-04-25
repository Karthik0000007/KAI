#!/usr/bin/env python3
"""
Aegis Emotion Calibration CLI

Interactive command-line interface for calibrating emotion detection thresholds.
Users record audio samples in different emotional states to personalize emotion detection.

Usage:
    python calibrate_emotion_cli.py [--user USER_ID]

Requirements: 9.7
"""

import sys
import time
import argparse
from pathlib import Path
from typing import List

import sounddevice as sd
import soundfile as sf
import numpy as np

from core.emotion import (
    calibrate_emotion,
    load_calibration_data,
    clear_calibration_data,
    compute_calibrated_thresholds,
    CALIBRATION_FILE
)
from core.config import SAMPLE_RATE, AUDIO_DIR

# Temporary directory for calibration samples
CALIBRATION_AUDIO_DIR = AUDIO_DIR / "calibration"
CALIBRATION_AUDIO_DIR.mkdir(parents=True, exist_ok=True)


def record_audio_sample(duration: int = 5, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """
    Record audio from microphone.
    
    Args:
        duration: Recording duration in seconds
        sample_rate: Audio sample rate in Hz
    
    Returns:
        NumPy array of audio samples
    """
    print(f"Recording for {duration} seconds...")
    print("Speak now!")
    
    # Record audio
    audio = sd.rec(
        int(duration * sample_rate),
        samplerate=sample_rate,
        channels=1,
        dtype='float32'
    )
    sd.wait()  # Wait until recording is finished
    
    print("Recording complete!")
    return audio


def save_audio_sample(audio: np.ndarray, filepath: Path, sample_rate: int = SAMPLE_RATE) -> None:
    """
    Save audio array to WAV file.
    
    Args:
        audio: NumPy array of audio samples
        filepath: Path to save WAV file
        sample_rate: Audio sample rate in Hz
    """
    sf.write(filepath, audio, sample_rate)
    print(f"Saved audio to: {filepath}")


def calibrate_emotion_state(
    emotion_state: str,
    user_id: str,
    num_samples: int = 5,
    duration: int = 5
) -> None:
    """
    Calibrate a specific emotional state by recording multiple samples.
    
    Args:
        emotion_state: Emotional state to calibrate (calm, stressed, anxious, fatigued)
        user_id: User identifier
        num_samples: Number of samples to record
        duration: Duration of each recording in seconds
    """
    print("\n" + "=" * 80)
    print(f"CALIBRATING EMOTION: {emotion_state.upper()}")
    print("=" * 80)
    
    # Provide instructions for each emotion state
    instructions = {
        "calm": "Please speak in a calm, relaxed manner. Take deep breaths and speak slowly.",
        "stressed": "Please speak as if you're feeling stressed or overwhelmed. Speak with urgency.",
        "anxious": "Please speak as if you're feeling anxious or worried. Let your voice show nervousness.",
        "fatigued": "Please speak as if you're very tired or exhausted. Speak slowly and with low energy."
    }
    
    print(f"\nInstructions: {instructions.get(emotion_state, 'Speak naturally.')}")
    print(f"\nYou will record {num_samples} samples of {duration} seconds each.")
    print("Try to maintain the emotional state throughout all recordings.")
    print("\nSuggested phrases to say:")
    print("  - 'How are you feeling today?'")
    print("  - 'I need to talk about my health.'")
    print("  - 'I'm having trouble sleeping lately.'")
    print("  - 'My energy levels have been low.'")
    
    input("\nPress Enter when you're ready to start recording...")
    
    # Record samples
    audio_files = []
    
    for i in range(num_samples):
        print(f"\n--- Sample {i+1}/{num_samples} ---")
        
        # Countdown
        for countdown in range(3, 0, -1):
            print(f"{countdown}...")
            time.sleep(1)
        
        # Record
        audio = record_audio_sample(duration=duration)
        
        # Save to file
        filename = f"{emotion_state}_{user_id}_{i+1}_{int(time.time())}.wav"
        filepath = CALIBRATION_AUDIO_DIR / filename
        save_audio_sample(audio, filepath)
        audio_files.append(str(filepath))
        
        if i < num_samples - 1:
            print("\nGet ready for the next recording...")
            time.sleep(2)
    
    # Run calibration
    print(f"\nProcessing {len(audio_files)} samples...")
    try:
        stats = calibrate_emotion(emotion_state, audio_files, user_id)
        
        print("\n✓ Calibration successful!")
        print(f"  Pitch: {stats['pitch_mean']:.1f} Hz (±{stats['pitch_std']:.1f})")
        print(f"  Energy: {stats['energy_mean']:.4f} (±{stats['energy_std']:.4f})")
        print(f"  Speech Rate: {stats['rate_mean']:.2f} (±{stats['rate_std']:.2f})")
        print(f"  Samples: {stats['sample_count']}")
    
    except Exception as e:
        print(f"\n✗ Calibration failed: {e}")
        sys.exit(1)


def show_calibration_status(user_id: str) -> None:
    """
    Display current calibration status for a user.
    
    Args:
        user_id: User identifier
    """
    print("\n" + "=" * 80)
    print("CALIBRATION STATUS")
    print("=" * 80)
    
    calibration_data = load_calibration_data()
    
    if user_id not in calibration_data:
        print(f"\nNo calibration data found for user '{user_id}'")
        print("You need to calibrate at least 'calm' and 'stressed' states.")
        return
    
    user_data = calibration_data[user_id]
    
    print(f"\nUser: {user_id}")
    print(f"Calibration file: {CALIBRATION_FILE}")
    print(f"\nCalibrated emotional states:")
    
    for emotion_state in ["calm", "stressed", "anxious", "fatigued"]:
        if emotion_state in user_data:
            stats = user_data[emotion_state]
            print(f"  ✓ {emotion_state.capitalize()}: {stats['sample_count']} samples")
            print(f"      Pitch: {stats['pitch_mean']:.1f} Hz, "
                  f"Energy: {stats['energy_mean']:.4f}, "
                  f"Rate: {stats['rate_mean']:.2f}")
        else:
            print(f"  ✗ {emotion_state.capitalize()}: Not calibrated")
    
    # Show computed thresholds
    thresholds = compute_calibrated_thresholds(user_id)
    if thresholds:
        print(f"\nComputed thresholds:")
        for key, value in thresholds.items():
            print(f"  {key}: {value:.4f}")
    else:
        print(f"\n⚠ Insufficient calibration data to compute thresholds")
        print("  Minimum required: 'calm' and 'stressed' states")


def run_full_calibration(user_id: str, num_samples: int = 5, duration: int = 5) -> None:
    """
    Run full calibration workflow for all emotion states.
    
    Args:
        user_id: User identifier
        num_samples: Number of samples per emotion state
        duration: Duration of each recording in seconds
    """
    print("\n" + "=" * 80)
    print("AEGIS EMOTION CALIBRATION")
    print("=" * 80)
    print(f"\nUser: {user_id}")
    print(f"Samples per emotion: {num_samples}")
    print(f"Recording duration: {duration} seconds")
    
    print("\nThis calibration will personalize emotion detection to your voice.")
    print("You will record samples in 4 different emotional states:")
    print("  1. Calm (relaxed, peaceful)")
    print("  2. Stressed (overwhelmed, pressured)")
    print("  3. Anxious (worried, nervous)")
    print("  4. Fatigued (tired, exhausted)")
    
    print("\nThe entire process will take approximately 15-20 minutes.")
    
    response = input("\nDo you want to continue? (yes/no): ").strip().lower()
    if response not in ['yes', 'y']:
        print("Calibration cancelled.")
        sys.exit(0)
    
    # Calibrate each emotion state
    emotion_states = ["calm", "stressed", "anxious", "fatigued"]
    
    for emotion_state in emotion_states:
        calibrate_emotion_state(emotion_state, user_id, num_samples, duration)
        
        if emotion_state != emotion_states[-1]:
            print("\n" + "-" * 80)
            input("Press Enter to continue to the next emotion state...")
    
    # Show final status
    show_calibration_status(user_id)
    
    print("\n" + "=" * 80)
    print("CALIBRATION COMPLETE!")
    print("=" * 80)
    print("\nYour personalized emotion detection is now active.")
    print("Aegis will use your calibrated thresholds for emotion classification.")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Calibrate Aegis emotion detection for personalized thresholds"
    )
    
    parser.add_argument(
        "--user",
        type=str,
        default="default",
        help="User identifier (default: 'default')"
    )
    
    parser.add_argument(
        "--samples",
        type=int,
        default=5,
        help="Number of samples per emotion state (default: 5)"
    )
    
    parser.add_argument(
        "--duration",
        type=int,
        default=5,
        help="Recording duration in seconds (default: 5)"
    )
    
    parser.add_argument(
        "--emotion",
        type=str,
        choices=["calm", "stressed", "anxious", "fatigued"],
        help="Calibrate only a specific emotion state"
    )
    
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show current calibration status"
    )
    
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear calibration data for this user"
    )
    
    args = parser.parse_args()
    
    # Handle status command
    if args.status:
        show_calibration_status(args.user)
        sys.exit(0)
    
    # Handle clear command
    if args.clear:
        response = input(f"Clear calibration data for user '{args.user}'? (yes/no): ").strip().lower()
        if response in ['yes', 'y']:
            clear_calibration_data(args.user)
            print(f"✓ Calibration data cleared for user '{args.user}'")
        else:
            print("Operation cancelled.")
        sys.exit(0)
    
    # Handle single emotion calibration
    if args.emotion:
        calibrate_emotion_state(args.emotion, args.user, args.samples, args.duration)
        show_calibration_status(args.user)
        sys.exit(0)
    
    # Run full calibration
    run_full_calibration(args.user, args.samples, args.duration)


if __name__ == "__main__":
    main()
