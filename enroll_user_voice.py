#!/usr/bin/env python3
"""
User Voice Enrollment Tool

Interactive CLI tool for enrolling users with voice biometrics.
Records multiple voice samples and creates a voice profile.

Requirements: 16.5

Usage:
    python enroll_user_voice.py
"""

import sys
import logging
from pathlib import Path
import time
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from core.voice_biometrics import VoiceBiometrics, SPEECHBRAIN_AVAILABLE
from core.user_manager import UserManager
from core.stt import record_audio

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def print_header():
    """Print enrollment header."""
    print("\n" + "="*60)
    print("  Aegis Voice Enrollment System")
    print("="*60)
    print()


def record_enrollment_samples(
    num_samples: int = 5,
    duration: int = 5
) -> list:
    """
    Record voice samples for enrollment.
    
    Args:
        num_samples: Number of samples to record
        duration: Duration of each sample in seconds
        
    Returns:
        List of paths to recorded audio files
    """
    print(f"\n📝 Recording {num_samples} voice samples...")
    print(f"   Each sample will be {duration} seconds long.")
    print()
    
    samples_dir = Path("data/audio/enrollment")
    samples_dir.mkdir(parents=True, exist_ok=True)
    
    audio_files = []
    
    for i in range(num_samples):
        print(f"Sample {i+1}/{num_samples}:")
        print("  Please read the following phrase:")
        print()
        
        # Provide different phrases for variety
        phrases = [
            "  'My voice is my password, verify me.'",
            "  'The quick brown fox jumps over the lazy dog.'",
            "  'I am enrolling my voice for Aegis health assistant.'",
            "  'Security and privacy are important to me.'",
            "  'This is my unique voice signature.'"
        ]
        print(phrases[i % len(phrases)])
        print()
        
        input("  Press Enter when ready to record...")
        
        # Record audio
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        audio_path = samples_dir / f"enrollment_sample_{i+1}_{timestamp}.wav"
        
        print(f"  🎤 Recording for {duration} seconds...")
        try:
            record_audio(str(audio_path), duration=duration)
            audio_files.append(str(audio_path))
            print("  ✓ Sample recorded successfully")
        except Exception as e:
            logger.error(f"Error recording sample: {e}")
            print(f"  ✗ Error recording sample: {e}")
            continue
        
        print()
        time.sleep(0.5)
    
    return audio_files


def enroll_new_user():
    """Enroll a new user with voice biometrics."""
    print_header()
    
    # Check if SpeechBrain is available
    if not SPEECHBRAIN_AVAILABLE:
        print("❌ Error: SpeechBrain is not installed.")
        print()
        print("To install SpeechBrain:")
        print("  pip install speechbrain torch torchaudio")
        print()
        return False
    
    # Initialize managers
    print("Initializing voice biometrics system...")
    voice_bio = VoiceBiometrics()
    user_manager = UserManager()
    
    if not voice_bio.is_available():
        print("❌ Error: Voice biometrics system not available.")
        return False
    
    print("✓ Voice biometrics system ready")
    print()
    
    # Get user information
    print("User Information")
    print("-" * 60)
    
    user_id = input("Enter user ID (e.g., john_doe): ").strip()
    if not user_id:
        print("❌ User ID cannot be empty")
        return False
    
    # Check if user already exists
    existing_user = user_manager.get_user(user_id)
    if existing_user:
        print(f"\n⚠️  User '{user_id}' already exists.")
        overwrite = input("Do you want to re-enroll? (yes/no): ").strip().lower()
        if overwrite != 'yes':
            print("Enrollment cancelled.")
            return False
    else:
        display_name = input("Enter display name: ").strip()
        if not display_name:
            display_name = user_id
        
        language = input("Enter preferred language (en/ja/es/fr/de) [en]: ").strip() or "en"
        
        is_child_input = input("Is this a child profile? (yes/no) [no]: ").strip().lower()
        is_child = is_child_input == 'yes'
        
        parent_user_id = None
        if is_child:
            parent_user_id = input("Enter parent user ID: ").strip()
            if not parent_user_id:
                print("❌ Parent user ID required for child profiles")
                return False
        
        # Create user profile
        print(f"\nCreating user profile for '{user_id}'...")
        user_manager.create_user(
            user_id=user_id,
            display_name=display_name,
            language=language,
            is_child=is_child,
            parent_user_id=parent_user_id
        )
        print("✓ User profile created")
    
    # Record voice samples
    print("\n" + "="*60)
    print("Voice Sample Recording")
    print("="*60)
    print()
    print("You will be asked to record 5 voice samples.")
    print("Please speak clearly and naturally.")
    print("Try to use your normal speaking voice.")
    print()
    
    input("Press Enter to begin recording...")
    
    audio_samples = record_enrollment_samples(num_samples=5, duration=5)
    
    if len(audio_samples) < 5:
        print(f"\n❌ Error: Only {len(audio_samples)} samples recorded.")
        print("   At least 5 samples are required for enrollment.")
        return False
    
    # Enroll user
    print("\n" + "="*60)
    print("Processing Voice Samples")
    print("="*60)
    print()
    print("Extracting voice features...")
    print("This may take a minute...")
    
    success = voice_bio.enroll_user(
        user_id=user_id,
        audio_samples=audio_samples,
        created_at=datetime.now().isoformat()
    )
    
    if success:
        # Mark user as enrolled
        user_manager.mark_voice_enrolled(user_id, len(audio_samples))
        
        print()
        print("="*60)
        print("✅ Enrollment Successful!")
        print("="*60)
        print()
        print(f"User '{user_id}' has been enrolled with voice biometrics.")
        print(f"Samples recorded: {len(audio_samples)}")
        print()
        print("You can now use voice identification with Aegis.")
        print()
        return True
    else:
        print()
        print("="*60)
        print("❌ Enrollment Failed")
        print("="*60)
        print()
        print("Please try again with clearer audio samples.")
        print()
        return False


def list_enrolled_users():
    """List all enrolled users."""
    print_header()
    
    user_manager = UserManager()
    voice_bio = VoiceBiometrics()
    
    users = user_manager.get_all_users()
    enrolled_users = voice_bio.get_enrolled_users()
    
    print("Enrolled Users")
    print("-" * 60)
    print()
    
    if not users:
        print("No users found.")
        return
    
    for user in users:
        status = "✓ Voice Enrolled" if user.user_id in enrolled_users else "✗ Not Enrolled"
        print(f"  {user.user_id}")
        print(f"    Name: {user.display_name}")
        print(f"    Language: {user.language}")
        print(f"    Status: {status}")
        if user.is_child:
            print(f"    Child Profile (Parent: {user.parent_user_id})")
        print()


def main():
    """Main enrollment interface."""
    while True:
        print("\n" + "="*60)
        print("  Aegis Voice Enrollment")
        print("="*60)
        print()
        print("1. Enroll new user")
        print("2. List enrolled users")
        print("3. Exit")
        print()
        
        choice = input("Select option (1-3): ").strip()
        
        if choice == '1':
            enroll_new_user()
        elif choice == '2':
            list_enrolled_users()
        elif choice == '3':
            print("\nGoodbye!")
            break
        else:
            print("\n❌ Invalid option. Please select 1-3.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nEnrollment cancelled by user.")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        print(f"\n❌ Error: {e}")
        sys.exit(1)
