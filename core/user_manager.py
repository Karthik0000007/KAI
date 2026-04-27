"""
User Manager - Multi-User Profile Management

Manages multiple user profiles with separate databases and encryption keys.
Handles user creation, switching, and profile management.

Requirements: 16.1, 16.5, 16.6, 16.7
"""

import logging
import json
from pathlib import Path
from typing import Optional, Dict, List
from dataclasses import dataclass, asdict
from datetime import datetime
import hashlib

from core.health_db import HealthDatabase
from core.key_manager import KeyManager

logger = logging.getLogger(__name__)


@dataclass
class UserProfile:
    """User profile with preferences and settings."""
    user_id: str
    display_name: str
    created_at: str
    last_active: str
    
    # Preferences
    language: str = "en"  # en, ja, es, fr, de
    tts_voice: str = "default"
    speech_rate: float = 1.0
    volume: float = 1.0
    
    # Privacy settings
    dp_epsilon: float = 1.0
    retention_days: int = 365
    vision_enabled: bool = True
    wearable_enabled: bool = True
    
    # Family sharing
    family_group_id: Optional[str] = None
    is_child: bool = False
    parent_user_id: Optional[str] = None
    shared_data_types: List[str] = None
    
    # Voice biometrics
    voice_enrolled: bool = False
    enrollment_sample_count: int = 0
    
    def __post_init__(self):
        if self.shared_data_types is None:
            self.shared_data_types = []
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'UserProfile':
        """Create from dictionary."""
        return cls(**data)


@dataclass
class FamilyGroup:
    """Family group for data sharing."""
    group_id: str
    group_name: str
    created_at: str
    admin_user_id: str
    member_user_ids: List[str]
    shared_data_types: List[str]  # mood, sleep, energy, vitals, alerts
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'FamilyGroup':
        """Create from dictionary."""
        return cls(**data)


class UserManager:
    """
    Manages multiple user profiles with separate databases and encryption keys.
    
    Requirements: 16.1, 16.5, 16.6, 16.7
    """
    
    def __init__(
        self,
        users_dir: Path = Path("data/users"),
        default_user_id: str = "default"
    ):
        """
        Initialize user manager.
        
        Args:
            users_dir: Directory to store user data
            default_user_id: Default user ID for single-user mode
        """
        self.users_dir = Path(users_dir)
        self.users_dir.mkdir(parents=True, exist_ok=True)
        self.default_user_id = default_user_id
        
        self.profiles: Dict[str, UserProfile] = {}
        self.family_groups: Dict[str, FamilyGroup] = {}
        self.current_user_id: Optional[str] = None
        self.current_db: Optional[HealthDatabase] = None
        
        self._load_profiles()
        self._load_family_groups()
        
        # Create default user if no users exist
        if not self.profiles:
            self.create_user(
                user_id=default_user_id,
                display_name="Default User"
            )
    
    def _load_profiles(self):
        """Load user profiles from disk."""
        profiles_file = self.users_dir / "profiles.json"
        if profiles_file.exists():
            try:
                with open(profiles_file, 'r') as f:
                    data = json.load(f)
                    for profile_data in data:
                        profile = UserProfile.from_dict(profile_data)
                        self.profiles[profile.user_id] = profile
                logger.info(f"Loaded {len(self.profiles)} user profiles")
            except Exception as e:
                logger.error(f"Error loading profiles: {e}")
    
    def _save_profiles(self):
        """Save user profiles to disk."""
        profiles_file = self.users_dir / "profiles.json"
        try:
            data = [profile.to_dict() for profile in self.profiles.values()]
            with open(profiles_file, 'w') as f:
                json.dump(data, f, indent=2)
            logger.info(f"Saved {len(self.profiles)} user profiles")
        except Exception as e:
            logger.error(f"Error saving profiles: {e}")
    
    def _load_family_groups(self):
        """Load family groups from disk."""
        groups_file = self.users_dir / "family_groups.json"
        if groups_file.exists():
            try:
                with open(groups_file, 'r') as f:
                    data = json.load(f)
                    for group_data in data:
                        group = FamilyGroup.from_dict(group_data)
                        self.family_groups[group.group_id] = group
                logger.info(f"Loaded {len(self.family_groups)} family groups")
            except Exception as e:
                logger.error(f"Error loading family groups: {e}")
    
    def _save_family_groups(self):
        """Save family groups to disk."""
        groups_file = self.users_dir / "family_groups.json"
        try:
            data = [group.to_dict() for group in self.family_groups.values()]
            with open(groups_file, 'w') as f:
                json.dump(data, f, indent=2)
            logger.info(f"Saved {len(self.family_groups)} family groups")
        except Exception as e:
            logger.error(f"Error saving family groups: {e}")
    
    def create_user(
        self,
        user_id: str,
        display_name: str,
        language: str = "en",
        is_child: bool = False,
        parent_user_id: Optional[str] = None
    ) -> UserProfile:
        """
        Create a new user profile.
        
        Args:
            user_id: Unique user identifier
            display_name: User's display name
            language: Preferred language
            is_child: Whether this is a child profile
            parent_user_id: Parent user ID for child profiles
            
        Returns:
            Created UserProfile
            
        Requirements: 16.1, 16.5
        """
        if user_id in self.profiles:
            logger.warning(f"User {user_id} already exists")
            return self.profiles[user_id]
        
        # Create user directory
        user_dir = self.users_dir / user_id
        user_dir.mkdir(parents=True, exist_ok=True)
        
        # Create user profile
        now = datetime.now().isoformat()
        profile = UserProfile(
            user_id=user_id,
            display_name=display_name,
            created_at=now,
            last_active=now,
            language=language,
            is_child=is_child,
            parent_user_id=parent_user_id
        )
        
        self.profiles[user_id] = profile
        self._save_profiles()
        
        logger.info(f"Created user {user_id} ({display_name})")
        return profile
    
    def get_user(self, user_id: str) -> Optional[UserProfile]:
        """Get user profile by ID."""
        return self.profiles.get(user_id)
    
    def get_all_users(self) -> List[UserProfile]:
        """Get all user profiles."""
        return list(self.profiles.values())
    
    def switch_user(self, user_id: str) -> bool:
        """
        Switch to a different user.
        
        Args:
            user_id: User ID to switch to
            
        Returns:
            True if switch successful, False otherwise
            
        Requirements: 16.3, 16.7
        """
        if user_id not in self.profiles:
            logger.error(f"User {user_id} not found")
            return False
        
        # Close current database
        if self.current_db:
            self.current_db.close()
        
        # Load user's database
        user_dir = self.users_dir / user_id
        db_path = user_dir / "health.db"
        
        try:
            # Open database (uses global encryption key for now)
            # TODO: Implement per-user encryption keys
            self.current_db = HealthDatabase(db_path=db_path)
            self.current_user_id = user_id
            
            # Update last active
            profile = self.profiles[user_id]
            profile.last_active = datetime.now().isoformat()
            self._save_profiles()
            
            logger.info(f"Switched to user {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error switching to user {user_id}: {e}")
            return False
    
    def get_current_user(self) -> Optional[UserProfile]:
        """Get current active user profile."""
        if self.current_user_id:
            return self.profiles.get(self.current_user_id)
        return None
    
    def get_current_database(self) -> Optional[HealthDatabase]:
        """Get current user's database."""
        return self.current_db
    
    def update_user_preferences(
        self,
        user_id: str,
        **preferences
    ) -> bool:
        """
        Update user preferences.
        
        Args:
            user_id: User ID
            **preferences: Preference key-value pairs
            
        Returns:
            True if update successful
            
        Requirements: 16.6
        """
        if user_id not in self.profiles:
            logger.error(f"User {user_id} not found")
            return False
        
        profile = self.profiles[user_id]
        
        # Update preferences
        for key, value in preferences.items():
            if hasattr(profile, key):
                setattr(profile, key, value)
            else:
                logger.warning(f"Unknown preference: {key}")
        
        self._save_profiles()
        logger.info(f"Updated preferences for user {user_id}")
        return True
    
    def mark_voice_enrolled(
        self,
        user_id: str,
        sample_count: int
    ) -> bool:
        """Mark user as voice enrolled."""
        if user_id not in self.profiles:
            return False
        
        profile = self.profiles[user_id]
        profile.voice_enrolled = True
        profile.enrollment_sample_count = sample_count
        self._save_profiles()
        
        logger.info(f"User {user_id} marked as voice enrolled")
        return True
    
    def create_family_group(
        self,
        group_name: str,
        admin_user_id: str,
        shared_data_types: List[str] = None
    ) -> FamilyGroup:
        """
        Create a family group for data sharing.
        
        Args:
            group_name: Name of the family group
            admin_user_id: User ID of the group administrator
            shared_data_types: Types of data to share
            
        Returns:
            Created FamilyGroup
            
        Requirements: 16.8
        """
        if shared_data_types is None:
            shared_data_types = ["mood", "sleep", "energy"]
        
        # Generate group ID
        group_id = hashlib.sha256(
            f"{group_name}{admin_user_id}{datetime.now().isoformat()}".encode()
        ).hexdigest()[:16]
        
        group = FamilyGroup(
            group_id=group_id,
            group_name=group_name,
            created_at=datetime.now().isoformat(),
            admin_user_id=admin_user_id,
            member_user_ids=[admin_user_id],
            shared_data_types=shared_data_types
        )
        
        self.family_groups[group_id] = group
        self._save_family_groups()
        
        # Update admin user's profile
        if admin_user_id in self.profiles:
            self.profiles[admin_user_id].family_group_id = group_id
            self._save_profiles()
        
        logger.info(f"Created family group {group_id} ({group_name})")
        return group
    
    def add_family_member(
        self,
        group_id: str,
        user_id: str
    ) -> bool:
        """
        Add user to family group.
        
        Requirements: 16.8
        """
        if group_id not in self.family_groups:
            logger.error(f"Family group {group_id} not found")
            return False
        
        if user_id not in self.profiles:
            logger.error(f"User {user_id} not found")
            return False
        
        group = self.family_groups[group_id]
        if user_id not in group.member_user_ids:
            group.member_user_ids.append(user_id)
            self._save_family_groups()
        
        # Update user's profile
        profile = self.profiles[user_id]
        profile.family_group_id = group_id
        self._save_profiles()
        
        logger.info(f"Added user {user_id} to family group {group_id}")
        return True
    
    def get_family_members(self, group_id: str) -> List[UserProfile]:
        """Get all members of a family group."""
        if group_id not in self.family_groups:
            return []
        
        group = self.family_groups[group_id]
        return [
            self.profiles[user_id]
            for user_id in group.member_user_ids
            if user_id in self.profiles
        ]
    
    def can_access_data(
        self,
        requesting_user_id: str,
        target_user_id: str,
        data_type: str
    ) -> bool:
        """
        Check if user can access another user's data.
        
        Args:
            requesting_user_id: User requesting access
            target_user_id: User whose data is being accessed
            data_type: Type of data (mood, sleep, energy, etc.)
            
        Returns:
            True if access allowed
            
        Requirements: 16.7, 16.9
        """
        # Users can always access their own data
        if requesting_user_id == target_user_id:
            return True
        
        # Check if users are in the same family group
        requesting_profile = self.profiles.get(requesting_user_id)
        target_profile = self.profiles.get(target_user_id)
        
        if not requesting_profile or not target_profile:
            return False
        
        # Check family group membership
        if requesting_profile.family_group_id == target_profile.family_group_id:
            group_id = requesting_profile.family_group_id
            if group_id and group_id in self.family_groups:
                group = self.family_groups[group_id]
                
                # Check if data type is shared
                if data_type in group.shared_data_types:
                    return True
                
                # Parents can access child data
                if target_profile.is_child and target_profile.parent_user_id == requesting_user_id:
                    return True
        
        return False
    
    def delete_user(self, user_id: str) -> bool:
        """Delete user profile and data."""
        if user_id not in self.profiles:
            return False
        
        # Remove from family groups
        profile = self.profiles[user_id]
        if profile.family_group_id:
            group = self.family_groups.get(profile.family_group_id)
            if group and user_id in group.member_user_ids:
                group.member_user_ids.remove(user_id)
                self._save_family_groups()
        
        # Delete profile
        del self.profiles[user_id]
        self._save_profiles()
        
        logger.info(f"Deleted user {user_id}")
        return True
