"""
Tests for User Manager Module

Requirements: 16.1, 16.5, 16.6, 16.7, 16.8, 16.9, 18.1
"""

import pytest
from pathlib import Path
import tempfile
import shutil
from datetime import datetime

from core.user_manager import (
    UserManager,
    UserProfile,
    FamilyGroup
)


@pytest.fixture
def temp_users_dir():
    """Create temporary directory for user data."""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def user_manager(temp_users_dir):
    """Create UserManager instance."""
    return UserManager(users_dir=temp_users_dir, default_user_id="test_default")


# ─── UserProfile Tests ──────────────────────────────────────────────────

def test_user_profile_creation():
    """Test UserProfile creation."""
    profile = UserProfile(
        user_id="test_user",
        display_name="Test User",
        created_at="2024-01-01T00:00:00",
        last_active="2024-01-01T00:00:00"
    )
    
    assert profile.user_id == "test_user"
    assert profile.display_name == "Test User"
    assert profile.language == "en"
    assert not profile.is_child


def test_user_profile_serialization():
    """Test UserProfile to/from dict."""
    profile = UserProfile(
        user_id="test_user",
        display_name="Test User",
        created_at="2024-01-01T00:00:00",
        last_active="2024-01-01T00:00:00",
        language="ja",
        is_child=True,
        parent_user_id="parent_user"
    )
    
    # To dict
    data = profile.to_dict()
    assert data['user_id'] == "test_user"
    assert data['language'] == "ja"
    assert data['is_child'] is True
    
    # From dict
    restored = UserProfile.from_dict(data)
    assert restored.user_id == profile.user_id
    assert restored.language == profile.language
    assert restored.is_child == profile.is_child


# ─── UserManager Initialization Tests ───────────────────────────────────

def test_user_manager_initialization(user_manager):
    """Test UserManager initialization."""
    assert user_manager is not None
    assert user_manager.users_dir.exists()
    
    # Should create default user
    assert "test_default" in user_manager.profiles


def test_user_manager_creates_default_user(temp_users_dir):
    """Test that UserManager creates default user if none exist."""
    manager = UserManager(users_dir=temp_users_dir)
    
    users = manager.get_all_users()
    assert len(users) >= 1
    assert any(u.user_id == "default" for u in users)


# ─── User Creation Tests ────────────────────────────────────────────────

def test_create_user(user_manager):
    """Test creating a new user."""
    profile = user_manager.create_user(
        user_id="john_doe",
        display_name="John Doe",
        language="en"
    )
    
    assert profile.user_id == "john_doe"
    assert profile.display_name == "John Doe"
    assert profile.language == "en"
    assert not profile.is_child
    
    # Check user directory created
    user_dir = user_manager.users_dir / "john_doe"
    assert user_dir.exists()


def test_create_child_user(user_manager):
    """Test creating a child user profile."""
    # Create parent first
    parent = user_manager.create_user(
        user_id="parent",
        display_name="Parent User"
    )
    
    # Create child
    child = user_manager.create_user(
        user_id="child",
        display_name="Child User",
        is_child=True,
        parent_user_id="parent"
    )
    
    assert child.is_child
    assert child.parent_user_id == "parent"


def test_create_duplicate_user(user_manager):
    """Test creating user with existing ID."""
    # Create first user
    user1 = user_manager.create_user(
        user_id="duplicate",
        display_name="First"
    )
    
    # Try to create duplicate
    user2 = user_manager.create_user(
        user_id="duplicate",
        display_name="Second"
    )
    
    # Should return existing user
    assert user2.user_id == user1.user_id
    assert user2.display_name == "First"  # Original name preserved


# ─── User Retrieval Tests ───────────────────────────────────────────────

def test_get_user(user_manager):
    """Test getting user by ID."""
    user_manager.create_user(
        user_id="test_user",
        display_name="Test User"
    )
    
    profile = user_manager.get_user("test_user")
    assert profile is not None
    assert profile.user_id == "test_user"


def test_get_nonexistent_user(user_manager):
    """Test getting user that doesn't exist."""
    profile = user_manager.get_user("nonexistent")
    assert profile is None


def test_get_all_users(user_manager):
    """Test getting all users."""
    # Create multiple users
    user_manager.create_user("user1", "User 1")
    user_manager.create_user("user2", "User 2")
    user_manager.create_user("user3", "User 3")
    
    users = user_manager.get_all_users()
    assert len(users) >= 4  # 3 + default user
    
    user_ids = [u.user_id for u in users]
    assert "user1" in user_ids
    assert "user2" in user_ids
    assert "user3" in user_ids


# ─── User Switching Tests ───────────────────────────────────────────────

def test_switch_user(user_manager):
    """Test switching between users."""
    # Create users
    user_manager.create_user("user1", "User 1")
    user_manager.create_user("user2", "User 2")
    
    # Switch to user1
    success = user_manager.switch_user("user1")
    assert success
    assert user_manager.current_user_id == "user1"
    
    current = user_manager.get_current_user()
    assert current.user_id == "user1"


def test_switch_to_nonexistent_user(user_manager):
    """Test switching to user that doesn't exist."""
    success = user_manager.switch_user("nonexistent")
    assert not success
    assert user_manager.current_user_id is None


def test_get_current_database(user_manager):
    """Test getting current user's database."""
    user_manager.create_user("test_user", "Test User")
    user_manager.switch_user("test_user")
    
    db = user_manager.get_current_database()
    assert db is not None


# ─── User Preferences Tests ─────────────────────────────────────────────

def test_update_user_preferences(user_manager):
    """Test updating user preferences."""
    user_manager.create_user("test_user", "Test User")
    
    success = user_manager.update_user_preferences(
        "test_user",
        language="ja",
        speech_rate=1.5,
        volume=0.8
    )
    
    assert success
    
    profile = user_manager.get_user("test_user")
    assert profile.language == "ja"
    assert profile.speech_rate == 1.5
    assert profile.volume == 0.8


def test_update_preferences_nonexistent_user(user_manager):
    """Test updating preferences for non-existent user."""
    success = user_manager.update_user_preferences(
        "nonexistent",
        language="ja"
    )
    
    assert not success


def test_update_invalid_preference(user_manager):
    """Test updating invalid preference."""
    user_manager.create_user("test_user", "Test User")
    
    # Should not raise error, just log warning
    success = user_manager.update_user_preferences(
        "test_user",
        invalid_field="value"
    )
    
    assert success  # Still succeeds, just ignores invalid field


# ─── Voice Enrollment Tests ─────────────────────────────────────────────

def test_mark_voice_enrolled(user_manager):
    """Test marking user as voice enrolled."""
    user_manager.create_user("test_user", "Test User")
    
    success = user_manager.mark_voice_enrolled("test_user", 5)
    assert success
    
    profile = user_manager.get_user("test_user")
    assert profile.voice_enrolled
    assert profile.enrollment_sample_count == 5


def test_mark_voice_enrolled_nonexistent_user(user_manager):
    """Test marking non-existent user as enrolled."""
    success = user_manager.mark_voice_enrolled("nonexistent", 5)
    assert not success


# ─── Family Group Tests ─────────────────────────────────────────────────

def test_create_family_group(user_manager):
    """Test creating a family group."""
    # Create admin user
    user_manager.create_user("admin", "Admin User")
    
    # Create family group
    group = user_manager.create_family_group(
        group_name="Smith Family",
        admin_user_id="admin",
        shared_data_types=["mood", "sleep"]
    )
    
    assert group.group_name == "Smith Family"
    assert group.admin_user_id == "admin"
    assert "admin" in group.member_user_ids
    assert "mood" in group.shared_data_types
    
    # Check admin user's profile updated
    admin_profile = user_manager.get_user("admin")
    assert admin_profile.family_group_id == group.group_id


def test_add_family_member(user_manager):
    """Test adding member to family group."""
    # Create users
    user_manager.create_user("admin", "Admin")
    user_manager.create_user("member", "Member")
    
    # Create group
    group = user_manager.create_family_group(
        group_name="Test Family",
        admin_user_id="admin"
    )
    
    # Add member
    success = user_manager.add_family_member(group.group_id, "member")
    assert success
    
    # Check member added
    updated_group = user_manager.family_groups[group.group_id]
    assert "member" in updated_group.member_user_ids
    
    # Check member's profile updated
    member_profile = user_manager.get_user("member")
    assert member_profile.family_group_id == group.group_id


def test_get_family_members(user_manager):
    """Test getting all family members."""
    # Create users
    user_manager.create_user("admin", "Admin")
    user_manager.create_user("member1", "Member 1")
    user_manager.create_user("member2", "Member 2")
    
    # Create group and add members
    group = user_manager.create_family_group(
        group_name="Test Family",
        admin_user_id="admin"
    )
    user_manager.add_family_member(group.group_id, "member1")
    user_manager.add_family_member(group.group_id, "member2")
    
    # Get members
    members = user_manager.get_family_members(group.group_id)
    assert len(members) == 3
    
    member_ids = [m.user_id for m in members]
    assert "admin" in member_ids
    assert "member1" in member_ids
    assert "member2" in member_ids


# ─── Data Access Control Tests ──────────────────────────────────────────

def test_can_access_own_data(user_manager):
    """Test that users can access their own data."""
    user_manager.create_user("test_user", "Test User")
    
    can_access = user_manager.can_access_data(
        "test_user",
        "test_user",
        "mood"
    )
    
    assert can_access


def test_cannot_access_other_user_data(user_manager):
    """Test that users cannot access other users' data by default."""
    user_manager.create_user("user1", "User 1")
    user_manager.create_user("user2", "User 2")
    
    can_access = user_manager.can_access_data(
        "user1",
        "user2",
        "mood"
    )
    
    assert not can_access


def test_can_access_family_shared_data(user_manager):
    """Test that family members can access shared data."""
    # Create users
    user_manager.create_user("user1", "User 1")
    user_manager.create_user("user2", "User 2")
    
    # Create family group with shared data
    group = user_manager.create_family_group(
        group_name="Test Family",
        admin_user_id="user1",
        shared_data_types=["mood", "sleep"]
    )
    user_manager.add_family_member(group.group_id, "user2")
    
    # Check access to shared data
    can_access_mood = user_manager.can_access_data("user1", "user2", "mood")
    can_access_sleep = user_manager.can_access_data("user1", "user2", "sleep")
    can_access_energy = user_manager.can_access_data("user1", "user2", "energy")
    
    assert can_access_mood
    assert can_access_sleep
    assert not can_access_energy  # Not shared


def test_parent_can_access_child_data(user_manager):
    """Test that parents can access child data."""
    # Create parent and child
    user_manager.create_user("parent", "Parent")
    user_manager.create_user(
        "child",
        "Child",
        is_child=True,
        parent_user_id="parent"
    )
    
    # Create family group
    group = user_manager.create_family_group(
        group_name="Family",
        admin_user_id="parent"
    )
    user_manager.add_family_member(group.group_id, "child")
    
    # Parent should access child data
    can_access = user_manager.can_access_data("parent", "child", "mood")
    assert can_access


# ─── User Deletion Tests ────────────────────────────────────────────────

def test_delete_user(user_manager):
    """Test deleting a user."""
    user_manager.create_user("test_user", "Test User")
    
    success = user_manager.delete_user("test_user")
    assert success
    
    # User should be gone
    profile = user_manager.get_user("test_user")
    assert profile is None


def test_delete_nonexistent_user(user_manager):
    """Test deleting user that doesn't exist."""
    success = user_manager.delete_user("nonexistent")
    assert not success


def test_delete_user_removes_from_family(user_manager):
    """Test that deleting user removes them from family group."""
    # Create users and family
    user_manager.create_user("admin", "Admin")
    user_manager.create_user("member", "Member")
    
    group = user_manager.create_family_group(
        group_name="Family",
        admin_user_id="admin"
    )
    user_manager.add_family_member(group.group_id, "member")
    
    # Delete member
    user_manager.delete_user("member")
    
    # Check member removed from group
    updated_group = user_manager.family_groups[group.group_id]
    assert "member" not in updated_group.member_user_ids


# ─── Persistence Tests ──────────────────────────────────────────────────

def test_profiles_persist_across_instances(temp_users_dir):
    """Test that profiles persist across UserManager instances."""
    # Create first instance and add user
    manager1 = UserManager(users_dir=temp_users_dir)
    manager1.create_user("persistent_user", "Persistent User")
    
    # Create second instance
    manager2 = UserManager(users_dir=temp_users_dir)
    
    # Check if user persisted
    profile = manager2.get_user("persistent_user")
    assert profile is not None
    assert profile.display_name == "Persistent User"


def test_family_groups_persist_across_instances(temp_users_dir):
    """Test that family groups persist across instances."""
    # Create first instance
    manager1 = UserManager(users_dir=temp_users_dir)
    manager1.create_user("admin", "Admin")
    group = manager1.create_family_group(
        group_name="Persistent Family",
        admin_user_id="admin"
    )
    
    # Create second instance
    manager2 = UserManager(users_dir=temp_users_dir)
    
    # Check if group persisted
    assert group.group_id in manager2.family_groups
    loaded_group = manager2.family_groups[group.group_id]
    assert loaded_group.group_name == "Persistent Family"
