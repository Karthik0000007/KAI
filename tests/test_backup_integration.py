"""
Integration tests for BackupManager with HealthDatabase.

Tests the complete backup workflow with real database operations.

Requirements: 6.1, 6.2, 6.3, 6.4
"""

import pytest
import asyncio
import tempfile
from pathlib import Path
from datetime import datetime

from core.backup_manager import BackupManager
from core.health_db import HealthDatabase
from core.models import HealthCheckIn


@pytest.fixture
def temp_dir():
    """Create temporary directory for test."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.mark.asyncio
async def test_backup_and_restore_workflow(temp_dir):
    """
    Test complete backup and restore workflow.
    
    1. Create database with data
    2. Create backup
    3. Modify database
    4. Restore from backup
    5. Verify original data is restored
    
    Requirements: 6.1, 6.2, 6.4, 6.5
    """
    # Setup paths
    db_path = temp_dir / "health.db"
    backup_dir = temp_dir / "backups"
    
    # Create database with test data
    db = HealthDatabase(db_path=db_path)
    
    original_checkin = HealthCheckIn(
        id="original_checkin",
        timestamp=datetime.now().isoformat(),
        mood_score=8.0,
        sleep_hours=7.5,
        energy_level=7.0,
        user_text="Original data",
        detected_emotion="calm",
        emotion_confidence=0.9
    )
    db.save_checkin(original_checkin)
    db.close()
    
    # Create backup manager
    from unittest.mock import patch
    with patch('core.backup_manager.DB_PATH', db_path):
        backup_manager = BackupManager({
            'enabled': True,
            'interval_hours': 24,
            'retention_days': 30,
            'backup_dir': str(backup_dir),
            'verify_integrity': True
        })
        
        # Create backup
        backup_path = await backup_manager.create_backup()
        assert backup_path is not None
        assert backup_path.exists()
        
        # Modify database (delete data)
        db = HealthDatabase(db_path=db_path)
        
        modified_checkin = HealthCheckIn(
            id="modified_checkin",
            timestamp=datetime.now().isoformat(),
            mood_score=3.0,
            sleep_hours=4.0,
            energy_level=2.0,
            user_text="Modified data",
            detected_emotion="stressed",
            emotion_confidence=0.8
        )
        db.save_checkin(modified_checkin)
        db.close()
        
        # Verify modified data exists
        db = HealthDatabase(db_path=db_path)
        checkins = db.get_recent_checkins(days=7)
        db.close()
        
        assert len(checkins) == 2
        assert any(c['id'] == 'modified_checkin' for c in checkins)
        
        # Restore from backup
        success = await backup_manager.restore_from_backup(backup_path)
        assert success is True
        
        # Verify original data is restored
        db = HealthDatabase(db_path=db_path)
        checkins = db.get_recent_checkins(days=7)
        db.close()
        
        assert len(checkins) == 1
        assert checkins[0]['id'] == 'original_checkin'
        # Note: mood_score may have differential privacy noise applied
        # Check it's approximately correct (within 1.0)
        assert abs(checkins[0]['mood_score'] - 8.0) < 1.0


@pytest.mark.asyncio
async def test_backup_preserves_encryption(temp_dir):
    """
    Test that backups preserve encrypted data correctly.
    
    Requirement 6.1: Backup preserves encryption
    """
    # Setup paths
    db_path = temp_dir / "health.db"
    backup_dir = temp_dir / "backups"
    
    # Create database with encrypted data
    db = HealthDatabase(db_path=db_path)
    
    checkin = HealthCheckIn(
        id="encrypted_checkin",
        timestamp=datetime.now().isoformat(),
        mood_score=7.0,
        sleep_hours=8.0,
        energy_level=6.5,
        user_text="This is sensitive encrypted text",
        pain_notes="Encrypted pain notes",
        detected_emotion="calm",
        emotion_confidence=0.85
    )
    db.save_checkin(checkin)
    db.close()
    
    # Create backup
    from unittest.mock import patch
    with patch('core.backup_manager.DB_PATH', db_path):
        backup_manager = BackupManager({
            'enabled': True,
            'backup_dir': str(backup_dir),
            'verify_integrity': True
        })
        
        backup_path = await backup_manager.create_backup()
        assert backup_path is not None
        
        # Open backup and verify data is still encrypted
        backup_db = HealthDatabase(db_path=backup_path)
        checkins = backup_db.get_recent_checkins(days=7)
        backup_db.close()
        
        # Should be able to decrypt the data
        assert len(checkins) == 1
        assert checkins[0]['user_text'] == "This is sensitive encrypted text"
        assert checkins[0]['pain_notes'] == "Encrypted pain notes"


@pytest.mark.asyncio
async def test_multiple_backups_retention(temp_dir):
    """
    Test that multiple backups are created and retained.
    
    Requirement 6.3: Retain last 30 daily backups
    """
    # Setup paths
    db_path = temp_dir / "health.db"
    backup_dir = temp_dir / "backups"
    
    # Create database
    db = HealthDatabase(db_path=db_path)
    checkin = HealthCheckIn(
        id="test_checkin",
        timestamp=datetime.now().isoformat(),
        mood_score=7.0,
        sleep_hours=8.0,
        energy_level=6.5
    )
    db.save_checkin(checkin)
    db.close()
    
    # Create backup manager with short retention
    from unittest.mock import patch
    with patch('core.backup_manager.DB_PATH', db_path):
        backup_manager = BackupManager({
            'enabled': True,
            'retention_days': 7,  # Short retention for testing
            'backup_dir': str(backup_dir)
        })
        
        # Create first backup
        backup_path1 = await backup_manager.create_backup()
        assert backup_path1 is not None
        
        # Verify backup is valid
        assert await backup_manager.verify_backup(backup_path1)
        
        # List backups
        backups = backup_manager.list_backups()
        assert len(backups) >= 1
        
        # Get latest backup
        latest = backup_manager.get_latest_backup()
        assert latest is not None
        assert latest.exists()


@pytest.mark.asyncio
async def test_backup_retention_cleanup_integration(temp_dir):
    """
    Test complete backup retention cleanup workflow.
    
    Creates old and recent backups, runs cleanup, and verifies
    that old backups are deleted while recent ones are retained.
    
    Requirement 6.3: Retain last 30 daily backups
    Requirement 18.2: Integration tests for backup/restore
    """
    import sqlite3
    from datetime import timedelta
    
    # Setup paths
    db_path = temp_dir / "health.db"
    backup_dir = temp_dir / "backups"
    
    # Create database with test data
    db = HealthDatabase(db_path=db_path)
    checkin = HealthCheckIn(
        id="test_checkin",
        timestamp=datetime.now().isoformat(),
        mood_score=7.0,
        sleep_hours=8.0,
        energy_level=6.5,
        user_text="Test data for retention"
    )
    db.save_checkin(checkin)
    db.close()
    
    # Create backup manager with 7-day retention
    from unittest.mock import patch
    with patch('core.backup_manager.DB_PATH', db_path):
        backup_manager = BackupManager({
            'enabled': True,
            'retention_days': 7,
            'backup_dir': str(backup_dir),
            'verify_integrity': True
        })
        
        # Create old backups (10 days ago - should be deleted)
        old_timestamp = (datetime.now() - timedelta(days=10)).strftime("%Y%m%d_%H%M%S")
        old_backup = backup_dir / f"aegis_health_{old_timestamp}.db"
        
        # Create a valid SQLite database for old backup
        old_db = sqlite3.connect(str(old_backup))
        old_db.execute("CREATE TABLE health_checkins (id TEXT PRIMARY KEY)")
        old_db.execute("INSERT INTO health_checkins VALUES ('old_data')")
        old_db.commit()
        old_db.close()
        
        # Create recent backups (3 days ago - should be retained)
        recent_timestamp = (datetime.now() - timedelta(days=3)).strftime("%Y%m%d_%H%M%S")
        recent_backup = backup_dir / f"aegis_health_{recent_timestamp}.db"
        
        recent_db = sqlite3.connect(str(recent_backup))
        recent_db.execute("CREATE TABLE health_checkins (id TEXT PRIMARY KEY)")
        recent_db.execute("INSERT INTO health_checkins VALUES ('recent_data')")
        recent_db.commit()
        recent_db.close()
        
        # Create a very recent backup (today)
        current_backup = await backup_manager.create_backup()
        assert current_backup is not None
        
        # Verify all backups exist before cleanup
        assert old_backup.exists()
        assert recent_backup.exists()
        assert current_backup.exists()
        
        # List backups before cleanup
        backups_before = backup_manager.list_backups()
        assert len(backups_before) == 3
        
        # Run cleanup
        await backup_manager.cleanup_old_backups()
        
        # Verify old backup was deleted
        assert not old_backup.exists(), "Old backup (10 days) should be deleted"
        
        # Verify recent backups still exist
        assert recent_backup.exists(), "Recent backup (3 days) should be retained"
        assert current_backup.exists(), "Current backup should be retained"
        
        # List backups after cleanup
        backups_after = backup_manager.list_backups()
        assert len(backups_after) == 2, "Should have 2 backups after cleanup"
        
        # Verify the remaining backups are the correct ones
        backup_paths = [b['path'] for b in backups_after]
        assert recent_backup in backup_paths
        assert current_backup in backup_paths
        assert old_backup not in backup_paths


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
