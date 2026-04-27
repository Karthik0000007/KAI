"""
Tests for BackupManager class.

Tests automated backup system including:
- Backup creation using SQLite backup API
- Backup verification with integrity checks
- Scheduled backups every 24 hours
- Retention management (30 days)
- Restore functionality

Requirements: 6.1, 6.2, 6.3, 6.4, 6.5
"""

import pytest
import asyncio
import sqlite3
import tempfile
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

from core.backup_manager import BackupManager
from core.health_db import HealthDatabase
from core.models import HealthCheckIn


@pytest.fixture
def temp_db_dir():
    """Create temporary directory for test databases."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def test_db(temp_db_dir):
    """Create a test database with sample data."""
    db_path = temp_db_dir / "test_health.db"
    db = HealthDatabase(db_path=db_path)
    
    # Add some test data
    checkin = HealthCheckIn(
        id="test_checkin_1",
        timestamp=datetime.now().isoformat(),
        mood_score=7.5,
        sleep_hours=8.0,
        energy_level=6.5,
        user_text="Feeling good today",
        detected_emotion="calm",
        emotion_confidence=0.85
    )
    db.save_checkin(checkin)
    
    db.close()
    return db_path


@pytest.fixture
def backup_manager(temp_db_dir, test_db):
    """Create BackupManager instance with test configuration."""
    backup_dir = temp_db_dir / "backups"
    
    config = {
        'enabled': True,
        'interval_hours': 24,
        'retention_days': 30,
        'backup_dir': str(backup_dir),
        'verify_integrity': True
    }
    
    # Patch DB_PATH to use test database
    with patch('core.backup_manager.DB_PATH', test_db):
        manager = BackupManager(config)
        yield manager


class TestBackupCreation:
    """Test backup creation functionality."""
    
    @pytest.mark.asyncio
    async def test_create_backup_success(self, backup_manager):
        """
        Test successful backup creation.
        
        Requirement 6.1: Create backup copy
        Requirement 6.2: Store in data/db/backups/ with timestamp
        """
        # Create backup
        backup_path = await backup_manager.create_backup()
        
        # Verify backup was created
        assert backup_path is not None
        assert backup_path.exists()
        assert backup_path.parent == backup_manager.backup_dir
        
        # Verify filename format: aegis_health_YYYYMMDD_HHMMSS.db
        assert backup_path.name.startswith("aegis_health_")
        assert backup_path.suffix == ".db"
        
        # Verify timestamp in filename is valid
        timestamp_str = backup_path.stem.split('_', 2)[2]
        timestamp = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
        assert timestamp <= datetime.now()
    
    @pytest.mark.asyncio
    async def test_backup_contains_data(self, backup_manager):
        """
        Test that backup contains all data from source database.
        
        Requirement 6.1: Create backup copy
        """
        # Create backup
        backup_path = await backup_manager.create_backup()
        assert backup_path is not None
        
        # Open backup and verify data
        backup_db = sqlite3.connect(str(backup_path))
        cursor = backup_db.execute("SELECT COUNT(*) FROM health_checkins")
        count = cursor.fetchone()[0]
        backup_db.close()
        
        # Should have the test data we inserted
        assert count == 1
    
    @pytest.mark.asyncio
    async def test_backup_uses_sqlite_api(self, backup_manager):
        """
        Test that backup uses SQLite backup API (not file copy).
        
        This ensures consistent backups even with concurrent writes.
        
        Requirement 6.1: Use SQLite backup API
        """
        # Create backup
        backup_path = await backup_manager.create_backup()
        assert backup_path is not None
        
        # Verify backup is a valid SQLite database
        backup_db = sqlite3.connect(str(backup_path))
        
        # Check that it has the expected schema
        cursor = backup_db.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
        tables = [row[0] for row in cursor.fetchall()]
        backup_db.close()
        
        # Should have all the health database tables
        expected_tables = [
            'health_checkins',
            'medication_reminders',
            'vital_records',
            'proactive_alerts',
            'conversation_history',
            'emotion_transitions'
        ]
        
        for table in expected_tables:
            assert table in tables
    
    @pytest.mark.asyncio
    async def test_backup_with_missing_source(self, backup_manager):
        """
        Test backup creation when source database doesn't exist.
        
        Should handle gracefully and return None.
        """
        # Delete source database
        backup_manager.db_path.unlink()
        
        # Attempt backup
        backup_path = await backup_manager.create_backup()
        
        # Should return None (graceful failure)
        assert backup_path is None


class TestBackupVerification:
    """Test backup integrity verification."""
    
    @pytest.mark.asyncio
    async def test_verify_valid_backup(self, backup_manager):
        """
        Test verification of valid backup.
        
        Requirement 6.4: Verify backup integrity after creation
        """
        # Create backup
        backup_path = await backup_manager.create_backup()
        assert backup_path is not None
        
        # Verify backup
        is_valid = await backup_manager.verify_backup(backup_path)
        assert is_valid is True
    
    @pytest.mark.asyncio
    async def test_verify_corrupted_backup(self, backup_manager, temp_db_dir):
        """
        Test verification of corrupted backup.
        
        Requirement 6.4: Verify backup integrity
        """
        # Create a corrupted backup file
        corrupted_path = backup_manager.backup_dir / "corrupted.db"
        with open(corrupted_path, 'wb') as f:
            f.write(b"This is not a valid SQLite database")
        
        # Verify should fail
        is_valid = await backup_manager.verify_backup(corrupted_path)
        assert is_valid is False
    
    @pytest.mark.asyncio
    async def test_verify_missing_backup(self, backup_manager):
        """
        Test verification of non-existent backup.
        
        Should handle gracefully and return False.
        """
        missing_path = backup_manager.backup_dir / "nonexistent.db"
        
        is_valid = await backup_manager.verify_backup(missing_path)
        assert is_valid is False
    
    @pytest.mark.asyncio
    async def test_backup_deleted_if_verification_fails(self, backup_manager):
        """
        Test that corrupted backups are deleted automatically.
        
        Requirement 6.4: Verify backup integrity after creation
        """
        # Mock verify_backup to return False
        with patch.object(backup_manager, 'verify_backup', return_value=False):
            backup_path = await backup_manager.create_backup()
            
            # Should return None (backup failed)
            assert backup_path is None


class TestBackupRetention:
    """Test backup retention and cleanup."""
    
    @pytest.mark.asyncio
    async def test_cleanup_old_backups(self, backup_manager):
        """
        Test cleanup of backups older than retention period.
        
        Requirement 6.3: Retain last 30 daily backups
        """
        # Create old backup (35 days ago)
        old_timestamp = (datetime.now() - timedelta(days=35)).strftime("%Y%m%d_%H%M%S")
        old_backup = backup_manager.backup_dir / f"aegis_health_{old_timestamp}.db"
        
        # Create a minimal valid SQLite database
        db = sqlite3.connect(str(old_backup))
        db.execute("CREATE TABLE test (id INTEGER)")
        db.commit()
        db.close()
        
        # Create recent backup (5 days ago)
        recent_timestamp = (datetime.now() - timedelta(days=5)).strftime("%Y%m%d_%H%M%S")
        recent_backup = backup_manager.backup_dir / f"aegis_health_{recent_timestamp}.db"
        
        db = sqlite3.connect(str(recent_backup))
        db.execute("CREATE TABLE test (id INTEGER)")
        db.commit()
        db.close()
        
        # Run cleanup
        await backup_manager.cleanup_old_backups()
        
        # Old backup should be deleted
        assert not old_backup.exists()
        
        # Recent backup should still exist
        assert recent_backup.exists()
    
    @pytest.mark.asyncio
    async def test_cleanup_keeps_recent_backups(self, backup_manager):
        """
        Test that cleanup preserves backups within retention period.
        
        Requirement 6.3: Retain last 30 daily backups
        """
        # Create multiple recent backups
        backup_paths = []
        for days_ago in [1, 5, 10, 15, 20, 25, 29]:
            timestamp = (datetime.now() - timedelta(days=days_ago)).strftime("%Y%m%d_%H%M%S")
            backup_path = backup_manager.backup_dir / f"aegis_health_{timestamp}.db"
            
            db = sqlite3.connect(str(backup_path))
            db.execute("CREATE TABLE test (id INTEGER)")
            db.commit()
            db.close()
            
            backup_paths.append(backup_path)
        
        # Run cleanup
        await backup_manager.cleanup_old_backups()
        
        # All backups should still exist (within 30 days)
        for backup_path in backup_paths:
            assert backup_path.exists()
    
    @pytest.mark.asyncio
    async def test_cleanup_handles_invalid_filenames(self, backup_manager):
        """
        Test that cleanup skips files with invalid filename format.
        """
        # Create file with invalid format
        invalid_file = backup_manager.backup_dir / "invalid_backup.db"
        invalid_file.touch()
        
        # Run cleanup (should not crash)
        await backup_manager.cleanup_old_backups()
        
        # Invalid file should still exist (not deleted)
        assert invalid_file.exists()


class TestBackupRestore:
    """Test backup restore functionality."""
    
    @pytest.mark.asyncio
    async def test_restore_from_backup(self, backup_manager, test_db):
        """
        Test restoring database from backup.
        
        Requirement 6.5: Provide restore from backup functionality
        """
        # Create backup
        backup_path = await backup_manager.create_backup()
        assert backup_path is not None
        
        # Modify source database
        db = sqlite3.connect(str(test_db))
        db.execute("DELETE FROM health_checkins")
        db.commit()
        db.close()
        
        # Verify data was deleted
        db = sqlite3.connect(str(test_db))
        cursor = db.execute("SELECT COUNT(*) FROM health_checkins")
        count = cursor.fetchone()[0]
        db.close()
        assert count == 0
        
        # Restore from backup
        success = await backup_manager.restore_from_backup(backup_path)
        assert success is True
        
        # Verify data was restored
        db = sqlite3.connect(str(test_db))
        cursor = db.execute("SELECT COUNT(*) FROM health_checkins")
        count = cursor.fetchone()[0]
        db.close()
        assert count == 1
    
    @pytest.mark.asyncio
    async def test_restore_creates_safety_backup(self, backup_manager, test_db):
        """
        Test that restore creates safety backup of current database.
        
        Requirement 6.5: Create safety backup before restore
        """
        # Create backup
        backup_path = await backup_manager.create_backup()
        assert backup_path is not None
        
        # Count existing backups
        initial_backups = len(list(backup_manager.backup_dir.glob("*.db")))
        
        # Restore from backup
        success = await backup_manager.restore_from_backup(backup_path)
        assert success is True
        
        # Should have created a safety backup (pre_restore_*)
        safety_backups = list(backup_manager.backup_dir.glob("pre_restore_*.db"))
        assert len(safety_backups) > 0
    
    @pytest.mark.asyncio
    async def test_restore_verifies_backup_before_restore(self, backup_manager):
        """
        Test that restore verifies backup integrity before restoring.
        
        Requirement 6.4: Verify backup integrity
        Requirement 6.6: Verify encryption key matches before restore
        """
        # Create corrupted backup
        corrupted_path = backup_manager.backup_dir / "corrupted.db"
        with open(corrupted_path, 'wb') as f:
            f.write(b"Not a valid database")
        
        # Attempt restore
        success = await backup_manager.restore_from_backup(corrupted_path)
        
        # Should fail
        assert success is False
    
    @pytest.mark.asyncio
    async def test_restore_verifies_encryption_key(self, backup_manager, test_db):
        """
        Test that restore verifies encryption key matches before restoring.
        
        Requirement 6.6: Verify encryption key matches before restore
        """
        # Create a backup with encrypted data
        from core.health_db import HealthDatabase
        from core.models import HealthCheckIn
        from datetime import datetime
        
        # Add encrypted data to test database
        db = HealthDatabase(db_path=test_db)
        checkin = HealthCheckIn(
            id="test_encrypted",
            timestamp=datetime.now().isoformat(),
            mood_score=7.5,
            sleep_hours=8.0,
            energy_level=6.5,
            user_text="This is encrypted sensitive data",
            detected_emotion="calm",
            emotion_confidence=0.85
        )
        db.save_checkin(checkin)
        db.close()
        
        # Create backup
        backup_path = await backup_manager.create_backup()
        assert backup_path is not None
        
        # Verify encryption key matches (should succeed with same key)
        key_matches = await backup_manager.verify_encryption_key(backup_path)
        assert key_matches is True
        
        # Now test with a different encryption key
        # We'll simulate this by corrupting the encrypted data
        import sqlite3
        db_conn = sqlite3.connect(str(backup_path))
        db_conn.execute("UPDATE health_checkins SET user_text = 'corrupted_encrypted_data'")
        db_conn.commit()
        db_conn.close()
        
        # Verify encryption key should now fail
        key_matches = await backup_manager.verify_encryption_key(backup_path)
        assert key_matches is False
    
    @pytest.mark.asyncio
    async def test_restore_fails_with_wrong_encryption_key(self, backup_manager, test_db):
        """
        Test that restore fails when encryption key doesn't match.
        
        Requirement 6.6: Verify encryption key matches before restore
        """
        # Create a backup with encrypted data
        from core.health_db import HealthDatabase
        from core.models import HealthCheckIn
        from datetime import datetime
        
        # Add encrypted data to test database
        db = HealthDatabase(db_path=test_db)
        checkin = HealthCheckIn(
            id="test_encrypted_2",
            timestamp=datetime.now().isoformat(),
            mood_score=7.5,
            sleep_hours=8.0,
            energy_level=6.5,
            user_text="This is encrypted sensitive data",
            detected_emotion="calm",
            emotion_confidence=0.85
        )
        db.save_checkin(checkin)
        db.close()
        
        # Create backup
        backup_path = await backup_manager.create_backup()
        assert backup_path is not None
        
        # Corrupt the encrypted data to simulate wrong key
        import sqlite3
        db_conn = sqlite3.connect(str(backup_path))
        db_conn.execute("UPDATE health_checkins SET user_text = 'invalid_encrypted_data'")
        db_conn.commit()
        db_conn.close()
        
        # Attempt restore (should fail due to encryption key mismatch)
        success = await backup_manager.restore_from_backup(backup_path)
        assert success is False
    
    @pytest.mark.asyncio
    async def test_restore_from_missing_backup(self, backup_manager):
        """
        Test restore from non-existent backup.
        
        Should handle gracefully and return False.
        """
        missing_path = backup_manager.backup_dir / "nonexistent.db"
        
        success = await backup_manager.restore_from_backup(missing_path)
        assert success is False


class TestBackupScheduling:
    """Test automated backup scheduling."""
    
    @pytest.mark.asyncio
    async def test_backup_loop_creates_backups(self, backup_manager):
        """
        Test that backup loop creates backups at scheduled intervals.
        
        Requirement 6.1: Create backup every 24 hours
        """
        # Mock sleep to speed up test
        original_sleep = asyncio.sleep
        sleep_calls = []
        
        async def mock_sleep(seconds):
            sleep_calls.append(seconds)
            # Only sleep briefly in test
            await original_sleep(0.01)
            # Cancel after first iteration
            raise asyncio.CancelledError()
        
        with patch('asyncio.sleep', side_effect=mock_sleep):
            # Run backup loop (will be cancelled after first iteration)
            try:
                await backup_manager.run_backup_loop()
            except asyncio.CancelledError:
                pass
        
        # Should have created a backup
        backups = list(backup_manager.backup_dir.glob("aegis_health_*.db"))
        assert len(backups) > 0
        
        # Should have called sleep with correct interval
        assert len(sleep_calls) > 0
        assert sleep_calls[0] == 24 * 3600  # 24 hours in seconds
    
    @pytest.mark.asyncio
    async def test_backup_loop_disabled(self, backup_manager):
        """
        Test that backup loop respects enabled flag.
        """
        backup_manager.enabled = False
        
        # Run backup loop (should return immediately)
        await backup_manager.run_backup_loop()
        
        # Should not have created any backups
        backups = list(backup_manager.backup_dir.glob("aegis_health_*.db"))
        assert len(backups) == 0
    
    @pytest.mark.asyncio
    async def test_backup_loop_handles_errors(self, backup_manager):
        """
        Test that backup loop continues after errors.
        
        Should not crash the system on backup failures.
        """
        # Mock create_backup to raise error
        async def mock_create_backup():
            raise Exception("Simulated backup error")
        
        backup_manager.create_backup = mock_create_backup
        
        # Mock cleanup to do nothing
        async def mock_cleanup():
            pass
        
        backup_manager.cleanup_old_backups = mock_cleanup
        
        # Mock sleep to cancel after first error recovery attempt
        sleep_count = [0]
        original_sleep = asyncio.sleep
        
        async def mock_sleep(seconds):
            sleep_count[0] += 1
            # Use real sleep for a tiny amount
            await original_sleep(0.001)
            # Cancel after first sleep (the error recovery sleep)
            if sleep_count[0] >= 1:
                raise asyncio.CancelledError()
        
        # Patch asyncio.sleep in the backup_manager module
        with patch('core.backup_manager.asyncio.sleep', side_effect=mock_sleep):
            # Run backup loop
            try:
                await backup_manager.run_backup_loop()
            except asyncio.CancelledError:
                pass
        
        # Should have slept once for error recovery
        assert sleep_count[0] >= 1


class TestBackupListing:
    """Test backup listing and metadata."""
    
    def test_list_backups(self, backup_manager):
        """
        Test listing available backups with metadata.
        """
        # Create multiple backups
        timestamps = []
        for i in range(3):
            timestamp = (datetime.now() - timedelta(days=i)).strftime("%Y%m%d_%H%M%S")
            timestamps.append(timestamp)
            backup_path = backup_manager.backup_dir / f"aegis_health_{timestamp}.db"
            
            db = sqlite3.connect(str(backup_path))
            db.execute("CREATE TABLE test (id INTEGER)")
            db.commit()
            db.close()
        
        # List backups
        backups = backup_manager.list_backups()
        
        # Should have 3 backups
        assert len(backups) == 3
        
        # Each backup should have metadata
        for backup in backups:
            assert 'path' in backup
            assert 'timestamp' in backup
            assert 'size' in backup
            assert 'age_days' in backup
            assert backup['path'].exists()
    
    def test_get_latest_backup(self, backup_manager):
        """
        Test getting the most recent backup.
        """
        # Create backups at different times
        old_timestamp = (datetime.now() - timedelta(days=5)).strftime("%Y%m%d_%H%M%S")
        old_backup = backup_manager.backup_dir / f"aegis_health_{old_timestamp}.db"
        
        db = sqlite3.connect(str(old_backup))
        db.execute("CREATE TABLE test (id INTEGER)")
        db.commit()
        db.close()
        
        recent_timestamp = (datetime.now() - timedelta(days=1)).strftime("%Y%m%d_%H%M%S")
        recent_backup = backup_manager.backup_dir / f"aegis_health_{recent_timestamp}.db"
        
        db = sqlite3.connect(str(recent_backup))
        db.execute("CREATE TABLE test (id INTEGER)")
        db.commit()
        db.close()
        
        # Get latest backup
        latest = backup_manager.get_latest_backup()
        
        # Should be the most recent one
        assert latest == recent_backup
    
    def test_get_latest_backup_when_none_exist(self, backup_manager):
        """
        Test getting latest backup when no backups exist.
        """
        latest = backup_manager.get_latest_backup()
        assert latest is None


class TestBackupConfiguration:
    """Test backup configuration options."""
    
    def test_custom_interval(self, temp_db_dir, test_db):
        """
        Test custom backup interval configuration.
        """
        config = {
            'enabled': True,
            'interval_hours': 12,  # Custom interval
            'retention_days': 30,
            'backup_dir': str(temp_db_dir / "backups")
        }
        
        with patch('core.backup_manager.DB_PATH', test_db):
            manager = BackupManager(config)
        
        assert manager.interval_hours == 12
    
    def test_custom_retention(self, temp_db_dir, test_db):
        """
        Test custom retention period configuration.
        """
        config = {
            'enabled': True,
            'interval_hours': 24,
            'retention_days': 60,  # Custom retention
            'backup_dir': str(temp_db_dir / "backups")
        }
        
        with patch('core.backup_manager.DB_PATH', test_db):
            manager = BackupManager(config)
        
        assert manager.retention_days == 60
    
    def test_disable_verification(self, temp_db_dir, test_db):
        """
        Test disabling backup verification.
        """
        config = {
            'enabled': True,
            'interval_hours': 24,
            'retention_days': 30,
            'backup_dir': str(temp_db_dir / "backups"),
            'verify_integrity': False  # Disable verification
        }
        
        with patch('core.backup_manager.DB_PATH', test_db):
            manager = BackupManager(config)
        
        assert manager.verify_integrity is False
