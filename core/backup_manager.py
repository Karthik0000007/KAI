"""
Aegis Backup Manager
Automated database backup and recovery system.

Implements:
- Automated backups every 24 hours
- SQLite backup API for consistent backups
- Backup verification using PRAGMA integrity_check
- Retention management (30 days)
- Graceful error handling

Requirements: 6.1, 6.2, 6.3, 6.4
"""

import sqlite3
import logging
import asyncio
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List

from core.config import DB_PATH

logger = logging.getLogger("aegis.backup_manager")


class BackupManager:
    """
    Automated backup and recovery manager for Aegis health database.
    
    Features:
    - Periodic backups using SQLite backup API
    - Integrity verification after each backup
    - Automatic cleanup of old backups
    - Restore functionality with verification
    
    Requirements:
    - 6.1: Create backup every 24 hours
    - 6.2: Store backups in data/db/backups/ with timestamp
    - 6.3: Retain last 30 daily backups
    - 6.4: Verify backup integrity after creation
    """
    
    def __init__(self, config: Optional[dict] = None):
        """
        Initialize backup manager.
        
        Args:
            config: Optional configuration dictionary with keys:
                - enabled: Enable/disable backups (default: True)
                - interval_hours: Backup interval in hours (default: 24)
                - retention_days: Days to retain backups (default: 30)
                - backup_dir: Backup directory path (default: data/db/backups)
                - verify_integrity: Verify backups after creation (default: True)
        """
        config = config or {}
        
        self.enabled = config.get('enabled', True)
        self.interval_hours = config.get('interval_hours', 24)
        self.retention_days = config.get('retention_days', 30)
        self.verify_integrity = config.get('verify_integrity', True)
        
        # Setup backup directory
        backup_dir_str = config.get('backup_dir', 'data/db/backups')
        self.backup_dir = Path(backup_dir_str)
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
        # Source database path
        self.db_path = Path(DB_PATH)
        
        logger.info(
            f"BackupManager initialized: enabled={self.enabled}, "
            f"interval={self.interval_hours}h, retention={self.retention_days}d"
        )
    
    async def run_backup_loop(self):
        """
        Run periodic backup loop.
        
        Creates backups at configured intervals and cleans up old backups.
        Runs indefinitely until disabled or cancelled.
        
        Requirement 6.1: Create backup every 24 hours
        """
        if not self.enabled:
            logger.info("Backup loop disabled by configuration")
            return
        
        logger.info(f"Starting backup loop (interval: {self.interval_hours}h)")
        
        while self.enabled:
            try:
                # Create backup
                await self.create_backup()
                
                # Cleanup old backups
                await self.cleanup_old_backups()
                
                # Wait for next backup interval
                await asyncio.sleep(self.interval_hours * 3600)
                
            except asyncio.CancelledError:
                logger.info("Backup loop cancelled")
                break
            except Exception as e:
                logger.error(f"Error in backup loop: {e}", exc_info=True)
                # Continue loop even on error - don't crash the system
                await asyncio.sleep(60)  # Wait 1 minute before retrying
    
    async def create_backup(self) -> Optional[Path]:
        """
        Create a database backup using SQLite backup API.
        
        Returns:
            Path to created backup file, or None if backup failed
        
        Requirements:
        - 6.1: Create backup copy
        - 6.2: Store in data/db/backups/ with timestamp
        - 6.4: Verify integrity after creation
        """
        try:
            # Generate backup filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = self.backup_dir / f"aegis_health_{timestamp}.db"
            
            logger.info(f"Creating backup: {backup_path}")
            
            # Check if source database exists
            if not self.db_path.exists():
                logger.error(f"Source database not found: {self.db_path}")
                return None
            
            # Use SQLite backup API for consistent backup
            # This is crash-safe and handles WAL mode correctly
            source_db = None
            backup_db = None
            
            try:
                source_db = sqlite3.connect(str(self.db_path))
                backup_db = sqlite3.connect(str(backup_path))
                
                # Perform backup using SQLite's backup API
                # This copies the database page-by-page while handling concurrent writes
                with backup_db:
                    source_db.backup(backup_db)
                
                logger.info(f"Backup created successfully: {backup_path}")
                
            finally:
                # Ensure connections are closed
                if source_db:
                    source_db.close()
                if backup_db:
                    backup_db.close()
            
            # Verify backup integrity
            if self.verify_integrity:
                if await self.verify_backup(backup_path):
                    logger.info(f"Backup verification passed: {backup_path}")
                else:
                    logger.error(f"Backup verification failed: {backup_path}")
                    # Delete corrupted backup
                    backup_path.unlink()
                    return None
            
            return backup_path
            
        except Exception as e:
            logger.error(f"Failed to create backup: {e}", exc_info=True)
            return None
    
    async def verify_backup(self, backup_path: Path) -> bool:
        """
        Verify backup integrity using SQLite PRAGMA integrity_check.
        
        Args:
            backup_path: Path to backup file to verify
        
        Returns:
            True if backup is valid, False otherwise
        
        Requirement 6.4: Verify backup integrity after creation
        """
        db = None
        try:
            if not backup_path.exists():
                logger.error(f"Backup file not found: {backup_path}")
                return False
            
            # Open backup database and run integrity check
            db = sqlite3.connect(str(backup_path))
            cursor = db.execute("PRAGMA integrity_check")
            result = cursor.fetchone()[0]
            
            # SQLite returns "ok" if database is valid
            is_valid = result == "ok"
            
            if not is_valid:
                logger.error(f"Backup integrity check failed: {result}")
            
            return is_valid
            
        except Exception as e:
            logger.error(f"Error verifying backup: {e}", exc_info=True)
            return False
        finally:
            # Ensure connection is closed to avoid file locks
            if db:
                db.close()
    
    async def cleanup_old_backups(self):
        """
        Remove backups older than retention period.
        
        Requirement 6.3: Retain last 30 daily backups
        """
        try:
            cutoff = datetime.now() - timedelta(days=self.retention_days)
            
            # Find all backup files
            backup_files = list(self.backup_dir.glob("aegis_health_*.db"))
            
            deleted_count = 0
            for backup_file in backup_files:
                try:
                    # Parse timestamp from filename
                    # Format: aegis_health_YYYYMMDD_HHMMSS.db
                    timestamp_str = backup_file.stem.split('_', 2)[2]
                    timestamp = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
                    
                    # Delete if older than retention period
                    if timestamp < cutoff:
                        backup_file.unlink()
                        deleted_count += 1
                        logger.info(f"Removed old backup: {backup_file.name}")
                        
                except (ValueError, IndexError) as e:
                    # Skip files with invalid filename format
                    logger.warning(f"Skipping file with invalid format: {backup_file.name}")
                    continue
            
            if deleted_count > 0:
                logger.info(f"Cleaned up {deleted_count} old backup(s)")
            
        except Exception as e:
            logger.error(f"Error cleaning up old backups: {e}", exc_info=True)
    
    async def verify_encryption_key(self, backup_path: Path) -> bool:
        """
        Verify that the backup was encrypted with the current encryption key.
        
        Attempts to decrypt a sample encrypted field from the backup database.
        If decryption succeeds, the encryption key matches.
        
        Args:
            backup_path: Path to backup file to verify
        
        Returns:
            True if encryption key matches, False otherwise
        
        Requirement 6.6: Verify encryption key matches before restore
        """
        db = None
        try:
            # Import encryption utilities
            from core.encryption import decrypt_string, get_fernet
            
            # Open backup database
            db = sqlite3.connect(str(backup_path))
            
            # Try to find an encrypted field to test
            # Check health_checkins table for user_text field
            cursor = db.execute("""
                SELECT user_text FROM health_checkins 
                WHERE user_text IS NOT NULL 
                LIMIT 1
            """)
            row = cursor.fetchone()
            
            if row and row[0]:
                # Try to decrypt the field
                try:
                    fernet = get_fernet()
                    decrypt_string(row[0], fernet)
                    logger.info("Encryption key verification passed")
                    return True
                except Exception as e:
                    logger.error(f"Encryption key verification failed: {e}")
                    return False
            
            # If no encrypted data found, check conversation_history
            cursor = db.execute("""
                SELECT content FROM conversation_history 
                WHERE content IS NOT NULL 
                LIMIT 1
            """)
            row = cursor.fetchone()
            
            if row and row[0]:
                # Try to decrypt the field
                try:
                    fernet = get_fernet()
                    decrypt_string(row[0], fernet)
                    logger.info("Encryption key verification passed")
                    return True
                except Exception as e:
                    logger.error(f"Encryption key verification failed: {e}")
                    return False
            
            # If no encrypted data found in backup, assume key matches
            # (backup might be empty or from before encryption was enabled)
            logger.warning("No encrypted data found in backup to verify key, assuming key matches")
            return True
            
        except Exception as e:
            logger.error(f"Error verifying encryption key: {e}", exc_info=True)
            return False
        finally:
            if db:
                db.close()
    
    async def restore_from_backup(self, backup_path: Path) -> bool:
        """
        Restore database from a backup file.
        
        Creates a safety backup of current database before restoring.
        Verifies backup integrity before restoration.
        Verifies encryption key matches before restoration.
        
        Args:
            backup_path: Path to backup file to restore from
        
        Returns:
            True if restore succeeded, False otherwise
        
        Requirements:
        - 6.5: Provide command-line option to restore from backup
        - 6.6: Verify encryption key matches before restore
        """
        try:
            # Validate backup file exists
            if not backup_path.exists():
                logger.error(f"Backup file not found: {backup_path}")
                return False
            
            # Verify backup integrity before restoring
            logger.info(f"Verifying backup before restore: {backup_path}")
            if not await self.verify_backup(backup_path):
                logger.error(f"Backup verification failed, cannot restore: {backup_path}")
                return False
            
            # Verify encryption key matches
            logger.info(f"Verifying encryption key matches: {backup_path}")
            if not await self.verify_encryption_key(backup_path):
                logger.error(f"Encryption key verification failed, cannot restore: {backup_path}")
                logger.error("The backup was encrypted with a different key than the current one")
                return False
            
            # Create safety backup of current database
            if self.db_path.exists():
                safety_backup = self.backup_dir / f"pre_restore_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db"
                logger.info(f"Creating safety backup: {safety_backup}")
                shutil.copy(str(self.db_path), str(safety_backup))
            
            # Restore from backup
            logger.info(f"Restoring database from: {backup_path}")
            shutil.copy(str(backup_path), str(self.db_path))
            
            # Verify restored database
            if await self.verify_backup(self.db_path):
                logger.info(f"Database restored successfully from: {backup_path}")
                return True
            else:
                logger.error("Restored database failed verification")
                return False
            
        except Exception as e:
            logger.error(f"Error restoring from backup: {e}", exc_info=True)
            return False
    
    def list_backups(self) -> List[dict]:
        """
        List all available backups with metadata.
        
        Returns:
            List of dictionaries with backup information:
            - path: Path to backup file
            - timestamp: Backup creation timestamp
            - size: File size in bytes
            - age_days: Age in days
        """
        backups = []
        
        try:
            backup_files = sorted(self.backup_dir.glob("aegis_health_*.db"))
            
            for backup_file in backup_files:
                try:
                    # Parse timestamp from filename
                    timestamp_str = backup_file.stem.split('_', 2)[2]
                    timestamp = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
                    
                    # Calculate age
                    age = datetime.now() - timestamp
                    
                    backups.append({
                        'path': backup_file,
                        'timestamp': timestamp,
                        'size': backup_file.stat().st_size,
                        'age_days': age.days
                    })
                    
                except (ValueError, IndexError):
                    # Skip files with invalid format
                    continue
            
        except Exception as e:
            logger.error(f"Error listing backups: {e}", exc_info=True)
        
        return backups
    
    def get_latest_backup(self) -> Optional[Path]:
        """
        Get the most recent backup file.
        
        Returns:
            Path to latest backup, or None if no backups exist
        """
        backups = self.list_backups()
        
        if not backups:
            return None
        
        # Sort by timestamp (most recent first)
        backups.sort(key=lambda b: b['timestamp'], reverse=True)
        
        return backups[0]['path']
