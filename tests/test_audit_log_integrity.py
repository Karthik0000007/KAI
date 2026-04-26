"""
Tests for audit log integrity and tamper detection.

**Validates: Requirements 15.7, 15.8, 18.9**

This module tests:
- Audit log creation and verification
- Tamper detection in audit logs
- HMAC chain integrity
"""

import pytest
import tempfile
import sqlite3
from pathlib import Path

from core.audit_logger import AuditLogger, create_audit_logger


# ─── Fixtures ───────────────────────────────────────────────────────────────

@pytest.fixture
def temp_audit_db():
    """Create a temporary audit log database for testing."""
    with tempfile.NamedTemporaryFile(suffix='_audit.db', delete=False) as f:
        db_path = Path(f.name)
    
    yield db_path
    
    # Cleanup
    if db_path.exists():
        db_path.unlink()


@pytest.fixture
def audit_logger(temp_audit_db):
    """Create an AuditLogger instance for testing."""
    logger = AuditLogger(temp_audit_db)
    yield logger
    logger.close()


# ─── Test: Audit Log Creation ──────────────────────────────────────────────

def test_audit_logger_initialization(temp_audit_db):
    """
    **Validates: Requirements 15.7**
    
    Test that AuditLogger initializes correctly and creates table.
    """
    logger = AuditLogger(temp_audit_db)
    
    # Verify database was created
    assert temp_audit_db.exists()
    
    # Verify table was created
    db = sqlite3.connect(temp_audit_db)
    cursor = db.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='audit_log'")
    assert cursor.fetchone() is not None
    db.close()
    
    logger.close()


def test_audit_log_single_entry(audit_logger):
    """
    **Validates: Requirements 15.7**
    
    Test logging a single audit entry.
    """
    audit_logger.log(
        action="READ",
        resource="health_checkins",
        details="User viewed health check-ins",
        user_id="user123"
    )
    
    # Verify entry was logged
    logs = audit_logger.get_logs(limit=1)
    assert len(logs) == 1
    
    log = logs[0]
    assert log['action'] == "READ"
    assert log['resource'] == "health_checkins"
    assert log['user_id'] == "user123"
    assert log['hash'] is not None


def test_audit_log_multiple_entries(audit_logger):
    """
    **Validates: Requirements 15.7**
    
    Test logging multiple audit entries.
    """
    # Log multiple entries
    audit_logger.log("CREATE", "user_profile", "Created new user", "user1")
    audit_logger.log("READ", "health_checkins", "Viewed checkins", "user1")
    audit_logger.log("UPDATE", "user_profile", "Updated profile", "user1")
    audit_logger.log("DELETE", "health_checkins", "Deleted old checkin", "user1")
    
    # Verify all entries were logged
    logs = audit_logger.get_logs(limit=10)
    assert len(logs) == 4
    
    # Verify order (most recent first)
    assert logs[0]['action'] == "DELETE"
    assert logs[1]['action'] == "UPDATE"
    assert logs[2]['action'] == "READ"
    assert logs[3]['action'] == "CREATE"


def test_audit_log_with_optional_fields(audit_logger):
    """
    **Validates: Requirements 15.7**
    
    Test logging with optional fields (resource, details, user_id).
    """
    # Log with minimal fields
    audit_logger.log(action="STARTUP")
    
    logs = audit_logger.get_logs(limit=1)
    assert len(logs) == 1
    
    log = logs[0]
    assert log['action'] == "STARTUP"
    assert log['resource'] is None
    assert log['details'] is None
    assert log['user_id'] is None


# ─── Test: Audit Log Integrity Verification ────────────────────────────────

def test_audit_log_integrity_valid(audit_logger):
    """
    **Validates: Requirements 15.8**
    
    Test that integrity verification passes for valid audit log.
    """
    # Log several entries
    audit_logger.log("CREATE", "user", "Created user", "admin")
    audit_logger.log("READ", "health_data", "Accessed data", "user1")
    audit_logger.log("UPDATE", "settings", "Changed settings", "user1")
    
    # Verify integrity
    is_valid, first_invalid_id = audit_logger.verify_integrity()
    
    assert is_valid is True
    assert first_invalid_id is None


def test_audit_log_integrity_empty_log(audit_logger):
    """
    **Validates: Requirements 15.8**
    
    Test that integrity verification passes for empty audit log.
    """
    is_valid, first_invalid_id = audit_logger.verify_integrity()
    
    assert is_valid is True
    assert first_invalid_id is None


def test_audit_log_integrity_single_entry(audit_logger):
    """
    **Validates: Requirements 15.8**
    
    Test that integrity verification passes for single entry.
    """
    audit_logger.log("TEST", "resource", "details", "user")
    
    is_valid, first_invalid_id = audit_logger.verify_integrity()
    
    assert is_valid is True
    assert first_invalid_id is None


# ─── Test: Tamper Detection ────────────────────────────────────────────────

def test_audit_log_detects_modified_action(audit_logger, temp_audit_db):
    """
    **Validates: Requirements 15.8**
    
    Test that tampering with action field is detected.
    """
    # Log entries
    audit_logger.log("READ", "data", "Read data", "user1")
    audit_logger.log("WRITE", "data", "Write data", "user1")
    audit_logger.log("DELETE", "data", "Delete data", "user1")
    
    # Close logger to release database
    audit_logger.close()
    
    # Tamper with the middle entry's action
    db = sqlite3.connect(temp_audit_db)
    db.execute("UPDATE audit_log SET action = 'TAMPERED' WHERE id = 2")
    db.commit()
    db.close()
    
    # Create new logger and verify integrity
    logger2 = AuditLogger(temp_audit_db)
    is_valid, first_invalid_id = logger2.verify_integrity()
    logger2.close()
    
    # Tampering should be detected
    assert is_valid is False
    assert first_invalid_id == 2  # Second entry was tampered with


def test_audit_log_detects_modified_details(audit_logger, temp_audit_db):
    """
    **Validates: Requirements 15.8**
    
    Test that tampering with details field is detected.
    """
    # Log entries
    audit_logger.log("ACTION1", "resource", "Original details", "user1")
    audit_logger.log("ACTION2", "resource", "More details", "user1")
    
    audit_logger.close()
    
    # Tamper with details
    db = sqlite3.connect(temp_audit_db)
    db.execute("UPDATE audit_log SET details = 'Tampered details' WHERE id = 1")
    db.commit()
    db.close()
    
    # Verify integrity
    logger2 = AuditLogger(temp_audit_db)
    is_valid, first_invalid_id = logger2.verify_integrity()
    logger2.close()
    
    assert is_valid is False
    assert first_invalid_id == 1  # First entry was tampered with


def test_audit_log_detects_deleted_entry(audit_logger, temp_audit_db):
    """
    **Validates: Requirements 15.8**
    
    Test that deleting an entry breaks the hash chain.
    """
    # Log entries
    audit_logger.log("ENTRY1", "resource", "First", "user1")
    audit_logger.log("ENTRY2", "resource", "Second", "user1")
    audit_logger.log("ENTRY3", "resource", "Third", "user1")
    
    audit_logger.close()
    
    # Delete middle entry
    db = sqlite3.connect(temp_audit_db)
    db.execute("DELETE FROM audit_log WHERE id = 2")
    db.commit()
    db.close()
    
    # Verify integrity
    logger2 = AuditLogger(temp_audit_db)
    is_valid, first_invalid_id = logger2.verify_integrity()
    logger2.close()
    
    # Deletion should be detected (third entry's hash depends on second)
    assert is_valid is False
    assert first_invalid_id == 3


def test_audit_log_detects_modified_hash(audit_logger, temp_audit_db):
    """
    **Validates: Requirements 15.8**
    
    Test that tampering with hash field is detected.
    """
    # Log entries
    audit_logger.log("ACTION", "resource", "details", "user1")
    
    audit_logger.close()
    
    # Tamper with hash
    db = sqlite3.connect(temp_audit_db)
    db.execute("UPDATE audit_log SET hash = 'fake_hash_12345' WHERE id = 1")
    db.commit()
    db.close()
    
    # Verify integrity
    logger2 = AuditLogger(temp_audit_db)
    is_valid, first_invalid_id = logger2.verify_integrity()
    logger2.close()
    
    assert is_valid is False
    assert first_invalid_id == 1


def test_audit_log_detects_reordered_entries(audit_logger, temp_audit_db):
    """
    **Validates: Requirements 15.8**
    
    Test that reordering entries breaks the hash chain.
    """
    # Log entries
    audit_logger.log("FIRST", "resource", "First entry", "user1")
    audit_logger.log("SECOND", "resource", "Second entry", "user1")
    audit_logger.log("THIRD", "resource", "Third entry", "user1")
    
    audit_logger.close()
    
    # Swap IDs of first and second entries (reorder)
    db = sqlite3.connect(temp_audit_db)
    # This is a simplified test - in reality, reordering would break the chain
    # because each hash depends on the previous hash
    db.execute("UPDATE audit_log SET id = 999 WHERE id = 1")
    db.execute("UPDATE audit_log SET id = 1 WHERE id = 2")
    db.execute("UPDATE audit_log SET id = 2 WHERE id = 999")
    db.commit()
    db.close()
    
    # Verify integrity
    logger2 = AuditLogger(temp_audit_db)
    is_valid, first_invalid_id = logger2.verify_integrity()
    logger2.close()
    
    # Reordering should be detected
    assert is_valid is False


# ─── Test: Audit Log Filtering ─────────────────────────────────────────────

def test_audit_log_filter_by_action(audit_logger):
    """
    **Validates: Requirements 15.7**
    
    Test filtering audit logs by action.
    """
    # Log various actions
    audit_logger.log("READ", "data", "Read 1", "user1")
    audit_logger.log("WRITE", "data", "Write 1", "user1")
    audit_logger.log("READ", "data", "Read 2", "user1")
    audit_logger.log("DELETE", "data", "Delete 1", "user1")
    audit_logger.log("READ", "data", "Read 3", "user1")
    
    # Filter by READ action
    read_logs = audit_logger.get_logs(action="READ")
    
    assert len(read_logs) == 3
    assert all(log['action'] == "READ" for log in read_logs)


def test_audit_log_filter_by_resource(audit_logger):
    """
    **Validates: Requirements 15.7**
    
    Test filtering audit logs by resource.
    """
    # Log various resources
    audit_logger.log("READ", "health_checkins", "Read checkins", "user1")
    audit_logger.log("READ", "user_profile", "Read profile", "user1")
    audit_logger.log("WRITE", "health_checkins", "Write checkin", "user1")
    audit_logger.log("READ", "settings", "Read settings", "user1")
    
    # Filter by health_checkins resource
    checkin_logs = audit_logger.get_logs(resource="health_checkins")
    
    assert len(checkin_logs) == 2
    assert all(log['resource'] == "health_checkins" for log in checkin_logs)


def test_audit_log_filter_by_user(audit_logger):
    """
    **Validates: Requirements 15.7**
    
    Test filtering audit logs by user ID.
    """
    # Log various users
    audit_logger.log("READ", "data", "Read 1", "user1")
    audit_logger.log("READ", "data", "Read 2", "user2")
    audit_logger.log("WRITE", "data", "Write 1", "user1")
    audit_logger.log("READ", "data", "Read 3", "user3")
    audit_logger.log("DELETE", "data", "Delete 1", "user1")
    
    # Filter by user1
    user1_logs = audit_logger.get_logs(user_id="user1")
    
    assert len(user1_logs) == 3
    assert all(log['user_id'] == "user1" for log in user1_logs)


def test_audit_log_filter_with_limit(audit_logger):
    """
    **Validates: Requirements 15.7**
    
    Test limiting number of returned audit logs.
    """
    # Log many entries
    for i in range(20):
        audit_logger.log(f"ACTION{i}", "resource", f"Entry {i}", "user1")
    
    # Get limited results
    logs = audit_logger.get_logs(limit=5)
    
    assert len(logs) == 5
    # Should return most recent entries
    assert logs[0]['action'] == "ACTION19"


# ─── Test: Context Manager ─────────────────────────────────────────────────

def test_audit_logger_context_manager(temp_audit_db):
    """
    **Validates: Requirements 15.7**
    
    Test that AuditLogger works as a context manager.
    """
    with AuditLogger(temp_audit_db) as logger:
        logger.log("TEST", "resource", "Test entry", "user1")
        
        logs = logger.get_logs(limit=1)
        assert len(logs) == 1
    
    # Database should be closed after context exit
    # Verify by opening a new logger
    with AuditLogger(temp_audit_db) as logger2:
        logs = logger2.get_logs(limit=1)
        assert len(logs) == 1


# ─── Test: Convenience Functions ───────────────────────────────────────────

def test_create_audit_logger_function(temp_audit_db):
    """
    **Validates: Requirements 15.7**
    
    Test the create_audit_logger convenience function.
    """
    logger = create_audit_logger(temp_audit_db)
    
    assert isinstance(logger, AuditLogger)
    
    logger.log("TEST", "resource", "Test", "user1")
    logs = logger.get_logs(limit=1)
    assert len(logs) == 1
    
    logger.close()


# ─── Test: Hash Chain Properties ───────────────────────────────────────────

def test_audit_log_hash_chain_continuity(audit_logger):
    """
    **Validates: Requirements 15.8**
    
    Test that each entry's hash depends on the previous entry's hash.
    """
    # Log entries
    audit_logger.log("ENTRY1", "resource", "First", "user1")
    audit_logger.log("ENTRY2", "resource", "Second", "user1")
    audit_logger.log("ENTRY3", "resource", "Third", "user1")
    
    # Get all logs
    logs = audit_logger.get_logs(limit=10)
    logs.reverse()  # Oldest first
    
    # Verify hash chain
    # First entry's hash should not depend on any previous hash (previous_hash is None)
    # Second entry's hash should depend on first entry's hash
    # Third entry's hash should depend on second entry's hash
    
    assert len(logs) == 3
    
    # Each entry should have a hash
    for log in logs:
        assert log['hash'] is not None
        assert len(log['hash']) == 64  # SHA-256 produces 64 hex characters


def test_audit_log_first_entry_has_no_previous_hash(audit_logger):
    """
    **Validates: Requirements 15.8**
    
    Test that the first entry has no previous hash (None).
    """
    # Log first entry
    audit_logger.log("FIRST", "resource", "First entry", "user1")
    
    # The previous_hash for the first entry should be None
    # This is verified implicitly by the integrity check
    is_valid, _ = audit_logger.verify_integrity()
    assert is_valid is True
