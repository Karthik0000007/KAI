"""
Aegis Audit Logger
Tamper-evident audit logging with HMAC chain for integrity verification.

**Validates: Requirements 15.7, 15.8**
"""

import hashlib
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Tuple

from core.logger import get_logger

logger = get_logger(__name__)


class AuditLogger:
    """
    Tamper-evident audit logger with HMAC chain.
    
    Each audit log entry includes a hash that depends on:
    - The current entry's data
    - The previous entry's hash
    
    This creates a chain where tampering with any entry breaks the chain,
    making tampering detectable.
    
    **Validates: Requirements 15.7, 15.8**
    """
    
    def __init__(self, db_path: Path):
        """
        Initialize AuditLogger.
        
        Args:
            db_path: Path to audit log database
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.db = sqlite3.connect(str(self.db_path))
        self.create_audit_table()
        self.previous_hash = self.get_last_hash()
    
    def create_audit_table(self):
        """Create audit log table if it doesn't exist."""
        self.db.execute("""
            CREATE TABLE IF NOT EXISTS audit_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                user_id TEXT,
                action TEXT NOT NULL,
                resource TEXT,
                details TEXT,
                hash TEXT NOT NULL
            )
        """)
        self.db.commit()
        logger.debug("Audit log table created/verified")
    
    def get_last_hash(self) -> Optional[str]:
        """
        Get the hash of the last audit log entry.
        
        Returns:
            Last hash, or None if no entries exist
        """
        cursor = self.db.execute(
            "SELECT hash FROM audit_log ORDER BY id DESC LIMIT 1"
        )
        row = cursor.fetchone()
        return row[0] if row else None
    
    def log(
        self,
        action: str,
        resource: Optional[str] = None,
        details: Optional[str] = None,
        user_id: Optional[str] = None,
    ):
        """
        Log an audit event with tamper-evident hash.
        
        **Validates: Requirements 15.7**
        
        Args:
            action: Action performed (e.g., "READ", "WRITE", "DELETE")
            resource: Resource accessed (e.g., "health_checkins", "user_profile")
            details: Additional details about the action
            user_id: User who performed the action
        """
        timestamp = datetime.now().isoformat()
        
        # Compute tamper-evident hash
        # Hash includes: timestamp, user_id, action, resource, details, previous_hash
        record = f"{timestamp}|{user_id or ''}|{action}|{resource or ''}|{details or ''}|{self.previous_hash or ''}"
        current_hash = hashlib.sha256(record.encode('utf-8')).hexdigest()
        
        # Insert audit log entry
        self.db.execute("""
            INSERT INTO audit_log (timestamp, user_id, action, resource, details, hash)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (timestamp, user_id, action, resource, details, current_hash))
        
        self.db.commit()
        self.previous_hash = current_hash
        
        logger.debug(f"Audit log: {action} on {resource} by {user_id}")
    
    def verify_integrity(self) -> Tuple[bool, Optional[int]]:
        """
        Verify audit log integrity by checking the hash chain.
        
        **Validates: Requirements 15.8**
        
        Returns:
            Tuple of (is_valid, first_invalid_id)
            - is_valid: True if all hashes are valid
            - first_invalid_id: ID of first invalid entry, or None if all valid
        """
        cursor = self.db.execute("SELECT * FROM audit_log ORDER BY id")
        previous_hash = None
        
        for row in cursor:
            entry_id, timestamp, user_id, action, resource, details, stored_hash = row
            
            # Recompute hash
            record = f"{timestamp}|{user_id or ''}|{action}|{resource or ''}|{details or ''}|{previous_hash or ''}"
            computed_hash = hashlib.sha256(record.encode('utf-8')).hexdigest()
            
            # Check if hash matches
            if computed_hash != stored_hash:
                logger.error(f"Audit log integrity violation at entry {entry_id}")
                return False, entry_id
            
            previous_hash = stored_hash
        
        logger.info("Audit log integrity verified successfully")
        return True, None
    
    def get_logs(
        self,
        action: Optional[str] = None,
        resource: Optional[str] = None,
        user_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[dict]:
        """
        Retrieve audit logs with optional filtering.
        
        Args:
            action: Filter by action type
            resource: Filter by resource
            user_id: Filter by user ID
            limit: Maximum number of entries to return
        
        Returns:
            List of audit log entries as dictionaries
        """
        query = "SELECT * FROM audit_log WHERE 1=1"
        params = []
        
        if action:
            query += " AND action = ?"
            params.append(action)
        
        if resource:
            query += " AND resource = ?"
            params.append(resource)
        
        if user_id:
            query += " AND user_id = ?"
            params.append(user_id)
        
        query += " ORDER BY id DESC LIMIT ?"
        params.append(limit)
        
        cursor = self.db.execute(query, params)
        
        logs = []
        for row in cursor:
            entry_id, timestamp, user_id, action, resource, details, hash_val = row
            logs.append({
                'id': entry_id,
                'timestamp': timestamp,
                'user_id': user_id,
                'action': action,
                'resource': resource,
                'details': details,
                'hash': hash_val,
            })
        
        return logs
    
    def close(self):
        """Close the database connection."""
        if self.db:
            self.db.close()
            logger.debug("Audit logger database connection closed")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


# ─── Convenience Functions ──────────────────────────────────────────────────

def create_audit_logger(db_path: Path) -> AuditLogger:
    """
    Create and return an AuditLogger instance.
    
    Args:
        db_path: Path to audit log database
    
    Returns:
        AuditLogger instance
    """
    return AuditLogger(db_path)
