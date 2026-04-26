"""
Aegis Key Manager
Manages encryption keys with AES-256-GCM, keyring integration, and passphrase support.

**Validates: Requirements 15.1, 15.2, 15.3, 15.4**
"""

import os
import hashlib
import sqlite3
from pathlib import Path
from typing import Optional
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes

from core.logger import get_logger

logger = get_logger(__name__)


class KeyManager:
    """
    Manages encryption keys with multiple storage backends.
    
    Supports:
    - System keyring storage (Windows Credential Manager, macOS Keychain, Linux Secret Service)
    - Passphrase-based key derivation with PBKDF2
    - AES-256-GCM encryption/decryption
    
    **Validates: Requirements 15.1, 15.2, 15.3, 15.4**
    """
    
    SERVICE_NAME = "aegis-health-ai"
    KEY_NAME = "master-encryption-key"
    PBKDF2_ITERATIONS = 100000
    KEY_SIZE_BYTES = 32  # 256 bits
    NONCE_SIZE_BYTES = 12  # 96 bits for GCM
    
    def __init__(
        self,
        use_keyring: bool = True,
        passphrase: Optional[str] = None,
        db_path: Optional[Path] = None,
    ):
        """
        Initialize KeyManager.
        
        Args:
            use_keyring: Whether to use system keyring for key storage
            passphrase: User passphrase for key derivation (required if keyring unavailable)
            db_path: Path to database for storing salt (required for passphrase mode)
        """
        self.use_keyring = use_keyring
        self.passphrase = passphrase
        self.db_path = db_path
        self.master_key: Optional[bytes] = None
        
        # Load or create the master key
        self.master_key = self.load_or_create_key()
    
    def load_or_create_key(self) -> bytes:
        """
        Load existing key or create a new one.
        
        Priority:
        1. Try keyring if enabled
        2. Fall back to passphrase mode if keyring unavailable
        3. Raise error if neither available
        
        Returns:
            32-byte master encryption key
        
        Raises:
            ValueError: If neither keyring nor passphrase is available
        """
        if self.use_keyring:
            try:
                import keyring
                
                # Try to load existing key from keyring
                key_hex = keyring.get_password(self.SERVICE_NAME, self.KEY_NAME)
                if key_hex:
                    logger.info("Loaded encryption key from system keyring")
                    return bytes.fromhex(key_hex)
                else:
                    # Generate new key and store in keyring
                    key = AESGCM.generate_key(bit_length=256)
                    keyring.set_password(
                        self.SERVICE_NAME,
                        self.KEY_NAME,
                        key.hex()
                    )
                    logger.info("Generated new encryption key and stored in system keyring")
                    return key
            except ImportError:
                logger.warning("keyring library not installed, falling back to passphrase mode")
                self.use_keyring = False
            except Exception as e:
                logger.warning(f"Keyring unavailable ({e}), falling back to passphrase mode")
                self.use_keyring = False
        
        # Fall back to passphrase mode
        if self.passphrase:
            logger.info("Using passphrase-based key derivation")
            return self.derive_key_from_passphrase(self.passphrase)
        else:
            raise ValueError(
                "Passphrase required when keyring unavailable. "
                "Please provide a passphrase or enable keyring support."
            )
    
    def derive_key_from_passphrase(self, passphrase: str) -> bytes:
        """
        Derive encryption key from passphrase using PBKDF2-HMAC-SHA256.
        
        **Validates: Requirements 15.3, 15.4**
        
        Args:
            passphrase: User-provided passphrase
        
        Returns:
            32-byte derived key
        
        Raises:
            ValueError: If db_path not provided for salt storage
        """
        if not self.db_path:
            raise ValueError("db_path required for passphrase mode (to store salt)")
        
        # Load or generate salt
        salt = self.load_or_create_salt()
        
        # Derive key using PBKDF2
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=self.KEY_SIZE_BYTES,
            salt=salt,
            iterations=self.PBKDF2_ITERATIONS,
        )
        
        key = kdf.derive(passphrase.encode('utf-8'))
        logger.info(f"Derived key from passphrase using PBKDF2 ({self.PBKDF2_ITERATIONS} iterations)")
        return key
    
    def load_or_create_salt(self) -> bytes:
        """
        Load existing salt from database or create a new one.
        
        Salt is stored in a metadata table in the database.
        
        Returns:
            16-byte salt
        """
        db = sqlite3.connect(self.db_path)
        cursor = db.cursor()
        
        # Create metadata table if it doesn't exist
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS encryption_metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
        """)
        
        # Try to load existing salt
        cursor.execute("SELECT value FROM encryption_metadata WHERE key = 'salt'")
        row = cursor.fetchone()
        
        if row:
            salt = bytes.fromhex(row[0])
            logger.debug("Loaded existing salt from database")
        else:
            # Generate new salt
            salt = os.urandom(16)
            cursor.execute(
                "INSERT INTO encryption_metadata (key, value) VALUES ('salt', ?)",
                (salt.hex(),)
            )
            db.commit()
            logger.info("Generated new salt and stored in database")
        
        db.close()
        return salt
    
    def encrypt(self, plaintext: str) -> bytes:
        """
        Encrypt plaintext using AES-256-GCM.
        
        **Validates: Requirements 15.1**
        
        Args:
            plaintext: String to encrypt
        
        Returns:
            Encrypted bytes (nonce + ciphertext)
        """
        if not self.master_key:
            raise ValueError("Master key not initialized")
        
        aesgcm = AESGCM(self.master_key)
        nonce = os.urandom(self.NONCE_SIZE_BYTES)
        ciphertext = aesgcm.encrypt(nonce, plaintext.encode('utf-8'), None)
        
        # Prepend nonce to ciphertext for storage
        return nonce + ciphertext
    
    def decrypt(self, ciphertext: bytes) -> str:
        """
        Decrypt ciphertext using AES-256-GCM.
        
        **Validates: Requirements 15.1**
        
        Args:
            ciphertext: Encrypted bytes (nonce + ciphertext)
        
        Returns:
            Decrypted plaintext string
        
        Raises:
            ValueError: If ciphertext is too short or decryption fails
        """
        if not self.master_key:
            raise ValueError("Master key not initialized")
        
        if len(ciphertext) < self.NONCE_SIZE_BYTES:
            raise ValueError("Ciphertext too short to contain nonce")
        
        # Extract nonce and actual ciphertext
        nonce = ciphertext[:self.NONCE_SIZE_BYTES]
        actual_ciphertext = ciphertext[self.NONCE_SIZE_BYTES:]
        
        aesgcm = AESGCM(self.master_key)
        plaintext_bytes = aesgcm.decrypt(nonce, actual_ciphertext, None)
        
        return plaintext_bytes.decode('utf-8')
    
    def encrypt_dict(self, data: dict) -> bytes:
        """
        Encrypt a dictionary by serializing to JSON first.
        
        Args:
            data: Dictionary to encrypt
        
        Returns:
            Encrypted bytes
        """
        import json
        json_str = json.dumps(data, ensure_ascii=False)
        return self.encrypt(json_str)
    
    def decrypt_dict(self, ciphertext: bytes) -> dict:
        """
        Decrypt ciphertext and deserialize from JSON.
        
        Args:
            ciphertext: Encrypted bytes
        
        Returns:
            Decrypted dictionary
        """
        import json
        json_str = self.decrypt(ciphertext)
        return json.loads(json_str)
    
    def delete_key_from_keyring(self) -> bool:
        """
        Delete the encryption key from the system keyring.
        
        Useful for testing or when switching to passphrase mode.
        
        Returns:
            True if key was deleted, False if keyring unavailable
        """
        if not self.use_keyring:
            return False
        
        try:
            import keyring
            keyring.delete_password(self.SERVICE_NAME, self.KEY_NAME)
            logger.info("Deleted encryption key from system keyring")
            return True
        except Exception as e:
            logger.warning(f"Failed to delete key from keyring: {e}")
            return False
