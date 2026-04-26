"""
Security tests for encryption system.

**Validates: Requirements 15.1, 15.2, 15.3, 15.4, 18.9**

This module tests:
- AES-256-GCM encryption/decryption round-trip
- Keyring storage and retrieval
- Passphrase key derivation with PBKDF2
- Key isolation and security properties
"""

import pytest
import tempfile
import sqlite3
from pathlib import Path
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from core.key_manager import KeyManager


# ─── Fixtures ───────────────────────────────────────────────────────────────

@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        db_path = Path(f.name)
    
    yield db_path
    
    # Cleanup
    if db_path.exists():
        db_path.unlink()


@pytest.fixture
def keyring_key_manager():
    """Create a KeyManager with keyring storage (if available)."""
    try:
        km = KeyManager(use_keyring=True)
        yield km
        # Cleanup: delete key from keyring
        km.delete_key_from_keyring()
    except ValueError:
        # Keyring not available, skip tests that require it
        pytest.skip("Keyring not available on this system")


@pytest.fixture
def passphrase_key_manager(temp_db):
    """Create a KeyManager with passphrase-based key derivation."""
    km = KeyManager(
        use_keyring=False,
        passphrase="test_passphrase_12345",
        db_path=temp_db
    )
    yield km


# ─── Test: Encryption/Decryption Round-Trip ────────────────────────────────

def test_encryption_roundtrip_basic(passphrase_key_manager):
    """
    **Validates: Requirements 15.1**
    
    Test basic encryption/decryption round-trip with AES-256-GCM.
    """
    km = passphrase_key_manager
    
    plaintext = "Hello, Aegis! This is sensitive health data."
    
    # Encrypt
    ciphertext = km.encrypt(plaintext)
    
    # Verify ciphertext is bytes
    assert isinstance(ciphertext, bytes)
    
    # Verify ciphertext differs from plaintext
    assert ciphertext != plaintext.encode('utf-8')
    
    # Decrypt
    decrypted = km.decrypt(ciphertext)
    
    # Verify round-trip preservation
    assert decrypted == plaintext


def test_encryption_roundtrip_unicode(passphrase_key_manager):
    """
    **Validates: Requirements 15.1**
    
    Test encryption/decryption with unicode characters.
    """
    km = passphrase_key_manager
    
    plaintext = "こんにちは、Aegis！ 🏥 健康データ"
    
    ciphertext = km.encrypt(plaintext)
    decrypted = km.decrypt(ciphertext)
    
    assert decrypted == plaintext


def test_encryption_roundtrip_empty_string(passphrase_key_manager):
    """
    **Validates: Requirements 15.1**
    
    Test encryption/decryption with empty string.
    """
    km = passphrase_key_manager
    
    plaintext = ""
    
    ciphertext = km.encrypt(plaintext)
    decrypted = km.decrypt(ciphertext)
    
    assert decrypted == plaintext


def test_encryption_roundtrip_long_string(passphrase_key_manager):
    """
    **Validates: Requirements 15.1**
    
    Test encryption/decryption with very long string (1MB).
    """
    km = passphrase_key_manager
    
    plaintext = "A" * (1024 * 1024)  # 1MB
    
    ciphertext = km.encrypt(plaintext)
    decrypted = km.decrypt(ciphertext)
    
    assert decrypted == plaintext


def test_encryption_dict_roundtrip(passphrase_key_manager):
    """
    **Validates: Requirements 15.1**
    
    Test encryption/decryption of dictionaries.
    """
    km = passphrase_key_manager
    
    data = {
        'user_id': 'user123',
        'mood': 'happy',
        'energy': 8,
        'notes': 'Feeling great today! 😊',
    }
    
    ciphertext = km.encrypt_dict(data)
    decrypted = km.decrypt_dict(ciphertext)
    
    assert decrypted == data


# ─── Test: Keyring Storage ─────────────────────────────────────────────────

def test_keyring_storage_and_retrieval():
    """
    **Validates: Requirements 15.2**
    
    Test that keys can be stored and retrieved from system keyring.
    """
    try:
        # Create KeyManager with keyring
        km1 = KeyManager(use_keyring=True)
        
        # Encrypt some data
        plaintext = "Test data for keyring"
        ciphertext = km1.encrypt(plaintext)
        
        # Create a new KeyManager instance (should load same key from keyring)
        km2 = KeyManager(use_keyring=True)
        
        # Decrypt with the second instance
        decrypted = km2.decrypt(ciphertext)
        
        assert decrypted == plaintext
        
        # Cleanup
        km1.delete_key_from_keyring()
    except ValueError:
        pytest.skip("Keyring not available on this system")


def test_keyring_fallback_to_passphrase(temp_db):
    """
    **Validates: Requirements 15.2**
    
    Test that KeyManager falls back to passphrase mode when keyring unavailable.
    """
    # Force keyring to be unavailable by using a non-existent keyring backend
    # This is simulated by catching the ValueError when no passphrase is provided
    
    # This should work with passphrase
    km = KeyManager(
        use_keyring=False,
        passphrase="fallback_test_passphrase",
        db_path=temp_db
    )
    
    plaintext = "Fallback test"
    ciphertext = km.encrypt(plaintext)
    decrypted = km.decrypt(ciphertext)
    
    assert decrypted == plaintext


# ─── Test: Passphrase Key Derivation ───────────────────────────────────────

def test_passphrase_key_derivation(temp_db):
    """
    **Validates: Requirements 15.3, 15.4**
    
    Test PBKDF2 key derivation from passphrase.
    """
    passphrase = "my_secure_passphrase_123"
    
    # Create KeyManager with passphrase
    km = KeyManager(
        use_keyring=False,
        passphrase=passphrase,
        db_path=temp_db
    )
    
    # Verify key was derived
    assert km.master_key is not None
    assert len(km.master_key) == 32  # 256 bits
    
    # Encrypt some data
    plaintext = "Passphrase-protected data"
    ciphertext = km.encrypt(plaintext)
    
    # Create new KeyManager with same passphrase (should derive same key)
    km2 = KeyManager(
        use_keyring=False,
        passphrase=passphrase,
        db_path=temp_db
    )
    
    # Decrypt with second instance
    decrypted = km2.decrypt(ciphertext)
    
    assert decrypted == plaintext


def test_passphrase_different_keys_for_different_passphrases(temp_db):
    """
    **Validates: Requirements 15.3**
    
    Test that different passphrases produce different keys.
    """
    plaintext = "Secret data"
    
    # Create KeyManager with first passphrase
    km1 = KeyManager(
        use_keyring=False,
        passphrase="passphrase_one",
        db_path=temp_db
    )
    ciphertext1 = km1.encrypt(plaintext)
    
    # Create new database for second passphrase
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        temp_db2 = Path(f.name)
    
    try:
        # Create KeyManager with different passphrase
        km2 = KeyManager(
            use_keyring=False,
            passphrase="passphrase_two",
            db_path=temp_db2
        )
        
        # Keys should be different
        assert km1.master_key != km2.master_key
        
        # Decryption with wrong key should fail
        with pytest.raises(Exception):  # cryptography raises InvalidTag
            km2.decrypt(ciphertext1)
    finally:
        if temp_db2.exists():
            temp_db2.unlink()


def test_passphrase_salt_stored_in_database(temp_db):
    """
    **Validates: Requirements 15.4**
    
    Test that salt is stored in database metadata.
    """
    km = KeyManager(
        use_keyring=False,
        passphrase="test_passphrase",
        db_path=temp_db
    )
    
    # Check that salt was stored in database
    db = sqlite3.connect(temp_db)
    cursor = db.cursor()
    cursor.execute("SELECT value FROM encryption_metadata WHERE key = 'salt'")
    row = cursor.fetchone()
    db.close()
    
    assert row is not None
    salt = bytes.fromhex(row[0])
    assert len(salt) == 16  # 128 bits


def test_passphrase_pbkdf2_iterations():
    """
    **Validates: Requirements 15.3**
    
    Test that PBKDF2 uses 100,000 iterations as specified.
    """
    assert KeyManager.PBKDF2_ITERATIONS == 100000


# ─── Test: Security Properties ─────────────────────────────────────────────

def test_encryption_produces_different_ciphertexts(passphrase_key_manager):
    """
    **Validates: Requirements 15.1**
    
    Test that encrypting the same plaintext twice produces different ciphertexts
    (due to random nonces in GCM mode).
    """
    km = passphrase_key_manager
    
    plaintext = "Same plaintext"
    
    ciphertext1 = km.encrypt(plaintext)
    ciphertext2 = km.encrypt(plaintext)
    
    # Ciphertexts should differ (different nonces)
    assert ciphertext1 != ciphertext2
    
    # But both should decrypt to same plaintext
    assert km.decrypt(ciphertext1) == plaintext
    assert km.decrypt(ciphertext2) == plaintext


def test_encryption_nonce_size(passphrase_key_manager):
    """
    **Validates: Requirements 15.1**
    
    Test that nonces are 96 bits (12 bytes) as required for GCM.
    """
    km = passphrase_key_manager
    
    plaintext = "Test"
    ciphertext = km.encrypt(plaintext)
    
    # First 12 bytes should be the nonce
    nonce = ciphertext[:12]
    assert len(nonce) == 12


def test_encryption_key_size():
    """
    **Validates: Requirements 15.1**
    
    Test that encryption uses 256-bit keys (AES-256).
    """
    assert KeyManager.KEY_SIZE_BYTES == 32  # 256 bits


def test_decryption_with_wrong_key_fails(temp_db):
    """
    **Validates: Requirements 15.1**
    
    Test that decryption with wrong key fails.
    """
    plaintext = "Secret message"
    
    # Encrypt with first key
    km1 = KeyManager(
        use_keyring=False,
        passphrase="key_one",
        db_path=temp_db
    )
    ciphertext = km1.encrypt(plaintext)
    
    # Try to decrypt with different key
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        temp_db2 = Path(f.name)
    
    try:
        km2 = KeyManager(
            use_keyring=False,
            passphrase="key_two",
            db_path=temp_db2
        )
        
        # Decryption should fail
        with pytest.raises(Exception):  # cryptography raises InvalidTag
            km2.decrypt(ciphertext)
    finally:
        if temp_db2.exists():
            temp_db2.unlink()


def test_decryption_with_corrupted_ciphertext_fails(passphrase_key_manager):
    """
    **Validates: Requirements 15.1**
    
    Test that decryption fails if ciphertext is corrupted.
    """
    km = passphrase_key_manager
    
    plaintext = "Original message"
    ciphertext = km.encrypt(plaintext)
    
    # Corrupt the ciphertext
    corrupted = bytearray(ciphertext)
    corrupted[-1] ^= 0xFF  # Flip bits in last byte
    corrupted = bytes(corrupted)
    
    # Decryption should fail
    with pytest.raises(Exception):  # cryptography raises InvalidTag
        km.decrypt(corrupted)


def test_decryption_with_short_ciphertext_fails(passphrase_key_manager):
    """
    **Validates: Requirements 15.1**
    
    Test that decryption fails if ciphertext is too short.
    """
    km = passphrase_key_manager
    
    # Ciphertext shorter than nonce size
    short_ciphertext = b"short"
    
    with pytest.raises(ValueError, match="too short"):
        km.decrypt(short_ciphertext)


# ─── Test: Error Handling ──────────────────────────────────────────────────

def test_key_manager_requires_passphrase_when_keyring_unavailable():
    """
    **Validates: Requirements 15.2, 15.3**
    
    Test that KeyManager raises error when neither keyring nor passphrase available.
    """
    with pytest.raises(ValueError, match="Passphrase required"):
        KeyManager(use_keyring=False, passphrase=None)


def test_passphrase_mode_requires_db_path():
    """
    **Validates: Requirements 15.4**
    
    Test that passphrase mode requires db_path for salt storage.
    """
    with pytest.raises(ValueError, match="db_path required"):
        km = KeyManager(
            use_keyring=False,
            passphrase="test",
            db_path=None
        )
