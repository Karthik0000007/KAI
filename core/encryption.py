"""
Aegis Encryption Utilities
Local encryption for health data at rest.

**DEPRECATED**: This module provides backward compatibility with Fernet (AES-128-CBC).
New code should use `core.key_manager.KeyManager` for AES-256-GCM encryption.

Includes optional differential privacy noise injection.
"""

import os
import json
import secrets
import hashlib
import base64
from pathlib import Path
from typing import Any, Dict, Optional
from cryptography.fernet import Fernet

from core.config import ENCRYPTION_KEY_FILE, DIFFERENTIAL_PRIVACY_EPSILON, ENABLE_DIFFERENTIAL_PRIVACY

# Note: For new code, use KeyManager instead
# from core.key_manager import KeyManager


# ─── Key Management ─────────────────────────────────────────────────────────

def _derive_key_from_passphrase(passphrase: str) -> bytes:
    """Derive a Fernet-compatible key from a passphrase using SHA-256."""
    digest = hashlib.sha256(passphrase.encode("utf-8")).digest()
    return base64.urlsafe_b64encode(digest)


def generate_encryption_key() -> bytes:
    """Generate a new Fernet encryption key."""
    return Fernet.generate_key()


def load_or_create_key(key_path: Optional[Path] = None) -> bytes:
    """Load existing key or create a new one. Stores to disk."""
    key_path = key_path or ENCRYPTION_KEY_FILE
    key_path = Path(key_path)

    if key_path.exists():
        return key_path.read_bytes().strip()

    key = generate_encryption_key()
    key_path.parent.mkdir(parents=True, exist_ok=True)
    key_path.write_bytes(key)
    # Restrict permissions on Unix-like systems
    try:
        os.chmod(key_path, 0o600)
    except (OSError, AttributeError):
        pass  # Windows doesn't support chmod the same way
    return key


def get_fernet(key: Optional[bytes] = None) -> Fernet:
    """Get a Fernet instance, loading/creating the key if needed."""
    if key is None:
        key = load_or_create_key()
    return Fernet(key)


# ─── Encrypt / Decrypt ──────────────────────────────────────────────────────

def encrypt_string(plaintext: str, fernet: Optional[Fernet] = None) -> str:
    """Encrypt a string, return base64-encoded ciphertext."""
    f = fernet or get_fernet()
    token = f.encrypt(plaintext.encode("utf-8"))
    return token.decode("utf-8")


def decrypt_string(ciphertext: str, fernet: Optional[Fernet] = None) -> str:
    """Decrypt a base64-encoded ciphertext back to plaintext."""
    f = fernet or get_fernet()
    plaintext = f.decrypt(ciphertext.encode("utf-8"))
    return plaintext.decode("utf-8")


def encrypt_dict(data: Dict[str, Any], fernet: Optional[Fernet] = None) -> str:
    """Serialize a dict to JSON and encrypt it."""
    json_str = json.dumps(data, ensure_ascii=False)
    return encrypt_string(json_str, fernet)


def decrypt_dict(ciphertext: str, fernet: Optional[Fernet] = None) -> Dict[str, Any]:
    """Decrypt ciphertext and deserialize from JSON."""
    json_str = decrypt_string(ciphertext, fernet)
    return json.loads(json_str)


# ─── Differential Privacy ───────────────────────────────────────────────────

def add_laplace_noise(value: float, sensitivity: float = 1.0,
                      epsilon: Optional[float] = None) -> float:
    """
    Add Laplace noise to a numeric value for differential privacy.
    
    Args:
        value: The original value.
        sensitivity: The sensitivity of the query (max change one record can cause).
        epsilon: Privacy budget. Lower = more privacy, more noise.
    
    Returns:
        Noised value.
    """
    if not ENABLE_DIFFERENTIAL_PRIVACY:
        return value

    eps = epsilon or DIFFERENTIAL_PRIVACY_EPSILON
    if eps <= 0:
        return value

    scale = sensitivity / eps
    # Laplace noise via inverse CDF
    import random
    u = random.random() - 0.5
    noise = -scale * (1 if u >= 0 else -1) * __import__("math").log(1 - 2 * abs(u))
    return value + noise


def sanitize_for_storage(data: Dict[str, Any],
                         numeric_fields: Optional[list] = None) -> Dict[str, Any]:
    """
    Apply differential privacy noise to specified numeric fields before storage.
    """
    if not ENABLE_DIFFERENTIAL_PRIVACY or not numeric_fields:
        return data

    sanitized = data.copy()
    for field_name in numeric_fields:
        if field_name in sanitized and sanitized[field_name] is not None:
            try:
                original = float(sanitized[field_name])
                sanitized[field_name] = round(add_laplace_noise(original), 2)
            except (ValueError, TypeError):
                pass
    return sanitized


# ─── Secure Random ──────────────────────────────────────────────────────────

def secure_random_id(length: int = 16) -> str:
    """Generate a cryptographically secure random hex ID."""
    return secrets.token_hex(length // 2)
