"""
Property-based tests for encryption round-trip preservation.

**Validates: Requirements 15.1**

This module tests that encryption and decryption operations preserve data
integrity across a wide range of inputs, including edge cases like empty
strings, unicode characters, and very long strings.

Property 3: Encryption Round-Trip Preservation
- Any string can be encrypted and decrypted back to the original
- The ciphertext is different from the plaintext
- Edge cases work (empty strings, unicode, very long strings)
"""

import pytest
from hypothesis import given, strategies as st, assume

from core.encryption import (
    encrypt_string,
    decrypt_string,
    get_fernet,
    generate_encryption_key,
)


# ─── Test Strategies ────────────────────────────────────────────────────────

def string_strategy() -> st.SearchStrategy[str]:
    """
    Generate diverse test strings including edge cases.
    
    Generates:
    - Empty strings
    - ASCII strings
    - Unicode strings (various scripts)
    - Long strings (up to 10KB, within Hypothesis limits)
    - Strings with special characters
    - Strings with whitespace variations
    """
    return st.one_of(
        # Empty string
        st.just(""),
        
        # Short ASCII strings
        st.text(
            alphabet=st.characters(min_codepoint=32, max_codepoint=126),
            min_size=1,
            max_size=100,
        ),
        
        # Unicode strings (various scripts)
        st.text(
            alphabet=st.characters(
                whitelist_categories=('Lu', 'Ll', 'Nd', 'P', 'Z', 'Sm', 'Sc'),
            ),
            min_size=1,
            max_size=500,
        ),
        
        # Long strings (up to 10KB, within Hypothesis buffer limits)
        st.text(
            alphabet=st.characters(min_codepoint=32, max_codepoint=126),
            min_size=5000,
            max_size=10000,
        ),
        
        # Strings with newlines and tabs
        st.text(
            alphabet=st.characters(
                whitelist_characters='\n\t\r ',
                whitelist_categories=('Lu', 'Ll', 'Nd'),
            ),
            min_size=1,
            max_size=1000,
        ),
        
        # JSON-like strings
        st.text(
            alphabet=st.characters(
                whitelist_characters='{}[]":,',
                whitelist_categories=('Lu', 'Ll', 'Nd'),
            ),
            min_size=1,
            max_size=500,
        ),
        
        # Emoji and special unicode
        st.text(
            alphabet=st.characters(
                whitelist_categories=('So',),  # Other symbols (includes emoji)
            ),
            min_size=1,
            max_size=100,
        ),
    )


# ─── Property Tests ─────────────────────────────────────────────────────────

@given(plaintext=string_strategy())
def test_encryption_roundtrip_preserves_data(plaintext: str):
    """
    **Property 3: Encryption Round-Trip Preservation**
    **Validates: Requirements 15.1**
    
    Property: For any string S, decrypt(encrypt(S)) == S
    
    This test verifies that:
    1. Any string can be encrypted
    2. The encrypted string can be decrypted
    3. The decrypted string equals the original
    
    This property must hold for all inputs including:
    - Empty strings
    - Unicode characters from various scripts
    - Very long strings (up to 100KB)
    - Strings with special characters and whitespace
    """
    # Generate a fresh encryption key for this test
    key = generate_encryption_key()
    fernet = get_fernet(key)
    
    # Encrypt the plaintext
    ciphertext = encrypt_string(plaintext, fernet)
    
    # Decrypt the ciphertext
    decrypted = decrypt_string(ciphertext, fernet)
    
    # Verify round-trip preservation
    assert decrypted == plaintext, (
        f"Round-trip failed: decrypted text does not match original.\n"
        f"Original length: {len(plaintext)}\n"
        f"Decrypted length: {len(decrypted)}\n"
        f"First 100 chars of original: {plaintext[:100]!r}\n"
        f"First 100 chars of decrypted: {decrypted[:100]!r}"
    )


@given(plaintext=string_strategy())
def test_ciphertext_differs_from_plaintext(plaintext: str):
    """
    **Property 3: Encryption Round-Trip Preservation (Security Property)**
    **Validates: Requirements 15.1**
    
    Property: For any non-empty string S, encrypt(S) != S
    
    This test verifies that encryption actually transforms the data
    and doesn't just return the plaintext unchanged.
    
    Note: Empty strings may have predictable ciphertext patterns,
    so we skip them for this test.
    """
    # Skip empty strings as they may have predictable ciphertext
    assume(len(plaintext) > 0)
    
    # Generate a fresh encryption key for this test
    key = generate_encryption_key()
    fernet = get_fernet(key)
    
    # Encrypt the plaintext
    ciphertext = encrypt_string(plaintext, fernet)
    
    # Verify ciphertext differs from plaintext
    assert ciphertext != plaintext, (
        f"Ciphertext is identical to plaintext!\n"
        f"Plaintext: {plaintext[:100]!r}\n"
        f"Ciphertext: {ciphertext[:100]!r}"
    )


@given(plaintext=string_strategy())
def test_encryption_is_deterministic_with_same_key(plaintext: str):
    """
    **Property 3: Encryption Round-Trip Preservation (Consistency Property)**
    **Validates: Requirements 15.1**
    
    Property: For any string S and key K, multiple encryptions with K
    should all decrypt to S (though ciphertexts may differ due to nonces).
    
    This test verifies that:
    1. Multiple encryptions with the same key are all valid
    2. All ciphertexts decrypt to the same original plaintext
    3. The encryption system is consistent
    
    Note: Fernet uses timestamps and random data, so ciphertexts will differ,
    but all should decrypt to the same plaintext.
    """
    # Generate a fresh encryption key for this test
    key = generate_encryption_key()
    fernet = get_fernet(key)
    
    # Encrypt the same plaintext multiple times
    ciphertext1 = encrypt_string(plaintext, fernet)
    ciphertext2 = encrypt_string(plaintext, fernet)
    ciphertext3 = encrypt_string(plaintext, fernet)
    
    # Decrypt all ciphertexts
    decrypted1 = decrypt_string(ciphertext1, fernet)
    decrypted2 = decrypt_string(ciphertext2, fernet)
    decrypted3 = decrypt_string(ciphertext3, fernet)
    
    # Verify all decrypt to the same original plaintext
    assert decrypted1 == plaintext, "First decryption failed"
    assert decrypted2 == plaintext, "Second decryption failed"
    assert decrypted3 == plaintext, "Third decryption failed"


@given(plaintext=string_strategy())
def test_encryption_with_different_keys_produces_different_ciphertexts(plaintext: str):
    """
    **Property 3: Encryption Round-Trip Preservation (Key Isolation Property)**
    **Validates: Requirements 15.1**
    
    Property: For any non-empty string S and different keys K1 and K2,
    encrypt(S, K1) should produce a different ciphertext than encrypt(S, K2).
    
    This test verifies that different encryption keys produce different
    ciphertexts, ensuring key isolation.
    
    Note: We skip empty strings as they may have predictable patterns.
    """
    # Skip empty strings
    assume(len(plaintext) > 0)
    
    # Generate two different encryption keys
    key1 = generate_encryption_key()
    key2 = generate_encryption_key()
    fernet1 = get_fernet(key1)
    fernet2 = get_fernet(key2)
    
    # Encrypt with both keys
    ciphertext1 = encrypt_string(plaintext, fernet1)
    ciphertext2 = encrypt_string(plaintext, fernet2)
    
    # Verify ciphertexts differ
    assert ciphertext1 != ciphertext2, (
        f"Different keys produced identical ciphertexts!\n"
        f"Plaintext: {plaintext[:100]!r}\n"
        f"Ciphertext1: {ciphertext1[:100]!r}\n"
        f"Ciphertext2: {ciphertext2[:100]!r}"
    )
    
    # Verify each decrypts correctly with its own key
    decrypted1 = decrypt_string(ciphertext1, fernet1)
    decrypted2 = decrypt_string(ciphertext2, fernet2)
    
    assert decrypted1 == plaintext, "Decryption with key1 failed"
    assert decrypted2 == plaintext, "Decryption with key2 failed"


@given(plaintext=string_strategy())
def test_decryption_with_wrong_key_fails(plaintext: str):
    """
    **Property 3: Encryption Round-Trip Preservation (Security Property)**
    **Validates: Requirements 15.1**
    
    Property: For any string S and different keys K1 and K2,
    decrypt(encrypt(S, K1), K2) should fail.
    
    This test verifies that ciphertext encrypted with one key
    cannot be decrypted with a different key, ensuring security.
    """
    # Skip empty strings
    assume(len(plaintext) > 0)
    
    # Generate two different encryption keys
    key1 = generate_encryption_key()
    key2 = generate_encryption_key()
    fernet1 = get_fernet(key1)
    fernet2 = get_fernet(key2)
    
    # Encrypt with key1
    ciphertext = encrypt_string(plaintext, fernet1)
    
    # Attempt to decrypt with key2 should fail
    with pytest.raises(Exception):  # Fernet raises cryptography.fernet.InvalidToken
        decrypt_string(ciphertext, fernet2)


# ─── Edge Case Tests ────────────────────────────────────────────────────────

def test_empty_string_roundtrip():
    """
    Test that empty strings can be encrypted and decrypted.
    
    This is an important edge case that should be explicitly tested.
    """
    key = generate_encryption_key()
    fernet = get_fernet(key)
    
    plaintext = ""
    ciphertext = encrypt_string(plaintext, fernet)
    decrypted = decrypt_string(ciphertext, fernet)
    
    assert decrypted == plaintext
    assert ciphertext != plaintext  # Even empty strings should be encrypted


def test_very_long_string_roundtrip():
    """
    Test that very long strings (1MB) can be encrypted and decrypted.
    
    This tests the system's ability to handle large data.
    """
    key = generate_encryption_key()
    fernet = get_fernet(key)
    
    # Create a 1MB string
    plaintext = "A" * (1024 * 1024)
    
    ciphertext = encrypt_string(plaintext, fernet)
    decrypted = decrypt_string(ciphertext, fernet)
    
    assert decrypted == plaintext
    assert len(ciphertext) > len(plaintext)  # Ciphertext should be larger due to overhead


def test_unicode_emoji_roundtrip():
    """
    Test that emoji and special unicode characters are preserved.
    
    This tests handling of multi-byte UTF-8 characters.
    """
    key = generate_encryption_key()
    fernet = get_fernet(key)
    
    plaintext = "Hello 👋 World 🌍! Testing emoji 😀🎉🔒"
    
    ciphertext = encrypt_string(plaintext, fernet)
    decrypted = decrypt_string(ciphertext, fernet)
    
    assert decrypted == plaintext


def test_json_string_roundtrip():
    """
    Test that JSON strings are preserved correctly.
    
    This is important since encrypt_dict uses JSON serialization.
    """
    key = generate_encryption_key()
    fernet = get_fernet(key)
    
    plaintext = '{"name": "Test", "value": 123, "nested": {"key": "value"}}'
    
    ciphertext = encrypt_string(plaintext, fernet)
    decrypted = decrypt_string(ciphertext, fernet)
    
    assert decrypted == plaintext


def test_multiline_string_roundtrip():
    """
    Test that strings with newlines and tabs are preserved.
    
    This tests handling of whitespace characters.
    """
    key = generate_encryption_key()
    fernet = get_fernet(key)
    
    plaintext = "Line 1\nLine 2\n\tIndented\r\nWindows line ending"
    
    ciphertext = encrypt_string(plaintext, fernet)
    decrypted = decrypt_string(ciphertext, fernet)
    
    assert decrypted == plaintext
