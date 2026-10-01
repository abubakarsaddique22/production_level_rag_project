import time

import pytest
from jose import JWTError, jwt

from nexora_rag.core.config import settings
from nexora_rag.core.security import (
    create_access_token,
    decode_access_token,
    hash_password,
    verify_password,
)

# ---------- passwords ----------

def test_password_is_hashed_and_verifies():
    hashed = hash_password("S3cret!")
    assert hashed != "S3cret!"
    assert verify_password("S3cret!", hashed) is True


def test_wrong_password_fails():
    assert verify_password("wrong", hash_password("right")) is False


def test_same_password_gives_different_hashes():
    assert hash_password("same") != hash_password("same")  # random salt


# ---------- JWT ----------

def test_token_roundtrip_keeps_user_and_role():
    payload = decode_access_token(create_access_token("u1", "engineer"))
    assert payload["sub"] == "u1"
    assert payload["role"] == "engineer"
    assert "exp" in payload


def test_expired_token_is_rejected(monkeypatch):
    monkeypatch.setattr(settings, "jwt_expire_minutes", -1)
    token = create_access_token("u1", "employee")
    with pytest.raises(JWTError):
        decode_access_token(token)


def test_token_signed_with_another_secret_is_rejected():
    # an attacker who forges "role: admin" without knowing the secret
    forged = jwt.encode(
        {"sub": "u1", "role": "admin", "exp": time.time() + 60},
        "attacker-secret",
        algorithm=settings.jwt_algorithm,
    )
    with pytest.raises(JWTError):
        decode_access_token(forged)


def test_garbage_token_is_rejected():
    with pytest.raises(JWTError):
        decode_access_token("not.a.token")