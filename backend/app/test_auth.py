"""Self-check minimo del hashing de contraseñas. Correr con:
    python -m app.test_auth
"""
from .auth import hash_password, verify_password


def test_password_roundtrip():
    hashed = hash_password("correcta")
    assert verify_password(hashed, "correcta")
    assert not verify_password(hashed, "incorrecta")
    assert not verify_password("basura", "correcta")


if __name__ == "__main__":
    test_password_roundtrip()
    print("ok")
