import hashlib
import hmac
import secrets
import time

from fastapi import Depends, HTTPException, Request
from sqlalchemy.orm import Session

from .db import get_db
from .models import AuthToken, User

PBKDF2_ITERATIONS = 600_000
TOKEN_TTL_SECONDS = 30 * 24 * 3600  # 30 dias
COOKIE_NAME = "nere_token"


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, PBKDF2_ITERATIONS)
    return f"pbkdf2_sha256${PBKDF2_ITERATIONS}${salt.hex()}${digest.hex()}"


def verify_password(stored: str, password: str) -> bool:
    try:
        scheme, iterations, salt_hex, hash_hex = stored.split("$")
    except ValueError:
        return False
    if scheme != "pbkdf2_sha256":
        return False
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt_hex), int(iterations))
    return hmac.compare_digest(digest.hex(), hash_hex)


def create_token(db: Session, user: User) -> str:
    token = secrets.token_urlsafe(32)
    db.add(AuthToken(token=token, user_id=user.id, expires_at=time.time() + TOKEN_TTL_SECONDS))
    db.commit()
    return token


def revoke_token(db: Session, token: str) -> None:
    db.query(AuthToken).filter(AuthToken.token == token).delete()
    db.commit()


def get_current_user(request: Request, db: Session = Depends(get_db)) -> User:
    token = request.cookies.get(COOKIE_NAME)
    auth_token = db.query(AuthToken).filter(AuthToken.token == token).first() if token else None
    if not auth_token or auth_token.expires_at < time.time():
        raise HTTPException(401, "no logueado")
    user = db.get(User, auth_token.user_id)
    if not user:
        raise HTTPException(401, "no logueado")
    return user


def get_current_user_optional(request: Request, db: Session = Depends(get_db)) -> User | None:
    try:
        return get_current_user(request, db)
    except HTTPException:
        return None
