"""Diagnostico local: confirma si un email+password matchea contra la DB, sin
mandar la password a ningun lado. Uso, desde backend/:

    python scripts/check_login.py
"""
import getpass
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.auth import verify_password
from app.db import SessionLocal
from app.models import User


def main():
    email = input("Email: ").strip()
    password = getpass.getpass("Password: ")

    db = SessionLocal()
    try:
        user = db.query(User).filter(User.email == email).first()
    finally:
        db.close()

    if not user:
        print(f"No existe ningun usuario con email {email!r} (revisá mayúsculas/espacios)")
        return

    ok = verify_password(user.password_hash, password)
    print("Match:", ok)


if __name__ == "__main__":
    main()
