"""Alta manual de un cliente (organizador de eventos). Uso, desde backend/:

    python scripts/create_user.py
"""
import getpass
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.auth import hash_password
from app.db import SessionLocal, init_db
from app.models import User


def main():
    init_db()
    email = input("Email: ").strip()
    password = getpass.getpass("Password: ")
    if not email or not password:
        print("Email y password son obligatorios")
        return

    db = SessionLocal()
    try:
        if db.query(User).filter(User.email == email).first():
            print(f"Ya existe un usuario con email {email}")
            return
        user = User(email=email, password_hash=hash_password(password))
        db.add(user)
        db.commit()
        print(f"Usuario creado: {email} (id={user.id})")
    finally:
        db.close()


if __name__ == "__main__":
    main()
