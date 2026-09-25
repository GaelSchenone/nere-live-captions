from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from .config import settings
from .models import Base

# SQLite serializa escrituras a nivel de archivo de todos modos (un solo
# writer a la vez) -- async (aiosqlite) no compra paralelismo real acá, solo
# complejidad. check_same_thread=False porque FastAPI puede correr el
# handler en un thread distinto al que abrió la conexión.
engine = create_engine(settings.database_url, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    Base.metadata.create_all(engine)
