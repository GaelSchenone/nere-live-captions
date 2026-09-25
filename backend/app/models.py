import time

from sqlalchemy import ForeignKey, JSON, String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True)
    email: Mapped[str] = mapped_column(String, unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String)
    created_at: Mapped[float] = mapped_column(default=time.time)


class SessionRow(Base):
    """Metadata persistente de una charla (la dataclass `Session` en
    sessions.py es el estado en vivo del proceso; esta fila es lo que
    sobrevive a un reinicio). Nombre distinto para no chocar con esa clase."""

    __tablename__ = "sessions"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    owner_id: Mapped[int] = mapped_column(ForeignKey("users.id"))
    name: Mapped[str] = mapped_column(String)
    source_lang: Mapped[str | None] = mapped_column(String, nullable=True)
    glossary: Mapped[list] = mapped_column(JSON, default=list)
    engine: Mapped[str] = mapped_column(String)
    chunking: Mapped[str] = mapped_column(String)
    whisper_preset: Mapped[str] = mapped_column(String)
    created_at: Mapped[float] = mapped_column(default=time.time)
    status: Mapped[str] = mapped_column(String, default="live")

    segments: Mapped[list["SegmentRow"]] = relationship(
        back_populates="session", cascade="all, delete-orphan", order_by="SegmentRow.seq"
    )


class SegmentRow(Base):
    __tablename__ = "segments"

    id: Mapped[int] = mapped_column(primary_key=True)
    session_id: Mapped[str] = mapped_column(ForeignKey("sessions.id"))
    seq: Mapped[int] = mapped_column()
    start_ts: Mapped[float] = mapped_column()
    end_ts: Mapped[float] = mapped_column()
    source_lang: Mapped[str] = mapped_column(String)
    original_text: Mapped[str] = mapped_column(String)
    translations: Mapped[dict] = mapped_column(JSON, default=dict)

    session: Mapped[SessionRow] = relationship(back_populates="segments")


class AuthToken(Base):
    __tablename__ = "auth_tokens"

    token: Mapped[str] = mapped_column(String, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"))
    created_at: Mapped[float] = mapped_column(default=time.time)
    expires_at: Mapped[float] = mapped_column()
