import asyncio
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from .asr import DEFAULT_WHISPER_PRESET, DEFAULT_WHISPERCPP_PRESET
from .db import SessionLocal
from .engines import DEFAULT_ENGINE
from .models import SegmentRow, SessionRow


@dataclass
class CaptionSegment:
    seq: int
    start_ts: float
    end_ts: float
    source_lang: str
    original_text: str
    translations: dict[str, str]


@dataclass
class Session:
    id: str
    owner_id: int
    name: str
    source_lang: str | None
    glossary: list[str]
    engine: str = DEFAULT_ENGINE  # local_whisper | cloud_whisper | gemini_audio
    chunking: str = "vad"  # vad | fixed (fixed = sin VAD, para diagnostico)
    whisper_preset: str = DEFAULT_WHISPER_PRESET  # solo aplica con engine local_whisper o whispercpp
    created_at: float = field(default_factory=time.time)
    status: str = "live"  # live | ended
    segments: list[CaptionSegment] = field(default_factory=list)
    audience_ws: set = field(default_factory=set)
    ingest_ws: set = field(default_factory=set)  # WS del operador mandando audio; se cierran al terminar la sesion
    # Solo para engine == "gemini_live": conexion persistente + tarea que
    # drena sus resultados, viven mientras dura la sesion (no por segmento).
    gemini_live: Any = None
    gemini_live_drain_task: Any = None
    # Estilo de cada ventana de subtitulos, configurado por separado -- OBS
    # (/overlay) y la pantalla fisica de la sala (/screen) suelen querer looks
    # distintos (ej. OBS con fondo transparente, pantalla con fondo solido y
    # letra grande para leerse de lejos), aunque ambas se controlan desde el
    # mismo panel del operador. /audience (el celular de cada persona) no se
    # controla desde aca -- cada quien elige su propia fuente ahi.
    styles: dict = field(
        default_factory=lambda: {
            "overlay": {
                "lang": "es",
                "font_family": "system",
                "text_color": "#ffffff",
                "font_size": 56,
                # bg_mode: "transparent" (nada) | "solid" (toda la ventana) | "behind_text"
                # (una caja de fondo ajustada al texto, no a toda la ventana)
                "bg_mode": "transparent",
                "bg_color": "#000000",
                "outline_enabled": False,
                "outline_color": "#000000",
            },
            "screen": {
                "lang": "es",
                "font_family": "system",
                "text_color": "#ffffff",
                "font_size": 48,
                "bg_mode": "solid",
                "bg_color": "#000000",
                "outline_enabled": False,
                "outline_color": "#000000",
                # QR para que la audiencia entre a /audience escaneando -- solo
                # tiene sentido en /screen (la pantalla fisica de la sala), no
                # en /overlay (OBS/vMix, donde nadie escanea nada en vivo).
                "qr_enabled": True,
                "qr_size": 110,
            },
        }
    )
    monitor_stats: dict = field(
        default_factory=lambda: {
            "chunks_received": 0,
            "segments_emitted": 0,
            "errors": 0,
            "last_activity": None,
            "last_latency_ms": None,
            "speech_ratio": None,  # % de audio clasificado como voz por el VAD (diagnostico de ruido)
            "last_error": None,
            "audio_rms_last": None,  # RMS del ultimo chunk de audio crudo (0-32767)
            "audio_peak_max": None,  # pico maximo visto en toda la sesion (0-32767)
        }
    )


def _hydrate(row: SessionRow) -> Session:
    """Reconstruye una Session en RAM a partir de lo que quedo en DB -- es lo
    que permite que una sesion siga respondiendo despues de un reinicio del
    proceso, sin un sistema de resume mas elaborado: en cuanto alguien la
    vuelve a pedir (operador reconecta el ingest, audiencia reconecta
    captions), se hidrata sola."""
    session = Session(
        id=row.id,
        owner_id=row.owner_id,
        name=row.name,
        source_lang=row.source_lang,
        glossary=row.glossary or [],
        engine=row.engine,
        chunking=row.chunking,
        whisper_preset=row.whisper_preset,
        created_at=row.created_at,
        status=row.status,
    )
    session.segments = [
        CaptionSegment(
            seq=s.seq,
            start_ts=s.start_ts,
            end_ts=s.end_ts,
            source_lang=s.source_lang,
            original_text=s.original_text,
            translations=s.translations or {},
        )
        for s in row.segments
    ]
    return session


class SessionManager:
    def __init__(self):
        self.sessions: dict[str, Session] = {}
        self._lock = asyncio.Lock()

    async def create(
        self,
        owner_id: int,
        name: str,
        source_lang: str | None,
        glossary: list[str] | None,
        session_id: str | None = None,
        engine: str | None = None,
        chunking: str | None = None,
        whisper_preset: str | None = None,
    ) -> Session:
        sid = session_id or str(uuid.uuid4())[:8]
        resolved_engine = engine or DEFAULT_ENGINE
        default_preset = DEFAULT_WHISPERCPP_PRESET if resolved_engine == "whispercpp" else DEFAULT_WHISPER_PRESET
        async with self._lock:
            if sid in self.sessions:
                raise ValueError(f"session '{sid}' already exists")
            db = SessionLocal()
            try:
                if db.get(SessionRow, sid):
                    raise ValueError(f"session '{sid}' already exists")
                session = Session(
                    id=sid,
                    owner_id=owner_id,
                    name=name,
                    source_lang=source_lang,
                    glossary=glossary or [],
                    engine=resolved_engine,
                    chunking=chunking or "vad",
                    whisper_preset=whisper_preset or default_preset,
                )
                db.add(
                    SessionRow(
                        id=session.id,
                        owner_id=owner_id,
                        name=session.name,
                        source_lang=session.source_lang,
                        glossary=session.glossary,
                        engine=session.engine,
                        chunking=session.chunking,
                        whisper_preset=session.whisper_preset,
                        created_at=session.created_at,
                        status=session.status,
                    )
                )
                db.commit()
            finally:
                db.close()
            self.sessions[sid] = session
        return session

    def get(self, session_id: str) -> Session | None:
        if session_id in self.sessions:
            return self.sessions[session_id]
        db = SessionLocal()
        try:
            row = db.get(SessionRow, session_id)
            if not row:
                return None
            session = _hydrate(row)
        finally:
            db.close()
        self.sessions[session_id] = session
        return session

    def list(self, owner_id: int) -> list[Session]:
        db = SessionLocal()
        try:
            rows = db.query(SessionRow).filter(SessionRow.owner_id == owner_id).all()
            return [self.sessions.get(row.id) or _hydrate(row) for row in rows]
        finally:
            db.close()

    def remove(self, session_id: str) -> bool:
        self.sessions.pop(session_id, None)
        db = SessionLocal()
        try:
            row = db.get(SessionRow, session_id)
            if not row:
                return False
            db.delete(row)
            db.commit()
            return True
        finally:
            db.close()

    def end(self, session: Session) -> None:
        session.status = "ended"
        db = SessionLocal()
        try:
            row = db.get(SessionRow, session.id)
            if row:
                row.status = "ended"
                db.commit()
        finally:
            db.close()

    async def close_ingest(self, session: Session):
        # Corta las conexiones del operador que siguen mandando audio -- si no,
        # una sesion "terminada" sigue consumiendo CPU/GPU compartida (VAD +
        # transcripcion) por una pestaña que nadie cerro.
        for ws in list(session.ingest_ws):
            try:
                await ws.close(code=4000)
            except Exception:
                pass
        session.ingest_ws.clear()

    def style_payload(self, session: Session, target: str) -> dict:
        return {"type": f"{target}_style", **session.styles[target]}

    async def broadcast_style(self, session: Session, target: str):
        await self.broadcast(session, self.style_payload(session, target))

    async def broadcast(self, session: Session, payload: dict):
        dead = []
        for ws in list(session.audience_ws):
            try:
                await ws.send_json(payload)
            except Exception:
                dead.append(ws)
        for ws in dead:
            session.audience_ws.discard(ws)

    async def add_segment(self, session: Session, segment: CaptionSegment):
        session.segments.append(segment)
        session.monitor_stats["segments_emitted"] += 1
        session.monitor_stats["last_activity"] = time.time()
        db = SessionLocal()
        try:
            # Insert sincronico por segmento -- a esta escala (subtitulos, no
            # cientos por segundo) no hace falta cola/batch.
            # ponytail: si el volumen real lo justifica, batchear.
            db.add(
                SegmentRow(
                    session_id=session.id,
                    seq=segment.seq,
                    start_ts=segment.start_ts,
                    end_ts=segment.end_ts,
                    source_lang=segment.source_lang,
                    original_text=segment.original_text,
                    translations=segment.translations,
                )
            )
            db.commit()
        finally:
            db.close()
        payload = {
            "type": "caption",
            "seq": segment.seq,
            "session_id": session.id,
            "start": segment.start_ts,
            "end": segment.end_ts,
            "source_lang": segment.source_lang,
            "original": segment.original_text,
            "translations": segment.translations,
        }
        await self.broadcast(session, payload)


manager = SessionManager()
