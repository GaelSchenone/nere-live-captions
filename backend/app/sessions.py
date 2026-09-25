import asyncio
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from .asr import DEFAULT_WHISPER_PRESET, DEFAULT_WHISPERCPP_PRESET
from .engines import DEFAULT_ENGINE


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


class SessionManager:
    def __init__(self):
        self.sessions: dict[str, Session] = {}
        self._lock = asyncio.Lock()

    async def create(
        self,
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
            session = Session(
                id=sid,
                name=name,
                source_lang=source_lang,
                glossary=glossary or [],
                engine=resolved_engine,
                chunking=chunking or "vad",
                whisper_preset=whisper_preset or default_preset,
            )
            self.sessions[sid] = session
        return session

    def get(self, session_id: str) -> Session | None:
        return self.sessions.get(session_id)

    def list(self) -> list[Session]:
        return list(self.sessions.values())

    def remove(self, session_id: str) -> bool:
        return self.sessions.pop(session_id, None) is not None

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
