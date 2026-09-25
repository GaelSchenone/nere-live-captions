import io
from typing import Optional

import qrcode
import qrcode.image.svg
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response
from pydantic import BaseModel

from ..asr import (
    DEFAULT_WHISPER_PRESET,
    DEFAULT_WHISPERCPP_PRESET,
    WHISPER_PRESETS,
    WHISPERCPP_PRESETS,
    whisper_preset_downloaded,
    whispercpp_preset_downloaded,
)
from ..config import settings
from ..engines import DEFAULT_ENGINE, ENGINE_LABELS, ENGINES, engine_available
from ..export import export_srt, export_txt, export_vtt
from ..sessions import manager

router = APIRouter(prefix="/api")

CHUNKING_MODES = ["vad", "fixed"]


class VadSettingsUpdate(BaseModel):
    vad_aggressiveness: Optional[int] = None
    vad_silence_ms: Optional[int] = None
    vad_max_buffer_seconds: Optional[float] = None
    vad_min_speech_ms: Optional[int] = None


def _vad_settings_dict():
    return {
        "vad_aggressiveness": settings.vad_aggressiveness,
        "vad_silence_ms": settings.vad_silence_ms,
        "vad_max_buffer_seconds": settings.vad_max_buffer_seconds,
        "vad_min_speech_ms": settings.vad_min_speech_ms,
    }


@router.get("/engines")
async def get_engines():
    return {
        "default": DEFAULT_ENGINE,
        "engines": [
            {"key": key, "label": ENGINE_LABELS[key], "available": engine_available(key)} for key in ENGINES
        ],
    }


@router.get("/whisper-presets")
async def get_whisper_presets():
    return {
        "local_whisper": {
            "default": DEFAULT_WHISPER_PRESET,
            "presets": [
                {"key": key, "label": p["label"], "downloaded": whisper_preset_downloaded(key)}
                for key, p in WHISPER_PRESETS.items()
            ],
        },
        "whispercpp": {
            "default": DEFAULT_WHISPERCPP_PRESET,
            "presets": [
                {"key": key, "label": p["label"], "downloaded": whispercpp_preset_downloaded(key)}
                for key, p in WHISPERCPP_PRESETS.items()
            ],
        },
    }


@router.get("/vad-settings")
async def get_vad_settings():
    return _vad_settings_dict()


@router.post("/vad-settings")
async def update_vad_settings(body: VadSettingsUpdate):
    # Cambia la config en memoria del proceso (no se escribe al .env). Las
    # sesiones que ya estan corriendo siguen con los valores que tenian; las
    # sesiones nuevas que se creen de aca en mas usan estos.
    if body.vad_aggressiveness is not None:
        if body.vad_aggressiveness not in (0, 1, 2, 3):
            raise HTTPException(400, "vad_aggressiveness debe ser 0, 1, 2 o 3")
        settings.vad_aggressiveness = body.vad_aggressiveness
    if body.vad_silence_ms is not None:
        if body.vad_silence_ms <= 0:
            raise HTTPException(400, "vad_silence_ms debe ser mayor a 0")
        settings.vad_silence_ms = body.vad_silence_ms
    if body.vad_max_buffer_seconds is not None:
        if body.vad_max_buffer_seconds <= 0:
            raise HTTPException(400, "vad_max_buffer_seconds debe ser mayor a 0")
        settings.vad_max_buffer_seconds = body.vad_max_buffer_seconds
    if body.vad_min_speech_ms is not None:
        if body.vad_min_speech_ms < 0:
            raise HTTPException(400, "vad_min_speech_ms debe ser 0 o mayor")
        settings.vad_min_speech_ms = body.vad_min_speech_ms
    return _vad_settings_dict()


class CreateSessionRequest(BaseModel):
    name: str
    source_lang: Optional[str] = None
    glossary: Optional[list[str]] = None
    session_id: Optional[str] = None
    engine: Optional[str] = None
    chunking: Optional[str] = None
    whisper_preset: Optional[str] = None


@router.post("/sessions")
async def create_session(req: CreateSessionRequest):
    if req.engine and req.engine not in ENGINES:
        raise HTTPException(400, f"engine debe ser uno de: {', '.join(ENGINES)}")
    if req.chunking and req.chunking not in CHUNKING_MODES:
        raise HTTPException(400, f"chunking debe ser uno de: {', '.join(CHUNKING_MODES)}")
    if req.whisper_preset:
        valid_presets = WHISPERCPP_PRESETS if req.engine == "whispercpp" else WHISPER_PRESETS
        if req.whisper_preset not in valid_presets:
            raise HTTPException(400, f"whisper_preset debe ser uno de: {', '.join(valid_presets)}")
    try:
        session = await manager.create(
            req.name, req.source_lang, req.glossary, req.session_id, req.engine, req.chunking, req.whisper_preset
        )
    except ValueError as e:
        raise HTTPException(409, str(e))
    return {
        "id": session.id,
        "name": session.name,
        "source_lang": session.source_lang,
        "engine": session.engine,
        "chunking": session.chunking,
        "whisper_preset": session.whisper_preset,
    }


@router.get("/sessions")
async def list_sessions():
    return [
        {
            "id": s.id,
            "name": s.name,
            "status": s.status,
            "source_lang": s.source_lang,
            "engine": s.engine,
            "chunking": s.chunking,
            "whisper_preset": s.whisper_preset,
            "created_at": s.created_at,
            "stats": s.monitor_stats,
            "audience_count": len(s.audience_ws),
            "segment_count": len(s.segments),
        }
        for s in manager.list()
    ]


@router.get("/sessions/{session_id}")
async def get_session(session_id: str):
    s = manager.get(session_id)
    if not s:
        raise HTTPException(404, "session not found")
    return {
        "id": s.id,
        "name": s.name,
        "status": s.status,
        "source_lang": s.source_lang,
        "engine": s.engine,
        "chunking": s.chunking,
        "whisper_preset": s.whisper_preset,
        "stats": s.monitor_stats,
        "segment_count": len(s.segments),
    }


@router.post("/sessions/{session_id}/end")
async def end_session(session_id: str):
    s = manager.get(session_id)
    if not s:
        raise HTTPException(404, "session not found")
    s.status = "ended"
    await manager.close_ingest(s)
    return {"ok": True}


@router.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    s = manager.get(session_id)
    if not s:
        raise HTTPException(404, "session not found")
    await manager.close_ingest(s)
    manager.remove(session_id)
    return {"ok": True}


class StyleUpdate(BaseModel):
    lang: Optional[str] = None
    font_family: Optional[str] = None
    text_color: Optional[str] = None
    font_size: Optional[int] = None
    bg_mode: Optional[str] = None
    bg_color: Optional[str] = None
    outline_enabled: Optional[bool] = None
    outline_color: Optional[str] = None


BG_MODES = ["transparent", "solid", "behind_text"]
STYLE_TARGETS = ["overlay", "screen"]
STYLE_FIELDS = ("lang", "font_family", "text_color", "font_size", "bg_mode", "bg_color", "outline_enabled", "outline_color")


@router.get("/sessions/{session_id}/style/{target}")
async def get_style(session_id: str, target: str):
    if target not in STYLE_TARGETS:
        raise HTTPException(400, f"target debe ser uno de: {', '.join(STYLE_TARGETS)}")
    s = manager.get(session_id)
    if not s:
        raise HTTPException(404, "session not found")
    return manager.style_payload(s, target)


@router.post("/sessions/{session_id}/style/{target}")
async def update_style(session_id: str, target: str, body: StyleUpdate):
    if target not in STYLE_TARGETS:
        raise HTTPException(400, f"target debe ser uno de: {', '.join(STYLE_TARGETS)}")
    s = manager.get(session_id)
    if not s:
        raise HTTPException(404, "session not found")
    if body.bg_mode is not None and body.bg_mode not in BG_MODES:
        raise HTTPException(400, f"bg_mode debe ser uno de: {', '.join(BG_MODES)}")

    for field in STYLE_FIELDS:
        value = getattr(body, field)
        if value is not None:
            s.styles[target][field] = value

    await manager.broadcast_style(s, target)
    return manager.style_payload(s, target)


@router.get("/sessions/{session_id}/audience-qr.svg")
async def audience_qr(session_id: str, request: Request):
    s = manager.get(session_id)
    if not s:
        raise HTTPException(404, "session not found")
    # request.base_url refleja el host/puerto con el que realmente se accedio
    # al servidor (ej. la IP de la red local del evento) -- asi el QR apunta a
    # algo que el celular de la audiencia puede alcanzar, no a "localhost".
    url = f"{request.base_url}audience?session={session_id}"
    img = qrcode.make(url, image_factory=qrcode.image.svg.SvgImage, box_size=8, border=1)
    buf = io.BytesIO()
    img.save(buf)
    return Response(content=buf.getvalue(), media_type="image/svg+xml", headers={"Cache-Control": "no-store"})


@router.get("/sessions/{session_id}/export")
async def export_session(session_id: str, format: str = "srt", lang: str = "original"):
    s = manager.get(session_id)
    if not s:
        raise HTTPException(404, "session not found")

    if format == "srt":
        content, media = export_srt(s, lang), "application/x-subrip"
    elif format == "vtt":
        content, media = export_vtt(s, lang), "text/vtt"
    elif format == "txt":
        content, media = export_txt(s, lang), "text/plain"
    else:
        raise HTTPException(400, "format debe ser srt, vtt o txt")

    return Response(
        content=content,
        media_type=media,
        headers={"Content-Disposition": f'attachment; filename="{session_id}_{lang}.{format}"'},
    )
