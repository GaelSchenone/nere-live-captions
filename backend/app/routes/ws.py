import time

import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from .. import engines
from ..asr import FixedChunker, StreamBuffer
from ..config import settings
from ..sessions import CaptionSegment, manager
from ..translate import translate_text

router = APIRouter()


def _pcm16_levels(data: bytes) -> tuple[float, int]:
    """RMS y pico (0-32767) del audio crudo que llega, antes de cualquier VAD.
    Sirve para diagnosticar si el problema es de captura (nivel siempre ~0)
    o de clasificacion del VAD (hay señal pero no se marca como voz)."""
    if len(data) < 2:
        return 0.0, 0
    samples = np.frombuffer(data, dtype=np.int16).astype(np.float32)
    rms = float(np.sqrt(np.mean(samples**2))) if samples.size else 0.0
    peak = int(np.max(np.abs(samples))) if samples.size else 0
    return rms, peak


def _targets_for(lang: str) -> set[str]:
    # Espanol para charlas en otros idiomas; ademas ingles cuando la charla es en espanol.
    return {"en"} if lang == "es" else {"es"}


async def _process_segment(session, seq: int, audio_f32) -> CaptionSegment | None:
    t0 = time.time()
    text, detected_lang = await engines.transcribe(session.engine, audio_f32, session.source_lang)
    if not text:
        return None

    lang = session.source_lang or detected_lang
    translations = {}
    # La traduccion es best-effort: un fallo (key faltante, cuota 429 de Gemini,
    # etc.) no puede descartar el subtitulo -- se emite igual con el texto original
    # y las traducciones que hayan salido; todo queda visible en /monitor.
    if settings.gemini_api_key:
        for tgt in _targets_for(lang):
            try:
                translations[tgt] = await translate_text(text, lang, tgt, session.glossary)
            except Exception as e:
                session.monitor_stats["errors"] += 1
                session.monitor_stats["last_error"] = f"traduccion a {tgt}: {e}"

    t1 = time.time()
    session.monitor_stats["last_latency_ms"] = round((t1 - t0) * 1000)
    return CaptionSegment(
        seq=seq,
        start_ts=t0,
        end_ts=t1,
        source_lang=lang,
        original_text=text,
        translations=translations,
    )


@router.websocket("/ws/ingest/{session_id}")
async def ingest(ws: WebSocket, session_id: str):
    await ws.accept()
    session = manager.get(session_id)
    if not session:
        await ws.close(code=4404)
        return

    buf = StreamBuffer() if session.chunking != "fixed" else FixedChunker()
    seq = 0
    try:
        while True:
            data = await ws.receive_bytes()
            session.monitor_stats["chunks_received"] += 1

            rms, peak = _pcm16_levels(data)
            session.monitor_stats["audio_rms_last"] = round(rms, 1)
            session.monitor_stats["audio_peak_max"] = max(session.monitor_stats["audio_peak_max"] or 0, peak)

            new_segments = buf.add_pcm16(data)
            total_frames = getattr(buf, "total_frames", 0)
            if total_frames:
                session.monitor_stats["speech_ratio"] = round(buf.speech_frames / total_frames, 3)
            for audio_f32 in new_segments:
                try:
                    seq += 1
                    seg = await _process_segment(session, seq, audio_f32)
                    if seg:
                        await manager.add_segment(session, seg)
                except Exception as e:
                    session.monitor_stats["errors"] += 1
                    session.monitor_stats["last_error"] = str(e)
    except WebSocketDisconnect:
        pass
    finally:
        final_audio = buf.flush_final()
        if final_audio is not None:
            try:
                seq += 1
                seg = await _process_segment(session, seq, final_audio)
                if seg:
                    await manager.add_segment(session, seg)
            except Exception as e:
                session.monitor_stats["errors"] += 1
                session.monitor_stats["last_error"] = str(e)


@router.websocket("/ws/captions/{session_id}")
async def captions(ws: WebSocket, session_id: str):
    await ws.accept()
    session = manager.get(session_id)
    if not session:
        await ws.close(code=4404)
        return

    session.audience_ws.add(ws)
    try:
        while True:
            await ws.receive_text()  # solo mantiene viva la conexion
    except WebSocketDisconnect:
        pass
    finally:
        session.audience_ws.discard(ws)
