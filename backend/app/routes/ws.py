import asyncio
import time

import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from .. import engines
from ..asr import FixedChunker, StreamBuffer
from ..config import settings
from ..gemini_live import GeminiLiveTranscriber
from ..sessions import CaptionSegment, manager
from ..translate import PACKAGE_URLS, translate_text

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


# Idiomas de origen para los que de verdad tenemos un paquete de traduccion
# (derivado de PACKAGE_URLS, no hardcodeado aparte -- si se agrega un idioma
# nuevo alla, esto se actualiza solo). Sin esto, un idioma detectado que no
# reconocemos (ej. "?" cuando gemini_live/gemini_audio no devuelven idioma y
# la sesion no fijo uno) intentaba traducir igual y fallaba SIEMPRE, sin
# emitir ninguna traduccion nunca -- probado con una sesion real: 76
# segmentos, 76 errores identicos.
_KNOWN_SOURCE_LANGS = {src for src, _ in PACKAGE_URLS}


def _targets_for(lang: str) -> set[str]:
    if lang not in _KNOWN_SOURCE_LANGS:
        return set()
    if lang == "es":
        return {"en"}
    if lang == "pt":
        # a diferencia de otros idiomas, para portugues tenemos paquete
        # directo a los dos -- asi el selector de idioma de la audiencia
        # (es/en) funciona sea cual sea el que elija cada persona.
        return {"es", "en"}
    # cualquier otro idioma con paquete disponible: espanol, como default.
    return {"es"}


async def _build_caption(session, seq: int, t0: float, text: str, detected_lang: str) -> CaptionSegment:
    lang = session.source_lang or detected_lang
    translations = {}
    # Traduccion local (CTranslate2, sin API externa): best-effort igual --
    # un idioma sin paquete de traduccion disponible no puede tirar abajo el
    # subtitulo, se emite igual con el texto original y las traducciones que
    # hayan salido.
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


async def _process_segment(session, seq: int, audio_f32) -> CaptionSegment | None:
    t0 = time.time()
    text, detected_lang = await engines.transcribe(
        session.engine, audio_f32, session.source_lang, session.whisper_preset
    )
    if not text:
        return None
    return await _build_caption(session, seq, t0, text, detected_lang)


async def _drain_gemini_live(session, live: GeminiLiveTranscriber, seq_holder: list[int]):
    """Corre en paralelo al loop de ingest: los resultados de la Live API
    llegan de forma asincronica (no atados a cuando llega un chunk de audio),
    asi que se van consumiendo de la cola a medida que estan listos."""
    while True:
        text, detected_lang = await live.results.get()
        try:
            seq_holder[0] += 1
            t0 = time.time()
            seg = await _build_caption(session, seq_holder[0], t0, text, detected_lang)
            await manager.add_segment(session, seg)
        except Exception as e:
            session.monitor_stats["errors"] += 1
            session.monitor_stats["last_error"] = str(e)


async def _ingest_gemini_live(ws: WebSocket, session):
    live = GeminiLiveTranscriber(
        language=session.source_lang,
        vad_aggressiveness=settings.vad_aggressiveness,
        silence_ms=settings.vad_silence_ms,
        min_speech_ms=settings.vad_min_speech_ms,
        max_buffer_seconds=settings.vad_max_buffer_seconds,
    )
    session.gemini_live = live
    seq_holder = [0]
    drain_task = asyncio.create_task(_drain_gemini_live(session, live, seq_holder))
    session.gemini_live_drain_task = drain_task
    try:
        while True:
            data = await ws.receive_bytes()
            session.monitor_stats["chunks_received"] += 1
            rms, peak = _pcm16_levels(data)
            session.monitor_stats["audio_rms_last"] = round(rms, 1)
            session.monitor_stats["audio_peak_max"] = max(session.monitor_stats["audio_peak_max"] or 0, peak)
            if live.total_frames:
                session.monitor_stats["speech_ratio"] = round(live.speech_frames / live.total_frames, 3)
            try:
                await live.feed(data)
            except Exception as e:
                session.monitor_stats["errors"] += 1
                session.monitor_stats["last_error"] = f"gemini_live: {e}"
    except WebSocketDisconnect:
        pass
    finally:
        # ojo con el orden: si se cancela drain_task antes de darle tiempo a
        # que llegue la transcripcion final del ultimo turno abierto, esa
        # ultima frase se pierde (probado). Primero flush + esperar un toque,
        # recien ahi cancelar y cerrar la conexion con Gemini.
        try:
            await live.flush_final()
            await asyncio.sleep(2)
        except Exception:
            pass
        drain_task.cancel()
        await live.close()
        session.gemini_live = None
        session.gemini_live_drain_task = None


async def _ingest_batch(ws: WebSocket, session):
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


@router.websocket("/ws/ingest/{session_id}")
async def ingest(ws: WebSocket, session_id: str):
    await ws.accept()
    session = manager.get(session_id)
    if not session:
        await ws.close(code=4404)
        return

    session.ingest_ws.add(ws)
    try:
        if session.engine == "gemini_live":
            await _ingest_gemini_live(ws, session)
        else:
            await _ingest_batch(ws, session)
    finally:
        session.ingest_ws.discard(ws)


@router.websocket("/ws/captions/{session_id}")
async def captions(ws: WebSocket, session_id: str):
    await ws.accept()
    session = manager.get(session_id)
    if not session:
        await ws.close(code=4404)
        return

    session.audience_ws.add(ws)
    # Se mandan los dos -- cada pagina (overlay/screen) filtra por "type" y
    # se queda con el suyo, asi no hace falta que el cliente diga quien es.
    # /audience no escucha ninguno de los dos (usa su propio look fijo).
    await ws.send_json(manager.style_payload(session, "overlay"))
    await ws.send_json(manager.style_payload(session, "screen"))
    try:
        while True:
            await ws.receive_text()  # solo mantiene viva la conexion
    except WebSocketDisconnect:
        pass
    finally:
        session.audience_ws.discard(ws)
