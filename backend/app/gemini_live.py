import asyncio
import logging

import webrtcvad
from google.genai import types

from .asr import FRAME_BYTES, FRAME_MS, SAMPLE_RATE
from .engines import _get_gemini_client

logger = logging.getLogger("uvicorn.error")

DEFAULT_GEMINI_LIVE_MODEL = "gemini-3.5-transcribe-live"


class GeminiLiveTranscriber:
    """Motor de transcripcion via la Live API de Gemini (conexion persistente
    por sesion, streaming real -- no un archivo completo por segmento como los
    demas motores).

    OJO -- probado con audio real: si se deja la deteccion de actividad en
    automatico, el modelo transcribe el primer turno de habla bien pero
    despues deja de responder aunque se le siga mandando audio. Por eso los
    limites de turno (activity_start/activity_end) se marcan a mano, usando
    nuestro propio VAD frame a frame (mismo esquema de StreamBuffer) -- el
    modelo transcribe, nuestro VAD solo decide donde arranca/termina cada
    turno de habla.

    Se alimenta frame a frame (30ms) a medida que llega el audio real desde
    el operador, sin esperar a juntar el segmento completo -- eso es lo que
    permite que el resultado llegue ~0.5-0.6s despues de que la persona dejo
    de hablar (medido), en vez de esperar a cortar-y-despues-transcribir.
    """

    def __init__(
        self,
        language: str | None,
        vad_aggressiveness: int,
        silence_ms: int,
        min_speech_ms: int,
        max_buffer_seconds: float,
    ):
        self.language = language
        self.vad = webrtcvad.Vad(vad_aggressiveness)
        self.silence_frames_to_flush = max(1, round(silence_ms / FRAME_MS))
        self.min_speech_frames = max(1, round(min_speech_ms / FRAME_MS))
        # sin esto, alguien hablando sin pausas de 300ms+ (charla de verdad,
        # no un dictado con pausas marcadas) nunca cortaria turno -- mismo
        # corte forzado que ya usa StreamBuffer.
        self.max_speech_frames = max(1, round(max_buffer_seconds * 1000 / FRAME_MS))

        self.frame_buffer = bytearray()
        self.has_speech = False
        self.silence_run = 0
        self.speech_frame_count = 0
        self.total_frames = 0
        self.speech_frames = 0
        self._pending_frames: list[bytes] = []  # frames de habla sin mandar aun, mientras no se cruza min_speech_frames

        self.results: asyncio.Queue = asyncio.Queue()
        self._session_cm = None
        self._session = None
        self._recv_task: asyncio.Task | None = None
        self._turn_open = False
        self._connect_lock = asyncio.Lock()

    async def _ensure_connected(self):
        if self._session is not None:
            return
        async with self._connect_lock:
            if self._session is not None:
                return
            client = _get_gemini_client()
            config = types.LiveConnectConfig(
                response_modalities=[types.Modality.TEXT],
                input_audio_transcription=types.AudioTranscriptionConfig(
                    language_codes=[self.language] if self.language else None,
                ),
                realtime_input_config=types.RealtimeInputConfig(
                    automatic_activity_detection=types.AutomaticActivityDetection(disabled=True),
                ),
            )
            self._session_cm = client.aio.live.connect(model=DEFAULT_GEMINI_LIVE_MODEL, config=config)
            self._session = await self._session_cm.__aenter__()
            self._recv_task = asyncio.create_task(self._receive_loop())

    async def _receive_loop(self):
        # OJO -- probado con audio real: `Transcription.finished` nunca sale
        # en True para este modelo. La señal real de "esto es el final" (no
        # un interino) es que venga en el campo `input_transcription` en vez
        # de `interim_input_transcription` -- el "generation_complete=True"
        # que manda despues es informativo, no hace falta esperarlo.
        try:
            async for msg in self._session.receive():
                sc = msg.server_content
                if sc and sc.input_transcription:
                    text = (sc.input_transcription.text or "").strip()
                    raw_lang = sc.input_transcription.language_code or self.language or "?"
                    lang = raw_lang.split("-")[0].lower()  # "en-US" -> "en", si algun dia lo devuelve asi
                    if text:
                        await self.results.put((text, lang))
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.warning(f"gemini_live: se corto la conexion ({e})")

    async def feed(self, chunk: bytes) -> None:
        await self._ensure_connected()
        self.frame_buffer.extend(chunk)
        while len(self.frame_buffer) >= FRAME_BYTES:
            frame = bytes(self.frame_buffer[:FRAME_BYTES])
            del self.frame_buffer[:FRAME_BYTES]

            is_speech = self.vad.is_speech(frame, SAMPLE_RATE)
            self.total_frames += 1
            if is_speech:
                self.speech_frames += 1

            if is_speech:
                if not self.has_speech:
                    self.has_speech = True
                    self.speech_frame_count = 0
                    self._pending_frames = []
                self.silence_run = 0
                self.speech_frame_count += 1

                if self._turn_open:
                    await self._send_audio_frame(frame)
                else:
                    # todavia no cruzamos min_speech_frames -- puede ser un
                    # blip de ruido, no vale la pena abrir turno con Gemini
                    # todavia (mismo criterio que min_speech_bytes en
                    # StreamBuffer para los demas motores).
                    self._pending_frames.append(frame)
                    if self.speech_frame_count >= self.min_speech_frames:
                        await self._session.send_realtime_input(activity_start=types.ActivityStart())
                        self._turn_open = True
                        for pending in self._pending_frames:
                            await self._send_audio_frame(pending)
                        self._pending_frames = []

                if self.speech_frame_count >= self.max_speech_frames:
                    await self._end_turn()
            elif self.has_speech:
                self.speech_frame_count += 1
                if self._turn_open:
                    await self._send_audio_frame(frame)
                self.silence_run += 1
                if self.silence_run >= self.silence_frames_to_flush:
                    await self._end_turn()

    async def _send_audio_frame(self, frame: bytes) -> None:
        await self._session.send_realtime_input(
            audio=types.Blob(data=frame, mime_type=f"audio/pcm;rate={SAMPLE_RATE}")
        )

    async def _end_turn(self) -> None:
        if self._turn_open:
            await self._session.send_realtime_input(activity_end=types.ActivityEnd())
            self._turn_open = False
        self.has_speech = False
        self.silence_run = 0
        self.speech_frame_count = 0
        self._pending_frames = []

    async def flush_final(self) -> None:
        if self.has_speech:
            await self._end_turn()

    async def close(self) -> None:
        try:
            await self.flush_final()
        except Exception:
            pass
        if self._recv_task:
            self._recv_task.cancel()
        if self._session_cm:
            try:
                await self._session_cm.__aexit__(None, None, None)
            except Exception:
                pass
