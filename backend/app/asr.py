import asyncio
import io
import wave
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import webrtcvad
from faster_whisper import WhisperModel

from .config import settings

SAMPLE_RATE = 16000
FRAME_MS = 30
FRAME_BYTES = int(SAMPLE_RATE * FRAME_MS / 1000) * 2  # 16-bit mono PCM
FIXED_CHUNK_SECONDS = 3.0  # duracion de corte en modo "sin VAD"


def pcm_f32_to_pcm16_bytes(audio_f32: np.ndarray) -> bytes:
    return np.clip(audio_f32 * 32768.0, -32768, 32767).astype(np.int16).tobytes()


def pcm_f32_to_wav_bytes(audio_f32: np.ndarray, sample_rate: int = SAMPLE_RATE) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_f32_to_pcm16_bytes(audio_f32))
    return buf.getvalue()

_executor = ThreadPoolExecutor(max_workers=settings.whisper_workers)
_model: WhisperModel | None = None
_model_lock = asyncio.Lock()


def get_model() -> WhisperModel:
    global _model
    if _model is None:
        _model = WhisperModel(
            settings.whisper_model,
            device=settings.whisper_device,
            compute_type=settings.whisper_compute_type,
        )
    return _model


def _transcribe_sync(audio_f32: np.ndarray, language: str | None):
    model = get_model()
    segments, info = model.transcribe(
        audio_f32,
        language=language,
        vad_filter=False,  # el VAD ya lo hicimos nosotros al armar el segmento
        beam_size=1,
        condition_on_previous_text=False,
    )
    text = " ".join(s.text.strip() for s in segments).strip()
    return text, info.language


async def transcribe_chunk(audio_f32: np.ndarray, language: str | None = None):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_executor, _transcribe_sync, audio_f32, language)


_whispercpp_model = None


def get_whispercpp_model():
    global _whispercpp_model
    if _whispercpp_model is None:
        from pywhispercpp.model import Model as WhisperCppModel

        _whispercpp_model = WhisperCppModel(
            settings.whispercpp_model,
            n_threads=settings.whispercpp_threads,
            redirect_whispercpp_logs_to=None,
            print_progress=False,
            print_realtime=False,
        )
    return _whispercpp_model


def _transcribe_whispercpp_sync(audio_f32: np.ndarray, language: str | None):
    model = get_whispercpp_model()
    lang_kwargs = {"language": language} if language else {"detect_language": True}
    segments = model.transcribe(audio_f32, **lang_kwargs)
    text = " ".join(s.text.strip() for s in segments).strip()
    return text, language or "en"


async def transcribe_chunk_whispercpp(audio_f32: np.ndarray, language: str | None = None):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_executor, _transcribe_whispercpp_sync, audio_f32, language)


class StreamBuffer:
    """Acumula audio PCM16 crudo por sesion y corta segmentos de habla usando VAD.

    Recibe chunks binarios (16kHz, mono, 16-bit) desde el operador y devuelve
    arrays float32 listos para faster-whisper cada vez que detecta una pausa
    (o cuando el buffer se pasa de MAX_BUFFER_SECONDS hablando sin parar).
    """

    def __init__(self):
        # se leen de `settings` al crear el buffer (no al importar el modulo),
        # asi una sesion nueva ya usa los valores mas recientes configurados
        # desde el dashboard -- una sesion en curso sigue con los que tenia.
        self.vad = webrtcvad.Vad(settings.vad_aggressiveness)
        self.silence_frames_to_flush = max(1, round(settings.vad_silence_ms / FRAME_MS))
        self.max_buffer_bytes = int(settings.vad_max_buffer_seconds * SAMPLE_RATE * 2)
        self.min_speech_bytes = int(settings.vad_min_speech_ms / 1000 * SAMPLE_RATE * 2)

        self.frame_buffer = bytearray()
        self.speech_bytes = bytearray()
        self.silence_run = 0
        self.has_speech = False
        # contadores solo para diagnostico (ver que tanto del audio se detecta como voz)
        self.total_frames = 0
        self.speech_frames = 0

    def add_pcm16(self, chunk: bytes) -> list[np.ndarray]:
        self.frame_buffer.extend(chunk)
        segments: list[np.ndarray] = []

        while len(self.frame_buffer) >= FRAME_BYTES:
            frame = bytes(self.frame_buffer[:FRAME_BYTES])
            del self.frame_buffer[:FRAME_BYTES]

            is_speech = self.vad.is_speech(frame, SAMPLE_RATE)
            self.total_frames += 1
            if is_speech:
                self.speech_frames += 1
            if is_speech:
                self.speech_bytes.extend(frame)
                self.has_speech = True
                self.silence_run = 0
            elif self.has_speech:
                self.speech_bytes.extend(frame)
                self.silence_run += 1
                if self.silence_run >= self.silence_frames_to_flush:
                    seg = self._flush()
                    if seg is not None:
                        segments.append(seg)

            if self.has_speech and len(self.speech_bytes) >= self.max_buffer_bytes:
                seg = self._flush()
                if seg is not None:
                    segments.append(seg)

        return segments

    def flush_final(self) -> np.ndarray | None:
        if self.has_speech and len(self.speech_bytes) > 0:
            return self._flush()
        return None

    def _flush(self) -> np.ndarray | None:
        pcm = bytes(self.speech_bytes)
        self.speech_bytes = bytearray()
        self.has_speech = False
        self.silence_run = 0
        if len(pcm) < self.min_speech_bytes:
            # probablemente un blip de ruido/eco, no vale la pena mandarlo al motor
            return None
        audio_i16 = np.frombuffer(pcm, dtype=np.int16)
        return audio_i16.astype(np.float32) / 32768.0


class FixedChunker:
    """Alternativa sin VAD: corta por tiempo fijo, sin importar si hay voz o
    silencio. Sirve para diagnosticar si el problema es del VAD (mal ajustado,
    ruido) o de la captura de audio en si (si tampoco transcribe nada acá con
    silencio real habria que revisar el mic/permisos, no el VAD).
    """

    def __init__(self):
        self.buf = bytearray()

    def add_pcm16(self, chunk: bytes) -> list[np.ndarray]:
        self.buf.extend(chunk)
        segments: list[np.ndarray] = []
        max_bytes = int(FIXED_CHUNK_SECONDS * SAMPLE_RATE * 2)
        while len(self.buf) >= max_bytes:
            piece = bytes(self.buf[:max_bytes])
            del self.buf[:max_bytes]
            audio_i16 = np.frombuffer(piece, dtype=np.int16)
            segments.append(audio_i16.astype(np.float32) / 32768.0)
        return segments

    def flush_final(self) -> np.ndarray | None:
        if len(self.buf) > SAMPLE_RATE:  # al menos ~1s
            audio_i16 = np.frombuffer(bytes(self.buf), dtype=np.int16)
            self.buf = bytearray()
            return audio_i16.astype(np.float32) / 32768.0
        return None
