import asyncio
import ctypes
import io
import pathlib
import re
import wave
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import webrtcvad
from faster_whisper import WhisperModel

from .config import settings


def _preload_cuda_libs() -> None:
    """CTranslate2 busca libcublas.so.12/libcudnn*.so.9 por nombre via dlopen,
    pero LD_LIBRARY_PATH seteado en caliente (os.environ, ya con el proceso
    corriendo) no alcanza a influir esa busqueda. Precargando los .so de los
    paquetes pip nvidia-cublas-cu12/nvidia-cudnn-cu12 con ctypes+RTLD_GLOBAL
    antes de crear el modelo, un dlopen posterior por nombre los encuentra
    porque el linker ya los tiene cargados en el proceso."""
    try:
        import nvidia.cublas
        import nvidia.cudnn
    except ImportError:
        return

    cublas_dir = pathlib.Path(next(iter(nvidia.cublas.__path__))) / "lib"
    cudnn_dir = pathlib.Path(next(iter(nvidia.cudnn.__path__))) / "lib"
    for name in (
        "libcublasLt.so.12",
        "libcublas.so.12",
        "libcudnn_graph.so.9",
        "libcudnn_ops.so.9",
        "libcudnn_engines_precompiled.so.9",
        "libcudnn_engines_runtime_compiled.so.9",
        "libcudnn_heuristic.so.9",
        "libcudnn.so.9",
    ):
        for lib_dir in (cublas_dir, cudnn_dir):
            path = lib_dir / name
            if path.exists():
                ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)
                break


_preload_cuda_libs()  # no-op si no estan instalados nvidia-cublas-cu12/nvidia-cudnn-cu12

# Presets de faster-whisper: eleccion explicita de perfil por sesion en vez de
# una sola config global -- pensado para que el mismo backend sirva tanto una
# MacBook sin GPU (perfiles cpu_*) como una maquina con GPU NVIDIA (perfiles
# gpu_*, requieren `pip install nvidia-cublas-cu12 nvidia-cudnn-cu12`).
#
# Los perfiles "pesados" NO son "maxima calidad sin importar la latencia" --
# son la mejor calidad posible DENTRO de un presupuesto de tiempo real, pensado
# para subtitulos en vivo (medido con audio real, ~11s -> tiempo de inferencia):
#   - large-v3-turbo: mismo encoder multilingue que large-v3 (ahi vive la
#     robustez a acentos), decoder podado. ~4.9s -- similar calidad a large-v3
#     pero ~30% mas rapido en esta GPU, y el beam_size le sale casi gratis
#     (el cuello de botella es el encoder, no la busqueda del decoder).
#   - distil-large-v3.5: SOLO INGLES (no multilingue), pero toda su capacidad
#     esta dedicada a un unico idioma -- buena opcion si las charlas van a ser
#     en ingles con acentos internacionales (frances, fines, etc.) y no vas a
#     necesitar detectar/transcribir otros idiomas.
WHISPER_PRESETS: dict[str, dict] = {
    "cpu_liviano": {"label": "CPU - Liviano (tiny)", "model": "tiny", "device": "cpu", "compute_type": "int8", "beam_size": 1},
    "cpu_medio": {"label": "CPU - Medio (small)", "model": "small", "device": "cpu", "compute_type": "int8", "beam_size": 1},
    "cpu_pesado": {"label": "CPU - Pesado (medium, requiere CPU multi-core)", "model": "medium", "device": "cpu", "compute_type": "int8", "beam_size": 1},
    "gpu_liviano": {"label": "GPU NVIDIA - Liviano (small)", "model": "small", "device": "cuda", "compute_type": "float16", "beam_size": 5},
    "gpu_medio": {"label": "GPU NVIDIA - Medio (medium)", "model": "medium", "device": "cuda", "compute_type": "float16", "beam_size": 5},
    "gpu_pesado": {"label": "GPU NVIDIA - Pesado, multilingue (large-v3-turbo)", "model": "large-v3-turbo", "device": "cuda", "compute_type": "float16", "beam_size": 5},
    "gpu_pesado_en": {"label": "GPU NVIDIA - Pesado, SOLO INGLES (distil-large-v3.5)", "model": "distil-large-v3.5", "device": "cuda", "compute_type": "float16", "beam_size": 5},
}
DEFAULT_WHISPER_PRESET = settings.default_whisper_preset if settings.default_whisper_preset in WHISPER_PRESETS else "cpu_medio"

# Presets de whisper.cpp (motor "whispercpp"): a diferencia de faster-whisper,
# pywhispercpp no expone un parametro de "device" en runtime -- la aceleracion
# de hardware (Metal en Apple Silicon, CUDA si se compilo con esa opcion) esta
# fija en como se compilo el binario que bajaste de PyPI, no se elige por sesion.
# Medido en CPU (Ryzen 3 4100, 4 threads) con audio real de ~11s: `medium` y
# `large-v3-turbo`, aun cuantizados a q5_0, superan tiempo real (RTF 1.2 y 1.7)
# -- por eso NO se ofrecen como preset aca, el techo util en CPU modesta es
# `small` (RTF ~0.42). En Apple Silicon con Metal podrian andar mejor, pero no
# esta validado en este proyecto -- si probas ahi y te va bien, se puede sumar
# un preset especifico.
WHISPERCPP_PRESETS: dict[str, dict] = {
    "wcpp_liviano": {"label": "whisper.cpp - Liviano (tiny)", "model": "tiny"},
    "wcpp_medio": {"label": "whisper.cpp - Medio (base)", "model": "base"},
    "wcpp_pesado": {"label": "whisper.cpp - Pesado (small)", "model": "small"},
}
DEFAULT_WHISPERCPP_PRESET = (
    settings.default_whispercpp_preset if settings.default_whispercpp_preset in WHISPERCPP_PRESETS else "wcpp_medio"
)


def whisper_preset_downloaded(preset_key: str) -> bool:
    """Chequea si el modelo de faster-whisper ya esta en el cache de
    HuggingFace, SIN dispararle una descarga (a diferencia de crear el
    modelo). Sirve para no mostrar en la UI un preset que tardaria varios
    minutos en la primera transcripcion real de un evento."""
    from faster_whisper.utils import _MODELS as _FASTER_WHISPER_REPOS
    from huggingface_hub import try_to_load_from_cache

    preset = WHISPER_PRESETS.get(preset_key)
    if not preset:
        return False
    repo = _FASTER_WHISPER_REPOS.get(preset["model"])
    if not repo:
        return False
    return try_to_load_from_cache(repo, "model.bin") is not None


def whispercpp_preset_downloaded(preset_key: str) -> bool:
    """Mismo chequeo que arriba pero para el modelo ggml de whisper.cpp."""
    from pywhispercpp import constants as pwc

    preset = WHISPERCPP_PRESETS.get(preset_key)
    if not preset:
        return False
    return (pwc.MODELS_DIR / f"ggml-{preset['model']}.bin").exists()

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
_models: dict[str, WhisperModel] = {}  # cacheados por preset -- cada uno pesa su propia RAM/VRAM


def get_model(preset_key: str) -> WhisperModel:
    preset = WHISPER_PRESETS.get(preset_key, WHISPER_PRESETS[DEFAULT_WHISPER_PRESET])
    if preset_key not in _models:
        _models[preset_key] = WhisperModel(
            preset["model"],
            device=preset["device"],
            compute_type=preset["compute_type"],
        )
    return _models[preset_key]


# Umbral para detectar "alucinaciones" de whisper: cuando el audio es silencio
# o ruido de fondo que nuestro VAD (webrtcvad) clasifico como voz por error, el
# modelo no siempre devuelve texto vacio -- a veces "alucina" frases comunes de
# su entrenamiento (tipicamente cierres de video de YouTube, tipo "Thanks for
# watching!"). OJO: en esos casos avg_logprob suele salir normal o incluso alto
# (el modelo esta "seguro" de lo que inventa), asi que no sirve como señal --
# medido con audio real: no_speech_prob ronda 0.11 en habla real y 0.88-0.94
# en silencio/ruido puro, con un salto bien limpio entre ambos.
HALLUCINATION_NO_SPEECH_PROB = 0.6
HALLUCINATION_COMPRESSION_RATIO = 2.4  # texto repetitivo/gibberish, señal aparte


def _is_hallucinated_segment(segment) -> bool:
    return segment.no_speech_prob > HALLUCINATION_NO_SPEECH_PROB or segment.compression_ratio > HALLUCINATION_COMPRESSION_RATIO


def _transcribe_sync(audio_f32: np.ndarray, language: str | None, preset_key: str):
    preset = WHISPER_PRESETS.get(preset_key, WHISPER_PRESETS[DEFAULT_WHISPER_PRESET])
    model = get_model(preset_key)
    segments, info = model.transcribe(
        audio_f32,
        language=language,
        vad_filter=False,  # el VAD ya lo hicimos nosotros al armar el segmento
        beam_size=preset["beam_size"],
        condition_on_previous_text=False,
    )
    text = " ".join(s.text.strip() for s in segments if not _is_hallucinated_segment(s)).strip()
    return text, info.language


async def transcribe_chunk(audio_f32: np.ndarray, language: str | None = None, preset_key: str = DEFAULT_WHISPER_PRESET):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_executor, _transcribe_sync, audio_f32, language, preset_key)


_whispercpp_models: dict = {}  # cacheados por preset (Model de pywhispercpp)

# whisper.cpp marca eventos de no-habla con un token entre corchetes/parentesis
# (ej. "[BLANK_AUDIO]", "[SILENCE]", "(wind blowing)") en vez de alucinar texto
# como faster-whisper -- pero si no lo filtramos, ese marcador literal se manda
# igual como si fuera el subtitulo.
_NON_SPEECH_MARKER_RE = re.compile(r"^[\[\(].*[\]\)]$")


def get_whispercpp_model(preset_key: str):
    from pywhispercpp.model import Model as WhisperCppModel

    preset = WHISPERCPP_PRESETS.get(preset_key, WHISPERCPP_PRESETS[DEFAULT_WHISPERCPP_PRESET])
    if preset_key not in _whispercpp_models:
        _whispercpp_models[preset_key] = WhisperCppModel(
            preset["model"],
            n_threads=settings.whispercpp_threads,
            redirect_whispercpp_logs_to=None,
            print_progress=False,
            print_realtime=False,
        )
    return _whispercpp_models[preset_key]


def _transcribe_whispercpp_sync(audio_f32: np.ndarray, language: str | None, preset_key: str):
    model = get_whispercpp_model(preset_key)
    # OJO: `detect_language=True` en whisper.cpp/pywhispercpp NO transcribe --
    # solo hace deteccion de idioma y devuelve 0 segmentos (bug silencioso: el
    # segmento se descarta como si no hubiera voz). Para auto-detectar Y
    # transcribir en la misma pasada hay que pasar language="auto".
    lang_kwargs = {"language": language} if language else {"language": "auto"}
    segments = model.transcribe(audio_f32, **lang_kwargs)
    parts = [s.text.strip() for s in segments if not _NON_SPEECH_MARKER_RE.match(s.text.strip())]
    text = " ".join(parts).strip()
    return text, language or "en"


async def transcribe_chunk_whispercpp(
    audio_f32: np.ndarray, language: str | None = None, preset_key: str = DEFAULT_WHISPERCPP_PRESET
):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_executor, _transcribe_whispercpp_sync, audio_f32, language, preset_key)


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
