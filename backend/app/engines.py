import numpy as np
import httpx
from google import genai
from google.genai import types

from . import asr
from .config import settings

ENGINES = ["local_whisper", "whispercpp", "cloud_whisper", "gemini_audio", "gemini_live"]
DEFAULT_ENGINE = settings.default_engine if settings.default_engine in ENGINES else "local_whisper"

ENGINE_LABELS = {
    "local_whisper": "Whisper local (faster-whisper)",
    "whispercpp": "Whisper local (whisper.cpp)",
    "cloud_whisper": "Whisper en la nube (OpenAI)",
    "gemini_audio": "Gemini (audio directo)",
    "gemini_live": "Gemini Live (streaming, baja latencia)",
}

LANG_NAMES = {
    "es": "español",
    "en": "inglés",
    "pt": "portugués",
}


def engine_available(engine: str) -> bool:
    """Si un motor no tiene nada usable ahora mismo (sin modelo descargado,
    sin API key configurada), no tiene sentido mostrarlo como opcion -- evita
    que alguien elija un motor que va a fallar o tardar minutos en el primer
    segmento de un evento real."""
    if engine == "local_whisper":
        return any(asr.whisper_preset_downloaded(k) for k in asr.WHISPER_PRESETS)
    if engine == "whispercpp":
        return any(asr.whispercpp_preset_downloaded(k) for k in asr.WHISPERCPP_PRESETS)
    if engine == "cloud_whisper":
        return bool(settings.openai_api_key)
    if engine in ("gemini_audio", "gemini_live"):
        return bool(settings.gemini_api_key)
    return False

_gemini_client: genai.Client | None = None


def _get_gemini_client() -> genai.Client:
    global _gemini_client
    if _gemini_client is None:
        _gemini_client = genai.Client(api_key=settings.gemini_api_key)
    return _gemini_client


async def transcribe(
    engine: str, audio_f32: np.ndarray, language: str | None, whisper_preset: str | None = None
) -> tuple[str, str]:
    if engine == "whispercpp":
        preset = whisper_preset if whisper_preset in asr.WHISPERCPP_PRESETS else asr.DEFAULT_WHISPERCPP_PRESET
        return await asr.transcribe_chunk_whispercpp(audio_f32, language, preset)
    if engine == "cloud_whisper":
        return await _transcribe_cloud_whisper(audio_f32, language)
    if engine == "gemini_audio":
        return await _transcribe_gemini_audio(audio_f32, language)
    preset = whisper_preset if whisper_preset in asr.WHISPER_PRESETS else asr.DEFAULT_WHISPER_PRESET
    return await asr.transcribe_chunk(audio_f32, language, preset)


async def _transcribe_cloud_whisper(audio_f32: np.ndarray, language: str | None) -> tuple[str, str]:
    if not settings.openai_api_key:
        raise RuntimeError("Falta OPENAI_API_KEY para usar el motor cloud_whisper")

    wav_bytes = asr.pcm_f32_to_wav_bytes(audio_f32)
    data = {"model": settings.cloud_whisper_model}
    if language:
        data["language"] = language
    else:
        data["response_format"] = "verbose_json"

    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            "https://api.openai.com/v1/audio/transcriptions",
            headers={"Authorization": f"Bearer {settings.openai_api_key}"},
            files={"file": ("audio.wav", wav_bytes, "audio/wav")},
            data=data,
        )
        resp.raise_for_status()
        result = resp.json()

    text = (result.get("text") or "").strip()
    detected_lang = language or result.get("language") or "en"
    return text, detected_lang


# Frases distintivas del prompt: si el modelo las devuelve tal cual (en vez de
# transcribir), es que se lo "escapó" -- ecoa la instruccion en vez de seguirla.
# Pasa con audio corto/ambiguo. Lo tratamos como si no hubiera habla.
_PROMPT_ECHO_MARKERS = ("exactamente lo que se dice", "sin comentarios ni comillas")


def _is_prompt_echo(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in _PROMPT_ECHO_MARKERS)


async def _transcribe_gemini_audio(audio_f32: np.ndarray, language: str | None) -> tuple[str, str]:
    if not settings.gemini_api_key:
        raise RuntimeError("Falta GEMINI_API_KEY para usar el motor gemini_audio")

    wav_bytes = asr.pcm_f32_to_wav_bytes(audio_f32)
    client = _get_gemini_client()
    lang_hint = f" El idioma hablado deberia ser {LANG_NAMES.get(language, language)}." if language else ""
    prompt = (
        "Transcribi EXACTAMENTE lo que se dice en este audio, en el idioma original "
        "(no traduzcas)." + lang_hint + " Si no hay habla, devolve una cadena vacia. "
        "Devolve SOLO la transcripcion, sin comentarios ni comillas."
    )

    response = await client.aio.models.generate_content(
        model=settings.gemini_audio_model,
        contents=[
            types.Part.from_bytes(data=wav_bytes, mime_type="audio/wav"),
            prompt,
        ],
        config=types.GenerateContentConfig(temperature=0.0, max_output_tokens=500),
    )
    text = (response.text or "").strip()
    if _is_prompt_echo(text):
        text = ""
    return text, language or "?"
