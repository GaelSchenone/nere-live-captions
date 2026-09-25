import numpy as np
import httpx
from google.genai import types

from . import asr
from .config import settings
from .translate import LANG_NAMES, get_client

ENGINES = ["local_whisper", "whispercpp", "cloud_whisper", "gemini_audio"]
DEFAULT_ENGINE = settings.default_engine if settings.default_engine in ENGINES else "local_whisper"


async def transcribe(engine: str, audio_f32: np.ndarray, language: str | None) -> tuple[str, str]:
    if engine == "whispercpp":
        return await asr.transcribe_chunk_whispercpp(audio_f32, language)
    if engine == "cloud_whisper":
        return await _transcribe_cloud_whisper(audio_f32, language)
    if engine == "gemini_audio":
        return await _transcribe_gemini_audio(audio_f32, language)
    return await asr.transcribe_chunk(audio_f32, language)


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


async def _transcribe_gemini_audio(audio_f32: np.ndarray, language: str | None) -> tuple[str, str]:
    if not settings.gemini_api_key:
        raise RuntimeError("Falta GEMINI_API_KEY para usar el motor gemini_audio")

    wav_bytes = asr.pcm_f32_to_wav_bytes(audio_f32)
    client = get_client()
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
    return text, language or "?"
