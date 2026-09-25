import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass
class Settings:
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    whisper_workers: int = int(os.getenv("WHISPER_WORKERS", "2"))
    # Preset default para sesiones nuevas si no se elige uno explicito (ver
    # WHISPER_PRESETS en asr.py). "cpu_medio" anda en cualquier maquina, sin GPU.
    default_whisper_preset: str = os.getenv("DEFAULT_WHISPER_PRESET", "cpu_medio")
    vad_aggressiveness: int = int(os.getenv("VAD_AGGRESSIVENESS", "2"))
    vad_silence_ms: int = int(os.getenv("VAD_SILENCE_MS", "300"))
    vad_max_buffer_seconds: float = float(os.getenv("VAD_MAX_BUFFER_SECONDS", "6.0"))
    vad_min_speech_ms: int = int(os.getenv("VAD_MIN_SPEECH_MS", "250"))

    # whisper.cpp (via pywhispercpp) -- otro motor 100% local, para comparar
    # contra faster-whisper (distinto backend de inferencia, misma familia de modelos)
    whispercpp_threads: int = int(os.getenv("WHISPERCPP_THREADS", "4"))
    # Preset default para el motor whispercpp (ver WHISPERCPP_PRESETS en asr.py)
    default_whispercpp_preset: str = os.getenv("DEFAULT_WHISPERCPP_PRESET", "wcpp_medio")

    default_engine: str = os.getenv("DEFAULT_ASR_ENGINE", "local_whisper")
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    cloud_whisper_model: str = os.getenv("CLOUD_WHISPER_MODEL", "whisper-1")
    gemini_audio_model: str = os.getenv("GEMINI_AUDIO_MODEL", "gemini-2.5-flash")

    # Google Cloud Speech-to-Text: soporta dos formas de autenticar.
    # 1) API key simple -> pega la key acá y usa la REST API directo.
    # 2) Service Account -> dejá esto vacio y apunta GOOGLE_APPLICATION_CREDENTIALS
    #    (variable de entorno estandar de Google) al archivo JSON de la cuenta de
    #    servicio; ahí se usa el SDK oficial con Application Default Credentials.
    google_cloud_api_key: str = os.getenv("GOOGLE_CLOUD_API_KEY", "")
    google_speech_model: str = os.getenv("GOOGLE_SPEECH_MODEL", "latest_short")


settings = Settings()
