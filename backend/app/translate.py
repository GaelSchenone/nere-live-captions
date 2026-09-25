from google import genai
from google.genai import types

from .config import settings

_client: genai.Client | None = None

LANG_NAMES = {
    "es": "español",
    "en": "inglés",
    "pt": "portugués",
}


def get_client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client(api_key=settings.gemini_api_key)
    return _client


async def translate_text(
    text: str,
    source_lang: str,
    target_lang: str,
    glossary: list[str] | None = None,
) -> str:
    if not text.strip():
        return ""

    client = get_client()
    src_name = LANG_NAMES.get(source_lang, source_lang)
    tgt_name = LANG_NAMES.get(target_lang, target_lang)

    glossary_block = ""
    if glossary:
        terms = "\n".join(f"- {g}" for g in glossary)
        glossary_block = (
            "Glosario de terminos tecnicos y nombres propios a preservar tal cual "
            f"(no traducir estos terminos):\n{terms}\n\n"
        )

    prompt = (
        f"Traduci el siguiente texto de {src_name} a {tgt_name}. "
        "Es un fragmento de la transcripcion en vivo de una charla de una conferencia tech, "
        "puede tener errores de transcripcion, estar cortado o incompleto. "
        "Devolve SOLO la traduccion, sin comentarios, sin comillas y sin explicaciones.\n\n"
        f"{glossary_block}"
        f"Texto: {text}"
    )

    response = await client.aio.models.generate_content(
        model=settings.translation_model,
        contents=prompt,
        config=types.GenerateContentConfig(temperature=0.2, max_output_tokens=500),
    )
    return (response.text or "").strip()
