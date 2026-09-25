import asyncio
import re
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import ctranslate2
import httpx
import sentencepiece as spm
from platformdirs import user_data_dir
from sacremoses import MosesDetokenizer, MosesPunctNormalizer, MosesTokenizer
from subword_nmt.apply_bpe import BPE

MODELS_DIR = Path(user_data_dir("nere-live-captions")) / "translate_models"

# Paquetes de Argos Translate (https://github.com/argosopentech/argospm-index):
# cada uno es un .zip con un modelo CTranslate2 + su tokenizer, sin la
# libreria python `argostranslate` (que arrastra torch+spacy+stanza solo para
# deteccion de limites de oracion que no necesitamos -- nuestros segmentos ya
# vienen cortados por el VAD). Bajamos y usamos el modelo directo.
PACKAGE_URLS = {
    ("en", "es"): "https://argos-net.com/v1/translate-en_es-1_0.argosmodel",
    ("es", "en"): "https://argos-net.com/v1/translate-es_en-1_9.argosmodel",
    ("pt", "es"): "https://argos-net.com/v1/translate-pt_es-1_0.argosmodel",
    ("pt", "en"): "https://argos-net.com/v1/translate-pt_en-1_9.argosmodel",
}

_executor = ThreadPoolExecutor(max_workers=2)
_backends: dict[tuple[str, str], object] = {}


def package_downloaded(pair: tuple[str, str]) -> bool:
    """Chequea si el paquete de traduccion ya esta descargado, sin dispararle
    una descarga -- mismo criterio que usa _download_and_extract para decidir
    si hace falta bajarlo."""
    if pair not in PACKAGE_URLS:
        return False
    dest = MODELS_DIR / f"{pair[0]}_{pair[1]}"
    return dest.exists() and any(dest.iterdir())


def ensure_package(pair: tuple[str, str]) -> Path:
    """Wrapper publico de _download_and_extract, para usar desde el script
    de instalacion/precalentamiento sin tocar el nombre privado del modulo."""
    return _download_and_extract(pair)


def _download_and_extract(pair: tuple[str, str]) -> Path:
    src, tgt = pair
    url = PACKAGE_URLS.get(pair)
    if not url:
        raise RuntimeError(f"No hay paquete de traduccion para {src} -> {tgt}")

    dest = MODELS_DIR / f"{src}_{tgt}"
    if dest.exists() and any(dest.iterdir()):
        return dest

    dest.mkdir(parents=True, exist_ok=True)
    zip_path = dest / "package.argosmodel"
    with httpx.stream("GET", url, follow_redirects=True, timeout=120) as resp:
        resp.raise_for_status()
        with open(zip_path, "wb") as f:
            for chunk in resp.iter_bytes():
                f.write(chunk)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest)
    zip_path.unlink()
    return dest


def _find_package_root(dest: Path) -> Path:
    # El .argosmodel trae todo adentro de una unica carpeta top-level, pero el
    # nombre de esa carpeta varia segun la version del paquete (ej. "en_es" en
    # unos, "translate-es_en-1_9" en otros) -- la buscamos en vez de asumir.
    subdirs = [d for d in dest.iterdir() if d.is_dir()]
    return subdirs[0] if len(subdirs) == 1 else dest


class _SentencePieceBackend:
    """Paquetes de Argos que tokenizan con SentencePiece (un solo modelo,
    sin normalizacion Moses)."""

    def __init__(self, model_dir: Path, sp_path: Path):
        self.translator = ctranslate2.Translator(str(model_dir), device="cpu")
        self.sp = spm.SentencePieceProcessor(model_file=str(sp_path))

    def translate(self, text: str) -> str:
        tokens = self.sp.encode(text, out_type=str)
        result = self.translator.translate_batch([tokens])
        out_tokens = result[0].hypotheses[0]
        # OJO: probado que `self.sp.decode(out_tokens)` puede unir mal las
        # piezas para algunos paquetes (ej. pt->en) -- deja el marcador "▁"
        # pegado a la palabra de al lado en vez de convertirlo en espacio,
        # de forma inconsistente. El detokenizado manual (unir + reemplazar
        # "▁" por espacio) es el estandar de SentencePiece y no depende de
        # ese comportamiento -- probado que da identico resultado en los
        # paquetes donde decode() si andaba bien, y arregla el que no.
        return "".join(out_tokens).replace("▁", " ").strip()


class _MosesBpeBackend:
    """Paquetes de Argos derivados directo de OPUS-MT/Marian: BPE clasico
    (subword-nmt) sobre texto pre-tokenizado al estilo Moses. Es el mismo
    pipeline que usa `transformers.MarianTokenizer` para estos modelos."""

    def __init__(self, model_dir: Path, bpe_path: Path, source_lang: str, target_lang: str):
        self.translator = ctranslate2.Translator(str(model_dir), device="cpu")
        with open(bpe_path, encoding="utf-8") as f:
            self.bpe = BPE(f)
        self.normalizer = MosesPunctNormalizer(lang=source_lang)
        self.tokenizer = MosesTokenizer(lang=source_lang)
        self.detokenizer = MosesDetokenizer(lang=target_lang)

    def translate(self, text: str) -> str:
        normalized = self.normalizer.normalize(text)
        tokens = self.tokenizer.tokenize(normalized, escape=True)
        bpe_applied = self.bpe.process_line(" ".join(tokens))
        src_tokens = bpe_applied.split(" ")
        result = self.translator.translate_batch([src_tokens])
        joined = " ".join(result[0].hypotheses[0]).replace("@@ ", "")
        return self.detokenizer.detokenize(joined.split(" "))


def _get_backend(source_lang: str, target_lang: str):
    pair = (source_lang, target_lang)
    if pair not in _backends:
        root = _find_package_root(_download_and_extract(pair))
        model_dir = root / "model"
        sp_path = root / "sentencepiece.model"
        bpe_path = root / "bpe.model"
        if sp_path.exists():
            _backends[pair] = _SentencePieceBackend(model_dir, sp_path)
        elif bpe_path.exists():
            _backends[pair] = _MosesBpeBackend(model_dir, bpe_path, source_lang, target_lang)
        else:
            raise RuntimeError(f"Paquete de traduccion {source_lang}->{target_lang} sin tokenizer reconocido")
    return _backends[pair]


# OJO: probado que un placeholder con letras (incluso sin ser una palabra
# real, tipo "QQZ0QQZ") puede llegar mutado del otro lado en el pipeline
# Moses+BPE (pierde/gana una letra al tokenizar y reensamblar) -- rompe la
# restauracion porque el texto ya no matchea exacto. Un placeholder puramente
# numerico sobrevive intacto en los dos esquemas de tokenizacion probados.
_GLOSSARY_PLACEHOLDER = "90{:03d}09"


def _apply_glossary(text: str, glossary: list[str]) -> tuple[str, dict[str, str]]:
    # Un NMT local no tiene forma de "pedirle" que no traduzca un termino (no
    # hay prompt). Lo sacamos del texto antes de traducir y lo devolvemos
    # despues -- best-effort: el modelo puede correr el placeholder de lugar
    # en la oracion, pero en general sobrevive bien (probado con nombres
    # propios y siglas tecnicas).
    placeholders: dict[str, str] = {}
    for i, term in enumerate(glossary):
        placeholder = _GLOSSARY_PLACEHOLDER.format(i)
        pattern = re.compile(r"\b" + re.escape(term) + r"\b", re.IGNORECASE)
        if pattern.search(text):
            text = pattern.sub(placeholder, text)
            placeholders[placeholder] = term
    return text, placeholders


def _restore_glossary(text: str, placeholders: dict[str, str]) -> str:
    for placeholder, term in placeholders.items():
        text = re.sub(re.escape(placeholder), term, text, flags=re.IGNORECASE)
    return text


def _translate_sync(text: str, source_lang: str, target_lang: str, glossary: list[str] | None) -> str:
    if not text.strip() or source_lang == target_lang:
        return text
    masked, placeholders = _apply_glossary(text, glossary or [])
    backend = _get_backend(source_lang, target_lang)
    translated = backend.translate(masked)
    return _restore_glossary(translated, placeholders)


async def translate_text(
    text: str, source_lang: str, target_lang: str, glossary: list[str] | None = None
) -> str:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_executor, _translate_sync, text, source_lang, target_lang, glossary)
