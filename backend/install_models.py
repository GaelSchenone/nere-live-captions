#!/usr/bin/env python3
"""Instalacion/precalentamiento de N.E.R.E: descarga (opcionalmente) los
paquetes de traduccion y los modelos de Whisper/whisper.cpp ANTES de un
evento real, para que nadie tenga que esperar minutos de descarga en medio
de una charla en vivo.

Todo es opcional e interactivo -- si no queres bajar nada ahora, contesta
que no a todo y el sistema va a intentar descargar los modelos on-demand
la primera vez que se usen (mas lento, pero funciona igual).

Uso: python install_models.py [--yes]
  --yes   no pregunta nada, instala todo (util para CI / setup automatizado)
"""

import argparse

from app import asr, translate
from app.config import settings


def ask(prompt: str, default_yes: bool, auto_yes: bool) -> bool:
    if auto_yes:
        return True
    suffix = " [S/n] " if default_yes else " [s/N] "
    resp = input(prompt + suffix).strip().lower()
    if not resp:
        return default_yes
    return resp in ("s", "si", "sí", "y", "yes")


def section(title: str):
    print(f"\n== {title} ==")


def install_translation(auto_yes: bool):
    section("Paquetes de traduccion (NMT local)")
    print("Idiomas soportados: en<->es, pt->es, pt->en")
    for pair in translate.PACKAGE_URLS:
        src, tgt = pair
        label = f"{src} -> {tgt}"
        if translate.package_downloaded(pair):
            print(f"  [ya instalado] {label}")
            continue
        if ask(f"Descargar paquete de traduccion {label}?", default_yes=True, auto_yes=auto_yes):
            print(f"  Descargando {label}...")
            try:
                translate.ensure_package(pair)
                print(f"  OK: {label}")
            except Exception as e:
                print(f"  ERROR descargando {label}: {e}")
        else:
            print(f"  Salteado: {label}")


def install_whisper(auto_yes: bool):
    section("Modelos de Whisper local (faster-whisper)")
    if not ask("Queres descargar/precalentar algun modelo de faster-whisper?", default_yes=False, auto_yes=auto_yes):
        print("  Salteado.")
        return
    print("Perfiles disponibles:")
    keys = list(asr.WHISPER_PRESETS.keys())
    for i, key in enumerate(keys):
        preset = asr.WHISPER_PRESETS[key]
        status = "ya descargado" if asr.whisper_preset_downloaded(key) else "sin descargar"
        marker = " (default)" if key == asr.DEFAULT_WHISPER_PRESET else ""
        print(f"  {i + 1}. {preset['label']}{marker} -- {status}")
    raw = input(
        "Elegi los numeros a descargar separados por coma (ej: 1,3), 'todos', o enter para ninguno: "
    ).strip().lower()
    if not raw:
        print("  Salteado.")
        return
    if raw == "todos":
        selected = keys
    else:
        selected = []
        for part in raw.split(","):
            part = part.strip()
            if part.isdigit() and 1 <= int(part) <= len(keys):
                selected.append(keys[int(part) - 1])
    for key in selected:
        if asr.whisper_preset_downloaded(key):
            print(f"  [ya instalado] {asr.WHISPER_PRESETS[key]['label']}")
            continue
        print(f"  Descargando/cargando {asr.WHISPER_PRESETS[key]['label']}...")
        try:
            asr.get_model(key)
            print("  OK")
        except Exception as e:
            print(f"  ERROR: {e}")


def install_whispercpp(auto_yes: bool):
    section("Modelos de whisper.cpp (pywhispercpp)")
    if not ask("Queres descargar/precalentar algun modelo de whisper.cpp?", default_yes=False, auto_yes=auto_yes):
        print("  Salteado.")
        return
    print("Perfiles disponibles:")
    keys = list(asr.WHISPERCPP_PRESETS.keys())
    for i, key in enumerate(keys):
        preset = asr.WHISPERCPP_PRESETS[key]
        status = "ya descargado" if asr.whispercpp_preset_downloaded(key) else "sin descargar"
        marker = " (default)" if key == asr.DEFAULT_WHISPERCPP_PRESET else ""
        print(f"  {i + 1}. {preset['label']}{marker} -- {status}")
    raw = input(
        "Elegi los numeros a descargar separados por coma (ej: 1,3), 'todos', o enter para ninguno: "
    ).strip().lower()
    if not raw:
        print("  Salteado.")
        return
    if raw == "todos":
        selected = keys
    else:
        selected = []
        for part in raw.split(","):
            part = part.strip()
            if part.isdigit() and 1 <= int(part) <= len(keys):
                selected.append(keys[int(part) - 1])
    for key in selected:
        if asr.whispercpp_preset_downloaded(key):
            print(f"  [ya instalado] {asr.WHISPERCPP_PRESETS[key]['label']}")
            continue
        print(f"  Descargando/cargando {asr.WHISPERCPP_PRESETS[key]['label']}...")
        try:
            asr.get_whispercpp_model(key)
            print("  OK")
        except Exception as e:
            print(f"  ERROR: {e}")


def check_api_keys():
    section("API keys")
    print(f"  GEMINI_API_KEY: {'configurada' if settings.gemini_api_key else 'NO configurada'} (gemini_audio, gemini_live)")
    print(f"  OPENAI_API_KEY: {'configurada' if settings.openai_api_key else 'NO configurada'} (cloud_whisper)")
    if not settings.gemini_api_key:
        print("  Sin GEMINI_API_KEY, gemini_live (motor default) no va a estar disponible en la UI.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--yes", action="store_true", help="instala todo sin preguntar")
    args = parser.parse_args()

    print("N.E.R.E -- instalacion y precalentamiento de modelos")
    print("Todo lo que sigue es opcional: podes cortar en cualquier momento con Ctrl+C.")

    check_api_keys()
    install_translation(args.yes)
    install_whisper(args.yes)
    install_whispercpp(args.yes)

    section("Listo")
    print("Corré el servidor con: uvicorn app.main:app --reload")
    print("y entrá a http://localhost:8000 -- los motores/perfiles que instalaste ya van a aparecer en los combos.")


if __name__ == "__main__":
    main()
