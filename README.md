# N.E.R.E — Núcleo de Escucha y Reconocimiento en Eventos

Transcripción y traducción simultánea open source para conferencias con muchas sesiones
en paralelo (nacido en la Vibeathon de Nerdearla, pensado para ser reusable por cualquier evento).

Toma audio en vivo por sesión (charla) y genera subtítulos casi en tiempo real: transcripción
en el idioma original + traducción automática (español↔inglés), distribuidos por WebSocket
a cuantas sesiones simultáneas hagan falta.

## Por qué esta arquitectura

Las bases del desafío sugieren mandar el audio directo a un modelo multimodal (Gemini/Gemma)
por chunk. Eso funciona, pero mete 4-8s de latencia por vuelta (audio grande + modelo
multimodal grande) y ata todo a una sola API para *todo* el trabajo.

En cambio, acá **desacoplamos transcripción de traducción**:

1. **ASR** cortando segmentos con VAD (`webrtcvad`) en pausas naturales del habla. El motor de
   transcripción es intercambiable por sesión (ver [Motores de transcripción](#motores-de-transcripción-asr)):
   por defecto es local (`faster-whisper`, sin red de por medio), pero se puede elegir uno en
   la nube para comparar calidad/latencia.
2. **Traducción con Gemini Flash**, solo del texto ya transcripto (payload chico), que
   responde en cientos de milisegundos.
3. **Distribución por WebSocket**: cada sesión tiene su propio canal de audiencia; agregar
   sesiones nuevas es solo crear otro `Session` en memoria, no levantar infraestructura nueva.

Esto escala mejor a 30+ sesiones simultáneas: con el motor local, el trabajo pesado (ASR) es
local y liviano, y la única dependencia externa (Gemini) recibe texto corto, no audio.

> **100% local**: con el motor `local_whisper` (default) y cambiando `translate.py` para
> pegarle a un Gemma local (vía Ollama, por ejemplo) en vez de la API de Gemini, el sistema
> entero corre sin conexión a internet.

## Arquitectura

```
Operador (mic/audio de sala)  --PCM16 16kHz-->  WebSocket /ws/ingest/{session}
                                                      │
                                          VAD corta segmentos de habla
                                                      │
                                        faster-whisper (ASR local)
                                                      │
                                    texto transcripto + idioma detectado
                                                      │
                                    Gemini Flash (traducción de texto)
                                                      │
                                          Session.segments[] (memoria)
                                                      │
                                WebSocket /ws/captions/{session}  -->  Audiencia (N clientes)
```

- **Backend**: FastAPI (Python), un `SessionManager` en memoria orquesta N sesiones
  concurrentes, cada una con su propio buffer de audio y su propio set de WebSockets
  de audiencia.
- **Frontend**: HTML/JS vanilla servido por el mismo FastAPI (sin build step):
  - `/` — crear sesiones (charlas), elegir motor de transcripción, y ver cuáles están activas.
  - `/operator?session=ID` — dashboard del expositor: captura audio del mic/sala, espectrograma
    en vivo para verificar que el audio llega, y transcripción local de verificación.
  - `/audience?session=ID` — la audiencia elige charla + idioma y ve los subtítulos en vivo.
  - `/overlay?session=ID` — ventana de solo texto (color/fondo/tamaño configurables, con
    fondo transparente) para proyectar o usar como Browser Source en OBS/vMix.
  - `/monitor` — panel para el equipo de producción: estado, motor, % de audio detectado
    como voz, latencia, errores por sesión.

## Cómo correrlo

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# completá GEMINI_API_KEY en .env (conseguila en https://aistudio.google.com/apikey)

uvicorn app.main:app --reload --port 8000
```

Abrí `http://localhost:8000`, creá una sesión, y desde otra pestaña/dispositivo abrí
`/operator?session=<id>` para empezar a transmitir audio, y `/audience?session=<id>`
para ver los subtítulos.

La primera vez que se transcribe algo, `faster-whisper` descarga el modelo configurado
(`WHISPER_MODEL=small` por defecto — balance razonable entre velocidad y calidad en CPU;
`base` es más rápido/menos preciso, `medium`/`large-v3` mejor calidad si hay GPU).

> Las sesiones viven en memoria (no en una base de datos): si el proceso se reinicia (por
> ejemplo, `--reload` detectando un cambio de código mientras hay algo corriendo) se pierden.
> Para un evento real, no reinicies el proceso durante el show.

## Correr varias sesiones al mismo tiempo

Cada charla es una `Session` independiente (creada vía `POST /api/sessions`). El operador de
cada sala transmite a su propio `/ws/ingest/{session_id}`, y cada uno corre su propio buffer
de VAD. El modelo de Whisper se comparte pero las llamadas de transcripción corren en un
`ThreadPoolExecutor` (`WHISPER_WORKERS` en `.env`), así que varias sesiones pueden transcribir
en paralelo sin bloquearse entre sí. Para 30+ sesiones reales en producción, lo recomendable
es escalar horizontalmente: correr varios workers/procesos (o contenedores) detrás de un
balanceador, cada uno sirviendo un subconjunto de sesiones.

## Motores de transcripción (ASR)

Cada sesión elige su motor de forma independiente (dropdown al crearla en `/`), para poder
comparar calidad/latencia entre varias sesiones de prueba en simultáneo:

| Motor | Dónde corre | Credencial necesaria | Notas |
|---|---|---|---|
| `local_whisper` (default) | Local (`faster-whisper`, CTranslate2) | ninguna | Sin latencia de red, corre sin internet. |
| `whispercpp` | Local (`whisper.cpp` vía `pywhispercpp`) | ninguna | Otro backend de inferencia 100% local, para comparar velocidad/calidad contra `faster-whisper` con el mismo tipo de modelo. Descarga el `.bin` ggml la primera vez. |
| `cloud_whisper` | Nube (API de OpenAI) | `OPENAI_API_KEY` | `whisper-1` devuelve idioma detectado; `gpt-4o-mini-transcribe` es más nuevo pero no lo devuelve. Ojo: OpenAI ya no da créditos gratis a cuentas nuevas. |
| `gemini_audio` | Nube (Gemini, audio directo) | `GEMINI_API_KEY` (la misma de traducción) | Sin paso de VAD-a-texto intermedio: manda el audio del segmento directo a Gemini pidiendo la transcripción. Sin idioma explícito en la sesión, no hay forma de saber qué detectó — conviene fijar `source_lang` al crear la sesión. |

Si un motor en la nube falla (por ejemplo, falta la API key), el error queda visible en
`/monitor` (columna Errores, con el mensaje al pasar el mouse) en vez de fallar en silencio.

### Modo de corte sin VAD (diagnóstico)

Además del motor, cada sesión elige cómo se cortan los segmentos de audio:

- **Por pausas de voz (VAD)** — el modo normal: corta cuando `webrtcvad` detecta una pausa.
- **Por tiempo fijo (sin VAD)** — ignora el VAD y corta cada ~3s pase lo que pase. Sirve para
  diagnosticar: si en `/monitor` el `% detectado como voz` da 0% con el modo VAD, probá una
  sesión con corte fijo. Si tampoco transcribe nada ahí, el problema es de captura de audio
  (mic equivocado, permisos, volumen) — no del VAD ni del motor elegido.

## Glosario de términos técnicos

Al crear una sesión se puede pasar una lista de términos/nombres propios (`glossary`) que se
inyectan en el prompt de traducción para que Gemini no los traduzca ni los distorsione
(nombres de proyectos, speakers, tecnologías).

## Exportar transcripción

Desde el panel de monitoreo (`/monitor`) o directo por API:

```
GET /api/sessions/{id}/export?format=srt&lang=es
GET /api/sessions/{id}/export?format=vtt&lang=en
GET /api/sessions/{id}/export?format=txt&lang=original
```

## Panel de monitoreo

`/monitor` muestra, por sesión: estado (live/ended), cantidad de audiencia conectada, chunks
de audio recibidos, segmentos emitidos, última latencia de transcripción+traducción, y
contador de errores — pensado para que el equipo de producción detecte al toque si una sala
se quedó sin transmitir.

## Roadmap / opcionales pendientes

- [x] Transcripción + traducción español↔inglés en tiempo casi real
- [x] Múltiples sesiones simultáneas
- [x] Vista de audiencia con selección de charla + idioma
- [x] Exportar SRT/VTT/TXT
- [x] Panel de monitoreo básico (latencia, errores, estado)
- [x] Glosario de términos técnicos por sesión
- [x] Integración OBS/vMix: `/overlay?session=ID`, texto solo con fondo transparente,
      pensado como Browser Source.
- [x] Motor de transcripción seleccionable por sesión (local / OpenAI / Gemini audio) para
      comparar calidad y latencia.
- [ ] Portugués como idioma adicional (agregar `pt` a los targets en `ws.py` y probarlo)
- [ ] Modo 100% local (Gemma vía Ollama en vez de Gemini) para `translate.py`.
- [ ] Autenticación básica para las vistas de operador/monitoreo (hoy son abiertas).

## Licencia

Open source — pensado para que cualquier conferencia lo pueda desplegar y adaptar.
