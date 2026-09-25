# N.E.R.E — Núcleo de Escucha y Reconocimiento en Eventos

Transcripción y traducción simultánea open source para conferencias con muchas sesiones
en paralelo (nacido en la Vibeathon de Nerdearla, pensado para ser reusable por cualquier evento).

Toma audio en vivo por sesión (charla) y genera subtítulos casi en tiempo real: transcripción
en el idioma original, distribuidos por WebSocket a cuantas sesiones simultáneas hagan falta.

> **Traducción automática (español↔inglés): en construcción.** El traductor anterior
> pegaba a la API de Gemini por cada segmento; se sacó del camino crítico porque un LLM
> es overkill para traducir frases cortas y agregaba una dependencia externa innecesaria.
> Los subtítulos hoy salen solo en el idioma original — la idea es reemplazarlo por un
> motor de traducción automática (NMT) sin LLM, corriendo local.

## Por qué esta arquitectura

Las bases del desafío sugieren mandar el audio directo a un modelo multimodal (Gemini/Gemma)
por chunk. Eso funciona, pero mete 4-8s de latencia por vuelta (audio grande + modelo
multimodal grande) y ata todo a una sola API para *todo* el trabajo.

En cambio, acá **desacoplamos transcripción de traducción**:

1. **ASR** cortando segmentos con VAD (`webrtcvad`) en pausas naturales del habla. El motor de
   transcripción es intercambiable por sesión (ver [Motores de transcripción](#motores-de-transcripción-asr)):
   por defecto es local (`faster-whisper`, sin red de por medio), pero se puede elegir uno en
   la nube para comparar calidad/latencia.
2. **Traducción** (pendiente de reintegrar), solo del texto ya transcripto (payload chico),
   pensada para responder en cientos de milisegundos sin depender de un LLM.
3. **Distribución por WebSocket**: cada sesión tiene su propio canal de audiencia; agregar
   sesiones nuevas es solo crear otro `Session` en memoria, no levantar infraestructura nueva.

Esto escala mejor a 30+ sesiones simultáneas: con el motor local, todo el trabajo pesado (ASR
y traducción) corre local, sin depender de una API externa para el camino crítico.

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
                                    (traducción -- pendiente de reintegrar)
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
# GEMINI_API_KEY en .env solo hace falta si vas a usar el motor gemini_audio

uvicorn app.main:app --reload --port 8000
```

Abrí `http://localhost:8000`, creá una sesión, y desde otra pestaña/dispositivo abrí
`/operator?session=<id>` para empezar a transmitir audio, y `/audience?session=<id>`
para ver los subtítulos.

La primera vez que se usa un perfil, `faster-whisper` descarga el modelo correspondiente
(ver [Perfiles de Whisper local](#perfiles-de-whisper-local-cpugpu) abajo).

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
| `local_whisper` (default) | Local (`faster-whisper`, CTranslate2) | ninguna | Sin latencia de red, corre sin internet. Mejor opción con GPU NVIDIA (no tiene backend Metal para Apple Silicon). |
| `whispercpp` | Local (`whisper.cpp` vía `pywhispercpp`) | ninguna | Otro backend de inferencia 100% local. Tiene soporte nativo de Metal en macOS — mejor opción para Apple Silicon. Descarga el `.bin` ggml la primera vez. |
| `cloud_whisper` | Nube (API de OpenAI) | `OPENAI_API_KEY` | `whisper-1` devuelve idioma detectado; `gpt-4o-mini-transcribe` es más nuevo pero no lo devuelve. Ojo: OpenAI ya no da créditos gratis a cuentas nuevas. |
| `gemini_audio` | Nube (Gemini, audio directo) | `GEMINI_API_KEY` | Sin paso de VAD-a-texto intermedio: manda el audio del segmento directo a Gemini pidiendo la transcripción. Sin idioma explícito en la sesión, no hay forma de saber qué detectó — conviene fijar `source_lang` al crear la sesión. |

Si un motor en la nube falla (por ejemplo, falta la API key), el error queda visible en
`/monitor` (columna Errores, con el mensaje al pasar el mouse) en vez de fallar en silencio.

### Perfiles de Whisper (CPU/GPU, por motor)

Con los motores `local_whisper` y `whispercpp`, cada sesión además elige un **perfil**
(dropdown al crearla) que define modelo + dispositivo + precisión. Pensado para que el
mismo backend sirva tanto una laptop sin GPU (ej. una MacBook) como una máquina con GPU
NVIDIA, sin tocar código ni `.env` por sesión.

**Importante:** los perfiles "pesados" no son "máxima calidad sin importar la latencia" —
son la mejor calidad posible **dentro de un presupuesto de tiempo real**, para que el
subtítulo siga siendo útil en vivo. Con un modelo grande a `beam_size` alto, la
transcripción de un segmento puede tardar varios segundos más de lo que duró hablarlo —
técnicamente "por debajo de tiempo real" pero inutilizable como subtítulo en vivo. Todos
los perfiles de esta tabla están elegidos y medidos con audio real para evitar eso (y los
que no dieron la talla, como `whisper.cpp` con `medium`, quedaron afuera del catálogo).

#### `local_whisper` (faster-whisper / CTranslate2)

| Perfil | Modelo | Dónde corre | Notas |
|---|---|---|---|
| `cpu_liviano` | `tiny` | CPU | El más rápido, calidad más baja. Para hardware modesto o probar el pipeline. |
| `cpu_medio` (default sin GPU) | `small` | CPU | Buen balance en cualquier máquina, sin instalar nada extra. |
| `cpu_pesado` | `medium` | CPU | Mejor calidad en CPU; requiere una CPU multi-core razonable para seguir en tiempo real. |
| `gpu_liviano` | `small` | GPU NVIDIA | Rápido y con `beam_size` alto (mejor calidad que el mismo modelo en CPU). |
| `gpu_medio` | `medium` | GPU NVIDIA | Buen balance calidad/velocidad con una GPU de gama media (ej. 6GB VRAM). |
| `gpu_pesado` | `large-v3-turbo` | GPU NVIDIA | **Multilingüe** (mismo encoder que `large-v3`, decoder podado para velocidad). Mejor calidad práctica sin perder tiempo real. |
| `gpu_pesado_en` | `distil-large-v3.5` | GPU NVIDIA | **Solo inglés** (no multilingüe) — toda su capacidad dedicada a un idioma. Mejor opción si las charlas van a ser en inglés con acentos internacionales y no hace falta detectar otros idiomas. |

Los perfiles `gpu_*` necesitan `pip install nvidia-cublas-cu12 nvidia-cudnn-cu12` (~1.3GB,
por eso no están en `requirements.txt`) y una GPU NVIDIA en la máquina que corre el backend
— si no están instaladas esas libs, simplemente no aparecen como opción utilizable (fallarían
al crear el modelo). `DEFAULT_WHISPER_PRESET` en `.env` define cuál se preselecciona.

#### `whispercpp` (whisper.cpp / pywhispercpp)

A diferencia de `faster-whisper`, `pywhispercpp` no expone un parámetro de dispositivo en
runtime: la aceleración de hardware (Metal en Apple Silicon, CUDA si se compiló con esa
opción) queda fija en cómo se compiló el `.whl` que bajaste de PyPI, no se elige por sesión.

| Perfil | Modelo | Notas |
|---|---|---|
| `wcpp_liviano` | `tiny` | El más rápido. |
| `wcpp_medio` (default) | `base` | Buen balance. |
| `wcpp_pesado` | `small` | Mejor calidad que se mantiene en tiempo real en CPU x86 modesta. |

`medium` y `large-v3-turbo` (incluso cuantizados a `q5_0`) quedaron **afuera** del catálogo:
medidos en CPU dieron más lento que tiempo real (ver benchmark abajo) — no sirven para vivo
en ese tipo de hardware. `DEFAULT_WHISPERCPP_PRESET` en `.env` define cuál se preselecciona.

### Benchmark (medido con audio real, ~11s de habla)

Hardware de referencia: CPU AMD Ryzen 3 4100 (4 núcleos/8 hilos), GPU NVIDIA GTX 1660 (6GB).
Un RTF (real-time factor) menor a 1 significa que tarda menos que la duración del audio —
más bajo es mejor. Los perfiles ya excluidos del catálogo (RTF > 1) se muestran igual acá
como referencia de por qué se descartaron:

| Motor | Perfil | Modelo | Tiempo | RTF | ¿Ofrecido como preset? |
|---|---|---|---|---|---|
| local_whisper | `cpu_liviano` | tiny | 0.5s | 0.05 | Sí |
| local_whisper | `cpu_medio` | small | 2.2s | 0.20 | Sí |
| local_whisper | `cpu_pesado` | medium | 6.3s | 0.57 | Sí |
| local_whisper | `gpu_liviano` | small | 1.3s | 0.12 | Sí |
| local_whisper | `gpu_medio` | medium | 3.9s | 0.35 | Sí |
| local_whisper | `gpu_pesado` | large-v3-turbo | 4.9s | 0.45 | Sí |
| local_whisper | `gpu_pesado_en` | distil-large-v3.5 | 4.8s | 0.44 | Sí |
| whispercpp | `wcpp_liviano` | tiny | 0.6s | 0.05 | Sí |
| whispercpp | `wcpp_medio` | base | 1.2s | 0.11 | Sí |
| whispercpp | `wcpp_pesado` | small | 4.6s | 0.42 | Sí |
| whispercpp | — | medium-q5_0 | 13.4s | **1.22** | **No** (más lento que tiempo real) |
| whispercpp | — | large-v3-turbo-q5_0 | 18.7s | **1.70** | **No** (más lento que tiempo real) |

### Qué motor/perfil usar según tu hardware

- **NVIDIA con GPU (Linux/Windows)**: `local_whisper` con un perfil `gpu_*`. Es el mejor
  combo calidad/latencia de los tres, y a diferencia de `whispercpp` sí acelera con CUDA acá
  (requiere `pip install nvidia-cublas-cu12 nvidia-cudnn-cu12`, ver arriba).
- **CPU sin GPU dedicada (AMD, Intel, o una NVIDIA sin esas libs instaladas)**: `local_whisper`
  con un perfil `cpu_*`. En el benchmark, CTranslate2 (faster-whisper) tiene mejores kernels
  de CPU int8 que whisper.cpp para modelos medianos — `medium` corrió a RTF 0.57 en
  faster-whisper vs. 1.22 en whisper.cpp cuantizado, en el mismo CPU. `cpu_medio` es un
  default seguro; `cpu_pesado` si tenés 4+ núcleos.
- **Apple Silicon (M1/M2/M3/M4)**: `faster-whisper`/CTranslate2 **no tiene backend Metal** —
  en una Mac correría igual por CPU aunque el chip tenga GPU. `whisper.cpp` sí tiene soporte
  nativo de Metal, así que ahí conviene el motor `whispercpp`. **Sin validar en este proyecto**
  (no hay Mac disponible para medir) — si probás ahí y `wcpp_pesado` te sobra de margen
  gracias a Metal, es candidato a sumar un perfil más pesado específico para Apple Silicon.

### Filtrado de alucinaciones

Cuando el VAD clasifica un fragmento de silencio/ruido de fondo como voz por error, Whisper
no siempre devuelve texto vacío: a veces "alucina" frases de su entrenamiento (frecuente:
cierres de video de YouTube, tipo "Thanks for watching!"). Ambos motores filtran esto antes
de emitir el subtítulo:

- `local_whisper`: descarta el segmento si `no_speech_prob > 0.6` (con habla real ronda 0.11,
  hay un salto bien limpio) o si el texto es muy repetitivo (`compression_ratio > 2.4`). Ojo:
  `avg_logprob` no sirve como señal acá — en una alucinación suele salir normal o hasta alto,
  el modelo está "seguro" de lo que inventó.
- `whispercpp`: whisper.cpp ya tiene sus propios umbrales nativos (`no_speech_thold`/
  `entropy_thold`) y en silencio devuelve un marcador tipo `[BLANK_AUDIO]` en vez de alucinar
  texto — pero hay que descartar ese marcador explícitamente, si no se manda igual como si
  fuera el subtítulo real.

### Modo de corte sin VAD (diagnóstico)

Además del motor, cada sesión elige cómo se cortan los segmentos de audio:

- **Por pausas de voz (VAD)** — el modo normal: corta cuando `webrtcvad` detecta una pausa.
- **Por tiempo fijo (sin VAD)** — ignora el VAD y corta cada ~3s pase lo que pase. Sirve para
  diagnosticar: si en `/monitor` el `% detectado como voz` da 0% con el modo VAD, probá una
  sesión con corte fijo. Si tampoco transcribe nada ahí, el problema es de captura de audio
  (mic equivocado, permisos, volumen) — no del VAD ni del motor elegido.

## Glosario de términos técnicos

Al crear una sesión se puede pasar una lista de términos/nombres propios (`glossary`) para
que el traductor no los traduzca ni los distorsione (nombres de proyectos, speakers,
tecnologías). Se guarda por sesión; falta conectarlo al nuevo traductor una vez reintegrado.

## Exportar transcripción

Desde el panel de monitoreo (`/monitor`) o directo por API:

```
GET /api/sessions/{id}/export?format=srt&lang=es
GET /api/sessions/{id}/export?format=vtt&lang=en
GET /api/sessions/{id}/export?format=txt&lang=original
```

## Panel de monitoreo

`/monitor` muestra, por sesión: estado (live/ended), cantidad de audiencia conectada, chunks
de audio recibidos, segmentos emitidos, última latencia de transcripción, y contador de
errores — pensado para que el equipo de producción detecte al toque si una sala se quedó
sin transmitir.

## Roadmap / opcionales pendientes

- [x] Transcripción en tiempo casi real
- [x] Múltiples sesiones simultáneas
- [x] Vista de audiencia con selección de charla + idioma
- [x] Exportar SRT/VTT/TXT
- [x] Panel de monitoreo básico (latencia, errores, estado)
- [x] Integración OBS/vMix: `/overlay?session=ID`, texto solo con fondo transparente,
      pensado como Browser Source.
- [x] Motor de transcripción seleccionable por sesión (local / OpenAI / Gemini audio) para
      comparar calidad y latencia.
- [ ] Traducción automática español↔inglés con un motor NMT local (sin LLM), reintegrando
      el glosario de términos técnicos por sesión.
- [ ] Portugués como idioma adicional (agregar `pt` a los targets de traducción y probarlo)
- [ ] Autenticación básica para las vistas de operador/monitoreo (hoy son abiertas).

## Licencia

[MIT](LICENSE) — open source, pensado para que cualquier conferencia lo pueda desplegar y adaptar.
