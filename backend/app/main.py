import pathlib

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .routes import api, ws

BASE_DIR = pathlib.Path(__file__).parent
STATIC_DIR = BASE_DIR / "static"

NO_CACHE_HEADERS = {"Cache-Control": "no-store"}


class NoCacheStaticFiles(StaticFiles):
    """Evita que el navegador cachee HTML/CSS/JS agresivamente durante
    desarrollo activo -- si no, una pestaña abierta puede quedar con JS/CSS
    viejo que ya no matchea el HTML actual y romper el layout en formas
    confusas de diagnosticar."""

    def file_response(self, *args, **kwargs):
        response = super().file_response(*args, **kwargs)
        response.headers["Cache-Control"] = "no-store"
        return response


app = FastAPI(title="N.E.R.E — Núcleo de Escucha y Reconocimiento en Eventos")

app.include_router(api.router)
app.include_router(ws.router)

app.mount("/assets", NoCacheStaticFiles(directory=STATIC_DIR), name="assets")


@app.get("/")
async def index():
    return FileResponse(STATIC_DIR / "index.html", headers=NO_CACHE_HEADERS)


@app.get("/operator")
async def operator_page():
    return FileResponse(STATIC_DIR / "operator.html", headers=NO_CACHE_HEADERS)


@app.get("/audience")
async def audience_page():
    return FileResponse(STATIC_DIR / "audience.html", headers=NO_CACHE_HEADERS)


@app.get("/monitor")
async def monitor_page():
    return FileResponse(STATIC_DIR / "monitor.html", headers=NO_CACHE_HEADERS)


@app.get("/overlay")
async def overlay_page():
    return FileResponse(STATIC_DIR / "overlay.html", headers=NO_CACHE_HEADERS)


@app.get("/screen")
async def screen_page():
    return FileResponse(STATIC_DIR / "screen.html", headers=NO_CACHE_HEADERS)
