import pathlib

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from .auth import get_current_user_optional
from .db import SessionLocal, init_db
from .routes import api, auth, ws

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
app.include_router(auth.router)
app.include_router(ws.router)

app.mount("/assets", NoCacheStaticFiles(directory=STATIC_DIR), name="assets")


@app.on_event("startup")
def _startup():
    init_db()


def _private_page(request: Request, filename: str):
    """Paginas de gestion (crear/operar/monitorear sesiones): sin cookie
    valida, redirige a /login en vez de mostrar contenido de otro cliente."""
    db = SessionLocal()
    try:
        if not get_current_user_optional(request, db):
            return RedirectResponse(f"/login?next={request.url.path}")
    finally:
        db.close()
    return FileResponse(STATIC_DIR / filename, headers=NO_CACHE_HEADERS)


@app.get("/login")
async def login_page():
    return FileResponse(STATIC_DIR / "login.html", headers=NO_CACHE_HEADERS)


@app.get("/")
async def index(request: Request):
    return _private_page(request, "index.html")


@app.get("/operator")
async def operator_page(request: Request):
    return _private_page(request, "operator.html")


@app.get("/audience")
async def audience_page():
    return FileResponse(STATIC_DIR / "audience.html", headers=NO_CACHE_HEADERS)


@app.get("/monitor")
async def monitor_page(request: Request):
    return _private_page(request, "monitor.html")


@app.get("/overlay")
async def overlay_page():
    return FileResponse(STATIC_DIR / "overlay.html", headers=NO_CACHE_HEADERS)


@app.get("/screen")
async def screen_page():
    return FileResponse(STATIC_DIR / "screen.html", headers=NO_CACHE_HEADERS)
