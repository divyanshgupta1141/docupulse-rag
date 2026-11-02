import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse

from app.api.routes_chat import router as chat_router
from app.api.routes_documents import router as documents_router
from app.api.routes_health import router as health_router
from app.core.config import get_settings
from app.core.database import init_db
from app.services.vector_store import get_vector_store

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: initialize database tables and vector collection
    init_db()
    vector_store = get_vector_store()
    vector_store.ensure_collection()
    yield
    # Shutdown logic


app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="Production-grade, source-grounded RAG Document Intelligence Assistant",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register route modules
app.include_router(health_router)
app.include_router(documents_router)
app.include_router(chat_router)

# Template path
TEMPLATE_PATH = os.path.join(os.path.dirname(__file__), "templates", "index.html")


@app.get("/", response_class=HTMLResponse)
def root(request: Request):
    """
    Serves the single-file reviewer inspection UI.
    If the caller explicitly requests application/json, returns API metadata.
    """
    accept = request.headers.get("accept", "")
    if "application/json" in accept and "text/html" not in accept:
        return JSONResponse({
            "name": settings.PROJECT_NAME,
            "version": settings.VERSION,
            "docs": "/docs",
            "health": "/health",
        })

    if os.path.exists(TEMPLATE_PATH):
        with open(TEMPLATE_PATH, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())

    return HTMLResponse(content="<h1>DocuPulse RAG Assistant API</h1><p>Visit <a href='/docs'>/docs</a> for Swagger.</p>")
