import logging
import os
import sys
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse

from app.api.routes_chat import router as chat_router
from app.api.routes_documents import router as documents_router
from app.api.routes_health import router as health_router
from app.core.config import get_settings
from app.core.database import SessionLocal, init_db
from app.models.document import Document
from app.services.ingestion import IngestionService
from app.services.vector_store import get_vector_store

logger = logging.getLogger(__name__)
settings = get_settings()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STARTER_SAMPLE_FILES = [
    os.path.join(BASE_DIR, "sample_data", "hr_leave_policy_v1.txt"),
    os.path.join(BASE_DIR, "sample_data", "engineering_handbook.pdf"),
    os.path.join(BASE_DIR, "sample_data", "clinical_safety_protocol.txt"),
]


def seed_starter_knowledge_base() -> None:
    """
    On live server startup (outside pytest), ensures the starter sample documents
    are indexed and synchronized between SQLite and in-memory Qdrant so the UI
    and suggested prompts work out of the box.
    """
    if "pytest" in sys.modules or "PYTEST_CURRENT_TEST" in os.environ:
        return

    vector_store = get_vector_store()
    try:
        point_count = vector_store.client.count(
            collection_name=vector_store.collection_name
        ).count
    except Exception:
        point_count = 0

    db = SessionLocal()
    try:
        doc_count = db.query(Document).count()
        # If in-memory Qdrant restarted with 0 vectors while SQLite still had rows, resync
        if point_count == 0 and doc_count > 0:
            db.query(Document).delete()
            db.commit()
            doc_count = 0

        if doc_count == 0:
            ingestion_service = IngestionService(vector_store=vector_store)
            for file_path in STARTER_SAMPLE_FILES:
                if not os.path.exists(file_path):
                    continue
                filename = os.path.basename(file_path)
                with open(file_path, "rb") as f:
                    content = f.read()
                pages, content_hash = ingestion_service.validate_and_extract_file(
                    filename=filename, content=content
                )
                ingestion_service.ingest_new_document(
                    db=db,
                    filename=filename,
                    pages=pages,
                    content_hash=content_hash,
                )
    except Exception as e:
        logger.warning("Starter sample seeding skipped: %s", e)
        db.rollback()
    finally:
        db.close()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: initialize database tables, vector collection, and starter knowledge base
    init_db()
    vector_store = get_vector_store()
    vector_store.ensure_collection()
    seed_starter_knowledge_base()
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
