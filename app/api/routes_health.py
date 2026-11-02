from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.database import get_db
from app.schemas.health import HealthResponse
from app.services.vector_store import get_vector_store

router = APIRouter(tags=["health"])
settings = get_settings()


@router.get("/health", response_model=HealthResponse)
def health_check(db: Session = Depends(get_db)):
    """
    Health check endpoint.
    Verifies SQLite database and Qdrant vector database connectivity without leaking credentials.
    """
    services = {}

    # Check SQLite
    try:
        db.execute(text("SELECT 1"))
        services["database"] = "healthy"
    except Exception as e:
        services["database"] = f"unhealthy: {type(e).__name__}"

    # Check Qdrant
    try:
        vector_store = get_vector_store()
        exists = vector_store.client.collection_exists(vector_store.collection_name)
        services["vector_store"] = "healthy" if exists else "initializing"
    except Exception as e:
        services["vector_store"] = f"unhealthy: {type(e).__name__}"

    # Provider status (names only, no secrets)
    services["embedding_provider"] = "mock" if settings.USE_MOCK_AI else settings.EMBEDDING_PROVIDER
    services["llm_provider"] = "mock" if settings.USE_MOCK_AI else settings.LLM_PROVIDER

    overall_status = "healthy" if all("healthy" in v for v in [services["database"], services["vector_store"]]) else "degraded"

    return HealthResponse(
        status=overall_status,
        version=settings.VERSION,
        environment=settings.ENV,
        services=services,
    )
