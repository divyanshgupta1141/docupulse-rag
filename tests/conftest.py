import os
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.core.database import Base, get_db
from app.main import app
from app.services.vector_store import get_vector_store, reset_vector_store_instance


@pytest.fixture(autouse=True)
def isolate_test_environment(monkeypatch, tmp_path):
    """
    Ensures every test runs against an isolated, clean in-memory database and vector store,
    preventing state leakage and duplicate hash collisions across tests.
    """
    # Create isolated SQLite in-memory engine with StaticPool for thread-safety in TestClient
    test_engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=test_engine)
    TestingSession = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)

    def override_get_db():
        db = TestingSession()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db

    # Reset vector store to fresh in-memory collection
    reset_vector_store_instance()
    vector_store = get_vector_store()
    vector_store.ensure_collection()

    yield

    app.dependency_overrides.clear()
    Base.metadata.drop_all(bind=test_engine)
