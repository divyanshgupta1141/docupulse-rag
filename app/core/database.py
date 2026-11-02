from typing import Generator
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from app.core.config import get_settings

settings = get_settings()

# For SQLite, check same thread must be False for multithreaded FastAPI requests
connect_args = {}
if settings.SQLITE_DB_PATH.startswith("sqlite"):
    connect_args = {"check_same_thread": False}

engine = create_engine(
    settings.SQLITE_DB_PATH,
    connect_args=connect_args,
    echo=False,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db() -> Generator[Session, None, None]:
    """Dependency for obtaining a SQLite database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """Create all database tables."""
    # Ensure all models are registered with Base.metadata before creating tables
    import app.models  # noqa: F401
    Base.metadata.create_all(bind=engine)
