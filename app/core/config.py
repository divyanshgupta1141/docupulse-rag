from functools import lru_cache
from typing import Literal, Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    PROJECT_NAME: str = "DocuPulse RAG Assistant"
    VERSION: str = "0.1.0"
    ENV: str = "development"
    DEBUG: bool = True

    # SQLite Database
    SQLITE_DB_PATH: str = "sqlite:///./docupulse.db"

    # Vector Database
    QDRANT_MODE: Literal["memory", "local"] = "memory"
    QDRANT_PATH: str = "./qdrant_storage"
    QDRANT_COLLECTION: str = "document_chunks"
    VECTOR_DIM: int = 384

    # Chunking & Retrieval
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 64
    TOP_K: int = 5
    SIMILARITY_THRESHOLD: float = 0.20
    MAX_FILE_SIZE_MB: int = 10

    # AI Providers Configuration
    USE_MOCK_AI: bool = True
    EMBEDDING_PROVIDER: Literal["local", "openai", "mock"] = "mock"
    LOCAL_EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-small"

    LLM_PROVIDER: Literal["groq", "openai", "gemini", "mock"] = "mock"
    GROQ_API_KEY: Optional[str] = None
    GROQ_MODEL: str = "llama-3.3-70b-versatile"

    OPENAI_API_KEY: Optional[str] = None
    OPENAI_MODEL: str = "gpt-4o-mini"

    GEMINI_API_KEY: Optional[str] = None
    GEMINI_MODEL: str = "gemini-1.5-flash"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


@lru_cache()
def get_settings() -> Settings:
    return Settings()
