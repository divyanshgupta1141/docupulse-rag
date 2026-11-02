from typing import Any, Dict, List, Optional
from pydantic import BaseModel, HttpUrl


class DocumentResponse(BaseModel):
    document_id: str
    filename: str
    version: int
    content_hash: str
    status: str
    chunk_count: int
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class DocumentListResponse(BaseModel):
    total: int
    documents: List[DocumentResponse]


class URLIngestRequest(BaseModel):
    url: HttpUrl
