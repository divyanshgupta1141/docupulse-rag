from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class Citation(BaseModel):
    document_id: str
    filename: str
    page_number: int
    chunk_id: str
    version: int


class RetrievalTraceItem(BaseModel):
    chunk_id: str
    document_id: str
    filename: str
    version: int
    similarity_score: float
    page_number: Optional[int] = 1


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, description="User question or follow-up query")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of context chunks to retrieve")


class ChatResponse(BaseModel):
    answer: str
    insufficient_evidence: bool
    citations: List[Citation]
    retrieval_trace: List[RetrievalTraceItem]


class MessageItem(BaseModel):
    id: int
    conversation_id: str
    role: str
    content: str
    citations: List[Citation] = []
    created_at: Optional[str] = None


class ConversationHistoryResponse(BaseModel):
    conversation_id: str
    total_messages: int
    messages: List[MessageItem]
