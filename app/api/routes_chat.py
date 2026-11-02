from fastapi import APIRouter, Depends, Path
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.schemas.chat import ChatRequest, ChatResponse, ConversationHistoryResponse, MessageItem, Citation
from app.services.memory import memory_service
from app.services.rag_engine import rag_engine

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("/{conversation_id}", response_model=ChatResponse)
def chat_turn(
    payload: ChatRequest,
    conversation_id: str = Path(..., description="Unique conversation identifier"),
    db: Session = Depends(get_db),
):
    """
    Execute a source-grounded chat turn:
    1. Evaluates latest 10 messages for pronoun/reference contextualization.
    2. Performs active document vector retrieval in Qdrant with threshold gating.
    3. Prompts LLM using strict <untrusted_document_context> prompt fence.
    4. Persists the user and assistant turns in SQLite.
    5. Returns answer, insufficient_evidence flag, citations, and retrieval trace.
    """
    response = rag_engine.chat(
        db=db,
        conversation_id=conversation_id,
        question=payload.question,
        top_k=payload.top_k,
    )
    return response


@router.get("/{conversation_id}/history", response_model=ConversationHistoryResponse)
def get_chat_history(
    conversation_id: str = Path(..., description="Unique conversation identifier"),
    db: Session = Depends(get_db),
):
    """Retrieve full chronological conversation message history with citations."""
    messages = memory_service.get_full_history(db, conversation_id)
    formatted_items = []
    for m in messages:
        cites = [Citation(**c) for c in m.get("citations", [])]
        formatted_items.append(
            MessageItem(
                id=m["id"],
                conversation_id=m["conversation_id"],
                role=m["role"],
                content=m["content"],
                citations=cites,
                created_at=m.get("created_at"),
            )
        )

    return ConversationHistoryResponse(
        conversation_id=conversation_id,
        total_messages=len(formatted_items),
        messages=formatted_items,
    )
