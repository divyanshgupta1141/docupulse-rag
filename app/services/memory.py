import json
from typing import Any, Dict, List, Optional, Tuple
from sqlalchemy.orm import Session
from app.models.message import Message, utc_now


class ConversationMemoryService:
    """
    Manages bounded conversation memory persisted in SQLite.
    Enforces strict latest-10-messages sliding window per conversation and boundary continuation rules.
    """

    def add_message(
        self,
        db: Session,
        conversation_id: str,
        role: str,
        content: str,
        citations: Optional[List[Dict[str, Any]]] = None,
    ) -> Message:
        """Store a new message into SQLite for the conversation."""
        citations_json = json.dumps(citations or [])
        msg = Message(
            conversation_id=conversation_id,
            role=role,
            content=content,
            citations_json=citations_json,
            created_at=utc_now(),
        )
        db.add(msg)
        db.commit()
        db.refresh(msg)
        return msg

    def get_recent_messages(
        self,
        db: Session,
        conversation_id: str,
        limit: int = 10,
    ) -> Tuple[List[Dict[str, Any]], bool]:
        """
        Fetch strictly the latest `limit` individual messages for this conversation.
        ORDER BY created_at DESC, id DESC LIMIT 10, then reversed to chronological order.

        Boundary Rule:
        If slicing the latest 10 messages starts with an orphaned assistant message,
        we keep all 10 messages and flag `has_orphaned_boundary=True` so a lightweight continuation header
        is prepended, ensuring deterministic 10-message counting without dropping turns.
        """
        records = (
            db.query(Message)
            .filter(Message.conversation_id == conversation_id)
            .order_by(Message.created_at.desc(), Message.id.desc())
            .limit(limit)
            .all()
        )

        # Reverse to chronological order (oldest to newest among the latest 10)
        chronological_records = list(reversed(records))
        has_orphaned_boundary = False

        if len(chronological_records) == limit and chronological_records[0].role == "assistant":
            has_orphaned_boundary = True

        formatted = []
        for r in chronological_records:
            try:
                cites = json.loads(r.citations_json)
            except Exception:
                cites = []
            formatted.append({
                "id": r.id,
                "conversation_id": r.conversation_id,
                "role": r.role,
                "content": r.content,
                "citations": cites,
                "created_at": r.created_at.isoformat() if r.created_at else None,
            })

        return formatted, has_orphaned_boundary

    def format_history_for_context(
        self,
        recent_messages: List[Dict[str, Any]],
        has_orphaned_boundary: bool,
    ) -> str:
        """
        Formats the recent messages into a clean chronological text block for query contextualization,
        applying the continuation header if an orphaned assistant boundary is detected.
        """
        if not recent_messages:
            return ""

        lines = []
        if has_orphaned_boundary:
            lines.append("[System: Prior conversation truncated; continuing conversation turn.]")

        for msg in recent_messages:
            role_label = "User" if msg["role"] == "user" else "Assistant"
            lines.append(f"{role_label}: {msg['content']}")

        return "\n".join(lines)

    def get_full_history(self, db: Session, conversation_id: str) -> List[Dict[str, Any]]:
        """Fetch all messages for the specified conversation in chronological order."""
        records = (
            db.query(Message)
            .filter(Message.conversation_id == conversation_id)
            .order_by(Message.created_at.asc(), Message.id.asc())
            .all()
        )

        formatted = []
        for r in records:
            try:
                cites = json.loads(r.citations_json)
            except Exception:
                cites = []
            formatted.append({
                "id": r.id,
                "conversation_id": r.conversation_id,
                "role": r.role,
                "content": r.content,
                "citations": cites,
                "created_at": r.created_at.isoformat() if r.created_at else None,
            })
        return formatted


memory_service = ConversationMemoryService()
