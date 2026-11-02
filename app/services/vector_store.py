import os
import uuid
from typing import Any, Dict, List, Optional
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue, Range
from app.core.config import get_settings

settings = get_settings()


class QdrantVectorStore:
    """Encapsulates Qdrant vector database interactions (in-memory or local disk)."""

    def __init__(
        self,
        mode: Optional[str] = None,
        path: Optional[str] = None,
        collection_name: Optional[str] = None,
        vector_dim: Optional[int] = None,
    ):
        self.mode = mode or settings.QDRANT_MODE
        self.path = path or settings.QDRANT_PATH
        self.collection_name = collection_name or settings.QDRANT_COLLECTION
        if vector_dim is not None:
            self.vector_dim = vector_dim
        else:
            try:
                from app.services.ai_providers import get_embedding_provider
                self.vector_dim = get_embedding_provider().dimension
            except Exception:
                self.vector_dim = settings.VECTOR_DIM

        if self.mode == "local":
            os.makedirs(self.path, exist_ok=True)
            self.client = QdrantClient(path=self.path)
        else:
            # Default zero-config in-memory mode for portable reviewer execution
            self.client = QdrantClient(":memory:")

        self.ensure_collection()

    def ensure_collection(self) -> None:
        """
        Ensure collection and payload indices exist while protecting existing indexed data:
        - If collection doesn't exist: create it.
        - If dimensions match: continue.
        - If dimensions differ and collection is empty: recreate it safely.
        - If dimensions differ and collection contains data: raise RuntimeError requiring explicit migration/re-indexing.
        """
        exists = self.client.collection_exists(self.collection_name)
        if exists:
            info = self.client.get_collection(self.collection_name)
            existing_dim = getattr(info.config.params.vectors, "size", None)  # type: ignore[union-attr]
            if existing_dim and int(existing_dim) != int(self.vector_dim):
                points_count = self.client.count(collection_name=self.collection_name, exact=True).count
                if points_count > 0:
                    raise RuntimeError(
                        f"Qdrant collection '{self.collection_name}' has vector dimension {existing_dim}, "
                        f"which conflicts with the active embedding provider dimension ({self.vector_dim}) "
                        f"and contains {points_count} indexed vector(s). Refusing to destructively recreate "
                        f"the collection; please run an explicit migration or clear/re-index the collection."
                    )
                self.client.delete_collection(self.collection_name)
                exists = False

        if not exists:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=self.vector_dim, distance=Distance.COSINE),
            )
            # Create payload indices for fast filtered queries and deletions (on-disk / server mode)
            if self.mode != "memory":
                try:
                    self.client.create_payload_index(
                        collection_name=self.collection_name,
                        field_name="document_id",
                        field_schema=models.PayloadSchemaType.KEYWORD,
                    )
                    self.client.create_payload_index(
                        collection_name=self.collection_name,
                        field_name="version",
                        field_schema=models.PayloadSchemaType.INTEGER,
                    )
                except Exception:
                    pass

    def upsert_chunks(self, chunks: List[Dict[str, Any]]) -> int:
        """
        Upsert a batch of vector chunks with full metadata into Qdrant.
        Each chunk is expected to have:
            - chunk_id: str
            - vector: List[float]
            - document_id: str
            - filename: str
            - version: int
            - page_number: int
            - chunk_index: int
            - text: str
            - content_hash: str
        """
        if not chunks:
            return 0

        points = []
        for c in chunks:
            raw_id = c["chunk_id"]
            ver = int(c.get("version", 1))
            # Generate deterministic version-isolated UUID so Blue-Green staging (v + 1)
            # never overwrites active version (v) points in-place prior to SQLite commit
            point_uuid = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{c['document_id']}_v{ver}_{raw_id}"))

            payload = {
                "chunk_id": c["chunk_id"],
                "document_id": c["document_id"],
                "filename": c.get("filename", ""),
                "version": int(c.get("version", 1)),
                "page_number": int(c.get("page_number", 1)),
                "chunk_index": int(c.get("chunk_index", 0)),
                "text": c.get("text", ""),
                "content_hash": c.get("content_hash", ""),
            }

            points.append(
                PointStruct(
                    id=point_uuid,
                    vector=c["vector"],
                    payload=payload,
                )
            )

        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
            wait=True,
        )
        return len(points)

    def search(
        self,
        query_vector: List[float],
        top_k: int = 5,
        score_threshold: Optional[float] = None,
        filter_criteria: Optional[Filter] = None,
        active_versions: Optional[Dict[str, int]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search for nearest neighbor vectors.
        If active_versions dict {doc_id: ver} is provided, filters for active document versions only.
        """
        query_filter = filter_criteria

        if active_versions:
            version_conditions = [
                Filter(
                    must=[
                        FieldCondition(key="document_id", match=MatchValue(value=doc_id)),
                        FieldCondition(key="version", match=MatchValue(value=ver)),
                    ]
                )
                for doc_id, ver in active_versions.items()
            ]
            if version_conditions:
                active_filter = Filter(should=version_conditions)
                if query_filter:
                    query_filter = Filter(must=[query_filter, active_filter])
                else:
                    query_filter = active_filter

        # Use query_points or search depending on qdrant-client version
        if hasattr(self.client, "query_points"):
            results = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                query_filter=query_filter,
                limit=top_k,
                score_threshold=score_threshold,
            ).points
        else:
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                query_filter=query_filter,
                limit=top_k,
                score_threshold=score_threshold,
            )

        hits = []
        for pt in results:
            payload = pt.payload or {}
            hits.append({
                "chunk_id": payload.get("chunk_id", str(pt.id)),
                "document_id": payload.get("document_id", ""),
                "filename": payload.get("filename", ""),
                "version": payload.get("version", 1),
                "page_number": payload.get("page_number", 1),
                "chunk_index": payload.get("chunk_index", 0),
                "text": payload.get("text", ""),
                "content_hash": payload.get("content_hash", ""),
                "similarity_score": round(float(pt.score), 4),
            })
        return hits

    def delete_by_document(
        self,
        document_id: str,
        version_lt: Optional[int] = None,
        version_eq: Optional[int] = None,
    ) -> None:
        """
        Delete vectors for a document.
        - If version_lt is set, only deletes vectors with version < version_lt (Blue-Green Swap purge).
        - If version_eq is set, only deletes vectors with version == version_eq (Rollback cleanup of staged version).
        - If neither is set, deletes all vectors belonging to document_id (Hard deletion).
        """
        must_conditions = [
            FieldCondition(key="document_id", match=MatchValue(value=document_id))
        ]
        if version_lt is not None:
            must_conditions.append(
                FieldCondition(key="version", range=Range(lt=version_lt))
            )
        if version_eq is not None:
            must_conditions.append(
                FieldCondition(key="version", match=MatchValue(value=version_eq))
            )

        del_filter = Filter(must=must_conditions)
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=models.FilterSelector(filter=del_filter),
            wait=True,
        )


# Global singleton instance
_vector_store_instance: Optional[QdrantVectorStore] = None


def get_vector_store() -> QdrantVectorStore:
    global _vector_store_instance
    if _vector_store_instance is None:
        _vector_store_instance = QdrantVectorStore()
    return _vector_store_instance


def reset_vector_store_instance() -> None:
    """Helper for testing to reset the singleton store."""
    global _vector_store_instance
    _vector_store_instance = None
