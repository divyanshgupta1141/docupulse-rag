import hashlib
import io
import logging
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse
from fastapi import HTTPException
import httpx
from bs4 import BeautifulSoup
from pypdf import PdfReader
from pypdf.errors import PdfReadError
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.models.document import Document, utc_now
from app.services.ai_providers import get_embedding_provider
from app.services.vector_store import get_vector_store

logger = logging.getLogger(__name__)
settings = get_settings()


class IngestionService:
    """Handles document validation, parsing, chunking, indexing, and blue-green atomic updates."""

    def __init__(
        self,
        embedding_provider=None,
        vector_store=None,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
    ):
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._embedding_provider = embedding_provider
        self._vector_store = vector_store
        self.max_file_size_bytes = settings.MAX_FILE_SIZE_MB * 1024 * 1024

    @property
    def chunk_size(self) -> int:
        return self._chunk_size or settings.CHUNK_SIZE

    @property
    def chunk_overlap(self) -> int:
        return self._chunk_overlap or settings.CHUNK_OVERLAP

    @property
    def embedding_provider(self):
        return self._embedding_provider or get_embedding_provider()

    @embedding_provider.setter
    def embedding_provider(self, val):
        self._embedding_provider = val

    @property
    def vector_store(self):
        return self._vector_store or get_vector_store()

    @vector_store.setter
    def vector_store(self, val):
        self._vector_store = val

    def validate_and_extract_file(self, filename: str, content: bytes) -> Tuple[List[Dict[str, Any]], str]:
        """
        Validates file size, type, content, and extracts text pages.
        Returns: (pages_list, content_hash)
        """
        # 1. Size validation
        if len(content) == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        if len(content) > self.max_file_size_bytes:
            raise HTTPException(
                status_code=400,
                detail=f"File exceeds maximum allowed size of {settings.MAX_FILE_SIZE_MB}MB."
            )

        lower_filename = filename.lower()
        pages: List[Dict[str, Any]] = []

        # 2. TXT Processing
        if lower_filename.endswith(".txt"):
            try:
                text = content.decode("utf-8").strip()
            except UnicodeDecodeError:
                try:
                    text = content.decode("latin-1").strip()
                except Exception as e:
                    raise HTTPException(status_code=422, detail=f"Failed to decode text file: {e}")

            if not text:
                raise HTTPException(status_code=422, detail="Text file contains no readable content.")

            pages.append({"page_number": 1, "text": text})

        # 3. PDF Processing
        elif lower_filename.endswith(".pdf"):
            try:
                reader = PdfReader(io.BytesIO(content))
            except (PdfReadError, Exception) as e:
                raise HTTPException(status_code=400, detail=f"Invalid or corrupted PDF file: {e}")

            if len(reader.pages) == 0:
                raise HTTPException(status_code=422, detail="PDF has 0 pages.")

            for idx, page in enumerate(reader.pages):
                page_text = page.extract_text() or ""
                pages.append({"page_number": idx + 1, "text": page_text})

            full_extracted = "".join(p["text"] for p in pages).strip()
            if not full_extracted:
                raise HTTPException(
                    status_code=422,
                    detail="PDF contains no extractable text. Scanned or image-only PDFs are not supported."
                )
        else:
            raise HTTPException(
                status_code=400,
                detail="Unsupported file type. Only .pdf and .txt files are supported."
            )

        # 4. Compute SHA-256 Content Hash
        full_text = "\n".join(p["text"] for p in pages).strip()
        content_hash = hashlib.sha256(full_text.encode("utf-8")).hexdigest()

        return pages, content_hash

    async def fetch_url_content(self, url: str) -> Tuple[List[Dict[str, Any]], str, str]:
        """
        Fetches and extracts clean text from a public web page.
        Returns: (pages_list, content_hash, derived_filename)
        """
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            raise HTTPException(status_code=400, detail="Invalid URL format. Only http and https URLs are supported.")

        hostname = (parsed.hostname or "").lower()
        if hostname in ("localhost", "127.0.0.1", "0.0.0.0", "::1", "169.254.169.254"):
            raise HTTPException(status_code=400, detail="Fetching from internal or loopback addresses is forbidden.")

        try:
            async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as client:
                headers = {"User-Agent": "DocuPulse-RAG-Crawler/1.0"}
                resp = await client.get(url, headers=headers)
                if resp.status_code != 200:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Failed to fetch URL. HTTP status: {resp.status_code}"
                    )
                html = resp.text
        except httpx.RequestError as e:
            raise HTTPException(status_code=400, detail=f"Network error while fetching URL: {e}")

        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()

        text = soup.get_text(separator="\n", strip=True)
        if not text or len(text) < 20:
            raise HTTPException(
                status_code=422,
                detail="Webpage contains insufficient readable text."
            )

        title = soup.title.string.strip() if soup.title and soup.title.string else parsed.netloc
        filename = f"url_{parsed.netloc}_{title[:30].replace(' ', '_')}.txt"
        pages = [{"page_number": 1, "text": text}]
        content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()

        return pages, content_hash, filename

    def chunk_pages(
        self,
        document_id: str,
        pages: List[Dict[str, Any]],
        chunk_size: int,
        chunk_overlap: int,
        version: int = 1,
    ) -> List[Dict[str, Any]]:
        """Splits page text into overlapping chunks, tracking chunk_id, page_number, and chunk_index."""
        chunks = []
        chunk_index = 0
        step = max(1, chunk_size - chunk_overlap)

        for page in pages:
            page_text = page["text"].strip()
            page_number = page["page_number"]
            if not page_text:
                continue

            # Word-boundary-aware sliding window chunking (prevents slicing tokens mid-word)
            start = 0
            text_len = len(page_text)
            while start < text_len:
                end = min(start + chunk_size, text_len)
                if end < text_len:
                    window_floor = start + int(chunk_size * 0.92)
                    space_idx = page_text.rfind(" ", window_floor, end)
                    if space_idx != -1:
                        end = space_idx

                chunk_str = page_text[start:end].strip()
                if chunk_str:
                    chunk_id = f"{document_id}_v{version}_p{page_number}_c{chunk_index}"
                    chunks.append({
                        "chunk_id": chunk_id,
                        "chunk_index": chunk_index,
                        "page_number": page_number,
                        "text": chunk_str,
                    })
                    chunk_index += 1

                if end >= text_len:
                    break
                start += step

        return chunks

    def stage_and_index_vectors(
        self,
        document_id: str,
        filename: str,
        version: int,
        content_hash: str,
        chunks: List[Dict[str, Any]],
    ) -> int:
        """Embeds text chunks and upserts them to Qdrant under the specified version."""
        if not chunks:
            return 0

        texts = [c["text"] for c in chunks]
        embeddings = self.embedding_provider.embed_documents(texts)

        vector_payloads = []
        for c, emb in zip(chunks, embeddings):
            vector_payloads.append({
                "chunk_id": c["chunk_id"],
                "document_id": document_id,
                "filename": filename,
                "version": version,
                "page_number": c["page_number"],
                "chunk_index": c["chunk_index"],
                "text": c["text"],
                "content_hash": content_hash,
                "vector": emb,
            })

        return self.vector_store.upsert_chunks(vector_payloads)

    def ingest_new_document(
        self,
        db: Session,
        filename: str,
        pages: List[Dict[str, Any]],
        content_hash: str,
    ) -> Document:
        """
        Creates a new document record, chunks, embeds, and indexes it into Qdrant.
        Checks for exact duplicate uploads.
        """
        # Detect exact duplicate upload
        duplicate = db.query(Document).filter(
            Document.content_hash == content_hash,
            Document.status == "indexed",
        ).first()

        if duplicate:
            raise HTTPException(
                status_code=409,
                detail=f"Exact duplicate document already indexed with ID: {duplicate.document_id}."
            )

        # Create record in SQLite
        doc = Document(
            filename=filename,
            version=1,
            content_hash=content_hash,
            status="processing",
            chunk_count=0,
        )
        db.add(doc)
        db.commit()
        db.refresh(doc)

        try:
            chunks = self.chunk_pages(
                document_id=doc.document_id,
                pages=pages,
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
            )
            count = self.stage_and_index_vectors(
                document_id=doc.document_id,
                filename=filename,
                version=1,
                content_hash=content_hash,
                chunks=chunks,
            )
            doc.status = "indexed"
            doc.chunk_count = count
            doc.updated_at = utc_now()
            db.commit()
            db.refresh(doc)
            return doc
        except Exception as e:
            logger.exception("Ingestion failed for document %s: %s", doc.document_id, e)
            doc.status = "failed"
            doc.updated_at = utc_now()
            db.commit()
            raise HTTPException(status_code=500, detail=f"Failed to index document: {str(e)}")

    def atomic_update_document(
        self,
        db: Session,
        document_id: str,
        filename: str,
        pages: List[Dict[str, Any]],
        content_hash: str,
    ) -> Document:
        """
        CRITICAL: Safe Atomic Update Strategy (Blue-Green Swap)
        1. Keep old version (v) active in SQLite and Qdrant while processing replacement.
        2. Parse, chunk, and embed new content with version = v + 1.
        3. Upsert the new v + 1 vectors into Qdrant.
        4. ONLY after Qdrant upsert succeeds: update SQLite document record to version = v + 1,
           status = 'indexed', and delete all vectors from Qdrant where document_id == id AND version < v + 1.
        5. If embedding/indexing fails during update, leave version = v untouched and active,
           log the failure, and return HTTP 500/502 without corrupting existing retrievable state.
        """
        doc = db.query(Document).filter(Document.document_id == document_id).first()
        if not doc:
            raise HTTPException(status_code=404, detail=f"Document with ID {document_id} not found.")

        old_version = doc.version
        new_version = old_version + 1

        logger.info(
            "Starting Blue-Green update for doc=%s: staging version=%d (active=%d)",
            document_id, new_version, old_version
        )

        try:
            # Step 2: Chunk new content for v + 1
            chunks = self.chunk_pages(
                document_id=document_id,
                pages=pages,
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                version=new_version,
            )

            # Step 3: Embed & upsert v + 1 vectors to Qdrant
            count = self.stage_and_index_vectors(
                document_id=document_id,
                filename=filename,
                version=new_version,
                content_hash=content_hash,
                chunks=chunks,
            )

            # Step 4: ONLY after Qdrant upsert succeeds:
            # Update SQLite document record
            doc.version = new_version
            doc.filename = filename
            doc.content_hash = content_hash
            doc.chunk_count = count
            doc.status = "indexed"
            doc.updated_at = utc_now()
            db.commit()
            db.refresh(doc)

        except Exception as e:
            # Step 5: Pre-commit failure & rollback handling:
            # Leave version = old_version untouched and active in SQLite, and purge any partially staged new_version points
            db.rollback()
            try:
                self.vector_store.delete_by_document(document_id=document_id, version_eq=new_version)
            except Exception:
                pass
            logger.error(
                "Blue-Green atomic update failed for doc=%s. Preserving active version=%d untouched. Error: %s",
                document_id, old_version, str(e)
            )
            raise HTTPException(
                status_code=502,
                detail=f"Atomic document update failed: {str(e)}. Previous version {old_version} remains active."
            )

        # Post-commit retirement of older vectors (version < new_version).
        # Isolated from pre-commit rollback so a cleanup error never deletes the newly committed active version.
        try:
            self.vector_store.delete_by_document(document_id=document_id, version_lt=new_version)
            logger.info(
                "Blue-Green update succeeded for doc=%s. Swapped active version to %d and purged older vectors.",
                document_id, new_version
            )
        except Exception as cleanup_err:
            logger.warning(
                "Blue-Green update committed v=%d for doc=%s, but stale vector cleanup failed (mitigated by active_versions filter): %s",
                new_version, document_id, cleanup_err
            )

        return doc

    def delete_document(self, db: Session, document_id: str) -> None:
        """Deletes all vectors from Qdrant and removes document record from SQLite."""
        doc = db.query(Document).filter(Document.document_id == document_id).first()
        if not doc:
            raise HTTPException(status_code=404, detail=f"Document with ID {document_id} not found.")

        # Hard-delete vectors from Qdrant
        self.vector_store.delete_by_document(document_id=document_id)

        # Remove from SQLite
        db.delete(doc)
        db.commit()
