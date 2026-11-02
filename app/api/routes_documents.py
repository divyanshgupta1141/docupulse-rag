from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.models.document import Document
from app.schemas.document import DocumentListResponse, DocumentResponse, URLIngestRequest
from app.services.ingestion import IngestionService

router = APIRouter(prefix="/documents", tags=["documents"])


def get_ingestion_service() -> IngestionService:
    return IngestionService()


@router.post("", response_model=DocumentResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    service: IngestionService = Depends(get_ingestion_service),
):
    """
    Upload and index a PDF or TXT document.
    Validates file size (max 10MB), file type, unreadable PDFs, and detects exact duplicate content.
    """
    content = await file.read()
    filename = file.filename or "uploaded_document.txt"

    pages, content_hash = service.validate_and_extract_file(filename=filename, content=content)
    doc = service.ingest_new_document(
        db=db,
        filename=filename,
        pages=pages,
        content_hash=content_hash,
    )
    return DocumentResponse(**doc.to_dict())


@router.post("/url", response_model=DocumentResponse, status_code=status.HTTP_201_CREATED)
async def ingest_url(
    payload: URLIngestRequest,
    db: Session = Depends(get_db),
    service: IngestionService = Depends(get_ingestion_service),
):
    """Ingest clean readable text from a public web page URL."""
    url_str = str(payload.url)
    pages, content_hash, filename = await service.fetch_url_content(url_str)
    doc = service.ingest_new_document(
        db=db,
        filename=filename,
        pages=pages,
        content_hash=content_hash,
    )
    return DocumentResponse(**doc.to_dict())


@router.get("", response_model=DocumentListResponse)
def list_documents(db: Session = Depends(get_db)):
    """List all documents with their status, version, content_hash, and timestamps."""
    docs = db.query(Document).order_by(Document.created_at.desc()).all()
    items = [DocumentResponse(**d.to_dict()) for d in docs]
    return DocumentListResponse(total=len(items), documents=items)


@router.get("/{document_id}", response_model=DocumentResponse)
def get_document(document_id: str, db: Session = Depends(get_db)):
    """Get status and metadata for a specific document."""
    doc = db.query(Document).filter(Document.document_id == document_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail=f"Document with ID {document_id} not found.")
    return DocumentResponse(**doc.to_dict())


@router.put("/{document_id}", response_model=DocumentResponse)
async def replace_document(
    document_id: str,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    service: IngestionService = Depends(get_ingestion_service),
):
    """
    CRITICAL: Atomic Blue-Green document update.
    Stages and indexes version = v + 1 into Qdrant before switching active version in SQLite.
    Purges older vectors only after successful indexing.
    Preserves version = v if processing fails.
    """
    content = await file.read()
    filename = file.filename or "updated_document.txt"

    pages, content_hash = service.validate_and_extract_file(filename=filename, content=content)
    updated_doc = service.atomic_update_document(
        db=db,
        document_id=document_id,
        filename=filename,
        pages=pages,
        content_hash=content_hash,
    )
    return DocumentResponse(**updated_doc.to_dict())


@router.delete("/{document_id}", status_code=status.HTTP_200_OK)
def delete_document(
    document_id: str,
    db: Session = Depends(get_db),
    service: IngestionService = Depends(get_ingestion_service),
):
    """Hard-delete all vectors for document_id from Qdrant and remove from SQLite."""
    service.delete_document(db=db, document_id=document_id)
    return {"status": "deleted", "document_id": document_id}
