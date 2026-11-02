import io
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.core.database import Base
from app.main import app
from app.services.ai_providers import MockEmbeddingProvider, MockLLMProvider
from app.services.ingestion import IngestionService
from app.services.vector_store import QdrantVectorStore


@pytest.fixture
def test_db():
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()
        Base.metadata.drop_all(bind=engine)


@pytest.fixture
def vector_store():
    store = QdrantVectorStore(mode="memory", collection_name="test_collection", vector_dim=384)
    return store


@pytest.fixture
def ingestion_service(vector_store):
    service = IngestionService()
    service.vector_store = vector_store
    service.embedding_provider = MockEmbeddingProvider(dimension=384)
    return service


def test_mock_ai_providers():
    emb = MockEmbeddingProvider(dimension=384)
    vec1 = emb.embed_query("Security and VPN policy")
    vec2 = emb.embed_query("Corporate VPN and security guidelines")
    vec3 = emb.embed_query("Completely unrelated cooking recipe for chocolate cake")

    assert len(vec1) == 384
    # Check normalization
    norm1 = sum(x * x for x in vec1)
    assert pytest.approx(norm1, rel=1e-3) == 1.0

    # Cosine similarity between related should be higher than unrelated
    sim_related = sum(a * b for a, b in zip(vec1, vec2))
    sim_unrelated = sum(a * b for a, b in zip(vec1, vec3))
    assert sim_related > sim_unrelated

    llm = MockLLMProvider()
    resp = llm.generate_response(
        prompt="Question: What is required for corporate VPN access?\nContext: <untrusted_document_context>Employees must use corporate VPN for all internal access.</untrusted_document_context>"
    )
    assert "corporate VPN" in resp

    # Insufficient evidence test
    resp_empty = llm.generate_response(
        prompt="Question: What is the cake recipe?\nContext: <untrusted_document_context>Nothing here about desserts.</untrusted_document_context>"
    )
    assert "not have enough information" in resp_empty.lower()


def test_vector_store_operations(vector_store):
    chunks = [
        {
            "chunk_id": "doc1_p1_c0",
            "document_id": "doc1",
            "filename": "test.txt",
            "version": 1,
            "page_number": 1,
            "chunk_index": 0,
            "text": "First version text content.",
            "content_hash": "hash1",
            "vector": [0.1] * 384,
        },
        {
            "chunk_id": "doc1_p1_c1",
            "document_id": "doc1",
            "filename": "test.txt",
            "version": 2,
            "page_number": 1,
            "chunk_index": 0,
            "text": "Second version text content updated.",
            "content_hash": "hash2",
            "vector": [0.2] * 384,
        },
    ]

    count = vector_store.upsert_chunks(chunks)
    assert count == 2

    # Query active version 2
    res_v2 = vector_store.search(query_vector=[0.2] * 384, top_k=5, active_versions={"doc1": 2})
    assert len(res_v2) == 1
    assert res_v2[0]["version"] == 2

    # Delete version < 2 (Blue-Green purge)
    vector_store.delete_by_document("doc1", version_lt=2)
    # Search all
    res_all = vector_store.search(query_vector=[0.1] * 384, top_k=5)
    assert len(res_all) == 1
    assert res_all[0]["version"] == 2

    # Non-empty collection dimension mismatch must refuse destructive recreation
    vector_store.vector_dim = 1536
    with pytest.raises(RuntimeError, match="Refusing to destructively recreate"):
        vector_store.ensure_collection()

    # Empty collection dimension mismatch recreates safely
    vector_store.delete_by_document("doc1")
    vector_store.ensure_collection()
    info = vector_store.client.get_collection(vector_store.collection_name)
    assert info.config.params.vectors.size == 1536


def test_ingestion_and_blue_green_atomic_update(test_db, ingestion_service):
    # 1. Ingest initial document (v1)
    pages, content_hash = ingestion_service.validate_and_extract_file(
        "policy.txt", b"All employees must connect to corporate VPN."
    )
    doc = ingestion_service.ingest_new_document(
        db=test_db,
        filename="policy.txt",
        pages=pages,
        content_hash=content_hash,
    )
    assert doc.version == 1
    assert doc.status == "indexed"
    assert doc.chunk_count > 0

    # 2. Duplicate detection
    with pytest.raises(Exception) as exc_info:
        ingestion_service.ingest_new_document(
            db=test_db,
            filename="policy_duplicate.txt",
            pages=pages,
            content_hash=content_hash,
        )
    assert "Exact duplicate document already indexed" in str(exc_info.value.detail)

    # 3. Safe Blue-Green update to v2
    new_pages, new_hash = ingestion_service.validate_and_extract_file(
        "policy_v2.txt", b"Updated policy: Employees must use VPN and hardware keys."
    )
    updated_doc = ingestion_service.atomic_update_document(
        db=test_db,
        document_id=doc.document_id,
        filename="policy_v2.txt",
        pages=new_pages,
        content_hash=new_hash,
    )
    assert updated_doc.version == 2
    assert updated_doc.status == "indexed"
    assert updated_doc.filename == "policy_v2.txt"

    # Verify Qdrant has only v2 for this document
    active_search = ingestion_service.vector_store.search(
        query_vector=[0.1] * 384,
        top_k=5,
    )
    assert all(hit["version"] == 2 for hit in active_search)


def test_health_endpoint():
    with TestClient(app) as client:
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert "database" in data["services"]
        assert "vector_store" in data["services"]


def test_document_routes():
    with TestClient(app) as client:
        # 1. Upload Document
        file_bytes = b"Corporate security policy requires 2FA on all email accounts."
        upload_resp = client.post(
            "/documents",
            files={"file": ("security_policy.txt", io.BytesIO(file_bytes), "text/plain")},
        )
        assert upload_resp.status_code == 201
        doc_data = upload_resp.json()
        doc_id = doc_data["document_id"]
        assert doc_data["filename"] == "security_policy.txt"
        assert doc_data["version"] == 1
        assert doc_data["status"] == "indexed"
        assert doc_data["chunk_count"] > 0

        # 2. List Documents
        list_resp = client.get("/documents")
        assert list_resp.status_code == 200
        list_data = list_resp.json()
        assert list_data["total"] >= 1
        assert any(d["document_id"] == doc_id for d in list_data["documents"])

        # 3. Get Document
        get_resp = client.get(f"/documents/{doc_id}")
        assert get_resp.status_code == 200
        assert get_resp.json()["document_id"] == doc_id

        # 4. Atomic Blue-Green Update (PUT /documents/{id})
        updated_bytes = b"Corporate security policy v2 requires hardware FIDO2 keys and 2FA."
        put_resp = client.put(
            f"/documents/{doc_id}",
            files={"file": ("security_policy_v2.txt", io.BytesIO(updated_bytes), "text/plain")},
        )
        assert put_resp.status_code == 200
        updated_data = put_resp.json()
        assert updated_data["version"] == 2
        assert updated_data["filename"] == "security_policy_v2.txt"

        # 5. Delete Document
        del_resp = client.delete(f"/documents/{doc_id}")
        assert del_resp.status_code == 200
        assert del_resp.json()["status"] == "deleted"

        # Verify 404 after deletion
        get_deleted = client.get(f"/documents/{doc_id}")
        assert get_deleted.status_code == 404


def test_pdf_validations():
    from pypdf import PdfWriter

    client = TestClient(app)

    # Blank PDF test (scanned/unreadable check)
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    pdf_bytes = io.BytesIO()
    writer.write(pdf_bytes)
    pdf_bytes.seek(0)

    resp = client.post(
        "/documents",
        files={"file": ("scanned_blank.pdf", pdf_bytes, "application/pdf")},
    )
    assert resp.status_code == 422
    assert "no extractable text" in resp.json()["detail"].lower()

    # Empty file test
    resp_empty = client.post(
        "/documents",
        files={"file": ("empty.txt", io.BytesIO(b""), "text/plain")},
    )
    assert resp_empty.status_code == 400
    assert "empty" in resp_empty.json()["detail"].lower()


