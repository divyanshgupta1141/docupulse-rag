import io
from unittest.mock import patch
import pytest
from fastapi.testclient import TestClient
from pypdf import PdfWriter

from app.main import app
from app.services.ingestion import IngestionService
from app.services.rag_engine import rag_engine


# ==============================================================================
# Scenario 1: Basic Retrieval
# Upload a document with a clear fact, ask a question, verify correct answer
# and matching citation (filename, page_number, chunk_id).
# ==============================================================================
def test_basic_retrieval():
    with TestClient(app) as client:
        content = (
            "Building Security Guidelines.\n"
            "The primary office access code is 8492.\n"
            "All staff members must never disclose this access code to unauthorized visitors."
        )
        upload_resp = client.post(
            "/documents",
            files={"file": ("building_security.txt", io.BytesIO(content.encode("utf-8")), "text/plain")},
        )
        assert upload_resp.status_code == 201
        doc_info = upload_resp.json()
        doc_id = doc_info["document_id"]
        assert doc_info["filename"] == "building_security.txt"
        assert doc_info["version"] == 1

        conv_id = "conv_test_basic_001"
        chat_resp = client.post(
            f"/chat/{conv_id}",
            json={"question": "What is the primary office access code?", "top_k": 3},
        )
        assert chat_resp.status_code == 200
        data = chat_resp.json()

        assert data["insufficient_evidence"] is False
        assert "8492" in data["answer"]
        assert len(data["citations"]) >= 1

        citation = data["citations"][0]
        assert citation["filename"] == "building_security.txt"
        assert citation["document_id"] == doc_id
        assert citation["version"] == 1
        assert citation["page_number"] == 1
        assert "chunk_id" in citation and len(citation["chunk_id"]) > 0

        # Verify retrieval trace contains score and chunk metadata
        assert len(data["retrieval_trace"]) >= 1
        trace_item = data["retrieval_trace"][0]
        assert trace_item["document_id"] == doc_id
        assert trace_item["similarity_score"] > 0


# ==============================================================================
# Scenario 2: Unsupported Question
# Ask for information absent from all sources; verify insufficient_evidence=True,
# clear refusal message, and zero fabricated citations.
# ==============================================================================
def test_unsupported_question():
    with TestClient(app) as client:
        # Ingest document containing only network policy
        content = "Internal WiFi SSID is CorpNet-Secure. Authentication requires enterprise WPA3 credentials."
        client.post(
            "/documents",
            files={"file": ("wifi_policy.txt", io.BytesIO(content.encode("utf-8")), "text/plain")},
        )

        conv_id = "conv_unsupported_002"
        # Ask question about entirely absent knowledge
        chat_resp = client.post(
            f"/chat/{conv_id}",
            json={"question": "What is the secret recipe for Aunt May's apple pie?", "top_k": 3},
        )
        assert chat_resp.status_code == 200
        data = chat_resp.json()

        assert data["insufficient_evidence"] is True
        assert "not have enough information" in data["answer"].lower()
        assert data["citations"] == []  # Zero fabricated citations


# ==============================================================================
# Scenario 3: Follow-up Context
# Ask an initial question about two policies, then ask a pronoun follow-up
# ("What about the second policy?"); verify recent chat context resolves the
# reference and retrieves fresh evidence.
# ==============================================================================
def test_followup_context():
    with TestClient(app) as client:
        content = (
            "Company Policies.\n"
            "Policy One governs Expense Reimbursement: All meals over $50 require an itemized receipt.\n"
            "Policy Two governs Travel Booking: All employee flights must be booked at least 14 days in advance through the corporate travel portal."
        )
        client.post(
            "/documents",
            files={"file": ("corporate_rules.txt", io.BytesIO(content.encode("utf-8")), "text/plain")},
        )

        conv_id = "conv_followup_003"

        # Turn 1: Initial question introducing both policies
        turn1_resp = client.post(
            f"/chat/{conv_id}",
            json={"question": "What are the rules for Policy One and Policy Two?", "top_k": 5},
        )
        assert turn1_resp.status_code == 200
        turn1_data = turn1_resp.json()
        assert turn1_data["insufficient_evidence"] is False
        assert "Policy" in turn1_data["answer"]

        # Turn 2: Pronoun follow-up referencing "the second policy"
        turn2_resp = client.post(
            f"/chat/{conv_id}",
            json={"question": "What about the second policy?", "top_k": 5},
        )
        assert turn2_resp.status_code == 200
        turn2_data = turn2_resp.json()

        assert turn2_data["insufficient_evidence"] is False
        assert (
            "14 days" in turn2_data["answer"]
            or "travel" in turn2_data["answer"].lower()
            or "flights" in turn2_data["answer"].lower()
        )
        assert len(turn2_data["citations"]) >= 1
        assert turn2_data["citations"][0]["filename"] == "corporate_rules.txt"
        assert len(turn2_data["retrieval_trace"]) >= 1


# ==============================================================================
# Scenario 4: History Boundary (10-message limit & Session Isolation)
# Send 14 messages (7 turns) in a conversation; verify strictly latest 10 messages
# are injected into context, and verify isolation between two conversation_id sessions.
# ==============================================================================
def test_history_boundary():
    with TestClient(app) as client:
        content = "Operational manual: Core server runs on port 8080."
        client.post(
            "/documents",
            files={"file": ("ops_manual.txt", io.BytesIO(content.encode("utf-8")), "text/plain")},
        )

        conv_a = "conv_session_alpha"
        conv_b = "conv_session_beta"

        # Send 7 turns (14 messages total) to conversation A
        for i in range(1, 8):
            resp = client.post(
                f"/chat/{conv_a}",
                json={"question": f"Turn {i}: What is the core server port?", "top_k": 2},
            )
            assert resp.status_code == 200

        # Verify /history endpoint returns all 14 messages for conversation A
        hist_a = client.get(f"/chat/{conv_a}/history").json()
        assert hist_a["total_messages"] == 14

        # Verify that for turn 8, strictly the latest 10 messages were pulled into memory
        # and passed both to contextualize_query and into the LLM generation prompt
        from app.services.ai_providers import MockLLMProvider
        with patch.object(rag_engine, "contextualize_query", wraps=rag_engine.contextualize_query) as spy_context, \
             patch.object(MockLLMProvider, "generate_response", autospec=True, side_effect=MockLLMProvider.generate_response) as spy_llm:
            client.post(
                f"/chat/{conv_a}",
                json={"question": "What port does it run on?", "top_k": 2},
            )
            assert spy_context.called
            # Check the history text passed to contextualize_query
            passed_history = spy_context.call_args[0][1]
            history_lines = [l for l in passed_history.splitlines() if l.startswith("User:") or l.startswith("Assistant:")]
            assert len(history_lines) == 10

            # Check that the exact 10-message <conversation_history> block was passed into generate_response
            assert spy_llm.called
            gen_prompt = spy_llm.call_args[1].get("prompt") or spy_llm.call_args[0][1]
            assert "<conversation_history>" in gen_prompt
            assert passed_history in gen_prompt

        # Now test the boundary rule where the 10-message window starts with an orphaned assistant message
        from app.services.memory import memory_service
        from app.core.database import get_db

        db_gen = app.dependency_overrides.get(get_db, get_db)
        db = next(db_gen())
        try:
            # Add an extra user message so total messages = 17 (8 pairs + 1 = 17)
            # The latest 10 messages in 17 messages will be messages 8..17.
            # Message 8 is role == 'assistant' (orphaned assistant message at boundary)
            memory_service.add_message(db, conv_a, "user", "Interim user query")
            recent, has_boundary = memory_service.get_recent_messages(db, conv_a, limit=10)
            assert len(recent) == 10
            assert recent[0]["role"] == "assistant"
            assert has_boundary is True
            formatted_boundary_ctx = memory_service.format_history_for_context(recent, has_boundary)
            assert "[System: Prior conversation truncated; continuing conversation turn.]" in formatted_boundary_ctx
        finally:
            db.close()

        # Send a message to conversation B
        resp_b = client.post(
            f"/chat/{conv_b}",
            json={"question": "Question in Beta session only", "top_k": 2},
        )
        assert resp_b.status_code == 200

        # Verify isolation: conversation B has strictly 2 messages (1 user, 1 assistant)
        hist_b = client.get(f"/chat/{conv_b}/history").json()
        assert hist_b["total_messages"] == 2
        assert not any("Alpha" in m["content"] for m in hist_b["messages"])


# ==============================================================================
# Scenario 5: Document Update with No Stale Chunks
# Upload "Annual leave is 15 days", query (confirm 15 days), PUT /documents/{id}
# with "Annual leave is 20 days", query again; assert answer reflects 20 days and
# zero chunks from version 1 exist in Qdrant retrieval results.
# ==============================================================================
def test_document_update_no_stale_chunks():
    with TestClient(app) as client:
        # 1. Upload initial policy (v1)
        v1_text = "HR Leave Policy: Standard annual leave is 15 days for all full-time employees."
        up_resp = client.post(
            "/documents",
            files={"file": ("hr_policy.txt", io.BytesIO(v1_text.encode("utf-8")), "text/plain")},
        )
        assert up_resp.status_code == 201
        doc_id = up_resp.json()["document_id"]
        assert up_resp.json()["version"] == 1

        conv_id = "conv_leave_test"

        # Query v1
        q1_resp = client.post(
            f"/chat/{conv_id}",
            json={"question": "What is the standard annual leave allowance?", "top_k": 3},
        )
        assert q1_resp.status_code == 200
        assert "15 days" in q1_resp.json()["answer"]
        assert q1_resp.json()["citations"][0]["version"] == 1

        # 2. Update document via PUT /documents/{id} to v2
        v2_text = "HR Leave Policy: Standard annual leave is 20 days for all full-time employees."
        put_resp = client.put(
            f"/documents/{doc_id}",
            files={"file": ("hr_policy_v2.txt", io.BytesIO(v2_text.encode("utf-8")), "text/plain")},
        )
        assert put_resp.status_code == 200
        assert put_resp.json()["version"] == 2

        # 3. Query v2
        q2_resp = client.post(
            f"/chat/{conv_id}",
            json={"question": "What is the standard annual leave allowance now?", "top_k": 5},
        )
        assert q2_resp.status_code == 200
        q2_data = q2_resp.json()
        assert "20 days" in q2_data["answer"]
        assert "15 days" not in q2_data["answer"]

        # 4. Assert zero chunks from version 1 exist in Qdrant retrieval results
        for trace_item in q2_data["retrieval_trace"]:
            if trace_item["document_id"] == doc_id:
                assert trace_item["version"] == 2, f"Found stale version {trace_item['version']} chunk in retrieval trace!"


# ==============================================================================
# Scenario 6: Document Deletion
# Delete a document via DELETE /documents/{document_id}, query its unique fact,
# and verify chunks are completely unretrievable.
# ==============================================================================
def test_document_deletion():
    with TestClient(app) as client:
        # Ingest document with unique secret fact
        unique_text = "Project Zephyr release target is definitively November 2029."
        up_resp = client.post(
            "/documents",
            files={"file": ("zephyr.txt", io.BytesIO(unique_text.encode("utf-8")), "text/plain")},
        )
        doc_id = up_resp.json()["document_id"]

        conv_id = "conv_zephyr_test"

        # Verify fact is queryable
        q1_resp = client.post(
            f"/chat/{conv_id}",
            json={"question": "When is Project Zephyr release target?", "top_k": 3},
        )
        assert "November 2029" in q1_resp.json()["answer"]

        # Delete document
        del_resp = client.delete(f"/documents/{doc_id}")
        assert del_resp.status_code == 200
        assert del_resp.json()["status"] == "deleted"

        # Verify document is removed from SQLite
        get_resp = client.get(f"/documents/{doc_id}")
        assert get_resp.status_code == 404

        # Query again: chunks must be completely unretrievable
        q2_resp = client.post(
            f"/chat/{conv_id}",
            json={"question": "When is Project Zephyr release target?", "top_k": 3},
        )
        q2_data = q2_resp.json()
        assert q2_data["insufficient_evidence"] is True
        assert "not have enough information" in q2_data["answer"].lower()
        assert q2_data["citations"] == []
        assert not any(t["document_id"] == doc_id for t in q2_data["retrieval_trace"])

        # Multi-document deletion isolation check:
        # Index both hr_leave_policy_v1.txt and clinical_safety_protocol.txt,
        # delete ONLY hr_leave_policy_v1.txt, and query "How many days of annual leave are provided?"
        with open("sample_data/hr_leave_policy_v1.txt", "rb") as f_hr:
            hr_resp = client.post(
                "/documents",
                files={"file": ("hr_leave_policy_v1.txt", io.BytesIO(f_hr.read()), "text/plain")},
            )
        assert hr_resp.status_code == 201
        hr_doc_id = hr_resp.json()["document_id"]

        with open("sample_data/clinical_safety_protocol.txt", "rb") as f_clin:
            clin_resp = client.post(
                "/documents",
                files={"file": ("clinical_safety_protocol.txt", io.BytesIO(f_clin.read()), "text/plain")},
            )
        assert clin_resp.status_code == 201

        del_hr = client.delete(f"/documents/{hr_doc_id}")
        assert del_hr.status_code == 200

        q_deleted_hr = client.post(
            "/chat/conv_deleted_hr_isolation",
            json={"question": "How many days of annual leave are provided?", "top_k": 5},
        )
        assert q_deleted_hr.status_code == 200
        hr_del_data = q_deleted_hr.json()
        assert hr_del_data["insufficient_evidence"] is True
        assert hr_del_data["citations"] == []
        assert "not have enough information" in hr_del_data["answer"].lower()


# ==============================================================================
# Scenario 7: Duplicate Upload
# Upload exact same file twice and verify deliberate duplicate handling
# (via SHA-256 content_hash).
# ==============================================================================
def test_duplicate_upload():
    with TestClient(app) as client:
        file_content = b"Database backup schedule: Every Sunday at 02:00 UTC."

        # First upload: succeeds (HTTP 201)
        resp1 = client.post(
            "/documents",
            files={"file": ("backup_schedule.txt", io.BytesIO(file_content), "text/plain")},
        )
        assert resp1.status_code == 201
        first_doc_id = resp1.json()["document_id"]

        # Second upload with exact same content: HTTP 409 Conflict
        resp2 = client.post(
            "/documents",
            files={"file": ("backup_schedule_copy.txt", io.BytesIO(file_content), "text/plain")},
        )
        assert resp2.status_code == 409
        assert "duplicate" in resp2.json()["detail"].lower()
        assert first_doc_id in resp2.json()["detail"]


# ==============================================================================
# Scenario 8: Bad Input Validation
# Upload empty .txt, unsupported .exe, and blank/corrupted PDF; verify HTTP 400/422.
# ==============================================================================
def test_bad_input():
    with TestClient(app) as client:
        # 1. Empty .txt file
        resp_empty = client.post(
            "/documents",
            files={"file": ("empty.txt", io.BytesIO(b""), "text/plain")},
        )
        assert resp_empty.status_code == 400
        assert "empty" in resp_empty.json()["detail"].lower()

        # 2. Unsupported file extension (.exe)
        resp_exe = client.post(
            "/documents",
            files={"file": ("malware.exe", io.BytesIO(b"MZ\x90\x00\x03\x00\x00\x00"), "application/octet-stream")},
        )
        assert resp_exe.status_code == 400
        assert "unsupported" in resp_exe.json()["detail"].lower()

        # 3. Blank PDF (scanned/unreadable check)
        writer = PdfWriter()
        writer.add_blank_page(width=100, height=100)
        blank_pdf_bytes = io.BytesIO()
        writer.write(blank_pdf_bytes)
        blank_pdf_bytes.seek(0)

        resp_blank_pdf = client.post(
            "/documents",
            files={"file": ("blank_scan.pdf", blank_pdf_bytes, "application/pdf")},
        )
        assert resp_blank_pdf.status_code == 422
        assert "no extractable text" in resp_blank_pdf.json()["detail"].lower()

        # 4. Corrupted PDF bytes
        resp_corrupted = client.post(
            "/documents",
            files={"file": ("corrupted.pdf", io.BytesIO(b"not a valid pdf header"), "application/pdf")},
        )
        assert resp_corrupted.status_code == 400
        assert "corrupted" in resp_corrupted.json()["detail"].lower()

        # System should remain healthy
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["status"] == "healthy"


# ==============================================================================
# Scenario 9: Multiple Sources Synthesis
# Upload two distinct documents, ask a comparative question spanning both,
# and verify both documents appear in citations.
# ==============================================================================
def test_multiple_sources():
    with TestClient(app) as client:
        # Document 1: Workstation Security
        doc1_text = (
            "Workstation Security Guidelines.\n"
            "All remote workstations must use full-disk encryption and corporate VPN."
        )
        resp1 = client.post(
            "/documents",
            files={"file": ("workstation_security.txt", io.BytesIO(doc1_text.encode("utf-8")), "text/plain")},
        )
        assert resp1.status_code == 201
        doc1_id = resp1.json()["document_id"]

        # Document 2: Compliance Audit
        doc2_text = (
            "Audit Compliance Policy.\n"
            "Annual SOC2 compliance requires audit logs to be retained for at least 365 days."
        )
        resp2 = client.post(
            "/documents",
            files={"file": ("compliance_audit.txt", io.BytesIO(doc2_text.encode("utf-8")), "text/plain")},
        )
        assert resp2.status_code == 201
        doc2_id = resp2.json()["document_id"]

        conv_id = "conv_multi_sources"
        chat_resp = client.post(
            f"/chat/{conv_id}",
            json={
                "question": "What are the requirements for remote workstations and how long must audit logs be retained?",
                "top_k": 5,
            },
        )
        assert chat_resp.status_code == 200
        data = chat_resp.json()

        assert data["insufficient_evidence"] is False
        answer = data["answer"]
        assert "workstation" in answer.lower() or "encryption" in answer.lower()
        assert "365 days" in answer or "audit" in answer.lower()

        # Verify citations include BOTH documents
        cited_doc_ids = {c["document_id"] for c in data["citations"]}
        assert doc1_id in cited_doc_ids
        assert doc2_id in cited_doc_ids


# ==============================================================================
# Scenario 10: Failure Handling & Rollback Recovery
# Simulate an embedding/indexing failure during PUT /documents/{id} update;
# verify HTTP 502 error, no false success, and version 1 remains active and retrievable.
# ==============================================================================
def test_failure_handling():
    with TestClient(app) as client:
        # 1. Ingest initial document (v1)
        v1_text = "Support SLA guarantees response within 4 hours for all critical tickets."
        up_resp = client.post(
            "/documents",
            files={"file": ("support_sla.txt", io.BytesIO(v1_text.encode("utf-8")), "text/plain")},
        )
        assert up_resp.status_code == 201
        doc_id = up_resp.json()["document_id"]

        conv_id = "conv_failure_sim"

        # Verify v1 is active and retrievable
        q1_resp = client.post(
            f"/chat/{conv_id}",
            json={"question": "What is the support SLA response time for critical tickets?", "top_k": 3},
        )
        assert "4 hours" in q1_resp.json()["answer"]

        # 2. Simulate Qdrant/Embedding failure during PUT /documents/{id}
        with patch.object(
            IngestionService,
            "stage_and_index_vectors",
            side_effect=RuntimeError("Simulated Qdrant connection timeout during vector indexing"),
        ):
            v2_failed_text = "Support SLA updated: Response within 1 hour for all critical tickets."
            put_resp = client.put(
                f"/documents/{doc_id}",
                files={"file": ("support_sla_v2.txt", io.BytesIO(v2_failed_text.encode("utf-8")), "text/plain")},
            )
            # Must return error (HTTP 502 / 500) and NOT report false success
            assert put_resp.status_code in [500, 502]
            err_detail = put_resp.json()["detail"]
            assert "failed" in err_detail.lower()
            assert "remains active" in err_detail.lower()

        # 3. Verify SQLite document record was NOT corrupted and is still version 1
        get_doc = client.get(f"/documents/{doc_id}")
        assert get_doc.status_code == 200
        assert get_doc.json()["version"] == 1
        assert get_doc.json()["status"] == "indexed"

        # 4. Verify version 1 remains active and retrievable in chat
        q2_resp = client.post(
            f"/chat/{conv_id}",
            json={"question": "What is the support SLA response time for critical tickets?", "top_k": 3},
        )
        assert q2_resp.status_code == 200
        q2_data = q2_resp.json()
        assert q2_data["insufficient_evidence"] is False
        assert "4 hours" in q2_data["answer"]
        assert "1 hour" not in q2_data["answer"]
        assert q2_data["citations"][0]["version"] == 1

        # 5. Simulate a post-commit old-vector cleanup failure:
        # Once SQLite commits version = 2, a failure in delete_by_document(version_lt=2)
        # MUST NOT delete version 2 vectors or brick the document.
        from app.services.vector_store import QdrantVectorStore

        with patch.object(
            QdrantVectorStore,
            "delete_by_document",
            side_effect=RuntimeError("Simulated Qdrant error during old-vector cleanup"),
        ):
            v2_success_text = "Support SLA updated: Response within 1 hour for all critical tickets."
            put_v2_resp = client.put(
                f"/documents/{doc_id}",
                files={"file": ("support_sla_v2.txt", io.BytesIO(v2_success_text.encode("utf-8")), "text/plain")},
            )
            assert put_v2_resp.status_code == 200
            assert put_v2_resp.json()["version"] == 2

        # Verify version 2 remains active and retrievable even though old-vector cleanup failed
        q3_resp = client.post(
            f"/chat/{conv_id}_v2",
            json={"question": "What is the support SLA response time for critical tickets?", "top_k": 3},
        )
        assert q3_resp.status_code == 200
        q3_data = q3_resp.json()
        assert q3_data["insufficient_evidence"] is False
        assert "1 hour" in q3_data["answer"]
        assert q3_data["citations"][0]["version"] == 2
