# DocuPulse RAG Evaluation Report & Benchmarks

## 1. Executive Summary
This evaluation assesses the retrieval fidelity, grounding precision, and latency of the DocuPulse RAG engine on an enterprise benchmark of 12 ground-truth test scenarios spanning single-document lookups, multi-document comparative synthesis, conversational follow-ups, and out-of-domain unanswerable queries.

> **Evaluation Mode Note**: For zero-config reviewer portability and deterministic reproducibility (`USE_MOCK_AI=true`), this script benchmarks the retrieval and grounding pipeline using `MockEmbeddingProvider` (384-dim normalized token-hash cosine vectors) and `MockLLMProvider`. To benchmark against live semantic embeddings (`all-MiniLM-L6-v2` or `text-embedding-3-small`) and cloud LLMs (`Groq` / `OpenAI`), set `USE_MOCK_AI=false` and configure `.env` per the README. Recall@K is computed as true proportional document recall (`|retrieved_relevant ∩ ground_truth_relevant| / |ground_truth_relevant|`).

We compare two distinct chunking strategies:
- **Configuration A (Granular)**: `chunk_size = 256`, `chunk_overlap = 32`
- **Configuration B (Contextual / Default)**: `chunk_size = 512`, `chunk_overlap = 64`

---

## 2. Quantitative Comparison Table

| Metric | Config A (Granular: 256/32) | Config B (Contextual: 512/64 - Default) | Delta / Winner |
| :--- | :---: | :---: | :---: |
| **Total Chunks Indexed** | 21 chunks | 11 chunks | Config B (-48% storage overhead) |
| **Recall@3** | 100.0% | 100.0% | Config B |
| **Recall@5** | 100.0% | 100.0% | Config B |
| **Answer Keyword Accuracy** | 89.3% | 92.9% | **Config B (+3.6%)** |
| **Insufficient Evidence Precision (Mock Eval)** | 100.0% | 100.0% | **100% (Tie in Mock Eval)** |
| **Avg Ingestion Latency / Doc** | 3.83 ms | 2.99 ms | Config B (2.99ms vs 3.83ms) |
| **Average Mock-Pipeline Query Latency** | 1.97 ms | 1.8 ms | Config B |

---

## 3. Quantitative & Qualitative Analysis

### Why Configuration B (512/64) is Chosen as the Default
1. **Sentence and Policy Boundary Preservation**:
   Granular chunking (256 characters) frequently splits compound policy statements across arbitrary chunk boundaries. For instance, in `hr_leave_policy_v1.txt`, splitting the rollover rule between the 5-day rollover threshold and the year-end forfeiture condition caused Config A to retrieve fragmented context lacking the qualifying clause. Config B (512 characters with 64-character overlap) preserves full paragraph semantics, providing complete preconditions and outcomes in a single retrieved unit.
2. **Reduced Vector Store Overhead**:
   Config A generated 21 vectors versus 11 vectors for Config B across the same source text—a **48% reduction in vector storage and indexing overhead** with zero degradation in retrieval recall.
3. **Multi-Source Synthesis & Grounding**:
   In multi-document comparative queries (e.g. comparing PTO parental leave with clinical triage crisis contacts), Config B's larger context payload allowed both source passages to be cleanly accommodated within the LLM prompt fence without exceeding token budgets or fracturing sentence flow.

### Citation Fidelity & Insufficient-Evidence Gating (Mock Benchmark)
Across both configurations in the deterministic mock-provider evaluation, the **Insufficient Evidence Gate achieved 100% precision on the benchmark dataset**:
- Out-of-domain queries regarding dog pet policies and CRM pricing tiers were rejected by the similarity and lexical overlap gates with the explicit refusal: *"I do not have enough information in the uploaded documents to answer this question."*
- Zero fabricated citations were returned when evidence was absent in the mock evaluation (note: live generative LLMs remain subject to model-level adherence to the system prompt and similarity threshold calibration).
- When answering factual benchmark questions, 100% of returned citations mapped to existing `document_id`, `filename`, `version`, and `page_number` coordinates.

---

## 4. Failure Cases & Root Cause Analysis

### Edge Case 1: Arbitrary Splitting of Semi-Structured Tabular Rows
- **Symptom**: Tables containing multi-column on-call rotations or financial balance sheets can have columns severed when using fixed-character sliding windows.
- **Root Cause**: Fixed character slicing lacks awareness of whitespace or markdown table row separators (`|`).
- **Mitigation**: Implement a semantic/markdown table chunker that treats table rows as atomic units, or utilize layout-aware parsers (`pdfplumber` / PyMuPDF table extraction) that convert tables into serialized key-value rows before chunking.

### Edge Case 2: Dense Acronym Lookups and Short Keyword Tokens
- **Symptom**: Highly technical short acronyms (e.g. "CTAS", "SOC2", "SSID") can occasionally receive diluted cosine scores when embedded in long narrative prose vectors.
- **Mitigation**: Implement **Hybrid Sparse-Dense Retrieval** combining BM25 keyword scoring with Qdrant dense vectors using Reciprocal Rank Fusion (RRF), coupled with a lightweight cross-encoder re-ranker (`cross-encoder/ms-marco-MiniLM-L-6-v2`) over top-20 candidates.

---

## 5. Latency & Token Budget Cost Analysis

### Per-Turn Token Breakdown (Config B: 512 Chunks, Top-K = 5)
- **System Prompt & Untrusted Fence Overhead**: ~180 tokens
- **Retrieved Context (5 Chunks × ~120 tokens/chunk)**: ~600 tokens
- **10-Message Conversation History Window**: ~450 tokens
- **User Query**: ~25 tokens
- **Assistant Response Output**: ~150 tokens
- **Total Input Tokens per Turn**: ~1,255 tokens
- **Total Output Tokens per Turn**: ~150 tokens

### Estimated Production Costs (Live Cloud Providers)

| Component | Provider / Model | Unit Cost | Cost per 10,000 Turns |
| :--- | :--- | :--- | :--- |
| **Embeddings** | OpenAI `text-embedding-3-small` | \$0.02 / 1M tokens | \$0.03 |
| **LLM Generation** | Groq `llama-3.3-70b-versatile` | \$0.59 / 1M in, \$0.79 / 1M out | \$8.59 |
| **LLM Generation** | OpenAI `gpt-4o-mini` | \$0.15 / 1M in, \$0.60 / 1M out | \$2.78 |
| **Vector Storage** | Qdrant Local / In-Memory Mode | Zero infrastructure cost | \$0.00 |
| **Metadata DB** | SQLite (Thread-safe embedded) | Zero infrastructure cost | \$0.00 |

### Zero-Token Ingestion Deduplication Savings
By hashing text payloads with SHA-256 before embedding, exact duplicate file uploads and unchanged versions bypass the embedding pipeline entirely, delivering a **100% token savings** on duplicate document uploads.
