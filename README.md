# RAG Pipeline API

A production-ready **Retrieval-Augmented Generation (RAG)** API built with FastAPI. Upload documents (PDF, DOCX, TXT), ask questions, and receive accurate answers grounded in your content using vector search and an LLM.

## Live Deployment (Start Here)
👉 **Interactive API Dashboard:** [https://intern-assignment-2y95.onrender.com/docs](https://intern-assignment-2y95.onrender.com/docs)

> **Note:** This API is currently deployed on a free Render instance to save costs. It automatically spins down when inactive. When you click the link, **please allow ~50 seconds for the server to wake up** on the initial load!

Services like GCP, AWS, Azure, etc. were not used as they were requiring credits and prepayments which were not available to me. Instead, Render was used for deployment.

## Architecture

```
┌──────────┐       ┌───────────────────────────────────────────┐
│  Client  │──────▶│  FastAPI  (uvicorn :8000)                 │
└──────────┘       │                                           │
                   │  ┌─────────────┐  ┌────────────────────┐  │
                   │  │  Routers    │  │  Services           │  │
                   │  │  /documents │  │  ingestion.py       │  │
                   │  │  /query     │  │  retrieval.py       │  │
                   │  │  /health    │  │  llm.py             │  │
                   │  └──────┬──────┘  └────────┬───────────┘  │
                   │         │                  │              │
                   │  ┌──────▼──────┐  ┌────────▼───────────┐  │
                   │  │  SQLite     │  │  ChromaDB          │  │
                   │  │  (metadata) │  │  (vectors/chunks)  │  │
                   │  └─────────────┘  └────────────────────┘  │
                   │                                           │
                   │         ┌──────────────────┐              │
                   │         │  Google Gemini   │              │
                   │         │  embeddings+LLM  │              │
                   │         └──────────────────┘              │
                   └───────────────────────────────────────────┘
```

## Ingestion & Retrieval Pipeline

**Ingestion:** extract text per page (PDF/DOCX/TXT) → split into **token-sized chunks** (`RecursiveCharacterTextSplitter` + `tiktoken` `cl100k_base`, 256 tokens, 40 overlap) → embed with Gemini `gemini-embedding-001` (768-d) → store in ChromaDB.

**Query:**

```
question ─┬─▶ vector search (ChromaDB, top 30) ─┐
          │                                     ├─▶ Reciprocal Rank Fusion ─▶ cross-encoder rerank ─▶ top 5 ─▶ LLM
          └─▶ BM25 keyword search (top 30) ─────┘
```

- **Token-based chunking** — chunk size is measured in tokens (tiktoken `cl100k_base`, a close approximation of Gemini's tokenizer) instead of characters.
- **Hybrid search** — dense vectors catch paraphrases; BM25 catches exact terms (IDs, names, acronyms). RRF merges the two ranked lists without needing comparable scores.
- **Reranking** — a local cross-encoder ([FlashRank](https://github.com/PrithivirajDamodaran/FlashRank), `ms-marco-MiniLM-L-12-v2`, ONNX/CPU, ~22MB) re-scores the fused candidates so only the best 5 reach the LLM. If the reranker fails, retrieval falls back to the fused order.
- Each stage can be toggled with env vars (`HYBRID_SEARCH_ENABLED`, `RERANKER_ENABLED`) for ablation.

> **Upgrading from an earlier version:** documents indexed before the Gemini/token-chunking changes live in a different vector space. Delete and re-upload them (they are stored in the new `documents_gemini` collection).

## Prerequisites

- **Docker** & **Docker Compose** (recommended), or Python 3.11+
- A free **Google Gemini API key** from [Google AI Studio](https://aistudio.google.com/apikey)

## Quickstart with Docker Compose

1. **Clone the repo:**
   ```bash
   git clone <repo-url>
   cd rag-pipeline
   ```

2. **Create your `.env` file:**
   ```bash
   cp .env.example .env
   # Edit .env and add your Gemini API key
   ```

3. **Build and run:**
   ```bash
   docker-compose up --build
   ```

4. **Open the API docs:**
   - Swagger UI: [http://localhost:8000/docs](http://localhost:8000/docs)
   - ReDoc: [http://localhost:8000/redoc](http://localhost:8000/redoc)

## Running Locally (Without Docker)

1. **Create a virtual environment and install dependencies:**
   ```bash
   uv venv
   source .venv/bin/activate   # Linux/Mac
   .venv\Scripts\activate      # Windows
   uv pip install -r requirements.txt
   ```

2. **Create your `.env` file:**
   ```bash
   cp .env.example .env
   # Add your GEMINI_API_KEY
   ```

4. **Run the server:**
   ```bash
   uvicorn app.main:app --reload --port 8000
   ```

## API Endpoints

### 1. Health Check

```bash
curl http://localhost:8000/health
```

**Response:**
```json
{
  "status": "ok",
  "vector_db": "connected",
  "metadata_db": "connected"
}
```

### 2. Upload Document

```bash
curl -X POST http://localhost:8000/documents/upload \
  -F "file=@report.pdf"
```

**Response (200):**
```json
{
  "id": 1,
  "filename": "report.pdf",
  "page_count": 12,
  "chunk_count": 47,
  "file_size_kb": 204.5,
  "status": "ready",
  "upload_time": "2024-01-01T10:00:00"
}
```

### 3. List All Documents

```bash
curl http://localhost:8000/documents/
```

**Response (200):**
```json
{
  "documents": [
    {
      "id": 1,
      "filename": "report.pdf",
      "page_count": 12,
      "chunk_count": 47,
      "file_size_kb": 204.5,
      "status": "ready",
      "upload_time": "2024-01-01T10:00:00"
    }
  ],
  "total": 1
}
```

### 4. Get Single Document

```bash
curl http://localhost:8000/documents/1
```

### 5. Delete Document

```bash
curl -X DELETE http://localhost:8000/documents/1
```

**Response (200):**
```json
{
  "message": "Document deleted successfully."
}
```

### 6. Query Documents

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What are the key findings of the report?",
    "document_ids": [1, 2]
  }'
```

**Response (200):**
```json
{
  "answer": "The key findings are...",
  "sources": [
    {
      "document_id": 1,
      "filename": "report.pdf",
      "page_number": 4,
      "chunk_preview": "First 200 characters of the chunk..."
    }
  ]
}
```

> **Tip:** Omit `document_ids` to search across all uploaded documents.

## Running Tests

```bash
# Install dependencies (if not already)
uv pip install -r requirements.txt

# Run all tests
pytest tests/ -v
```

All tests fake the Gemini client, so no API key or network is needed to run the suite (the first run downloads the tokenizer data once).

## RAGAS Evaluation

The project includes an offline evaluation command that exercises the **same**
retrieval and answer-generation services used by `POST /query`. It measures:

- `faithfulness` — whether the answer is supported by retrieved chunks;
- `answer_relevancy` — whether the answer addresses the question;
- `context_precision` and `context_recall` — retrieval quality against a
  human-verified reference answer; and
- `answer_correctness` — agreement with that reference answer.

Create a labelled JSON array based on
[`evaluation/dataset.example.json`](rag-pipeline/evaluation/dataset.example.json).
Each example requires a `question` and a human-verified `reference`; it may
also include `document_ids` to evaluate a filtered search.

```bash
cd rag-pipeline
pip install -r requirements.txt
python scripts/evaluate_rag.py --dataset evaluation/dataset.json
```

The command needs the same `GEMINI_API_KEY` as the application because both the
RAG pipeline and RAGAS's Gemini evaluator make model calls. It prints aggregate
scores and saves per-example answers, contexts, sources, scores, and evaluator
reasons in `evaluation/results/` (or the path passed through `--output`).

## Gemini Setup & Free-Tier Notes

All LLM and embedding calls go through one module, [`app/services/gemini.py`](rag-pipeline/app/services/gemini.py), so swapping providers later only means rewriting that file.

1. Get a free key at <https://aistudio.google.com/apikey> and set `GEMINI_API_KEY` in `.env`.
2. Verify your key and models: `python scripts/check_gemini.py` (it lists the models your key can use if one of the configured ones isn't available).

- **Embeddings:** `gemini-embedding-001`, truncated to 768 dimensions (Matryoshka) and re-normalised. Documents use task type `RETRIEVAL_DOCUMENT` and queries use `RETRIEVAL_QUERY`.
- **Generation:** `gemini-2.5-flash-lite` by default (`GEMINI_LLM_MODEL`).
- **Rate limits:** the free tier is limited (roughly 100 embedding requests/min and 30k tokens/min). Embedding is batched (`EMBEDDING_BATCH_SIZE`) and 429/5xx errors are retried with jittered exponential backoff, so large uploads are slower but succeed. If the quota is still exhausted, `/query` returns `429`.
- **Changing embedding model or dimension** produces vectors that are incompatible with existing ones. Use a new `CHROMA_COLLECTION` name and re-upload your documents.

## Environment Variable Reference

| Variable | Default | Description |
|---|---|---|
| `GEMINI_API_KEY` | *(required)* | Google AI Studio API key |
| `GEMINI_LLM_MODEL` | `gemini-2.5-flash-lite` | Model used to generate answers |
| `GEMINI_EMBEDDING_MODEL` | `gemini-embedding-001` | Model used for embeddings |
| `EMBEDDING_DIMENSIONS` | `768` | Embedding size (truncated and re-normalised) |
| `EMBEDDING_BATCH_SIZE` | `50` | Texts per embedding request |
| `LLM_TEMPERATURE` | `0.3` | Generation temperature |
| `LLM_MAX_OUTPUT_TOKENS` | `2048` | Max tokens in an answer |
| `CHROMA_COLLECTION` | `documents_gemini` | ChromaDB collection name (tied to the embedding model) |
| `CHROMA_PERSIST_DIR` | `./chroma_data` | ChromaDB persistence directory |
| `UPLOAD_DIR` | `./uploads` | Temporary upload directory |
| `DATABASE_URL` | `sqlite:///./metadata.db` | SQLAlchemy database URL |
| `MAX_DOCUMENTS` | `20` | Maximum number of documents allowed |
| `CHUNK_SIZE_TOKENS` | `256` | Chunk size in tokens |
| `CHUNK_OVERLAP_TOKENS` | `40` | Overlap between chunks in tokens |
| `TOKENIZER_ENCODING` | `cl100k_base` | tiktoken encoding used to count tokens |
| `TOP_K_RESULTS` | `5` | Final number of chunks sent to the LLM |
| `RETRIEVAL_CANDIDATES` | `30` | Candidates fetched from each retriever before fusion |
| `HYBRID_SEARCH_ENABLED` | `true` | Combine vector + BM25 search (set `false` for vector only) |
| `RRF_K` | `60` | Reciprocal Rank Fusion constant |
| `RERANKER_ENABLED` | `true` | Rerank fused candidates with a cross-encoder |
| `RERANKER_MODEL` | `ms-marco-MiniLM-L-12-v2` | FlashRank model name |
| `RERANKER_CACHE_DIR` | `./.flashrank_cache` | Where the reranker model is cached |

## Known Limitations

- **No authentication/authorization** — the API is open. Add API key middleware or OAuth for production use.
- **Synchronous processing** — document upload is blocking. For large documents, consider background task processing with Celery or FastAPI `BackgroundTasks`.
- **DOCX page numbers** — DOCX files don't expose reliable page numbers; all content is assigned to page 1.
- **Single-container architecture** — ChromaDB runs embedded in the API process. For high-traffic deployments, consider running ChromaDB as a separate service.
- **File size limits** — no explicit file size cap is enforced beyond what uvicorn allows by default.
- **No OCR support** — scanned PDFs (image-only) will not have extractable text.
