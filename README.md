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
                   │         │  OpenAI API      │              │
                   │         │  embeddings+LLM  │              │
                   │         └──────────────────┘              │
                   └───────────────────────────────────────────┘
```

## Prerequisites

- **Docker** & **Docker Compose** (recommended), or Python 3.11+
- An **OpenAI API key** with access to `text-embedding-3-small` and `gpt-4o-mini`

## Quickstart with Docker Compose

1. **Clone the repo:**
   ```bash
   git clone <repo-url>
   cd rag-pipeline
   ```

2. **Create your `.env` file:**
   ```bash
   cp .env.example .env
   # Edit .env and add your OpenAI API key
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
   # Add your OPENAI_API_KEY
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

All tests use mocked OpenAI calls, so no API key is needed to run the test suite.

## How to Swap LLM Provider (e.g., Google Gemini)

To switch from OpenAI `gpt-4o-mini` to Google Gemini:

1. **Install the Google AI SDK:**
   ```bash
   pip install google-generativeai
   ```

2. **Update `.env`:**
   ```env
   GEMINI_API_KEY=your_gemini_api_key
   ```

3. **Modify `app/services/llm.py`:**
   ```python
   import google.generativeai as genai
   from app.config import settings

   genai.configure(api_key=settings.GEMINI_API_KEY)
   model = genai.GenerativeModel("gemini-1.5-flash")

   def generate_answer(question: str, chunks: list) -> str:
       context = "\n\n".join([c["text"] for c in chunks])
       prompt = f"Context:\n{context}\n\nQuestion: {question}"
       response = model.generate_content(prompt)
       return response.text
   ```

4. **For embeddings**, you would also swap `retrieval.py` and `ingestion.py` to use a Gemini embedding model or another provider.

## Environment Variable Reference

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | *(required)* | Your OpenAI API key |
| `CHROMA_PERSIST_DIR` | `./chroma_data` | ChromaDB persistence directory |
| `UPLOAD_DIR` | `./uploads` | Temporary upload directory |
| `DATABASE_URL` | `sqlite:///./metadata.db` | SQLAlchemy database URL |
| `MAX_DOCUMENTS` | `20` | Maximum number of documents allowed |
| `CHUNK_SIZE` | `500` | Text chunk size (characters) |
| `CHUNK_OVERLAP` | `50` | Overlap between chunks (characters) |
| `TOP_K_RESULTS` | `40` | Number of chunks returned per query |

## Known Limitations

- **No authentication/authorization** — the API is open. Add API key middleware or OAuth for production use.
- **Synchronous processing** — document upload is blocking. For large documents, consider background task processing with Celery or FastAPI `BackgroundTasks`.
- **DOCX page numbers** — DOCX files don't expose reliable page numbers; all content is assigned to page 1.
- **Single-container architecture** — ChromaDB runs embedded in the API process. For high-traffic deployments, consider running ChromaDB as a separate service.
- **File size limits** — no explicit file size cap is enforced beyond what uvicorn allows by default.
- **No OCR support** — scanned PDFs (image-only) will not have extractable text.
