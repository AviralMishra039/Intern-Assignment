"""FastAPI application entry point."""

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text

from app.config import settings
from app.database import init_db, get_chroma_collection, SessionLocal
from app.models.schemas import HealthResponse
from app.routers import documents, query


# ---------------------------------------------------------------------------
# Lifespan handler (replaces deprecated on_event)
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Run once on startup; yield for app lifetime; cleanup on shutdown."""
    # Ensure required directories exist
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    os.makedirs(settings.CHROMA_PERSIST_DIR, exist_ok=True)
    # Create database tables
    init_db()
    yield


# ---------------------------------------------------------------------------
# App initialization
# ---------------------------------------------------------------------------

app = FastAPI(
    title="RAG Pipeline API",
    description=(
        "A production-ready Retrieval-Augmented Generation API. "
        "Upload documents (PDF, DOCX, TXT) and ask questions — "
        "answers are grounded in your uploaded content using vector search and an LLM."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allow all origins for development convenience
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(documents.router)
app.include_router(query.router)


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------


@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health_check():
    """Check connectivity to the metadata DB and vector DB."""
    # Check metadata DB
    metadata_status = "disconnected"
    try:
        db = SessionLocal()
        db.execute(text("SELECT 1"))
        db.close()
        metadata_status = "connected"
    except Exception:
        metadata_status = "disconnected"

    # Check vector DB
    vector_status = "disconnected"
    try:
        collection = get_chroma_collection()
        collection.count()  # Simple operation to verify connectivity
        vector_status = "connected"
    except Exception:
        vector_status = "disconnected"

    overall = "ok" if metadata_status == "connected" and vector_status == "connected" else "degraded"

    return HealthResponse(
        status=overall,
        vector_db=vector_status,
        metadata_db=metadata_status,
    )
