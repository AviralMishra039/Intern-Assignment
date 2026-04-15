"""Pydantic request/response models for all API endpoints."""

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# Document schemas
# ---------------------------------------------------------------------------


class DocumentResponse(BaseModel):
    """Response model for a single document."""

    id: int
    filename: str
    page_count: int
    chunk_count: int
    file_size_kb: float
    status: str
    upload_time: datetime

    model_config = ConfigDict(from_attributes=True)


class DocumentListResponse(BaseModel):
    """Response model for listing all documents."""

    documents: List[DocumentResponse]
    total: int


class DeleteResponse(BaseModel):
    """Response model for document deletion."""

    message: str


# ---------------------------------------------------------------------------
# Query schemas
# ---------------------------------------------------------------------------


class QueryRequest(BaseModel):
    """Request model for querying documents."""

    question: str = Field(..., min_length=1, description="The question to ask")
    document_ids: Optional[List[int]] = Field(
        default=None,
        description="Optional list of document IDs to search. If empty, searches all.",
    )


class SourceInfo(BaseModel):
    """A single source chunk returned with a query answer."""

    document_id: int
    filename: str
    page_number: int
    chunk_preview: str


class QueryResponse(BaseModel):
    """Response model for a query answer."""

    answer: str
    sources: List[SourceInfo]


# ---------------------------------------------------------------------------
# Health check schema
# ---------------------------------------------------------------------------


class HealthResponse(BaseModel):
    """Response model for the health check endpoint."""

    status: str
    vector_db: str
    metadata_db: str
