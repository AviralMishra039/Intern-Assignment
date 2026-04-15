"""Document management endpoints — upload, list, get, delete."""

import os
import uuid
from typing import List

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from sqlalchemy.orm import Session

from app.config import settings
from app.database import Document, get_chroma_collection, get_db
from app.models.schemas import (
    DeleteResponse,
    DocumentListResponse,
    DocumentResponse,
)
from app.services.ingestion import process_document

router = APIRouter(prefix="/documents", tags=["Documents"])

ALLOWED_EXTENSIONS = {"pdf", "docx", "txt"}


def _get_file_extension(filename: str) -> str:
    """Return the lowercase file extension without the dot."""
    return filename.rsplit(".", 1)[-1].lower() if "." in filename else ""


# ---------------------------------------------------------------------------
# POST /documents/upload
# ---------------------------------------------------------------------------


@router.post("/upload", response_model=DocumentResponse)
async def upload_document(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    """Upload a document (PDF, DOCX, or TXT) for processing and indexing."""

    # Validate file type
    file_ext = _get_file_extension(file.filename or "")
    if file_ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="File type not supported. Upload PDF, DOCX, or TXT.",
        )

    # Check document limit
    doc_count = db.query(Document).count()
    if doc_count >= settings.MAX_DOCUMENTS:
        raise HTTPException(
            status_code=400,
            detail=f"Maximum document limit of {settings.MAX_DOCUMENTS} reached.",
        )

    # Save file to disk
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    safe_name = f"{uuid.uuid4().hex}.{file_ext}"
    file_path = os.path.join(settings.UPLOAD_DIR, safe_name)

    content = await file.read()
    file_size_kb = len(content) / 1024.0

    with open(file_path, "wb") as f:
        f.write(content)

    # Create DB record (status = processing)
    doc_record = Document(
        filename=file.filename,
        original_name=file.filename,
        file_type=file_ext,
        file_size_kb=round(file_size_kb, 2),
        status="processing",
    )
    db.add(doc_record)
    db.commit()
    db.refresh(doc_record)

    # Process document (extract → chunk → embed → store)
    try:
        chunk_count, page_count = process_document(
            file_path=file_path,
            file_type=file_ext,
            document_id=doc_record.id,
            filename=file.filename,
        )
        doc_record.chunk_count = chunk_count
        doc_record.page_count = page_count
        doc_record.status = "ready"
    except Exception as e:
        doc_record.status = "failed"
        # Clean up the file if processing failed
        if os.path.exists(file_path):
            os.remove(file_path)
        db.commit()
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

    db.commit()
    db.refresh(doc_record)

    return DocumentResponse(
        id=doc_record.id,
        filename=doc_record.filename,
        page_count=doc_record.page_count,
        chunk_count=doc_record.chunk_count,
        file_size_kb=doc_record.file_size_kb,
        status=doc_record.status,
        upload_time=doc_record.upload_time,
    )


# ---------------------------------------------------------------------------
# GET /documents/
# ---------------------------------------------------------------------------


@router.get("/", response_model=DocumentListResponse)
def list_documents(db: Session = Depends(get_db)):
    """List all uploaded documents."""
    docs = db.query(Document).all()
    return DocumentListResponse(
        documents=[
            DocumentResponse(
                id=d.id,
                filename=d.filename,
                page_count=d.page_count,
                chunk_count=d.chunk_count,
                file_size_kb=d.file_size_kb,
                status=d.status,
                upload_time=d.upload_time,
            )
            for d in docs
        ],
        total=len(docs),
    )


# ---------------------------------------------------------------------------
# GET /documents/{document_id}
# ---------------------------------------------------------------------------


@router.get("/{document_id}", response_model=DocumentResponse)
def get_document(document_id: int, db: Session = Depends(get_db)):
    """Get metadata for a single document by ID."""
    doc = db.query(Document).filter(Document.id == document_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found.")

    return DocumentResponse(
        id=doc.id,
        filename=doc.filename,
        page_count=doc.page_count,
        chunk_count=doc.chunk_count,
        file_size_kb=doc.file_size_kb,
        status=doc.status,
        upload_time=doc.upload_time,
    )


# ---------------------------------------------------------------------------
# DELETE /documents/{document_id}
# ---------------------------------------------------------------------------


@router.delete("/{document_id}", response_model=DeleteResponse)
def delete_document(document_id: int, db: Session = Depends(get_db)):
    """Delete a document and its chunks from the database and vector store."""
    doc = db.query(Document).filter(Document.id == document_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found.")

    # Remove chunks from ChromaDB
    collection = get_chroma_collection()
    # Get all chunk IDs for this document
    chunk_ids = [f"{document_id}_{i}" for i in range(doc.chunk_count)]
    if chunk_ids:
        try:
            collection.delete(ids=chunk_ids)
        except Exception:
            pass  # Best-effort deletion from vector store

    # Remove from metadata DB
    db.delete(doc)
    db.commit()

    return DeleteResponse(message="Document deleted successfully.")
