"""Document ingestion service — parsing, chunking, embedding, and storage."""

import os
from typing import Dict, List, Tuple

import fitz  # PyMuPDF
from docx import Document as DocxDocument
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI

from app.config import settings
from app.database import get_chroma_collection

client = OpenAI(api_key=settings.OPENAI_API_KEY)


# ---------------------------------------------------------------------------
# Text extraction
# ---------------------------------------------------------------------------


def extract_text_from_pdf(file_path: str) -> Tuple[List[Dict], int]:
    """Extract text from a PDF file, page by page.

    Returns:
        A tuple of (list of {text, page_number} dicts, total page count).
    """
    pages = []
    doc = fitz.open(file_path)
    page_count = len(doc)
    for page_num in range(page_count):
        page = doc[page_num]
        text = page.get_text()
        if text.strip():
            pages.append({"text": text, "page_number": page_num + 1})
    doc.close()
    return pages, page_count


def extract_text_from_docx(file_path: str) -> Tuple[List[Dict], int]:
    """Extract text from a DOCX file.

    Returns:
        A tuple of (list with a single {text, page_number} dict, page count=1).
    """
    doc = DocxDocument(file_path)
    full_text = "\n".join([para.text for para in doc.paragraphs if para.text.strip()])
    # DOCX doesn't have reliable page numbers; we treat the whole doc as page 1
    return [{"text": full_text, "page_number": 1}] if full_text else [], 1


def extract_text_from_txt(file_path: str) -> Tuple[List[Dict], int]:
    """Extract text from a plain TXT file.

    Returns:
        A tuple of (list with a single {text, page_number} dict, page count=1).
    """
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    return [{"text": text, "page_number": 1}] if text.strip() else [], 1


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------


def chunk_text(pages: List[Dict]) -> List[Dict]:
    """Split extracted pages into smaller chunks with metadata.

    Each chunk retains the page_number from which it was extracted.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
    )

    chunks = []
    for page in pages:
        page_chunks = splitter.split_text(page["text"])
        for chunk_text_piece in page_chunks:
            chunks.append(
                {
                    "text": chunk_text_piece,
                    "page_number": page["page_number"],
                }
            )
    return chunks


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------


def generate_embeddings(texts: List[str]) -> List[List[float]]:
    """Generate embeddings for a list of texts via OpenAI text-embedding-3-small."""
    embeddings = []
    # OpenAI has a hard limit of 2048 inputs per request. We batch at 1000 for safety.
    batch_size = 1000
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        response = client.embeddings.create(
            input=batch,
            model="text-embedding-3-small",
        )
        embeddings.extend([item.embedding for item in response.data])
    return embeddings


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------


def process_document(
    file_path: str, file_type: str, document_id: int, filename: str
) -> Tuple[int, int]:
    """End-to-end document processing pipeline.

    1. Extract text from the file.
    2. Chunk the text.
    3. Generate embeddings.
    4. Store chunks + embeddings in ChromaDB.

    Returns:
        A tuple of (chunk_count, page_count).
    """
    # 1. Extract text based on file type
    extractors = {
        "pdf": extract_text_from_pdf,
        "docx": extract_text_from_docx,
        "txt": extract_text_from_txt,
    }
    extractor = extractors.get(file_type)
    if not extractor:
        raise ValueError(f"Unsupported file type: {file_type}")

    pages, page_count = extractor(file_path)

    if not pages:
        return 0, page_count

    # 2. Chunk
    chunks = chunk_text(pages)

    if not chunks:
        return 0, page_count

    # 3. Generate embeddings
    texts = [c["text"] for c in chunks]
    embeddings = generate_embeddings(texts)

    # 4. Store in ChromaDB
    collection = get_chroma_collection()
    ids = [f"{document_id}_{i}" for i in range(len(chunks))]
    metadatas = [
        {
            "document_id": document_id,
            "page_number": c["page_number"],
            "filename": filename,
        }
        for c in chunks
    ]
    collection.add(
        ids=ids,
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    # 5. Clean up uploaded file
    if os.path.exists(file_path):
        os.remove(file_path)

    return len(chunks), page_count
