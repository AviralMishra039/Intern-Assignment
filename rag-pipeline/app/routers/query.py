"""Query endpoint — retrieve relevant chunks and generate an LLM answer."""

from fastapi import APIRouter, HTTPException

from app.models.schemas import QueryRequest, QueryResponse, SourceInfo
from app.services.llm import generate_answer
from app.services.retrieval import retrieve_chunks

router = APIRouter(tags=["Query"])


@router.post("/query", response_model=QueryResponse)
def query_documents(request: QueryRequest):
    """Ask a question across uploaded documents and get an LLM-generated answer.

    Optionally filter by specific document IDs. If no document_ids are provided,
    all documents are searched.
    """
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    # Retrieve relevant chunks
    chunks = retrieve_chunks(
        question=question,
        document_ids=request.document_ids,
    )

    # Generate answer
    answer = generate_answer(question=question, chunks=chunks)

    # Build sources
    sources = [
        SourceInfo(
            document_id=c["document_id"],
            filename=c["filename"],
            page_number=c["page_number"],
            chunk_preview=c["text"][:200],
        )
        for c in chunks
    ]

    return QueryResponse(answer=answer, sources=sources)
