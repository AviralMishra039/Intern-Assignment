"""Retrieval service — embed a question and find relevant chunks from ChromaDB."""

from typing import Dict, List, Optional

from openai import OpenAI

from app.config import settings
from app.database import get_chroma_collection

client = OpenAI(api_key=settings.OPENAI_API_KEY)


def retrieve_chunks(
    question: str,
    document_ids: Optional[List[int]] = None,
    top_k: int = settings.TOP_K_RESULTS,
) -> List[Dict]:
    """Retrieve the most relevant chunks for a given question.

    Args:
        question: The user's natural-language question.
        document_ids: Optional filter — only search within these document IDs.
        top_k: Number of results to return.

    Returns:
        A list of dicts with keys: text, document_id, filename, page_number.
    """
    # 1. Embed the question
    response = client.embeddings.create(
        input=[question],
        model="text-embedding-3-small",
    )
    query_embedding = response.data[0].embedding

    # 2. Build optional where-filter for ChromaDB
    collection = get_chroma_collection()
    query_params = {
        "query_embeddings": [query_embedding],
        "n_results": top_k,
    }

    if document_ids:
        if len(document_ids) == 1:
            query_params["where"] = {"document_id": document_ids[0]}
        else:
            query_params["where"] = {
                "$or": [{"document_id": doc_id} for doc_id in document_ids]
            }

    # 3. Query ChromaDB
    results = collection.query(**query_params)

    # 4. Format results
    chunks = []
    if results and results["documents"]:
        for i, doc_text in enumerate(results["documents"][0]):
            metadata = results["metadatas"][0][i]
            chunks.append(
                {
                    "text": doc_text,
                    "document_id": metadata["document_id"],
                    "filename": metadata["filename"],
                    "page_number": metadata["page_number"],
                }
            )

    return chunks
