"""Retrieval service — hybrid search (vector + BM25) fused with RRF, then reranked.

Pipeline:
    1. Dense retrieval: embed the question, query ChromaDB (semantic matches).
    2. Sparse retrieval: BM25 keyword search (exact terms, IDs, acronyms).
    3. Reciprocal Rank Fusion merges both ranked lists without needing their
       (incomparable) scores to be normalised.
    4. A cross-encoder reranks the fused candidates; the top_k go to the LLM.

Each stage can be switched off via settings (HYBRID_SEARCH_ENABLED,
RERANKER_ENABLED), which makes ablation / evaluation straightforward.
"""

import logging
from typing import Dict, List, Optional

from app.config import settings
from app.database import get_chroma_collection
from app.services.gemini import embed_query
from app.services.lexical import bm25_search
from app.services.reranker import rerank

logger = logging.getLogger(__name__)


def _build_where(document_ids: Optional[List[int]]) -> Optional[Dict]:
    """Build a ChromaDB where-filter for the given document IDs."""
    if not document_ids:
        return None
    if len(document_ids) == 1:
        return {"document_id": document_ids[0]}
    return {"$or": [{"document_id": doc_id} for doc_id in document_ids]}


def _to_candidate(chunk_id: str, text: str, metadata: Dict) -> Dict:
    return {
        "id": chunk_id,
        "text": text,
        "document_id": metadata["document_id"],
        "filename": metadata["filename"],
        "page_number": metadata["page_number"],
    }


def _vector_search(
    collection, question: str, top_n: int, document_ids: Optional[List[int]]
) -> List[Dict]:
    """Dense retrieval: nearest chunks to the question embedding."""
    query_params = {
        "query_embeddings": [embed_query(question)],
        "n_results": top_n,
    }
    where = _build_where(document_ids)
    if where:
        query_params["where"] = where

    results = collection.query(**query_params)

    candidates: List[Dict] = []
    if results and results.get("documents") and results["documents"][0]:
        for chunk_id, text, metadata in zip(
            results["ids"][0], results["documents"][0], results["metadatas"][0]
        ):
            candidates.append(_to_candidate(chunk_id, text, metadata))
    return candidates


def reciprocal_rank_fusion(
    ranked_lists: List[List[Dict]], k: int = settings.RRF_K
) -> List[Dict]:
    """Fuse several ranked candidate lists with Reciprocal Rank Fusion.

    score(chunk) = sum over lists of 1 / (k + rank), rank starting at 1.
    Chunks ranked highly by multiple retrievers rise to the top. Returns the
    unique candidates ordered by fused score (a "rrf_score" key is added).
    """
    scores: Dict[str, float] = {}
    by_id: Dict[str, Dict] = {}
    for ranked in ranked_lists:
        for rank, candidate in enumerate(ranked, start=1):
            cid = candidate["id"]
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
            by_id.setdefault(cid, candidate)

    fused = []
    for cid in sorted(scores, key=scores.get, reverse=True):
        item = dict(by_id[cid])
        item["rrf_score"] = scores[cid]
        fused.append(item)
    return fused


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
        A list of dicts with keys: text, document_id, filename, page_number
        (plus id and the stage scores), best first.
    """
    collection = get_chroma_collection()
    total_chunks = collection.count()
    if total_chunks == 0:
        return []

    n_candidates = min(settings.RETRIEVAL_CANDIDATES, total_chunks)

    # 1. Dense retrieval
    ranked_lists = [_vector_search(collection, question, n_candidates, document_ids)]

    # 2. Sparse retrieval (BM25) — optional
    if settings.HYBRID_SEARCH_ENABLED:
        lexical_hits = bm25_search(collection, question, n_candidates, document_ids)
        ranked_lists.append(
            [_to_candidate(h["id"], h["text"], h["metadata"]) for h in lexical_hits]
        )

    # 3. Fuse (a single list passes through unchanged, in its original order)
    candidates = reciprocal_rank_fusion(ranked_lists)

    # 4. Rerank with a cross-encoder — optional, degrade gracefully on failure
    if settings.RERANKER_ENABLED and candidates:
        try:
            return rerank(question, candidates, top_k)
        except Exception:
            logger.exception("Reranker failed; falling back to fused ordering.")

    return candidates[:top_k]
