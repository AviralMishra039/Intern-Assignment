"""Cross-encoder reranking via FlashRank (ONNX, CPU, no PyTorch required).

A bi-encoder (the embedding model) scores query and chunk independently, which is
fast but coarse. A cross-encoder reads query + chunk together and is much more
accurate, so we use it to re-order the small fused candidate set.

FlashRank was chosen over sentence-transformers/Cohere because the model is
~22MB, runs on CPU in tens of milliseconds, needs no API key and fits small
(free-tier) deployments.
"""

import threading
from typing import Dict, List

from app.config import settings

_ranker = None
_ranker_lock = threading.Lock()


def _get_ranker():
    """Lazily load the FlashRank model (downloads it on first use)."""
    global _ranker
    with _ranker_lock:
        if _ranker is None:
            from flashrank import Ranker

            _ranker = Ranker(
                model_name=settings.RERANKER_MODEL,
                cache_dir=settings.RERANKER_CACHE_DIR,
            )
        return _ranker


def rerank(query: str, chunks: List[Dict], top_k: int) -> List[Dict]:
    """Re-order `chunks` by cross-encoder relevance to `query` and keep top_k.

    Each chunk must have a "text" key. A "rerank_score" key is added.
    """
    if not chunks:
        return []

    from flashrank import RerankRequest

    passages = [{"id": i, "text": c["text"]} for i, c in enumerate(chunks)]
    results = _get_ranker().rerank(RerankRequest(query=query, passages=passages))

    reranked = []
    for r in results[:top_k]:
        chunk = dict(chunks[r["id"]])
        chunk["rerank_score"] = float(r["score"])
        reranked.append(chunk)
    return reranked
