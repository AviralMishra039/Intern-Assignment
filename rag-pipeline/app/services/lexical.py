"""Lexical (BM25) search over the chunks stored in ChromaDB.

The index is built lazily from the Chroma collection (the single source of
truth) and cached in memory. It is invalidated whenever documents are added or
deleted, and also rebuilt defensively if the collection size changes underneath
it. With the app's document cap (MAX_DOCUMENTS) the corpus is small enough that
an in-memory index is simple and fast; for a larger corpus this would move to a
dedicated engine (Postgres FTS, OpenSearch, Qdrant sparse vectors, ...).
"""

import re
import threading
from typing import Dict, List, Optional

from rank_bm25 import BM25Okapi

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)

_lock = threading.Lock()
_index: Optional[Dict] = None


def tokenize(text: str) -> List[str]:
    """Lowercase and split on word characters (keeps IDs, acronyms, numbers)."""
    return _TOKEN_RE.findall(text.lower())


def invalidate_index() -> None:
    """Mark the cached BM25 index stale; it is rebuilt on the next search."""
    global _index
    with _lock:
        _index = None


def _build_index(collection) -> Dict:
    data = collection.get(include=["documents", "metadatas"])
    ids = data.get("ids") or []
    texts = data.get("documents") or []
    metadatas = data.get("metadatas") or []
    bm25 = BM25Okapi([tokenize(t) for t in texts]) if texts else None
    return {
        "bm25": bm25,
        "ids": ids,
        "texts": texts,
        "metadatas": metadatas,
        "count": len(ids),
    }


def _get_index(collection) -> Dict:
    global _index
    count = collection.count()
    with _lock:
        if _index is None or _index["count"] != count:
            _index = _build_index(collection)
        return _index


def bm25_search(
    collection,
    query: str,
    top_n: int,
    document_ids: Optional[List[int]] = None,
) -> List[Dict]:
    """Return the top_n chunks by BM25 score for `query`.

    Args:
        collection: The ChromaDB collection holding the chunks.
        query: The user's question.
        top_n: Maximum number of results.
        document_ids: Optional filter — only consider chunks from these documents.

    Returns:
        A list of {id, text, metadata, score} dicts, best first. Chunks with no
        keyword overlap (score <= 0) are dropped.
    """
    index = _get_index(collection)
    bm25 = index["bm25"]
    query_tokens = tokenize(query)
    if bm25 is None or not query_tokens:
        return []

    scores = bm25.get_scores(query_tokens)
    allowed = set(document_ids) if document_ids else None

    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    results: List[Dict] = []
    for i in ranked:
        if scores[i] <= 0:
            break
        metadata = index["metadatas"][i]
        if allowed is not None and metadata["document_id"] not in allowed:
            continue
        results.append(
            {
                "id": index["ids"][i],
                "text": index["texts"][i],
                "metadata": metadata,
                "score": float(scores[i]),
            }
        )
        if len(results) >= top_n:
            break
    return results
