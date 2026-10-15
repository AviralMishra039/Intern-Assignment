"""Tests for token-based chunking, BM25 search, RRF fusion, and hybrid retrieval."""

from unittest.mock import MagicMock

import chromadb
import pytest
import tiktoken

from app.config import settings
from app.services import ingestion, lexical, retrieval


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

# 3-d toy embeddings: each chunk points along its own axis.
CHUNKS = [
    ("c0", "The invoice number INV-2024-0042 was issued in March.", 1, [1.0, 0.0, 0.0]),
    ("c1", "Photosynthesis converts sunlight into chemical energy in plants.", 2, [0.0, 1.0, 0.0]),
    ("c2", "Cats are small domesticated carnivorous mammals.", 2, [0.0, 0.0, 1.0]),
    ("c3", "The quarterly budget review covers marketing and engineering spend.", 1, [0.6, 0.6, 0.6]),
]


@pytest.fixture()
def collection(tmp_path):
    client = chromadb.PersistentClient(path=str(tmp_path))
    col = client.get_or_create_collection(name="retrieval_test")
    col.add(
        ids=[c[0] for c in CHUNKS],
        documents=[c[1] for c in CHUNKS],
        embeddings=[c[3] for c in CHUNKS],
        metadatas=[
            {"document_id": c[2], "page_number": 1, "filename": f"doc{c[2]}.txt"}
            for c in CHUNKS
        ],
    )
    lexical.invalidate_index()
    yield col
    lexical.invalidate_index()


@pytest.fixture()
def patched_retrieval(collection, monkeypatch):
    """Point retrieval at the temp collection and fake the query embedding."""
    query_embedding = {"value": [0.0, 1.0, 0.0]}  # semantically closest to c1

    monkeypatch.setattr(retrieval, "embed_query", lambda text: query_embedding["value"])
    monkeypatch.setattr(retrieval, "get_chroma_collection", lambda: collection)
    monkeypatch.setattr(settings, "RERANKER_ENABLED", False)
    monkeypatch.setattr(settings, "HYBRID_SEARCH_ENABLED", True)
    return query_embedding


# ---------------------------------------------------------------------------
# Token-based chunking
# ---------------------------------------------------------------------------


def test_chunks_respect_token_budget_and_overlap(monkeypatch):
    monkeypatch.setattr(ingestion, "_splitter", None)  # rebuild with these settings
    monkeypatch.setattr(settings, "CHUNK_SIZE_TOKENS", 64)
    monkeypatch.setattr(settings, "CHUNK_OVERLAP_TOKENS", 16)

    text = " ".join(f"sentence{i} talks about topic{i}." for i in range(300))
    chunks = ingestion.chunk_text([{"text": text, "page_number": 3}])
    monkeypatch.setattr(ingestion, "_splitter", None)  # don't leak to other tests

    assert len(chunks) > 1
    encoding = tiktoken.get_encoding(settings.TOKENIZER_ENCODING)
    for chunk in chunks:
        assert len(encoding.encode(chunk["text"])) <= 64
        assert chunk["page_number"] == 3

    # Overlap: the tail of a chunk is repeated at the start of the next one.
    assert chunks[0]["text"].split()[-1] in chunks[1]["text"]


# ---------------------------------------------------------------------------
# BM25
# ---------------------------------------------------------------------------


def test_bm25_finds_exact_identifier(collection):
    hits = lexical.bm25_search(collection, "INV-2024-0042", top_n=3)
    assert hits and hits[0]["id"] == "c0"


def test_bm25_respects_document_filter(collection):
    hits = lexical.bm25_search(collection, "invoice cats", top_n=5, document_ids=[2])
    assert hits
    assert all(h["metadata"]["document_id"] == 2 for h in hits)


def test_bm25_no_keyword_overlap_returns_nothing(collection):
    assert lexical.bm25_search(collection, "zzzunknownzzz", top_n=5) == []


def test_bm25_index_reflects_new_chunks_after_invalidation(collection):
    assert lexical.bm25_search(collection, "kubernetes", top_n=3) == []
    collection.add(
        ids=["c9"],
        documents=["Kubernetes orchestrates containers."],
        embeddings=[[0.1, 0.2, 0.3]],
        metadatas=[{"document_id": 3, "page_number": 1, "filename": "k8s.txt"}],
    )
    lexical.invalidate_index()
    hits = lexical.bm25_search(collection, "kubernetes", top_n=3)
    assert hits and hits[0]["id"] == "c9"


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion
# ---------------------------------------------------------------------------


def test_rrf_promotes_items_ranked_by_both_retrievers():
    a, b, c = ({"id": x, "text": x} for x in "abc")
    # "b" is 2nd in both lists, "a" is 1st in only one, "c" is 3rd in one.
    fused = retrieval.reciprocal_rank_fusion([[a, b], [c, b]], k=60)
    assert [f["id"] for f in fused][0] == "b"
    assert {f["id"] for f in fused} == {"a", "b", "c"}
    assert all("rrf_score" in f for f in fused)


def test_rrf_single_list_keeps_order():
    items = [{"id": str(i), "text": str(i)} for i in range(4)]
    fused = retrieval.reciprocal_rank_fusion([items], k=60)
    assert [f["id"] for f in fused] == ["0", "1", "2", "3"]


# ---------------------------------------------------------------------------
# End-to-end retrieval
# ---------------------------------------------------------------------------


def test_hybrid_surfaces_keyword_match_that_vector_search_misses(patched_retrieval):
    # Query embedding points at c1 (photosynthesis) but the query text names the invoice ID.
    settings_hybrid = retrieval.retrieve_chunks("INV-2024-0042", top_k=2)
    ids_hybrid = [c["id"] for c in settings_hybrid]
    assert "c0" in ids_hybrid and "c1" in ids_hybrid


def test_vector_only_when_hybrid_disabled(patched_retrieval, monkeypatch):
    monkeypatch.setattr(settings, "HYBRID_SEARCH_ENABLED", False)
    results = retrieval.retrieve_chunks("INV-2024-0042", top_k=1)
    assert [c["id"] for c in results] == ["c1"]  # purely semantic nearest neighbour


def test_retrieval_respects_document_filter(patched_retrieval):
    results = retrieval.retrieve_chunks("INV-2024-0042", document_ids=[2], top_k=5)
    assert results
    assert all(c["document_id"] == 2 for c in results)


def test_result_shape(patched_retrieval):
    chunk = retrieval.retrieve_chunks("photosynthesis", top_k=1)[0]
    assert {"text", "document_id", "filename", "page_number"} <= set(chunk)


def test_empty_collection_returns_no_chunks(patched_retrieval, collection):
    collection.delete(ids=[c[0] for c in CHUNKS])
    assert retrieval.retrieve_chunks("anything") == []


def test_reranker_reorders_and_truncates(patched_retrieval, monkeypatch):
    monkeypatch.setattr(settings, "RERANKER_ENABLED", True)
    seen = {}

    def fake_rerank(query, chunks, top_k):
        seen["n_candidates"] = len(chunks)
        return list(reversed(chunks))[:top_k]

    monkeypatch.setattr(retrieval, "rerank", fake_rerank)
    results = retrieval.retrieve_chunks("photosynthesis", top_k=2)
    assert len(results) == 2
    assert seen["n_candidates"] >= 2


def test_reranker_failure_falls_back_to_fused_order(patched_retrieval, monkeypatch):
    monkeypatch.setattr(settings, "RERANKER_ENABLED", True)
    monkeypatch.setattr(
        retrieval, "rerank", MagicMock(side_effect=RuntimeError("model missing"))
    )
    results = retrieval.retrieve_chunks("photosynthesis", top_k=2)
    assert len(results) == 2
