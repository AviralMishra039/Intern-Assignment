"""Tests for the query endpoint."""

import io
from unittest.mock import patch, MagicMock


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------


def _mock_retrieve_chunks(question, document_ids=None, top_k=5):
    """Return fake chunks without calling OpenAI embeddings or ChromaDB."""
    return [
        {
            "text": "The capital of France is Paris. It is known for the Eiffel Tower.",
            "document_id": 1,
            "filename": "test.txt",
            "page_number": 1,
        },
    ]


def _mock_retrieve_chunks_filtered(question, document_ids=None, top_k=5):
    """Return chunks that respect the document_ids filter."""
    all_chunks = [
        {
            "text": "Document one content about history.",
            "document_id": 1,
            "filename": "doc1.txt",
            "page_number": 1,
        },
        {
            "text": "Document two content about science.",
            "document_id": 2,
            "filename": "doc2.txt",
            "page_number": 1,
        },
    ]
    if document_ids:
        return [c for c in all_chunks if c["document_id"] in document_ids]
    return all_chunks


def _mock_retrieve_no_chunks(question, document_ids=None, top_k=5):
    """Return empty chunks to simulate no documents uploaded."""
    return []


def _mock_generate_answer(question, chunks):
    """Return a fake answer without calling OpenAI."""
    if not chunks:
        return "I could not find relevant information in the uploaded documents."
    return "Paris is the capital of France, famous for the Eiffel Tower."


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@patch("app.routers.query.generate_answer", side_effect=_mock_generate_answer)
@patch("app.routers.query.retrieve_chunks", side_effect=_mock_retrieve_chunks)
def test_basic_query(mock_retrieve, mock_llm, client):
    """A basic query should return a non-empty answer with sources."""
    response = client.post(
        "/query",
        json={"question": "What is the capital of France?"},
    )
    assert response.status_code == 200
    data = response.json()
    assert len(data["answer"]) > 0
    assert len(data["sources"]) > 0
    assert data["sources"][0]["document_id"] == 1


def test_query_empty_question(client):
    """An empty question should return 400."""
    response = client.post("/query", json={"question": ""})
    assert response.status_code == 422  # Pydantic validation (min_length=1)


@patch("app.routers.query.generate_answer", side_effect=_mock_generate_answer)
@patch("app.routers.query.retrieve_chunks", side_effect=_mock_retrieve_chunks_filtered)
def test_query_with_document_filter(mock_retrieve, mock_llm, client):
    """Querying with a document_ids filter should only return sources from that doc."""
    response = client.post(
        "/query",
        json={"question": "Tell me about history", "document_ids": [1]},
    )
    assert response.status_code == 200
    data = response.json()
    # All sources should be from document 1
    for source in data["sources"]:
        assert source["document_id"] == 1


@patch("app.routers.query.generate_answer", side_effect=_mock_generate_answer)
@patch("app.routers.query.retrieve_chunks", side_effect=_mock_retrieve_no_chunks)
def test_query_no_documents(mock_retrieve, mock_llm, client):
    """Querying with no documents should return a graceful 'no info' answer."""
    response = client.post(
        "/query",
        json={"question": "What is the meaning of life?"},
    )
    assert response.status_code == 200
    data = response.json()
    assert "could not find" in data["answer"].lower()
    assert data["sources"] == []
