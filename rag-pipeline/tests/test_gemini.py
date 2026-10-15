"""Tests for the Gemini wrapper — embeddings, batching, normalisation and retry policy.

The real SDK client is replaced with a fake, so no API key or network is needed.
"""

import math
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from google.genai import errors

from app.config import settings
from app.services import gemini


def _fake_embed_response(n, dim=4):
    # Un-normalised vectors so we can verify normalisation happens.
    return SimpleNamespace(
        embeddings=[SimpleNamespace(values=[2.0] * dim) for _ in range(n)]
    )


@pytest.fixture()
def fake_client(monkeypatch):
    client = MagicMock()
    client.models.embed_content.side_effect = lambda **kw: _fake_embed_response(
        len(kw["contents"])
    )
    monkeypatch.setattr(gemini, "_get_client", lambda: client)
    # Make retries instant.
    for fn in (gemini._embed_batch, gemini.generate_text):
        monkeypatch.setattr(fn.retry, "sleep", lambda s: None)
    return client


def test_embeddings_are_unit_length(fake_client):
    vector = gemini.embed_query("hello")
    assert math.isclose(math.sqrt(sum(v * v for v in vector)), 1.0, rel_tol=1e-9)


def test_query_and_document_use_different_task_types(fake_client):
    gemini.embed_query("q")
    gemini.embed_documents(["d"])
    task_types = [
        c.kwargs["config"].task_type for c in fake_client.models.embed_content.call_args_list
    ]
    assert task_types == ["RETRIEVAL_QUERY", "RETRIEVAL_DOCUMENT"]


def test_requested_dimensionality_and_model_are_passed(fake_client):
    gemini.embed_query("q")
    kwargs = fake_client.models.embed_content.call_args.kwargs
    assert kwargs["model"] == settings.GEMINI_EMBEDDING_MODEL
    assert kwargs["config"].output_dimensionality == settings.EMBEDDING_DIMENSIONS


def test_documents_are_batched(fake_client, monkeypatch):
    monkeypatch.setattr(settings, "EMBEDDING_BATCH_SIZE", 3)
    result = gemini.embed_documents([f"t{i}" for i in range(7)])
    assert len(result) == 7
    batch_sizes = [
        len(c.kwargs["contents"]) for c in fake_client.models.embed_content.call_args_list
    ]
    assert batch_sizes == [3, 3, 1]


def test_rate_limit_is_retried_then_succeeds(fake_client):
    rate_limit = errors.ClientError(429, {"error": {"message": "quota"}})
    fake_client.models.embed_content.side_effect = [
        rate_limit,
        rate_limit,
        _fake_embed_response(1),
    ]
    assert len(gemini.embed_query("q")) == 4
    assert fake_client.models.embed_content.call_count == 3


def test_non_retryable_error_is_raised_immediately(fake_client):
    fake_client.models.embed_content.side_effect = errors.ClientError(
        400, {"error": {"message": "bad request"}}
    )
    with pytest.raises(errors.ClientError):
        gemini.embed_query("q")
    assert fake_client.models.embed_content.call_count == 1


def test_persistent_rate_limit_eventually_raises(fake_client):
    fake_client.models.embed_content.side_effect = errors.ClientError(
        429, {"error": {"message": "quota"}}
    )
    with pytest.raises(errors.ClientError):
        gemini.embed_query("q")
    assert fake_client.models.embed_content.call_count == 6  # stop_after_attempt(6)


def test_generate_text_passes_system_prompt(fake_client):
    fake_client.models.generate_content.return_value = SimpleNamespace(text="an answer")
    assert gemini.generate_text("be strict", "question?") == "an answer"
    kwargs = fake_client.models.generate_content.call_args.kwargs
    assert kwargs["contents"] == "question?"
    assert kwargs["config"].system_instruction == "be strict"
    assert kwargs["model"] == settings.GEMINI_LLM_MODEL


def test_generate_text_handles_empty_response(fake_client):
    fake_client.models.generate_content.return_value = SimpleNamespace(text=None)
    assert "empty response" in gemini.generate_text("s", "q").lower()


def test_missing_api_key_gives_clear_error(monkeypatch):
    monkeypatch.setattr(gemini, "_client", None)
    monkeypatch.setattr(settings, "GEMINI_API_KEY", "")
    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        gemini._get_client()
