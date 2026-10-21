"""Thin wrapper around the Google Gemini API (embeddings + text generation).

Centralising the SDK here keeps the rest of the app provider-agnostic: ingestion,
retrieval and the LLM service only call `embed_documents`, `embed_query` and
`generate_text`. It also owns the retry policy, which matters on the free tier
where 429 (RESOURCE_EXHAUSTED) responses are routine.
"""

import logging
import math
from typing import List, Optional

from google import genai
from google.genai import errors, types
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_random_exponential,
)

from app.config import settings

logger = logging.getLogger(__name__)

_client: Optional[genai.Client] = None


def _get_client() -> genai.Client:
    """Create the Gemini client lazily so importing the app never needs a key."""
    global _client
    if _client is None:
        if not settings.GEMINI_API_KEY:
            raise RuntimeError(
                "GEMINI_API_KEY is not set. Get a free key at "
                "https://aistudio.google.com/apikey and add it to your .env file."
            )
        _client = genai.Client(api_key=settings.GEMINI_API_KEY)
    return _client


def _is_retryable(exc: BaseException) -> bool:
    """Retry rate limits (429) and transient server errors (500, 502, 503, 504)."""
    code = getattr(exc, "code", None)
    if code in (429, 500, 502, 503, 504):
        return True
    if isinstance(exc, errors.ServerError):
        return True
    return isinstance(exc, errors.ClientError) and code == 429


_retry_policy = retry(
    retry=retry_if_exception(_is_retryable),
    wait=wait_random_exponential(multiplier=1.5, max=15),
    stop=stop_after_attempt(6),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)


def _normalize(vector: List[float]) -> List[float]:
    """L2-normalise a vector.

    Gemini only returns unit-length embeddings at the full 3072 dimensions;
    truncated (Matryoshka) vectors must be re-normalised before cosine/L2 search.
    """
    norm = math.sqrt(sum(v * v for v in vector))
    return [v / norm for v in vector] if norm else vector


@_retry_policy
def _embed_batch(texts: List[str], task_type: str) -> List[List[float]]:
    response = _get_client().models.embed_content(
        model=settings.GEMINI_EMBEDDING_MODEL,
        contents=texts,
        config=types.EmbedContentConfig(
            task_type=task_type,
            output_dimensionality=settings.EMBEDDING_DIMENSIONS,
        ),
    )
    return [_normalize(list(e.values)) for e in response.embeddings]


def embed_documents(texts: List[str]) -> List[List[float]]:
    """Embed document chunks (task type RETRIEVAL_DOCUMENT), in batches."""
    embeddings: List[List[float]] = []
    size = settings.EMBEDDING_BATCH_SIZE
    for i in range(0, len(texts), size):
        embeddings.extend(_embed_batch(texts[i : i + size], "RETRIEVAL_DOCUMENT"))
    return embeddings


def embed_query(text: str) -> List[float]:
    """Embed a search query (task type RETRIEVAL_QUERY).

    Gemini embeddings are asymmetric: queries and documents use different task
    types so that short questions land near the passages that answer them.
    """
    return _embed_batch([text], "RETRIEVAL_QUERY")[0]


@_retry_policy
def _call_generate_content(model: str, system_prompt: str, user_prompt: str) -> str:
    response = _get_client().models.generate_content(
        model=model,
        contents=user_prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=settings.LLM_TEMPERATURE,
            max_output_tokens=settings.LLM_MAX_OUTPUT_TOKENS,
        ),
    )
    return response.text or ""


def generate_text(system_prompt: str, user_prompt: str, model_name: Optional[str] = None) -> str:
    """Generate a completion with automatic fallback if a model experiences 503 high demand."""
    primary_model = model_name or settings.GEMINI_LLM_MODEL
    fallback_models = ["gemini-3.5-flash", "gemini-3.5-flash-lite"]

    candidate_models = [primary_model] + [m for m in fallback_models if m != primary_model]

    last_error = None
    for model in candidate_models:
        try:
            text = _call_generate_content(model, system_prompt, user_prompt)
            if not text:
                logger.warning("Model %s returned empty text.", model)
                return "The model returned an empty response. Please try rephrasing your question."
            return text
        except Exception as exc:
            code = getattr(exc, "code", None)
            if code in (503, 500, 502, 504, 429):
                logger.warning("Model %s failed with %s. Attempting fallback...", model, exc)
                last_error = exc
                continue
            raise

    if last_error:
        raise last_error
    return "Generation failed across all models."


generate_text.retry = _call_generate_content.retry
