"""Sanity-check your Gemini API key and the configured models.

Usage (from the rag-pipeline directory):
    python scripts/check_gemini.py

It makes one embedding call and one generation call with the models from your
.env, and lists the models your key can use if either fails. The key is never
printed.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from google import genai  # noqa: E402

from app.config import settings  # noqa: E402
from app.services.gemini import embed_query, generate_text  # noqa: E402


def _list_models():
    client = genai.Client(api_key=settings.GEMINI_API_KEY)
    print("\nModels available to your key:")
    for m in client.models.list():
        actions = ", ".join(getattr(m, "supported_actions", None) or [])
        print(f"  {m.name.replace('models/', ''):45s} {actions}")


def main() -> int:
    if not settings.GEMINI_API_KEY:
        print("GEMINI_API_KEY is not set in .env")
        return 1

    ok = True
    try:
        vec = embed_query("hello world")
        print(f"[ok] embeddings: {settings.GEMINI_EMBEDDING_MODEL} -> {len(vec)} dims")
    except Exception as exc:  # noqa: BLE001
        ok = False
        print(f"[FAIL] embeddings ({settings.GEMINI_EMBEDDING_MODEL}): {exc}")

    try:
        reply = generate_text("Answer in one word.", "What is the capital of France?")
        print(f"[ok] generation: {settings.GEMINI_LLM_MODEL} -> {reply.strip()[:60]!r}")
    except Exception as exc:  # noqa: BLE001
        ok = False
        print(f"[FAIL] generation ({settings.GEMINI_LLM_MODEL}): {exc}")

    if not ok:
        _list_models()
        print("\nSet GEMINI_LLM_MODEL / GEMINI_EMBEDDING_MODEL in .env to a listed model.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
