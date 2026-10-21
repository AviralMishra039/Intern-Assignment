"""LLM service — build prompts and generate answers via Google Gemini."""

from typing import Dict, List, Optional

from app.services.gemini import generate_text

SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer questions strictly based "
    "on the provided context. If the answer is not in the context, "
    "say 'I could not find relevant information in the uploaded documents.' "
    "Always be concise and accurate."
)


def generate_answer(question: str, chunks: List[Dict], model_name: Optional[str] = None) -> str:
    """Generate an LLM answer grounded in the retrieved chunks.

    Args:
        question: The user's question.
        chunks: List of dicts with at least a "text" key.
        model_name: Optional override for the Gemini model.

    Returns:
        The assistant's answer string.
    """
    if not chunks:
        return "I could not find relevant information in the uploaded documents."

    # Build context from chunks
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        context_parts.append(f"[Chunk {i}]\n{chunk['text']}")
    context = "\n\n---\n\n".join(context_parts)

    user_prompt = f"Context:\n{context}\n\nQuestion: {question}"

    return generate_text(SYSTEM_PROMPT, user_prompt, model_name=model_name)
