"""LLM service — build prompts and generate answers via OpenAI gpt-4o-mini."""

from typing import Dict, List

from openai import OpenAI

from app.config import settings

client = OpenAI(api_key=settings.OPENAI_API_KEY)

SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer questions strictly based "
    "on the provided context. If the answer is not in the context, "
    "say 'I could not find relevant information in the uploaded documents.' "
    "Always be concise and accurate."
)


def generate_answer(question: str, chunks: List[Dict]) -> str:
    """Generate an LLM answer grounded in the retrieved chunks.

    Args:
        question: The user's question.
        chunks: List of dicts with at least a "text" key.

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

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.3,
        max_tokens=1024,
    )

    return response.choices[0].message.content
