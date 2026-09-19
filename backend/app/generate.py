"""Answer generation grounded in retrieved chunks, with inline citations.

The prompt explicitly instructs the model to answer only from the provided
context and to cite chunk numbers, and to say so if the context is
insufficient -- this is what keeps the system honest instead of letting the
LLM fall back on its own parametric knowledge unchecked.
"""
from __future__ import annotations

from anthropic import Anthropic

from .config import settings
from .retriever import RetrievedChunk

SYSTEM_PROMPT = """You are a research assistant that answers questions ONLY using the numbered \
excerpts provided below, drawn from a corpus of machine learning papers. Rules:

1. Every factual claim in your answer must be traceable to one of the excerpts. \
Cite the excerpt number in square brackets right after the claim, e.g. "vLLM improves \
throughput 2-4x [2]."
2. If the excerpts do not contain enough information to answer the question, say so \
explicitly instead of guessing or using outside knowledge.
3. Be concise. Do not repeat the question back or add unnecessary preamble.
"""


def _format_context(chunks: list[RetrievedChunk]) -> str:
    parts = []
    for i, c in enumerate(chunks, start=1):
        parts.append(f"[{i}] ({c.doc_title})\n{c.text}")
    return "\n\n".join(parts)


def generate_answer(question: str, chunks: list[RetrievedChunk]) -> str:
    if not chunks:
        return "I couldn't find any relevant passages in the corpus to answer this."

    if not settings.anthropic_api_key:
        # Deterministic fallback so the API is still usable/testable without a key.
        top = chunks[0]
        return (
            "[No ANTHROPIC_API_KEY configured -- returning the top retrieved passage "
            f"instead of a generated answer]\n\n({top.doc_title}) {top.text}"
        )

    client = Anthropic(api_key=settings.anthropic_api_key)
    context = _format_context(chunks)
    message = client.messages.create(
        model=settings.generation_model,
        max_tokens=600,
        system=SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": f"Excerpts:\n\n{context}\n\nQuestion: {question}",
            }
        ],
    )
    return "".join(block.text for block in message.content if hasattr(block, "text"))
