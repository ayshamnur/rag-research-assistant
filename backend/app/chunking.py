"""Simple, dependency-light word-based chunker with overlap.

A real production system would chunk on sentence/paragraph boundaries with a
tokenizer that matches the embedding model. We chunk on whitespace-split
"words" as a token proxy (~0.75 words/token for English), which is accurate
enough for consistent chunk sizes and keeps the pipeline free of an extra
tokenizer dependency.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    doc_title: str
    doc_url: str
    text: str
    chunk_index: int


def chunk_document(
    doc_id: str,
    title: str,
    url: str,
    text: str,
    chunk_size_tokens: int = 220,
    overlap_tokens: int = 40,
) -> list[Chunk]:
    words = text.split()
    if not words:
        return []

    # ~0.75 words per token for English -> convert token budget to word budget
    chunk_size_words = max(20, int(chunk_size_tokens * 0.75))
    overlap_words = max(0, int(overlap_tokens * 0.75))
    stride = max(1, chunk_size_words - overlap_words)

    chunks: list[Chunk] = []
    idx = 0
    start = 0
    while start < len(words):
        end = min(start + chunk_size_words, len(words))
        chunk_text = " ".join(words[start:end])
        chunks.append(
            Chunk(
                chunk_id=f"{doc_id}::chunk{idx}",
                doc_id=doc_id,
                doc_title=title,
                doc_url=url,
                text=chunk_text,
                chunk_index=idx,
            )
        )
        idx += 1
        if end == len(words):
            break
        start += stride

    return chunks
