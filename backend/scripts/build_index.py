"""Builds the Chroma vector index from backend/data/corpus.json.

Usage:
    python -m scripts.build_index [--reset]

Run this once after changing the corpus or chunking parameters. It is
idempotent-ish: pass --reset to wipe and rebuild the collection from scratch.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.chunking import chunk_document
from app.config import settings
from app.vectorstore import get_or_create_collection


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Wipe the collection first")
    args = parser.parse_args()

    corpus_path = Path(settings.corpus_path)
    if not corpus_path.exists():
        raise SystemExit(f"Corpus file not found: {corpus_path}")

    with open(corpus_path) as f:
        papers = json.load(f)

    collection = get_or_create_collection(reset=args.reset)

    ids, docs, metas = [], [], []
    for paper in papers:
        chunks = chunk_document(
            doc_id=paper["id"],
            title=paper["title"],
            url=paper["url"],
            text=paper["text"],
            chunk_size_tokens=settings.chunk_size_tokens,
            overlap_tokens=settings.chunk_overlap_tokens,
        )
        for c in chunks:
            ids.append(c.chunk_id)
            docs.append(c.text)
            metas.append(
                {
                    "doc_id": c.doc_id,
                    "doc_title": c.doc_title,
                    "doc_url": c.doc_url,
                    "chunk_index": c.chunk_index,
                }
            )

    if not ids:
        raise SystemExit("No chunks produced -- check the corpus file.")

    # Chroma upsert handles re-runs without duplicating rows.
    collection.upsert(ids=ids, documents=docs, metadatas=metas)
    print(f"Indexed {len(ids)} chunks from {len(papers)} papers into '{collection.name}'.")
    print(f"Persisted at: {settings.chroma_persist_dir}")


if __name__ == "__main__":
    main()
