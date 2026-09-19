"""Vector store wrapper around ChromaDB (persistent, local, no external service).

Embeddings are computed with a local sentence-transformers model, so the
index can be rebuilt offline with no API costs.
"""
from __future__ import annotations

import chromadb
from chromadb.utils import embedding_functions

from .config import settings

_COLLECTION_NAME = "papers"


def get_client() -> chromadb.ClientAPI:
    return chromadb.PersistentClient(
        path=settings.chroma_persist_dir,
        settings=chromadb.Settings(anonymized_telemetry=False),
    )


def get_embedding_function() -> embedding_functions.EmbeddingFunction:
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=settings.embedding_model
    )


def get_or_create_collection(reset: bool = False):
    client = get_client()
    if reset:
        try:
            client.delete_collection(_COLLECTION_NAME)
        except Exception:
            pass
    return client.get_or_create_collection(
        name=_COLLECTION_NAME,
        embedding_function=get_embedding_function(),
        metadata={"hnsw:space": "cosine"},
    )
