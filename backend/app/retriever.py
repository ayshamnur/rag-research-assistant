"""Hybrid retrieval: dense (embedding) search fused with BM25 (lexical) search.

Pure dense retrieval misses exact-term matches (model names, acronyms like
"GPTQ" or "ColBERT"); pure BM25 misses paraphrases and semantic matches.
Fusing both with reciprocal-rank fusion is a well-known way to get the
strengths of each without needing a trained reranker.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

from rank_bm25 import BM25Okapi

from .config import settings
from .vectorstore import get_or_create_collection

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


@dataclass
class RetrievedChunk:
    chunk_id: str
    doc_id: str
    doc_title: str
    doc_url: str
    text: str
    dense_rank: int | None
    bm25_rank: int | None
    fused_score: float


class HybridRetriever:
    """Loads all chunk metadata into memory once and builds a BM25 index
    alongside the persistent Chroma collection for dense search."""

    def __init__(self):
        self.collection = get_or_create_collection()
        all_docs = self.collection.get(include=["documents", "metadatas"])
        self.ids: list[str] = all_docs["ids"]
        self.texts: list[str] = all_docs["documents"]
        self.metadatas: list[dict] = all_docs["metadatas"]
        self._id_to_index = {cid: i for i, cid in enumerate(self.ids)}

        tokenized_corpus = [_tokenize(t) for t in self.texts]
        self.bm25 = BM25Okapi(tokenized_corpus) if tokenized_corpus else None

    def _dense_search(self, query: str, k: int) -> list[tuple[str, int]]:
        if not self.ids:
            return []
        results = self.collection.query(query_texts=[query], n_results=min(k, len(self.ids)))
        return [(cid, rank) for rank, cid in enumerate(results["ids"][0])]

    def _bm25_search(self, query: str, k: int) -> list[tuple[str, int]]:
        if self.bm25 is None:
            return []
        scores = self.bm25.get_scores(_tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [(self.ids[i], rank) for rank, i in enumerate(ranked)]

    def search(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        top_k = top_k or settings.top_k_final
        dense_hits = dict(self._dense_search(query, settings.top_k_dense))
        bm25_hits = dict(self._bm25_search(query, settings.top_k_bm25))

        all_ids = set(dense_hits) | set(bm25_hits)
        rrf_k = 60  # standard reciprocal-rank-fusion smoothing constant
        scored: list[RetrievedChunk] = []
        for cid in all_ids:
            d_rank = dense_hits.get(cid)
            b_rank = bm25_hits.get(cid)
            score = 0.0
            if d_rank is not None:
                score += settings.dense_weight * (1.0 / (rrf_k + d_rank + 1))
            if b_rank is not None:
                score += (1 - settings.dense_weight) * (1.0 / (rrf_k + b_rank + 1))

            idx = self._id_to_index[cid]
            meta = self.metadatas[idx]
            scored.append(
                RetrievedChunk(
                    chunk_id=cid,
                    doc_id=meta["doc_id"],
                    doc_title=meta["doc_title"],
                    doc_url=meta["doc_url"],
                    text=self.texts[idx],
                    dense_rank=d_rank,
                    bm25_rank=b_rank,
                    fused_score=score,
                )
            )

        scored.sort(key=lambda c: c.fused_score, reverse=True)
        return scored[:top_k]
