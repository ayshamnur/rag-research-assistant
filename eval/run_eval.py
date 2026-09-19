"""Retrieval evaluation harness.

Measures, against a hand-labeled query set (eval_set.json), whether the
hybrid retriever surfaces at least one chunk from the correct source paper(s)
in its top-k results. This is a retrieval-quality check (precision/recall
proxy at the document level), separate from and complementary to judging
generated-answer quality, which an LLM-judge or human review would cover.

Usage (from backend/, with the venv activated and the index already built):
    python -m eval.run_eval
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent / "backend"
sys.path.insert(0, str(BACKEND_DIR))

from app.config import settings  # noqa: E402
from app.retriever import HybridRetriever  # noqa: E402

EVAL_SET_PATH = Path(__file__).resolve().parent / "eval_set.json"
RESULTS_DIR = Path(__file__).resolve().parent / "results"


def recall_at_k(retrieved_doc_ids: list[str], relevant_doc_ids: list[str]) -> float:
    if not relevant_doc_ids:
        return 0.0
    hit = any(doc_id in retrieved_doc_ids for doc_id in relevant_doc_ids)
    return 1.0 if hit else 0.0


def precision_at_k(retrieved_doc_ids: list[str], relevant_doc_ids: list[str]) -> float:
    if not retrieved_doc_ids:
        return 0.0
    relevant_set = set(relevant_doc_ids)
    hits = sum(1 for d in retrieved_doc_ids if d in relevant_set)
    return hits / len(retrieved_doc_ids)


def main():
    with open(EVAL_SET_PATH) as f:
        eval_set = json.load(f)

    retriever = HybridRetriever()

    rows = []
    recalls, precisions, latencies = [], [], []

    for item in eval_set:
        question = item["question"]
        relevant = item["relevant_doc_ids"]

        t0 = time.perf_counter()
        chunks = retriever.search(question)
        latency_ms = (time.perf_counter() - t0) * 1000

        retrieved_doc_ids = [c.doc_id for c in chunks]
        r = recall_at_k(retrieved_doc_ids, relevant)
        p = precision_at_k(retrieved_doc_ids, relevant)

        recalls.append(r)
        precisions.append(p)
        latencies.append(latency_ms)

        rows.append(
            {
                "question": question,
                "relevant_doc_ids": relevant,
                "retrieved_doc_ids": retrieved_doc_ids,
                "recall_at_k": r,
                "precision_at_k": round(p, 3),
                "latency_ms": round(latency_ms, 1),
            }
        )

    n = len(eval_set)
    summary = {
        "n_queries": n,
        "mean_recall_at_k": round(sum(recalls) / n, 3),
        "mean_precision_at_k": round(sum(precisions) / n, 3),
        "mean_latency_ms": round(sum(latencies) / n, 1),
        "top_k": settings.top_k_final,
    }

    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / "eval_results.json"
    with open(out_path, "w") as f:
        json.dump({"summary": summary, "rows": rows}, f, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"\nFull per-query results written to {out_path}")


if __name__ == "__main__":
    main()
