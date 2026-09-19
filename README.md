# RAG Research Assistant

A retrieval-augmented question-answering system over a curated corpus of
machine learning papers (retrieval-augmented generation, efficient LLM
inference, dense/lexical retrieval). Built as a personal project to work
through the full RAG pipeline end to end — not just call an API, but build
and evaluate the retrieval layer that makes RAG work or fail.

**Live demo:** _add your deployed URL here once deployed_
**Author:** Aysha — MSc AI student, software engineer

## Why this project

Most "RAG demos" are a LangChain quickstart wrapping a single embedding call.
This one exists to demonstrate the parts that actually matter in production
RAG systems:

- **Hybrid retrieval** (dense + BM25, fused with reciprocal-rank fusion) —
  because pure embedding search misses exact terms like model names and
  acronyms ("GPTQ", "ColBERT"), and pure BM25 misses paraphrase and semantic
  matches.
- **Grounded generation with citations** — the model is instructed to answer
  only from retrieved passages and to cite them, and to say so explicitly
  when the corpus doesn't have the answer, rather than silently falling back
  to its own parametric knowledge.
- **A real evaluation harness** — retrieval recall/precision measured against
  a hand-labeled query set, not eyeballed "it looks right" testing.

## Architecture

```
                    ┌─────────────────┐
   question ──────▶ │   FastAPI /query │
                    └────────┬─────────┘
                             │
              ┌──────────────┴──────────────┐
              ▼                              ▼
     ┌─────────────────┐           ┌──────────────────┐
     │ Dense search     │           │ BM25 search       │
     │ (Chroma +        │           │ (rank_bm25,        │
     │  MiniLM embeds)  │           │  lexical)          │
     └────────┬─────────┘           └─────────┬─────────┘
              │                                │
              └───────────┬────────────────────┘
                           ▼
              Reciprocal-rank fusion (top-k)
                           │
                           ▼
              LLM answer generation, grounded
              in retrieved excerpts, with citations
                           │
                           ▼
              { answer, sources[] } ──▶ frontend
```

## Corpus

12 real, published papers (titles, authors, venues, and abstracts/excerpts
pulled directly from arXiv), covering:

- Foundational RAG: Lewis et al. 2020, RAG survey, RAG evaluation survey
- Retrieval: DPR, ColBERT, HyDE, Contriever
- Efficient inference: PagedAttention/vLLM, FlashAttention, speculative
  decoding, GPTQ
- Self-RAG (adaptive retrieval + self-reflection)

Full list with links in `backend/data/corpus.json`. This is a scoped demo
corpus (abstract/excerpt-level text, not full PDFs) — the pipeline itself
(chunking, indexing, hybrid retrieval, evaluation) is what scales to a larger
corpus, not the specific 12 papers.

## Running locally

```bash
cd backend
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt

cp .env.example .env          # add your ANTHROPIC_API_KEY (optional --
                               # without it, /query returns the top retrieved
                               # passage instead of a generated answer, so
                               # retrieval still works and is testable)

python -m scripts.build_index --reset   # builds the local Chroma index
uvicorn app.main:app --reload --port 8000
```

Then open `frontend/index.html` in a browser and point it at
`http://localhost:8000`.

### Tests

```bash
cd backend
pytest tests/ -v
```

### Evaluation

```bash
cd .   # repo root
python -m eval.run_eval
```

This runs 15 hand-labeled questions against the retriever and reports mean
recall@k, precision@k, and latency, writing full per-query results to
`eval/results/eval_results.json`. See [`docs/eval_results.md`](docs/eval_results.md)
for the latest numbers.

## API

`POST /query`
```json
{ "question": "How does PagedAttention reduce KV cache memory waste?" }
```
returns
```json
{
  "answer": "...cited answer...",
  "sources": [{ "doc_title": "...", "doc_url": "...", "text": "...", "fused_score": 0.031 }]
}
```

## What I'd do next with more time

- Swap the abstract-level corpus for full paper text (PDF extraction +
  section-aware chunking) to test retrieval on longer, noisier documents.
- Add a cross-encoder reranking stage on top of the RRF-fused candidates.
- Add an LLM-judge faithfulness/answer-relevance score to the eval harness,
  alongside the current recall/precision-at-doc-level metric.
- Swap MiniLM for a larger embedding model and A/B the retrieval quality
  difference on the eval set.

## Stack

FastAPI · ChromaDB (persistent, local) · sentence-transformers (MiniLM-L6-v2)
· rank_bm25 · Anthropic API (generation) · vanilla JS frontend
