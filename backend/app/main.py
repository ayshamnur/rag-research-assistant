from __future__ import annotations

from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .config import settings
from .generate import generate_answer
from .retriever import HybridRetriever

app = FastAPI(
    title="RAG Research Assistant",
    description="Hybrid (dense + BM25) retrieval over a curated ML-papers corpus, "
    "with citation-grounded answer generation.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_allow_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

_retriever: Optional[HybridRetriever] = None


def get_retriever() -> HybridRetriever:
    global _retriever
    if _retriever is None:
        _retriever = HybridRetriever()
    return _retriever


class QueryRequest(BaseModel):
    question: str
    top_k: Optional[int] = None


class SourceOut(BaseModel):
    doc_title: str
    doc_url: str
    text: str
    fused_score: float


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceOut]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    retriever = get_retriever()
    chunks = retriever.search(req.question, top_k=req.top_k)
    answer = generate_answer(req.question, chunks)
    return QueryResponse(
        answer=answer,
        sources=[
            SourceOut(
                doc_title=c.doc_title,
                doc_url=c.doc_url,
                text=c.text,
                fused_score=round(c.fused_score, 4),
            )
            for c in chunks
        ],
    )
