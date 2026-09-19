"""End-to-end API tests. Requires the index to already be built
(run `python -m scripts.build_index --reset` from backend/ first).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health():
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json() == {"status": "ok"}


def test_query_returns_sources():
    res = client.post("/query", json={"question": "How does PagedAttention work?"})
    assert res.status_code == 200
    data = res.json()
    assert "answer" in data
    assert len(data["sources"]) > 0
    # top source should plausibly be the vLLM/PagedAttention paper
    titles = [s["doc_title"] for s in data["sources"]]
    assert any("PagedAttention" in t or "vLLM" in t or "Memory Management" in t for t in titles)


def test_query_top_k_respected():
    res = client.post("/query", json={"question": "retrieval augmented generation", "top_k": 2})
    assert res.status_code == 200
    assert len(res.json()["sources"]) <= 2
