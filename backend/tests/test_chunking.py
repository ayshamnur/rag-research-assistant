import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.chunking import chunk_document


def test_short_text_single_chunk():
    chunks = chunk_document("d1", "Title", "http://x", "one two three", chunk_size_tokens=220, overlap_tokens=40)
    assert len(chunks) == 1
    assert chunks[0].text == "one two three"


def test_empty_text_no_chunks():
    assert chunk_document("d1", "Title", "http://x", "", 220, 40) == []


def test_long_text_produces_overlapping_chunks():
    words = [f"word{i}" for i in range(1000)]
    text = " ".join(words)
    chunks = chunk_document("d1", "Title", "http://x", text, chunk_size_tokens=100, overlap_tokens=20)
    assert len(chunks) > 1
    # consecutive chunks should share some words (overlap)
    first_words = set(chunks[0].text.split())
    second_words = set(chunks[1].text.split())
    assert first_words & second_words

    # chunk ids are unique and sequential
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


def test_last_chunk_reaches_end_of_text():
    words = [f"w{i}" for i in range(53)]
    text = " ".join(words)
    chunks = chunk_document("d1", "Title", "http://x", text, chunk_size_tokens=40, overlap_tokens=10)
    assert chunks[-1].text.split()[-1] == "w52"
