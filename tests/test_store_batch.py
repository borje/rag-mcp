"""Batch delete, deferred BM25 rebuild, tokenizer, and dropped page_* fields."""

import numpy as np
import pytest

import store as store_module
from store import RAGStore


def _keyword_embed(texts: list[str]) -> np.ndarray:
    """Deterministic embedding: 'zebra' and 'apple' axes plus a small bias."""
    return np.array(
        [[float("zebra" in t), float("apple" in t), 1e-3] for t in texts],
        dtype=np.float32,
    )


@pytest.fixture
def rag_store(tmp_path, monkeypatch):
    monkeypatch.setattr(store_module, "STORE_DIR", tmp_path)
    monkeypatch.setenv("RAG_MCP_ADJACENT_CHUNKS", "0")
    s = RAGStore()
    monkeypatch.setattr(s, "_embed", _keyword_embed)
    return s


def _chunk(source: str, i: int, body: str) -> dict:
    return {
        "id": f"{source}-{i}",
        "source": f"/data/docs/{source}",
        "relative_source": f"docs/{source}",
        "doc_title": source,
        "chunk_type": "section",
        "section_path": f"{source} section {i}",
        "chunk_index": 0,
        "chunk_total": 1,
        "title": f"Section {i}",
        "body": body,
    }


def _score_of(results: list[dict], chunk_id: str) -> float:
    return next(r["score"] for r in results if r["id"] == chunk_id)


def test_delete_sources_removes_several_with_one_save(rag_store, monkeypatch):
    for name, n in (("a.md", 2), ("b.md", 3), ("c.md", 1)):
        rag_store.ingest(
            [_chunk(name, i, f"apple body {name} {i}") for i in range(n)], mtime=1.0
        )
    saves = []
    original_save = rag_store._save
    monkeypatch.setattr(rag_store, "_save", lambda: (saves.append(1), original_save()))

    removed = rag_store.delete_sources(["/data/docs/a.md", "/data/docs/b.md", "/nope"])

    assert removed == {"/data/docs/a.md": 2, "/data/docs/b.md": 3, "/nope": 0}
    assert len(saves) == 1
    assert rag_store.list_sources() == ["/data/docs/c.md"]
    assert rag_store.source_mtime("/data/docs/a.md") is None
    assert rag_store.source_mtime("/data/docs/c.md") == 1.0
    assert len(rag_store.load_bodies()) == 1
    assert [r["id"] for r in rag_store.search("apple", n=5)] == ["c.md-0"]


def test_delete_source_still_returns_count(rag_store):
    rag_store.ingest([_chunk("a.md", i, f"apple {i}") for i in range(2)])
    assert rag_store.delete_source("/data/docs/a.md") == 2
    assert rag_store.delete_source("/data/docs/a.md") == 0


def test_ingest_without_bm25_rebuild_still_searches_by_vector(rag_store):
    rag_store.ingest([_chunk("a.md", i, f"apple body {i}") for i in range(3)])
    rag_store.ingest([_chunk("b.md", 0, "zebra body quokka")], rebuild_bm25=False)

    # Scoped search indexes bm25 scores by chunk position, so it would crash
    # if the stale BM25 array were not padded to the new chunk count.
    results = rag_store.search("zebra", n=4, scope="docs")
    assert "b.md-0" in [r["id"] for r in results]


def test_rebuild_bm25_makes_new_chunk_keyword_searchable(rag_store):
    rag_store.ingest([_chunk("a.md", i, f"apple body {i}") for i in range(3)])
    rag_store.ingest([_chunk("b.md", 0, "zebra body quokka")], rebuild_bm25=False)
    before = _score_of(rag_store.search("quokka", n=4, scope="docs"), "b.md-0")

    rag_store.rebuild_bm25()

    after = _score_of(rag_store.search("quokka", n=4, scope="docs"), "b.md-0")
    assert after > before


def test_tokenizer_splits_on_punctuation():
    assert store_module._tokenize("GET /api/users.") == ["get", "api", "users"]


def test_bm25_matches_path_tokens(rag_store):
    rag_store.ingest(
        [
            _chunk("a.md", 0, "apple pie recipe"),
            _chunk("b.md", 0, "GET /api/users."),
        ]
    )
    scores = rag_store._bm25.get_scores(store_module._tokenize("api users"))
    assert scores[1] > 0
    assert scores[0] == 0


def test_ingested_chunks_have_no_page_fields(rag_store):
    rag_store.ingest([_chunk("a.md", 0, "apple body")])
    result = rag_store.search("apple", n=1)[0]
    assert "page_start" not in result
    assert "page_end" not in result


# ── BM25 index construction ──────────────────────────────────────────────────


def _reference_bm25(corpus, query, k1=1.5, b=0.75):
    """Straightforward BM25Okapi (same IDF formula as _SparseBM25) for cross-checking."""
    import math

    n = len(corpus)
    avgdl = sum(len(d) for d in corpus) / n
    df = {}
    for d in corpus:
        for t in set(d):
            df[t] = df.get(t, 0) + 1
    scores = []
    for d in corpus:
        s = 0.0
        for t in query:
            tf = d.count(t)
            if tf == 0:
                continue
            idf = math.log1p((n - df[t] + 0.5) / (df[t] + 0.5))
            s += idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * len(d) / avgdl))
        scores.append(s)
    return np.array(scores, dtype=np.float32)


def test_sparse_bm25_matches_reference_on_random_corpus():
    import random

    from store import _SparseBM25

    rng = random.Random(7)
    vocab = [f"w{i}" for i in range(50)]
    corpus = [rng.choices(vocab, k=rng.randint(1, 40)) for _ in range(200)]
    corpus[3] = []  # empty document must not break construction
    idx = _SparseBM25(corpus)
    for query in (["w1"], ["w1", "w2", "w2", "unknown"], vocab[:10]):
        np.testing.assert_allclose(
            idx.get_scores(query), _reference_bm25(corpus, query), rtol=1e-5, atol=1e-6
        )


def test_sparse_bm25_empty_corpus_and_no_tokens():
    from store import _SparseBM25

    assert _SparseBM25([]).get_scores(["x"]).shape == (0,)
    assert list(_SparseBM25([[], []]).get_scores(["x"])) == [0.0, 0.0]


def test_delete_sources_can_defer_bm25_rebuild(rag_store, monkeypatch):
    rag_store.ingest([_chunk("a.md", 0, "alpha body one"), _chunk("a.md", 1, "alpha body two")])
    rag_store.ingest([_chunk("b.md", 0, "beta body one"), _chunk("b.md", 1, "beta body two")])
    calls = []
    monkeypatch.setattr(rag_store, "_rebuild_bm25", lambda: calls.append(1))
    rag_store.delete_sources(["/data/docs/a.md"], rebuild_bm25=False)
    assert calls == []
    # Positions shifted, so the stale index is dropped and search is vector-only until rebuild.
    assert rag_store._bm25 is None
    results = rag_store.search("beta body", n=4)
    assert results and all(r["source"] == "/data/docs/b.md" for r in results)
    rag_store.rebuild_bm25()
    assert calls == [1]
