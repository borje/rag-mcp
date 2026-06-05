"""Tests for mtime-based staleness detection in RAGStore and _ingest_files_root.

Tests marked "FAILS before fix" use source_mtime() / mtime= param that did not
exist in the old code, or rely on re-ingest behaviour the old code lacked.
"""

import os
import json
from pathlib import Path

import numpy as np
import pytest

import store as store_module
from store import RAGStore


# ── fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def rag_store(tmp_path, monkeypatch):
    """Isolated RAGStore backed by tmp_path; _embed returns zero vectors."""
    monkeypatch.setattr(store_module, "STORE_DIR", tmp_path)
    s = RAGStore()
    monkeypatch.setattr(
        s, "_embed", lambda texts: np.zeros((len(texts), 4), dtype=np.float32)
    )
    return s


def _chunks(source: str, tag: str = "v1", n: int = 2) -> list[dict]:
    return [
        {
            "id": f"{tag}-{i}",
            "source": source,
            "doc_title": "test",
            "chunk_type": "section",
            "title": f"Section {i}",
            "body": f"[{tag}] Body text for chunk {i}. Enough content to exceed minimum length filter.",
        }
        for i in range(n)
    ]


def test_default_mode_reports_fastembed_provider(rag_store, monkeypatch):
    monkeypatch.setattr(store_module, "OPENAI_CONFIGURED", False)
    monkeypatch.setattr(store_module, "MODEL_NAME", "BAAI/bge-small-en-v1.5")
    monkeypatch.setattr(store_module, "OPENAI_BASE_URL", None)
    monkeypatch.setattr(store_module, "OPENAI_API_KEY", None)
    monkeypatch.setattr(store_module, "OPENAI_EMBEDDINGS_PATH", "/embeddings")
    monkeypatch.setattr(store_module, "OPENAI_TIMEOUT", 60.0)

    assert rag_store.stats()["model"] == "fastembed:BAAI/bge-small-en-v1.5"


def test_openai_mode_sends_expected_request(tmp_path, monkeypatch):
    monkeypatch.setattr(store_module, "STORE_DIR", tmp_path)
    monkeypatch.setattr(store_module, "OPENAI_CONFIGURED", True)
    monkeypatch.setattr(store_module, "MODEL_NAME", "text-embedding-3-small")
    monkeypatch.setattr(store_module, "OPENAI_BASE_URL", "https://api.openai.com/v1")
    monkeypatch.setattr(store_module, "OPENAI_API_KEY", "sk-test")
    monkeypatch.setattr(store_module, "OPENAI_EMBEDDINGS_PATH", "/embeddings")
    monkeypatch.setattr(store_module, "OPENAI_TIMEOUT", 12.0)
    captured = {}

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return json.dumps({"data": [{"index": 0, "embedding": [1.0, 2.0]}]}).encode()

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["headers"] = dict(request.header_items())
        captured["body"] = json.loads(request.data.decode())
        captured["timeout"] = timeout
        return Response()

    monkeypatch.setattr(store_module.urllib.request, "urlopen", fake_urlopen)

    s = RAGStore()
    s.ingest(_chunks("/docs/api.md", n=1))

    assert captured == {
        "url": "https://api.openai.com/v1/embeddings",
        "headers": {
            "Content-type": "application/json",
            "Authorization": "Bearer sk-test",
        },
        "body": {"model": "text-embedding-3-small", "input": [_chunks("/docs/api.md", n=1)[0]["body"]]},
        "timeout": 12.0,
    }


def test_openai_mode_allows_missing_key_for_local_base_url(tmp_path, monkeypatch):
    monkeypatch.setattr(store_module, "STORE_DIR", tmp_path)
    monkeypatch.setattr(store_module, "OPENAI_CONFIGURED", True)
    monkeypatch.setattr(store_module, "MODEL_NAME", "nomic-embed-text")
    monkeypatch.setattr(store_module, "OPENAI_BASE_URL", "http://localhost:11434/v1")
    monkeypatch.setattr(store_module, "OPENAI_API_KEY", None)
    monkeypatch.setattr(store_module, "OPENAI_EMBEDDINGS_PATH", "/embeddings")
    monkeypatch.setattr(store_module, "OPENAI_TIMEOUT", 60.0)

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return json.dumps({"data": [{"index": 0, "embedding": [1.0, 2.0]}]}).encode()

    def fake_urlopen(request, timeout):
        assert "Authorization" not in dict(request.header_items())
        return Response()

    monkeypatch.setattr(store_module.urllib.request, "urlopen", fake_urlopen)

    assert RAGStore().ingest(_chunks("/docs/api.md", n=1)) == 1


def test_openai_embeddings_are_sorted_by_index(tmp_path, monkeypatch):
    monkeypatch.setattr(store_module, "STORE_DIR", tmp_path)
    monkeypatch.setattr(store_module, "OPENAI_CONFIGURED", True)
    monkeypatch.setattr(store_module, "MODEL_NAME", "text-embedding-3-small")
    monkeypatch.setattr(store_module, "OPENAI_BASE_URL", "https://api.openai.com/v1")
    monkeypatch.setattr(store_module, "OPENAI_API_KEY", "sk-test")
    monkeypatch.setattr(store_module, "OPENAI_EMBEDDINGS_PATH", "/embeddings")
    monkeypatch.setattr(store_module, "OPENAI_TIMEOUT", 60.0)

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return json.dumps(
                {
                    "data": [
                        {"index": 1, "embedding": [2.0, 2.0]},
                        {"index": 0, "embedding": [1.0, 1.0]},
                    ]
                }
            ).encode()

    monkeypatch.setattr(store_module.urllib.request, "urlopen", lambda request, timeout: Response())

    s = RAGStore()
    s.ingest(_chunks("/docs/api.md", n=2))

    assert s._vectors.tolist() == [[1.0, 1.0], [2.0, 2.0]]


def test_dimension_mismatch_raises_clear_error(rag_store, monkeypatch):
    message = "Embedding dimension mismatch. The selected embedding model differs from the stored vectors. Clear and re-ingest the store."
    rag_store.ingest(_chunks("/docs/api.md", n=1))
    monkeypatch.setattr(
        rag_store, "_embed", lambda texts: np.zeros((len(texts), 3), dtype=np.float32)
    )

    with pytest.raises(ValueError, match=message):
        rag_store.ingest(_chunks("/docs/other.md", n=1))

    with pytest.raises(ValueError, match=message):
        rag_store.search("api")


# ── store unit tests: FAIL before fix ─────────────────────────────────────────


def test_source_mtime_returns_none_for_unknown(rag_store):
    """source_mtime() must exist and return None for an unknown source."""
    assert rag_store.source_mtime("/does/not/exist.md") is None


def test_source_mtime_recorded_after_ingest(rag_store):
    """mtime passed to ingest() must be retrievable via source_mtime()."""
    rag_store.ingest(_chunks("/docs/api.md"), mtime=1_000_000.0)
    assert rag_store.source_mtime("/docs/api.md") == 1_000_000.0


def test_source_mtime_not_set_when_mtime_omitted(rag_store):
    """ingest() without mtime must not record any mtime (backwards compat)."""
    rag_store.ingest(_chunks("/docs/api.md"))
    assert rag_store.source_mtime("/docs/api.md") is None


def test_delete_source_clears_mtime(rag_store):
    """delete_source() must remove the mtime entry for that source."""
    rag_store.ingest(_chunks("/docs/api.md"), mtime=1_000_000.0)
    rag_store.delete_source("/docs/api.md")
    assert rag_store.source_mtime("/docs/api.md") is None


def test_mtime_persists_across_reload(tmp_path, monkeypatch):
    """mtimes.json must survive a store reload (simulates server restart)."""
    monkeypatch.setattr(store_module, "STORE_DIR", tmp_path)
    s1 = RAGStore()
    monkeypatch.setattr(
        s1, "_embed", lambda texts: np.zeros((len(texts), 4), dtype=np.float32)
    )
    s1.ingest(_chunks("/docs/api.md"), mtime=42.0)

    s2 = RAGStore()  # fresh instance reads from same tmp_path
    assert s2.source_mtime("/docs/api.md") == 42.0


# ── integration test: FAIL before fix ─────────────────────────────────────────


def test_changed_file_triggers_reingest(tmp_path, monkeypatch):
    """When a file's mtime changes, _ingest_files_root must delete old chunks
    and re-ingest. Old code skipped already-ingested sources unconditionally."""
    import server as server_module
    from server import _ingest_files_root

    files_root = tmp_path / "files"
    files_root.mkdir()
    store_dir = tmp_path / "store"
    store_dir.mkdir()

    monkeypatch.setattr(store_module, "STORE_DIR", store_dir)
    monkeypatch.setattr(server_module, "FILES_ROOT", files_root)

    fresh = RAGStore()
    monkeypatch.setattr(
        fresh, "_embed", lambda texts: np.zeros((len(texts), 4), dtype=np.float32)
    )
    monkeypatch.setattr(server_module, "store", fresh)

    doc = files_root / "api.md"

    # First ingest: short doc → small chunk count
    doc.write_text("# API\n\n" + "First version content. " * 10)
    _ingest_files_root()
    count_v1 = len(fresh._chunks)
    assert count_v1 > 0

    # Modify file: advance mtime explicitly so the test is not time-dependent
    new_mtime = doc.stat().st_mtime + 1.0
    os.utime(doc, (new_mtime, new_mtime))
    doc.write_text(
        "# API\n\n"
        + "Updated version content. " * 10
        + "\n\n## New Section\n\n"
        + "Extra section content. " * 10
    )

    _ingest_files_root()

    all_bodies = " ".join(c["body"] for c in fresh._chunks)
    assert "Updated version content" in all_bodies, (
        "Re-ingested content not found in store"
    )
    assert "First version content" not in all_bodies, "Stale v1 content still in store"


def test_unchanged_file_is_skipped(tmp_path, monkeypatch):
    """File with unchanged mtime must produce 0 new chunks on second ingest."""
    import server as server_module
    from server import _ingest_files_root

    files_root = tmp_path / "files"
    files_root.mkdir()
    store_dir = tmp_path / "store"
    store_dir.mkdir()

    monkeypatch.setattr(store_module, "STORE_DIR", store_dir)
    monkeypatch.setattr(server_module, "FILES_ROOT", files_root)

    fresh = RAGStore()
    monkeypatch.setattr(
        fresh, "_embed", lambda texts: np.zeros((len(texts), 4), dtype=np.float32)
    )
    monkeypatch.setattr(server_module, "store", fresh)

    doc = files_root / "api.md"
    doc.write_text("# API\n\n" + "Content. " * 20)

    r1 = _ingest_files_root()
    assert r1.total_files == 1

    r2 = _ingest_files_root()  # same mtime → must skip
    assert r2.total_files == 0, f"Expected 0 files re-ingested, got {r2.total_files}"


def test_deleted_file_chunks_removed_on_ingest(tmp_path, monkeypatch):
    """Deleting a file from FILES_ROOT and calling ingest must remove its chunks."""
    import server as server_module
    from server import _ingest_files_root

    files_root = tmp_path / "files"
    files_root.mkdir()
    store_dir = tmp_path / "store"
    store_dir.mkdir()

    monkeypatch.setattr(store_module, "STORE_DIR", store_dir)
    monkeypatch.setattr(server_module, "FILES_ROOT", files_root)

    fresh = RAGStore()
    monkeypatch.setattr(
        fresh, "_embed", lambda texts: np.zeros((len(texts), 4), dtype=np.float32)
    )
    monkeypatch.setattr(server_module, "store", fresh)

    doc = files_root / "api.md"
    doc.write_text("# API\n\n" + "Content. " * 20)

    _ingest_files_root()
    assert len(fresh._chunks) > 0

    doc.unlink()
    result = _ingest_files_root()

    assert len(fresh._chunks) == 0
    assert str(doc) in result.removed_sources


# ── regression tests: PASS before AND after ───────────────────────────────────


def test_ingest_without_mtime_still_works(rag_store):
    """Callers that omit mtime must not get a TypeError."""
    assert rag_store.ingest(_chunks("/docs/api.md")) == 2


def test_delete_source_without_prior_mtime_does_not_raise(rag_store):
    """delete_source on a source ingested without mtime must not raise."""
    rag_store.ingest(_chunks("/docs/no-mtime.md"))
    assert rag_store.delete_source("/docs/no-mtime.md") == 2
