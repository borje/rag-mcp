"""Tests for mtime-based staleness detection in RAGStore and _ingest_files_root.

Tests marked "FAILS before fix" use source_mtime() / mtime= param that did not
exist in the old code, or rely on re-ingest behaviour the old code lacked.
"""

import os
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


# ── regression tests: PASS before AND after ───────────────────────────────────


def test_ingest_without_mtime_still_works(rag_store):
    """Callers that omit mtime must not get a TypeError."""
    assert rag_store.ingest(_chunks("/docs/api.md")) == 2


def test_delete_source_without_prior_mtime_does_not_raise(rag_store):
    """delete_source on a source ingested without mtime must not raise."""
    rag_store.ingest(_chunks("/docs/no-mtime.md"))
    assert rag_store.delete_source("/docs/no-mtime.md") == 2
