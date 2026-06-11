"""Tests for mtime-based staleness detection in RAGStore and _ingest_files_root.

Tests marked "FAILS before fix" use source_mtime() / mtime= param that did not
exist in the old code, or rely on re-ingest behaviour the old code lacked.
"""

import json
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
            "source_name": Path(source).name,
            "doc_title": "test",
            "chunk_type": "section",
            "section_path": "test",
            "chunk_index": i,
            "chunk_total": n,
            "title": f"Section {i}",
            "body": f"[{tag}] Body text for chunk {i}. Enough content to exceed minimum length filter.",
        }
        for i in range(n)
    ]


def _search_chunks(source: str, section: str, n: int = 5) -> list[dict]:
    chunks = _chunks(source, "search", n)
    for i, chunk in enumerate(chunks):
        chunk["id"] = f"c{i}"
        chunk["section_path"] = section
        chunk["body"] = f"chunk-{i} ordinary content for adjacent expansion testing"
    chunks[2]["body"] += " needle"
    return chunks


def _scoped_chunks(source: str, relative_source: str, n: int = 3) -> list[dict]:
    chunks = _chunks(source, "scope", n)
    for i, chunk in enumerate(chunks):
        chunk["id"] = f"scope-{Path(relative_source).stem}-{i}"
        chunk["relative_source"] = relative_source
        chunk["section_path"] = "Scoped Section"
        chunk["title"] = "Scoped Section"
        chunk["body"] = f"chunk-{i} scoped content {'pad ' * 12}"
    return chunks


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


def test_store_resets_when_index_manifest_changes(tmp_path, monkeypatch):
    """Chunk config / model changes must invalidate persisted vectors and mtimes."""
    monkeypatch.setattr(store_module, "STORE_DIR", tmp_path)
    manifest_v1 = {
        "index_schema_version": 1,
        "chunker_version": 1,
        "model": "test-model",
        "chunk_config": {
            "MD_CHUNK_MAX_CHARS": 1000,
            "MD_CHUNK_OVERLAP_CHARS": 150,
            "MIN_CHUNK_BODY": 80,
        },
    }
    manifest_v2 = {
        **manifest_v1,
        "chunk_config": {**manifest_v1["chunk_config"], "MD_CHUNK_MAX_CHARS": 600},
    }

    monkeypatch.setattr(store_module, "current_index_manifest", lambda: manifest_v1)
    s1 = RAGStore()
    monkeypatch.setattr(
        s1, "_embed", lambda texts: np.zeros((len(texts), 4), dtype=np.float32)
    )
    s1.ingest(_chunks("/docs/api.md"), mtime=42.0)
    assert len(s1._chunks) == 2

    monkeypatch.setattr(store_module, "current_index_manifest", lambda: manifest_v2)
    s2 = RAGStore()

    assert s2._chunks == []
    assert s2.source_mtime("/docs/api.md") is None
    assert s2._vectors is None
    assert s2.manifest_reset_reason is not None
    assert json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8")) == manifest_v2


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

    all_bodies = " ".join(fresh._load_bodies())
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


def test_manifest_mismatch_forces_reingest_of_unchanged_file(tmp_path, monkeypatch):
    """A chunk-config change must clear mtimes so startup re-ingests unchanged files."""
    import server as server_module
    from server import _ingest_files_root

    files_root = tmp_path / "files"
    files_root.mkdir()
    store_dir = tmp_path / "store"
    store_dir.mkdir()
    doc = files_root / "api.md"
    doc.write_text("# API\n\n" + "Content. " * 20)

    manifest_v1 = {
        "index_schema_version": 1,
        "chunker_version": 1,
        "model": "test-model",
        "chunk_config": {
            "MD_CHUNK_MAX_CHARS": 1000,
            "MD_CHUNK_OVERLAP_CHARS": 150,
            "MIN_CHUNK_BODY": 80,
        },
    }
    manifest_v2 = {
        **manifest_v1,
        "chunk_config": {**manifest_v1["chunk_config"], "MD_CHUNK_MAX_CHARS": 600},
    }

    monkeypatch.setattr(store_module, "STORE_DIR", store_dir)
    monkeypatch.setattr(server_module, "FILES_ROOT", files_root)
    monkeypatch.setattr(store_module, "current_index_manifest", lambda: manifest_v1)

    first_store = RAGStore()
    monkeypatch.setattr(
        first_store, "_embed", lambda texts: np.zeros((len(texts), 4), dtype=np.float32)
    )
    monkeypatch.setattr(server_module, "store", first_store)

    first_result = _ingest_files_root()
    assert first_result.total_files == 1

    monkeypatch.setattr(store_module, "current_index_manifest", lambda: manifest_v2)
    second_store = RAGStore()
    monkeypatch.setattr(
        second_store, "_embed", lambda texts: np.zeros((len(texts), 4), dtype=np.float32)
    )
    monkeypatch.setattr(server_module, "store", second_store)

    second_result = _ingest_files_root()

    assert second_store.manifest_reset_reason is not None
    assert second_result.total_files == 1
    assert len(second_store._chunks) > 0


# ── regression tests: PASS before AND after ───────────────────────────────────


def test_ingest_without_mtime_still_works(rag_store):
    """Callers that omit mtime must not get a TypeError."""
    assert rag_store.ingest(_chunks("/docs/api.md")) == 2


def test_delete_source_without_prior_mtime_does_not_raise(rag_store):
    """delete_source on a source ingested without mtime must not raise."""
    rag_store.ingest(_chunks("/docs/no-mtime.md"))
    assert rag_store.delete_source("/docs/no-mtime.md") == 2


def test_search_interleaves_adjacent_chunks_and_honors_limit(rag_store, monkeypatch):
    chunks = _search_chunks("/docs/api.md", "Authentication", 5)

    def embed(texts):
        vectors = np.zeros((len(texts), 5), dtype=np.float32)
        for row, text in enumerate(texts):
            if text == "needle":
                vectors[row, 2] = 1
                continue
            for i in range(5):
                if f"chunk-{i}" in text:
                    vectors[row, i] = 1
                    break
        return vectors

    monkeypatch.setattr(rag_store, "_embed", embed)
    rag_store.ingest(chunks)

    results = rag_store.search("needle", n=3)

    assert [r["id"] for r in results] == ["c2", "c1", "c3"]
    assert [r["match_type"] for r in results] == ["hit", "adjacent", "adjacent"]


def test_search_adjacent_chunks_do_not_cross_section(rag_store, monkeypatch):
    chunks = _search_chunks("/docs/api.md", "Authentication", 5)
    chunks[3]["section_path"] = "Other Section"

    def embed(texts):
        vectors = np.zeros((len(texts), 5), dtype=np.float32)
        for row, text in enumerate(texts):
            if text == "needle":
                vectors[row, 2] = 1
            elif "chunk-2" in text:
                vectors[row, 2] = 1
        return vectors

    monkeypatch.setattr(rag_store, "_embed", embed)
    rag_store.ingest(chunks)

    results = rag_store.search("needle", n=2)

    assert [r["id"] for r in results] == ["c2", "c1"]


def test_search_scope_filters_nested_subtree(rag_store, monkeypatch):
    momentum = _scoped_chunks(
        "/data/library/trading/momentum/book-a/ch1.md",
        "library/trading/momentum/book-a/ch1.md",
    )
    risk = _scoped_chunks(
        "/data/library/trading/risk/book-b/ch1.md",
        "library/trading/risk/book-b/ch1.md",
    )
    momentum[1]["body"] += " needle"
    risk[1]["body"] += " needle"

    def embed(texts):
        vectors = np.zeros((len(texts), 3), dtype=np.float32)
        for row, text in enumerate(texts):
            if text == "needle":
                vectors[row, 1] = 1
            elif "needle" in text:
                vectors[row, 1] = 1
        return vectors

    monkeypatch.setattr(rag_store, "_embed", embed)
    rag_store.ingest(momentum + risk)

    results = rag_store.search("needle", n=4, scope="library/trading/momentum")

    assert results
    assert all(
        r["relative_source"].startswith("library/trading/momentum/") for r in results
    )


def test_search_scope_normalizes_slashes_and_respects_prefix_boundary(
    rag_store, monkeypatch
):
    wanted = _scoped_chunks(
        "/data/library/trading/momentum/book-a/ch1.md",
        "library/trading/momentum/book-a/ch1.md",
    )
    sibling = _scoped_chunks(
        "/data/library/trading/momentum-advanced/book-b/ch1.md",
        "library/trading/momentum-advanced/book-b/ch1.md",
    )
    wanted[1]["body"] += " needle"
    sibling[1]["body"] += " needle"

    def embed(texts):
        vectors = np.zeros((len(texts), 3), dtype=np.float32)
        for row, text in enumerate(texts):
            if text == "needle":
                vectors[row, 1] = 1
            elif "needle" in text:
                vectors[row, 1] = 1
        return vectors

    monkeypatch.setattr(rag_store, "_embed", embed)
    rag_store.ingest(wanted + sibling)

    results = rag_store.search("needle", n=4, scope="/library//trading/momentum/")

    assert results
    assert all("momentum-advanced" not in r["relative_source"] for r in results)


def test_scoped_search_adjacent_chunks_stay_inside_scope(rag_store, monkeypatch):
    in_scope = _scoped_chunks(
        "/data/library/trading/momentum/book-a/ch1.md",
        "library/trading/momentum/book-a/ch1.md",
    )
    out_of_scope = _scoped_chunks(
        "/data/library/trading/risk/book-b/ch1.md",
        "library/trading/risk/book-b/ch1.md",
    )
    in_scope[1]["body"] += " needle"

    def embed(texts):
        vectors = np.zeros((len(texts), 3), dtype=np.float32)
        for row, text in enumerate(texts):
            if text == "needle":
                vectors[row, 1] = 1
                continue
            for i in range(3):
                if f"chunk-{i}" in text:
                    vectors[row, i] = 1
                    break
        return vectors

    monkeypatch.setattr(rag_store, "_embed", embed)
    rag_store.ingest(in_scope + out_of_scope)

    results = rag_store.search("needle", n=3, scope="library/trading/momentum")

    assert [r["relative_source"] for r in results] == [
        "library/trading/momentum/book-a/ch1.md",
        "library/trading/momentum/book-a/ch1.md",
        "library/trading/momentum/book-a/ch1.md",
    ]


def test_ingest_records_relative_source_for_nested_files(tmp_path, monkeypatch):
    import server as server_module

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

    doc = files_root / "library" / "trading" / "momentum" / "book-a.md"
    doc.parent.mkdir(parents=True)
    doc.write_text("# Book\n\n" + "Momentum content. " * 20, encoding="utf-8")

    server_module._ingest_files_root()

    assert fresh._chunks
    assert {c["relative_source"] for c in fresh._chunks} == {
        "library/trading/momentum/book-a.md"
    }


# ── list_scopes ────────────────────────────────────────────────────────────────


def test_list_scopes_returns_all_ancestor_scopes(rag_store):
    chunks = _scoped_chunks(
        "/data/library/trading/momentum/book-a/ch1.md",
        "library/trading/momentum/book-a/ch1.md",
    )
    rag_store.ingest(chunks)

    scopes = {s["scope"] for s in rag_store.list_scopes()}

    assert scopes == {
        "library",
        "library/trading",
        "library/trading/momentum",
        "library/trading/momentum/book-a",
    }


def test_list_scopes_dedupes_and_counts_shared_ancestors(rag_store):
    momentum = _scoped_chunks(
        "/data/library/trading/momentum/book-a/ch1.md",
        "library/trading/momentum/book-a/ch1.md",
    )
    risk = _scoped_chunks(
        "/data/library/trading/risk/book-b/ch1.md",
        "library/trading/risk/book-b/ch1.md",
    )
    rag_store.ingest(momentum + risk)

    by_scope = {s["scope"]: s["n_docs"] for s in rag_store.list_scopes()}

    assert by_scope["library"] == 2
    assert by_scope["library/trading"] == 2
    assert by_scope["library/trading/momentum"] == 1
    assert by_scope["library/trading/momentum/book-a"] == 1
    assert by_scope["library/trading/risk"] == 1
    assert by_scope["library/trading/risk/book-b"] == 1


def test_list_scopes_ignores_root_level_files(rag_store):
    chunks = _scoped_chunks("/data/readme.md", "readme.md")
    rag_store.ingest(chunks)

    assert rag_store.list_scopes() == []


def test_list_scopes_skips_chunks_without_relative_source(rag_store):
    chunks = _chunks("/data/legacy.md")  # no relative_source set
    rag_store.ingest(chunks)

    assert rag_store.list_scopes() == []


def test_list_scopes_normalizes_slashes_and_sorts(rag_store):
    chunks = _scoped_chunks(
        "/data/library/trading/momentum/book-a/ch1.md",
        "/library//trading/momentum/book-a/ch1.md",
    )
    rag_store.ingest(chunks)

    scopes = [s["scope"] for s in rag_store.list_scopes()]

    assert scopes == sorted(scopes)
    assert "library" in scopes
    assert "library/trading" in scopes
