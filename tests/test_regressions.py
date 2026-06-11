"""Regression tests for the code-review fixes (store consistency, chunker
robustness, ingest isolation)."""

import json
from pathlib import Path

import numpy as np
import pytest

import store as store_module
from chunkers import chunk_docx, chunk_file, chunk_markdown, chunk_text
from store import RAGStore


@pytest.fixture
def rag_store(tmp_path, monkeypatch):
    monkeypatch.setattr(store_module, "STORE_DIR", tmp_path)
    s = RAGStore()
    monkeypatch.setattr(
        s, "_embed", lambda texts: np.zeros((len(texts), 4), dtype=np.float32)
    )
    return s


def _mk(source: str, n: int = 3, tag: str = "a") -> list[dict]:
    return [
        {
            "id": f"{tag}{i}",
            "source": source,
            "source_name": Path(source).name,
            "doc_title": "t",
            "chunk_type": "section",
            "section_path": "S",
            "chunk_index": i,
            "chunk_total": n,
            "page_start": None,
            "page_end": None,
            "title": f"T{i}",
            "body": f"{tag} body {i} with sufficient content to pass the minimum length filter",
        }
        for i in range(n)
    ]


# ── store: BM25 / bodies / vectors stay aligned ──────────────────────────────


def test_search_works_after_delete_source(rag_store):
    """BM25 must be rebuilt from the post-delete bodies, not stale disk state."""
    rag_store.ingest(_mk("/a.md", 3, "alpha"))
    rag_store.ingest(_mk("/b.md", 3, "beta"))
    rag_store.delete_source("/a.md")

    results = rag_store.search("beta body content", n=8)

    assert results, "search crashed or returned nothing after delete"
    assert all(r["source"] == "/b.md" for r in results)


def test_legacy_store_without_manifest_is_reset(tmp_path, monkeypatch):
    """A store without manifest.json (built before the manifest feature) must be
    reset on load so a fresh ingest rebuilds it with current chunker settings."""
    monkeypatch.setattr(store_module, "STORE_DIR", tmp_path)
    legacy = [
        {
            "id": "1",
            "source": "/docs/x.md",
            "doc_title": "x",
            "chunk_type": "section",
            "title": "T",
            "body": "legacy body content " * 5,
        }
    ]
    (tmp_path / "meta.json").write_text(json.dumps(legacy), encoding="utf-8")
    np.save(str(tmp_path / "vectors.npy"), np.zeros((1, 4), dtype=np.float32))

    s = RAGStore()

    assert s.stats()["total_chunks"] == 0, "legacy store must be cleared on first load"
    assert s.manifest_reset_reason is not None
    assert (tmp_path / "manifest.json").exists(), "fresh manifest must be written"


def test_missing_bodies_file_does_not_crash(tmp_path, monkeypatch):
    """meta.json without bodies.json must degrade safely (trim + clear mtimes
    so the next scan re-ingests), not crash every search."""
    monkeypatch.setattr(store_module, "STORE_DIR", tmp_path)
    s1 = RAGStore()
    monkeypatch.setattr(
        s1, "_embed", lambda texts: np.zeros((len(texts), 4), dtype=np.float32)
    )
    s1.ingest(_mk("/a.md"), mtime=1.0)
    (tmp_path / "bodies.json").unlink()

    s2 = RAGStore()
    monkeypatch.setattr(
        s2, "_embed", lambda texts: np.zeros((len(texts), 4), dtype=np.float32)
    )

    assert s2.search("anything at all") == []
    assert s2.source_mtime("/a.md") is None


def test_corrupt_meta_does_not_prevent_boot(tmp_path, monkeypatch):
    """A truncated meta.json (crash mid-write) must not make RAGStore() raise."""
    monkeypatch.setattr(store_module, "STORE_DIR", tmp_path)
    (tmp_path / "meta.json").write_text('[{"id": "x", "sou', encoding="utf-8")

    s = RAGStore()

    assert s.stats()["total_chunks"] == 0


def test_save_leaves_no_tmp_files(rag_store, tmp_path):
    rag_store.ingest(_mk("/a.md"), mtime=1.0)
    leftovers = list(tmp_path.glob("*.tmp")) + list(tmp_path.glob("*.tmp.npy"))
    assert leftovers == []


def test_search_non_positive_n_returns_empty(rag_store):
    rag_store.ingest(_mk("/a.md"))
    assert rag_store.search("body", n=-1) == []
    assert rag_store.search("body", n=0) == []


def test_adjacent_expansion_does_not_cross_duplicate_section_titles(
    rag_store, monkeypatch
):
    """Two same-titled sections used to collide in the (source, section_path,
    chunk_index) lookup, returning adjacent chunks from the wrong section."""
    chunks = []
    for tag in ("first", "second"):
        for i in range(2):
            chunks.append(
                {
                    "id": f"{tag}-{i}",
                    "source": "/docs/x.md",
                    "source_name": "x.md",
                    "doc_title": "x",
                    "chunk_type": "section",
                    "section_path": "Examples",
                    "chunk_index": i,
                    "chunk_total": 2,
                    "page_start": None,
                    "page_end": None,
                    "title": "Examples",
                    "body": f"{tag} section chunk {i} content padded {'pad ' * 15}",
                }
            )
    chunks[1]["body"] += " needle"  # hit lands on first-1

    def embed(texts):
        v = np.zeros((len(texts), 4), dtype=np.float32)
        for row, t in enumerate(texts):
            if "needle" in t:
                v[row, 0] = 1
        return v

    monkeypatch.setattr(rag_store, "_embed", embed)
    rag_store.ingest(chunks)

    results = rag_store.search("needle", n=2)

    assert [r["id"] for r in results] == ["first-1", "first-0"], (
        "adjacent expansion pulled a chunk from the duplicate-titled section"
    )


# ── chunkers: robustness and contiguous indices ──────────────────────────────


def test_chunk_file_yaml_edge_cases_no_crash(tmp_path):
    empty = tmp_path / "empty.yaml"
    empty.write_text("", encoding="utf-8")
    listroot = tmp_path / "list.yaml"
    listroot.write_text("- a\n- b\n", encoding="utf-8")
    no_servers = tmp_path / "spec.yaml"
    no_servers.write_text(
        "openapi: 3.0.0\n"
        "info:\n  title: X\n"
        "servers: []\n"
        "paths:\n  /a:\n    get:\n      summary: "
        + "long summary text " * 8
        + "\n",
        encoding="utf-8",
    )

    assert chunk_file(empty) == []
    assert isinstance(chunk_file(listroot), list)
    endpoint_chunks = chunk_file(no_servers)
    assert endpoint_chunks
    assert endpoint_chunks[0]["chunk_type"] == "endpoint"


def test_non_openapi_yaml_falls_back_to_text(tmp_path):
    f = tmp_path / "config.yaml"
    f.write_text(
        "services:\n  app:\n    image: nginx\n    description: "
        + "word " * 30
        + "\n",
        encoding="utf-8",
    )

    chunks = chunk_file(f)

    assert chunks, ".yaml without OpenAPI paths should be indexed as text"
    assert chunks[0]["chunk_type"] == "paragraph"


def test_markdown_heading_inside_code_fence_not_split(tmp_path):
    md = tmp_path / "doc.md"
    md.write_text(
        "# Setup\n\n"
        "Intro paragraph with enough length to survive the filter. "
        + "pad " * 10
        + "\n\n```bash\n# install dependencies first\napt-get install foo\n```\n\n"
        "After-fence paragraph also long enough to survive. " + "pad " * 10 + "\n",
        encoding="utf-8",
    )

    chunks = list(chunk_markdown(md))

    assert {c["section_path"] for c in chunks} == {"Setup"}
    joined = " ".join(c["body"] for c in chunks)
    assert "# install dependencies first" in joined


def test_short_paragraph_filtered_before_indexing(tmp_path):
    f = tmp_path / "notes.txt"
    f.write_text(
        "first long paragraph " * 8 + "\n\nshort\n\n" + "third long paragraph " * 8,
        encoding="utf-8",
    )

    chunks = list(chunk_text(f))

    assert [c["chunk_index"] for c in chunks] == [0, 1], (
        "min-length filter must run before chunk_index assignment"
    )
    assert {c["chunk_total"] for c in chunks} == {2}


def test_docx_table_content_is_indexed(tmp_path):
    docx = pytest.importorskip("docx")

    doc = docx.Document()
    doc.add_heading("Parameters", level=1)
    table = doc.add_table(rows=2, cols=2)
    table.cell(0, 0).text = "name"
    table.cell(0, 1).text = "description"
    table.cell(1, 0).text = "timeout_seconds"
    table.cell(1, 1).text = "How long to wait before aborting the request entirely"
    path = tmp_path / "spec.docx"
    doc.save(str(path))

    chunks = list(chunk_docx(path))

    joined = " ".join(c["body"] for c in chunks)
    assert "timeout_seconds" in joined, "table cell text missing from chunks"


# ── server: per-file isolation and orphan handling ───────────────────────────


def _server_env(tmp_path, monkeypatch):
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
    return server_module, fresh, files_root


def test_bad_file_does_not_abort_scan(tmp_path, monkeypatch):
    server_module, fresh, files_root = _server_env(tmp_path, monkeypatch)
    bad = files_root / "a-bad.md"
    bad.write_bytes(b"\xff\xfe invalid utf-8 \xff" * 20)
    good = files_root / "b-good.md"
    good.write_text("# Good\n\n" + "content " * 30, encoding="utf-8")

    result = server_module._ingest_files_root()

    assert result.total_files == 1, "good file after the bad one was not ingested"
    assert "a-bad.md" in result.failed_files


def test_orphan_chunks_not_duplicated_on_rescan(tmp_path, monkeypatch):
    """Chunks persisted without an mtime record (crash mid-ingest) must be
    replaced, not duplicated, by the next scan."""
    server_module, fresh, files_root = _server_env(tmp_path, monkeypatch)
    doc = files_root / "a.md"
    doc.write_text("# T\n\n" + "content " * 30, encoding="utf-8")

    server_module._ingest_files_root()
    count = len(fresh._chunks)
    assert count > 0

    # Simulate a crash that saved chunks but never recorded the mtime
    fresh._mtimes = {}
    fresh._save_mtimes()

    server_module._ingest_files_root()

    assert len(fresh._chunks) == count, "orphan chunks were duplicated"


# ── dashboard: URL encoding ──────────────────────────────────────────────────


def test_file_url_percent_encodes_special_chars(tmp_path):
    import dashboard

    src = str(tmp_path / "api spec#v2.md")

    url = dashboard.file_url(src, tmp_path, "http://h")

    assert url == "http://h/files/api%20spec%23v2.md"
