import numpy as np

import dashboard
import store as store_module
from store import RAGStore


def test_dashboard_data_groups_chunks_by_relative_source(tmp_path, monkeypatch):
    files_root = tmp_path / "files"
    files_root.mkdir()
    store_dir = tmp_path / "store"
    store_dir.mkdir()
    source_a = files_root / "docs" / "guide.md"
    source_a.parent.mkdir()
    source_a.write_text("# Guide\n")
    source_b = files_root / "api.md"
    source_b.write_text("# API\n")

    monkeypatch.setattr(store_module, "STORE_DIR", store_dir)

    fresh = RAGStore()
    monkeypatch.setattr(
        fresh, "_embed", lambda texts: np.zeros((len(texts), 4), dtype=np.float32)
    )
    fresh.ingest(
        [
            {
                "id": "chunk-2",
                "source": str(source_a),
                "doc_title": "guide",
                "chunk_type": "section",
                "title": "Second",
                "body": "Second body text " * 40,
            },
            {
                "id": "chunk-1",
                "source": str(source_a),
                "doc_title": "guide",
                "chunk_type": "section",
                "title": "First",
                "body": "First body text " * 40,
            },
            {
                "id": "chunk-3",
                "source": str(source_b),
                "doc_title": "api",
                "chunk_type": "section",
                "title": "API",
                "body": "API body text " * 40,
            },
        ]
    )
    payload = dashboard.dashboard_data(fresh, files_root, "http://testserver")

    assert payload["total_sources"] == 2
    assert payload["total_chunks"] == 3
    assert [f["path"] for f in payload["files"]] == ["api.md", "docs/guide.md"]
    assert payload["files"][1]["chunk_count"] == 2
    assert payload["files"][1]["source"] == str(source_a)
    assert payload["files"][1]["file_url"] == "http://testserver/files/docs/guide.md"
    assert payload["files"][1]["chunks"][0]["preview"].startswith("Second body")
    assert "body" not in payload["files"][1]["chunks"][0]


def test_dashboard_chunk_returns_full_body(tmp_path, monkeypatch):
    store_dir = tmp_path / "store"
    store_dir.mkdir()
    monkeypatch.setattr(store_module, "STORE_DIR", store_dir)

    fresh = RAGStore()
    monkeypatch.setattr(
        fresh, "_embed", lambda texts: np.zeros((len(texts), 4), dtype=np.float32)
    )
    body = "Full chunk body text. " * 30
    fresh.ingest(
        [
            {
                "id": "full-body-chunk",
                "source": str(tmp_path / "guide.md"),
                "doc_title": "guide",
                "chunk_type": "section",
                "title": "Guide",
                "body": body,
            }
        ]
    )
    assert dashboard.dashboard_chunk(fresh, tmp_path, "full-body-chunk")["body"] == body
    assert dashboard.dashboard_chunk(fresh, tmp_path, "missing") is None


def test_dashboard_html_fetches_dashboard_data_and_chunks():
    page = dashboard.dashboard_html("http://testserver")

    assert "fetch('/dashboard/data')" in page
    assert "/dashboard/chunk/" in page
    assert "id=\"filter\"" in page


def test_server_dashboard_wrappers_delegate_to_dashboard_module(tmp_path, monkeypatch):
    import server as server_module

    store_dir = tmp_path / "store"
    store_dir.mkdir()
    monkeypatch.setattr(store_module, "STORE_DIR", store_dir)
    monkeypatch.setattr(server_module, "FILES_ROOT", tmp_path)
    monkeypatch.setattr(server_module, "BASE_URL", "http://testserver")

    fresh = RAGStore()
    monkeypatch.setattr(
        fresh, "_embed", lambda texts: np.zeros((len(texts), 4), dtype=np.float32)
    )
    fresh.ingest(
        [
            {
                "id": "wrapped-chunk",
                "source": str(tmp_path / "guide.md"),
                "doc_title": "guide",
                "chunk_type": "section",
                "title": "Guide",
                "body": "Wrapped chunk body. " * 30,
            }
        ]
    )
    monkeypatch.setattr(server_module, "store", fresh)

    assert server_module.dashboard_data()["files"][0]["path"] == "guide.md"
    assert server_module.dashboard_chunk("wrapped-chunk")["title"] == "Guide"
    assert "http://testserver/dashboard" in server_module.dashboard_html()
