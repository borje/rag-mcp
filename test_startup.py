"""Tests for startup-auto-ingest branch features.

These tests cover:
  - _ingest_files_root: skips already-ingested, skips unsupported, ingests new
  - ingest: scans FILES_ROOT without client-supplied paths
  - _cleanup_stale_sources: removes missing paths, keeps existing paths

These helpers are module-level so startup and MCP-triggered ingestion share the
same behavior.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import server
from server import _cleanup_stale_sources, _ingest_files_root, ingest


@pytest.fixture()
def mock_store():
    s = MagicMock()
    s.list_sources.return_value = []
    s.source_mtime.return_value = None
    s.ingest.return_value = 3
    s.source_fingerprint.return_value = None
    s.delete_source.return_value = 2
    s.delete_sources.return_value = {}
    s.known_sources.side_effect = lambda: set(s.list_sources.return_value)
    s.stats.return_value = {
        "total_chunks": 0,
        "total_sources": 0,
        "model": "m",
        "store_dir": "/tmp/x",
    }
    return s


# ---------------------------------------------------------------------------
# _ingest_files_root
# ---------------------------------------------------------------------------


def test_ingest_files_root_ingests_new_markdown(mock_store, tmp_path):
    (tmp_path / "doc.md").write_text("# Title\n\n" + "word " * 30)
    with (
        patch.object(server, "store", mock_store),
        patch.object(server, "FILES_ROOT", tmp_path),
    ):
        result = _ingest_files_root()
    assert result.total_files == 1
    mock_store.ingest.assert_called_once()


def test_ingest_files_root_skips_already_ingested(mock_store, tmp_path):
    f = tmp_path / "doc.md"
    f.write_text("# Title\n\n" + "word " * 30)
    mock_store.list_sources.return_value = [str(f)]
    st = f.stat()
    mock_store.source_fingerprint.return_value = {
        "mtime": st.st_mtime,
        "size": st.st_size,
    }
    with (
        patch.object(server, "store", mock_store),
        patch.object(server, "FILES_ROOT", tmp_path),
    ):
        result = _ingest_files_root()
    assert result.total_files == 0
    mock_store.ingest.assert_not_called()


def test_ingest_files_root_skips_unsupported_extensions(mock_store, tmp_path):
    (tmp_path / "binary.exe").write_bytes(b"\x00\x01\x02")
    (tmp_path / "archive.zip").write_bytes(b"PK\x03\x04")
    with (
        patch.object(server, "store", mock_store),
        patch.object(server, "FILES_ROOT", tmp_path),
    ):
        result = _ingest_files_root()
    assert result.total_files == 0
    mock_store.ingest.assert_not_called()


def test_ingest_files_root_empty_dir(mock_store, tmp_path):
    with (
        patch.object(server, "store", mock_store),
        patch.object(server, "FILES_ROOT", tmp_path),
    ):
        result = _ingest_files_root()
    assert result.total_files == 0
    assert result.total_chunks == 0


def test_ingest_files_root_skipped_files_when_no_chunks(mock_store, tmp_path):
    (tmp_path / "empty.txt").write_text("")
    mock_store.list_sources.return_value = []
    with (
        patch.object(server, "store", mock_store),
        patch.object(server, "FILES_ROOT", tmp_path),
    ):
        result = _ingest_files_root()
    assert result.total_files == 0
    assert result.skipped_files == ["empty.txt"]
    mock_store.ingest.assert_not_called()


def test_ingest_uses_files_root(mock_store, tmp_path):
    (tmp_path / "doc.md").write_text("# Title\n\n" + "word " * 30)
    with (
        patch.object(server, "store", mock_store),
        patch.object(server, "FILES_ROOT", tmp_path),
    ):
        result = ingest()
    assert result.total_files == 1
    mock_store.ingest.assert_called_once()


# ---------------------------------------------------------------------------
# _cleanup_stale_sources
# ---------------------------------------------------------------------------


def test_cleanup_removes_missing_source(mock_store, tmp_path):
    missing = str(tmp_path / "gone.md")
    mock_store.list_sources.return_value = [missing]
    with patch.object(server, "store", mock_store):
        removed = _cleanup_stale_sources()
    assert removed == [missing]
    mock_store.delete_sources.assert_called_once_with([missing], rebuild_bm25=False)


def test_cleanup_keeps_existing_source(mock_store, tmp_path):
    existing = tmp_path / "present.md"
    existing.write_text("hello")
    mock_store.list_sources.return_value = [str(existing)]
    with patch.object(server, "store", mock_store):
        removed = _cleanup_stale_sources()
    assert removed == []
    mock_store.delete_sources.assert_not_called()


def test_cleanup_mixed(mock_store, tmp_path):
    present = tmp_path / "present.md"
    present.write_text("hello")
    missing = str(tmp_path / "gone.md")
    mock_store.list_sources.return_value = [str(present), missing]
    with patch.object(server, "store", mock_store):
        removed = _cleanup_stale_sources()
    assert removed == [missing]
    mock_store.delete_sources.assert_called_once_with([missing], rebuild_bm25=False)


def test_cleanup_removes_existing_unsupported_suffix(mock_store, tmp_path):
    pdf = tmp_path / "old.pdf"
    pdf.write_bytes(b"%PDF")
    md = tmp_path / "keep.md"
    md.write_text("hello")
    mock_store.list_sources.return_value = [str(pdf), str(md)]
    with (
        patch.object(server, "store", mock_store),
        patch.object(server, "SUPPORTED_EXTENSIONS", {".md"}),
    ):
        removed = _cleanup_stale_sources()
    assert removed == [str(pdf)]
    mock_store.delete_sources.assert_called_once_with([str(pdf)], rebuild_bm25=False)


# ---------------------------------------------------------------------------
# ignored files, status, reindex
# ---------------------------------------------------------------------------


def test_ignored_files_lists_unsupported_pdf(mock_store, tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "paper.pdf").write_bytes(b"%PDF")
    (tmp_path / "doc.md").write_text("# Title\n\n" + "word " * 30)
    with (
        patch.object(server, "store", mock_store),
        patch.object(server, "FILES_ROOT", tmp_path),
        patch.object(server, "SUPPORTED_EXTENSIONS", {".md"}),
    ):
        result = _ingest_files_root()
    assert result.ignored_files == ["sub/paper.pdf"]


def test_ignored_files_capped(mock_store, tmp_path):
    for i in range(205):
        (tmp_path / f"f{i:03}.bin").write_bytes(b"x")
    with (
        patch.object(server, "store", mock_store),
        patch.object(server, "FILES_ROOT", tmp_path),
    ):
        result = _ingest_files_root()
    assert len(result.ignored_files) == 201
    assert result.ignored_files[-1] == "... and 5 more"


def test_scan_defers_bm25_rebuild(mock_store, tmp_path):
    (tmp_path / "a.md").write_text("# A\n\n" + "word " * 30)
    (tmp_path / "b.md").write_text("# B\n\n" + "word " * 30)
    with (
        patch.object(server, "store", mock_store),
        patch.object(server, "FILES_ROOT", tmp_path),
    ):
        _ingest_files_root()
    for call in mock_store.ingest.call_args_list:
        assert call.kwargs["rebuild_bm25"] is False
    mock_store.rebuild_bm25.assert_called_once()


def test_rag_status_has_last_scan_after_scan(mock_store, tmp_path):
    (tmp_path / "doc.md").write_text("# Title\n\n" + "word " * 30)
    (tmp_path / "x.bin").write_bytes(b"x")
    with (
        patch.object(server, "store", mock_store),
        patch.object(server, "FILES_ROOT", tmp_path),
    ):
        _ingest_files_root()
        status = server.rag_status()
    assert status.last_scan["files"] == 1
    assert status.last_scan["ignored"] == 1
    assert status.last_scan["at"].endswith("+00:00")
    assert status.watch_interval == server.WATCH_INTERVAL
    assert ".md" in status.supported_extensions


def test_reindex_outside_files_root_raises(mock_store, tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    with (
        patch.object(server, "store", mock_store),
        patch.object(server, "FILES_ROOT", root),
    ):
        with pytest.raises(ValueError):
            server.reindex("../outside.md")
        with pytest.raises(ValueError):
            server.reindex(str(tmp_path / "outside.md"))
        with pytest.raises(ValueError):
            server.reindex(".")
        with pytest.raises(FileNotFoundError):
            server.reindex("missing.md")
    mock_store.touch_source.assert_not_called()
    mock_store.ingest.assert_not_called()


def test_reindex_valid_file_reingests(mock_store, tmp_path):
    f = tmp_path / "doc.md"
    f.write_text("# Title\n\n" + "word " * 30)
    st = f.stat()
    # Store believes the file is unchanged, so only reindex forces re-embedding.
    mock_store.list_sources.return_value = [str(f)]
    mock_store.source_fingerprint.side_effect = lambda s: (
        None
        if mock_store.touch_source.called
        else {"mtime": st.st_mtime, "size": st.st_size}
    )
    with (
        patch.object(server, "store", mock_store),
        patch.object(server, "FILES_ROOT", tmp_path),
    ):
        result = server.reindex("doc.md")
    mock_store.touch_source.assert_any_call(str(f), {})
    mock_store.delete_sources.assert_not_called()
    assert result.total_files == 1
    mock_store.ingest.assert_called_once()


def test_reindex_directory_clears_only_sources_under_it(mock_store, tmp_path):
    (tmp_path / "sub").mkdir()
    inside = [str(tmp_path / "sub" / "a.md"), str(tmp_path / "sub" / "deep" / "b.md")]
    outside = str(tmp_path / "subway.md")  # shares the prefix string, not the directory
    mock_store.list_sources.return_value = inside + [outside]
    with (
        patch.object(server, "store", mock_store),
        patch.object(server, "FILES_ROOT", tmp_path),
    ):
        server.reindex("sub")
    assert sorted(c.args[0] for c in mock_store.touch_source.call_args_list) == sorted(inside)


def test_reindex_symlink_targets_the_link_not_its_target(mock_store, tmp_path):
    real = tmp_path / "real.md"
    real.write_text("# T\n\n" + "word " * 30)
    link = tmp_path / "link.md"
    link.symlink_to(real)
    mock_store.list_sources.return_value = [str(real), str(link)]
    with (
        patch.object(server, "store", mock_store),
        patch.object(server, "FILES_ROOT", tmp_path),
    ):
        server.reindex("link.md")
    assert [c.args[0] for c in mock_store.touch_source.call_args_list if c.args[1] == {}] == [str(link)]


def test_failed_file_is_fingerprinted_and_not_retried(mock_store, tmp_path):
    bad = tmp_path / "bad.md"
    bad.write_text("# T\n\n" + "word " * 30)
    with (
        patch.object(server, "store", mock_store),
        patch.object(server, "FILES_ROOT", tmp_path),
        patch.object(server, "chunk_file", side_effect=RuntimeError("boom")),
    ):
        result = _ingest_files_root()
    assert result.failed_files == ["bad.md"]
    (source, fp), _ = mock_store.touch_source.call_args
    assert source == str(bad) and "sha256" in fp
