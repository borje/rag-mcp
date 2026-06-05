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
    s.ingest.return_value = 3
    s.delete_source.return_value = 2
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
    mock_store.delete_source.assert_called_once_with(missing)


def test_cleanup_keeps_existing_source(mock_store, tmp_path):
    existing = tmp_path / "present.md"
    existing.write_text("hello")
    mock_store.list_sources.return_value = [str(existing)]
    with patch.object(server, "store", mock_store):
        removed = _cleanup_stale_sources()
    assert removed == []
    mock_store.delete_source.assert_not_called()


def test_cleanup_mixed(mock_store, tmp_path):
    present = tmp_path / "present.md"
    present.write_text("hello")
    missing = str(tmp_path / "gone.md")
    mock_store.list_sources.return_value = [str(present), missing]
    with patch.object(server, "store", mock_store):
        removed = _cleanup_stale_sources()
    assert removed == [missing]
    mock_store.delete_source.assert_called_once_with(missing)
