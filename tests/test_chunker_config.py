import importlib
from pathlib import Path

import pytest

import chunkers as chunkers_module


_CONFIG_VARS = ("MD_CHUNK_MAX_CHARS", "MD_CHUNK_OVERLAP_CHARS", "MIN_CHUNK_BODY")


@pytest.fixture(autouse=True)
def _restore_default_chunker_config(monkeypatch):
    yield
    for name in _CONFIG_VARS:
        monkeypatch.delenv(name, raising=False)
    importlib.reload(chunkers_module)


def _reload_chunkers(monkeypatch, **env):
    for name in _CONFIG_VARS:
        monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, str(value))
    return importlib.reload(chunkers_module)


def test_chunker_config_defaults(monkeypatch):
    chunkers = _reload_chunkers(monkeypatch)

    assert chunkers._MD_MAX_CHARS == 1000
    assert chunkers._MD_OVERLAP_CHARS == 150
    assert chunkers._MIN_CHUNK_BODY == 80


def test_md_chunk_max_chars_changes_split_behavior(tmp_path: Path, monkeypatch):
    md = tmp_path / "guide.md"
    md.write_text(
        "# Guide\n\n## Long Section\n\n"
        + "\n\n".join(
            f"Paragraph {i}: " + ("payment flow details " * 12) for i in range(8)
        )
    )

    default_chunkers = _reload_chunkers(monkeypatch)
    default_chunks = list(default_chunkers.chunk_markdown(md))

    tighter_chunkers = _reload_chunkers(
        monkeypatch, MD_CHUNK_MAX_CHARS=300, MD_CHUNK_OVERLAP_CHARS=50
    )
    tighter_chunks = list(tighter_chunkers.chunk_markdown(md))

    assert len(tighter_chunks) > len(default_chunks)


def test_min_chunk_body_filters_short_chunks(tmp_path: Path, monkeypatch):
    txt = tmp_path / "notes.txt"
    body = "A" * 120
    txt.write_text(body)

    default_chunkers = _reload_chunkers(monkeypatch)
    assert len(default_chunkers.chunk_file(txt)) == 1

    stricter_chunkers = _reload_chunkers(monkeypatch, MIN_CHUNK_BODY=200)
    assert stricter_chunkers.chunk_file(txt) == []


@pytest.mark.parametrize(
    ("env", "message"),
    [
        ({"MD_CHUNK_MAX_CHARS": "abc"}, "MD_CHUNK_MAX_CHARS must be an integer"),
        ({"MD_CHUNK_OVERLAP_CHARS": -1}, "MD_CHUNK_OVERLAP_CHARS must be >= 0"),
        (
            {"MD_CHUNK_MAX_CHARS": 100, "MD_CHUNK_OVERLAP_CHARS": 100},
            "MD_CHUNK_OVERLAP_CHARS must be smaller than MD_CHUNK_MAX_CHARS",
        ),
        ({"MIN_CHUNK_BODY": -1}, "MIN_CHUNK_BODY must be >= 0"),
    ],
)
def test_invalid_chunker_config_raises_clear_error(monkeypatch, env, message):
    with pytest.raises(ValueError, match=message):
        _reload_chunkers(monkeypatch, **env)
