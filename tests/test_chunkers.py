"""Benchmark tests for chunk_markdown quality.

Failing tests = baseline (before fix).
All green = chunking meets retrieval-quality bar.
"""

import pytest
from pathlib import Path
from chunkers import chunk_markdown

# Target: no chunk should exceed this many chars.
MAX_CHUNK_CHARS = 1200

# Overlap: adjacent sub-chunks of a split section must share at least this many chars.
MIN_OVERLAP_CHARS = 50

REQUIRED_METADATA = {
    "source_name",
    "section_path",
    "chunk_index",
    "chunk_total",
}

# ----- fixtures ----------------------------------------------------------------

_BASE_PARA = (
    "Each entry includes a unique identifier, timestamp, and status code. "
    "The response payload is JSON. Pagination uses cursor-based tokens. "
    "Retry logic should use exponential backoff with jitter."
)  # ~200 chars


def _make_long_section(tag: str, n_paras: int = 12) -> str:
    """Generates a long section; `tag` is unique per section for targeted lookup."""
    paras = [
        f"{tag}-para-{i}: {_BASE_PARA} "
        f"Sequence index {i} applies here. Max value is {i * 100}."
        for i in range(n_paras)
    ]
    return "\n\n".join(paras)


SAMPLE_MD = f"""\
# Payment API

## Overview

This API provides payment processing for merchants.
Use the endpoints below to create and manage transactions.

## Authentication

{_make_long_section("auth", 12)}

## Create Payment

{_make_long_section("payment", 10)}

## Short Section

Brief content only.
"""


@pytest.fixture
def md_file(tmp_path: Path) -> Path:
    f = tmp_path / "api_docs.md"
    f.write_text(SAMPLE_MD)
    return f


# ----- helpers ----------------------------------------------------------------


def chunks_list(md_file: Path) -> list[dict]:
    return list(chunk_markdown(md_file))


def _chunks_containing(chunks: list[dict], keyword: str) -> list[dict]:
    return [c for c in chunks if keyword.lower() in c["body"].lower()]


# ----- tests that FAIL before fix, PASS after ---------------------------------


def test_no_chunk_exceeds_max_chars(md_file: Path):
    """Every chunk body must be short enough for embedding to carry signal."""
    chunks = chunks_list(md_file)
    oversized = [c for c in chunks if len(c["body"]) > MAX_CHUNK_CHARS]
    sizes = [len(c["body"]) for c in oversized]
    assert oversized == [], (
        f"{len(oversized)} chunk(s) exceed {MAX_CHUNK_CHARS} chars; sizes={sizes}"
    )


def test_large_section_split_into_multiple_chunks(md_file: Path):
    """A 2000+ char section must produce more than one chunk."""
    chunks = chunks_list(md_file)
    # "auth-para" is unique to the Authentication section
    auth_chunks = _chunks_containing(chunks, "auth-para")
    assert len(auth_chunks) > 1, (
        f"Authentication section only produced {len(auth_chunks)} chunk(s); "
        "expected multiple due to length"
    )


def test_adjacent_chunks_overlap(md_file: Path):
    """When a section is split, consecutive chunks share text (overlap window)."""
    chunks = chunks_list(md_file)
    auth_chunks = _chunks_containing(chunks, "auth-para")
    if len(auth_chunks) < 2:
        pytest.skip(
            "section not split — covered by test_large_section_split_into_multiple_chunks"
        )

    found_overlap = False
    for a, b in zip(auth_chunks, auth_chunks[1:]):
        lines_a = {l.strip() for l in a["body"].splitlines() if l.strip()}
        lines_b = {l.strip() for l in b["body"].splitlines() if l.strip()}
        shared_chars = sum(len(l) for l in lines_a & lines_b)
        if shared_chars >= MIN_OVERLAP_CHARS:
            found_overlap = True
            break

    assert found_overlap, "Adjacent split-chunks share no overlapping content"


# ----- regression tests: PASS before AND after --------------------------------


def test_content_not_lost(md_file: Path):
    """Every paragraph in the source appears in at least one chunk."""
    chunks = chunks_list(md_file)
    all_bodies = " ".join(c["body"] for c in chunks)
    for i in range(12):
        assert f"auth-para-{i}:" in all_bodies, f"auth-para-{i} missing from all chunks"


def test_short_section_is_single_chunk(md_file: Path):
    """Sections already under the limit must not be artificially split."""
    chunks = chunks_list(md_file)
    overview_chunks = _chunks_containing(chunks, "Overview")
    assert len(overview_chunks) == 1, (
        f"'Overview' section produced {len(overview_chunks)} chunks; expected 1"
    )


def test_doc_title_is_filename_stem(md_file: Path):
    chunks = chunks_list(md_file)
    for c in chunks:
        assert c["doc_title"] == "api_docs", f"doc_title wrong: {c['doc_title']}"


def test_chunk_type_is_section(md_file: Path):
    chunks = chunks_list(md_file)
    for c in chunks:
        assert c["chunk_type"] == "section"


def test_markdown_chunks_include_required_metadata(md_file: Path):
    chunks = chunks_list(md_file)
    for c in chunks:
        assert REQUIRED_METADATA <= c.keys()
        assert c["source_name"] == "api_docs.md"


def test_markdown_split_chunks_have_section_indices(md_file: Path):
    chunks = chunks_list(md_file)
    auth_chunks = _chunks_containing(chunks, "auth-para")

    assert len(auth_chunks) > 1
    assert [c["chunk_index"] for c in auth_chunks] == list(range(len(auth_chunks)))
    assert {c["chunk_total"] for c in auth_chunks} == {len(auth_chunks)}
    assert {c["section_path"] for c in auth_chunks} == {"Authentication"}


# ----- convert-to-markdown hook ------------------------------------------------

import importlib

import chunkers as _chunkers_module

_CONVERT_VARS = ("RAG_MCP_CONVERT_CMD", "RAG_MCP_CONVERT_EXTS", "RAG_MCP_CONVERT_TIMEOUT")


@pytest.fixture
def reload_chunkers(monkeypatch):
    def _reload(**env):
        for name in _CONVERT_VARS:
            monkeypatch.delenv(name, raising=False)
        for name, value in env.items():
            monkeypatch.setenv(name, value)
        return importlib.reload(_chunkers_module)

    yield _reload
    for name in _CONVERT_VARS:
        monkeypatch.delenv(name, raising=False)
    importlib.reload(_chunkers_module)


def test_convert_off_by_default(reload_chunkers):
    c = reload_chunkers()
    assert ".pdf" not in c.SUPPORTED_EXTENSIONS
    assert "RAG_MCP_CONVERT_CMD" not in c.current_chunk_config()


def test_convert_cmd_registers_exts_and_chunks(reload_chunkers, tmp_path: Path):
    body = "Converted body text " * 6  # >= 80 chars
    c = reload_chunkers(
        # %.0s consumes the path argument without printing it
        RAG_MCP_CONVERT_CMD=f"printf '# Title\\n\\n{body}%.0s' {{input}}",
        RAG_MCP_CONVERT_EXTS="pdf, .DOCX",
    )
    assert {".pdf", ".docx"} <= c.SUPPORTED_EXTENSIONS
    assert c._CONVERT_EXTS == [".pdf", ".docx"]
    assert "RAG_MCP_CONVERT_EXTS" not in c.current_chunk_config()
    pdf = tmp_path / "my doc.pdf"
    pdf.write_bytes(b"%PDF-fake")
    chunks = c.chunk_file(pdf)
    assert len(chunks) == 1
    assert chunks[0]["source"] == str(pdf)
    assert chunks[0]["chunk_type"] == "section"
    assert chunks[0]["doc_title"] == "my doc"
    assert chunks[0]["title"] == "Title"


@pytest.mark.parametrize("template", ["cat {input}", 'cat "{input}"', "cat '{input}'"])
def test_convert_hostile_filename_never_reaches_a_shell(
    reload_chunkers, tmp_path: Path, template: str
):
    c = reload_chunkers(RAG_MCP_CONVERT_CMD=template)
    pdf = tmp_path / "it's $(touch pwned) `id` doc.pdf"
    pdf.write_text("# Heading\n\n" + "quoted path content " * 6)
    chunks = c.chunk_file(pdf)
    assert chunks and "quoted path content" in chunks[0]["body"]
    assert not (Path.cwd() / "pwned").exists() and not (tmp_path / "pwned").exists()


def test_convert_does_not_inherit_stdin(reload_chunkers, tmp_path: Path):
    # A converter that reads stdin must see EOF, not the MCP stdio pipe.
    c = reload_chunkers(RAG_MCP_CONVERT_CMD="cat", RAG_MCP_CONVERT_TIMEOUT="5")
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"x")
    assert c.chunk_file(pdf) == []


def test_convert_failure_raises(reload_chunkers, tmp_path: Path):
    # Shell features still work when the operator opts in explicitly.
    c = reload_chunkers(RAG_MCP_CONVERT_CMD="sh -c 'echo boom >&2; exit 1' x {input}")
    pdf = tmp_path / "bad.pdf"
    pdf.write_bytes(b"x")
    with pytest.raises(RuntimeError, match="convert failed for bad.pdf: boom"):
        c.chunk_file(pdf)


def test_convert_exts_do_not_override_openapi_sniffing(reload_chunkers):
    c = reload_chunkers(
        RAG_MCP_CONVERT_CMD="cat {input}", RAG_MCP_CONVERT_EXTS=".pdf,.json,.yaml"
    )
    assert c._EXT_MAP[".yaml"] is c.chunk_openapi
    assert ".json" not in c._EXT_MAP


def test_chunker_version_bumped():
    assert _chunkers_module.CHUNKER_VERSION == 2
