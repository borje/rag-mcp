"""Benchmark tests for chunk_markdown quality.

Failing tests = baseline (before fix).
All green = chunking meets retrieval-quality bar.
"""

import pytest
from pathlib import Path
from chunkers import chunk_markdown, chunk_pdf

# Target: no chunk should exceed this many chars.
MAX_CHUNK_CHARS = 1200

# Overlap: adjacent sub-chunks of a split section must share at least this many chars.
MIN_OVERLAP_CHARS = 50

REQUIRED_METADATA = {
    "source_name",
    "section_path",
    "chunk_index",
    "chunk_total",
    "page_start",
    "page_end",
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
        assert c["page_start"] is None
        assert c["page_end"] is None


def test_markdown_split_chunks_have_section_indices(md_file: Path):
    chunks = chunks_list(md_file)
    auth_chunks = _chunks_containing(chunks, "auth-para")

    assert len(auth_chunks) > 1
    assert [c["chunk_index"] for c in auth_chunks] == list(range(len(auth_chunks)))
    assert {c["chunk_total"] for c in auth_chunks} == {len(auth_chunks)}
    assert {c["section_path"] for c in auth_chunks} == {"Authentication"}


def test_pdf_chunks_include_page_metadata(tmp_path: Path):
    fitz = pytest.importorskip("fitz")

    pdf = tmp_path / "guide.pdf"
    doc = fitz.open()
    for page_text in ["First page content " * 10, "Second page content " * 10]:
        page = doc.new_page()
        page.insert_text((72, 72), page_text)
    doc.save(pdf)
    doc.close()

    chunks = list(chunk_pdf(pdf))

    assert len(chunks) == 2
    assert [c["page_start"] for c in chunks] == [1, 2]
    assert [c["page_end"] for c in chunks] == [1, 2]
    assert [c["chunk_index"] for c in chunks] == [0, 1]
    assert {c["chunk_total"] for c in chunks} == {2}
    assert {c["section_path"] for c in chunks} == {"guide"}
