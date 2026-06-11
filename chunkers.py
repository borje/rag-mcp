"""Document chunkers: OpenAPI, Markdown, PDF, DOCX, plain text."""

import json
import os
import re
import uuid
from pathlib import Path
from typing import Iterator

# Chunks with a body shorter than this are dropped by the chunkers BEFORE
# chunk_index/chunk_total are assigned, so indices stay contiguous and
# adjacent-chunk expansion in the store keeps working.
_MIN_CHUNK_BODY = 80


def _id() -> str:
    return str(uuid.uuid4())


def _meta(
    path: Path,
    section_path: str,
    chunk_index: int,
    chunk_total: int,
    page_start: int | None = None,
    page_end: int | None = None,
) -> dict:
    return {
        "source_name": path.name,
        "section_path": section_path,
        "chunk_index": chunk_index,
        "chunk_total": chunk_total,
        "page_start": page_start,
        "page_end": page_end,
    }


# ── OpenAPI (OAS2 / OAS3) ────────────────────────────────────────────────────


def chunk_openapi(path: Path) -> Iterator[dict]:
    import yaml

    text = path.read_text(encoding="utf-8")
    spec = (
        yaml.safe_load(text) if path.suffix in (".yaml", ".yml") else json.loads(text)
    )
    if not isinstance(spec, dict):
        return

    info = spec.get("info") or {}
    title = info.get("title", path.stem) if isinstance(info, dict) else path.stem
    servers = spec.get("servers") or []
    first_server = (
        servers[0]
        if isinstance(servers, list) and servers and isinstance(servers[0], dict)
        else {}
    )
    base = first_server.get("url", "") or spec.get("host", "")

    operations = []
    for path_str, methods in (spec.get("paths") or {}).items():
        if not isinstance(methods, dict):
            continue
        for method, op in methods.items():
            if method.startswith("x-") or not isinstance(op, dict):
                continue
            operations.append((path_str, method, op))

    entries: list[tuple[str, str]] = []
    for path_str, method, op in operations:
        summary = op.get("summary") or op.get("operationId") or ""
        description = op.get("description") or ""
        params = op.get("parameters") or []
        req_body = op.get("requestBody") or {}
        responses = op.get("responses") or {}
        tags = ", ".join(op.get("tags") or [])

        lines = [f"{method.upper()} {path_str}"]
        if base:
            lines.append(f"Base: {base}")
        if tags:
            lines.append(f"Tags: {tags}")
        if summary:
            lines.append(f"Summary: {summary}")
        if description:
            lines.append(f"Description: {description.strip()[:600]}")
        if params:
            plines = []
            for p in params[:20]:
                loc = p.get("in", "")
                name = p.get("name", "")
                req = " (required)" if p.get("required") else ""
                desc = p.get("description") or ""
                schema = p.get("schema") or {}
                typ = schema.get("type") or p.get("type") or ""
                plines.append(f"  [{loc}] {name}{req} {typ}: {desc}")
            lines.append("Parameters:\n" + "\n".join(plines))
        if req_body:
            content_types = ", ".join((req_body.get("content") or {}).keys())
            rb_desc = req_body.get("description") or ""
            lines.append(f"Request body ({content_types}): {rb_desc}")
        if responses:
            rlines = [
                f"  {code}: {(v or {}).get('description', '')}"
                for code, v in list(responses.items())[:8]
            ]
            lines.append("Responses:\n" + "\n".join(rlines))

        entries.append((f"{method.upper()} {path_str}", "\n".join(lines)))

    entries = [(t, b) for t, b in entries if len(b) >= _MIN_CHUNK_BODY]
    total = len(entries)
    for idx, (op_title, body) in enumerate(entries):
        yield {
            "id": _id(),
            "source": str(path),
            **_meta(path, op_title, idx, total),
            "doc_title": title,
            "chunk_type": "endpoint",
            "title": op_title,
            "body": body,
        }


# ── Markdown ─────────────────────────────────────────────────────────────────


def _env_int(name: str, default: int, *, minimum: int | None = None) -> int:
    raw = os.environ.get(name)
    if raw in {None, ""}:
        value = default
    else:
        try:
            value = int(raw)
        except ValueError as exc:
            raise ValueError(f"{name} must be an integer, got {raw!r}") from exc
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")
    return value


CHUNKER_VERSION = 1

_MD_MAX_CHARS = _env_int("MD_CHUNK_MAX_CHARS", 1000, minimum=1)
_MD_OVERLAP_CHARS = _env_int("MD_CHUNK_OVERLAP_CHARS", 150, minimum=0)
if _MD_OVERLAP_CHARS >= _MD_MAX_CHARS:
    raise ValueError(
        "MD_CHUNK_OVERLAP_CHARS must be smaller than MD_CHUNK_MAX_CHARS, "
        f"got overlap={_MD_OVERLAP_CHARS}, max={_MD_MAX_CHARS}"
    )


def _split_long_section(header: str, body: str) -> list[str]:
    """Split body into overlapping paragraph-aware chunks prefixed with header."""
    prefix = (header + "\n\n") if header else ""
    effective_max = _MD_MAX_CHARS - len(prefix)
    paras = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
    if not paras:
        return [prefix.rstrip()]

    result: list[str] = []
    start = 0
    while start < len(paras):
        window: list[str] = []
        length = 0
        i = start
        while i < len(paras):
            add = len(paras[i]) + (2 if window else 0)
            if length + add > effective_max and window:
                break
            window.append(paras[i])
            length += add
            i += 1
        if not window:  # single para exceeds limit — include it anyway
            window = [paras[start]]
            i = start + 1
        result.append(prefix + "\n\n".join(window))
        if i >= len(paras):
            break
        # step back to include at least _MD_OVERLAP_CHARS worth of paragraphs
        back, back_len = 0, 0
        for p in reversed(window):
            back_len += len(p)
            back += 1
            if back_len >= _MD_OVERLAP_CHARS:
                break
        start = max(start + 1, i - back)
    return result or [prefix + "\n\n".join(paras)]


def _split_md_sections(text: str) -> list[str]:
    """Split on #/##/### headings at line start, ignoring fenced code blocks.

    A naive regex split treats '# comment' lines inside ``` fences as
    section boundaries, producing bogus titles and broken adjacency keys.
    """
    sections: list[list[str]] = [[]]
    in_fence = False
    for line in text.splitlines(keepends=True):
        stripped = line.lstrip()
        if stripped.startswith("```") or stripped.startswith("~~~"):
            in_fence = not in_fence
            sections[-1].append(line)
            continue
        if not in_fence and re.match(r"#{1,3} ", line):
            sections.append([line])
        else:
            sections[-1].append(line)
    return ["".join(s) for s in sections if s]


def chunk_markdown(path: Path) -> Iterator[dict]:
    text = path.read_text(encoding="utf-8")
    for section in _split_md_sections(text):
        section = section.strip()
        if not section:
            continue
        m = re.match(r"^(#{1,3} .+)", section)
        header_line = m.group(1).strip() if m else ""
        title_m = re.match(r"^#{1,3} (.+)", header_line)
        title = title_m.group(1).strip() if title_m else path.stem
        body_text = section[len(header_line) :].strip() if header_line else section

        sub_chunks = [
            c
            for c in _split_long_section(header_line, body_text)
            if len(c) >= _MIN_CHUNK_BODY
        ]
        total = len(sub_chunks)
        for idx, chunk_body in enumerate(sub_chunks):
            yield {
                "id": _id(),
                "source": str(path),
                **_meta(path, title, idx, total),
                "doc_title": path.stem,
                "chunk_type": "section",
                "title": title if total == 1 else f"{title} ({idx + 1}/{total})",
                "body": chunk_body,
            }


# ── PDF ───────────────────────────────────────────────────────────────────────


def chunk_pdf(path: Path) -> Iterator[dict]:
    import fitz  # pymupdf

    doc = fitz.open(str(path))
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text().strip()
        if len(text) >= _MIN_CHUNK_BODY:
            pages.append((i, text))
    total = len(pages)
    for idx, (page_index, text) in enumerate(pages):
        page_number = page_index + 1
        yield {
            "id": _id(),
            "source": str(path),
            **_meta(path, path.stem, idx, total, page_number, page_number),
            "doc_title": path.stem,
            "chunk_type": "page",
            "title": f"{path.stem} p.{page_number}",
            "body": text,
        }
    doc.close()


# ── DOCX ──────────────────────────────────────────────────────────────────────


def _docx_blocks(doc):
    """Yield Paragraph and Table objects in document order.

    doc.paragraphs alone skips all table-cell text, silently dropping
    tabular content from the index.
    """
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    for child in doc.element.body.iterchildren():
        if child.tag.endswith("}p"):
            yield Paragraph(child, doc)
        elif child.tag.endswith("}tbl"):
            yield Table(child, doc)


def chunk_docx(path: Path) -> Iterator[dict]:
    from docx import Document
    from docx.text.paragraph import Paragraph

    doc = Document(str(path))
    heading = path.stem
    paras: list[str] = []
    sections: list[tuple[str, str]] = []

    def flush():
        if paras:
            sections.append((heading, "\n".join(paras)))

    for block in _docx_blocks(doc):
        if isinstance(block, Paragraph):
            if block.style.name.startswith("Heading"):
                flush()
                paras.clear()
                heading = block.text or heading
            elif block.text.strip():
                paras.append(block.text)
        else:  # Table
            for row in block.rows:
                cells = [c.text.strip() for c in row.cells]
                if any(cells):
                    paras.append(" | ".join(cells))

    flush()
    sections = [(h, b) for h, b in sections if len(b) >= _MIN_CHUNK_BODY]
    total = len(sections)
    for idx, (section_heading, body) in enumerate(sections):
        yield {
            "id": _id(),
            "source": str(path),
            **_meta(path, section_heading, idx, total),
            "doc_title": path.stem,
            "chunk_type": "section",
            "title": section_heading,
            "body": body,
        }


# ── Plain text ────────────────────────────────────────────────────────────────


def chunk_text(path: Path) -> Iterator[dict]:
    text = path.read_text(encoding="utf-8", errors="replace")
    paragraphs = [
        p.strip()
        for p in re.split(r"\n\s*\n", text)
        if len(p.strip()) >= _MIN_CHUNK_BODY
    ]
    total = len(paragraphs)
    for i, para in enumerate(paragraphs):
        yield {
            "id": _id(),
            "source": str(path),
            **_meta(path, path.stem, i, total),
            "doc_title": path.stem,
            "chunk_type": "paragraph",
            "title": f"{path.stem} [{i + 1}]",
            "body": para,
        }


# ── Dispatch ──────────────────────────────────────────────────────────────────

_EXT_MAP = {
    ".yaml": chunk_openapi,
    ".yml": chunk_openapi,
    ".md": chunk_markdown,
    ".markdown": chunk_markdown,
    ".pdf": chunk_pdf,
    ".docx": chunk_docx,
    ".txt": chunk_text,
    ".rst": chunk_text,
}

# Single source of truth for the directory scanner in server.py.
SUPPORTED_EXTENSIONS = frozenset(_EXT_MAP) | {".json"}

_MIN_CHUNK_BODY = _env_int("MIN_CHUNK_BODY", 80, minimum=0)


def current_chunk_config() -> dict[str, int]:
    return {
        "MD_CHUNK_MAX_CHARS": _MD_MAX_CHARS,
        "MD_CHUNK_OVERLAP_CHARS": _MD_OVERLAP_CHARS,
        "MIN_CHUNK_BODY": _MIN_CHUNK_BODY,
    }


def _looks_like_openapi(data) -> bool:
    return isinstance(data, dict) and isinstance(data.get("paths"), dict)


def chunk_file(path: Path) -> list[dict]:
    path = Path(path)
    ext = path.suffix.lower()

    if ext in (".json", ".yaml", ".yml"):
        # Sniff content: OpenAPI specs get endpoint chunks, anything else
        # (configs, data files, malformed specs) falls back to plain text.
        try:
            import yaml

            text = path.read_text(encoding="utf-8")
            data = json.loads(text) if ext == ".json" else yaml.safe_load(text)
            if _looks_like_openapi(data):
                return list(chunk_openapi(path))
            return list(chunk_text(path))
        except Exception:
            return list(chunk_text(path))

    fn = _EXT_MAP.get(ext)
    return list(fn(path)) if fn else []
