"""Document chunkers: OpenAPI, Markdown, PDF, DOCX, plain text."""
import json
import re
import uuid
from pathlib import Path
from typing import Iterator


def _id() -> str:
    return str(uuid.uuid4())


# ── OpenAPI (OAS2 / OAS3) ────────────────────────────────────────────────────

def chunk_openapi(path: Path) -> Iterator[dict]:
    import yaml
    text = path.read_text(encoding="utf-8")
    spec = yaml.safe_load(text) if path.suffix in (".yaml", ".yml") else json.loads(text)

    title = spec.get("info", {}).get("title", path.stem)
    base = spec.get("servers", [{}])[0].get("url", "") or spec.get("host", "")

    for path_str, methods in (spec.get("paths") or {}).items():
        if not isinstance(methods, dict):
            continue
        for method, op in methods.items():
            if method.startswith("x-") or not isinstance(op, dict):
                continue

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
                rlines = [f"  {code}: {(v or {}).get('description','')}" for code, v in list(responses.items())[:8]]
                lines.append("Responses:\n" + "\n".join(rlines))

            yield {
                "id": _id(),
                "source": str(path),
                "doc_title": title,
                "chunk_type": "endpoint",
                "title": f"{method.upper()} {path_str}",
                "body": "\n".join(lines),
            }


# ── Markdown ─────────────────────────────────────────────────────────────────

def chunk_markdown(path: Path) -> Iterator[dict]:
    text = path.read_text(encoding="utf-8")
    sections = re.split(r'(?m)^(?=#{1,3} )', text)
    for section in sections:
        section = section.strip()
        if not section:
            continue
        m = re.match(r'^(#{1,3}) (.+)', section)
        title = m.group(2).strip() if m else path.stem
        yield {
            "id": _id(),
            "source": str(path),
            "doc_title": path.stem,
            "chunk_type": "section",
            "title": title,
            "body": section,
        }


# ── PDF ───────────────────────────────────────────────────────────────────────

def chunk_pdf(path: Path) -> Iterator[dict]:
    import fitz  # pymupdf
    doc = fitz.open(str(path))
    for i, page in enumerate(doc):
        text = page.get_text().strip()
        if text:
            yield {
                "id": _id(),
                "source": str(path),
                "doc_title": path.stem,
                "chunk_type": "page",
                "title": f"{path.stem} p.{i + 1}",
                "body": text,
            }
    doc.close()


# ── DOCX ──────────────────────────────────────────────────────────────────────

def chunk_docx(path: Path) -> Iterator[dict]:
    from docx import Document
    doc = Document(str(path))
    heading = path.stem
    paras: list[str] = []

    def flush():
        if paras:
            yield {
                "id": _id(),
                "source": str(path),
                "doc_title": path.stem,
                "chunk_type": "section",
                "title": heading,
                "body": "\n".join(paras),
            }

    for para in doc.paragraphs:
        if para.style.name.startswith("Heading"):
            yield from flush()
            paras.clear()
            heading = para.text or heading
        elif para.text.strip():
            paras.append(para.text)

    yield from flush()


# ── Plain text ────────────────────────────────────────────────────────────────

def chunk_text(path: Path) -> Iterator[dict]:
    text = path.read_text(encoding="utf-8", errors="replace")
    paragraphs = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]
    for i, para in enumerate(paragraphs):
        yield {
            "id": _id(),
            "source": str(path),
            "doc_title": path.stem,
            "chunk_type": "paragraph",
            "title": f"{path.stem} [{i + 1}]",
            "body": para,
        }


# ── Dispatch ──────────────────────────────────────────────────────────────────

_EXT_MAP = {
    ".yaml": chunk_openapi,
    ".yml":  chunk_openapi,
    ".md":   chunk_markdown,
    ".markdown": chunk_markdown,
    ".pdf":  chunk_pdf,
    ".docx": chunk_docx,
    ".txt":  chunk_text,
    ".rst":  chunk_text,
}


def chunk_file(path: Path) -> list[dict]:
    path = Path(path)
    ext = path.suffix.lower()

    if ext == ".json":
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if "paths" in data and isinstance(data["paths"], dict):
                return list(chunk_openapi(path))
        except Exception:
            pass
        return list(chunk_text(path))

    fn = _EXT_MAP.get(ext)
    return list(fn(path)) if fn else []
