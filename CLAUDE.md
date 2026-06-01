# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Setup

```bash
uv venv
uv pip install --python .venv/bin/python -r requirements.txt

# Pre-download embedding model
python -c "from fastembed import TextEmbedding; list(TextEmbedding('BAAI/bge-small-en-v1.5').embed(['warmup']))"
```

No test suite. Manual testing via MCP tools or direct Python invocation.

## Running

**stdio (Claude Code):**
```bash
python server.py
```

**SSE/HTTP:**
```bash
MCP_TRANSPORT=sse python server.py
```

**Docker:**
```bash
docker compose up --build
```

## Architecture

Three modules, no framework beyond FastMCP:

- **`chunkers.py`** — format-specific chunkers (`chunk_openapi`, `chunk_markdown`, `chunk_pdf`, `chunk_docx`, `chunk_text`) dispatched by `chunk_file()`. Each yields dicts with keys: `id`, `source`, `doc_title`, `chunk_type`, `title`, `body`. Chunks with `body` < 80 chars are dropped.

- **`store.py`** — `RAGStore` persists chunks as `meta.json` + `vectors.npy` in `RAG_MCP_DATA`. Lazy-loads fastembed model on first embed. Search: cosine similarity + BM25Okapi → Reciprocal Rank Fusion (k=60). BM25 index rebuilt in-memory on every ingest/delete.

- **`server.py`** — FastMCP tool definitions wiring chunkers → store. SSE mode mounts static file server at `/files/` for `file_url` construction.

## Chunk dict schema

```python
{
    "id": str,           # UUID
    "source": str,       # absolute file path
    "doc_title": str,    # document-level title
    "chunk_type": str,   # "endpoint" | "section" | "page" | "paragraph"
    "title": str,        # chunk-level title
    "body": str,         # searchable text (≥80 chars)
}
```

## Key env vars

| Var | Default |
|-----|---------|
| `RAG_MCP_DATA` | `~/.local/share/rag-mcp` |
| `RAG_MCP_MODEL` | `BAAI/bge-small-en-v1.5` |
| `MCP_TRANSPORT` | `stdio` |
| `BASE_URL` | `http://localhost:8000` |
| `FILES_ROOT` | `/data` |
