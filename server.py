#!/usr/bin/env python3
"""Hybrid RAG MCP server — offline, fastembed ONNX embeddings, no cloud."""
import json
import os
from pathlib import Path

from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel

from chunkers import chunk_file
from store import RAGStore

store = RAGStore()
mcp = FastMCP(
    "rag-mcp",
    host=os.environ.get("FASTMCP_HOST", "127.0.0.1"),
    port=int(os.environ.get("FASTMCP_PORT", "8000")),
)

BASE_URL = os.environ.get("BASE_URL", "http://localhost:8000").rstrip("/")
FILES_ROOT = Path(os.environ.get("FILES_ROOT", "/data"))


def _file_url(source: str) -> str | None:
    try:
        rel = Path(source).relative_to(FILES_ROOT)
        return f"{BASE_URL}/files/{rel}"
    except ValueError:
        return None


class SearchResult(BaseModel):
    title: str
    doc_title: str | None
    chunk_type: str | None
    file_url: str | None
    score: float
    body: str


class StoreStatus(BaseModel):
    total_chunks: int
    total_sources: int
    model: str
    store_dir: str


class FileIngestResult(BaseModel):
    file: str
    chunks: int


class DirectoryIngestResult(BaseModel):
    total_chunks: int
    total_files: int
    files: list[FileIngestResult]


@mcp.tool()
def ingest_file(path: str) -> str:
    """Ingest a file into the RAG store.
    Supported: OpenAPI (.yaml/.yml/.json), PDF, DOCX, Markdown, plain text.
    """
    p = Path(path).expanduser().resolve()
    if not p.exists():
        return f"Error: file not found: {path}"
    if not p.is_file():
        return f"Error: not a file: {path}"
    chunks = chunk_file(p)
    if not chunks:
        return f"No chunks extracted from {p.name} — unsupported format or empty file"
    n = store.ingest(chunks)
    return f"Ingested {n} chunks from {p.name}"


@mcp.tool()
def ingest_directory(directory: str, glob: str = "**/*") -> DirectoryIngestResult:
    """Ingest all supported files in a directory tree.
    glob defaults to '**/*' (recursive). Example: '*.yaml' for top-level only.
    """
    d = Path(directory).expanduser().resolve()
    if not d.is_dir():
        raise ValueError(f"Not a directory: {directory}")

    supported = {".yaml", ".yml", ".json", ".md", ".markdown", ".pdf", ".docx", ".txt", ".rst"}
    files: list[FileIngestResult] = []
    for f in sorted(d.glob(glob)):
        if f.is_file() and f.suffix.lower() in supported:
            chunks = chunk_file(f)
            if chunks:
                n = store.ingest(chunks)
                files.append(FileIngestResult(file=f.name, chunks=n))

    return DirectoryIngestResult(
        total_chunks=sum(r.chunks for r in files),
        total_files=len(files),
        files=files,
    )


@mcp.tool()
def search(query: str, n_results: int = 8) -> list[SearchResult]:
    """Hybrid vector + BM25 search over ingested documents.
    Returns a JSON array; each item includes file_url for direct download of the source file.
    """
    results = store.search(query, n=n_results)
    return [
        SearchResult(
            title=r["title"],
            doc_title=r.get("doc_title"),
            chunk_type=r.get("chunk_type"),
            file_url=_file_url(r["source"]),
            score=r["score"],
            body=r["body"],
        )
        for r in results
    ]


@mcp.tool()
def list_sources() -> str:
    """List all ingested document source paths."""
    return json.dumps(store.list_sources())


@mcp.tool()
def delete_source(source: str) -> str:
    """Remove all chunks belonging to a source path (exact match from list_sources)."""
    n = store.delete_source(source)
    if n == 0:
        return f"Source not found: {source}"
    return f"Removed {n} chunks from {source}"


@mcp.tool()
def rag_status() -> StoreStatus:
    """Show store statistics: chunk count, source count, model, storage path."""
    return StoreStatus(**store.stats())


if __name__ == "__main__":
    import uvicorn
    transport = os.environ.get("MCP_TRANSPORT", "stdio")
    host = os.environ.get("FASTMCP_HOST", "127.0.0.1")
    port = int(os.environ.get("FASTMCP_PORT", "8000"))

    if transport == "sse":
        from starlette.applications import Starlette
        from starlette.routing import Mount
        from starlette.staticfiles import StaticFiles

        FILES_ROOT.mkdir(parents=True, exist_ok=True)
        app = Starlette(routes=[
            Mount("/files", StaticFiles(directory=str(FILES_ROOT))),
            Mount("/", app=mcp.sse_app()),
        ])
        uvicorn.run(app, host=host, port=port)
    else:
        mcp.run(transport=transport)
