#!/usr/bin/env python3
"""Hybrid RAG MCP server — offline, fastembed ONNX embeddings, no cloud."""
import json
import os
import sys
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
    if str(p) in store.list_sources():
        return f"Already ingested: {p.name}"
    chunks = chunk_file(p)
    if not chunks:
        return f"No chunks extracted from {p.name} — unsupported format or empty file"
    n = store.ingest(chunks)
    return f"Ingested {n} chunks from {p.name}"


def _ingest_directory(directory: Path, glob: str = "**/*") -> DirectoryIngestResult:
    """Walk a directory, ingesting supported files not already in the store."""
    supported = {".yaml", ".yml", ".json", ".md", ".markdown", ".pdf", ".docx", ".txt", ".rst"}
    files: list[FileIngestResult] = []
    ingested_sources = set(store.list_sources())
    for f in sorted(directory.glob(glob)):
        if f.is_file() and f.suffix.lower() in supported and str(f) not in ingested_sources:
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
def ingest_directory(directory: str, glob: str = "**/*") -> DirectoryIngestResult:
    """Ingest all supported files in a directory tree.
    glob defaults to '**/*' (recursive). Example: '*.yaml' for top-level only.
    """
    d = Path(directory).expanduser().resolve()
    if not d.is_dir():
        raise ValueError(f"Not a directory: {directory}")
    return _ingest_directory(d, glob)


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
        from contextlib import asynccontextmanager

        from starlette.applications import Starlette
        from starlette.routing import Mount
        from starlette.staticfiles import StaticFiles

        FILES_ROOT.mkdir(parents=True, exist_ok=True)

        def _startup_ingest():
            print(f"[startup] scanning {FILES_ROOT} for new documents…", flush=True)
            try:
                supported = {".yaml", ".yml", ".json", ".md", ".markdown", ".pdf", ".docx", ".txt", ".rst"}
                ingested_sources = set(store.list_sources())

                removed = [s for s in ingested_sources if not Path(s).exists()]
                for s in removed:
                    n = store.delete_source(s)
                    print(f"[startup] removed stale source {Path(s).name} ({n} chunks)", flush=True)

                pending = [
                    f for f in sorted(FILES_ROOT.glob("**/*"))
                    if f.is_file() and f.suffix.lower() in supported and str(f) not in ingested_sources
                ]
                total = len(pending)
                if total == 0:
                    print("[startup] no new documents", flush=True)
                    return
                all_chunks = 0
                for i, f in enumerate(pending, 1):
                    chunks = chunk_file(f)
                    if chunks:
                        n = store.ingest(chunks)
                        all_chunks += n
                        print(f"[startup] [{i}/{total}] {f.name} ({n} chunks)", flush=True)
                    else:
                        print(f"[startup] [{i}/{total}] {f.name} (skipped — no chunks)", flush=True)
                print(f"[startup] done — {all_chunks} chunks from {total} file(s)", flush=True)
            except Exception as e:
                print(f"[startup] ingestion failed: {e}", file=sys.stderr, flush=True)

        @asynccontextmanager
        async def lifespan(app):
            try:
                _startup_ingest()
            except KeyboardInterrupt:
                print("[startup] interrupted", flush=True)
            yield

        app = Starlette(
            lifespan=lifespan,
            routes=[
                Mount("/files", StaticFiles(directory=str(FILES_ROOT))),
                Mount("/", app=mcp.sse_app()),
            ],
        )
        try:
            uvicorn.run(app, host=host, port=port)
        except (KeyboardInterrupt, SystemExit):
            pass
    else:
        mcp.run(transport=transport)
