#!/usr/bin/env python3
"""Hybrid RAG MCP server — offline, fastembed ONNX embeddings, no cloud."""

import asyncio
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
WATCH_INTERVAL = int(os.environ.get("RAG_MCP_WATCH_INTERVAL", "30"))


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


class IngestResult(BaseModel):
    total_chunks: int
    total_files: int
    files: list[FileIngestResult]
    skipped_files: list[str]
    removed_sources: list[str]


def _ingest_files_root() -> IngestResult:
    """Ingest supported files under FILES_ROOT and remove stale sources."""
    supported = {
        ".yaml",
        ".yml",
        ".json",
        ".md",
        ".markdown",
        ".pdf",
        ".docx",
        ".txt",
        ".rst",
    }
    FILES_ROOT.mkdir(parents=True, exist_ok=True)
    removed_sources = _cleanup_stale_sources()
    files: list[FileIngestResult] = []
    skipped: list[str] = []
    for f in sorted(FILES_ROOT.glob("**/*")):
        if not f.is_file() or f.suffix.lower() not in supported:
            continue
        current_mtime = f.stat().st_mtime
        if store.source_mtime(str(f)) == current_mtime:
            continue
        if store.source_mtime(str(f)) is not None:
            store.delete_source(str(f))
        chunks = chunk_file(f)
        if chunks:
            n = store.ingest(chunks, mtime=current_mtime)
            files.append(FileIngestResult(file=f.name, chunks=n))
        else:
            skipped.append(f.name)

    return IngestResult(
        total_chunks=sum(r.chunks for r in files),
        total_files=len(files),
        files=files,
        skipped_files=skipped,
        removed_sources=removed_sources,
    )


def _cleanup_stale_sources() -> list[str]:
    """Delete store entries whose source files no longer exist. Returns removed paths."""
    removed = []
    for s in store.list_sources():
        if not Path(s).exists():
            store.delete_source(s)
            removed.append(s)
    return removed


@mcp.tool()
def ingest() -> IngestResult:
    """Scan FILES_ROOT for supported documents and ingest new/changed store state.

    Remove files from FILES_ROOT, then run this tool to remove stale chunks.
    """
    return _ingest_files_root()


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
def rag_status() -> StoreStatus:
    """Show store statistics: chunk count, source count, model, storage path."""
    return StoreStatus(**store.stats())


if __name__ == "__main__":
    import uvicorn

    transport = os.environ.get("MCP_TRANSPORT", "streamable-http")
    host = os.environ.get("FASTMCP_HOST", "127.0.0.1")
    port = int(os.environ.get("FASTMCP_PORT", "8000"))

    if transport in {"sse", "streamable-http"}:
        from contextlib import asynccontextmanager

        from starlette.applications import Starlette
        from starlette.routing import Mount
        from starlette.staticfiles import StaticFiles

        mcp_app = mcp.sse_app() if transport == "sse" else mcp.streamable_http_app()

        def _startup_ingest():
            print(f"[startup] scanning {FILES_ROOT} for new documents…", flush=True)
            try:
                result = _ingest_files_root()
                for s in result.removed_sources:
                    print(f"[startup] removed stale source {Path(s).name}", flush=True)
                for name in result.skipped_files:
                    print(f"[startup] skipped {name} (no chunks extracted)", flush=True)
                if (
                    result.total_files == 0
                    and not result.removed_sources
                    and not result.skipped_files
                ):
                    print("[startup] no new documents", flush=True)
                    return
                for i, f in enumerate(result.files, 1):
                    print(
                        f"[startup] [{i}/{result.total_files}] {f.file} ({f.chunks} chunks)",
                        flush=True,
                    )
                if result.total_files > 0:
                    print(
                        f"[startup] done — {result.total_chunks} chunks from {result.total_files} file(s)",
                        flush=True,
                    )
            except Exception as e:
                print(f"[startup] ingestion failed: {e}", file=sys.stderr, flush=True)

        async def _watch_loop(interval: int) -> None:
            while True:
                await asyncio.sleep(interval)
                _startup_ingest()

        @asynccontextmanager
        async def lifespan(app):
            try:
                _startup_ingest()
            except KeyboardInterrupt:
                print("[startup] interrupted", flush=True)
            if WATCH_INTERVAL > 0:
                asyncio.create_task(_watch_loop(WATCH_INTERVAL))
            if transport == "streamable-http":
                async with mcp_app.router.lifespan_context(mcp_app):
                    yield
            else:
                yield

        app = Starlette(
            lifespan=lifespan,
            routes=[
                Mount("/files", StaticFiles(directory=str(FILES_ROOT))),
                Mount("/", app=mcp_app),
            ],
        )
        try:
            uvicorn.run(app, host=host, port=port)
        except (KeyboardInterrupt, SystemExit):
            pass
    else:
        mcp.run(transport=transport)
