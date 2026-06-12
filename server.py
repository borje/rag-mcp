#!/usr/bin/env python3
"""Hybrid RAG MCP server — offline, fastembed ONNX embeddings, no cloud."""

import asyncio
import json
import os
import sys
import threading
from pathlib import Path

from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel

import dashboard
from chunkers import SUPPORTED_EXTENSIONS, chunk_file
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

_ingest_lock = threading.Lock()


def _file_url(source: str) -> str | None:
    return dashboard.file_url(source, FILES_ROOT, BASE_URL)



def _relative_source(path: Path) -> str:
    try:
        return path.relative_to(FILES_ROOT).as_posix()
    except ValueError:
        return path.name


def dashboard_data() -> dict:
    return dashboard.dashboard_data(store, FILES_ROOT, BASE_URL)


def dashboard_chunk(chunk_id: str) -> dict | None:
    return dashboard.dashboard_chunk(store, FILES_ROOT, chunk_id)


def dashboard_html() -> str:
    return dashboard.dashboard_html(BASE_URL)


class SearchResult(BaseModel):
    title: str
    source_name: str
    doc_title: str | None
    chunk_type: str | None
    section_path: str
    chunk_index: int
    chunk_total: int
    page_start: int | None
    page_end: int | None
    file_url: str | None
    score: float
    match_type: str
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
    failed_files: list[str] = []
    removed_sources: list[str]


def _ingest_files_root(log=None) -> IngestResult:
    """Ingest supported files under FILES_ROOT and remove stale sources."""
    with _ingest_lock:
        FILES_ROOT.mkdir(parents=True, exist_ok=True)
        removed_sources = _cleanup_stale_sources()
        files: list[FileIngestResult] = []
        skipped: list[str] = []
        failed: list[str] = []
        candidates = [
            f
            for f in sorted(FILES_ROOT.glob("**/*"))
            if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
        ]
        if log:
            log(f"found {len(candidates)} supported file(s)")
        for i, f in enumerate(candidates, 1):
            label = f.relative_to(FILES_ROOT)
            try:
                current_mtime = f.stat().st_mtime
                if store.source_mtime(str(f)) == current_mtime:
                    if log:
                        log(f"[{i}/{len(candidates)}] unchanged {label}")
                    continue
                # Always remove existing chunks for this source first: an ingest
                # interrupted before the mtime was recorded leaves orphan chunks
                # that would otherwise be duplicated by the re-ingest.
                stale = store.delete_source(str(f))
                if stale and log:
                    log(f"[{i}/{len(candidates)}] {label}: removed {stale} old chunks")
                if log:
                    log(f"[{i}/{len(candidates)}] chunking {label}")
                chunks = chunk_file(f)
                if chunks:
                    relative_source = _relative_source(f)
                    for chunk in chunks:
                        chunk["relative_source"] = relative_source
                    if log:
                        log(
                            f"[{i}/{len(candidates)}] embedding {label} "
                            f"({len(chunks)} chunks)"
                        )
                    n = store.ingest(
                        chunks,
                        mtime=current_mtime,
                        log=(
                            lambda message: log(
                                f"[{i}/{len(candidates)}] {label}: {message}"
                            )
                        )
                        if log
                        else None,
                    )
                    if log:
                        log(f"[{i}/{len(candidates)}] ingested {label} ({n} chunks)")
                    files.append(FileIngestResult(file=f.name, chunks=n))
                else:
                    if log:
                        log(
                            f"[{i}/{len(candidates)}] skipped {label} "
                            f"(no chunks extracted)"
                        )
                    skipped.append(f.name)
            except Exception as e:
                # One bad file must not abort the scan for the remaining files.
                print(f"[ingest] failed {label}: {e}", file=sys.stderr, flush=True)
                if log:
                    log(f"[{i}/{len(candidates)}] FAILED {label}: {e}")
                failed.append(f.name)

        return IngestResult(
            total_chunks=sum(r.chunks for r in files),
            total_files=len(files),
            files=files,
            skipped_files=skipped,
            failed_files=failed,
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
def search(
    query: str, n_results: int = 8, scope: str | None = None
) -> list[SearchResult]:
    """Hybrid vector + BM25 search over ingested documents.
    Optionally restrict matches to a subtree under FILES_ROOT using scope,
    expressed as a relative path prefix like "books/trading".
    Returns a JSON array; each item includes file_url for direct download of the source file.
    """
    results = store.search(query, n=n_results, scope=scope)
    return [
        SearchResult(
            title=r["title"],
            source_name=r["source_name"],
            doc_title=r.get("doc_title"),
            chunk_type=r.get("chunk_type"),
            section_path=r["section_path"],
            chunk_index=r["chunk_index"],
            chunk_total=r["chunk_total"],
            page_start=r["page_start"],
            page_end=r["page_end"],
            file_url=_file_url(r["source"]),
            score=r["score"],
            match_type=r["match_type"],
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

    # Default matches the documented Claude Code stdio setup; the Docker
    # images set MCP_TRANSPORT=streamable-http explicitly.
    transport = os.environ.get("MCP_TRANSPORT", "stdio")
    host = os.environ.get("FASTMCP_HOST", "127.0.0.1")
    port = int(os.environ.get("FASTMCP_PORT", "8000"))

    if transport in {"sse", "streamable-http"}:
        from contextlib import asynccontextmanager

        from starlette.applications import Starlette
        from starlette.responses import HTMLResponse, JSONResponse
        from starlette.routing import Mount, Route
        from starlette.staticfiles import StaticFiles

        mcp_app = mcp.sse_app() if transport == "sse" else mcp.streamable_http_app()

        async def _dashboard_page(request):
            return HTMLResponse(dashboard_html())

        async def _dashboard_data_route(request):
            return JSONResponse(dashboard_data())

        async def _dashboard_chunk_route(request):
            chunk = dashboard_chunk(request.path_params["chunk_id"])
            if chunk is None:
                return JSONResponse({"error": "chunk not found"}, status_code=404)
            return JSONResponse(chunk)

        def _startup_ingest():
            print(f"[startup] scanning {FILES_ROOT} for new documents…", flush=True)
            if store.manifest_reset_reason:
                print(f"[startup] {store.manifest_reset_reason}", flush=True)
                store.manifest_reset_reason = None
            try:
                result = _ingest_files_root(
                    lambda message: print(f"[startup] {message}", flush=True)
                )
                for s in result.removed_sources:
                    print(f"[startup] removed stale source {Path(s).name}", flush=True)
                for name in result.skipped_files:
                    print(f"[startup] skipped {name} (no chunks extracted)", flush=True)
                for name in result.failed_files:
                    print(f"[startup] FAILED {name}", file=sys.stderr, flush=True)
                if (
                    result.total_files == 0
                    and not result.removed_sources
                    and not result.skipped_files
                    and not result.failed_files
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
                # Off the event loop: a synchronous ingest (chunking + ONNX
                # embedding) here would freeze every MCP/dashboard request.
                await asyncio.to_thread(_startup_ingest)

        @asynccontextmanager
        async def lifespan(app):
            try:
                _startup_ingest()
            except KeyboardInterrupt:
                print("[startup] interrupted", flush=True)
            if WATCH_INTERVAL > 0:
                # Keep a strong reference: the event loop only holds weak
                # refs to tasks, so an unreferenced task can be GC'd.
                app.state.watch_task = asyncio.create_task(
                    _watch_loop(WATCH_INTERVAL)
                )
            if transport == "streamable-http":
                async with mcp_app.router.lifespan_context(mcp_app):
                    yield
            else:
                yield

        app = Starlette(
            lifespan=lifespan,
            routes=[
                Route("/dashboard", _dashboard_page),
                Route("/dashboard/data", _dashboard_data_route),
                Route("/dashboard/chunk/{chunk_id:path}", _dashboard_chunk_route),
                Mount(
                    "/files",
                    StaticFiles(directory=str(FILES_ROOT), check_dir=False),
                ),
                Mount("/", app=mcp_app),
            ],
        )
        try:
            uvicorn.run(app, host=host, port=port)
        except (KeyboardInterrupt, SystemExit):
            pass
    else:
        mcp.run(transport=transport)
