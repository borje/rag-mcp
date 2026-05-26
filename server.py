#!/usr/bin/env python3
"""Hybrid RAG MCP server — offline, fastembed ONNX embeddings, no cloud."""
import os
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from chunkers import chunk_file
from store import RAGStore

store = RAGStore()
mcp = FastMCP(
    "rag-mcp",
    host=os.environ.get("FASTMCP_HOST", "127.0.0.1"),
    port=int(os.environ.get("FASTMCP_PORT", "8000")),
)


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
def ingest_directory(directory: str, glob: str = "**/*") -> str:
    """Ingest all supported files in a directory tree.
    glob defaults to '**/*' (recursive). Example: '*.yaml' for top-level only.
    """
    d = Path(directory).expanduser().resolve()
    if not d.is_dir():
        return f"Error: not a directory: {directory}"

    supported = {".yaml", ".yml", ".json", ".md", ".markdown", ".pdf", ".docx", ".txt", ".rst"}
    total = 0
    lines: list[str] = []
    for f in sorted(d.glob(glob)):
        if f.is_file() and f.suffix.lower() in supported:
            chunks = chunk_file(f)
            if chunks:
                n = store.ingest(chunks)
                total += n
                lines.append(f"  {f.name}: {n} chunks")

    if not lines:
        return f"No supported files found in {directory}"
    return f"Ingested {total} chunks from {len(lines)} files:\n" + "\n".join(lines)


@mcp.tool()
def search(query: str, n_results: int = 8) -> str:
    """Hybrid vector + BM25 search over ingested documents.
    Returns the top matching chunks with source paths and relevance scores.
    """
    results = store.search(query, n=n_results)
    if not results:
        return "No results. Use ingest_file or ingest_directory to add documents first."

    parts = []
    for r in results:
        parts.append(
            f"## {r['title']}\n"
            f"Source: {r['source']}  |  Score: {r['score']:.4f}\n\n"
            f"{r['body']}"
        )
    return "\n\n---\n\n".join(parts)


@mcp.tool()
def list_sources() -> str:
    """List all ingested document source paths."""
    sources = store.list_sources()
    if not sources:
        return "No documents ingested yet."
    return "\n".join(sources)


@mcp.tool()
def delete_source(source: str) -> str:
    """Remove all chunks belonging to a source path (exact match from list_sources)."""
    n = store.delete_source(source)
    if n == 0:
        return f"Source not found: {source}"
    return f"Removed {n} chunks from {source}"


@mcp.tool()
def rag_status() -> str:
    """Show store statistics: chunk count, source count, model, storage path."""
    s = store.stats()
    return (
        f"Chunks: {s['total_chunks']}\n"
        f"Sources: {s['total_sources']}\n"
        f"Embedding model: {s['model']}\n"
        f"Store path: {s['store_dir']}"
    )


if __name__ == "__main__":
    mcp.run(transport=os.environ.get("MCP_TRANSPORT", "stdio"))
