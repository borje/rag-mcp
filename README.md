# rag-mcp

A hybrid RAG (Retrieval-Augmented Generation) server exposed as an [MCP](https://modelcontextprotocol.io) tool. Runs fully offline — no cloud APIs, no telemetry. Embeddings use [fastembed](https://github.com/qdrant/fastembed) with a bundled ONNX model.

## How it works

Search combines two signals via [Reciprocal Rank Fusion](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf):

- **Vector search** — cosine similarity over `BAAI/bge-small-en-v1.5` embeddings
- **BM25** — classic keyword search

This means exact-term queries and semantic queries both work well.

By default, search also includes adjacent chunks from the same source and section (`RAG_MCP_ADJACENT_CHUNKS=1`) while keeping `n_results` as a hard cap.

## Supported file types

| Format | Chunking strategy |
|--------|------------------|
| OpenAPI (`.yaml`, `.yml`, `.json`) | One chunk per endpoint |
| Markdown / RST | One chunk per heading section |
| Plain text | One chunk per paragraph |
| Other (PDF, DOCX, …) via `RAG_MCP_CONVERT_CMD` | Converted to Markdown, then one chunk per heading section. Off by default, see [Environment variables](#environment-variables) |

Files with other extensions are skipped and listed in `ignored_files` of the `ingest` result.

## MCP tools

| Tool | Description |
|------|-------------|
| `ingest` | Scan `FILES_ROOT` and ingest new/changed supported documents. Result includes `ignored_files` (unsupported files) |
| `reindex` | Force re-ingest of one file, or every file under a directory (path relative to `FILES_ROOT` or absolute under it; the root itself, missing paths and paths outside are rejected). Old chunks stay searchable until each file is replaced |
| `search` | Hybrid vector + BM25 search, optional `scope` subtree filter, returns JSON |
| `list_sources` | List all ingested source paths |
| `list_scopes` | List subtree scopes under `FILES_ROOT` with doc counts, for `search(scope=...)` |
| `rag_status` | Chunk/source counts, model, store path, `last_scan`, `watch_interval`, `supported_extensions` |

### Search result format

```json
[
  {
    "title": "POST /api/users",
    "source_name": "api-spec.yaml",
    "doc_title": "My API",
    "chunk_type": "endpoint",
    "section_path": "POST /api/users",
    "chunk_index": 0,
    "chunk_total": 1,
    "file_url": "http://localhost:8000/files/api-spec.yaml",
    "score": 0.0312,
    "match_type": "hit",
    "body": "POST /api/users\nSummary: Create a new user\n..."
  }
]
```

`match_type` is `hit` for directly ranked chunks and `adjacent` for context chunks included from the same source and section.

`file_url` points to the source file served by the built-in HTTP server (SSE mode). It is `null` if the file is not under `FILES_ROOT`.

## Running with Docker (recommended)

```bash
docker compose up --build
```

The image build and first embedding model download require internet. The server listens on `EXTERNAL_PORT` (`8001` by default in Docker Compose). Source files placed in `DATA_DIR` are served under `/files/`. The vector store lives in the `rag-store` Docker volume and the model cache lives in `rag-models`.

Chunking can be tuned from Docker Compose or a `.env` file:

```yaml
environment:
  BASE_URL: http://${FQDN:-localhost}:${EXTERNAL_PORT:-8001}
  MD_CHUNK_MAX_CHARS: ${MD_CHUNK_MAX_CHARS:-1500}
  MD_CHUNK_OVERLAP_CHARS: ${MD_CHUNK_OVERLAP_CHARS:-150}
  MIN_CHUNK_BODY: ${MIN_CHUNK_BODY:-80}
```

Example `.env` override:

```env
MD_CHUNK_MAX_CHARS=600
MD_CHUNK_OVERLAP_CHARS=100
MIN_CHUNK_BODY=60
```

Chunking settings are tracked in the persisted store manifest. If chunking config or the embedding model changes, rag-mcp automatically clears the old index and rebuilds it from `FILES_ROOT` on the next startup or ingest.

To build the Docker image fully offline after creating/copying `transfer/`, use:

```bash
docker compose -f docker-compose.yaml -f docker-compose.offline.yaml build
```

**Ingest your docs:**
```
ingest
```

To remove documents from the index, delete them from `FILES_ROOT`, then run `ingest` or restart the server.

**Force full reingestion:**

```bash
docker compose exec rag-mcp /app/reset-store.sh
docker compose restart rag-mcp
```

This clears persisted store files inside the container store volume. Restarting triggers startup ingest and rebuilds the index from `FILES_ROOT`.

> Set `BASE_URL` in `docker-compose.yaml` to your externally-accessible hostname when deploying behind a reverse proxy.

## Typical workflows

**Add documents**
1. Copy files into `FILES_ROOT` (or `DATA_DIR` in Docker)
2. Call `ingest` — only new/changed files are re-chunked (mtime+size first, then a SHA-256 of the content when those differ; a touched but identical file is not re-embedded)

**Remove documents**
1. Delete files from `FILES_ROOT`
2. Call `ingest` — stale chunks are purged automatically

**Update a document**
1. Overwrite the file in `FILES_ROOT`
2. Call `ingest` — a content change (detected via mtime/size, confirmed by SHA-256) triggers re-chunk

**Force re-ingest of one file**
- `reindex` with the path relative to `FILES_ROOT`

**Index PDF / DOCX (optional converter)**
1. Install a converter that prints Markdown to stdout, e.g. `pip install markitdown` or `pandoc`
2. Set `RAG_MCP_CONVERT_CMD="markitdown {input}"` (or `pandoc -t gfm {input}` for DOCX only)
3. Restart — the changed config triggers a rebuild; converted files are chunked like Markdown

**Check index health**
- `rag_status` — chunk + source counts, model, store path, last scan summary, supported extensions
- `list_sources` — paths of every ingested file

**Docker restart as shortcut**
`docker compose restart` triggers startup ingest — equivalent to calling `ingest` manually.

**Force full reingestion**
`docker compose exec rag-mcp /app/reset-store.sh`, then `docker compose restart rag-mcp`.

## Running locally (stdio, for Claude Code)

```bash
# Install deps
python3.13 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Pre-download the embedding model
python -c "from fastembed import TextEmbedding; list(TextEmbedding('BAAI/bge-small-en-v1.5').embed(['warmup']))"

# Register with Claude Code
claude mcp add rag-mcp \
  -e MCP_TRANSPORT=stdio \
  -e FASTEMBED_CACHE_PATH="$HOME/.local/share/rag-mcp/models" \
  -- python server.py
```

## Offline / air-gapped deployment

Run `prepare-transfer.sh` on an internet-connected machine to produce a `transfer/` bundle containing wheels, the ONNX model, source, and an `install.sh`:

```bash
bash prepare-transfer.sh
# Copy transfer/ to the offline machine, then:
bash transfer/install.sh
```

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `RAG_MCP_DATA` | `~/.local/share/rag-mcp` | Vector store directory |
| `FASTEMBED_CACHE_PATH` | *(fastembed default)* | ONNX model cache directory |
| `BASE_URL` | `http://localhost:8000` | Public base URL for `file_url` construction |
| `FILES_ROOT` | `/data` | Directory served at `/files/` |
| `MCP_TRANSPORT` | `streamable-http` | `stdio`, `sse`, or `streamable-http` |
| `FASTMCP_HOST` | `127.0.0.1` | Bind address (HTTP/SSE modes) |
| `FASTMCP_PORT` | `8000` | Port (HTTP/SSE modes) |
| `RAG_MCP_MODEL` | `BAAI/bge-small-en-v1.5` | fastembed model name |
| `RAG_MCP_WATCH_INTERVAL` | `30` | Seconds between auto-ingest polls (SSE/HTTP only). Set to `0` to disable. |
| `RAG_MCP_ADJACENT_CHUNKS` | `1` | Adjacent chunks before/after each hit to include from the same source and section. Set to `0` to disable. |
| `MD_CHUNK_MAX_CHARS` | `1500` | Maximum size of a markdown sub-chunk in characters. |
| `MD_CHUNK_OVERLAP_CHARS` | `150` | Overlap to keep between adjacent markdown sub-chunks. Must be smaller than `MD_CHUNK_MAX_CHARS`. |
| `MIN_CHUNK_BODY` | `80` | Drop chunks whose body is shorter than this many characters. |
| `RAG_MCP_CONVERT_CMD` | *(unset)* | Command template that converts a file to Markdown on stdout; `{input}` is replaced with the file path. Split like a shell command line but run without a shell (no pipes or redirects; wrap those in a script). Changing it does not rebuild the store: run `reindex` on the affected directory. Unset disables conversion. |
| `RAG_MCP_CONVERT_EXTS` | `.pdf,.docx` | Comma-separated extensions routed through `RAG_MCP_CONVERT_CMD`. |
| `RAG_MCP_CONVERT_TIMEOUT` | `120` | Seconds before a conversion is aborted (file is reported as failed). |
