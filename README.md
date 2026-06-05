# rag-mcp

A hybrid RAG (Retrieval-Augmented Generation) server exposed as an [MCP](https://modelcontextprotocol.io) tool. Runs fully offline — no cloud APIs, no telemetry. Embeddings use [fastembed](https://github.com/qdrant/fastembed) with a bundled ONNX model.

## How it works

Search combines two signals via [Reciprocal Rank Fusion](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf):

- **Vector search** — cosine similarity over `BAAI/bge-small-en-v1.5` embeddings
- **BM25** — classic keyword search

This means exact-term queries and semantic queries both work well.

## Supported file types

| Format | Chunking strategy |
|--------|------------------|
| OpenAPI (`.yaml`, `.yml`, `.json`) | One chunk per endpoint |
| Markdown / RST | One chunk per heading section |
| PDF | One chunk per page |
| DOCX | One chunk per heading section |
| Plain text | One chunk per paragraph |

## MCP tools

| Tool | Description |
|------|-------------|
| `ingest` | Scan `FILES_ROOT` and ingest supported new documents |
| `search` | Hybrid vector + BM25 search, returns JSON |
| `list_sources` | List all ingested source paths |
| `rag_status` | Chunk count, source count, model, store path |

### Search result format

```json
[
  {
    "title": "POST /api/users",
    "doc_title": "My API",
    "chunk_type": "endpoint",
    "file_url": "http://localhost:8000/files/api-spec.yaml",
    "score": 0.0312,
    "body": "POST /api/users\nSummary: Create a new user\n..."
  }
]
```

`file_url` points to the source file served by the built-in HTTP server (SSE mode). It is `null` if the file is not under `FILES_ROOT`.

## Running with Docker (recommended)

```bash
# First time: build the offline bundle (requires internet)
bash prepare-transfer.sh

# Start the server
docker compose up --build
```

The server listens on `EXTERNAL_PORT` (`8001` by default in Docker Compose). Source files placed in `DATA_DIR` are served under `/files/`. The vector store lives in the `rag-store` Docker volume.

**Ingest your docs:**
```
ingest
```

To remove documents from the index, delete them from `FILES_ROOT`, then run `ingest` or restart the server.

> Set `BASE_URL` in `docker-compose.yaml` to your externally-accessible hostname when deploying behind a reverse proxy.

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
