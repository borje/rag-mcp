# OpenAI-Compatible Embeddings Plan

## Goal

Add optional HTTP(S) OpenAI-compatible embedding support while keeping local fastembed as default.

Supported targets:

- OpenAI `/v1/embeddings`
- OpenRouter OpenAI-compatible embeddings
- Ollama OpenAI-compatible `/v1/embeddings`

## Behavior

If no `OPENAI_*` env vars are set, use current local fastembed model.

If any relevant `OPENAI_*` env var is set, use OpenAI-compatible HTTP embeddings.

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `RAG_MCP_MODEL` | Local: `BAAI/bge-small-en-v1.5`; OpenAI mode: `text-embedding-3-small` | Embedding model name |
| `OPENAI_BASE_URL` | `https://api.openai.com/v1` | OpenAI-compatible API base URL |
| `OPENAI_API_KEY` | unset | Optional bearer token |
| `OPENAI_EMBEDDINGS_PATH` | `/embeddings` | Embeddings endpoint path |
| `OPENAI_TIMEOUT` | `60` | HTTP timeout seconds |

## Examples

### Local fastembed default

```bash
# No OPENAI_* vars set
RAG_MCP_MODEL=BAAI/bge-small-en-v1.5
```

### OpenAI

```bash
OPENAI_API_KEY=sk-...
RAG_MCP_MODEL=text-embedding-3-small
```

### OpenRouter

```bash
OPENAI_BASE_URL=https://openrouter.ai/api/v1
OPENAI_API_KEY=...
RAG_MCP_MODEL=<embedding-model-id>
```

### Ollama

```bash
OPENAI_BASE_URL=http://localhost:11434/v1
RAG_MCP_MODEL=nomic-embed-text
```

No `OPENAI_API_KEY` required for local Ollama.

## Implementation

### `store.py`

1. Keep local fastembed lazy-loaded.
2. Detect OpenAI-compatible mode from `OPENAI_BASE_URL` or `OPENAI_API_KEY`.
3. Add `_embed_openai(texts: list[str]) -> np.ndarray`.
4. Use stdlib `urllib.request`, no new dependency.
5. POST JSON:

```json
{
  "model": "<RAG_MCP_MODEL>",
  "input": ["text 1", "text 2"]
}
```

6. Add headers:
   - `Content-Type: application/json`
   - `Authorization: Bearer <OPENAI_API_KEY>` only if key is set
7. Parse response:
   - Read `data[].embedding`
   - Sort by `data[].index`
   - Return `np.float32` array
8. Add clear errors for:
   - HTTP failures
   - invalid JSON
   - missing `data`
   - missing `embedding`
   - missing `OPENAI_API_KEY` only when using default OpenAI cloud URL
9. Update `stats()["model"]` to show provider:
   - `fastembed:BAAI/bge-small-en-v1.5`
   - `openai:text-embedding-3-small`

### Dimension Safety

Stored vectors from different embedding models cannot mix.

Add checks:

- During ingest: new vector dimension must match existing store vectors.
- During search: query vector dimension must match stored vectors.

Error message:

```text
Embedding dimension mismatch. The selected embedding model differs from the stored vectors. Clear and re-ingest the store.
```

### Tests

Add tests in `tests/test_store.py`:

1. Default mode remains fastembed when no `OPENAI_*` vars are set.
2. OpenAI mode sends expected URL, headers, and JSON body.
3. OpenAI mode allows missing API key for non-default local base URL, e.g. Ollama.
4. Response embeddings are sorted by `index`.
5. Dimension mismatch raises clear error.
6. Existing tests still pass.

Use monkeypatching for HTTP calls; do not hit network.

### Docs

Update:

- `README.md`
- `.env.example`

Document:

- default offline behavior
- OpenAI/OpenRouter/Ollama examples
- warning that changing embedding model requires clearing/re-ingesting store

## Verification

Run:

```bash
uv run pytest tests/
```

Optional focused run:

```bash
uv run pytest tests/test_store.py
```
