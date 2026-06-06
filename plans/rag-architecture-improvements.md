# RAG Architecture Improvements

Source: recommendations from "Designing a Production-Grade RAG Architecture", compared against this repo.

## Already Covered

- Hybrid vector + BM25 search.
- Reciprocal Rank Fusion.
- `BAAI/bge-small-en-v1.5` embeddings.
- Offline/container-friendly operation.
- Rich chunk metadata: `source_name`, `section_path`, `chunk_index`, `chunk_total`, `page_start`, `page_end`.
- Adjacent chunk expansion, default `RAG_MCP_ADJACENT_CHUNKS=1`, hard-capped by `n_results`.
- mtime-based ingest skipping and stale-source cleanup.

## Recommended Improvements

1. Add weighted hybrid search.

Current RRF weights vector and BM25 equally in `store.py`.

Add configurable weights, defaulting near `dense=0.7`, `bm25=0.3`.

Why: embeddings usually carry semantic intent better, BM25 still catches exact terms/codes.

2. Embed metadata with chunk body.

Current embeddings use only `body`.

Now that chunks have `doc_title`, `source_name`, and `section_path`, use a template like:

```text
Document: {doc_title}
Source: {source_name}
Section: {section_path}

{body}
```

Why: headings and document names often carry key semantic context.

3. Improve non-Markdown chunking.

Markdown already has splitting and overlap. Adjacent expansion now helps all formats, but extraction/chunk quality still matters.

Weak spots:

- PDF: one chunk per page with page metadata, but no section extraction or overlap.
- DOCX: section chunks, but no long-section splitting.
- TXT/RST: paragraph chunks only, no split/merge/overlap.

Refactor `_split_long_section()` into a reusable splitter and apply it across formats.

4. Use token-aware chunk limits.

Current Markdown chunking is char-based.

Move toward token-aware limits or a better approximation aligned with embedding model constraints.

Why: avoids overlong chunks and improves embedding quality.

5. Add optional reranking.

Pipeline:

- Retrieve `n * 2` or `n * 3` candidates.
- Apply existing adjacent chunk expansion.
- Rerank down to `n`.

Keep env-gated because CPU latency and dependency weight may not fit local/offline use.

6. Improve BM25 preprocessing.

Current tokenization is `lower().split()`.

Use shared regex tokenization for ingest and query. Strip punctuation while preserving useful code/API tokens.

Why: small, low-risk retrieval quality win.

7. Use content-hash ingest detection.

Current ingest change detection uses mtime only.

Store hash, or hash + size + mtime.

Why: avoids missed changes and false re-ingests.

8. Consider Qdrant only if scale requires it.

Current numpy + JSON store is simpler and fits local MCP/offline goals. Rich metadata and adjacent expansion reduce the immediate need for Qdrant.

Move to Qdrant only when needing large corpora, concurrent writers, metadata indexes, persistent vector/BM25 indexes, or lower memory pressure.

## Implemented

1. Rich chunk metadata.

Implemented fields: `source_name`, `section_path`, `chunk_index`, `chunk_total`, `page_start`, `page_end`.

Notes:

- PDF chunks include `page_start` and `page_end`.
- Markdown split chunks use per-section indices and totals.
- Existing stores should be force reingested because backward compatibility was intentionally not added.

2. Adjacent chunk expansion.

Implemented behavior:

- Default `RAG_MCP_ADJACENT_CHUNKS=1`.
- `0` disables adjacent expansion.
- `n_results` remains a hard cap.
- Ordering is `hit`, `previous`, `next`.
- Neighbors must match same `source` and `section_path`.
- Results are deduplicated.

Why: improves context completion when retrieval finds the right area but the answer spans chunk boundaries.

## Reingestion

Force reingestion after this schema change by deleting store files, then running `ingest`:

```bash
rm -rf ~/.local/share/rag-mcp/meta.json ~/.local/share/rag-mcp/vectors.npy ~/.local/share/rag-mcp/mtimes.json
```

Docker volume reset alternative:

```bash
docker compose down
docker volume rm rag-mcp_rag-store
docker compose up --build
```

## Not Urgent

- Generation/tool-message orchestration: MCP already exposes search as a tool; clients decide prompt format.
- Separate embedder/reranker services: overkill unless deploying multi-user production.
- Full cloud/self-host architecture split: outside this repo's current local/offline scope.

## Best Next Implementation Order

1. Weighted RRF.
2. Metadata-aware embedding text.
3. Reusable splitter for PDF/DOCX/TXT/RST.
4. Better BM25 tokenizer.
5. Content hash ingest detection.
6. Optional reranker.
7. Token-aware chunk limits.
