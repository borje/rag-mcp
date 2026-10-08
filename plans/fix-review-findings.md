# Fix review findings for `7155f83 remove PDF and DOCX support`

Status: done 2026-10-08 (T1-T4 implemented, transfer bundle regenerated, 98 tests passing). Source: `/code-review` on HEAD, plus its product suggestions.
Owner runs phases; each task is sized for one Opus subagent. Tasks inside a phase
touch disjoint files and run in parallel in git worktrees; phases run in order.

Ground rules for every task
- TDD (red → green → refactor). Tests in `tests/`. Run `.venv-313/bin/python -m pytest tests/ -q`.
- No new runtime dependencies. `requirements.txt` stays as is.
- Shortest working diff. No abstractions for one caller. Mark deliberate ceilings with a `# ponytail:` comment.
- Only touch the files listed under "Files". If something outside is needed, stop and report instead.
- Do not commit. The owner merges and commits.

## Phase 1 (parallel)

### T1 — store.py: batch delete, deferred BM25 rebuild, tokenizer, drop page_* (store owner)
Files: `store.py`, new `tests/test_store_batch.py`.
1. Remove the two `page_start`/`page_end` `setdefault` lines in `_backfill_meta` (store.py:117). Old persisted chunks with those keys keep loading; nothing reads them.
2. `delete_sources(sources: Iterable[str]) -> dict[str, int]`: filter chunks/bodies/vectors once, drop mtimes, one `_save()`, one `_rebuild_bm25()`, one `_reload_vectors_mmapped()`. Reimplement `delete_source(s)` as `self.delete_sources([s]).get(s, 0)`.
3. Deferred BM25 rebuild for scans: `ingest(..., rebuild_bm25: bool = True)`; when False skip `_rebuild_bm25()`. Add public `rebuild_bm25()`. In `search`, the snapshot `bm25` may cover fewer docs than `chunks` while a scan is running: pad `bm25_scores` with zeros to `len(chunks)` before scope indexing (`np.pad`). Deletes always rebuild (positions shift). Comment: `# ponytail: BM25 is eventually consistent during a scan; new chunks rank by vector only until rebuild_bm25()`.
4. BM25 tokenizer: one module-level `_tokenize(text) -> list[str]` = `re.findall(r"\w+", text.lower())`, used in `_rebuild_bm25` and for `query_terms` in `search`. Bump nothing in the manifest for this: BM25 is in-memory and rebuilt on load.
5. Tests (new file): delete_sources removes several sources in one call and returns per-source counts; `_save` called once (monkeypatch); after `ingest(rebuild_bm25=False)` search still works and returns the new chunk by vector; after `rebuild_bm25()` a BM25-only term in the new chunk is found; tokenizer test: body "GET /api/users." is found by query "api users"; `page_start` absent from ingested chunk dicts.

### T2 — docs, scripts, test fixtures (no production code)
Files: `README.md`, `CLAUDE.md`, `plans/rag-architecture-improvements.md`, `prepare-transfer.sh`, `tests/test_regressions.py`, `tests/test_store.py`.
1. `tests/test_store.py` ~line 177: flaky `test_changed_file_triggers_reingest`. Move `os.utime(...)` AFTER `doc.write_text(...)` and set it to `stat().st_mtime + 10`. Run the test 5 times to confirm.
2. `tests/test_regressions.py` lines 38-39, 124-125, 183-184, 361-362: delete the `page_start`/`page_end` pairs.
3. `README.md`: remove PDF/DOCX rows from "Supported file types" (line 22-23); remove `page_start`/`page_end` from the search-result JSON (47-48); add a row to "Supported file types": "Other (PDF, DOCX, …) via `RAG_MCP_CONVERT_CMD`, see Environment variables" (T3 implements it; the env var is `RAG_MCP_CONVERT_CMD`, a shell template with `{input}` that prints Markdown to stdout, and `RAG_MCP_CONVERT_EXTS`, default `.pdf,.docx`). Add both to the env table. Add `list_scopes` to the MCP tools table (already exists, undocumented).
4. `CLAUDE.md` line 42: list the actual chunkers (`chunk_openapi`, `chunk_markdown`, `chunk_text`, plus `chunk_converted` from T3); line 55 chunk_type: `"endpoint" | "section" | "paragraph"`.
5. `prepare-transfer.sh:121`: `.md, .yaml, .json, .txt, .rst`.
6. `plans/rag-architecture-improvements.md`: lines 11, 97, 101 drop page fields; 47-48 replace PDF/DOCX bullets with one line noting removal in 7155f83 and the convert hook; 144 → "Reusable splitter for TXT/RST"; mark item 6 (BM25 tokenizer) and 7 (content-hash detection) as implemented 2026-10-08.

### T3 — chunkers.py: convert-to-markdown hook, CHUNKER_VERSION 2
Files: `chunkers.py`, `tests/test_chunkers.py`.
1. `CHUNKER_VERSION = 2` (forces one rebuild of existing stores, which also purges orphaned PDF/DOCX chunks).
2. Refactor `chunk_markdown(path)` into `_chunk_markdown_text(path, text)` + thin `chunk_markdown(path)` wrapper. Same output as today.
3. Env: `RAG_MCP_CONVERT_CMD` (shell template, `{input}` is replaced with `shlex.quote(str(path))`, must print Markdown to stdout; empty/unset = feature off) and `RAG_MCP_CONVERT_EXTS` (default `.pdf,.docx`, comma-separated, lowercased, leading dot added if missing). Read at import like the `_MD_*` settings.
4. `chunk_converted(path)`: `subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=RAG_MCP_CONVERT_TIMEOUT default 120, check=True)`, then `_chunk_markdown_text(path, stdout)`. Chunks keep `source=str(path)`, `doc_title=path.stem`, `chunk_type="section"`. On non-zero exit raise `RuntimeError(f"convert failed for {path.name}: {stderr[-500:]}")` so the server's per-file try/except records it in `failed_files`.
5. Register: when CMD is set, add each ext in CONVERT_EXTS to `_EXT_MAP` → `chunk_converted` and therefore to `SUPPORTED_EXTENSIONS`. Add `RAG_MCP_CONVERT_CMD` and `RAG_MCP_CONVERT_EXTS` to `current_chunk_config()` so changing them invalidates the manifest.
6. Tests: with `monkeypatch.setenv` + `importlib.reload(chunkers)`: CMD unset → `.pdf` not in SUPPORTED_EXTENSIONS; CMD=`printf '# Title\n\n<80+ chars>' # {input}` → `.pdf` is supported and `chunk_file(tmp.pdf)` yields a section chunk with `source` = pdf path; CMD=`false` → `chunk_file` raises RuntimeError. Restore module state after each test (reload with env cleared). Existing tests must still pass.

## Phase 2 (one agent, after Phase 1 merged)

### T4 — server.py: stale definition, fingerprint change detection, ignored files, status, reindex
Files: `server.py`, `store.py`, `dashboard.py`, `test_startup.py`, `tests/test_store.py`, `tests/test_dashboard.py`, `README.md` (tools + env tables only).
1. `_cleanup_stale_sources`: stale = `not Path(s).exists() or Path(s).suffix.lower() not in SUPPORTED_EXTENSIONS`. Use `store.delete_sources(stale_list)` once. Keep return type `list[str]`.
2. Change detection. Store side (`store.py`): `mtimes.json` values become `{"mtime": float, "size": int, "sha256": str}`; `_load` upgrades bare floats to `{"mtime": v}`; `source_mtime()` keeps returning the float (existing tests). New `source_fingerprint(source) -> dict | None` and `ingest(..., fingerprint: dict | None = None)` (when given, store it; `mtime=` keeps working and becomes `{"mtime": mtime}`). New `touch_source(source, fingerprint)` that updates the record and saves only `mtimes.json`. Server side: `_fingerprint(path) -> dict` with mtime+size, sha256 computed lazily. Rule: stored mtime and size equal → unchanged, no read. Else hash the file; if sha256 equals stored → `touch_source` (log "unchanged content, mtime updated") and skip re-embed. Else re-ingest. `# ponytail: hash only when mtime/size changed; full-hash-every-scan if someone rewrites files with preserved stat`.
3. `IngestResult.ignored_files: list[str] = []`: files under FILES_ROOT whose suffix is not supported (relative paths, sorted, cap at 200 with a trailing "... and N more" entry). Log a one-line summary at startup. Add `ignored_files` count to `dashboard_data()` and a card on the dashboard page (`dashboard.py`); the server stores the last result so the dashboard can read it (see 4).
4. Status: module-level `_last_scan: dict | None` set at the end of `_ingest_files_root` with `{"at": iso-8601 UTC, "files": n, "skipped": n, "failed": n, "removed": n, "ignored": n, "duration_s": float}`. Extend `StoreStatus` with `last_scan: dict | None`, `watch_interval: int`, `supported_extensions: list[str]`. Expose in `dashboard_data()` too.
5. `reindex(path: str) -> IngestResult` MCP tool: path relative to FILES_ROOT (or absolute under it); reject paths outside FILES_ROOT with a ValueError (trust boundary). Deletes the source via `store.delete_sources` then runs `_ingest_files_root()`. Document it in README tools table.
6. Scan loop: pass `rebuild_bm25=False` to `store.ingest` inside `_ingest_files_root`, call `store.rebuild_bm25()` in a `finally` after the loop (only if anything was ingested).
7. Tests: cleanup removes an existing file with unsupported suffix; same-tick rewrite (write twice without utime) re-ingests when size differs; touch without content change does not call `store.ingest` but updates the stored mtime; `ignored_files` lists a `.pdf` when the converter is off; `rag_status()` has `last_scan` after a scan; `reindex` on a path outside FILES_ROOT raises; `reindex` on a valid file re-ingests it; dashboard data has `ignored_files` and `last_scan`.

## Phase 3 (owner)
- `bash prepare-transfer.sh` to regenerate `transfer/` so `requirements.frozen.txt` matches `requirements.txt`; verify no PyMuPDF/python-docx/rank-bm25 in it.
- `uv venv && uv pip install --python .venv/bin/python -r requirements.txt pytest` so `uv run pytest tests/` from CLAUDE.md works.
- Full suite 5× to confirm no flake; commit in logical commits; final `/code-review`.

## Deviations recorded during implementation
- `requirements.txt` now pins `mcp[cli]>=1.0.0,<2`: a fresh install resolved mcp 2.x, which removed `mcp.server.fastmcp.FastMCP` and broke `server.py` on import. Found while recreating `.venv`.
- `chunk_converted` uses `check=False` plus an explicit return-code check so the `RuntimeError` carries stderr.
- `_ingest_lock` is an `RLock` so `reindex()` can hold it across its delete and the rescan.
- `reindex(path)` also accepts a directory and re-ingests every source under it.
- Pre-change stores re-embed once because their records lack size/hash; `CHUNKER_VERSION=2` forces that rebuild anyway.
- `prepare-transfer.sh` could not replace the root-owned `.venv-build` left by an earlier docker run; the bundle was rebuilt with `prepare-transfer-docker.sh` and ownership fixed afterwards.

## Post-implementation review fixes (same day)
- Converter runs as an argv list (`shlex.split`, no shell), with `stdin=DEVNULL` and UTF-8 decoding: a quoted `{input}` plus a hostile file name could otherwise reach a shell, and a converter reading stdin would eat the MCP stdio pipe.
- Skipped and failed files are fingerprinted too, so a broken converter is not re-run every watch tick. `reindex(path)` forces a retry. `RAGStore.known_sources()` makes those chunk-less sources visible to stale cleanup.
- `reindex` contains the path lexically (`normpath`, no `resolve`), rejects the root and missing paths, and clears fingerprints instead of deleting chunks, so search keeps serving until each file is replaced. `_ingest_lock` is a plain `Lock` again.
- Converter settings are no longer in the store manifest: changing the command does not wipe Markdown/OpenAPI chunks.
- `ignored_files` skips dot-paths.
- Not fixed: the pre-existing search snapshot race between `_chunks` and `_vectors`/`_bm25` (would need a single atomically swapped state tuple); `ingest(mtime=)`/`source_mtime()` kept for test compatibility.

## Performance follow-up (2026-10-08, measured on a Core Ultra 5 125U under WSL2)
- Embedding is hardware-bound here at 5-7 chunks/s; batch size, ONNX threads, fp32 vs int8 and process parallelism all made it equal or slower. Defaults kept. Measure again on the Docker host.
- `_SparseBM25` build vectorized (`np.unique` on term*n+doc pairs): rebuild at 20k chunks 5.2s → 2.0s (now split roughly evenly between `re.findall` tokenization and index construction), cold load 3.4s → 1.1s, search 17ms → 9ms.
- Deletes during a scan no longer rebuild BM25 per file (`rebuild_bm25=False` drops the index; search is vector-only until the single rebuild at scan end).
- Startup scan runs in a daemon thread: dashboard reachable 2s after start while embedding continues; SIGTERM exits in ~0.2s instead of after the scan.

## Deliberately not done
- Persisting the BM25 index to disk: rebuild on load is seconds for stores this size. Add when cold start is measured as a problem.
- Full content hashing on every scan: see T4.2 ceiling comment.
