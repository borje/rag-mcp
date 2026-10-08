"""Vector + BM25 hybrid store backed by numpy arrays and JSON metadata."""

import json
import os
import re
import sys
import threading
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

from chunkers import CHUNKER_VERSION, current_chunk_config

STORE_DIR = Path(os.environ.get("RAG_MCP_DATA", Path.home() / ".local/share/rag-mcp"))
MODEL_NAME = os.environ.get("RAG_MCP_MODEL", "BAAI/bge-small-en-v1.5")
INDEX_SCHEMA_VERSION = 2


def current_index_manifest() -> dict:
    return {
        "index_schema_version": INDEX_SCHEMA_VERSION,
        "chunker_version": CHUNKER_VERSION,
        "model": MODEL_NAME,
        "chunk_config": current_chunk_config(),
    }


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


class _SparseBM25:
    """BM25Okapi using flat numpy arrays instead of per-doc Python dicts.

    Replaces rank_bm25.BM25Okapi. Identical interface (get_scores), ~5-10x
    less memory because term-doc data lives in compact numpy arrays rather
    than a list of Counter dicts.
    """

    def __init__(self, corpus: list[list[str]], k1: float = 1.5, b: float = 0.75):
        n = len(corpus)
        self._n = n
        self._vocab: dict[str, int] = {}
        self._idf = np.zeros(0, dtype=np.float32)
        self._term_ptr = np.zeros(1, dtype=np.int32)
        self._doc_indices = np.zeros(0, dtype=np.int32)
        self._bm25_vals = np.zeros(0, dtype=np.float32)
        if n == 0:
            return

        # One Python pass assigns term ids; everything after is numpy.
        vocab = self._vocab
        doc_lens = np.fromiter((len(d) for d in corpus), dtype=np.int32, count=n)
        term_ids = np.fromiter(
            (vocab.setdefault(tok, len(vocab)) for doc in corpus for tok in doc),
            dtype=np.int64,
            count=int(doc_lens.sum()),
        )
        V = len(vocab)
        if V == 0:
            return
        doc_ids = np.repeat(np.arange(n, dtype=np.int64), doc_lens)

        # Unique (term, doc) pairs with counts = tf; sorted by term then doc,
        # which is exactly the CSC order the flat arrays need.
        pairs, tf = np.unique(term_ids * n + doc_ids, return_counts=True)
        term_of = pairs // n
        doc_of = (pairs % n).astype(np.int32)
        df = np.bincount(term_of, minlength=V)

        # IDF: log(1 + (n - df + 0.5) / (df + 0.5)) — always non-negative
        self._idf = np.log1p((n - df + 0.5) / (df + 0.5)).astype(np.float32)

        avgdl = float(doc_lens.mean())
        norm = k1 * (1.0 - b + b * doc_lens[doc_of] / avgdl)
        self._bm25_vals = (tf * (k1 + 1.0) / (tf + norm)).astype(np.float32)
        self._doc_indices = doc_of
        term_ptr = np.zeros(V + 1, dtype=np.int32)
        np.cumsum(df, out=term_ptr[1:])
        self._term_ptr = term_ptr

    def get_scores(self, query: list[str]) -> np.ndarray:
        scores = np.zeros(self._n, dtype=np.float32)
        for tok in query:
            t = self._vocab.get(tok)
            if t is None:
                continue
            idf = float(self._idf[t])
            s, e = int(self._term_ptr[t]), int(self._term_ptr[t + 1])
            scores[self._doc_indices[s:e]] += idf * self._bm25_vals[s:e]
        return scores


def _backfill_meta(chunk: dict) -> dict:
    """Fill metadata keys that pre-adjacent-chunk stores/callers lack."""
    chunk.setdefault("source_name", Path(chunk.get("source", "")).name)
    chunk.setdefault("section_path", chunk.get("title", ""))
    chunk.setdefault("chunk_index", 0)
    chunk.setdefault("chunk_total", 1)
    return chunk



def _normalize_scope_path(path: str | None) -> str | None:
    if path is None:
        return None
    parts = [p for p in str(path).replace("\\", "/").split("/") if p and p != "."]
    normalized = "/".join(parts)
    return normalized or None



def _chunk_relative_source(chunk: dict) -> str:
    relative_source = chunk.get("relative_source")
    if relative_source:
        return _normalize_scope_path(relative_source) or ""
    source = chunk.get("source", "")
    return _normalize_scope_path(source) or ""



def _chunk_in_scope(chunk: dict, scope: str | None) -> bool:
    if scope is None:
        return True
    relative_source = _chunk_relative_source(chunk)
    return relative_source == scope or relative_source.startswith(scope + "/")


def _expand_scopes(relative_source: str) -> list[str]:
    norm = _normalize_scope_path(relative_source)
    if not norm:
        return []
    dirs = norm.split("/")[:-1]  # drop filename
    return ["/".join(dirs[: i + 1]) for i in range(len(dirs))]


class RAGStore:
    def __init__(self):
        STORE_DIR.mkdir(parents=True, exist_ok=True)
        self._meta_path = STORE_DIR / "meta.json"
        self._vec_path = STORE_DIR / "vectors.npy"
        self._mtimes_path = STORE_DIR / "mtimes.json"
        self._bodies_path = STORE_DIR / "bodies.json"
        self._manifest_path = STORE_DIR / "manifest.json"
        self._model = None
        self._chunks: list[dict] = []
        self._bodies: list[str] = []
        self._vectors: Optional[np.ndarray] = None
        self._norms: Optional[np.ndarray] = None
        self._bm25: Optional[_SparseBM25] = None
        self._mtimes: dict[str, dict] = {}  # source -> {"mtime", "size", "sha256"}
        self.manifest_reset_reason: str | None = None
        self._write_lock = threading.Lock()
        self._load()

    def _persisted_paths(self) -> list[Path]:
        return [
            self._meta_path,
            self._vec_path,
            self._mtimes_path,
            self._bodies_path,
            self._manifest_path,
        ]

    def _has_persisted_store(self) -> bool:
        return any(path.exists() for path in self._persisted_paths() if path != self._manifest_path)

    def _load_manifest(self) -> dict | None:
        if not self._manifest_path.exists():
            return None
        return self._read_json(self._manifest_path, None)

    def _save_manifest(self) -> None:
        self._write_atomic(
            self._manifest_path,
            json.dumps(current_index_manifest(), ensure_ascii=False),
        )

    def _reset_persisted_store(self, reason: str) -> None:
        self.manifest_reset_reason = reason
        self._chunks = []
        self._bodies = []
        self._vectors = None
        self._bm25 = None
        self._mtimes = {}
        for path in self._persisted_paths():
            if path.exists():
                path.unlink()

    @staticmethod
    def _write_atomic(path: Path, text: str) -> None:
        tmp = path.parent / (path.name + ".tmp")
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)

    @staticmethod
    def _read_json(path: Path, default):
        if not path.exists():
            return default
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            print(f"[store] could not read {path.name}: {e}", file=sys.stderr)
            return default


    def _load(self):
        saved_manifest = self._load_manifest()
        expected_manifest = current_index_manifest()
        if saved_manifest != expected_manifest:
            if saved_manifest is not None or self._has_persisted_store():
                self._reset_persisted_store(
                    "index manifest mismatch; clearing persisted store so it can be rebuilt"
                )
            self._save_manifest()
        self._chunks = self._read_json(self._meta_path, [])
        # Migrate old format: meta.json had 'body' in each chunk dict
        if self._chunks and not self._bodies_path.exists() and "body" in self._chunks[0]:
            self._bodies = [c.pop("body", "") for c in self._chunks]
            self._write_atomic(
                self._bodies_path, json.dumps(self._bodies, ensure_ascii=False)
            )
            self._write_atomic(
                self._meta_path, json.dumps(self._chunks, ensure_ascii=False)
            )
        elif self._chunks:
            self._bodies = self._read_json(self._bodies_path, [])
        else:
            self._bodies = []
        for c in self._chunks:
            _backfill_meta(c)
        if self._vec_path.exists() and self._chunks:
            try:
                self._vectors = np.load(str(self._vec_path), mmap_mode="r")
            except (OSError, ValueError) as e:
                print(f"[store] could not read vectors.npy: {e}", file=sys.stderr)
                self._vectors = None
        if self._mtimes_path.exists():
            self._mtimes = {
                k: v if isinstance(v, dict) else {"mtime": v}
                for k, v in self._read_json(self._mtimes_path, {}).items()
            }
        self._validate_alignment()
        self._rebuild_bm25()

    def _validate_alignment(self) -> None:
        """Trim chunks/bodies/vectors to a consistent common length.

        A crash between the individual file writes (or an external partial
        wipe) can leave the three artifacts at different lengths; positional
        misalignment silently attributes the wrong body/vector to a chunk.
        """
        sizes = [len(self._chunks), len(self._bodies)]
        vector_size: int | None = None
        if self._vectors is not None:
            vector_size = int(self._vectors.shape[0])
        elif self._chunks or self._bodies:
            # Missing/corrupt vectors.npy with surviving metadata must be
            # treated as an inconsistent store so the next ingest re-embeds
            # the affected sources instead of skipping them forever.
            vector_size = 0
        if vector_size is not None:
            sizes.append(vector_size)
        m = min(sizes)
        if any(s != m for s in sizes):
            print(
                f"[store] inconsistent store files "
                f"(chunks={len(self._chunks)}, bodies={len(self._bodies)}, "
                f"vectors={vector_size if vector_size is not None else 'absent'}); "
                f"truncating to {m} — re-run ingest to restore",
                file=sys.stderr,
            )
            self._chunks = self._chunks[:m]
            self._bodies = self._bodies[:m]
            if self._vectors is not None:
                self._vectors = np.array(self._vectors[:m]) if m else None
            self._norms = None
            # Drop all mtimes so the next ingest scan re-ingests the trimmed
            # sources instead of skipping them as "unchanged".
            self._mtimes = {}
            self._save()

    def _load_bodies(self) -> list[str]:
        return self._read_json(self._bodies_path, [])

    def load_bodies(self) -> list[str]:
        """Public accessor for chunk bodies (positionally aligned with _chunks)."""
        return self._bodies

    def _save(self) -> None:
        self._save_manifest()
        self._write_atomic(
            self._meta_path, json.dumps(self._chunks, ensure_ascii=False)
        )
        self._write_atomic(
            self._bodies_path, json.dumps(self._bodies, ensure_ascii=False)
        )
        if self._vectors is not None:
            vecs = (
                np.array(self._vectors)
                if isinstance(self._vectors, np.memmap)
                else self._vectors
            )
            tmp = self._vec_path.parent / "vectors.tmp.npy"
            np.save(str(tmp), vecs)
            os.replace(tmp, self._vec_path)
        elif self._vec_path.exists():
            self._vec_path.unlink()
        self._save_mtimes()

    def _save_mtimes(self) -> None:
        self._write_atomic(
            self._mtimes_path, json.dumps(self._mtimes, ensure_ascii=False)
        )

    def _reload_vectors_mmapped(self) -> None:
        if self._vec_path.exists() and self._chunks:
            self._vectors = np.load(str(self._vec_path), mmap_mode="r")

    def _rebuild_bm25(self):
        if self._bodies:
            self._bm25 = _SparseBM25([_tokenize(b) for b in self._bodies])
        else:
            self._bm25 = None

    @property
    def model(self):
        if self._model is None:
            from fastembed import TextEmbedding

            self._model = TextEmbedding(MODEL_NAME)
        return self._model

    def _embed(self, texts: list[str]) -> np.ndarray:
        return np.array(list(self.model.embed(texts)), dtype=np.float32)

    @staticmethod
    def _malloc_trim() -> None:
        """Return fragmented heap pages to the OS (Linux only, no-op elsewhere)."""
        try:
            import ctypes
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except Exception:
            pass

    def ingest(
        self,
        chunks: list[dict],
        mtime: float | None = None,
        batch_size: int | None = None,
        log=None,
        rebuild_bm25: bool = True,
        fingerprint: dict | None = None,
    ) -> int:
        if not chunks:
            return 0
        if batch_size is None:
            batch_size = int(os.environ.get("RAG_MCP_EMBED_BATCH_SIZE", "16"))
        with self._write_lock:
            new_meta: list[dict] = []
            new_bodies: list[str] = []
            vec_blocks: list[np.ndarray] = []
            total_batches = (len(chunks) + batch_size - 1) // batch_size
            for batch_number, i in enumerate(range(0, len(chunks), batch_size), 1):
                batch = chunks[i : i + batch_size]
                if log:
                    log(
                        f"embedding batch {batch_number}/{total_batches} "
                        f"({len(batch)} chunks)"
                    )
                vec_blocks.append(self._embed([c["body"] for c in batch]))
                new_bodies.extend(c["body"] for c in batch)
                new_meta.extend(
                    _backfill_meta({k: v for k, v in c.items() if k != "body"})
                    for c in batch
                )
            new_vecs = np.vstack(vec_blocks)
            # Rebind (don't mutate in place) so concurrent readers see either
            # the old or the new state, never a partially-extended one.
            self._chunks = self._chunks + new_meta
            self._bodies = self._bodies + new_bodies
            self._vectors = (
                new_vecs
                if self._vectors is None
                else np.vstack([np.array(self._vectors), new_vecs])
            )
            self._norms = None
            if fingerprint is None and mtime is not None:
                fingerprint = {"mtime": mtime}
            if fingerprint is not None:
                for chunk in chunks:
                    self._mtimes[chunk["source"]] = dict(fingerprint)
            self._save()
            if log:
                log(f"saved {len(chunks)} chunks")
            if rebuild_bm25:
                self._rebuild_bm25()
            self._reload_vectors_mmapped()
        self._malloc_trim()
        return len(chunks)

    def source_mtime(self, source: str) -> float | None:
        return (self._mtimes.get(source) or {}).get("mtime")

    def source_fingerprint(self, source: str) -> dict | None:
        return self._mtimes.get(source)

    def touch_source(self, source: str, fingerprint: dict) -> None:
        """Record a new fingerprint for unchanged content; rewrites only mtimes.json."""
        with self._write_lock:
            self._mtimes[source] = dict(fingerprint)
            self._save_mtimes()

    def known_sources(self) -> set[str]:
        """Sources with chunks or a fingerprint (skipped/failed files have only the latter)."""
        return set(self.list_sources()) | set(self._mtimes)

    def rebuild_bm25(self) -> None:
        with self._write_lock:
            self._rebuild_bm25()

    def delete_source(self, source: str, rebuild_bm25: bool = True) -> int:
        return self.delete_sources([source], rebuild_bm25=rebuild_bm25).get(source, 0)

    def delete_sources(
        self, sources: Iterable[str], rebuild_bm25: bool = True
    ) -> dict[str, int]:
        sources = set(sources)
        removed = dict.fromkeys(sources, 0)
        with self._write_lock:
            keep = []
            for i, c in enumerate(self._chunks):
                if c["source"] in sources:
                    removed[c["source"]] += 1
                else:
                    keep.append(i)
            if len(keep) == len(self._chunks):
                return removed
            self._chunks = [self._chunks[i] for i in keep]
            self._bodies = [
                self._bodies[i] if i < len(self._bodies) else "" for i in keep
            ]
            self._vectors = (
                np.array(self._vectors)[np.array(keep)] if keep else None
            )
            self._norms = None
            for source in sources:
                self._mtimes.pop(source, None)
            self._save()
            if rebuild_bm25:
                self._rebuild_bm25()
            else:
                # Positions shifted, so the old index would misattribute scores:
                # drop it (search falls back to vector-only) until rebuild_bm25().
                self._bm25 = None
            self._reload_vectors_mmapped()
        return removed

    def list_sources(self) -> list[str]:
        return sorted(set(c["source"] for c in self._chunks))

    def list_scopes(self) -> list[dict]:
        scope_sources: dict[str, set[str]] = {}
        for c in self._chunks:
            relative_source = c.get("relative_source")
            if not relative_source:
                continue
            for scope in _expand_scopes(relative_source):
                scope_sources.setdefault(scope, set()).add(c["source"])
        return [
            {"scope": scope, "n_docs": len(sources)}
            for scope, sources in sorted(scope_sources.items())
        ]

    def stats(self) -> dict:
        return {
            "total_chunks": len(self._chunks),
            "total_sources": len(self.list_sources()),
            "model": MODEL_NAME,
            "store_dir": str(STORE_DIR),
        }

    def search(self, query: str, n: int = 8, scope: str | None = None) -> list[dict]:
        # Snapshot references so a concurrent ingest/delete (which rebinds,
        # never mutates) cannot change them mid-search.
        chunks = self._chunks
        vectors = self._vectors
        bodies = self._bodies
        bm25 = self._bm25
        if n <= 0 or not chunks or vectors is None:
            return []

        scope = _normalize_scope_path(scope)
        if scope is None:
            candidate_indices = None
            n_candidates = len(chunks)
        else:
            candidate_indices = np.array(
                [i for i, chunk in enumerate(chunks) if _chunk_in_scope(chunk, scope)],
                dtype=np.int32,
            )
            if candidate_indices.size == 0:
                return []
            n_candidates = int(candidate_indices.size)

        n = min(n, n_candidates)
        pool = min(n * 4, n_candidates)
        adjacent = max(0, int(os.environ.get("RAG_MCP_ADJACENT_CHUNKS", "1")))

        # Cosine similarity (vector search)
        q = self._embed([query])[0]
        if self._norms is None or len(self._norms) != vectors.shape[0]:
            self._norms = np.linalg.norm(vectors, axis=1)
        if candidate_indices is None:
            dot = vectors @ q
            norms = self._norms * np.linalg.norm(q)
        else:
            dot = vectors[candidate_indices] @ q
            norms = self._norms[candidate_indices] * np.linalg.norm(q)
        norms = np.where(norms < 1e-10, 1e-10, norms)
        cos_sims = dot / norms
        top_local = np.argsort(-cos_sims)[:pool]
        vec_ranks = top_local if candidate_indices is None else candidate_indices[top_local]

        # BM25 keyword search
        query_terms = _tokenize(query)
        if bm25 is not None:
            bm25_scores = bm25.get_scores(query_terms)
            # ponytail: BM25 is eventually consistent during a scan; new chunks rank by vector only until rebuild_bm25()
            if len(bm25_scores) < len(chunks):
                bm25_scores = np.pad(bm25_scores, (0, len(chunks) - len(bm25_scores)))
            if candidate_indices is not None:
                bm25_scores = bm25_scores[candidate_indices]
        else:
            bm25_scores = np.zeros(n_candidates, dtype=np.float32)
        top_local = np.argsort(-bm25_scores)[:pool]
        bm25_ranks = top_local if candidate_indices is None else candidate_indices[top_local]

        # Reciprocal Rank Fusion (k=60)
        k = 60
        rrf: dict[int, float] = {}
        for rank, idx in enumerate(vec_ranks):
            if int(idx) < len(chunks):
                rrf[int(idx)] = rrf.get(int(idx), 0.0) + 1.0 / (k + rank + 1)
        for rank, idx in enumerate(bm25_ranks):
            if int(idx) < len(chunks):
                rrf[int(idx)] = rrf.get(int(idx), 0.0) + 1.0 / (k + rank + 1)

        ranked = sorted(rrf, key=lambda i: -rrf[i])
        match_types = {idx: "hit" for idx in ranked}
        allowed = None if candidate_indices is None else set(int(i) for i in candidate_indices)

        def _neighbor(idx: int, offset: int) -> int | None:
            """Adjacent chunk by store position, validated by metadata.

            Chunks of one section are stored contiguously, so position-based
            lookup cannot confuse two same-titled sections the way a
            (source, section_path, chunk_index) key can.
            """
            j = idx + offset
            if 0 <= j < len(chunks):
                a, b = chunks[idx], chunks[j]
                if (
                    (allowed is None or j in allowed)
                    and b.get("source") == a.get("source")
                    and b.get("section_path") == a.get("section_path")
                    and b.get("chunk_index") == a.get("chunk_index", 0) + offset
                ):
                    return j
            return None

        if adjacent:
            expanded: list[int] = []
            seen: set[int] = set()
            for idx in ranked:
                candidates = [idx]
                for offset in range(1, adjacent + 1):
                    candidates.append(_neighbor(idx, -offset))
                    candidates.append(_neighbor(idx, offset))
                for candidate in candidates:
                    if candidate is None or candidate in seen or (allowed is not None and candidate not in allowed):
                        continue
                    expanded.append(candidate)
                    if candidate != idx:
                        match_types[candidate] = "adjacent"
                    seen.add(candidate)
                    if len(expanded) == n:
                        break
                if len(expanded) == n:
                    break
            top = expanded
        else:
            top = ranked[:n]

        return [
            {
                **chunks[i],
                "body": bodies[i] if i < len(bodies) else "",
                "score": float(rrf.get(i, 0.0)),
                "match_type": match_types[i],
            }
            for i in top
        ]
