"""Vector + BM25 hybrid store backed by numpy arrays and JSON metadata."""

import json
import os
from pathlib import Path
from typing import Optional

import numpy as np

from chunkers import CHUNKER_VERSION, current_chunk_config

STORE_DIR = Path(os.environ.get("RAG_MCP_DATA", Path.home() / ".local/share/rag-mcp"))
MODEL_NAME = os.environ.get("RAG_MCP_MODEL", "BAAI/bge-small-en-v1.5")
INDEX_SCHEMA_VERSION = 1


def current_index_manifest() -> dict:
    return {
        "index_schema_version": INDEX_SCHEMA_VERSION,
        "chunker_version": CHUNKER_VERSION,
        "model": MODEL_NAME,
        "chunk_config": current_chunk_config(),
    }


class _SparseBM25:
    """BM25Okapi using flat numpy arrays instead of per-doc Python dicts.

    Replaces rank_bm25.BM25Okapi. Identical interface (get_scores), ~5-10x
    less memory because term-doc data lives in compact numpy arrays rather
    than a list of Counter dicts.
    """

    def __init__(self, corpus: list[list[str]], k1: float = 1.5, b: float = 0.75):
        n = len(corpus)
        self._n = n
        if n == 0:
            return

        # Build vocabulary and per-term posting lists {term_id: [(doc, tf)]}
        vocab: dict[str, int] = {}
        posting: dict[int, list] = {}
        doc_lens = np.zeros(n, dtype=np.int32)

        for d, doc in enumerate(corpus):
            doc_lens[d] = len(doc)
            counts: dict[str, int] = {}
            for tok in doc:
                counts[tok] = counts.get(tok, 0) + 1
            for tok, cnt in counts.items():
                if tok not in vocab:
                    t = len(vocab)
                    vocab[tok] = t
                else:
                    t = vocab[tok]
                if t not in posting:
                    posting[t] = []
                posting[t].append((d, cnt))

        self._vocab = vocab
        avgdl = float(doc_lens.mean())
        V = len(vocab)

        # IDF: log(1 + (n - df + 0.5) / (df + 0.5)) — always non-negative
        idf = np.zeros(V, dtype=np.float32)
        for t, posts in posting.items():
            df = len(posts)
            idf[t] = np.log1p((n - df + 0.5) / (df + 0.5))
        self._idf = idf

        # Precompute BM25 term-doc scores in CSC-like flat arrays.
        # _term_ptr[t]:_term_ptr[t+1] → slice of _doc_indices / _bm25_vals for term t.
        term_ptr = np.zeros(V + 1, dtype=np.int32)
        for t in range(V):
            term_ptr[t + 1] = term_ptr[t] + len(posting.get(t, []))
        nnz = int(term_ptr[-1])

        doc_indices = np.zeros(nnz, dtype=np.int32)
        bm25_vals = np.zeros(nnz, dtype=np.float32)

        for t in range(V):
            start = int(term_ptr[t])
            for i, (d, tf) in enumerate(posting.get(t, [])):
                norm = k1 * (1.0 - b + b * float(doc_lens[d]) / avgdl)
                doc_indices[start + i] = d
                bm25_vals[start + i] = tf * (k1 + 1.0) / (tf + norm)

        self._term_ptr = term_ptr
        self._doc_indices = doc_indices
        self._bm25_vals = bm25_vals

    def get_scores(self, query: list[str]) -> np.ndarray:
        scores = np.zeros(self._n, dtype=np.float32)
        for tok in query:
            t = self._vocab.get(tok)
            if t is None:
                continue
            idf = float(self._idf[t])
            if idf <= 0:
                continue
            s, e = int(self._term_ptr[t]), int(self._term_ptr[t + 1])
            if s == e:
                continue
            scores[self._doc_indices[s:e]] += idf * self._bm25_vals[s:e]
        return scores


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
        self._vectors: Optional[np.ndarray] = None
        self._bm25: Optional[_SparseBM25] = None
        self._mtimes: dict[str, float] = {}
        self.manifest_reset_reason: str | None = None
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
        return json.loads(self._manifest_path.read_text(encoding="utf-8"))

    def _save_manifest(self) -> None:
        self._manifest_path.write_text(
            json.dumps(current_index_manifest(), ensure_ascii=False),
            encoding="utf-8",
        )

    def _reset_persisted_store(self, reason: str) -> None:
        self.manifest_reset_reason = reason
        self._chunks = []
        self._vectors = None
        self._bm25 = None
        self._mtimes = {}
        for path in self._persisted_paths():
            if path.exists():
                path.unlink()

    def _load(self):
        saved_manifest = self._load_manifest()
        expected_manifest = current_index_manifest()
        if saved_manifest != expected_manifest:
            if saved_manifest is not None or self._has_persisted_store():
                self._reset_persisted_store(
                    "index manifest mismatch; clearing persisted store so it can be rebuilt"
                )
            self._save_manifest()
        if self._meta_path.exists():
            self._chunks = json.loads(self._meta_path.read_text(encoding="utf-8"))
        # Migrate old format: meta.json had 'body' in each chunk dict
        if self._chunks and not self._bodies_path.exists() and "body" in self._chunks[0]:
            bodies = [c.pop("body") for c in self._chunks]
            self._bodies_path.write_text(
                json.dumps(bodies, ensure_ascii=False), encoding="utf-8"
            )
            self._meta_path.write_text(
                json.dumps(self._chunks, ensure_ascii=False), encoding="utf-8"
            )
        if self._vec_path.exists() and self._chunks:
            self._vectors = np.load(str(self._vec_path), mmap_mode="r")
        if self._mtimes_path.exists():
            self._mtimes = json.loads(self._mtimes_path.read_text(encoding="utf-8"))
        self._rebuild_bm25()

    def _load_bodies(self) -> list[str]:
        if self._bodies_path.exists():
            return json.loads(self._bodies_path.read_text(encoding="utf-8"))
        return []

    def load_bodies(self) -> list[str]:
        """Public accessor for chunk bodies (positionally aligned with _chunks)."""
        return self._load_bodies()

    def _save(self, bodies: list[str]) -> None:
        self._save_manifest()
        self._meta_path.write_text(
            json.dumps(self._chunks, ensure_ascii=False, indent=None),
            encoding="utf-8",
        )
        self._bodies_path.write_text(
            json.dumps(bodies, ensure_ascii=False, indent=None),
            encoding="utf-8",
        )
        if self._vectors is not None:
            vecs = (
                np.array(self._vectors)
                if isinstance(self._vectors, np.memmap)
                else self._vectors
            )
            np.save(str(self._vec_path), vecs)
        elif self._vec_path.exists():
            self._vec_path.unlink()
        self._mtimes_path.write_text(
            json.dumps(self._mtimes, ensure_ascii=False),
            encoding="utf-8",
        )

    def _save_mtimes(self) -> None:
        self._save_manifest()
        self._mtimes_path.write_text(
            json.dumps(self._mtimes, ensure_ascii=False),
            encoding="utf-8",
        )

    def _reload_vectors_mmapped(self) -> None:
        if self._vec_path.exists() and self._chunks:
            self._vectors = np.load(str(self._vec_path), mmap_mode="r")

    def _rebuild_bm25(self):
        if self._chunks:
            bodies = self._load_bodies()
            self._bm25 = _SparseBM25([b.lower().split() for b in bodies])
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
        batch_size: int = int(os.environ.get("RAG_MCP_EMBED_BATCH_SIZE", "16")),
        log=None,
    ) -> int:
        if not chunks:
            return 0
        # Release mmap before modifying vectors in-place
        if isinstance(self._vectors, np.memmap):
            self._vectors = np.array(self._vectors)
        existing_bodies = self._load_bodies()
        total = 0
        total_batches = (len(chunks) + batch_size - 1) // batch_size
        for batch_number, i in enumerate(range(0, len(chunks), batch_size), 1):
            batch = chunks[i : i + batch_size]
            if log:
                log(f"embedding batch {batch_number}/{total_batches} ({len(batch)} chunks)")
            new_vecs = self._embed([c["body"] for c in batch])
            batch_bodies = [c["body"] for c in batch]
            batch_meta = [{k: v for k, v in c.items() if k != "body"} for c in batch]
            self._chunks.extend(batch_meta)
            existing_bodies.extend(batch_bodies)
            self._vectors = (
                new_vecs
                if self._vectors is None
                else np.vstack([self._vectors, new_vecs])
            )
            total += len(batch)
            self._save(bodies=existing_bodies)
            if log:
                log(
                    f"saved batch {batch_number}/{total_batches} "
                    f"({total}/{len(chunks)} chunks)"
                )
        if mtime is not None:
            for chunk in chunks:
                self._mtimes[chunk["source"]] = mtime
            self._save_mtimes()
        self._rebuild_bm25()
        self._reload_vectors_mmapped()
        self._malloc_trim()
        return total

    def source_mtime(self, source: str) -> float | None:
        return self._mtimes.get(source)

    def delete_source(self, source: str) -> int:
        if not self._chunks:
            return 0
        bodies = self._load_bodies()
        keep = [i for i, c in enumerate(self._chunks) if c["source"] != source]
        removed = len(self._chunks) - len(keep)
        if removed == 0:
            return 0
        self._chunks = [self._chunks[i] for i in keep]
        new_bodies = [bodies[i] for i in keep] if bodies else []
        self._vectors = self._vectors[np.array(keep)] if keep else None
        self._mtimes.pop(source, None)
        self._rebuild_bm25()
        self._save(bodies=new_bodies)
        self._reload_vectors_mmapped()
        return removed

    def list_sources(self) -> list[str]:
        return sorted(set(c["source"] for c in self._chunks))

    def stats(self) -> dict:
        return {
            "total_chunks": len(self._chunks),
            "total_sources": len(self.list_sources()),
            "model": MODEL_NAME,
            "store_dir": str(STORE_DIR),
        }

    def search(self, query: str, n: int = 8) -> list[dict]:
        if not self._chunks or self._vectors is None:
            return []
        n = min(n, len(self._chunks))
        pool = min(n * 4, len(self._chunks))
        adjacent = max(0, int(os.environ.get("RAG_MCP_ADJACENT_CHUNKS", "1")))

        # Cosine similarity (vector search)
        q = self._embed([query])[0]
        dot = self._vectors @ q
        norms = np.linalg.norm(self._vectors, axis=1) * np.linalg.norm(q)
        norms = np.where(norms < 1e-10, 1e-10, norms)
        cos_sims = dot / norms
        vec_ranks = np.argsort(-cos_sims)[:pool]

        # BM25 keyword search
        bm25_scores = self._bm25.get_scores(query.lower().split())
        bm25_ranks = np.argsort(-bm25_scores)[:pool]

        # Reciprocal Rank Fusion (k=60)
        k = 60
        rrf: dict[int, float] = {}
        for rank, idx in enumerate(vec_ranks):
            rrf[int(idx)] = rrf.get(int(idx), 0.0) + 1.0 / (k + rank + 1)
        for rank, idx in enumerate(bm25_ranks):
            rrf[int(idx)] = rrf.get(int(idx), 0.0) + 1.0 / (k + rank + 1)

        ranked = sorted(rrf, key=lambda i: -rrf[i])
        match_types = {idx: "hit" for idx in ranked}
        if adjacent:
            by_location = {
                (c["source"], c["section_path"], c["chunk_index"]): i
                for i, c in enumerate(self._chunks)
            }
            expanded: list[int] = []
            seen: set[int] = set()
            for idx in ranked:
                chunk = self._chunks[idx]
                candidates = [idx]
                for offset in range(1, adjacent + 1):
                    candidates.append(
                        by_location.get(
                            (
                                chunk["source"],
                                chunk["section_path"],
                                chunk["chunk_index"] - offset,
                            )
                        )
                    )
                    candidates.append(
                        by_location.get(
                            (
                                chunk["source"],
                                chunk["section_path"],
                                chunk["chunk_index"] + offset,
                            )
                        )
                    )
                for candidate in candidates:
                    if candidate is None or candidate in seen:
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

        bodies = self._load_bodies()
        return [
            {
                **self._chunks[i],
                "body": bodies[i] if i < len(bodies) else "",
                "score": float(rrf.get(i, 0.0)),
                "match_type": match_types[i],
            }
            for i in top
        ]
