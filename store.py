"""Vector + BM25 hybrid store backed by numpy arrays and JSON metadata."""
import json
import os
from pathlib import Path
from typing import Optional

import numpy as np
from rank_bm25 import BM25Okapi

STORE_DIR = Path(os.environ.get("RAG_MCP_DATA", Path.home() / ".local/share/rag-mcp"))
MODEL_NAME = os.environ.get("RAG_MCP_MODEL", "BAAI/bge-small-en-v1.5")


class RAGStore:
    def __init__(self):
        STORE_DIR.mkdir(parents=True, exist_ok=True)
        self._meta_path = STORE_DIR / "meta.json"
        self._vec_path = STORE_DIR / "vectors.npy"
        self._model = None
        self._chunks: list[dict] = []
        self._vectors: Optional[np.ndarray] = None
        self._bm25: Optional[BM25Okapi] = None
        self._load()

    def _load(self):
        if self._meta_path.exists():
            self._chunks = json.loads(self._meta_path.read_text(encoding="utf-8"))
        if self._vec_path.exists() and self._chunks:
            self._vectors = np.load(str(self._vec_path))
        self._rebuild_bm25()

    def _save(self):
        self._meta_path.write_text(
            json.dumps(self._chunks, ensure_ascii=False, indent=None),
            encoding="utf-8",
        )
        if self._vectors is not None:
            np.save(str(self._vec_path), self._vectors)
        elif self._vec_path.exists():
            self._vec_path.unlink()

    def _rebuild_bm25(self):
        if self._chunks:
            self._bm25 = BM25Okapi([c["body"].lower().split() for c in self._chunks])
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

    def ingest(self, chunks: list[dict], batch_size: int = 64) -> int:
        if not chunks:
            return 0
        total = 0
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            new_vecs = self._embed([c["body"] for c in batch])
            self._chunks.extend(batch)
            self._vectors = new_vecs if self._vectors is None else np.vstack([self._vectors, new_vecs])
            total += len(batch)
            self._save()
        self._rebuild_bm25()
        return total

    def delete_source(self, source: str) -> int:
        if not self._chunks:
            return 0
        keep = [i for i, c in enumerate(self._chunks) if c["source"] != source]
        removed = len(self._chunks) - len(keep)
        if removed == 0:
            return 0
        self._chunks = [self._chunks[i] for i in keep]
        self._vectors = self._vectors[np.array(keep)] if keep else None
        self._rebuild_bm25()
        self._save()
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

        top = sorted(rrf, key=lambda i: -rrf[i])[:n]
        return [{**self._chunks[i], "score": float(rrf[i])} for i in top]
