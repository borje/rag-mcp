"""Vector + BM25 hybrid store backed by numpy arrays and JSON metadata."""

import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional

import numpy as np
from rank_bm25 import BM25Okapi

STORE_DIR = Path(os.environ.get("RAG_MCP_DATA", Path.home() / ".local/share/rag-mcp"))
OPENAI_ENV_VARS = (
    "OPENAI_BASE_URL",
    "OPENAI_API_KEY",
    "OPENAI_EMBEDDINGS_PATH",
    "OPENAI_TIMEOUT",
)
OPENAI_CONFIGURED = any(name in os.environ for name in OPENAI_ENV_VARS)
OPENAI_DEFAULT_BASE_URL = "https://api.openai.com/v1"
LOCAL_DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
OPENAI_DEFAULT_MODEL = "text-embedding-3-small"
MODEL_NAME = os.environ.get(
    "RAG_MCP_MODEL", OPENAI_DEFAULT_MODEL if OPENAI_CONFIGURED else LOCAL_DEFAULT_MODEL
)
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_EMBEDDINGS_PATH = os.environ.get("OPENAI_EMBEDDINGS_PATH", "/embeddings")
OPENAI_TIMEOUT = float(os.environ.get("OPENAI_TIMEOUT", "60"))
DIMENSION_MISMATCH_ERROR = (
    "Embedding dimension mismatch. The selected embedding model differs from the "
    "stored vectors. Clear and re-ingest the store."
)


class RAGStore:
    def __init__(self):
        STORE_DIR.mkdir(parents=True, exist_ok=True)
        self._meta_path = STORE_DIR / "meta.json"
        self._vec_path = STORE_DIR / "vectors.npy"
        self._mtimes_path = STORE_DIR / "mtimes.json"
        self._model = None
        self._chunks: list[dict] = []
        self._vectors: Optional[np.ndarray] = None
        self._bm25: Optional[BM25Okapi] = None
        self._mtimes: dict[str, float] = {}
        self._load()

    def _load(self):
        if self._meta_path.exists():
            self._chunks = json.loads(self._meta_path.read_text(encoding="utf-8"))
        if self._vec_path.exists() and self._chunks:
            self._vectors = np.load(str(self._vec_path))
        if self._mtimes_path.exists():
            self._mtimes = json.loads(self._mtimes_path.read_text(encoding="utf-8"))
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
        self._mtimes_path.write_text(
            json.dumps(self._mtimes, ensure_ascii=False),
            encoding="utf-8",
        )

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

    def _using_openai(self) -> bool:
        return OPENAI_CONFIGURED or any(value for value in (OPENAI_BASE_URL, OPENAI_API_KEY))

    def _embed(self, texts: list[str]) -> np.ndarray:
        if self._using_openai():
            return self._embed_openai(texts)
        return np.array(list(self.model.embed(texts)), dtype=np.float32)

    def _embed_openai(self, texts: list[str]) -> np.ndarray:
        base_url = (OPENAI_BASE_URL or OPENAI_DEFAULT_BASE_URL).rstrip("/")
        if base_url == OPENAI_DEFAULT_BASE_URL and not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is required for the default OpenAI API URL")
        url = f"{base_url}/{OPENAI_EMBEDDINGS_PATH.lstrip('/')}"
        headers = {"Content-Type": "application/json"}
        if OPENAI_API_KEY:
            headers["Authorization"] = f"Bearer {OPENAI_API_KEY}"
        request = urllib.request.Request(
            url,
            data=json.dumps({"model": MODEL_NAME, "input": texts}).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=OPENAI_TIMEOUT) as response:
                raw = response.read()
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"OpenAI embeddings request failed: HTTP {exc.code}: {detail}"
            ) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"OpenAI embeddings request failed: {exc.reason}") from exc
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("OpenAI embeddings response was not valid JSON") from exc
        if "data" not in payload:
            raise ValueError("OpenAI embeddings response missing data")
        embeddings = []
        for item in sorted(payload["data"], key=lambda item: item.get("index", 0)):
            if "embedding" not in item:
                raise ValueError("OpenAI embeddings response missing embedding")
            embeddings.append(item["embedding"])
        return np.array(embeddings, dtype=np.float32)

    def _check_dimension(self, vectors: np.ndarray):
        if self._vectors is not None and vectors.shape[1] != self._vectors.shape[1]:
            raise ValueError(DIMENSION_MISMATCH_ERROR)

    def ingest(
        self, chunks: list[dict], mtime: float | None = None, batch_size: int = 64
    ) -> int:
        if not chunks:
            return 0
        total = 0
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            new_vecs = self._embed([c["body"] for c in batch])
            self._check_dimension(new_vecs)
            self._chunks.extend(batch)
            self._vectors = (
                new_vecs
                if self._vectors is None
                else np.vstack([self._vectors, new_vecs])
            )
            total += len(batch)
            self._save()
        if mtime is not None:
            for chunk in chunks:
                self._mtimes[chunk["source"]] = mtime
            self._save()
        self._rebuild_bm25()
        return total

    def source_mtime(self, source: str) -> float | None:
        return self._mtimes.get(source)

    def delete_source(self, source: str) -> int:
        if not self._chunks:
            return 0
        keep = [i for i, c in enumerate(self._chunks) if c["source"] != source]
        removed = len(self._chunks) - len(keep)
        if removed == 0:
            return 0
        self._chunks = [self._chunks[i] for i in keep]
        self._vectors = self._vectors[np.array(keep)] if keep else None
        self._mtimes.pop(source, None)
        self._rebuild_bm25()
        self._save()
        return removed

    def list_sources(self) -> list[str]:
        return sorted(set(c["source"] for c in self._chunks))

    def stats(self) -> dict:
        return {
            "total_chunks": len(self._chunks),
            "total_sources": len(self.list_sources()),
            "model": f"{'openai' if self._using_openai() else 'fastembed'}:{MODEL_NAME}",
            "store_dir": str(STORE_DIR),
        }

    def search(self, query: str, n: int = 8) -> list[dict]:
        if not self._chunks or self._vectors is None:
            return []
        n = min(n, len(self._chunks))
        pool = min(n * 4, len(self._chunks))

        # Cosine similarity (vector search)
        q = self._embed([query])[0]
        if q.shape[0] != self._vectors.shape[1]:
            raise ValueError(DIMENSION_MISMATCH_ERROR)
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
