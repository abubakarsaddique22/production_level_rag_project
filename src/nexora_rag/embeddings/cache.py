"""
Embedding cache — content-hash keyed, so re-running the embedder on
unchanged chunks makes zero new model calls.

Backed by a local SQLite database (sqlite3 is in the Python standard
library, so this adds no new dependency) at:
    data/processed/.cache/embeddings.db

Key:   (content_hash, embedding_model)
Value: the embedding vector, JSON-encoded.

Why key by model too: if you switch from bge-small to bge-m3, the old
vectors have a different dimensionality and meaning — they must never be
silently reused. Changing settings.embed_model automatically busts the
cache for every chunk, without deleting anything.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from ..core.logging import get_logger

log = get_logger(__name__)

DEFAULT_CACHE_PATH = Path("data/processed/.cache/embeddings.db")


class EmbeddingCache:
    """Simple get/set/flush cache in front of a SQLite table."""

    def __init__(self, cache_path: Path | None = None):
        self.cache_path = cache_path or DEFAULT_CACHE_PATH
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)

        self._conn = sqlite3.connect(self.cache_path)
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS embeddings (
                content_hash    TEXT NOT NULL,
                embedding_model TEXT NOT NULL,
                vector          TEXT NOT NULL,
                PRIMARY KEY (content_hash, embedding_model)
            )
            """
        )
        self._conn.commit()
        self._dirty = False

        log.info("embedding_cache_opened", extra={"cache_path": str(self.cache_path)})

    def get(self, content_hash: str, embedding_model: str) -> list[float] | None:
        """Return the cached vector, or None if this (hash, model) pair
        has never been embedded before."""
        row = self._conn.execute(
            "SELECT vector FROM embeddings WHERE content_hash = ? AND embedding_model = ?",
            (content_hash, embedding_model),
        ).fetchone()

        if row is None:
            return None

        try:
            return json.loads(row[0])
        except json.JSONDecodeError:
            log.warning("cache_corrupt_entry", extra={"content_hash": content_hash})
            return None

    def set(self, content_hash: str, embedding_model: str, vector: list[float]) -> None:
        """Store (or overwrite) the vector for this (hash, model) pair."""
        self._conn.execute(
            """
            INSERT INTO embeddings (content_hash, embedding_model, vector)
            VALUES (?, ?, ?)
            ON CONFLICT(content_hash, embedding_model)
            DO UPDATE SET vector = excluded.vector
            """,
            (content_hash, embedding_model, json.dumps(vector)),
        )
        self._dirty = True

    def flush(self) -> None:
        """Commit pending writes to disk. Safe to call often — it's a
        no-op when nothing changed since the last flush."""
        if self._dirty:
            self._conn.commit()
            self._dirty = False

    def stats(self) -> dict:
        total = self._conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
        return {"cache_path": str(self.cache_path), "total_cached_vectors": total}

    def close(self) -> None:
        self.flush()
        self._conn.close()

    def __enter__(self) -> "EmbeddingCache":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()