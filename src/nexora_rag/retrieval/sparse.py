"""
BM25 (sparse / keyword) search over all chunks.

Dense search (vector_store.py) is good at "meaning" but can miss exact
IDs, numbers and rare terms (e.g. "NX-FIN-002", "20 days") because an
embedding model doesn't treat these tokens specially. BM25 scores chunks
by literal word overlap, so it catches exactly what dense search misses --
this is why Step M combines both (see hybrid.py).

The whole corpus is small, so every chunks.json under data/processed/ is
simply loaded into memory and one BM25 index is built -- no persistence
needed, it rebuilds in well under a second.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from rank_bm25 import BM25Okapi

from ..core.logging import get_logger

log = get_logger(__name__)

PROCESSED_DIR = Path("data/processed")
CHUNK_FILENAME = "chunks.json"

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _load_all_chunks(processed_dir: Path = PROCESSED_DIR) -> list[dict]:
    chunks: list[dict] = []
    for chunk_path in sorted(processed_dir.glob(f"*/{CHUNK_FILENAME}")):
        with chunk_path.open("r", encoding="utf-8") as f:
            chunks.extend(json.load(f))
    return chunks


class SparseIndex:
    """BM25 index over every chunk's content, keyed by chunk_id."""

    def __init__(self, processed_dir: Path = PROCESSED_DIR):
        self.processed_dir = processed_dir
        self.chunk_ids: list[str] = []
        self.chunks_by_id: dict[str, dict] = {}
        self._bm25: BM25Okapi | None = None
        self._build()

    def _build(self) -> None:
        chunks = _load_all_chunks(self.processed_dir)
        if not chunks:
            log.warning(
                "no_chunks_found_for_sparse_index",
                extra={"processed_dir": str(self.processed_dir)},
            )
            return

        tokenized_corpus = []
        for chunk in chunks:
            chunk_id = chunk["chunk_id"]
            self.chunk_ids.append(chunk_id)
            self.chunks_by_id[chunk_id] = chunk
            tokenized_corpus.append(_tokenize(chunk["content"]))

        self._bm25 = BM25Okapi(tokenized_corpus)
        log.info("sparse_index_built", extra={"n_chunks": len(self.chunk_ids)})

    def search(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        """Returns [(chunk_id, bm25_score), ...] sorted best-first."""
        if self._bm25 is None:
            return []

        tokenized_query = _tokenize(query)
        scores = self._bm25.get_scores(tokenized_query)

        ranked = sorted(
            zip(self.chunk_ids, scores),
            key=lambda pair: pair[1],
            reverse=True,
        )
        return ranked[:top_k]

    def get_chunk(self, chunk_id: str) -> dict | None:
        return self.chunks_by_id.get(chunk_id)


if __name__ == "__main__":
    index = SparseIndex()
    print(f"Built BM25 index over {len(index.chunk_ids)} chunks.\n")

    test_query = "expense claim submission days"
    results = index.search(test_query, top_k=5)
    print(f"Query: {test_query!r}\n")
    for chunk_id, score in results:
        chunk = index.get_chunk(chunk_id)
        preview = chunk["content"][:80] if chunk else ""
        print(f"  {score:6.3f}  {chunk_id}  -> {preview}")