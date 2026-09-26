"""
Embeds RAG chunks into dense vectors using a local sentence-transformers model.

Reads:  data/processed/<Document Folder>/chunk.json
Writes: data/processed/<Document Folder>/embeddings.json

Idempotency: re-running this on unchanged chunks does NOT call the model
again — every chunk's content_hash is checked against embeddings/cache.py
before encoding.
"""

from __future__ import annotations

import json
from pathlib import Path

from sentence_transformers import SentenceTransformer

from ..core.config import settings
from ..core.exceptions import EmbeddingError
from ..core.logging import get_logger
from ..embeddings.cache import EmbeddingCache

log = get_logger(__name__)

PROCESSED_DIR = Path("data/processed")
CHUNK_FILENAME = "chunks.json"
EMBEDDINGS_FILENAME = "embeddings.json"


class Embedder:
    def __init__(self, model_name: str | None = None, batch_size: int | None = None):
        self.model_name = model_name or settings.embed_model
        self.batch_size = batch_size or settings.embed_batch_size

        log.info("loading_embedding_model", extra={"model": self.model_name})
        try:
            self._model = SentenceTransformer(self.model_name)
        except Exception as exc:  # noqa: BLE001
            raise EmbeddingError(
                f"Could not load embedding model '{self.model_name}': {exc}"
            ) from exc

        self.embedding_dim = self._model.get_embedding_dimension()

    def encode(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        try:
            vectors = self._model.encode(
                texts,
                batch_size=self.batch_size,
                show_progress_bar=False,
                normalize_embeddings=True,
                convert_to_numpy=True,
            )
        except Exception as exc:  # noqa: BLE001
            raise EmbeddingError(f"Embedding call failed: {exc}") from exc
        return vectors.tolist()


def _load_chunks(chunk_path: Path) -> list[dict]:
    with chunk_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "chunks" in data:
        return data["chunks"]
    if isinstance(data, list):
        return data
    raise EmbeddingError(f"Unrecognized chunk.json structure in {chunk_path}")


def _content_hash_of(chunk: dict) -> str | None:
    meta = chunk.get("metadata", {})
    return meta.get("content_hash") or chunk.get("content_hash")


def _write_embeddings(embeddings_path: Path, records: list[dict]) -> None:
    embeddings_path.parent.mkdir(parents=True, exist_ok=True)
    with embeddings_path.open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)


def embed_chunk_file(chunk_path: Path, embedder: Embedder, cache: EmbeddingCache) -> dict:
    chunks = _load_chunks(chunk_path)
    doc_folder = chunk_path.parent
    embeddings_path = doc_folder / EMBEDDINGS_FILENAME

    records: list[dict | None] = [None] * len(chunks)
    to_embed_texts: list[str] = []
    to_embed_idx: list[int] = []

    for i, chunk in enumerate(chunks):
        chunk_id = chunk.get("chunk_id")
        content = chunk.get("content", "")
        content_hash = _content_hash_of(chunk)

        if not chunk_id or not content_hash:
            log.warning("chunk_missing_fields", extra={"chunk_path": str(chunk_path), "index": i})
            continue

        cached_vector = cache.get(content_hash, embedder.model_name)
        if cached_vector is not None:
            records[i] = {
                "chunk_id": chunk_id,
                "content_hash": content_hash,
                "embedding_model": embedder.model_name,
                "embedding_dim": embedder.embedding_dim,
                "vector": cached_vector,
            }
        else:
            to_embed_texts.append(content)
            to_embed_idx.append(i)

    if to_embed_texts:
        vectors = embedder.encode(to_embed_texts)
        for idx, vector in zip(to_embed_idx, vectors):
            chunk = chunks[idx]
            chunk_id = chunk["chunk_id"]
            content_hash = _content_hash_of(chunk)
            records[idx] = {
                "chunk_id": chunk_id,
                "content_hash": content_hash,
                "embedding_model": embedder.model_name,
                "embedding_dim": embedder.embedding_dim,
                "vector": vector,
            }
            cache.set(content_hash, embedder.model_name, vector)

    final_records = [r for r in records if r is not None]
    _write_embeddings(embeddings_path, final_records)
    cache.flush()

    stats = {
        "doc_folder": doc_folder.name,
        "total_chunks": len(chunks),
        "newly_embedded": len(to_embed_texts),
        "from_cache": len(final_records) - len(to_embed_texts),
        "output": str(embeddings_path),
    }
    log.info("embedded_document", extra=stats)
    return stats


def find_chunk_files(processed_dir: Path = PROCESSED_DIR) -> list[Path]:
    return sorted(processed_dir.glob(f"*/{CHUNK_FILENAME}"))


def embed_all(processed_dir: Path = PROCESSED_DIR) -> list[dict]:
    chunk_files = find_chunk_files(processed_dir)
    if not chunk_files:
        log.warning("no_chunk_files_found", extra={"processed_dir": str(processed_dir)})
        return []

    embedder = Embedder()
    cache = EmbeddingCache()

    return [embed_chunk_file(cf, embedder, cache) for cf in chunk_files]


if __name__ == "__main__":
    results = embed_all()
    total = sum(r["total_chunks"] for r in results)
    new = sum(r["newly_embedded"] for r in results)
    print(f"Embedded {total} chunks across {len(results)} documents ({new} new, {total - new} cached).")