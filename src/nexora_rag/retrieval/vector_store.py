"""
Qdrant collection setup, upsert and dense (semantic) search.

Reads, per document folder under data/processed/<Document>/:
    chunks.json       -- chunk_id, content, metadata (doc_id, department, page, ...)
    embeddings.json   -- chunk_id, content_hash, embedding_model, embedding_dim, vector

Writes: nothing to disk -- everything is upserted into the Qdrant collection
named by settings.collection at settings.qdrant_url.

Idempotent indexing: point IDs are deterministically derived from chunk_id
(a UUID5 hash), so re-running index_all() on unchanged chunks re-upserts the
SAME points instead of creating duplicates.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from ..core.config import settings
from ..core.exceptions import VectorStoreError
from ..core.logging import get_logger

log = get_logger(__name__)

PROCESSED_DIR = Path("data/processed")
CHUNK_FILENAME = "chunks.json"
EMBEDDINGS_FILENAME = "embeddings.json"

# Qdrant point IDs must be an unsigned integer or a UUID. chunk_id is a
# human-readable string (e.g. "NX-ENG-001-p1-c0-55e58e54b2e0"), so we
# deterministically derive a UUID from it: same chunk_id -> same UUID,
# every run, on every machine -- which is exactly what makes upsert
# overwrite the same point instead of creating a duplicate.
_ID_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_DNS, "nexora-rag.chunks")


def _point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(_ID_NAMESPACE, chunk_id))


def _load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


class VectorStore:
    """Thin wrapper around qdrant-client for this project's one collection."""

    def __init__(
        self,
        url: str | None = None,
        collection: str | None = None,
        api_key: str | None = None,
    ):
        self.url = url or settings.qdrant_url
        self.collection = collection or settings.collection
        self._embedder = None  # lazy-loaded on first search_by_text() call

        try:
            self.client = QdrantClient(
                url=self.url,
                api_key=api_key or settings.qdrant_api_key,
            )
        except Exception as exc:
            raise VectorStoreError(f"Could not connect to Qdrant at {self.url}: {exc}") from exc

    def collection_exists(self) -> bool:
        existing = [c.name for c in self.client.get_collections().collections]
        return self.collection in existing

    def ensure_collection(self, vector_size: int) -> None:
        """Creates the collection (with payload indexes for role-based
        filtering) if it doesn't already exist. Safe to call every run."""
        if self.collection_exists():
            log.info("qdrant_collection_exists", extra={"collection": self.collection})
            return

        self.client.create_collection(
            collection_name=self.collection,
            vectors_config=qmodels.VectorParams(
                size=vector_size,
                distance=qmodels.Distance.COSINE,
            ),
        )

        for field in ("department", "doc_id", "version"):
            self.client.create_payload_index(
                collection_name=self.collection,
                field_name=field,
                field_schema=qmodels.PayloadSchemaType.KEYWORD,
            )

        log.info(
            "qdrant_collection_created",
            extra={"collection": self.collection, "vector_size": vector_size},
        )

    def upsert_chunks(self, chunks: list[dict], embeddings_by_chunk_id: dict[str, dict]) -> int:
        """Upserts one document's chunks. Chunks without a matching
        embedding are skipped with a warning (should not normally happen
        if the embedder ran successfully first)."""
        points: list[qmodels.PointStruct] = []

        for chunk in chunks:
            chunk_id = chunk["chunk_id"]
            emb = embeddings_by_chunk_id.get(chunk_id)

            if emb is None:
                log.warning("missing_embedding_for_chunk", extra={"chunk_id": chunk_id})
                continue

            meta = chunk.get("metadata", {})
            payload = {
                "chunk_id": chunk_id,
                "content": chunk["content"],
                "doc_id": meta.get("doc_id"),
                "title": meta.get("title"),
                "department": meta.get("department"),
                "version": meta.get("version"),
                "effective_date": meta.get("effective_date"),
                "page": meta.get("page"),
                "source_pdf": meta.get("source_pdf"),
                "chunk_type": meta.get("chunk_type"),
                "chunk_index": meta.get("chunk_index"),
                "content_hash": meta.get("content_hash"),
            }

            points.append(
                qmodels.PointStruct(
                    id=_point_id(chunk_id),
                    vector=emb["vector"],
                    payload=payload,
                )
            )

        if not points:
            return 0

        try:
            self.client.upsert(collection_name=self.collection, points=points)
        except Exception as exc:
            raise VectorStoreError(f"Qdrant upsert failed: {exc}") from exc

        return len(points)

    def search(
        self,
        query_vector: list[float],
        top_k: int = 10,
        department_filter: list[str] | None = None,
        ) -> list:
        """Dense (semantic) search. department_filter enforces role-based
        access at the RETRIEVAL layer (never only in the prompt -- see
        Step Q / Table 10 in the blueprint).

        Handles two qdrant-client API generations: older versions expose
        `.search()`; qdrant-client >= 1.10 removed it in favour of
        `.query_points()`, which wraps the same list of scored points in
        a QueryResponse object. We detect which one is available at
        runtime so this code keeps working across client upgrades.
        """
        query_filter = None
        if department_filter is not None:
            query_filter = qmodels.Filter(
                must=[
                    qmodels.FieldCondition(
                        key="department",
                        match=qmodels.MatchAny(any=department_filter),
                    )
                ]
            )

        try:
            if hasattr(self.client, "query_points"):
                response = self.client.query_points(
                    collection_name=self.collection,
                    query=query_vector,
                    query_filter=query_filter,
                    limit=top_k,
                    with_payload=True,
                )
                return response.points
            return self.client.search(
                collection_name=self.collection,
                query_vector=query_vector,
                query_filter=query_filter,
                limit=top_k,
                with_payload=True,
        )
        except Exception as exc:
            raise VectorStoreError(f"Qdrant search failed: {exc}") from exc

    def search_by_text(
        self,
        query_text: str,
        top_k: int = 10,
        department_filter: list[str] | None = None,
    ) -> list:
        """Convenience wrapper: embeds query_text with the SAME embedding
        model used at ingestion time, then runs a dense search.

        This is what evaluation/metrics.py and (later) the /v1/chat
        endpoint call -- callers never touch a raw vector, they just pass
        a question as plain text.

        The embedding model is loaded lazily (on first call) and reused
        for every later call, so calling this in a loop over 30 golden-set
        questions loads the model exactly once.
        """
        if self._embedder is None:
            # local import avoids forcing every VectorStore user to pay
            # the sentence-transformers import cost even if they never
            # call search_by_text() (e.g. the indexing script only embeds
            # via embeddings/embedder.py directly, never through here).
            from ..embeddings.embedder import Embedder

            self._embedder = Embedder()

        query_vector = self._embedder.encode([query_text])[0]
        return self.search(query_vector, top_k=top_k, department_filter=department_filter)

    def count(self) -> int:
        info = self.client.get_collection(self.collection)
        return info.points_count


def index_document_folder(doc_folder: Path, store: VectorStore) -> dict:
    chunk_path = doc_folder / CHUNK_FILENAME
    embeddings_path = doc_folder / EMBEDDINGS_FILENAME

    if not chunk_path.exists() or not embeddings_path.exists():
        log.warning("missing_files_for_indexing", extra={"doc_folder": doc_folder.name})
        return {"doc_folder": doc_folder.name, "indexed": 0}

    chunks = _load_json(chunk_path)
    embedding_records = _load_json(embeddings_path)
    embeddings_by_chunk_id = {e["chunk_id"]: e for e in embedding_records}

    indexed = store.upsert_chunks(chunks, embeddings_by_chunk_id)
    log.info("indexed_document", extra={"doc_folder": doc_folder.name, "indexed": indexed})
    return {"doc_folder": doc_folder.name, "indexed": indexed}


def index_all(processed_dir: Path = PROCESSED_DIR) -> list[dict]:
    """Entry point used by scripts/ingest.py and ingestion/pipeline.py.

    Infers the vector size from the first embeddings.json found, creates
    the collection if needed, then upserts every document folder.
    """
    doc_folders = sorted(
        d for d in processed_dir.iterdir() if d.is_dir() and not d.name.startswith(".")
    )
    if not doc_folders:
        log.warning("no_document_folders_found", extra={"processed_dir": str(processed_dir)})
        return []

    vector_size = None
    for d in doc_folders:
        emb_path = d / EMBEDDINGS_FILENAME
        if emb_path.exists():
            records = _load_json(emb_path)
            if records:
                vector_size = records[0]["embedding_dim"]
                break

    if vector_size is None:
        raise VectorStoreError(
            "Could not determine vector size -- no embeddings.json found. Run the embedder first."
        )

    store = VectorStore()
    store.ensure_collection(vector_size)

    results = [index_document_folder(d, store) for d in doc_folders]
    total = sum(r["indexed"] for r in results)
    log.info("indexing_complete", extra={"total_indexed": total, "collection": store.collection})
    return results


if __name__ == "__main__":
    results = index_all()
    total = sum(r["indexed"] for r in results)
    print(
        f"Indexed {total} chunks across {len(results)} documents "
        f"into Qdrant collection '{settings.collection}'."
    )