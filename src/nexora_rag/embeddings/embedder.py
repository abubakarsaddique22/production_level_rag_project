import json
from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer

from ..core.config import settings


class Embedder:
    def __init__(self) -> None:
        self.device = "cpu"

        self.model = SentenceTransformer(
            settings.embed_model,
            device=self.device,
        )

        self.embedding_dimension = self.model.get_sentence_embedding_dimension()

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        embeddings = self.model.encode(
            texts,
            batch_size=settings.embed_batch_size,
            normalize_embeddings=True,
            show_progress_bar=True,
            convert_to_numpy=True,
        )

        return embeddings.tolist()

    def embed_chunks(self, chunks: list[dict]) -> list[dict]:
        if not chunks:
            return []

        texts = [chunk["content"] for chunk in chunks]
        embeddings = self.embed_texts(texts)

        embedded_chunks = []

        for chunk, embedding in zip(chunks, embeddings):
            embedded_chunks.append(
                {
                    **chunk,
                    "embedding": embedding,
                }
            )

        return embedded_chunks


def load_chunks(chunks_path: Path) -> list[dict]:
    with open(chunks_path, "r", encoding="utf-8") as f:
        return json.load(f)


def embed_document(document_dir: Path) -> list[dict]:
    chunks_path = document_dir / "chunks.json"

    if not chunks_path.exists():
        print(f"SKIP: {document_dir.name} -> chunks.json not found")
        return []

    chunks = load_chunks(chunks_path)

    if not chunks:
        print(f"SKIP: {document_dir.name} -> no chunks found")
        return []

    embedder = Embedder()
    embedded_chunks = embedder.embed_chunks(chunks)

    output_path = document_dir / "embedded_chunks.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            embedded_chunks,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(
        f"OK: {document_dir.name} -> "
        f"{len(embedded_chunks)} embeddings -> {output_path}"
    )
    print(f"Embedding dimension: {embedder.embedding_dimension}")

    return embedded_chunks


def embed_all(processed_dir: Path | None = None) -> None:
    processed_dir = processed_dir or settings.data_processed_dir

    document_dirs = [
        d for d in processed_dir.iterdir()
        if d.is_dir()
    ]

    if not document_dirs:
        print("No processed document directories found.")
        return

    embedder = Embedder()

    total_documents = 0
    total_chunks = 0

    for document_dir in document_dirs:
        chunks_path = document_dir / "chunks.json"

        if not chunks_path.exists():
            print(
                f"SKIP: {document_dir.name} -> "
                "chunks.json not found"
            )
            continue

        chunks = load_chunks(chunks_path)

        if not chunks:
            print(
                f"SKIP: {document_dir.name} -> "
                "no chunks found"
            )
            continue

        embedded_chunks = embedder.embed_chunks(chunks)

        output_path = document_dir / "embedded_chunks.json"

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(
                embedded_chunks,
                f,
                indent=2,
                ensure_ascii=False,
            )

        total_documents += 1
        total_chunks += len(embedded_chunks)

        print(
            f"OK: {document_dir.name} -> "
            f"{len(embedded_chunks)} embeddings"
        )

    print(f"\nTotal documents: {total_documents}")
    print(f"Total embedded chunks: {total_chunks}")
    print(
        f"Embedding dimension: "
        f"{embedder.embedding_dimension}"
    )


if __name__ == "__main__":
    print(f"Torch version: {torch.__version__}")
    print(f"Device: cpu")
    print(f"Embedding model: {settings.embed_model}")

    embed_all()