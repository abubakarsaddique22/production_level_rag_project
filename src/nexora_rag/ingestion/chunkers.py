import json
import hashlib
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ..core.config import settings


def create_chunk_id(doc_id: str, page: int, chunk_index: int, content: str) -> str:
    content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()[:12]
    return f"{doc_id}-p{page}-c{chunk_index}-{content_hash}"


def detect_chunk_type(content: str) -> str:
    has_table = "Tables:" in content
    has_image = "Image OCR:" in content

    if has_table and has_image:
        return "mixed"
    if has_table:
        return "table"
    if has_image:
        return "image_ocr"
    return "prose"


def load_clean_pages(clean_pages_path: Path) -> list[Document]:
    with open(clean_pages_path, "r", encoding="utf-8") as f:
        pages = json.load(f)

    documents = []

    for page in pages:
        metadata = {
            "doc_id": page["doc_id"],
            "title": page["title"],
            "department": page["department"],
            "version": page["version"],
            "effective_date": page["effective_date"],
            "page": page["page"],
            "source_pdf": page["source_pdf"],
            "image_count": page["image_count"],
            "table_count": page["table_count"],
            "content_hash": page["content_hash"],
            "chunk_type_hint": page["chunk_type_hint"],
        }

        documents.append(
            Document(
                page_content=page["content"],
                metadata=metadata,
            )
        )

    return documents


def create_splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_tokens * 4,
        chunk_overlap=settings.chunk_overlap * 4,
        separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""],
        length_function=len,
    )


def chunk_document(document_dir: Path) -> list[dict]:
    clean_pages_path = document_dir / "clean_pages.json"

    if not clean_pages_path.exists():
        print(f"SKIP: {document_dir.name} -> clean_pages.json not found")
        return []

    documents = load_clean_pages(clean_pages_path)
    splitter = create_splitter()
    chunks = splitter.split_documents(documents)

    output_chunks = []
    page_chunk_counts = {}

    for chunk in chunks:
        page = chunk.metadata["page"]
        chunk_index = page_chunk_counts.get(page, 0)
        page_chunk_counts[page] = chunk_index + 1

        content = chunk.page_content.strip()

        if not content:
            continue

        chunk_id = create_chunk_id(
            chunk.metadata["doc_id"],
            page,
            chunk_index,
            content,
        )

        output_chunks.append({
            "chunk_id": chunk_id,
            "content": content,
            "metadata": {
                **chunk.metadata,
                "chunk_index": chunk_index,
                "chunk_type": detect_chunk_type(content),
                "chunk_size": len(content),
            },
        })

    return output_chunks


def chunk_all(processed_dir: Path | None = None) -> None:
    processed_dir = processed_dir or settings.data_processed_dir
    total_documents = 0
    total_chunks = 0

    for document_dir in processed_dir.iterdir():
        if not document_dir.is_dir():
            continue

        chunks = chunk_document(document_dir)

        if not chunks:
            continue

        output_path = document_dir / "chunks.json"

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(chunks, f, indent=2, ensure_ascii=False)

        total_documents += 1
        total_chunks += len(chunks)

        print(
            f"OK: {document_dir.name} -> "
            f"{len(chunks)} chunks -> {output_path}"
        )

    print(f"\nTotal documents: {total_documents}")
    print(f"Total chunks: {total_chunks}")


if __name__ == "__main__":
    chunk_all()