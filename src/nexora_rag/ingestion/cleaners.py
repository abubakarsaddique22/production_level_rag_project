"""
Noise removal, whitespace/unicode normalization, metadata enrichment.

ingestion/cleaners.py
======================

loaders.py ke raw output (page_records.json, table_records.json,
image_records.json) ko leta hai aur:

1. Har page ko document_registry.yaml se doc_id/department/title deta hai.
2. Selectable text vs OCR text mein se BEHTAR wala chunta hai.
3. Text clean karta hai (extra whitespace, repeated headers/footers).
4. Image OCR text ko image_records.json se corresponding page mein add karta hai.
5. Har page ke liye content_hash banata hai.
6. Pydantic se validate karta hai.

Output: data/processed/<pdf_name>/clean_pages.json
"""

import json
import re
import hashlib
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from ..core.config import settings

# ------------------------------------------------------------------
# 1. Pydantic model
# ------------------------------------------------------------------

class CleanPage(BaseModel):
    doc_id: str
    title: str
    department: str
    version: str
    effective_date: str
    page: int
    content: str = Field(..., min_length=1)
    chunk_type_hint: str  # "prose" | "table_heavy" | "mixed"
    image_count: int
    table_count: int
    content_hash: str
    source_pdf: str

# ------------------------------------------------------------------
# 2. Document registry load
# ------------------------------------------------------------------

def load_registry() -> dict:
    with open(settings.document_registry_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return {doc["filename"]: doc for doc in raw["documents"]}

def get_doc_metadata(pdf_stem: str, registry: dict) -> dict:
    meta = registry.get(pdf_stem)
    if meta is None:
        raise ValueError(
            f"'{pdf_stem}' document_registry.yaml mein nahi mila. "
            f"Pehle usme is document ki entry add karo."
        )
    return meta

# ------------------------------------------------------------------
# 3. Text cleaning
# ------------------------------------------------------------------

_WHITESPACE_RE = re.compile(r"[ \t]+")
_MULTI_BLANK_RE = re.compile(r"\n{3,}")

def clean_text(text: str) -> str:
    """Extra spaces, zyada blank lines aur trailing spaces hatata hai."""
    if not text:
        return ""
    text = _WHITESPACE_RE.sub(" ", text)
    text = _MULTI_BLANK_RE.sub("\n\n", text)
    return text.strip()

def choose_best_text(selectable_text: str, ocr_text: str) -> str:
    """
    Selectable text kaafi ho to usi ko use karta hai.
    Warna OCR text fallback ke taur par use hota hai.
    Dono ko ek sath nahi milata.
    """
    selectable_clean = clean_text(selectable_text)
    ocr_clean = clean_text(ocr_text)

    if len(selectable_clean) >= settings.min_selectable_chars_per_page:
        return selectable_clean
    if len(ocr_clean) > len(selectable_clean):
        return ocr_clean
    return selectable_clean

def compute_content_hash(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]

def guess_chunk_type(table_count: int, image_count: int) -> str:
    if table_count >= 1 and table_count >= image_count:
        return "table_heavy"
    if image_count >= 1:
        return "mixed"
    return "prose"

# ------------------------------------------------------------------
# 4. Ek PDF ke raw records ko clean karna
# ------------------------------------------------------------------

def clean_pdf_output(pdf_output_dir: Path, registry: dict) -> list[CleanPage]:
    pdf_name = pdf_output_dir.name
    doc_meta = get_doc_metadata(pdf_name, registry)

    with open(pdf_output_dir / "page_records.json", encoding="utf-8") as f:
        page_records = json.load(f)

    with open(pdf_output_dir / "table_records.json", encoding="utf-8") as f:
        table_records = json.load(f)

    with open(pdf_output_dir / "image_records.json", encoding="utf-8") as f:
        image_records = json.load(f)

    tables_by_page: dict[int, list] = {}
    for t in table_records:
        tables_by_page.setdefault(t["page_number"], []).append(t)

    images_by_page: dict[int, list] = {}
    for image in image_records:
        images_by_page.setdefault(image["page_number"], []).append(image)

    clean_pages: list[CleanPage] = []

    for page in page_records:
        page_number = page["page_number"]

        best_text = choose_best_text(
            page["text"],
            page["ocr_text"]
        )

        page_tables = tables_by_page.get(page_number, [])
        table_markdown_blocks = "\n\n".join(
            t["markdown"] for t in page_tables
        )

        page_images = images_by_page.get(page_number, [])
        image_ocr_blocks = []

        for image in page_images:
            image_ocr_text = clean_text(
                image.get("image_ocr_text", "")
            )
            if image_ocr_text:
                image_ocr_blocks.append(image_ocr_text)

        heading = f"{doc_meta['title']} > Page {page_number}"
        content_parts = [heading, best_text]

        if image_ocr_blocks:
            content_parts.append(
                "Image OCR:\n" + "\n\n".join(image_ocr_blocks)
            )

        if table_markdown_blocks:
            content_parts.append(
                "Tables:\n" + table_markdown_blocks
            )

        full_content = clean_text(
            "\n\n".join(content_parts)
        )

        clean_page = CleanPage(
            doc_id=doc_meta["doc_id"],
            title=doc_meta["title"],
            department=doc_meta["department"],
            version=doc_meta["version"],
            effective_date=doc_meta["effective_date"],
            page=page_number,
            content=full_content,
            chunk_type_hint=guess_chunk_type(
                len(page_tables),
                len(page_images)
            ),
            image_count=len(page_images),
            table_count=len(page_tables),
            content_hash=compute_content_hash(full_content),
            source_pdf=pdf_name,
        )

        clean_pages.append(clean_page)

    return clean_pages

# ------------------------------------------------------------------
# 5. Sab PDFs process karna
# ------------------------------------------------------------------

def clean_all(processed_dir: Path | None = None) -> None:
    processed_dir = processed_dir or settings.data_processed_dir
    registry = load_registry()
    pdf_dirs = [d for d in processed_dir.iterdir() if d.is_dir()]
    total_pages = 0

    for pdf_dir in pdf_dirs:
        if pdf_dir.name not in registry:
            print(f"SKIP: '{pdf_dir.name}' registry mein nahi hai.")
            continue

        clean_pages = clean_pdf_output(pdf_dir, registry)
        out_path = pdf_dir / "clean_pages.json"

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(
                [p.model_dump() for p in clean_pages],
                f,
                indent=2,
                ensure_ascii=False
            )

        print(
            f"OK: {pdf_dir.name} -> "
            f"{len(clean_pages)} clean pages -> {out_path}"
        )
        total_pages += len(clean_pages)

    print(f"\nTotal cleaned pages across all documents: {total_pages}")

if __name__ == "__main__":
    clean_all()