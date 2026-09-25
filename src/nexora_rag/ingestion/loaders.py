import os
import sys
import json
import argparse
import fitz  # PyMuPDF
import pdfplumber
import pandas as pd
import pytesseract
from PIL import Image
from pathlib import Path
from langchain_core.documents import Document

# ------------------------------------------------------------------
# WINDOWS: Tesseract path set kar diya hai (confirmed installed here).
# ------------------------------------------------------------------
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# ------------------------------------------------------------------
# NAYA: Full-page OCR sirf tab chalana hai jab selectable text kam ho
# (matlab yeh ek scanned/image-only page hai). Agar page pe already
# accurate selectable text maujood hai, poori page ki OCR fazool hai —
# isse processing time bachta hai aur cleaners.py ko clean signal milta
# hai ke "yeh selectable-text page thi, OCR ki zaroorat nahi thi".
# ------------------------------------------------------------------
MIN_SELECTABLE_CHARS = 40


# ============================================================
# 0. CLI args
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Parse ALL PDFs in a folder (text + OCR + tables + images) "
                     "into per-page LangChain Documents."
    )
    parser.add_argument(
        "pdf_dir",
        nargs="?",
        default="data/raw",
        help="Folder containing one or more PDF files (default: data/raw)",
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed",
        help="Directory to save extracted images and parsed output (default: data/processed)",
    )
    return parser.parse_args()


# ============================================================
# 1. Discover / load all PDFs in a folder using LangChain DirectoryLoader
# ============================================================

def load_pdf_paths_from_directory(pdf_dir):
    """
    Finds every PDF under pdf_dir (recursively).

    NOTE: We deliberately do NOT use LangChain's DirectoryLoader/PyPDFLoader
    here. Importing langchain_community.document_loaders pulls in
    langchain_text_splitters -> sentence_transformers -> transformers -> torch
    as a side effect, which is a heavy, fragile dependency chain that can
    crash (e.g. broken torch DLL installs on Windows) even though we only
    need the file PATHS, not any actual document loading. Plain pathlib
    globbing does the same job here with zero extra dependencies.
    """
    pdf_dir = Path(pdf_dir)
    pdf_paths = sorted(str(p) for p in pdf_dir.rglob("*.pdf"))
    return pdf_paths


# ============================================================
# 2. Helper: Safe OCR (Tesseract via pytesseract)
# ============================================================

def run_ocr_on_image(image_path):
    try:
        img = Image.open(image_path)
        text = pytesseract.image_to_string(img)
        return text.strip()
    except Exception as e:
        return f"[OCR_SKIPPED_OR_FAILED: {str(e)}]"


# ============================================================
# 3. Extract Text + Images using PyMuPDF (with Tesseract OCR)
# ============================================================

def extract_text_and_images(pdf_path, image_dir, page_image_dir):
    doc = fitz.open(pdf_path)

    page_records = []
    image_records = []

    for page_index in range(len(doc)):
        page = doc[page_index]
        page_number = page_index + 1

        text = page.get_text("text")

        page_info = {
            "page_number": page_number,
            "text": text.strip(),
            "image_count": len(page.get_images(full=True)),
            "width": page.rect.width,
            "height": page.rect.height,
        }
        page_records.append(page_info)

        # Page image (render) — yeh hamesha banti hai, kyunke isi se
        # baad mein tables crop hoti hain (save_extracted_tables mein).
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
        page_image_path = page_image_dir / f"page_{page_number:03d}.png"
        pix.save(str(page_image_path))
        page_info["page_image_path"] = str(page_image_path)

        # === CHANGE: full-page OCR sirf zaroorat pe (scanned pages) ===
        if len(page_info["text"]) < MIN_SELECTABLE_CHARS:
            ocr_text = run_ocr_on_image(page_image_path)
        else:
            ocr_text = ""  # selectable text kaafi hai, poori-page OCR fazool hai
        page_info["ocr_text"] = ocr_text

        # Individual embedded images pe OCR rehne diya hai — yeh zaroori
        # hai kyunke diagram/screenshot ke andar ka text selectable text
        # mein kabhi nahi aata.
        images = page.get_images(full=True)
        for img_index, img in enumerate(images):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]

            image_path = image_dir / f"page_{page_number:03d}_image_{img_index + 1}.{image_ext}"
            with open(image_path, "wb") as f:
                f.write(image_bytes)

            image_ocr_text = run_ocr_on_image(image_path)

            image_records.append({
                "page_number": page_number,
                "image_index": img_index + 1,
                "image_path": str(image_path),
                "image_ext": image_ext,
                "image_ocr_text": image_ocr_text
            })

    return page_records, image_records


# ============================================================
# 4. Extract Tables using pdfplumber
# ============================================================

def extract_tables(pdf_path):
    table_records = []

    with pdfplumber.open(pdf_path) as pdf:
        for page_index, page in enumerate(pdf.pages):
            page_number = page_index + 1
            try:
                found_tables = page.find_tables()
            except Exception as e:
                found_tables = []
                print(f"Table extraction failed on page {page_number}: {e}")

            for table_index, table_obj in enumerate(found_tables):
                try:
                    table = table_obj.extract()
                except Exception:
                    continue

                if not table:
                    continue

                cleaned_table = []
                for row in table:
                    cleaned_row = [
                        cell.strip() if isinstance(cell, str) else cell
                        for cell in row
                    ]
                    cleaned_table.append(cleaned_row)

                try:
                    df = pd.DataFrame(cleaned_table[1:], columns=cleaned_table[0])
                except Exception:
                    df = pd.DataFrame(cleaned_table)

                table_records.append({
                    "page_number": page_number,
                    "table_index": table_index + 1,
                    "raw_table": cleaned_table,
                    "markdown": df.to_markdown(index=False),
                    "csv": df.to_csv(index=False),
                    "bbox": table_obj.bbox,  # (x0, top, x1, bottom) in PDF points
                })

    return table_records


# ============================================================
# 5. Build one LangChain Document PER PAGE
#    (page text + OCR + that page's tables + that page's images, combined)
# ============================================================

def build_per_page_documents(pdf_path, page_records, table_records, image_records):
    tables_by_page = {}
    for t in table_records:
        tables_by_page.setdefault(t["page_number"], []).append(t)

    images_by_page = {}
    for im in image_records:
        images_by_page.setdefault(im["page_number"], []).append(im)

    page_documents = []

    for page in page_records:
        page_number = page["page_number"]

        parts = [
            f"PAGE {page_number}",
            "",
            "SELECTABLE TEXT:",
            page["text"],
            "",
            "OCR TEXT:",
            page["ocr_text"],
        ]

        page_tables = tables_by_page.get(page_number, [])
        if page_tables:
            parts.append("")
            parts.append("TABLES ON THIS PAGE:")
            for t in page_tables:
                parts.append(f"\n-- Table {t['table_index']} --\n{t['markdown']}")

        page_images = images_by_page.get(page_number, [])
        if page_images:
            parts.append("")
            parts.append("IMAGES ON THIS PAGE:")
            for im in page_images:
                parts.append(
                    f"\n-- Image {im['image_index']} ({im['image_path']}) --\n"
                    f"OCR TEXT:\n{im['image_ocr_text']}"
                )

        combined_text = "\n".join(parts).strip()

        doc = Document(
            page_content=combined_text,
            metadata={
                "source": pdf_path,
                "page_number": page_number,
                "content_type": "full_page",
                "image_count": page["image_count"],
                "table_count": len(page_tables),
                "page_image_path": page["page_image_path"],
            }
        )
        page_documents.append(doc)

    return page_documents


# ============================================================
# 6. Save ONE separate final output file PER PAGE
# ============================================================

def save_extracted_tables(pdf_output_dir, page_records, table_records, zoom=2):
    """
    extracted_tables/ folder holds ONLY images — one cropped image per table,
    cropped out of that page's full-page render using the table's bbox from
    pdfplumber. No .md / .csv files here (that raw data still lives inside
    table_records.json).
    """
    tables_dir = pdf_output_dir / "extracted_tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    page_image_by_number = {p["page_number"]: p["page_image_path"] for p in page_records}

    saved = 0
    for t in table_records:
        page_number = t["page_number"]
        table_index = t["table_index"]
        bbox = t.get("bbox")
        page_image_path = page_image_by_number.get(page_number)

        if not bbox or not page_image_path:
            continue

        try:
            img = Image.open(page_image_path)
            x0, top, x1, bottom = bbox
            # page image was rendered at `zoom`x scale (fitz.Matrix(zoom, zoom)),
            # so PDF-point coordinates need to be scaled up by the same factor.
            crop_box = (int(x0 * zoom), int(top * zoom), int(x1 * zoom), int(bottom * zoom))
            cropped = img.crop(crop_box)

            out_path = tables_dir / f"page_{page_number:03d}_table_{table_index}.png"
            cropped.save(out_path)
            saved += 1
        except Exception as e:
            print(f"Failed to crop table image (page {page_number}, table {table_index}): {e}")

    print(f"Saved {saved} table image(s) in:", tables_dir)


def save_combined_rag_ready(pdf_output_dir, page_documents):
    """
    One combined rag_ready.md per PDF — all of that PDF's pages, one after
    another, in a single file. (So 5 PDFs in -> 5 rag_ready.md files out,
    one inside each PDF's own output folder.)
    """
    rag_path = pdf_output_dir / "rag_ready.md"
    with open(rag_path, "w", encoding="utf-8") as f:
        for doc in page_documents:
            page_number = doc.metadata["page_number"]
            f.write(f"\n\n# Page {page_number}\n")
            f.write(f"\nMetadata:\n```json\n{json.dumps(doc.metadata, indent=2)}\n```\n")
            f.write("\nContent:\n\n")
            f.write(doc.page_content)
            f.write("\n\n---\n")

    print("Saved combined rag_ready.md at:", rag_path)


def save_raw_records(pdf_output_dir, page_records, image_records, table_records):
    with open(pdf_output_dir / "page_records.json", "w", encoding="utf-8") as f:
        json.dump(page_records, f, indent=2, ensure_ascii=False)
    with open(pdf_output_dir / "image_records.json", "w", encoding="utf-8") as f:
        json.dump(image_records, f, indent=2, ensure_ascii=False)
    with open(pdf_output_dir / "table_records.json", "w", encoding="utf-8") as f:
        json.dump(table_records, f, indent=2, ensure_ascii=False)


# ============================================================
# 7. Process a single PDF end-to-end
# ============================================================

def process_single_pdf(pdf_path, base_output_dir):
    pdf_name = Path(pdf_path).stem
    pdf_output_dir = base_output_dir / pdf_name
    image_dir = pdf_output_dir / "extracted_images"
    page_image_dir = pdf_output_dir / "page_images"

    pdf_output_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)
    page_image_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}\nParsing PDF: {pdf_path}\n{'='*60}")

    page_records, image_records = extract_text_and_images(pdf_path, image_dir, page_image_dir)
    print("Total pages parsed:", len(page_records))
    print("Total images extracted:", len(image_records))

    table_records = extract_tables(pdf_path)
    print("Total tables extracted:", len(table_records))

    page_documents = build_per_page_documents(pdf_path, page_records, table_records, image_records)

    save_raw_records(pdf_output_dir, page_records, image_records, table_records)
    save_extracted_tables(pdf_output_dir, page_records, table_records)
    save_combined_rag_ready(pdf_output_dir, page_documents)

    return page_documents


def build_final_merged_rag(base_output_dir, pdf_paths):
    """
    After ALL PDFs are processed, merge every PDF's rag_ready.md into ONE
    final file, appended in the same sequence the PDFs were processed in.
    """
    final_path = base_output_dir / "final_rag_ready.md"

    with open(final_path, "w", encoding="utf-8") as out:
        for pdf_path in pdf_paths:
            pdf_name = Path(pdf_path).stem
            rag_path = base_output_dir / pdf_name / "rag_ready.md"

            if not rag_path.exists():
                continue

            out.write(f"\n\n<!-- ==================== SOURCE PDF: {pdf_name} ==================== -->\n")
            with open(rag_path, "r", encoding="utf-8") as f:
                out.write(f.read())
            out.write("\n\n")

    print("Saved final merged RAG file at:", final_path)


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()

    pdf_dir = args.pdf_dir
    if not os.path.isdir(pdf_dir):
        print(f"ERROR: Folder not found: {pdf_dir}")
        sys.exit(1)

    try:
        print("Tesseract version:", pytesseract.get_tesseract_version())
    except Exception as e:
        print(f"WARNING: Could not detect Tesseract on PATH ({e}). "
              f"OCR steps will fail gracefully and return placeholder text.")

    base_output_dir = Path(args.output_dir)
    base_output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Discover all PDFs in the folder via LangChain DirectoryLoader
    pdf_paths = load_pdf_paths_from_directory(pdf_dir)
    if not pdf_paths:
        print(f"No PDFs found in {pdf_dir}")
        sys.exit(0)

    print(f"Found {len(pdf_paths)} PDF(s):")
    for p in pdf_paths:
        print("  -", p)

    # 2. Pass each discovered PDF into the parsing pipeline
    all_page_documents = []
    for pdf_path in pdf_paths:
        page_documents = process_single_pdf(pdf_path, base_output_dir)
        all_page_documents.extend(page_documents)

    print(f"\nDone. Total pages processed across all PDFs: {len(all_page_documents)}")

    # 3. Merge every PDF's rag_ready.md into one final file, in sequence
    build_final_merged_rag(base_output_dir, pdf_paths)

    print(f"All outputs saved under: {base_output_dir}")


if __name__ == "__main__":
    main()