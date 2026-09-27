"""
Orchestrates the FULL offline ingestion pipeline end to end:

    loaders (parse PDFs)  ->  cleaners (clean_pages.json)
        ->  chunkers (chunks.json)  ->  embedder (embeddings.json)
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from ..core.config import settings
from ..core.logging import configure_logging, get_logger
from ..embeddings.embedder import embed_all
from . import chunkers, cleaners, loaders

log = get_logger(__name__)


def _run_parse(pdf_dir: Path, processed_dir: Path) -> None:
    """Step E: raw PDFs -> page_records.json / table_records.json /
    image_records.json / rag_ready.md (via loaders.py)."""
    pdf_paths = loaders.load_pdf_paths_from_directory(pdf_dir)

    if not pdf_paths:
        print(f"No PDFs found in {pdf_dir} -- nothing to parse.")
        return

    print(f"Found {len(pdf_paths)} PDF(s) in {pdf_dir}:")
    for p in pdf_paths:
        print("  -", p)

    for pdf_path in pdf_paths:
        loaders.process_single_pdf(pdf_path, processed_dir)

    loaders.build_final_merged_rag(processed_dir, pdf_paths)


def run_pipeline(
    pdf_dir: str | Path = "data/raw",
    processed_dir: str | Path | None = None,
    skip_parse: bool = False,
    skip_clean: bool = False,
    skip_chunk: bool = False,
    skip_embed: bool = False,
) -> None:
    pdf_dir = Path(pdf_dir)
    processed_dir = Path(processed_dir or settings.data_processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    timings: list[tuple[str, float]] = []

    def _timed_step(name: str, skip: bool, fn, *args) -> None:
        if skip:
            print(f"\nSKIP: {name} (--skip-{name})")
            return
        print(f"\n{'=' * 60}\nSTEP: {name}\n{'=' * 60}")
        log.info("pipeline_step_start", extra={"step": name})
        t0 = time.time()
        fn(*args)
        elapsed = time.time() - t0
        timings.append((name, elapsed))
        log.info("pipeline_step_done", extra={"step": name, "seconds": round(elapsed, 1)})

    _timed_step("parse", skip_parse, _run_parse, pdf_dir, processed_dir)
    _timed_step("clean", skip_clean, cleaners.clean_all, processed_dir)
    _timed_step("chunk", skip_chunk, chunkers.chunk_all, processed_dir)
    _timed_step("embed", skip_embed, embed_all, processed_dir)

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    if not timings:
        print("  (every step was skipped)")
    for step_name, duration in timings:
        print(f"  {step_name:8s} : {duration:7.1f}s")
    print("=" * 60)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the full RAG ingestion pipeline: parse -> clean -> chunk -> embed"
    )
    parser.add_argument("--pdf-dir", default="data/raw", help="Folder of source PDFs")
    parser.add_argument("--processed-dir", default=None, help="Output folder (default: settings.data_processed_dir)")
    parser.add_argument("--skip-parse", action="store_true", help="Reuse existing page/table/image records")
    parser.add_argument("--skip-clean", action="store_true", help="Reuse existing clean_pages.json")
    parser.add_argument("--skip-chunk", action="store_true", help="Reuse existing chunks.json")
    parser.add_argument("--skip-embed", action="store_true", help="Don't (re)compute embeddings")
    return parser.parse_args()


if __name__ == "__main__":
    configure_logging(settings.log_level)
    args = parse_args()
    run_pipeline(
        pdf_dir=args.pdf_dir,
        processed_dir=args.processed_dir,
        skip_parse=args.skip_parse,
        skip_clean=args.skip_clean,
        skip_chunk=args.skip_chunk,
        skip_embed=args.skip_embed,
    )