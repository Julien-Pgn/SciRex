"""Run Chandra OCR on all PDFs in a folder, save artefacts, and update the DuckDB tables."""

import argparse
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import duckdb

os.environ.setdefault("VLLM_API_BASE", "http://chandra-vllm:8000/v1")
os.environ.setdefault("VLLM_MODEL_NAME", "chandra")

# Import the functions from the pipeline module
from scirex.ocr.modern.pipeline import (
    make_paper_dir,
    ocr_images,
    pdf_to_images,
    save_all,
    update_db,
)

# To modify later with a real call
OCR_MODEL = "chandra0.2.0"

log_dir = Path("data/logs")
log_dir.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(log_dir / "ocr.log"),
        logging.StreamHandler(),
    ],
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


def process_one_pdf(pdf_path, i, total):
    """Process a single PDF: skip if done, otherwise OCR, save and update the table from the DuckDB."""
    arxiv_id = pdf_path.stem  # Assuming the PDF filename is the arxivid.pdf
    out_md = Path("data/processed") / arxiv_id / f"{arxiv_id}.md"
    if out_md.exists():
        logger.info(f"[{i}/{total}] Skipping {pdf_path.name} (already done)")
        return None
    try:
        img_list = pdf_to_images(pdf_path)
        md, stats, figures, html = ocr_images(img_list)
        paper_dir = make_paper_dir(arxiv_id)
        save_all(paper_dir, arxiv_id, html, md, figures)
        logger.info(f"[{i}/{total}] ✓ OCR {pdf_path.name} ({len(img_list)} pages)")
        return {
            "arxiv_id": arxiv_id,
            "stats": stats,
            "nb_pages_pdf": len(img_list),
            "nb_pages_ocr": len(stats),
            "paper_dir": paper_dir,
            "pdf_path": pdf_path,
        }
    except Exception:
        logger.exception(f"[{i}/{total}] ✗ Failed to process {pdf_path.name}")
        return None


def main():
    # 1. Parse arguments
    parser = argparse.ArgumentParser(description="Batch OCR PDFs with Chandra.")
    parser.add_argument(
        "--input-dir", type=Path, default=Path("data/raw/pdfs"), help="Folder containing PDFs."
    )
    parser.add_argument(
        "--db", type=Path, default=Path("data/arxiv_metadata.duckdb"), help="DuckDB database file."
    )
    parser.add_argument(
        "--keyword",
        type=str,
        required=True,
        help="Topic/subset of this batch (stored in paper_local.keyword_for_ocr).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional: process only the first N PDFs (for testing).",
    )
    parser.add_argument(
        "--max-concurrent-pdfs",
        type=int,
        default=6,
        help="Number of PDFs to OCR in parallel (default 6).",
    )
    args = parser.parse_args()

    # 2. List the PDFs
    pdf_paths = sorted(args.input_dir.glob("*.pdf"))
    if args.limit:
        pdf_paths = pdf_paths[: args.limit]
    logger.info(f"Found {len(pdf_paths)} PDFs in {args.input_dir}")

    # 3. Process PDFs in parallel
    conn = duckdb.connect(str(args.db))
    try:
        # OCR in parallel (workers), writing DB in series (principal thread).
        with ThreadPoolExecutor(max_workers=args.max_concurrent_pdfs) as executor:
            futures = [
                executor.submit(process_one_pdf, pdf_path, i, len(pdf_paths))
                for i, pdf_path in enumerate(pdf_paths, start=1)
            ]
            for future in as_completed(futures):
                result = future.result()
                if result is None:
                    continue
                update_db(
                    conn,
                    result["arxiv_id"],
                    result["stats"],
                    nb_pages_pdf=result["nb_pages_pdf"],
                    nb_pages_ocr=result["nb_pages_ocr"],
                    keyword_for_ocr=args.keyword,
                    ocr_model=OCR_MODEL,
                    paper_dir=result["paper_dir"],
                    pdf_path=result["pdf_path"],
                )
                logger.info(f"  DB updated for {result['arxiv_id']}")
    finally:
        conn.close()  # libère le verrou d'écriture pour les prochains runs


if __name__ == "__main__":
    main()
