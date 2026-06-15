"""Run Chandra OCR on all PDFs in a folder and write markdown outputs to another folder."""

import argparse
import os
from pathlib import Path

import logging

os.environ.setdefault("VLLM_API_BASE", "http://chandra-vllm:8000/v1")
os.environ.setdefault("VLLM_MODEL_NAME", "chandra")

from scirex.ocr.modern.pipeline import pdf_to_images, images_to_markdown, save_markdown, output_path_for, stats_path_for, save_stats
from concurrent.futures import ThreadPoolExecutor, as_completed


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

def process_one_pdf(pdf_path, i, total, output_dir):
    """Process a single PDF: skip if done, otherwise OCR and save."""
    output_path = output_path_for(pdf_path, output_dir)
    arxiv_id = pdf.path.stem  # Assuming the PDF filename is the arxivid.pdf 
    if output_path.exists():
        return f"[{i}/{total}] Skipping {pdf_path.name} (already done)"
    try:
        arxiv_id = pdf.path.stem  # Assuming the PDF filename is the arxivid.pdf 
        pages = pdf_to_images(pdf_path)
        md, stats = images_to_markdown(pages)
        saved_path = save_markdown(md, pdf_path, output_dir)
        save_stats(stats, pdf_path, output_dir)
        return f"[{i}/{total}] ✓ Wrote {saved_path.name} + stats ({len(pages)} pages)"
    except Exception:
        logger.exception(f"[{i}/{total}] ✗ Failed to process {pdf_path.name}")
        return None

def main():
    # 1. Parse arguments
    parser = argparse.ArgumentParser(description="Batch OCR PDFs with Chandra.")
    parser.add_argument("--input-dir", type=Path, required=True, help="Folder containing PDFs.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Folder where markdown files will be written.")
    parser.add_argument("--limit", type=int, default=None, help="Optional: process only the first N PDFs (for testing).")
    parser.add_argument("--max-concurrent-pdfs", type=int, default=3, help="Number of PDFs to process in parallel (default 3).")
    args = parser.parse_args()

    # 2. List the PDFs
    pdf_paths = sorted(args.input_dir.glob("*.pdf"))
    if args.limit:
        pdf_paths = pdf_paths[:args.limit]

    logger.info(f"Found {len(pdf_paths)} PDFs in {args.input_dir}")

    # 3. Process PDFs in parallel
    with ThreadPoolExecutor(max_workers=args.max_concurrent_pdfs) as executor:
        futures = [
            executor.submit(process_one_pdf, pdf_path, i, len(pdf_paths), args.output_dir)
            for i, pdf_path in enumerate(pdf_paths, start=1)
        ]
        for future in as_completed(futures):
            result = future.result()
            if result:
                logger.info(result)

    
if __name__ == "__main__":
    main()