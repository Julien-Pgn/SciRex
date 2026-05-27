"""Run Chandra OCR on all PDFs in a folder and write markdown outputs to another folder."""

import argparse
import os
from pathlib import Path

import logging

os.environ.setdefault("VLLM_API_BASE", "http://chandra-vllm:8000/v1")
os.environ.setdefault("VLLM_MODEL_NAME", "chandra")

from scirex.ocr.modern.pipeline import pdf_to_images, images_to_markdown, save_markdown, output_path_for, stats_path_for, save_stats

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler("ocr.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

def main():
    # 1. Parse arguments
    parser = argparse.ArgumentParser(description="Batch OCR PDFs with Chandra.")
    parser.add_argument("--input-dir", type=Path, required=True, help="Folder containing PDFs.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Folder where markdown files will be written.")
    parser.add_argument("--limit", type=int, default=None, help="Optional: process only the first N PDFs (for testing).")
    args = parser.parse_args()
    
    # 2. List the PDFs
    pdf_paths = sorted(args.input_dir.glob("*.pdf"))
    if args.limit:
        pdf_paths = pdf_paths[:args.limit]
    
    logger.info(f"Found {len(pdf_paths)} PDFs in {args.input_dir}")
    
    # 3. Loop over each PDF
    for i, pdf_path in enumerate(pdf_paths, start=1):
        output_path = output_path_for(pdf_path, output_dir= args.output_dir)
        if output_path.exists():
            logger.info(f"[{i}/{len(pdf_paths)}] Skipping {pdf_path.name} (already done)")
            continue
        try:
            pages = pdf_to_images(pdf_path)
            md, stats = images_to_markdown(pages)
            saved_path = save_markdown(md, pdf_path, args.output_dir)
            stats_csv = save_stats(stats, pdf_path, args.output_dir)
            logger.info(f"[{i}/{len(pdf_paths)}] ✓ Wrote {saved_path.name} + stats ({len(pages)} pages)")

        except Exception:
            logger.exception(f"[{i}/{len(pdf_paths)}] ✗ Failed to process {pdf_path.name}")
            
if __name__ == "__main__":
    main()