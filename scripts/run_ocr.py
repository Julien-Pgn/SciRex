"""Run Chandra OCR on all PDFs in a folder and write markdown outputs to another folder."""

import argparse
import os
from pathlib import Path

os.environ.setdefault("VLLM_API_BASE", "http://chandra-vllm:8000/v1")
os.environ.setdefault("VLLM_MODEL_NAME", "chandra")

from scirex.ocr.modern.pipeline import pdf_to_images, images_to_markdown, save_markdown


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
    
    print(f"Found {len(pdf_paths)} PDFs in {args.input_dir}")
    
    # 3. Loop over each PDF
    for i, pdf_path in enumerate(pdf_paths, start=1):
        print(f"[{i}/{len(pdf_paths)}] Processing {pdf_path.name}...")
        try:
            pages = pdf_to_images(pdf_path)
            md, stats = images_to_markdown(pages)
            saved_path = save_markdown(md, pdf_path, args.output_dir)
            print(f"  ✓ Wrote {saved_path} ({len(pages)} pages)")
        except Exception as e:
            print(f"  ✗ FAILED on {pdf_path.name}: {type(e).__name__}: {e}")

if __name__ == "__main__":
    main()