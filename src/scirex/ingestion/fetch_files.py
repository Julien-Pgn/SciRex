"""
arXiv file fetcher: downloads PDFs and LaTeX source tarballs for the
benchmark subset selected in the DuckDB `benchmark_subset` table.

Failures per paper are logged but do not stop the run.
Resumable: re-running skips papers whose `pdf_path` is already set in `papers`.

Usage:
    python -m scirex.ingestion.fetch_files --table benchmark_subset
"""
import argparse
import logging
import time
import traceback
from pathlib import Path
from typing import Optional

import duckdb
import requests
from requests.exceptions import RequestException, HTTPError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

# Configuration
DB_PATH = Path("data/arxiv_metadata.duckdb")
PDF_DIR = Path("data/raw/pdfs")
SRC_DIR = Path("data/raw/sources")
LOG_DIR = Path("data/logs")

# Polite identifier — arXiv asks API users to identify themselves
HEADERS = {"User-Agent": "SciRex/0.1 (https://github.com/Julien-Pgn/scirex)"}

# Pause between papers (arXiv asks for >= 3s between requests)
SLEEP_BETWEEN_PAPERS = 3

# Shared retry policy for any HTTP-flavored I/O against arXiv
HTTP_RETRY = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=2, max=10),
    retry=retry_if_exception_type(RequestException),
    reraise=True,
)


# Fetch the pdfs
@HTTP_RETRY
def fetch_pdf(arxiv_id: str, output_dir: Path) -> Path:
    """Download a paper's PDF. Returns the local path on success."""
    url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    pdf_path = output_dir / f"{arxiv_id}.pdf"

    with requests.get(url, stream=True, timeout=60, headers=HEADERS) as response:
        response.raise_for_status()
        with open(pdf_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

    return pdf_path

# Fetch the LaTeX source tarballs
@HTTP_RETRY
def fetch_source(arxiv_id: str, output_dir: Path) -> Path:
    """Download a paper's LaTeX source tarball."""
    url = f"https://arxiv.org/e-print/{arxiv_id}"
    latex_path = output_dir / f"{arxiv_id}.tar.gz"

    with requests.get(url, stream=True, timeout=60, headers=HEADERS) as response:
        response.raise_for_status()
        with open(latex_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

    return latex_path


# Get the list of ArXiv papers to dowload
def get_papers_to_fetch(conn: duckdb.DuckDBPyConnection, table: str) -> list[str]:
    """Return arxiv_ids from 'table' whose pdf_path is still NULL in paper_local."""

    rows = conn.execute(f"""
        SELECT s.arxiv_id
        FROM {table} s
        LEFT JOIN paper_local pl USING (arxiv_id)
        WHERE pl.pdf_path IS NULL
        ORDER BY s.arxiv_id
    """).fetchall()
    return [row[0] for row in rows]

# Update the db with the paths of the fetched files when done
def update_paper_paths(conn: duckdb.DuckDBPyConnection, arxiv_id: str, pdf_path: Path, source_path: Path, table: str) -> None:
    """Mark a paper as fetched by writing its file paths into the paper_local table."""
    conn.execute(
        """
        INSERT INTO paper_local (arxiv_id, pdf_path, latex_source_path, ocr_reason)
        VALUES (?, ?, ?, ?)
        ON CONFLICT (arxiv_id) DO UPDATE SET
            pdf_path = EXCLUDED.pdf_path,
            latex_source_path = EXCLUDED.latex_source_path,
            ocr_reason = EXCLUDED.ocr_reason
        """,
        [arxiv_id, str(pdf_path), str(source_path), str(table)],  
    )

# Main loop:
def main() -> None:
    """Fetch PDFs and LaTeX sources for the benchmark subset."""

    parser = argparse.ArgumentParser(description="Fetch PDFs/LaTeX for a subset table.")
    parser.add_argument(
        "--table",
        required=True ,
        help="DuckDB table listing all papers to download (must exist in the database).",
    )
    args: argparse.Namespace = parser.parse_args()

    # Security
    if not args.table.isidentifier():
        raise ValueError(f"Invalid table name: {args.table}")

    # Create all necessary directories if they don't exist yet
    for d in (PDF_DIR, SRC_DIR, LOG_DIR):
        d.mkdir(parents=True, exist_ok=True)

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(LOG_DIR / "fetch.log"),
            logging.StreamHandler(),
        ],
    )
    log = logging.getLogger(__name__)

    with duckdb.connect(DB_PATH) as conn:

        # Verify that the table exists
        exists = conn.execute("""
            SELECT 1 FROM information_schema.tables WHERE table_name = ?""", [args.table],).fetchone() is not None
        if not exists:
            log.error(f"Table '{args.table}' does not exist in the database. Exiting.")
            raise ValueError(f"Table '{args.table}' does not exist in the database.")

        arxiv_ids = get_papers_to_fetch(conn, args.table)
        n = len(arxiv_ids)
        log.info(f"Starting fetch run: {n} papers to download")

        n_ok, n_fail = 0, 0
        for i, arxiv_id in enumerate(arxiv_ids, start=1):
            try:
                pdf_path = fetch_pdf(arxiv_id, PDF_DIR)
                source_path = fetch_source(arxiv_id, SRC_DIR)
                update_paper_paths(conn, arxiv_id, pdf_path, source_path, args.table)
                n_ok += 1
                log.info(f"[{i}/{n}] OK   {arxiv_id}")
            except Exception as e:
                n_fail += 1
                log.error(f"[{i}/{n}] FAIL {arxiv_id}: {type(e).__name__}: {e}\n{traceback.format_exc()}")
            time.sleep(SLEEP_BETWEEN_PAPERS)

        log.info(f"Done. Success: {n_ok}, Failed: {n_fail}")

        # Prints a summary to the console: for me to monitor progress without opening the log file. 
        total = n_ok + n_fail
        print()
        print("=" * 60)
        print(f"  FETCH RUN COMPLETE")
        print("=" * 60)
        print(f"  Total attempted:  {total}")
        print(f"  Successful:       {n_ok}")
        print(f"  Failed:           {n_fail}")
        if total > 0:
            print(f"  Success rate:     {100 * n_ok / total:.1f}%")
        print(f"  Log file:         {LOG_DIR / 'fetch.log'}")
        if n_fail > 0:
            print(f"  Re-run to retry the {n_fail} failed papers (resume is automatic).")
        print("=" * 60)


if __name__ == "__main__":
    main()