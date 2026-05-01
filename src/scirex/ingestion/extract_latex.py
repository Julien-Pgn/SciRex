"""
After downloading the PDFs and the LaTeX sources, let's unzip them to build our benchmarking database.
Updates which arXiv papers have LaTeX source available in our DuckDB.

Usage:
In the container run: python -m scirex.ingestion.extract_latex
"""

import logging
import tarfile
from pathlib import Path

import duckdb


# Configuration
DB_PATH = "data/arxiv_metadata.duckdb"
SRC_DIR = Path("data/raw/sources")
LOG_DIR = Path("data/logs")
OUT_DIR = Path("data/interim/latex")


def get_papers_to_unzip(conn: duckdb.DuckDBPyConnection) -> list[tuple[str, str]]:
    """Return (arxiv_id, latex_source_path) pairs to extract."""
    return conn.execute("""
        SELECT bs.arxiv_id, p.latex_source_path
        FROM benchmark_subset bs
        JOIN papers p USING (arxiv_id)
        WHERE p.latex_source_path IS NOT NULL AND p.latex_extracted = FALSE
        ORDER BY bs.arxiv_id
    """).fetchall()


def extract_latex(arxiv_id: str, source_path: Path, output_dir: Path) -> bool:
    """Extract a tar.gz source archive. Returns True if at least one .tex file was found."""
    target = output_dir / arxiv_id
    target.mkdir(parents=True, exist_ok=True)

    # tarfile.open with mode "r:*" auto-detects compression (gz, bz2, xz, plain).
    # filter="data" prevents malicious tarballs from writing outside `target`.
    with tarfile.open(source_path, "r:*") as tar:
        tar.extractall(target, filter="data")

    # rglob walks the directory recursively. any() short-circuits on first match.
    return any(target.rglob("*.tex"))


def main() -> None:
    """Extract LaTeX sources and update the DB accordingly."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(LOG_DIR / "extract.log"),
            logging.StreamHandler(),
        ],
    )
    log = logging.getLogger(__name__)

    with duckdb.connect(DB_PATH) as conn:
        rows = get_papers_to_unzip(conn)
        n = len(rows)
        log.info(f"Starting extraction: {n} archives to process")

        n_tex, n_no_tex, n_fail = 0, 0, 0
        for i, (arxiv_id, source_path) in enumerate(rows, start=1):
            try:
                got_tex = extract_latex(arxiv_id, Path(source_path), OUT_DIR)
                if got_tex:
                    conn.execute(
                        "UPDATE papers SET latex_extracted = TRUE WHERE arxiv_id = ?",
                        [arxiv_id],
                    )
                    n_tex += 1
                    log.info(f"[{i}/{n}] OK   {arxiv_id} (LaTeX)")
                else:
                    n_no_tex += 1
                    log.warning(f"[{i}/{n}] NOTEX {arxiv_id} (extracted, no .tex)")
            except Exception as e:
                n_fail += 1
                log.error(f"[{i}/{n}] FAIL {arxiv_id}: {type(e).__name__}: {e}")

        total = n_tex + n_no_tex + n_fail
        print()
        print("=" * 60)
        print("  LATEX EXTRACTION COMPLETE")
        print("=" * 60)
        print(f"  Processed:        {total}")
        print(f"  With LaTeX:       {n_tex}")
        print(f"  No .tex found:    {n_no_tex}")
        print(f"  Failed:           {n_fail}")
        print(f"  Log file:         {LOG_DIR / 'extract.log'}")
        print("=" * 60)


if __name__ == "__main__":
    main()