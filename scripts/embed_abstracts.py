"""Batch-embed paper abstracts (title + abstract) into abstract_embeddings.

Idempotent: only embeds papers in --categories that don't already have a row
in abstract_embeddings. Saves every --save-every papers, so a crash partway
through only loses that chunk's GPU compute, not the whole run.

Usage:
    python scripts/embed_abstracts.py --categories cs.LG cs.CL cs.AI cs.CV
"""

import argparse
import logging
from functools import partial
from pathlib import Path

import duckdb

from scirex.retrieval.embed import (
    embed_texts,
    get_unembedded_abstracts,
    load_model,
    save_embeddings,
)

EMBED_MODEL = "BAAI/bge-m3"

DB_PATH = Path("data/arxiv_metadata.duckdb")
LOG_DIR = Path("data/logs")

DDL = """
CREATE TABLE IF NOT EXISTS abstract_embeddings (
    arxiv_id       VARCHAR PRIMARY KEY,
    dense_vec      FLOAT[1024],
    sparse_weights MAP(INTEGER, FLOAT),
    embed_model    VARCHAR,
    embed_date     TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch-embed abstracts with BGE-M3.")
    parser.add_argument("--db", type=Path, default=DB_PATH, help="DuckDB database file.")
    parser.add_argument(
        "--categories",
        nargs="+",
        required=True,
        help="arXiv categories to embed, e.g. cs.LG cs.CL.",
    )
    parser.add_argument("--batch-size", type=int, default=64, help="Texts per model.encode() call.")
    parser.add_argument(
        "--save-every",
        type=int,
        default=500,
        help="Papers per DB write (crash-resilience granularity).",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Optional: embed only the first N papers."
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(LOG_DIR / "embed.log"), logging.StreamHandler()],
    )
    log = logging.getLogger(__name__)

    conn = duckdb.connect(str(args.db))
    conn.execute(DDL)

    n_done, n_failed = 0, 0
    try:
        todo = get_unembedded_abstracts(conn, args.categories)
        if args.limit:
            todo = todo.head(args.limit)
        n = len(todo)
        log.info(f"Found {n} un-embedded papers in categories {args.categories}")

        if n > 0:
            log.info("Loading BGE-M3...")
            model = load_model(device=args.device)
            encode_fn = partial(model.encode, return_dense=True, return_sparse=True)

            for start in range(0, n, args.save_every):
                chunk = todo.iloc[start : start + args.save_every]
                texts = (chunk["title"] + ". " + chunk["abstract"]).tolist()
                try:
                    dense, sparse = embed_texts(texts, encode_fn, batch_size=args.batch_size)
                    save_embeddings(conn, chunk["arxiv_id"].tolist(), dense, sparse, EMBED_MODEL)
                    n_done += len(chunk)
                    log.info(f"[{n_done}/{n}] embedded and saved")
                except Exception:
                    n_failed += len(chunk)
                    log.exception(f"Failed to embed chunk starting at row {start}")
    finally:
        conn.close()

    print()
    print("=" * 60)
    print("  EMBEDDING RUN COMPLETE")
    print("=" * 60)
    print(f"  Categories:   {args.categories}")
    print(f"  Embedded:     {n_done}")
    print(f"  Failed:       {n_failed}")
    print(f"  Log file:     {LOG_DIR / 'embed.log'}")
    if n_failed > 0:
        print("  Re-run to retry the failed chunks (resume is automatic).")
    print("=" * 60)


if __name__ == "__main__":
    main()
