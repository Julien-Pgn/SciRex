"""BGE-M3 embedding: batches abstracts through the model and persists them to DuckDB.

load_model imports FlagEmbedding lazily (inside the function) so importing this
module — and testing embed_texts/get_unembedded_abstracts/save_embeddings — never
requires FlagEmbedding/torch to be installed.
"""

from collections.abc import Callable

import duckdb
import numpy as np
import pandas as pd


def load_model(device: str = "cuda:0"):
    """Load BGE-M3. Not unit-tested — needs real GPU weights, exercised manually."""
    from FlagEmbedding import BGEM3FlagModel

    return BGEM3FlagModel(
        "BAAI/bge-m3", use_fp16=True, device=device, max_seq_len=2048, use_cache=True
    )


def embed_texts(
    texts: list[str],
    encode_fn: Callable[[list[str]], dict],
    batch_size: int = 64,
) -> tuple[np.ndarray, list[dict[int, float]]]:
    """Batch texts through encode_fn, concatenating dense vecs and collecting sparse weights.

    encode_fn should be model.encode with return_dense=True, return_sparse=True
    already bound — injected so this function has no FlagEmbedding dependency
    and is testable with a fake encoder (same trick as sparse_scores in hybrid_search.py).
    """
    all_dense, all_sparse = [], []
    for i in range(0, len(texts), batch_size):
        out = encode_fn(texts[i : i + batch_size])
        all_dense.append(out["dense_vecs"])
        all_sparse.extend(out["lexical_weights"])
    dense = np.concatenate(all_dense, axis=0) if all_dense else np.empty((0, 0))
    return dense, all_sparse


def get_unembedded_abstracts(
    conn: duckdb.DuckDBPyConnection, categories: list[str]
) -> pd.DataFrame:
    """Return arxiv_id/title/abstract for papers in `categories` not yet in abstract_embeddings."""
    placeholders = ", ".join("?" * len(categories))
    return conn.execute(
        f"""
        SELECT p.arxiv_id, p.title, p.abstract
        FROM papers p
        LEFT JOIN abstract_embeddings ae USING (arxiv_id)
        WHERE p.primary_category IN ({placeholders})
          AND ae.arxiv_id IS NULL
        ORDER BY p.arxiv_id
        """,
        categories,
    ).fetchdf()


def save_embeddings(
    conn: duckdb.DuckDBPyConnection,
    arxiv_ids: list[str],
    dense: np.ndarray,
    sparse: list[dict[int, float]],
    embed_model: str,
) -> None:
    """Write one row per paper via a bulk columnar insert.

    ON CONFLICT DO NOTHING (not DO UPDATE, unlike paper_local): an embedding is
    an immutable fact about a point in time. Re-embedding with a different model
    should be a deliberate, explicit operation later, not something a plain rerun
    silently overwrites.

    BGE-M3 runs in fp16 (load_model uses use_fp16=True for GPU speed), so
    sparse dict values come back as numpy.float16 — DuckDB's MAP binding can't
    convert that dtype (unlike its array/LIST binding, which handles fp16 fine
    via .tolist()), so we cast each value to a plain Python float explicitly.

    Bulk "INSERT ... SELECT FROM df" instead of executemany: row-by-row parameter
    binding of 1024-float arrays + ~100-entry MAP dicts measured at ~20-40s per
    500 rows (this was the actual bottleneck behind slow embed_abstracts.py runs,
    not the GPU) — the columnar bulk insert does the same in well under a second.
    """
    df = pd.DataFrame(  # noqa: F841 -- referenced by name in the SQL below (DuckDB replacement scan)
        {
            "arxiv_id": arxiv_ids,
            "dense_vec": [dense_vec.tolist() for dense_vec in dense],
            "sparse_weights": [
                {int(token_id): float(weight) for token_id, weight in sparse_weights.items()}
                for sparse_weights in sparse
            ],
            "embed_model": embed_model,
        }
    )
    conn.execute("""
        INSERT INTO abstract_embeddings (arxiv_id, dense_vec, sparse_weights, embed_model)
        SELECT arxiv_id, dense_vec, sparse_weights, embed_model FROM df
        ON CONFLICT (arxiv_id) DO NOTHING
    """)


def load_corpus_slice(conn: duckdb.DuckDBPyConnection, categories: list[str]) -> pd.DataFrame:
    """Load every embedded paper in `categories`: arxiv_id, title, abstract, dense_vec, sparse_weights.

    This is what search_topic.py scores the query against — only papers that
    have already been through embed_abstracts.py show up here.
    """
    placeholders = ", ".join("?" * len(categories))
    return conn.execute(
        f"""
        SELECT p.arxiv_id, p.title, p.abstract, ae.dense_vec, ae.sparse_weights
        FROM abstract_embeddings ae
        JOIN papers p USING (arxiv_id)
        WHERE p.primary_category IN ({placeholders})
        """,
        categories,
    ).fetchdf()
