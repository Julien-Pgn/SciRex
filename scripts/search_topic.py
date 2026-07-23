"""Hybrid search + LLM classification for one topic. Writes results to topic_subset.

top-k is recall-based by default, not a fixed guess: we look up where the golden
set's known positives actually rank in the full corpus and pick the depth that
catches --target-recall of them. Pass --top-k to override with a fixed number.

Usage:
    python scripts/search_topic.py \
        --topic quantization \
        --query "quantization of neural network weights for efficient inference" \
        --categories cs.LG cs.CL cs.AI cs.CV \
        --rubric-file data/quantization_rubric.txt \
        --golden-set data/interim/quantization_golden_set_v2.parquet
"""

import argparse
import gc
import logging
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import torch

from scirex.retrieval.classify import (
    build_few_shot_block,
    build_prompt,
    classify_paper,
    load_classifier,
    select_few_shot_examples,
)
from scirex.retrieval.embed import load_corpus_slice, load_model
from scirex.retrieval.hybrid_search import dense_scores, recommend_top_k, rrf_fuse, sparse_scores

DB_PATH = Path("data/arxiv_metadata.duckdb")
LOG_DIR = Path("data/logs")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Hybrid search + LLM classification for one topic."
    )
    parser.add_argument("--db", type=Path, default=DB_PATH)
    parser.add_argument(
        "--topic", required=True, help="Topic name, e.g. 'quantization' — key in topic_subset."
    )
    parser.add_argument(
        "--query", required=True, help="Natural-language query for the hybrid search."
    )
    parser.add_argument("--categories", nargs="+", required=True)
    parser.add_argument("--rubric-file", type=Path, required=True)
    parser.add_argument(
        "--golden-set",
        type=Path,
        required=True,
        help="Labeled parquet — also sizes top-k, see below.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Fixed number of hybrid-ranked candidates to classify. If omitted (default), "
        "computed from the golden set: the depth needed to catch --target-recall of its "
        "known positives.",
    )
    parser.add_argument(
        "--target-recall",
        type=float,
        default=0.97,
        help="Used only when --top-k is omitted. Missing a real paper is worse than including "
        "a borderline one, so default high — the LLM classifier does precision cleanup after.",
    )
    parser.add_argument(
        "--n-few-shot",
        type=int,
        default=3,
        help="Positive/negative examples each, from the golden set.",
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(LOG_DIR / "search_topic.log"), logging.StreamHandler()],
    )
    log = logging.getLogger(__name__)

    conn = duckdb.connect(str(args.db))
    conn.execute("CREATE SEQUENCE IF NOT EXISTS seq_topic_run_id START 1")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS topic_subset (
            topic         VARCHAR,
            arxiv_id      VARCHAR,
            hybrid_score  DOUBLE,
            llm_verdict   BOOLEAN,
            rank          INTEGER,
            run_id        INTEGER,
            created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (topic, arxiv_id)
        )
    """)
    run_id = conn.execute("SELECT nextval('seq_topic_run_id')").fetchone()[0]

    try:
        # --- Stage 1: hybrid search over the whole category slice ---
        log.info(f"Loading embeddings for categories {args.categories}...")
        corpus = load_corpus_slice(conn, args.categories)
        log.info(f"{len(corpus)} embedded papers in the category slice.")
        if len(corpus) == 0:
            log.error(
                "No embedded papers found — run embed_abstracts.py for these categories first."
            )
            return

        doc_dense = np.array(corpus["dense_vec"].tolist())
        doc_sparse = corpus["sparse_weights"].tolist()

        log.info("Loading BGE-M3 to embed the query...")
        bge_model = load_model(device=args.device)
        q_out = bge_model.encode([args.query], return_dense=True, return_sparse=True)
        query_dense = q_out["dense_vecs"][0]
        query_sparse = q_out["lexical_weights"][0]

        d_scores = dense_scores(query_dense, doc_dense)
        s_scores = sparse_scores(query_sparse, doc_sparse, bge_model.compute_lexical_matching_score)
        fused = rrf_fuse(d_scores, s_scores)

        # Free BGE-M3's VRAM before loading the LLM — same as the notebook (cell 22)
        del bge_model
        gc.collect()
        torch.cuda.empty_cache()

        golden = pd.read_parquet(args.golden_set).dropna(subset=["label"])
        golden["label"] = golden["label"].astype(int)

        if args.top_k is not None:
            top_k = args.top_k
            log.info(f"Using fixed top_k={top_k} (--top-k override).")
        else:
            top_k = recommend_top_k(
                fused,
                corpus["arxiv_id"].tolist(),
                golden["arxiv_id"].tolist(),
                golden["label"].tolist(),
                target_recall=args.target_recall,
            )
            log.info(
                f"Recall-based sizing: top_k={top_k} needed to catch "
                f"{args.target_recall:.0%} of the golden set's known positives."
            )

        top_idx = np.argsort(-fused)[:top_k]
        candidates = corpus.iloc[top_idx].copy().reset_index(drop=True)
        candidates["hybrid_score"] = fused[top_idx]
        log.info(f"Top {len(candidates)} candidates from hybrid search.")

        # --- Stage 2: LLM classification pass ---
        rubric = args.rubric_file.read_text()
        few_shot_df = select_few_shot_examples(
            golden, n_positive=args.n_few_shot, n_negative=args.n_few_shot
        )
        few_shot_examples = [
            (row["title"], row["abstract"], bool(row["label"])) for _, row in few_shot_df.iterrows()
        ]
        few_shot_block = build_few_shot_block(few_shot_examples)

        log.info("Loading Qwen2.5-7B-Instruct classifier...")
        tokenizer, llm = load_classifier(device=args.device)

        verdicts = []
        for i, row in enumerate(candidates.itertuples(), start=1):
            try:
                prompt = build_prompt(rubric, few_shot_block, row.title, row.abstract)
                verdicts.append(classify_paper(tokenizer, llm, prompt))
            except Exception:
                log.exception(f"Classification failed for {row.arxiv_id}")
                verdicts.append(None)
            if i % 50 == 0:
                log.info(f"Classified {i}/{len(candidates)}")
        candidates["llm_verdict"] = verdicts
        candidates["rank"] = range(1, len(candidates) + 1)

        # --- Write results: replace this topic's previous rows ---
        conn.execute("DELETE FROM topic_subset WHERE topic = ?", [args.topic])
        rows = [
            (args.topic, r.arxiv_id, r.hybrid_score, r.llm_verdict, r.rank, run_id)
            for r in candidates.itertuples()
        ]
        conn.executemany(
            """
            INSERT INTO topic_subset (topic, arxiv_id, hybrid_score, llm_verdict, rank, run_id)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
    finally:
        conn.close()

    n_positive = sum(v is True for v in candidates["llm_verdict"])
    print()
    print("=" * 60)
    print("  TOPIC SEARCH COMPLETE")
    print("=" * 60)
    print(f"  Topic:              {args.topic}")
    print(f"  Candidates ranked:  {len(candidates)}")
    print(f"  LLM verdict=yes:    {n_positive}")
    print(f"  run_id:             {run_id}")
    print(f"  Log file:           {LOG_DIR / 'search_topic.log'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
