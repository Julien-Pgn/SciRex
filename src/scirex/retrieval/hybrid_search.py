"""Pure scoring/fusion functions for hybrid (dense + sparse) retrieval.

No model or DB imports here — embed.py/classify.py call the actual model,
this module only ranks numbers that already exist. That split is what makes
these functions testable without a GPU.
"""

import math
from collections.abc import Callable

import numpy as np


def dense_scores(query_dense: np.ndarray, doc_dense: np.ndarray) -> np.ndarray:
    """Similarity between one query vector and every doc's dense vector.

    BGE-M3's dense output is already L2-normalized, so a plain dot product
    IS the cosine similarity — no extra normalization step needed here.
    """
    return doc_dense @ query_dense


def sparse_scores(
    query_sparse: dict[int, float],
    doc_sparse: list[dict[int, float]],
    lexical_match_fn: Callable[[dict[int, float], dict[int, float]], float],
) -> np.ndarray:
    """Lexical-matching score between one query and every doc's sparse weights.

    lexical_match_fn is injected (pass BGEM3FlagModel.compute_lexical_matching_score
    at the call site) so this module never has to import the embedding model.
    """
    return np.array([lexical_match_fn(query_sparse, doc) for doc in doc_sparse])


def rrf_fuse(*score_arrays: np.ndarray, k: int = 60) -> np.ndarray:
    """Reciprocal Rank Fusion: combine several same-length score arrays into one ranking.

    Each channel is converted to a rank first (0 = best), then summed as
    1 / (k + rank + 1). Fusing on RANK rather than raw score is what lets us
    combine dense scores (roughly -1..1) with sparse scores (unbounded, often
    much bigger numbers) without one channel dominating just because its
    numbers happen to be larger.
    """
    fused = np.zeros(len(score_arrays[0]))
    for scores in score_arrays:
        ranks = np.argsort(np.argsort(-scores))  # 0 = best in that channel
        fused += 1.0 / (k + ranks + 1)
    return fused


def recommend_top_k(
    fused_scores: np.ndarray,
    corpus_arxiv_ids: list[str],
    golden_arxiv_ids: list[str],
    golden_labels: list[int],
    target_recall: float = 0.97,
) -> int:
    """How deep into the hybrid ranking to go to catch `target_recall` of the
    golden set's known-positive papers, instead of guessing a fixed top-k.

    Only golden papers that are actually part of the scored corpus count —
    anything else is skipped. This has to be their rank in the FULL corpus
    ranking, not a re-ranking among just the golden set: RRF fuses on rank,
    so a paper's rank among ~100 curated papers is a different number than
    its rank among 400,000 real ones, even for the same underlying score.
    """
    order_by_score_desc = [corpus_arxiv_ids[i] for i in np.argsort(-fused_scores)]
    rank_of_id = {arxiv_id: rank for rank, arxiv_id in enumerate(order_by_score_desc)}

    positive_ranks = sorted(
        rank_of_id[arxiv_id]
        for arxiv_id, label in zip(golden_arxiv_ids, golden_labels, strict=True)
        if label == 1 and arxiv_id in rank_of_id
    )
    if not positive_ranks:
        raise ValueError(
            "None of the golden set's positive papers were found in the scored corpus."
        )

    n_needed = min(math.ceil(target_recall * len(positive_ranks)), len(positive_ranks))
    return positive_ranks[n_needed - 1] + 1
