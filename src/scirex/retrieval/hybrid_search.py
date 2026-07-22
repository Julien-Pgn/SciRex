"""Pure scoring/fusion functions for hybrid (dense + sparse) retrieval.

No model or DB imports here — embed.py/classify.py call the actual model,
this module only ranks numbers that already exist. That split is what makes
these functions testable without a GPU.
"""

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