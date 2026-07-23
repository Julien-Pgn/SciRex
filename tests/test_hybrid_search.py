import numpy as np
import pytest

from scirex.retrieval.hybrid_search import dense_scores, recommend_top_k, rrf_fuse, sparse_scores


def test_dense_scores_is_dot_product():
    query = np.array([1.0, 0.0])
    docs = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])
    result = dense_scores(query, docs)
    np.testing.assert_allclose(result, [1.0, 0.0, 0.5])


def test_sparse_scores_applies_lexical_match_fn_per_document():
    query_sparse = {1: 0.5, 2: 0.3}
    doc_sparse = [{1: 1.0}, {2: 1.0}, {}]

    def fake_match(q, d):
        return sum(q.get(tok, 0.0) * w for tok, w in d.items())

    result = sparse_scores(query_sparse, doc_sparse, fake_match)
    np.testing.assert_allclose(result, [0.5, 0.3, 0.0])


def test_rrf_fuse_ranks_items_that_win_both_channels_highest():
    dense = np.array([0.9, 0.5, 0.1])
    sparse = np.array([0.8, 0.6, 0.05])
    fused = rrf_fuse(dense, sparse, k=60)

    assert fused[0] > fused[1] > fused[2]
    # exact value: item 0 is rank 0 in both channels -> 1/61 + 1/61... wait, 2 channels each contribute 1/(60+0+1)
    assert fused[0] == 2 / 61


def test_rrf_fuse_is_invariant_to_score_scale():
    """The whole point of RRF: only relative order matters, not raw magnitude.

    This is why it's safe to fuse dense scores (~-1..1) with sparse scores
    (unbounded) — rescaling one channel by 1000x must not change the result.
    """
    dense = np.array([0.9, 0.5, 0.1])
    sparse = np.array([0.02, 0.6, 0.8])

    fused = rrf_fuse(dense, sparse)
    fused_rescaled = rrf_fuse(dense * 1000, sparse * 1000)

    np.testing.assert_allclose(fused, fused_rescaled)


# Corpus of 10 papers already in descending-score order, so rank(p_i) == i —
# makes the expected top-k values easy to verify by hand.
CORPUS_IDS = [f"p{i}" for i in range(10)]
FUSED = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05])


def test_recommend_top_k_at_full_recall_reaches_deepest_positive():
    # positives at ranks 1, 3, 7, 9 -> need to go all the way to rank 9 -> top_k = 10
    top_k = recommend_top_k(
        FUSED, CORPUS_IDS, ["p1", "p3", "p7", "p9"], [1, 1, 1, 1], target_recall=1.0
    )
    assert top_k == 10


def test_recommend_top_k_at_partial_recall_uses_fewer_items():
    # target_recall=0.5 of 4 positives needs the 2 shallowest -> ranks 1, 3 -> top_k = 4
    top_k = recommend_top_k(
        FUSED, CORPUS_IDS, ["p1", "p3", "p7", "p9"], [1, 1, 1, 1], target_recall=0.5
    )
    assert top_k == 4


def test_recommend_top_k_skips_golden_papers_missing_from_corpus():
    top_k = recommend_top_k(FUSED, CORPUS_IDS, ["p1", "p99"], [1, 1], target_recall=1.0)
    assert top_k == 2


def test_recommend_top_k_excludes_negatives_even_at_low_rank():
    # p2 (rank 2) is a negative — it must not shrink the required depth
    top_k = recommend_top_k(FUSED, CORPUS_IDS, ["p1", "p2"], [1, 0], target_recall=1.0)
    assert top_k == 2


def test_recommend_top_k_raises_when_no_positives_found_in_corpus():
    with pytest.raises(ValueError):
        recommend_top_k(FUSED, CORPUS_IDS, ["p99"], [1], target_recall=1.0)
