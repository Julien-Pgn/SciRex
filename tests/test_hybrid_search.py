import numpy as np

from scirex.retrieval.hybrid_search import dense_scores, rrf_fuse, sparse_scores


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