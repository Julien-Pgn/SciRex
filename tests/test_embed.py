import duckdb
import numpy as np
import pytest

from scirex.retrieval.embed import (
    embed_texts,
    get_unembedded_abstracts,
    load_corpus_slice,
    save_embeddings,
)


def test_embed_texts_batches_and_concatenates():
    texts = ["a", "b", "c", "d", "e"]
    calls = []

    def fake_encode(batch):
        calls.append(batch)
        return {
            "dense_vecs": np.ones((len(batch), 2)),
            "lexical_weights": [{1: 0.1}] * len(batch),
        }

    dense, sparse = embed_texts(texts, fake_encode, batch_size=2)

    assert calls == [["a", "b"], ["c", "d"], ["e"]]
    assert dense.shape == (5, 2)
    assert len(sparse) == 5


def test_embed_texts_handles_empty_input():
    dense, sparse = embed_texts(
        [], lambda batch: {"dense_vecs": np.empty((0, 2)), "lexical_weights": []}
    )
    assert dense.shape[0] == 0
    assert sparse == []


@pytest.fixture
def conn():
    connection = duckdb.connect(":memory:")
    connection.execute("""
        CREATE TABLE papers (arxiv_id VARCHAR, title VARCHAR, abstract VARCHAR, primary_category VARCHAR)
    """)
    connection.execute("""
        CREATE TABLE abstract_embeddings (
            arxiv_id VARCHAR PRIMARY KEY, dense_vec FLOAT[4],
            sparse_weights MAP(INTEGER, FLOAT), embed_model VARCHAR,
            embed_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    connection.execute("""
        INSERT INTO papers VALUES
            ('1111.1111', 'A', 'abstract a', 'cs.LG'),
            ('2222.2222', 'B', 'abstract b', 'cs.LG'),
            ('3333.3333', 'C', 'abstract c', 'cs.CV')
    """)
    yield connection
    connection.close()


def test_get_unembedded_abstracts_filters_category_and_already_embedded(conn):
    conn.execute(
        "INSERT INTO abstract_embeddings (arxiv_id, embed_model) VALUES ('1111.1111', 'test-model')"
    )
    result = get_unembedded_abstracts(conn, ["cs.LG"])
    assert result["arxiv_id"].tolist() == ["2222.2222"]


def test_save_embeddings_writes_rows(conn):
    save_embeddings(conn, ["1111.1111"], np.array([[0.1, 0.2, 0.3, 0.4]]), [{1: 0.5}], "test-model")
    row = conn.execute(
        "SELECT arxiv_id, sparse_weights, embed_model FROM abstract_embeddings WHERE arxiv_id = '1111.1111'"
    ).fetchone()
    assert row == ("1111.1111", {1: 0.5}, "test-model")


def test_save_embeddings_casts_numpy_float16_sparse_weights(conn):
    """Regression test: BGE-M3's use_fp16=True means lexical_weights dict values
    come back as numpy.float16, which DuckDB's MAP binding can't convert directly."""
    fp16_sparse = {np.int64(1): np.float16(0.5)}
    save_embeddings(
        conn, ["1111.1111"], np.array([[0.1, 0.2, 0.3, 0.4]]), [fp16_sparse], "test-model"
    )
    row = conn.execute(
        "SELECT sparse_weights FROM abstract_embeddings WHERE arxiv_id = '1111.1111'"
    ).fetchone()
    assert row[0] == pytest.approx({1: 0.5})


def test_save_embeddings_is_idempotent_on_rerun(conn):
    save_embeddings(conn, ["1111.1111"], np.array([[0.1, 0.2, 0.3, 0.4]]), [{1: 0.5}], "test-model")
    save_embeddings(conn, ["1111.1111"], np.array([[0.9, 0.9, 0.9, 0.9]]), [{2: 0.9}], "test-model")
    count = conn.execute("SELECT COUNT(*) FROM abstract_embeddings").fetchone()[0]
    assert count == 1


def test_load_corpus_slice_joins_embeddings_with_paper_text(conn):
    save_embeddings(conn, ["1111.1111"], np.array([[0.1, 0.2, 0.3, 0.4]]), [{1: 0.5}], "test-model")
    result = load_corpus_slice(conn, ["cs.LG"])
    assert result["arxiv_id"].tolist() == ["1111.1111"]
    assert result["title"][0] == "A"
    np.testing.assert_allclose(result["dense_vec"][0], [0.1, 0.2, 0.3, 0.4])
