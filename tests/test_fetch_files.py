from pathlib import Path

import duckdb
import pytest

from scirex.ingestion.fetch_files import get_papers_to_fetch, update_paper_paths


@pytest.fixture
def conn():
    connection = duckdb.connect(":memory:")
    connection.execute("CREATE TABLE benchmark_subset (arxiv_id VARCHAR)")
    connection.execute(
        """
        CREATE TABLE paper_local (
            arxiv_id VARCHAR PRIMARY KEY,
            pdf_path VARCHAR,
            latex_source_path VARCHAR,
            keyword_for_ocr VARCHAR
        )
        """
    )
    connection.execute(
        "INSERT INTO benchmark_subset VALUES ('1111.1111'), ('2222.2222'), ('3333.3333')"
    )
    yield connection
    connection.close()


def test_get_papers_to_fetch_returns_papers_without_pdf_path(conn):
    conn.execute(
        "INSERT INTO paper_local (arxiv_id, pdf_path) VALUES ('1111.1111', '/some/path.pdf')"
    )
    assert get_papers_to_fetch(conn, "benchmark_subset") == ["2222.2222", "3333.3333"]


def test_get_papers_to_fetch_returns_all_when_none_fetched(conn):
    assert get_papers_to_fetch(conn, "benchmark_subset") == ["1111.1111", "2222.2222", "3333.3333"]


def test_update_paper_paths_inserts_new_row(conn):
    update_paper_paths(
        conn,
        "1111.1111",
        Path("/pdf/1111.1111.pdf"),
        Path("/src/1111.1111.tar.gz"),
        "benchmark_subset",
    )
    row = conn.execute(
        "SELECT pdf_path, latex_source_path, keyword_for_ocr FROM paper_local WHERE arxiv_id = '1111.1111'"
    ).fetchone()
    assert row == ("/pdf/1111.1111.pdf", "/src/1111.1111.tar.gz", "benchmark_subset")


def test_update_paper_paths_upserts_existing_row(conn):
    update_paper_paths(
        conn, "1111.1111", Path("/pdf/old.pdf"), Path("/src/old.tar.gz"), "benchmark_subset"
    )
    update_paper_paths(
        conn, "1111.1111", Path("/pdf/new.pdf"), Path("/src/new.tar.gz"), "benchmark_subset"
    )
    rows = conn.execute("SELECT COUNT(*) FROM paper_local WHERE arxiv_id = '1111.1111'").fetchone()
    assert rows[0] == 1
    row = conn.execute("SELECT pdf_path FROM paper_local WHERE arxiv_id = '1111.1111'").fetchone()
    assert row == ("/pdf/new.pdf",)


@pytest.fixture
def topic_conn():
    connection = duckdb.connect(":memory:")
    connection.execute("""
        CREATE TABLE topic_subset (
            topic VARCHAR,
            arxiv_id VARCHAR,
            hybrid_score DOUBLE,
            llm_verdict BOOLEAN,
            rank INTEGER,
            run_id INTEGER
        )
        """)
    connection.execute(
        """
        CREATE TABLE paper_local (
            arxiv_id VARCHAR PRIMARY KEY,
            pdf_path VARCHAR,
            latex_source_path VARCHAR,
            keyword_for_ocr VARCHAR
        )
        """
    )
    connection.execute("""
        INSERT INTO topic_subset (topic, arxiv_id, llm_verdict) VALUES
            ('quantization', '1111.1111', TRUE),
            ('quantization', '2222.2222', FALSE),
            ('quantization', '3333.3333', NULL),
            ('genai', '4444.4444', TRUE)
        """)
    yield connection
    connection.close()


def test_get_papers_to_fetch_with_topic_filters_to_matching_topic_and_true_verdict(topic_conn):
    assert get_papers_to_fetch(topic_conn, "topic_subset", topic="quantization") == ["1111.1111"]


def test_get_papers_to_fetch_with_topic_excludes_other_topics_even_if_verdict_true(topic_conn):
    result = get_papers_to_fetch(topic_conn, "topic_subset", topic="genai")
    assert result == ["4444.4444"]


def test_get_papers_to_fetch_with_topic_excludes_already_fetched(topic_conn):
    topic_conn.execute(
        "INSERT INTO paper_local (arxiv_id, pdf_path) VALUES ('1111.1111', '/some/path.pdf')"
    )
    assert get_papers_to_fetch(topic_conn, "topic_subset", topic="quantization") == []


def test_get_papers_to_fetch_without_topic_ignores_verdict_and_topic(topic_conn):
    # No --topic -> every row is eligible regardless of topic or llm_verdict.
    # This is the pre-existing behavior for single-topic tables like
    # benchmark_subset; documented here since it's exactly why forgetting
    # --topic against topic_subset would be wrong (it would fetch PDFs the
    # LLM said "no" to, and mix topics together).
    result = get_papers_to_fetch(topic_conn, "topic_subset")
    assert set(result) == {"1111.1111", "2222.2222", "3333.3333", "4444.4444"}
