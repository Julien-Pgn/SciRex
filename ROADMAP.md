# Roadmap

# Roadmap

Critical path: **pipeline state → OCR corpus → chunking + vector store → retrieval eval → agent layer → digest + MCP**.
Everything off this path is parked until the path is shippable end-to-end.

| Phase | Scope                                                                                          | Status      |
| ----- | ---------------------------------------------------------------------------------------------- | ----------- |
| 0     | Environment, repo, CI (workflow under `.github/workflows/`, Python 3.12)                       | done        |
| 1     | Ingestion (Kaggle dump → DuckDB, arXiv fetcher, LaTeX extraction)                              | done        |
| 2     | Pipeline state in DuckDB: run journal, per-paper stage tracking, idempotent stages, backfill   | in progress |
| 3     | OCR corpus: Chandra-2 via vLLM on ~900 papers (truncation-flagged); bootstrap RAG with HF 27k  | todo        |
| 4     | Chunking + vector store: markdown-aware chunker, `chunks` table, Qdrant hybrid (BGE-M3), golden eval set, Recall@k / MRR | todo |
| 5     | Agent layer: tools over DuckDB + Qdrant (sql_query, vector_search, get_paper, get_repo), local LLM, RAGAS end-to-end eval | todo |
| 6     | Intelligence layer: weekly delta ingest + interest-profile digest; expose agent as MCP server  | todo        |
| 7     | Parked: OCR benchmark (OlmOCRBench, 5 engines), knowledge graph (Semantic Scholar), Streamlit demo, GPT-from-scratch (→ separate repo) | parked |
| 8     | Polish: blog posts, architecture diagram, demo video                                           | ongoing     |