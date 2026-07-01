# PROJECT.md — SciRex working state

Last updated: 2026-07-01
Repo: github.com/Julien-Pgn/scirex

## Current phase
✅ Phase 0 complete. Starting Phase 1 — ingestion.
✅ Phase 1 complete (2026-05-01). See full close-out in Decisions log.

## Environment
- Base image: `nvcr.io/nvidia/pytorch:26.01-py3` (Python 3.12.3, PyTorch 2.10.0a NV-patched, CUDA 13.1, sm_120 supported)
- Image tag: `scirex:dev`
- Hardware: RTX 5070 Ti (Blackwell, 16 GB VRAM)
- Host driver: 590.48.01 (CUDA 13.0) 
- Package manager: `uv` 0.9.24 (preinstalled in NGC). `--system --break-system-packages` for installs.
- Editable install: confirmed working via `_editable_impl_scirex.pth`. Host edits are live in container.

## Decisions log
### Phase 0 (setup)
2026-04-29: Repo named scirex. License MIT (file: LICENSE.txt).
2026-04-29: Base = NGC PyTorch 26.01-py3, NOT NeMo (NeMo adds ~20 GB framework overhead, not needed for single-GPU research).
2026-04-29: NumPy 2.x acceptable — NGC ships numpy==2.1.0 with PyTorch built against it.
2026-04-29: uv for dependency mgmt. pyproject.toml declares loose constraints; uv.lock records exact versions.
2026-04-29: Editable install requires COPY src/ and COPY tests/ BEFORE uv pip install -e . in Dockerfile.
2026-04-29: Dependency groups in pyproject map to phases: dev, training, ocr, llm, graph, rag, demo. [all] is meta-group for full dev container.
2026-04-29: Excluded from pyproject (NGC provides): torch family, transformers tokenizers, datasets, huggingface_hub, einops, safetensors, scipy, scikit-learn, pillow, networkx, beautifulsoup4, pytest, ipykernel, jupyter, tensorboard.
2026-04-29: Skipped Hydra (overkill); using Pydantic Settings. Skipped Makefile (use scripts/). Skipped pre-commit (CI handles linting). Skipped FastAPI (Streamlit is sufficient).

### Phase 1 
#### strategic decisions
2026-04-30: Pivoted away from OAI-PMH async client. Original plan: build a resumable async harvester. Revised plan: bulk Kaggle metadata dump + targeted per-paper API fetches for the benchmark subset.
Rationale: Kaggle dump (~1GB JSONL, ~2.5M papers, weekly refresh, free) covers the metadata layer without rate limits. arXiv API serves only the ~1k benchmark papers we actually need files for.
Cost comparison: 30k API calls × 3s rate limit = 25h of waiting + retry complexity. Kaggle = one-time download.
S3 requester-pays bucket considered and rejected: ~$100 + 1.1TB for content we don't need (HuggingFace already OCR'd 27k papers via Chandra-2).
2026-04-30: ELT over ETL. Stage raw JSONL into DuckDB, transform via SQL, retain staging for re-derivation if schema changes.
2026-04-30: Schema = 5 normalized tables: papers, authors, paper_authors (M:N with order), paper_categories (M:N with is_primary flag), ingestion_runs (lineage).
2026-04-30: arXiv ID as natural primary key (no surrogate keys).
2026-04-30: GitHub URL extraction = regex on abstract + comments (Layer 1, ~30-40% coverage). Layer 2 (Papers With Code cross-reference) deferred to Phase 5.
2026-04-30: Removed arxiv and feedparser from base deps after committing to Kaggle ingestion strategy.
2026-04-30: Author dedup = LOWER(strip_accents(...)). Known limitation: collides homonyms. Acknowledged trade-off, not solved.
2026-04-30: Stratified random sampling for benchmark subset: ROW_NUMBER() OVER (PARTITION BY primary_category ORDER BY HASH(arxiv_id || '_seed42')). Reproducible at code+data level; not wall-clock reproducible (would need Kaggle dump checksum in ingestion_runs for that).
2026-04-30: Typo bug in benchmark_subset selection: cs.RP (doesn't exist) instead of cs.RO. Result: 9 categories × 100 = 900 papers, not 1000. Decision: don't fix. 900 sufficient for benchmarking; regenerating mid-run risks corrupting in-flight data. Documented; future work could create benchmark_subset_v2 adding cs.RO.
2026-04-30: Conference enrichment (NeurIPS, ICLR, etc.) deferred to Phase 5 (Knowledge Graph). Plan: use Semantic Scholar API for arxiv_id → venue/year resolution, NOT HTML scraping. APIs to investigate: OpenReview, DBLP, Semantic Scholar.

#### implementation decisions
2026-04-30: Sync over async for the fetcher. 1k papers × 3s = 50 min. Async would save ~30 min wall-clock and add ~200 lines. Bad trade.
2026-04-30: Retries via tenacity decorator at I/O boundary (fetch_pdf, fetch_source), NOT at orchestration level. Each I/O function self-protects; orchestrator stays thin.
2026-04-30: Retry policy: 3 attempts, exponential backoff (2s, 4s, 8s), only on RequestException. Known wart: HTTPError 4xx is a subclass and gets retried 3x even when pointless. Acceptable at this scale; would refine with custom predicate in production.
2026-04-30: Resumability via DB state (pdf_path IS NULL), not filesystem checks. DB is source of truth; partial files don't lie about completion.
2026-04-30: Best-effort orchestration: per-paper try/except, log failures, continue loop. Failed papers re-enter queue on next run automatically.
2026-04-30: Structured logging with logging.basicConfig + dual handlers (file + stderr). Final summary banner via print for visibility regardless of log level.
2026-04-30: Polite User-Agent identifying project + contact URL on every request to arxiv.org (PDFs and source tarballs both — all programmatic access counts as API consumption).
2026-05-01: LaTeX extraction = tarfile.open("r:*") (auto-detect compression) + extractall(filter="data") (security: prevents path traversal). Single-format approach validated empirically: 92% success on first run.
2026-05-01: Three-state extraction outcome: OK (got .tex, DB updated), NOTEX (extracted, no .tex found, DB not updated → re-eligible), FAIL (exception, DB not updated → re-eligible).

### Phase2a
2026-05-06: Skip this phase for now and focus on OCR

### Phase2b:
2026-05-06: Let's replicate the OCR usage from Landing.AI course on DeepLearning.ai and then move to LLMs and RAG building at the same time to go faster on the learning aspect. 
2026-05-06: add a .env.example file to the project. It should only refer to the keys you will use. Then copy this into .env and add the correct APIs key. Verify you have .env in the .gitignore so your keys are never commited to your repo.
2026-05-07: Add paddleocr in pyproject.toml file and rebuilt the docker image
2026-05-07: Pivoted from PaddleOCR to EasyOCR for the OCR step. PaddleOCR 3.3.x raised persistent Intel MKL dispatch errors on the NGC PyTorch container despite a clean install (paddle.utils.run_check passed, MKL libraries present); after FLAGS_use_mkldnn=0 and verbose tracing produced no fix, EasyOCR (pure PyTorch, no native-library drama) was chosen as the pragmatic substitute for the same architectural role in the agentic pipeline.
2026-05-09: LayoutReader is not that good for getting the reading order of a scientific apper correctly. 
2026-05-09: I will use a YOLO model for detecting the layout from the document as paddle ocr would do. 
2026-05-20: Skip the agentic OCR workflow for now as it is not optimized (recent VLM are much better at it, so let's dive in this directly)
2026-05-22: Problems with Chandra-ocr in my environemnt because transformers was too old. I had to get a newer version and remove marker-pdf because of conflicting versions. 
2026-05-22: Now Chandra 2 works well but is long: 1min for 1 page with HF (transformers with flashattention). Bitsandbytes doesn't work with CUDA 13 so i I used torchao which is already installed in the NGC container. Int8 quantization is fine to run on my GPU. 
2026-05-22: Adding ipywidgets to pyproject.toml for interactive progress bars. 
2026-05-25: Downloaded the Docker image for vLLM: "vllm/vllm-openai:v0.17.0" and created a run_vllm.sh script to run it in jupyter notebook to measure inference time compared to HF. It runs 6 times faster than HF inference mode.
2026-05-26: It is not because of the lack of authentification. I tested it once more but still 1 min per page.
2026-05-26: I modified the run_container.sh files for better understanding and to launch a mode without GPU access so the vLLM container gets it all for OCR. 
2026-05-26: When the NGC container doesn't use the GPU, it takes 7sec per page on average with vllm.
2026-05-27: Runnning OCR on 100 papers with the 2 containers and adding logging, stats saving.  
2026-05-28: Multithreading improves overall performances on large pdf. But no gains when many small pdf: reduction of max workers to increase the number of pdf treated in parallel. 
2026-06-10: pdfs in parallel, 8 max workers and 28 requests at the same time for the GPU works well. 
2026-06-10: Scope cut after external review. Critical path = pipeline state → OCR → chunking → Qdrant → retrieval eval → agent → digest/MCP. Deferred (not deleted): full OCR benchmark, knowledge graph phase, GPT-from-scratch (separate repo), Streamlit demo. Rationale: each deferred item is a project of its own; none blocks the stated goal.
2026-06-10: Vector DB decided: Qdrant over ChromaDB. Reasons: runs as a Docker service (matches the vLLM pattern), payload filtering for metadata-constrained queries, native dense+sparse hybrid in one collection, quantization headroom. DuckDB stays the source of truth for metadata; Qdrant stores chunk vectors + minimal payload keyed on arxiv_id. Never duplicate full metadata into Qdrant.
2026-06-10: Embeddings: BGE-M3 (dense + sparse from one model, 8k context) — validate VRAM fit on the 5070 Ti alongside other loads before locking.
2026-06-10: Bootstrap the RAG with the HF 27k Chandra corpus while own OCR run completes; own 900-paper pipeline is the validation set, not the bottleneck.
2026-06-10: CI fixed: workflow moved to .github/workflows/ (was .github/, so it never ran), Python bumped to 3.12 to match requires-python.
2026-06-16: Pivoting the project by recreating the duckdb and tables for regular updates and getting more informations. Modification of the scripts to run the ocr and update the tables when md and html files are saved. 
2026-06-16: How it works: you select a subset of papers of interest from the papers table and use fetch_files.py to download the pdfs in data/raw then the scripts/run_ocr.py will OCR all the papers found and create new subfolders in data/processed/{arxiv_id}/ to store the md file, the html and all the image for later multimodal RAG. 
2026-07-01: the initial paper selection is not precise so I need to do a hybrid search that combines a keyword and semantic meaning followed by reranking. Then OCR and the proper RAG. 


## Project scope (locked) — v2, 2026-06-10
 
Replan after external review: cut to the critical path that delivers the actual
goal (a queryable scientific knowledge base with an intelligence layer).
Deferred items move to Phase 7 (parked), not deleted.
 
1. **Phase 2 — Pipeline state & idempotency (2-3 days):** DuckDB becomes the single source of truth AND the working journal. `pipeline_runs` (one row per script execution: stage, config JSON, git sha, outcome) + `paper_stage_state` (one row per paper × stage: done/failed, artifact path, detail JSON). All stage scripts select work from the DB, write artifacts atomically (tmp + os.replace), record outcome last. Backfill from existing `papers.pdf_path` / `source_path` and any OCR markdown already on disk. Refactor `run_ocr.py` to this pattern; fetcher migrates later (same helpers).
2. **Phase 3 — OCR corpus (2-3 days):** Finish Chandra-2/vLLM run on ~900 papers with truncation flags (`tokens >= max_output_tokens` recorded per page). In parallel, pull the HF 27k Chandra corpus to bootstrap the RAG — own pipeline serves as validation, not the bottleneck.
3. **Phase 4 — Chunking + vector store (4-5 days):** Markdown-aware chunker (header boundaries, tables/equations kept intact, ~512-1024 tokens, title + section path prepended). `chunks` table in DuckDB (lineage: chunker version, text hash, run_id). Qdrant as a Docker service: dense + sparse vectors (BGE-M3), payload = {arxiv_id, section, page, category, published_date}. Golden eval set (~50 question → relevant-chunk pairs) BEFORE tuning. Retrieval eval: Recall@k, MRR. Timebox the chunking ablation to 1 day.
4. **Phase 5 — Agent layer (4-5 days):** LLM with tools over both stores: `sql_query` (DuckDB: trends, filters, counts), `vector_search` (Qdrant: hybrid + reranker), `get_paper` (full markdown), `get_repo` (GitHub URLs from Phase 1 extraction). Local generation via Ollama/vLLM. RAGAS for end-to-end eval.
5. **Phase 6 — Intelligence layer (3-4 days):** Weekly job: Kaggle dump delta → embed new abstracts → score against interest profile (seed queries/papers) → novelty-ranked digest with arXiv + repo links. Expose the agent as an MCP server so coding agents can query the corpus while building.
6. **Phase 7 — Parked:** OCR benchmark (OlmOCRBench, 5 engines), knowledge graph enrichment (Semantic Scholar venue resolution, citation edges), Streamlit demo, GPT-from-scratch (→ separate pedagogical repo).
7. **Phase 8 — Polish (ongoing):** per-phase blog posts, README quickstart, architecture diagram, demo video.

### Phase 1 talking points:
"Evaluated three ingestion paths, picked the hybrid that fit the data scale."
"ELT in DuckDB — staged raw, transformed in SQL, kept lineage in ingestion_runs."
"Retries at the I/O boundary, resumability via DB state, not filesystem."
"92% LaTeX-source extraction rate on cs.* papers = OCR ground truth."

## Known risks
- bitsandbytes for sm_120 — installed but not yet exercised. Doesn't work with CUDA 13 - use torchao instead.
- ragas + langchain-community installed; potential conflict if transformers upgraded.
- Chandra-2 weights ~5B — 8-bit quantization works on RTX 5070Ti with 16 GB VRAM.
- TurboQuant (Google KV-cache compression) flagged for Phase 7 if VRAM becomes binding.
- NEW: Phase 2a math depth. CTC derivation requires solid grasp of dynamic programming + log-space arithmetic. Daily math exercises in progress; if math feels shaky after Day 2, take a math-only day before continuing.

## Open questions
- BGE-M3 VRAM fit next to vLLM/Ollama on 16 GB — validate at Phase 4 entry; fallback: smaller dense model + Qdrant-side BM25 sparse.
- Chunk text storage: in the DuckDB `chunks` table (ground truth, simple) vs files + hash only. Leaning DuckDB; decide at Phase 4 entry.
- Reranker: bge-reranker (already in deps) vs none — measure on golden set before adding latency.
- Fetcher migration to paper_stage_state helpers: do when next ingestion batch is needed (10-line change, not urgent).
- Interest profile representation (Phase 6): seed queries vs seed-paper centroids vs both.
- bioRxiv ingestion: still v2.

## Phase 1 artifacts (for reference)
src/scirex/ingestion/fetch_files.py — async-free fetcher with tenacity retries
src/scirex/ingestion/extract_latex.py — tarball extractor with security-hardened extractall
data/arxiv_metadata.duckdb — 5-table normalized schema
data/raw/pdfs/ — ~900 PDFs
data/raw/sources/ — ~900 source tarballs
data/interim/latex/{arxiv_id}/ — 916 extracted LaTeX trees
data/logs/fetch.log, data/logs/extract.log — structured run logs

## Resources
- HF blog: https://huggingface.co/blog/nielsr/ocr-papers-jobs
- HF corpus: `nielsr/arxiv-chandra-ocr-full-markdown-20260406` (27k papers)
- Reference repo: https://github.com/NielsRogge/arxiv-ocr
- ArXiv APIs and bulk data : https://info.arxiv.org/help/bulk_data/index.html
- OlmOCRBench: https://huggingface.co/datasets/allenai/olmOCR-bench
- Nemotron-OCR-v2 (released 2026-04-15): https://huggingface.co/nvidia/nemotron-ocr-v2
- TurboQuant: https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/
- Phase 2a — Graves 2006 CTC paper: https://www.cs.toronto.edu/~graves/icml_2006.pdf
- Phase 2a — CRNN paper (Shi 2015): https://arxiv.org/abs/1507.05717
- Phase 2a — Hannun's CTC explainer (Distill): https://distill.pub/2017/ctc/
- Phase 5 — Semantic Scholar API: https://api.semanticscholar.org/