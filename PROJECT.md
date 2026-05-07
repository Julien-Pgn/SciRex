# PROJECT.md — SciRex working state

Last updated: 2026-05-01
Repo: github.com/Julien-Pgn/scirex

## Current phase
✅ Phase 0 complete. Starting Phase 1 — ingestion.
✅ Phase 1 complete (2026-05-01). See full close-out in Decisions log.
⏭ Starting Phase 2a — CRNN from scratch.

## Environment
- Base image: `nvcr.io/nvidia/pytorch:26.01-py3` (Python 3.12.3, PyTorch 2.10.0a NV-patched, CUDA 13.1, sm_120 supported)
- Image tag: `scirex:dev`
- Hardware: RTX 5070 Ti (Blackwell, 16 GB VRAM)
- Host driver: 580.126.09 (CUDA 13.0) — runs container in MVC mode; upgrade before Phase 2b benchmarks
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
2026-05-07: I have to find a way to get the 
## Project scope (locked)
1. **Phase 1 — Ingestion - DONE - (3 days):** Kaggle dump → DuckDB schema → benchmark_subset selection → API fetcher → LaTeX extractor.
2. **Phase 2a — CRNN from scratch (5-6 days):** CNN + BiLSTM + CTC, hand-derived forward-backward CTC pass.
3. **Phase 2b — OCR benchmark (5-6 days):** Chandra-2 + Marker + Nougat + olmOCR + Nemotron-OCR-v2 + CRNN baseline. Evaluated on OlmOCRBench (field-standard) AND ~900 arXiv papers with LaTeX-source ground truth.
4. **Phase 3 — Metadata extraction → markdown corpus (2-3 days):** Pydantic + instructor structured output. Reuse HF's 27k Chandra corpus for scale; produce 2k via own pipeline as validation.
5. **Phase 4 — Streamlit demo (2 days):** Upload PDF → OCR → markdown + metadata.
6. **Phase 5 — Knowledge graph (3-4 days):** Citation extraction, NetworkX, gap-signal heuristics.
7. **Phase 6 — GPT-from-scratch PoC (5-6 days):** ~30M params, BPE from scratch, training loop. Pretrain on subset of HF corpus. SFT on small instruction set. Pedagogical, framed as such.
8. **Phase 7 — Production RAG (4-5 days):** Chunking ablation, embeddings, Qdrant or ChromaDB, retrieval eval (Recall@k, MRR, RAGAS). Local generation via Ollama with 4-bit quantized 7-14B model.
9. **Phase 8 — Polish (ongoing + 2 days):** Per-phase blog posts, README, architecture diagram, demo video.

Target: mid-May for Phases 1-4 + 8 (apply to jobs). Phases 5-7 marked "in progress" on ROADMAP.md.

## Strategic narrative for interviews
"HuggingFace solved bulk OCR (27k papers via Chandra-2). I solved the **evaluation problem** with LaTeX-aligned ground truth, and built the **knowledge layer** on top. CRNN from scratch as pedagogical baseline. RAG, not pretraining, for the queryable assistant — pretraining is the educational PoC."

### Phase 1 talking points:
"Evaluated three ingestion paths, picked the hybrid that fit the data scale."
"ELT in DuckDB — staged raw, transformed in SQL, kept lineage in ingestion_runs."
"Retries at the I/O boundary, resumability via DB state, not filesystem."
"92% LaTeX-source extraction rate on cs.* papers — that's my OCR ground truth."

## Known risks
- MVC mode warning on every container start until host driver upgrade.
- bitsandbytes for sm_120 — installed but not yet exercised. Validate in Phase 6.
- ragas + langchain-community installed; potential conflict if transformers upgraded.
- Chandra-2 weights ~5B — 4-bit quantization or model offload required for 16 GB VRAM.
- TurboQuant (Google KV-cache compression) flagged for Phase 7 if VRAM becomes binding.
- NEW: Phase 2a math depth. CTC derivation requires solid grasp of dynamic programming + log-space arithmetic. Daily math exercises in progress; if math feels shaky after Day 2, take a math-only day before continuing.

## Open questions
- bioRxiv ingestion in Phase 1: include now or defer to v2? (current default: defer; arXiv only for v1)
- Vector DB choice in Phase 7: Qdrant vs. ChromaDB. Decide at Phase 7 entry.
- Phase 2a backward-pass implementation vs. derivation-only: decide at end of Day 2 based on math comfort.
- Phase 2a beam search: decide at Day 10 based on time remaining.

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