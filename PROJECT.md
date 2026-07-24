# PROJECT.md — SciRex working state

Last updated: 2026-07-24
Repo: github.com/Julien-Pgn/scirex

## Current phase
✅ Phase 0 complete. Starting Phase 1 — ingestion.
✅ Phase 1 complete (2026-05-01). See full close-out in Decisions log.
✅ Phase 3.5 complete (2026-07-24) — topic retrieval built and run end-to-end for "quantization":
870 papers confirmed relevant (hybrid search + local LLM classifier), 863 fetched and OCR'd
(7 permanent arXiv 404s). See full close-out in Decisions log.
▶️ Next: Phase 4 (chunking + vector store) then Phase 5 (agent layer) — together, a working
local RAG system over this corpus.

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
2026-07-18: Repo audit + cleanup. Removed src/scirex/layoutreader/ (LayoutLMv3 training/eval code, abandoned per 2026-05-09/2026-05-20 decisions above; broken imports, undeclared nltk/datasets/typer deps, 6 mypy errors) and src/scirex/ocr/crnn/ (empty dir, vestige of the abandoned EasyOCR/CRNN path). Removed data/ocr_hf/, data/ocr_vllm/ (V1 notebook exploration outputs, superseded by data/processed/) and debug_page.html (stray container artifact). Added unit tests for the pure logic in ocr/modern/pipeline.py, ingestion/fetch_files.py, ingestion/extract_latex.py (0%→43% coverage on those modules); CI now installs chandra-ocr (lightweight, no torch/GPU deps) so these tests run in the workflow. README's "Contributions" section corrected to distinguish what runs today (ingestion, OCR) from what's planned (chunking/RAG/KG) — it previously implied llm/rag/extraction modules had code, but those directories are still empty. ROADMAP.md Phase 2 status intentionally left unchanged pending review.
2026-07-18: CI broken by a newer uv release enforcing PEP 668 more strictly — `uv pip install --system` started failing with "externally managed" even against uv's own managed Python (not just the OS one at /usr). Fix: added `--break-system-packages` to the install step. Safe here since the CI runner is a disposable container, not a persistent machine.
2026-07-22: corpus curation should happen on abstracts already in the duck db
2026-07-22: hybrid serach combining dense and sparse embeddings using BGE-M3 is the best approach to find all relevant papers (precision at 0.7 and recall at 0.7) -> classifying using a local LLM (Qwen2.5-7B-Instruct) increases precision up to 0.82 with a recall to 0.82 which validates the two stage approach. 

### Phase 3.5 — Topic retrieval (built, first production run complete — 2026-07-23)

**Problem:** regex-based paper selection (`genai_subset`) had poor precision and recall for
topic-specific corpus curation — spot-checked sample showed most matches were topic-adjacent, not
topic-relevant.

**Validated approach** (prototype: `notebooks/topic_retrieval.ipynb`, test topic: "quantization"):
1. Embed paper abstracts (already in DuckDB `papers.abstract` for all 3.07M rows — no fetch/OCR
   needed) with BGE-M3, producing dense + sparse (lexical) vectors from one model call.
2. Hybrid search: fuse dense + sparse rankings via Reciprocal Rank Fusion (RRF). Beats dense-only
   retrieval (AP 0.866 vs 0.767 on an 85-paper hand-labeled golden set).
3. LLM classification pass over hybrid's candidates: local Qwen2.5-7B-Instruct (int8 via torchao),
   rubric + few-shot prompt (examples pulled from the golden set, encoding real confusions found
   during labeling: "uncertainty quantification," "vector quantization"/VQ-VAE, papers that only
   evaluate already-quantized models). Measurably improves precision over hybrid alone at matched
   recall: 0.82 vs 0.74 precision at recall=0.82.
4. Two-stage architecture: hybrid = recall engine (wide net), LLM classifier = precision cleanup.

**Hard constraint:** 100% local — no cloud APIs. BGE-M3 and the classifier LLM both run on-device
(RTX 5070 Ti, 16GB VRAM). DuckDB is the only datastore for this stage — not Qdrant, which stays
scoped to Phase 4's post-OCR chunk vectors.

**Built (2026-07-22/23):**
- `src/scirex/retrieval/hybrid_search.py` — pure `dense_scores`/`sparse_scores`/`rrf_fuse`/
  `recommend_top_k`, no model/DB imports, unit-tested without GPU.
- `src/scirex/retrieval/embed.py`, `classify.py` — BGE-M3 and Qwen2.5-7B-Instruct wrappers; heavy
  imports (FlagEmbedding/transformers/torchao) are lazy, loaded only inside the functions that need
  them, so the rest of each module (batching, DB I/O, prompt construction, verdict parsing) is
  testable without GPU/torch installed at all.
- DDL: `abstract_embeddings(arxiv_id PK, dense_vec FLOAT[1024], sparse_weights MAP(INTEGER, FLOAT),
  embed_model, embed_date)` and `topic_subset(topic, arxiv_id, hybrid_score, llm_verdict, rank,
  run_id, created_at, PRIMARY KEY(topic, arxiv_id))`. Resolved the open decisions: MAP over JSON for
  sparse weights (round-trips to a native Python dict via both `fetchone()`/`fetchdf()`, verified
  empirically); skipped the `vss` extension (brute-force numpy scoring is fast enough at this scale,
  no ANN index needed for batch topic curation).
- `scripts/embed_abstracts.py`, `scripts/search_topic.py` — same shape as `fetch_files.py`/
  `run_ocr.py` (argparse, dual file+stderr logging, summary banner, idempotent via DB state).
- `fetch_files.py` got the planned `--topic` filter: `get_papers_to_fetch(conn, table, topic=None)`
  now requires topic match + `llm_verdict = TRUE` when `topic` is given, so `topic_subset`'s mixed
  verdicts/topics can't silently leak into a fetch run. `keyword_for_ocr` is set to the topic string
  itself (e.g. `"quantization"`), not the table name, so `run_ocr.py --keyword` stays meaningful.

**Bug found and fixed during the build:** `save_embeddings`'s first version wrote rows one at a time
via `executemany`. Row-by-row binding of a 1024-float array + ~100-entry MAP dict measured at
~20-40s per 500 rows — this was the actual bottleneck in the first production embed run (not the
GPU, which sat at 0% SM utilization the whole time), compounded by DuckDB's periodic WAL
auto-checkpoint (a clean ~4-chunk sawtooth in the run's timing was the tell). Fixed with a bulk
columnar insert (`INSERT ... SELECT FROM df`, DuckDB's native pandas integration): same 500 rows in
~0.2s, ~100x faster. Also: BGE-M3's `use_fp16=True` means `lexical_weights` dict values come back as
`numpy.float16`, which DuckDB's MAP binding can't convert (unlike its LIST/array binding, which
handles fp16 fine via `.tolist()`) — cast explicitly to plain `float`/`int` before insert.

**`recommend_top_k` — recall-based sizing, and a real limitation found in production:** instead of
guessing a fixed `--top-k`, look up where the golden set's known positives actually rank in the
*full* corpus ranking (not a re-ranking among just the golden set — RRF fuses on rank, so a paper's
rank among ~20 curated papers is a different number than its rank among 421k real ones) and return
the depth needed to catch `--target-recall` of them. On the real "quantization" category slice
(421,574 papers), with only 20 labeled positives in the golden set and `target_recall=0.97`, this
came out to `top_k=231,007` — over half the corpus — because at n=20, missing even one positive
already drops recall to 95%, so the single worst-ranked outlier (a KV-cache-compression paper
sharing little vocabulary with the query) single-handedly set the cutoff. Not a bug —
`recommend_top_k` did exactly what it's specified to do — but a real fragility: with a small golden
set, high recall targets end up dominated by 1-2 outliers rather than being a robust estimate. For
tonight's production run, overrode with a fixed `--top-k 1000`, chosen to match the existing OCR
throughput ceiling (~1000 papers/day) rather than derived from the golden set. Follow-up for a
future session: either enlarge the golden set so percentile-based sizing is statistically stable, or
make `recommend_top_k` robust to outliers (e.g. cap by percentile of positives instead of requiring
literally all of them).

**First production run (2026-07-22/23, topic="quantization", run overnight/unattended):**
- `embed_abstracts.py --categories cs.LG cs.CL cs.AI cs.CV`: embedded all 421,574 papers in these
  categories (0 missing on verification).
- `search_topic.py --topic quantization --top-k 1000`: 1000 hybrid-ranked candidates classified,
  310 verdict=TRUE (~1.8s/paper on the RTX 5070 Ti). Spot-checked: TRUE verdicts are all genuinely
  about weight/activation quantization; FALSE verdicts correctly exclude the rubric's named
  look-alikes in the wild ("Pyramid Vector Quantization for LLMs", "An Overview of Uncertainty
  Quantification Methods for Infinite Neural Networks"), and correctly overrode hybrid search's own
  #1-ranked candidate ("On Irrelevance of Attributes in Flexible Prediction") as irrelevant — the
  two-stage architecture validated on real production data, not just the notebook's 85-paper eval set.
- `fetch_files.py --table topic_subset --topic quantization`: 307/310 PDFs+sources fetched
  successfully (99%); 3 failed on genuine arXiv-side 404s (not a systemic issue — resumable,
  re-running the same command will retry just those 3).
- Deliberately NOT run that night: `run_ocr.py` — the `chandra-vllm` OCR service wasn't running, and at
  the established ~1000-papers/day OCR throughput, 310 papers would take multiple hours, well past
  the session's time budget. Left for a follow-up session.
- `genai` was explicitly scoped out as a second validation topic for this phase (2026-07-23 decision)
  — it's a separate, unrelated topic, not a generalization test for this work.

**`run_ocr.py` bug found and fixed (2026-07-23):** it selected PDFs via `args.input_dir.glob("*.pdf")`
— every topic's fetched PDFs land in the same shared `data/raw/pdfs/`, so this silently OCR'd
whichever topic's backlog happened to be sitting there (caught in practice: it picked up ~3000 old
`genai_subset` PDFs instead of the intended quantization batch). `--keyword` was already documented
as "the topic for this batch" but was never actually used to *select* files, only to label results
afterward. Fixed: PDF selection is now `get_pdf_paths_for_keyword()` (new, in
`src/scirex/ocr/modern/pipeline.py`), querying `paper_local WHERE keyword_for_ocr = --keyword AND
pdf_path IS NOT NULL` — `--input-dir` removed entirely (the DB already has the full path, so a
directory glob was redundant as well as unsafe). 4 new tests in `tests/test_ocr_pipeline.py`.

**Second production run (2026-07-23, user's explicit choice): `search_topic.py --top-k 5000`**
(overriding the recall-based default — see the `recommend_top_k` limitation above) — 870 verdict=TRUE,
up from 310 at top_k=1000. `fetch_files.py` re-run picked up the delta (561 new candidates): 863/870
fetched total (99.2%); the remaining 7 are confirmed *permanent* arXiv-side 404s (several repeat-failed
across multiple fetch re-runs, not transient). `run_ocr.py` (with the fix above) processed all 863
successfully.

**Verification caught a subtle 2-paper gap (2026-07-24):** don't trust the DB flag alone — cross-checking
`ocr_done` against the actual filesystem surfaced 2 papers whose `keyword_for_ocr` didn't say
`"quantization"` despite having a `pdf_path` set. Both had been fetched under an *earlier* run before
Step 6's `--topic` filter existed; since their `pdf_path` was already non-NULL, this quantization
fetch correctly skipped re-downloading them (no need — the file was already there) but had no reason to
touch their `keyword_for_ocr` label. One (`2605.12327`) already had valid Chandra OCR output from
2026-06-17 (verified: proper markdown, extracted images, matches the current pipeline's output shape)
— just needed its DB metadata corrected, not reprocessing, since `run_ocr.py`'s skip-if-exists check
means a plain rerun never repairs stale metadata for a file that's already on disk. The other
(`2605.29705`) had genuinely never been OCR'd — processed directly. Lesson: `keyword_for_ocr` is a
single string, so a paper relevant to two topics fetched in different runs can only ever carry one
label; fine for a single active topic, but worth remembering before assuming "labeled X" means
"never touched by anything else."

**Final verified state (2026-07-24): 863/863 quantization papers have both `ocr_done = TRUE` in
`paper_local` and a real `.md` file on disk** (checked both ways, not just the DB flag). **Phase 3.5
is done.** This corpus (plus the original ~900-paper benchmark set from Phase 3) is what Phase 4
chunking will consume.

### Phase 4 — Chunking + vector store: next steps (not started — planned 2026-07-24)

**Prerequisite corpus:** any paper with `paper_local.ocr_done = TRUE` — **not** gated on having a
`topic_subset` row. The original ~900-paper Phase 3 benchmark corpus predates topic tracking
entirely (no `topic_subset` row at all) but is equally valid, chunkable content. `topic_subset` only
explains *why* a paper got fetched/OCR'd, not whether it's usable downstream.

**Two open questions from the locked scope below, resolved now (their notes said "decide at Phase 4
entry" — this is that):**
1. **Chunk text storage → DuckDB `chunks` table**, not files+hash. Consistent with this project's
   established pattern (`abstract_embeddings`, `topic_subset`, `paper_local` all live in DuckDB too,
   not on disk). A `chunks` table already exists as an empty stub from Phase 1
   (`notebooks/db_build.ipynb`, cell 17): `chunk_id, arxiv_id, chunk_index, char_start, char_end,
   n_tokens, text, section, embed_model, UNIQUE(arxiv_id, chunk_index)`. Needs 2 more columns for
   lineage per the locked scope's requirement: `chunker_version`, `run_id`.
2. **BGE-M3 VRAM fit "next to vLLM/Ollama"** — re-scoped: that's actually a **Phase 5** question
   (agent generation running *alongside* retrieval at query time), not a Phase 4 one. Phase 4 just
   batch-embeds chunks with the same load-model → use → free-VRAM pattern already proven in
   `search_topic.py` — no real concurrency risk here. Revisit at Phase 5 entry instead.
3. **Reranker (bge-reranker, already in `pyproject.toml`'s `rag` group)** — deliberately left open:
   build the golden eval set and a baseline retrieval score first (steps 1 and 7 below), then measure
   with/without the reranker before deciding. Same "evaluate before tuning" discipline the topic
   retrieval's golden set already established — don't add latency on a hunch.

**Concrete steps, in dependency order (estimate: 4-5 days total, matching the locked-scope estimate):**
1. **Golden eval set first, before any chunking code** (~0.5-1 day). ~50 question → relevant-chunk
   pairs, hand-written against real papers already in the corpus (mix of quantization + benchmark
   papers, so both populations are represented). Same methodology as the topic-retrieval golden set:
   write it once, store as parquet, reuse it for every tuning decision below so nothing gets tuned by
   vibes.
2. **Notebook-prototype the chunker first** (~1 day) — matches this project's established pattern
   (every phase so far prototyped in `notebooks/` before graduating to `src/scirex/`). Try a
   markdown-aware splitter on a handful of real `.md` files from `data/processed/`: split on header
   boundaries, keep tables/equations intact (don't split mid-table or mid-equation), target
   ~512-1024 tokens per chunk, prepend title + section path to each chunk's text so a chunk read in
   isolation still carries context. Eyeball a few real chunks before formalizing anything.
3. **Graduate into `src/scirex/chunking/chunker.py`** (~0.5 day) — pure function(s): markdown text
   in, list of chunk dicts out (text, section, char_start/end, n_tokens). No DB/model imports,
   unit-testable — same "pure logic separated from I/O" split already used for `hybrid_search.py`.
4. **`chunks` table migration + `scripts/chunk_papers.py`** (~1 day) — same shape as
   `embed_abstracts.py`: idempotent (only chunk papers not yet chunked for the current
   `chunker_version`, so a future re-chunk with a new version doesn't collide with old chunks), dual
   file+stderr logging, summary banner.
5. **Qdrant as a Docker service** (~0.5 day) — new `run_qdrant.sh`, same shape as `run_vllm.sh`,
   joined to the existing `scirex-net` Docker network so it's reachable by hostname from the dev
   container, matching how `chandra-vllm` already works.
6. **`scripts/embed_chunks.py`** (~1 day) — BGE-M3 dense+sparse over chunk text, reusing
   `src/scirex/retrieval/embed.py`'s patterns, upserting into Qdrant with payload
   `{arxiv_id, section, page, category, published_date}`. Idempotent, same style as everything else.
7. **Retrieval eval against the golden set** (~0.5-1 day) — Recall@k, MRR. Run once without a
   reranker (baseline), then once with `bge-reranker` — keep it only if it measurably helps at
   acceptable latency. This is where open question 3 above actually gets closed, with data instead
   of a guess.
8. **Timebox a chunking ablation to 1 day** (per the locked scope) — if time allows, try 1-2
   alternate chunk sizes/overlaps against the same golden set, but don't let this expand past a day.

### Phase 5 — Agent layer (after Phase 4): preview, not detailed yet
LLM with tools over both stores — `sql_query` (DuckDB), `vector_search` (Qdrant, hybrid + reranker),
`get_paper` (full markdown), `get_repo` (GitHub URLs already extracted in Phase 1). Local generation
via Ollama or vLLM. Worth reusing rather than re-validating from scratch: Qwen2.5-7B-Instruct (int8
via torchao) is already confirmed to fit comfortably on the RTX 5070 Ti from Phase 3.5's classifier
work — a strong default candidate for the agent's LLM backbone too. RAGAS for end-to-end eval. Revisit
and flesh this out once Phase 4 is actually done — no point over-planning it now.

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

## Phase 3.5 artifacts (for reference)
src/scirex/retrieval/{hybrid_search,embed,classify}.py — hybrid search + BGE-M3/Qwen2.5-7B wrappers
scripts/embed_abstracts.py, scripts/search_topic.py — CLI scripts, same shape as fetch_files.py/run_ocr.py
data/arxiv_metadata.duckdb — abstract_embeddings, topic_subset tables added
data/quantization_rubric.txt — hand-written classification rubric for the "quantization" topic
data/interim/quantization_golden_set_v2.parquet — hand-labeled golden set (20 positives)
data/logs/embed.log, search_topic.log — structured run logs

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