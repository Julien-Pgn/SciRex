# PROJECT.md — SciRex working state

Last updated: 2026-04-29
Repo: github.com/Julien-Pgn/scirex

## Current phase
✅ Phase 0 complete. Starting Phase 1 — ingestion.

## Environment
- Base image: `nvcr.io/nvidia/pytorch:26.01-py3` (Python 3.12.3, PyTorch 2.10.0a NV-patched, CUDA 13.1, sm_120 supported)
- Image tag: `scirex:dev`
- Hardware: RTX 5070 Ti (Blackwell, 16 GB VRAM)
- Host driver: 580.126.09 (CUDA 13.0) — runs container in MVC mode; upgrade before Phase 2b benchmarks
- Package manager: `uv` 0.9.24 (preinstalled in NGC). `--system --break-system-packages` for installs.
- Editable install: confirmed working via `_editable_impl_scirex.pth`. Host edits are live in container.

## Decisions log
- 2026-04-29: Repo named `scirex`. License MIT (file: `LICENSE.txt`).
- 2026-04-29: Base = NGC PyTorch 26.01-py3, NOT NeMo (NeMo adds ~20 GB framework overhead, not needed for single-GPU research).
- 2026-04-29: NumPy 2.x acceptable — NGC ships numpy==2.1.0 with PyTorch built against it.
- 2026-04-29: `uv` for dependency mgmt. `pyproject.toml` declares loose constraints; `uv.lock` records exact versions.
- 2026-04-29: Editable install requires `COPY src/` and `COPY tests/` BEFORE `uv pip install -e .` in Dockerfile.
- 2026-04-29: Dependency groups in pyproject map to phases: dev, training, ocr, llm, graph, rag, demo. `[all]` is meta-group for full dev container.
- 2026-04-29: Excluded from pyproject (NGC provides): torch family, transformers tokenizers, datasets, huggingface_hub, einops, safetensors, scipy, scikit-learn, pillow, networkx, beautifulsoup4, pytest, ipykernel, jupyter, tensorboard.
- 2026-04-29: Skipped Hydra (overkill); using Pydantic Settings. Skipped Makefile (use scripts/). Skipped pre-commit (CI handles linting). Skipped FastAPI (Streamlit is sufficient).

## Project scope (locked)
1. **Phase 1 — Ingestion (3-4 days):** ArXiv OAI-PMH async client, LaTeX-source download, DuckDB metadata store, resumable.
2. **Phase 2a — CRNN from scratch (5-6 days):** CNN + BiLSTM + CTC, hand-derived forward-backward CTC pass.
3. **Phase 2b — OCR benchmark (5-6 days):** Chandra-2 + Marker + Nougat + olmOCR + Nemotron-OCR-v2 + CRNN baseline. Evaluated on OlmOCRBench (field-standard) AND 1-2k arXiv papers with LaTeX-source ground truth (math-specific breakdowns).
4. **Phase 3 — Metadata extraction → markdown corpus (2-3 days):** Pydantic + instructor structured output. Reuse HF's 27k Chandra corpus for scale; produce 2k via own pipeline as validation.
5. **Phase 4 — Streamlit demo (2 days):** Upload PDF → OCR → markdown + metadata.
6. **Phase 5 — Knowledge graph (3-4 days):** Citation extraction, NetworkX, gap-signal heuristics.
7. **Phase 6 — GPT-from-scratch PoC (5-6 days):** ~30M params, BPE from scratch, training loop. Pretrain on subset of HF corpus. SFT on small instruction set. Pedagogical, framed as such.
8. **Phase 7 — Production RAG (4-5 days):** Chunking ablation, embeddings, Qdrant or ChromaDB, retrieval eval (Recall@k, MRR, RAGAS). Local generation via Ollama with 4-bit quantized 7-14B model.
9. **Phase 8 — Polish (ongoing + 2 days):** Per-phase blog posts, README, architecture diagram, demo video.

Target: mid-May for Phases 1-4 + 8 (apply to jobs). Phases 5-7 marked "in progress" on ROADMAP.md.

## Strategic narrative for interviews
"HuggingFace solved bulk OCR (27k papers via Chandra-2). I solved the **evaluation problem** with LaTeX-aligned ground truth, and built the **knowledge layer** on top. CRNN from scratch as pedagogical baseline. RAG, not pretraining, for the queryable assistant — pretraining is the educational PoC."

## Known risks
- MVC mode warning on every container start until host driver upgrade.
- bitsandbytes for sm_120 — installed but not yet exercised. Validate in Phase 6.
- ragas + langchain-community installed; potential conflict if transformers upgraded.
- Chandra-2 weights ~5B — 4-bit quantization or model offload required for 16 GB VRAM.
- TurboQuant (Google KV-cache compression) flagged for Phase 7 if VRAM becomes binding.

## Open questions
- bioRxiv ingestion in Phase 1: include now or defer to v2? (current default: defer; arXiv only for v1)
- Vector DB choice in Phase 7: Qdrant vs. ChromaDB. Decide at Phase 7 entry.
- Final paper count for LaTeX-aligned benchmark (Phase 2b): target 1-2k.

## Next actions (Phase 1)
1. ArXiv OAI-PMH client design — async, resumable, rate-limited via tenacity.
2. Schema design in DuckDB: papers, authors, categories, files, ingestion_runs.
3. Async download of PDFs + LaTeX-source tarballs with retry/backoff.
4. CLI entry point via typer-slim: `scirex ingest --max-papers 50 --category cs.LG`.
5. Tests: smoke + integration on 3-5 papers.
6. Decide: how/where do we keep raw downloaded data on disk vs. just metadata in DuckDB?

## Resources
- HF blog: https://huggingface.co/blog/nielsr/ocr-papers-jobs
- HF corpus: `nielsr/arxiv-chandra-ocr-full-markdown-20260406` (27k papers)
- Reference repo: https://github.com/NielsRogge/arxiv-ocr
- OlmOCRBench: https://huggingface.co/datasets/allenai/olmOCR-bench
- Nemotron-OCR-v2 (released 2026-04-15): https://huggingface.co/nvidia/nemotron-ocr-v2
- TurboQuant: https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/