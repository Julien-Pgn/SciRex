# PROJECT.md — working state

Last updated: 2026-04-23

## Current phase
Phase 0 — environment & repo setup.

## Decisions log
- 2026-04-23: Use NVIDIA NGC container `nvcr.io/nvidia/pytorch:25.01-py3` (GPU verified, Blackwell sm_120 OK).
- 2026-04-23: Package manager = `uv`; src-layout; `pyproject.toml` PEP 621.
- 2026-04-23: Experiment tracker = Weights & Biases.
- 2026-04-23: License = MIT (code), CC-BY-4.0 (derived data).
- 2026-04-23: Config = Pydantic Settings (skip Hydra for v1).
- 2026-04-23: Skip Makefile (use `scripts/`), skip pre-commit (ruff in CI), skip FastAPI (Streamlit only).
- 2026-04-23: OCR benchmark = OlmOCRBench + 1–2k arXiv papers with LaTeX-source ground truth.
- 2026-04-23: Corpus strategy = OCR 2k ourselves end-to-end; consume HF's 27k bucket for downstream RAG/LLM scale.
- 2026-04-23: LLM-from-scratch = GPT-2-small-scale (~30M), pedagogical PoC only.

## Known risks
- OCR inference may require newer NGC image (25.08+) if vLLM enforces torch>=2.7.
- Blackwell (sm_120) support is recent; watch for silent numerical issues — validate CRNN training loss curves against a known baseline.
- Chandra-2 weights are ~5B; inference on 16 GB VRAM will require 4-bit quantization or offload.

## Open questions
- Which modern OCR models beyond Chandra-2, Marker, Nougat, olmOCR should we include?
- Final paper count for LaTeX-aligned benchmark (target: 1–2k).
- Whether to use Qdrant vs. ChromaDB for RAG.
- Implementing TurboQuant to the final LLM used for inference.

## Next actions
1. Complete repo scaffold, first commit, GitHub push.
2. Add `.github/workflows/ci.yml` with ruff + pytest.
3. Verify Python version in NGC container; pin `.python-version` accordingly.
4. Install package inside container, run `import arxiv_knowledge`.
5. Math exercise #1: MSE gradient (see `docs/math-journal/2026-04-23-mse-gradient.md`).

## Resources
- HF blog: https://huggingface.co/blog/nielsr/ocr-papers-jobs
- HF corpus: `nielsr/arxiv-chandra-ocr-full-markdown-20260406` (27k papers).
- Reference repo: https://github.com/NielsRogge/arxiv-ocr
- OlmOCRBench: https://huggingface.co/datasets/allenai/olmOCR-bench