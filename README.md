# SciRex

A PDF-to-markdown knowledge pipeline with rigorous OCR benchmarking to build RAG databases for AI query.

**Status:** in active development — see `ROADMAP.md`.

## What this project is

A general pipeline for turning PDF corpora into a queryable knowledge base, using
scientific literature as a case study. Contributions:

- **OCR benchmark** on OlmOCRBench and LaTeX-aligned arXiv ground truth, comparing
  Chandra-OCR-2, Marker, Nougat, olmOCR, and a from-scratch CRNN baseline.
- **From-scratch CRNN** with a hand-derived CTC forward-backward pass (pedagogical).
- **Metadata extraction** via structured LLM output into per-paper markdown.
- **Knowledge graph + RAG** over the resulting corpus, with a locally-hosted LLM.
- **From-scratch GPT** at ~30M params as a pedagogical reimplementation.

## Quickstart

Coming soon.

## Running the dev container

All container operations go through `./run_container.sh`.

| Command | Purpose |
|---|---|
| `./run_container.sh shell` | Open an interactive bash session in the container |
| `./run_container.sh exec <cmd>` | Run a one-shot command (e.g. `exec pytest`) |
| `./run_container.sh jupyter` | Start JupyterLab on http://localhost:8888 |
| `./run_container.sh streamlit` | Start the Streamlit demo on http://localhost:8501 |
| `./run_container.sh stop jupyter` | Stop the Jupyter daemon |
| `./run_container.sh logs jupyter` | Follow Jupyter logs |
| `./run_container.sh build` | Rebuild the image |

Optional environment variables (read from your shell): `WANDB_API_KEY`, `HF_TOKEN`.

## License

MIT (code) — see `LICENSE`.
Derived datasets are licensed CC-BY-4.0.