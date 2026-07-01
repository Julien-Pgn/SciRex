# SciRex

A PDF-to-markdown knowledge pipeline to build RAG databases for AI query on scientific topics.

**Status:** in active development — see `ROADMAP.md`.

## What this project is

A general pipeline for turning PDFs into a queryable knowledge base, using
scientific literature as a case study. Contributions:

- **Metadata extraction** via structured LLM output into per-paper markdown.
- **Knowledge graph + RAG** over the resulting corpus, with a locally-hosted LLM.

## Quickstart

Coming soon.

## Running the dev container

All container operations go through `./run_container.sh`.

| Command | Purpose |
|---|---|
| `./run_container.sh shell` | Open an interactive bash session in the container |
| `./run_container.sh exec <cmd>` | Run a one-shot command (e.g. `exec pytest`) |
| `./run_container.sh jupyter` | Start JupyterLab on http://localhost:8888 |
| `./run_container.sh jupyter_nogpu` | Start JupyterLab with access to the GPU: for OCR |
| `./run_container.sh streamlit` | Start the Streamlit demo on http://localhost:8501 |
| `./run_container.sh stop jupyter` | Stop the Jupyter daemon |
| `./run_container.sh logs jupyter` | Follow Jupyter logs |
| `./run_container.sh build` | Rebuild the image |

Optional environment variables (read from your shell): `WANDB_API_KEY`, `HF_TOKEN`.

## Perspective

This project aims to integrate various domain knowledge databases into a single AI layer that can reason across it. So far, it integrates scientific literature (mostly computer science, but soon biology and medicine), and hopefully one day integrate code from Github.

## License

MIT (code) — see `LICENSE`.
Derived datasets are licensed CC-BY-4.0.