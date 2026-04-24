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

## License

MIT (code) — see `LICENSE`.
Derived datasets are licensed CC-BY-4.0.