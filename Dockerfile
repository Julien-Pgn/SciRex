# syntax=docker/dockerfile:1.7
# SciRex development image
# Base: NVIDIA NGC PyTorch 26.01 (PyTorch 2.10.0a, CUDA 13.1.1, Python 3.12, Ubuntu 24.04)
# Supports Blackwell sm_120 (RTX 50-series) natively.

FROM nvcr.io/nvidia/pytorch:26.01-py3

LABEL project="scirex"
LABEL description="PDF-to-markdown scientific-knowledge pipeline"
LABEL maintainer="Julien Pigeon"

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# System packages beyond NGC defaults.
# poppler-utils -> pdftotext, pdfinfo (used by pdf2image and for fallbacks)
# tesseract-ocr + libtesseract-dev -> classical OCR baseline for CRNN comparison
# libgl1, libglib2.0-0 -> OpenCV runtime deps
# git-lfs -> large file handling in git
# graphviz -> knowledge-graph visualization (pyvis fallback)
RUN apt-get update && apt-get install -y --no-install-recommends \
    poppler-utils \
    tesseract-ocr \
    libtesseract-dev \
    libgl1 \
    libglib2.0-0 \
    git-lfs \
    graphviz \
    && rm -rf /var/lib/apt/lists/*

# uv is already installed in NGC 26.01 (uv==0.9.24). No install step needed.

WORKDIR /workspace

# Copy dependency manifests first for layer caching.
# Source is mounted at runtime via -v, not copied into the image.
COPY pyproject.toml uv.lock LICENSE.txt README.md /workspace/
COPY src /workspace/src
COPY tests /workspace/tests

# Install our Python dependencies into the container's system interpreter.
# --system        : target system Python (no venv creation)
# --break-system-packages : allow overwriting NGC's preinstalled packages (e.g. torch, torchvision) -> is in the NGC documentation as the way to update packages in the base image
# --no-cache-dir  : keep image size down
# -e .[all]       : editable install with all optional groups

RUN uv pip install --system --break-system-packages --no-cache-dir -e ".[all]"

EXPOSE 8888 8501

CMD ["/bin/bash"]