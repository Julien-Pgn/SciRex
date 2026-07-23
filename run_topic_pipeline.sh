#!/usr/bin/env bash
# Sequential fetch -> OCR pipeline for one topic, with basic retry on outright failure.
# Per-paper failures (a stray arXiv 404, one bad PDF) are already handled by
# fetch_files.py/run_ocr.py's own resumability — simply rerunning retries just those.
# This wrapper's retries are for the whole script crashing outright (container
# start failure, vLLM not ready yet, etc.), not for those expected per-item cases.
#
# Usage: ./run_topic_pipeline.sh <topic>
# Run this inside tmux/screen on the GPU machine so it survives SSH/VSCode
# disconnects: tmux new -s pipeline, then ./run_topic_pipeline.sh quantization,
# then Ctrl+B D to detach. Reattach anytime with: tmux attach -t pipeline

set -uo pipefail  # not -e: failures are handled explicitly below, not aborted on

TOPIC="${1:?Usage: ./run_topic_pipeline.sh <topic>}"
MAX_RETRIES=3
LOG="data/logs/${TOPIC}_pipeline.log"

mkdir -p data/logs

log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') $*" | tee -a "$LOG"
}

log "=== Starting pipeline for topic '$TOPIC' ==="

# --- Step 1: fetch PDFs/sources for this topic's confirmed-relevant papers ---
fetch_ok=false
for attempt in $(seq 1 "$MAX_RETRIES"); do
    log "Fetch attempt $attempt/$MAX_RETRIES"
    if ./run_container.sh exec python -m scirex.ingestion.fetch_files \
        --table topic_subset --topic "$TOPIC" 2>&1 | tee -a "$LOG"; then
        fetch_ok=true
        break
    fi
    log "Fetch attempt $attempt crashed outright — retrying in 30s..."
    sleep 30
done

if [ "$fetch_ok" != true ]; then
    log "=== ABORTING: fetch never completed after $MAX_RETRIES attempts. OCR not started. ==="
    exit 1
fi

# --- Step 2: start the OCR service ---
log "Starting vLLM OCR server..."
./run_vllm.sh up
log "Waiting for model load..."
for _ in $(seq 1 30); do
    if docker logs chandra-vllm 2>&1 | grep -q "Application startup complete"; then
        log "vLLM server ready."
        break
    fi
    sleep 5
done

# --- Step 3: OCR this topic's PDFs ---
ocr_ok=false
for attempt in $(seq 1 "$MAX_RETRIES"); do
    log "OCR attempt $attempt/$MAX_RETRIES"
    if ./run_container.sh exec python scripts/run_ocr.py \
        --keyword "$TOPIC" --max-concurrent-pdfs 8 2>&1 | tee -a "$LOG"; then
        ocr_ok=true
        break
    fi
    log "OCR attempt $attempt crashed outright — retrying in 30s..."
    sleep 30
done

./run_vllm.sh stop

if [ "$ocr_ok" != true ]; then
    log "=== FINISHED WITH ERRORS: OCR never completed after $MAX_RETRIES attempts. ==="
    exit 1
fi

log "=== Pipeline for topic '$TOPIC' finished successfully. ==="
