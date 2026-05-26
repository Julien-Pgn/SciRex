#!/usr/bin/env bash
# vLLM server launcher for SciRex.
# Usage: ./run_vllm.sh <mode>
# Modes:
#   up        start the vLLM server (detached), serving Chandra-OCR-2 on port 8000
#   logs      follow vLLM server logs
#   stop      stop and remove the vLLM container
#   status    show whether the container is running
#   exec      open an interactive shell inside the running container (debug)
#   help      show this message

set -euo pipefail

IMAGE="vllm/vllm-openai:v0.17.0"
NAME="chandra-vllm"

CMD="${1:-help}"

case "$CMD" in

    # "up" starts the vLLM server in detached mode. The model is loaded once into VRAM and exposed via an OpenAI-compatible HTTP API on port 8000. Bound to 127.0.0.1 so only the local host can reach it.
    up)
        # Idempotent restart: remove any previous container with the same name.
        docker rm -f "$NAME" >/dev/null 2>&1 || true
        docker run -d \
            --name "$NAME" \
            --gpus all \
            --ipc=host \
            -p 127.0.0.1:8000:8000 \
            --network scirex-net \
            -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
            "$IMAGE" \
                --model datalab-to/chandra-ocr-2 \
                --dtype bfloat16 \
                --max-model-len 8192 \
                --gpu-memory-utilization 0.70 \
                --quantization bitsandbytes \
                --served-model-name chandra
        echo "vLLM server starting (model load ~30-60s)."
        echo "Check readiness: ./run_vllm.sh logs   (wait for 'Application startup complete')"
        echo "Stop with:       ./run_vllm.sh stop"
        ;;

    # "logs" follows the vLLM server's stdout/stderr. Useful to confirm the model loaded and to debug inference errors.
    logs)
        docker logs -f "$NAME"
        ;;

    # "stop" stops and removes the vLLM container, freeing the GPU.
    stop)
        docker rm -f "$NAME" >/dev/null 2>&1 \
            && echo "Stopped $NAME" \
            || echo "$NAME was not running"
        ;;

    # "status" shows the running container, if any.
    status)
        docker ps --filter "name=$NAME"
        ;;

    # "exec" drops you into a bash shell inside the running container. For debugging only.
    exec)
        docker exec -it "$NAME" bash
        ;;

    help|-h|--help)
        sed -n '2,11p' "$0"
        ;;

    *)
        echo "Unknown mode: $CMD" >&2
        echo "Run './run_vllm.sh help' for usage." >&2
        exit 1
        ;;
esac