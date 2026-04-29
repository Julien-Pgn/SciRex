#!/usr/bin/env bash
# SciRex container launcher.
# Usage: ./run_container.sh <mode> [args...]
# Modes:
#   shell                         interactive bash inside the container (default)
#   exec <cmd> [args...]          run a single command, e.g. exec pytest
#   jupyter                       start JupyterLab on http://localhost:8888 (detached)
#   streamlit                     start Streamlit demo on http://localhost:8501 (detached)
#   stop <jupyter|streamlit>      stop and remove a daemon
#   logs <jupyter|streamlit>      follow logs from a daemon
#   build                         rebuild the dev image
#   help                          show this message

set -euo pipefail

IMAGE="scirex:dev"

COMMON_FLAGS=(
    --gpus all
    --ipc=host
    --ulimit memlock=-1
    --ulimit stack=67108864
    --shm-size=16g
    -v "$(pwd):/workspace"
    -v "$HOME/.cache/huggingface:/root/.cache/huggingface"
    -e "WANDB_API_KEY=${WANDB_API_KEY:-}"
    -e "HF_TOKEN=${HF_TOKEN:-}"
    -e "PYTHONDONTWRITEBYTECODE=1"
    -w /workspace
)

# Ensure the HF cache directory exists on the host (else Docker creates it as root)
mkdir -p "$HOME/.cache/huggingface"

mode="${1:-shell}"

case "$mode" in
    shell)
        docker run --rm -it "${COMMON_FLAGS[@]}" "$IMAGE" bash
        ;;

    exec)
        shift
        if [ $# -eq 0 ]; then
            echo "exec requires a command, e.g. ./run_container.sh exec pytest" >&2
            exit 1
        fi
        docker run --rm "${COMMON_FLAGS[@]}" "$IMAGE" "$@"
        ;;

    jupyter)
        docker rm -f scirex-jupyter >/dev/null 2>&1 || true
        docker run -d --name scirex-jupyter \
            "${COMMON_FLAGS[@]}" \
            -p 8888:8888 \
            "$IMAGE" \
            jupyter lab \
                --ip=0.0.0.0 --port=8888 --no-browser --allow-root \
                --NotebookApp.token='' --NotebookApp.password=''
        echo "JupyterLab: http://localhost:8888"
        echo "Stop with:  ./run_container.sh stop jupyter"
        echo "Logs with:  ./run_container.sh logs jupyter"
        ;;

    streamlit)
        docker rm -f scirex-streamlit >/dev/null 2>&1 || true
        docker run -d --name scirex-streamlit \
            "${COMMON_FLAGS[@]}" \
            -p 8501:8501 \
            "$IMAGE" \
            streamlit run src/scirex/demo/app.py \
                --server.address 0.0.0.0 --server.port 8501 \
                --server.headless true
        echo "Streamlit: http://localhost:8501"
        echo "Stop with: ./run_container.sh stop streamlit"
        ;;

    stop)
        target="${2:-}"
        if [ -z "$target" ]; then
            echo "stop requires a target, e.g. stop jupyter" >&2
            exit 1
        fi
        docker rm -f "scirex-${target}" >/dev/null 2>&1 \
            && echo "Stopped scirex-${target}" \
            || echo "scirex-${target} was not running"
        ;;

    logs)
        target="${2:-}"
        if [ -z "$target" ]; then
            echo "logs requires a target, e.g. logs jupyter" >&2
            exit 1
        fi
        docker logs -f "scirex-${target}"
        ;;

    build)
        docker build -t "$IMAGE" .
        ;;

    help|-h|--help)
        sed -n '2,12p' "$0"
        ;;

    *)
        echo "Unknown mode: $mode" >&2
        echo "Run './run_container.sh help' for usage." >&2
        exit 1
        ;;
esac