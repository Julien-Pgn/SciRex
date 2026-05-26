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

# No GPU on purpose. This is to avoid conflicts with other Docker containers that require full GPU access, e.g. for training. If you want GPU access in the container, you can modify the COMMON_FLAGS below to include "--gpus all" and ensure you have the NVIDIA Container Toolkit installed on your system.
COMMON_FLAGS=(
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

# Default to "shell" mode if no argument is provided to ./run_container.sh
mode="${1:-shell}"


case "$mode" in

    # "shell" is the default mode, so that running ./run_container.sh without arguments drops you into a bash shell inside the container.
    shell)
        docker run --rm -it --gpus all "${COMMON_FLAGS[@]}" "$IMAGE" bash
        ;;

    # "exec" allows you to run a single command inside the container without starting an interactive shell, e.g. "./run_container.sh exec pytest"
    exec)
        shift
        if [ $# -eq 0 ]; then
            echo "exec requires a command, e.g. ./run_container.sh exec pytest" >&2
            exit 1
        fi
        docker run --rm "${COMMON_FLAGS[@]}" "$IMAGE" "$@"
        ;;

    # "jupyter" starts JupyterLab in detached mode, mapping port 8888 to the host. It also sets up Jupyter with no token or password for easy access. Logs can be viewed with "./run_container.sh logs jupyter" and the container can be stopped with "./run_container.sh stop jupyter".
    jupyter)
        # Remove any existing container with the same name to avoid conflicts - idempotent restart (=restarts the container even if another is running). 
        docker rm -f scirex-jupyter >/dev/null 2>&1 || true
        docker run -d --name scirex-jupyter \
            --gpus all \
            "${COMMON_FLAGS[@]}" \
            -p 127.0.0.1:8888:8888 \
            "$IMAGE" \
            jupyter lab \
                --ip=0.0.0.0 --port=8888 --no-browser --allow-root 
        # Wait for Jupyter to print its startup URL (max ~15s), then show it
        echo "Waiting for Jupyter to start..."
        for i in {1..15}; do
            url=$(docker exec scirex-jupyter jupyter server list 2>/dev/null \
                    | grep -oE 'http://[^ ]+' | head -1 || true)
            if [ -n "$url" ]; then
                # Replace 0.0.0.0 with localhost since we bound to 127.0.0.1
                echo "JupyterLab URL (paste into VS Code):"
                # Replace whatever hostname Jupyter printed (0.0.0.0, container ID, etc.) with localhost
                echo "  ${url/http:\/\/*:8888/http:\/\/localhost:8888}"
                break
            fi
            sleep 1
        done
        echo "Stop with:  ./run_container.sh stop jupyter"
        echo "Logs with:  ./run_container.sh logs jupyter"
        ;;

    # "jupyter_nogpu" starts JupyterLab in detached mode, mapping port 8888 to the host without any access to GPU to avoid conflicts with other Docker container requiring full GPU access. 
    jupyter_nogpu)
        # Remove any existing container with the same name to avoid conflicts - idempotent restart (=restarts the container even if another is running). 
        docker rm -f scirex-jupyter >/dev/null 2>&1 || true
        docker run -d --name scirex-jupyter \
            "${COMMON_FLAGS[@]}" \
            -p 127.0.0.1:8888:8888 \
            "$IMAGE" \
            jupyter lab \
                --ip=0.0.0.0 --port=8888 --no-browser --allow-root 
        # Wait for Jupyter to print its startup URL (max ~15s), then show it
        echo "Waiting for Jupyter to start..."
        for i in {1..15}; do
            url=$(docker exec scirex-jupyter jupyter server list 2>/dev/null \
                    | grep -oE 'http://[^ ]+' | head -1 || true)
            if [ -n "$url" ]; then
                # Replace 0.0.0.0 with localhost since we bound to 127.0.0.1
                echo "JupyterLab URL (paste into VS Code):"
                # Replace whatever hostname Jupyter printed (0.0.0.0, container ID, etc.) with localhost
                echo "  ${url/http:\/\/*:8888/http:\/\/localhost:8888}"
                break
            fi
            sleep 1
        done
        echo "Stop with:  ./run_container.sh stop jupyter"
        echo "Logs with:  ./run_container.sh logs jupyter"
        ;;

    # "streamlit" starts the Streamlit demo app in detached mode, mapping port 8501 to the host. Logs can be viewed with "./run_container.sh logs streamlit" and the container can be stopped with "./run_container.sh stop streamlit".
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

    # "stop" allows you to stop and remove a detached container by name, e.g. "./run_container.sh stop jupyter" or "./run_container.sh stop streamlit"
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

    # "logs" allows you to follow logs from a detached container by name, e.g. "./run_container.sh logs jupyter" or "./run_container.sh logs streamlit"
    logs)
        target="${2:-}"
        if [ -z "$target" ]; then
            echo "logs requires a target, e.g. logs jupyter" >&2
            exit 1
        fi
        docker logs -f "scirex-${target}"
        ;;

    # "build" allows you to rebuild the Docker image after making changes to the Dockerfile or dependencies, e.g. "./run_container.sh build"
    build)
        docker build -t "$IMAGE" .
        ;;

    help|-h|--help)
        sed -n '2,12p' "$0"
        ;;

    # Catch-all for unknown modes - prints an error message and usage instructions
    *)
        echo "Unknown mode: $mode" >&2
        echo "Run './run_container.sh help' for usage." >&2
        exit 1
        ;;
esac