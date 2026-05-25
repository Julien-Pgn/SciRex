#!/usr/bin/env bash
set -euo pipefail

IMAGE="vllm/vllm-openai:v0.17.0"
NAME="chandra-vllm"

CMD="${1:-up}"

case "$CMD" in
  up)
    docker run -d --rm \
      --gpus all \
      --ipc=host \
      -p 8000:8000 \
      -p 8888:8888 \
      -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
      -v "$(pwd):/workspace" \
      -w /workspace \
      --name "$NAME" \
      "$IMAGE" \
      --model datalab-to/chandra-ocr-2 \
      --dtype bfloat16 \
      --max-model-len 8192 \
      --gpu-memory-utilization 0.70 \
      --quantization bitsandbytes \
      --served-model-name chandra

    echo "Container started. Following logs (Ctrl+C to detach)."
    echo "Wait for 'Starting vLLM API server' before running 'jupyter' subcommand."
    docker logs -f "$NAME"
    ;;

  jupyter)
    # Installe jupyter + ipykernel + libs dont on a besoin, puis lance un server
    docker exec -d "$NAME" bash -c "
      pip install -q jupyter ipykernel pymupdf 2>&1 | tail -3
      jupyter server \
        --ip=0.0.0.0 \
        --port=8888 \
        --no-browser \
        --allow-root \
        --ServerApp.token='' \
        --ServerApp.password='' \
        --ServerApp.disable_check_xsrf=True \
        > /tmp/jupyter.log 2>&1
    "
    sleep 5
    echo ""
    echo "Jupyter server starting in container."
    echo "Check it's up:  ./run_vllm.sh jupyter-url"
    echo "Use this URL in VS Code:  http://localhost:8888/"
    ;;

  jupyter-url)
    # Récupère l'URL avec token (vide ici, donc juste l'URL)
    docker exec "$NAME" bash -c "jupyter server list 2>&1 | tail -n +2"
    ;;

  jupyter-logs)
    docker exec "$NAME" cat /tmp/jupyter.log
    ;;

  exec)
    docker exec -it "$NAME" bash
    ;;

  logs)
    docker logs -f "$NAME"
    ;;

  stop)
    docker stop "$NAME"
    ;;

  status)
    docker ps --filter "name=$NAME"
    ;;

  *)
    echo "Usage: $0 {up|jupyter|jupyter-url|jupyter-logs|exec|logs|stop|status}"
    exit 1
    ;;
esac