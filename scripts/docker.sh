#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="wonderdesk:local"
ACTION="${1:-}"

case "$ACTION" in
  build)
    docker build --tag "$IMAGE" "$ROOT"
    ;;
  run)
    token_file="${HF_HOME:-$HOME/.cache/huggingface}/token"
    docker_args=(run --rm --name wonderdesk --publish 127.0.0.1:8080:8080
      --volume wonderdesk-model-cache:/home/llm/.cache/huggingface)
    if [[ "${2:-}" == "--gpu" ]]; then
      docker_args+=(--gpus all)
    elif [[ $# -gt 1 ]]; then
      echo "Usage: ./scripts/docker.sh build|run [--gpu]" >&2
      exit 2
    fi
    if [[ -f "$token_file" ]]; then
      docker_args+=(--mount "type=bind,source=$token_file,target=/run/secrets/hf_token,readonly"
        --env HF_TOKEN_PATH=/run/secrets/hf_token)
    fi
    docker "${docker_args[@]}" "$IMAGE"
    ;;
  *)
    echo "Usage: ./scripts/docker.sh build|run [--gpu]" >&2
    exit 2
    ;;
esac
