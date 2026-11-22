#!/usr/bin/env bash
# usage: scripts/serve.sh <model> [extra vllm serve args]   (docker; GPU 0 only, OpenAI API on $PORT; removed on exit)
set -euo pipefail
cd "$(dirname "$0")/.."; M=$1; shift; . models/$M/model.env; . scripts/docker-args.sh
exec docker run --rm --name "redshift-$M" "${DOCKER_ARGS[@]}" "$IMAGE" "${SERVE_CMD[@]}" "$@"
