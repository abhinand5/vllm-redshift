#!/usr/bin/env bash
# usage: scripts/serve.sh <model> [extra vllm serve args]   (docker; GPU 0 only, OpenAI API on $PORT)
set -euo pipefail
cd "$(dirname "$0")/.."; M=$1; shift; . models/$M/model.env
exec docker run --rm --name "redshift-$M" --device /dev/kfd --device /dev/dri --group-add video \
  --security-opt seccomp=unconfined --ipc host --shm-size 8g -p 127.0.0.1:$PORT:$PORT \
  -e HIP_VISIBLE_DEVICES=0 -e ROCR_VISIBLE_DEVICES=0 \
  -v "$MODEL_DIR":/model:ro -v redshift-cache:/root/.cache \
  "$IMAGE" vllm serve /model --served-model-name "$SERVED_NAME" --host 0.0.0.0 --port "$PORT" \
  --max-model-len "$MAX_MODEL_LEN" $SERVE_ARGS "$@"
