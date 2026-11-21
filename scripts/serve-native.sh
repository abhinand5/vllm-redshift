#!/usr/bin/env bash
# Without docker: run build/<model>/src with an existing pip-ROCm-10 venv whose vLLM was built from the same
# commit + patches (its compiled .so files are symlinked in). usage: VENV=... SO_DIR=... scripts/serve-native.sh <model>
set -euo pipefail
cd "$(dirname "$0")/.."; M=$1; shift; . models/$M/model.env
VENV=${VENV:-$HOME/dev/inference-engines/vllm-inkling2-mini-r9700/.venv-rocm}
SO_DIR=${SO_DIR:-$HOME/dev/inference-engines/vllm-inkling2-mini-r9700/vllm}
[ -d build/$M/src ] || scripts/prepare.sh "$M"
for f in "$SO_DIR"/*.so; do ln -sf "$f" build/$M/src/vllm/; done
. "$VENV/bin/activate"; sp=$VENV/lib/python3.12/site-packages
export LD_LIBRARY_PATH="$sp/_rocm_sdk_core/lib:$sp/_rocm_sdk_libraries/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export ROCM_PATH=$sp/_rocm_sdk_devel ROCM_HOME=$sp/_rocm_sdk_devel HIP_PATH=$sp/_rocm_sdk_devel   # never /opt/rocm
export HIP_VISIBLE_DEVICES=0 ROCR_VISIBLE_DEVICES=0 PYTHONPATH=$PWD/build/$M/src
exec python -m vllm.entrypoints.cli.main serve "$MODEL_DIR" --served-model-name "$SERVED_NAME" --host 127.0.0.1 \
  --port "$PORT" --max-model-len "$MAX_MODEL_LEN" $SERVE_ARGS "$@"
