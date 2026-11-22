# Sourced by serve.sh and create-container.sh: the container's docker args and command, from model.env.
# Sets DOCKER_ARGS and SERVE_CMD (bash arrays). Requires model.env to be sourced first.
# ROCPROFILER_REGISTER_ENABLED=0: with the profiler hook attached, ROCr's AsyncEventsLoop thread busy-polls one CPU
# core at 100% from the first kernel on, even when idle (pip ROCm 10.0 + torch). Serving does not use rocprofiler.
DOCKER_ARGS=(--device /dev/kfd --device /dev/dri --group-add video --security-opt seccomp=unconfined --ipc host
  --shm-size 8g -p "127.0.0.1:$PORT:$PORT" -e HIP_VISIBLE_DEVICES=0 -e ROCR_VISIBLE_DEVICES=0
  -e ROCPROFILER_REGISTER_ENABLED=0
  -v "$MODEL_DIR":/model:ro -v redshift-cache:/root/.cache)
# shellcheck disable=SC2206  # SERVE_ARGS / SPEC_ARGS are flag strings meant to split on spaces
SERVE_CMD=(vllm serve /model --served-model-name "$SERVED_NAME" --host 0.0.0.0 --port "$PORT"
  --max-model-len "$MAX_MODEL_LEN" $SERVE_ARGS $SPEC_ARGS)
