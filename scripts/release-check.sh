#!/usr/bin/env bash
# Serve the model's built image, measure it with scripts/release_check.py, stop it, and print the numbers next to the
# previous run. Reports only: it never blocks a release; whether to ship is your call.
# usage: scripts/release-check.sh <model> [--quick] [extra release_check.py args]
#   holds the shared GPU lock (/tmp/r9700-gpu.lock) for the whole run; results/<model>/<date>-<image>.json
set -euo pipefail
cd "$(dirname "$0")/.."; M=$1; shift; . models/$M/model.env
LOCK=${GPU_LOCK:-/tmp/r9700-gpu.lock}
if [ -z "${_RC_LOCKED:-}" ]; then _RC_LOCKED=1 exec flock -w 3600 "$LOCK" "$0" "$M" "$@"; fi
IMG=$(docker image inspect -f '{{.Id}}' "$IMAGE" | cut -c8-19)
# opt-in flags in effect (passed into the container by docker-args.sh) are recorded and suffix the result name
OPTS=$(compgen -e | { grep -E '^(VLLM_INKLING2_|VLLM_ADAPTIVE_VERIFICATION_)' || true; } | while read -r v; do printf '%s=%s ' "$v" "${!v}"; done)
OPTS=${OPTS% }
TAG=v$VERSION-$(date +%Y%m%d-%H%M)-$IMG${OPTS:+-opt}; mkdir -p build/logs results/$M; LOG=build/logs/release-check-$TAG.log
scripts/serve.sh "$M" > "$LOG" 2>&1 &
trap 'docker stop "redshift-$M" >/dev/null 2>&1 || true' EXIT
for i in $(seq 1 120); do
  curl -sf "http://127.0.0.1:$PORT/health" >/dev/null && break
  docker ps --format '{{.Names}}' | grep -qx "redshift-$M" || { [ $i -gt 5 ] && { echo "server exited; see $LOG"; tail -20 "$LOG"; exit 1; }; }
  grep -vE 'torchcodec|_has_module|find_spec' "$LOG" | grep -qE 'out of memory|EngineCore failed' && { echo "server failed; see $LOG"; exit 1; }
  sleep 5
done
curl -sf "http://127.0.0.1:$PORT/health" >/dev/null || { echo "server not ready after 10 min; see $LOG"; exit 1; }
POOL=$(grep -o 'GPU KV cache size: [0-9,]* tokens, Maximum concurrency for [0-9,]* tokens per request: [0-9.]*x' "$LOG" | head -1)
META=$(printf '{"version":"v%s","image":"%s","patches":"%s","vllm_commit":"%s","kv_pool":"%s","opts":"%s"}' "$VERSION" "$IMG" "$PATCHES" "$VLLM_COMMIT" "$POOL" "$OPTS")
echo "server ready; $POOL${OPTS:+; opts: $OPTS}"
python3 scripts/release_check.py --model-dir models/$M --out results/$M/$TAG.json --port "$PORT" \
  --served-name "$SERVED_NAME" --meta "$META" "$@"
echo "results/$M/$TAG.json  (server log $LOG)"
