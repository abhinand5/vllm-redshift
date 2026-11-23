#!/usr/bin/env bash
# usage: scripts/create-container.sh <model> [name] [extra vllm serve args]
# A persistent (not --rm) container with serve.sh's exact args, for managers that `docker start`/`docker stop` it
# (e.g. modelctl kind="docker"). Default name vllm-<model>. Recreate after a new image or model.env change.
set -euo pipefail
cd "$(dirname "$0")/.."; M=$1; N=${2:-vllm-$M}; shift $(( $# > 1 ? 2 : 1 )); . models/$M/model.env; . scripts/docker-args.sh
if docker inspect "$N" >/dev/null 2>&1; then
  [ "$(docker inspect -f '{{.State.Running}}' "$N")" = false ] || { echo "$N is running; stop it first"; exit 1; }
  docker rm "$N" >/dev/null; echo "removed old $N"
fi
# modelctl shows this label as the engine, so a recreate keeps its version current.
OPTS=$(compgen -e | { grep -E '^VLLM_INKLING2_' || true; } | while read -r v; do printf ' %s=%s' "$v" "${!v}"; done)
ENGINE="vllm-redshift v$VERSION (vLLM ${VLLM_COMMIT:0:10} + gfx1201 series)${OPTS:+ opts:$OPTS}"
docker create --name "$N" --label "modelctl.engine=$ENGINE" "${DOCKER_ARGS[@]}" "$IMAGE" "${SERVE_CMD[@]}" "$@" >/dev/null
echo "created $N ($ENGINE) from $IMAGE ($(docker image inspect -f '{{.Id}}' "$IMAGE" | cut -c8-19)); start with: docker start $N"
