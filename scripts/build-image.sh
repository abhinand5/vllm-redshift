#!/usr/bin/env bash
# usage: scripts/build-image.sh <model>   (MAX_JOBS default 6: 30 GiB RAM box; DEPS_IDX: see the Dockerfile)
set -euo pipefail
cd "$(dirname "$0")/.."; M=$1; . models/$M/model.env
ARGS=$(scripts/build-context.sh "$M")
EXTRA=(); [ -n "${DEPS_IDX:-}" ] && EXTRA=(--build-arg "DEPS_IDX=$DEPS_IDX")
docker build -f docker/Dockerfile.rocm-gfx1201 --build-arg MAX_JOBS=${MAX_JOBS:-6} "${EXTRA[@]}" \
  $(sed 's/^/--build-arg /' <<<"$ARGS") -t "$IMAGE" build/$M/src
