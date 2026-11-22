#!/usr/bin/env bash
# usage: scripts/build-image.sh <model>   (MAX_JOBS default 6: 30 GiB RAM box)
set -euo pipefail
cd "$(dirname "$0")/.."; M=$1; . models/$M/model.env
ARGS=$(scripts/build-context.sh "$M")
docker build -f docker/Dockerfile.rocm-gfx1201 --build-arg MAX_JOBS=${MAX_JOBS:-6} \
  $(sed 's/^/--build-arg /' <<<"$ARGS") -t "$IMAGE" build/$M/src
