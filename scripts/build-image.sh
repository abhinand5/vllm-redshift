#!/usr/bin/env bash
# usage: scripts/build-image.sh <model>   (MAX_JOBS default 6: 30 GiB RAM box)
set -euo pipefail
cd "$(dirname "$0")/.."; M=$1; . models/$M/model.env
scripts/prepare.sh "$M"
cp docker/constraints-rocm10.txt docker/requirements-rocm-r9700.txt build/$M/src/requirements/
docker build -f docker/Dockerfile.rocm-gfx1201 --build-arg MAX_JOBS=${MAX_JOBS:-6} \
  --build-arg VLLM_VERSION=0.34.0+redshift.${M}.${VLLM_COMMIT:0:10} -t "$IMAGE" build/$M/src
