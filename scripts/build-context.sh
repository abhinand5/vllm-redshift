#!/usr/bin/env bash
# usage: scripts/build-context.sh <model>   -> build/<model>/src ready as the docker build context; prints the
# build args on stdout (KEY=VALUE lines) so CI and scripts/build-image.sh build the same thing.
set -euo pipefail
cd "$(dirname "$0")/.."; M=$1; . models/$M/model.env
scripts/prepare.sh "$M" >&2
cp docker/constraints-rocm10.txt docker/requirements-rocm-r9700.txt build/$M/src/requirements/
echo "VLLM_VERSION=0.34.0+redshift.${M}.${VLLM_COMMIT:0:10}"
