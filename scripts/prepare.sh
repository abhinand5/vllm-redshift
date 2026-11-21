#!/usr/bin/env bash
# usage: scripts/prepare.sh <model>   -> build/<model>/src = upstream vLLM at the model's commit + its patches
set -euo pipefail
cd "$(dirname "$0")/.."; M=$1; . models/$M/model.env
git submodule update --init vllm
git -C vllm cat-file -e "$VLLM_COMMIT^{commit}" 2>/dev/null || git -C vllm fetch origin "$VLLM_COMMIT"
OUT=build/$M/src; rm -rf "$OUT"; mkdir -p "$OUT"
git -C vllm archive "$VLLM_COMMIT" | tar -x -C "$OUT"
# Apply in a throwaway repo: inside this repo's tree, git apply would silently skip the paths.
git -C "$OUT" init -q
for p in $PATCHES; do git -C "$OUT" apply --whitespace=nowarn "$PWD/models/$M/patches/$p"; echo "applied $p"; done
rm -rf "$OUT/.git"
test -n "$(ls "$OUT"/vllm/models/*/amd 2>/dev/null)" || { echo "patch did not apply"; exit 1; }
echo "$OUT ready (vLLM $VLLM_COMMIT + ${PATCHES})"
