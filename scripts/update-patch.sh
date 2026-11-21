#!/usr/bin/env bash
# Regenerate a model's patch from a dev branch in a full vLLM clone:
# usage: scripts/update-patch.sh <model> <vllm-clone> <branch> [base-commit]   (base defaults to VLLM_COMMIT)
set -euo pipefail
cd "$(dirname "$0")/.."; M=$1; C=$2; BR=$3; . models/$M/model.env; BASE=${4:-$VLLM_COMMIT}
git -C "$C" diff --binary "$BASE" "$BR" -- . ':!.claude' ':!.buildkite' > models/$M/patches/0001-inkling2-mini-r9700.patch
sed -i "s/^VLLM_COMMIT=[0-9a-f]*/VLLM_COMMIT=$BASE/" models/$M/model.env
echo "patch from $BASE..$BR ($(git -C "$C" rev-parse --short "$BR")); run scripts/prepare.sh $M to check it applies"
