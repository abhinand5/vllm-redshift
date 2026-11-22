#!/usr/bin/env bash
# Kept for the old command line; the patch series is now cut by scripts/export-series.sh from models/<model>/series.conf.
# usage: scripts/update-patch.sh <model> <vllm-clone> <branch>
exec "$(dirname "$0")/export-series.sh" "$@"
