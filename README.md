# vllm-redshift

<p align="center">
  <img src="./assets/banner.png" alt="vLLM RedShift — Fast vLLM ports for Radeon RDNA GPUs" width="640">
</p>

Fast vLLM ports for **one AMD Radeon AI PRO R9700 (gfx1201, RDNA4, 32 GB)**, one model at a time.
Upstream vLLM is a pinned git submodule (`vllm/`, never edited). Each model directory holds the upstream
commit it targets and a patch series on top of it, so models can sit on different vLLM versions and a rebase is
"bump the commit, refresh the patch".

```
vllm/                         upstream vllm-project/vllm (submodule)
models/<model>/model.env      VLLM_COMMIT, PATCHES (in order), image tag, serve flags
models/<model>/series.conf    how each patch is cut from the dev branch (revisions + paths)
models/<model>/patches/       upstream/  open upstream PRs the model needs (dropped as they merge)
                              core/      small generic vLLM fixes, one per patch
                              model/     the model's AMD backend + its tests
docker/Dockerfile.rocm-gfx1201  pip ROCm 10.0 (TheRock) + torch 2.13 + Triton 3.8, no system ROCm
scripts/prepare.sh <model>      build/<model>/src = vllm@VLLM_COMMIT + patches
scripts/build-image.sh <model>  docker image (builds vLLM's C++/Rust for gfx1201; ~30-60 min, MAX_JOBS=6)
scripts/serve.sh <model>        OpenAI server in docker on 127.0.0.1:$PORT
scripts/serve-native.sh <model> same without docker, from an existing pip-ROCm venv (see below)
scripts/export-series.sh <model> <vllm-clone> <branch>   re-cut the series from a dev branch, check it rebuilds it
scripts/release-check.sh <model> serve the image, measure it, compare with the last run (reports only)
results/<model>/*.json          numbers from each release check
```

### Workflow
Development happens on a branch in a full vLLM clone (no fork or remote needed). To ship:
1. `scripts/export-series.sh <model> <clone> <branch>`: writes the patches, fails if a changed path is not covered by
   `series.conf`, and proves `VLLM_COMMIT + series` is byte-identical to the branch.
2. `scripts/build-image.sh <model>`, then `scripts/release-check.sh <model>`: prints this build next to the previous one.
3. Decide, commit, push. The check never blocks a release.

## Models

| model | vLLM base | decode, batch 1 | notes |
|---|---|---|---|
| `inkling2-mini` (thinkingmachines/Inkling-2-Mini, NVFP4 + MXFP8 weights as shipped) | `bf151b7eb1` (main, 2026-11-21) | **77-104 tok/s with DSpark** (adaptive verification; API, 512 tokens), **~78 tok/s without**; llama.cpp Q8: 31 | defaults: 196,608 context, 2 concurrent requests sharing a 416,701-token FP8 KV pool (2.12x). Tested: two concurrent 191K-token needle prompts both answered, 155 s wall. Prefill 4.6K / 4.4K / 3.7K tok/s at 4K / 16K / 64K (`results/inkling2-mini/`). Greedy top-1 vs GGUF Q8 (BF16 KV): 97-99%, varies with fp32 summation order at near-ties |

The Inkling series: two open upstream PRs (#67685 ROCm port of Inkling 2 Small, #69028 Mini loop core), four generic fixes
(`GPU_MAX_HW_QUEUES=2` default on RDNA4: 38 -> 60 tok/s on its own; gfx12 bring-up; a tail-only sliding-window KV manager
for conv state: 7.2 -> 3.9 GiB per 196K request; adaptive-verification profiling with routed inputs) and the gfx1201
backend (NVFP4/MXFP8 expert GEMMs, decode GEMV, prefill attention, multi-row verify GEMV, launch fusion, FP8 KV).

## Quick start

```bash
git clone --recursive https://github.com/abhinand5/vllm-redshift && cd vllm-redshift
scripts/build-image.sh inkling2-mini                  # once
MODEL_DIR=~/models/Inkling-2-Mini scripts/serve.sh inkling2-mini
curl localhost:8005/v1/chat/completions -H content-type:application/json \
  -d '{"model":"Inkling-2-Mini","messages":[{"role":"user","content":"hi"}],"reasoning_effort":"none"}'
```
Env overrides: `MAX_MODEL_LEN` (196608), `PORT` (8005), `KV_CACHE_BYTES` (8.5 GiB: one 196K request needs 3.9 GiB
with DSpark, so two fit), `SPEC_ARGS=` (empty: no DSpark, ~78 tok/s, frees the
draft's memory). Extra args go to `vllm serve`. DSpark only supports `num_speculative_tokens` 7 (checkpoint block 8).
The KV budget is explicit because vLLM's automatic sizing keeps a fixed 3 GiB ROCm margin on top of the measured
activation peak (it would offer 6.97 GiB, too little for 196K).

Native (no docker): needs a venv with torch 2.13+rocm10.0, Triton 3.8 and a vLLM build of the same commit+patch
(its `.so` files are symlinked into `build/<model>/src/vllm`):
`VENV=/path/.venv-rocm SO_DIR=/path/vllm-build/vllm scripts/serve-native.sh inkling2-mini`.

## Rules of the road (R9700)
- Never use a system ROCm 10.1 HIP runtime (segfaults in `hipStreamCreateWithFlags`); the image ships its own.
- `HIP_VISIBLE_DEVICES=0` when an iGPU is present.
- Updating vLLM: in the vLLM clone, rebase the model branch onto the new upstream commit (drop `upstream/` patches whose
  PR has merged), set `VLLM_COMMIT` and the revisions in `series.conf`, then `scripts/export-series.sh` and rebuild.
