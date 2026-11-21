# vllm-redshift

Fast vLLM ports for **one AMD Radeon AI PRO R9700 (gfx1201, RDNA4, 32 GB)**, one model at a time.
Upstream vLLM is a pinned git submodule (`vllm/`, never edited). Each model directory holds the upstream
commit it targets and a patch series on top of it, so models can sit on different vLLM versions and a rebase is
"bump the commit, refresh the patch".

```
vllm/                         upstream vllm-project/vllm (submodule)
models/<model>/model.env      VLLM_COMMIT, PATCHES, image tag, serve flags
models/<model>/patches/*.patch
docker/Dockerfile.rocm-gfx1201  pip ROCm 10.0 (TheRock) + torch 2.13 + Triton 3.8, no system ROCm
scripts/prepare.sh <model>      build/<model>/src = vllm@VLLM_COMMIT + patches
scripts/build-image.sh <model>  docker image (builds vLLM's C++/Rust for gfx1201; ~30-60 min, MAX_JOBS=6)
scripts/serve.sh <model>        OpenAI server in docker on 127.0.0.1:$PORT
scripts/serve-native.sh <model> same without docker, from an existing pip-ROCm venv (see below)
scripts/update-patch.sh <model> <vllm-clone> <branch>   refresh the patch from a dev branch
```

## Models

| model | vLLM base | decode, batch 1, no spec decoding | notes |
|---|---|---|---|
| `inkling2-mini` (thinkingmachines/Inkling-2-Mini, NVFP4 + MXFP8 as shipped, lossless) | `bf151b7eb1` (main, 2026-11-21) | **70 tok/s** (128-token prompt, 256 out); llama.cpp Q8: 31 | 99.2% greedy top-1 vs GGUF Q8 reference; 24.8 GiB VRAM; DSpark speculation ported but not tuned on the real checkpoint; validated up to 32K context (196K target not yet soaked) |

The Inkling patch = ROCm port of Inkling 2 (PR #67685) + Mini loop core (PR #69028) + gfx1201 kernels
(NVFP4/MXFP8 expert GEMMs, decode GEMV, tuning) + `GPU_MAX_HW_QUEUES=2` default on RDNA4 (removes ~15 us
gaps between kernels in HIP graphs: 38 -> 60 tok/s on its own).

## Quick start

```bash
git clone --recursive <this repo> && cd vllm-redshift
scripts/build-image.sh inkling2-mini                  # once
MODEL_DIR=~/models/Inkling-2-Mini scripts/serve.sh inkling2-mini
curl localhost:8005/v1/chat/completions -H content-type:application/json \
  -d '{"model":"Inkling-2-Mini","messages":[{"role":"user","content":"hi"}],"reasoning_effort":"none"}'
```
`MAX_MODEL_LEN` (default 32768) and `PORT` (8005) are env overrides; extra args go to `vllm serve`
(e.g. `--speculative-config '{"method":"dspark","num_speculative_tokens":3}'`).

Native (no docker): needs a venv with torch 2.13+rocm10.0, Triton 3.8 and a vLLM build of the same commit+patch
(its `.so` files are symlinked into `build/<model>/src/vllm`):
`VENV=/path/.venv-rocm SO_DIR=/path/vllm-build/vllm scripts/serve-native.sh inkling2-mini`.

## Rules of the road (R9700)
- Never use a system ROCm 10.1 HIP runtime (segfaults in `hipStreamCreateWithFlags`); the image ships its own.
- `HIP_VISIBLE_DEVICES=0` when an iGPU is present.
- Updating vLLM: in a full vLLM clone, rebase the model branch onto the new upstream commit, then
  `scripts/update-patch.sh <model> <clone> <branch> <new-commit>` and rebuild.
