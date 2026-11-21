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

| model | vLLM base | decode, batch 1 | notes |
|---|---|---|---|
| `inkling2-mini` (thinkingmachines/Inkling-2-Mini, NVFP4 + MXFP8 weights as shipped) | `bf151b7eb1` (main, 2026-11-21) | **87 tok/s with DSpark** (coding prompt), **70 tok/s without**; llama.cpp Q8: 31 | defaults: 196,608 context, 2 concurrent requests sharing a 232K-token FP8 KV pool, DSpark 7-token drafts. Tested: two concurrent 94K-token needle prompts both answered, 69 tok/s decode at 94K. Prefill ~1.1K tok/s (not tuned). Greedy top-1 vs GGUF Q8: 99.2% BF16 KV, 97.9% FP8 KV |

The Inkling patch = ROCm port of Inkling 2 (PR #67685) + Mini loop core (PR #69028) + gfx1201 kernels
(NVFP4/MXFP8 expert GEMMs, decode GEMV, unit-scale FP8 KV, tuning) + `GPU_MAX_HW_QUEUES=2` default on RDNA4 (removes ~15 us
gaps between kernels in HIP graphs: 38 -> 60 tok/s on its own).

## Quick start

```bash
git clone --recursive https://github.com/abhinand5/vllm-redshift && cd vllm-redshift
scripts/build-image.sh inkling2-mini                  # once
MODEL_DIR=~/models/Inkling-2-Mini scripts/serve.sh inkling2-mini
curl localhost:8005/v1/chat/completions -H content-type:application/json \
  -d '{"model":"Inkling-2-Mini","messages":[{"role":"user","content":"hi"}],"reasoning_effort":"none"}'
```
Env overrides: `MAX_MODEL_LEN` (196608), `PORT` (8005), `KV_CACHE_BYTES` (8.5 GiB: one 196K request needs 7.2 GiB
with DSpark; the rest lets a second request share the pool), `SPEC_ARGS=` (empty: no DSpark, 70 tok/s, frees the
draft's memory). Extra args go to `vllm serve`. DSpark only supports `num_speculative_tokens` 7 (checkpoint block 8).
The KV budget is explicit because vLLM's automatic sizing keeps a fixed 3 GiB ROCm margin on top of the measured
activation peak (it would offer 6.97 GiB, too little for 196K).

Native (no docker): needs a venv with torch 2.13+rocm10.0, Triton 3.8 and a vLLM build of the same commit+patch
(its `.so` files are symlinked into `build/<model>/src/vllm`):
`VENV=/path/.venv-rocm SO_DIR=/path/vllm-build/vllm scripts/serve-native.sh inkling2-mini`.

## Rules of the road (R9700)
- Never use a system ROCm 10.1 HIP runtime (segfaults in `hipStreamCreateWithFlags`); the image ships its own.
- `HIP_VISIBLE_DEVICES=0` when an iGPU is present.
- Updating vLLM: in a full vLLM clone, rebase the model branch onto the new upstream commit, then
  `scripts/update-patch.sh <model> <clone> <branch> <new-commit>` and rebuild.
