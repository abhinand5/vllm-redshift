# inkling2-mini changelog

Versions follow semver for people serving the model: MAJOR = incompatible serve flags/defaults, API behaviour or
weights; MINOR = new vLLM base or new optimizations; PATCH = fixes only. Release tag: `inkling2-mini/vX.Y.Z`
(must equal `VERSION` in `model.env`). Numbers: 1x R9700, FP8 KV, `scripts/release-check.sh` unless noted.

## v0.3.1 (2026-11-22)
vLLM base `bf151b7eb1` (unchanged). Same weights and image; serve flag change only.
- DSpark drafts are now sampled (`"draft_sample_method":"probabilistic"`) instead of argmax. Requests at the model's
  default sampling (T=1.0, top_p 1.0, top_k 0, min_p 0) used to accept only 1.6-2.2 tokens/step, because a greedy draft
  is accepted with the target's probability of that one token. 1024-token decode, 11 seeds at T=1.0: 45K-token code
  prompt 101.2 -> 118.6 tok/s (+17%), prose 73.2 -> 79.6 (+9%), short code 102.3 -> 102.0 (same). T=0 unchanged.

## v0.3.0 (2026-11-22)
vLLM base `bf151b7eb1` (unchanged). Same weights, numbers as v0.2.0 (decode 78-113 tok/s, prefill 4.6K / 3.7K at 4K / 64K).
- Image input on (`--limit-mm-per-prompt {"image":64}`; agents resend history, so the cap counts every image in a session): the ViT runs vLLM's Triton varlen attention on ROCm (no
  upstream flash-attn needed) and keeps encoder graphs. Costs ~0.6 GiB VRAM; KV pool unchanged (416,701 tokens).
  Text decode/prefill unchanged.
- Idle CPU: `ROCPROFILER_REGISTER_ENABLED=0` in the image and container args; ROCr's AsyncEventsLoop no longer
  busy-polls a CPU core at 100% while the server is idle.

## v0.2.0 (2026-11-22)
vLLM base `bf151b7eb1` (unchanged). Patch series: 2 upstream PRs (#67685, #69028), 4 core fixes, the gfx1201 backend.
- Decode without speculation 70 -> ~78 tok/s: NVFP4 decode GEMV for 1-2 tokens, launch fusion (1002 -> 762 launches/step).
- DSpark: multi-row verify GEMV and adaptive verification (on by default); FP8 draft KV. API decode 77-104 tok/s.
- Prefill 3.3K / 1.4K -> 4.6K / 3.7K tok/s at 4K / 64K (new prefill attention kernel, exact FP8 KV decode).
- KV per 196K request 7.2 -> 3.9 GiB (tail-only conv-state cache): pool 232K -> 416,701 tokens, 2.12x at 196,608.
  Tested: two concurrent 191K-token needle prompts both answered.
- `SPEC_ARGS` default adds `"kv_cache_dtype":"fp8","enable_adaptive_verification":true`; other serve flags unchanged.

## v0.1.0 (2026-11-21)
First release (`43cfec0`). vLLM base `bf151b7eb1` + one patch.
- 196,608 context, 2 concurrent requests sharing a 232K-token FP8 KV pool, DSpark 7-token drafts on by default.
- Decode 70 tok/s without speculation, 87 with DSpark (coding prompt); prefill 3.3K / 1.4K tok/s at 4K / 64K (offline).
