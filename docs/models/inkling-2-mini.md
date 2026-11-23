# Inkling-2-Mini on vllm-redshift: user guide

How to run [`thinkingmachines/Inkling-2-Mini`](https://huggingface.co/thinkingmachines/Inkling-2-Mini) on one Radeon AI
PRO R9700 (gfx1201, 32 GB), what the defaults are, and what every switch costs. Numbers are from this repo's release
check (`scripts/release-check.sh`, T=0, 512 generated tokens) unless a row says otherwise; per-version details are in
[`models/inkling2-mini/CHANGELOG.md`](../../models/inkling2-mini/CHANGELOG.md).

## The default setup

The default is **lossless with respect to the released weights**: NVFP4 and MXFP8 experts are read as shipped, and every
other weight stays BF16. Nothing is requantized. Two approximations are on by default, as in upstream vLLM deployments of
this model:

- **FP8 (e4m3) KV cache.** It halves the cache, which is what makes 196K context fit twice.
- **DSpark speculative decoding.** A 3-layer drafter proposes 7 tokens, and the full model checks them. Output follows
  the full model's distribution: greedy output is the full model's argmax, and sampling uses rejection sampling.

| setting | default | where |
|---|---|---|
| context | 196,608 tokens (the model's full window) | `MAX_MODEL_LEN` |
| concurrent requests | 2 | `--max-num-seqs 2` in `SERVE_ARGS` |
| KV cache | FP8, 9.13 GB = 416,701 tokens (2.12 full-length requests) | `KV_CACHE_BYTES` |
| speculative decoding | DSpark, 7 draft tokens, adaptive verify width, sampled drafts | `SPEC_ARGS` |
| reasoning / tools | `--reasoning-parser inkling --tool-call-parser inkling --enable-auto-tool-choice` | `SERVE_ARGS` |
| images | up to 64 per prompt | `--limit-mm-per-prompt` |
| port | 8005, bound to 127.0.0.1 | `PORT` |

Measured with the defaults (v0.5.1 image; one release-check run each, so decode varies by a few percent between runs):

| | tok/s |
|---|---|
| decode: story / code / explain | 89.9 / 119.6 / 95.8 |
| prefill: 4K / 16K / 64K tokens | 4,582 / 4,495 / 3,741 |
| decode with a ~41K-token prompt (dev server, T=1, median) | ~137 |
| decode at ~159K context (5 runs) | 118-136 |
| decode without the drafter (`SPEC_ARGS=`) | 78 |

Decode speed with the drafter depends on how predictable the text is. Code and long, repetitive contexts run fastest;
free prose is slowest.

## Running it

```bash
scripts/build-image.sh inkling2-mini                       # or IMAGE=ghcr.io/abhinand5/vllm-redshift:inkling2-mini
MODEL_DIR=~/models/Inkling-2-Mini scripts/serve.sh inkling2-mini      # foreground, removed on exit
scripts/create-container.sh inkling2-mini                  # persistent container `vllm-inkling2-mini`; docker start/stop it
```

Everything after the model name is passed to `vllm serve` and placed after the defaults. Since the last copy of a flag
wins, `scripts/serve.sh inkling2-mini --max-num-seqs 4` overrides the default. The environment variables below (`MODEL_DIR`,
`PORT`, `MAX_MODEL_LEN`, `KV_CACHE_BYTES`, `SPEC_ARGS`, `IMAGE`) are read by `models/inkling2-mini/model.env`. Any
`VLLM_INKLING2_*` or `VLLM_ADAPTIVE_VERIFICATION_*` variable set on the host is passed into the container (this works for
`serve.sh`, `create-container.sh` and `release-check.sh`).

**First start after an install or upgrade is slow.** Startup takes about 1.5-2 minutes: loading the weights (~20 GB),
capturing HIP graphs, and a short speculative-decoding profile. New Triton kernels are compiled while the server serves
its first requests, and cached in the `redshift-cache` docker volume. The first release check after the v0.5.0 upgrade
measured 65 / 84 / 68 tok/s decode and 2,973 tok/s at 4K prefill. The next start, with a warm cache, measured 91 / 134 /
92 and 4,610. Send a few warm-up requests (short and long) before you benchmark. Keep the volume between upgrades;
removing it only costs a recompile.

## Context, concurrency and KV cache

The KV pool is a fixed byte budget (`KV_CACHE_BYTES`, default 9,126,805,504), shared by all running requests. At FP8 one
196K-token request takes about half of it, so:

| goal | how |
|---|---|
| two full-length (196K) requests at once | default |
| more short requests at once | `scripts/serve.sh inkling2-mini --max-num-seqs 4` (each still up to 196K, but the pool holds 416,701 tokens in total; long requests then wait) |
| shorter context, faster startup, more free VRAM | `MAX_MODEL_LEN=65536 KV_CACHE_BYTES=4000000000 scripts/serve.sh inkling2-mini` |
| use the VRAM freed by `VLLM_INKLING2_FP8_LM_HEAD` for KV | raise `KV_CACHE_BYTES` by up to ~0.8 GB (≈+36K tokens) |

Prefix caching is on (vLLM default). Repeated prompt prefixes, such as a chat history or a fixed system prompt, are not
prefilled again.

## Reasoning, tools and images

- **Reasoning.** The model thinks before it answers. With the `inkling` reasoning parser, the thinking comes back in
  the message's `reasoning` field, separate from `content`. Set how hard it thinks per request with
  `"chat_template_kwargs": {"reasoning_effort": "low"}`. Values: `none`, `minimal`, `low`, `medium`, `high` (default),
  `xhigh` or `max`, or a number in [0, 0.99]. The value is a hint to the model, not a hard switch: on "Is 221 prime?",
  `none` still produced 294 characters of reasoning, against 370 at `high`. Cap total output with `max_tokens`.
- **Tools.** OpenAI-style `tools` / `tool_choice` work, including `"auto"`. Tool calls come back as `tool_calls`.
- **Images.** Send them as OpenAI `image_url` content parts, up to 64 per prompt. Every image costs prompt tokens and
  prefill time. To reserve less memory for vision, cap them with `--limit-mm-per-prompt '{"image":4}'`.
- **Sampling.** The model's own default is T=1.0 (`generation_config.json`). The drafter's sampled drafts are tuned
  for that: at T=1 DSpark gives about 92 tok/s on prose and 116 on code (dev-server medians, v0.5.0).

## Opt-in flags (off by default)

Each of these trades a little fidelity, or a little of the defaults' tuning, for speed or memory. Leave them off for the
exact default behaviour.

| flag | what it does | speed | VRAM | quality vs default |
|---|---|---|---|---|
| `VLLM_INKLING2_FP8_LM_HEAD=1` | reads the output head (201K x 2304) as FP8 (e4m3, one scale per row) instead of BF16; the drafter shares it | −0.7 ms per DSpark step (fixed verify width 8: 28.9 → 28.15 ms short, 30.0 → 29.35 ms at 41K, repeat runs within 0.6 ms), about +2.5%; no drafter 72.2 → 74.1 tok/s | −0.86 GiB with DSpark (model memory 19.80 → 18.94 GiB), −0.43 GiB without the drafter | KL 0.00003-0.0001 nats/token; teacher-forced top-1 99.4-99.9% identical; perplexity change < 0.05% (code / prose / 32K context) |
| `SPEC_ARGS=` (empty) | turns the drafter off | 78 tok/s flat (slower on most text) | the drafter's weights and KV (not measured) | none (same model) |
| `VLLM_ADAPTIVE_VERIFICATION_ONLINE=0` | verify-width policy uses only the startup profile instead of also learning step costs while serving | at 55K context: 126 vs 143-189 tok/s with it on; elsewhere within noise | - | none |
| `VLLM_ADAPTIVE_VERIFICATION_STATS=N` | logs the chosen verify widths and the learned cost fit every N decode steps | - | - | none (logging) |

The FP8 head's quality numbers come from rounding the BF16 head to e4m3 in place, then comparing next-token log-probabilities
with the BF16 head on the same server stack (6K code, 6K prose, 32K context). For scale: running the same BF16 model
with only the batch composition changed gives KL 0.004 and 97-98% top-1, so the FP8 head is about 100x below the stack's
own numerical noise.

## Debug switches (restore an older code path)

For A/B tests and bug reports only. Each one disables a speed-up, and none of them changes the weights.

| flag | restores |
|---|---|
| `VLLM_INKLING2_DRAFT_HEAD_BF16=1` | BF16 drafter output head (default: FP8 copy, draft only) |
| `VLLM_INKLING2_DRAFT_LAYERS_BF16=1` | BF16 drafter layers (default: FP8 copies, draft only) |
| `VLLM_INKLING2_MARKOV_UNFUSED=1` | drafter's per-token recurrence as separate kernels |
| `VLLM_INKLING2_UNFUSED_SOFTCAP=1` | final logit softcap as three torch ops |
| `VLLM_INKLING2_SEPARATE_GEMV_ROWS=1` | one launch per attention projection on verify steps |
| `VLLM_INKLING2_ATTN_SPLIT_LEGACY=1` | the pre-v0.5.0 split-KV FP8 attention kernel |
| `VLLM_INKLING2_DIRECT_MAX_CTAS=N` | caps the BF16-activation NVFP4 expert path at N routed tiles (default 48, about 6 tokens); larger batches use the MXFP8 path, and 0 sends every batch there |
| `VLLM_INKLING2_MXFP8_DECODE_ACT=1` | MXFP8-quantized activations in NVFP4 expert decode (default: BF16 activations) |

The drafter switches change only which tokens are proposed, never what the model accepts. Output distributions
stay the same; only speed changes.
