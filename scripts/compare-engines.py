#!/usr/bin/env python3
"""Same OpenAI-API workload against one engine (redshift / Strata / llama.cpp), one JSON line per measurement.

usage: scripts/compare-engines.py LABEL PORT MODEL OUT.jsonl [--long FILE] [--skip-prefill]
Decode: release_check's 3 prompts, 512 tokens (ignore_eos where the server honours it), T=0 x2 and T=1.0 x3 seeds;
tok/s = (completion_tokens - 1) / (time after first token), so it excludes prefill. Prefill: fresh random filler
(no shared prefix) at 4K/16K/64K tokens, max_tokens 1, tok/s = prompt_tokens / TTFT.
"""
import json, sys, time, random, urllib.request
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from release_check import filler  # noqa: E402

label, port, model, out = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
long_file = sys.argv[sys.argv.index("--long") + 1] if "--long" in sys.argv else None
URL = f"http://127.0.0.1:{port}/v1/chat/completions"
PROMPTS = [("story", "Write a long story about a lighthouse keeper."),
           ("code", "Write a Python implementation of a red-black tree with comments."),
           ("explain", "Explain in detail how a CPU cache hierarchy works.")]


def post(body):
    req = urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=3600)


def stream(content, n, temp, seed):
    body = {"model": model, "messages": [{"role": "user", "content": content}], "max_tokens": n, "temperature": temp,
            "seed": seed, "ignore_eos": True, "stream": True, "stream_options": {"include_usage": True}}
    t0 = time.perf_counter(); first = None; usage = None; chunks = 0; timings = None
    with post(body) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            ev = json.loads(line[6:])
            usage = ev.get("usage") or usage
            timings = ev.get("timings") or timings
            ch = ev.get("choices") or []
            delta = (ch[0].get("delta") or {}) if ch else {}
            if any(isinstance(v, str) and v and k != "role" for k, v in delta.items()):
                chunks += 1
                if first is None:
                    first = time.perf_counter()
    end = time.perf_counter()
    ct = (usage or {}).get("completion_tokens") or chunks
    rec = {"tokens": ct, "prompt_tokens": (usage or {}).get("prompt_tokens"), "usage_reported": usage is not None}
    if timings:  # engine-side numbers (llama.cpp, Strata): decode excludes prefill
        rec["engine_decode_tok_s"] = round(timings["predicted_n"] / timings["predicted_ms"] * 1000, 1)
        rec["engine_prompt_ms"] = timings.get("prompt_ms")
    if first is not None:  # wall clock from the first streamed token
        rec["ttft_s"] = round(first - t0, 3)
        rec["decode_tok_s"] = round((ct - 1) / (end - first), 1)
    else:  # nothing streamed (Strata hides reasoning tokens): engine timings, plus whole-request wall clock
        rec["decode_tok_s"] = rec.get("engine_decode_tok_s")
        rec["wall_tok_s"] = round(ct / (end - t0), 1)
        rec["streamed"] = False
    return rec


def emit(rec):
    rec = {"engine": label, **rec}
    print(json.dumps(rec), flush=True)
    with open(out, "a") as f:
        f.write(json.dumps(rec) + "\n")


stream("Say hi.", 16, 0.0, 0)  # warm-up
for name, p in PROMPTS:
    for temp, seeds in ((0.0, (1, 2)), (1.0, (1, 2, 3))):
        for s in seeds:
            emit({"kind": "decode", "prompt": name, "temp": temp, "seed": s, **stream(p, 512, temp, s)})
if long_file:
    text = open(long_file).read()
    for temp, s in ((0.0, 1), (1.0, 1), (1.0, 2)):
        emit({"kind": "decode", "prompt": "long", "temp": temp, "seed": s, **stream(text, 512, temp, s)})
if "--skip-prefill" not in sys.argv:
    with post({"model": model, "messages": [{"role": "user", "content": filler(3000, 1) + "\n\nSay ok."}],
               "max_tokens": 1, "temperature": 0}) as r:
        json.load(r)  # warm-up of the long-prompt path
    for seed, (lab, words) in enumerate([("4k", 3000), ("4k", 3000), ("16k", 12000), ("64k", 48000)], start=11):
        t0 = time.perf_counter()
        with post({"model": model, "messages": [{"role": "user", "content": filler(words, seed) + "\n\nSay ok."}],
                   "max_tokens": 1, "temperature": 0}) as r:
            o = json.load(r)
        dt = time.perf_counter() - t0
        pt = o["usage"]["prompt_tokens"]
        emit({"kind": "prefill", "size": lab, "prompt_tokens": pt, "ttft_s": round(dt, 2), "prefill_tok_s": round(pt / dt)})
