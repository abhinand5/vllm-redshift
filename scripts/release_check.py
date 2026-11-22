"""Measure a running redshift server and report; it does not pass or fail anything (the release call is yours).

usage: python scripts/release_check.py --model-dir models/<model> --out results/<model>/<tag>.json
         [--port 8005] [--needle-words 140000] [--conc 2] [--quick]
Checks: short chat answer, tool call, streamed decode speed (3 prompts, 512 tokens), prefill TTFT at 4K/16K/64K
tokens, N concurrent long-context needles. Writes a JSON of numbers and prints a table next to the previous results.
"""
import argparse
import concurrent.futures as cf
import glob
import json
import os
import random
import time
import urllib.request

WORDS = "river stone lamp orchard ledger copper window meadow signal harbor".split()


class Client:
    def __init__(self, port, model):
        self.url, self.model = f"http://127.0.0.1:{port}/v1", model

    def _post(self, body):
        req = urllib.request.Request(self.url + "/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        return urllib.request.urlopen(req, timeout=3600)

    def chat(self, content, **kw):
        body = {"model": self.model, "messages": [{"role": "user", "content": content}], "temperature": 0, **kw}
        with self._post(body) as r:
            return json.load(r)

    def stream(self, content, n=512):
        body = {"model": self.model, "messages": [{"role": "user", "content": content}], "max_tokens": n,
                "temperature": 0, "ignore_eos": True, "stream": True, "stream_options": {"include_usage": True}}
        t0 = time.perf_counter(); first = None; usage = None
        with self._post(body) as r:
            for line in r:
                line = line.decode().strip()
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                ev = json.loads(line[6:])
                usage = ev.get("usage") or usage
                if first is None and ev.get("choices"):
                    first = time.perf_counter()
        ct = usage["completion_tokens"]
        return {"tokens": ct, "ttft_s": round(first - t0, 3),
                "decode_tok_s": round((ct - 1) / (time.perf_counter() - first), 1)}


def filler(n_words, seed):
    r = random.Random(seed)
    return " ".join(" ".join(r.choice(WORDS) for _ in range(12)) + "." for _ in range(n_words // 13))


def needle(cl, n_words, seed):
    r = random.Random(seed); code = r.randint(10000, 99999)
    sents = filler(n_words, seed).split(". ")
    sents.insert(int(len(sents) * 0.37), f"The secret vault code for project {seed} is {code}")
    p = ". ".join(sents) + f"\n\nWhat is the secret vault code for project {seed}? Reply with the number only."
    t0 = time.perf_counter()
    out = cl.chat(p, max_tokens=600)
    content = out["choices"][0]["message"].get("content") or ""
    return {"prompt_tokens": out["usage"]["prompt_tokens"], "found": str(code) in content,
            "seconds": round(time.perf_counter() - t0, 1)}


def prefill(cl, n_words, seed):
    # a fresh seed per call: prompts must not share a prefix, or prefix caching inflates the number
    t0 = time.perf_counter()
    out = cl.chat(filler(n_words, seed) + "\n\nSay ok.", max_tokens=1)
    dt = time.perf_counter() - t0
    pt = out["usage"]["prompt_tokens"]
    return {"prompt_tokens": pt, "ttft_s": round(dt, 2), "prefill_tok_s": round(pt / dt)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--port", type=int, default=8005)
    ap.add_argument("--served-name", default="Inkling-2-Mini")
    ap.add_argument("--needle-words", type=int, default=140000)
    ap.add_argument("--conc", type=int, default=2)
    ap.add_argument("--quick", action="store_true", help="skip 64K prefill and the long needles")
    ap.add_argument("--meta", default="{}", help="JSON merged into the result (image id, patch series, KV pool)")
    a = ap.parse_args()
    cl = Client(a.port, a.served_name)
    res = {"meta": json.loads(a.meta), "time": time.strftime("%Y-%m-%d %H:%M")}

    c = cl.chat("What is 17 * 23? Answer with the number only.", max_tokens=400)
    res["chat_391"] = "391" in (c["choices"][0]["message"].get("content") or "")
    tools = [{"type": "function", "function": {"name": "get_weather", "description": "Get the weather for a city",
              "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]
    t = cl.chat("What's the weather in Paris right now?", tools=tools, tool_choice="auto", max_tokens=600)
    tc = t["choices"][0]["message"].get("tool_calls") or []
    res["tool_call"] = bool(tc) and tc[0]["function"]["name"] == "get_weather"
    for name, p in [("story", "Write a long story about a lighthouse keeper."),
                    ("code", "Write a Python implementation of a red-black tree with comments."),
                    ("explain", "Explain in detail how a CPU cache hierarchy works.")]:
        res[f"decode_{name}"] = cl.stream(p)
        print(f"decode {name}: {res[f'decode_{name}']}", flush=True)
    prefill(cl, 3000, 1)  # warm-up: first long request after start pays one-time costs
    sizes = [("4k", 3000), ("16k", 12000)] + ([] if a.quick else [("64k", 48000)])
    for seed, (label, words) in enumerate(sizes, start=11):
        res[f"prefill_{label}"] = prefill(cl, words, seed)
        print(f"prefill {label}: {res[f'prefill_{label}']}", flush=True)
    if not a.quick:
        t0 = time.perf_counter()
        with cf.ThreadPoolExecutor(a.conc) as ex:
            res["needles"] = list(ex.map(lambda s: needle(cl, a.needle_words, s), range(101, 101 + a.conc)))
        res["needles_wall_s"] = round(time.perf_counter() - t0, 1)
        print(f"needles: {res['needles']}", flush=True)

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    # previous run = latest by the time recorded in each file (file names start with the version, which does not sort)
    prev = sorted((f for f in glob.glob(os.path.join(os.path.dirname(a.out), "*.json")) if f != a.out),
                  key=lambda f: json.load(open(f)).get("time", ""))
    json.dump(res, open(a.out, "w"), indent=1)
    old = json.load(open(prev[-1])) if prev else {}
    print(f"\n| metric | previous ({os.path.basename(prev[-1]) if prev else '-'}) | this run |\n|---|---|---|")
    rows = [("chat 17*23 = 391", "chat_391", None), ("tool call", "tool_call", None)]
    rows += [(f"decode {n} tok/s", f"decode_{n}", "decode_tok_s") for n in ("story", "code", "explain")]
    rows += [(f"prefill {n} tok/s", f"prefill_{n}", "prefill_tok_s") for n in ("4k", "16k", "64k")]
    for label, k, sub in rows:
        get = lambda d: (d.get(k) or {}).get(sub, "-") if sub else d.get(k, "-")
        print(f"| {label} | {get(old)} | {get(res)} |")
    if "needles" in res:
        fmt = lambda d: ", ".join(f"{n['prompt_tokens']} tok {'found' if n['found'] else 'MISSED'}"
                                  for n in d.get("needles", [])) or "-"
        print(f"| needles ({a.conc} concurrent) | {fmt(old)} | {fmt(res)} ({res['needles_wall_s']} s) |")


if __name__ == "__main__":
    main()
