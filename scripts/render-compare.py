#!/usr/bin/env python3
"""Render the README engine comparison (light + dark SVG) from a compare JSON in results/.

usage: scripts/render-compare.py results/inkling2-mini/compare-v0.4.0-20261122.json assets/inkling2-mini-compare
writes <prefix>-light.svg and <prefix>-dark.svg. Series hues validated per mode (dataviz validate_palette.js,
--pairs all): light #e34948 #2a78d6 #eda100, dark #d44545 #3987e5 #bf8800; every bar also carries its value and
the legend names each engine, so identity never rests on colour alone.
Decode bars are the mean over runs at T=0 (greedy); Strata's decode rate is its engine timing (its stream hides
reasoning tokens). Prefill bars are the best run per size.
"""
import json
import statistics as st
import sys
from html import escape

src, prefix = sys.argv[1], sys.argv[2]
data = json.load(open(src))
rows = data["rows"]

ENGINES = [  # fixed order = fixed colour slot
    ("redshift", "vllm-redshift v0.4.0", "official weights, as shipped"),
    ("strata", "Strata 0.2.19", "SQ-20 requantized pack"),
    ("llamacpp", "llama.cpp", "Unsloth UD-Q8_K_XL GGUF"),
]
THEMES = {
    "light": {"bg0": "#ffffff", "bg1": "#faf9f7", "glow": "#e34948", "glow_a": 0.10,
              "border": "#e7e5e0", "grid": "#efede9", "baseline": "#cfcdc6", "card": "#ffffff",
              "ink": "#0b0b0b", "ink2": "#4a4945", "ink3": "#7a7972",
              "series": ["#e34948", "#2a78d6", "#eda100"], "hero": ("#f0605a", "#d9363a"),
              "accent": "#c8292c", "pill_bg": "#e34948", "pill_a": 0.10, "track": "#f3f1ed"},
    "dark": {"bg0": "#161615", "bg1": "#1c1c1b", "glow": "#e04848", "glow_a": 0.22,
             "border": "#2f2f2d", "grid": "#262624", "baseline": "#45453f", "card": "#1f1f1e",
             "ink": "#f4f3f0", "ink2": "#c9c8bf", "ink3": "#8f8e86",
             "series": ["#d44545", "#3987e5", "#bf8800"], "hero": ("#f06a64", "#d44545"),
             "accent": "#ff7b73", "pill_bg": "#e34948", "pill_a": 0.16, "track": "#242423"},
}
FONT = "-apple-system, BlinkMacSystemFont, 'Segoe UI Variable Display', 'Segoe UI', Inter, Helvetica, Arial, sans-serif"
NUM = ' style="font-variant-numeric:tabular-nums"'


def decode(engine, prompt):
    v = [r["engine_decode_tok_s"] if engine == "strata" else r["decode_tok_s"] for r in rows
         if r["engine"] == engine and r["kind"] == "decode" and r["prompt"] == prompt and r["temp"] == 0.0]
    return st.mean(v)


def prefill(engine, size):
    return max(r["prefill_tok_s"] for r in rows if r["engine"] == engine and r["kind"] == "prefill" and r["size"] == size)


DECODE = [("Story", "story"), ("Code", "code"), ("Explain", "explain"), ("41K-token code prompt", "long")]
PREFILL = [("4K-token prompt", "4k"), ("16K-token prompt", "16k"), ("64K-token prompt", "64k")]
dec = {p: [decode(e, p) for e, *_ in ENGINES] for _, p in DECODE}
pre = {s: [prefill(e, s) for e, *_ in ENGINES] for _, s in PREFILL}
assert all(v[0] > max(v[1:]) for v in [*dec.values(), *pre.values()]), "headline claims redshift leads every test"


def ratios(table, i):
    return [v[0] / v[i] for v in table.values()]


W, PAD = 1200, 48
HERO_H, RIVAL_H, GAP = 22, 11, 4  # redshift is the thick bar; rivals are slim
LABEL_ROW, GROUP_GAP = 30, 26


def char_w(s, size, bold=False):  # rough advance width of a system sans; only sizes pills, with slack
    w = sum(0.62 if c.isdigit() or c in "×." else 0.30 if c in " ·" else 0.56 for c in s)
    return w * size * (1.06 if bold else 1.0)


def svg(mode):
    t = THEMES[mode]
    out, defs = [], []
    add = out.append

    def text(x, y, s, size=13, fill=None, weight=400, anchor="start", extra=""):
        add(f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" font-weight="{weight}" fill="{fill or t["ink"]}" '
            f'text-anchor="{anchor}"{extra}>{escape(s)}</text>')

    def bar(x, y, w, h, fill, r=None):  # square at the baseline, rounded data end
        r = min(r if r is not None else h / 2, w / 2, h / 2)
        add(f'<path d="M{x:.1f},{y:.1f} H{x + w - r:.1f} A{r:.1f},{r:.1f} 0 0 1 {x + w:.1f},{y + r:.1f} '
            f'V{y + h - r:.1f} A{r:.1f},{r:.1f} 0 0 1 {x + w - r:.1f},{y + h:.1f} H{x:.1f} Z" fill="{fill}"/>')

    defs.append(f'<linearGradient id="bg" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="{t["bg0"]}"/>'
                f'<stop offset="1" stop-color="{t["bg1"]}"/></linearGradient>')
    defs.append(f'<radialGradient id="glow" cx="0.92" cy="0" r="0.75"><stop offset="0" stop-color="{t["glow"]}" '
                f'stop-opacity="{t["glow_a"]}"/><stop offset="1" stop-color="{t["glow"]}" stop-opacity="0"/></radialGradient>')
    defs.append(f'<linearGradient id="hero" x1="0" y1="0" x2="1" y2="0"><stop offset="0" stop-color="{t["hero"][1]}"/>'
                f'<stop offset="1" stop-color="{t["hero"][0]}"/></linearGradient>')
    defs.append(f'<linearGradient id="num" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="{t["hero"][0]}"/>'
                f'<stop offset="1" stop-color="{t["hero"][1]}"/></linearGradient>')

    # header
    y = PAD + 14
    text(PAD, y, "VLLM-REDSHIFT  ·  INKLING-2-MINI  ·  ONE RADEON AI PRO R9700", 12, t["accent"], 700,
         extra=' letter-spacing="1.6"')
    y += 40
    text(PAD, y, "Faster than Strata and llama.cpp on every test", 32, weight=750, extra=' letter-spacing="-0.6"')
    y += 28
    text(PAD, y, "Same requests through each engine's OpenAI API on the same GPU. Only vllm-redshift serves the "
         "official weights, unmodified.", 15, t["ink2"])

    # hero stats: the moat in four numbers
    y += 30
    rd, rl, pd, pl = ratios(dec, 1), ratios(dec, 2), ratios(pre, 1), ratios(pre, 2)
    tiles = [(f"{max(rd):.1f}×", "faster decode than Strata", f"41K-token prompt · {min(rd):.1f}× on short ones"),
             (f"{min(rl):.1f}×", "faster decode than llama.cpp", f"at least, on every prompt · up to {max(rl):.1f}×"),
             (f"{min(pd):.1f}×", "faster prefill than Strata", "4K to 64K-token prompts"),
             (f"{max(pl):.1f}×", "faster prefill than llama.cpp", f"64K-token prompt · {min(pl):.1f}× at 4K")]
    gap = 16
    tw = (W - 2 * PAD - 3 * gap) / 4
    th = 112
    for i, (big, line, sub) in enumerate(tiles):
        x = PAD + i * (tw + gap)
        add(f'<rect x="{x:.1f}" y="{y:.1f}" width="{tw:.1f}" height="{th}" rx="14" fill="{t["card"]}" '
            f'stroke="{t["border"]}"/>')
        add(f'<rect x="{x + 20:.1f}" y="{y:.1f}" width="28" height="3" rx="1.5" fill="url(#hero)"/>')
        text(x + 20, y + 52, big, 40, "url(#num)", 760, extra=NUM + ' letter-spacing="-1"')
        text(x + 20, y + 78, line, 14, weight=600)
        text(x + 20, y + 97, sub, 12, t["ink3"])
    y += th + 40

    # legend, fixed columns aligned with the tiles
    for i, (_, name, sub) in enumerate(ENGINES):
        lx = PAD + i * (tw + gap)
        h = 14 if i == 0 else 8
        bar(lx, y - 7 - h / 2, 26, h, "url(#hero)" if i == 0 else t["series"][i], r=h / 2)
        text(lx + 36, y - 2, name, 14, weight=650)
        text(lx + 36, y + 16, sub, 12, t["ink3"])
    y += 52

    # two panels, each its own scale (never a dual axis)
    pgap = 56
    pw = (W - 2 * PAD - pgap) / 2
    top = y

    def panel(x0, top, title, unit, groups, table, vmax, ticks, fmt, lower_better=False, word="×"):
        add(f'<text x="{x0:.1f}" y="{top:.1f}" font-size="18" font-weight="700" fill="{t["ink"]}">{escape(title)}'
            f'<tspan dx="10" font-size="13" font-weight="400" fill="{t["ink3"]}">{escape(unit)}</tspan></text>')
        value_w = 64
        plot_w = pw - value_w
        n = len(ENGINES)
        group_h = LABEL_ROW + HERO_H + (n - 1) * (RIVAL_H + GAP) + GAP
        gy = top + 22
        plot_bot = gy + len(groups) * (group_h + GROUP_GAP) - GROUP_GAP
        for tv in ticks:
            gx = x0 + plot_w * tv / vmax
            add(f'<line x1="{gx:.1f}" y1="{gy + LABEL_ROW - 4:.1f}" x2="{gx:.1f}" y2="{plot_bot + 2:.1f}" '
                f'stroke="{t["baseline"] if tv == 0 else t["grid"]}" stroke-width="1"/>')
            text(gx, plot_bot + 22, fmt(tv), 11.5, t["ink3"], anchor="middle", extra=NUM)
        for label, key in groups:
            vals = table[key]
            text(x0, gy + 19, label, 14, t["ink2"], 600)
            # lead badges, right-aligned on the label row: redshift's multiple over each rival
            bx = x0 + pw
            for i in (2, 1):
                lead = vals[i] / vals[0] if lower_better else vals[0] / vals[i]
                s = f"{lead:.1f}{word} {ENGINES[i][1].split()[0]}"
                w = char_w(s, 12, True) + 22
                bx -= w
                add(f'<rect x="{bx:.1f}" y="{gy + 4:.1f}" width="{w:.1f}" height="22" rx="11" '
                    f'fill="{t["pill_bg"]}" fill-opacity="{t["pill_a"]}"/>')
                text(bx + w / 2, gy + 19.5, s, 12, t["accent"], 700, anchor="middle", extra=NUM)
                bx -= 8
            by = gy + LABEL_ROW
            for i, v in enumerate(vals):
                h = HERO_H if i == 0 else RIVAL_H
                w = max(3.0, plot_w * v / vmax)
                bar(x0, by, w, h, "url(#hero)" if i == 0 else t["series"][i], r=6 if i == 0 else h / 2)
                add(f'<title>{escape(ENGINES[i][1])}, {escape(label)}: {fmt(v)} tok/s</title>')
                if i == 0:
                    text(x0 + w + 10, by + h / 2 + 6, fmt(v), 17, t["ink"], 750, extra=NUM)
                else:
                    text(x0 + w + 8, by + h / 2 + 4, fmt(v), 11.5, t["ink3"], 500, extra=NUM)
                by += h + GAP
            gy += group_h + GROUP_GAP
        return plot_bot + 22

    b1 = panel(PAD, top, "Decode", "tokens / s, mean of runs", DECODE, dec, 120, [0, 30, 60, 90, 120],
               lambda v: f"{v:.0f}" if v == int(v) else f"{v:.1f}")
    b2 = panel(PAD + pw + pgap, top, "Prefill", "tokens / s, cold: fresh prompt, nothing cached", PREFILL, pre, 5000,
               [0, 1000, 2000, 3000, 4000, 5000], lambda v: f"{v:,.0f}")
    y = max(b1, b2) + 40

    add(f'<line x1="{PAD}" y1="{y - 18:.1f}" x2="{W - PAD}" y2="{y - 18:.1f}" stroke="{t["border"]}"/>')
    notes = ["512 output tokens, greedy. DSpark speculative decoding on for vllm-redshift and Strata; llama.cpp shown without a drafter "
             "(with DSpark it was slower: 21–38 tok/s).",
             "FP8 KV cache, 196K context, 2 slots for vllm-redshift and Strata. Strata's AMD engine loads only its own requantized "
             ".strata packs.",
             f"Measured {data['meta']['date']} with scripts/compare-engines.py · raw runs in results/inkling2-mini/"]
    for n_ in notes:
        text(PAD, y, n_, 12, t["ink3"])
        y += 18
    H = y + PAD - 18
    head = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H:.0f}" viewBox="0 0 {W} {H:.0f}" '
            f'font-family="{FONT}" role="img" aria-label="Inkling-2-Mini on one Radeon AI PRO R9700: vllm-redshift is faster '
            f'than Strata and llama.cpp at decode and prefill on every test">'
            f'<defs>{"".join(defs)}</defs>'
            f'<rect x="0.5" y="0.5" width="{W - 1}" height="{H - 1:.0f}" rx="20" fill="url(#bg)" stroke="{t["border"]}"/>'
            f'<rect x="0.5" y="0.5" width="{W - 1}" height="{H - 1:.0f}" rx="20" fill="url(#glow)"/>')
    return head + "".join(out) + "</svg>\n"


for mode in THEMES:
    open(f"{prefix}-{mode}.svg", "w").write(svg(mode))
    print(f"{prefix}-{mode}.svg")
