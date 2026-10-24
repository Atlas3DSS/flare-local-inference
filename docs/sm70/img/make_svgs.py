"""Generate SM70 infographics as self-contained SVGs, light and dark.

hero, prefill, long-decode and kv-pool take their numbers from a JSON file filled from docs/sm70/receipts/g128/
(docs/sm70/img/numbers.json); comparison uses the values tabulated in docs/sm70/WHY-NOT-LOW-BIT.md, with the receipts
named next to them. latency and e2e were drawn by an earlier version of this generator and are not regenerated here.
usage: python3 docs/sm70/img/make_svgs.py docs/sm70/img docs/sm70/img/numbers.json [graphic ...]"""
import sys
from pathlib import Path
from xml.sax.saxutils import escape

import json
OUT = Path(sys.argv[1])
N = json.loads(Path(sys.argv[2]).read_text())
W = 1200
PAD = 56
SANS = "system-ui, -apple-system, 'Segoe UI', Roboto, 'Helvetica Neue', 'Noto Sans', Arial, sans-serif"
MONO = "ui-monospace, SFMono-Regular, Menlo, Consolas, 'Liberation Mono', 'DejaVu Sans Mono', monospace"

# Validated tokens (see README.md in the output folder for the validator run).
MODES = {
    "light": dict(surface="#f3f5f4", tile="#ffffff", border="rgba(37,41,43,0.14)",
                  primary="#25292b", secondary="#596166", grid="#dde2e0", axis="#b3babd",
                  accent="#5f9524", context="#858d91"),
    "dark": dict(surface="#222629", tile="#2a2f33", border="rgba(255,255,255,0.10)",
                 primary="#f3f5f4", secondary="#b0b8bc", grid="#353b3f", axis="#50585d",
                 accent="#6c9f27", context="#737b80"),
}

HW = "Tesla V100 PCIe 32 GB"


def t(x, y, s, size, fill, weight=400, anchor="start", family=SANS, ls=None, extra=""):
    lsa = f' letter-spacing="{ls}"' if ls else ""
    return (f'<text x="{x}" y="{y}" font-family="{family}" font-size="{size}" font-weight="{weight}" '
            f'fill="{fill}" text-anchor="{anchor}"{lsa}{extra}>{s}</text>')


def mono(s, fill=None):
    f = f' fill="{fill}"' if fill else ""
    return f'<tspan font-family="{MONO}" font-size="0.92em"{f}>{escape(s)}</tspan>'


def chip_w(label):
    # generous estimate (bold caps ~0.72em at 12.5px + tracking); text is centred
    return round(len(label) * 9.6 + 30)


def header(c, chip, kicker, title, subtitle):
    cw = chip_w(chip)
    out = [
        f'<rect x="{PAD}" y="40" width="{cw}" height="28" rx="14" fill="none" stroke="{c["secondary"]}" stroke-width="1"/>',
        t(PAD + cw / 2, 59, escape(chip), 12.5, c["primary"], 700, "middle", ls="0.08em"),
        t(PAD + cw + 14, 60, kicker, 15, c["secondary"]),
        t(PAD, 120, escape(title), 36, c["primary"], 650),
        t(PAD, 156, subtitle, 17, c["secondary"]),
    ]
    return "\n".join(out)


def footer(c, h, meta, caption):
    y0 = h - 92
    return "\n".join([
        f'<line x1="{PAD}" y1="{y0}" x2="{W - PAD}" y2="{y0}" stroke="{c["grid"]}" stroke-width="1"/>',
        t(PAD, y0 + 30, meta, 14, c["primary"]),
        t(PAD, y0 + 56, caption, 13, c["secondary"]),
    ])


def frame(c, h, title, desc, body):
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{h}" viewBox="0 0 {W} {h}" role="img" aria-labelledby="title desc">
<title id="title">{escape(title)}</title>
<desc id="desc">{escape(desc)}</desc>
<rect x="0.5" y="0.5" width="{W - 1}" height="{h - 1}" rx="20" fill="{c["surface"]}" stroke="{c["border"]}" stroke-width="1"/>
{body}
</svg>
'''


def hbar(x0, y, w, h, fill, tip):
    """Horizontal bar: square at the baseline, 4px rounded data-end."""
    r = 4
    w = max(w, r)
    d = (f"M{x0},{y} H{x0 + w - r} Q{x0 + w},{y} {x0 + w},{y + r} V{y + h - r} "
         f"Q{x0 + w},{y + h} {x0 + w - r},{y + h} H{x0} Z")
    return f'<path d="{d}" fill="{fill}"><title>{escape(tip)}</title></path>'


def tile(c, x, y, w, h, stripe, label, value, unit, note, value_fill=None, value_size=84, tip=""):
    vf = value_fill or c["primary"]
    return "\n".join([
        f'<g><title>{escape(tip)}</title>',
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="14" fill="{c["tile"]}" stroke="{c["border"]}" stroke-width="1"/>',
        f'<rect x="{x + 28}" y="{y + 26}" width="36" height="4" rx="2" fill="{stripe}"/>',
        t(x + 28, y + 62, escape(label), 18, c["secondary"]),
        t(x + 28, y + 152,
          f'{escape(value)}<tspan font-size="26" font-weight="400" fill="{c["secondary"]}" dx="12">{escape(unit)}</tspan>',
          value_size, vf, 650),
        t(x + 28, y + h - 22, note, 14, c["secondary"]) if note else "",
        '</g>',
    ])


REL = "package f21b9abb · server binary 48b69ed4"
PREV = "previous package fd7c24dc, release binary d4ffb1fa"


def dot(x, y, r, fill, ring, tip, square=False):
    """Marker >= 8px with a 2px surface ring; square for the second series."""
    if square:
        return (f'<rect x="{x - r}" y="{y - r}" width="{2 * r}" height="{2 * r}" rx="1.5" fill="{fill}" '
                f'stroke="{ring}" stroke-width="2"><title>{escape(tip)}</title></rect>')
    return (f'<circle cx="{x}" cy="{y}" r="{r}" fill="{fill}" stroke="{ring}" stroke-width="2">'
            f'<title>{escape(tip)}</title></circle>')



def f1(v):
    return f"{v:.1f}"


# ---------------------------------------------------------------- hero
def hero(c):
    h = 588
    c1, c4 = N["short"]["c1_agent"], N["short"]["c4_agent"]
    body = [header(c, "DECODE-ONLY", f"Flare on SM70 · Qwen3.8-27B NVFP4 · MTP draft depth 4 · {REL}",
                   f"{f1(c1)} tok/s on one V100, {f1(c4)} per agent with four",
                   "Median per-request decode rate on the public short prompts. Excludes prefill and queueing.")]
    tw = (W - 2 * PAD - 24) / 2
    body.append(tile(c, PAD, 192, tw, 196, c["accent"], "Median decode, 1 agent", f1(c1), "tok/s", "",
                     tip=f"Package f21b9abb, median decode at 1 agent: {f1(c1)} tok/s"))
    body.append(tile(c, PAD + tw + 24, 192, tw, 196, c["accent"], "Median decode per agent, 4 concurrent agents",
                     f1(c4), "tok/s per agent", "",
                     tip=f"Package f21b9abb, median decode per agent at 4 agents: {f1(c4)} tok/s"))
    body.append(t(PAD, 422, f"Previous package {mono('fd7c24dc')} on the release binary {mono('d4ffb1fa')}, 2026-10-23: "
                  "83.0 tok/s at 1 agent and 64.8 per agent at 4.", 15, c["secondary"]))
    body.append(t(PAD, 446, "Best recorded: 90.4 tok/s at 1 agent and 70.7 per agent at 4, on 2026-10-22 with the older "
                  f"binary {mono('2c792073')}", 15, c["secondary"]))
    body.append(t(PAD, 468, "and the internal 1.1k–1.7k-token prompts: the best of 11 runs (median 87.6 at 1 agent).",
                  15, c["secondary"]))
    meta = (f"{HW} · TP1 · {mono('--memory-limit 31G')} · public prompts of 578–582 tokens, 128 output tokens · "
            f"{N['short']['requests']}/24 requests ok · 2026-10-24")
    cap = ("Source: BENCHMARKS-SM70.md (package f21b9abb; previous package; measurement history) · receipts "
           "docs/sm70/receipts/g128/")
    body.append(footer(c, h, meta, cap))
    return frame(c, h, "Flare SM70 decode-only headline",
                 f"Decode-only, Qwen3.8-27B NVFP4 on one Tesla V100 PCIe 32 GB, package f21b9abb on server binary "
                 f"48b69ed4, memory limit 31G, public prompts of 578 to 582 tokens, 2026-10-24: median decode {f1(c1)} "
                 f"tok/s at 1 agent and {f1(c4)} tok/s per agent at 4 agents. Previous package fd7c24dc on release "
                 "binary d4ffb1fa, 2026-10-23: 83.0 and 64.8. Best recorded: 90.4 tok/s at 1 agent and 70.7 per agent "
                 "at 4 on 2026-10-22 with the older binary 2c792073 and internal prompts. Source: BENCHMARKS-SM70.md.",
                 "\n".join(body))


# ---------------------------------------------------------------- prefill
PREV_RATE = {2048: 807, 8192: 785, 22528: 741, 30720: 719}


def prefill(c):
    h = 990
    P = [(r["tokens"], (r["c1_ttft"], r["c1_rate"]), (r["c4_ttft"], r["c4_rate"])) for r in N["prefill"]]
    rates = [one[1] for _, one, _ in P]
    g_lo = min(100 * (one[1] / PREV_RATE[L] - 1) for L, one, _ in P)
    g_hi = max(100 * (one[1] / PREV_RATE[L] - 1) for L, one, _ in P)
    body = [header(c, "END-TO-END · PREFILL", f"Time to first token · uncached token-ID prompts · {REL}",
                   f"Prefill runs at {min(rates):,}–{max(rates):,} tok/s, one request at a time",
                   f"{g_lo:.0f}–{g_hi:.0f} % faster than the previous package's 128-token chunks, with the same tokens."
                   " Four concurrent requests still share it.")]
    x0, x1, xmax = 350, 1000, 32_000
    top, axis_y, ymax = 214, 494, 180
    sx = lambda v: x0 + (x1 - x0) * v / xmax
    sy = lambda v: axis_y - (axis_y - top) * v / ymax
    for v in range(0, ymax + 1, 30):
        y = sy(v)
        if v:
            body.append(f'<line x1="{x0}" y1="{y}" x2="{x1}" y2="{y}" stroke="{c["grid"]}" stroke-width="1"/>')
        body.append(t(x0 - 12, y + 4, str(v), 13, c["secondary"], anchor="end",
                      extra=' style="font-variant-numeric: tabular-nums"'))
    body.append(t(x0 - 12, top - 18, "TTFT, s", 13, c["secondary"], anchor="end"))
    body.append(f'<line x1="{x0}" y1="{axis_y}" x2="{x1}" y2="{axis_y}" stroke="{c["axis"]}" stroke-width="1"/>')
    for v in range(0, xmax + 1, 8000):
        body.append(t(sx(v), axis_y + 22, "0" if v == 0 else f"{v // 1000}K", 13, c["secondary"], anchor="middle",
                      extra=' style="font-variant-numeric: tabular-nums"'))
    body.append(t(x1, axis_y + 44, "prompt tokens", 13, c["secondary"], anchor="end"))
    series = [("4 concurrent requests", 2, c["accent"], False), ("1 request", 1, c["context"], True)]
    for name, i, fill, square in series:
        pts = [(sx(r[0]), sy(r[i][0])) for r in P]
        d = "M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in pts)
        body.append(f'<path d="{d}" fill="none" stroke="{fill}" stroke-width="2"/>')
    for name, i, fill, square in series:
        for L, one, four in P:
            ttft, rate = (one, four)[i - 1]
            body.append(dot(sx(L), sy(ttft), 6, fill, c["surface"],
                            f"{name}, {L:,} prompt tokens: TTFT p50 {ttft:.2f} s, effective prefill {rate:,} tok/s",
                            square=square))
    L, one, four = P[-1]
    body.append(t(sx(L) + 14, sy(four[0]) - 4, "4 concurrent", 15, c["primary"], 650))
    body.append(t(sx(L) + 14, sy(four[0]) + 15, f"{four[0]:.1f} s", 15, c["primary"]))
    body.append(t(sx(L) + 14, sy(one[0]) - 4, "1 request", 15, c["primary"], 650))
    body.append(t(sx(L) + 14, sy(one[0]) + 15, f"{one[0]:.1f} s", 15, c["primary"]))
    lx = x0
    for name, i, fill, square in series:
        body.append(dot(lx + 6, 196, 6, fill, c["surface"], name, square=square))
        body.append(t(lx + 20, 201, name, 14, c["primary"]))
        lx += 230
    ry = axis_y + 84
    body.append(f'<line x1="{PAD}" y1="{ry - 22}" x2="{W - PAD}" y2="{ry - 22}" stroke="{c["grid"]}" stroke-width="1"/>')
    body.append(t(PAD, ry, "Prompt tokens", 13, c["secondary"], 600))
    body.append(t(PAD, ry + 24, "Effective tok/s, 1 request", 14, c["secondary"]))
    body.append(t(PAD, ry + 48, "Effective tok/s, 4 concurrent (aggregate)", 14, c["secondary"]))
    body.append(t(PAD, ry + 72, "Previous package, 1 request", 14, c["secondary"]))
    num = ' style="font-variant-numeric: tabular-nums"'
    for L, one, four in P:
        x = sx(L)
        body.append(t(x, ry, f"{L:,}", 13, c["secondary"], 600, "middle", extra=num))
        body.append(t(x, ry + 24, f"{one[1]:,}", 15, c["primary"], 600, "middle", extra=num))
        body.append(t(x, ry + 48, f"{four[1]:,}", 15, c["primary"], 600, "middle", extra=num))
        body.append(t(x, ry + 72, f"{PREV_RATE[L]:,}", 15, c["secondary"], 400, "middle", extra=num))
    s = N["srpt"]
    py = ry + 120
    body.append(f'<line x1="{PAD}" y1="{py - 26}" x2="{W - PAD}" y2="{py - 26}" stroke="{c["grid"]}" stroke-width="1"/>')
    body.append(t(PAD, py, "Four concurrent 30,720-token prompts: first token per request", 15, c["primary"], 650))
    body.append(t(PAD, py + 20, f"Opt-in {mono('--prefill-srpt')} (off by default), same package and protocol; "
                  "mean of 3 measured batches by order of arrival", 13, c["secondary"]))
    gx0, gx1, gmax = x0, x1, 180
    gs = lambda v: gx0 + (gx1 - gx0) * v / gmax
    gtop, gaxis = py + 44, py + 150
    for v in range(0, gmax + 1, 30):
        x = gs(v)
        if v:
            body.append(f'<line x1="{x}" y1="{gtop}" x2="{x}" y2="{gaxis}" stroke="{c["grid"]}" stroke-width="1"/>')
        body.append(t(x, gaxis + 20, str(v), 13, c["secondary"], anchor="middle", extra=num))
    body.append(t(gx1, gaxis + 40, "seconds to first token", 13, c["secondary"], anchor="end"))
    body.append(f'<line x1="{gx0}" y1="{gtop}" x2="{gx0}" y2="{gaxis}" stroke="{c["axis"]}" stroke-width="1"/>')
    ya, yb = gtop + 26, gtop + 78
    lo, hi = s["default_c4_30720_range"]
    body.append(t(gx0 - 24, ya + 5, "Round-robin (default)", 15, c["primary"], 600, "end"))
    body.append(f'<rect x="{gs(lo)}" y="{ya - 6}" width="{max(gs(hi) - gs(lo), 4)}" height="12" rx="3" '
                f'fill="{c["context"]}"><title>Round-robin: all four first tokens at {lo:.2f}-{hi:.2f} s</title></rect>')
    body.append(t(gs(lo) - 10, ya + 5, f"all four at {lo:.1f}–{hi:.1f} s", 14, c["primary"], anchor="end"))
    body.append(t(gx0 - 24, yb + 5, "Opt-in prefill ordering", 15, c["primary"], 600, "end"))
    for v in s["srpt_c4_30720_ranks"]:
        body.append(dot(gs(v), yb, 6, c["accent"], c["surface"], f"--prefill-srpt: first token at {v:.2f} s"))
        body.append(t(gs(v), yb - 14, f"{v:.1f}", 14, c["primary"], 600, "middle"))
    meta = (f"{HW} · {mono('--memory-limit 31G')} · 1 warmup + 3 measured batches per point · 32 output tokens · "
            "2026-10-24")
    cap = ("Source: BENCHMARKS-SM70.md (package f21b9abb; previous package) · receipts docs/sm70/receipts/g128/ · "
           "TTFT: POST to the first SSE chunk with output")
    body.append(footer(c, h, meta, cap))
    rows = "; ".join(f"{L:,} tokens: {one[0]:.2f} s and {one[1]:,} tok/s alone (previous package {PREV_RATE[L]:,}), "
                     f"{four[0]:.2f} s and {four[1]:,} tok/s aggregate with four" for L, one, four in P)
    ranks = ", ".join(f"{v:.2f}" for v in s["srpt_c4_30720_ranks"])
    return frame(c, h, "Flare SM70 prefill and time to first token",
                 "Time to first token against prompt length on one Tesla V100 PCIe 32 GB, package f21b9abb with "
                 f"512-token prefill chunks on server binary 48b69ed4, 2026-10-24, uncached prompts: {rows}. "
                 "Concurrent prefills do not overlap. With --prefill-srpt (off by default), four concurrent "
                 f"30,720-token prompts get their first tokens at {ranks} s instead of all at {lo:.2f} to {hi:.2f} s. "
                 "Source: BENCHMARKS-SM70.md.", "\n".join(body))


# ---------------------------------------------------------------- long-context decode
def long_decode(c):
    h = 560
    g = N["long"]
    C1, C4 = g["c1_cold"], g["c4_first"]
    m1, m4 = g["c1_cold_median"], g["c4_first_median"]
    body = [header(c, "DECODE-ONLY", f"Four public prompts of 28.7k tokens · 512 output tokens · {REL}",
                   f"At 28.7k tokens: {f1(m1)} tok/s alone, {f1(m4)} per agent with four",
                   "Decode rate per request after its first token. Bars: medians; dots: single requests. With four,"
                   " some decode while others are still in prefill.")]
    x0, x1, dmax = 430, 1080, 90
    sx = lambda v: x0 + (x1 - x0) * v / dmax
    top, axis_y = 200, 400
    for v in range(0, dmax + 1, 10):
        x = sx(v)
        if v:
            body.append(f'<line x1="{x}" y1="{top}" x2="{x}" y2="{axis_y}" stroke="{c["grid"]}" stroke-width="1"/>')
        body.append(t(x, axis_y + 24, str(v), 13, c["secondary"], anchor="middle",
                      extra=' style="font-variant-numeric: tabular-nums"'))
    body.append(t(x1, axis_y + 46, "tok/s per request", 13, c["secondary"], anchor="end"))
    body.append(f'<line x1="{x0}" y1="{top}" x2="{x0}" y2="{axis_y}" stroke="{c["axis"]}" stroke-width="1"/>')
    bh = 24
    rows = [(252, "1 agent", f"{g['c1_label']} ({len(C1)})", C1, m1, c["context"], True),
            (348, "4 agents", f"first batch, per agent ({len(C4)})", C4, m4, c["accent"], False)]
    for yc, name, sub, values, median, fill, square in rows:
        body.append(t(x0 - 24, yc - 3, name, 17, c["primary"], 600, "end"))
        body.append(t(x0 - 24, yc + 17, escape(sub), 13, c["secondary"], anchor="end"))
        body.append(hbar(x0, yc - bh / 2, sx(median) - x0, bh, fill, f"{name}: median {median:.1f} tok/s"))
        for v in values:
            body.append(dot(sx(v), yc - bh / 2 - 12, 5, fill, c["surface"], f"{name}: one request at {v:.1f} tok/s",
                            square=square))
        body.append(t(sx(max(values)) + 14, yc + 6, f"median {median:.1f}", 17, c["primary"], 650))
        body.append(t(sx(max(values)) + 14, yc + 26, f"range {min(values):.1f}–{max(values):.1f}", 13, c["secondary"]))
    meta = (f"{HW} · {mono('--memory-limit 31G')} · 124,928-token KV pool · {g['meta_resident']} · 2026-10-24")
    cap = ("Source: BENCHMARKS-SM70.md (package f21b9abb long-context table; previous package: 72.4 and 41.9) · "
           "receipts docs/sm70/receipts/g128/")
    body.append(footer(c, h, meta, cap))
    return frame(c, h, "Flare SM70 long-context decode",
                 "Decode-only at 28.7k tokens of context, one Tesla V100 PCIe 32 GB, package f21b9abb on server binary "
                 f"48b69ed4, 2026-10-24: one agent, {g['c1_label']}, median {m1:.1f} tok/s ({min(C1):.1f} to "
                 f"{max(C1):.1f}); four agents in the first batch, per agent "
                 + ", ".join(f"{v:.1f}" for v in C4) + f" tok/s (median {m4:.1f}); some decode while others are still in "
                 "prefill. Previous "
                 "package fd7c24dc: 72.4 and 41.9. Source: BENCHMARKS-SM70.md.", "\n".join(body))


# ---------------------------------------------------------------- kv pool
def kv_pool(c):
    h = 720
    body = [header(c, "MEMORY", f"KV pool at {mono('--memory-limit 31G')}, 4 agents · Qwen3.8-27B NVFP4",
                   "512-token prefill leaves a 124,928-token KV pool",
                   "6,656 tokens (5.1 %) below the previous package's cap: four 28.7k-token prompts fit, four"
                   " 32,768-token contexts do not.")]
    x0, x1, dmax = 380, 1120, 140_000
    sx = lambda v: x0 + (x1 - x0) * v / dmax
    top, axis_y = 270, 580
    rows = [  # y centre, name, sub lines, value, fill
        (330, "Baseline", ["before replay-commit, 2026-10-21", "limited by memory"], 98_304, c["context"]),
        (420, "Previous package", [f'{mono("--kv-cache-limit concurrency")} cap', "replay-commit, 128-token chunks"],
         131_584, c["context"]),
        (510, "This package", ["limited by memory", "512-token prefill chunks"], 124_928, c["accent"]),
    ]
    for v in range(0, dmax + 1, 20_000):
        x = sx(v)
        if v:
            body.append(f'<line x1="{x}" y1="{top}" x2="{x}" y2="{axis_y}" stroke="{c["grid"]}" stroke-width="1"/>')
        body.append(t(x, axis_y + 24, "0" if v == 0 else f"{v // 1000}K", 13, c["secondary"], anchor="middle",
                      extra=' style="font-variant-numeric: tabular-nums"'))
    body.append(t(x1, axis_y + 46, "KV tokens", 13, c["secondary"], anchor="end"))
    bh = 24
    body.append(f'<line x1="{x0}" y1="{top}" x2="{x0}" y2="{axis_y}" stroke="{c["axis"]}" stroke-width="1"/>')
    for xv, l1, l2, dy in ((114_664, "114,664 prompt tokens", "four 28.7k-token prompts", -26),
                           (131_072, "131,072 tokens", "four 32,768-token contexts", -68)):
        xr = sx(xv)
        body.append(f'<line x1="{xr}" y1="{top + dy - 14}" x2="{xr}" y2="{axis_y}" stroke="{c["primary"]}" '
                    f'stroke-width="1" stroke-dasharray="{"none" if xv == 114_664 else "4 3"}"/>')
        body.append(t(xr - 8, top + dy, l1, 14, c["primary"], 600, "end"))
        body.append(t(xr - 8, top + dy + 17, l2, 13, c["secondary"], anchor="end"))
    for yc, name, subs, v, fill in rows:
        body.append(t(x0 - 24, yc - 12, name, 17, c["primary"], 600, "end"))
        for k, sub in enumerate(subs):
            body.append(t(x0 - 24, yc + 10 + 18 * k, sub, 13, c["secondary"], anchor="end"))
        body.append(hbar(x0, yc - bh / 2, sx(v) - x0, bh, fill, f"{name}: {v:,} KV tokens"))
        body.append(t(x0 + 10, yc - 22, f"{v:,} tokens", 17, c["primary"], 650))
    meta = (f"{HW} · {mono('--memory-limit 31G')} · 4 agents · baseline 2026-10-21 · previous package 2026-10-23 · "
            "package f21b9abb 2026-10-24")
    cap = ("Source: BENCHMARKS-SM70.md (both release sections; historical Memory and capacity) · STATUS.md line 49 · "
           "receipts docs/sm70/receipts/g128/")
    body.append(footer(c, h, meta, cap))
    return frame(c, h, "Flare SM70 KV pool capacity",
                 "BF16 KV pool at memory limit 31G with 4 agents on one Tesla V100 PCIe 32 GB: 98,304 tokens before "
                 "replay-commit (2026-10-21, limited by memory); 131,584 tokens for the previous package fd7c24dc, the "
                 "--kv-cache-limit concurrency cap; 124,928 tokens for package f21b9abb, limited by memory, because "
                 "its 512-token prefill graphs take more memory (6,656 tokens or 5.1 % less). Four 28.7k-token "
                 f"prompts, 114,664 prompt tokens, {N['long']['resident_phrase']}; four 32,768-token contexts, "
                 "131,072 tokens, do not fit. Source: BENCHMARKS-SM70.md, STATUS.md line 49.", "\n".join(body))



# ---------------------------------------------------------------- comparison (same model, same card)
# Values from the published receipts, as tabulated in docs/sm70/WHY-NOT-LOW-BIT.md and BENCHMARKS-SM70.md:
# docs/sm70/receipts/measurements/speculation-a1-summary.json, speculation-a2-summary.json, a1-spec-results.json,
# a2-spec-results.json, a1-flare-d0-run.json (decode, 2026-10-23), a1-summary.json (llama.cpp TTFT and four agents),
# a1-llamacpp-kl.json and reference-kl.json (KL, 141 positions), g128/default-short-native-depth4.json,
# g128/default-prefill-summary.json and g128/srpt-prefill-summary.json (this port, package f21b9abb, 2026-10-24).
DECODE = [  # label, 1 agent, 4 agents (per agent), this port?, speculation on?
    ("This port, MTP speculation", 83.2, 64.9, True, True),
    ("llama.cpp Q4_K_M, checkpoint MTP head", 64.6, 31.8, False, True),
    ("This port, speculation off", 30.9, 28.3, True, False),
    ("llama.cpp Q4_K_M, no speculation", 33.2, 20.5, False, False),
]
KL = [("This port (NVFP4 weights unchanged)", 0.0011, True), ("llama.cpp Q8_0", 0.0013, False),
      ("llama.cpp Q4_K_M", 0.0088, False)]


def bar_h(c, x0, y, w, h, this_port, spec_on, tip):
    fill = c["accent"] if this_port else c["context"]
    if spec_on:
        return hbar(x0, y, w, h, fill, tip)
    # speculation off: outlined bar (shape encoding in addition to colour)
    return (f'<rect x="{x0 + 1}" y="{y + 1}" width="{max(w - 2, 2)}" height="{h - 2}" rx="3" fill="{c["surface"]}" '
            f'stroke="{fill}" stroke-width="2"><title>{escape(tip)}</title></rect>')


def comparison(c):
    num = ' style="font-variant-numeric: tabular-nums"'
    cw = chip_w("SAME MODEL · SAME CARD")
    body = [
        f'<rect x="{PAD}" y="40" width="{cw}" height="28" rx="14" fill="none" stroke="{c["secondary"]}" stroke-width="1"/>',
        t(PAD + cw / 2, 59, escape("SAME MODEL · SAME CARD"), 12.5, c["primary"], 700, "middle", ls="0.08em"),
        t(PAD + cw + 14, 60, "Measured on one Tesla V100 PCIe 32 GB · engine and speculation named on every bar", 15,
          c["secondary"]),
        t(PAD, 118, escape("Same model, same card: Qwen3.8-27B on one Tesla V100 32 GB"), 32, c["primary"], 650),
        t(PAD, 154, "This port runs the NVFP4 checkpoint as published; llama.cpp runs GGUFs converted from the same "
          "checkpoint.", 17, c["secondary"]),
    ]
    lx, x0, x1 = PAD, 470, 1040
    # ---- panel A: decode
    y = 214
    body.append(t(lx, y, "Decode speed per user, tokens/s (higher is better)", 21, c["primary"], 650))
    body.append(t(lx, y + 24, "Decode-only: after the first token; short public prompts, 128 output tokens", 16, c["secondary"]))
    dmax = 90
    sx = lambda v: x0 + (x1 - x0) * v / dmax
    bh, gap = 26, 8
    yy = y + 56
    grid_at = len(body)
    for gi, (gname, idx) in enumerate((("1 user", 1), ("4 users at once", 2))):
        body.append(t(lx, yy + 18, gname, 18, c["primary"], 650))
        yy += 30
        for label, one, four, port, spec in DECODE:
            v = (one, four)[idx - 1]
            body.append(t(x0 - 16, yy + 19, escape(label), 16, c["primary"], 400, "end"))
            body.append(bar_h(c, x0, yy, sx(v) - x0, bh, port, spec, f"{label}, {gname}: {v} tok/s"))
            body.append(t(sx(v) + 10, yy + 19, f"{v:.1f}", 17, c["primary"], 650, extra=num))
            yy += bh + gap
        yy += 14
    grid = [f'<line x1="{sx(v)}" y1="{y + 50}" x2="{sx(v)}" y2="{yy - 10}" stroke="{c["grid"]}" stroke-width="1"/>'
            for v in range(30, dmax + 1, 30)]
    grid.append(f'<line x1="{x0}" y1="{y + 50}" x2="{x0}" y2="{yy - 10}" stroke="{c["axis"]}" stroke-width="1"/>')
    body[grid_at:grid_at] = grid
    for v in range(0, dmax + 1, 30):
        body.append(t(sx(v), yy + 8, str(v), 14, c["secondary"], anchor="middle", extra=num))
    # legend for the encoding
    ly = yy + 36
    body.append(hbar(lx, ly - 13, 28, 16, c["accent"], "This port"))
    body.append(t(lx + 38, ly, "this port", 15, c["primary"]))
    body.append(hbar(lx + 140, ly - 13, 28, 16, c["context"], "llama.cpp"))
    body.append(t(lx + 178, ly, "llama.cpp", 15, c["primary"]))
    body.append(f'<rect x="{lx + 291}" y="{ly - 12}" width="26" height="14" rx="3" fill="{c["surface"]}" stroke="{c["secondary"]}" stroke-width="2"/>')
    body.append(t(lx + 328, ly, "outlined = speculation off", 15, c["primary"]))
    # ---- panel B: time to first token, 30,720-token prompts
    y = ly + 46
    body.append(f'<line x1="{PAD}" y1="{y - 24}" x2="{W - PAD}" y2="{y - 24}" stroke="{c["grid"]}" stroke-width="1"/>')
    body.append(t(lx, y, "Waiting for the first token, 30,720-token prompts, seconds (lower is better)", 21, c["primary"], 650))
    body.append(t(lx, y + 24, "Includes prefill (reading the prompt) and queueing; this port: package f21b9abb", 16, c["secondary"]))
    tmax = 210
    tx = lambda v: x0 + (x1 - x0) * v / tmax
    yy = y + 56
    body.append(t(lx, yy + 18, "1 prompt", 18, c["primary"], 650))
    yy += 30
    for label, v, port in (("This port", 35.46, True), ("llama.cpp Q4_K_M", 49.27, False)):
        body.append(t(x0 - 16, yy + 19, label, 16, c["primary"], 400, "end"))
        body.append(hbar(x0, yy, tx(v) - x0, bh, c["accent"] if port else c["context"], f"{label}, one 30,720-token prompt: {v} s"))
        body.append(t(tx(v) + 10, yy + 19, f"{v:.1f} s", 17, c["primary"], 650, extra=num))
        yy += bh + gap
    yy += 14
    body.append(t(lx, yy + 18, "4 prompts at once: first token of each", 18, c["primary"], 650))
    yy += 34
    rows = [("llama.cpp Q4_K_M (one prompt after another)", [59.8, 107.9, 156.7, 200.3], False, None),
            ("This port, default (taking turns)", None, True, (139.7, 142.4)),
            ("This port, --prefill-srpt", [36.2, 71.9, 107.6, 145.1], True, None)]
    for label, pts, port, rng in rows:
        yc = yy + 13
        body.append(t(x0 - 16, yc + 6, escape(label), 16, c["primary"], 400, "end"))
        body.append(f'<line x1="{x0}" y1="{yc}" x2="{x1}" y2="{yc}" stroke="{c["grid"]}" stroke-width="1"/>')
        fill = c["accent"] if port else c["context"]
        if rng:
            body.append(f'<rect x="{tx(rng[0])}" y="{yc - 8}" width="{max(tx(rng[1]) - tx(rng[0]), 6)}" height="16" rx="4" '
                        f'fill="{fill}"><title>{escape(label)}: all four at {rng[0]}-{rng[1]} s</title></rect>')
            body.append(t(tx(rng[0]) - 10, yc + 6, f"all four at {rng[0]:.0f}–{rng[1]:.0f} s", 15, c["primary"], 600, "end"))
        else:
            for v in pts:
                body.append(dot(tx(v), yc, 7, fill, c["surface"], f"{label}: first token at {v} s", square=not port))
                body.append(t(tx(v), yc - 14, f"{v:.0f}", 14, c["primary"], 600, "middle", extra=num))
        yy += 52
    for v in range(0, tmax + 1, 30):
        body.append(t(tx(v), yy + 4, str(v), 14, c["secondary"], anchor="middle", extra=num))
    body.append(t(x1, yy + 26, "seconds", 14, c["secondary"], anchor="end"))
    # ---- panel C: KL
    y = yy + 74
    body.append(f'<line x1="{PAD}" y1="{y - 24}" x2="{W - PAD}" y2="{y - 24}" stroke="{c["grid"]}" stroke-width="1"/>')
    body.append(t(lx, y, "Distance from a full-precision (FP32) run of the same checkpoint (lower is better)", 21, c["primary"], 650))
    body.append(t(lx, y + 24, "Mean KL divergence over 141 positions of 3 public prompts, log scale; this port: SM70 capture", 16, c["secondary"]))
    import math
    lo, hi = 0.0005, 0.02
    kx = lambda v: x0 + (x1 - x0) * (math.log10(v) - math.log10(lo)) / (math.log10(hi) - math.log10(lo))
    yy = y + 50
    for label, v, port in KL:
        body.append(t(x0 - 16, yy + 19, escape(label), 16, c["primary"], 400, "end"))
        body.append(hbar(x0, yy, kx(v) - x0, bh, c["accent"] if port else c["context"], f"{label}: mean KL {v}"))
        body.append(t(kx(v) + 10, yy + 19, f"{v:.4f}", 17, c["primary"], 650, extra=num))
        yy += bh + gap
    for v, lab in ((0.0005, "0.0005"), (0.001, "0.001"), (0.002, "0.002"), (0.005, "0.005"), (0.01, "0.01"), (0.02, "0.02")):
        body.append(t(kx(v), yy + 12, lab, 14, c["secondary"], anchor="middle", extra=num))
    body.append(t(lx, yy + 44, "Top-1 agreement with the FP32 run is 135-137 of 141 positions for all three, so it is not drawn.",
                  15, c["secondary"]))
    # ---- footnotes
    notes = [
        "All numbers on one Tesla V100 PCIe 32 GB. llama.cpp: commit 8e5a58df built for sm_70, GGUFs converted from the same",
        "checkpoint (head kept NVFP4), 2026-10-23; speculation off and KL for this port: previous package, 2026-10-23; this port",
        "with MTP and first-token times: package f21b9abb, 2026-10-24. Four-prompt runs: 256 output tokens on llama.cpp, 32 here.",
        "Acceptance is similar on both engines (about 0.70 at depth 4): this port's lead comes from the cost of each verification",
        "step and from batching. Strata and Project Maya cannot serve this 27B model, so they are not drawn (WHY-NOT-LOW-BIT.md).",
    ]
    fy = yy + 110
    h = fy + 22 * len(notes) + 30
    body.append(f'<line x1="{PAD}" y1="{fy - 30}" x2="{W - PAD}" y2="{fy - 30}" stroke="{c["grid"]}" stroke-width="1"/>')
    for i, n in enumerate(notes):
        body.append(t(PAD, fy + 22 * i, escape(n), 15, c["secondary"]))
    desc = ("Same model, same card: Qwen3.8-27B on one Tesla V100 PCIe 32 GB. Decode speed per user, tokens per second: "
            "1 user: this port with MTP speculation 83.2, llama.cpp Q4_K_M with the checkpoint MTP head 64.6, this port with "
            "speculation off 30.9, llama.cpp Q4_K_M without speculation 33.2. 4 users at once, per user: 64.9, 31.8, 28.3 and "
            "20.5. First token for a 30,720-token prompt: this port 35.5 s, llama.cpp Q4_K_M 49.3 s. Four such prompts at "
            "once: llama.cpp first tokens at 60, 108, 157 and 200 s; this port all four at 140 to 142 s by default, or 36, 72, "
            "108 and 145 s with --prefill-srpt. Mean KL from a full-precision run of the same checkpoint: this port 0.0011, "
            "llama.cpp Q8_0 0.0013, llama.cpp Q4_K_M 0.0088. Sources: WHY-NOT-LOW-BIT.md and BENCHMARKS-SM70.md.")
    return frame(c, h, "Same model, same card: Qwen3.8-27B on one Tesla V100 32 GB", desc, "\n".join(body))


OUT.mkdir(parents=True, exist_ok=True)
GRAPHICS = [("hero", hero), ("prefill", prefill), ("long-decode", long_decode), ("kv-pool", kv_pool),
            ("comparison", comparison)]
only = set(sys.argv[3:])
for name, fn in GRAPHICS:
    if only and name not in only:
        continue
    for mode, c in MODES.items():
        (OUT / f"{name}-{mode}.svg").write_text(fn(c), encoding="utf-8")
        print(OUT / f"{name}-{mode}.svg")
