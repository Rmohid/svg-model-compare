#!/usr/bin/env python3
"""Generate the LLM Wiki "Frontier LLM Cost vs Intelligence" synthesis page.

Reads the canonical AA snapshot from generate.py (via aa_data) and prints the
complete wiki page (frontmatter + body) to stdout. The caller pipes this into
the `mcp__llm-wiki__wiki_write` tool. No side effects, no credentials needed.

Usage:  python3 build_wiki_page.py            # full page to stdout
        python3 build_wiki_page.py --mermaid  # just the quadrantChart block
"""

import math
import os
import sys

# aa_data lives at the repo root (.claude/skills/<skill>/build_wiki_page.py).
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, REPO_ROOT)
import aa_data  # noqa: E402

WIKI_PATH = "wiki/syntheses/llm-cost-vs-intelligence-frontier.md"
FREE_SENTINEL = 0.01  # free models snap here so the log x-axis still works


def fmt_price(p):
    return "Free" if not p else f"{p:.2f}"


def fmt_value(m):
    return "free" if not m["price"] else f"{m['intel'] / m['price']:.1f}"


def fmt_speed(s):
    return "n/a" if s is None else f"{s:.0f}"


def sanitize_label(name):
    """quadrantChart axis/quadrant labels reject ( ) + : — strip defensively."""
    for ch in "():+":
        name = name.replace(ch, " ")
    return " ".join(name.split())


def quadrant_block(snapshot, models, colors):
    """A readable curated subset: the frontier plus each provider's flagship."""
    frontier = aa_data.pareto_frontier(models)
    flagship = {}
    for m in models:
        if m["provider"] not in flagship:  # models are sorted by intel desc
            flagship[m["provider"]] = m
    chosen = {m["name"]: m for m in frontier}
    chosen.update({m["name"]: m for m in flagship.values()})
    chosen = sorted(chosen.values(), key=lambda m: m["intel"], reverse=True)

    xs = [math.log(m["price"] if m["price"] > 0 else FREE_SENTINEL) for m in chosen]
    ys = [m["intel"] for m in chosen]
    xlo, xhi = min(xs), max(xs)
    ylo, yhi = min(ys), max(ys)
    smax = max((m["speed"] for m in chosen if m["speed"]), default=1)

    def clamp(v):  # quadrant coords must be strictly inside (0, 1)
        return max(0.02, min(0.98, v))

    lines = [
        "quadrantChart",
        f"  title Frontier LLMs - Cost vs Intelligence ({snapshot})",
        "  x-axis Low price --> High price",
        "  y-axis Low intelligence --> High intelligence",
        "  quadrant-1 Smart and pricey",
        "  quadrant-2 Best value",
        "  quadrant-3 Budget",
        "  quadrant-4 Overpriced",
    ]
    for m in chosen:
        x = clamp((math.log(m["price"] if m["price"] > 0 else FREE_SENTINEL) - xlo) / (xhi - xlo))
        y = clamp((m["intel"] - ylo) / (yhi - ylo))
        r = 4 + 18 * ((m["speed"] or 0) / smax)
        color = colors.get(m["provider"], "#888888")
        lines.append(
            f'  "{sanitize_label(m["name"])}": [{x:.3f}, {y:.3f}] '
            f"radius: {r:.0f}, color: {color}, stroke-color: #ffffff"
        )
    return "\n".join(lines), chosen


def build_page():
    snapshot, models, colors = aa_data.load()
    frontier = aa_data.pareto_frontier(models)
    ranked = aa_data.value_ranking(models)

    smartest_intel = models[0]["intel"]
    smartest = [m["name"] for m in models if m["intel"] == smartest_intel]
    priciest = max(models, key=lambda m: m["price"])
    smart_cheap = min((m for m in models if m["intel"] >= 50),
                      key=lambda m: m["price"], default=None)
    value_leaders = [m for m in ranked if m["price"] > 0][:2]
    top5 = models[:5]

    mermaid, _ = quadrant_block(snapshot, models, colors)

    smartest_str = " and ".join(
        f"**{n}** (${next(m['price'] for m in models if m['name'] == n):.2f})"
        for n in smartest
    )
    cheap_str = (f"**{smart_cheap['name']}** (Intelligence {smart_cheap['intel']} at "
                 f"**${smart_cheap['price']:.2f}/Mtok**)") if smart_cheap else "n/a"
    value_str = " and ".join(
        f"**{m['name']}** ({m['intel'] / m['price']:.0f} intel/$)" for m in value_leaders
    )

    rows = "\n".join(
        f"| {m['name']}{' *(est)*' if m['est'] else ''} | {m['provider']} | {m['intel']} | "
        f"{fmt_price(m['price'])} | {fmt_speed(m['speed'])} | {fmt_value(m)} |"
        for m in models
    )

    frontier_list = "\n".join(
        f"{i}. **{m['name']}** — {'Free' if not m['price'] else '$' + format(m['price'], '.2f')}, "
        f"Intelligence {m['intel']}"
        for i, m in enumerate(frontier, 1)
    )

    fastest = max((m for m in models if m["speed"]), key=lambda m: m["speed"])
    slow_value = [m["name"] for m in value_leaders if (m["speed"] or 999) < 60]
    top5_str = ", ".join(f"{m['name']} (I{m['intel']})" for m in top5)
    frontier_providers = sorted({m["provider"] for m in frontier})

    return f"""---
project: shared
type: synthesis
summary: "Frontier LLM cost vs intelligence snapshot (Artificial Analysis, {snapshot}) with actual blended $/Mtok prices, the best-value Pareto frontier, and a Mermaid quadrantChart rendering. Auto-generated from generate.py INTEL_DATA."
tags:
- topic/ai-engineering
- domain/ai-engineering
last_verified: {snapshot}
sources:
- "https://artificialanalysis.ai/leaderboards/models"
aliases:
- LLM Cost vs Intelligence
- Frontier LLM Pricing
- Cost vs Intelligence Frontier
confidence: medium
---

# Frontier LLM Cost vs Intelligence

> Auto-generated from `~/code/svg-model-compare` (`generate.py` `INTEL_DATA`, the
> table that drives the main-page Chart view). Regenerate with the
> `update-cost-intelligence-wiki` skill — do not hand-edit.

## TL;DR

A snapshot of {len(models)} frontier LLMs ranked on **intelligence vs price vs speed**,
sourced from the **Artificial Analysis** leaderboard (`artificialanalysis.ai/leaderboards/models`).
Best value = high intelligence at low price (upper-left of the scatter). Numbers
drift weekly; this is the **{snapshot}** snapshot and prices are headline blended $/Mtok.

**Cheat sheet:** the cheapest model that is still genuinely smart (Intelligence ≥ 50)
is {cheap_str}. The smartest model(s): {smartest_str} (Intelligence {smartest_intel}).
The most expensive is **{priciest['name']}** at **${priciest['price']:.2f}/Mtok**.
On raw intelligence-per-dollar, {value_str} lead among paid models.

## Snapshot — actual prices ({snapshot})

Intelligence = Artificial Analysis Intelligence Index. Price = blended USD per 1M tokens.
Speed = output tokens/sec. Value = Intelligence per $/Mtok. Sorted by intelligence.
*(est)* marks an estimated price. Models with no published price are omitted.

| Model | Provider | Intelligence | Price ($/Mtok) | Speed (tok/s) | Value (I/$) |
|---|---|---:|---:|---:|---:|
{rows}

## Best-value Pareto frontier

The non-dominated set — at each price tier, the highest intelligence available.
Nothing is both cheaper *and* smarter than these:

{frontier_list}

Everything else is off the frontier: you pay more for equal-or-lower intelligence.

## Reading the chart

- **Top-left = best value** (smart + cheap), top-right = premium (smart + pricey),
  bottom = weak regardless of price.
- **Speed is a third axis** the price table flattens: the fastest model is
  **{fastest['name']}** at {fastest['speed']:.0f} tok/s.{' Some value leaders are slow (' + ', '.join(slow_value) + '), so if latency matters the value ranking shifts.' if slow_value else ''}
- **Top of the intelligence index:** {top5_str}.
- **Best-value frontier providers:** {', '.join(frontier_providers)}.

## As a Mermaid diagram

The scatter can be approximated with a Mermaid **`quadrantChart`** (not `xychart-beta`,
which only does bar/line). Position encodes price (x) and intelligence (y), `radius:`
encodes speed, `color:` encodes provider. It is a faithful *qualitative* view but loses
the real axes (coords are forced to 0–1), the log price scale, exact values, and the
frontier line — keep `value-chart.svg` as the quantitative reference. To stay readable
this shows a **curated subset** (the Pareto frontier plus each provider's flagship).

**Gotchas (Mermaid 11):** quadrant point coordinates must be strictly less than 1.0
(a literal 1.0 is a lexical error — clamp to 0.98); quadrant/axis labels reject
parentheses, plus signs, and colons.

```mermaid
{mermaid}
```

## Provenance and refresh

- **Source:** Artificial Analysis leaderboard (`artificialanalysis.ai/leaderboards/models`).
- **Single source of truth:** `~/code/svg-model-compare/generate.py` — the `INTEL_DATA`
  dict and `AA_SNAPSHOT_DATE`. This page is generated from it (via `aa_data.py`), so it
  matches the main-page Chart view exactly.
- **Regenerate:** run the `update-cost-intelligence-wiki` skill in that repo, or
  `python3 .claude/skills/update-cost-intelligence-wiki/build_wiki_page.py`.
- **Drift warning:** prices and intelligence scores move on roughly weekly cadence;
  treat any snapshot older than ~2 weeks as stale and re-pull. Pricing lookups for
  individual models can also go through [[openrouter]] (use per-model pages, not the
  bulk endpoint).

## Related

- [[openrouter]] — multi-provider gateway and pricing-lookup surface for these models
- [[metr]] — independent frontier-model capability evaluations (complements the AA index)
- [[simon-willison]] — fastest source for new-model coverage as the snapshot drifts
"""


def main():
    if "--mermaid" in sys.argv:
        snapshot, models, colors = aa_data.load()
        print(quadrant_block(snapshot, models, colors)[0])
    else:
        print(build_page(), end="")


if __name__ == "__main__":
    main()
