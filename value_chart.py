#!/usr/bin/env python3
"""
Render a static scatter chart ranking frontier LLMs on price, intelligence,
and speed. Best value = upper-left (high intelligence, low price).

Data source: Artificial Analysis leaderboard (artificialanalysis.ai/leaderboards/models).
Snapshot date in the title -- numbers drift, re-pull manually when stale.

Outputs: value-chart.png and value-chart.svg in the repo root.
"""

import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

SNAPSHOT_DATE = "2026-04-16"

# (display_name, provider, intelligence_index, price_blended_per_mtok, output_tok_per_sec)
# Hand-picked frontier models from the AA leaderboard. "max" / "xhigh" reasoning
# variants used where they are the headline configuration for the model.
MODELS = [
    ("Gemini 3.1 Pro",    "Google",    57, 4.50, 29.16),
    ("GPT-5.4",           "OpenAI",    57, 5.63, 205.54),
    ("Claude Opus 4.6",   "Anthropic", 53, 10.00, 16.78),
    ("Claude Sonnet 4.6", "Anthropic", 52, 6.00, 90.36),
    ("GLM-5.1",           "Z AI",      51, 2.15, 1.54),
    ("Qwen3.6 Plus",      "Alibaba",   50, 1.13, 2.66),
    ("MiniMax M2.7",      "MiniMax",   50, 0.53, 2.11),
    ("Grok 4.20",         "xAI",       49, 3.00, 16.17),
    ("GPT-5.4 mini",      "OpenAI",    49, 1.69, 9.79),
    ("Kimi K2.5",         "Kimi",      47, 1.20, 1.65),
    ("Gemini 3 Flash",    "Google",    46, 1.13, 7.32),
    ("GPT-5.4 nano",      "OpenAI",    44, 0.46, 4.53),
    ("DeepSeek V3.2",     "DeepSeek",  42, 0.32, 1.81),
    ("o3",                "OpenAI",    38, 3.50, 9.07),
    ("Claude Haiku 4.5",  "Anthropic", 37, 2.00, 11.26),
    ("Grok 4.1 Fast",     "xAI",       39, 0.28, 7.45),
    ("Mistral Large 3",   "Mistral",   23, 0.75, 23.67),
    ("Llama 4 Maverick",  "Meta",      18, 0.50, 1.08),
    ("Nova Premier",      "Amazon",    19, 5.00, 3.06),
    ("Command A",         "Cohere",    13, 4.38, 2.06),
]

PROVIDER_COLORS = {
    "Anthropic": "#D97757",
    "OpenAI":    "#10A37F",
    "Google":    "#4285F4",
    "xAI":       "#111111",
    "Meta":      "#1877F2",
    "DeepSeek":  "#4D6BFE",
    "Alibaba":   "#FF6A00",
    "Z AI":      "#7B68EE",
    "Mistral":   "#FA520F",
    "MiniMax":   "#E91E63",
    "Kimi":      "#6B46C1",
    "Cohere":    "#39594D",
    "Amazon":    "#FF9900",
}


def pareto_frontier(points):
    """Return points on the upper-left frontier: max intelligence at min price.
    points: list of (price, intelligence, label). Returns sorted-by-price subset."""
    sorted_pts = sorted(points, key=lambda p: p[0])
    frontier = []
    best_y = -np.inf
    for p in sorted_pts:
        if p[1] > best_y:
            frontier.append(p)
            best_y = p[1]
    return frontier


def render():
    fig, ax = plt.subplots(figsize=(14, 9), dpi=140)
    fig.patch.set_facecolor("white")

    speeds = [m[4] for m in MODELS]
    max_speed = max(speeds)
    # Bubble area scales with sqrt(speed) so visual diameter ~= linear in speed
    sizes = [80 + 1500 * (s / max_speed) for s in speeds]

    for (name, provider, intel, price, speed), size in zip(MODELS, sizes):
        ax.scatter(
            price, intel,
            s=size,
            c=PROVIDER_COLORS.get(provider, "#888"),
            alpha=0.72,
            edgecolors="white",
            linewidths=1.5,
            zorder=3,
        )

    # Pareto frontier: best intelligence at each price tier
    frontier_pts = pareto_frontier([(m[3], m[2], m[0]) for m in MODELS])
    fx = [p[0] for p in frontier_pts]
    fy = [p[1] for p in frontier_pts]
    ax.plot(fx, fy, "--", color="#444", linewidth=1.5, alpha=0.6,
            zorder=2, label="Best-value frontier")

    # Labels: nudge to avoid bubble overlap
    for name, provider, intel, price, speed in MODELS:
        ax.annotate(
            name,
            xy=(price, intel),
            xytext=(8, 6),
            textcoords="offset points",
            fontsize=8.5,
            color="#222",
            fontweight="500",
            zorder=4,
        )

    ax.set_xscale("log")
    ax.set_xlabel("Blended price (USD per 1M tokens, log scale)",
                  fontsize=11, color="#333")
    ax.set_ylabel("Artificial Analysis Intelligence Index",
                  fontsize=11, color="#333")
    ax.set_title(
        f"Frontier LLMs: Intelligence vs Price vs Speed  ·  {SNAPSHOT_DATE}\n"
        "Bubble size = output tokens/sec  ·  Upper-left = best value",
        fontsize=13, pad=14, color="#111",
    )

    ax.grid(True, which="both", linestyle=":", alpha=0.35, zorder=1)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    # Provider legend
    providers_used = []
    seen = set()
    for _, p, *_ in MODELS:
        if p not in seen:
            seen.add(p)
            providers_used.append(p)

    provider_handles = [
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor=PROVIDER_COLORS.get(p, "#888"),
               markersize=10, label=p)
        for p in providers_used
    ]
    leg1 = ax.legend(handles=provider_handles, loc="lower right",
                     fontsize=9, frameon=True, framealpha=0.95,
                     title="Provider", title_fontsize=9)
    ax.add_artist(leg1)

    # Speed legend (bubble size)
    speed_ticks = [10, 50, 200]
    speed_handles = [
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor="#bbb", markeredgecolor="white",
               markersize=np.sqrt(80 + 1500 * (s / max_speed)) * 0.5,
               label=f"{s} tok/s")
        for s in speed_ticks
    ]
    speed_handles.append(
        Line2D([0], [0], linestyle="--", color="#444",
               label="Best-value frontier")
    )
    ax.legend(handles=speed_handles, loc="lower left",
              fontsize=9, frameon=True, framealpha=0.95,
              title="Output speed", title_fontsize=9, labelspacing=1.4)

    plt.tight_layout()

    out_dir = os.path.dirname(os.path.abspath(__file__))
    png_path = os.path.join(out_dir, "value-chart.png")
    svg_path = os.path.join(out_dir, "value-chart.svg")
    fig.savefig(png_path, dpi=160, bbox_inches="tight", facecolor="white")
    fig.savefig(svg_path, bbox_inches="tight", facecolor="white")
    print(f"Wrote {png_path}")
    print(f"Wrote {svg_path}")

    # Also print value ranking (intelligence per $) to stdout
    print("\nValue ranking (Intelligence Index per $/Mtok):")
    ranked = sorted(MODELS, key=lambda m: m[2] / m[3], reverse=True)
    for i, (name, provider, intel, price, speed) in enumerate(ranked, 1):
        print(f"  {i:2}. {name:20} {intel/price:6.1f}  "
              f"(intel={intel}, ${price:.2f}/Mtok, {speed:.1f} tok/s)")


if __name__ == "__main__":
    render()
