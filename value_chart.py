#!/usr/bin/env python3
"""
Render a static scatter chart ranking frontier LLMs on price, intelligence,
and speed. Best value = upper-left (high intelligence, low price).

Data source: the canonical `INTEL_DATA` table in generate.py (Artificial
Analysis leaderboard snapshot), read via aa_data so this chart never drifts
from the main-page Chart view. The snapshot date is taken from the data.

Outputs: value-chart.png and value-chart.svg in the repo root.
"""

import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

import aa_data

# Free models (price 0) snap to this x so the log axis still works.
FREE_PRICE_SENTINEL = 0.01


def render():
    snapshot_date, models, provider_colors = aa_data.load()

    fig, ax = plt.subplots(figsize=(16, 10), dpi=140)
    fig.patch.set_facecolor("white")

    # Speed drives bubble area; models with no speed get the floor size.
    speeds = [m["speed"] for m in models if m["speed"] is not None]
    max_speed = max(speeds) if speeds else 1

    def xpos(m):
        return m["price"] if m["price"] > 0 else FREE_PRICE_SENTINEL

    def bubble(m):
        s = m["speed"] if m["speed"] is not None else 0
        return 70 + 1400 * (s / max_speed)

    for m in models:
        ax.scatter(
            xpos(m), m["intel"],
            s=bubble(m),
            c=provider_colors.get(m["provider"], "#888"),
            alpha=0.72,
            edgecolors="white",
            linewidths=1.4,
            zorder=3,
        )

    # Pareto frontier: best intelligence at each price tier.
    frontier = aa_data.pareto_frontier(models)
    ax.plot([xpos(m) for m in frontier], [m["intel"] for m in frontier],
            "--", color="#444", linewidth=1.5, alpha=0.6, zorder=2,
            label="Best-value frontier")

    for m in models:
        ax.annotate(
            m["name"],
            xy=(xpos(m), m["intel"]),
            xytext=(7, 5),
            textcoords="offset points",
            fontsize=7.5,
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
        f"Frontier LLMs: Intelligence vs Price vs Speed  ·  {snapshot_date}\n"
        "Bubble size = output tokens/sec  ·  Upper-left = best value",
        fontsize=13, pad=14, color="#111",
    )

    ax.grid(True, which="both", linestyle=":", alpha=0.35, zorder=1)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    # Provider legend (first appearance order).
    providers_used, seen = [], set()
    for m in models:
        if m["provider"] not in seen:
            seen.add(m["provider"])
            providers_used.append(m["provider"])
    provider_handles = [
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor=provider_colors.get(p, "#888"),
               markersize=9, label=p)
        for p in providers_used
    ]
    leg1 = ax.legend(handles=provider_handles, loc="lower right",
                     fontsize=8, frameon=True, framealpha=0.95,
                     title="Provider", title_fontsize=9, ncol=2)
    ax.add_artist(leg1)

    # Speed legend (bubble size).
    speed_ticks = [50, 150, 300]
    speed_handles = [
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor="#bbb", markeredgecolor="white",
               markersize=np.sqrt(70 + 1400 * (s / max_speed)) * 0.5,
               label=f"{s} tok/s")
        for s in speed_ticks
    ]
    speed_handles.append(
        Line2D([0], [0], linestyle="--", color="#444", label="Best-value frontier")
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

    # Value ranking (intelligence per $) to stdout.
    print(f"\nValue ranking (Intelligence Index per $/Mtok)  ·  {snapshot_date}:")
    for i, m in enumerate(aa_data.value_ranking(models), 1):
        ratio = "free" if not m["price"] else f"{m['intel']/m['price']:6.1f}"
        speed = "  n/a" if m["speed"] is None else f"{m['speed']:.0f}"
        print(f"  {i:2}. {m['name']:24} {ratio:>6}  "
              f"(intel={m['intel']}, ${m['price']:.2f}/Mtok, {speed} tok/s)")


if __name__ == "__main__":
    render()
