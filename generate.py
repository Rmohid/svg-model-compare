#!/usr/bin/env python3
"""
Compare animated SVG generation across multiple AI models via OpenRouter.
Caches results to avoid re-calling models that already succeeded.
"""

import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.request import Request, urlopen
from urllib.error import HTTPError

import subprocess
API_KEY = os.environ.get("OPENROUTER_API_KEY") or subprocess.check_output(
    ["secrets", "get", "OPENROUTER_API_KEY"], text=True
).strip()
API_URL = "https://openrouter.ai/api/v1/chat/completions"
CACHE_PATH = os.path.join(os.path.dirname(__file__), "cache.json")

PROMPT = """Create an animated SVG image of a pelican riding a bicycle.
The pelican should be pedaling and the wheels should be spinning.
Use SVG animations (animate, animateTransform, etc).
Output ONLY the SVG code, nothing else. No markdown fences, no explanation.
Start with <svg and end with </svg>."""

# (display_name, model_id, release_date_str)
MODELS = [
    # --- Anthropic ---
    ("Claude Opus 4.6", "anthropic/claude-opus-4.6", "Feb 2026"),
    ("Claude Sonnet 4.6", "anthropic/claude-sonnet-4.6", "Feb 2026"),
    ("Claude Opus 4.5", "anthropic/claude-opus-4.5", "Nov 2025"),
    ("Claude Haiku 4.5", "anthropic/claude-haiku-4.5", "Oct 2025"),
    ("Claude Sonnet 4", "anthropic/claude-sonnet-4", "May 2025"),        # 3mo ago SOTA
    ("Claude Opus 4.1", "anthropic/claude-opus-4.1", "Aug 2025"),        # 6mo ago SOTA
    # --- OpenAI ---
    ("GPT-5.4", "openai/gpt-5.4", "Mar 2026"),
    ("GPT-5.4 Mini", "openai/gpt-5.4-mini", "Mar 2026"),
    ("GPT-5.2", "openai/gpt-5.2", "Dec 2025"),
    ("GPT-5.1", "openai/gpt-5.1", "Nov 2025"),
    ("GPT-5", "openai/gpt-5", "Jun 2025"),                              # 3mo ago SOTA
    ("GPT-5 Mini", "openai/gpt-5-mini", "Jun 2025"),                    # 6mo ago fast
    ("GPT-4.1", "openai/gpt-4.1", "Apr 2025"),
    ("GPT-4.1 Mini", "openai/gpt-4.1-mini", "Apr 2025"),
    # --- Google ---
    ("Gemini 3.1 Pro", "google/gemini-3.1-pro-preview", "Feb 2026"),
    ("Gemini 3.1 Flash Lite", "google/gemini-3.1-flash-lite-preview", "Mar 2026"),
    ("Gemini 3 Pro", "google/gemini-3-pro-preview", "Nov 2025"),
    ("Gemini 3 Flash", "google/gemini-3-flash-preview", "Dec 2025"),
    ("Gemini 2.5 Pro", "google/gemini-2.5-pro", "Jun 2025"),
    ("Gemini 2.5 Flash", "google/gemini-2.5-flash", "Jun 2025"),
    # --- xAI / Grok ---
    ("Grok 4.20 Beta", "x-ai/grok-4.20-beta", "Mar 2026"),
    ("Grok 4", "x-ai/grok-4", "Jul 2025"),
    ("Grok 4.1 Fast", "x-ai/grok-4.1-fast", "Nov 2025"),
    ("Grok 4 Fast", "x-ai/grok-4-fast", "Sep 2025"),
    ("Grok 3", "x-ai/grok-3", "Jun 2025"),
    ("Grok 3 Mini", "x-ai/grok-3-mini", "Jun 2025"),                    # 6mo ago fast
    # --- Chinese Models ---
    ("MiniMax M2.7", "minimax/minimax-m2.7", "Mar 2026"),
    ("GLM-5 Turbo", "z-ai/glm-5-turbo", "Mar 2026"),
    ("Xiaomi MiMo-V2-Pro", "xiaomi/mimo-v2-pro", "Mar 2026"),
    ("ByteDance Seed 2.0", "bytedance-seed/seed-2.0-lite", "Mar 2026"),
    ("DeepSeek V3.2 Speciale", "deepseek/deepseek-v3.2-speciale", "Dec 2025"),
    ("DeepSeek V3.2", "deepseek/deepseek-v3.2", "Oct 2025"),
    ("DeepSeek V3.1", "deepseek/deepseek-chat-v3.1", "Sep 2025"),
    ("DeepSeek R1", "deepseek/deepseek-r1", "Jan 2025"),                 # 6mo ago SOTA
    ("Kimi K2.5", "moonshotai/kimi-k2.5", "Jan 2026"),
    ("Kimi K2", "moonshotai/kimi-k2", "Jul 2025"),                       # 6mo ago SOTA
    ("MiniMax M2.5", "minimax/minimax-m2.5", "Feb 2026"),
    ("GLM-5", "z-ai/glm-5", "Feb 2026"),
    # --- NVIDIA ---
    ("Nemotron 3 Super", "nvidia/nemotron-3-super-120b-a12b", "Mar 2026"),
    # --- Mistral ---
    ("Mistral Small 4", "mistralai/mistral-small-2603", "Mar 2026"),
    # --- Qwen: full model then smaller quantizations ---
    ("Qwen3 Max Thinking", "qwen/qwen3-max-thinking", "Feb 2026"),
    ("Qwen 3.5 397B", "qwen/qwen3.5-397b-a17b", "Feb 2026"),
    ("Qwen 3.5 122B", "qwen/qwen3.5-122b-a10b", "Mar 2026"),
    ("Qwen 3.5 35B", "qwen/qwen3.5-35b-a3b", "Mar 2026"),
    ("Qwen 3.5 27B", "qwen/qwen3.5-27b", "Mar 2026"),
    ("Qwen3 235B (Full)", "qwen/qwen3-235b-a22b", "Apr 2025"),
    ("Qwen3 32B", "qwen/qwen3-32b", "Apr 2025"),
    ("Qwen3 14B", "qwen/qwen3-14b", "Apr 2025"),
    ("Qwen3 8B", "qwen/qwen3-8b", "Apr 2025"),
    ("Qwen 3.5 9B", "qwen/qwen3.5-9b", "Mar 2026"),
    ("Qwen 2.5 7B", "qwen/qwen-2.5-7b-instruct", "Oct 2024"),
]

# Categories: (vendor_group, [(family_label, [model_names_newest_first]), ...])
# Models in the same family (lineage) share a row in the timeline table.
CATEGORIES = [
    ("Anthropic", [
        ("Claude Opus", ["Claude Opus 4.6", "Claude Opus 4.5", "Claude Opus 4.1"]),
        ("Claude Sonnet", ["Claude Sonnet 4.6", "Claude Sonnet 4"]),
        ("Claude Haiku", ["Claude Haiku 4.5"]),
    ]),
    ("OpenAI", [
        ("GPT (flagship)", ["GPT-5.4", "GPT-5.2", "GPT-5.1", "GPT-5", "GPT-4.1"]),
        ("GPT Mini", ["GPT-5.4 Mini", "GPT-5 Mini", "GPT-4.1 Mini"]),
    ]),
    ("Google", [
        ("Gemini Pro", ["Gemini 3.1 Pro", "Gemini 3 Pro", "Gemini 2.5 Pro"]),
        ("Gemini Flash", ["Gemini 3.1 Flash Lite", "Gemini 3 Flash", "Gemini 2.5 Flash"]),
    ]),
    ("xAI / Grok", [
        ("Grok (flagship)", ["Grok 4.20 Beta", "Grok 4", "Grok 3"]),
        ("Grok Fast", ["Grok 4.1 Fast", "Grok 4 Fast"]),
        ("Grok Mini", ["Grok 3 Mini"]),
    ]),
    ("Chinese Models", [
        ("DeepSeek", ["DeepSeek V3.2 Speciale", "DeepSeek V3.2", "DeepSeek V3.1", "DeepSeek R1"]),
        ("Kimi", ["Kimi K2.5", "Kimi K2"]),
        ("MiniMax", ["MiniMax M2.7", "MiniMax M2.5"]),
        ("GLM", ["GLM-5 Turbo", "GLM-5"]),
        ("Xiaomi", ["Xiaomi MiMo-V2-Pro"]),
        ("ByteDance", ["ByteDance Seed 2.0"]),
    ]),
    ("NVIDIA", [
        ("Nemotron", ["Nemotron 3 Super"]),
    ]),
    ("Mistral", [
        ("Mistral Small", ["Mistral Small 4"]),
    ]),
    ("Qwen", [
        ("Qwen Max Thinking", ["Qwen3 Max Thinking"]),
        ("Qwen Flagship", ["Qwen 3.5 397B", "Qwen 3.5 122B", "Qwen3 235B (Full)"]),
        ("Qwen 35B", ["Qwen 3.5 35B"]),
        ("Qwen 27-32B", ["Qwen 3.5 27B", "Qwen3 32B"]),
        ("Qwen 14B", ["Qwen3 14B"]),
        ("Qwen 8-9B", ["Qwen 3.5 9B", "Qwen3 8B"]),
        ("Qwen 7B", ["Qwen 2.5 7B"]),
    ]),
]


def load_cache():
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH) as f:
            return json.load(f)
    return {}


def save_cache(cache):
    with open(CACHE_PATH, "w") as f:
        json.dump(cache, f)


def call_model(name, model_id):
    """Call a single model and return (name, svg_output, elapsed, error)."""
    print(f"  [{name}] Requesting...", flush=True)
    start = time.time()
    payload = json.dumps({
        "model": model_id,
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": 16000,
        "temperature": 0.7,
    }).encode()

    req = Request(API_URL, data=payload, method="POST")
    req.add_header("Authorization", f"Bearer {API_KEY}")
    req.add_header("Content-Type", "application/json")
    req.add_header("HTTP-Referer", "https://localhost")

    try:
        with urlopen(req, timeout=300) as resp:
            data = json.loads(resp.read())
        elapsed = time.time() - start
        content = data["choices"][0]["message"]["content"]
        # Strip markdown fences if present
        content = re.sub(r"```(?:svg|xml|html)?\s*\n?", "", content)
        content = content.replace("```", "")
        svg_match = re.search(r"(<svg[\s\S]*?</svg>)", content, re.IGNORECASE)
        if svg_match:
            svg = svg_match.group(1)
            print(f"  [{name}] Done in {elapsed:.1f}s ({len(svg)} chars)", flush=True)
            return name, svg, elapsed, None
        else:
            print(f"  [{name}] Done in {elapsed:.1f}s but no SVG found", flush=True)
            return name, None, elapsed, "No <svg> tag found in response"
    except HTTPError as e:
        elapsed = time.time() - start
        body = e.read().decode() if e.fp else ""
        print(f"  [{name}] Error: {e.code} in {elapsed:.1f}s", flush=True)
        return name, None, elapsed, f"HTTP {e.code}: {body[:200]}"
    except Exception as e:
        elapsed = time.time() - start
        print(f"  [{name}] Error: {e} in {elapsed:.1f}s", flush=True)
        return name, None, elapsed, str(e)


def build_html(results, model_dates):
    """Build per-vendor timeline tables. Each row = model family, columns = release months."""
    from datetime import datetime

    def sort_months(month_set):
        return sorted(month_set, key=lambda d: datetime.strptime(d, "%b %Y"), reverse=True)

    sections_html = []
    for cat_name, families in CATEGORIES:
        # Collect months that have at least one model in this vendor group
        group_months = set()
        for _, model_names in families:
            for name in model_names:
                if name in model_dates and name in results:
                    group_months.add(model_dates[name])
        if not group_months:
            continue
        months = sort_months(group_months)

        # Header row
        header = '<tr><th class="corner"></th>'
        for m in months:
            header += f"<th>{m}</th>"
        header += "</tr>"

        # Family rows
        rows = []
        for family_label, model_names in families:
            cells = [f'<td class="model-name">{family_label}</td>']
            for m in months:
                # Find the model in this family released this month
                matched = None
                for name in model_names:
                    if model_dates.get(name) == m and name in results:
                        matched = name
                        break
                if matched:
                    svg, elapsed, error = results[matched]
                    if error:
                        cells.append(
                            f'<td class="svg-cell">'
                            f'<div class="cell-label">{matched}'
                            f' <span class="time">{elapsed:.1f}s</span></div>'
                            f'<div class="error">Error: {error}</div></td>'
                        )
                    else:
                        cells.append(
                            f'<td class="svg-cell">'
                            f'<div class="cell-label">{matched}'
                            f' <span class="time">{elapsed:.1f}s</span></div>'
                            f'<div class="svg-container">{svg}</div></td>'
                        )
                else:
                    cells.append('<td class="empty-cell"></td>')
            rows.append(f'<tr>{"".join(cells)}</tr>')

        sections_html.append(f"""
        <section>
            <h2>{cat_name}</h2>
            <div class="table-wrap">
            <table>
            <thead>{header}</thead>
            <tbody>{"".join(rows)}</tbody>
            </table>
            </div>
        </section>""")

    total = len(results)
    success = sum(1 for v in results.values() if v[2] is None)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Animated SVG Model Comparison - Pelican Riding a Bicycle</title>
<style>
    * {{ margin: 0; padding: 0; box-sizing: border-box; }}
    body {{
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        background: #0a0a0a;
        color: #e0e0e0;
        padding: 2rem;
    }}
    h1 {{
        text-align: center;
        margin-bottom: 0.5rem;
        font-size: 1.8rem;
        color: #fff;
    }}
    .subtitle {{
        text-align: center;
        color: #888;
        margin-bottom: 2rem;
        font-size: 0.9rem;
        line-height: 1.6;
    }}
    section {{
        margin-bottom: 2.5rem;
    }}
    h2 {{
        font-size: 1.3rem;
        color: #aaa;
        border-bottom: 1px solid #333;
        padding-bottom: 0.5rem;
        margin-bottom: 1rem;
    }}
    .table-wrap {{
        overflow-x: auto;
    }}
    table {{
        border-collapse: separate;
        border-spacing: 0;
    }}
    thead th {{
        background: #111;
        color: #ccc;
        font-size: 0.85rem;
        font-weight: 600;
        padding: 0.6rem 1rem;
        border-bottom: 2px solid #444;
        white-space: nowrap;
        text-align: center;
    }}
    thead th.corner {{
        min-width: 160px;
        background: #111;
    }}
    td.model-name {{
        background: #141414;
        min-width: 160px;
        max-width: 160px;
        padding: 0.75rem 1rem;
        border-right: 2px solid #333;
        border-bottom: 1px solid #222;
        vertical-align: middle;
        font-weight: 600;
        font-size: 0.9rem;
        color: #fff;
    }}
    td.svg-cell {{
        padding: 0;
        border-bottom: 1px solid #222;
        min-width: 380px;
        vertical-align: top;
    }}
    .cell-label {{
        padding: 0.4rem 0.75rem;
        background: #222;
        font-size: 0.8rem;
        color: #ccc;
        border-bottom: 1px solid #333;
        white-space: nowrap;
    }}
    .cell-label .time {{
        color: #888;
        background: #2a2a2a;
        padding: 1px 6px;
        border-radius: 3px;
        font-size: 0.75rem;
        margin-left: 0.5rem;
    }}
    .svg-container {{
        padding: 1rem;
        display: flex;
        justify-content: center;
        align-items: center;
        min-height: 350px;
        background: #fff;
    }}
    .svg-container svg {{
        width: 100%;
        height: auto;
        max-height: 400px;
    }}
    .error {{
        padding: 1.5rem;
        color: #f44;
        font-size: 0.85rem;
        min-height: 200px;
        display: flex;
        align-items: center;
        justify-content: center;
        text-align: center;
        background: #1a1a1a;
    }}
    td.empty-cell {{
        background: #0e0e0e;
        border-bottom: 1px solid #222;
        min-width: 0;
        width: 0;
        padding: 0;
    }}
</style>
</head>
<body>
<h1>Animated SVG: Pelican Riding a Bicycle</h1>
<p class="subtitle">
    Same prompt sent to {total} models via OpenRouter ({success} returned valid SVG)<br>
    Generated {time.strftime('%Y-%m-%d %H:%M')}
</p>
{"".join(sections_html)}
</body>
</html>"""
    return html


def main():
    model_dates = {name: date for name, _, date in MODELS}
    model_map = {name: mid for name, mid, _ in MODELS}

    # Load cache of previous successful results
    cache = load_cache()
    results = {}
    to_call = {}

    for name, mid in model_map.items():
        if name in cache and cache[name].get("svg"):
            print(f"  [{name}] Using cached result", flush=True)
            results[name] = (cache[name]["svg"], cache[name]["elapsed"], None)
        else:
            to_call[name] = mid

    if to_call:
        print(f"Calling {len(to_call)} models ({len(results)} cached)...", flush=True)
        with ThreadPoolExecutor(max_workers=6) as pool:
            futures = {pool.submit(call_model, name, mid): name for name, mid in to_call.items()}
            for future in as_completed(futures):
                name, svg, elapsed, error = future.result()
                results[name] = (svg, elapsed, error)
                if svg and not error:
                    cache[name] = {"svg": svg, "elapsed": elapsed}
        save_cache(cache)
    else:
        print("All models cached, building HTML...", flush=True)

    print(f"\nBuilding HTML...", flush=True)
    html = build_html(results, model_dates)
    out_path = os.path.join(os.path.dirname(__file__), "index.html")
    with open(out_path, "w") as f:
        f.write(html)
    print(f"Output: {out_path}")

    success = sum(1 for v in results.values() if v[2] is None)
    print(f"\nResults: {success}/{len(results)} models returned valid SVG")
    for name in model_map:
        if name in results:
            svg, elapsed, error = results[name]
            status = "OK" if not error else f"FAIL: {error[:60]}"
            print(f"  {name:25s} {elapsed:6.1f}s  {status}")


if __name__ == "__main__":
    main()
