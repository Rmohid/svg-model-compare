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
    ("Claude Opus 4.5", "anthropic/claude-opus-4.5", "May 2025"),
    ("Claude Haiku 4.5", "anthropic/claude-haiku-4.5", "Jun 2025"),
    ("Claude Sonnet 4", "anthropic/claude-sonnet-4", "May 2025"),        # 3mo ago SOTA
    ("Claude Opus 4.1", "anthropic/claude-opus-4.1", "Aug 2025"),        # 6mo ago SOTA
    # --- OpenAI ---
    ("GPT-5.2", "openai/gpt-5.2", "Dec 2025"),
    ("GPT-5.1", "openai/gpt-5.1", "Nov 2025"),
    ("GPT-5", "openai/gpt-5", "Jun 2025"),                              # 3mo ago SOTA
    ("GPT-5 Mini", "openai/gpt-5-mini", "Jun 2025"),                    # 6mo ago fast
    ("GPT-4.1", "openai/gpt-4.1", "Apr 2025"),
    ("GPT-4.1 Mini", "openai/gpt-4.1-mini", "Apr 2025"),
    # --- Google ---
    ("Gemini 3.1 Pro", "google/gemini-3.1-pro-preview", "Feb 2026"),
    ("Gemini 3 Pro", "google/gemini-3-pro-preview", "Nov 2025"),
    ("Gemini 3 Flash", "google/gemini-3-flash-preview", "Dec 2025"),
    ("Gemini 2.5 Pro", "google/gemini-2.5-pro", "Jun 2025"),
    ("Gemini 2.5 Flash", "google/gemini-2.5-flash", "Jun 2025"),
    # --- xAI / Grok ---
    ("Grok 4", "x-ai/grok-4", "Jul 2025"),
    ("Grok 4.1 Fast", "x-ai/grok-4.1-fast", "Nov 2025"),
    ("Grok 4 Fast", "x-ai/grok-4-fast", "Sep 2025"),
    ("Grok 3", "x-ai/grok-3", "Jun 2025"),
    ("Grok 3 Mini", "x-ai/grok-3-mini", "Jun 2025"),                    # 6mo ago fast
    # --- Chinese Models ---
    ("DeepSeek V3.2", "deepseek/deepseek-v3.2", "Oct 2025"),
    ("DeepSeek V3.1", "deepseek/deepseek-chat-v3.1", "Sep 2025"),
    ("DeepSeek R1", "deepseek/deepseek-r1", "Jan 2025"),                 # 6mo ago SOTA
    ("Kimi K2.5", "moonshotai/kimi-k2.5", "Jan 2026"),
    ("Kimi K2", "moonshotai/kimi-k2", "Jul 2025"),                       # 6mo ago SOTA
    ("MiniMax M2.5", "minimax/minimax-m2.5", "Feb 2026"),
    ("GLM-5", "z-ai/glm-5", "Feb 2026"),
    # --- Qwen: full model then smaller quantizations ---
    ("Qwen3 Max Thinking", "qwen/qwen3-max-thinking", "Feb 2026"),
    ("Qwen 3.5 397B", "qwen/qwen3.5-397b-a17b", "Feb 2026"),
    ("Qwen3 235B (Full)", "qwen/qwen3-235b-a22b", "Apr 2025"),
    ("Qwen3 32B", "qwen/qwen3-32b", "Apr 2025"),
    ("Qwen3 14B", "qwen/qwen3-14b", "Apr 2025"),
    ("Qwen3 8B", "qwen/qwen3-8b", "Apr 2025"),
    ("Qwen 2.5 7B", "qwen/qwen-2.5-7b-instruct", "Oct 2024"),
]

# Categories: newest first within each section, older SOTA models at the end
CATEGORIES = [
    ("Anthropic (current + historical)", [
        "Claude Opus 4.6", "Claude Sonnet 4.6",
        "Claude Opus 4.5", "Claude Haiku 4.5",
        "Claude Opus 4.1",
        "Claude Sonnet 4",
    ]),
    ("OpenAI (current + historical)", [
        "GPT-5.2", "GPT-5.1",
        "GPT-5", "GPT-5 Mini",
        "GPT-4.1", "GPT-4.1 Mini",
    ]),
    ("Google", [
        "Gemini 3.1 Pro", "Gemini 3 Pro", "Gemini 3 Flash",
        "Gemini 2.5 Pro", "Gemini 2.5 Flash",
    ]),
    ("xAI / Grok (current + historical)", [
        "Grok 4.1 Fast", "Grok 4", "Grok 4 Fast",
        "Grok 3", "Grok 3 Mini",
    ]),
    ("Chinese Models (current + historical)", [
        "GLM-5", "MiniMax M2.5", "Kimi K2.5", "Kimi K2",
        "DeepSeek V3.2", "DeepSeek V3.1", "DeepSeek R1",
    ]),
    ("Qwen -- Full to Small", [
        "Qwen3 Max Thinking", "Qwen 3.5 397B",
        "Qwen3 235B (Full)", "Qwen3 32B", "Qwen3 14B", "Qwen3 8B",
        "Qwen 2.5 7B",
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
    """Build comparison HTML from results dict."""
    sections_html = []
    for cat_name, model_names in CATEGORIES:
        cards_html = []
        for name in model_names:
            r = results.get(name)
            if not r:
                continue
            svg, elapsed, error = r
            date = model_dates.get(name, "")
            if error:
                content = f'<div class="error">Error: {error}</div>'
            else:
                content = f'<div class="svg-container">{svg}</div>'
            cards_html.append(f"""
            <div class="card">
                <div class="card-header">
                    <div>
                        <h3>{name}</h3>
                        <span class="release">Released: {date}</span>
                    </div>
                    <span class="time">{elapsed:.1f}s</span>
                </div>
                {content}
            </div>""")

        sections_html.append(f"""
        <section>
            <h2>{cat_name}</h2>
            <div class="grid">{"".join(cards_html)}</div>
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
    .grid {{
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(380px, 1fr));
        gap: 1.5rem;
    }}
    .card {{
        background: #1a1a1a;
        border: 1px solid #333;
        border-radius: 8px;
        overflow: hidden;
    }}
    .card-header {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.75rem 1rem;
        background: #222;
        border-bottom: 1px solid #333;
    }}
    .card-header h3 {{
        font-size: 0.95rem;
        color: #fff;
    }}
    .release {{
        font-size: 0.75rem;
        color: #666;
    }}
    .time {{
        font-size: 0.8rem;
        color: #888;
        background: #2a2a2a;
        padding: 2px 8px;
        border-radius: 4px;
        white-space: nowrap;
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
        max-width: 100%;
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
