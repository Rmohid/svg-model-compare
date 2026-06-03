#!/usr/bin/env python3
"""Single source of truth accessor for the Artificial Analysis cost/intelligence data.

`generate.py` holds the canonical `INTEL_DATA` table (it drives the main-page
Chart view). We read it via AST instead of importing, because importing
`generate.py` shells out to the `secrets` CLI at module load. AST parsing only
reads literals, so it has no side effects and needs no credentials.

Consumers: `value_chart.py` (standalone matplotlib chart) and the
`update-cost-intelligence-wiki` skill.
"""

import ast
import os

# Provider resolution mirrors generate.py:model_provider() (prefix -> vendor).
_PROVIDER_PREFIXES = [
    (("Claude",), "Anthropic"),
    (("GPT", "o3", "o4"), "OpenAI"),
    (("Gemini", "Gemma"), "Google"),
    (("Grok",), "xAI"),
    (("Llama",), "Meta"),
    (("DeepSeek",), "DeepSeek"),
    (("Qwen",), "Alibaba"),
    (("GLM",), "Z AI"),
    (("Mistral",), "Mistral"),
    (("MiniMax",), "MiniMax"),
    (("Kimi",), "Kimi"),
    (("Command",), "Cohere"),
    (("Nova",), "Amazon"),
    (("Nemotron",), "NVIDIA"),
    (("Xiaomi",), "Xiaomi"),
    (("Ling", "Ring"), "inclusionAI"),
    (("IBM",), "IBM"),
    (("Tencent",), "Tencent"),
    (("ByteDance",), "ByteDance"),
    (("StepFun",), "StepFun"),
]


def provider_for(name):
    for prefixes, vendor in _PROVIDER_PREFIXES:
        if name.startswith(prefixes):
            return vendor
    return "Other"


def _default_generate_path():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "generate.py")


def _extract_literals(path, names):
    """Return {name: literal_value} for top-level assignments in `path`."""
    tree = ast.parse(open(path).read())
    found = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            nm = node.targets[0].id
            if nm in names:
                found[nm] = ast.literal_eval(node.value)
    return found


def load(generate_path=None):
    """Load the AA snapshot from generate.py.

    Returns (snapshot_date, models, colors) where `models` is a list of dicts
    {name, provider, intel, price, speed, est} for every model that has a
    numeric price, sorted by intelligence descending. Models with no price
    cannot be placed on the cost axis and are dropped.
    """
    path = generate_path or _default_generate_path()
    lits = _extract_literals(path, {"AA_SNAPSHOT_DATE", "INTEL_DATA", "PROVIDER_COLORS"})
    snapshot = lits.get("AA_SNAPSHOT_DATE")
    raw = lits.get("INTEL_DATA")
    colors = lits.get("PROVIDER_COLORS", {})
    if raw is None:
        raise ValueError(f"INTEL_DATA not found in {path}")

    models = []
    for name, tup in raw.items():
        intel, price, speed = tup[0], tup[1], tup[2]
        est = len(tup) > 3 and tup[3] == "est"
        if price is None:
            continue
        models.append({
            "name": name,
            "provider": provider_for(name),
            "intel": intel,
            "price": price,
            "speed": speed,
            "est": est,
        })
    models.sort(key=lambda m: m["intel"], reverse=True)
    return snapshot, models, colors


def pareto_frontier(models):
    """Non-dominated set: highest intelligence available at each price tier."""
    by_price = sorted(models, key=lambda m: m["price"])
    out, best = [], -1
    for m in by_price:
        if m["intel"] > best:
            out.append(m)
            best = m["intel"]
    return out


def value_ranking(models):
    """Models sorted by intelligence-per-dollar. Free models (price 0) first."""
    def ratio(m):
        return m["intel"] / m["price"] if m["price"] else float("inf")
    return sorted(models, key=ratio, reverse=True)
