#!/usr/bin/env python3
"""Probe which `reasoning.effort` value actually maximizes a model's thinking.

WHY THIS EXISTS
---------------
generate.py must call every model at the highest reasoning effort its provider
exposes (see CLAUDE.md / Task D). But OpenRouter's unified `reasoning.effort`
param is NOT uniform across providers, and an unsupported value is silently
ignored rather than rejected -- so you cannot tell from an HTTP 200 whether the
effort you asked for did anything at all.

Confirmed 2026-07-31 on the GPT-5.6 family (reasoning tokens, same prompt):

    effort     Sol    Terra   Luna
    default     25       59     54
    high       121      119    219
    xhigh      138      201   2070
    max       4142     5178   3624

`xhigh` -- the value GLM-5.2 uses as its maximum -- is very nearly a no-op on
Sol and Terra. Copying another model's effort label is how a model ends up
quietly generating at ~zero reasoning. Always probe.

USAGE
-----
    export SOPS_AGE_KEY_FILE="$HOME/.config/sops/age/keys.txt"
    python3 probe_reasoning_effort.py openai/gpt-5.6-sol            # full ladder
    python3 probe_reasoning_effort.py --screen openai/gpt-5.6-sol …  # default-vs-max only

--screen is the cheap two-call version for triaging many models at once; run the
full ladder only on models it flags. Reads the SVG prompt's cousin (a
reasoning-inducing prompt) rather than the real prompt so probes stay cheap.

Interpreting results: if `max` (or whichever level is highest) shows reasoning
tokens far above `default`, the model has headroom and belongs in
REASONING_EFFORT_OVERRIDES. If every level is flat, the model either isn't
reasoning-capable or doesn't honor the param -- leave it out.
"""
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from urllib.request import Request, urlopen
from urllib.error import HTTPError

API_URL = "https://openrouter.ai/api/v1/chat/completions"

# Deliberately reasoning-inducing but short-answer, so probes stay cheap: the
# thinking budget is what we're measuring, not the visible completion.
PROBE_PROMPT = (
    "Design an animated SVG of a pelican riding a bicycle with spinning wheels "
    "and pedaling legs. Think carefully about the joint kinematics and timing "
    "before answering. Output only a brief plan, not the SVG."
)

FULL_LADDER = [None, "low", "medium", "high", "xhigh", "max"]
SCREEN_LADDER = [None, "max"]


def _key(secret_name="OPENROUTER_911_API_KEY"):
    return subprocess.check_output(["secrets", "get", secret_name], text=True).strip()


def probe(model_id, effort, api_key, max_tokens=8000):
    """One probe call. Returns dict with reasoning/completion token counts or an error."""
    body = {
        "model": model_id,
        "messages": [{"role": "user", "content": PROBE_PROMPT}],
        "max_tokens": max_tokens,
    }
    if effort:
        body["reasoning"] = {"effort": effort}

    req = Request(API_URL, data=json.dumps(body).encode(), method="POST")
    req.add_header("Authorization", f"Bearer {api_key}")
    req.add_header("Content-Type", "application/json")
    try:
        with urlopen(req, timeout=300) as resp:
            d = json.loads(resp.read())
    except HTTPError as e:
        detail = (e.read().decode() if e.fp else "")[:120]
        return {"effort": effort, "error": f"HTTP {e.code}: {detail}"}
    except Exception as e:
        return {"effort": effort, "error": str(e)[:120]}

    if "error" in d:
        return {"effort": effort, "error": str(d["error"].get("message"))[:120]}

    u = d.get("usage") or {}
    cd = u.get("completion_tokens_details") or {}
    return {
        "effort": effort,
        "reasoning": cd.get("reasoning_tokens") or 0,
        "completion": u.get("completion_tokens") or 0,
        "finish": d["choices"][0].get("finish_reason"),
    }


def probe_model(model_id, ladder, api_key):
    results = []
    for effort in ladder:
        results.append(probe(model_id, effort, api_key))
    return model_id, results


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    screen = "--screen" in sys.argv
    if not args:
        print(__doc__)
        sys.exit(1)

    ladder = SCREEN_LADDER if screen else FULL_LADDER
    api_key = _key()

    # Models probe independently; run them concurrently but keep each model's
    # own ladder sequential so one model can't stampede a provider's rate limit.
    with ThreadPoolExecutor(max_workers=4) as pool:
        outputs = list(pool.map(lambda m: probe_model(m, ladder, api_key), args))

    for model_id, results in outputs:
        print(f"\n=== {model_id} ===")
        print(f"{'effort':<10} {'reasoning':>10} {'completion':>11}  note")
        print("-" * 50)
        for r in results:
            label = r["effort"] or "(default)"
            if "error" in r:
                print(f"{label:<10} {'-':>10} {'-':>11}  {r['error']}")
            else:
                print(f"{label:<10} {r['reasoning']:>10} {r['completion']:>11}  finish={r['finish']}")
        ok = [r for r in results if "error" not in r]
        if len(ok) >= 2:
            base = ok[0]["reasoning"]
            best = max(ok, key=lambda r: r["reasoning"])
            if best["reasoning"] > max(base * 3, base + 500):
                print(f"  -> HEADROOM: {best['effort']!r} gives {best['reasoning']} vs "
                      f"{base} at default. Add to REASONING_EFFORT_OVERRIDES.")
            else:
                print("  -> flat: no meaningful headroom; leave out of REASONING_EFFORT_OVERRIDES.")


if __name__ == "__main__":
    main()
