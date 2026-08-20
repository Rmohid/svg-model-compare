# SVG Model Compare

## Rules
- **Only providers with an AA intelligence ≥ 50 model are shown** (Robert's policy, 2026-08-19). A provider whose best model scores < 50 on the Artificial Analysis leaderboard is pruned entirely — no exceptions. Qualifying set (Aug 2026): Anthropic, OpenAI, xAI, Kimi, Z AI (GLM), Alibaba (Qwen), Meta, Google, DeepSeek. All models from non-qualifying providers are removed from `MODELS`, `PRICING`, `INTEL_DATA`, `CATEGORIES`, and `PROVIDER_COLORS`.
- **New models from qualifying providers are added as they ship** (from the AA leaderboard / OpenRouter), even if that specific model scores below 50 — the provider gate is per-provider, not per-model.
- **Never regenerate cached images.** The `cache.json` file stores SVG outputs from prior API calls. When adding new models, only call the API for models not already in the cache. The `generate.py` script already handles this via its cache mechanism — do not clear or rebuild the cache.
  - *Narrow exception:* a single entry generated at the wrong reasoning effort may be deleted and re-run. Back the old entry up first (max effort can produce a **worse** result — see below), and never bulk-clear.
- When adding a new model, add it to both the `MODELS` list and the appropriate `CATEGORIES` section in `generate.py`, then run the script.
- **Never remove a model that disappears from OpenRouter.** The site is a historical record. A delisted model keeps rendering from `cache.json` (which is keyed by **display name**, not model ID), so it costs nothing to retain. Mark it in place with a `# DELISTED <date>:` comment above its `MODELS` row and leave its `PRICING`, `CATEGORIES`, and `INTEL_DATA` entries intact. The comment exists so a later session doesn't spend a probe call rediscovering it.
  - This also means changing a model's ID is free — it can never invalidate a cached image. When a `:free` variant is retired to paid, repoint the ID **and** fix `PRICING` in the same commit: the Chart keys off the literal string `"Free"` in `PRICING` and snaps such models to `FREE_PRICE_SENTINEL` ($0.01), overriding their real price.

### Checking whether an ID still resolves

Costs nothing — do this before any inference probe. Two signals must agree:

```bash
python3 -c 'import json,urllib.request,sys
for i in sys.argv[1:]:
    d=json.load(urllib.request.urlopen(f"https://openrouter.ai/api/v1/models/{i}/endpoints"))["data"]
    print(i, "DEAD" if not d.get("endpoints") else "OK")' <model-id> ...
```

A delisted model still returns HTTP 200 with full metadata, so check the `endpoints` **array**, not the status code. Cross-check against `GET /api/v1/models`, which lists only models with live endpoints. Neither can see account-level guardrail 404s — only a real call catches those.

## Reasoning effort — always probe, never infer

Every model must generate at the **highest reasoning effort its provider actually exposes**, set via `REASONING_EFFORT_OVERRIDES` in `generate.py`. Determine that value by measurement:

```bash
python3 probe_reasoning_effort.py --screen <model-id> [<model-id> ...]   # cheap default-vs-max triage
python3 probe_reasoning_effort.py <model-id>                             # full ladder
```

Two traps make guessing unreliable:

1. **An unsupported effort value is silently ignored, not rejected.** You get HTTP 200 and near-zero thinking. A clean response is not evidence the effort applied. (Some providers *do* reject — o3/o4-mini 400 on `xhigh`/`max`, exposing only up to `high`.)
2. **Effort labels are not portable between providers.** Measured reasoning tokens for GPT-5.6: Sol `default` 25 → `xhigh` 138 → `max` 4142; Terra 59 → 201 → 5178. `xhigh` is GLM-5.2's maximum but is nearly a no-op on Sol/Terra. Copying another model's label is how a model ends up generating at ~zero reasoning.

A "Thinking"/"Reasoning" name does **not** mean a model reasons by default — Gemini 3.5 Flash Lite and Qwen3 Max Thinking both measured **0** reasoning tokens at their provider default.

Equally, most models have **no** headroom (default ≈ max); don't add overrides reflexively. Adding one also promotes the model to `REASONING_MAX_TOKENS` (96K) and the retry-and-keep-best path, because high effort can starve the visible completion and yield a *worse* SVG — confirmed on GLM-5.2.
