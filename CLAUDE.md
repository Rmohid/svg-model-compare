# SVG Model Compare

## Rules
- **Never regenerate cached images.** The `cache.json` file stores SVG outputs from prior API calls. When adding new models, only call the API for models not already in the cache. The `generate.py` script already handles this via its cache mechanism — do not clear or rebuild the cache.
  - *Narrow exception:* a single entry generated at the wrong reasoning effort may be deleted and re-run. Back the old entry up first (max effort can produce a **worse** result — see below), and never bulk-clear.
- When adding a new model, add it to both the `MODELS` list and the appropriate `CATEGORIES` section in `generate.py`, then run the script.

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
