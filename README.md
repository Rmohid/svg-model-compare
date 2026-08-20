# SVG Model Comparison

Compares animated SVG generation across many AI models by sending the same prompt and displaying the results side by side. Non-Anthropic models are generated via OpenRouter; Claude models are generated through the Claude Max account using the local `claude` CLI.

**Live page:** https://rmohid.github.io/svg-model-compare/

## Prompt

> Create an animated SVG image of a pelican riding a bicycle with spinning wheels and pedaling motion.

## Provider policy (2026-08-19)

Only providers with at least one model scoring **AA intelligence ≥ 50** are shown. Providers whose best model is below 50 are pruned entirely. New models from a qualifying provider are added as they ship; results older than ~2 weeks are served from cache (never regenerated).

## Models (97)

| Category | Models |
|----------|--------|
| Anthropic (15) | Opus 5, Opus 5 Fast, Opus 4.8 / Fast, Opus 4.7, Opus 4.6, Opus 4.5, Opus 4.1, Opus 4; Fable 5; Sonnet 5, 4.6, 4.5, 4; Haiku 4.5 |
| OpenAI (16) | GPT-5.6 Sol / Terra / Luna; GPT-5.5 Pro / 5.5 / 5.4; GPT-5.4 Mini / Nano; GPT-5.2 / 5.1 / 5 / 5 Mini; GPT-4.1 / 4.1 Mini; o3, o4 Mini |
| Google (12) | Gemini 3.7 Flash, 3.6 Flash, 3.5 Flash / Flash Lite, 3.1 Flash Lite, 3 Flash, 2.5 Flash; 3.1 Pro, 3 Pro, 2.5 Pro; Gemma 4 31B, 26B |
| xAI / Grok (10) | Grok 4.6, 4.5, 4.3, 4.20 Beta, 4, 4.1 Fast, 4 Fast, 3, 3 Mini, Build 0.1 |
| Chinese (19) | DeepSeek V4 Pro / 0813 / Flash 0731 / V4 Flash / V3.2 / V3.1 / R1; Kimi K3, K2.7 Code, K2.6, K2.5, K2; GLM-5.3, 5.2, 5V-Turbo, 5.1, 5 Turbo, 5 |
| Meta (3) | Muse Spark 1.2; Llama 4 Maverick, Scout |
| Qwen (22) | 3.8 Max, 2.4T A95B, 27B; 3.7 Max / Flash / Plus; 3.6 Max / Plus / Flash / 35B / 27B; 3.5 397B / 122B / 35B / 27B / 9B; 3 Max Thinking; 3 235B / 32B / 14B / 8B; 2.5 7B |

Each card shows the model name, release date, and response time.

## Re-running

Requires an [OpenRouter](https://openrouter.ai/) API key (in the `secrets` vault as `OPENROUTER_911_API_KEY`) for non-Anthropic models, and a signed-in `claude` CLI (Claude Max account) for Claude models. Export `SOPS_AGE_KEY_FILE` so the OpenRouter key can be read from the vault.

```bash
# Regenerate with cached results (only calls new/failed models)
python3 generate.py

# Force full regeneration (only if you really mean it)
rm cache.json
python3 generate.py

# Push updated results
git add -A && git commit -m "Update comparison" && git push
```

## Files

- `generate.py` -- calls models, builds HTML, manages cache
- `cache.json` -- cached SVG results (avoids re-calling successful models; cache is keyed by display name)
- `index.html` -- generated comparison page (served by GitHub Pages)

## Last updated

August 2026