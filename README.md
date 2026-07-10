# SVG Model Comparison

Compares animated SVG generation across many AI models by sending the same prompt and displaying the results side by side. Each model is served by one of four backends, chosen by the model's name:

- **Claude Max CLI** — Anthropic models, via the local `claude` CLI.
- **OpenAI (native)** — the latest GPT models (`OPENAI_DIRECT_MODELS` in `generate.py`), via OpenAI's own API.
- **xAI / Grok (native)** — the latest Grok models (`GROK_DIRECT_MODELS`), via xAI's own API.
- **OpenRouter** — every other model, and any native model whose key is missing at run time (it falls back automatically with a printed notice).

**Live page:** https://rmohid.github.io/svg-model-compare/

## Prompt

> Create an animated SVG image of a pelican riding a bicycle with spinning wheels and pedaling motion.

## Models (34)

| Provider | Models |
|----------|--------|
| Anthropic | Opus 4.6, Sonnet 4.6, Opus 4.5, Haiku 4.5, Opus 4.1, Sonnet 4 |
| OpenAI | GPT-5.2, 5.1, 5, 5 Mini, 4.1, 4.1 Mini |
| Google | Gemini 3.1 Pro, 3 Pro, 3 Flash, 2.5 Pro, 2.5 Flash |
| xAI | Grok 4.1 Fast, 4, 4 Fast, 3, 3 Mini |
| Chinese | DeepSeek V3.2, V3.1, R1 / Kimi K2.5, K2 / MiniMax M2.5 |
| Qwen (size scaling) | 3.5 397B, 3 235B, 32B, 14B, 8B / 2.5 7B |

Each card shows the model name, release date, and response time.

## Re-running

Keys are read from the environment first, then the `secrets` vault (export `SOPS_AGE_KEY_FILE` so the vault can be read):

| Backend | Key | Needed for |
|---------|-----|-----------|
| OpenRouter | `OPENROUTER_911_API_KEY` | all OpenRouter-served models (required) |
| Claude Max | — (signed-in `claude` CLI) | Anthropic models |
| OpenAI | `OPENAI_API_KEY` | the latest GPT models (optional; falls back to OpenRouter) |
| xAI / Grok | `XAI_API_KEY` | the latest Grok models (optional; falls back to OpenRouter) |

```bash
# Route a model to a native backend by exporting its key (or `secrets set` it):
export OPENAI_API_KEY=sk-...     # latest GPT models via OpenAI directly
export XAI_API_KEY=xai-...       # latest Grok models via xAI directly
```

To move a model onto (or off) a native backend, edit `OPENAI_DIRECT_MODELS` / `GROK_DIRECT_MODELS` in `generate.py`. Already-cached models are not re-called, so change the map *and* remove the model from `cache.json` to regenerate it through the new backend.

```bash
# Regenerate with cached results (only calls new/failed models)
python3 generate.py

# Force full regeneration
rm cache.json
python3 generate.py

# Push updated results
git add -A && git commit -m "Update comparison" && git push
```

## Files

- `generate.py` -- calls models, builds HTML, manages cache
- `cache.json` -- cached SVG results (avoids re-calling successful models)
- `index.html` -- generated comparison page (served by GitHub Pages)

## Last updated

July 2026
