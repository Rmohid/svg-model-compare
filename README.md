# SVG Model Comparison

Compares animated SVG generation across 34 AI models by sending the same prompt via OpenRouter and displaying the results side by side.

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

Requires an [OpenRouter](https://openrouter.ai/) API key in `~/.config/jobsearch/api_keys.json`.

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

February 2026
