# LLM Performance vs Cost — 2026-06-29

Snapshot of 50 LLMs comparing **Intelligence Index** (Artificial Analysis composite, higher is better), **blended price** (USD per million tokens, weighted 3·input + 1·output / 4), and **output speed** (tokens/sec). Data sourced from `INTEL_DATA` in `generate.py`. Underlying AA snapshot date: `2026-05-31`.

## Pareto best-value frontier

| Tier | Model | Intel | Blended $/Mtok | Notes |
|---|---|---:|---:|---|
| Entry | Qwen 3.5 9B | 32 | $0.11 | Cheapest priced entry on frontier |
| Budget | DeepSeek V4 Flash | 47 | $0.17 | +15 intel for $0.06/Mtok more; sub-$0.20 budget pick |
| Mid | MiniMax M2.7 | 50 | $0.53 | Sub-$1 model reaching intel=50 |
| Performance | Xiaomi MiMo-V2.5-Pro | 54 | $1.50 | Best value under $2/Mtok; ties Kimi K2.6 at same intel |
| Premium | Qwen 3.7 Max | 57 | $1.88 | ★ New: intel=57 at $1.88 — 2.4× cheaper than prior leader Gemini 3.1 Pro ($4.50/Mtok) |
| Top | Claude Opus 4.8 | 61 | $10.00 | ★ New: highest intelligence on the board (61); prior peak was GPT-5.5 at 60 |

**Notable free options:** **Qwen 3.6 Plus** (intel=50), **Ling 2.6 1T** (intel=34), **Ling 2.6 Flash** (intel=26)

## Highlights

- **New intelligence leader:** Claude Opus 4.8 debuts at intel=61, surpassing the prior peak of 60 (GPT-5.5/GPT-5.5 Pro) and becoming the new Pareto top at $10.00/Mtok.
- **Qwen 3.7 Max reshapes the premium tier:** At $1.88/Mtok (AA), it reaches intel=57 — matching Gemini 3.1 Pro ($4.50) and GPT-5.4 ($5.63) at 2.4–3× lower cost. New entrant since the May 2026 snapshot.
- **Speed champion:** Gemini 3.1 Flash Lite leads at 321 tok/s (intel=34, $0.56/Mtok AA). For high-intelligence tasks (intel≥57), Qwen 3.7 Max is fastest at 189 tok/s.
- **Cheapest Pareto step:** DeepSeek V4 Flash at $0.17/Mtok (intel=47) bests GPT-5.4 Nano ($0.46, intel=44) — 47 vs 44 intel points for less than half the price.
- **Free high-value:** Qwen 3.6 Plus (free on OpenRouter, intel=50) outperforms many $1–2/Mtok paid models; Ling 2.6 1T (intel=34) and Ling 2.6 Flash (intel=26) also free.
- **AA vs OR divergences >50%:** Gemini 2.5 Flash (AA $0.26 est vs OR $0.85, 227% gap — historical data); Llama 4 Scout (AA $0.29 vs OR $0.14, 115%), Nemotron 3 Super (AA $0.41 vs OR $0.20, 105%), Llama 4 Maverick (AA $0.47 vs OR $0.26, 79%) — Meta and NVIDIA models benchmark against higher-priced AA providers than OR routes use.

## Full table (sorted by Intelligence Index, descending)

| Model | Provider | Intel | Blended price (AA, $/Mtok) | OR blended ($/Mtok) | Speed (tok/s) | OR pricing |
|---|---|---:|---:|---:|---:|---|
| Claude Opus 4.8 | Anthropic | 61 | $10.00 | $10.00 | 56 | $5 / $25 /M |
| GPT-5.5 | OpenAI | 60 | $11.25 | $11.25 | — | $5 / $30 /M |
| GPT-5.5 Pro | OpenAI | 60 | $67.50 (est) | $67.50 | — | $30 / $180 /M |
| Qwen 3.7 Max | Alibaba | 57 | $1.88 | $1.88 | 189 | $1.25 / $3.75 /M |
| Gemini 3.1 Pro | Google | 57 | $4.50 | $4.50 | 130 | $2 / $12 /M |
| GPT-5.4 | OpenAI | 57 | $5.63 | $5.62 | 79 | $2.50 / $15 /M |
| Claude Opus 4.7 | Anthropic | 57 | $10.00 | $10.00 | 46 | $5 / $25 /M |
| Gemini 3.5 Flash | Google | 55 | $3.38 | $3.38 | 176 | $1.50 / $9 /M |
| Xiaomi MiMo-V2.5-Pro | Xiaomi | 54 | $1.50 | $1.50 | 60 | $1 / $3 /M |
| Kimi K2.6 | Kimi | 54 | $1.71 | $1.72 | 112 | $0.74 / $4.66 /M |
| DeepSeek V4 Pro | DeepSeek | 52 | $2.17 | $2.17 | 33 | $1.74 / $3.48 /M |
| Claude Sonnet 4.6 | Anthropic | 52 | $6.00 | $6.00 | 51 | $3 / $15 /M |
| GLM-5.2 | Z AI | 51 | $2.15 | $2.15 | — | $1.40 / $4.40 /M |
| MiniMax M2.7 | MiniMax | 50 | $0.53 | $0.52 | 51 | $0.30 / $1.20 /M |
| Qwen 3.6 Plus | Alibaba | 50 | $1.13 | Free | 53 | Free /M |
| GLM-5 | Z AI | 50 | $1.55 | $1.35 | 68 | $0.95 / $2.55 /M |
| GPT-5.4 Mini | OpenAI | 49 | $1.69 | $1.69 | 162 | $0.75 / $4.50 /M |
| Grok 4.20 Beta | xAI | 49 | $3.00 | $3.00 | 163 | $2 / $6 /M |
| Xiaomi MiMo-V2-Pro | Xiaomi | 49 | — | $1.50 | — | $1 / $3 /M |
| DeepSeek V4 Flash | DeepSeek | 47 | $0.17 | $0.18 | 84 | $0.14 / $0.28 /M |
| Gemini 3 Flash | Google | 46 | $1.13 | $1.12 | 176 | $0.50 / $3 /M |
| Qwen 3.5 397B | Alibaba | 45 | $1.35 | $1.29 | 53 | $0.55 / $3.50 /M |
| GPT-5.4 Nano | OpenAI | 44 | $0.46 | $0.46 | 157 | $0.20 / $1.25 /M |
| MiniMax M3 | MiniMax | 44 | $1.05 | $1.05 | 89 | $0.60 / $2.40 /M |
| DeepSeek V3.2 | DeepSeek | 42 | $0.32 | $0.29 | 63 | $0.25 / $0.40 /M |
| Qwen 3.5 122B | Alibaba | 42 | $1.10 | $0.72 | 142 | $0.26 / $2.08 /M |
| Kimi K2.7 Code | Kimi | 42 | $1.23 | $1.23 | — | $0.61 / $3.07 /M |
| Gemini 3 Pro | Google | 41 | $4.50 | $4.50 | — | $2 / $12 /M |
| Gemma 4 31B | Google | 39 | $0.00 | $0.21 | 35 | $0.14 / $0.40 /M |
| Grok 4.1 Fast | xAI | 39 | $0.28 | $0.28 | 142 | $0.20 / $0.50 /M |
| Qwen 3.7 Plus | Alibaba | 39 | $0.56 | $0.56 | 51 | $0.32 / $1.28 /M |
| o3 | OpenAI | 38 | $3.50 | $3.50 | 93 | $2 / $8 /M |
| Kimi K2.5 | Kimi | 37 | $1.20 | $0.89 | 38 | $0.45 / $2.20 /M |
| Claude Haiku 4.5 | Anthropic | 37 | $2.00 | $2.00 | 98 | $1 / $5 /M |
| Nemotron 3 Super | NVIDIA | 36 | $0.41 | $0.20 | 154 | $0.10 / $0.50 /M |
| Gemini 3.1 Flash Lite | Google | 34 | $0.56 | $0.56 | 321 | $0.25 / $1.50 /M |
| Ling 2.6 1T | inclusionAI | 34 | $0.85 | Free | 68 | Free /M |
| Qwen 3.5 9B | Alibaba | 32 | $0.11 | $0.08 | 48 | $0.05 / $0.15 /M |
| Grok 3 Mini | xAI | 32 | $0.35 | $0.35 | 207 | $0.30 / $0.50 /M |
| Gemma 4 26B-A4B | Google | 31 | $0.20 | $0.20 | — | $0.13 / $0.40 /M |
| Qwen 3.5 35B | Alibaba | 31 | $0.69 | $0.45 | 154 | $0.16 / $1.30 /M |
| Gemini 2.5 Flash | Google | 30 | $0.26 (est) | $0.85 | 250 | $0.30 / $2.50 /M |
| DeepSeek V3.2 Speciale | DeepSeek | 29 | — | $0.60 | — | $0.40 / $1.20 /M |
| Mistral Small 4 | Mistral | 28 | $0.26 | $0.26 | 151 | $0.15 / $0.60 /M |
| Ling 2.6 Flash | inclusionAI | 26 | $0.15 | Free | 199 | Free /M |
| Mistral Large 3 | Mistral | 23 | $0.75 | $0.75 | 50 | $0.50 / $1.50 /M |
| Nova Premier | Amazon | 19 | $5.00 | $5.00 | 26 | $2.50 / $12.50 /M |
| Llama 4 Maverick | Meta | 18 | $0.47 | $0.26 | 112 | $0.15 / $0.60 /M |
| Llama 4 Scout | Meta | 14 | $0.29 | $0.14 | 143 | $0.08 / $0.30 /M |
| Command A | Cohere | 13 | $4.38 | $4.38 | 40 | $2.50 / $10 /M |

## Methodology notes

- Intelligence Index is the Artificial Analysis composite; higher is better.
- Blended price = (3 × input + 1 × output) / 4, USD per million tokens.
- AA price comes from `INTEL_DATA`; `(est)` flags estimated entries.
- OR blended price is recomputed from the OpenRouter `$X / $Y /M` pricing string. Discrepancies usually mean OR has a cheaper provider than AA's benchmark, or there's a free variant.
- Speed is output tokens/sec from the model's primary provider on AA.
- Free models are billed at $0 on OpenRouter; they participate in intel comparison but get a sentinel x-coordinate on the value chart.
