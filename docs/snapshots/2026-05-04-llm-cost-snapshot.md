# LLM Performance vs Cost — 2026-05-04

Snapshot of 43 LLMs comparing **Intelligence Index** (Artificial Analysis composite, higher is better), **blended price** (USD per million tokens, weighted 3·input + 1·output / 4), and **output speed** (tokens/sec). Data sourced from `INTEL_DATA` in `generate.py`. Underlying AA snapshot date: `2026-04-24`.

## Pareto best-value frontier

| Tier | Model | Intel | Blended $/Mtok | Notes |
|---|---|---:|---:|---|
| T1 | Gemma 4 31B | 39 | $0.00 | AA benchmarks at $0; OR charges $0.21/Mtok |
| T2 | DeepSeek V4 Flash | 47 | $0.17 | Best value under $1/Mtok at this intelligence tier |
| T3 | MiniMax M2.7 | 50 | $0.53 | Best value under $1/Mtok at this intelligence tier |
| T4 | Xiaomi MiMo-V2.5-Pro | 54 | $1.50 | Cheapest route to intel 54 |
| T5 | Gemini 3.1 Pro | 57 | $4.50 | Cheapest route to intel 57 |
| T6 | GPT-5.5 | 60 | $11.25 | Peak intelligence in this snapshot |

**Notable free options:** Qwen 3.6 Plus (intel 50), Ling 2.6 1T (intel 34), Ling 2.6 Flash (intel 26)

## Highlights

- **Top-intel tier:** GPT-5.5 and GPT-5.5 Pro share the snapshot ceiling at intelligence 60. GPT-5.5 at $11.25/Mtok (AA blended) is the practical option; GPT-5.5 Pro carries an estimated $67.50/Mtok for extended-context tiers.
- **Cheapest path to intel 57:** Gemini 3.1 Pro reaches intelligence 57 at $4.50/Mtok (AA blended), matching GPT-5.4 and Claude Opus 4.7 at lower cost.
- **Fastest model with intel ≥ 30:** Gemini 3.1 Flash Lite leads at 321 tok/s with intelligence 34.
- **Best Pareto value under $1/Mtok:** MiniMax M2.7 delivers intelligence 50 at $0.53/Mtok (AA), the highest intelligence point reachable below $1/Mtok on the best-value frontier.
- **Largest AA vs OR price gap:** Gemini 2.5 Flash shows the starkest divergence—AA logs $0.26/Mtok while OR charges $0.85/Mtok (227% gap). This is flagged `(est)` in AA data, likely reflecting a different benchmark provider tier than current OR rates.
- **Free tier:** Qwen 3.6 Plus (intel 50) is available at $0 on OpenRouter, matching the priced MiniMax M2.7 and GLM-5 at the intelligence-50 level. Ling 2.6 1T (intel 34) and Ling 2.6 Flash (intel 26) also offer free tiers.

## Full table (sorted by Intelligence Index, descending)

| Model | Provider | Intel | Blended price (AA, $/Mtok) | OR blended ($/Mtok) | Speed (tok/s) | OR pricing |
|---|---|---:|---:|---:|---:|---|
| GPT-5.5 | OpenAI | 60 | $11.25 | $11.25 | — | $5 / $30 /M |
| GPT-5.5 Pro | OpenAI | 60 | $67.50 (est) | $67.50 | — | $30 / $180 /M |
| Gemini 3.1 Pro | Google | 57 | $4.50 | $4.50 | 130 | $2 / $12 /M |
| GPT-5.4 | OpenAI | 57 | $5.63 | $5.62 | 79 | $2.50 / $15 /M |
| Claude Opus 4.7 | Anthropic | 57 | $10.00 | $10.00 | 46 | $5 / $25 /M |
| Xiaomi MiMo-V2.5-Pro | Xiaomi | 54 | $1.50 | $1.50 | 60 | $1 / $3 /M |
| Kimi K2.6 | Kimi | 54 | $1.71 | $1.72 | 112 | $0.74 / $4.66 /M |
| DeepSeek V4 Pro | DeepSeek | 52 | $2.17 | $2.17 | 33 | $1.74 / $3.48 /M |
| Claude Sonnet 4.6 | Anthropic | 52 | $6.00 | $6.00 | 51 | $3 / $15 /M |
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
| DeepSeek V3.2 | DeepSeek | 42 | $0.32 | $0.29 | 63 | $0.25 / $0.40 /M |
| Qwen 3.5 122B | Alibaba | 42 | $1.10 | $0.72 | 142 | $0.26 / $2.08 /M |
| Gemini 3 Pro | Google | 41 | $4.50 | $4.50 | — | $2 / $12 /M |
| Gemma 4 31B | Google | 39 | $0.00 | $0.21 | 35 | $0.14 / $0.40 /M |
| Grok 4.1 Fast | xAI | 39 | $0.28 | $0.28 | 142 | $0.20 / $0.50 /M |
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
