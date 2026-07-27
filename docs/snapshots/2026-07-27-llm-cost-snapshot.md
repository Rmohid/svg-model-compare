# LLM Performance vs Cost — 2026-07-27

Snapshot of 60 LLMs comparing **Intelligence Index** (Artificial Analysis composite, higher is better), **blended price** (USD per million tokens, weighted 3·input + 1·output / 4), and **output speed** (tokens/sec). Data sourced from `INTEL_DATA` in `generate.py`. Underlying AA snapshot date: `2026-05-31`.

## Pareto best-value frontier

| Tier | Model | Intel | Blended $/Mtok | Notes |
|---|---|---:|---:|---|
| Entry | Qwen 3.5 9B | 32 | $0.11 | Cheapest priced entry on frontier |
| Budget | DeepSeek V4 Flash | 47 | $0.17 | +15 intel for $0.06/Mtok more; best sub-$0.20 pick |
| Mid | MiniMax M2.7 | 50 | $0.53 | Sub-$1 model reaching intel=50 |
| Performance | Gemini 3.6 Flash | 55 | $1.16 | ★ New: displaces Xiaomi MiMo-V2.5-Pro; highest intel under $1.50 |
| Premium | Qwen 3.7 Max | 57 | $1.88 | Cheapest way past intel=55; beats Gemini 3.1 Pro ($4.50) at same tier |
| Advanced | Kimi K3 | 62 | $2.31 | ★ New: knocks Grok 4.5, GPT-5.6 Terra, and Claude Opus 4.8 off frontier |
| Frontier | GPT-5.6 Sol | 64 | $11.25 (est) | Closest rival to Claude Fable 5 at 1 intel point below |
| Apex | Claude Fable 5 | 65 | $20.00 (est) | Highest intelligence on the board |

**Notable free options:** **Qwen 3.6 Plus** (intel=50), **Ling 2.6 1T** (intel=34), **Ling 2.6 Flash** (intel=26)

## Highlights

- **Kimi K3 reshapes the upper frontier:** At intel=62 for $2.31/Mtok (AA), it is the highest-intel model under $5 — simultaneously pushing Grok 4.5 (59, $3.00), GPT-5.6 Terra (60, $5.63), and Claude Opus 4.8 (61, $10.00) all off the Pareto frontier in a single addition.
- **Gemini 3.6 Flash enters sub-$1.50 tier:** intel=55 at $1.16/Mtok (AA) unseats Xiaomi MiMo-V2.5-Pro (54, $1.50) — higher intelligence at lower cost. Note: OR lists it at $3.00/Mtok, a 159% divergence from the AA benchmark, likely reflecting a cheaper AA provider route.
- **Top-3 tightest cluster yet:** Claude Fable 5 leads at intel=65 (est, $20/Mtok), GPT-5.6 Sol trails by 1 point at 64 ($11.25 est), and Kimi K3 now reaches 62 for $2.31 — a 3-point gap spanning a 9× price range.
- **Speed standouts:** Gemini 3.1 Flash Lite tops at 321 tok/s (intel=34, $0.56/Mtok); Gemini 3.6 Flash delivers 244 tok/s at intel=55 — fastest model in the 50+ intelligence band. Grok 3 Mini hits 207 tok/s at intel=32 for $0.35.
- **Budget Pareto anchor:** DeepSeek V4 Flash ($0.17/Mtok, intel=47) gains +15 intel vs Qwen 3.5 9B ($0.11) for just $0.06 more — steepest intel-per-dollar jump on the frontier.
- **OR vs AA divergences >50%:** Kimi K3 (AA $2.31 vs OR $6.00, +160%); Gemini 3.6 Flash (AA $1.16 vs OR $3.00, +159%); Gemini 2.5 Flash (AA $0.26 est vs OR $0.85, +227%); Inkling (AA $1.10 vs OR $1.76, +60%); Llama 4 Scout (OR $0.14 vs AA $0.29, OR 52% cheaper). These reflect AA benchmarking against cheaper provider tiers than the default OR route.

## Full table (sorted by Intelligence Index, descending)

| Model | Provider | Intel | Blended price (AA, $/Mtok) | OR blended ($/Mtok) | Speed (tok/s) | OR pricing |
|---|---|---:|---:|---:|---:|---|
| Claude Fable 5 | Anthropic | 65 | $20.00 (est) | $20.00 | — | $10 / $50 /M |
| GPT-5.6 Sol | OpenAI | 64 | $11.25 (est) | $11.25 | — | $5 / $30 /M |
| Kimi K3 | Kimi | 62 | $2.31 | $6.00 | 33 | $3 / $15 /M |
| Claude Opus 4.8 | Anthropic | 61 | $10.00 | $10.00 | 56 | $5 / $25 /M |
| GPT-5.6 Terra | OpenAI | 60 | $5.63 (est) | $5.63 | — | $2.50 / $15 /M |
| GPT-5.5 | OpenAI | 60 | $11.25 | $11.25 | — | $5 / $30 /M |
| GPT-5.5 Pro | OpenAI | 60 | $67.50 (est) | $67.50 | — | $30 / $180 /M |
| Grok 4.5 | xAI | 59 | $3.00 (est) | $3.00 | 90 | $2 / $6 /M |
| Claude Sonnet 5 | Anthropic | 58 | $6.00 (est) | $6.00 | 79 | $3 / $15 /M |
| Qwen 3.7 Max | Alibaba | 57 | $1.88 | $1.88 | 189 | $1.25 / $3.75 /M |
| Gemini 3.1 Pro | Google | 57 | $4.50 | $4.50 | 130 | $2 / $12 /M |
| GPT-5.4 | OpenAI | 57 | $5.63 | $5.63 | 79 | $2.50 / $15 /M |
| Claude Opus 4.7 | Anthropic | 57 | $10.00 | $10.00 | 46 | $5 / $25 /M |
| GPT-5.6 Luna | OpenAI | 56 | $2.25 (est) | $2.25 | — | $1 / $6 /M |
| Gemini 3.6 Flash | Google | 55 | $1.16 | $3.00 | 244 | $1.50 / $7.50 /M |
| Gemini 3.5 Flash | Google | 55 | $3.38 | $3.38 | 176 | $1.50 / $9 /M |
| Xiaomi MiMo-V2.5-Pro | Xiaomi | 54 | $1.50 | $1.50 | 60 | $1 / $3 /M |
| Kimi K2.6 | Kimi | 54 | $1.71 | $1.72 | 112 | $0.74 / $4.66 /M |
| DeepSeek V4 Pro | DeepSeek | 52 | $2.17 | $2.18 | 33 | $1.74 / $3.48 /M |
| Claude Sonnet 4.6 | Anthropic | 52 | $6.00 | $6.00 | 51 | $3 / $15 /M |
| GLM-5.2 | Z AI | 51 | $2.15 | $2.15 | — | $1.40 / $4.40 /M |
| MiniMax M2.7 | MiniMax | 50 | $0.53 | $0.53 | 51 | $0.30 / $1.20 /M |
| Qwen 3.6 Plus | Alibaba | 50 | $1.13 | Free | 53 | Free /M |
| GLM-5 | Z AI | 50 | $1.55 | $1.35 | 68 | $0.95 / $2.55 /M |
| GPT-5.4 Mini | OpenAI | 49 | $1.69 | $1.69 | 162 | $0.75 / $4.50 /M |
| Grok 4.20 Beta | xAI | 49 | $3.00 | $3.00 | 163 | $2 / $6 /M |
| Xiaomi MiMo-V2-Pro | Xiaomi | 49 | — | $1.50 | — | $1 / $3 /M |
| DeepSeek V4 Flash | DeepSeek | 47 | $0.17 | $0.18 | 84 | $0.14 / $0.28 /M |
| Inkling | Thinking Machines | 46 | $1.10 | $1.76 | 65 | $1 / $4.05 /M |
| Gemini 3 Flash | Google | 46 | $1.13 | $1.13 | 176 | $0.50 / $3 /M |
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
| Tencent Hy3 | Tencent | 35 | $0.25 (est) | $0.25 | — | $0.14 / $0.58 /M |
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
