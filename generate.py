#!/usr/bin/env python3
"""
Compare animated SVG generation across multiple AI models via OpenRouter.
Caches results to avoid re-calling models that already succeeded.
"""

import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.request import Request, urlopen
from urllib.error import HTTPError

import subprocess
API_KEY = os.environ.get("OPENROUTER_911_API_KEY") or subprocess.check_output(
    ["secrets", "get", "OPENROUTER_911_API_KEY"], text=True
).strip()
API_URL = "https://openrouter.ai/api/v1/chat/completions"
CACHE_PATH = os.path.join(os.path.dirname(__file__), "cache.json")

# Anthropic models are generated through the Claude Max account via the local
# `claude` CLI (-p print mode) instead of OpenRouter. Maps the display name used
# in MODELS -> the CLI `--model` id. Any Claude model NOT listed here still falls
# back to OpenRouter. Reachability probed 2026-07-02 against the Max account:
# claude-fable-5, claude-sonnet-5, opus-4-8, sonnet-4-6, haiku-4-5 all serve;
# opus-4-8-fast does not.
CLAUDE_MAX_MODELS = {
    "Claude Opus 5":     "claude-opus-5",
    "Claude Fable 5":    "claude-fable-5",
    "Claude Sonnet 5":   "claude-sonnet-5",
    "Claude Opus 4.8":   "claude-opus-4-8",
    "Claude Opus 4.7":   "claude-opus-4-7",
    "Claude Opus 4.6":   "claude-opus-4-6",
    "Claude Sonnet 4.6": "claude-sonnet-4-6",
    "Claude Opus 4.5":   "claude-opus-4-5",
    "Claude Sonnet 4.5": "claude-sonnet-4-5",
    "Claude Haiku 4.5":  "claude-haiku-4-5",
}

# Reasoning-capable OpenRouter models where we force the highest available
# thinking effort via the unified `reasoning.effort` param, instead of the
# provider's default. GLM-5.2 supports "high"/"xhigh" only, where "xhigh"
# maps to the model's native max reasoning mode.
REASONING_EFFORT_OVERRIDES = {
    "GLM-5.2": "xhigh",
    "Gemini 3.6 Flash": "high",
    "Inkling": "xhigh",
}

# OpenRouter's unified reasoning.effort can claim up to ~95% of max_tokens for
# thinking (e.g. "xhigh"), starving the visible completion if max_tokens stays
# at the plain default. Confirmed 2026-07-02: GLM-5.2 at xhigh with the default
# 32000 budget produced a 3055-char SVG (3 animations) vs. 9196 chars (11
# animations) at the provider's default effort -- same prompt, worse output,
# 3x slower. Give any model with a forced effort override a much larger ceiling
# so reasoning and completion both have room.
DEFAULT_MAX_TOKENS = 32000
REASONING_MAX_TOKENS = 96000

# Below this length, a reasoning-boosted model's SVG is suspiciously sparse
# compared to what non-boosted models typically produce (3-9k chars). Not a
# hard failure -- some models are legitimately terse -- but worth flagging in
# the cache for review rather than silently accepting.
MIN_REASONING_SVG_CHARS = 4000

# Extra attempts (beyond the first) for reasoning-boosted models whose result
# comes back suspiciously short. Confirmed 2026-07-02: a clean, non-truncated
# GLM-5.2 xhigh response can still land far terser than a typical run -- this
# is generation variance under expensive thinking, not a bug in a single call.
REASONING_RETRIES = 2


def reasoning_effort_for(name):
    """Reasoning/thinking effort used to generate this model's SVG, recorded in the
    cache for provenance. Claude Max models use the CLI's adaptive-thinking default
    (no explicit override configured); OpenRouter models use their override if set,
    else the provider's own default effort."""
    if name in CLAUDE_MAX_MODELS:
        return "adaptive"
    return REASONING_EFFORT_OVERRIDES.get(name, "default")

PROMPT = """Create an animated SVG image of a pelican riding a bicycle.
The pelican should be pedaling and the wheels should be spinning.
Use SVG animations (animate, animateTransform, etc).
Output ONLY the SVG code, nothing else. No markdown fences, no explanation.
Start with <svg and end with </svg>."""

# (display_name, model_id, release_date_str)
MODELS = [
    # --- Anthropic ---
    ("Claude Opus 5", "anthropic/claude-opus-5", "Jul 2026"),
    ("Claude Fable 5", "anthropic/claude-fable-5", "Jun 2026"),
    ("Claude Sonnet 5", "anthropic/claude-sonnet-5", "Jun 2026"),
    ("Claude Opus 4.8", "anthropic/claude-opus-4.8", "May 2026"),
    ("Claude Opus 4.8 Fast", "anthropic/claude-opus-4.8-fast", "May 2026"),
    ("Claude Opus 4.7", "anthropic/claude-opus-4.7", "Apr 2026"),
    ("Claude Opus 4.6", "anthropic/claude-opus-4.6", "Feb 2026"),
    ("Claude Sonnet 4.6", "anthropic/claude-sonnet-4.6", "Feb 2026"),
    ("Claude Opus 4.5", "anthropic/claude-opus-4.5", "Nov 2025"),
    ("Claude Sonnet 4.5", "anthropic/claude-sonnet-4.5", "Sep 2025"),
    ("Claude Haiku 4.5", "anthropic/claude-haiku-4.5", "Oct 2025"),
    ("Claude Opus 4.1", "anthropic/claude-opus-4.1", "Aug 2025"),
    ("Claude Sonnet 4", "anthropic/claude-sonnet-4", "May 2025"),
    ("Claude Opus 4", "anthropic/claude-opus-4", "May 2025"),
    # --- OpenAI ---
    ("GPT-5.6 Sol", "openai/gpt-5.6-sol", "Jul 2026"),
    ("GPT-5.6 Terra", "openai/gpt-5.6-terra", "Jul 2026"),
    ("GPT-5.6 Luna", "openai/gpt-5.6-luna", "Jul 2026"),
    ("GPT-5.5 Pro", "openai/gpt-5.5-pro", "Apr 2026"),
    ("GPT-5.5", "openai/gpt-5.5", "Apr 2026"),
    ("GPT-5.4", "openai/gpt-5.4", "Mar 2026"),
    ("GPT-5.4 Mini", "openai/gpt-5.4-mini", "Mar 2026"),
    ("GPT-5.4 Nano", "openai/gpt-5.4-nano", "Mar 2026"),
    ("GPT-5.2", "openai/gpt-5.2", "Dec 2025"),
    ("GPT-5.1", "openai/gpt-5.1", "Nov 2025"),
    ("GPT-5", "openai/gpt-5", "Jun 2025"),
    ("GPT-5 Mini", "openai/gpt-5-mini", "Jun 2025"),
    ("GPT-4.1", "openai/gpt-4.1", "Apr 2025"),
    ("GPT-4.1 Mini", "openai/gpt-4.1-mini", "Apr 2025"),
    ("o3", "openai/o3", "Apr 2025"),
    ("o4 Mini", "openai/o4-mini", "Apr 2025"),
    # --- Google ---
    ("Gemini 3.1 Pro", "google/gemini-3.1-pro-preview", "Feb 2026"),
    ("Gemini 3.6 Flash", "google/gemini-3.6-flash", "Jul 2026"),
    ("Gemini 3.5 Flash Lite", "google/gemini-3.5-flash-lite", "Jul 2026"),
    ("Gemini 3.5 Flash", "google/gemini-3.5-flash", "May 2026"),
    ("Gemini 3.1 Flash Lite", "google/gemini-3.1-flash-lite-preview", "Mar 2026"),
    ("Gemini 3 Pro", "google/gemini-3-pro-preview", "Nov 2025"),
    ("Gemini 3 Flash", "google/gemini-3-flash-preview", "Dec 2025"),
    ("Gemini 2.5 Pro", "google/gemini-2.5-pro", "Jun 2025"),
    ("Gemini 2.5 Flash", "google/gemini-2.5-flash", "Jun 2025"),
    # --- Google (Open Weight) ---
    ("Gemma 4 31B", "google/gemma-4-31b-it", "Apr 2026"),
    ("Gemma 4 26B-A4B", "google/gemma-4-26b-a4b-it", "Apr 2026"),
    # --- xAI / Grok ---
    ("Grok 4.5", "x-ai/grok-4.5", "Jul 2026"),
    ("Grok 4.3", "x-ai/grok-4.3", "Apr 2026"),
    ("Grok Build 0.1", "x-ai/grok-build-0.1", "May 2026"),
    ("Grok 4.20 Beta", "x-ai/grok-4.20-beta", "Mar 2026"),
    ("Grok 4", "x-ai/grok-4", "Jul 2025"),
    ("Grok 4.1 Fast", "x-ai/grok-4.1-fast", "Nov 2025"),
    ("Grok 4 Fast", "x-ai/grok-4-fast", "Sep 2025"),
    ("Grok 3", "x-ai/grok-3", "Jun 2025"),
    ("Grok 3 Mini", "x-ai/grok-3-mini", "Jun 2025"),                    # 6mo ago fast
    # --- Chinese Models ---
    ("MiniMax M3", "minimax/minimax-m3", "Jun 2026"),
    ("MiniMax M2.7", "minimax/minimax-m2.7", "Mar 2026"),
    ("GLM-5.2", "z-ai/glm-5.2", "Jun 2026"),
    ("GLM-5V-Turbo", "z-ai/glm-5v-turbo", "Apr 2026"),
    ("GLM-5.1", "z-ai/glm-5.1", "Apr 2026"),
    ("GLM-5 Turbo", "z-ai/glm-5-turbo", "Mar 2026"),
    ("Xiaomi MiMo-V2.5-Pro", "xiaomi/mimo-v2.5-pro", "Apr 2026"),
    ("Xiaomi MiMo-V2.5", "xiaomi/mimo-v2.5", "Apr 2026"),
    ("Xiaomi MiMo-V2-Pro", "xiaomi/mimo-v2-pro", "Mar 2026"),
    ("ByteDance Seed 2.0", "bytedance-seed/seed-2.0-lite", "Mar 2026"),
    ("StepFun Step 3.7 Flash", "stepfun/step-3.7-flash", "May 2026"),
    ("Tencent Hy3", "tencent/hy3", "Jul 2026"),
    ("Tencent Hy3 Preview", "tencent/hy3-preview:free", "Apr 2026"),
    ("Ring 2.6 1T", "inclusionai/ring-2.6-1t:free", "May 2026"),
    ("Ling 2.6 1T", "inclusionai/ling-2.6-1t:free", "Apr 2026"),
    ("Ling 2.6 Flash", "inclusionai/ling-2.6-flash:free", "Apr 2026"),
    ("DeepSeek V4 Pro", "deepseek/deepseek-v4-pro", "Apr 2026"),
    ("DeepSeek V4 Flash", "deepseek/deepseek-v4-flash", "Apr 2026"),
    ("DeepSeek V3.2 Speciale", "deepseek/deepseek-v3.2-speciale", "Dec 2025"),
    ("DeepSeek V3.2", "deepseek/deepseek-v3.2", "Oct 2025"),
    ("DeepSeek V3.1", "deepseek/deepseek-chat-v3.1", "Sep 2025"),
    ("DeepSeek R1", "deepseek/deepseek-r1", "Jan 2025"),                 # 6mo ago SOTA
    ("Kimi K3", "moonshotai/kimi-k3", "Jul 2026"),
    ("Kimi K2.7 Code", "moonshotai/kimi-k2.7-code", "Jun 2026"),
    ("Kimi K2.6", "moonshotai/kimi-k2.6", "Apr 2026"),
    ("Kimi K2.5", "moonshotai/kimi-k2.5", "Jan 2026"),
    ("Kimi K2", "moonshotai/kimi-k2", "Jul 2025"),                       # 6mo ago SOTA
    ("MiniMax M2.5", "minimax/minimax-m2.5", "Feb 2026"),
    ("GLM-5", "z-ai/glm-5", "Feb 2026"),
    # --- Meta ---
    ("Llama 4 Maverick", "meta-llama/llama-4-maverick", "Apr 2025"),
    ("Llama 4 Scout", "meta-llama/llama-4-scout", "Apr 2025"),
    # --- NVIDIA ---
    ("Nemotron 3 Super", "nvidia/nemotron-3-super-120b-a12b", "Mar 2026"),
    ("Nemotron 3 Nano", "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free", "Apr 2026"),
    # --- Mistral ---
    ("Mistral Large 3", "mistralai/mistral-large-2512", "Dec 2025"),
    ("Mistral Medium 3.5", "mistralai/mistral-medium-3-5", "Apr 2026"),
    ("Mistral Small 4", "mistralai/mistral-small-2603", "Mar 2026"),
    # --- Amazon ---
    ("Nova Premier", "amazon/nova-premier-v1", "Oct 2025"),
    # --- Cohere ---
    ("Command A", "cohere/command-a", "Mar 2025"),
    # --- IBM ---
    ("IBM Granite 4.1 8B", "ibm-granite/granite-4.1-8b", "Apr 2026"),
    # --- Emerging Labs ---
    ("Sakana Fugu Ultra", "sakana/fugu-ultra", "Jun 2026"),
    ("Nex AGI Nex-N2-Mini", "nex-agi/nex-n2-mini", "Jun 2026"),
    ("Poolside Laguna XS 2.1", "poolside/laguna-xs-2.1", "Jul 2026"),
    ("Inkling", "thinkingmachines/inkling", "Jul 2026"),
    # --- Qwen: full model then smaller quantizations ---
    ("Qwen 3.7 Flash", "qwen/qwen3.7-flash", "Jul 2026"),
    ("Qwen 3.7 Plus", "qwen/qwen3.7-plus", "Jun 2026"),
    ("Qwen 3.7 Max", "qwen/qwen3.7-max", "May 2026"),
    ("Qwen 3.6 Max", "qwen/qwen3.6-max-preview", "Apr 2026"),
    ("Qwen 3.6 Plus", "qwen/qwen3.6-plus:free", "Apr 2026"),
    ("Qwen 3.6 Flash", "qwen/qwen3.6-flash", "Apr 2026"),
    ("Qwen 3.6 35B", "qwen/qwen3.6-35b-a3b", "Apr 2026"),
    ("Qwen 3.6 27B", "qwen/qwen3.6-27b", "Apr 2026"),
    ("Qwen3 Max Thinking", "qwen/qwen3-max-thinking", "Feb 2026"),
    ("Qwen 3.5 397B", "qwen/qwen3.5-397b-a17b", "Feb 2026"),
    ("Qwen 3.5 122B", "qwen/qwen3.5-122b-a10b", "Mar 2026"),
    ("Qwen 3.5 35B", "qwen/qwen3.5-35b-a3b", "Mar 2026"),
    ("Qwen 3.5 27B", "qwen/qwen3.5-27b", "Mar 2026"),
    ("Qwen3 235B (Full)", "qwen/qwen3-235b-a22b", "Apr 2025"),
    ("Qwen3 32B", "qwen/qwen3-32b", "Apr 2025"),
    ("Qwen3 14B", "qwen/qwen3-14b", "Apr 2025"),
    ("Qwen3 8B", "qwen/qwen3-8b", "Apr 2025"),
    ("Qwen 3.5 9B", "qwen/qwen3.5-9b", "Mar 2026"),
    ("Qwen 2.5 7B", "qwen/qwen-2.5-7b-instruct", "Oct 2024"),
]

# Per-model pricing (input / output per million tokens) from OpenRouter
PRICING = {
    "Claude Opus 5": "$5 / $25 /M",
    "Claude Fable 5": "$10 / $50 /M",
    "Claude Sonnet 5": "$3 / $15 /M",
    "Claude Opus 4.8": "$5 / $25 /M",
    "Claude Opus 4.8 Fast": "$10 / $50 /M",
    "Claude Opus 4.7": "$5 / $25 /M",
    "Claude Opus 4.6": "$5 / $25 /M",
    "Claude Sonnet 4.6": "$3 / $15 /M",
    "Claude Opus 4.5": "$5 / $25 /M",
    "Claude Haiku 4.5": "$1 / $5 /M",
    "Claude Sonnet 4": "$3 / $15 /M",
    "Claude Sonnet 4.5": "$3 / $15 /M",
    "Claude Opus 4.1": "$15 / $75 /M",
    "Claude Opus 4": "$15 / $75 /M",
    "GPT-5.6 Sol": "$5 / $30 /M",
    "GPT-5.6 Terra": "$2.50 / $15 /M",
    "GPT-5.6 Luna": "$1 / $6 /M",
    "GPT-5.5 Pro": "$30 / $180 /M",
    "GPT-5.5": "$5 / $30 /M",
    "GPT-5.4": "$2.50 / $15 /M",
    "GPT-5.4 Mini": "$0.75 / $4.50 /M",
    "GPT-5.4 Nano": "$0.20 / $1.25 /M",
    "GPT-5.2": "$1.75 / $14 /M",
    "GPT-5.1": "$1.25 / $10 /M",
    "GPT-5": "$1.25 / $10 /M",
    "GPT-5 Mini": "$0.25 / $2 /M",
    "GPT-4.1": "$2 / $8 /M",
    "GPT-4.1 Mini": "$0.40 / $1.60 /M",
    "o3": "$2 / $8 /M",
    "o4 Mini": "$1.10 / $4.40 /M",
    "Gemini 3.1 Pro": "$2 / $12 /M",
    "Gemini 3.6 Flash": "$1.50 / $7.50 /M",
    "Gemini 3.5 Flash Lite": "$0.30 / $2.50 /M",
    "Gemini 3.5 Flash": "$1.50 / $9 /M",
    "Gemini 3.1 Flash Lite": "$0.25 / $1.50 /M",
    "Gemini 3 Pro": "$2 / $12 /M",
    "Gemini 3 Flash": "$0.50 / $3 /M",
    "Gemini 2.5 Pro": "$1.25 / $10 /M",
    "Gemini 2.5 Flash": "$0.30 / $2.50 /M",
    "Gemma 4 31B": "$0.14 / $0.40 /M",
    "Gemma 4 26B-A4B": "$0.13 / $0.40 /M",
    "Grok 4.5": "$2 / $6 /M",
    "Grok 4.3": "$1.25 / $2.50 /M",
    "Grok Build 0.1": "$1 / $2 /M",
    "Grok 4.20 Beta": "$2 / $6 /M",
    "Grok 4": "$3 / $15 /M",
    "Grok 4.1 Fast": "$0.20 / $0.50 /M",
    "Grok 4 Fast": "$0.20 / $0.50 /M",
    "Grok 3": "$3 / $15 /M",
    "Grok 3 Mini": "$0.30 / $0.50 /M",
    "MiniMax M3": "$0.60 / $2.40 /M",
    "MiniMax M2.7": "$0.30 / $1.20 /M",
    "GLM-5.2": "$1.40 / $4.40 /M",
    "GLM-5V-Turbo": "$1.20 / $4.00 /M",
    "GLM-5.1": "$0.95 / $3.15 /M",
    "GLM-5 Turbo": "$0.96 / $3.20 /M",
    "Xiaomi MiMo-V2.5-Pro": "$1 / $3 /M",
    "Xiaomi MiMo-V2.5": "$0.40 / $2 /M",
    "Xiaomi MiMo-V2-Pro": "$1 / $3 /M",
    "ByteDance Seed 2.0": "$0.25 / $2 /M",
    "StepFun Step 3.7 Flash": "$0.20 / $1.15 /M",
    "Tencent Hy3": "$0.14 / $0.58 /M",
    "Tencent Hy3 Preview": "Free /M",
    "Ring 2.6 1T": "Free /M",
    "Ling 2.6 1T": "Free /M",
    "Ling 2.6 Flash": "Free /M",
    "DeepSeek V4 Pro": "$1.74 / $3.48 /M",
    "DeepSeek V4 Flash": "$0.14 / $0.28 /M",
    "DeepSeek V3.2 Speciale": "$0.40 / $1.20 /M",
    "DeepSeek V3.2": "$0.25 / $0.40 /M",
    "DeepSeek V3.1": "$0.19 / $0.87 /M",
    "DeepSeek R1": "$0.70 / $2.50 /M",
    "Kimi K3": "$3 / $15 /M",
    "Kimi K2.7 Code": "$0.61 / $3.07 /M",
    "Kimi K2.6": "$0.74 / $4.66 /M",
    "Kimi K2.5": "$0.45 / $2.20 /M",
    "Kimi K2": "$0.50 / $2.40 /M",
    "MiniMax M2.5": "$0.30 / $1.10 /M",
    "GLM-5": "$0.95 / $2.55 /M",
    "Llama 4 Maverick": "$0.15 / $0.60 /M",
    "Llama 4 Scout": "$0.08 / $0.30 /M",
    "Nemotron 3 Super": "$0.10 / $0.50 /M",
    "Nemotron 3 Nano": "Free /M",
    "Mistral Large 3": "$0.50 / $1.50 /M",
    "Mistral Medium 3.5": "$1.50 / $7.50 /M",
    "Mistral Small 4": "$0.15 / $0.60 /M",
    "Nova Premier": "$2.50 / $12.50 /M",
    "Command A": "$2.50 / $10 /M",
    "IBM Granite 4.1 8B": "$0.05 / $0.10 /M",
    "Sakana Fugu Ultra": "$5 / $30 /M",
    "Nex AGI Nex-N2-Mini": "$0.025 / $0.10 /M",
    "Poolside Laguna XS 2.1": "$0.06 / $0.12 /M",
    "Inkling": "$1 / $4.05 /M",
    "Qwen 3.7 Flash": "$0.03 / $0.13 /M",
    "Qwen 3.7 Plus": "$0.32 / $1.28 /M",
    "Qwen 3.7 Max": "$1.25 / $3.75 /M",
    "Qwen 3.6 Max": "$1.04 / $6.24 /M",
    "Qwen 3.6 Plus": "Free /M",
    "Qwen 3.6 Flash": "$0.25 / $1.50 /M",
    "Qwen 3.6 35B": "$0.15 / $1 /M",
    "Qwen 3.6 27B": "$0.32 / $3.20 /M",
    "Qwen3 Max Thinking": "$1.20 / $6 /M",
    "Qwen 3.5 397B": "$0.55 / $3.50 /M",
    "Qwen 3.5 122B": "$0.26 / $2.08 /M",
    "Qwen 3.5 35B": "$0.16 / $1.30 /M",
    "Qwen 3.5 27B": "$0.20 / $1.56 /M",
    "Qwen3 235B (Full)": "$0.46 / $1.82 /M",
    "Qwen3 32B": "$0.08 / $0.24 /M",
    "Qwen3 14B": "$0.06 / $0.24 /M",
    "Qwen3 8B": "$0.05 / $0.40 /M",
    "Qwen 3.5 9B": "$0.05 / $0.15 /M",
    "Qwen 2.5 7B": "$0.04 / $0.10 /M",
}

# Intelligence / price / speed data sourced from the Artificial Analysis leaderboard.
# (intel_index, blended_price_usd_per_Mtok, output_tokens_per_sec)
# Models not in this dict are simply not plotted on the Chart view.
AA_SNAPSHOT_DATE = "2026-05-31"
INTEL_DATA = {
    "Claude Opus 5":           (66, 10.00, 54),  # AA raw 61 (max effort), mapped +5 vs Opus 4.8 anchor (raw 56 -> table 61); narrowly tops Fable 5 per AA 2026-07-25
    "Claude Fable 5":          (65, 20.00, None, "est"),  # AA rescaled index mapped onto the 2026-05-31 scale (+4 vs Opus 4.8)
    "Claude Sonnet 5":         (58, 6.00, 79, "est"),     # AA rescaled index mapped onto the 2026-05-31 scale (+6 vs Sonnet 4.6)
    "Claude Opus 4.8":         (61, 10.00, 56),
    "Claude Opus 4.7":         (57, 10.00, 46),
    "Claude Sonnet 4.6":       (52, 6.00, 51),
    "Claude Haiku 4.5":        (37, 2.00, 98),
    "GPT-5.6 Sol":             (64, 11.25, None, "est"),  # AA max-mode index mapped onto the 2026-05-31 scale (1 below Fable 5)
    "GPT-5.6 Terra":           (60, 5.63, None, "est"),   # ~GPT-5.5-class, mapped onto the 2026-05-31 scale
    "GPT-5.6 Luna":            (56, 2.25, None, "est"),   # cheapest GPT-5.6 tier, AA max-mode mapped to the 2026-05-31 scale
    "GPT-5.5":                 (60, 11.25, None),
    "GPT-5.5 Pro":             (60, 67.50, None, "est"),  # Same model tier as GPT-5.5 xhigh; OR-blended pricing
    "GPT-5.4":                 (57, 5.63, 79),
    "GPT-5.4 Mini":            (49, 1.69, 162),
    "GPT-5.4 Nano":            (44, 0.46, 157),
    "o3":                      (38, 3.50, 93),
    "Gemini 3.1 Pro":          (57, 4.50, 130),
    "Gemini 3.6 Flash":        (55, 1.16, 244),  # AA raw 50, mapped +5 vs Opus 4.8 anchor (raw 56 -> table 61)
    "Gemini 3.5 Flash Lite":   (41, 0.85, 350),  # AA raw 36 (high effort), mapped +5 vs Opus 4.8 anchor
    "Gemini 3.5 Flash":        (55, 3.38, 176),
    "Gemini 3.1 Flash Lite":   (34, 0.56, 321),
    "Gemini 3 Pro":            (41, 4.50, None),
    "Gemini 3 Flash":          (46, 1.13, 176),
    "Gemini 2.5 Flash":        (30, 0.26, 250, "est"),  # Historical AA numbers; kept as Google Flash exception
    "Gemma 4 31B":             (39, 0.00, 35),
    "Gemma 4 26B-A4B":         (31, 0.20, None),
    "Grok 4.5":                (59, 3.00, 90, "est"),  # AA rescaled index mapped onto the 2026-05-31 scale (Opus-class, ~2 below Opus 4.8)
    "Grok 4.20 Beta":          (49, 3.00, 163),
    "Grok 4.1 Fast":           (39, 0.28, 142),
    "Grok 3 Mini":             (32, 0.35, 207),
    "MiniMax M3":              (44, 1.05, 89),
    "MiniMax M2.7":            (50, 0.53, 51),
    "DeepSeek V4 Pro":         (52, 2.17, 33),
    "DeepSeek V4 Flash":       (47, 0.17, 84),
    "DeepSeek V3.2":           (42, 0.32, 63),
    "DeepSeek V3.2 Speciale":  (29, None, None),
    "Kimi K3":                 (62, 2.31, 33),  # AA raw 57, mapped +5 vs Opus 4.8 anchor (raw 56 -> table 61)
"Kimi K2.7 Code":          (42, 1.23, None),
    "Kimi K2.6":               (54, 1.71, 112),
    "Kimi K2.5":               (37, 1.20, 38),
    "GLM-5.2":                 (51, 2.15, None),
    "GLM-5":                   (50, 1.55, 68),
    "Tencent Hy3":             (35, 0.25, None, "est"),  # AA 33.6 (current scale) nudged onto the 2026-05-31 scale
    "Xiaomi MiMo-V2-Pro":      (49, None, None),
    "Xiaomi MiMo-V2.5-Pro":    (54, 1.50, 60),
    "Ling 2.6 1T":             (34, 0.85, 68),
    "Ling 2.6 Flash":          (26, 0.15, 199),
    "Llama 4 Maverick":        (18, 0.47, 112),
    "Llama 4 Scout":           (14, 0.29, 143),
    "Nemotron 3 Super":        (36, 0.41, 154),
    "Mistral Large 3":         (23, 0.75, 50),
    "Mistral Small 4":         (28, 0.26, 151),
    "Nova Premier":            (19, 5.00, 26),
    "Command A":               (13, 4.38, 40),
    "Qwen 3.7 Plus":           (39, 0.56, 51),
    "Qwen 3.7 Max":            (57, 1.88, 189),
    "Qwen 3.6 Plus":           (50, 1.13, 53),
    "Qwen 3.5 397B":           (45, 1.35, 53),
    "Qwen 3.5 122B":           (42, 1.10, 142),
    "Qwen 3.5 35B":            (31, 0.69, 154),
    "Qwen 3.5 9B":             (32, 0.11, 48),
    "Inkling":                 (46, 1.10, 65),  # AA raw 41, mapped +5 vs Opus 4.8 anchor (raw 56 -> table 61)
}

# Models marked Free on OpenRouter snap to this x-value so log scale still works.
FREE_PRICE_SENTINEL = 0.01

def model_provider(name):
    if name.startswith("Claude"): return "Anthropic"
    if name.startswith(("GPT", "o3", "o4")): return "OpenAI"
    if name.startswith(("Gemini", "Gemma")): return "Google"
    if name.startswith("Grok"): return "xAI"
    if name.startswith("Llama"): return "Meta"
    if name.startswith("DeepSeek"): return "DeepSeek"
    if name.startswith("Qwen"): return "Alibaba"
    if name.startswith("GLM"): return "Z AI"
    if name.startswith("Mistral"): return "Mistral"
    if name.startswith("MiniMax"): return "MiniMax"
    if name.startswith("Kimi"): return "Kimi"
    if name.startswith("Command"): return "Cohere"
    if name.startswith("Nova"): return "Amazon"
    if name.startswith("Nemotron"): return "NVIDIA"
    if name.startswith("Xiaomi"): return "Xiaomi"
    if name.startswith(("Ling", "Ring")): return "inclusionAI"
    if name.startswith("IBM"): return "IBM"
    if name.startswith("Tencent"): return "Tencent"
    if name.startswith("ByteDance"): return "ByteDance"
    if name.startswith("StepFun"): return "StepFun"
    if name.startswith("Inkling"): return "Thinking Machines"
    return "Other"

PROVIDER_COLORS = {
    # Western vendors — distinctive primary hues
    "Anthropic":   "#D97757",  # terracotta (brand)
    "OpenAI":      "#10A37F",  # jade teal (brand)
    "Google":      "#4285F4",  # Google blue (brand)
    "Meta":        "#8B5CF6",  # violet (differentiated from Google blue)
    "xAI":         "#FFD700",  # gold
    "Amazon":      "#FF9900",  # amazon orange (brand)
    "NVIDIA":      "#76B900",  # nvidia green (brand)
    "Mistral":     "#FA520F",  # vermillion (brand)
    "Cohere":      "#39594D",  # forest green (brand)
    "IBM":         "#0F62FE",  # IBM blue (brand)
    # Chinese & other vendors — no orange to avoid colliding with Anthropic/Amazon
    "Alibaba":     "#C026D3",  # magenta-purple (was orange)
    "DeepSeek":    "#1E3A8A",  # deep navy (distinct from Google's bright blue)
    "Z AI":        "#7B68EE",  # medium purple
    "MiniMax":     "#E91E63",  # hot pink
    "Kimi":        "#A78BFA",  # lavender
    "Xiaomi":      "#DC2626",  # deep red (was orange)
    "inclusionAI": "#00BCD4",  # cyan
    "Tencent":     "#00A1E0",  # sky blue
    "ByteDance":   "#FF3366",  # rose
    "StepFun":     "#14B8A6",  # teal
    "Thinking Machines": "#F59E0B",  # amber
    "Other":       "#888888",
}

# Categories: (vendor_group, [(family_label, [model_names_newest_first]), ...])
# Models in the same family (lineage) share a row in the timeline table.
CATEGORIES = [
    ("Anthropic", [
        ("Claude Fable", ["Claude Fable 5"]),
        ("Claude Opus", ["Claude Opus 5", "Claude Opus 4.8", "Claude Opus 4.8 Fast", "Claude Opus 4.7", "Claude Opus 4.6", "Claude Opus 4.5", "Claude Opus 4.1", "Claude Opus 4"]),
        ("Claude Sonnet", ["Claude Sonnet 5", "Claude Sonnet 4.6", "Claude Sonnet 4.5", "Claude Sonnet 4"]),
        ("Claude Haiku", ["Claude Haiku 4.5"]),
    ]),
    ("OpenAI", [
        ("GPT-5.6", ["GPT-5.6 Sol", "GPT-5.6 Terra", "GPT-5.6 Luna"]),
        ("GPT Pro", ["GPT-5.5 Pro"]),
        ("GPT (flagship)", ["GPT-5.5", "GPT-5.4", "GPT-5.2", "GPT-5.1", "GPT-5", "GPT-4.1"]),
        ("GPT Mini", ["GPT-5.4 Mini", "GPT-5.4 Nano", "GPT-5 Mini", "GPT-4.1 Mini"]),
        ("Reasoning", ["o3", "o4 Mini"]),
    ]),
    ("Google", [
        ("Gemini Pro", ["Gemini 3.1 Pro", "Gemini 3 Pro", "Gemini 2.5 Pro"]),
        ("Gemini Flash", ["Gemini 3.6 Flash", "Gemini 3.5 Flash Lite", "Gemini 3.5 Flash", "Gemini 3.1 Flash Lite", "Gemini 3 Flash", "Gemini 2.5 Flash"]),
        ("Gemma", ["Gemma 4 31B", "Gemma 4 26B-A4B"]),
    ]),
    ("xAI / Grok", [
        ("Grok (flagship)", ["Grok 4.5", "Grok 4.3", "Grok 4.20 Beta", "Grok 4", "Grok 3"]),
        ("Grok Fast", ["Grok 4.1 Fast", "Grok 4 Fast"]),
        ("Grok Build", ["Grok Build 0.1"]),
        ("Grok Mini", ["Grok 3 Mini"]),
    ]),
    ("Chinese Models", [
        ("DeepSeek", ["DeepSeek V4 Pro", "DeepSeek V3.2 Speciale", "DeepSeek V3.2", "DeepSeek V3.1", "DeepSeek R1"]),
        ("DeepSeek Flash", ["DeepSeek V4 Flash"]),
        ("Kimi", ["Kimi K3", "Kimi K2.7 Code", "Kimi K2.6", "Kimi K2.5", "Kimi K2"]),
        ("MiniMax", ["MiniMax M3", "MiniMax M2.7", "MiniMax M2.5"]),
        ("GLM", ["GLM-5.2", "GLM-5V-Turbo", "GLM-5.1", "GLM-5 Turbo", "GLM-5"]),
        ("Xiaomi Pro", ["Xiaomi MiMo-V2.5-Pro", "Xiaomi MiMo-V2-Pro"]),
        ("Xiaomi", ["Xiaomi MiMo-V2.5"]),
        ("ByteDance", ["ByteDance Seed 2.0"]),
        ("StepFun", ["StepFun Step 3.7 Flash"]),
        ("Tencent", ["Tencent Hy3", "Tencent Hy3 Preview"]),
        ("inclusionAI Ling (1T)", ["Ling 2.6 1T"]),
        ("inclusionAI Ling (Flash)", ["Ling 2.6 Flash"]),
        ("inclusionAI Ring", ["Ring 2.6 1T"]),
    ]),
    ("Meta", [
        ("Llama 4", ["Llama 4 Maverick", "Llama 4 Scout"]),
    ]),
    ("NVIDIA", [
        ("Nemotron", ["Nemotron 3 Super"]),
        ("Nemotron Nano", ["Nemotron 3 Nano"]),
    ]),
    ("Mistral", [
        ("Mistral Large", ["Mistral Large 3"]),
        ("Mistral Medium", ["Mistral Medium 3.5"]),
        ("Mistral Small", ["Mistral Small 4"]),
    ]),
    ("Amazon", [
        ("Nova", ["Nova Premier"]),
    ]),
    ("Cohere", [
        ("Command", ["Command A"]),
    ]),
    ("IBM", [
        ("Granite", ["IBM Granite 4.1 8B"]),
    ]),
    ("Sakana AI", [
        ("Fugu", ["Sakana Fugu Ultra"]),
    ]),
    ("Nex AGI", [
        ("Nex-N2", ["Nex AGI Nex-N2-Mini"]),
    ]),
    ("Poolside", [
        ("Laguna", ["Poolside Laguna XS 2.1"]),
    ]),
    ("Thinking Machines", [
        ("Inkling", ["Inkling"]),
    ]),
    ("Qwen", [
        ("Qwen Max", ["Qwen 3.7 Max", "Qwen 3.6 Max", "Qwen3 Max Thinking"]),
        ("Qwen Plus", ["Qwen 3.7 Plus", "Qwen 3.6 Plus"]),
        ("Qwen Flash", ["Qwen 3.7 Flash", "Qwen 3.6 Flash"]),
        ("Qwen Flagship", ["Qwen 3.5 397B", "Qwen 3.5 122B", "Qwen3 235B (Full)"]),
        ("Qwen 35B", ["Qwen 3.6 35B", "Qwen 3.5 35B"]),
        ("Qwen 27-32B", ["Qwen 3.6 27B", "Qwen 3.5 27B", "Qwen3 32B"]),
        ("Qwen 14B", ["Qwen3 14B"]),
        ("Qwen 8-9B", ["Qwen 3.5 9B", "Qwen3 8B"]),
        ("Qwen 7B", ["Qwen 2.5 7B"]),
    ]),
]


def load_cache():
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH) as f:
            return json.load(f)
    return {}


def save_cache(cache):
    with open(CACHE_PATH, "w") as f:
        json.dump(cache, f)


def extract_svg(content):
    """Pull a usable <svg>...</svg> out of a model response. Returns svg or None."""
    # Strip markdown fences if present
    content = re.sub(r"```(?:svg|xml|html)?\s*\n?", "", content)
    content = content.replace("```", "")
    # Match <svg ...> with at least one attribute and a real body (not the prompt's literal example)
    svg_match = re.search(r"(<svg\s[^>]*>[\s\S]+?</svg>)", content, re.IGNORECASE)
    if svg_match and len(svg_match.group(1)) >= 200:
        return svg_match.group(1)
    return None


def _openrouter_attempt(name, model_id, has_reasoning_override):
    """One raw OpenRouter call. Returns (svg, elapsed, error, usage)."""
    start = time.time()
    payload_dict = {
        "model": model_id,
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": REASONING_MAX_TOKENS if has_reasoning_override else DEFAULT_MAX_TOKENS,
        "temperature": 0.7,
    }
    if has_reasoning_override:
        payload_dict["reasoning"] = {"effort": REASONING_EFFORT_OVERRIDES[name]}
    payload = json.dumps(payload_dict).encode()

    req = Request(API_URL, data=payload, method="POST")
    req.add_header("Authorization", f"Bearer {API_KEY}")
    req.add_header("Content-Type", "application/json")
    req.add_header("HTTP-Referer", "https://localhost")

    try:
        with urlopen(req, timeout=600) as resp:
            data = json.loads(resp.read())
        elapsed = time.time() - start
        usage = data.get("usage")
        choice = data["choices"][0]
        if choice.get("finish_reason") == "length":
            return None, elapsed, "Truncated: hit max_tokens before finishing (reasoning likely ate the budget)", usage
        msg = choice["message"]
        content = msg.get("content") or msg.get("reasoning") or ""
        svg = extract_svg(content)
        if svg:
            return svg, elapsed, None, usage
        return None, elapsed, "No usable <svg> in response", usage
    except HTTPError as e:
        elapsed = time.time() - start
        body = e.read().decode() if e.fp else ""
        return None, elapsed, f"HTTP {e.code}: {body[:200]}", None
    except Exception as e:
        elapsed = time.time() - start
        return None, elapsed, str(e), None


def call_openrouter(name, model_id):
    """Call a model via OpenRouter and return (name, svg, elapsed, error, usage).

    Reasoning-boosted models (REASONING_EFFORT_OVERRIDES) are non-deterministic
    under expensive thinking -- the same request can occasionally come back much
    terser than usual with a perfectly clean finish_reason (confirmed 2026-07-02
    on GLM-5.2: one run produced an 18k-reasoning-token response with only ~3k
    tokens of visible completion). Retry up to REASONING_RETRIES extra times
    when the result looks suspiciously short, and keep the best of all attempts
    rather than the first roll.
    """
    has_reasoning_override = name in REASONING_EFFORT_OVERRIDES
    attempts = 1 + REASONING_RETRIES if has_reasoning_override else 1
    best = None  # (svg, elapsed, error, usage)
    total_elapsed = 0.0

    for attempt in range(attempts):
        label = f"attempt {attempt + 1}/{attempts}" if attempts > 1 else "Requesting"
        print(f"  [{name}] {label} via OpenRouter...", flush=True)
        svg, elapsed, error, usage = _openrouter_attempt(name, model_id, has_reasoning_override)
        total_elapsed += elapsed
        if svg and (best is None or len(svg) > len(best[0])):
            best = (svg, elapsed, error, usage)
        if svg and not (has_reasoning_override and len(svg) < MIN_REASONING_SVG_CHARS):
            break  # good result, no need to retry
        if not svg:
            print(f"  [{name}] {label} failed in {elapsed:.1f}s: {error}", flush=True)
        elif attempt < attempts - 1:
            print(f"  [{name}] {label} suspiciously short ({len(svg)} chars), retrying...", flush=True)

    if best:
        svg, _, _, usage = best
        suspect = has_reasoning_override and len(svg) < MIN_REASONING_SVG_CHARS
        flag = " [SUSPECT: short for a max-reasoning model even after retries]" if suspect else ""
        print(f"  [{name}] Done in {total_elapsed:.1f}s ({len(svg)} chars){flag}", flush=True)
        return name, svg, total_elapsed, None, usage
    print(f"  [{name}] Error: all {attempts} attempt(s) failed in {total_elapsed:.1f}s", flush=True)
    return name, None, total_elapsed, error, usage


def call_claude_max(name, cli_model):
    """Generate via the Claude Max account using the local `claude` CLI (-p print mode).
    Used for Anthropic models so the SVGs come from the Max subscription rather than
    OpenRouter. Returns (name, svg, elapsed, error)."""
    print(f"  [{name}] Requesting via Claude Max ({cli_model})...", flush=True)
    start = time.time()
    try:
        proc = subprocess.run(
            ["claude", "-p", PROMPT, "--model", cli_model],
            capture_output=True, text=True, timeout=600,
        )
        elapsed = time.time() - start
        content = proc.stdout or ""
        if proc.returncode != 0:
            err = (proc.stderr or content).strip()[:200] or f"exit {proc.returncode}"
            print(f"  [{name}] Error in {elapsed:.1f}s: {err}", flush=True)
            return name, None, elapsed, err
        # The CLI surfaces gating/errors as a plain-text message on stdout with exit 0.
        svg = extract_svg(content)
        if svg:
            print(f"  [{name}] Done in {elapsed:.1f}s ({len(svg)} chars)", flush=True)
            return name, svg, elapsed, None
        print(f"  [{name}] Done in {elapsed:.1f}s but no usable SVG", flush=True)
        return name, None, elapsed, content.strip()[:200] or "No usable <svg> in response"
    except Exception as e:
        elapsed = time.time() - start
        print(f"  [{name}] Error: {e} in {elapsed:.1f}s", flush=True)
        return name, None, elapsed, str(e)


def call_model(name, model_id):
    """Dispatch a model to its backend and normalize to (name, svg, elapsed, error, usage).
    Anthropic models go through the Claude Max account (`claude` CLI, no usage data);
    everything else uses OpenRouter."""
    if name in CLAUDE_MAX_MODELS:
        return call_claude_max(name, CLAUDE_MAX_MODELS[name]) + (None,)
    return call_openrouter(name, model_id)


def build_html(results, model_dates):
    """Build dual-view HTML: timeline tables + gallery cards with a toggle."""
    from datetime import datetime

    def sort_months(month_set):
        return sorted(month_set, key=lambda d: datetime.strptime(d, "%b %Y"), reverse=True)

    # --- Common columns: union of all months across all models ---
    all_months = set()
    for _, families in CATEGORIES:
        for _, model_names in families:
            for name in model_names:
                if name in model_dates and name in results:
                    all_months.add(model_dates[name])
    global_months = sort_months(all_months)

    # --- Timeline view (single table, all vendors share columns) ---
    num_cols = len(global_months) + 1  # +1 for model-name column
    header = '<tr><th class="corner"></th>'
    for m in global_months:
        header += f"<th>{m}</th>"
    header += "</tr>"

    timeline_rows = []
    for cat_name, families in CATEGORIES:
        has_models = any(
            name in results
            for _, model_names in families
            for name in model_names
        )
        if not has_models:
            continue

        # Vendor group header row
        timeline_rows.append(
            f'<tr><td class="group-header" colspan="{num_cols}">{cat_name}</td></tr>'
        )

        for family_label, model_names in families:
            cells = [f'<td class="model-name">{family_label}</td>']
            for m in global_months:
                matched = [n for n in model_names if model_dates.get(n) == m and n in results]
                if matched:
                    blocks = []
                    for name in matched:
                        svg, elapsed, error = results[name]
                        price = PRICING.get(name, "")
                        price_span = f' <span class="price">{price}</span>' if price else ""
                        if error:
                            body = f'<div class="error">Error: {error}</div>'
                        else:
                            body = f'<div class="svg-container">{svg}</div>'
                        blocks.append(
                            f'<div class="cell-label">{name}'
                            f' <span class="time">{elapsed:.1f}s</span>{price_span}</div>'
                            f'{body}'
                        )
                    cells.append(f'<td class="svg-cell">{"".join(blocks)}</td>')
                else:
                    cells.append('<td class="empty-cell"></td>')
            timeline_rows.append(f'<tr>{"".join(cells)}</tr>')

    timeline_html = f"""
        <div class="table-wrap">
        <table>
        <thead>{header}</thead>
        <tbody>{"".join(timeline_rows)}</tbody>
        </table>
        </div>"""

    # --- Gallery view (mirrors timeline rows: one row per family) ---
    gallery_sections = []
    for cat_name, families in CATEGORIES:
        family_rows = []
        for family_label, model_names in families:
            cards_html = []
            for name in model_names:
                r = results.get(name)
                if not r:
                    continue
                svg, elapsed, error = r
                date = model_dates.get(name, "")
                if error:
                    content = f'<div class="error">Error: {error}</div>'
                else:
                    content = f'<div class="svg-container">{svg}</div>'
                price = PRICING.get(name, "")
                price_html = f'<span class="price">{price}</span>' if price else ""
                cards_html.append(f"""
            <div class="card">
                <div class="card-header">
                    <div>
                        <h3>{name}</h3>
                        <span class="release">Released: {date}</span>
                    </div>
                    <div class="card-meta"><span class="time">{elapsed:.1f}s</span>{price_html}</div>
                </div>
                {content}
            </div>""")
            if cards_html:
                family_rows.append(f"""
        <div class="family-row">
            <div class="family-label">{family_label}</div>
            <div class="grid">{"".join(cards_html)}</div>
        </div>""")
        if family_rows:
            gallery_sections.append(f"""
        <section>
            <h2>{cat_name}</h2>
            {"".join(family_rows)}
        </section>""")

    # --- Chart view (intelligence vs cost scatter) ---
    def openrouter_blended(model_name):
        """Parse '$X / $Y /M' → (3X + Y)/4. Returns None for 'Free' or malformed."""
        s = PRICING.get(model_name, "")
        m = re.match(r"\$([\d.]+)\s*/\s*\$([\d.]+)", s)
        if not m:
            return None
        inp, out = float(m.group(1)), float(m.group(2))
        return (inp * 3 + out) / 4

    # Speed stats exclude None
    speeds_present = [row[2] for row in INTEL_DATA.values() if row[2] is not None]
    max_speed = max(speeds_present) if speeds_present else 1
    default_speed = sum(speeds_present) / len(speeds_present) if speeds_present else 50

    by_provider = {}
    plotted_points = []  # flat list for Pareto
    for name, row in INTEL_DATA.items():
        intel, price, speed = row[0], row[1], row[2]
        est = (len(row) > 3 and row[3] == "est")
        is_free_on_openrouter = "Free" in PRICING.get(name, "")

        # Choose x (price) — AA first, then OR blended fallback, then sentinel for free
        if is_free_on_openrouter:
            x = FREE_PRICE_SENTINEL
            display_free = True
        elif price is not None and price > 0:
            x = price
            display_free = False
        else:
            # AA says 0 or None: use OR blended if we have it
            or_price = openrouter_blended(name)
            if or_price is not None and or_price > 0:
                x = or_price
                display_free = False
            else:
                # No usable price anywhere — skip rather than pretend
                continue

        prov = model_provider(name)
        eff_speed = speed if speed is not None else default_speed
        r = 6 + 22 * (eff_speed / max_speed) ** 0.5
        by_provider.setdefault(prov, []).append({
            "x": x,
            "y": intel,
            "r": r,
            "label": name,
            "price_raw": price if (price is not None and price > 0) else None,
            "speed_raw": speed,
            "free": display_free,
            "est": est,
        })
        plotted_points.append((x, intel))

    plotted = sum(len(v) for v in by_provider.values())

    # Pareto frontier: sorted by price ascending, keeping points that raise the max intel.
    sorted_pts = sorted(plotted_points, key=lambda p: p[0])
    frontier = []
    best_y = -1
    for x, y in sorted_pts:
        if y > best_y:
            frontier.append({"x": x, "y": y})
            best_y = y

    datasets_js = []
    for prov, pts in sorted(by_provider.items(), key=lambda kv: kv[0]):
        color = PROVIDER_COLORS.get(prov, "#888")
        datasets_js.append({
            "label": prov,
            "type": "bubble",
            "data": pts,
            "backgroundColor": color + "CC",
            "borderColor": color,
            "borderWidth": 1.5,
        })

    # Pareto frontier — drawn as a dashed line behind the bubbles.
    frontier_dataset = {
        "label": "Best-value frontier",
        "type": "line",
        "data": frontier,
        "borderColor": "rgba(255,255,255,0.35)",
        "borderDash": [6, 4],
        "borderWidth": 1.5,
        "pointRadius": 0,
        "fill": False,
        "tension": 0,
        "order": 99,
    }
    chart_payload = json.dumps([frontier_dataset] + datasets_js)

    chart_html = f"""
    <div class="chart-meta">
      Data from <a href="https://artificialanalysis.ai/leaderboards/models" target="_blank" rel="noopener">Artificial Analysis</a>
      (snapshot {AA_SNAPSHOT_DATE}).
      X = blended price USD/Mtok (log). Y = intelligence index. Bubble area = output tokens/sec.
      Free-on-OpenRouter models are plotted at $0.01 so they remain visible on a log scale.
      Showing {plotted} of {len(results)} models that have AA intel + price.
    </div>
    <div class="chart-wrap"><canvas id="value-chart"></canvas></div>
    <script>
    document.addEventListener('DOMContentLoaded', function() {{
      var datasets = {chart_payload};
      var ctx = document.getElementById('value-chart').getContext('2d');
      new Chart(ctx, {{
        type: 'bubble',
        data: {{ datasets: datasets }},
        options: {{
          responsive: true,
          maintainAspectRatio: false,
          scales: {{
            x: {{
              type: 'logarithmic',
              title: {{ display: true, text: 'Blended price (USD / Mtok, log)', color: '#ccc' }},
              ticks: {{
                color: '#aaa',
                callback: function(v) {{
                  var allowed = [0.01, 0.1, 0.3, 1, 3, 10, 30, 100];
                  return allowed.indexOf(v) >= 0 ? (v === 0.01 ? 'Free' : '$' + v) : '';
                }}
              }},
              grid: {{ color: 'rgba(255,255,255,0.06)' }}
            }},
            y: {{
              title: {{ display: true, text: 'Intelligence Index (AA)', color: '#ccc' }},
              ticks: {{ color: '#aaa' }},
              grid: {{ color: 'rgba(255,255,255,0.06)' }}
            }}
          }},
          plugins: {{
            legend: {{
              position: 'right',
              labels: {{
                color: '#ddd', usePointStyle: true, boxWidth: 8,
                filter: function(item) {{ return item.text !== 'Best-value frontier'; }}
              }}
            }},
            tooltip: {{
              filter: function(ctx) {{ return ctx.dataset.type !== 'line'; }},
              callbacks: {{
                label: function(ctx) {{
                  var p = ctx.raw;
                  var price = p.free ? 'Free'
                    : (p.price_raw == null ? '$' + p.x.toFixed(2) + '/Mtok (OR)' : '$' + p.price_raw.toFixed(2) + '/Mtok');
                  var sp = p.speed_raw == null ? 'speed: n/a' : p.speed_raw + ' tok/s';
                  var suffix = p.est ? '  (est.)' : '';
                  return p.label + ' · intel ' + p.y + ' · ' + price + ' · ' + sp + suffix;
                }}
              }}
            }}
          }}
        }}
      }});
    }});
    </script>"""

    total = len(results)
    success = sum(1 for v in results.values() if v[2] is None)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Animated SVG Model Comparison - Pelican Riding a Bicycle</title>
<style>
    * {{ margin: 0; padding: 0; box-sizing: border-box; }}
    body {{
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        background: #0a0a0a;
        color: #e0e0e0;
        padding: 2rem;
    }}
    h1 {{
        text-align: center;
        margin-bottom: 0.5rem;
        font-size: 1.8rem;
        color: #fff;
    }}
    .subtitle {{
        text-align: center;
        color: #888;
        margin-bottom: 2rem;
        font-size: 0.9rem;
        line-height: 1.6;
    }}
    /* View toggle */
    .view-controls {{
        text-align: center;
        margin-bottom: 1.5rem;
    }}
    .view-btn {{
        background: #222;
        color: #888;
        border: 1px solid #444;
        padding: 0.5rem 1.5rem;
        font-size: 0.9rem;
        cursor: pointer;
        transition: all 0.2s;
    }}
    .view-btn:first-child {{
        border-radius: 6px 0 0 6px;
    }}
    .view-btn:last-child {{
        border-radius: 0 6px 6px 0;
    }}
    .view-btn.active {{
        background: #444;
        color: #fff;
        border-color: #666;
    }}
    .view-btn:hover:not(.active) {{
        background: #333;
        color: #ccc;
    }}
    /* Shared */
    section {{
        margin-bottom: 2.5rem;
    }}
    h2 {{
        font-size: 1.3rem;
        color: #aaa;
        border-bottom: 1px solid #333;
        padding-bottom: 0.5rem;
        margin-bottom: 1rem;
    }}
    .time {{
        font-size: 0.8rem;
        color: #888;
        background: #2a2a2a;
        padding: 2px 8px;
        border-radius: 4px;
        white-space: nowrap;
    }}
    .svg-container {{
        padding: 1rem;
        display: flex;
        justify-content: center;
        align-items: center;
        min-height: 350px;
        background: #fff;
    }}
    .svg-container svg {{
        width: 100%;
        height: auto;
        max-height: 400px;
    }}
    .error {{
        padding: 1.5rem;
        color: #f44;
        font-size: 0.85rem;
        min-height: 200px;
        display: flex;
        align-items: center;
        justify-content: center;
        text-align: center;
        background: #1a1a1a;
    }}
    /* Timeline view */
    .table-wrap {{
        overflow-x: auto;
    }}
    table {{
        border-collapse: separate;
        border-spacing: 0;
    }}
    thead th {{
        background: #111;
        color: #ccc;
        font-size: 0.85rem;
        font-weight: 600;
        padding: 0.6rem 1rem;
        border-bottom: 2px solid #444;
        white-space: nowrap;
        text-align: center;
        position: sticky;
        top: 0;
        z-index: 2;
    }}
    thead th.corner {{
        min-width: 160px;
        background: #111;
        position: sticky;
        left: 0;
        z-index: 3;
    }}
    td.model-name {{
        background: #141414;
        min-width: 160px;
        max-width: 160px;
        padding: 0.75rem 1rem;
        border-right: 2px solid #333;
        border-bottom: 1px solid #222;
        vertical-align: middle;
        font-weight: 600;
        font-size: 0.9rem;
        color: #fff;
        position: sticky;
        left: 0;
        z-index: 1;
    }}
    td.svg-cell {{
        padding: 0;
        border-bottom: 1px solid #222;
        min-width: 380px;
        vertical-align: top;
    }}
    .cell-label {{
        padding: 0.4rem 0.75rem;
        background: #222;
        font-size: 0.8rem;
        color: #ccc;
        border-bottom: 1px solid #333;
        white-space: nowrap;
    }}
    .cell-label .time {{
        padding: 1px 6px;
        border-radius: 3px;
        font-size: 0.75rem;
        margin-left: 0.5rem;
    }}
    td.empty-cell {{
        background: #0e0e0e;
        border-bottom: 1px solid #222;
        min-width: 0;
        width: 0;
        padding: 0;
    }}
    td.group-header {{
        background: #1a1a1a;
        color: #aaa;
        font-size: 1.1rem;
        font-weight: 700;
        padding: 0.8rem 1rem;
        border-bottom: 2px solid #333;
        border-top: 2px solid #333;
    }}
    /* Gallery view */
    .family-row {{
        margin-bottom: 1.5rem;
    }}
    .family-label {{
        font-size: 0.9rem;
        font-weight: 600;
        color: #ccc;
        padding: 0.4rem 0.75rem;
        background: #1a1a1a;
        border-left: 3px solid #444;
        border-bottom: 1px solid #2a2a2a;
        margin-bottom: 0.75rem;
    }}
    .grid {{
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(380px, 1fr));
        gap: 1.5rem;
    }}
    .card {{
        background: #1a1a1a;
        border: 1px solid #333;
        border-radius: 8px;
        overflow: hidden;
    }}
    .card-header {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.75rem 1rem;
        background: #222;
        border-bottom: 1px solid #333;
    }}
    .card-header h3 {{
        font-size: 0.95rem;
        color: #fff;
    }}
    .release {{
        font-size: 0.75rem;
        color: #666;
    }}
    .card-meta {{
        display: flex;
        gap: 0.5rem;
        align-items: center;
    }}
    .price {{
        font-size: 0.7rem;
        color: #6a6;
        background: #1a2a1a;
        padding: 2px 8px;
        border-radius: 4px;
        white-space: nowrap;
    }}
    /* Chart view */
    .chart-meta {{
        max-width: 980px;
        margin: 0 auto 1rem;
        font-size: 0.85rem;
        color: #aaa;
        line-height: 1.5;
    }}
    .chart-meta a {{ color: #7aa9ff; }}
    .chart-wrap {{
        max-width: 1400px;
        height: 700px;
        margin: 0 auto;
        padding: 1rem;
        background: #141414;
        border: 1px solid #2a2a2a;
        border-radius: 8px;
    }}
</style>
</head>
<body>
<h1>Animated SVG: Pelican Riding a Bicycle</h1>
<p class="subtitle">
    Same prompt sent to {total} models ({success} returned valid SVG)<br>
    Non-Anthropic models via OpenRouter; Claude models via the Claude Max account<br>
    Generated {time.strftime('%Y-%m-%d %H:%M')}
</p>
<div class="view-controls">
    <button class="view-btn active" data-view="chart">Chart</button>
    <button class="view-btn" data-view="gallery">Gallery</button>
    <button class="view-btn" data-view="timeline">Timeline</button>
</div>
<div id="chart-view">
{chart_html}
</div>
<div id="gallery-view" style="display: none;">
{"".join(gallery_sections)}
</div>
<div id="timeline-view" style="display: none;">
{timeline_html}
</div>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<script>
document.addEventListener('DOMContentLoaded', function() {{
    var btns = document.querySelectorAll('.view-btn');
    var views = {{
        chart: document.getElementById('chart-view'),
        timeline: document.getElementById('timeline-view'),
        gallery: document.getElementById('gallery-view')
    }};
    btns.forEach(function(btn) {{
        btn.addEventListener('click', function() {{
            btns.forEach(function(b) {{ b.classList.remove('active'); }});
            btn.classList.add('active');
            var v = btn.getAttribute('data-view');
            Object.keys(views).forEach(function(k) {{
                views[k].style.display = k === v ? '' : 'none';
            }});
        }});
    }});
}});
</script>
</body>
</html>"""
    return html


def main():
    model_dates = {name: date for name, _, date in MODELS}
    model_map = {name: mid for name, mid, _ in MODELS}

    # Load cache of previous successful results
    cache = load_cache()
    results = {}
    to_call = {}
    cached_names = set()

    for name, mid in model_map.items():
        if name in cache and cache[name].get("svg"):
            results[name] = (cache[name]["svg"], cache[name]["elapsed"], None)
            cached_names.add(name)
        else:
            to_call[name] = mid

    print(f"Cache: {len(cached_names)} pelicans already cached, will NOT be re-generated", flush=True)
    if to_call:
        print(f"To call: {len(to_call)} models -> {sorted(to_call.keys())}", flush=True)

    if to_call:
        with ThreadPoolExecutor(max_workers=6) as pool:
            futures = {pool.submit(call_model, name, mid): name for name, mid in to_call.items()}
            for future in as_completed(futures):
                name, svg, elapsed, error, usage = future.result()
                results[name] = (svg, elapsed, error)
                if svg and not error:
                    entry = {"svg": svg, "elapsed": elapsed, "reasoning_effort": reasoning_effort_for(name)}
                    if usage:
                        entry["usage"] = usage
                    if name in REASONING_EFFORT_OVERRIDES and len(svg) < MIN_REASONING_SVG_CHARS:
                        entry["suspect_short"] = True
                        print(f"  [{name}] WARNING: cached anyway, but flagged suspect_short for review", flush=True)
                    cache[name] = entry
                    save_cache(cache)
    else:
        print("All models cached, building HTML...", flush=True)

    print(f"\nBuilding HTML...", flush=True)
    html = build_html(results, model_dates)
    out_path = os.path.join(os.path.dirname(__file__), "index.html")
    with open(out_path, "w") as f:
        f.write(html)
    print(f"Output: {out_path}")

    success = sum(1 for v in results.values() if v[2] is None)
    print(f"\nResults: {success}/{len(results)} models returned valid SVG")
    for name in model_map:
        if name in results:
            svg, elapsed, error = results[name]
            source = "CACHED" if name in cached_names else "api   "
            status = "OK" if not error else f"FAIL: {error[:60]}"
            print(f"  [{source}] {name:25s} {elapsed:6.1f}s  {status}")


if __name__ == "__main__":
    main()
