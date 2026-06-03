---
name: update-cost-intelligence-wiki
description: Update the LLM Wiki "Frontier LLM Cost vs Intelligence" synthesis page from this repo's current Artificial Analysis data. Use after bumping AA_SNAPSHOT_DATE / INTEL_DATA in generate.py, or when the user asks to refresh, regenerate, or sync the cost-vs-intelligence (cost vs quality / price vs intelligence) wiki page.
---

# Update Cost-vs-Intelligence Wiki Page

Regenerates the LLM Wiki synthesis page `wiki/syntheses/llm-cost-vs-intelligence-frontier.md`
from this repo's canonical data and publishes it through the wiki's lint gate.

**Single source of truth:** the `INTEL_DATA` dict and `AA_SNAPSHOT_DATE` in
`generate.py` (the same table that drives the main-page Chart view). The page is
generated from it via `aa_data.py`, so the wiki always matches the chart. Never
hand-edit the wiki page — change `generate.py`, then run this skill.

## Steps

1. **Generate the page.** From the repo root run:
   ```
   python3 .claude/skills/update-cost-intelligence-wiki/build_wiki_page.py
   ```
   It prints the complete page (frontmatter + body) to stdout. No credentials
   needed — it AST-reads `generate.py` and never executes it.

2. **(Recommended) Verify the Mermaid block renders** before publishing:
   ```
   python3 .claude/skills/update-cost-intelligence-wiki/build_wiki_page.py --mermaid > /tmp/cvi.mmd
   npx --yes @mermaid-js/mermaid-cli@11 -i /tmp/cvi.mmd -o /tmp/cvi.svg
   ```
   Require a clean exit (rc=0). If `npx` is rewritten in this environment, run it
   as `rtk proxy npx ...`.

3. **Publish via the wiki tool.** Call `mcp__llm-wiki__wiki_write` with:
   - `pages`: a single `{path, content}` where
     `path = "wiki/syntheses/llm-cost-vs-intelligence-frontier.md"` and
     `content` is the full stdout from step 1.
   - `summary`: e.g. `"Refresh frontier LLM cost-vs-intelligence page to <AA_SNAPSHOT_DATE> snapshot"`.

4. **Report** the returned `committed` sha and `pushed` status to the user.

## Notes / guardrails

- The page uses only established wiki tags (`topic/ai-engineering`,
  `domain/ai-engineering`) and wikilinks that already exist
  (`[[openrouter]]`, `[[metr]]`, `[[simon-willison]]`). Do not introduce a novel
  `topic/*` tag — the lint gate rejects orphan concepts.
- If lint fails (`rolled_back: true`), read `lint.judgment`, fix the generator
  (`build_wiki_page.py`), regenerate, and call `wiki_write` again.
- Mermaid `quadrantChart` constraints the generator already enforces: point
  coordinates must be strictly `< 1.0` (clamped to 0.98); axis/quadrant labels
  must not contain `(`, `)`, `+`, or `:`.
- This skill does **not** touch `cache.json` or regenerate `index.html`; it only
  reads `INTEL_DATA`. Refreshing the gallery is the separate model-update workflow.
