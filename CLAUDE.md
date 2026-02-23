# SVG Model Compare

## Rules
- **Never regenerate cached images.** The `cache.json` file stores SVG outputs from prior API calls. When adding new models, only call the API for models not already in the cache. The `generate.py` script already handles this via its cache mechanism — do not clear or rebuild the cache.
- When adding a new model, add it to both the `MODELS` list and the appropriate `CATEGORIES` section in `generate.py`, then run the script.
