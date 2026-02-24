# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AI API price comparison and performance monitoring platform for API aggregators (OpenRouter, Together AI, Groq, etc.). Python script collects pricing/latency data into SQLite, exports JSON, and a static HTML frontend renders it.

## Commands

```bash
# Fetch prices (requires `pip install requests`)
python3 scripts/fetch_prices.py

# Fetch prices without latency enrichment (faster)
python3 scripts/fetch_prices.py --skip-latency

# Fetch prices without Playwright-based scraping (API-only collectors)
python3 scripts/fetch_prices.py --skip-scraping

# Serve the frontend locally
cd web && python3 -m http.server 8000

# Optional: install Playwright for SiliconFlow/OhMyGPT price scraping
pip install playwright && playwright install chromium
```

## Environment Variables

- `TOGETHER_API_KEY` — required for Together AI data collection
- `GROQ_API_KEY` — required for Groq data collection
- `SILICONFLOW_API_KEY` — required for SiliconFlow data collection
- `OHMYGPT_API_KEY` — required for OhMyGPT data collection
- OpenRouter requires no API key

## Optional Dependencies

- `playwright` — required for scraping SiliconFlow/OhMyGPT pricing pages (JS-rendered). Without it, these collectors still return models but with price=0.

## Architecture

**Data pipeline:** `scripts/fetch_prices.py` → `data/prices.db` (SQLite) + `data/latest_prices.json` → `web/index.html`

**Collector pattern:** Each provider has a class (e.g., `OpenRouterCollector`, `TogetherAICollector`) with a `fetch_models()` method returning a normalized list of dicts. OpenRouter additionally has `enrich_latency()` for performance metrics via its endpoints API. SiliconFlow and OhMyGPT collectors accept an optional `scraper` kwarg for Playwright-based price scraping.

**Pricing normalization:** All prices are converted to USD per million tokens. OpenRouter's API returns per-token prices (multiplied by 1M), Together AI returns per-token prices similarly.

**Frontend:** Single-file vanilla HTML/CSS/JS app (`web/index.html`) — no build step, no framework. Loads `./data/latest_prices.json` (symlinked from `../data`). Features i18n (Chinese/English), filtering, sorting, and benchmark comparison.

**Storage:** SQLite `price_snapshots` table stores timestamped snapshots indexed on `(provider, model_id, timestamp)` for historical tracking.

## Key Files

- `scripts/fetch_prices.py` — all data collection logic and DB schema
- `web/index.html` — entire frontend (HTML + CSS + JS in one file)
- `data/` — generated artifacts (gitignored): `prices.db`, `latest_prices.json`
