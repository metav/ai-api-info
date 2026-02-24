#!/usr/bin/env python3
"""
AI API 聚合器价格采集器 - MVP v0.2
采集 OpenRouter 等聚合器的模型价格 + 延迟数据
"""

import json
import re
import time
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

try:
    import requests
except ImportError:
    print("请先安装 requests: pip install requests")
    exit(1)

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env")
except ImportError:
    pass

try:
    from playwright.sync_api import sync_playwright
    HAS_PLAYWRIGHT = True
except ImportError:
    HAS_PLAYWRIGHT = False

# ============================================================
# 配置
# ============================================================
DB_PATH = Path(__file__).parent.parent / "data" / "prices.db"
OUTPUT_DIR = Path(__file__).parent.parent / "data"

BENCHMARK_MODELS = [
    "gpt-4o", "gpt-4o-mini", "gpt-4.1", "gpt-4.1-mini",
    "claude-sonnet-4-20250514", "claude-opus-4-20250514",
    "claude-3.5-sonnet", "claude-3-haiku",
    "gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.0-flash",
    "deepseek-r1", "deepseek-v3", "deepseek-chat",
    "llama-3.1-405b", "llama-3.1-70b", "llama-3.1-8b",
    "qwen-2.5-72b",
]


# ============================================================
# Playwright 辅助
# ============================================================

class PriceScraper:
    """Shared browser context manager for scraping JS-rendered pricing pages."""

    def __init__(self):
        self._pw = None
        self._browser = None

    @property
    def available(self):
        return HAS_PLAYWRIGHT

    def __enter__(self):
        if not HAS_PLAYWRIGHT:
            return self
        self._pw = sync_playwright().start()
        self._browser = self._pw.chromium.launch(headless=True)
        return self

    def __exit__(self, *exc):
        if self._browser:
            self._browser.close()
        if self._pw:
            self._pw.stop()

    def get_page_content(self, url: str, wait_selector: str = None,
                         wait_ms: int = 5000, click_load_more: bool = False) -> str:
        """Load a URL in a browser page and return the rendered HTML."""
        if not self._browser:
            return ""
        page = self._browser.new_page()
        try:
            page.goto(url, timeout=30000)
            if wait_selector:
                try:
                    page.wait_for_selector(wait_selector, timeout=wait_ms)
                except Exception:
                    pass
            page.wait_for_timeout(wait_ms)
            if click_load_more:
                for _ in range(20):
                    buttons = page.query_selector_all('text=Load More')
                    if not buttons:
                        break
                    for btn in buttons:
                        try:
                            btn.click()
                        except Exception:
                            pass
                    page.wait_for_timeout(2000)
            return page.content()
        finally:
            page.close()


# ============================================================
# 模型名称归一化
# ============================================================

def _normalize_for_match(name: str) -> str:
    """Normalize a model name for fuzzy matching.

    Strips vendor prefixes, common suffixes, date stamps,
    and normalizes separators to produce a canonical lowercase key.
    """
    s = name.lower().strip()
    # Remove known vendor prefixes (e.g. Qwen/, deepseek-ai/, meta-llama/)
    s = re.sub(r'^[a-z0-9_-]+/', '', s)
    # Remove common suffixes
    for suffix in ['-instruct', '-chat', '-turbo', '-preview', '-online', '-free']:
        s = s.replace(suffix, '')
    # Remove date stamps like -20250514, -2025-05-14
    s = re.sub(r'-?\d{4}-?\d{2}-?\d{2}', '', s)
    # Normalize separators: dots, underscores -> hyphens
    s = re.sub(r'[._]', '-', s)
    # Collapse multiple hyphens
    s = re.sub(r'-+', '-', s).strip('-')
    return s


def _match_prices_to_models(models: list[dict], scraped: dict[str, tuple],
                            overrides: dict[str, str] = None) -> int:
    """Match scraped (input_price, output_price) tuples to model dicts.

    ``scraped`` maps scraped display name -> (input_price_per_mtok, output_price_per_mtok).
    ``overrides`` maps model_id -> scraped display name for known mismatches.
    Returns the number of models matched.
    """
    overrides = overrides or {}
    # Build normalized lookup from scraped names
    norm_scraped = {}
    for display_name, prices in scraped.items():
        norm_scraped[_normalize_for_match(display_name)] = prices

    matched = 0
    for m in models:
        # Try explicit override first
        if m["model_id"] in overrides:
            key = _normalize_for_match(overrides[m["model_id"]])
            if key in norm_scraped:
                inp, out = norm_scraped[key]
                m["input_price_per_mtok"] = inp
                m["output_price_per_mtok"] = out
                matched += 1
                continue

        # Fuzzy match on normalized model_id
        key = _normalize_for_match(m["model_id"])
        if key in norm_scraped:
            inp, out = norm_scraped[key]
            m["input_price_per_mtok"] = inp
            m["output_price_per_mtok"] = out
            matched += 1
            continue

        # Also try normalized model_name
        key = _normalize_for_match(m["model_name"])
        if key in norm_scraped:
            inp, out = norm_scraped[key]
            m["input_price_per_mtok"] = inp
            m["output_price_per_mtok"] = out
            matched += 1

    return matched


# ============================================================
# 聚合器采集器
# ============================================================

class OpenRouterCollector:
    """OpenRouter - 采集价格 + 延迟"""
    name = "openrouter"
    MODELS_URL = "https://openrouter.ai/api/v1/models"
    ENDPOINTS_URL = "https://openrouter.ai/api/v1/models/{model_id}/endpoints"

    def fetch_models(self) -> list[dict]:
        resp = requests.get(self.MODELS_URL, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        results = []
        for m in data.get("data", []):
            pricing = m.get("pricing", {})
            input_price = float(pricing.get("prompt", 0)) * 1_000_000
            output_price = float(pricing.get("completion", 0)) * 1_000_000
            results.append({
                "provider": self.name,
                "model_id": m["id"],
                "model_name": m.get("name", m["id"]),
                "input_price_per_mtok": round(input_price, 4),
                "output_price_per_mtok": round(output_price, 4),
                "context_length": m.get("context_length", 0),
                "currency": "USD",
                "latency_ms": None,
                "throughput_tps": None,
                "uptime_pct": None,
            })
        return results

    def enrich_latency(self, models: list[dict]) -> list[dict]:
        """通过 endpoints API 补充延迟数据"""
        print("  📊 采集延迟数据...")
        enriched = 0
        for m in models:
            if m["provider"] != self.name:
                continue
            try:
                url = self.ENDPOINTS_URL.format(model_id=m["model_id"])
                resp = requests.get(url, timeout=10)
                if resp.status_code != 200:
                    continue
                data = resp.json()
                endpoints = data.get("data", {}).get("endpoints", [])
                
                # Aggregate: pick best latency across endpoints
                latencies = []
                throughputs = []
                uptimes = []
                for ep in endpoints:
                    lat = ep.get("latency_last_30m")
                    thr = ep.get("throughput_last_30m")
                    upt = ep.get("uptime_last_30m")
                    if lat is not None:
                        latencies.append(float(lat))
                    if thr is not None:
                        throughputs.append(float(thr))
                    if upt is not None:
                        uptimes.append(float(upt))
                
                if latencies:
                    m["latency_ms"] = round(min(latencies))
                if throughputs:
                    m["throughput_tps"] = round(max(throughputs), 1)
                if uptimes:
                    m["uptime_pct"] = round(max(uptimes), 2)
                
                if latencies or throughputs:
                    enriched += 1
                
                time.sleep(0.1)  # Rate limit
            except Exception:
                continue
        
        print(f"  ✅ 延迟数据补充完成: {enriched} 个模型")
        return models


class TogetherAICollector:
    name = "together_ai"
    API_URL = "https://api.together.xyz/v1/models"

    def fetch_models(self) -> list[dict]:
        api_key = os.environ.get("TOGETHER_API_KEY", "")
        if not api_key:
            print(f"  ⚠️  TOGETHER_API_KEY not set, skipping")
            return []
        resp = requests.get(self.API_URL, headers={"Authorization": f"Bearer {api_key}"}, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        results = []
        for m in (data if isinstance(data, list) else data.get("data", [])):
            pricing = m.get("pricing", {})
            if not pricing:
                continue
            results.append({
                "provider": self.name,
                "model_id": m.get("id", ""),
                "model_name": m.get("display_name", m.get("id", "")),
                "input_price_per_mtok": float(pricing.get("input", 0)) * 1_000_000,
                "output_price_per_mtok": float(pricing.get("output", 0)) * 1_000_000,
                "context_length": m.get("context_length", 0),
                "currency": "USD",
                "latency_ms": None,
                "throughput_tps": None,
                "uptime_pct": None,
            })
        return results


class GroqCollector:
    name = "groq"
    API_URL = "https://api.groq.com/openai/v1/models"

    def fetch_models(self) -> list[dict]:
        api_key = os.environ.get("GROQ_API_KEY", "")
        if not api_key:
            print(f"  ⚠️  GROQ_API_KEY not set, skipping")
            return []
        resp = requests.get(self.API_URL, headers={"Authorization": f"Bearer {api_key}"}, timeout=30)
        resp.raise_for_status()
        results = []
        for m in resp.json().get("data", []):
            results.append({
                "provider": self.name,
                "model_id": m["id"],
                "model_name": m.get("id", ""),
                "input_price_per_mtok": 0,
                "output_price_per_mtok": 0,
                "context_length": m.get("context_window", 0),
                "currency": "USD",
                "latency_ms": None,
                "throughput_tps": None,
                "uptime_pct": None,
            })
        return results


class SiliconFlowCollector:
    """SiliconFlow - API for model list, Playwright for pricing."""
    name = "siliconflow"
    API_URL = "https://api.siliconflow.com/v1/models"
    PRICING_URL = "https://www.siliconflow.com/pricing"

    def fetch_models(self, scraper: PriceScraper = None) -> list[dict]:
        api_key = os.environ.get("SILICONFLOW_API_KEY", "")
        if not api_key:
            print("  ⚠️  SILICONFLOW_API_KEY not set, skipping")
            return []

        resp = requests.get(
            self.API_URL,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        models_data = data if isinstance(data, list) else data.get("data", [])

        results = []
        for m in models_data:
            model_id = m.get("id", "")
            results.append({
                "provider": self.name,
                "model_id": model_id,
                "model_name": model_id,
                "input_price_per_mtok": 0,
                "output_price_per_mtok": 0,
                "context_length": m.get("context_length", 0),
                "currency": "USD",
                "latency_ms": None,
                "throughput_tps": None,
                "uptime_pct": None,
            })

        # Scrape prices if Playwright is available
        if scraper and scraper.available:
            scraped = self._scrape_prices(scraper)
            if scraped:
                matched = _match_prices_to_models(results, scraped)
                print(f"  💰 价格匹配: {matched}/{len(results)} 个模型")
        elif not scraper or not scraper.available:
            print("  ⚠️  Playwright not available, prices will be 0")

        return results

    def _scrape_prices(self, scraper: PriceScraper) -> dict[str, tuple]:
        """Scrape pricing from SiliconFlow's Framer-rendered pricing page.

        Page text nodes follow this pattern per model row:
          ModelName, ContextSize, $, InputPrice, $, OutputPrice, Details
        Prices are already per 1M tokens (USD).
        """
        print("  🌐 Scraping SiliconFlow pricing page...")
        try:
            html = scraper.get_page_content(
                self.PRICING_URL, wait_ms=8000, click_load_more=True,
            )
            if not html:
                return {}

            from html.parser import HTMLParser

            class TextExtractor(HTMLParser):
                def __init__(self):
                    super().__init__()
                    self.texts = []

                def handle_data(self, data):
                    text = data.strip()
                    if text:
                        self.texts.append(text)

            extractor = TextExtractor()
            extractor.feed(html)
            texts = extractor.texts

            prices = {}

            # Scan for pattern: ModelName, ContextSize, $, InputPrice, $, OutputPrice
            i = 0
            while i < len(texts) - 5:
                # Look for "$" followed by a number, then another "$" + number
                if (texts[i] == '$'
                        and re.match(r'^\d+(?:\.\d+)?$', texts[i + 1])
                        and texts[i + 2] == '$'
                        and re.match(r'^\d+(?:\.\d+)?$', texts[i + 3])):
                    inp = float(texts[i + 1])
                    out = float(texts[i + 3])
                    # Walk back to find model name (skip context size like "164K")
                    model_name = None
                    for j in range(i - 1, max(i - 3, -1), -1):
                        if j >= 0 and not re.match(r'^\d+[KMB]?$', texts[j], re.IGNORECASE):
                            model_name = texts[j]
                            break
                    if model_name and model_name not in ('$', 'Input', 'Output', 'Actions', 'Model Name'):
                        prices[model_name] = (round(inp, 4), round(out, 4))
                    i += 4
                    continue
                i += 1

            print(f"  📋 Scraped {len(prices)} model prices from SiliconFlow")
            return prices

        except Exception as e:
            print(f"  ❌ SiliconFlow scraping failed: {e}")
            return {}


class OhMyGPTCollector:
    """OhMyGPT - API for model list, Playwright for pricing."""
    name = "ohmygpt"
    API_URL = "https://api.ohmygpt.com/v1/models"
    PRICING_URL = "https://www.ohmygpt.com/models"

    def fetch_models(self, scraper: PriceScraper = None) -> list[dict]:
        api_key = os.environ.get("OHMYGPT_API_KEY", "")
        if not api_key:
            print("  ⚠️  OHMYGPT_API_KEY not set, skipping")
            return []

        resp = requests.get(
            self.API_URL,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        models_data = data if isinstance(data, list) else data.get("data", [])

        results = []
        for m in models_data:
            model_id = m.get("id", "")
            results.append({
                "provider": self.name,
                "model_id": model_id,
                "model_name": model_id,
                "input_price_per_mtok": 0,
                "output_price_per_mtok": 0,
                "context_length": m.get("context_length", 0),
                "currency": "USD",
                "latency_ms": None,
                "throughput_tps": None,
                "uptime_pct": None,
            })

        # Scrape prices if Playwright is available
        if scraper and scraper.available:
            scraped = self._scrape_prices(scraper)
            if scraped:
                matched = _match_prices_to_models(results, scraped)
                print(f"  💰 价格匹配: {matched}/{len(results)} 个模型")
        elif not scraper or not scraper.available:
            print("  ⚠️  Playwright not available, prices will be 0")

        return results

    def _scrape_prices(self, scraper: PriceScraper) -> dict[str, tuple]:
        """Scrape pricing from OhMyGPT's Next.js-rendered models page.

        Page text nodes follow this pattern per model card:
          DisplayName, model-id, $INPUT, /, $OUTPUT, /M, ctx, SIZE, ...
        Prices are split across separate text nodes.
        """
        print("  🌐 Scraping OhMyGPT pricing page...")
        try:
            html = scraper.get_page_content(self.PRICING_URL, wait_ms=8000)
            if not html:
                return {}

            prices = {}

            from html.parser import HTMLParser

            class TextExtractor(HTMLParser):
                def __init__(self):
                    super().__init__()
                    self.texts = []

                def handle_data(self, data):
                    text = data.strip()
                    if text:
                        self.texts.append(text)

            extractor = TextExtractor()
            extractor.feed(html)
            texts = extractor.texts

            # Scan for pattern: model_id, $INPUT, /, $OUTPUT, /M
            i = 0
            while i < len(texts) - 4:
                text = texts[i]
                # Model IDs: lowercase with hyphens/dots/slashes, e.g. "gpt-5.1", "claude-sonnet-4-6"
                if (re.match(r'^[a-z][a-z0-9._/-]+$', text)
                        and len(text) > 3
                        and i + 4 < len(texts)
                        and re.match(r'^\$[\d.]+$', texts[i + 1])
                        and texts[i + 2] == '/'
                        and re.match(r'^\$[\d.]+$', texts[i + 3])
                        and texts[i + 4] == '/M'):
                    model_id = text
                    inp = float(texts[i + 1][1:])  # strip $
                    out = float(texts[i + 3][1:])  # strip $
                    # Check for ¥ (CNY) — convert to USD
                    if texts[i + 1].startswith('¥') or texts[i + 3].startswith('¥'):
                        inp *= 0.14
                        out *= 0.14
                    prices[model_id] = (round(inp, 4), round(out, 4))
                    i += 5
                    continue
                # Also handle ¥ prices
                if (re.match(r'^[a-z][a-z0-9._/-]+$', text)
                        and len(text) > 3
                        and i + 4 < len(texts)
                        and re.match(r'^[¥][\d.]+$', texts[i + 1])
                        and texts[i + 2] == '/'
                        and re.match(r'^[¥][\d.]+$', texts[i + 3])
                        and texts[i + 4] == '/M'):
                    model_id = text
                    inp = float(texts[i + 1][1:]) * 0.14
                    out = float(texts[i + 3][1:]) * 0.14
                    prices[model_id] = (round(inp, 4), round(out, 4))
                    i += 5
                    continue
                i += 1

            print(f"  📋 Scraped {len(prices)} model prices from OhMyGPT")
            return prices

        except Exception as e:
            print(f"  ❌ OhMyGPT scraping failed: {e}")
            return {}


# ============================================================
# 数据存储
# ============================================================

def init_db(db_path: Path):
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS price_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            provider TEXT NOT NULL,
            model_id TEXT NOT NULL,
            model_name TEXT,
            input_price_per_mtok REAL,
            output_price_per_mtok REAL,
            context_length INTEGER,
            currency TEXT DEFAULT 'USD',
            latency_ms INTEGER,
            throughput_tps REAL,
            uptime_pct REAL
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_provider_model_time
        ON price_snapshots(provider, model_id, timestamp)
    """)
    conn.commit()
    return conn


def save_snapshot(conn, models):
    ts = datetime.now(timezone.utc).isoformat()
    for m in models:
        conn.execute("""
            INSERT INTO price_snapshots
            (timestamp, provider, model_id, model_name, input_price_per_mtok,
             output_price_per_mtok, context_length, currency, latency_ms, throughput_tps, uptime_pct)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            ts, m["provider"], m["model_id"], m["model_name"],
            m["input_price_per_mtok"], m["output_price_per_mtok"],
            m["context_length"], m["currency"],
            m.get("latency_ms"), m.get("throughput_tps"), m.get("uptime_pct")
        ))
    conn.commit()


# ============================================================
# 主程序
# ============================================================

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-latency", action="store_true", help="Skip latency enrichment (faster)")
    parser.add_argument("--skip-scraping", action="store_true", help="Skip Playwright-based price scraping")
    args = parser.parse_args()

    print("🚀 AI API 聚合器价格采集器 v0.3")
    print("=" * 50)

    or_collector = OpenRouterCollector()
    api_collectors = [or_collector, TogetherAICollector(), GroqCollector()]
    scraping_collectors = [SiliconFlowCollector(), OhMyGPTCollector()]

    all_models = []

    # API-only collectors (no scraper needed)
    for collector in api_collectors:
        print(f"\n📡 采集 {collector.name}...")
        try:
            models = collector.fetch_models()
            print(f"  ✅ 获取到 {len(models)} 个模型")
            all_models.extend(models)
        except Exception as e:
            print(f"  ❌ 采集失败: {e}")

    # Scraping collectors (shared browser instance)
    if not args.skip_scraping:
        with PriceScraper() as scraper:
            if not scraper.available:
                print("\n⚠️  Playwright not installed — scraping collectors will have price=0")
                print("   Install with: pip install playwright && playwright install chromium")
            for collector in scraping_collectors:
                print(f"\n📡 采集 {collector.name}...")
                try:
                    models = collector.fetch_models(scraper=scraper)
                    print(f"  ✅ 获取到 {len(models)} 个模型")
                    all_models.extend(models)
                except Exception as e:
                    print(f"  ❌ 采集失败: {e}")
    else:
        print("\n⏭️  Skipping scraping collectors (--skip-scraping)")
        for collector in scraping_collectors:
            print(f"\n📡 采集 {collector.name} (no scraping)...")
            try:
                models = collector.fetch_models(scraper=None)
                print(f"  ✅ 获取到 {len(models)} 个模型")
                all_models.extend(models)
            except Exception as e:
                print(f"  ❌ 采集失败: {e}")

    # Enrich latency for OpenRouter
    if not args.skip_latency:
        or_collector.enrich_latency(all_models)

    print(f"\n📊 总计采集 {len(all_models)} 个模型")
    lat_count = sum(1 for m in all_models if m.get("latency_ms") is not None)
    print(f"   其中 {lat_count} 个有延迟数据")

    # Save
    conn = init_db(DB_PATH)
    save_snapshot(conn, all_models)
    conn.close()

    # Output JSON
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = OUTPUT_DIR / "latest_prices.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "total_models": len(all_models),
            "models": all_models
        }, f, indent=2, ensure_ascii=False)
    print(f"📄 JSON: {json_path}")


if __name__ == "__main__":
    main()
