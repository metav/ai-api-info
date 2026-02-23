#!/usr/bin/env python3
"""
AI API 聚合器价格采集器 - MVP v0.2
采集 OpenRouter 等聚合器的模型价格 + 延迟数据
"""

import json
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
    args = parser.parse_args()

    print("🚀 AI API 聚合器价格采集器 v0.2")
    print("=" * 50)

    or_collector = OpenRouterCollector()
    collectors = [or_collector, TogetherAICollector(), GroqCollector()]

    all_models = []
    for collector in collectors:
        print(f"\n📡 采集 {collector.name}...")
        try:
            models = collector.fetch_models()
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
