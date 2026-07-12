#!/usr/bin/env python3
"""Download recent 1m bars from OKX and save with raw sqlite3."""
import os, sys, time, json, base64, hmac
from datetime import datetime, timezone, timedelta
from pathlib import Path
import requests
from dotenv import load_dotenv

raise SystemExit(
    "DISABLED: this script hardcodes DEMO header "
    "(contamination channel, see reports/MR5M_postmortem.md). "
    "Use download_mainnet_history.py instead."
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # 2026-07 重构批次3：迁入 data_engineering/scripts/，深度 1→2
sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env")

DB = str(PROJECT_ROOT / ".vntrader" / "database.db")
OKX_REST = "https://www.okx.com"

SYMBOLS = [
    ("BTC-USDT-SWAP", "BTCUSDT_SWAP_OKX", "GLOBAL"),
    ("ETH-USDT-SWAP", "ETHUSDT_SWAP_OKX", "GLOBAL"),
    ("SOL-USDT-SWAP", "SOLUSDT_SWAP_OKX", "GLOBAL"),
    ("LINK-USDT-SWAP", "LINKUSDT_SWAP_OKX", "GLOBAL"),
    ("DOGE-USDT-SWAP", "DOGEUSDT_SWAP_OKX", "GLOBAL"),
]

api_key = os.getenv("OKX_API_KEY", "")
secret = os.getenv("OKX_SECRET_KEY", "")
passphrase = os.getenv("OKX_PASSPHRASE", "")


def okx_get(path, params=None):
    now = datetime.now(timezone.utc)
    ts = now.strftime("%Y-%m-%dT%H:%M:%S.") + f"{now.microsecond // 1000:03d}Z"
    q = "?" + "&".join(f"{k}={v}" for k, v in (params or {}).items()) if params else ""
    sign_str = ts + "GET" + path + q
    sign = base64.b64encode(hmac.new(secret.encode(), sign_str.encode(), "sha256").digest()).decode()
    headers = {
        "OK-ACCESS-KEY": api_key, "OK-ACCESS-SIGN": sign,
        "OK-ACCESS-TIMESTAMP": ts, "OK-ACCESS-PASSPHRASE": passphrase,
        "Content-Type": "application/json", "x-simulated-trading": "1",
    }
    return requests.get(OKX_REST + path + q, headers=headers, timeout=30).json()


def download(inst_id, hours=24):
    end_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    start_ms = int((datetime.now(timezone.utc) - timedelta(hours=hours)).timestamp() * 1000)
    bars = []
    after = str(end_ms)
    while True:
        data = okx_get("/api/v5/market/candles", {"instId": inst_id, "bar": "1m", "limit": "100", "after": after})
        if data.get("code") != "0":
            break
        candles = data.get("data", [])
        if not candles:
            break
        for c in candles:
            ts = int(c[0])
            if ts < start_ms:
                return sorted(bars, key=lambda x: x[0])
            bars.append((ts, float(c[1]), float(c[2]), float(c[3]), float(c[4]), float(c[5])))
        after = str(int(candles[-1][0]) - 1)
        time.sleep(0.15)
    return sorted(bars, key=lambda x: x[0])


def save(sqlite_path, symbol, exchange, bars):
    import sqlite3
    conn = sqlite3.connect(sqlite_path)
    for ts, o, h, l, c, vol in bars:
        dt = datetime.fromtimestamp(ts / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        try:
            conn.execute(
                "INSERT OR IGNORE INTO dbbardata (symbol,exchange,datetime,interval,volume,turnover,open_interest,open_price,high_price,low_price,close_price) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (symbol, exchange, dt, "1m", vol, 0.0, 0.0, o, h, l, c),
            )
        except Exception as e:
            print(f"  insert error: {e}")
    conn.commit()
    conn.close()


for inst_id, symbol, exchange in SYMBOLS:
    print(f"[{inst_id}] Downloading...")
    bars = download(inst_id, hours=36)  # get a bit more than 24h to overlap
    print(f"  Got {len(bars)} bars")
    save(DB, symbol, exchange, bars)
    time.sleep(0.5)

print("Done!")
