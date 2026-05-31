#!/usr/bin/env python3
"""Download 1m kline history from OKX REST API and save to vnpy database."""

import os
import sys
import time
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests
from dotenv import load_dotenv

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load env
env_path = PROJECT_ROOT / ".env"
if env_path.exists():
    load_dotenv(env_path)

OKX_REST = "https://www.okx.com"

SYMBOLS = [
    ("BTC-USDT-SWAP", "BTCUSDT_SWAP_OKX", "GLOBAL"),
    ("ETH-USDT-SWAP", "ETHUSDT_SWAP_OKX", "GLOBAL"),
    ("SOL-USDT-SWAP", "SOLUSDT_SWAP_OKX", "GLOBAL"),
    ("LINK-USDT-SWAP", "LINKUSDT_SWAP_OKX", "GLOBAL"),
    ("DOGE-USDT-SWAP", "DOGEUSDT_SWAP_OKX", "GLOBAL"),
]

DAYS = 60
BAR_SIZE = "1m"  # OKX uses "1m" for 1-minute bars
LIMIT = 100  # Max bars per request


def okx_request(path: str, params: dict | None = None) -> dict:
    """Make an authenticated OKX REST API request."""
    import base64
    import hmac

    api_key = os.getenv("OKX_API_KEY", "")
    secret = os.getenv("OKX_SECRET_KEY", "")
    passphrase = os.getenv("OKX_PASSPHRASE", "")

    now = datetime.now(timezone.utc)
    ts = now.strftime("%Y-%m-%dT%H:%M:%S.") + f"{now.microsecond // 1000:03d}Z"

    if params:
        query = "?" + "&".join(f"{k}={v}" for k, v in params.items())
    else:
        query = ""

    sign_str = ts + "GET" + path + query
    sign = base64.b64encode(
        hmac.new(secret.encode(), sign_str.encode(), "sha256").digest()
    ).decode()

    headers = {
        "OK-ACCESS-KEY": api_key,
        "OK-ACCESS-SIGN": sign,
        "OK-ACCESS-TIMESTAMP": ts,
        "OK-ACCESS-PASSPHRASE": passphrase,
        "Content-Type": "application/json",
        "x-simulated-trading": "1",
    }

    resp = requests.get(OKX_REST + path + query, headers=headers, timeout=30)
    return resp.json()


def download_bars(inst_id: str, days: int) -> list[tuple]:
    """Download 1m bars for an instrument, return list of (ts_ms, o, h, l, c, vol).

    Returns bars sorted oldest-first.
    """
    end_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    start_ms = int((datetime.now(timezone.utc) - timedelta(days=days)).timestamp() * 1000)

    all_bars = []
    # OKX 'after' returns candles BEFORE this timestamp (descending order)
    after = str(end_ms)

    while True:
        params = {
            "instId": inst_id,
            "bar": BAR_SIZE,
            "limit": str(LIMIT),
            "after": after,
        }

        try:
            data = okx_request("/api/v5/market/candles", params)
        except Exception as e:
            print(f"  Request error: {e}, retrying in 2s...")
            time.sleep(2)
            continue

        if data.get("code") != "0":
            print(f"  API error: {data.get('msg', 'unknown')}")
            break

        candles = data.get("data", [])
        if not candles:
            break

        for c in candles:
            # OKX returns: [ts, o, h, l, c, vol, volCcy, ...]
            ts_ms = int(c[0])
            if ts_ms < start_ms:
                # Reached beyond our desired range, stop after this batch
                all_bars.sort(key=lambda x: x[0])
                return all_bars
            o = float(c[1])
            h = float(c[2])
            l = float(c[3])
            cl = float(c[4])
            vol = float(c[5])
            all_bars.append((ts_ms, o, h, l, cl, vol))

        # Use oldest candle's timestamp - 1ms for next page
        oldest_ts = int(candles[-1][0])
        after = str(oldest_ts - 1)

        print(f"    Fetched {len(candles)} bars, total={len(all_bars)}, oldest_ts={oldest_ts}", end="\r")

        # Rate limiting
        time.sleep(0.15)

    # Sort oldest-first
    all_bars.sort(key=lambda x: x[0])
    return all_bars


def save_to_vnpy_db(inst_id: str, vt_symbol: str, exchange: str, bars: list[tuple]):
    """Save bars to vnpy's SQLite database using peewee models."""
    from vnpy.trader.database import get_database
    from vnpy.trader.object import BarData
    from vnpy.trader.constant import Interval, Exchange

    database = get_database()

    exchange_enum = Exchange.GLOBAL

    bar_objects = []
    for ts_ms, o, h, l, c, vol in bars:
        dt = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
        bar = BarData(
            symbol=vt_symbol,
            exchange=exchange_enum,
            datetime=dt,
            interval=Interval.MINUTE,
            open_price=o,
            high_price=h,
            low_price=l,
            close_price=c,
            volume=vol,
            turnover=0.0,
            open_interest=0.0,
            gateway_name="OKX",
        )
        bar_objects.append(bar)

    # Save in batches
    batch_size = 500
    for i in range(0, len(bar_objects), batch_size):
        batch = bar_objects[i : i + batch_size]
        database.save_bar_data(batch)
        print(f"    Saved {i + len(batch)}/{len(bar_objects)} bars", end="\r")
    print(f"    Saved {len(bar_objects)} bars total")


def main():
    # Initialize vnpy database
    from vnpy.trader.setting import SETTINGS

    SETTINGS["database.name"] = "sqlite"
    SETTINGS["database.database"] = str(PROJECT_ROOT / ".vntrader" / "database.db")

    print(f"Database: {SETTINGS['database.database']}")
    print(f"Downloading {DAYS} days of 1m bars for {len(SYMBOLS)} symbols...")

    for inst_id, vt_symbol, exchange in SYMBOLS:
        print(f"\n[{inst_id}] Downloading...")
        bars = download_bars(inst_id, DAYS)
        print(f"  Downloaded {len(bars)} bars")
        if bars:
            save_to_vnpy_db(inst_id, vt_symbol, exchange, bars)
        time.sleep(0.5)  # Pause between symbols

    print("\nDone!")


if __name__ == "__main__":
    main()
