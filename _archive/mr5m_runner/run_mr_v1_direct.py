#!/usr/bin/env python3
"""MR-v1 Direct Runner: OKX WebSocket feed + strategy + REST orders.

Bypasses vnpy gateway — connects directly to OKX DEMO WebSocket for ticker data,
builds 1m/4h bars, runs MrV1Strategy logic, and places orders via REST API.

Deploy: PYTHONUNBUFFERED=1 nohup python -u scripts/run_mr_v1_direct.py > /tmp/mr_v1_direct.log 2>&1 &
"""

from __future__ import annotations

import argparse
import base64
import datetime
import hmac
import json
import os
import signal
import sys
import time
import threading
from collections import defaultdict
from copy import copy
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # 2026-07 重构批次6：迁入 _archive/mr5m_runner/，深度 1→2
sys.path.insert(0, str(PROJECT_ROOT))


# ─── Constants ───────────────────────────────────────────────────────────────

DEMO_WS_URL = "wss://wspap.okx.com:8443/ws/v5/public"
DEMO_REST_URL = "https://www.okx.com"

SYMBOLS = [
    ("BTC-USDT-SWAP", "BTCUSDT", "mr_v1_btc"),
    ("ETH-USDT-SWAP", "ETHUSDT", "mr_v1_eth"),
    ("SOL-USDT-SWAP", "SOLUSDT", "mr_v1_sol"),
    ("LINK-USDT-SWAP", "LINKUSDT", "mr_v1_lin"),
    ("DOGE-USDT-SWAP", "DOGEUSDT", "mr_v1_dog"),
]

# Fast lookup: inst_id → strategy_name
_SYM_NAME = {s[0]: s[2] for s in SYMBOLS}

# Strategy parameters (from MR-v1)
LOOKBACK = 8
ATR_WINDOW = 14
ATR_STOP = 1.0
MAX_HOLD = 60
NOTIONAL_PER_TRADE = 500.0
PRICE_OFFSET = 2

# Contract specs (from OKX API)
CONTRACT_SPECS = {
    "BTC-USDT-SWAP":  {"ctVal": 0.01, "tickSz": 0.1,  "minSz": 0.01},
    "ETH-USDT-SWAP":  {"ctVal": 0.1,  "tickSz": 0.01, "minSz": 0.01},
    "SOL-USDT-SWAP":  {"ctVal": 1.0,  "tickSz": 0.01, "minSz": 0.01},
    "LINK-USDT-SWAP": {"ctVal": 1.0,  "tickSz": 0.001,"minSz": 0.1},
    "DOGE-USDT-SWAP": {"ctVal": 1000, "tickSz": 0.00001, "minSz": 1.0},
}


# ─── OKX REST API helpers ────────────────────────────────────────────────────

def okx_sign(timestamp: str, method: str, path: str, body: str = "") -> str:
    secret = os.getenv("OKX_SECRET_KEY", "")
    sign_str = timestamp + method + path + body
    return base64.b64encode(hmac.new(secret.encode(), sign_str.encode(), "sha256").digest()).decode()


def okx_headers(method: str, path: str, body: str = "") -> dict:
    now = datetime.datetime.now(datetime.timezone.utc)
    ts = now.strftime("%Y-%m-%dT%H:%M:%S.") + f"{now.microsecond // 1000:03d}Z"
    return {
        "OK-ACCESS-KEY": os.getenv("OKX_API_KEY", ""),
        "OK-ACCESS-SIGN": okx_sign(ts, method, path, body),
        "OK-ACCESS-TIMESTAMP": ts,
        "OK-ACCESS-PASSPHRASE": os.getenv("OKX_PASSPHRASE", ""),
        "Content-Type": "application/json",
        "x-simulated-trading": "1",
    }


def okx_post(path: str, body: dict) -> dict:
    body_str = json.dumps(body)
    resp = requests.post(f"{DEMO_REST_URL}{path}", data=body_str,
                         headers=okx_headers("POST", path, body_str), timeout=10)
    return resp.json()


def okx_get(path: str) -> dict:
    resp = requests.get(f"{DEMO_REST_URL}{path}", headers=okx_headers("GET", path), timeout=10)
    return resp.json()


# ─── Bar aggregation ─────────────────────────────────────────────────────────

class BarAggregator:
    """Aggregate ticks into 1m bars, then 4h bars."""
    
    def __init__(self, inst_id: str):
        self.inst_id = inst_id
        self._1m_current: dict | None = None
        self._1m_minute: int = -1
        self._1m_count: int = 0
        
        self._4h_current: dict | None = None
        self._4h_slot: int = -1
        self._4h_count: int = 0
        self._4h_minute_count: int = 0
        
        # History for indicators
        self._4h_history: list[dict] = []
        self._4h_max_keep = 200  # enough for ATR(14) + lookback(8)
        
        # Indicators
        self.atr: float = 0.0
        self.donchian_high: float = 0.0
        self.donchian_low: float = 0.0
        
        # Position tracking
        self.pos: float = 0.0
        self.entry_price: float = 0.0
        self.hold_bars: int = 0
        self.highest_since_entry: float = 0.0
        self.lowest_since_entry: float = float("inf")
        self.active_order: bool = False
        
    def load_history(self, db_path: str, days: int = 90):
        """Load historical 1m bars from vnpy database to warm up indicators."""
        import sqlite3
        base = self.inst_id.replace("-", "").replace("SWAP", "_SWAP_OKX")
        try:
            conn = sqlite3.connect(db_path)
            cutoff = (datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=days)).strftime("%Y-%m-%d")
            rows = conn.execute(
                "SELECT datetime, open_price, high_price, low_price, close_price, volume "
                "FROM dbbardata WHERE symbol=? AND exchange=? AND interval=? AND datetime>=? "
                "ORDER BY datetime",
                (base, "GLOBAL", "1m", cutoff)
            ).fetchall()
            conn.close()
        except Exception as e:
            print(f"  [{self.inst_id}] History load failed: {e}")
            return
        if not rows:
            print(f"  [{self.inst_id}] No history for {base}")
            return
        bars_loaded = 0
        for row in rows:
            dt_str, o, h, l, c, v = row
            dt = datetime.datetime.fromisoformat(dt_str)
            ts_ms = str(int(dt.timestamp() * 1000))
            result_1m, result_4h = self.on_ticker(float(c), float(c), float(c), ts_ms)
            if result_1m:
                bars_loaded += 1
        print(f"  [{self.inst_id}] Loaded {bars_loaded} 1m bars → {self._4h_count} 4h bars", flush=True)
        
    def _compute_atr(self) -> float:
        if len(self._4h_history) < ATR_WINDOW + 1:
            return 0.0
        tr_values = []
        for i in range(-ATR_WINDOW, 0):
            cur = self._4h_history[i]
            prev = self._4h_history[i - 1]
            tr = max(
                cur["high"] - cur["low"],
                abs(cur["high"] - prev["close"]),
                abs(cur["low"] - prev["close"]),
            )
            tr_values.append(tr)
        return sum(tr_values) / len(tr_values)
    
    def _compute_donchian(self):
        if len(self._4h_history) < LOOKBACK:
            self.donchian_high = 0.0
            self.donchian_low = 0.0
            return
        highs = [b["high"] for b in self._4h_history[-LOOKBACK:]]
        lows = [b["low"] for b in self._4h_history[-LOOKBACK:]]
        self.donchian_high = max(highs)
        self.donchian_low = min(lows)
    
    def _add_4h_bar(self, bar: dict):
        self._4h_history.append(bar)
        if len(self._4h_history) > self._4h_max_keep:
            self._4h_history.pop(0)
        self.atr = self._compute_atr()
        self._compute_donchian()
    
    def on_ticker(self, last: float, bid: float, ask: float, ts: str):
        """Process a ticker update. Returns (1m_bar, 4h_bar) if completed, else None."""
        now = datetime.datetime.fromtimestamp(int(ts) / 1000, tz=datetime.timezone.utc)
        minute = now.minute
        hour_slot = now.hour // 4
        
        result_1m = None
        result_4h = None
        
        # 1m bar
        if self._1m_current is None or minute != self._1m_minute:
            if self._1m_current is not None:
                result_1m = self._1m_current
            self._1m_current = {"open": last, "high": last, "low": last, "close": last,
                               "volume": 0, "ts": ts}
            self._1m_minute = minute
            self._1m_count += 1
        else:
            bar = self._1m_current
            bar["high"] = max(bar["high"], last)
            bar["low"] = min(bar["low"], last)
            bar["close"] = last
        
        # 4h bar
        if result_1m is not None:
            if self._4h_current is None or hour_slot != self._4h_slot:
                if self._4h_current is not None and self._4h_minute_count >= 180:
                    result_4h = dict(self._4h_current)
                    self._4h_count += 1
                    self._add_4h_bar(result_4h)
                self._4h_current = dict(result_1m)
                self._4h_slot = hour_slot
                self._4h_minute_count = 1
            else:
                bar_4h = self._4h_current
                bar_4h["high"] = max(bar_4h["high"], result_1m["high"])
                bar_4h["low"] = min(bar_4h["low"], result_1m["low"])
                bar_4h["close"] = result_1m["close"]
                self._4h_minute_count += 1
        
        return result_1m, result_4h


# ─── Order execution ─────────────────────────────────────────────────────────

def calc_size(inst_id: str, price: float) -> int:
    spec = CONTRACT_SPECS[inst_id]
    multiplier = spec["ctVal"]
    contract_value = price * multiplier
    size = round(NOTIONAL_PER_TRADE / contract_value)
    return max(1, min(size, 1000))


def round_price(inst_id: str, price: float) -> float:
    tick = CONTRACT_SPECS[inst_id]["tickSz"]
    return round(price / tick) * tick


def place_order(inst_id: str, side: str, price: float, sz: int) -> dict | None:
    """Place a limit order via OKX REST API. side: 'buy' or 'sell'."""
    px = round_price(inst_id, price)
    body = {
        "instId": inst_id,
        "tdMode": "cross",
        "side": side,
        "ordType": "limit",
        "px": str(px),
        "sz": str(sz),
    }
    result = okx_post("/api/v5/trade/order", body)
    return result


def set_leverage(inst_id: str, leverage: int = 1):
    body = {"instId": inst_id, "lever": str(leverage), "mgnMode": "cross"}
    result = okx_post("/api/v5/account/set-leverage", body)
    if result.get("code") == "0":
        print(f"  Leverage set: {inst_id} → {leverage}x")
    else:
        print(f"  Leverage FAILED {inst_id}: {result.get('msg', 'unknown')}")


# ─── Strategy logic ──────────────────────────────────────────────────────────

def check_entry(agg: BarAggregator, bar: dict) -> tuple[str | None, float | None]:
    """Check MR-v1 entry signal. Returns (side, stop_price) or (None, None)."""
    if agg.pos != 0 or agg.active_order:
        return None, None
    
    close = bar["close"]
    if close <= 0 or agg.atr <= 0:
        return None, None
    
    long_breakout = close > agg.donchian_high > 0
    short_breakout = close < agg.donchian_low > 0
    
    if long_breakout:
        return "short", close + ATR_STOP * agg.atr  # fade: short on long breakout
    elif short_breakout:
        return "long", close - ATR_STOP * agg.atr   # fade: long on short breakout
    
    return None, None


def check_exit(agg: BarAggregator, bar: dict) -> tuple[bool, str]:
    """Check MR-v1 exit signal. Returns (should_exit, reason)."""
    if agg.pos == 0:
        return False, ""
    
    agg.hold_bars += 1
    agg.highest_since_entry = max(agg.highest_since_entry, bar["high"])
    agg.lowest_since_entry = min(agg.lowest_since_entry, bar["low"])
    
    # Stop loss
    if agg.pos > 0:
        stop_price = agg.entry_price - ATR_STOP * agg.atr
        if bar["low"] <= stop_price:
            return True, "stop"
    else:
        stop_price = agg.entry_price + ATR_STOP * agg.atr
        if bar["high"] >= stop_price:
            return True, "stop"
    
    # Max hold
    if agg.hold_bars >= MAX_HOLD:
        return True, "max_hold"
    
    return False, ""


def reset_position(agg: BarAggregator):
    agg.pos = 0.0
    agg.entry_price = 0.0
    agg.hold_bars = 0
    agg.highest_since_entry = 0.0
    agg.lowest_since_entry = float("inf")
    agg.active_order = False


# ─── WebSocket client ────────────────────────────────────────────────────────

class OKXTickerFeed:
    """WebSocket feed for OKX ticker data."""
    
    def __init__(self):
        self.ws = None
        self.running = False
        self.aggregators: dict[str, BarAggregator] = {}
        self._1m_counters: dict[str, int] = defaultdict(int)
        
        for inst_id, _, name in SYMBOLS:
            self.aggregators[inst_id] = BarAggregator(inst_id)
    
    def _connect_ws(self):
        import websocket
        self.ws = websocket.WebSocket()
        self.ws.connect(DEMO_WS_URL)
        self.ws.settimeout(10)  # 10s timeout for recv
        print(f"WebSocket connected to {DEMO_WS_URL}")
    
    def _subscribe(self):
        args = []
        for inst_id, _, _ in SYMBOLS:
            args.append({"channel": "tickers", "instId": inst_id})
        sub_msg = json.dumps({"op": "subscribe", "args": args})
        self.ws.send(sub_msg)
        print(f"Subscribed to tickers for {len(SYMBOLS)} symbols")
    
    def run(self):
        import websocket
        
        self.running = True
        reconnect_delay = 1
        
        while self.running:
            try:
                self._connect_ws()
                self._subscribe()
                reconnect_delay = 1
                
                while self.running:
                    try:
                        msg = self.ws.recv()
                        if msg == "pong":
                            continue
                        self._on_message(msg)
                    except websocket.WebSocketTimeoutException:
                        # Send ping
                        try:
                            self.ws.send("ping")
                        except Exception:
                            break
                    except Exception as e:
                        print(f"WS recv error: {type(e).__name__}: {e}", flush=True)
                        break
            except Exception as e:
                print(f"WS connection error: {e}, reconnecting in {reconnect_delay}s...", flush=True)
                time.sleep(reconnect_delay)
                reconnect_delay = min(reconnect_delay * 2, 60)
            finally:
                try:
                    self.ws.close()
                except Exception:
                    pass
    
    def _on_message(self, msg: str):
        try:
            data = json.loads(msg)
        except json.JSONDecodeError:
            return
        
        if "data" not in data or "arg" not in data:
            return
        
        channel = data.get("arg", {}).get("channel", "")
        if channel != "tickers":
            return
        
        for d in data["data"]:
            inst_id = d.get("instId", "")
            agg = self.aggregators.get(inst_id)
            if not agg:
                continue
            
            last = float(d.get("last", 0))
            bid = float(d.get("bidPx", 0))
            ask = float(d.get("askPx", 0))
            ts = d.get("ts", "0")
            
            result_1m, result_4h = agg.on_ticker(last, bid, ask, ts)
            
            # Print 1m bar heartbeat (every 10 bars)
            if result_1m is not None:
                self._1m_counters[inst_id] += 1
                cnt = self._1m_counters[inst_id]
                if cnt == 1 or cnt % 10 == 0:
                    name = _SYM_NAME[inst_id]
                    print(f"[{name}] 1m #{cnt} | c={result_1m['close']}", flush=True)
            
            # Process 4h bar
            if result_4h is not None:
                name = _SYM_NAME[inst_id]
                self._on_4h_bar(inst_id, agg, result_4h, name)
    
    def _on_4h_bar(self, inst_id: str, agg: BarAggregator, bar: dict, name: str):
        """Strategy logic on completed 4h bar."""
        close = bar["close"]
        print(
            f"[{name}] 4h #{agg._4h_count} | c={close:.4f} "
            f"ATR={agg.atr:.4f} DH={agg.donchian_high:.4f} DL={agg.donchian_low:.4f} "
            f"pos={agg.pos}",
            flush=True,
        )
        
        # Check exit
        should_exit, reason = check_exit(agg, bar)
        if should_exit:
            side = "sell" if agg.pos > 0 else "buy"
            exit_price = bar["close"]
            sz = abs(int(agg.pos))
            print(f"[{name}] EXIT {reason} | side={side} px={exit_price} sz={sz}", flush=True)
            result = place_order(inst_id, side, exit_price, sz)
            if result and result.get("code") == "0":
                print(f"[{name}] EXIT order placed: {result['data'][0].get('ordId', '?')[:8]}", flush=True)
                reset_position(agg)
            else:
                print(f"[{name}] EXIT FAILED: {result}", flush=True)
            return
        
        # Check entry
        side, stop_price = check_entry(agg, bar)
        if side is not None:
            price = bar["close"] + PRICE_OFFSET * CONTRACT_SPECS[inst_id]["tickSz"] if side == "buy" else bar["close"] - PRICE_OFFSET * CONTRACT_SPECS[inst_id]["tickSz"]
            sz = calc_size(inst_id, bar["close"])
            okx_side = "buy" if side == "long" else "sell"
            print(f"[{name}] ENTRY {side} | px={price} sz={sz} stop={stop_price:.4f}", flush=True)
            result = place_order(inst_id, okx_side, price, sz)
            if result and result.get("code") == "0":
                print(f"[{name}] ENTRY order placed: {result['data'][0].get('ordId', '?')[:8]}", flush=True)
                agg.active_order = True
                # In a real system, we'd wait for order fill confirmation
                # For now, assume fill at bar close
                agg.pos = sz if side == "long" else -sz
                agg.entry_price = bar["close"]
                agg.hold_bars = 0
                agg.highest_since_entry = bar["high"] if side == "long" else float("-inf")
                agg.lowest_since_entry = float("inf") if side == "long" else bar["low"]
                agg.active_order = False
            else:
                print(f"[{name}] ENTRY FAILED: {result}", flush=True)
    
    def stop(self):
        self.running = False
        try:
            self.ws.close()
        except Exception:
            pass


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-trade", action="store_true", help="Monitor only, no orders")
    args = parser.parse_args()
    
    # Load env
    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        load_dotenv(env_path)
    
    required = ["OKX_API_KEY", "OKX_SECRET_KEY", "OKX_PASSPHRASE"]
    missing = [k for k in required if not os.getenv(k)]
    if missing:
        print(f"ERROR: missing env vars: {missing}")
        sys.exit(1)
    
    print(f"=== MR-v1 Direct Runner ===")
    print(f"Instruments: {[s[0] for s in SYMBOLS]}")
    print(f"Trade mode: {'OFF (--no-trade)' if args.no_trade else 'ON'}")
    
    # Set leverage
    if not args.no_trade:
        for inst_id, _, _ in SYMBOLS:
            set_leverage(inst_id)
    
    # Start WebSocket feed
    feed = OKXTickerFeed()
    
    # Load historical bars for indicator warmup
    db_path = str(PROJECT_ROOT / ".vntrader" / "database.db")
    print("Loading historical bars...")
    for inst_id in feed.aggregators:
        feed.aggregators[inst_id].load_history(db_path)
    print("History loaded.\n")
    
    running = True
    def on_shutdown(sig, frame):
        nonlocal running
        print("\nShutting down...")
        running = False
    signal.signal(signal.SIGINT, on_shutdown)
    signal.signal(signal.SIGTERM, on_shutdown)
    
    # Run feed in background thread
    thread = threading.Thread(target=feed.run, daemon=True)
    thread.start()
    
    print("Running. Press Ctrl+C to stop.\n")
    
    try:
        while running:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    
    feed.stop()
    thread.join(timeout=5)
    print("Done.")


if __name__ == "__main__":
    main()
