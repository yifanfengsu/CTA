#!/usr/bin/env python3
"""MR-5m Direct Runner: OKX WebSocket feed + 5-min MR strategy + REST orders.

Bypasses vnpy gateway — connects directly to OKX DEMO WebSocket for ticker data,
builds 1m/5m bars, runs Mr5mStrategy logic, and places orders via REST API.

PushPlus notifications on every entry/exit + daily summary.

Deploy: PYTHONUNBUFFERED=1 nohup .venv/bin/python -u scripts/run_mr_5m_direct.py > /tmp/mr_5m_direct.log 2>&1 &
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
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# ═══════════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════════

DEMO_WS_URL = "wss://wspap.okx.com:8443/ws/v5/public"
DEMO_REST_URL = "https://www.okx.com"

SYMBOLS = [
    ("BTC-USDT-SWAP", "BTCUSDT", "mr_5m_btc"),
    ("ETH-USDT-SWAP", "ETHUSDT", "mr_5m_eth"),
    ("SOL-USDT-SWAP", "SOLUSDT", "mr_5m_sol"),
    ("LINK-USDT-SWAP", "LINKUSDT", "mr_5m_lin"),
    ("DOGE-USDT-SWAP", "DOGEUSDT", "mr_5m_dog"),
]

_SYM_NAME = {s[0]: s[2] for s in SYMBOLS}

# Strategy parameters (MR-5m)
LOOKBACK = 24
ATR_WINDOW = 14
ATR_STOP = 1.0
MAX_HOLD = 48
NOTIONAL_PER_TRADE = 500.0
PRICE_OFFSET = 2
BARS_PER_5M = 5  # 5 × 1m = 5m

# ATR regime filter thresholds (p30 from 3yr backtest)
ATR_THRESHOLDS = {
    "BTC-USDT-SWAP": 81.5,
    "ETH-USDT-SWAP": 4.64,
    "SOL-USDT-SWAP": 0.245,
    "LINK-USDT-SWAP": 0.0212,
    "DOGE-USDT-SWAP": 0.0002,
}

# Contract specs (from OKX API)
CONTRACT_SPECS = {
    "BTC-USDT-SWAP":  {"ctVal": 0.01, "tickSz": 0.1,   "minSz": 0.01},
    "ETH-USDT-SWAP":  {"ctVal": 0.1,  "tickSz": 0.01,  "minSz": 0.01},
    "SOL-USDT-SWAP":  {"ctVal": 1.0,  "tickSz": 0.01,  "minSz": 0.01},
    "LINK-USDT-SWAP": {"ctVal": 1.0,  "tickSz": 0.001, "minSz": 0.1},
    "DOGE-USDT-SWAP": {"ctVal": 1000, "tickSz": 0.00001, "minSz": 1.0},
}

# ═══════════════════════════════════════════════════════════════════════════════
# PushPlus Notifier
# ═══════════════════════════════════════════════════════════════════════════════

PUSHPLUS_URL = "http://www.pushplus.plus/send"


class Notifier:
    """PushPlus notification sender. Set PUSHPLUS_TOKEN in .env."""

    def __init__(self):
        self.token = os.getenv("PUSHPLUS_TOKEN", "")
        self.enabled = bool(self.token)

    def send(self, title: str, content: str) -> bool:
        if not self.enabled:
            return False
        try:
            resp = requests.post(
                PUSHPLUS_URL,
                json={"token": self.token, "title": title, "content": content},
                timeout=10,
            )
            data = resp.json()
            if data.get("code") == 200:
                print(f"[NOTIFY] {title}", flush=True)
                return True
            else:
                print(f"[NOTIFY] FAIL: {data.get('msg', 'unknown')}", flush=True)
                return False
        except Exception as e:
            print(f"[NOTIFY] ERROR: {e}", flush=True)
            return False


# ═══════════════════════════════════════════════════════════════════════════════
# OKX REST API helpers
# ═══════════════════════════════════════════════════════════════════════════════

def okx_sign(timestamp: str, method: str, path: str, body: str = "") -> str:
    secret = os.getenv("OKX_SECRET_KEY", "")
    sign_str = timestamp + method + path + body
    return base64.b64encode(
        hmac.new(secret.encode(), sign_str.encode(), "sha256").digest()
    ).decode()


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
    resp = requests.post(
        f"{DEMO_REST_URL}{path}",
        data=body_str,
        headers=okx_headers("POST", path, body_str),
        timeout=10,
    )
    return resp.json()


def okx_get(path: str) -> dict:
    resp = requests.get(
        f"{DEMO_REST_URL}{path}",
        headers=okx_headers("GET", path),
        timeout=10,
    )
    return resp.json()


# ═══════════════════════════════════════════════════════════════════════════════
# Bar Aggregator: 1m → 5m
# ═══════════════════════════════════════════════════════════════════════════════

class BarAggregator:
    """Aggregate 1m ticker updates into 5m bars."""

    def __init__(self, inst_id: str):
        self.inst_id = inst_id
        self._1m_current: dict | None = None
        self._1m_minute: int = -1
        self._1m_count: int = 0
        self._1m_in_5m: int = 0  # bars accumulated in current 5m window

        self._5m_current: dict | None = None
        self._5m_count: int = 0

        # History for indicators
        self._5m_history: list[dict] = []
        self._5m_max_keep = 200

        # Indicators
        self.atr: float = 0.0
        self.donchian_high: float = 0.0
        self.donchian_low: float = 0.0
        self.midline: float = 0.0

        # ATR filter
        self.atr_threshold: float = ATR_THRESHOLDS.get(inst_id, 0.0)
        self.filtered_count: int = 0

        # Position tracking
        self.pos: float = 0.0
        self.entry_price: float = 0.0
        self.entry_time: str = ""
        self.hold_bars: int = 0
        self.highest_since_entry: float = 0.0
        self.lowest_since_entry: float = float("inf")

        # Order tracking — wait for fill confirmation
        self.pending_order_id: str = ""
        self.pending_order_side: str = ""   # "entry" or "exit"
        self.pending_order_time: float = 0.0
        self.pending_order_sz: int = 0
        self.pending_entry_side: str = ""    # "long" or "short"
        self.ORDER_TIMEOUT: int = 30  # seconds

    def load_history(self, db_path: str, days: int = 90):
        """Load historical 1m bars from vnpy database to warm up indicators."""
        import sqlite3
        base = self.inst_id.replace("-", "").replace("SWAP", "_SWAP_OKX")
        try:
            conn = sqlite3.connect(db_path)
            cutoff = (
                datetime.datetime.now(datetime.timezone.utc)
                - datetime.timedelta(days=days)
            ).strftime("%Y-%m-%d")
            rows = conn.execute(
                "SELECT datetime, open_price, high_price, low_price, close_price, volume "
                "FROM dbbardata WHERE symbol=? AND exchange=? AND interval=? AND datetime>=? "
                "ORDER BY datetime",
                (base, "GLOBAL", "1m", cutoff),
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
            result_5m = self._process_1m(float(o), float(h), float(l), float(c), ts_ms)
            if result_5m is None:
                bars_loaded += 1  # 1m bar processed (but not a completed 5m)
        print(
            f"  [{self.inst_id}] Loaded {bars_loaded} 1m → {self._5m_count} 5m bars",
            flush=True,
        )

    def _process_1m(self, o: float, h: float, l: float, c: float, ts: str) -> dict | None:
        """Process one 1m bar. Returns completed 5m bar dict or None."""
        # Start/update 5m bar
        if self._5m_current is None:
            self._5m_current = {"open": o, "high": h, "low": l, "close": c, "ts": ts}
            self._1m_in_5m = 1
        else:
            bar = self._5m_current
            bar["high"] = max(bar["high"], h)
            bar["low"] = min(bar["low"], l)
            bar["close"] = c
            self._1m_in_5m += 1

        # Close 5m bar when 5 x 1m bars accumulated
        if self._1m_in_5m >= BARS_PER_5M:
            result = dict(self._5m_current)
            self._5m_count += 1
            self._5m_current = None
            self._1m_in_5m = 0
            self._add_5m_bar(result)
            return result
        return None

    def on_ticker(self, last: float, bid: float, ask: float, ts: str):
        """Process a ticker update. Returns (1m_bar, 5m_bar) if completed, else None."""
        now = datetime.datetime.fromtimestamp(int(ts) / 1000, tz=datetime.timezone.utc)
        minute = now.minute

        result_1m = None

        # 1m bar
        if self._1m_current is None or minute != self._1m_minute:
            if self._1m_current is not None:
                result_1m = self._1m_current
            self._1m_current = {
                "open": last, "high": last, "low": last, "close": last,
                "volume": 0, "ts": ts,
            }
            self._1m_minute = minute
            self._1m_count += 1
        else:
            bar = self._1m_current
            bar["high"] = max(bar["high"], last)
            bar["low"] = min(bar["low"], last)
            bar["close"] = last

        # Build 5m from completed 1m
        result_5m = None
        if result_1m is not None:
            result_5m = self._process_1m(
                result_1m["open"],
                result_1m["high"],
                result_1m["low"],
                result_1m["close"],
                result_1m["ts"],
            )

        return result_1m, result_5m

    def _add_5m_bar(self, bar: dict):
        self._5m_history.append(bar)
        if len(self._5m_history) > self._5m_max_keep:
            self._5m_history.pop(0)
        self.atr = self._compute_atr()
        self._compute_donchian()
        if self.donchian_high > 0 and self.donchian_low > 0:
            self.midline = (self.donchian_high + self.donchian_low) / 2

    def _compute_atr(self) -> float:
        if len(self._5m_history) < ATR_WINDOW + 1:
            return 0.0
        tr_values = []
        for i in range(-ATR_WINDOW, 0):
            cur = self._5m_history[i]
            prev = self._5m_history[i - 1]
            tr = max(
                cur["high"] - cur["low"],
                abs(cur["high"] - prev["close"]),
                abs(cur["low"] - prev["close"]),
            )
            tr_values.append(tr)
        return sum(tr_values) / len(tr_values)

    def _compute_donchian(self):
        """Donchian channel — excludes current bar (matching vnpy: high[-LB:-1])."""
        if len(self._5m_history) < LOOKBACK:
            self.donchian_high = 0.0
            self.donchian_low = 0.0
            return
        # [-LOOKBACK:-1] = last LOOKBACK bars excluding current (23 bars for LB=24)
        window = self._5m_history[-LOOKBACK:-1]
        self.donchian_high = max(b["high"] for b in window)
        self.donchian_low = min(b["low"] for b in window)


# ═══════════════════════════════════════════════════════════════════════════════
# Order execution
# ═══════════════════════════════════════════════════════════════════════════════

def calc_size(inst_id: str, price: float) -> int:
    spec = CONTRACT_SPECS[inst_id]
    multiplier = spec["ctVal"]
    contract_value = price * multiplier
    if contract_value <= 0:
        return 1
    size = round(NOTIONAL_PER_TRADE / contract_value)
    return max(1, min(size, 1000))


def round_price(inst_id: str, price: float) -> float:
    tick = CONTRACT_SPECS[inst_id]["tickSz"]
    return round(price / tick) * tick


def place_order(inst_id: str, side: str, price: float, sz: int) -> dict | None:
    """Place a limit order via OKX REST API."""
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


def check_order_filled(inst_id: str, ord_id: str) -> dict | None:
    """Query order status. Returns order dict if filled, None if still open/error."""
    result = okx_get(f"/api/v5/trade/order?instId={inst_id}&ordId={ord_id}")
    if not result or result.get("code") != "0":
        return None
    data = result.get("data", [])
    if not data:
        return None
    order = data[0]
    if order["state"] == "filled":
        return order
    if order["state"] == "canceled":
        return {"state": "canceled", "sz": order["sz"], "fillSz": order["fillSz"]}
    return None  # still live


def cancel_order(inst_id: str, ord_id: str) -> bool:
    """Cancel a pending order."""
    body = {"instId": inst_id, "ordId": ord_id}
    result = okx_post("/api/v5/trade/cancel-order", body)
    return result.get("code") == "0"


def sync_positions_from_okx(aggregators: dict):
    """Sync local position state from OKX REST API (call at startup)."""
    result = okx_get("/api/v5/account/positions?instType=SWAP")
    if not result or result.get("code") != "0":
        print("  WARNING: Could not fetch positions from OKX")
        return
    for p in result.get("data", []):
        inst_id = p["instId"]
        agg = aggregators.get(inst_id)
        if not agg:
            continue
        pos = float(p["pos"])
        if pos == 0:
            continue
        # Determine direction: pos>0 = long, pos<0 = short
        agg.pos = float(p["pos"])
        agg.entry_price = float(p["avgPx"])
        agg.entry_time = p.get("cTime", "")
        agg.hold_bars = 0  # unknown, reset
        side = "long" if agg.pos > 0 else "short"
        print(
            f"  [{inst_id}] Synced existing {side} pos={agg.pos} entry={agg.entry_price}",
            flush=True,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Trade log for PnL tracking
# ═══════════════════════════════════════════════════════════════════════════════

TRADE_LOG_PATH = PROJECT_ROOT / "trade_log_5m.jsonl"


def log_trade(entry: dict):
    """Append a trade record to JSONL file."""
    with open(TRADE_LOG_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")


# ═══════════════════════════════════════════════════════════════════════════════
# Strategy logic
# ═══════════════════════════════════════════════════════════════════════════════

def check_entry(agg: BarAggregator, bar: dict) -> tuple[str | None, float | None]:
    """MR-5m entry signal. Returns (side, stop_price) or (None, None)."""
    if agg.pos != 0 or agg.pending_order_id:
        return None, None

    close = bar["close"]
    if close <= 0 or agg.atr <= 0:
        return None, None

    # ATR regime filter
    if agg.atr_threshold > 0 and agg.atr < agg.atr_threshold:
        agg.filtered_count += 1
        return None, None

    # Donchian breakout (using pre-computed values which exclude current bar)
    long_breakout = close > agg.donchian_high > 0
    short_breakout = close < agg.donchian_low > 0

    if long_breakout:
        return "short", close + ATR_STOP * agg.atr  # fade: short on long breakout
    elif short_breakout:
        return "long", close - ATR_STOP * agg.atr   # fade: long on short breakout

    return None, None


def check_exit(agg: BarAggregator, bar: dict) -> tuple[bool, str]:
    """MR-5m exit signal. Priority: midline > stop > max_hold."""
    if agg.pos == 0:
        return False, ""

    agg.hold_bars += 1
    close = bar["close"]

    # Update extremes
    if agg.pos > 0:
        agg.highest_since_entry = max(agg.highest_since_entry, bar["high"])
    else:
        agg.lowest_since_entry = min(agg.lowest_since_entry, bar["low"])

    # 1) Midline take-profit
    if agg.midline > 0:
        if (agg.pos > 0 and close >= agg.midline) or (agg.pos < 0 and close <= agg.midline):
            return True, "midline"

    # 2) ATR stop
    stop_dist = ATR_STOP * agg.atr
    if agg.pos > 0:
        if bar["low"] <= agg.entry_price - stop_dist:
            return True, "stop"
    else:
        if bar["high"] >= agg.entry_price + stop_dist:
            return True, "stop"

    # 3) Max hold
    if agg.hold_bars >= MAX_HOLD:
        return True, "max_hold"

    return False, ""


def reset_position(agg: BarAggregator):
    agg.pos = 0.0
    agg.entry_price = 0.0
    agg.entry_time = ""
    agg.hold_bars = 0
    agg.highest_since_entry = 0.0
    agg.lowest_since_entry = float("inf")
    agg.pending_order_id = ""
    agg.pending_order_side = ""
    agg.pending_order_time = 0.0
    agg.pending_order_sz = 0
    agg.pending_entry_side = ""


# ═══════════════════════════════════════════════════════════════════════════════
# WebSocket client
# ═══════════════════════════════════════════════════════════════════════════════

class OKXTickerFeed:
    """WebSocket feed for OKX ticker data."""

    def __init__(self, notifier: Notifier):
        self.ws = None
        self.running = False
        self.aggregators: dict[str, BarAggregator] = {}
        self._1m_counters: dict[str, int] = defaultdict(int)
        self.notifier = notifier
        self._last_summary_time: float = 0.0
        self._summary_interval: int = 6 * 3600  # 6 hours

        for inst_id, _, name in SYMBOLS:
            self.aggregators[inst_id] = BarAggregator(inst_id)

    def _connect_ws(self):
        import websocket
        self.ws = websocket.WebSocket()
        self.ws.connect(DEMO_WS_URL)
        self.ws.settimeout(10)
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
                        try:
                            self.ws.send("ping")
                        except Exception:
                            break
                    except Exception as e:
                        print(f"WS recv error: {type(e).__name__}: {e}", flush=True)
                        break
            except Exception as e:
                print(
                    f"WS connection error: {e}, reconnecting in {reconnect_delay}s...",
                    flush=True,
                )
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

            result_1m, result_5m = agg.on_ticker(last, bid, ask, ts)

            # Print 1m bar heartbeat (every 10 bars)
            if result_1m is not None:
                self._1m_counters[inst_id] += 1
                cnt = self._1m_counters[inst_id]
                if cnt == 1 or cnt % 10 == 0:
                    name = _SYM_NAME[inst_id]
                    print(f"[{name}] 1m #{cnt} | c={result_1m['close']}", flush=True)

            # Process 5m bar
            if result_5m is not None:
                name = _SYM_NAME[inst_id]
                self._on_5m_bar(inst_id, agg, result_5m, name)

        # Periodic summary
        now = time.time()
        if now - self._last_summary_time > self._summary_interval:
            self._last_summary_time = now
            self._send_summary()

    def _on_5m_bar(self, inst_id: str, agg: BarAggregator, bar: dict, name: str):
        """Strategy logic on completed 5m bar."""
        close = bar["close"]

        # ── Pending order check ──────────────────────────────────────────
        if agg.pending_order_id:
            elapsed = time.time() - agg.pending_order_time
            order = check_order_filled(inst_id, agg.pending_order_id)

            if order and order.get("state") == "filled":
                # Order confirmed filled — use actual fill size, not requested
                fill_px = float(order.get("avgPx", order.get("fillPx", bar["close"])))
                fill_sz = float(order.get("fillSz", "0"))
                if fill_sz <= 0:
                    fill_sz = float(order.get("sz", agg.pending_order_sz))

                if agg.pending_order_side == "entry":
                    agg.pos = fill_sz if agg.pending_entry_side == "long" else -fill_sz
                    agg.entry_price = fill_px
                    agg.entry_time = datetime.datetime.now(datetime.timezone.utc).isoformat()
                    agg.hold_bars = 0
                    if agg.pending_entry_side == "long":
                        agg.highest_since_entry = bar["high"]
                        agg.lowest_since_entry = float("inf")
                    else:
                        agg.highest_since_entry = float("-inf")
                        agg.lowest_since_entry = bar["low"]
                    if fill_sz != agg.pending_order_sz:
                        print(f"[{name}] ENTRY PARTIAL FILL | req={agg.pending_order_sz} filled={fill_sz} pos={agg.pos}", flush=True)
                    else:
                        print(f"[{name}] ENTRY FILLED | px={fill_px} pos={agg.pos}", flush=True)

                else:  # exit
                    # PnL based on actual filled size
                    pnl = (fill_px - agg.entry_price) * fill_sz if agg.pos > 0 else (agg.entry_price - fill_px) * fill_sz
                    multiplier = CONTRACT_SPECS[inst_id]["ctVal"]
                    pnl_usd = pnl * multiplier
                    log_trade({
                        "time": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                        "symbol": name,
                        "side": "long" if agg.pos > 0 else "short",
                        "entry_price": agg.entry_price,
                        "entry_time": agg.entry_time,
                        "exit_price": fill_px,
                        "exit_reason": agg.pending_order_side,
                        "size": fill_sz,
                        "pnl_usd": round(pnl_usd, 2),
                    })
                    self.notifier.send(
                        f"MR-5m {name} EXIT filled",
                        f"{'多' if agg.pos > 0 else '空'}平仓成交 | 入场{agg.entry_price} → 出场{fill_px}\n"
                        f"盈亏: ${pnl_usd:+.2f} | 数量: {fill_sz}\n"
                        f"时间: {datetime.datetime.now().strftime('%m-%d %H:%M')}",
                    )

                    # Handle partial fill: reduce position, only reset if fully closed
                    remaining = abs(agg.pos) - fill_sz
                    if remaining <= 0.001:  # fully closed
                        reset_position(agg)
                        print(f"[{name}] EXIT FILLED | px={fill_px} pnl=${pnl_usd:+.2f}", flush=True)
                    else:
                        # Partial fill — update position, keep tracking
                        agg.pos = remaining if agg.pos > 0 else -remaining
                        print(f"[{name}] EXIT PARTIAL | filled={fill_sz} remaining={agg.pos} px={fill_px}", flush=True)
                agg.pending_order_id = ""
                agg.pending_order_side = ""
                agg.pending_order_time = 0.0
                agg.pending_order_sz = 0
                agg.pending_entry_side = ""

            elif order and order.get("state") == "canceled":
                print(f"[{name}] ORDER CANCELED | {agg.pending_order_id}", flush=True)
                agg.pending_order_id = ""
                agg.pending_order_side = ""
                agg.pending_order_time = 0.0
                agg.pending_order_sz = 0
                agg.pending_entry_side = ""

            elif elapsed > agg.ORDER_TIMEOUT:
                # Timeout — cancel order
                print(f"[{name}] ORDER TIMEOUT {elapsed:.0f}s — canceling {agg.pending_order_id}", flush=True)
                cancelled = cancel_order(inst_id, agg.pending_order_id)
                print(f"[{name}] Cancel result: {'OK' if cancelled else 'FAIL'}", flush=True)
                if agg.pending_order_side == "exit":
                    # Exit order timed out — reset position anyway (we tried)
                    reset_position(agg)
                else:
                    agg.pending_order_id = ""
                    agg.pending_order_side = ""
                    agg.pending_order_time = 0.0
                    agg.pending_order_sz = 0
                    agg.pending_entry_side = ""

            # Still pending — skip strategy logic this bar
            if agg.pending_order_id:
                return

        # ── Position management (only when no pending order) ────────────

        # Status log every 288 bars (daily)
        if agg._5m_count % 288 == 0:
            print(
                f"[{name}] 5m #{agg._5m_count} | c={close:.4f} "
                f"ATR={agg.atr:.4f} DH={agg.donchian_high:.4f} "
                f"DL={agg.donchian_low:.4f} MID={agg.midline:.4f} "
                f"pos={agg.pos} filtered={agg.filtered_count}",
                flush=True,
            )

        # Check exit
        should_exit, reason = check_exit(agg, bar)
        if should_exit:
            exit_side = "sell" if agg.pos > 0 else "buy"
            tick_sz = CONTRACT_SPECS[inst_id]["tickSz"]
            sz = abs(int(agg.pos))
            # Display PnL uses bar close (actual fill PnL confirmed separately)
            display_pnl = (bar["close"] - agg.entry_price) * sz if agg.pos > 0 else (agg.entry_price - bar["close"]) * sz
            multiplier = CONTRACT_SPECS[inst_id]["ctVal"]
            display_pnl_usd = display_pnl * multiplier

            # Order price uses offset to ensure fill
            if agg.pos > 0:
                exit_price = bar["close"] - PRICE_OFFSET * tick_sz  # sell below market
            else:
                exit_price = bar["close"] + PRICE_OFFSET * tick_sz  # buy above market

            print(
                f"[{name}] EXIT {reason} | side={exit_side} px={exit_price} sz={sz} "
                f"entry={agg.entry_price} est_pnl={display_pnl_usd:.2f}",
                flush=True,
            )

            result = place_order(inst_id, exit_side, exit_price, sz)
            if result and result.get("code") == "0":
                ord_id = result["data"][0].get("ordId", "")
                print(f"[{name}] EXIT order placed: {ord_id[:8]} — waiting fill", flush=True)
                # Store pending order — DON'T reset position until fill confirmed
                agg.pending_order_id = ord_id
                agg.pending_order_side = "exit"
                agg.pending_order_time = time.time()
                agg.pending_order_sz = sz
            else:
                print(f"[{name}] EXIT FAILED: {result}", flush=True)
            return

        # Check entry
        side, stop_price = check_entry(agg, bar)
        if side is not None:
            tick_sz = CONTRACT_SPECS[inst_id]["tickSz"]
            if side == "long":
                price = bar["close"] + PRICE_OFFSET * tick_sz
            else:
                price = bar["close"] - PRICE_OFFSET * tick_sz
            sz = calc_size(inst_id, bar["close"])
            okx_side = "buy" if side == "long" else "sell"

            print(
                f"[{name}] ENTRY {side} | px={price} sz={sz} "
                f"stop={stop_price:.4f} c={close} DH={agg.donchian_high:.4f} DL={agg.donchian_low:.4f}",
                flush=True,
            )

            self.notifier.send(
                f"MR-5m {name} ENTRY {side.upper()}",
                f"{'做多' if side == 'long' else '做空'}开仓 | 价格{price}\n"
                f"数量: {sz}张 | 止损: {stop_price:.4f}\n"
                f"突破: close={close} DH={agg.donchian_high} DL={agg.donchian_low}\n"
                f"时间: {datetime.datetime.now().strftime('%m-%d %H:%M')}",
            )

            result = place_order(inst_id, okx_side, price, sz)
            if result and result.get("code") == "0":
                ord_id = result["data"][0].get("ordId", "")
                print(f"[{name}] ENTRY order placed: {ord_id[:8]} — waiting fill", flush=True)
                # Store pending order — DON'T set pos until fill confirmed
                agg.pending_order_id = ord_id
                agg.pending_order_side = "entry"
                agg.pending_order_time = time.time()
                agg.pending_order_sz = sz
                agg.pending_entry_side = side
            else:
                print(f"[{name}] ENTRY FAILED: {result}", flush=True)

    def _send_summary(self):
        """Send periodic summary of all positions."""
        lines = []
        total_pnl = 0.0
        has_position = False
        now_str = datetime.datetime.now().strftime("%m-%d %H:%M")

        for inst_id, _, name in SYMBOLS:
            agg = self.aggregators[inst_id]
            if agg.pos != 0:
                has_position = True
                close = agg._5m_history[-1]["close"] if agg._5m_history else 0
                multiplier = CONTRACT_SPECS[inst_id]["ctVal"]
                if agg.pos > 0:
                    unrealized = (close - agg.entry_price) * abs(agg.pos) * multiplier
                else:
                    unrealized = (agg.entry_price - close) * abs(agg.pos) * multiplier
                total_pnl += unrealized
                lines.append(
                    f"{'多' if agg.pos > 0 else '空'} {name}: 入场{agg.entry_price} "
                    f"现价{close} 持仓{agg.hold_bars}bar | "
                    f"浮盈${unrealized:+.2f}"
                )
            else:
                lines.append(f"空仓 {name}: DH={agg.donchian_high:.4f} DL={agg.donchian_low:.4f} ATR={agg.atr:.4f}")

        if has_position:
            title = f"MR-5m 持仓摘要 {now_str} | 浮盈${total_pnl:+.2f}"
        else:
            title = f"MR-5m 状态 {now_str} | 无持仓"

        content = "\n".join(lines)
        self.notifier.send(title, content)

    def stop(self):
        self.running = False
        try:
            self.ws.close()
        except Exception:
            pass


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-trade", action="store_true", help="Monitor only, no orders")
    parser.add_argument("--no-notify", action="store_true", help="Disable PushPlus notifications")
    parser.add_argument("--summary-interval", type=int, default=6, help="Summary interval in hours (default: 6)")
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

    # PushPlus check
    pushplus_token = os.getenv("PUSHPLUS_TOKEN", "")
    if not pushplus_token and not args.no_notify:
        print("WARNING: PUSHPLUS_TOKEN not set in .env — notifications disabled.")
        print("  Get your token at https://www.pushplus.plus/")
        print("  Add to .env: PUSHPLUS_TOKEN=your_token_here")
    notifier = Notifier()

    print(f"=== MR-5m Direct Runner ===")
    print(f"Instruments: {[s[0] for s in SYMBOLS]}")
    print(f"Trade mode: {'OFF (--no-trade)' if args.no_trade else 'ON'}")
    print(f"Notify: {'ON' if notifier.enabled else 'OFF'}")
    print(f"Params: LB={LOOKBACK} ATR={ATR_STOP} MH={MAX_HOLD} Notional=${NOTIONAL_PER_TRADE}")
    print(f"Trade log: {TRADE_LOG_PATH}")

    # Set leverage
    if not args.no_trade:
        for inst_id, _, _ in SYMBOLS:
            set_leverage(inst_id)

    # Start WebSocket feed
    feed = OKXTickerFeed(notifier)
    feed._summary_interval = args.summary_interval * 3600

    # Load historical bars for indicator warmup
    db_path = str(PROJECT_ROOT / ".vntrader" / "database.db")
    print("Loading historical bars...")
    for inst_id in feed.aggregators:
        feed.aggregators[inst_id].load_history(db_path)
    print("History loaded.\n")

    # Sync existing positions from OKX
    if not args.no_trade:
        print("Syncing positions from OKX...")
        sync_positions_from_okx(feed.aggregators)

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

    # Startup notification
    notifier.send(
        "MR-5m 策略已启动",
        f"时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"币种: BTC ETH SOL LINK DOGE\n"
        f"参数: LB={LOOKBACK} STOP={ATR_STOP}ATR MH={MAX_HOLD}\n"
        f"每笔: ${NOTIONAL_PER_TRADE}\n"
        f"模式: {'仅监控' if args.no_trade else '实盘交易'}",
    )

    print("Running. Press Ctrl+C to stop.\n")

    try:
        while running:
            time.sleep(1)
    except KeyboardInterrupt:
        pass

    feed.stop()
    thread.join(timeout=5)

    # Shutdown notification
    notifier.send(
        "MR-5m 策略已停止",
        f"停止时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
    )

    print("Done.")


if __name__ == "__main__":
    main()
