#!/usr/bin/env python3
"""MR-v1 Donchian Midline Analysis: bar-by-bar path of winning trades.

Uses globally-computed Donchian midlines (from pre-entry history, .shift(1))
to track exact midline crossings from entry bar 0 onward. No blind spot.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# 2026-07 重构批次6：脚本迁入 _archive/legacy_scripts/；共享依赖真身在
# scripts/（前向冻结区）与 core/data_io/，此处按新深度注入 sys.path。
import sys as _sys
from pathlib import Path as _Path

_REPO_ROOT = _Path(__file__).resolve().parents[2]
for _p in (
    str(_REPO_ROOT / "core" / "data_io"),
    str(_REPO_ROOT / "scripts"),
    str(_REPO_ROOT / "data_engineering" / "scripts"),
    *sorted(str(_q) for _q in (_REPO_ROOT / "research" / "_closed").glob("*/*/scripts")),
):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)

from common_runtime import PROJECT_ROOT, ensure_headless_runtime, log_event, setup_logging, to_jsonable
from history_time_utils import HistoryRange, parse_history_range

# ---------------------------------------------------------------------------
DEFAULT_SYMBOLS = [
    "BTCUSDT_SWAP_OKX.GLOBAL", "ETHUSDT_SWAP_OKX.GLOBAL",
    "SOLUSDT_SWAP_OKX.GLOBAL", "LINKUSDT_SWAP_OKX.GLOBAL",
    "DOGEUSDT_SWAP_OKX.GLOBAL",
]
DEFAULT_START = "2023-01-01"
DEFAULT_END = "2026-03-31"
DEFAULT_TIMEZONE = "Asia/Shanghai"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "reports" / "research" / "mr_v1_midline"
DEFAULT_DATABASE_PATH = PROJECT_ROOT / ".vntrader" / "database.db"

LOOKBACK = 8
ATR_MULT = 1.0
MAX_HOLD = 60
ATR_PERIOD = 14
FIXED_NOTIONAL = 1000.0


class ResearchError(Exception):
    pass


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def split_vt_symbol(vt: str) -> tuple[str, str]:
    s, sep, e = str(vt).partition(".")
    if not sep:
        raise ResearchError(f"bad vt_symbol: {vt}")
    return s, e


def normalize_1m(df: pd.DataFrame, tz: str) -> pd.DataFrame:
    cols = ["datetime", "open", "high", "low", "close", "volume"]
    if df.empty:
        return pd.DataFrame(columns=cols)
    out = df.loc[:, cols].copy()
    ts = pd.to_datetime(out["datetime"], errors="coerce")
    out["datetime"] = ts.dt.tz_localize(tz) if ts.dt.tz is None else ts.dt.tz_convert(tz)
    for c in ["open", "high", "low", "close", "volume"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out.dropna(subset=cols).sort_values("datetime", kind="stable").drop_duplicates("datetime", keep="last").reset_index(drop=True)


def load_1m(sym: str, hr: HistoryRange, db: Path) -> pd.DataFrame:
    s, e = split_vt_symbol(sym)
    qs = hr.start.replace(tzinfo=None).isoformat(sep=" ", timespec="seconds")
    qe = hr.end_exclusive.replace(tzinfo=None).isoformat(sep=" ", timespec="seconds")
    with sqlite3.connect(db) as conn:
        df = pd.read_sql_query(
            "select datetime, open_price as open, high_price as high, low_price as low, close_price as close, volume "
            "from dbbardata where symbol=? and exchange=? and interval='1m' and datetime>=? and datetime<? order by datetime",
            conn, params=(s, e, qs, qe))
    return normalize_1m(df, hr.timezone_name)


def resample_4h(bars_1m: pd.DataFrame, hr: HistoryRange | None = None) -> pd.DataFrame:
    cols = ["open_time", "datetime", "open", "high", "low", "close", "volume"]
    if bars_1m.empty:
        return pd.DataFrame(columns=cols)
    w = bars_1m.sort_values("datetime", kind="stable").drop_duplicates("datetime", keep="last").copy()
    anchor = pd.Timestamp(hr.start if hr else w["datetime"].iloc[0])
    if anchor.tzinfo is None:
        anchor = anchor.tz_localize(w["datetime"].iloc[0].tz)
    deltas = (w["datetime"] - anchor) / pd.Timedelta(minutes=1)
    w = w.loc[deltas >= 0].copy()
    w["_slot"] = np.floor(deltas.loc[w.index].to_numpy(dtype=float) / 240).astype(np.int64)
    g = w.groupby("_slot", sort=True, dropna=False)
    r = g.agg(open_time=("datetime", "min"), datetime=("datetime", "max"),
              open=("open", "first"), high=("high", "max"), low=("low", "min"),
              close=("close", "last"), volume=("volume", "sum"), mc=("datetime", "size"))
    return r[r["mc"] == 240].drop(columns=["mc"]).dropna(subset=["open", "high", "low", "close"]).reset_index(drop=True).loc[:, cols]


def compute_atr(bars: pd.DataFrame, period: int) -> pd.Series:
    h, l, c = bars["high"], bars["low"], bars["close"]
    tr = pd.concat([(h - l).abs(), (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()


# ---------------------------------------------------------------------------
# Backtest with global midline tracking
# ---------------------------------------------------------------------------

@dataclass
class TradeRecord:
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    direction: int
    entry_price: float
    exit_price: float
    exit_reason: str
    symbol: str
    entry_bar_idx: int
    trade_closes: np.ndarray      # closes from entry_bar to exit_bar
    trade_midlines: np.ndarray    # global midlines for same range


def run_backtest_with_tracking(bars_4h: pd.DataFrame, symbol: str) -> list[TradeRecord]:
    """Run MR-v1 backtest, recording globally-computed midlines per trade."""
    if len(bars_4h) < LOOKBACK + 2:
        return []

    atr = compute_atr(bars_4h, ATR_PERIOD)
    highs = bars_4h["high"].rolling(LOOKBACK).max().shift(1)
    lows = bars_4h["low"].rolling(LOOKBACK).min().shift(1)
    close_series = bars_4h["close"]

    # Global midline: (DH + DL) / 2, using strict .shift(1) for no look-ahead
    global_midlines = (highs + lows) / 2

    trades: list[TradeRecord] = []
    pos = 0
    entry_bar = -1
    entry_price = 0.0
    entry_atr = 0.0

    for i in range(LOOKBACK, len(bars_4h)):
        bar = bars_4h.iloc[i]

        if pos != 0:
            hold_bars = i - entry_bar
            stop_dist = ATR_MULT * entry_atr
            exit_now = False
            reason = ""

            if pos == 1:
                if bar["low"] <= entry_price - stop_dist:
                    exit_now = True
                    reason = "stop"
            else:
                if bar["high"] >= entry_price + stop_dist:
                    exit_now = True
                    reason = "stop"

            if hold_bars >= MAX_HOLD and not exit_now:
                exit_now = True
                reason = "max_hold"

            if exit_now:
                # Realistic exit price: stop fills at stop level unless gapped
                if reason == "stop":
                    if pos == 1:
                        exit_price = min(bar["open"], entry_price - stop_dist)
                    else:
                        exit_price = max(bar["open"], entry_price + stop_dist)
                else:
                    exit_price = bar["close"]

                trade_closes = bars_4h["close"].iloc[entry_bar:i+1].to_numpy(dtype=float)
                trade_midlines = global_midlines.iloc[entry_bar:i+1].to_numpy(dtype=float)
                trades.append(TradeRecord(
                    entry_time=bars_4h["datetime"].iloc[entry_bar],
                    exit_time=bar["datetime"], direction=pos,
                    entry_price=entry_price, exit_price=exit_price,
                    exit_reason=reason, symbol=symbol,
                    entry_bar_idx=entry_bar,
                    trade_closes=trade_closes,
                    trade_midlines=trade_midlines,
                ))
                pos = 0
                continue

        if pos == 0:
            lb = close_series.iloc[i] > highs.iloc[i]
            sb = close_series.iloc[i] < lows.iloc[i]
            if lb:
                pos = -1
                entry_bar = i
                entry_price = bar["close"]
                entry_atr = atr.iloc[i] if i < len(atr) and not pd.isna(atr.iloc[i]) and atr.iloc[i] > 0 else bar["close"] * 0.01
            elif sb:
                pos = 1
                entry_bar = i
                entry_price = bar["close"]
                entry_atr = atr.iloc[i] if i < len(atr) and not pd.isna(atr.iloc[i]) and atr.iloc[i] > 0 else bar["close"] * 0.01

    return trades


# ---------------------------------------------------------------------------
# Midline analysis — FULL SAMPLE (all exit reasons)
# ---------------------------------------------------------------------------

@dataclass
class CrossResult:
    """Result for a single trade's midline crossing analysis."""
    symbol: str
    direction: int
    exit_reason: str
    entry_price: float
    exit_price: float
    hold_bars: int
    first_cross_bar: int | None     # None = never crossed
    total_crossings: int
    is_profitable_at_cross: bool    # True = price moved toward profit before/at cross
    pnl_at_cross: float             # PnL if exit at first cross ($)
    final_pnl: float                # Actual final PnL ($)
    max_pnl_after_cross: float      # Best PnL achievable after first cross ($)


def analyze_all_trades(trades: list[TradeRecord]) -> dict[str, Any]:
    """Full-sample midline analysis across ALL exit reasons.

    Returns stratified results: by exit reason, by profitable_at_cross,
    and computes expected value impact if midline exit were applied.
    """
    results: list[CrossResult] = []
    for t in trades:
        r = _analyze_one_trade(t)
        results.append(r)

    if not results:
        return {"error": "No trades"}

    # --- Stratify by exit reason ---
    by_reason: dict[str, list[CrossResult]] = defaultdict(list)
    for r in results:
        by_reason[r.exit_reason].append(r)

    stratified = {}
    for reason, rlist in by_reason.items():
        crossed = [r for r in rlist if r.first_cross_bar is not None]
        never = [r for r in rlist if r.first_cross_bar is None]
        profitable_cross = [r for r in crossed if r.is_profitable_at_cross]
        ghost_cross = [r for r in crossed if not r.is_profitable_at_cross]

        # Actual PnL for this group
        actual_total = sum(r.final_pnl for r in rlist)
        # Expected value if midline exit applied to ALL trades in this group
        midline_ev = sum(r.pnl_at_cross for r in crossed) + sum(r.final_pnl for r in never)

        cross_bars = [r.first_cross_bar for r in crossed]
        stratified[reason] = {
            "total": len(rlist),
            "crossed": len(crossed),
            "never_crossed": len(never),
            "profitable_at_cross": len(profitable_cross),
            "ghost_cross": len(ghost_cross),
            "actual_total_pnl": actual_total,
            "midline_ev_total_pnl": midline_ev,
            "pnl_delta": midline_ev - actual_total,
            "first_cross_stats": _cross_stats(cross_bars) if cross_bars else None,
            "first_cross_distribution": _bar_distribution(np.array(cross_bars)) if cross_bars else {},
            "avg_actual_pnl": actual_total / len(rlist) if rlist else 0,
            "avg_midline_pnl": midline_ev / len(rlist) if rlist else 0,
            "cross_details": [
                {
                    "bar": r.first_cross_bar,
                    "pnl_at_cross": r.pnl_at_cross,
                    "final_pnl": r.final_pnl,
                    "profitable_at_cross": r.is_profitable_at_cross,
                }
                for r in crossed
            ],
        }

    # --- Full sample summary ---
    all_crossed = [r for r in results if r.first_cross_bar is not None]
    all_never = [r for r in results if r.first_cross_bar is None]
    cross_bars_all = [r.first_cross_bar for r in all_crossed]

    actual_total_all = sum(r.final_pnl for r in results)
    midline_ev_all = sum(r.pnl_at_cross for r in all_crossed) + sum(r.final_pnl for r in all_never)
    profitable_cross_all = [r for r in all_crossed if r.is_profitable_at_cross]
    ghost_cross_all = [r for r in all_crossed if not r.is_profitable_at_cross]

    return {
        "total_trades": len(results),
        "trades_crossed": len(all_crossed),
        "trades_never_crossed": len(all_never),
        "profitable_at_cross": len(profitable_cross_all),
        "ghost_crossings": len(ghost_cross_all),
        "ghost_cross_pct": round(len(ghost_cross_all) / max(len(all_crossed), 1) * 100, 1),
        "actual_total_pnl": actual_total_all,
        "midline_ev_total_pnl": midline_ev_all,
        "pnl_delta": midline_ev_all - actual_total_all,
        "first_cross_stats": _cross_stats(cross_bars_all) if cross_bars_all else None,
        "first_cross_distribution": _bar_distribution(np.array(cross_bars_all)) if cross_bars_all else {},
        "by_exit_reason": stratified,
    }


def _cross_stats(bars: list[int]) -> dict:
    arr = np.array(bars)
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr)),
        "min": int(np.min(arr)),
        "max": int(np.max(arr)),
        "p25": float(np.percentile(arr, 25)),
        "p75": float(np.percentile(arr, 75)),
    }


def _bar_distribution(bars: np.ndarray) -> dict[str, int]:
    dist = {}
    for lo, hi, label in [(0, 3, "0-3"), (4, 6, "4-6"), (7, 12, "7-12"), (13, 24, "13-24"), (25, 40, "25-40"), (41, 60, "41-60")]:
        dist[label] = int(((bars >= lo) & (bars <= hi)).sum())
    return dist


def _analyze_one_trade(t: TradeRecord) -> CrossResult:
    """Analyze midline crossings for ONE trade (any exit reason).

    Key addition: tracks whether the price is profitable at the crossing point.
    A "ghost crossing" is when midline drops INTO the price (indicator decay),
    not when the price rises to meet midline (true reversion).
    """
    closes = t.trade_closes
    midlines = t.trade_midlines

    # Compute PnL at each bar (ratio)
    if t.direction == 1:  # LONG
        pnl_ratios = (closes - t.entry_price) / t.entry_price
    else:  # SHORT
        pnl_ratios = (t.entry_price - closes) / t.entry_price

    crossings = []
    for i in range(len(closes)):
        if pd.isna(midlines[i]) or midlines[i] <= 0:
            continue

        if t.direction == 1:
            if closes[i] >= midlines[i]:
                crossings.append({"bar": i, "pnl_ratio": float(pnl_ratios[i])})
        else:
            if closes[i] <= midlines[i]:
                crossings.append({"bar": i, "pnl_ratio": float(pnl_ratios[i])})

    if not crossings:
        # Never crossed midline — use actual final PnL
        if t.direction == 1:
            final_pnl = (t.exit_price - t.entry_price) / t.entry_price * FIXED_NOTIONAL
        else:
            final_pnl = (t.entry_price - t.exit_price) / t.entry_price * FIXED_NOTIONAL
        return CrossResult(
            symbol=t.symbol, direction=t.direction, exit_reason=t.exit_reason,
            entry_price=t.entry_price, exit_price=t.exit_price,
            hold_bars=len(closes) - 1,
            first_cross_bar=None, total_crossings=0,
            is_profitable_at_cross=False,
            pnl_at_cross=final_pnl,  # never crossed → keep actual PnL
            final_pnl=final_pnl,
            max_pnl_after_cross=final_pnl,
        )

    first = crossings[0]
    first_cross_bar = first["bar"]
    is_profitable = first["pnl_ratio"] > 0

    # PnL if exit at first cross
    pnl_at_cross = first["pnl_ratio"] * FIXED_NOTIONAL

    # Actual final PnL
    if t.direction == 1:
        final_pnl = (t.exit_price - t.entry_price) / t.entry_price * FIXED_NOTIONAL
    else:
        final_pnl = (t.entry_price - t.exit_price) / t.entry_price * FIXED_NOTIONAL

    # Max PnL achievable after first cross
    max_after = float(max(pnl_ratios[first_cross_bar:])) * FIXED_NOTIONAL

    return CrossResult(
        symbol=t.symbol, direction=t.direction, exit_reason=t.exit_reason,
        entry_price=t.entry_price, exit_price=t.exit_price,
        hold_bars=len(closes) - 1,
        first_cross_bar=first_cross_bar,
        total_crossings=len(crossings),
        is_profitable_at_cross=is_profitable,
        pnl_at_cross=pnl_at_cross,
        final_pnl=final_pnl,
        max_pnl_after_cross=max_after,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(description="MR-v1 Donchian Midline Analysis")
    p.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS))
    p.add_argument("--start", default=DEFAULT_START)
    p.add_argument("--end", default=DEFAULT_END)
    p.add_argument("--timezone", default=DEFAULT_TIMEZONE)
    p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    p.add_argument("--database-path", default=str(DEFAULT_DATABASE_PATH))
    p.add_argument("--verbose", action="store_true")
    return p.parse_args(argv)


def main(argv=None):
    ensure_headless_runtime()
    args = parse_args(argv)
    logger = setup_logging("research_mr_v1_midline", verbose=bool(args.verbose))
    symbols = [s.strip() for s in re.split(r"[\s,]+", args.symbols) if s.strip()]
    out = Path(args.output_dir) if Path(args.output_dir).is_absolute() else PROJECT_ROOT / args.output_dir
    db = Path(args.database_path) if Path(args.database_path).is_absolute() else PROJECT_ROOT / args.database_path
    hr = parse_history_range(args.start, args.end, timedelta(minutes=1), args.timezone)

    all_trades: list[TradeRecord] = []
    for sym in symbols:
        log_event(logger, logging.INFO, "backtest", "Running backtest with tracking", symbol=sym)
        bars_1m = load_1m(sym, hr, db)
        bars_4h = resample_4h(bars_1m, hr)
        if bars_4h.empty:
            log_event(logger, logging.WARNING, "no_data", "No 4h bars", symbol=sym)
            continue
        trades = run_backtest_with_tracking(bars_4h, sym)
        all_trades.extend(trades)

    if not all_trades:
        logger.error("No trades generated")
        return 1

    result = analyze_all_trades(all_trades)

    out.mkdir(parents=True, exist_ok=True)
    (out / "midline_analysis.json").write_text(
        json.dumps(to_jsonable(result), ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n========== Donchian Midline Analysis (Full Sample, All Exit Reasons) ==========")
    print(f"Total trades analyzed: {result.get('total_trades', 0)}")
    print(f"Trades that crossed midline: {result.get('trades_crossed', 0)}")
    print(f"Trades that NEVER crossed: {result.get('trades_never_crossed', 0)}")
    print(f"True reversions (profitable at cross): {result.get('profitable_at_cross', 0)}")
    print(f"GHOST crossings (indicator decay, NOT profitable): {result.get('ghost_crossings', 0)} ({result.get('ghost_cross_pct', 0)}%)")
    print(f"\nActual total PnL: ${result.get('actual_total_pnl', 0):,.0f}")
    print(f"Midline-exit EV:    ${result.get('midline_ev_total_pnl', 0):,.0f}")
    print(f"Net delta:          ${result.get('pnl_delta', 0):,.0f}")

    # Full-sample crossing timing
    fcs = result.get("first_cross_stats")
    if fcs:
        print(f"\nFirst midline crossing timing (all trades):")
        print(f"  Mean: {fcs['mean']:.1f} bars ({fcs['mean']*4:.0f}h)")
        print(f"  Median: {fcs['median']:.0f} bars ({fcs['median']*4:.0f}h)")
        print(f"  25th-75th: {fcs['p25']:.0f} - {fcs['p75']:.0f} bars")
        print(f"\n  Bar distribution:")
        for label, count in result.get("first_cross_distribution", {}).items():
            bar = "█" * min(count, 60)
            print(f"    {label:>6}: {count:>4} {bar}")

    # Stratified by exit reason
    by_reason = result.get("by_exit_reason", {})
    for reason in ["stop", "max_hold", "end_of_data"]:
        s = by_reason.get(reason)
        if not s:
            continue
        print(f"\n--- {reason.upper()} exits ({s['total']} trades) ---")
        print(f"  Crossed: {s['crossed']}, Never: {s['never_crossed']}")
        print(f"  True reversions at cross: {s['profitable_at_cross']}, Ghost: {s['ghost_cross']}")
        print(f"  Actual PnL: ${s['actual_total_pnl']:,.0f} (avg ${s['avg_actual_pnl']:,.0f}/trade)")
        print(f"  Midline EV:  ${s['midline_ev_total_pnl']:,.0f} (avg ${s['avg_midline_pnl']:,.0f}/trade)")
        print(f"  Delta:       ${s['pnl_delta']:,.0f}")

    log_event(logger, logging.INFO, "done", "Midline analysis complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
