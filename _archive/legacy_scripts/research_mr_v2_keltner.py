#!/usr/bin/env python3
"""MR-v2 Keltner Channel Strategy Research.

Replaces Donchian Channel with Keltner Channel (EMA + ATR bands).
Entry: right-side re-entry — price was outside band, now back inside = fade.
Exit: midline (EMA) take-profit, dynamic ATR stop at outer band, max_hold.

Grid search over: ema_period, atr_mult, atr_period.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sqlite3
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
DEFAULT_FUNDING_DIR = PROJECT_ROOT / "data" / "funding" / "okx"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "reports" / "research" / "mr_v2_keltner"
DEFAULT_DATABASE_PATH = PROJECT_ROOT / ".vntrader" / "database.db"

MAX_HOLD = 60
FIXED_NOTIONAL = 1000.0
FEE_BPS = 5.0
SLIPPAGE_BPS = 5.0

# Grid search space
GRID_EMA_PERIOD = [20, 30]
GRID_ATR_MULT = [1.5, 2.0, 2.5]
GRID_ATR_PERIOD = [14, 20]


class ResearchError(Exception):
    pass


# ---------------------------------------------------------------------------
# Data (same as existing scripts)
# ---------------------------------------------------------------------------

def split_vt_symbol(vt: str) -> tuple[str, str]:
    s, sep, e = str(vt).partition(".")
    if not sep:
        raise ResearchError(f"bad vt_symbol: {vt}")
    return s, e


def symbol_to_inst_id(vt: str) -> str:
    s, _ = split_vt_symbol(vt)
    r = s.removesuffix("_OKX")
    p = r[:-len("_SWAP")] if r.endswith("_SWAP") else r
    return f"{p[:-4]}-USDT-SWAP" if p.endswith("USDT") else r.replace("_", "-")


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


def load_funding(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if df.empty:
        return pd.DataFrame(columns=["funding_time_utc", "funding_rate"])
    if "funding_time_utc" in df.columns:
        ts = pd.to_datetime(df["funding_time_utc"], utc=True, errors="coerce")
    elif "funding_time" in df.columns:
        ts = pd.to_datetime(pd.to_numeric(df["funding_time"], errors="coerce"), unit="ms", utc=True, errors="coerce")
    else:
        return pd.DataFrame(columns=["funding_time_utc", "funding_rate"])
    r = pd.DataFrame({"funding_time_utc": ts, "funding_rate": pd.to_numeric(df.get("funding_rate"), errors="coerce")})
    return r.dropna().sort_values("funding_time_utc", kind="stable").drop_duplicates("funding_time_utc", keep="last").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Keltner Channel (pure pandas)
# ---------------------------------------------------------------------------

def compute_keltner(bars: pd.DataFrame, ema_period: int, atr_period: int, atr_mult: float) -> pd.DataFrame:
    """Compute Keltner Channel: EMA midline + ATR bands."""
    out = bars.copy()
    close = bars["close"]
    high, low = bars["high"], bars["low"]

    # EMA midline
    out["ema_mid"] = close.ewm(span=ema_period, adjust=False).mean()

    # ATR
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    out["atr"] = tr.rolling(atr_period).mean()

    # Bands
    out["upper"] = out["ema_mid"] + atr_mult * out["atr"]
    out["lower"] = out["ema_mid"] - atr_mult * out["atr"]

    # Bandwidth (for potential squeeze filter)
    out["bandwidth"] = (out["upper"] - out["lower"]) / out["ema_mid"]

    return out


# ---------------------------------------------------------------------------
# Trade record
# ---------------------------------------------------------------------------

@dataclass
class Trade:
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    direction: int
    entry_price: float
    exit_price: float
    exit_reason: str
    symbol: str


# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------

def run_keltner_backtest(bars: pd.DataFrame, symbol: str, ema_period: int, atr_period: int, atr_mult: float) -> list[Trade]:
    """Run Keltner Channel MR backtest.

    Entry: price was outside band last bar, closed inside this bar (fade).
    Exit: midline (EMA) profit > ATR stop at outer band > max_hold.
    """
    # Bars must already have keltner columns
    min_bars = max(ema_period, atr_period) + 5
    if len(bars) < min_bars:
        return []

    ema_mid = bars["ema_mid"]
    upper = bars["upper"]
    lower = bars["lower"]
    atr = bars["atr"]
    close = bars["close"]

    trades: list[Trade] = []
    pos = 0
    entry_bar = -1
    entry_price = 0.0
    entry_ema = 0.0    # EMA midline at entry (for take-profit)
    entry_atr = 0.0

    for i in range(min_bars, len(bars)):
        bar = bars.iloc[i]

        # Skip if indicators not ready
        if pd.isna(ema_mid.iloc[i]) or pd.isna(lower.iloc[i]) or pd.isna(upper.iloc[i]):
            continue

        if pos != 0:
            hold_bars = i - entry_bar
            exit_now = False
            reason = ""

            # 1. Midline take-profit: price crosses EMA midline
            if (pos == 1 and bar["close"] >= ema_mid.iloc[i]) or (pos == -1 and bar["close"] <= ema_mid.iloc[i]):
                exit_now = True
                reason = "midline"

            # 2. Dynamic ATR stop: stop at outer band
            if not exit_now:
                if pos == 1:
                    stop_price = lower.iloc[i]  # long stop at lower band
                    if bar["low"] <= stop_price:
                        exit_now = True
                        reason = "stop"
                else:
                    stop_price = upper.iloc[i]  # short stop at upper band
                    if bar["high"] >= stop_price:
                        exit_now = True
                        reason = "stop"

            # 3. Max hold
            if hold_bars >= MAX_HOLD and not exit_now:
                exit_now = True
                reason = "max_hold"

            if exit_now:
                if reason == "stop":
                    if pos == 1:
                        exit_price = min(bar["open"], lower.iloc[i])
                    else:
                        exit_price = max(bar["open"], upper.iloc[i])
                else:
                    exit_price = bar["close"]
                trades.append(Trade(
                    entry_time=bars["datetime"].iloc[entry_bar],
                    exit_time=bar["datetime"], direction=pos,
                    entry_price=entry_price, exit_price=exit_price,
                    exit_reason=reason, symbol=symbol,
                ))
                pos = 0
                continue

        if pos == 0:
            # Right-side re-entry pattern:
            # Previous bar close was OUTSIDE the band, current bar close is INSIDE → fade
            prev_close = close.iloc[i - 1]
            prev_upper = upper.iloc[i - 1]
            prev_lower = lower.iloc[i - 1]
            cur_close = bar["close"]

            if pd.isna(prev_close) or pd.isna(prev_upper) or pd.isna(prev_lower):
                continue

            # SHORT: prev close > upper, now close <= upper (pierced upper, came back)
            short_signal = prev_close > prev_upper and cur_close <= upper.iloc[i]
            # LONG: prev close < lower, now close >= lower (pierced lower, came back)
            long_signal = prev_close < prev_lower and cur_close >= lower.iloc[i]

            if short_signal:
                pos = -1
                entry_bar = i
                entry_price = cur_close
                entry_ema = ema_mid.iloc[i]
                entry_atr = atr.iloc[i] if not pd.isna(atr.iloc[i]) and atr.iloc[i] > 0 else cur_close * 0.01

            elif long_signal:
                pos = 1
                entry_bar = i
                entry_price = cur_close
                entry_ema = ema_mid.iloc[i]
                entry_atr = atr.iloc[i] if not pd.isna(atr.iloc[i]) and atr.iloc[i] > 0 else cur_close * 0.01

    # Close open position
    if pos != 0:
        last = bars.iloc[-1]
        trades.append(Trade(
            entry_time=bars["datetime"].iloc[entry_bar],
            exit_time=last["datetime"], direction=pos,
            entry_price=entry_price, exit_price=last["close"],
            exit_reason="end_of_data", symbol=symbol,
        ))
    return trades


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(trades: list[Trade], funding_map: dict, tz_name: str) -> dict[str, Any]:
    if not trades:
        return {"total_trades": 0, "total_pnl": 0, "sharpe": 0, "win_rate": 0}
    cost_per_trade = FIXED_NOTIONAL * (2 * FEE_BPS + 2 * SLIPPAGE_BPS) / 10000.0
    import zoneinfo

    records = []
    for t in trades:
        if t.direction == 1:
            ret = (t.exit_price - t.entry_price) / t.entry_price
        else:
            ret = (t.entry_price - t.exit_price) / t.entry_price
        no_cost = ret * FIXED_NOTIONAL
        cost_aware = no_cost - cost_per_trade

        tz = zoneinfo.ZoneInfo(tz_name)
        et = t.entry_time.tz_convert("UTC") if t.entry_time.tzinfo else t.entry_time.tz_localize(tz).tz_convert("UTC")
        xt = t.exit_time.tz_convert("UTC") if t.exit_time.tzinfo else t.exit_time.tz_localize(tz).tz_convert("UTC")
        inst = symbol_to_inst_id(t.symbol)
        fund = funding_map.get(inst)
        funding_paid = 0.0
        if fund is not None and not fund.empty:
            mask = (fund["funding_time_utc"] >= et) & (fund["funding_time_utc"] < xt)
            if mask.any():
                fp = fund.loc[mask, "funding_rate"].sum() * FIXED_NOTIONAL
                funding_paid = fp if t.direction == 1 else -fp
        records.append({
            "exit_reason": t.exit_reason, "symbol": t.symbol,
            "no_cost_pnl": no_cost, "cost_aware_pnl": cost_aware,
            "funding_adjusted_pnl": cost_aware - funding_paid,
        })

    df = pd.DataFrame(records)
    df["date"] = df.index  # placeholder, actual dates from trades not needed for aggregate metrics

    wins = df[df["funding_adjusted_pnl"] > 0]["funding_adjusted_pnl"].sum()
    losses = abs(df[df["funding_adjusted_pnl"] < 0]["funding_adjusted_pnl"].sum())
    pf = float(wins / losses) if losses > 0 else float("inf")
    win_rate = float((df["funding_adjusted_pnl"] > 0).mean()) * 100 if len(df) > 0 else 0

    exit_breakdown = {}
    for reason, grp in df.groupby("exit_reason"):
        exit_breakdown[reason] = {"count": len(grp), "total_pnl": float(grp["funding_adjusted_pnl"].sum())}

    return {
        "total_trades": len(df),
        "total_pnl_funding_adjusted": float(df["funding_adjusted_pnl"].sum()),
        "profit_factor": round(pf, 2),
        "win_rate": round(win_rate, 1),
        "by_exit_reason": exit_breakdown,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(description="MR-v2 Keltner Channel Strategy Research")
    p.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS))
    p.add_argument("--start", default=DEFAULT_START)
    p.add_argument("--end", default=DEFAULT_END)
    p.add_argument("--timezone", default=DEFAULT_TIMEZONE)
    p.add_argument("--funding-dir", default=str(DEFAULT_FUNDING_DIR))
    p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    p.add_argument("--database-path", default=str(DEFAULT_DATABASE_PATH))
    p.add_argument("--verbose", action="store_true")
    return p.parse_args(argv)


def main(argv=None):
    ensure_headless_runtime()
    args = parse_args(argv)
    logger = setup_logging("research_mr_v2_keltner", verbose=bool(args.verbose))
    symbols = [s.strip() for s in re.split(r"[\s,]+", args.symbols) if s.strip()]
    out = Path(args.output_dir) if Path(args.output_dir).is_absolute() else PROJECT_ROOT / args.output_dir
    db = Path(args.database_path) if Path(args.database_path).is_absolute() else PROJECT_ROOT / args.database_path
    fdir = Path(args.funding_dir) if Path(args.funding_dir).is_absolute() else PROJECT_ROOT / args.funding_dir
    hr = parse_history_range(args.start, args.end, timedelta(minutes=1), args.timezone)

    # Load funding
    funding_map = {}
    for sym in symbols:
        inst = symbol_to_inst_id(sym)
        cf = fdir / f"{inst}_funding_{DEFAULT_START}_{DEFAULT_END}.csv"
        if cf.exists():
            funding_map[inst] = load_funding(cf)
        else:
            matches = sorted(fdir.glob(f"{inst}_funding_*.csv"))
            if matches:
                funding_map[inst] = load_funding(matches[-1])

    # Load all 4h data
    raw_bars: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        log_event(logger, logging.INFO, "load_data", "Loading data", symbol=sym)
        bars_1m = load_1m(sym, hr, db)
        bars_4h = resample_4h(bars_1m, hr)
        if not bars_4h.empty:
            raw_bars[sym] = bars_4h
        else:
            log_event(logger, logging.WARNING, "no_data", "No 4h bars", symbol=sym)

    if not raw_bars:
        logger.error("No data loaded")
        return 1

    # Build grid
    grid = []
    for ema_p in GRID_EMA_PERIOD:
        for atr_p in GRID_ATR_PERIOD:
            for atr_m in GRID_ATR_MULT:
                grid.append((ema_p, atr_p, atr_m))

    results = []
    for idx, (ema_p, atr_p, atr_m) in enumerate(grid):
        log_event(logger, logging.INFO, "run_grid", f"[{idx+1}/{len(grid)}] Running",
                  ema_period=ema_p, atr_period=atr_p, atr_mult=atr_m)

        all_trades: list[Trade] = []
        for sym, raw in raw_bars.items():
            bars = compute_keltner(raw, ema_p, atr_p, atr_m)
            trades = run_keltner_backtest(bars, sym, ema_p, atr_p, atr_m)
            all_trades.extend(trades)

        m = compute_metrics(all_trades, funding_map, args.timezone)
        stop_info = m.get("by_exit_reason", {}).get("stop", {})
        midline_info = m.get("by_exit_reason", {}).get("midline", {})
        maxhold_info = m.get("by_exit_reason", {}).get("max_hold", {})

        results.append({
            "ema_period": ema_p,
            "atr_period": atr_p,
            "atr_mult": atr_m,
            "total_trades": m["total_trades"],
            "total_pnl": m["total_pnl_funding_adjusted"],
            "pf": m["profit_factor"],
            "win_rate": m["win_rate"],
            "stop_trades": stop_info.get("count", 0),
            "stop_pnl": stop_info.get("total_pnl", 0),
            "midline_trades": midline_info.get("count", 0),
            "midline_pnl": midline_info.get("total_pnl", 0),
            "maxhold_trades": maxhold_info.get("count", 0),
            "maxhold_pnl": maxhold_info.get("total_pnl", 0),
        })

    results.sort(key=lambda x: x["total_pnl"], reverse=True)

    out.mkdir(parents=True, exist_ok=True)
    (out / "keltner_grid_results.json").write_text(json.dumps(to_jsonable(results), ensure_ascii=False, indent=2), encoding="utf-8")

    # Print
    print(f"\n{'='*120}")
    print(f"MR-v2 Keltner Channel Grid Search: {len(grid)} combinations")
    print(f"{'='*120}")
    print(f"{'EMA':>4} {'ATR_P':>5} {'ATR_M':>6} {'Trades':>7} {'PnL':>10} {'PF':>6} {'Win%':>6} {'Stop_T':>7} {'Stop_PnL':>10} {'Mid_T':>7} {'Mid_PnL':>10} {'MH_T':>5} {'MH_PnL':>10}")
    print(f"{'-'*120}")
    for r in results:
        print(f"{r['ema_period']:>4} {r['atr_period']:>5} {r['atr_mult']:>6.1f} "
              f"{r['total_trades']:>7} ${r['total_pnl']:>9,.0f} {r['pf']:>6.2f} {r['win_rate']:>5.1f}% "
              f"{r['stop_trades']:>7} ${r['stop_pnl']:>10,.0f} "
              f"{r['midline_trades']:>7} ${r['midline_pnl']:>10,.0f} "
              f"{r['maxhold_trades']:>5} ${r['maxhold_pnl']:>10,.0f}")

    print(f"\n{'='*60}")
    print("TOP 3:")
    for i, r in enumerate(results[:3]):
        print(f"  #{i+1}: ema={r['ema_period']}, atr_period={r['atr_period']}, atr_mult={r['atr_mult']} → "
              f"PnL=${r['total_pnl']:,.0f}, Trades={r['total_trades']}, Win%={r['win_rate']:.1f}%, "
              f"Stop=${r['stop_pnl']:,.0f} ({r['stop_trades']}t), Mid=${r['midline_pnl']:,.0f} ({r['midline_trades']}t)")

    # Compare vs Donchian baseline
    print(f"\n--- vs MR-v1.2 Donchian Baseline ---")
    print(f"  Donchian:  PnL=+$4,048,  Trades=2216,  Stop=-$36,213 (1208t),  Mid=+$40,262 (1007t)")
    print(f"  Best KC:   PnL=${results[0]['total_pnl']:+,.0f},  Trades={results[0]['total_trades']},  "
          f"Stop=${results[0]['stop_pnl']:+,.0f} ({results[0]['stop_trades']}t),  "
          f"Mid=${results[0]['midline_pnl']:+,.0f} ({results[0]['midline_trades']}t)")

    log_event(logger, logging.INFO, "done", "Keltner grid search complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
