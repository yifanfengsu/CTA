#!/usr/bin/env python3
"""MR-v1 Filter Research: ADX + EMA slope entry filters.

Tests whether filtering out strong-trend entries improves the strategy.
Grid search over: adx_period, adx_threshold, ema_period, slope_floor.

Entry gates:
  1. ADX > threshold → BLOCK (trend too strong for mean reversion)
  2. EMA slope direction → SHORT blocked if slope > 0 (uptrend, don't short)
                           LONG blocked if slope < 0 (downtrend, don't long)
  slope_floor creates a "neutral zone" where both directions are allowed.

v1.2 baseline (no filter): PnL +$4,048, Stop PnL -$36,213, 2216 trades.
Goal: reduce stop losses while keeping >500 trades.
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
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "reports" / "research" / "mr_v1_filter"
DEFAULT_DATABASE_PATH = PROJECT_ROOT / ".vntrader" / "database.db"

LOOKBACK = 8
ATR_MULT = 1.0
MAX_HOLD = 60
ATR_PERIOD = 14
FIXED_NOTIONAL = 1000.0
FEE_BPS = 5.0
SLIPPAGE_BPS = 5.0

SPLIT_RANGES = {
    "train_ext": ("2023-01-01", "2024-07-01"),
    "validation_ext": ("2024-07-01", "2025-07-01"),
    "oos_ext": ("2025-07-01", "2026-04-01"),
}

# Grid search space
GRID_ADX_PERIOD = [10, 14]
GRID_ADX_THRESHOLD = [25.0, 28.0, 32.0]
GRID_EMA_PERIOD = [120, 168]
GRID_SLOPE_FLOOR = [0.0005, 0.001]


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
# Indicators (pure pandas, no ta-lib dependency)
# ---------------------------------------------------------------------------

def compute_adx(bars: pd.DataFrame, period: int = 14) -> pd.Series:
    """Pure-pandas ADX. Returns NaN until enough data is available."""
    high, low, close = bars["high"], bars["low"], bars["close"]

    # True Range
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()

    # Directional Movement
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low

    plus_dm = pd.Series(0.0, index=bars.index)
    minus_dm = pd.Series(0.0, index=bars.index)

    mask_up = (up_move > down_move) & (up_move > 0)
    mask_down = (down_move > up_move) & (down_move > 0)

    plus_dm[mask_up] = up_move[mask_up]
    minus_dm[mask_down] = down_move[mask_down]

    # Smoothed DMs
    plus_di = 100 * (plus_dm.rolling(period).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(period).mean() / atr)

    # ADX
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    adx = dx.rolling(period).mean()

    return adx


def compute_ema_slope(bars: pd.DataFrame, period: int = 120, slope_lookback: int = 3) -> pd.Series:
    """EMA slope: (EMA(t) - EMA(t-slope_lookback)) / EMA(t-slope_lookback).

    slope > 0 → uptrend bias (don't short)
    slope < 0 → downtrend bias (don't long)
    """
    ema = bars["close"].ewm(span=period, adjust=False).mean()
    slope = (ema - ema.shift(slope_lookback)) / ema.shift(slope_lookback)
    return slope


def compute_indicators(bars: pd.DataFrame, adx_period: int, ema_period: int, slope_lookback: int = 3) -> pd.DataFrame:
    """Compute all filter indicators in one pass. Returns DataFrame with added columns."""
    out = bars.copy()
    out["adx"] = compute_adx(bars, adx_period)
    out["ema_slope"] = compute_ema_slope(bars, ema_period, slope_lookback)
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
    filtered: bool = False   # True if signal was blocked by filter
    filter_reason: str = ""  # "adx" or "slope" or ""


# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------

def run_filtered_backtest(
    bars_4h: pd.DataFrame,
    symbol: str,
    adx_threshold: float,
    slope_floor: float,
) -> list[Trade]:
    """Run MR-v1 backtest with ADX + EMA slope entry filters.

    Bars must already have 'adx' and 'ema_slope' columns added.
    """
    if len(bars_4h) < LOOKBACK + 2:
        return []

    # Use pre-computed ATR if not already present
    h, l, c = bars_4h["high"], bars_4h["low"], bars_4h["close"]
    tr = pd.concat([(h - l).abs(), (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    atr = tr.rolling(ATR_PERIOD).mean()

    highs = bars_4h["high"].rolling(LOOKBACK).max().shift(1)
    lows = bars_4h["low"].rolling(LOOKBACK).min().shift(1)
    close = bars_4h["close"]

    trades: list[Trade] = []
    pos = 0
    entry_bar = -1
    entry_price = 0.0
    entry_atr = 0.0

    blocked_adx = 0
    blocked_slope = 0

    for i in range(LOOKBACK, len(bars_4h)):
        bar = bars_4h.iloc[i]

        if pos != 0:
            hold_bars = i - entry_bar
            stop_dist = ATR_MULT * entry_atr
            exit_now = False
            reason = ""

            # Midline take-profit
            dh = highs.iloc[i] if i < len(highs) and not pd.isna(highs.iloc[i]) else 0
            dl = lows.iloc[i] if i < len(lows) and not pd.isna(lows.iloc[i]) else 0
            if dh > 0 and dl > 0:
                midline = (dh + dl) / 2
                if (pos == 1 and bar["close"] >= midline) or (pos == -1 and bar["close"] <= midline):
                    exit_now = True
                    reason = "midline"

            if not exit_now:
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
                if reason == "stop":
                    exit_price = min(bar["open"], entry_price - stop_dist) if pos == 1 else max(bar["open"], entry_price + stop_dist)
                else:
                    exit_price = bar["close"]
                trades.append(Trade(
                    entry_time=bars_4h["datetime"].iloc[entry_bar],
                    exit_time=bar["datetime"], direction=pos,
                    entry_price=entry_price, exit_price=exit_price,
                    exit_reason=reason, symbol=symbol,
                ))
                pos = 0
                continue

        if pos == 0:
            # Check entry signals
            lb = close.iloc[i] > highs.iloc[i]
            sb = close.iloc[i] < lows.iloc[i]

            if not (lb or sb):
                continue

            # --- Filter gates ---
            adx_val = bars_4h["adx"].iloc[i]
            slope_val = bars_4h["ema_slope"].iloc[i]

            # Skip if indicators not ready
            if pd.isna(adx_val) or pd.isna(slope_val):
                continue

            is_trend_strong = adx_val > adx_threshold

            # Direction filters with slope floor buffer
            can_short = not is_trend_strong and (slope_val <= slope_floor)
            can_long = not is_trend_strong and (slope_val >= -slope_floor)

            if lb:
                if not can_short:
                    filter_reason = "adx" if is_trend_strong else "slope"
                    trades.append(Trade(
                        entry_time=pd.NaT, exit_time=pd.NaT, direction=-1,
                        entry_price=0, exit_price=0, exit_reason="filtered",
                        symbol=symbol, filtered=True, filter_reason=filter_reason,
                    ))
                    if is_trend_strong:
                        blocked_adx += 1
                    else:
                        blocked_slope += 1
                    continue
                pos = -1
                entry_bar = i
                entry_price = bar["close"]
                entry_atr = atr.iloc[i] if i < len(atr) and not pd.isna(atr.iloc[i]) and atr.iloc[i] > 0 else bar["close"] * 0.01

            elif sb:
                if not can_long:
                    filter_reason = "adx" if is_trend_strong else "slope"
                    trades.append(Trade(
                        entry_time=pd.NaT, exit_time=pd.NaT, direction=1,
                        entry_price=0, exit_price=0, exit_reason="filtered",
                        symbol=symbol, filtered=True, filter_reason=filter_reason,
                    ))
                    if is_trend_strong:
                        blocked_adx += 1
                    else:
                        blocked_slope += 1
                    continue
                pos = 1
                entry_bar = i
                entry_price = bar["close"]
                entry_atr = atr.iloc[i] if i < len(atr) and not pd.isna(atr.iloc[i]) and atr.iloc[i] > 0 else bar["close"] * 0.01

    # Close open position
    if pos != 0:
        last = bars_4h.iloc[-1]
        trades.append(Trade(
            entry_time=bars_4h["datetime"].iloc[entry_bar],
            exit_time=last["datetime"], direction=pos,
            entry_price=entry_price, exit_price=last["close"],
            exit_reason="end_of_data", symbol=symbol,
        ))

    return trades


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(real_trades: list[Trade], funding_map: dict, tz_name: str) -> dict[str, Any]:
    """Compute metrics on actual (non-filtered) trades only."""
    trades = [t for t in real_trades if not t.filtered]
    if not trades:
        return {"total_trades": 0, "total_pnl": 0, "sharpe": 0, "max_drawdown_pct": 0}

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
            "entry_time": t.entry_time, "exit_time": t.exit_time,
            "direction": t.direction, "entry_price": t.entry_price,
            "exit_price": t.exit_price, "exit_reason": t.exit_reason,
            "symbol": t.symbol,
            "no_cost_pnl": no_cost, "cost_aware_pnl": cost_aware,
            "funding_adjusted_pnl": cost_aware - funding_paid,
        })

    df = pd.DataFrame(records)
    df["date"] = df["exit_time"].dt.date
    daily = df.groupby("date")["funding_adjusted_pnl"].sum()
    if len(daily) >= 2:
        all_dates = pd.date_range(start=daily.index.min(), end=daily.index.max(), freq="D").date
        daily = daily.reindex(all_dates, fill_value=0.0)
    equity = daily.cumsum()

    total_pnl = float(equity.iloc[-1]) if len(equity) > 0 else 0.0
    daily_returns = daily / FIXED_NOTIONAL
    if len(daily_returns) > 1:
        sharpe = float(np.sqrt(252) * daily_returns.mean() / daily_returns.std()) if daily_returns.std() > 0 else 0.0
    else:
        sharpe = 0.0

    peak = equity.cummax()
    dd = (equity - peak) / FIXED_NOTIONAL
    max_dd = float(dd.min()) if len(dd) > 0 else 0.0
    win_rate = float((df["funding_adjusted_pnl"] > 0).mean()) if len(df) > 0 else 0.0

    # Exit breakdown
    exit_breakdown = {}
    for reason, grp in df.groupby("exit_reason"):
        exit_breakdown[reason] = {"count": len(grp), "total_pnl": float(grp["funding_adjusted_pnl"].sum())}

    return {
        "total_trades": len(df),
        "total_pnl_funding_adjusted": total_pnl,
        "sharpe_ratio": round(sharpe, 2),
        "max_drawdown_pct": round(max_dd * 100, 2),
        "win_rate": round(win_rate * 100, 1),
        "by_exit_reason": exit_breakdown,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_variant(bars_map: dict[str, pd.DataFrame], funding_map: dict, tz: str,
                adx_period: int, adx_threshold: float, ema_period: int, slope_floor: float) -> dict:
    """Run one parameter combination."""
    all_trades: list[Trade] = []
    for sym, bars in bars_map.items():
        # Add indicators
        bars_with_ind = compute_indicators(bars, adx_period, ema_period)
        trades = run_filtered_backtest(bars_with_ind, sym, adx_threshold, slope_floor)
        all_trades.extend(trades)

    real = [t for t in all_trades if not t.filtered]
    filtered = [t for t in all_trades if t.filtered]
    blocked_adx = sum(1 for t in filtered if t.filter_reason == "adx")
    blocked_slope = sum(1 for t in filtered if t.filter_reason == "slope")

    m = compute_metrics(all_trades, funding_map, tz)
    stop_info = m.get("by_exit_reason", {}).get("stop", {})
    midline_info = m.get("by_exit_reason", {}).get("midline", {})

    return {
        "adx_period": adx_period,
        "adx_threshold": adx_threshold,
        "ema_period": ema_period,
        "slope_floor": slope_floor,
        "total_signals": len(all_trades),
        "filtered_adx": blocked_adx,
        "filtered_slope": blocked_slope,
        "filtered_total": len(filtered),
        "actual_trades": len(real),
        "total_pnl": m["total_pnl_funding_adjusted"],
        "sharpe": m["sharpe_ratio"],
        "win_rate": m["win_rate"],
        "max_dd": m["max_drawdown_pct"],
        "stop_trades": stop_info.get("count", 0),
        "stop_pnl": stop_info.get("total_pnl", 0),
        "midline_trades": midline_info.get("count", 0),
        "midline_pnl": midline_info.get("total_pnl", 0),
    }


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="MR-v1 Filter Research: ADX + EMA slope entry gates")
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
    logger = setup_logging("research_mr_v1_filter", verbose=bool(args.verbose))
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

    # Load all 4h data once
    bars_map: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        log_event(logger, logging.INFO, "load_data", "Loading data", symbol=sym)
        bars_1m = load_1m(sym, hr, db)
        bars_4h = resample_4h(bars_1m, hr)
        if not bars_4h.empty:
            bars_map[sym] = bars_4h
        else:
            log_event(logger, logging.WARNING, "no_data", "No 4h bars", symbol=sym)

    if not bars_map:
        logger.error("No data loaded")
        return 1

    # Build grid
    grid = []
    for adx_p in GRID_ADX_PERIOD:
        for adx_t in GRID_ADX_THRESHOLD:
            for ema_p in GRID_EMA_PERIOD:
                for sf in GRID_SLOPE_FLOOR:
                    grid.append((adx_p, adx_t, ema_p, sf))

    total_combos = len(grid)
    results = []

    for idx, (adx_p, adx_t, ema_p, sf) in enumerate(grid):
        log_event(logger, logging.INFO, "run_grid", f"[{idx+1}/{total_combos}] Running",
                  adx_period=adx_p, adx_threshold=adx_t, ema_period=ema_p, slope_floor=sf)
        r = run_variant(bars_map, funding_map, args.timezone, adx_p, adx_t, ema_p, sf)
        results.append(r)

    # Sort by total PnL descending
    results.sort(key=lambda x: x["total_pnl"], reverse=True)

    out.mkdir(parents=True, exist_ok=True)
    (out / "filter_grid_results.json").write_text(json.dumps(to_jsonable(results), ensure_ascii=False, indent=2), encoding="utf-8")

    # Print results table
    print(f"\n{'='*130}")
    print(f"MR-v1 Filter Grid Search: {total_combos} combinations")
    print(f"{'='*130}")
    print(f"{'ADX_P':>5} {'ADX_T':>6} {'EMA_P':>5} {'Floor':>8} {'Signals':>7} {'Filt_ADX':>8} {'Filt_SLP':>8} {'Trades':>7} {'PnL':>10} {'Sharpe':>7} {'Win%':>6} {'Stop_T':>7} {'Stop_PnL':>10} {'Mid_T':>7} {'Mid_PnL':>10}")
    print(f"{'-'*130}")

    for r in results:
        print(f"{r['adx_period']:>5} {r['adx_threshold']:>6.1f} {r['ema_period']:>5} {r['slope_floor']:>8.4f} "
              f"{r['total_signals']:>7} {r['filtered_adx']:>8} {r['filtered_slope']:>8} "
              f"{r['actual_trades']:>7} ${r['total_pnl']:>9,.0f} {r['sharpe']:>7.2f} {r['win_rate']:>5.1f}% "
              f"{r['stop_trades']:>7} ${r['stop_pnl']:>10,.0f} {r['midline_trades']:>7} ${r['midline_pnl']:>10,.0f}")

    # Top 5 summary
    print(f"\n{'='*80}")
    print("TOP 5 by Total PnL:")
    for i, r in enumerate(results[:5]):
        print(f"  #{i+1}: adx_period={r['adx_period']}, adx_threshold={r['adx_threshold']}, "
              f"ema_period={r['ema_period']}, slope_floor={r['slope_floor']} → "
              f"PnL=${r['total_pnl']:,.0f}, Trades={r['actual_trades']}, "
              f"STOP PnL=${r['stop_pnl']:,.0f} ({r['stop_trades']} trades), "
              f"Filtered: ADX={r['filtered_adx']}, Slope={r['filtered_slope']}")

    log_event(logger, logging.INFO, "done", "Filter grid search complete", combinations=total_combos)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
