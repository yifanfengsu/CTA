#!/usr/bin/env python3
"""MR-v1 Chandelier Exit Research: compare fixed stop vs trailing stop.

Sweeps tracking distances (0.5, 1.0, 1.5, 2.0 ATR) for Chandelier exit
and compares against baseline fixed-stop exit.
Also tests combined variants: Chandelier + max_hold reductions.
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
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "reports" / "research" / "mr_v1_chandelier"
DEFAULT_DATABASE_PATH = PROJECT_ROOT / ".vntrader" / "database.db"

LOOKBACK = 8
ATR_MULT = 1.0          # Baseline fixed stop
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

# Chandelier parameters to test
CHANDELIER_MULTS = [1.0, 1.5, 2.0, 2.5, 3.0]
MAX_HOLD_VARIANTS = [60, 40, 30, 20, 12]  # For combined tests


class ResearchError(Exception):
    pass


# ---------------------------------------------------------------------------
# Data (reused from backtest_mr_v1.py)
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


def compute_atr(bars: pd.DataFrame, period: int) -> pd.Series:
    h, l, c = bars["high"], bars["low"], bars["close"]
    tr = pd.concat([(h - l).abs(), (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()


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
    exit_detail: str = ""  # e.g. "chandelier_1.0" or "max_hold"


# ---------------------------------------------------------------------------
# Strategy variants
# ---------------------------------------------------------------------------

def _atr_val(i: int, atr: pd.Series, close: float) -> float:
    if i < len(atr) and not pd.isna(atr.iloc[i]) and atr.iloc[i] > 0:
        return atr.iloc[i]
    return close * 0.01


def run_baseline(bars_4h: pd.DataFrame, symbol: str) -> list[Trade]:
    """Baseline fixed-stop exit (current MR-v1)."""
    if len(bars_4h) < LOOKBACK + 2:
        return []
    atr = compute_atr(bars_4h, ATR_PERIOD)
    highs = bars_4h["high"].rolling(LOOKBACK).max().shift(1)
    lows = bars_4h["low"].rolling(LOOKBACK).min().shift(1)
    close = bars_4h["close"]

    trades: list[Trade] = []
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
                # Realistic stop exit: can't fill better than stop price.
                # If gap-down (long: open < stop), forced to fill at open.
                # Otherwise fill at the stop price itself.
                if reason == "stop":
                    if pos == 1:
                        exit_price = min(bar["open"], entry_price - stop_dist)
                    else:
                        exit_price = max(bar["open"], entry_price + stop_dist)
                else:
                    exit_price = bar["close"]
                trades.append(Trade(
                    entry_time=bars_4h["datetime"].iloc[entry_bar],
                    exit_time=bar["datetime"], direction=pos,
                    entry_price=entry_price, exit_price=exit_price,
                    exit_reason=reason, symbol=symbol,
                    exit_detail=f"baseline_{reason}",
                ))
                pos = 0
                continue

        if pos == 0:
            lb = close.iloc[i] > highs.iloc[i]
            sb = close.iloc[i] < lows.iloc[i]
            if lb:
                pos = -1
                entry_bar = i
                entry_price = bar["close"]
                entry_atr = _atr_val(i, atr, entry_price)
            elif sb:
                pos = 1
                entry_bar = i
                entry_price = bar["close"]
                entry_atr = _atr_val(i, atr, entry_price)

    if pos != 0:
        last = bars_4h.iloc[-1]
        trades.append(Trade(
            entry_time=bars_4h["datetime"].iloc[entry_bar],
            exit_time=last["datetime"], direction=pos,
            entry_price=entry_price, exit_price=last["close"],
            exit_reason="end_of_data", symbol=symbol, exit_detail="end_of_data",
        ))
    return trades


def run_chandelier(bars_4h: pd.DataFrame, symbol: str, tracking_atr: float, max_hold: int = MAX_HOLD) -> list[Trade]:
    """Chandelier exit: trailing stop based on running extreme."""
    if len(bars_4h) < LOOKBACK + 2:
        return []
    atr = compute_atr(bars_4h, ATR_PERIOD)
    highs = bars_4h["high"].rolling(LOOKBACK).max().shift(1)
    lows = bars_4h["low"].rolling(LOOKBACK).min().shift(1)
    close = bars_4h["close"]

    trades: list[Trade] = []
    pos = 0
    entry_bar = -1
    entry_price = 0.0
    highest_since = 0.0    # for long positions
    lowest_since = 0.0     # for short positions

    for i in range(LOOKBACK, len(bars_4h)):
        bar = bars_4h.iloc[i]

        if pos != 0:
            hold_bars = i - entry_bar

            # Current ATR — use PREVIOUS bar to avoid look-ahead
            cur_atr = _atr_val(i - 1, atr, bar["close"])
            stop_dist = tracking_atr * cur_atr

            exit_now = False
            reason = ""

            # Check stop using PREVIOUS extreme (before this bar updates it)
            if pos == 1:
                stop_price = highest_since - stop_dist
                if bar["low"] <= stop_price:
                    exit_now = True
                    reason = "chandelier"
            else:
                stop_price = lowest_since + stop_dist
                if bar["high"] >= stop_price:
                    exit_now = True
                    reason = "chandelier"

            if hold_bars >= max_hold and not exit_now:
                exit_now = True
                reason = "max_hold"

            if exit_now:
                # Realistic exit: fill at stop price (or worse if gapped)
                if reason == "chandelier":
                    if pos == 1:
                        exit_price = min(bar["open"], stop_price)
                    else:
                        exit_price = max(bar["open"], stop_price)
                else:
                    exit_price = bar["close"]
                detail = f"chandelier_{tracking_atr}" if reason == "chandelier" else f"max_hold_{max_hold}"
                trades.append(Trade(
                    entry_time=bars_4h["datetime"].iloc[entry_bar],
                    exit_time=bar["datetime"], direction=pos,
                    entry_price=entry_price, exit_price=exit_price,
                    exit_reason=reason, symbol=symbol, exit_detail=detail,
                ))
                pos = 0
                highest_since = 0.0
                lowest_since = 0.0
                continue

            # Only update extremes AFTER surviving this bar
            if pos == 1:
                highest_since = max(highest_since, bar["high"])
            else:
                lowest_since = min(lowest_since, bar["low"])

        if pos == 0:
            lb = close.iloc[i] > highs.iloc[i]
            sb = close.iloc[i] < lows.iloc[i]
            if lb:
                pos = -1
                entry_bar = i
                entry_price = bar["close"]
                lowest_since = bar["close"]
            elif sb:
                pos = 1
                entry_bar = i
                entry_price = bar["close"]
                highest_since = bar["close"]

    if pos != 0:
        last = bars_4h.iloc[-1]
        trades.append(Trade(
            entry_time=bars_4h["datetime"].iloc[entry_bar],
            exit_time=last["datetime"], direction=pos,
            entry_price=entry_price, exit_price=last["close"],
            exit_reason="end_of_data", symbol=symbol, exit_detail="end_of_data",
        ))
    return trades


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(trades: list[Trade], funding_map: dict, tz_name: str) -> dict[str, Any]:
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
            "exit_detail": t.exit_detail, "symbol": t.symbol,
            "no_cost_pnl": no_cost, "cost_aware_pnl": cost_aware,
            "funding_adjusted_pnl": cost_aware - funding_paid,
        })

    df = pd.DataFrame(records)
    df["date"] = df["exit_time"].dt.date

    daily = df.groupby("date")["funding_adjusted_pnl"].sum()
    # Reindex to full calendar — missing days are 0 PnL, not skipped
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
    avg_trade = total_pnl / len(df) if len(df) > 0 else 0.0

    # Profit factor
    wins = df[df["funding_adjusted_pnl"] > 0]["funding_adjusted_pnl"].sum()
    losses = abs(df[df["funding_adjusted_pnl"] < 0]["funding_adjusted_pnl"].sum())
    pf = float(wins / losses) if losses > 0 else float("inf")

    # Exit reason breakdown
    exit_breakdown = {}
    for reason, grp in df.groupby("exit_detail"):
        exit_breakdown[reason] = {"count": len(grp), "total_pnl": float(grp["funding_adjusted_pnl"].sum())}

    return {
        "total_trades": len(df),
        "total_pnl_funding_adjusted": total_pnl,
        "total_pnl_cost_aware": float(df["cost_aware_pnl"].sum()),
        "total_pnl_no_cost": float(df["no_cost_pnl"].sum()),
        "sharpe_ratio": round(sharpe, 2),
        "max_drawdown_pct": round(max_dd * 100, 2),
        "win_rate": round(win_rate * 100, 1),
        "avg_trade_pnl": round(avg_trade, 2),
        "profit_factor": round(pf, 2),
        "by_exit_detail": exit_breakdown,
        "by_exit_reason": {},
    }


def assign_split(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["split"] = "oos_ext"
    for sn, (s, e) in SPLIT_RANGES.items():
        mask = (pd.to_datetime(out["entry_time"]) >= pd.Timestamp(s).tz_localize(DEFAULT_TIMEZONE)) & \
               (pd.to_datetime(out["entry_time"]) < pd.Timestamp(e).tz_localize(DEFAULT_TIMEZONE))
        out.loc[mask, "split"] = sn
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_variant(name: str, trades: list[Trade], funding_map: dict, tz: str) -> dict:
    """Compute metrics for one variant."""
    m = compute_metrics(trades, funding_map, tz)
    trade_df = pd.DataFrame([{
        "entry_time": t.entry_time, "exit_time": t.exit_time,
        "direction": t.direction, "entry_price": t.entry_price,
        "exit_price": t.exit_price, "exit_reason": t.exit_reason,
        "exit_detail": t.exit_detail, "symbol": t.symbol,
    } for t in trades])
    trade_df = assign_split(trade_df)
    oos_mask = trade_df["split"] == "oos_ext"
    oos_trades = [t for i, t in enumerate(trades) if oos_mask.iloc[i]]
    oos_m = compute_metrics(oos_trades, funding_map, tz) if oos_trades else {}
    return {
        "name": name,
        "total_trades": m["total_trades"],
        "overall_pnl": m["total_pnl_funding_adjusted"],
        "overall_sharpe": m["sharpe_ratio"],
        "overall_win_rate": m["win_rate"],
        "overall_pf": m["profit_factor"],
        "overall_max_dd": m["max_drawdown_pct"],
        "overall_avg_trade": m["avg_trade_pnl"],
        "oos_pnl": oos_m.get("total_pnl_funding_adjusted", 0),
        "oos_sharpe": oos_m.get("sharpe_ratio", 0),
        "oos_win_rate": oos_m.get("win_rate", 0),
        "by_exit_detail": m["by_exit_detail"],
        "cost_aware_pnl": m["total_pnl_cost_aware"],
        "no_cost_pnl": m["total_pnl_no_cost"],
    }


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="MR-v1 Chandelier Exit Research")
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
    logger = setup_logging("research_mr_v1_chandelier", verbose=bool(args.verbose))
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
    all_bars: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        log_event(logger, logging.INFO, "load_data", "Loading data", symbol=sym)
        bars_1m = load_1m(sym, hr, db)
        bars_4h = resample_4h(bars_1m, hr)
        if not bars_4h.empty:
            all_bars[sym] = bars_4h
        else:
            log_event(logger, logging.WARNING, "no_data", "No 4h bars", symbol=sym)

    if not all_bars:
        logger.error("No data loaded for any symbol")
        return 1

    results: list[dict] = []

    # --- Baseline ---
    log_event(logger, logging.INFO, "run_variant", "Running", variant="baseline")
    baseline_trades = []
    for sym, bars in all_bars.items():
        baseline_trades.extend(run_baseline(bars, sym))
    results.append(run_variant("baseline (fixed 1.0 ATR, max_hold=60)", baseline_trades, funding_map, args.timezone))

    # --- Chandelier variants ---
    for mult in CHANDELIER_MULTS:
        log_event(logger, logging.INFO, "run_variant", "Running", variant=f"chandelier_{mult}")
        ch_trades = []
        for sym, bars in all_bars.items():
            ch_trades.extend(run_chandelier(bars, sym, mult, MAX_HOLD))
        results.append(run_variant(f"chandelier ({mult} ATR, max_hold={MAX_HOLD})", ch_trades, funding_map, args.timezone))

    # --- Chandelier + reduced max_hold ---
    for mult in CHANDELIER_MULTS:
        for mh in MAX_HOLD_VARIANTS:
            if mh >= MAX_HOLD:
                continue
            log_event(logger, logging.INFO, "run_variant", "Running", variant=f"chandelier_{mult}_mh{mh}")
            ch_trades = []
            for sym, bars in all_bars.items():
                ch_trades.extend(run_chandelier(bars, sym, mult, mh))
            results.append(run_variant(f"chandelier ({mult} ATR, max_hold={mh})", ch_trades, funding_map, args.timezone))

    # --- Output ---
    out.mkdir(parents=True, exist_ok=True)

    # Summary table
    rows = []
    for r in results:
        rows.append({
            "variant": r["name"],
            "trades": r["total_trades"],
            "pnl_overall": r["overall_pnl"],
            "sharpe_overall": r["overall_sharpe"],
            "win_rate": r["overall_win_rate"],
            "pf": r["overall_pf"],
            "max_dd": r["overall_max_dd"],
            "avg_trade": r["overall_avg_trade"],
            "pnl_oos": r["oos_pnl"],
            "sharpe_oos": r["oos_sharpe"],
            "win_oos": r["oos_win_rate"],
        })
    df_out = pd.DataFrame(rows)
    df_out.to_csv(out / "chandelier_summary.csv", index=False)

    # JSON
    summary = {"variants": results}
    (out / "chandelier_summary.json").write_text(json.dumps(to_jsonable(summary), ensure_ascii=False, indent=2), encoding="utf-8")

    # Print summary
    print("\n========== Chandelier Exit Research Results ==========")
    print(f"{'Variant':<45} {'Trades':>6} {'Overall PnL':>10} {'Sharpe':>7} {'Win%':>6} {'PF':>6} {'MaxDD%':>7} {'OOS PnL':>10} {'OOS Sharpe':>10}")
    print("-" * 120)
    for r in results:
        print(f"{r['name']:<45} {r['total_trades']:>6} ${r['overall_pnl']:>9,.0f} {r['overall_sharpe']:>7.2f} {r['overall_win_rate']:>5.1f}% {r['overall_pf']:>6.2f} {r['overall_max_dd']:>6.1f}% ${r['oos_pnl']:>9,.0f} {r['oos_sharpe']:>10.2f}")

    # Exit breakdown for top variants
    print("\n--- Exit Detail Breakdown (Top Variants) ---")
    for r in results[:6]:  # baseline + 4 chandelier + best combined
        print(f"\n{r['name']}:")
        for reason, v in sorted(r.get("by_exit_detail", {}).items(), key=lambda x: -x[1]["total_pnl"]):
            print(f"  {reason:<30} count={v['count']:>4}  PnL=${v['total_pnl']:>10,.0f}")

    log_event(logger, logging.INFO, "done", "Chandelier research complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
