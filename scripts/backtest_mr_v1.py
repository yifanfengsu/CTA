#!/usr/bin/env python3
"""MR-v1 Phase 3: Formal 4h backtest with vnpy-compatible reporting.

Research-only. Runs bar-by-bar 4h backtest on all 5 symbols using best
parameters from Phase 2 (lookback=8, atr_stop=1.0, max_hold=60).
Outputs standard backtest metrics: total PnL, Sharpe, max drawdown,
daily PnL, trade list, monthly breakdown.
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
from typing import Any, Iterable

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
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "reports" / "research" / "mr_v1_phase3"
DEFAULT_DATABASE_PATH = PROJECT_ROOT / ".vntrader" / "database.db"

# Best parameters from Phase 2
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


class BacktestError(Exception):
    pass


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def split_vt_symbol(vt: str) -> tuple[str, str]:
    s, sep, e = str(vt).partition(".")
    if not sep:
        raise BacktestError(f"bad vt_symbol: {vt}")
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
# Strategy
# ---------------------------------------------------------------------------

def compute_atr(bars: pd.DataFrame, period: int) -> pd.Series:
    h, l, c = bars["high"], bars["low"], bars["close"]
    tr = pd.concat([(h - l).abs(), (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()


@dataclass
class Trade:
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    direction: int  # 1=long, -1=short
    entry_price: float
    exit_price: float
    exit_reason: str
    symbol: str


def run_backtest(bars_4h: pd.DataFrame, symbol: str) -> list[Trade]:
    """Run MR-v1 backtest on 4h bars."""
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
            # Manage exit
            hold_bars = i - entry_bar
            stop_dist = ATR_MULT * entry_atr
            exit_now = False
            reason = ""

            if pos == 1:  # long (fading short breakout)
                if bar["low"] <= entry_price - stop_dist:
                    exit_now = True
                    reason = "stop"
            else:  # short (fading long breakout)
                if bar["high"] >= entry_price + stop_dist:
                    exit_now = True
                    reason = "stop"

            if hold_bars >= MAX_HOLD and not exit_now:
                exit_now = True
                reason = "max_hold"

            if exit_now:
                exit_price = bar["open"] if reason == "stop" else bar["close"]
                trades.append(Trade(
                    entry_time=bars_4h["datetime"].iloc[entry_bar],
                    exit_time=bar["datetime"],
                    direction=pos, entry_price=entry_price,
                    exit_price=exit_price, exit_reason=reason, symbol=symbol,
                ))
                pos = 0
                continue

        if pos == 0:
            # Check entry signals
            lb = close.iloc[i] > highs.iloc[i]  # long breakout → short (fade)
            sb = close.iloc[i] < lows.iloc[i]   # short breakout → long (fade)

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

    # Close any open position at end
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
# Analysis
# ---------------------------------------------------------------------------

def compute_metrics(trades: list[Trade], funding_map: dict, tz_name: str) -> dict[str, Any]:
    """Compute comprehensive backtest metrics."""
    if not trades:
        return {"total_trades": 0, "total_pnl": 0, "sharpe": 0, "max_drawdown_pct": 0}

    cost_per_trade = FIXED_NOTIONAL * (2 * FEE_BPS + 2 * SLIPPAGE_BPS) / 10000.0

    records = []
    for t in trades:
        if t.direction == 1:
            ret = (t.exit_price - t.entry_price) / t.entry_price
        else:
            ret = (t.entry_price - t.exit_price) / t.entry_price
        no_cost = ret * FIXED_NOTIONAL
        cost_aware = no_cost - cost_per_trade

        # Funding
        import zoneinfo
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
            "symbol": t.symbol, "no_cost_pnl": no_cost,
            "cost_aware_pnl": cost_aware, "funding_adjusted_pnl": cost_aware - funding_paid,
        })

    df = pd.DataFrame(records)
    df["date"] = df["exit_time"].dt.date

    # Daily PnL
    daily = df.groupby("date")["funding_adjusted_pnl"].sum()
    equity = daily.cumsum()

    # Metrics
    total_pnl = float(equity.iloc[-1]) if len(equity) > 0 else 0.0
    daily_returns = daily / FIXED_NOTIONAL

    if len(daily_returns) > 1:
        sharpe = float(np.sqrt(252) * daily_returns.mean() / daily_returns.std()) if daily_returns.std() > 0 else 0.0
    else:
        sharpe = 0.0

    # Max drawdown
    peak = equity.cummax()
    dd = (equity - peak) / FIXED_NOTIONAL
    max_dd = float(dd.min()) if len(dd) > 0 else 0.0

    # Win rate
    win_rate = float((df["funding_adjusted_pnl"] > 0).mean()) if len(df) > 0 else 0.0

    # Avg trade
    avg_trade = total_pnl / len(df) if len(df) > 0 else 0.0

    return {
        "total_trades": len(df),
        "total_pnl_funding_adjusted": total_pnl,
        "total_pnl_cost_aware": float(df["cost_aware_pnl"].sum()),
        "total_pnl_no_cost": float(df["no_cost_pnl"].sum()),
        "sharpe_ratio": round(sharpe, 2),
        "max_drawdown_pct": round(max_dd * 100, 2),
        "win_rate": round(win_rate * 100, 1),
        "avg_trade_pnl": round(avg_trade, 2),
        "profit_factor": _profit_factor(df),
        "daily_pnl": {str(d): float(v) for d, v in daily.items()},
        "equity_curve": {str(d): float(v) for d, v in equity.items()},
        "monthly_pnl": _monthly_pnl(daily),
        "by_exit_reason": _exit_reason_breakdown(df),
    }


def _profit_factor(df: pd.DataFrame) -> float:
    wins = df[df["funding_adjusted_pnl"] > 0]["funding_adjusted_pnl"].sum()
    losses = abs(df[df["funding_adjusted_pnl"] < 0]["funding_adjusted_pnl"].sum())
    return float(wins / losses) if losses > 0 else float("inf")


def _monthly_pnl(daily: pd.Series) -> dict[str, float]:
    daily.index = pd.to_datetime(daily.index)
    monthly = daily.resample("ME").sum()
    return {str(d.date()): float(v) for d, v in monthly.items()}


def _exit_reason_breakdown(df: pd.DataFrame) -> dict[str, dict]:
    breakdown = {}
    for reason, grp in df.groupby("exit_reason"):
        breakdown[reason] = {"count": len(grp), "total_pnl": float(grp["funding_adjusted_pnl"].sum())}
    return breakdown


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

def parse_args(argv=None):
    p = argparse.ArgumentParser(description="MR-v1 Phase 3: Formal 4h backtest.")
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
    logger = setup_logging("research_mr_v1_phase3", verbose=bool(args.verbose))
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

    # Run backtest
    all_trades: list[Trade] = []
    for sym in symbols:
        log_event(logger, logging.INFO, "backtest_symbol", "Running backtest", symbol=sym)
        bars_1m = load_1m(sym, hr, db)
        bars_4h = resample_4h(bars_1m, hr)
        if bars_4h.empty:
            log_event(logger, logging.WARNING, "no_data", "No 4h bars", symbol=sym)
            continue
        trades = run_backtest(bars_4h, sym)
        all_trades.extend(trades)

    if not all_trades:
        logger.error("No trades generated")
        return 1

    # Build trade records
    records = []
    for t in all_trades:
        records.append({"entry_time": t.entry_time, "exit_time": t.exit_time,
                         "direction": t.direction, "entry_price": t.entry_price,
                         "exit_price": t.exit_price, "exit_reason": t.exit_reason,
                         "symbol": t.symbol})
    trade_df = pd.DataFrame(records)

    # Assign splits
    trade_df = assign_split(trade_df)

    # Compute metrics
    metrics = compute_metrics(all_trades, funding_map, args.timezone)

    # Per-split metrics
    split_metrics = {}
    for split in ["train_ext", "validation_ext", "oos_ext"]:
        mask = trade_df["split"] == split
        split_trades = [t for i, t in enumerate(all_trades) if mask.iloc[i]]
        split_metrics[split] = compute_metrics(split_trades, funding_map, args.timezone)

    # Per-symbol OOS
    symbol_oos = {}
    for sym in symbols:
        mask = (trade_df["symbol"] == sym) & (trade_df["split"] == "oos_ext")
        sym_trades = [t for i, t in enumerate(all_trades) if mask.iloc[i]]
        if sym_trades:
            m = compute_metrics(sym_trades, funding_map, args.timezone)
            symbol_oos[sym] = {"pnl": m["total_pnl_funding_adjusted"], "trades": m["total_trades"], "win_rate": m["win_rate"]}

    # Summary
    summary = {
        "strategy": "MR-v1 Mean Reversion",
        "version": "v3.0 (Phase 3)",
        "parameters": {"lookback": LOOKBACK, "atr_mult": ATR_MULT, "max_hold": MAX_HOLD},
        "symbols": symbols,
        "data_range": f"{args.start} to {args.end}",
        "overall_metrics": metrics,
        "split_metrics": split_metrics,
        "symbol_oos": symbol_oos,
        "gates": {
            "oos_profitable": split_metrics.get("oos_ext", {}).get("total_pnl_funding_adjusted", 0) > 0,
            "all_splits_profitable": all(
                split_metrics.get(s, {}).get("total_pnl_funding_adjusted", 0) > 0
                for s in ["train_ext", "validation_ext", "oos_ext"]
            ),
            "strategy_development_allowed": False,
            "demo_live_allowed": False,
        },
    }

    out.mkdir(parents=True, exist_ok=True)
    (out / "phase3_summary.json").write_text(json.dumps(to_jsonable(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    trade_df.to_csv(out / "phase3_trades.csv", index=False)

    # Report
    m = metrics
    report = (
        f"# MR-v1 Phase 3: Formal 4h Backtest\n\n"
        f"## Parameters\n"
        f"- lookback={LOOKBACK}, atr_stop={ATR_MULT}×ATR, max_hold={MAX_HOLD} bars\n"
        f"- Notional: ${FIXED_NOTIONAL:.0f} per trade, Fees: {FEE_BPS}bps/side, Slippage: {SLIPPAGE_BPS}bps/side\n\n"
        f"## Overall Metrics (funding-adjusted)\n"
        f"| Metric | Value |\n|---|---|\n"
        f"| Total Trades | {m['total_trades']} |\n"
        f"| Total PnL | ${m['total_pnl_funding_adjusted']:,.2f} |\n"
        f"| Win Rate | {m['win_rate']}% |\n"
        f"| Avg Trade | ${m['avg_trade_pnl']:,.2f} |\n"
        f"| Sharpe (annualized) | {m['sharpe_ratio']} |\n"
        f"| Max Drawdown | {m['max_drawdown_pct']}% |\n"
        f"| Profit Factor | {m['profit_factor']:.2f} |\n\n"
        f"## Per-Split Metrics\n"
        f"| Split | Trades | PnL | Win% |\n|---|---|---|---|\n"
    )
    for s in ["train_ext", "validation_ext", "oos_ext"]:
        sm = split_metrics.get(s, {})
        report += f"| {s} | {sm.get('total_trades',0)} | ${sm.get('total_pnl_funding_adjusted',0):,.0f} | {sm.get('win_rate',0)}% |\n"

    report += f"\n## OOS Per-Symbol\n| Symbol | Trades | PnL | Win% |\n|---|---|---|---|\n"
    for sym, v in symbol_oos.items():
        report += f"| {sym.split('_')[0]} | {v['trades']} | ${v['pnl']:,.0f} | {v['win_rate']}% |\n"

    report += f"\n## Exit Reason Breakdown\n| Reason | Count | PnL |\n|---|---|---|\n"
    for reason, v in m.get("by_exit_reason", {}).items():
        report += f"| {reason} | {v['count']} | ${v['total_pnl']:,.0f} |\n"

    report += (
        f"\n## Monthly PnL\n"
        f"| Month | PnL |\n|---|---|\n"
    )
    for month, pnl in sorted(m.get("monthly_pnl", {}).items()):
        report += f"| {month[:7]} | ${pnl:,.0f} |\n"

    (out / "phase3_report.md").write_text(report, encoding="utf-8")

    log_event(logger, logging.INFO, "phase3_complete", "Phase 3 complete",
              oos_pnl=split_metrics.get("oos_ext", {}).get("total_pnl_funding_adjusted", 0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
