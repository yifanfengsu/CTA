#!/usr/bin/env python3
"""MR-v1 Mean Reversion Research: fade 20-bar breakouts with exit mechanisms.

This script is research-only. It fades (trades against) 20-bar breakouts on 4h bars:
long breakout → go SHORT, short breakout → go LONG. Then compares exit
mechanisms (baseline ATR stop, trend drawdown tolerance, 1d trend health lock,
and B+C combined) across train_ext / validation_ext / oos_ext splits.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import sqlite3
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Any, Iterable

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
from history_time_utils import HistoryRange, expected_bar_count, parse_history_range

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_SYMBOLS = [
    "BTCUSDT_SWAP_OKX.GLOBAL",
    "ETHUSDT_SWAP_OKX.GLOBAL",
    "SOLUSDT_SWAP_OKX.GLOBAL",
    "LINKUSDT_SWAP_OKX.GLOBAL",
    "DOGEUSDT_SWAP_OKX.GLOBAL",
]
DEFAULT_START = "2023-01-01"
DEFAULT_END = "2026-03-31"
DEFAULT_TIMEZONE = "Asia/Shanghai"
DEFAULT_FUNDING_DIR = PROJECT_ROOT / "data" / "funding" / "okx"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "reports" / "research" / "mr_v1"
DEFAULT_DATABASE_PATH = PROJECT_ROOT / ".vntrader" / "database.db"
TIMEFRAME_MINUTES = {"4h": 240, "1d": 1440}

SPLIT_RANGES = {
    "train_ext": ("2023-01-01", "2024-07-01"),
    "validation_ext": ("2024-07-01", "2025-07-01"),
    "oos_ext": ("2025-07-01", "2026-04-01"),
}
SPLIT_NAMES = ["train_ext", "validation_ext", "oos_ext"]

EXIT_VARIANTS = ["A", "B", "C", "BC", "D", "E"]

FIXED_NOTIONAL = 1000.0
FEE_BPS_PER_SIDE = 5.0
SLIPPAGE_BPS_PER_SIDE = 5.0
RANDOM_SEED = 42

# Entry: 20-bar breakout
ENTRY_LOOKBACK = 20

# Exit A (baseline): ATR stop
EXIT_A_ATR_STOP = 2.0
EXIT_A_ATR_PROFIT = 3.0
EXIT_A_MAX_HOLD_BARS = 60

# Exit B (trend drawdown): drawdown from peak
EXIT_B_DRAWDOWN_ATR = 2.0
EXIT_B_MAX_HOLD_BARS = 60

# Exit C (1d health): 1d EMA lookback
EXIT_C_1D_EMA = 20
# When C suppresses exit, still enforce ultimate max hold
EXIT_C_MAX_HOLD_BARS = 120

# Exit B+C: use B's drawdown + C's suppression
EXIT_BC_MAX_HOLD_BARS = 120


class ExitResearchError(Exception):
    """Raised when research cannot continue."""


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def split_vt_symbol(vt_symbol: str) -> tuple[str, str]:
    symbol, sep, exchange = str(vt_symbol).partition(".")
    if not sep or not symbol or not exchange:
        raise ExitResearchError(f"invalid vt_symbol: {vt_symbol}")
    return symbol, exchange


def symbol_to_inst_id(vt_symbol: str) -> str:
    symbol, _exchange = split_vt_symbol(vt_symbol)
    root = symbol.removesuffix("_OKX")
    pair = root[: -len("_SWAP")] if root.endswith("_SWAP") else root
    if pair.endswith("USDT"):
        return f"{pair[:-4]}-USDT-SWAP"
    return root.replace("_", "-")


def normalize_1m_bars(frame: pd.DataFrame, timezone_name: str) -> pd.DataFrame:
    columns = ["datetime", "open", "high", "low", "close", "volume"]
    if frame.empty:
        return pd.DataFrame(columns=columns)
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise ExitResearchError(f"1m bars missing columns: {missing}")
    out = frame.loc[:, columns].copy()
    timestamps = pd.to_datetime(out["datetime"], errors="coerce")
    if timestamps.isna().any():
        raise ExitResearchError("unparsable datetime in 1m bars")
    if timestamps.dt.tz is None:
        timestamps = timestamps.dt.tz_localize(timezone_name)
    else:
        timestamps = timestamps.dt.tz_convert(timezone_name)
    out["datetime"] = timestamps
    for c in ["open", "high", "low", "close", "volume"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.dropna(subset=columns).sort_values("datetime", kind="stable").drop_duplicates("datetime", keep="last")
    return out.reset_index(drop=True)


def load_1m_bars(vt_symbol: str, history_range: HistoryRange, database_path: Path) -> pd.DataFrame:
    if not database_path.exists():
        raise ExitResearchError(f"database not found: {database_path}")
    symbol, exchange = split_vt_symbol(vt_symbol)
    q_start = history_range.start.replace(tzinfo=None).isoformat(sep=" ", timespec="seconds")
    q_end = history_range.end_exclusive.replace(tzinfo=None).isoformat(sep=" ", timespec="seconds")
    sql = """select datetime, open_price as open, high_price as high,
                    low_price as low, close_price as close, volume
               from dbbardata
              where symbol = ? and exchange = ? and interval = ?
                and datetime >= ? and datetime < ?
              order by datetime"""
    with sqlite3.connect(database_path) as conn:
        frame = pd.read_sql_query(sql, conn, params=(symbol, exchange, "1m", q_start, q_end))
    return normalize_1m_bars(frame, history_range.timezone_name)


def resample_ohlcv_closed(bars_1m: pd.DataFrame, timeframe: str, history_range: HistoryRange | None = None) -> pd.DataFrame:
    minutes = TIMEFRAME_MINUTES[timeframe]
    columns = ["open_time", "datetime", "open", "high", "low", "close", "volume"]
    if bars_1m.empty:
        return pd.DataFrame(columns=columns)
    working = bars_1m.sort_values("datetime", kind="stable").drop_duplicates("datetime", keep="last").copy()
    anchor = pd.Timestamp(history_range.start if history_range else working["datetime"].iloc[0])
    if anchor.tzinfo is None:
        anchor = anchor.tz_localize(working["datetime"].iloc[0].tz)
    deltas = (working["datetime"] - anchor) / pd.Timedelta(minutes=1)
    working = working.loc[deltas >= 0].copy()
    working["_slot"] = np.floor(deltas.loc[working.index].to_numpy(dtype=float) / minutes).astype(np.int64)
    grouped = working.groupby("_slot", sort=True, dropna=False)
    result = grouped.agg(
        open_time=("datetime", "min"), datetime=("datetime", "max"),
        open=("open", "first"), high=("high", "max"),
        low=("low", "min"), close=("close", "last"),
        volume=("volume", "sum"), minute_count=("datetime", "size"),
    )
    result = result[result["minute_count"] == minutes].copy()
    result = result.drop(columns=["minute_count"]).dropna(subset=["open", "high", "low", "close"]).reset_index(drop=True)
    return result.loc[:, columns]


def load_funding_csv(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if frame.empty:
        return pd.DataFrame(columns=["funding_time_utc", "funding_rate"])
    if "funding_time_utc" in frame.columns:
        timestamps = pd.to_datetime(frame["funding_time_utc"], utc=True, errors="coerce")
    elif "funding_time" in frame.columns:
        timestamps = pd.to_datetime(pd.to_numeric(frame["funding_time"], errors="coerce"), unit="ms", utc=True, errors="coerce")
    else:
        return pd.DataFrame(columns=["funding_time_utc", "funding_rate"])
    result = pd.DataFrame({"funding_time_utc": timestamps, "funding_rate": pd.to_numeric(frame.get("funding_rate"), errors="coerce")})
    result = result.dropna(subset=["funding_time_utc", "funding_rate"])
    return result.sort_values("funding_time_utc", kind="stable").drop_duplicates("funding_time_utc", keep="last").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Entry logic
# ---------------------------------------------------------------------------

def detect_breakout_entries(bars_4h: pd.DataFrame) -> pd.DataFrame:
    """Detect 20-bar breakout entries on 4h closed bars.

    Returns DataFrame with columns: entry_time, direction (1=long, -1=short),
    entry_price.
    """
    if len(bars_4h) < ENTRY_LOOKBACK + 1:
        return pd.DataFrame(columns=["entry_time", "direction", "entry_price"])

    highs = bars_4h["high"].rolling(ENTRY_LOOKBACK).max().shift(1)
    lows = bars_4h["low"].rolling(ENTRY_LOOKBACK).min().shift(1)
    close = bars_4h["close"]

    long_signal = close > highs
    short_signal = close < lows

    entries = []
    position = 0  # 0=flat, 1=long, -1=short

    for i in range(ENTRY_LOOKBACK, len(bars_4h)):
        # Mean reversion: long breakout → go SHORT (fade breakout)
        if long_signal.iloc[i] and position != -1:
            if position == 1:
                entries.append({"entry_time": bars_4h["datetime"].iloc[i], "direction": 0, "entry_price": bars_4h["close"].iloc[i]})
            entries.append({"entry_time": bars_4h["datetime"].iloc[i], "direction": -1, "entry_price": bars_4h["close"].iloc[i]})
            position = -1
        # Mean reversion: short breakout → go LONG (fade breakdown)
        elif short_signal.iloc[i] and position != 1:
            if position == -1:
                entries.append({"entry_time": bars_4h["datetime"].iloc[i], "direction": 0, "entry_price": bars_4h["close"].iloc[i]})
            entries.append({"entry_time": bars_4h["datetime"].iloc[i], "direction": 1, "entry_price": bars_4h["close"].iloc[i]})
            position = 1

    return pd.DataFrame(entries)


# ---------------------------------------------------------------------------
# Exit logic
# ---------------------------------------------------------------------------

def compute_atr(bars: pd.DataFrame, period: int = 14) -> pd.Series:
    """Compute ATR on bars."""
    high, low, close = bars["high"], bars["low"], bars["close"]
    tr = pd.concat([
        (high - low).abs(),
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def compute_ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def simulate_trades(
    entries: pd.DataFrame,
    bars_4h: pd.DataFrame,
    bars_1d: pd.DataFrame | None,
    variant: str,
) -> pd.DataFrame:
    """Simulate trades with a given exit variant.

    Returns DataFrame with: entry_time, exit_time, direction, entry_price,
    exit_price, variant, exit_reason.
    """
    if entries.empty or len(bars_4h) < 2:
        return pd.DataFrame(columns=["entry_time", "exit_time", "direction", "entry_price", "exit_price", "variant", "exit_reason"])

    atr_14 = compute_atr(bars_4h, 14)
    # 1d EMA for variant C
    ema_1d = None
    if bars_1d is not None and not bars_1d.empty:
        ema_1d = compute_ema(bars_1d["close"], EXIT_C_1D_EMA)

    # Build lookup: for each 4h bar, what's the latest 1d EMA value
    ema_1d_lookup = None
    if ema_1d is not None and bars_1d is not None:
        ema_1d_lookup = pd.Series(np.nan, index=bars_4h.index)
        for i, ts in enumerate(bars_4h["datetime"]):
            mask = bars_1d["datetime"] <= ts
            if mask.any():
                idx = mask[mask].index[-1]
                if idx < len(ema_1d):
                    ema_1d_lookup.iloc[i] = ema_1d.iloc[idx]

    trades = []
    bar_idx_map = {ts: i for i, ts in enumerate(bars_4h["datetime"])}

    for _, entry in entries.iterrows():
        entry_ts = entry["entry_time"]
        direction = int(entry["direction"])
        if direction == 0:
            continue  # skip closing-only entries

        entry_price = entry["entry_price"]
        entry_idx = bar_idx_map.get(entry_ts)
        if entry_idx is None or entry_idx >= len(bars_4h) - 1:
            continue

        entry_atr = atr_14.iloc[entry_idx] if entry_idx < len(atr_14) and not pd.isna(atr_14.iloc[entry_idx]) else 0.0
        if entry_atr <= 0:
            entry_atr = bars_4h["close"].iloc[entry_idx] * 0.01  # fallback: 1% of price

        max_hold = EXIT_A_MAX_HOLD_BARS
        if variant == "C":
            max_hold = EXIT_C_MAX_HOLD_BARS
        elif variant in ("BC", "B+C"):
            max_hold = EXIT_BC_MAX_HOLD_BARS

        max_mfe = entry_price
        exit_idx = None
        exit_price = None
        exit_reason = None

        for j in range(entry_idx + 1, min(entry_idx + max_hold + 1, len(bars_4h))):
            bar = bars_4h.iloc[j]

            if direction == 1:
                mfe = bar["high"]
                if mfe > max_mfe:
                    max_mfe = mfe
            else:
                mfe = bar["low"]
                if mfe < max_mfe:
                    max_mfe = mfe

            should_exit = False

            if variant == "A":
                if direction == 1:
                    if bar["low"] <= entry_price - exit_a_atr(entry_atr):
                        should_exit = True
                        exit_reason = "atr_stop"
                else:
                    if bar["high"] >= entry_price + exit_a_atr(entry_atr):
                        should_exit = True
                        exit_reason = "atr_stop"

            elif variant == "B":
                if direction == 1:
                    drawdown = max_mfe - bar["low"]
                    if drawdown >= exit_b_drawdown_atr(entry_atr):
                        should_exit = True
                        exit_reason = "drawdown"
                else:
                    drawdown = bar["high"] - max_mfe
                    if drawdown >= exit_b_drawdown_atr(entry_atr):
                        should_exit = True
                        exit_reason = "drawdown"

            elif variant == "C":
                if direction == 1:
                    stop_hit = bar["low"] <= entry_price - exit_a_atr(entry_atr)
                else:
                    stop_hit = bar["high"] >= entry_price + exit_a_atr(entry_atr)

                if stop_hit:
                    trend_healthy = _is_1d_trend_healthy(direction, bars_4h, bars_1d, ema_1d_lookup, j)
                    if not trend_healthy:
                        should_exit = True
                        exit_reason = "atr_stop_unhealthy_1d"
                    # else: suppress exit, continue holding

            elif variant in ("BC", "B+C"):
                if direction == 1:
                    drawdown = max_mfe - bar["low"]
                    dd_hit = drawdown >= exit_b_drawdown_atr(entry_atr)
                else:
                    drawdown = bar["high"] - max_mfe
                    dd_hit = drawdown >= exit_b_drawdown_atr(entry_atr)

                if dd_hit:
                    trend_healthy = _is_1d_trend_healthy(direction, bars_4h, bars_1d, ema_1d_lookup, j)
                    if not trend_healthy:
                        should_exit = True
                        exit_reason = "drawdown_unhealthy_1d"

            if should_exit:
                exit_idx = j
                exit_price = bar["open"]
                break

        # Forced exit at max hold
        if exit_idx is None:
            exit_idx = min(entry_idx + max_hold, len(bars_4h) - 1)
            exit_price = bars_4h["close"].iloc[exit_idx]
            exit_reason = "max_hold"

        exit_ts = bars_4h["datetime"].iloc[exit_idx]

        trades.append({
            "entry_time": entry_ts,
            "exit_time": exit_ts,
            "direction": direction,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "variant": variant,
            "exit_reason": exit_reason,
            "entry_atr": entry_atr,
            "max_mfe": max_mfe,
        })

    return pd.DataFrame(trades)


def exit_a_atr(entry_atr: float) -> float:
    return EXIT_A_ATR_STOP * entry_atr


def exit_b_drawdown_atr(entry_atr: float) -> float:
    return EXIT_B_DRAWDOWN_ATR * entry_atr


def _is_1d_trend_healthy(direction: int, bars_4h: pd.DataFrame, bars_1d: pd.DataFrame | None, ema_1d_lookup: pd.Series | None, idx: int) -> bool:
    """Check if 1d trend is healthy (in direction's favor)."""
    if bars_1d is None or bars_1d.empty or ema_1d_lookup is None:
        return False
    if idx < 0 or idx >= len(ema_1d_lookup):
        return False
    ema_val = ema_1d_lookup.iloc[idx]
    if pd.isna(ema_val):
        return False
    close_1d = bars_4h["close"].iloc[idx]  # approximate with 4h close
    if direction == 1:
        return close_1d > ema_val
    else:
        return close_1d < ema_val


# ---------------------------------------------------------------------------
# PnL computation
# ---------------------------------------------------------------------------

def compute_trade_pnl(
    trades: pd.DataFrame,
    funding_map: dict[str, pd.DataFrame],
    symbol: str,
    timezone_name: str,
) -> pd.DataFrame:
    """Compute no-cost, cost-aware, and funding-adjusted PnL for each trade."""
    if trades.empty:
        return trades.assign(no_cost_pnl=0.0, cost_aware_pnl=0.0, funding_adjusted_pnl=0.0, funding_paid=0.0)

    out = trades.copy()
    cost_per_trade = FIXED_NOTIONAL * (2 * FEE_BPS_PER_SIDE + 2 * SLIPPAGE_BPS_PER_SIDE) / 10000.0

    out["gross_return"] = np.where(
        out["direction"] == 1,
        (out["exit_price"] - out["entry_price"]) / out["entry_price"],
        (out["entry_price"] - out["exit_price"]) / out["entry_price"],
    )
    out["no_cost_pnl"] = out["gross_return"] * FIXED_NOTIONAL
    out["cost_aware_pnl"] = out["no_cost_pnl"] - cost_per_trade

    # Funding
    inst_id = symbol_to_inst_id(symbol)
    funding = funding_map.get(inst_id)
    out["funding_paid"] = 0.0
    if funding is not None and not funding.empty:
        tz = resolve_timezone(timezone_name)
        for i, trade in out.iterrows():
            entry_ts = pd.Timestamp(trade["entry_time"])
            exit_ts = pd.Timestamp(trade["exit_time"])
            if entry_ts.tzinfo is None:
                entry_ts = entry_ts.tz_localize(tz)
            else:
                entry_ts = entry_ts.tz_convert(tz)
            if exit_ts.tzinfo is None:
                exit_ts = exit_ts.tz_localize(tz)
            else:
                exit_ts = exit_ts.tz_convert(tz)
            entry_utc = entry_ts.tz_convert("UTC")
            exit_utc = exit_ts.tz_convert("UTC")
            mask = (funding["funding_time_utc"] >= entry_utc) & (funding["funding_time_utc"] < exit_utc)
            if mask.any():
                total_rate = funding.loc[mask, "funding_rate"].sum()
                funding_paid = total_rate * FIXED_NOTIONAL
                if trade["direction"] == 1:
                    out.at[i, "funding_paid"] = funding_paid
                else:
                    out.at[i, "funding_paid"] = -funding_paid

    out["funding_adjusted_pnl"] = out["cost_aware_pnl"] - out["funding_paid"]
    return out


def resolve_timezone(timezone_name: str):
    import zoneinfo
    return zoneinfo.ZoneInfo(timezone_name)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def assign_split(trades: pd.DataFrame) -> pd.DataFrame:
    """Assign train_ext/validation_ext/oos_ext based on entry_time."""
    out = trades.copy()
    out["split"] = "oos_ext"
    for split_name, (start_str, end_str) in SPLIT_RANGES.items():
        start = pd.Timestamp(start_str).tz_localize(DEFAULT_TIMEZONE)
        end = pd.Timestamp(end_str).tz_localize(DEFAULT_TIMEZONE)
        mask = (out["entry_time"] >= start) & (out["entry_time"] < end)
        out.loc[mask, "split"] = split_name
    return out


def build_trade_summary(trades: pd.DataFrame) -> pd.DataFrame:
    """Build per-variant per-split summary."""
    if trades.empty:
        return pd.DataFrame()
    summary = trades.groupby(["variant", "split"]).agg(
        trade_count=("entry_time", "count"),
        avg_hold_bars=("exit_time", lambda x: np.nan if len(x) == 0 else np.mean([
            (pd.Timestamp(e) - pd.Timestamp(s)).total_seconds() / (240 * 60) for s, e in zip(trades.loc[x.index, "entry_time"], x)
        ])),
        no_cost_pnl=("no_cost_pnl", "sum"),
        cost_aware_pnl=("cost_aware_pnl", "sum"),
        funding_adjusted_pnl=("funding_adjusted_pnl", "sum"),
        winning_rate=("no_cost_pnl", lambda x: (x > 0).mean()),
    ).reset_index()
    return summary


def build_concentration(trades: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Build concentration analysis."""
    if trades.empty:
        return pd.DataFrame()
    trades = trades.copy()
    trades["symbol"] = symbol
    rows = []
    for (variant, split), grp in trades.groupby(["variant", "split"]):
        if grp.empty:
            continue
        pnl = grp["no_cost_pnl"].sort_values(ascending=False)
        total = pnl.sum()
        top5 = int(max(1, len(pnl) * 0.05))
        rows.append({
            "variant": variant, "split": split, "symbol": symbol,
            "trade_count": len(grp),
            "total_no_cost_pnl": total,
            "top_5pct_contribution": pnl.head(top5).sum() / total if total != 0 else np.inf,
            "concentration_pass": abs(pnl.head(top5).sum() / total) <= 2.0 if total != 0 else True,
        })
    return pd.DataFrame(rows)


def generate_random_entries(
    bars_4h: pd.DataFrame,
    entry_count: int,
    seed: int,
) -> pd.DataFrame:
    """Generate random entry times for control group."""
    if len(bars_4h) < ENTRY_LOOKBACK + 2:
        return pd.DataFrame(columns=["entry_time", "direction", "entry_price"])
    rng = deterministic_rng(seed)
    valid = bars_4h.iloc[ENTRY_LOOKBACK:-1].copy()
    if len(valid) < entry_count:
        entry_count = len(valid)
    chosen = rng.choice(valid.index, size=entry_count, replace=False)
    entries = []
    for idx in sorted(chosen):
        direction = 1 if rng.random() > 0.5 else -1
        entries.append({
            "entry_time": bars_4h["datetime"].iloc[idx],
            "direction": direction,
            "entry_price": bars_4h["close"].iloc[idx],
        })
    return pd.DataFrame(entries)


def generate_reverse_entries(entries: pd.DataFrame) -> pd.DataFrame:
    """Reverse direction of entries."""
    out = entries.copy()
    out["direction"] = -out["direction"]
    return out


def deterministic_rng(seed: int, *parts: Any) -> np.random.Generator:
    text = "|".join(str(p) for p in (seed, *parts))
    digest = hashlib.sha256(text.encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "little") % (2**32 - 1))


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def format_number(value: Any, digits: int = 6) -> str:
    try:
        n = float(value)
    except (TypeError, ValueError):
        return "N/A"
    if not np.isfinite(n):
        return "N/A"
    return f"{n:.{digits}f}"


def markdown_table(rows: list[dict[str, Any]], columns: list[str], limit: int = 50) -> str:
    if not rows:
        return "- N/A"
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join("---" for _ in columns) + "|"]
    for row in rows[:limit]:
        vals = []
        for c in columns:
            v = row.get(c)
            vals.append(format_number(v, 4) if isinstance(v, float) else str(v))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(payload), ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def dataframe_records(frame: pd.DataFrame, limit: int | None = None) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    work = frame.head(limit) if limit else frame.copy()
    work = work.replace({np.nan: None})
    return json.loads(work.to_json(orient="records", force_ascii=False, date_format="iso"))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MR-v1 Mean Reversion: fade 20-bar breakouts with exit mechanisms.")
    parser.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS))
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--end", default=DEFAULT_END)
    parser.add_argument("--timezone", default=DEFAULT_TIMEZONE)
    parser.add_argument("--funding-dir", default=str(DEFAULT_FUNDING_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--database-path", default=str(DEFAULT_DATABASE_PATH))
    parser.add_argument("--data-check-strict", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def parse_csv_list(value: str | Iterable[str]) -> list[str]:
    tokens = re.split(r"[\s,]+", value) if isinstance(value, str) else [str(x) for x in value]
    parsed, seen = [], set()
    for t in tokens:
        t = t.strip()
        if t and t not in seen:
            parsed.append(t); seen.add(t)
    return parsed


def main(argv: list[str] | None = None) -> int:
    ensure_headless_runtime()
    args = parse_args(argv)
    logger = setup_logging("research_mr_v1", verbose=bool(args.verbose))
    symbols = parse_csv_list(args.symbols)
    output_dir = Path(args.output_dir) if Path(args.output_dir).is_absolute() else PROJECT_ROOT / args.output_dir
    database_path = Path(args.database_path) if Path(args.database_path).is_absolute() else PROJECT_ROOT / args.database_path
    funding_dir = Path(args.funding_dir) if Path(args.funding_dir).is_absolute() else PROJECT_ROOT / args.funding_dir

    history_range = parse_history_range(args.start, args.end, timedelta(minutes=1), args.timezone)
    tz = resolve_timezone(args.timezone)

    # Load funding data
    funding_map: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        inst_id = symbol_to_inst_id(sym)
        canonical = funding_dir / f"{inst_id}_funding_{DEFAULT_START}_{DEFAULT_END}.csv"
        if canonical.exists():
            funding_map[inst_id] = load_funding_csv(canonical)
        else:
            matches = sorted(funding_dir.glob(f"{inst_id}_funding_*.csv"))
            if matches:
                funding_map[inst_id] = load_funding_csv(matches[-1])

    all_trades: list[pd.DataFrame] = []
    warnings: list[str] = []

    for sym in symbols:
        log_event(logger, logging.INFO, "load_symbol", "Loading symbol market data", symbol=sym)

        # Load 1m bars
        bars_1m = load_1m_bars(sym, history_range, database_path)

        # Resample to 4h and 1d
        bars_4h = resample_ohlcv_closed(bars_1m, "4h", history_range)
        bars_1d = resample_ohlcv_closed(bars_1m, "1d", history_range)

        if bars_4h.empty:
            warnings.append(f"no_4h_bars:{sym}")
            continue

        # Detect entries
        entries = detect_breakout_entries(bars_4h)
        if entries.empty:
            warnings.append(f"no_entries:{sym}")
            continue

        real_entries = entries[entries["direction"] != 0].copy()

        # A: Baseline
        trades_a = simulate_trades(real_entries, bars_4h, bars_1d, "A")
        # B: Trend drawdown
        trades_b = simulate_trades(real_entries, bars_4h, bars_1d, "B")
        # C: 1d health lock
        trades_c = simulate_trades(real_entries, bars_4h, bars_1d, "C")
        # BC: Combined
        trades_bc = simulate_trades(real_entries, bars_4h, bars_1d, "BC")
        # D: Random control
        random_entries = generate_random_entries(bars_4h, len(real_entries), RANDOM_SEED)
        trades_d = simulate_trades(random_entries, bars_4h, bars_1d, "A")  # use baseline exit for random
        trades_d["variant"] = "D"
        # E: Reverse test
        reverse_entries = generate_reverse_entries(real_entries)
        trades_e = simulate_trades(reverse_entries, bars_4h, bars_1d, "A")  # baseline exit
        trades_e["variant"] = "E"

        for trades in [trades_a, trades_b, trades_c, trades_bc, trades_d, trades_e]:
            if trades.empty:
                continue
            trades = compute_trade_pnl(trades, funding_map, sym, args.timezone)
            trades = assign_split(trades)
            trades["symbol"] = sym
            all_trades.append(trades)

    if not all_trades:
        logger.error("No trades generated for any symbol")
        return 1

    all_trades_df = pd.concat(all_trades, ignore_index=True)

    # Write output
    write_csv(output_dir / "all_trades.csv", all_trades_df)

    # Trade summary
    summary = build_trade_summary(all_trades_df)
    write_csv(output_dir / "trade_summary.csv", summary)

    # Concentration
    conc_parts = []
    for sym in symbols:
        sym_trades = all_trades_df[all_trades_df["symbol"] == sym]
        if not sym_trades.empty:
            conc_parts.append(build_concentration(sym_trades, sym))
    concentration = pd.concat(conc_parts, ignore_index=True) if conc_parts else pd.DataFrame()
    write_csv(output_dir / "concentration.csv", concentration)

    # By variant
    by_variant = all_trades_df.groupby(["variant", "split"]).agg(
        trade_count=("entry_time", "count"),
        no_cost_pnl=("no_cost_pnl", "sum"),
        cost_aware_pnl=("cost_aware_pnl", "sum"),
        funding_adjusted_pnl=("funding_adjusted_pnl", "sum"),
        winning_rate=("no_cost_pnl", lambda x: (x > 0).mean()),
    ).reset_index()
    write_csv(output_dir / "by_variant.csv", by_variant)

    # By symbol
    by_symbol = all_trades_df.groupby(["variant", "split", "symbol"]).agg(
        trade_count=("entry_time", "count"),
        no_cost_pnl=("no_cost_pnl", "sum"),
        cost_aware_pnl=("cost_aware_pnl", "sum"),
        funding_adjusted_pnl=("funding_adjusted_pnl", "sum"),
    ).reset_index()
    write_csv(output_dir / "by_symbol.csv", by_symbol)

    # Reverse test
    reverse_df = all_trades_df[all_trades_df["variant"].isin(["A", "E"])].copy()
    reverse_summary = reverse_df.groupby(["variant", "split"]).agg(no_cost_pnl=("no_cost_pnl", "sum")).reset_index()
    write_csv(output_dir / "reverse_test.csv", reverse_summary)

    # Gates
    gates = _evaluate_gates(by_variant, concentration)

    # Summary JSON
    summary_json = {
        "hypothesis": "MR-v1 Mean Reversion: fade 20-bar breakouts (long breakout → short, short breakout → long)",
        "version": "v1.0",
        "status": "research_only",
        "symbols": symbols,
        "data_range": f"{args.start} to {args.end}",
        "entry_rule": "20-bar breakout on 4h closed bars",
        "exit_variants": {
            "A": "Baseline: 2x ATR stop, 60-bar max hold",
            "B": "Trend drawdown tolerance: exit when drawdown from peak > 2x entry_ATR, 60-bar max hold",
            "C": "1d trend health lock: suppress A exit when 1d close > 1d 20-EMA, 120-bar max hold",
            "BC": "Combined B + C: drawdown exit suppressed by 1d health, 120-bar max hold",
            "D": "Random entry time control",
            "E": "Reverse direction test",
        },
        "by_variant": dataframe_records(by_variant),
        "gates": gates,
        "warnings": warnings,
    }
    write_json(output_dir / "mr_v1_summary.json", summary_json)

    # Markdown report
    report = _render_report(summary_json, by_variant, concentration, reverse_summary)
    (output_dir / "mr_v1_report.md").write_text(report, encoding="utf-8")

    log_event(logger, logging.INFO, "research_complete", "MR-v1 research complete",
              can_enter_phase2=gates.get("can_enter_phase2", False),
              strategy_development_allowed=False,
              demo_live_allowed=False)
    return 0


def _evaluate_gates(by_variant: pd.DataFrame, concentration: pd.DataFrame) -> dict[str, Any]:
    gates: dict[str, Any] = {
        "cost_aware_pass": False,
        "funding_adjusted_pass": False,
        "reverse_test_pass": False,
        "random_control_pass": False,
        "concentration_pass": False,
        "can_enter_phase2": False,
        "final_decision": "postmortem_or_pause",
    }

    # Check BC variant OOS cost-aware and funding-adjusted
    bc_oos = by_variant[(by_variant["variant"] == "BC") & (by_variant["split"] == "oos_ext")]
    if not bc_oos.empty:
        gates["cost_aware_pass"] = bool(bc_oos["cost_aware_pnl"].iloc[0] > 0)
        gates["funding_adjusted_pass"] = bool(bc_oos["funding_adjusted_pnl"].iloc[0] > 0)

    # Reverse test: forward (A) vs reverse (E) in OOS
    # E should be worse than A
    a_oos = by_variant[(by_variant["variant"] == "A") & (by_variant["split"] == "oos_ext")]
    e_oos = by_variant[(by_variant["variant"] == "E") & (by_variant["split"] == "oos_ext")]
    if not a_oos.empty and not e_oos.empty:
        gates["reverse_test_pass"] = bool(a_oos["no_cost_pnl"].iloc[0] > e_oos["no_cost_pnl"].iloc[0])

    # Random control: BC should be better than D in OOS
    bc_oos_pnl = by_variant[(by_variant["variant"] == "BC") & (by_variant["split"] == "oos_ext")]
    d_oos = by_variant[(by_variant["variant"] == "D") & (by_variant["split"] == "oos_ext")]
    if not bc_oos_pnl.empty and not d_oos.empty:
        gates["random_control_pass"] = bool(bc_oos_pnl["no_cost_pnl"].iloc[0] > d_oos["no_cost_pnl"].iloc[0])

    # Concentration
    if not concentration.empty:
        bc_conc = concentration[(concentration["variant"] == "BC") & (concentration["split"] == "oos_ext")]
        if not bc_conc.empty:
            gates["concentration_pass"] = bool(bc_conc["concentration_pass"].all())

    # All gates must pass
    if all([
        gates["cost_aware_pass"],
        gates["funding_adjusted_pass"],
        gates["reverse_test_pass"],
        gates["random_control_pass"],
        gates["concentration_pass"],
    ]):
        gates["can_enter_phase2"] = True
        gates["final_decision"] = "proceed_to_phase2"

    return gates


def _render_report(
    summary: dict[str, Any],
    by_variant: pd.DataFrame,
    concentration: pd.DataFrame,
    reverse_summary: pd.DataFrame,
) -> str:
    lines = [
        "# MR-v1 Mean Reversion: Fade 20-bar Breakouts with Exit Mechanisms",
        "",
        "## 1. 研究假设",
        "MR-v1 是对趋势跟踪失败后「突破失败 = 均值回归」假设的 Phase 1 验证。核心思路：",
        "- **入场**：fade 20-bar breakout（长突破→做空，短突破→做多）",
        "- **出场**：比较 A/B/C/BC 四种出场机制的 OOS 效果",
        "- **假设**：breakout 在 crypto 4h 级别倾向于失败并回归",
        "",
        "## 2. 入场规则",
        "- 4h closed bar close > 20-bar high → **做空**（fade breakout）",
        "- 4h closed bar close < 20-bar low → **做多**（fade breakdown）",
        "- 一次只持有一个方向",
        "- **不调参，不优化**",
        "",
        "## 3. 出场机制",
        "| Variant | 描述 | Max Hold |",
        "|---|---|---|",
        "| A (Baseline) | 2× ATR 止损 | 60 bars |",
        "| B (Drawdown) | drawdown from peak > 2× entry_ATR 退出 | 60 bars |",
        "| C (1d Lock) | A 的止损被 1d EMA 确认抑制 | 120 bars |",
        "| BC (B+C) | B 的 drawdown 被 1d EMA 确认抑制 | 120 bars |",
        "| D (Random) | 随机入场时间 + A exit | — |",
        "| E (Reverse) | 反向信号 + A exit | — |",
        "",
        "## 4. 各 Variant 汇总",
    ]

    if not by_variant.empty:
        cols = ["variant", "split", "trade_count", "no_cost_pnl", "cost_aware_pnl", "funding_adjusted_pnl", "winning_rate"]
        lines.append(markdown_table(dataframe_records(by_variant), cols))

    lines.extend([
        "",
        "## 5. Reverse Test",
        "正向 (A) vs 反向 (E) 对比：",
    ])
    if not reverse_summary.empty:
        lines.append(markdown_table(dataframe_records(reverse_summary), ["variant", "split", "no_cost_pnl"]))

    lines.extend([
        "",
        "## 6. 集中度",
    ])
    if not concentration.empty:
        conc_cols = ["variant", "split", "symbol", "trade_count", "total_no_cost_pnl", "top_5pct_contribution", "concentration_pass"]
        lines.append(markdown_table(dataframe_records(concentration, limit=20), conc_cols))

    gates = summary.get("gates", {})
    lines.extend([
        "",
        "## 7. Gate 裁决",
        f"- cost_aware_pass={gates.get('cost_aware_pass')}",
        f"- funding_adjusted_pass={gates.get('funding_adjusted_pass')}",
        f"- reverse_test_pass={gates.get('reverse_test_pass')}",
        f"- random_control_pass={gates.get('random_control_pass')}",
        f"- concentration_pass={gates.get('concentration_pass')}",
        f"- can_enter_phase2={gates.get('can_enter_phase2')}",
        f"- final_decision={gates.get('final_decision')}",
        "",
        "## 8. 限制",
        "- strategy_development_allowed=false",
        "- demo_live_allowed=false",
        "- 不修改 OkxAdaptiveMhfStrategy",
    ])

    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
