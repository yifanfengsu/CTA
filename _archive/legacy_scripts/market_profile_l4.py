#!/usr/bin/env python3
"""OKX Market Microstructure L4 Analysis — 市场统计特征复现脚本

Outputs quantitative values for:
  2.1 4h Breakout Reversal Rate
  2.2 ATR Distribution by Symbol
  2.3 Trend Duration Distribution
  2.4 Funding Rate History Stats

Produces a JSON summary for okx-market-microstructure skill.
Also generates per-symbol CSV for downstream analysis.
"""

import argparse
import json
import os
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# ─── Config ───────────────────────────────────────────────────
SYMBOLS = ["BTCUSDT_SWAP_OKX", "ETHUSDT_SWAP_OKX", "SOLUSDT_SWAP_OKX", "LINKUSDT_SWAP_OKX", "DOGEUSDT_SWAP_OKX"]
EXCHANGE = "GLOBAL"
INTERVAL = "1m"
START = "2023-01-01"
END = "2026-04-01"

# OKX contract specs (L1, from API 2026-05-30)
CTVAL = {"BTCUSDT": 0.01, "ETHUSDT": 0.1, "SOLUSDT": 1, "LINKUSDT": 1, "DOGEUSDT": 1000}

# Mapping: DB symbol → short name for funding CSV lookup
SYMBOL_SHORT = {
    "BTCUSDT_SWAP_OKX": "BTCUSDT",
    "ETHUSDT_SWAP_OKX": "ETHUSDT",
    "SOLUSDT_SWAP_OKX": "SOLUSDT",
    "LINKUSDT_SWAP_OKX": "LINKUSDT",
    "DOGEUSDT_SWAP_OKX": "DOGEUSDT",
}

# ─── Data Loading ──────────────────────────────────────────────


def load_1m_bars(db_path: str, symbol: str) -> pd.DataFrame:
    """Load 1m OHLCV bars from vnpy sqlite database."""
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql_query(
            """SELECT datetime, open_price as open, high_price as high,
                      low_price as low, close_price as close, volume
               FROM dbbardata
               WHERE symbol=? AND exchange=? AND interval=?
                 AND datetime>=? AND datetime<?
               ORDER BY datetime""",
            conn,
            params=[symbol, EXCHANGE, INTERVAL, START, END],
        )
    # Explicitly convert to datetime for proper DatetimeIndex
    df["datetime"] = pd.to_datetime(df["datetime"])
    return df


def resample_4h(bars_1m: pd.DataFrame) -> pd.DataFrame:
    """Resample 1m bars to 4h closed bars (no lookahead). Vectorized."""
    if bars_1m.empty:
        return pd.DataFrame()

    df = bars_1m.copy()
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.set_index("datetime").sort_index()

    # Vectorized 4h slot — must include year to avoid cross-year collisions
    slot = df.index.hour // 4 + df.index.dayofyear * 6 + df.index.year * 2200

    # OHLCV aggregation per slot
    result = df.groupby(slot).agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
        count=("close", "count"),
    )

    # Only keep slots with 180+ minutes (75%+ complete)
    result = result[result["count"] >= 180]

    # Build datetime index from bar timestamps
    bar_times = df.groupby(slot)["close"].apply(lambda x: x.index[0])
    bar_times = bar_times.reindex(result.index)  # align with filtered result
    result.index = pd.to_datetime(bar_times).dt.floor("4h")

    return result


# ─── 2.1 Breakout Reversal Rate ────────────────────────────────


def compute_breakout_reversal(
    bars_4h: pd.DataFrame, lookback: int = 8, forward_bars: int = 8
) -> dict:
    """Compute the probability that a breakout reverses within N bars."""
    if len(bars_4h) < lookback + forward_bars + 1:
        return {}

    highs = bars_4h["high"].rolling(lookback).max().shift(1)
    lows = bars_4h["low"].rolling(lookback).min().shift(1)

    # Long breakout: close > lookback high (excluding current bar)
    long_brk = bars_4h["close"] > highs
    # Short breakout: close < lookback low
    short_brk = bars_4h["close"] < lows

    # For long breakouts (fade → go short): reversed if close within N bars < breakout close
    long_reversed = 0
    long_total = 0
    short_reversed = 0
    short_total = 0

    for i in range(len(bars_4h) - forward_bars):
        if long_brk.iloc[i]:
            long_total += 1
            brk_close = bars_4h["close"].iloc[i]
            forward_closes = bars_4h["close"].iloc[i + 1 : i + 1 + forward_bars]
            if any(forward_closes < brk_close):
                long_reversed += 1

        if short_brk.iloc[i]:
            short_total += 1
            brk_close = bars_4h["close"].iloc[i]
            forward_closes = bars_4h["close"].iloc[i + 1 : i + 1 + forward_bars]
            if any(forward_closes > brk_close):
                short_reversed += 1

    return {
        "lookback": lookback,
        "forward_bars": forward_bars,
        "long_breakout_total": int(long_total),
        "long_reversed": int(long_reversed),
        "long_reversal_rate": round(long_reversed / long_total, 4) if long_total else None,
        "short_breakout_total": int(short_total),
        "short_reversed": int(short_reversed),
        "short_reversal_rate": round(short_reversed / short_total, 4) if short_total else None,
    }


# ─── 2.2 ATR Distribution ──────────────────────────────────────


def compute_atr_stats(bars_4h: pd.DataFrame, period: int = 14) -> dict:
    """Compute ATR distribution statistics."""
    if len(bars_4h) < period + 1:
        return {}

    high, low, close = bars_4h["high"], bars_4h["low"], bars_4h["close"]
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    atr = tr.rolling(period).mean().dropna()
    atr_pct = (atr / close.loc[atr.index]) * 100  # percentage ATR

    return {
        "period": period,
        "mean": round(float(atr.mean()), 6),
        "mean_pct": round(float(atr_pct.mean()), 2),
        "median": round(float(atr.median()), 6),
        "q25": round(float(atr.quantile(0.25)), 6),
        "q75": round(float(atr.quantile(0.75)), 6),
        "min": round(float(atr.min()), 6),
        "max": round(float(atr.max()), 6),
        "samples": int(len(atr)),
    }


# ─── 2.3 Trend Duration Distribution ───────────────────────────


def compute_trend_duration(bars_4h: pd.DataFrame) -> dict:
    """Compute the distribution of consecutive same-direction bar streaks."""
    direction = np.sign(bars_4h["close"].diff())
    direction = direction.dropna()

    streaks = []
    current_streak = 1
    for i in range(1, len(direction)):
        if direction.iloc[i] == direction.iloc[i - 1] and direction.iloc[i] != 0:
            current_streak += 1
        else:
            if current_streak > 1:
                streaks.append(current_streak)
            current_streak = 1
    if current_streak > 1:
        streaks.append(current_streak)

    if not streaks:
        return {}

    arr = np.array(streaks)
    return {
        "total_streaks": int(len(arr)),
        "mean": round(float(arr.mean()), 2),
        "median": round(float(np.median(arr)), 2),
        "q25": round(float(np.quantile(arr, 0.25)), 2),
        "q75": round(float(np.quantile(arr, 0.75)), 2),
        "max": int(arr.max()),
        "pct_lt_5": round(float((arr <= 5).sum() / len(arr)) * 100, 1),
        "pct_lt_10": round(float((arr <= 10).sum() / len(arr)) * 100, 1),
    }


# ─── 2.4 Funding Rate Stats ────────────────────────────────────


def compute_funding_stats(funding_dir: str, symbol: str) -> dict:
    """Compute funding rate history statistics from CSV files."""
    short = SYMBOL_SHORT.get(symbol, symbol)
    inst_id = short.replace("USDT", "-USDT-SWAP")
    csv_path = os.path.join(funding_dir, f"{inst_id}_funding_2023-01-01_2026-03-31.csv")

    if not os.path.exists(csv_path):
        return {"error": f"File not found: {csv_path}"}

    try:
        df = pd.read_csv(csv_path)
        if "funding_rate" not in df.columns:
            return {"error": "No funding_rate column"}

        rates = df["funding_rate"].dropna()
        annualized = rates * 3 * 365  # 3 settlements/day × 365 days

        return {
            "samples": int(len(rates)),
            "mean_rate": round(float(rates.mean()), 8),
            "annualized_mean_pct": round(float(annualized.mean() * 100), 2),
            "positive_pct": round(float((rates > 0).sum() / len(rates)) * 100, 1),
            "std_rate": round(float(rates.std()), 8),
            "min_rate": round(float(rates.min()), 8),
            "max_rate": round(float(rates.max()), 8),
        }
    except Exception as e:
        return {"error": str(e)}


# ─── Main ──────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="OKX Market Microstructure L4 Analysis")
    parser.add_argument("--db", default=os.path.expanduser("~/.vntrader/database.db"), help="vnpy sqlite database")
    parser.add_argument("--funding-dir", default="data/funding/okx", help="Funding rate CSV directory")
    parser.add_argument("--output", default="market_profile_summary.json", help="Output JSON path")
    parser.add_argument("--lookback", type=int, default=8, help="Breakout lookback (4h bars)")
    parser.add_argument("--forward", type=int, default=8, help="Forward window for reversal check")
    args = parser.parse_args()

    results = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "data_range": f"{START} to {END}",
        "timeframe": "4h (resampled from 1m)",
        "lookback": args.lookback,
        "forward_bars": args.forward,
        "by_symbol": {},
    }

    for sym in SYMBOLS:
        print(f"Processing {sym}...")
        bars_1m = load_1m_bars(args.db, sym)
        bars_4h = resample_4h(bars_1m)
        print(f"  {len(bars_1m):,} 1m bars → {len(bars_4h):,} 4h bars")

        symbol_result = {
            "bars_1m": len(bars_1m),
            "bars_4h": len(bars_4h),
            "atr_stats": compute_atr_stats(bars_4h),
            "breakout_reversal": compute_breakout_reversal(bars_4h, args.lookback, args.forward),
            "trend_duration": compute_trend_duration(bars_4h),
            "funding_stats": compute_funding_stats(args.funding_dir, sym),
        }
        results["by_symbol"][sym] = symbol_result

        # Print summary
        br = symbol_result["breakout_reversal"]
        atr = symbol_result["atr_stats"]
        td = symbol_result["trend_duration"]
        print(f"  ATR mean={atr.get('mean', 'N/A')}, "
              f"long_rev={br.get('long_reversal_rate', 'N/A')}, "
              f"short_rev={br.get('short_reversal_rate', 'N/A')}, "
              f"trend_median={td.get('median', 'N/A')} bars")

    # Write output
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nDone. Output: {args.output}")


if __name__ == "__main__":
    main()
