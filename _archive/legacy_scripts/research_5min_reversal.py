#!/usr/bin/env python3
"""5min Breakout Reversal Rate — vectorized version.

Checks: at 5min resolution, do breakouts reverse frequently enough
to support a fade strategy?

Uses numpy rolling windows for O(n) performance.
"""

from __future__ import annotations

import argparse
import re
import sqlite3
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

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

from common_runtime import PROJECT_ROOT, ensure_headless_runtime
from history_time_utils import HistoryRange, parse_history_range

DEFAULT_SYMBOLS = [
    "BTCUSDT_SWAP_OKX.GLOBAL", "ETHUSDT_SWAP_OKX.GLOBAL",
    "SOLUSDT_SWAP_OKX.GLOBAL", "LINKUSDT_SWAP_OKX.GLOBAL",
    "DOGEUSDT_SWAP_OKX.GLOBAL",
]
DEFAULT_START = "2023-01-01"
DEFAULT_END = "2026-03-31"
DEFAULT_TIMEZONE = "Asia/Shanghai"
DEFAULT_DATABASE_PATH = PROJECT_ROOT / ".vntrader" / "database.db"

# 5min bars: 12 = 1h, 24 = 2h, 48 = 4h, 96 = 8h
LOOKBACKS = [12, 24, 48, 96]
HOLD_MULTS = [1, 2]


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


def resample_tf(bars_1m: pd.DataFrame, minutes: int, hr: HistoryRange) -> pd.DataFrame:
    cols = ["open_time", "datetime", "open", "high", "low", "close", "volume"]
    if bars_1m.empty:
        return pd.DataFrame(columns=cols)
    w = bars_1m.sort_values("datetime", kind="stable").drop_duplicates("datetime", keep="last").copy()
    anchor = pd.Timestamp(hr.start)
    if anchor.tzinfo is None:
        anchor = anchor.tz_localize(w["datetime"].iloc[0].tz)
    deltas = (w["datetime"] - anchor) / pd.Timedelta(minutes=1)
    w = w.loc[deltas >= 0].copy()
    w["_slot"] = np.floor(deltas.loc[w.index].to_numpy(dtype=float) / minutes).astype(np.int64)
    g = w.groupby("_slot", sort=True, dropna=False)
    r = g.agg(open_time=("datetime", "min"), datetime=("datetime", "max"),
              open=("open", "first"), high=("high", "max"), low=("low", "min"),
              close=("close", "last"), volume=("volume", "sum"), mc=("datetime", "size"))
    min_req = max(1, int(minutes * 0.75))
    return r[r["mc"] >= min_req].drop(columns=["mc"]).dropna(subset=["open", "high", "low", "close"]).reset_index(drop=True).loc[:, cols]


@dataclass
class RevResult:
    name: str
    lookback: int
    hold_bars: int
    long_sig: int
    long_rev: int
    short_sig: int
    short_rev: int


def analyze_vectorized(bars: pd.DataFrame, lookback: int, hold_bars: int) -> RevResult:
    """Vectorized breakout reversal analysis.

    Returns count of signals and reversals for long and short.
    """
    close = bars["close"].values
    high = bars["high"].values
    low = bars["low"].values
    n = len(bars)

    if n < lookback + hold_bars + 2:
        return RevResult("", lookback, hold_bars, 0, 0, 0, 0)

    # Rolling Donchian with shift(1) — use pandas for speed
    dh = pd.Series(high).rolling(lookback).max().shift(1).values
    dl = pd.Series(low).rolling(lookback).min().shift(1).values

    valid = ~np.isnan(dh) & ~np.isnan(dl)
    valid[:lookback] = False
    valid[n - hold_bars:] = False  # need hold_bars of future data

    idx = np.where(valid)[0]
    if len(idx) == 0:
        return RevResult("", lookback, hold_bars, 0, 0, 0, 0)

    c = close[idx]
    dh_val = dh[idx]
    dl_val = dl[idx]

    long_mask = c < dl_val
    short_mask = c > dh_val

    long_idx = idx[long_mask]
    short_idx = idx[short_mask]

    long_rev = 0
    if len(long_idx) > 0:
        for i in long_idx:
            future = close[i + 1:i + hold_bars + 1]
            if np.any(future > close[i]):
                long_rev += 1

    short_rev = 0
    if len(short_idx) > 0:
        for i in short_idx:
            future = close[i + 1:i + hold_bars + 1]
            if np.any(future < close[i]):
                short_rev += 1

    return RevResult(
        name="", lookback=lookback, hold_bars=hold_bars,
        long_sig=len(long_idx), long_rev=long_rev,
        short_sig=len(short_idx), short_rev=short_rev,
    )


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS))
    p.add_argument("--start", default=DEFAULT_START)
    p.add_argument("--end", default=DEFAULT_END)
    p.add_argument("--timezone", default=DEFAULT_TIMEZONE)
    p.add_argument("--database-path", default=str(DEFAULT_DATABASE_PATH))
    return p.parse_args(argv)


def main(argv=None):
    ensure_headless_runtime()
    args = parse_args(argv)
    symbols = [s.strip() for s in re.split(r"[\s,]+", args.symbols) if s.strip()]
    db = Path(args.database_path)
    hr = parse_history_range(args.start, args.end, timedelta(minutes=1), args.timezone)

    all_bars: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        s, _, e = sym.partition(".")
        qs = hr.start.replace(tzinfo=None).isoformat(sep=" ", timespec="seconds")
        qe = hr.end_exclusive.replace(tzinfo=None).isoformat(sep=" ", timespec="seconds")
        with sqlite3.connect(db) as conn:
            df = pd.read_sql_query(
                "select datetime, open_price as open, high_price as high, low_price as low, close_price as close, volume "
                "from dbbardata where symbol=? and exchange=? and interval='1m' and datetime>=? and datetime<? order by datetime",
                conn, params=(s, e, qs, qe))
        df = normalize_1m(df, hr.timezone_name)
        if not df.empty:
            b5 = resample_tf(df, 5, hr)
            if not b5.empty:
                all_bars[sym] = b5
                print(f"  {sym.split('_')[0]:>5}: {len(b5):,} bars  |  first={b5['datetime'].iloc[0].strftime('%Y-%m-%d %H:%M')}  last={b5['datetime'].iloc[-1].strftime('%Y-%m-%d %H:%M')}")

    if not all_bars:
        print("ERROR: No data")
        return 1

    for hold_mult in HOLD_MULTS:
        print(f"\n{'='*100}")
        print(f"Holding period: {hold_mult}× lookback (must reverse within {hold_mult}×N bars)")
        print(f"{'='*100}")
        print(f"{'Lookback':>8} {'Hold':>5} {'Hours':>5} {'L_Sig':>7} {'L_Rev%':>7} {'S_Sig':>7} {'S_Rev%':>7} {'Total':>7} {'Total%':>7}")
        print("-" * 80)

        for lb in LOOKBACKS:
            comb = RevResult("", lb, lb * hold_mult, 0, 0, 0, 0)
            for sym, bars in all_bars.items():
                r = analyze_vectorized(bars, lb, lb * hold_mult)
                comb.long_sig += r.long_sig
                comb.long_rev += r.long_rev
                comb.short_sig += r.short_sig
                comb.short_rev += r.short_rev

            total_sig = comb.long_sig + comb.short_sig
            total_rev = comb.long_rev + comb.short_rev
            print(f"{lb:>8}  {lb * hold_mult:>5}  {lb * hold_mult // 12:>5}  "
                  f"{comb.long_sig:>7,}  {comb.long_rev/max(comb.long_sig,1)*100:>6.1f}%  "
                  f"{comb.short_sig:>7,}  {comb.short_rev/max(comb.short_sig,1)*100:>6.1f}%  "
                  f"{total_sig:>7,}  {total_rev/max(total_sig,1)*100:>6.1f}%")

        # Per-symbol
        print(f"\n  Per-symbol (best lookback):")
        for lb in LOOKBACKS:
            print(f"  Lookback={lb}:")
            for sym in all_bars:
                r = analyze_vectorized(all_bars[sym], lb, lb * hold_mult)
                ts = r.long_sig + r.short_sig
                tr = r.long_rev + r.short_rev
                print(f"    {sym.split('_')[0]:>5}:  long={r.long_sig:>5,} ({r.long_rev/max(r.long_sig,1)*100:>5.1f}%)  "
                      f"short={r.short_sig:>5,} ({r.short_rev/max(r.short_sig,1)*100:>5.1f}%)  "
                      f"total={ts:>6,} ({tr/max(ts,1)*100:>5.1f}%)")

    print(f"\n--- 4h Baseline (Donchian lookback=8, hold=8) ---")
    print(f"  4h: reversal rate = 78-82%")

    print(f"\n--- Cost note ---")
    print(f"  OKX Perp: Maker=0.02%, Taker=0.05% per side")
    print(f"  Round-trip maker-only: 4 bps, taker: 10 bps")
    print(f"  5min trades need sufficient edge per trade to overcome fee")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
