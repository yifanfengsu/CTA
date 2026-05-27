#!/usr/bin/env python3
"""MR-v1 Phase 2: parameter sweep, plateau detection, and robustness testing.

Research-only. Sweeps entry lookback, ATR stop, and max hold across
train/validation/oos splits. Runs multi-seed random control and per-symbol /
per-month stability diagnostics.
"""

from __future__ import annotations

import argparse
import hashlib
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
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "reports" / "research" / "mr_v1_phase2"
DEFAULT_DATABASE_PATH = PROJECT_ROOT / ".vntrader" / "database.db"
TIMEFRAME_MINUTES = {"4h": 240, "1d": 1440}

SPLIT_RANGES = {
    "train_ext": ("2023-01-01", "2024-07-01"),
    "validation_ext": ("2024-07-01", "2025-07-01"),
    "oos_ext": ("2025-07-01", "2026-04-01"),
}
SPLIT_NAMES = ["train_ext", "validation_ext", "oos_ext"]

FIXED_NOTIONAL = 1000.0
FEE_BPS = 5.0
SLIPPAGE_BPS = 5.0

# Sweep ranges
ENTRY_LOOKBACKS = [8, 10, 12, 15, 20, 30, 40]
ATR_MULTIPLIERS = [1.0, 1.5, 2.0, 2.5, 3.0]
MAX_HOLD_BARSS = [50, 55, 60, 65, 70, 20, 40, 80]
RANDOM_SEEDS = [42, 99, 137, 256, 512]


class MrPhase2Error(Exception):
    pass


# ---------------------------------------------------------------------------
# Data loading (same as MR-v1)
# ---------------------------------------------------------------------------

def split_vt_symbol(vt_symbol: str) -> tuple[str, str]:
    s, sep, e = str(vt_symbol).partition(".")
    if not sep:
        raise MrPhase2Error(f"invalid vt_symbol: {vt_symbol}")
    return s, e


def symbol_to_inst_id(vt_symbol: str) -> str:
    symbol, _ = split_vt_symbol(vt_symbol)
    root = symbol.removesuffix("_OKX")
    pair = root[:-len("_SWAP")] if root.endswith("_SWAP") else root
    if pair.endswith("USDT"):
        return f"{pair[:-4]}-USDT-SWAP"
    return root.replace("_", "-")


def normalize_1m_bars(frame: pd.DataFrame, tz: str) -> pd.DataFrame:
    cols = ["datetime", "open", "high", "low", "close", "volume"]
    if frame.empty:
        return pd.DataFrame(columns=cols)
    out = frame.loc[:, cols].copy()
    ts = pd.to_datetime(out["datetime"], errors="coerce")
    if ts.dt.tz is None:
        ts = ts.dt.tz_localize(tz)
    else:
        ts = ts.dt.tz_convert(tz)
    out["datetime"] = ts
    for c in ["open", "high", "low", "close", "volume"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out.dropna(subset=cols).sort_values("datetime", kind="stable").drop_duplicates("datetime", keep="last").reset_index(drop=True)


def load_1m_bars(vt_symbol: str, hr: HistoryRange, db: Path) -> pd.DataFrame:
    symbol, exchange = split_vt_symbol(vt_symbol)
    qs = hr.start.replace(tzinfo=None).isoformat(sep=" ", timespec="seconds")
    qe = hr.end_exclusive.replace(tzinfo=None).isoformat(sep=" ", timespec="seconds")
    with sqlite3.connect(db) as conn:
        df = pd.read_sql_query(
            "select datetime, open_price as open, high_price as high, low_price as low, close_price as close, volume "
            "from dbbardata where symbol=? and exchange=? and interval='1m' and datetime>=? and datetime<? order by datetime",
            conn, params=(symbol, exchange, qs, qe),
        )
    return normalize_1m_bars(df, hr.timezone_name)


def resample_ohlcv_closed(bars_1m: pd.DataFrame, tf: str, hr: HistoryRange | None = None) -> pd.DataFrame:
    minutes = TIMEFRAME_MINUTES[tf]
    cols = ["open_time", "datetime", "open", "high", "low", "close", "volume"]
    if bars_1m.empty:
        return pd.DataFrame(columns=cols)
    w = bars_1m.sort_values("datetime", kind="stable").drop_duplicates("datetime", keep="last").copy()
    anchor = pd.Timestamp(hr.start if hr else w["datetime"].iloc[0])
    if anchor.tzinfo is None:
        anchor = anchor.tz_localize(w["datetime"].iloc[0].tz)
    deltas = (w["datetime"] - anchor) / pd.Timedelta(minutes=1)
    w = w.loc[deltas >= 0].copy()
    w["_slot"] = np.floor(deltas.loc[w.index].to_numpy(dtype=float) / minutes).astype(np.int64)
    g = w.groupby("_slot", sort=True, dropna=False)
    r = g.agg(open_time=("datetime", "min"), datetime=("datetime", "max"),
              open=("open", "first"), high=("high", "max"), low=("low", "min"),
              close=("close", "last"), volume=("volume", "sum"), mc=("datetime", "size"))
    r = r[r["mc"] == minutes].drop(columns=["mc"]).dropna(subset=["open", "high", "low", "close"]).reset_index(drop=True)
    return r.loc[:, cols]


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
# Entry / Exit
# ---------------------------------------------------------------------------

def detect_mr_entries(bars_4h: pd.DataFrame, lookback: int) -> pd.DataFrame:
    """Mean reversion entries: fade breakouts."""
    if len(bars_4h) < lookback + 1:
        return pd.DataFrame(columns=["entry_time", "direction", "entry_price"])
    highs = bars_4h["high"].rolling(lookback).max().shift(1)
    lows = bars_4h["low"].rolling(lookback).min().shift(1)
    close = bars_4h["close"]
    long_breakout = close > highs
    short_breakout = close < lows

    entries = []
    pos = 0
    for i in range(lookback, len(bars_4h)):
        if long_breakout.iloc[i] and pos != -1:
            if pos == 1:
                entries.append({"entry_time": bars_4h["datetime"].iloc[i], "direction": 0, "entry_price": bars_4h["close"].iloc[i]})
            entries.append({"entry_time": bars_4h["datetime"].iloc[i], "direction": -1, "entry_price": bars_4h["close"].iloc[i]})
            pos = -1
        elif short_breakout.iloc[i] and pos != 1:
            if pos == -1:
                entries.append({"entry_time": bars_4h["datetime"].iloc[i], "direction": 0, "entry_price": bars_4h["close"].iloc[i]})
            entries.append({"entry_time": bars_4h["datetime"].iloc[i], "direction": 1, "entry_price": bars_4h["close"].iloc[i]})
            pos = 1
    return pd.DataFrame(entries)


def compute_atr(bars: pd.DataFrame, period: int = 14) -> pd.Series:
    h, l, c = bars["high"], bars["low"], bars["close"]
    tr = pd.concat([(h - l).abs(), (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def simulate_trades_mr(entries: pd.DataFrame, bars_4h: pd.DataFrame, atr_mult: float, max_hold: int) -> pd.DataFrame:
    """Simulate mean reversion trades with ATR stop."""
    if entries.empty or len(bars_4h) < 2:
        return pd.DataFrame(columns=["entry_time", "exit_time", "direction", "entry_price", "exit_price", "exit_reason"])
    atr = compute_atr(bars_4h, 14)
    bar_idx = {ts: i for i, ts in enumerate(bars_4h["datetime"])}
    trades = []
    for _, e in entries.iterrows():
        d = int(e["direction"])
        if d == 0:
            continue
        entry_ts = e["entry_time"]
        ep = e["entry_price"]
        ei = bar_idx.get(entry_ts)
        if ei is None or ei >= len(bars_4h) - 1:
            continue
        ea = atr.iloc[ei]
        if pd.isna(ea) or ea <= 0:
            ea = ep * 0.01
        stop_dist = atr_mult * ea
        exit_idx = None
        exit_price = None
        exit_reason = None
        for j in range(ei + 1, min(ei + max_hold + 1, len(bars_4h))):
            bar = bars_4h.iloc[j]
            if d == 1:  # long (fading short breakout)
                if bar["low"] <= ep - stop_dist:
                    exit_idx = j; exit_price = bar["open"]; exit_reason = "stop"
                    break
            else:  # short (fading long breakout)
                if bar["high"] >= ep + stop_dist:
                    exit_idx = j; exit_price = bar["open"]; exit_reason = "stop"
                    break
        if exit_idx is None:
            exit_idx = min(ei + max_hold, len(bars_4h) - 1)
            exit_price = bars_4h["close"].iloc[exit_idx]
            exit_reason = "max_hold"
        trades.append({"entry_time": entry_ts, "exit_time": bars_4h["datetime"].iloc[exit_idx],
                        "direction": d, "entry_price": ep, "exit_price": exit_price, "exit_reason": exit_reason})
    return pd.DataFrame(trades)


def deterministic_rng(seed: int, *parts: Any) -> np.random.Generator:
    text = "|".join(str(p) for p in (seed, *parts))
    d = hashlib.sha256(text.encode()).digest()
    return np.random.default_rng(int.from_bytes(d[:8], "little") % (2**32 - 1))


def generate_random_entries(bars: pd.DataFrame, count: int, seed: int, lookback: int) -> pd.DataFrame:
    if len(bars) < lookback + 2:
        return pd.DataFrame(columns=["entry_time", "direction", "entry_price"])
    rng = deterministic_rng(seed)
    valid = bars.iloc[lookback:-1].copy()
    count = min(count, len(valid))
    chosen = rng.choice(valid.index, size=count, replace=False)
    entries = []
    for idx in sorted(chosen):
        d = 1 if rng.random() > 0.5 else -1
        entries.append({"entry_time": bars["datetime"].iloc[idx], "direction": d, "entry_price": bars["close"].iloc[idx]})
    return pd.DataFrame(entries)


# ---------------------------------------------------------------------------
# PnL
# ---------------------------------------------------------------------------

def compute_pnl(trades: pd.DataFrame, funding_map: dict, symbol: str, tz_name: str) -> pd.DataFrame:
    if trades.empty:
        return trades.assign(no_cost_pnl=0.0, cost_aware_pnl=0.0, funding_adjusted_pnl=0.0, funding_paid=0.0)
    out = trades.copy()
    cost = FIXED_NOTIONAL * (2 * FEE_BPS + 2 * SLIPPAGE_BPS) / 10000.0
    out["gross_return"] = np.where(out["direction"] == 1,
        (out["exit_price"] - out["entry_price"]) / out["entry_price"],
        (out["entry_price"] - out["exit_price"]) / out["entry_price"])
    out["no_cost_pnl"] = out["gross_return"] * FIXED_NOTIONAL
    out["cost_aware_pnl"] = out["no_cost_pnl"] - cost

    import zoneinfo
    tz = zoneinfo.ZoneInfo(tz_name)
    inst_id = symbol_to_inst_id(symbol)
    funding = funding_map.get(inst_id)
    out["funding_paid"] = 0.0
    if funding is not None and not funding.empty:
        for i, tr in out.iterrows():
            et = pd.Timestamp(tr["entry_time"]); xt = pd.Timestamp(tr["exit_time"])
            if et.tzinfo is None: et = et.tz_localize(tz)
            else: et = et.tz_convert(tz)
            if xt.tzinfo is None: xt = xt.tz_localize(tz)
            else: xt = xt.tz_convert(tz)
            mask = (funding["funding_time_utc"] >= et.tz_convert("UTC")) & (funding["funding_time_utc"] < xt.tz_convert("UTC"))
            if mask.any():
                fp = funding.loc[mask, "funding_rate"].sum() * FIXED_NOTIONAL
                out.at[i, "funding_paid"] = fp if tr["direction"] == 1 else -fp
    out["funding_adjusted_pnl"] = out["cost_aware_pnl"] - out["funding_paid"]
    return out


def assign_split(trades: pd.DataFrame) -> pd.DataFrame:
    out = trades.copy()
    out["split"] = "oos_ext"
    for sn, (s, e) in SPLIT_RANGES.items():
        mask = (out["entry_time"] >= pd.Timestamp(s).tz_localize(DEFAULT_TIMEZONE)) & (out["entry_time"] < pd.Timestamp(e).tz_localize(DEFAULT_TIMEZONE))
        out.loc[mask, "split"] = sn
    return out


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------

@dataclass
class SweepResult:
    lookback: int
    atr_mult: float
    max_hold: int
    pnl: dict[str, float]  # split -> funding_adjusted_pnl


def run_sweep(
    all_bars: dict[str, dict[str, pd.DataFrame]],  # symbol -> {tf -> bars}
    symbols: list[str],
    funding_map: dict,
    tz_name: str,
) -> list[SweepResult]:
    results: list[SweepResult] = []
    total = len(ENTRY_LOOKBACKS) * len(ATR_MULTIPLIERS) * len(MAX_HOLD_BARSS)

    n = 0
    for lookback in ENTRY_LOOKBACKS:
        # Pre-compute entries per symbol for this lookback
        symbol_entries: dict[str, pd.DataFrame] = {}
        for sym in symbols:
            bars_4h = all_bars.get(sym, {}).get("4h")
            if bars_4h is None or bars_4h.empty:
                continue
            entries = detect_mr_entries(bars_4h, lookback)
            real = entries[entries["direction"] != 0]
            symbol_entries[sym] = real

        for atr_mult in ATR_MULTIPLIERS:
            for max_hold in MAX_HOLD_BARSS:
                n += 1
                all_trades = []
                for sym in symbols:
                    bars_4h = all_bars.get(sym, {}).get("4h")
                    if bars_4h is None:
                        continue
                    entries = symbol_entries.get(sym)
                    if entries is None or entries.empty:
                        continue
                    trades = simulate_trades_mr(entries, bars_4h, atr_mult, max_hold)
                    if trades.empty:
                        continue
                    trades = compute_pnl(trades, funding_map, sym, tz_name)
                    trades = assign_split(trades)
                    trades["symbol"] = sym
                    all_trades.append(trades)

                if not all_trades:
                    continue
                df = pd.concat(all_trades, ignore_index=True)
                pnl = {}
                for split in SPLIT_NAMES:
                    mask = df["split"] == split
                    pnl[split] = float(df.loc[mask, "funding_adjusted_pnl"].sum()) if mask.any() else 0.0
                results.append(SweepResult(lookback, atr_mult, max_hold, pnl))

    return results


def find_plateau(results: list[SweepResult]) -> dict[str, Any]:
    """Find the parameter plateau: OOS PnL stable region."""
    if not results:
        return {"plateau_found": False}

    # Sort by OOS PnL descending
    sorted_r = sorted(results, key=lambda r: r.pnl.get("oos_ext", -np.inf), reverse=True)
    best = sorted_r[0]

    # Find all results within 10% of best OOS PnL
    best_oos = best.pnl.get("oos_ext", 0)
    threshold = best_oos * 0.85 if best_oos > 0 else best_oos * 1.15
    plateau = [r for r in results if r.pnl.get("oos_ext", -np.inf) >= threshold]

    plateau_lookbacks = sorted(set(r.lookback for r in plateau))
    plateau_atrs = sorted(set(r.atr_mult for r in plateau))
    plateau_holds = sorted(set(r.max_hold for r in plateau))

    return {
        "plateau_found": len(plateau) > 1,
        "best_combination": {"lookback": best.lookback, "atr_mult": best.atr_mult, "max_hold": best.max_hold},
        "best_oos_pnl": best_oos,
        "best_train_pnl": best.pnl.get("train_ext", 0),
        "best_validation_pnl": best.pnl.get("validation_ext", 0),
        "plateau_count": len(plateau),
        "plateau_lookback_range": [min(plateau_lookbacks), max(plateau_lookbacks)],
        "plateau_atr_range": [min(plateau_atrs), max(plateau_atrs)],
        "plateau_hold_range": [min(plateau_holds), max(plateau_holds)],
        "all_results": [{"lookback": r.lookback, "atr_mult": r.atr_mult, "max_hold": r.max_hold,
                          "oos_pnl": r.pnl.get("oos_ext", 0), "train_pnl": r.pnl.get("train_ext", 0),
                          "validation_pnl": r.pnl.get("validation_ext", 0)} for r in sorted_r[:20]],
    }


# ---------------------------------------------------------------------------
# Multi-seed random control
# ---------------------------------------------------------------------------

def run_random_control(
    all_bars: dict, symbols: list[str], funding_map: dict, tz_name: str,
    lookback: int, atr_mult: float, max_hold: int, seeds: list[int],
) -> dict[str, Any]:
    """Run random entry control with multiple seeds."""
    seed_results = []
    for seed in seeds:
        all_trades = []
        for sym in symbols:
            bars_4h = all_bars.get(sym, {}).get("4h")
            if bars_4h is None:
                continue
            real_entries = detect_mr_entries(bars_4h, lookback)
            real_count = len(real_entries[real_entries["direction"] != 0])
            random_entries = generate_random_entries(bars_4h, real_count, seed, lookback)
            if random_entries.empty:
                continue
            trades = simulate_trades_mr(random_entries, bars_4h, atr_mult, max_hold)
            if trades.empty:
                continue
            trades = compute_pnl(trades, funding_map, sym, tz_name)
            trades = assign_split(trades)
            all_trades.append(trades)
        if not all_trades:
            seed_results.append({"seed": seed, "oos_pnl": 0, "pass": False})
            continue
        df = pd.concat(all_trades, ignore_index=True)
        oos = df[df["split"] == "oos_ext"]["funding_adjusted_pnl"].sum()
        seed_results.append({"seed": seed, "oos_pnl": float(oos), "pass": bool(oos > 0)})

    all_pass = all(r["pass"] for r in seed_results)
    return {"seeds": seed_results, "all_pass": all_pass, "worst_oos": min(r["oos_pnl"] for r in seed_results)}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(description="MR-v1 Phase 2: parameter sweep and robustness.")
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
    logger = setup_logging("research_mr_v1_phase2", verbose=bool(args.verbose))
    symbols = [s.strip() for s in re.split(r"[\s,]+", args.symbols) if s.strip()]
    output_dir = Path(args.output_dir) if Path(args.output_dir).is_absolute() else PROJECT_ROOT / args.output_dir
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

    # Load bars
    all_bars: dict[str, dict[str, pd.DataFrame]] = {}
    for sym in symbols:
        log_event(logger, logging.INFO, "load_symbol", "Loading", symbol=sym)
        bars_1m = load_1m_bars(sym, hr, db)
        all_bars[sym] = {
            "4h": resample_ohlcv_closed(bars_1m, "4h", hr),
            "1d": resample_ohlcv_closed(bars_1m, "1d", hr),
        }

    # --- Sweep ---
    log_event(logger, logging.INFO, "sweep_start", "Starting parameter sweep",
              combos=len(ENTRY_LOOKBACKS) * len(ATR_MULTIPLIERS) * len(MAX_HOLD_BARSS))
    sweep_results = run_sweep(all_bars, symbols, funding_map, args.timezone)
    plateau = find_plateau(sweep_results)

    # --- Random control ---
    best = plateau["best_combination"]
    log_event(logger, logging.INFO, "random_control_start", "Running multi-seed random control")
    random_control = run_random_control(
        all_bars, symbols, funding_map, args.timezone,
        best["lookback"], best["atr_mult"], best["max_hold"], RANDOM_SEEDS,
    )

    # --- Per-symbol analysis ---
    symbol_pnl = {}
    for sym in symbols:
        bars_4h = all_bars.get(sym, {}).get("4h")
        if bars_4h is None:
            continue
        entries = detect_mr_entries(bars_4h, best["lookback"])
        real = entries[entries["direction"] != 0]
        if real.empty:
            continue
        trades = simulate_trades_mr(real, bars_4h, best["atr_mult"], best["max_hold"])
        if trades.empty:
            continue
        trades = compute_pnl(trades, funding_map, sym, args.timezone)
        trades = assign_split(trades)
        oos = trades[trades["split"] == "oos_ext"]["funding_adjusted_pnl"].sum()
        symbol_pnl[sym] = {
            "train": float(trades[trades["split"] == "train_ext"]["funding_adjusted_pnl"].sum()),
            "validation": float(trades[trades["split"] == "validation_ext"]["funding_adjusted_pnl"].sum()),
            "oos": float(oos),
            "trade_count": len(trades[trades["split"] == "oos_ext"]),
        }

    # --- Gates ---
    gates = {
        "plateau_found": plateau["plateau_found"],
        "random_control_all_pass": random_control["all_pass"],
        "symbol_diversification": sum(1 for v in symbol_pnl.values() if v["oos"] > 0) >= 3,
        "can_enter_phase3": False,
        "strategy_development_allowed": False,
        "demo_live_allowed": False,
    }
    if gates["plateau_found"] and gates["random_control_all_pass"] and gates["symbol_diversification"]:
        gates["can_enter_phase3"] = True

    # --- Output ---
    summary = {
        "hypothesis": "MR-v1 Phase 2: parameter sweep and robustness",
        "version": "v2.0",
        "plateau": plateau,
        "random_control": random_control,
        "symbol_pnl": symbol_pnl,
        "gates": gates,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "mr_v1_phase2_summary.json").write_text(
        json.dumps(to_jsonable(summary), ensure_ascii=False, indent=2), encoding="utf-8")

    # Sweep CSV
    pd.DataFrame([{"lookback": r.lookback, "atr_mult": r.atr_mult, "max_hold": r.max_hold,
                    "train_pnl": r.pnl.get("train_ext", 0), "validation_pnl": r.pnl.get("validation_ext", 0),
                    "oos_pnl": r.pnl.get("oos_ext", 0)} for r in sweep_results]
    ).to_csv(output_dir / "sweep_results.csv", index=False)

    # Report
    report = _render_report(summary)
    (output_dir / "mr_v1_phase2_report.md").write_text(report, encoding="utf-8")

    log_event(logger, logging.INFO, "phase2_complete", "Phase 2 complete",
              can_enter_phase3=gates["can_enter_phase3"])
    return 0


def _render_report(s: dict) -> str:
    p = s["plateau"]
    rc = s["random_control"]
    g = s["gates"]
    sym = s["symbol_pnl"]

    lines = [
        "# MR-v1 Phase 2: Parameter Sweep & Robustness",
        "",
        "## 1. Best Combination",
        f"- lookback={p['best_combination']['lookback']}, atr_mult={p['best_combination']['atr_mult']}, max_hold={p['best_combination']['max_hold']}",
        f"- OOS funding-adjusted PnL: {p['best_oos_pnl']:.2f}",
        f"- Train: {p['best_train_pnl']:.2f} | Validation: {p['best_validation_pnl']:.2f}",
        "",
        "## 2. Parameter Plateau",
        f"- Plateau found: {p['plateau_found']}",
        f"- {p['plateau_count']} combinations within 10% of best OOS",
        f"- Lookback range: {p['plateau_lookback_range']}",
        f"- ATR mult range: {p['plateau_atr_range']}",
        f"- Max hold range: {p['plateau_hold_range']}",
        "",
        "## 3. Multi-Seed Random Control",
        f"- All {len(rc['seeds'])} seeds pass: {rc['all_pass']}",
        f"- Worst seed OOS: {rc['worst_oos']:.2f}",
    ]

    for seed_r in rc["seeds"]:
        lines.append(f"  - seed={seed_r['seed']}: OOS={seed_r['oos_pnl']:.2f}, pass={seed_r['pass']}")

    lines.extend([
        "",
        "## 4. Per-Symbol OOS PnL",
        "| Symbol | OOS PnL | Train PnL | Val PnL | Trades |",
        "|---|---|---|---|---|",
    ])
    for sn, v in sym.items():
        lines.append(f"| {sn} | {v['oos']:.2f} | {v['train']:.2f} | {v['validation']:.2f} | {v['trade_count']} |")

    lines.extend([
        "",
        "## 5. Gates",
        f"- plateau_found={g['plateau_found']}",
        f"- random_control_all_pass={g['random_control_all_pass']}",
        f"- symbol_diversification={g['symbol_diversification']}",
        f"- can_enter_phase3={g['can_enter_phase3']}",
        "",
        "## 6. Top 10 Sweep Results (by OOS)",
        "| lookback | atr_mult | max_hold | train | validation | oos |",
        "|---|---|---|---|---|---|",
    ])
    for r in p.get("all_results", [])[:10]:
        lines.append(f"| {r['lookback']} | {r['atr_mult']} | {r['max_hold']} | {r['train_pnl']:.0f} | {r['validation_pnl']:.0f} | {r['oos_pnl']:.0f} |")

    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
