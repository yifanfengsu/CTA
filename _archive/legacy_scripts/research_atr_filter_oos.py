#!/usr/bin/env python3
"""ATR regime-filter p30 threshold — out-of-sample validation + sensitivity.

PURPOSE
  The live MR-5m ATR regime filter uses static per-symbol thresholds
  (BTC=81.5 / ETH=4.64 / SOL=0.245 / LINK=0.0212 / DOGE=0.0002). These were
  originally derived as p30 of *SMA*-ATR over the *full* sample (see
  scripts/backtest_mr_5m_v2.py / fix_mr_5m.py) — i.e. in-sample. This script
  answers two questions WITHOUT touching the live runner, the strategy, or the
  baseline engine:

    Q1  Out-of-sample value: on a held-out test set, does filter-on (threshold =
        train-derived p30) beat filter-off (threshold = 0, all signals pass)?
    Q2  Sensitivity: across train p20/p25/p30/p35/p40, is test-set net PnL a
        smooth/monotone surface (robust) or a jagged spike at p30 (overfit)?

DESIGN (faithful reuse, no engine modification)
  * The baseline engine scripts/backtest_mr_5m_compare.py (`bt`) is the single
    source of truth for data loading, Wilder ATR, Donchian, fills, fees, sizing
    and the trade loop. We do NOT re-implement any of it.
  * The engine reads its threshold from the module global bt.ATR_THRESHOLDS.
    We run the exact engine loop bt.backtest_symbol() repeatedly, monkeypatching
    that global to the threshold under test. The engine file is never edited.
  * Train/test split is by calendar: train = earliest 2/3 of the data span,
    test = latest 1/3 (strict time order, never shuffled).
  * train_pXX = the XXth percentile of *Wilder* ATR (the quantity the engine
    actually thresholds) computed on TRAIN bars only.
  * OOS evaluation: run the engine over the FULL bar series at threshold T, then
    keep only trades whose ENTRY time falls in the test window (entry-based
    attribution, identical convention to bt.compute_metrics subperiods). Running
    on the full series preserves indicator warmup continuity into the test
    window. (Caveat: at most one position may straddle the split boundary; its
    entry is in train so it is discarded — a negligible, threshold-consistent
    edge effect that affects every variant identically.)

OUTPUT  reports/regime/atr_filter_oos_validation_<UTC-date>/
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

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

import backtest_mr_5m_compare as bt  # noqa: E402  baseline engine — reused, never edited
from history_time_utils import parse_history_range  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # 2026-07 重构批次6：迁入 _archive/legacy_scripts/，深度 1→2
SYMBOLS = ["BTC", "ETH", "SOL", "LINK", "DOGE"]
PERCENTILES = [20, 25, 30, 35, 40]
FULL_START, FULL_END = "2023-01-01", "2026-05-29"   # +1d end so 05-28 23:59 is inclusive
TRAIN_FRACTION = 2.0 / 3.0


def utc_now_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_5m(name: str, hr, db: Path) -> pd.DataFrame:
    """Load full 1m series and aggregate to 5m via the engine's own loaders."""
    b1 = bt.load_1m(bt.SYMBOLS[name][0], hr, db)
    return bt.r5(b1, 5, hr)


def wilder_atr_series(bars: pd.DataFrame) -> np.ndarray:
    return bt.wilder_atr(
        bars["high"].to_numpy(float),
        bars["low"].to_numpy(float),
        bars["close"].to_numpy(float),
    )


def run_engine(name: str, bars: pd.DataFrame, threshold: float) -> list[dict]:
    """Run the EXACT baseline engine loop at a chosen threshold.

    Monkeypatch bt.ATR_THRESHOLDS for `name`'s inst id, restore afterwards.
    """
    inst = bt.SYMBOLS[name][1]
    saved = bt.ATR_THRESHOLDS.get(inst)
    bt.ATR_THRESHOLDS[inst] = float(threshold)
    try:
        return bt.backtest_symbol(name, bars)
    finally:
        if saved is None:
            bt.ATR_THRESHOLDS.pop(inst, None)
        else:
            bt.ATR_THRESHOLDS[inst] = saved


def keep_test_entries(trades: list[dict], split_dt: pd.Timestamp) -> list[dict]:
    """Filter to trades whose ENTRY time is in the test window (>= split)."""
    out = []
    for t in trades:
        et = pd.Timestamp(t["entry_time"])
        if et.tzinfo is None:
            et = et.tz_localize("UTC")
        if et >= split_dt:
            out.append(t)
    return out


def trade_sharpe(trades: list[dict]) -> float:
    """Per-trade Sharpe = mean(net)/std(net)*sqrt(n). NOT annualized; a unitless
    consistency measure for ranking, labeled as such in the report."""
    if len(trades) < 2:
        return float("nan")
    net = np.array([t["net_pnl_usd"] for t in trades], dtype=float)
    sd = net.std(ddof=1)
    if sd == 0:
        return float("nan")
    return float(net.mean() / sd * np.sqrt(len(net)))


def slim_metrics(trades: list[dict]) -> dict:
    """Reuse the engine's compute_metrics, plus per-trade Sharpe; flatten the
    fields we report."""
    m = bt.compute_metrics(trades)
    out = {
        "n": m.get("n", 0),
        "net_pnl": m.get("net_pnl", 0.0),
        "pf": m.get("pf", float("nan")),
        "win_rate_profit": m.get("win_rate_profit", float("nan")),
        "max_dd_usd": m.get("max_dd_usd", float("nan")),
        "max_dd_pct": m.get("max_dd_pct", float("nan")),
        "avg_trade": m.get("avg_trade", float("nan")),
        "sharpe_per_trade": trade_sharpe(trades),
    }
    return out


def fmt(x, nd=2, money=False):
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return "inf" if (isinstance(x, float) and np.isinf(x)) else "n/a"
    if money:
        return f"${x:,.0f}"
    return f"{x:.{nd}f}"


def main() -> int:
    out_date = datetime.now(timezone.utc).strftime("%Y%m%d")
    out_dir = PROJECT_ROOT / "reports" / "regime" / f"atr_filter_oos_validation_{out_date}"
    by_sym_dir = out_dir / "by_symbol"
    by_sym_dir.mkdir(parents=True, exist_ok=True)

    log: list[str] = []

    def logp(s: str = ""):
        print(s, flush=True)
        log.append(s)

    started = utc_now_str()
    logp(f"# ATR p30 OOS validation run — started {started} (UTC)")
    logp(f"Engine: backtest_mr_5m_compare.py (reused unmodified)")
    logp(f"Full range: {FULL_START} -> {FULL_END} | train fraction = {TRAIN_FRACTION:.4f}")
    logp("")

    db = PROJECT_ROOT / ".vntrader" / "database.db"
    hr = parse_history_range(FULL_START, FULL_END, timedelta(minutes=1), bt.TIMEZONE)

    # ── Load all symbols, determine global split date ─────────────────────────
    bars_by_sym: dict[str, pd.DataFrame] = {}
    global_min = None
    global_max = None
    for name in SYMBOLS:
        logp(f"[{name}] loading 1m -> 5m ...")
        b5 = load_5m(name, hr, db)
        bars_by_sym[name] = b5
        dt0 = pd.Timestamp(b5["datetime"].iloc[0])
        dt1 = pd.Timestamp(b5["datetime"].iloc[-1])
        global_min = dt0 if global_min is None else min(global_min, dt0)
        global_max = dt1 if global_max is None else max(global_max, dt1)
        logp(f"[{name}] {len(b5):,} 5m bars  {dt0} -> {dt1}")

    span = global_max - global_min
    split_dt = global_min + span * TRAIN_FRACTION
    # normalize tz to UTC for comparisons
    if split_dt.tzinfo is None:
        split_dt = split_dt.tz_localize("UTC")
    logp("")
    logp(f"Global 5m span: {global_min} -> {global_max}")
    logp(f"SPLIT DATE (train|test boundary): {split_dt}")
    logp(f"  train = [{global_min}, split)   test = [split, {global_max}]")
    logp("")

    # Per-symbol train/test bar counts
    split_info = {"split_dt": str(split_dt), "global_min": str(global_min),
                  "global_max": str(global_max), "train_fraction": TRAIN_FRACTION,
                  "per_symbol": {}}
    logp(f"{'sym':>5} {'train_bars':>11} {'test_bars':>10} {'train_end':>20} {'test_start':>20}")
    for name in SYMBOLS:
        b5 = bars_by_sym[name]
        dts = pd.to_datetime(b5["datetime"])
        if dts.dt.tz is None:
            dts = dts.dt.tz_localize("UTC")
        train_mask = dts < split_dt
        n_tr = int(train_mask.sum())
        n_te = int((~train_mask).sum())
        tr_end = dts[train_mask].iloc[-1] if n_tr else None
        te_start = dts[~train_mask].iloc[0] if n_te else None
        split_info["per_symbol"][name] = {
            "train_bars": n_tr, "test_bars": n_te,
            "train_end": str(tr_end), "test_start": str(te_start),
        }
        logp(f"{name:>5} {n_tr:>11,} {n_te:>10,} {str(tr_end):>20} {str(te_start):>20}")
    logp("")

    # ── Per-symbol: train percentiles, Q1, Q2 ────────────────────────────────
    live_thr = {n: bt.ATR_THRESHOLDS[bt.SYMBOLS[n][1]] for n in SYMBOLS}
    results: dict[str, dict] = {}
    # collect test trades for combined portfolio
    combined_filter_on: list[dict] = []
    combined_filter_off: list[dict] = []
    combined_by_pct: dict[int, list[dict]] = {p: [] for p in PERCENTILES}

    for name in SYMBOLS:
        b5 = bars_by_sym[name]
        dts = pd.to_datetime(b5["datetime"])
        if dts.dt.tz is None:
            dts = dts.dt.tz_localize("UTC")
        train_mask = (dts < split_dt).to_numpy()

        # train Wilder ATR percentiles
        atr_full = wilder_atr_series(b5)
        atr_train = atr_full[train_mask]
        atr_train = atr_train[~np.isnan(atr_train)]
        pct_thr = {p: float(np.percentile(atr_train, p)) for p in PERCENTILES}
        train_p30 = pct_thr[30]
        live = live_thr[name]
        diff_pct = (train_p30 - live) / live * 100 if live else float("nan")

        logp(f"=== {name} ===")
        logp(f"  train Wilder-ATR percentiles: " +
             ", ".join(f"p{p}={fmt(v,4)}" for p, v in pct_thr.items()))
        logp(f"  live threshold = {live}  |  train_p30 = {fmt(train_p30,4)}  "
             f"|  diff = {fmt(diff_pct,1)}%")

        # Q1: filter ON (train_p30) vs OFF (0) on TEST entries
        tr_on = keep_test_entries(run_engine(name, b5, train_p30), split_dt)
        tr_off = keep_test_entries(run_engine(name, b5, 0.0), split_dt)
        m_on = slim_metrics(tr_on)
        m_off = slim_metrics(tr_off)
        combined_filter_on += tr_on
        combined_filter_off += tr_off

        logp(f"  [Q1 test set]  filter OFF: n={m_off['n']:,} net={fmt(m_off['net_pnl'],money=True)} "
             f"pf={fmt(m_off['pf'])} win%={fmt(m_off['win_rate_profit'],1)} dd={fmt(m_off['max_dd_usd'],money=True)} "
             f"sharpe/t={fmt(m_off['sharpe_per_trade'],3)}")
        logp(f"  [Q1 test set]  filter ON : n={m_on['n']:,} net={fmt(m_on['net_pnl'],money=True)} "
             f"pf={fmt(m_on['pf'])} win%={fmt(m_on['win_rate_profit'],1)} dd={fmt(m_on['max_dd_usd'],money=True)} "
             f"sharpe/t={fmt(m_on['sharpe_per_trade'],3)}")

        # Q2: sensitivity p20..p40 on TEST entries
        sens = {}
        for p in PERCENTILES:
            trs = keep_test_entries(run_engine(name, b5, pct_thr[p]), split_dt)
            sens[p] = slim_metrics(trs)
            combined_by_pct[p] += trs
            logp(f"  [Q2 p{p}={fmt(pct_thr[p],4)}]  n={sens[p]['n']:,} "
                 f"net={fmt(sens[p]['net_pnl'],money=True)} pf={fmt(sens[p]['pf'])} "
                 f"win%={fmt(sens[p]['win_rate_profit'],1)} dd={fmt(sens[p]['max_dd_usd'],money=True)}")
        logp("")

        results[name] = {
            "live_threshold": live,
            "train_percentiles": pct_thr,
            "train_p30": train_p30,
            "p30_vs_live_diff_pct": diff_pct,
            "q1_filter_off": m_off,
            "q1_filter_on": m_on,
            "q2_sensitivity": {str(p): sens[p] for p in PERCENTILES},
        }
        (by_sym_dir / f"{name}_results.json").write_text(
            json.dumps(results[name], indent=2, default=float))

    # ── Combined portfolio (5-symbol) ────────────────────────────────────────
    port = {
        "q1_filter_off": slim_metrics(combined_filter_off),
        "q1_filter_on": slim_metrics(combined_filter_on),
        "q2_sensitivity": {str(p): slim_metrics(combined_by_pct[p]) for p in PERCENTILES},
    }
    logp("=== PORTFOLIO (BTC+ETH+SOL+LINK+DOGE, test set) ===")
    logp(f"  filter OFF: n={port['q1_filter_off']['n']:,} net={fmt(port['q1_filter_off']['net_pnl'],money=True)} "
         f"pf={fmt(port['q1_filter_off']['pf'])} dd={fmt(port['q1_filter_off']['max_dd_usd'],money=True)}")
    logp(f"  filter ON : n={port['q1_filter_on']['n']:,} net={fmt(port['q1_filter_on']['net_pnl'],money=True)} "
         f"pf={fmt(port['q1_filter_on']['pf'])} dd={fmt(port['q1_filter_on']['max_dd_usd'],money=True)}")
    for p in PERCENTILES:
        s = port["q2_sensitivity"][str(p)]
        logp(f"  p{p}: n={s['n']:,} net={fmt(s['net_pnl'],money=True)} pf={fmt(s['pf'])} dd={fmt(s['max_dd_usd'],money=True)}")
    logp("")

    # ── Write artifacts ──────────────────────────────────────────────────────
    finished = utc_now_str()
    write_outputs(out_dir, split_info, results, port, started, finished, log)
    logp(f"-> wrote report to {out_dir}")
    (out_dir / "run_log.txt").write_text("\n".join(log) + "\n")
    return 0


def write_outputs(out_dir, split_info, results, port, started, finished, log):
    # sensitivity_curve.json
    sens_curve = {
        "percentiles": PERCENTILES,
        "portfolio": {p: port["q2_sensitivity"][str(p)] for p in PERCENTILES},
        "by_symbol": {n: {p: results[n]["q2_sensitivity"][str(p)] for p in PERCENTILES}
                      for n in SYMBOLS},
        "thresholds": {n: results[n]["train_percentiles"] for n in SYMBOLS},
    }
    (out_dir / "sensitivity_curve.json").write_text(json.dumps(sens_curve, indent=2, default=float))

    # sensitivity_curve.md
    L = ["# ATR filter — test-set sensitivity (net PnL by train-percentile threshold)\n"]
    L.append("All values are TEST-SET net PnL (USD). Threshold for percentile pXX = "
             "pXX of Wilder ATR over the TRAIN window.\n")
    header = "| symbol | " + " | ".join(f"p{p}" for p in PERCENTILES) + " | shape |"
    sep = "|--------|" + "|".join("-----:" for _ in PERCENTILES) + "|-------|"
    L.append(header)
    L.append(sep)
    for n in SYMBOLS + ["PORTFOLIO"]:
        if n == "PORTFOLIO":
            row = [port["q2_sensitivity"][str(p)]["net_pnl"] for p in PERCENTILES]
        else:
            row = [results[n]["q2_sensitivity"][str(p)]["net_pnl"] for p in PERCENTILES]
        shape = describe_shape(row)
        L.append("| " + n + " | " + " | ".join(fmt(v, money=True) for v in row) + f" | {shape} |")
    (out_dir / "sensitivity_curve.md").write_text("\n".join(L) + "\n")

    # README.md — built separately for readability
    build_readme(out_dir, split_info, results, port, started, finished)


def describe_shape(vals: list[float]) -> str:
    """Crude monotonicity/spike descriptor for the percentile curve."""
    v = np.array(vals, dtype=float)
    if np.all(np.diff(v) >= 0):
        return "monotone up"
    if np.all(np.diff(v) <= 0):
        return "monotone down"
    rng = v.max() - v.min()
    mean_abs = np.mean(np.abs(v)) + 1e-9
    # is p30 an isolated peak/trough?
    i30 = PERCENTILES.index(30)
    peak30 = v[i30] == v.max() and v[i30] - np.median(v) > 0.5 * rng
    if rng / mean_abs < 0.15:
        return "flat (robust)"
    if peak30:
        return "p30 spike (overfit?)"
    return "non-monotone"


def build_readme(out_dir, split_info, results, port, started, finished):
    L = []
    A = L.append
    A("# ATR regime-filter p30 — out-of-sample validation + sensitivity\n")
    A(f"- Run (UTC): start {started} | end {finished}")
    A("- Pure local backtest. Live runner / strategy / baseline engine NOT modified.")
    A("- Engine reused: `scripts/backtest_mr_5m_compare.py` (faithful live replica).")
    A(f"- Split (strict time order): train = earliest {TRAIN_FRACTION:.3f}, test = latest 1/3.")
    A(f"- Split boundary: **{split_info['split_dt']}**")
    A(f"  - train = [{split_info['global_min']}, split), test = [split, {split_info['global_max']}]")
    A("- Threshold for percentile pXX = pXX of **Wilder ATR over the TRAIN window** "
      "(Wilder = the quantity the engine actually thresholds).")
    A("- OOS eval = run engine over full series at threshold T, keep trades whose "
      "**entry** is in the test window (entry-based attribution).\n")

    A("## Split — bars per symbol\n")
    A("| symbol | train bars | test bars | train end | test start |")
    A("|--------|----------:|---------:|-----------|-----------|")
    for n in SYMBOLS:
        s = split_info["per_symbol"][n]
        A(f"| {n} | {s['train_bars']:,} | {s['test_bars']:,} | {s['train_end']} | {s['test_start']} |")
    A("\n> Note: this 2/3–1/3 calendar split differs from `reports/ablation/` "
      "(which uses a fixed 2026-Q1 OOS window on BTC for direction/hour filters). "
      "Per task spec we use 2/3–1/3 here; the two are unrelated experiments.\n")

    A("## train_p30 vs live threshold\n")
    A("Live thresholds were derived from *SMA*-ATR p30 on the *full* sample "
      "(`backtest_mr_5m_v2.py`); here train_p30 is *Wilder*-ATR p30 on *train only*. "
      "Differences are expected (different estimator + different window).\n")
    A("| symbol | live thr | train_p30 (Wilder) | diff % |")
    A("|--------|--------:|------------------:|------:|")
    for n in SYMBOLS:
        r = results[n]
        A(f"| {n} | {r['live_threshold']} | {fmt(r['train_p30'],6)} | {fmt(r['p30_vs_live_diff_pct'],1)}% |")
    A("")

    A("## Q1 — out-of-sample value of the filter (test set)\n")
    A("filter ON = threshold train_p30; filter OFF = threshold 0 (all signals pass). "
      "Other params identical (LB=24, ATR=14 Wilder, stop=1.0, max_hold=48, $500/trade).\n")
    A("| symbol | mode | trades | net PnL | PF | win% | max DD $ | Sharpe/trade |")
    A("|--------|------|------:|-------:|---:|----:|--------:|------------:|")
    for n in SYMBOLS + ["PORTFOLIO"]:
        src = port if n == "PORTFOLIO" else results[n]
        off, on = src["q1_filter_off"], src["q1_filter_on"]
        A(f"| {n} | OFF | {off['n']:,} | {fmt(off['net_pnl'],money=True)} | {fmt(off['pf'])} | "
          f"{fmt(off['win_rate_profit'],1)} | {fmt(off['max_dd_usd'],money=True)} | {fmt(off['sharpe_per_trade'],3)} |")
        A(f"| {n} | ON  | {on['n']:,} | {fmt(on['net_pnl'],money=True)} | {fmt(on['pf'])} | "
          f"{fmt(on['win_rate_profit'],1)} | {fmt(on['max_dd_usd'],money=True)} | {fmt(on['sharpe_per_trade'],3)} |")
    A("")

    A("## Q2 — threshold sensitivity (test-set net PnL)\n")
    A("| symbol | p20 | p25 | p30 | p35 | p40 | shape |")
    A("|--------|----:|----:|----:|----:|----:|-------|")
    for n in SYMBOLS + ["PORTFOLIO"]:
        if n == "PORTFOLIO":
            row = [port["q2_sensitivity"][str(p)]["net_pnl"] for p in PERCENTILES]
        else:
            row = [results[n]["q2_sensitivity"][str(p)]["net_pnl"] for p in PERCENTILES]
        A(f"| {n} | " + " | ".join(fmt(v, money=True) for v in row) + f" | {describe_shape(row)} |")
    A("\n> Conclusions (Q1/Q2 per-symbol judgements) are written by the analyst "
      "after reading these tables — see the task report.\n")

    (out_dir / "README.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
