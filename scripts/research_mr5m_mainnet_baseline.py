#!/usr/bin/env python3
"""MR-5m FLAT baseline re-validation on mainnet data (read-only).

Reuses backtest_mr_5m_compare.py verbatim (imported, never modified):
the only outer-script injections are (a) data loaded from database_mainnet.db
via sqlite mode=ro, (b) ATR_THRESHOLDS dict mutated in place per config and
restored afterwards.

Configs:
  C1: FLAT $500, no ATR filter (thresholds = 0)
  C2: FLAT $500, legacy live thresholds (demo-derived, 口径 A)
  C3: FLAT $500, mainnet-rederived Wilder-p40 train thresholds (口径 B)
"""

from __future__ import annotations

import json
import sys
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import backtest_mr_5m_compare as bm
from research_demo_vs_mainnet import load_1m_ro, DB_MAIN
from research_mr_5m import r5
from history_time_utils import parse_history_range

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT = PROJECT_ROOT / "reports/regime/mr5m_mainnet_baseline_20260611"

START, END, TIMEZONE = "2023-01-01", "2026-05-28", "UTC"
CUT = pd.Timestamp("2025-04-09 08:00:00", tz="UTC")  # train/test split, same as all prior research
TRAIN_P = 40  # live thresholds ≈ Wilder-p40 (per prior research)

DEMO_FLAT_TEST = {  # historical demo baseline (dyn_v2 step2_summary.json, FLAT, test period)
    "n": 21298, "net": 189227.38, "pf": 2.0615, "win": 0.4828,
    "max_dd": 2249.87, "mean_pnl": 8.8847,
}


def metrics_for(trades: list) -> dict:
    return bm.compute_metrics(trades)


def split_trades(trades: list) -> dict[str, list]:
    tr, te = [], []
    for t in trades:
        (tr if pd.Timestamp(t["entry_time"]).tz_convert("UTC") < CUT else te).append(t)
    return {"full": trades, "train": tr, "test": te}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    for sub in ("by_symbol", "by_year", "config_C1_C2_C3"):
        (OUT / sub).mkdir(exist_ok=True)

    hr = parse_history_range(start_arg=START, end_arg=END,
                             interval_delta=timedelta(minutes=1), timezone_name=TIMEZONE)

    bars5 = {}
    atr5 = {}
    for name, (vt_symbol, inst_id) in bm.SYMBOLS.items():
        d1 = load_1m_ro(vt_symbol, hr, DB_MAIN)
        b5 = r5(d1, 5, hr)
        bars5[name] = b5
        atr5[name] = bm.wilder_atr(b5["high"].to_numpy(dtype=float),
                                   b5["low"].to_numpy(dtype=float),
                                   b5["close"].to_numpy(dtype=float))
        print(f"loaded {name}: 1m={len(d1)} 5m={len(b5)}", flush=True)

    # ── step 1: dual-threshold preparation ──
    legacy = dict(bm.ATR_THRESHOLDS)  # 口径 A (and restore copy)
    rederived = {}
    thresholds_report = {}
    for name, (_vt, inst_id) in bm.SYMBOLS.items():
        atr = atr5[name]
        dtv = bars5[name]["datetime"]
        train_mask = (dtv < CUT).to_numpy() & ~np.isnan(atr)
        valid = ~np.isnan(atr)
        thr_b = float(np.percentile(atr[train_mask], TRAIN_P))
        rederived[inst_id] = thr_b
        thr_a = legacy[inst_id]
        thresholds_report[name] = {
            "A_legacy_demo_derived": thr_a,
            "B_mainnet_p40_train": thr_b,
            "B_over_A": thr_b / thr_a,
            "pass_rate_A_on_mainnet": float(np.mean(atr[valid] >= thr_a)),
            "pass_rate_B_on_mainnet": float(np.mean(atr[valid] >= thr_b)),
        }
    (OUT / "thresholds.json").write_text(json.dumps(thresholds_report, indent=2))
    print(json.dumps(thresholds_report, indent=2), flush=True)

    # ── step 2: run C1/C2/C3 ──
    configs = {
        "C1_no_filter": {k: 0.0 for k in legacy},
        "C2_legacy_thresholds": legacy,
        "C3_mainnet_p40": rederived,
    }
    results = {}
    try:
        for cfg_name, thr in configs.items():
            bm.ATR_THRESHOLDS.clear()
            bm.ATR_THRESHOLDS.update(thr)
            per_symbol_trades = {n: bm.backtest_symbol(n, bars5[n]) for n in bm.SYMBOLS}
            all_trades = [t for ts in per_symbol_trades.values() for t in ts]
            res = {"thresholds": thr}
            for period, subset in split_trades(all_trades).items():
                res[period] = metrics_for(subset)
            res["by_symbol"] = {}
            for n, ts in per_symbol_trades.items():
                res["by_symbol"][n] = {p: metrics_for(s) for p, s in split_trades(ts).items()}
            results[cfg_name] = res
            (OUT / "config_C1_C2_C3" / f"{cfg_name}.json").write_text(
                json.dumps(res, indent=2, default=str))
            f, t = res["full"], res["test"]
            print(f"{cfg_name}: full n={f['n']} net=${f.get('net_pnl',0):,.0f} "
                  f"pf={f.get('pf',float('nan')):.3f} dd=${f.get('max_dd_usd',0):,.0f} | "
                  f"test n={t['n']} net=${t.get('net_pnl',0):,.0f} "
                  f"pf={t.get('pf',float('nan')):.3f} dd=${t.get('max_dd_usd',0):,.0f}", flush=True)
    finally:
        bm.ATR_THRESHOLDS.clear()
        bm.ATR_THRESHOLDS.update(legacy)

    # by_symbol / by_year extracts
    for n in bm.SYMBOLS:
        (OUT / "by_symbol" / f"{n}.json").write_text(json.dumps(
            {cfg: results[cfg]["by_symbol"][n] for cfg in results}, indent=2, default=str))
    (OUT / "by_year" / "yearly.json").write_text(json.dumps(
        {cfg: {p: results[cfg][p].get("subperiods", {}) for p in ("full", "train", "test")}
         for cfg in results}, indent=2, default=str))

    # ── step 3: attribution vs demo (C2 test period) ──
    c2t = results["C2_legacy_thresholds"]["test"]
    attr = {
        "demo_FLAT_test": DEMO_FLAT_TEST,
        "mainnet_C2_test": {
            "n": c2t["n"], "net": c2t.get("net_pnl"), "pf": c2t.get("pf"),
            "win": c2t.get("win_rate_profit"), "max_dd": c2t.get("max_dd_usd"),
            "mean_pnl": c2t.get("avg_trade"),
        },
        "delta": {
            "net_change": c2t.get("net_pnl", 0) - DEMO_FLAT_TEST["net"],
            "net_ratio": c2t.get("net_pnl", 0) / DEMO_FLAT_TEST["net"],
            "pf_change": c2t.get("pf", 0) - DEMO_FLAT_TEST["pf"],
            "n_change": c2t["n"] - DEMO_FLAT_TEST["n"],
            "mean_pnl_change": (c2t.get("avg_trade") or 0) - DEMO_FLAT_TEST["mean_pnl"],
        },
        "by_symbol_test_C2": {
            n: {"n": results["C2_legacy_thresholds"]["by_symbol"][n]["test"].get("n"),
                "net": results["C2_legacy_thresholds"]["by_symbol"][n]["test"].get("net_pnl"),
                "pf": results["C2_legacy_thresholds"]["by_symbol"][n]["test"].get("pf")}
            for n in bm.SYMBOLS
        },
    }
    (OUT / "attribution.json").write_text(json.dumps(attr, indent=2, default=str))
    print("attribution:", json.dumps(attr["delta"], indent=2), flush=True)
    print("done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
