#!/usr/bin/env python3
"""atr_ratio factor decomposition — vol MOMENTUM vs vol ABSOLUTE LEVEL (local).

Question (from strategy diagnosis): the C2-1 dynamic-sizing system sizes by
`atr_ratio` (current 5m Wilder ATR / trailing-24-bar mean ATR), which measures
the short-term *change* in volatility (momentum), NOT the absolute *level* of
volatility. Does the MR-5m edge actually track vol momentum, vol level, both,
or neither? And do the live C2-1 cutpoints (1.050 / 1.302) sit where atr_ratio
has any real marginal difference?

Method (no in-sample peeking):
  * Reuse the validated engine's indicators / loader / contract specs from
    scripts/backtest_mr_5m_compare.py — that file is NOT modified.
  * Run a BASELINE pass per symbol: no ATR regime filter, fixed $500 notional
    (calc_size already targets $500), identical entry/exit logic to the live
    replica. This strips out DYN sizing so PnL/trade is the clean $500 baseline.
  * Split each symbol's 5m bars by bar-count: train = earliest 2/3, test =
    latest 1/3 (reproduces reports/regime/atr_filter_oos_validation_20260609/:
    238,848 train + 119,424 test per symbol). Analysis runs on TEST trades only
    (entry-bar attribution). Train is used ONLY to learn the per-symbol ATR
    empirical distribution for the absolute-level quantile (no leakage).
  * Feature A = atr_ratio at entry (vol momentum, raw).
    Feature B = ATR-absolute quantile = where the entry-bar ATR sits in the
    TRAIN ATR distribution for that symbol (0..1).
  * Single-dim buckets (3a/3b), 5x5 2-D pivot (4), C2-1 cutpoint check (5).

Does not touch run_mr_5m_direct.py, backtest_mr_5m_compare.py, or any params.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # 2026-07 重构批次6：迁入 _archive/legacy_scripts/，深度 1→2
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

# ── Reuse the validated engine (no modification) ──────────────────────────────
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

from backtest_mr_5m_compare import (  # noqa: E402
    wilder_atr, calc_size, SYMBOLS, CONTRACT_SPECS,
    LOOKBACK, ATR_WINDOW, ATR_STOP, MAX_HOLD,
    FEE_MAKER, FEE_TAKER, NOTIONAL_PER_TRADE,
)
from research_mr_5m import load_1m, r5  # noqa: E402
from history_time_utils import parse_history_range  # noqa: E402

# ── Config ────────────────────────────────────────────────────────────────────
START = "2023-01-01"
END = "2026-05-29"          # end-exclusive -> bars through 2026-05-28 (matches OOS run)
TIMEZONE = "UTC"
TRAIN_FRAC = 2.0 / 3.0      # earliest 2/3 = train, latest 1/3 = test

# Live C2-1 raw atr_ratio cutpoints (from dynamic_sizing C3: P50 / P80 of IS dist).
DYN_LO, DYN_HI = 1.050, 1.302

# Step-3a atr_ratio fixed buckets (task spec).
RATIO_EDGES = [0.0, 0.95, 1.05, 1.20, 1.45, np.inf]
RATIO_LABELS = ["[0,0.95)", "[0.95,1.05)", "[1.05,1.20)", "[1.20,1.45)", "[1.45,inf)"]

# Step-3b ATR-absolute quantile buckets (train-distribution quantiles).
Q_EDGES = [0.0, 0.20, 0.40, 0.60, 0.80, 1.0]
Q_LABELS = ["p0-20", "p20-40", "p40-60", "p60-80", "p80-100"]

OUT_DIR = PROJECT_ROOT / "reports" / "regime" / "atr_ratio_decomposition_20260609"


# ── Baseline trade engine (mirror of backtest_symbol, NO filter, tags features) ─
def run_symbol_baseline(name, b5):
    """MR-5m loop, NO ATR regime filter, fixed $500. Tag each entry with
    atr_ratio and the entry-bar absolute ATR. Returns (trades, split_idx, n, atr).
    Logic is line-for-line identical to backtest_mr_5m_compare.backtest_symbol
    except: (1) threshold check removed, (2) per-trade feature tagging.
    """
    inst_id = SYMBOLS[name][1]
    ct_val = CONTRACT_SPECS[inst_id]["ctVal"]
    tick = CONTRACT_SPECS[inst_id]["tickSz"]

    dt = b5["datetime"].to_numpy()
    o = b5["open"].to_numpy(dtype=float)  # noqa: F841 (kept for parity/readability)
    h = b5["high"].to_numpy(dtype=float)
    l = b5["low"].to_numpy(dtype=float)
    c = b5["close"].to_numpy(dtype=float)
    n = len(c)
    if n < LOOKBACK + 5:
        return [], 0, n, np.array([])

    atr = wilder_atr(h, l, c)
    dh = b5["high"].rolling(LOOKBACK).max().shift(1).to_numpy()
    dl = b5["low"].rolling(LOOKBACK).min().shift(1).to_numpy()
    # trailing-24 mean ATR excluding current bar (matches analyze_mr_signal_quality)
    atr_ma = pd.Series(atr).rolling(24).mean().shift(1).to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        atr_ratio = atr / atr_ma

    split_idx = int(n * TRAIN_FRAC)

    trades = []
    pos = 0
    eb = -1
    ep = 0.0
    esize = 0
    et = None
    e_ratio = np.nan
    e_atr = np.nan

    for i in range(LOOKBACK + 1, n):
        atr_i = atr[i]
        if np.isnan(atr_i) or atr_i <= 0:
            continue

        if pos != 0:
            hb = i - eb
            reason = ""
            d_h, d_l = dh[i], dl[i]
            if d_h > 0 and d_l > 0:
                mid = (d_h + d_l) / 2.0
                if (pos == 1 and c[i] >= mid) or (pos == -1 and c[i] <= mid):
                    reason = "midline"
            if not reason:
                stop_dist = ATR_STOP * atr_i
                if pos == 1 and l[i] <= ep - stop_dist:
                    reason = "stop"
                elif pos == -1 and h[i] >= ep + stop_dist:
                    reason = "stop"
            if not reason and hb >= MAX_HOLD:
                reason = "max_hold"

            if reason:
                exit_px = c[i] - tick if pos == 1 else c[i] + tick
                gross = (exit_px - ep) * esize * ct_val if pos == 1 \
                    else (ep - exit_px) * esize * ct_val
                entry_notional = ep * esize * ct_val
                exit_notional = exit_px * esize * ct_val
                maker_rebate = -FEE_MAKER * entry_notional
                taker_fee = FEE_TAKER * exit_notional
                fee_usd = maker_rebate - taker_fee
                net = gross + fee_usd
                trades.append({
                    "symbol": name,
                    "side": "long" if pos == 1 else "short",
                    "entry_idx": eb,
                    "entry_time": pd.Timestamp(et).isoformat(),
                    "exit_time": pd.Timestamp(dt[i]).isoformat(),
                    "exit_reason": reason,
                    "net_pnl_usd": float(net),
                    "atr_ratio": float(e_ratio),
                    "atr_abs": float(e_atr),
                })
                pos = 0
                continue

        if pos == 0:
            # NO ATR regime filter here (baseline = all signals)
            d_h, d_l = dh[i], dl[i]
            if np.isnan(d_h) or np.isnan(d_l) or d_h <= 0 or d_l <= 0:
                continue
            close = c[i]
            if close > d_h:
                pos = -1
            elif close < d_l:
                pos = 1
            else:
                continue
            ep = close
            esize = calc_size(inst_id, close)
            eb = i
            et = dt[i]
            e_ratio = atr_ratio[i]
            e_atr = atr_i

    return trades, split_idx, n, atr


# ── Stats helpers ─────────────────────────────────────────────────────────────
def group_stats(pnls):
    arr = np.asarray(pnls, dtype=float)
    n = len(arr)
    if n == 0:
        return {"n": 0, "win": None, "mean": None, "median": None,
                "pf": None, "total": 0.0}
    wins = arr[arr > 0].sum()
    losses = -arr[arr < 0].sum()
    if losses > 0:
        pf = wins / losses
    elif wins > 0:
        pf = float("inf")
    else:
        pf = 0.0
    return {
        "n": int(n),
        "win": float((arr > 0).mean()),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "pf": (None if pf == float("inf") else float(pf)),
        "pf_inf": bool(pf == float("inf")),
        "total": float(arr.sum()),
    }


def pf_str(g):
    if g["n"] == 0:
        return "—"
    if g.get("pf_inf"):
        return "inf"
    return f"{g['pf']:.2f}"


def train_quantile_mapper(train_atr):
    """Return f(x) -> empirical quantile (0..1) of x in train_atr (right-side)."""
    s = np.sort(train_atr[np.isfinite(train_atr) & (train_atr > 0)])
    m = len(s)
    def q(x):
        if m == 0 or not np.isfinite(x):
            return np.nan
        return float(np.searchsorted(s, x, side="right") / m)
    return q, m


def bucket_curve(df, col, edges, labels):
    """Single-dim grouped stats over the given bucket edges."""
    out = []
    cats = pd.cut(df[col], bins=edges, labels=labels, right=False, include_lowest=True)
    for lab in labels:
        sub = df[cats == lab]
        g = group_stats(sub["net_pnl_usd"].to_numpy())
        g["bucket"] = lab
        out.append(g)
    return out


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "by_symbol").mkdir(exist_ok=True)
    log = []

    def L(msg=""):
        print(msg, flush=True)
        log.append(msg)

    run_start = datetime.now(timezone.utc)
    L(f"# atr_ratio decomposition — run start {run_start.isoformat()}")
    L(f"Range: {START} -> {END} ({TIMEZONE}); train_frac={TRAIN_FRAC:.4f}")
    L(f"Reused engine: scripts/backtest_mr_5m_compare.py (UNMODIFIED)")
    L(f"Params: LB={LOOKBACK} ATR_W={ATR_WINDOW} stop={ATR_STOP} max_hold={MAX_HOLD} "
      f"notional=${NOTIONAL_PER_TRADE} (baseline, NO filter)")
    L(f"DYN cutpoints checked: atr_ratio {DYN_LO} / {DYN_HI}")
    L("")

    hr = parse_history_range(START, END, timedelta(minutes=1), TIMEZONE)
    db = PROJECT_ROOT / ".vntrader" / "database.db"

    all_test = []          # pooled test trades across symbols (with atr_q tagged)
    per_symbol_test = {}
    split_info = {}

    for name in SYMBOLS:
        vt = SYMBOLS[name][0]
        L(f"[{name}] loading 1m bars ...")
        b1 = load_1m(vt, hr, db)
        b5 = r5(b1, 5, hr)
        if b5.empty:
            L(f"[{name}] no data — skipped")
            continue
        trades, split_idx, n, atr = run_symbol_baseline(name, b5)
        # train ATR distribution = ATR over train bars
        train_atr = atr[:split_idx]
        qmap, m_train = train_quantile_mapper(train_atr)

        df = pd.DataFrame(trades)
        df["is_test"] = df["entry_idx"] >= split_idx
        df["atr_q"] = df["atr_abs"].map(qmap)
        test = df[df["is_test"]].copy()
        # drop rows with undefined features (early bars where trailing means nan)
        before = len(test)
        test = test[np.isfinite(test["atr_ratio"]) & np.isfinite(test["atr_q"])].copy()
        dropped = before - len(test)

        split_ts = pd.Timestamp(b5["datetime"].iloc[split_idx]).isoformat()
        split_info[name] = {
            "n_5m_bars": int(n), "split_idx": int(split_idx),
            "train_bars": int(split_idx), "test_bars": int(n - split_idx),
            "split_ts": split_ts, "train_atr_n": int(m_train),
            "total_trades": int(len(df)), "test_trades": int(len(test)),
            "test_trades_dropped_nan": int(dropped),
            "bars_first": pd.Timestamp(b5["datetime"].iloc[0]).isoformat(),
            "bars_last": pd.Timestamp(b5["datetime"].iloc[-1]).isoformat(),
        }
        L(f"[{name}] {n:,} 5m bars | split@idx {split_idx:,} ({split_ts}) | "
          f"test trades {len(test):,} (dropped {dropped} nan-feature)")
        per_symbol_test[name] = test
        all_test.append(test)

    pooled = pd.concat(all_test, ignore_index=True) if all_test else pd.DataFrame()
    L("")

    # ── Step 3: single-dim curves ─────────────────────────────────────────────
    single = {"by_symbol": {}, "portfolio": {}}

    def emit_single(df, key, container):
        ratio_curve = bucket_curve(df, "atr_ratio", RATIO_EDGES, RATIO_LABELS)
        # ATR-abs quantile buckets: cut on atr_q in [0,1]
        q_curve = []
        cats = pd.cut(df["atr_q"], bins=Q_EDGES, labels=Q_LABELS,
                      right=True, include_lowest=True)
        for lab in Q_LABELS:
            sub = df[cats == lab]
            g = group_stats(sub["net_pnl_usd"].to_numpy())
            g["bucket"] = lab
            q_curve.append(g)
        container[key] = {"atr_ratio": ratio_curve, "atr_abs_q": q_curve}

    for name, df in per_symbol_test.items():
        emit_single(df, name, single["by_symbol"])
    if not pooled.empty:
        emit_single(pooled, "PORTFOLIO", single["portfolio"])

    # ── Step 4: 5x5 2-D pivot (atr_ratio x atr_abs_q) ─────────────────────────
    def pivot_2d(df):
        rcat = pd.cut(df["atr_ratio"], bins=RATIO_EDGES, labels=RATIO_LABELS,
                      right=False, include_lowest=True)
        qcat = pd.cut(df["atr_q"], bins=Q_EDGES, labels=Q_LABELS,
                      right=True, include_lowest=True)
        cells = {}
        for rl in RATIO_LABELS:
            for ql in Q_LABELS:
                sub = df[(rcat == rl) & (qcat == ql)]
                cells[f"{rl}|{ql}"] = group_stats(sub["net_pnl_usd"].to_numpy())
        return cells

    pivots = {"by_symbol": {}, "portfolio": {}}
    for name, df in per_symbol_test.items():
        pivots["by_symbol"][name] = pivot_2d(df)
    if not pooled.empty:
        pivots["portfolio"]["PORTFOLIO"] = pivot_2d(pooled)

    # ── Step 5: C2-1 cutpoint check ───────────────────────────────────────────
    def dyn_segments(df):
        lo = df[df["atr_ratio"] < DYN_LO]["net_pnl_usd"].to_numpy()
        mid = df[(df["atr_ratio"] >= DYN_LO) & (df["atr_ratio"] < DYN_HI)]["net_pnl_usd"].to_numpy()
        hi = df[df["atr_ratio"] >= DYN_HI]["net_pnl_usd"].to_numpy()
        ntot = len(df)
        out = {}
        for seg, arr in [("small <1.050", lo), ("medium [1.050,1.302)", mid), ("large >=1.302", hi)]:
            g = group_stats(arr)
            g["share"] = (len(arr) / ntot) if ntot else 0.0
            out[seg] = g
        # where do the live cutpoints fall in the TEST atr_ratio distribution?
        r = df["atr_ratio"].to_numpy()
        out["_test_pctile_of_1.050"] = float((r < DYN_LO).mean())
        out["_test_pctile_of_1.302"] = float((r < DYN_HI).mean())
        return out

    dyn = {"by_symbol": {}, "portfolio": {}}
    for name, df in per_symbol_test.items():
        dyn["by_symbol"][name] = dyn_segments(df)
    if not pooled.empty:
        dyn["portfolio"]["PORTFOLIO"] = dyn_segments(pooled)

    # ── Write per-symbol pivot JSONs ──────────────────────────────────────────
    for name in per_symbol_test:
        payload = {
            "symbol": name,
            "split": split_info[name],
            "single_dim": single["by_symbol"][name],
            "pivot_2d": pivots["by_symbol"][name],
            "dyn_cutpoint_check": dyn["by_symbol"][name],
        }
        (OUT_DIR / "by_symbol" / f"{name}_pivot.json").write_text(
            json.dumps(payload, indent=2))
    if not pooled.empty:
        payload = {
            "symbol": "PORTFOLIO",
            "single_dim": single["portfolio"]["PORTFOLIO"],
            "pivot_2d": pivots["portfolio"]["PORTFOLIO"],
            "dyn_cutpoint_check": dyn["portfolio"]["PORTFOLIO"],
        }
        (OUT_DIR / "by_symbol" / "PORTFOLIO_pivot.json").write_text(
            json.dumps(payload, indent=2))

    # ── single_dim_curves.{json,md} ───────────────────────────────────────────
    (OUT_DIR / "single_dim_curves.json").write_text(json.dumps({
        "split_info": split_info, "single": single,
    }, indent=2))

    md = []
    def M(s=""):
        md.append(s)

    M("# MR-5m: atr_ratio (vol momentum) vs ATR absolute level — single-dim curves")
    M("")
    M(f"- Test set only (latest 1/3 by bar-count). Baseline: no filter, $500 fixed.")
    M(f"- atr_ratio = current Wilder ATR / trailing-24-bar mean ATR (raw).")
    M(f"- atr_abs_q = entry-bar ATR's quantile in the per-symbol TRAIN ATR distribution.")
    M("")

    def write_curve_block(title, curve):
        M(f"### {title}")
        M("")
        M("| bucket | n | win% | mean$ | median$ | PF | total$ |")
        M("|--------|--:|----:|-----:|-------:|---:|------:|")
        for g in curve:
            if g["n"] == 0:
                M(f"| {g['bucket']} | 0 | — | — | — | — | — |")
            else:
                M(f"| {g['bucket']} | {g['n']:,} | {g['win']*100:.1f} | "
                  f"{g['mean']:+.2f} | {g['median']:+.2f} | {pf_str(g)} | {g['total']:+,.0f} |")
        M("")

    M("## PORTFOLIO")
    M("")
    if not pooled.empty:
        write_curve_block("(3a) by atr_ratio (vol MOMENTUM)", single["portfolio"]["PORTFOLIO"]["atr_ratio"])
        write_curve_block("(3b) by ATR absolute quantile (vol LEVEL)", single["portfolio"]["PORTFOLIO"]["atr_abs_q"])
    for name in per_symbol_test:
        M(f"## {name}")
        M("")
        write_curve_block("(3a) by atr_ratio (vol MOMENTUM)", single["by_symbol"][name]["atr_ratio"])
        write_curve_block("(3b) by ATR absolute quantile (vol LEVEL)", single["by_symbol"][name]["atr_abs_q"])

    (OUT_DIR / "single_dim_curves.md").write_text("\n".join(md))

    # ── run_log + machine-readable dyn check ──────────────────────────────────
    (OUT_DIR / "dyn_cutpoint_check.json").write_text(json.dumps(dyn, indent=2))

    run_end = datetime.now(timezone.utc)
    L("")
    L(f"# run end {run_end.isoformat()} (elapsed {(run_end-run_start).total_seconds():.1f}s)")
    (OUT_DIR / "run_log.txt").write_text("\n".join(log))

    # ── console digest for the operator ───────────────────────────────────────
    print("\n\n================ DIGEST ================")
    _print_digest(single, pivots, dyn, per_symbol_test, pooled)
    return single, pivots, dyn, split_info


def _fmt_pivot(cells, metric):
    """Return markdown table string for a pivot (metric in {'mean','pf'})."""
    lines = []
    head = "| atr_ratio \\ atr_q | " + " | ".join(Q_LABELS) + " |"
    lines.append(head)
    lines.append("|" + "---|" * (len(Q_LABELS) + 1))
    for rl in RATIO_LABELS:
        row = [rl]
        for ql in Q_LABELS:
            g = cells[f"{rl}|{ql}"]
            if g["n"] == 0:
                row.append("—")
            elif metric == "mean":
                row.append(f"{g['mean']:+.1f} (n={g['n']})")
            else:
                row.append(f"{pf_str(g)} (n={g['n']})")
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _print_digest(single, pivots, dyn, per_symbol_test, pooled):
    if pooled.empty:
        print("no test trades")
        return
    print("\n--- PORTFOLIO single-dim: atr_ratio (momentum) ---")
    for g in single["portfolio"]["PORTFOLIO"]["atr_ratio"]:
        if g["n"]:
            print(f"  {g['bucket']:>14}  n={g['n']:>6}  win={g['win']*100:4.1f}%  "
                  f"mean={g['mean']:+6.2f}  PF={pf_str(g):>5}")
    print("\n--- PORTFOLIO single-dim: ATR absolute quantile (level) ---")
    for g in single["portfolio"]["PORTFOLIO"]["atr_abs_q"]:
        if g["n"]:
            print(f"  {g['bucket']:>10}  n={g['n']:>6}  win={g['win']*100:4.1f}%  "
                  f"mean={g['mean']:+6.2f}  PF={pf_str(g):>5}")
    print("\n--- PORTFOLIO 2-D pivot: mean PnL$/trade ---")
    print(_fmt_pivot(pivots["portfolio"]["PORTFOLIO"], "mean"))
    print("\n--- PORTFOLIO 2-D pivot: PF ---")
    print(_fmt_pivot(pivots["portfolio"]["PORTFOLIO"], "pf"))
    print("\n--- C2-1 cutpoint check (test set) ---")
    d = dyn["portfolio"]["PORTFOLIO"]
    print(f"  live 1.050 sits at test pctile {d['_test_pctile_of_1.050']*100:.1f}%")
    print(f"  live 1.302 sits at test pctile {d['_test_pctile_of_1.302']*100:.1f}%")
    for seg in ["small <1.050", "medium [1.050,1.302)", "large >=1.302"]:
        g = d[seg]
        print(f"  {seg:>22}  n={g['n']:>6} ({g['share']*100:4.1f}%)  "
              f"mean={g['mean']:+6.2f}  PF={pf_str(g):>5}")


if __name__ == "__main__":
    main()
