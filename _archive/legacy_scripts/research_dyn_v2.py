#!/usr/bin/env python3
"""DYN-v2 design + walk-forward validation — size by ATR ABSOLUTE level.

Premise (from reports/regime/atr_ratio_decomposition_20260609/): DYN-v1 (C2-1)
sizes by atr_ratio (vol momentum), which OOS-decomposition showed is a PROXY for
ATR absolute level. The real regime factor is the ATR absolute quantile (5/5
symbols monotone). v1 even UP-sizes the v1 pain cell "low abs level + high
atr_ratio" (PF 0.24). This script designs 3 ATR-absolute-level candidates and
runs the SAME validation rigor as DYN-v1's C2/C3.

Baseline is always FLAT $500 (clean null hypothesis for "is switching the axis
actually better?"). No path dependence: the entry/exit decisions are identical
across FLAT and all candidates (sizing never changes entries) — so we run the
trade engine ONCE per symbol and re-size each trade per candidate, exactly as
DYN-v1 did.

Reuses scripts/backtest_mr_5m_compare.py indicators/loader/specs (NOT modified).
Does NOT touch run_mr_5m_direct.py / mr_5m_strategy.py / any param / any cutpoint.
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
    wilder_atr, SYMBOLS, CONTRACT_SPECS,
    LOOKBACK, ATR_WINDOW, ATR_STOP, MAX_HOLD, LEVERAGE,
    FEE_MAKER, FEE_TAKER,
)
from research_mr_5m import load_1m, r5  # noqa: E402
from history_time_utils import parse_history_range  # noqa: E402

START, END, TZ = "2023-01-01", "2026-05-29", "UTC"
TRAIN_FRAC = 2.0 / 3.0
CAPITAL = 5000.0            # README CAPITAL; used only for step-5 daily-loss circuit
SMALL, MED, LARGE = 250.0, 500.0, 750.0
FLAT_NOTIONAL = 500.0

# Walk-forward windows (identical to DYN-v1 C3-1).
WF_IS = ("2023-01-01", "2025-01-01")    # [IS_start, IS_end_exclusive)
WF_OOS = ("2025-01-01", "2026-05-29")   # [OOS_start, OOS_end_exclusive)

OUT_DIR = PROJECT_ROOT / "reports" / "regime" / "dyn_v2_design_20260609"


# ── Run the trade engine ONCE per symbol (no filter, full context) ────────────
def build_trades(name, b5):
    inst_id = SYMBOLS[name][1]
    ct_val = CONTRACT_SPECS[inst_id]["ctVal"]
    tick = CONTRACT_SPECS[inst_id]["tickSz"]

    dt = b5["datetime"].to_numpy()
    h = b5["high"].to_numpy(dtype=float)
    l = b5["low"].to_numpy(dtype=float)
    c = b5["close"].to_numpy(dtype=float)
    n = len(c)
    if n < LOOKBACK + 5:
        return [], 0, n, np.array([]), dt

    atr = wilder_atr(h, l, c)
    dh = b5["high"].rolling(LOOKBACK).max().shift(1).to_numpy()
    dl = b5["low"].rolling(LOOKBACK).min().shift(1).to_numpy()
    atr_ma = pd.Series(atr).rolling(24).mean().shift(1).to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        atr_ratio = atr / atr_ma

    split_idx = int(n * TRAIN_FRAC)
    trades = []
    pos = 0
    eb = -1
    ep = 0.0
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
                sd = ATR_STOP * atr_i
                if pos == 1 and l[i] <= ep - sd:
                    reason = "stop"
                elif pos == -1 and h[i] >= ep + sd:
                    reason = "stop"
            if not reason and hb >= MAX_HOLD:
                reason = "max_hold"
            if reason:
                exit_px = c[i] - tick if pos == 1 else c[i] + tick
                trades.append({
                    "symbol": name, "inst_id": inst_id, "ct_val": ct_val,
                    "side": pos, "entry_idx": eb,
                    "entry_dt": pd.Timestamp(et), "exit_dt": pd.Timestamp(dt[i]),
                    "entry_px": ep, "exit_px": exit_px,
                    "atr_abs": float(e_atr), "atr_ratio": float(e_ratio),
                    "exit_reason": reason,
                })
                pos = 0
                continue
        if pos == 0:
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
            eb = i
            et = dt[i]
            e_ratio = atr_ratio[i]
            e_atr = atr_i
    return trades, split_idx, n, atr, dt


# ── Sizing: exact engine net at an arbitrary target notional ──────────────────
def trade_net(tr, notional):
    """Return (net_pnl, actual_deployed_notional, size) for this trade at the
    given target notional. notional<=0 -> trade skipped (returns None)."""
    if notional <= 0:
        return None
    ep, xp, ctv = tr["entry_px"], tr["exit_px"], tr["ct_val"]
    cv = ep * ctv
    size = max(1, min(round(notional * LEVERAGE / cv), 1000)) if cv > 0 else 1
    if tr["side"] == 1:
        gross = (xp - ep) * size * ctv
    else:
        gross = (ep - xp) * size * ctv
    entry_notional = ep * size * ctv
    exit_notional = xp * size * ctv
    fee = (-FEE_MAKER * entry_notional) - (FEE_TAKER * exit_notional)
    return gross + fee, entry_notional, size


# ── Candidate notional rules (parameters fixed on TRAIN, no leakage) ──────────
def make_rules(th):
    """th = dict of per-band ATR thresholds learned on the train window."""
    def flat(tr):
        return FLAT_NOTIONAL

    def v2A(tr):                    # single-dim abs level: p40/p80
        a = tr["atr_abs"]
        if a < th["p40"]:
            return SMALL
        if a < th["p80"]:
            return MED
        return LARGE

    def v2B(tr):                    # 2-D: abs level x atr_ratio
        a = tr["atr_abs"]
        if a < th["p40"]:
            return SMALL            # low abs = negative region -> minimise
        if a >= th["p80"]:
            return LARGE            # high abs -> large regardless of momentum
        r = tr["atr_ratio"]
        return MED if (np.isfinite(r) and r >= 1.05) else SMALL

    def v2C(tr):                    # abs level + low-level skip (integrates filter)
        a = tr["atr_abs"]
        if a < th["p30"]:
            return 0.0              # skip negative region entirely
        if a < th["p60"]:
            return SMALL
        if a < th["p85"]:
            return MED
        return LARGE
    return {"FLAT": flat, "v2A": v2A, "v2B": v2B, "v2C": v2C}


def learn_thresholds(atr, mask):
    a = atr[mask]
    a = a[np.isfinite(a) & (a > 0)]
    return {f"p{p}": float(np.percentile(a, p)) for p in (30, 40, 60, 80, 85)}


# ── Metrics ───────────────────────────────────────────────────────────────────
def eval_config(trades, rule):
    """Apply a sizing rule to a list of trades. Returns metric dict + per-trade
    arrays needed downstream."""
    rows = []
    for tr in trades:
        r = trade_net(tr, rule(tr))
        if r is None:
            continue
        net, notion, size = r
        rows.append((tr["exit_dt"], net, notion, tr["entry_dt"]))
    if not rows:
        return None
    rows.sort(key=lambda x: x[0])
    nets = np.array([r[1] for r in rows])
    notion = np.array([r[2] for r in rows])
    cum = np.cumsum(nets)
    peak = np.maximum.accumulate(cum)
    dd = float(np.max(peak - cum)) if len(cum) else 0.0
    wins = nets[nets > 0].sum()
    losses = -nets[nets < 0].sum()
    pf = (wins / losses) if losses > 0 else float("inf")
    std = nets.std(ddof=1) if len(nets) > 1 else 0.0
    sharpe = float(nets.mean() / std * np.sqrt(len(nets))) if std > 0 else 0.0
    return {
        "n": int(len(nets)),
        "net": float(nets.sum()),
        "pf": (None if pf == float("inf") else round(float(pf), 4)),
        "win": round(float((nets > 0).mean()), 4),
        "max_dd": round(dd, 2),
        "dd_over_net": round(dd / nets.sum(), 4) if nets.sum() > 0 else None,
        "sharpe_per_trade": round(sharpe, 4),
        "total_notional": round(float(notion.sum()), 0),
        "avg_notional": round(float(notion.mean()), 1),
        "max_notional": round(float(notion.max()), 1),
        "mean_pnl": round(float(nets.mean()), 4),
    }


def pf_disp(m):
    return "inf" if (m and m["pf"] is None and m["n"]) else (f"{m['pf']:.2f}" if m else "—")


# ── Cohort overlap stress (4b) ────────────────────────────────────────────────
def cohort_stress(trades_test, rules, n_syms):
    """Find 5m timestamps where >=4 symbols hold the SAME-direction position
    simultaneously, attribute the trades active during such 'cohort' bars, and
    sum their PnL under FLAT vs each candidate."""
    # Build a 5m grid of (timestamp -> {symbol: dir}) over open intervals.
    events = []  # (ts, symbol, dir, +1 open / -1 close)
    for tr in trades_test:
        events.append((tr["entry_dt"], tr["symbol"], tr["side"], +1))
        events.append((tr["exit_dt"], tr["symbol"], tr["side"], -1))
    events.sort(key=lambda x: x[0])
    open_dir = {}
    cohort_active = {}   # symbol-set sampling: mark trades active during cohort bars
    # We need per-bar concurrency; approximate by scanning event boundaries.
    # Track max same-direction concurrency reached at any moment.
    max_long = max_short = 0
    cohort_times = []
    for ts, sym, d, kind in events:
        if kind == +1:
            open_dir[(sym)] = d
        else:
            open_dir.pop(sym, None)
        nl = sum(1 for v in open_dir.values() if v == 1)
        ns = sum(1 for v in open_dir.values() if v == -1)
        max_long = max(max_long, nl)
        max_short = max(max_short, ns)
        if nl >= 4 or ns >= 4:
            cohort_times.append((ts, 1 if nl >= 4 else -1))
    # A trade is a "cohort trade" if its holding interval covers any cohort time
    # of its own direction.
    ct_sorted = cohort_times
    def is_cohort(tr):
        for ts, d in ct_sorted:
            if d == tr["side"] and tr["entry_dt"] <= ts <= tr["exit_dt"]:
                return True
        return False
    import bisect
    long_times = sorted(ts for ts, d in cohort_times if d == 1)
    short_times = sorted(ts for ts, d in cohort_times if d == -1)
    def is_cohort(tr):  # noqa: F811 (replaces the linear-scan version above)
        times = long_times if tr["side"] == 1 else short_times
        i = bisect.bisect_left(times, tr["entry_dt"])
        return i < len(times) and times[i] <= tr["exit_dt"]
    cohort_trades = [tr for tr in trades_test if is_cohort(tr)]
    out = {
        "max_concurrent_same_dir_long": int(max_long),
        "max_concurrent_same_dir_short": int(max_short),
        "n_cohort_bars(>=4 same dir)": len(cohort_times),
        "n_cohort_trades": len(cohort_trades),
        "cohort_pnl": {},
    }
    for cname, rule in rules.items():
        s = 0.0
        for tr in cohort_trades:
            r = trade_net(tr, rule(tr))
            if r:
                s += r[0]
        out["cohort_pnl"][cname] = round(s, 2)
    return out


# ── Step-5 exploratory portfolio risk overlay (NOT a real design) ─────────────
def risk_overlay(trades_test, rule, max_same_dir=3, daily_loss_pct=0.02):
    """Sequential sim: block a new entry if >=max_same_dir same-dir positions are
    already open, or if the current calendar day's realised PnL < -daily_loss_pct
    of running equity (24h halt on new entries). Path-dependent -> sequential.
    Exploratory only."""
    import heapq
    trs = sorted(trades_test, key=lambda x: x["entry_dt"])
    open_heap = []      # (exit_dt, dir, net) of currently-open positions
    realised_sum = 0.0
    day_pnl = {}        # date -> realised PnL closed that day
    halt_until = None
    nets = []
    for tr in trs:
        now = tr["entry_dt"]
        while open_heap and open_heap[0][0] <= now:        # release closed positions
            xdt, d, net = heapq.heappop(open_heap)
            realised_sum += net
            day_pnl[xdt.date()] = day_pnl.get(xdt.date(), 0.0) + net
        eq = CAPITAL + realised_sum
        dp = day_pnl.get(now.date(), 0.0)
        if halt_until is not None and now < halt_until:
            continue
        if eq > 0 and dp < -daily_loss_pct * eq:
            halt_until = now + pd.Timedelta(hours=24)
            continue
        same = sum(1 for _, d, _ in open_heap if d == tr["side"])
        if same >= max_same_dir:
            continue
        r = trade_net(tr, rule(tr))
        if r is None:
            continue
        net = r[0]
        nets.append((tr["exit_dt"], net))
        heapq.heappush(open_heap, (tr["exit_dt"], tr["side"], net))
    if not nets:
        return None
    nets.sort(key=lambda x: x[0])
    arr = np.array([n for _, n in nets])
    cum = np.cumsum(arr)
    dd = float(np.max(np.maximum.accumulate(cum) - cum)) if len(cum) else 0.0
    return {"n": len(arr), "net": round(float(arr.sum()), 2), "max_dd": round(dd, 2)}


# ── Sanity checks (step 6) ────────────────────────────────────────────────────
def sanity_checks(trades_test, atr_by_sym, dt_by_sym, split_by_sym, candidate, th_by_sym):
    """Run the 5 honesty checks for one candidate name."""
    res = {}
    # rebuild per-symbol rules so thresholds are symbol-correct
    def net_for(tr, cname, thr):
        return trade_net(tr, make_rules(thr)[cname](tr))

    # check 1: parameter robustness (shift abs-level cutpoints +/-1 percentile).
    # Precompute the shifted per-symbol thresholds ONCE (not per trade).
    th_up, th_dn = {}, {}
    for sym in SYMBOLS:
        a = atr_by_sym[sym][:split_by_sym[sym]]
        a = a[np.isfinite(a) & (a > 0)]
        if len(a) == 0:
            continue
        th_up[sym] = {f"p{p}": float(np.percentile(a, min(99, p + 1))) for p in (30, 40, 60, 80, 85)}
        th_dn[sym] = {f"p{p}": float(np.percentile(a, max(1, p - 1))) for p in (30, 40, 60, 80, 85)}
    base_net = up_net = dn_net = 0.0
    for tr in trades_test:
        sym = tr["symbol"]
        r = net_for(tr, candidate, th_by_sym[sym])
        if r:
            base_net += r[0]
        ru = net_for(tr, candidate, th_up[sym])
        if ru:
            up_net += ru[0]
        rd = net_for(tr, candidate, th_dn[sym])
        if rd:
            dn_net += rd[0]
    def pct(a, b):
        return round((a - b) / abs(b) * 100, 2) if b else None
    res["check1_param_robust"] = {
        "base_net": round(base_net, 0), "plus1pct_net": round(up_net, 0),
        "minus1pct_net": round(dn_net, 0),
        "delta_up_%": pct(up_net, base_net), "delta_dn_%": pct(dn_net, base_net),
        "pass": (abs(pct(up_net, base_net) or 0) < 10 and abs(pct(dn_net, base_net) or 0) < 10),
    }
    # check 2: sub-period robustness (split test in two halves by entry_dt median)
    ents = sorted(tr["entry_dt"] for tr in trades_test)
    midt = ents[len(ents) // 2]
    for half, sel in (("first", lambda t: t["entry_dt"] < midt),
                      ("second", lambda t: t["entry_dt"] >= midt)):
        sub = [t for t in trades_test if sel(t)]
        cand = sum(net_for(t, candidate, th_by_sym[t["symbol"]])[0]
                   for t in sub if net_for(t, candidate, th_by_sym[t["symbol"]]))
        flat = sum(trade_net(t, FLAT_NOTIONAL)[0] for t in sub)
        res.setdefault("check2_subperiod", {})[half] = {
            "cand_net": round(cand, 0), "flat_net": round(flat, 0),
            "cand_beats_flat": cand > flat}
    res["check2_subperiod"]["pass"] = (
        res["check2_subperiod"]["first"]["cand_beats_flat"] and
        res["check2_subperiod"]["second"]["cand_beats_flat"])
    # check 3: per-symbol (cand > flat for each)
    persym = {}
    for sym in SYMBOLS:
        sub = [t for t in trades_test if t["symbol"] == sym]
        if not sub:
            continue
        cand = sum(net_for(t, candidate, th_by_sym[sym])[0]
                   for t in sub if net_for(t, candidate, th_by_sym[sym]))
        flat = sum(trade_net(t, FLAT_NOTIONAL)[0] for t in sub)
        persym[sym] = {"cand": round(cand, 0), "flat": round(flat, 0), "win": cand > flat}
    nwin = sum(1 for v in persym.values() if v["win"])
    res["check3_persymbol"] = {"detail": persym, "n_win": nwin, "n": len(persym),
                               "pass": nwin == len(persym)}
    # check 4: avg notional alignment (<= 1.1x FLAT)
    notion = [trade_net(t, make_rules(th_by_sym[t["symbol"]])[candidate](t))
              for t in trades_test]
    notion = [x[1] for x in notion if x is not None]
    flat_notion = [trade_net(t, FLAT_NOTIONAL)[1] for t in trades_test]
    avg = np.mean(notion)
    favg = np.mean(flat_notion)
    res["check4_notional_align"] = {
        "cand_avg_notional": round(float(avg), 1),
        "flat_avg_notional": round(float(favg), 1),
        "ratio": round(float(avg / favg), 3), "pass": (avg <= 1.1 * favg)}
    # check 5: did v2 avoid the v1 pain cell (low abs level + high atr_ratio)?
    # v1 pain cell = atr_abs < train p40 AND atr_ratio >= 1.302 -> v1 would size LARGE.
    pain = []
    for t in trades_test:
        thr = th_by_sym[t["symbol"]]
        if t["atr_abs"] < thr["p40"] and np.isfinite(t["atr_ratio"]) and t["atr_ratio"] >= 1.302:
            r = make_rules(thr)[candidate](t)
            pain.append(r)
    res["check5_pain_cell"] = {
        "n_pain_trades": len(pain),
        "v1_would_size": LARGE,
        "cand_sizes": sorted(set(pain)),
        "cand_max_size_here": (max(pain) if pain else None),
        "pass": (len(pain) == 0 or max(pain) <= SMALL)}
    return res


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "candidates").mkdir(exist_ok=True)
    (OUT_DIR / "walk_forward").mkdir(exist_ok=True)
    log = []
    def L(m=""):
        print(m, flush=True)
        log.append(m)

    t0 = datetime.now(timezone.utc)
    L(f"# DYN-v2 design + walk-forward — start {t0.isoformat()}")
    L(f"Range {START}->{END} {TZ}; baseline=FLAT ${FLAT_NOTIONAL}; bands {SMALL}/{MED}/{LARGE}")
    L(f"Reused engine backtest_mr_5m_compare.py (UNMODIFIED); no path dependence (re-size only)")
    L("")

    hr = parse_history_range(START, END, timedelta(minutes=1), TZ)
    db = PROJECT_ROOT / ".vntrader" / "database.db"

    all_trades = []
    atr_by_sym, dt_by_sym, split_by_sym = {}, {}, {}
    th_train_by_sym = {}   # thresholds learned on 2/3 train (step2)
    for name in SYMBOLS:
        b1 = load_1m(SYMBOLS[name][0], hr, db)
        b5 = r5(b1, 5, hr)
        if b5.empty:
            L(f"[{name}] no data"); continue
        trades, split_idx, n, atr, dt = build_trades(name, b5)
        atr_by_sym[name] = atr
        dt_by_sym[name] = np.arange(n)            # index proxy for train mask
        split_by_sym[name] = split_idx
        th_train_by_sym[name] = learn_thresholds(atr, np.arange(n) < split_idx)
        all_trades.extend(trades)
        L(f"[{name}] {n:,} bars split@{split_idx:,} | {len(trades):,} trades | "
          f"train p40/p80 ATR = {th_train_by_sym[name]['p40']:.4g}/{th_train_by_sym[name]['p80']:.4g}")
    L("")

    # split into train/test by bar index (matches decomposition / ATR-filter OOS)
    test = [t for t in all_trades if t["entry_idx"] >= split_by_sym[t["symbol"]]]
    L(f"TEST trades (latest 1/3): {len(test):,}")

    # ── STEP 2: single train/test eval (FLAT + v2A/B/C) ───────────────────────
    def eval_all(trade_set, th_by_sym):
        out = {}
        # need per-symbol rules; wrap a rule that dispatches on symbol thresholds
        for cname in ("FLAT", "v2A", "v2B", "v2C"):
            def rule(tr, _c=cname):
                return make_rules(th_by_sym[tr["symbol"]])[_c](tr)
            out[cname] = eval_config(trade_set, rule)
        return out

    step2 = eval_all(test, th_train_by_sym)
    L("\n## STEP 2 — single train/test (TEST set)")
    L(f"{'cfg':6} {'n':>6} {'net$':>10} {'PF':>5} {'win%':>5} {'maxDD$':>8} "
      f"{'Sharpe':>7} {'avgNotion':>9} {'maxNotion':>9} {'Δnet%vsFLAT':>11}")
    flat_net = step2["FLAT"]["net"]
    survivors = []
    for cn in ("FLAT", "v2A", "v2B", "v2C"):
        m = step2[cn]
        dnet = (m["net"] - flat_net) / abs(flat_net) * 100
        L(f"{cn:6} {m['n']:>6} {m['net']:>10,.0f} {pf_disp(m):>5} {m['win']*100:>5.1f} "
          f"{m['max_dd']:>8,.0f} {m['sharpe_per_trade']:>7.2f} {m['avg_notional']:>9,.0f} "
          f"{m['max_notional']:>9,.0f} {dnet:>+11.1f}")
        if cn != "FLAT":
            # Step-2 is a COARSE "worth analysing" screen (net gain + DD not
            # catastrophic). The BINDING ≤1.5× DD gate is enforced at Step 4c.
            dd_ratio = m["max_dd"] / step2["FLAT"]["max_dd"]
            if dnet >= 5.0 and dd_ratio <= 2.0:
                survivors.append(cn)
    flat_dd = step2["FLAT"]["max_dd"]
    L(f"\nStep-2 coarse survivors (Δnet≥5% & maxDD≤2.0×FLAT): {survivors or 'NONE'}")
    L(f"  (binding ≤1.5×FLAT DD gate [={1.5*flat_dd:,.0f}] enforced at Step 4c:")
    for cn in survivors:
        r = step2[cn]["max_dd"] / flat_dd
        L(f"    {cn}: maxDD ${step2[cn]['max_dd']:,.0f} = {r:.3f}×FLAT "
          f"{'PASS' if r <= 1.5 else 'FAIL'})")

    artifacts = {"step2": step2, "survivors_step2": survivors,
                 "thresholds_train": th_train_by_sym}

    if not survivors:
        L("\n>>> All candidates fail Step-2 gate. DYN-v2 single-test validation "
          "fails; research terminated at Step 2 per task spec.")
        _dump(artifacts, OUT_DIR, log, t0, test, {}, {}, {})
        _digest(step2, None, None, None, None)
        return

    # ── STEP 3: walk-forward (C3-style IS->OOS, neighbours, per-symbol) ───────
    wf = walk_forward(all_trades, atr_by_sym, survivors)
    artifacts["walk_forward"] = wf

    # ── STEP 4: drawdown explicit + cohort ────────────────────────────────────
    dd_analysis = {"per_candidate": {}}
    for cn in ["FLAT"] + survivors:
        m = step2[cn]
        dd_analysis["per_candidate"][cn] = {
            "max_dd": m["max_dd"], "dd_over_net": m["dd_over_net"],
            "net": m["net"], "avg_notional": m["avg_notional"]}
    # worst single day
    for cn in ["FLAT"] + survivors:
        def rule(tr, _c=cn):
            return make_rules(th_train_by_sym[tr["symbol"]])[_c](tr)
        day = {}
        for tr in test:
            r = trade_net(tr, rule(tr))
            if r:
                day[tr["exit_dt"].date()] = day.get(tr["exit_dt"].date(), 0) + r[0]
        worst = min(day.values()) if day else 0
        dd_analysis["per_candidate"][cn]["worst_day"] = round(worst, 2)
        # gate: maxDD <= 1.5x FLAT
        dd_analysis["per_candidate"][cn]["dd_gate_vs_flat_1.5x"] = (
            step2[cn]["max_dd"] <= 1.5 * step2["FLAT"]["max_dd"])
    rules_full = {cn: (lambda tr, _c=cn: make_rules(th_train_by_sym[tr["symbol"]])[_c](tr))
                  for cn in ["FLAT"] + survivors}
    cohort = cohort_stress(test, rules_full, len(SYMBOLS))
    artifacts["drawdown_analysis"] = dd_analysis
    artifacts["cohort_overlap"] = cohort

    # ── STEP 5: exploratory portfolio-risk overlay ────────────────────────────
    overlay = {"note": "EXPLORATORY ONLY — not a portfolio-risk design",
               "no_overlay": {}, "with_overlay": {}}
    for cn in ["FLAT"] + survivors:
        def rule(tr, _c=cn):
            return make_rules(th_train_by_sym[tr["symbol"]])[_c](tr)
        overlay["no_overlay"][cn] = {"net": round(sum(
            trade_net(t, rule(t))[0] for t in test if trade_net(t, rule(t))), 0),
            "max_dd": step2[cn]["max_dd"]}
        overlay["with_overlay"][cn] = risk_overlay(test, rule)
    artifacts["risk_overlay"] = overlay

    # ── STEP 6: sanity checks ─────────────────────────────────────────────────
    checks = {}
    for cn in survivors:
        checks[cn] = sanity_checks(test, atr_by_sym, dt_by_sym, split_by_sym,
                                   cn, th_train_by_sym)
    artifacts["sanity_checks"] = checks

    _dump(artifacts, OUT_DIR, log, t0, test, wf, dd_analysis, checks)
    _digest(step2, wf, dd_analysis, cohort, checks)


def walk_forward(all_trades, atr_by_sym, survivors):
    """C3-style: learn thresholds on IS (2023-2024), apply to OOS (2025-2026).
    Also neighbour robustness and per-symbol gate (on OOS)."""
    out = {"IS_window": WF_IS, "OOS_window": WF_OOS, "candidates": {}}
    is_lo = pd.Timestamp(WF_IS[0], tz="UTC"); is_hi = pd.Timestamp(WF_IS[1], tz="UTC")
    oos_lo = pd.Timestamp(WF_OOS[0], tz="UTC"); oos_hi = pd.Timestamp(WF_OOS[1], tz="UTC")
    # thresholds learned from IS bars per symbol
    th_is = {}
    for sym, atr in atr_by_sym.items():
        # IS mask via bar datetimes — rebuild from entry distribution is hard;
        # approximate IS by using the same fraction of bars before 2025-01-01.
        th_is[sym] = None
    # We learn IS thresholds from IS *bars*; reuse atr arrays + need datetimes.
    # Simpler & faithful: learn from IS *entries'* atr is biased; use IS trades.
    is_tr = [t for t in all_trades if is_lo <= t["entry_dt"] < is_hi]
    oos_tr = [t for t in all_trades if oos_lo <= t["entry_dt"] < oos_hi]
    th_is = {}
    for sym in SYMBOLS:
        a = np.array([t["atr_abs"] for t in is_tr if t["symbol"] == sym])
        a = a[np.isfinite(a) & (a > 0)]
        if len(a) < 50:
            continue
        th_is[sym] = {f"p{p}": float(np.percentile(a, p)) for p in (30, 40, 60, 80, 85)}

    def evalset(trset, thmap, cname):
        def rule(tr):
            return make_rules(thmap[tr["symbol"]])[cname](tr)
        return eval_config(trset, rule)

    for cn in survivors:
        is_m = evalset(is_tr, th_is, cn)
        oos_m = evalset(oos_tr, th_is, cn)
        is_f = evalset(is_tr, th_is, "FLAT")
        oos_f = evalset(oos_tr, th_is, "FLAT")
        # relative advantage (net) IS vs OOS
        is_adv = (is_m["net"] - is_f["net"]) / abs(is_f["net"]) * 100
        oos_adv = (oos_m["net"] - oos_f["net"]) / abs(oos_f["net"]) * 100
        # PF ratio gate
        pf_ratio = (oos_m["pf"] / is_m["pf"]) if (is_m["pf"] and oos_m["pf"]) else None
        # neighbour robustness: shift p40/p80 (or p30/60/85) by +/-5 pct on IS
        neigh = {}
        for shift in (-5, +5):
            thn = {}
            for sym in th_is:
                a = np.array([t["atr_abs"] for t in is_tr if t["symbol"] == sym])
                a = a[np.isfinite(a) & (a > 0)]
                thn[sym] = {f"p{p}": float(np.percentile(a, min(99, max(1, p + shift))))
                            for p in (30, 40, 60, 80, 85)}
            mm = evalset(oos_tr, thn, cn)
            neigh[f"shift{shift:+d}"] = {"oos_net": round(mm["net"], 0),
                                         "beats_flat": mm["net"] > oos_f["net"]}
        # per-symbol on OOS
        ps = {}
        for sym in SYMBOLS:
            sub = [t for t in oos_tr if t["symbol"] == sym]
            if not sub or sym not in th_is:
                continue
            c = evalset(sub, th_is, cn); f = evalset(sub, th_is, "FLAT")
            ps[sym] = {"cand": round(c["net"], 0), "flat": round(f["net"], 0),
                       "win": c["net"] > f["net"]}
        nwin = sum(1 for v in ps.values() if v["win"])
        out["candidates"][cn] = {
            "IS": is_m, "OOS": oos_m, "IS_flat": is_f, "OOS_flat": oos_f,
            "IS_adv_%": round(is_adv, 1), "OOS_adv_%": round(oos_adv, 1),
            "pf_ratio_oos_is": round(pf_ratio, 3) if pf_ratio else None,
            "neighbours": neigh,
            "persymbol_oos": ps, "n_sym_win": nwin, "n_sym": len(ps),
            "gate_oos_adv_positive": oos_adv > 0,
            "gate_persym_4of5": nwin >= 4,
        }
    return out


def _dump(artifacts, out, log, t0, test, wf, dd, checks):
    th = artifacts.get("thresholds_train", {})
    for cn in ("v2A", "v2B", "v2C"):
        payload = {"candidate": cn,
                   "step2": artifacts["step2"].get(cn),
                   "thresholds_train": th,
                   "walk_forward": (wf.get("candidates", {}).get(cn) if wf else None),
                   "sanity_checks": checks.get(cn) if checks else None}
        (out / "candidates" / f"{cn}.json").write_text(json.dumps(payload, indent=2, default=str))
    if wf:
        (out / "walk_forward" / "walk_forward.json").write_text(json.dumps(wf, indent=2, default=str))
    if "drawdown_analysis" in artifacts:
        (out / "drawdown_analysis.json").write_text(json.dumps(artifacts["drawdown_analysis"], indent=2, default=str))
    if "cohort_overlap" in artifacts:
        (out / "cohort_overlap_check.json").write_text(json.dumps(artifacts["cohort_overlap"], indent=2, default=str))
    if "risk_overlay" in artifacts:
        (out / "risk_overlay.json").write_text(json.dumps(artifacts["risk_overlay"], indent=2, default=str))
    if checks:
        (out / "sanity_checks.json").write_text(json.dumps(checks, indent=2, default=str))
    (out / "step2_summary.json").write_text(json.dumps(artifacts["step2"], indent=2, default=str))
    t1 = datetime.now(timezone.utc)
    log.append(f"\n# end {t1.isoformat()} ({(t1-t0).total_seconds():.1f}s)")
    (out / "run_log.txt").write_text("\n".join(log))


def _digest(step2, wf, dd, cohort, checks):
    print("\n================ DIGEST ================")
    if wf:
        print("\n-- STEP 3 walk-forward (IS->OOS) --")
        for cn, v in wf["candidates"].items():
            print(f"  {cn}: IS_adv {v['IS_adv_%']:+.1f}%  OOS_adv {v['OOS_adv_%']:+.1f}%  "
                  f"pf_ratio(oos/is) {v['pf_ratio_oos_is']}  persym {v['n_sym_win']}/{v['n_sym']}  "
                  f"neigh {[ (k, x['beats_flat']) for k,x in v['neighbours'].items() ]}")
    if dd:
        print("\n-- STEP 4 drawdown --")
        for cn, v in dd["per_candidate"].items():
            print(f"  {cn}: maxDD ${v['max_dd']:,.0f}  worstDay ${v['worst_day']:,.0f}  "
                  f"DD/net {v['dd_over_net']}  ddGate1.5x {v.get('dd_gate_vs_flat_1.5x')}")
    if cohort:
        print("\n-- STEP 4b cohort stress --")
        print(f"  max same-dir concurrency L/S = {cohort['max_concurrent_same_dir_long']}/"
              f"{cohort['max_concurrent_same_dir_short']}; cohort trades {cohort['n_cohort_trades']}")
        print(f"  cohort PnL: {cohort['cohort_pnl']}")
    if checks:
        print("\n-- STEP 6 sanity checks (pass?) --")
        for cn, v in checks.items():
            print(f"  {cn}: c1param={v['check1_param_robust']['pass']} "
                  f"c2subperiod={v['check2_subperiod']['pass']} "
                  f"c3persym={v['check3_persymbol']['n_win']}/{v['check3_persymbol']['n']}({v['check3_persymbol']['pass']}) "
                  f"c4notional={v['check4_notional_align']['ratio']}x({v['check4_notional_align']['pass']}) "
                  f"c5paincell={v['check5_pain_cell']['pass']}")


if __name__ == "__main__":
    main()
