#!/usr/bin/env python3
"""DYN-v2B ATR-extreme size cap — V1 HARD truncation (Path A, stage 1).

Direct follow-up to reports/regime/portfolio_risk_phase_2a_20260609/ Q4(c), which
named "per-trade size cap at extreme ATR" as the only untested lever able to lower
v2B's maxDD (v2B maxDD is driven by LARGE single-volatility trades, NOT coordinated
events — 5/5 cohort events net POSITIVE +$3,009).

V1 = HARD truncation only. For each entry that v2B already sized (small/med/large),
look up the entry-bar Wilder ATR's percentile in that symbol's TRAIN ATR distribution;
if atr_q >= T, downsize per a cap rule. This is forward-looking (caps by structural
ATR level at entry), NOT a reactive-to-PnL circuit breaker (that family was rejected).

Scope is strictly limited to V1 (no continuous decay = V2, no dual-axis = V3).
Baseline = v2B (no cap); references = FLAT $500 and v2B-no-cap. NO parameter is
re-tuned: v2B cutpoints, ATR thresholds, DYN bands all unchanged.

Reuses research_dyn_v2.py (which reuses backtest_mr_5m_compare.py). Modifies nothing.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from research_dyn_v2 import (  # noqa: E402
    build_trades, trade_net, make_rules, learn_thresholds,
    SYMBOLS, SMALL, MED, LARGE, FLAT_NOTIONAL,
    START, END, TZ,
)
from research_mr_5m import load_1m, r5  # noqa: E402
from history_time_utils import parse_history_range  # noqa: E402

OUT_DIR = PROJECT_ROOT / "reports" / "regime" / "dyn_v2b_size_cap_v1_20260609"

T_KEYS = ["p90", "p93", "p95", "p97", "p98"]        # cap thresholds (high ATR quantile)
T_PCTS = {"p90": 90, "p93": 93, "p95": 95, "p97": 97, "p98": 98}
CAP_RULES = ["C1", "C2", "C3"]


# ── Cap rules ─────────────────────────────────────────────────────────────────
def drop_one(n):
    if abs(n - LARGE) < 1e-6:
        return MED
    if abs(n - MED) < 1e-6:
        return SMALL
    return SMALL                       # small stays small


def drop_two(n):
    if abs(n - LARGE) < 1e-6:
        return SMALL
    if abs(n - MED) < 1e-6:
        return SMALL
    return 0.0                         # small -> skip


def cap_apply(n, rule):
    if rule == "C1":
        return drop_one(n)
    if rule == "C2":
        return drop_two(n)
    if rule == "C3":
        return n / 2.0                 # halve by amount, not band
    raise ValueError(rule)


def learn_high_thresholds(atr, mask):
    a = atr[mask]
    a = a[np.isfinite(a) & (a > 0)]
    return {k: float(np.percentile(a, T_PCTS[k])) for k in T_KEYS}


# ── Per-trade record builder ──────────────────────────────────────────────────
def v2b_notional(tr, th_by_sym):
    return make_rules(th_by_sym[tr["symbol"]])["v2B"](tr)


def build_records(trades, th_by_sym, hi_by_sym, T_key, cap_rule):
    """Return per-trade records under v2B + (T_key, cap_rule) size cap.
    Each record: symbol, side, entry_dt, exit_dt, atr_abs, base_net, base_notion,
    cap_net, cap_notion, triggered."""
    recs = []
    for tr in trades:
        base_not = v2b_notional(tr, th_by_sym)
        br = trade_net(tr, base_not)            # v2B-no-cap
        triggered = tr["atr_abs"] >= hi_by_sym[tr["symbol"]][T_key]
        cap_not = cap_apply(base_not, cap_rule) if triggered else base_not
        cr = trade_net(tr, cap_not)
        recs.append({
            "symbol": tr["symbol"], "side": tr["side"],
            "entry_dt": tr["entry_dt"], "exit_dt": tr["exit_dt"],
            "atr_abs": tr["atr_abs"],
            "base_net": (br[0] if br else 0.0),
            "base_notion": (br[1] if br else 0.0),
            "cap_net": (cr[0] if cr else None),     # None => skipped
            "cap_notion": (cr[1] if cr else 0.0),
            "triggered": bool(triggered),
        })
    return recs


# ── Metrics from records (uses cap_net) ───────────────────────────────────────
def metrics_from(recs, key="cap_net", notion_key="cap_notion"):
    rows = [(r["exit_dt"], r[key], r[notion_key]) for r in recs if r[key] is not None]
    rows.sort(key=lambda x: x[0])
    nets = np.array([x[1] for x in rows], dtype=float)
    notion = np.array([x[2] for x in rows], dtype=float)
    if len(nets) == 0:
        return None
    cum = np.cumsum(nets)
    peak = np.maximum.accumulate(cum)
    dd = float(np.max(peak - cum))
    wins = nets[nets > 0].sum()
    losses = -nets[nets < 0].sum()
    pf = (wins / losses) if losses > 0 else float("inf")
    return {
        "n": int(len(nets)),
        "net": float(nets.sum()),
        "pf": (None if pf == float("inf") else round(float(pf), 4)),
        "win": round(float((nets > 0).mean()), 4),
        "max_dd": round(dd, 2),
        "dd_over_net": round(dd / nets.sum(), 5) if nets.sum() > 0 else None,
        "avg_notional": round(float(notion.mean()), 1),
        "max_notional": round(float(notion.max()), 1),
        "max_single_loss": round(float(nets.min()), 2),
    }


# ── DD window (peak->trough) of a net series, by exit_dt ───────────────────────
def dd_window(recs, key="base_net"):
    rows = [(r["exit_dt"], r[key], r) for r in recs if r[key] is not None]
    rows.sort(key=lambda x: x[0])
    cum = 0.0
    peak = -1e18
    peak_idx = -1
    best_dd = 0.0
    trough_idx = -1
    peak_at_trough = -1
    for i, (_, net, _) in enumerate(rows):
        cum += net
        if cum > peak:
            peak = cum
            peak_idx = i
        dd = peak - cum
        if dd > best_dd:
            best_dd = dd
            trough_idx = i
            peak_at_trough = peak_idx
    # DD-maker trades = those between the peak (exclusive) and trough (inclusive)
    window = [rows[i][2] for i in range(peak_at_trough + 1, trough_idx + 1)]
    return {
        "max_dd": round(best_dd, 2),
        "peak_idx": peak_at_trough, "trough_idx": trough_idx,
        "n_window_trades": len(window),
        "window_peak_dt": str(rows[peak_at_trough][0]) if peak_at_trough >= 0 else None,
        "window_trough_dt": str(rows[trough_idx][0]) if trough_idx >= 0 else None,
    }, window


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "configs").mkdir(exist_ok=True)
    log = []

    def L(m=""):
        print(m, flush=True)
        log.append(m)

    t0 = datetime.now(timezone.utc)
    L(f"# DYN-v2B size cap V1 — start {t0.isoformat()}")
    L(f"Range {START}->{END} {TZ}; baseline=v2B; refs=FLAT ${FLAT_NOTIONAL:.0f}, v2B-no-cap")
    L("Reused research_dyn_v2 / backtest_mr_5m_compare (UNMODIFIED). V1 hard-truncation only.")
    L("")

    hr = parse_history_range(START, END, timedelta(minutes=1), TZ)
    db = PROJECT_ROOT / ".vntrader" / "database.db"

    all_trades = []
    th_by_sym, hi_by_sym, split_by_sym = {}, {}, {}
    for name in SYMBOLS:
        b1 = load_1m(SYMBOLS[name][0], hr, db)
        b5 = r5(b1, 5, hr)
        if b5.empty:
            L(f"[{name}] no data"); continue
        trades, split_idx, n, atr, dt = build_trades(name, b5)
        mask = np.arange(n) < split_idx
        th_by_sym[name] = learn_thresholds(atr, mask)          # p30/40/60/80/85 (v2B)
        hi_by_sym[name] = learn_high_thresholds(atr, mask)     # p90..p98 (cap)
        split_by_sym[name] = split_idx
        all_trades.extend(trades)
        hi = hi_by_sym[name]
        L(f"[{name}] {n:,} bars split@{split_idx:,} | {len(trades):,} trades | "
          f"train ATR p90/p95/p98 = {hi['p90']:.4g}/{hi['p95']:.4g}/{hi['p98']:.4g}")
    L("")

    test = [t for t in all_trades if t["entry_idx"] >= split_by_sym[t["symbol"]]]
    L(f"TEST trades (latest 1/3): {len(test):,}")

    # ── references: FLAT and v2B-no-cap ───────────────────────────────────────
    flat_rows = [(t["exit_dt"], trade_net(t, FLAT_NOTIONAL)[0], trade_net(t, FLAT_NOTIONAL)[1])
                 for t in test]
    flat_recs = [{"exit_dt": e, "cap_net": n, "cap_notion": no, "base_net": n} for e, n, no in flat_rows]
    flat_m = metrics_from(flat_recs)
    # v2B-no-cap == any T with cap rule that never fires? simpler: build directly
    v2b_recs = build_records(test, th_by_sym, hi_by_sym, "p98", "C1")
    v2b_m = metrics_from(v2b_recs, key="base_net", notion_key="base_notion")

    flat_net, flat_dd = flat_m["net"], flat_m["max_dd"]
    gate1 = 1.5 * flat_dd        # maxDD <= 1.5x FLAT
    gate2 = 1.5 * flat_net       # net  >= 1.5x FLAT
    L(f"\nFLAT       : net ${flat_m['net']:,.0f}  maxDD ${flat_m['max_dd']:,.0f}")
    L(f"v2B-no-cap : net ${v2b_m['net']:,.0f}  maxDD ${v2b_m['max_dd']:,.0f} "
      f"({v2b_m['max_dd']/flat_dd:.3f}x)  avgNotion ${v2b_m['avg_notional']:,.0f}  "
      f"maxSingleLoss ${v2b_m['max_single_loss']:,.0f}")
    L(f"GATE1 maxDD <= 1.5xFLAT = ${gate1:,.0f}   GATE2 net >= 1.5xFLAT = ${gate2:,.0f}")
    L("")

    # v2B-no-cap DD window (the culprit window) — computed ONCE
    ddw, ddw_trades = dd_window(v2b_recs, key="base_net")
    L(f"v2B-no-cap maxDD window: {ddw['window_peak_dt']} -> {ddw['window_trough_dt']} "
      f"({ddw['n_window_trades']} trades, dd ${ddw['max_dd']:,.0f})")

    # ── full 15-config scan ───────────────────────────────────────────────────
    L("\n## STEP 2/3 — 15-config scan (v2B + size cap)")
    L(f"{'cfg':12} {'n':>6} {'net$':>9} {'dPnL%v2B':>8} {'dPnL%FLT':>8} {'PF':>5} "
      f"{'maxDD$':>7} {'xFLAT':>5} {'maxLoss$':>8} {'avgNot':>7} {'capN':>5} {'G1':>3} {'G2':>3}")
    scan = {}
    passers = []
    for T_key in T_KEYS:
        for rule in CAP_RULES:
            recs = build_records(test, th_by_sym, hi_by_sym, T_key, rule)
            m = metrics_from(recs)
            cap_n = sum(1 for r in recs if r["triggered"])
            dpnl_v2b = (m["net"] - v2b_m["net"]) / abs(v2b_m["net"]) * 100
            dpnl_flat = (m["net"] - flat_net) / abs(flat_net) * 100
            g1 = m["max_dd"] <= gate1
            g2 = m["net"] >= gate2
            cfg = f"T{T_key[1:]}_{rule}"
            # cap-triggered trade characteristics (base_net of triggered trades)
            trig = [r for r in recs if r["triggered"]]
            trig_base = np.array([r["base_net"] for r in trig]) if trig else np.array([0.0])
            # culprit hit-rate: of v2B-no-cap DD-window trades, how many are cap-eligible at T
            cul_total = len(ddw_trades)
            cul_hit = sum(1 for r in ddw_trades if r["atr_abs"] >= hi_by_sym[r["symbol"]][T_key])
            scan[cfg] = {
                "T": T_key, "rule": rule, **m,
                "dPnL_pct_vs_v2b": round(dpnl_v2b, 2),
                "dPnL_pct_vs_flat": round(dpnl_flat, 2),
                "dd_x_flat": round(m["max_dd"] / flat_dd, 4),
                "cap_count": cap_n,
                "cap_trigger_rate": round(cap_n / len(test), 4),
                "triggered_base_pnl_sum": round(float(trig_base.sum()), 2),
                "triggered_base_pnl_mean": round(float(trig_base.mean()), 3),
                "triggered_base_pnl_std": round(float(trig_base.std(ddof=1)) if len(trig_base) > 1 else 0.0, 3),
                "triggered_net_positive": bool(trig_base.sum() > 0),
                "dd_culprit_total": cul_total,
                "dd_culprit_hit": cul_hit,
                "dd_culprit_hit_rate": round(cul_hit / cul_total, 4) if cul_total else None,
                "gate1_dd": g1, "gate2_pnl": g2, "pass_both": (g1 and g2),
            }
            if g1 and g2:
                passers.append(cfg)
            L(f"{cfg:12} {m['n']:>6} {m['net']:>9,.0f} {dpnl_v2b:>+8.2f} {dpnl_flat:>+8.2f} "
              f"{(m['pf'] or 0):>5.2f} {m['max_dd']:>7,.0f} {m['max_dd']/flat_dd:>5.3f} "
              f"{m['max_single_loss']:>8,.0f} {m['avg_notional']:>7,.0f} {cap_n:>5} "
              f"{'Y' if g1 else '.':>3} {'Y' if g2 else '.':>3}")
            (OUT_DIR / "configs").mkdir(exist_ok=True)
            cdir = OUT_DIR / "configs" / f"V1_T{T_key[1:]}_{rule}"
            cdir.mkdir(exist_ok=True)
            (cdir / "result.json").write_text(json.dumps(scan[cfg], indent=2, default=str))

    L(f"\nGate1 ($maxDD<={gate1:,.0f}) AND Gate2 (net>=${gate2:,.0f}) passers: {passers or 'NONE'}")

    # DD culprit summary (per T; rule-independent)
    dd_culprit = {
        "v2b_no_cap_maxdd_window": ddw,
        "per_T_hit_rate": {},
        "note": ("culprit hit-rate = fraction of v2B-no-cap maxDD-window trades whose "
                 "entry ATR >= train pT (i.e. cap-eligible). Rule-independent (depends only on T)."),
    }
    for T_key in T_KEYS:
        hit = sum(1 for r in ddw_trades if r["atr_abs"] >= hi_by_sym[r["symbol"]][T_key])
        dd_culprit["per_T_hit_rate"][T_key] = {
            "total": len(ddw_trades), "hit": hit,
            "hit_rate": round(hit / len(ddw_trades), 4) if ddw_trades else None,
        }
    (OUT_DIR / "dd_culprit_analysis.json").write_text(json.dumps(dd_culprit, indent=2, default=str))

    L("\n## DD culprit hit-rate (v2B-no-cap maxDD window, per T)")
    for T_key in T_KEYS:
        d = dd_culprit["per_T_hit_rate"][T_key]
        L(f"  {T_key}: {d['hit']}/{d['total']} = {(d['hit_rate'] or 0)*100:.1f}% of DD-makers cap-eligible")

    # ── STEP 4 sanity (ONLY if a config passes both gates) ────────────────────
    sanity = {"ran": False, "reason": "no config passed both gates"}
    if passers:
        sanity = run_sanity(passers, scan, test, th_by_sym, hi_by_sym,
                            v2b_recs, v2b_m, flat_net, flat_dd, L)
    else:
        L("\n## STEP 4 — SKIPPED: no config passed both gates (per task spec).")
    (OUT_DIR / "sanity_checks.json").write_text(json.dumps(sanity, indent=2, default=str))

    # ── dump scan + log ───────────────────────────────────────────────────────
    summary = {
        "FLAT": flat_m, "v2B_no_cap": v2b_m,
        "gate1_maxdd_le": round(gate1, 2), "gate2_net_ge": round(gate2, 2),
        "scan": scan, "passers": passers,
    }
    (OUT_DIR / "scan_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    t1 = datetime.now(timezone.utc)
    log.append(f"\n# end {t1.isoformat()} ({(t1-t0).total_seconds():.1f}s)")
    (OUT_DIR / "run_log.txt").write_text("\n".join(log))
    L("\nDONE.")


def run_sanity(passers, scan, test, th_by_sym, hi_by_sym, v2b_recs, v2b_m, flat_net, flat_dd, L):
    """4 honesty checks on the BEST passing config (max net among passers)."""
    best = max(passers, key=lambda c: scan[c]["net"])
    T_key, rule = scan[best]["T"], scan[best]["rule"]
    L(f"\n## STEP 4 — sanity checks on best config {best}")
    out = {"ran": True, "best_config": best, "T": T_key, "rule": rule}

    # 4a parameter robustness: +/- 1 quantile (same rule)
    idx = T_KEYS.index(T_key)
    neighbors = {}
    for off, lbl in ((-1, "minus1"), (+1, "plus1")):
        j = idx + off
        if 0 <= j < len(T_KEYS):
            ncfg = f"T{T_KEYS[j][1:]}_{rule}"
            s = scan[ncfg]
            neighbors[lbl] = {
                "cfg": ncfg, "net": s["net"], "max_dd": s["max_dd"],
                "dnet_pct": round((s["net"] - scan[best]["net"]) / abs(scan[best]["net"]) * 100, 2),
                "ddd_pct": round((s["max_dd"] - scan[best]["max_dd"]) / abs(scan[best]["max_dd"]) * 100, 2),
            }
    worst_swing = max((abs(v["dnet_pct"]) for v in neighbors.values()), default=0)
    worst_dd_swing = max((abs(v["ddd_pct"]) for v in neighbors.values()), default=0)
    out["check4a_param_robust"] = {
        "neighbors": neighbors, "max_net_swing_pct": worst_swing,
        "max_dd_swing_pct": worst_dd_swing,
        "pass": worst_swing < 10 and worst_dd_swing < 10,
    }

    # 4b sub-period: split test by entry_dt median; gate in each half (vs FLAT-half)
    ents = sorted(t["entry_dt"] for t in test)
    midt = ents[len(ents) // 2]
    half = {}
    for hlbl, sel in (("first", lambda t: t["entry_dt"] < midt),
                      ("second", lambda t: t["entry_dt"] >= midt)):
        sub = [t for t in test if sel(t)]
        f_recs = [{"exit_dt": t["exit_dt"], "cap_net": trade_net(t, FLAT_NOTIONAL)[0],
                   "cap_notion": trade_net(t, FLAT_NOTIONAL)[1]} for t in sub]
        fm = metrics_from(f_recs)
        recs = build_records(sub, th_by_sym, hi_by_sym, T_key, rule)
        m = metrics_from(recs)
        g1 = m["max_dd"] <= 1.5 * fm["max_dd"]
        g2 = m["net"] >= 1.5 * fm["net"]
        half[hlbl] = {"flat_net": round(fm["net"], 0), "flat_dd": fm["max_dd"],
                      "cfg_net": round(m["net"], 0), "cfg_dd": m["max_dd"],
                      "gate1": g1, "gate2": g2, "pass_both": g1 and g2}
    out["check4b_subperiod"] = {**half, "pass": half["first"]["pass_both"] and half["second"]["pass_both"]}

    # 4c per-symbol: cfg net vs v2B-no-cap net per symbol
    persym = {}
    for sym in SYMBOLS:
        sub = [t for t in test if t["symbol"] == sym]
        if not sub:
            continue
        recs = build_records(sub, th_by_sym, hi_by_sym, T_key, rule)
        cfg_net = sum(r["cap_net"] for r in recs if r["cap_net"] is not None)
        v2b_net = sum(r["base_net"] for r in recs)
        persym[sym] = {"cfg_net": round(cfg_net, 0), "v2b_net": round(v2b_net, 0),
                       "cfg_beats_v2b": cfg_net >= v2b_net,
                       "delta": round(cfg_net - v2b_net, 0)}
    nwin = sum(1 for v in persym.values() if v["cfg_beats_v2b"])
    out["check4c_persymbol"] = {"detail": persym, "n_cfg_ge_v2b": nwin, "n": len(persym),
                                "note": "cap only lowers high-ATR size; cfg >= v2b not expected (info only)"}

    # 4d DD culprit hit-rate at best T
    out["check4d_dd_culprit_hit_rate"] = scan[best]["dd_culprit_hit_rate"]

    L(f"  4a param robust (+/-1 quantile): net swing {worst_swing:.1f}% dd swing {worst_dd_swing:.1f}% "
      f"-> {out['check4a_param_robust']['pass']}")
    L(f"  4b sub-period: first pass={half['first']['pass_both']} second pass={half['second']['pass_both']} "
      f"-> {out['check4b_subperiod']['pass']}")
    L(f"  4c per-symbol cfg>=v2b: {nwin}/{len(persym)}")
    L(f"  4d DD culprit hit-rate at {T_key}: {(scan[best]['dd_culprit_hit_rate'] or 0)*100:.1f}%")
    return out


if __name__ == "__main__":
    main()
