#!/usr/bin/env python3
"""Portfolio-layer risk control — Phase 2.A single-rule exploration (local).

Explores 3 portfolio-level rules in isolation (best threshold + standalone effect)
to screen which deserve Phase 2.B combination testing:
  R1 daily-loss halt   : day cum PnL <= -X% * day-start equity -> block new entries N h
  R2 same-dir cap      : block a new entry if >=K same-direction positions already open
  R3 cross-sectional   : >=M same-dir entry signals in one 5m bar -> downgrade that bar
                         (D1 skip all / D2 all->small / D3 scale: m=3 -1 tier, 4 -2, >=5 skip)

Baselines: FLAT $500 and v2B (dyn_v2 2-D candidate). Each rule x threshold is run on
BOTH. Test set only (latest 1/3, split @ 2025-04-09 08:00 UTC) — same as
atr_ratio_decomposition / dyn_v2.

Reuses scripts/research_dyn_v2.py (which reuses backtest_mr_5m_compare.py). Convention
identical to dyn_v2 Step-5 overlay and the reactive-breaker study: operate on the FIXED
baseline trade list; blocking/downsizing removes/reduces that trade's PnL, subsequent
same-symbol entries are NOT re-derived (a documented 2nd-order simplification, kept for
fair cross-config comparison). Does NOT modify any engine / live code / params.
"""

from __future__ import annotations

import csv
import heapq
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # 2026-07 重构批次6：迁入 _archive/legacy_scripts/，深度 1→2
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import research_dyn_v2 as RV  # noqa: E402
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

from backtest_mr_5m_compare import SYMBOLS  # noqa: E402
from research_mr_5m import load_1m, r5  # noqa: E402
from history_time_utils import parse_history_range  # noqa: E402

CAPITAL = RV.CAPITAL
FLAT_N = RV.FLAT_NOTIONAL
TIER = [250.0, 500.0, 750.0]
OUT = PROJECT_ROOT / "reports" / "regime" / "portfolio_risk_phase_2a_20260609"


# ── base sizing ───────────────────────────────────────────────────────────────
def base_notional(tr, base, th_by_sym):
    if base == "FLAT":
        return FLAT_N
    return RV.make_rules(th_by_sym[tr["symbol"]])["v2B"](tr)


def net_at(tr, notional):
    if notional <= 0:
        return 0.0
    r = RV.trade_net(tr, notional)
    return r[0] if r else 0.0


# ── metrics from a notional decision per trade ────────────────────────────────
def metrics(test, final_notion, base_notion, base_net_sum, cohort_ids, flat_ref_dd):
    rows = []
    cohort_pnl = 0.0
    virtual = 0.0           # sum(base_net - final_net) over changed trades (误伤)
    n_taken = 0
    for i, tr in enumerate(test):
        fn = final_notion[i]
        fnet = net_at(tr, fn)
        bnet = net_at(tr, base_notion[i])
        if fn != base_notion[i]:
            virtual += bnet - fnet
        if fn > 0:
            rows.append((tr["exit_dt"], fnet))
            n_taken += 1
        if i in cohort_ids:
            cohort_pnl += fnet
    if not rows:
        return None
    rows.sort(key=lambda x: x[0])
    nets = np.array([r[1] for r in rows])
    cum = np.cumsum(nets)
    dd = float(np.max(np.maximum.accumulate(cum) - cum)) if len(cum) else 0.0
    w = nets[nets > 0].sum()
    loss = -nets[nets < 0].sum()
    pf = (w / loss) if loss > 0 else float("inf")
    netsum = float(nets.sum())
    return {
        "n": n_taken, "net": round(netsum, 0),
        "dpnl_abs": round(netsum - base_net_sum, 0),
        "dpnl_pct": round((netsum - base_net_sum) / abs(base_net_sum) * 100, 2),
        "pf": (None if pf == float("inf") else round(float(pf), 3)),
        "win": round(float((nets > 0).mean()), 4),
        "max_dd": round(dd, 0),
        "dd_over_flatref": round(dd / flat_ref_dd, 3),
        "dd_over_net": round(dd / netsum, 5) if netsum > 0 else None,
        "cohort_pnl": round(cohort_pnl, 0),
        "virtual_blocked_pnl": round(virtual, 0),
    }


# ── rule resolvers: return per-trade final notional list ──────────────────────
def resolve_R1(test, base_notion, X, N_hours):
    """Daily-loss halt. Sequential in entry order. Realized PnL by exit date."""
    order = sorted(range(len(test)), key=lambda i: test[i]["entry_dt"])
    final = list(base_notion)
    open_heap = []          # (exit_dt, taken_net)
    realized = 0.0
    day_start_eq = {}
    day_pnl = {}
    halt_until = None
    triggers = 0
    blocked = 0
    for i in order:
        tr = test[i]
        now = tr["entry_dt"]
        while open_heap and open_heap[0][0] <= now:
            xdt, net = heapq.heappop(open_heap)
            realized += net
            day_pnl[xdt.date()] = day_pnl.get(xdt.date(), 0.0) + net
        d = now.date()
        if d not in day_start_eq:
            day_start_eq[d] = CAPITAL + realized
        if halt_until is not None and now < halt_until:
            final[i] = 0.0
            blocked += 1
            continue
        if day_pnl.get(d, 0.0) <= -X / 100.0 * day_start_eq[d]:
            halt_until = now + pd.Timedelta(hours=N_hours)
            triggers += 1
            final[i] = 0.0
            blocked += 1
            continue
        net = net_at(tr, base_notion[i])
        heapq.heappush(open_heap, (tr["exit_dt"], net))
    return final, {"triggers": triggers, "blocked": blocked, "applicable": len(test)}


def resolve_R2(test, base_notion, K):
    order = sorted(range(len(test)), key=lambda i: test[i]["entry_dt"])
    final = list(base_notion)
    open_heap = []          # (exit_dt, side)
    blocked = 0
    for i in order:
        tr = test[i]
        now = tr["entry_dt"]
        while open_heap and open_heap[0][0] <= now:
            heapq.heappop(open_heap)
        same = sum(1 for _, s in open_heap if s == tr["side"])
        if same >= K:
            final[i] = 0.0
            blocked += 1
            continue
        heapq.heappush(open_heap, (tr["exit_dt"], tr["side"]))
    return final, {"triggers": blocked, "blocked": blocked, "applicable": len(test)}


def resolve_R3(test, base_notion, M, D):
    # same-bar same-dir signal counts within this trade universe
    cnt = {}
    for tr in test:
        key = (tr["entry_dt"], tr["side"])
        cnt[key] = cnt.get(key, 0) + 1
    final = list(base_notion)
    affected = 0
    applicable = 0
    for i, tr in enumerate(test):
        m = cnt[(tr["entry_dt"], tr["side"])]
        if m >= M:
            applicable += 1
            base_idx = TIER.index(base_notion[i]) if base_notion[i] in TIER else 1
            if D == "D1":
                newn = 0.0
            elif D == "D2":
                newn = min(base_notion[i], TIER[0])
            else:  # D3 scale by m
                if m >= 5:
                    newn = 0.0
                else:
                    k = 1 if m == 3 else 2
                    newn = TIER[max(0, base_idx - k)]
                    newn = min(newn, base_notion[i])
            if newn != base_notion[i]:
                affected += 1
            final[i] = newn
    return final, {"triggers": affected, "blocked": affected, "applicable": applicable}


# ── cohort: trades active during any bar with ALL 5 symbols same-dir open ──────
def cohort_5of5(test):
    ev = []
    for tr in test:
        ev.append((tr["entry_dt"], tr["side"], +1))
        ev.append((tr["exit_dt"], tr["side"], -1))
    ev.sort(key=lambda x: x[0])
    open_long = open_short = 0
    long_t, short_t = [], []
    seen = {}
    # track per-symbol direction to count distinct symbols same dir
    # (simpler: count concurrent same-dir POSITIONS; >=5 implies 5 symbols since
    #  one position per symbol at a time)
    cl = cs = 0
    for ts, side, k in ev:
        if side == 1:
            cl += k
        else:
            cs += k
        if cl >= 5:
            long_t.append(ts)
        if cs >= 5:
            short_t.append(ts)
    import bisect
    long_t.sort(); short_t.sort()
    ids = set()
    for i, tr in enumerate(test):
        times = long_t if tr["side"] == 1 else short_t
        j = bisect.bisect_left(times, tr["entry_dt"])
        if j < len(times) and times[j] <= tr["exit_dt"]:
            ids.add(i)
    return ids, len(long_t) + len(short_t)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for sub in ("rule_R1_daily_halt", "rule_R2_position_cap", "rule_R3_cross_sectional"):
        (OUT / sub).mkdir(exist_ok=True)
    log = []
    def L(m=""):
        print(m, flush=True); log.append(m)

    t0 = datetime.now(timezone.utc)
    L(f"# Portfolio risk Phase 2.A — start {t0.isoformat()}")
    L("Reuses research_dyn_v2 engine (UNMODIFIED). Test set only. Baselines FLAT + v2B.")

    hr = parse_history_range(RV.START, RV.END, timedelta(minutes=1), RV.TZ)
    db = PROJECT_ROOT / ".vntrader" / "database.db"
    all_tr = []
    split = {}
    th_by_sym = {}
    for name in SYMBOLS:
        b5 = r5(load_1m(SYMBOLS[name][0], hr, db), 5, hr)
        if b5.empty:
            continue
        trs, si, n, atr, dt = RV.build_trades(name, b5)
        split[name] = si
        th_by_sym[name] = RV.learn_thresholds(atr, np.arange(n) < si)
        all_tr.extend(trs)
    test = [t for t in all_tr if t["entry_idx"] >= split[t["symbol"]]]
    L(f"TEST trades: {len(test):,}")

    cohort_ids, n_cohort_bars = cohort_5of5(test)
    L(f"5/5 same-dir cohort bars: {n_cohort_bars}; cohort trades: {len(cohort_ids)}")

    bases = {}
    for base in ("FLAT", "v2B"):
        bn = [base_notional(t, base, th_by_sym) for t in test]
        nosum = sum(net_at(t, bn[i]) for i, t in enumerate(test))
        bases[base] = {"bn": bn, "nosum": nosum}
    flat_ref_dd = metrics(test, bases["FLAT"]["bn"], bases["FLAT"]["bn"],
                          bases["FLAT"]["nosum"], cohort_ids, 1.0)["max_dd"]
    L(f"FLAT no-rule maxDD (reference for ≤1.5× gate) = ${flat_ref_dd:,.0f} "
      f"(gate=${1.5*flat_ref_dd:,.0f}); FLAT net ${bases['FLAT']['nosum']:,.0f}; "
      f"v2B net ${bases['v2B']['nosum']:,.0f}")

    # no-rule reference metrics per base
    ref = {}
    for base in ("FLAT", "v2B"):
        ref[base] = metrics(test, bases[base]["bn"], bases[base]["bn"],
                            bases[base]["nosum"], cohort_ids, flat_ref_dd)
    L(f"\nNO-RULE: FLAT maxDD ${ref['FLAT']['max_dd']:,.0f} ({ref['FLAT']['dd_over_flatref']}×) "
      f"cohort ${ref['FLAT']['cohort_pnl']:,.0f} | "
      f"v2B maxDD ${ref['v2B']['max_dd']:,.0f} ({ref['v2B']['dd_over_flatref']}×) "
      f"cohort ${ref['v2B']['cohort_pnl']:,.0f}")

    # ── sweep configs ─────────────────────────────────────────────────────────
    configs = []
    for X in (1.0, 1.5, 2.0, 3.0):
        for Nh in (12, 24):
            configs.append(("R1", {"X": X, "N": Nh}, f"X{X}_N{Nh}h"))
    for K in (2, 3, 4):
        configs.append(("R2", {"K": K}, f"K{K}"))
    for M in (3, 4, 5):
        for D in ("D1", "D2", "D3"):
            configs.append(("R3", {"M": M, "D": D}, f"M{M}_{D}"))

    results = {"R1": [], "R2": [], "R3": []}
    false_pos = {}
    for rule, params, tag in configs:
        row = {"tag": tag, **params}
        for base in ("FLAT", "v2B"):
            bn = bases[base]["bn"]
            if rule == "R1":
                final, trig = resolve_R1(test, bn, params["X"], params["N"])
            elif rule == "R2":
                final, trig = resolve_R2(test, bn, params["K"])
            else:
                final, trig = resolve_R3(test, bn, params["M"], params["D"])
            m = metrics(test, final, bn, bases[base]["nosum"], cohort_ids, flat_ref_dd)
            row[base] = m
            row[f"{base}_trig"] = trig
        results[rule].append(row)
        false_pos[f"{rule}_{tag}"] = {
            "v2B_virtual_blocked_pnl": row["v2B"]["virtual_blocked_pnl"],
            "FLAT_virtual_blocked_pnl": row["FLAT"]["virtual_blocked_pnl"],
            "v2B_blocked": row["v2B_trig"]["blocked"],
            "interpretation": ("误伤(blocked profitable)" if row["v2B"]["virtual_blocked_pnl"] > 0
                               else "拦坏的(blocked net-losing)"),
        }

    # ── dump per-rule csv/json ────────────────────────────────────────────────
    rule_dir = {"R1": "rule_R1_daily_halt", "R2": "rule_R2_position_cap",
                "R3": "rule_R3_cross_sectional"}
    for rule, rows in results.items():
        (OUT / rule_dir[rule] / "results.json").write_text(json.dumps(rows, indent=2, default=str))
        with open(OUT / rule_dir[rule] / "results.csv", "w", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(["tag", "base", "n", "net", "dpnl_pct", "pf", "win",
                         "max_dd", "dd/flatref", "dd/net", "cohort_pnl",
                         "virtual_blocked_pnl", "triggers", "blocked"])
            for r in rows:
                for base in ("FLAT", "v2B"):
                    m = r[base]; t = r[f"{base}_trig"]
                    wr.writerow([r["tag"], base, m["n"], m["net"], m["dpnl_pct"],
                                 m["pf"], m["win"], m["max_dd"], m["dd_over_flatref"],
                                 m["dd_over_net"], m["cohort_pnl"],
                                 m["virtual_blocked_pnl"], t["triggers"], t["blocked"]])
    (OUT / "false_positive_analysis.json").write_text(json.dumps(false_pos, indent=2, default=str))

    # ── Step 3: optimal threshold per rule ────────────────────────────────────
    def pick_optimal(rows):
        # criterion: v2B dpnl_pct > -3 ; minimise v2B max_dd ; tiebreak virtual asc
        elig = [r for r in rows if r["v2B"]["dpnl_pct"] > -3.0]
        if not elig:
            return None
        elig.sort(key=lambda r: (r["v2B"]["max_dd"], r["v2B"]["virtual_blocked_pnl"]))
        return elig[0]

    optimal = {}
    L("\n================ STEP 3 — optimal per rule ================")
    for rule, rows in results.items():
        best = pick_optimal(rows)
        optimal[rule] = best["tag"] if best else None
        if not best:
            L(f"\n[{rule}] NO config satisfies (ΔPnL>-3% on v2B). Rule ineffective "
              f"under this data — does NOT enter Phase 2.B.")
            continue
        v = best["v2B"]; fb = best["FLAT"]; tg = best["v2B_trig"]
        gate = "PASS" if v["dd_over_flatref"] <= 1.5 else "FAIL"
        L(f"\n[{rule}] optimal = {best['tag']}")
        L(f"   v2B: ΔPnL {v['dpnl_pct']:+.1f}%  maxDD ${v['max_dd']:,.0f} "
          f"({v['dd_over_flatref']}×FLAT → ≤1.5× {gate})  DD/net {v['dd_over_net']}  "
          f"cohort ${v['cohort_pnl']:,.0f} (no-rule ${ref['v2B']['cohort_pnl']:,.0f})")
        L(f"   FLAT: ΔPnL {fb['dpnl_pct']:+.1f}%  maxDD ${fb['max_dd']:,.0f} "
          f"({fb['dd_over_flatref']}×)  → net-positive on FLAT? "
          f"{'yes' if fb['dpnl_pct'] >= 0 else 'no'}")
        L(f"   triggers {tg['triggers']} / blocked {tg['blocked']} of {tg['applicable']} "
          f"applicable; 误伤(virtual blocked PnL on v2B) ${v['virtual_blocked_pnl']:,.0f}")

    # ── digest tables ─────────────────────────────────────────────────────────
    L("\n================ FULL SWEEP (v2B) ================")
    for rule, rows in results.items():
        L(f"\n-- {rule} (v2B) --   no-rule maxDD ${ref['v2B']['max_dd']:,.0f}")
        L(f"{'cfg':12} {'net$':>10} {'ΔPnL%':>7} {'PF':>5} {'maxDD$':>8} "
          f"{'×FLAT':>6} {'cohort$':>8} {'误伤$':>9} {'blk':>5}")
        for r in rows:
            v = r["v2B"]; t = r["v2B_trig"]
            L(f"{r['tag']:12} {v['net']:>10,.0f} {v['dpnl_pct']:>+7.1f} "
              f"{(v['pf'] if v['pf'] else 0):>5.2f} {v['max_dd']:>8,.0f} "
              f"{v['dd_over_flatref']:>6.3f} {v['cohort_pnl']:>8,.0f} "
              f"{v['virtual_blocked_pnl']:>9,.0f} {t['blocked']:>5}")

    summary = {"reference": ref, "optimal": optimal, "results": results,
               "cohort_trades": len(cohort_ids), "flat_ref_dd": flat_ref_dd}
    (OUT / "step3_optimal.json").write_text(json.dumps({"optimal": optimal,
        "reference": ref}, indent=2, default=str))
    t1 = datetime.now(timezone.utc)
    log.append(f"\n# end {t1.isoformat()} ({(t1-t0).total_seconds():.1f}s)")
    (OUT / "run_log.txt").write_text("\n".join(log))
    return summary


if __name__ == "__main__":
    main()
