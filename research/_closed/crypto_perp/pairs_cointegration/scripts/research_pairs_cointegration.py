#!/usr/bin/env python3
"""Pre-study: pair / spread mean-reversion via rolling cointegration (descriptive,
go/no-go). No strategy, no backtest, no PnL curve, no portfolio optimisation, no
pair whitelist output. Last leg of the MR family.

ONLY QUESTION: in the 22-coin pure-crypto set, do stably-cointegrated pairs exist
whose post-deviation spread reversion is positive AFTER cost, with cointegration
stable enough to trade — and crucially, do reversion profits cover the break tail?

MECHANISM (verbatim into header):
  价差是两个右偏序列的差，对冲掉共同方向运动后，剩余的相对偏离没有理由是右偏延续的
  ——这是配对交易长期存在的理论基础。本研究检验它在加密 22 币上是否成立。

PRIOR (verbatim): 配对理论基础扎实，但加密协整关系不稳定，可因基本面变化(升级/暴雷/
  叙事)永久断裂(如 LUNA 类)；22 币均主流大币、板块分散，天然协整对可能很少、样本小。
  先验中性，非稳赢。

SURVIVORSHIP (verbatim): 22 币幸存者集合，不含退市币。协整破裂最极端形态(一腿退市归零)
  在本样本缺失 → 协整稳定性估计系统性偏乐观。结论标注"幸存者未校正、破裂风险低估"。

POSITIONING: go/no-go only; no strategy/params/whitelist.

DATA: data/binance_vision/ 22-coin pure-crypto set (READ-ONLY). Contaminated DB never
touched; no existing DB touched. statsmodels for Engle-Granger/ADF (version in run_log).

============== PRE-REGISTERED (frozen; zero variants) =============================
FREQUENCY: daily (same as carry study, for comparability).
SPREAD: log spread. Per pair (A,B): OLS log(A)=a+b*log(B) on the FORMATION window ->
  (a,b); spread = log(A) - b*log(B) - a (residual). NOT price ratio (no basis for b=1).
PAIR SET: all C(22,2)=231 pairs. Tradeability per window decided by rolling
  cointegration screen (below); NEVER full-sample pair selection.

ROLLING SCREEN (anti-look-ahead, the life/death line):
  formation = 90 days (estimate b + cointegration test); trading = next 30 days (use
  formation b, mean, std; NOT re-estimated). Roll forward by 30 days (monthly).
  At each formation end: Engle-Granger on all 231 pairs (ADF on residual via
  statsmodels.coint, trend='c', maxlag=1, autolag=None — FROZEN for determinism/speed,
  uses MacKinnon EG p-values). Tradeable set = pairs with EG p < 0.05.
  A pair enters a window only if BOTH legs have a complete 90-day formation history
  (no forward-fill) and >=2 trading-period observations.
  Persistence = consecutive windows a pair stays EG-significant.

DEVIATION EVENTS (trading period, tradeable pairs only):
  z = (spread - formation_mean) / formation_std. FIRST |z|>=2 crossing per pair-window
  is the event (one event per pair-window; frozen). Resolve at first of: |z|<0.5
  (REVERT) / |z|>3 (BLOWOUT) / trading end (UNRESOLVED). Outcome gross (two-leg combined
  log return, %) = sign(z_trig)*(spread_trig - spread_exit)*100 (positive = converged).
  Reversion rate = REVERT share. Baseline = unconditional P(reach |z|<0.5 within
  remaining trading window | any trading day) across tradeable pair-windows.
  COST = 0.20% (pair = two legs * taker double-sided 0.05%; large caps, single tier).
  mean AND median reported.

VERDICT (frozen, may not change after results):
  C1 cointegration exists : mean tradeable pairs per window >= 3
  C2 cointegration stable : median persistence (consecutive sig windows) >= 2
  C3 reversion real       : event reversion rate > unconditional baseline, significant
                            (two-proportion z, |z|>2)
  C4 thickness            : reversion gross mean - 0.20% > 0 AND mean/median same sign
  C5 break controlled     : event-weighted net expectation > 0, i.e.
                            mean(ALL events outcome_gross) - 0.20% > 0 (reversion profit
                            covers blowout/unresolved losses) — the real pair-trade test.
  ALL pass = go-candidate; any fail = FAIL with named death cause.
  ARITHMETIC (pass only): coarse Sharpe from event-weighted net + freq -> verification
  years (1.96/S)^2 -> <=12mo go / 12-24 marginal / >24 not started.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
# 2026-07 重构批次5：脚本迁入 research/_closed/crypto_perp/pairs_cointegration/scripts/；共享依赖真身在
# scripts/（前向冻结区）与 core/data_io/，此处按新深度注入 sys.path。
import sys as _sys
from pathlib import Path as _Path

_REPO_ROOT = _Path(__file__).resolve().parents[5]
for _p in (
    str(_REPO_ROOT / "core" / "data_io"),
    str(_REPO_ROOT / "scripts"),
    *sorted(str(_q) for _q in (_REPO_ROOT / "research" / "_closed").glob("*/*/scripts")),
):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)

from research_cross_sectional_ic import COINS, daily_close

PROJECT_ROOT = Path(__file__).resolve().parents[5]  # 2026-07 重构批次5：迁入 research/_closed/crypto_perp/pairs_cointegration/scripts/，深度 1→5
OUT = PROJECT_ROOT / "reports" / "pairs_cointegration_20260613"

FORM, TRADE, ROLL = 90, 30, 30
P_COINT = 0.05
Z_ENTRY, Z_EXIT, Z_BLOW = 2.0, 0.5, 3.0
COST_PCT = 0.20
C1_MIN_PAIRS, C2_MIN_PERSIST = 3, 2

LOG: list[str] = []


def L(msg=""):
    print(msg, flush=True)
    LOG.append(str(msg))


def ols_ab(ya: np.ndarray, yb: np.ndarray):
    X = np.column_stack([np.ones(len(yb)), yb])
    coef, *_ = np.linalg.lstsq(X, ya, rcond=None)
    return float(coef[0]), float(coef[1])      # alpha, beta


def main() -> int:
    import statsmodels
    from statsmodels.tsa.stattools import coint
    L(f"run start (UTC): {datetime.now(timezone.utc).isoformat()}")
    L(f"statsmodels {statsmodels.__version__}; numpy {np.__version__}")
    L("DATA ENVIRONMENT: data/binance_vision 22-coin pure-crypto (read-only); "
      "contaminated DB not touched; survivorship NOT corrected (break-tail underestimated)")
    L("PRE-REGISTERED: form90/trade30/roll30, EG p<0.05 (trend=c,maxlag=1), z=+-2 entry "
      "/0.5 exit/3 blowout, cost 0.20%, C1-C5 frozen; rolling formation/trading split, "
      "NO full-sample pair selection; zero variants")
    OUT.mkdir(parents=True, exist_ok=True)

    L("\n== loading 22 coins daily close -> log price matrix ==")
    closes = {c: daily_close(c) for c in COINS}
    all_days = sorted(set().union(*[set(s.index) for s in closes.values()]))
    C = pd.DataFrame(index=all_days, columns=COINS, dtype=float)
    for c in COINS:
        C[c] = closes[c].reindex(all_days)
    LP = np.log(C.to_numpy(dtype=float))        # day x coin
    D = len(all_days)
    pairs = list(combinations(range(len(COINS)), 2))
    L(f"  {len(COINS)} coins, {D} day-grid, {len(pairs)} pairs")

    # rolling windows: formation [fe-FORM, fe), trading [fe, fe+TRADE)
    window_starts = list(range(FORM, D - TRADE + 1, ROLL))
    L(f"  {len(window_starts)} rolling windows (form {FORM}d / trade {TRADE}d / roll {ROLL}d)")

    pairs_per_window = []
    persist = {p: 0 for p in pairs}              # running consecutive-significant counter
    persist_runs = []                            # completed run lengths
    prev_sig = set()
    churn_list = []

    # event accumulators
    ev_outcomes = []        # outcome_gross % for ALL first-events
    ev_revert = []          # reversion gross % for REVERT events
    ev_class = {"revert": 0, "blowout": 0, "unresolved": 0}
    base_reach_num = base_reach_den = 0
    cond_reach_num = cond_reach_den = 0

    for wi, fe in enumerate(window_starts):
        f0, f1 = fe - FORM, fe
        t0, t1 = fe, min(fe + TRADE, D)
        sig_pairs = set()
        for (ia, ib) in pairs:
            a_f, b_f = LP[f0:f1, ia], LP[f0:f1, ib]
            if not (np.isfinite(a_f).all() and np.isfinite(b_f).all()):
                continue
            a_t, b_t = LP[t0:t1, ia], LP[t0:t1, ib]
            ok_t = np.isfinite(a_t) & np.isfinite(b_t)
            if ok_t.sum() < 2:
                continue
            try:
                pval = coint(a_f, b_f, trend="c", maxlag=1, autolag=None)[1]
            except Exception:
                continue
            if pval >= P_COINT:
                continue
            sig_pairs.add((ia, ib))
            # spread on formation b,a
            alpha, beta = ols_ab(a_f, b_f)
            sp_f = a_f - beta * b_f - alpha
            mu, sd = float(sp_f.mean()), float(sp_f.std(ddof=1))
            if sd <= 0:
                continue
            sp_t = (a_t - beta * b_t - alpha)
            z = (sp_t - mu) / sd
            zt = z[ok_t]; spt = sp_t[ok_t]
            n = len(zt)
            # baseline reach: from each day, does |z|<Z_EXIT occur later in window
            below = np.abs(zt) < Z_EXIT
            future_reach = np.zeros(n, bool)
            seen = False
            for j in range(n - 1, -1, -1):
                future_reach[j] = seen
                if below[j]:
                    seen = True
            for j in range(n - 1):
                base_reach_den += 1
                if future_reach[j]:
                    base_reach_num += 1
            # first |z|>=2 event
            trig = next((j for j in range(n) if abs(zt[j]) >= Z_ENTRY), None)
            if trig is None:
                continue
            cond_reach_den += 1
            sgn = np.sign(zt[trig]); sp_trig = spt[trig]
            outcome = None; klass = "unresolved"
            for j in range(trig + 1, n):
                if abs(zt[j]) < Z_EXIT:
                    outcome = sgn * (sp_trig - spt[j]) * 100; klass = "revert"; break
                if abs(zt[j]) > Z_BLOW:
                    outcome = sgn * (sp_trig - spt[j]) * 100; klass = "blowout"; break
            if outcome is None:
                outcome = sgn * (sp_trig - spt[-1]) * 100; klass = "unresolved"
            ev_outcomes.append(outcome)
            ev_class[klass] += 1
            if klass == "revert":
                ev_revert.append(outcome); cond_reach_num += 1

        pairs_per_window.append(len(sig_pairs))
        # persistence bookkeeping
        for p in list(persist):
            if p in sig_pairs:
                persist[p] += 1
            elif persist[p] > 0:
                persist_runs.append(persist[p]); persist[p] = 0
        if prev_sig or sig_pairs:
            inter = len(prev_sig & sig_pairs)
            union = len(prev_sig | sig_pairs)
            churn_list.append(1 - inter / union if union else 0.0)
        prev_sig = sig_pairs
        if wi % 10 == 0:
            d0 = pd.Timestamp(int(all_days[f1 - 1]) * 1440 * 60, unit="s", tz="UTC").date()
            L(f"  window {wi:>2} (form end {d0}): {len(sig_pairs)} cointegrated pairs")
    for p in persist:
        if persist[p] > 0:
            persist_runs.append(persist[p])

    # ── metrics ──────────────────────────────────────────────────────────────
    ppw = np.array(pairs_per_window)
    mean_pairs = float(ppw.mean())
    med_persist = float(np.median(persist_runs)) if persist_runs else 0.0
    mean_persist = float(np.mean(persist_runs)) if persist_runs else 0.0
    churn = float(np.mean(churn_list)) if churn_list else None

    base_rate = base_reach_num / base_reach_den if base_reach_den else float("nan")
    cond_rate = cond_reach_num / cond_reach_den if cond_reach_den else float("nan")
    # two-proportion z
    p_pool = ((base_reach_num + cond_reach_num) / (base_reach_den + cond_reach_den)
              if (base_reach_den + cond_reach_den) else float("nan"))
    se = np.sqrt(p_pool * (1 - p_pool) * (1 / base_reach_den + 1 / cond_reach_den)) \
        if base_reach_den and cond_reach_den else float("nan")
    z_prop = (cond_rate - base_rate) / se if se and se > 0 else float("nan")

    rev = np.array(ev_revert); allo = np.array(ev_outcomes)
    rev_mean = float(rev.mean()) if len(rev) else float("nan")
    rev_med = float(np.median(rev)) if len(rev) else float("nan")
    all_mean = float(allo.mean()) if len(allo) else float("nan")
    all_med = float(np.median(allo)) if len(allo) else float("nan")
    net_thick = rev_mean - COST_PCT
    net_eventwt = all_mean - COST_PCT

    c1 = mean_pairs >= C1_MIN_PAIRS
    c2 = med_persist >= C2_MIN_PERSIST
    c3 = (cond_rate > base_rate) and (abs(z_prop) > 2)
    c4 = (net_thick > 0) and (np.sign(rev_mean) == np.sign(rev_med))
    c5 = net_eventwt > 0
    all_pass = c1 and c2 and c3 and c4 and c5
    death = None if all_pass else ",".join(
        x for x, ok in [("C1_exist", c1), ("C2_stable", c2), ("C3_reversion", c3),
                        ("C4_thick", c4), ("C5_break", c5)] if not ok)

    res = {
        "n_windows": len(window_starts), "n_pairs_universe": len(pairs),
        "C1_mean_tradeable_pairs_per_window": mean_pairs,
        "pairs_per_window_min_max": [int(ppw.min()), int(ppw.max())],
        "C2_median_persistence_windows": med_persist,
        "mean_persistence_windows": mean_persist,
        "n_persistence_runs": len(persist_runs),
        "mean_churn_rate": churn,
        "n_deviation_events": int(len(allo)),
        "event_class": ev_class,
        "reversion_rate_conditional": cond_rate, "baseline_reach_rate": base_rate,
        "twoprop_z": z_prop,
        "reversion_gross_mean_pct": rev_mean, "reversion_gross_median_pct": rev_med,
        "all_events_gross_mean_pct": all_mean, "all_events_gross_median_pct": all_med,
        "cost_pct": COST_PCT,
        "net_thickness_mean_pct": net_thick, "net_eventweighted_mean_pct": net_eventwt,
        "gates": {"C1_exist": c1, "C2_stable": c2, "C3_reversion": c3,
                  "C4_thick": c4, "C5_break": c5, "ALL": all_pass},
        "death_cause": death,
    }

    L("\n== RESULTS ==")
    L(f"  C1 mean tradeable pairs/window: {mean_pairs:.2f} (min {ppw.min()} max {ppw.max()}) "
      f">=3? {c1}")
    L(f"  C2 median persistence: {med_persist:.1f} windows (mean {mean_persist:.2f}, "
      f"{len(persist_runs)} runs, churn {churn:.2f}) >=2? {c2}")
    L(f"  C3 reversion rate {cond_rate:.3f} vs baseline {base_rate:.3f} (two-prop z {z_prop:+.2f}) "
      f"cond>base&sig? {c3}")
    L(f"  events: {len(allo)} ({ev_class['revert']} revert / {ev_class['blowout']} blowout / "
      f"{ev_class['unresolved']} unresolved)")
    L(f"  C4 reversion gross mean {rev_mean:+.4f}% med {rev_med:+.4f}% - cost {COST_PCT}% "
      f"= net {net_thick:+.4f}% mean/med same sign? {c4}")
    L(f"  C5 ALL-event gross mean {all_mean:+.4f}% med {all_med:+.4f}% - cost = "
      f"net eventwt {net_eventwt:+.4f}% >0? {c5}")
    L(f"  GATES C1{int(c1)} C2{int(c2)} C3{int(c3)} C4{int(c4)} C5{int(c5)} -> "
      f"{'GO-CANDIDATE' if all_pass else 'FAIL ['+str(death)+']'}")

    final = ("GO-CANDIDATE -> survivorship sensitivity stage" if all_pass else
             "NOT STARTED (pairs cointegration not viable on survivor sample) -> "
             "MR FAMILY FULLY CLOSED")
    L(f"  FINAL: {final}")

    (OUT / "summary.json").write_text(json.dumps({
        "positioning": "descriptive pairs-cointegration pre-study, go/no-go only; rolling "
                       "formation/trading split (no full-sample selection); survivorship "
                       "NOT corrected (break tail underestimated); no PnL/optimisation/whitelist",
        "pre_registered": {"form_days": FORM, "trade_days": TRADE, "roll_days": ROLL,
                           "coint_p": P_COINT, "z_entry": Z_ENTRY, "z_exit": Z_EXIT,
                           "z_blowout": Z_BLOW, "cost_pct": COST_PCT,
                           "C1_min_pairs": C1_MIN_PAIRS, "C2_min_persist": C2_MIN_PERSIST,
                           "eg_test": "statsmodels.coint trend=c maxlag=1 autolag=None"},
        "results": res, "final_verdict": final,
    }, indent=2, default=lambda x: None if x is None else float(x)))
    (OUT / "cointegration_screen_timeseries.json").write_text(json.dumps({
        "pairs_per_window": pairs_per_window,
        "persistence_runs": persist_runs,
        "churn_rate_series": churn_list,
    }, indent=2, default=float))
    (OUT / "reversion_events.json").write_text(json.dumps({
        "n_events": int(len(allo)), "class": ev_class,
        "reversion_gross_pct": {"mean": rev_mean, "median": rev_med, "n": int(len(rev))},
        "all_events_gross_pct": {"mean": all_mean, "median": all_med, "n": int(len(allo))},
        "reversion_rate": cond_rate, "baseline_reach_rate": base_rate, "twoprop_z": z_prop,
        "net_thickness_pct": net_thick, "net_eventweighted_pct": net_eventwt,
    }, indent=2, default=lambda x: None if x is None else float(x)))
    L(f"run end (UTC): {datetime.now(timezone.utc).isoformat()}")
    (OUT / "run_log.txt").write_text("\n".join(LOG) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
