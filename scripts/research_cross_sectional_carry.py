#!/usr/bin/env python3
"""Pre-study (single-variable re-test): cross-sectional CARRY at longer holding
periods. Descriptive only, go/no-go. No strategy, no backtest, no portfolio opt.

ONLY HYPOTHESIS: in the cross-sectional IC study
(reports/cross_sectional_ic_20260613/), CAR (funding-sorted long/short) FAILED on
G2 (non-monotone) + G5 (recent decay) at k=1-day holding, yet was the ONLY factor
with BOTH significant full-sample IC (t=3.11) AND true beta-stripped alpha (t=2.08),
and its IC strengthened with horizon (k1 +0.019 / k3 +0.032 / k5 +0.035). This study
tests the single hypothesis: is carry buried by the WRONG (k=1) holding period — at
a mechanism-justified 3-5 day hold, does it pass G1-G5?

MECHANISM-vs-DATA-MINING (verbatim into header):
  funding 为 8h 结算的慢变量，其失衡缓慢回归，机制上合理持有期为数日级。选 3 日/
  5 日基于此机制推断，该推断在观察 IC 衰减曲线之前即成立，非纯数据挑选。但诚实标注：
  本研究系在 k=1 失败后开展，任何通过结论永久标注"持有期系事后选定"，证据等级低于
  一次通过（沿用 V1' 星号纪律）。

DISCIPLINE BOUNDARY (verbatim into header):
  ① 上次 CAR 的 k=1 FAIL 判定保持不变、永久存档；
  ② G1-G5 判定线与上次逐字一致、零修改，不因"重看"放宽任一门；
  ③ 测 3 日与 5 日两个持有期是稳健性检验（两者须一致才可信），不是"挑 IC 更高者
     立项"；若两持有期结论相反 = 参数尖峰，判可疑不立项；
  ④ 本次仅 CAR 因子、仅改持有期一个变量；MOM/REV 不重测。

SURVIVORSHIP (verbatim): 22 当前在市币（幸存者集合，不含 LUNA/FTT 等退市币），收益
  估计系统性偏乐观；全部结论标注"幸存者未校正"。立项后第一步即补退市币敏感性检验。

POSITIONING: go/no-go only; no strategy, params, or weights.

DATA: data/binance_vision/ 22-coin pure-crypto set (READ-ONLY, identical to prior
study). Contaminated DB never touched; no existing DB touched.

============== PRE-REGISTERED (engine reused verbatim; only N changes) =============
ONLY CHANGED VARIABLE: holding period N in {3, 5} days (two independent runs/verdicts).
  N=1 also computed as the holding-period-curve anchor + engine reproduction check
  (must reproduce prior k=1 result), NOT re-judged.
OVERLAPPING PORTFOLIO (classic Jegadeesh-Titman): rebalance daily, each batch held N
  days -> N staggered parallel sub-portfolios, combined weight = 1/N mean of the N
  most-recent batches. Day-d combined return = w_d . (close[d]/close[d-1]-1).
  Turnover (hence cost) falls ~1/N vs daily.
FROZEN from prior study (zero edits): universe (22 pure-crypto, per-day tradeable set,
  no look-ahead, >=10 coins); F-CAR = 7-day mean funding ascending (most negative =
  long), quintiles; fee top-5 {BTC,ETH,SOL,ZEC,XRP} 0.05% / others 0.07%, cost =
  sum|dw|*fee on combined net weights; beta diagnostic; IC sign convention (IC>0 = adds
  value); APY=365. Loaders/spearman/OLS imported from research_cross_sectional_ic.

IC: Spearman(CAR score at t, forward N-day return close[t-1+N]/close[t-1]-1), aligned
  to holding period N. Per-year decomposition.

VERDICT G1-G5 (VERBATIM identical to prior; NOT relaxed):
  G1 |t(IC)| > 2 AND annualised IC_IR > 0.3
  G2 quintile mean (overlapping) monotonic in long-pref direction (<=1 of 4 violations)
  G3 cost-adjusted annualised spread return > 0
  G4 BTC-beta-stripped alpha t > 2
  G5 2025+ IC retains full-sample sign AND |t| > 1
  ALL pass = go-candidate (with "post-hoc holding period" asterisk) -> survivorship stage.
CONSISTENCY (Part 3): N=3 and N=5 must agree (both pass / same character) to be
  credible; one-pass-one-fail or different death = parameter spike -> SUSPECT, not
  started. Holding curve k=1/3/5 must improve monotonically for a true horizon effect.
  ARITHMETIC (only if pass): verification years = (1.96/net_Sharpe)^2.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import research_cross_sectional_ic as base
from research_cross_sectional_ic import (COINS, FEE, MIN_COINS, CAR_W, APY, YEARS,
                                         daily_close, daily_funding, spearman,
                                         ols_alpha_beta)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT = PROJECT_ROOT / "reports" / "cross_sectional_carry_holding_20260613"
HOLDS = (1, 3, 5)            # 1 = anchor/reproduction (not judged); 3,5 = judged
JUDGED = (3, 5)
PRIOR_K1_NET_SHARPE = 0.61   # from reports/cross_sectional_ic_20260613 (reference)

LOG: list[str] = []


def L(msg=""):
    print(msg, flush=True)
    LOG.append(str(msg))


def run_hold(N: int, Cn, Fn, all_days, day_year, btc_i):
    """Full G1-G5 pipeline for CAR at holding period N (overlapping portfolios)."""
    D = len(all_days)
    fee_arr = np.array([FEE[c] for c in COINS])
    n_coins = len(COINS)
    imin = CAR_W + 1
    imax = D - N                       # need close[i-1+N]

    # per-rebalance-day batches: signed spread weights + 5 long-only quintile weights
    batch_spread = {}                  # day i -> signed weight vector
    batch_quint = {}                   # day i -> list of 5 long-only weight vectors
    ic_by_day, ic_year = [], {y: [] for y in YEARS}
    for i in range(imin, imax):
        c_prev = Cn[i - 1]
        score = -Fn[i - 1]             # most negative funding = long
        fwdN = Cn[i - 1 + N] / c_prev - 1.0
        ok = np.isfinite(score) & np.isfinite(fwdN) & np.isfinite(c_prev)
        idx = np.where(ok)[0]
        if len(idx) < MIN_COINS:
            continue
        sc_v = score[idx]
        val = spearman(sc_v, fwdN[idx])
        if np.isfinite(val):
            ic_by_day.append(val)
            ic_year[day_year[all_days[i]]].append(val)
        order = idx[np.argsort(sc_v, kind="stable")]
        n = len(order); m = n // 5
        if m < 1:
            continue
        groups = [order[:m], order[m:2 * m], order[2 * m:3 * m],
                  order[3 * m:4 * m], order[n - m:]]
        ws = np.zeros(n_coins)
        ws[groups[4]] = 1.0 / m
        ws[groups[0]] = -1.0 / m
        batch_spread[i] = ws
        qv = []
        for g in groups:
            wq = np.zeros(n_coins); wq[g] = 1.0 / len(g); qv.append(wq)
        batch_quint[i] = qv

    # overlapping combined daily returns (day d earns close[d]/close[d-1]-1)
    days_sorted = sorted(batch_spread)
    spread_ret, quint_ret = [], [[] for _ in range(5)]
    long_ret, short_ret, btc_ret = [], [], []
    cost_tot = gross_tot = 0.0
    prev_w = np.zeros(n_coins)
    for d in range(imin, imax):
        win = [t for t in range(d - N + 1, d + 1) if t in batch_spread]
        if not win:
            continue
        fwd1 = Cn[d] / Cn[d - 1] - 1.0
        w = np.mean([batch_spread[t] for t in win], axis=0)
        # only coins with finite fwd1 contribute; treat NaN as 0 weight contribution
        f = np.where(np.isfinite(fwd1), fwd1, 0.0)
        gross = float(w @ f)
        cost = float(np.sum(np.abs(w - prev_w) * fee_arr))
        prev_w = w
        spread_ret.append(gross - cost)            # store NET; gross tracked separately
        cost_tot += cost; gross_tot += abs(gross)
        long_w = np.mean([batch_quint[t][4] for t in win], axis=0)
        short_w = np.mean([batch_quint[t][0] for t in win], axis=0)
        long_ret.append(float(long_w @ f)); short_ret.append(float(short_w @ f))
        btc_ret.append(float(fwd1[btc_i]) if np.isfinite(fwd1[btc_i]) else np.nan)
        for qi in range(5):
            qw = np.mean([batch_quint[t][qi] for t in win], axis=0)
            quint_ret[qi].append(float(qw @ f))

    sn = np.array(spread_ret)                      # net daily spread
    sg = sn + 0.0                                  # placeholder; recompute gross below
    # recompute gross series for reporting (net + cost per day not stored; derive gross)
    # gross_daily = net_daily + cost_daily — store cost per day:
    # (re-loop cheaply to get gross/cost split)
    gross_series, cost_series = [], []
    prev_w = np.zeros(n_coins)
    for d in range(imin, imax):
        win = [t for t in range(d - N + 1, d + 1) if t in batch_spread]
        if not win:
            continue
        fwd1 = Cn[d] / Cn[d - 1] - 1.0
        w = np.mean([batch_spread[t] for t in win], axis=0)
        f = np.where(np.isfinite(fwd1), fwd1, 0.0)
        gross_series.append(float(w @ f))
        cost_series.append(float(np.sum(np.abs(w - prev_w) * fee_arr)))
        prev_w = w
    sg = np.array(gross_series); sc = np.array(cost_series)

    ics = np.array(ic_by_day); nrb = len(ics)
    mean_ic = float(ics.mean()); std_ic = float(ics.std(ddof=1))
    t_ic = mean_ic / std_ic * np.sqrt(nrb) if std_ic > 0 else np.nan
    ir_ann = (mean_ic / std_ic) * np.sqrt(APY) if std_ic > 0 else np.nan
    q_means = [float(np.mean(q)) if q else np.nan for q in quint_ret]
    viol = sum(1 for a, b in zip(q_means, q_means[1:]) if b < a)
    gross_ann = float(sg.mean() * APY * 100)
    gross_shp = float(sg.mean() / sg.std(ddof=1) * np.sqrt(APY)) if sg.std() > 0 else np.nan
    net_ann = float(sn.mean() * APY * 100)
    net_shp = float(sn.mean() / sn.std(ddof=1) * np.sqrt(APY)) if sn.std() > 0 else np.nan
    cost_ratio = float(sc.sum() / np.abs(sg).sum()) if np.abs(sg).sum() > 0 else np.nan
    br = np.array(btc_ret); ll = np.array(long_ret); ls = np.array(short_ret)
    msk = np.isfinite(br) & np.isfinite(sg)
    beta = ols_alpha_beta(sg[msk], br[msk])
    beta_long = ols_alpha_beta(ll[msk], br[msk])
    beta_short = ols_alpha_beta(ls[msk], br[msk])
    yr = {str(y): {"mean": float(np.mean(ic_year[y])) if ic_year[y] else None,
                   "n": len(ic_year[y])} for y in YEARS}
    ic25 = ic_year[2025] + ic_year[2026]
    m25 = float(np.mean(ic25)) if ic25 else np.nan
    t25 = (float(m25 / np.std(ic25, ddof=1) * np.sqrt(len(ic25)))
           if len(ic25) > 1 and np.std(ic25) > 0 else np.nan)

    g1 = abs(t_ic) > 2 and ir_ann > 0.3
    g2 = viol <= 1
    g3 = net_ann > 0
    g4 = beta["alpha_t"] > 2
    g5 = np.isfinite(m25) and (np.sign(m25) == np.sign(mean_ic)) and abs(t25) > 1
    all_pass = g1 and g2 and g3 and g4 and g5
    death = None if all_pass else ",".join(
        x for x, ok in [("G1_IC", g1), ("G2_mono", g2), ("G3_net", g3),
                        ("G4_alpha", g4), ("G5_stable", g5)] if not ok)
    verif = ((1.96 / net_shp) ** 2 if all_pass and net_shp and net_shp > 0 else None)

    res = {
        "hold_days": N, "n_rebalance_days": nrb,
        "ic_mean": mean_ic, "ic_std": std_ic, "ic_t": t_ic, "ic_ir_annual": ir_ann,
        "ic_pos_share": float((ics > 0).mean()),
        "quintile_mean_dailyret": q_means, "quintile_monotonic_violations": viol,
        "gross_annual_pct": gross_ann, "gross_sharpe": gross_shp,
        "net_annual_pct": net_ann, "net_sharpe": net_shp, "cost_over_gross_ratio": cost_ratio,
        "beta_spread": beta, "beta_long_leg": beta_long, "beta_short_leg": beta_short,
        "ic_by_year": yr, "ic_2025plus_mean": m25, "ic_2025plus_t": t25,
        "gates": {"G1_ic_sig": g1, "G2_monotonic": g2, "G3_net_positive": g3,
                  "G4_true_alpha": g4, "G5_stable_2025": g5, "ALL": all_pass},
        "death_cause": death, "verification_years": verif,
    }
    L(f"\n== CAR hold N={N}{'  (anchor/reproduction, NOT judged)' if N not in JUDGED else ''} ==")
    L(f"  IC mean {mean_ic:+.4f} t {t_ic:+.2f} IR_ann {ir_ann:+.2f} pos {res['ic_pos_share']:.1%} (n={nrb})")
    L(f"  quintile daily-ret: {[round(x,5) for x in q_means]} viol={viol}")
    L(f"  gross ann {gross_ann:+.1f}% Shp {gross_shp:+.2f} | net ann {net_ann:+.1f}% "
      f"Shp {net_shp:+.2f} | cost/gross {cost_ratio:.2%}")
    L(f"  beta spread {beta['beta']:+.2f} alpha_ann {beta['alpha_annual_pct']:+.1f}% "
      f"alpha_t {beta['alpha_t']:+.2f} | long beta {beta_long['beta']:+.2f} short beta {beta_short['beta']:+.2f}")
    L(f"  IC by year: " + " ".join(
        f"{y}:{yr[str(y)]['mean']:+.3f}" if yr[str(y)]['mean'] is not None else f"{y}:-" for y in YEARS))
    L(f"  2025+ IC {m25:+.4f} t {t25:+.2f}")
    if N in JUDGED:
        L(f"  GATES G1{int(g1)} G2{int(g2)} G3{int(g3)} G4{int(g4)} G5{int(g5)} -> "
          f"{'GO-CANDIDATE*' if all_pass else 'FAIL ['+str(death)+']'}")
    return res


def main() -> int:
    L(f"run start (UTC): {datetime.now(timezone.utc).isoformat()}")
    L("DATA ENVIRONMENT: data/binance_vision 22-coin pure-crypto (read-only); "
      "contaminated DB not touched; survivorship NOT corrected")
    L("SINGLE VARIABLE: holding period N in {3,5} (N=1 anchor); engine + G1-G5 reused "
      "VERBATIM from research_cross_sectional_ic; gates NOT relaxed; CAR only; post-hoc "
      "holding period -> asterisk on any pass")
    OUT.mkdir(parents=True, exist_ok=True)

    L("\n== loading 22 coins: daily close + daily funding (reused loaders) ==")
    closes, fundings, all_days = {}, {}, None
    for c in COINS:
        closes[c] = daily_close(c); fundings[c] = daily_funding(c)
    all_days = sorted(set().union(*[set(s.index) for s in closes.values()]))
    C = pd.DataFrame(index=all_days, columns=COINS, dtype=float)
    Fund = pd.DataFrame(index=all_days, columns=COINS, dtype=float)
    for c in COINS:
        C[c] = closes[c].reindex(all_days); Fund[c] = fundings[c].reindex(all_days)
    FundMA = Fund.rolling(CAR_W, min_periods=CAR_W).mean()
    Cn = C.to_numpy(float); Fn = FundMA.to_numpy(float)
    btc_i = COINS.index("BTCUSDT")
    day_year = {d: pd.Timestamp(int(d) * 1440 * 60, unit="s", tz="UTC").year for d in all_days}
    L(f"  {len(COINS)} coins, {len(all_days)} day-grid")

    res = {N: run_hold(N, Cn, Fn, all_days, day_year, btc_i) for N in HOLDS}

    # ── consistency adjudication ────────────────────────────────────────────
    p3, p5 = res[3]["gates"]["ALL"], res[5]["gates"]["ALL"]
    if p3 and p5:
        consistency, final = "CONSISTENT (both pass)", "GO-CANDIDATE* (post-hoc holding period)"
    elif (not p3) and (not p5):
        same = res[3]["death_cause"] == res[5]["death_cause"]
        consistency = f"CONSISTENT (both fail{'·same death' if same else '·diff death'})"
        final = "NOT STARTED (carry MR clue terminated on survivor sample)"
    else:
        consistency, final = "CONTRADICTORY (one pass one fail) -> parameter spike SUSPECT", \
                             "NOT STARTED (parameter-sensitive, suspect)"
    curve = {str(N): {"ic_mean": res[N]["ic_mean"],
                      "quintile_violations": res[N]["quintile_monotonic_violations"],
                      "net_sharpe": res[N]["net_sharpe"]} for N in HOLDS}
    ic_mono = res[1]["ic_mean"] <= res[3]["ic_mean"] <= res[5]["ic_mean"]

    L(f"\n== CONSISTENCY: {consistency} ==")
    L(f"  holding curve (k=1 anchor / 3 / 5):")
    for N in HOLDS:
        L(f"    N={N}: IC {res[N]['ic_mean']:+.4f}  viol {res[N]['quintile_monotonic_violations']}  "
          f"netShp {res[N]['net_sharpe']:+.2f}  G1-5 "
          f"{''.join(str(int(res[N]['gates'][g])) for g in ['G1_ic_sig','G2_monotonic','G3_net_positive','G4_true_alpha','G5_stable_2025'])}")
    L(f"  IC monotone in horizon (1<=3<=5): {ic_mono}")
    L(f"  N=1 reproduction check: IC {res[1]['ic_mean']:+.4f} (prior k1 +0.0188), "
      f"net Sharpe {res[1]['net_sharpe']:+.2f} (prior k1 ~{PRIOR_K1_NET_SHARPE})")
    L(f"\n  FINAL: {final}")

    for N in HOLDS:
        (OUT / f"hold_{N}d.json").write_text(
            json.dumps(res[N], indent=2, default=lambda x: None if x is None else float(x)))
    (OUT / "summary.json").write_text(json.dumps({
        "positioning": "single-variable re-test (holding period) of CAR only; engine + "
                       "G1-G5 reused verbatim; gates NOT relaxed; post-hoc holding period "
                       "-> asterisk; survivorship NOT corrected",
        "prior_k1_verdict": "FAIL (G2 non-monotone + G5 decay) — retained, archived",
        "judged_holds": list(JUDGED), "anchor_hold": 1,
        "results": {str(N): res[N] for N in HOLDS},
        "consistency": consistency, "ic_monotone_in_horizon": ic_mono,
        "holding_curve": curve, "final_verdict": final,
    }, indent=2, default=lambda x: None if x is None else float(x)))
    L(f"run end (UTC): {datetime.now(timezone.utc).isoformat()}")
    (OUT / "run_log.txt").write_text("\n".join(LOG) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
