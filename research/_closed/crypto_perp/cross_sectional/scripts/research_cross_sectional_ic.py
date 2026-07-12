#!/usr/bin/env python3
"""Pre-study: cross-sectional signal IC (descriptive only, go/no-go).
No strategy, no backtest, no PnL curve, no portfolio optimisation.

ONLY QUESTION: are crypto cross-sectional factors (momentum / carry / reversal)
informative (IC significant + stable) and is the cost-adjusted top-bottom spread
positive AND a true cross-sectional alpha (not disguised BTC-beta)?

PRIOR (verbatim into report header):
  横截面动量在股票市场有数十年文献支撑，但加密市场证据混合：早期(2018-2021)较强，
  近年随市场成熟可能衰减。加密横截面独有风险是与 BTC 的 beta 难以剥离（强者常为
  高 beta 币，多空两腿 beta 不对称→残留方向暴露）。先验中性偏正，非稳赢。

SURVIVORSHIP (verbatim into header):
  本研究用当前 22 币（幸存者集合，不含已退市币如 LUNA/FTT）。横截面收益估计因此
  系统性偏乐观（输家已从样本蒸发）。本阶段接受此局限以快速判生死；若信号存在，
  立项后第一步即补退市币做幸存者敏感性检验。全部结论标注"幸存者未校正"。

POSITIONING: produces go/no-go only; no strategy, parameters, or portfolio weights.

DATA: data/binance_vision/ 22-coin pure-crypto set (READ-ONLY). Contaminated DB
never touched; no existing DB touched.

================== PRE-REGISTERED DEFINITIONS (frozen; zero variants) ==============
UNIVERSE: 22 pure-crypto USDT perps. Tokenised traditional assets mechanically
  excluded (this batch confirmed to contain none). Tradeable set on each rebalance =
  coins listed with a FULL factor window of history as of the prior close (no
  forward-fill, no look-ahead). A rebalance day is used only if >=10 tradeable coins
  (so each quintile has >=2 names); else skipped.

REBALANCE: daily at UTC 00:00 using data through the PRIOR day's close. Single
  frequency, NOT scanned. Day index i: factor uses closes <= i-1; forward k-day
  return = close[i-1+k]/close[i-1]-1 (fwd1 = the return realised during day i).

FACTORS (three classic prototypes, params frozen, NOT scanned):
  F-MOM  momentum : 30-day cumulative return, long-preferred score = +ret30.
  F-CAR  carry    : 7-day mean funding rate, long-preferred score = -mean_funding
                    (most negative funding = longs get paid = long).
  F-REV  reversal : 3-day cumulative return, long-preferred score = -ret3
                    (recent relative losers = long; cross-sectional, not single-series MR).
  IC sign convention: IC = Spearman(long-preferred score, forward return); IC>0 means
  the factor adds value in its intended direction (uniform across factors).

PART 2 (IC, pure stats, no cost):
  (2a) IC series = per-day Spearman(score, fwd1). Report IC mean, IC std, IC_IR
       (=mean/std) and annualised IC_IR (xsqrt(365)), IC>0 share, t = mean/std*sqrt(n).
       IC decay for fwd k in {1,3,5}.
  (2b) quintile monotonicity: 5 groups by score; mean fwd1 per quintile.
  (2c) IC by year (2020..2026): does crypto cross-sectional alpha decay recently?

PART 3 (cost-adjusted spread + beta diagnostic):
  (3a) gross top-bottom: long top quintile, short bottom quintile, equal weight;
       daily spread return series; gross annualised return (x365) and Sharpe (xsqrt365).
  (3b) cost: per-name turnover * per-coin taker fee. Fee tiers (FROZEN): top-5 by
       2026-05 volume {BTC,ETH,SOL,ZEC,XRP} = 0.05%; others = 0.07%. Daily cost =
       sum_i |w[i] - w_prev[i]| * fee_i (sum over series counts every open AND close =
       double-sided). Report net annualised return, net Sharpe, cost/gross ratio.
  (3c) beta-neutral diagnostic: OLS of daily spread return on BTC daily return ->
       residual beta, alpha, alpha t. Long-leg and short-leg BTC betas separately
       (asymmetry check). Answers: relative-rank alpha vs disguised directional beta.

PART 4 (verdict, FROZEN, may not change after results) — per factor:
  G1 IC significant : |t(IC)| > 2 AND annualised IC_IR > 0.3
  G2 monotonic      : quintile mean fwd1 monotonic in long-preferred direction
                      (<=1 adjacent-pair violation of 4)
  G3 net spread >0  : 3b cost-adjusted annualised return > 0
  G4 true alpha     : 3c BTC-beta-stripped alpha t > 2 (else "disguised directional
                      exposure" -> handled as already-closed directional signal)
  G5 time stable    : 2025+ IC retains full-sample sign AND its own |t| > 1
                      (not collapsed to ~0, not sign-flipped)
  ALL pass = go-candidate (next stage: delisted-coin survivorship sensitivity +
  verification arithmetic); any fail = factor FAIL with named death cause.
  ARITHMETIC (passing factors only): verification years = (1.96/net_Sharpe)^2;
  <=12mo start / 12-24mo marginal / >24mo NOT STARTED. mean AND IR both reported.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
# 2026-07 重构批次5：脚本迁入 research/_closed/crypto_perp/cross_sectional/scripts/；共享依赖真身在
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

from research_trend_dualcycle import load_1m_bv
from binance_funding import load_funding_binance

PROJECT_ROOT = Path(__file__).resolve().parents[5]  # 2026-07 重构批次5：迁入 research/_closed/crypto_perp/cross_sectional/scripts/，深度 1→5
OUT = PROJECT_ROOT / "reports" / "cross_sectional_ic_20260613"

COINS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "ZECUSDT", "XRPUSDT", "DOGEUSDT",
         "BNBUSDT", "SUIUSDT", "NEARUSDT", "TONUSDT", "1000PEPEUSDT", "XLMUSDT",
         "WLDUSDT", "ONDOUSDT", "TAOUSDT", "ADAUSDT", "FILUSDT", "LINKUSDT",
         "AVAXUSDT", "BCHUSDT", "ENAUSDT", "INJUSDT"]
TOP5 = {"BTCUSDT", "ETHUSDT", "SOLUSDT", "ZECUSDT", "XRPUSDT"}
FEE = {c: (0.0005 if c in TOP5 else 0.0007) for c in COINS}

MOM_W, CAR_W, REV_W = 30, 7, 3
MIN_COINS = 10
KS = (1, 3, 5)
YEARS = (2020, 2021, 2022, 2023, 2024, 2025, 2026)
APY = 365.0

LOG: list[str] = []


def L(msg=""):
    print(msg, flush=True)
    LOG.append(str(msg))


def daily_close(sym: str) -> pd.Series:
    m1 = load_1m_bv(sym)
    day = (m1["min_utc"] // 1440).to_numpy()
    s = pd.Series(m1["close"].to_numpy(), index=day)
    return s.groupby(level=0).last()


def daily_funding(sym: str) -> pd.Series:
    m1 = load_1m_bv(sym)[["min_utc", "close"]]
    f = load_funding_binance(sym, m1)
    day = (f["slot_min"] // 1440).to_numpy()
    s = pd.Series(f["rate"].to_numpy(), index=day)
    return s.groupby(level=0).mean()


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = pd.Series(a).rank().to_numpy()
    rb = pd.Series(b).rank().to_numpy()
    ra = ra - ra.mean(); rb = rb - rb.mean()
    d = np.sqrt((ra * ra).sum() * (rb * rb).sum())
    return float((ra * rb).sum() / d) if d > 0 else np.nan


def ols_alpha_beta(y: np.ndarray, x: np.ndarray) -> dict:
    n = len(y)
    X = np.column_stack([np.ones(n), x])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    dof = n - 2
    sigma2 = (resid @ resid) / dof
    cov = sigma2 * np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(cov))
    return {"alpha_daily": float(coef[0]), "beta": float(coef[1]),
            "alpha_t": float(coef[0] / se[0]) if se[0] > 0 else np.nan,
            "alpha_annual_pct": float(coef[0] * APY * 100)}


def main() -> int:
    L(f"run start (UTC): {datetime.now(timezone.utc).isoformat()}")
    L("DATA ENVIRONMENT: data/binance_vision 22-coin pure-crypto set (read-only); "
      "contaminated DB not touched; survivorship NOT corrected (current-listed set)")
    L("PRE-REGISTERED: MOM30/CAR7/REV3, daily rebalance, quintiles, >=10 coins, "
      "fee 0.05%(top5)/0.07%, G1-G5 frozen; zero variants")
    OUT.mkdir(parents=True, exist_ok=True)

    L("\n== loading 22 coins: daily close + daily funding ==")
    closes, fundings = {}, {}
    for c in COINS:
        closes[c] = daily_close(c)
        fundings[c] = daily_funding(c)
        L(f"  {c:<14} days={len(closes[c]):>5} ({closes[c].index.min()}..{closes[c].index.max()})  "
          f"funding-days={len(fundings[c])}")

    all_days = sorted(set().union(*[set(s.index) for s in closes.values()]))
    C = pd.DataFrame(index=all_days, columns=COINS, dtype=float)
    Fund = pd.DataFrame(index=all_days, columns=COINS, dtype=float)
    for c in COINS:
        C[c] = closes[c].reindex(all_days)
        Fund[c] = fundings[c].reindex(all_days)
    # 7-day trailing mean funding (per coin), aligned to day index
    FundMA = Fund.rolling(CAR_W, min_periods=CAR_W).mean()
    Cn = C.to_numpy(dtype=float)
    Fn = FundMA.to_numpy(dtype=float)
    D = len(all_days)
    btc_i = COINS.index("BTCUSDT")

    # per-factor accumulators
    factors = ["F-MOM", "F-CAR", "F-REV"]
    ic = {f: {k: [] for k in KS} for f in factors}
    ic_year = {f: {y: [] for y in YEARS} for f in factors}
    quint_ret = {f: [[] for _ in range(5)] for f in factors}
    spread_gross = {f: [] for f in factors}
    spread_net = {f: [] for f in factors}
    leg_long = {f: [] for f in factors}
    leg_short = {f: [] for f in factors}
    btc_ret_series = {f: [] for f in factors}
    cost_tot = {f: 0.0 for f in factors}
    gross_tot = {f: 0.0 for f in factors}
    prev_w = {f: np.zeros(len(COINS)) for f in factors}
    fee_arr = np.array([FEE[c] for c in COINS])

    day_year = {d: pd.Timestamp(int(d) * 1440 * 60, unit="s", tz="UTC").year for d in all_days}

    imin = MOM_W + 1
    imax = D - max(KS)
    for i in range(imin, imax):
        c_prev = Cn[i - 1]
        c_now = Cn[i]
        # forward returns
        fwd = {k: Cn[i - 1 + k] / c_prev - 1.0 for k in KS}
        # long-preferred scores
        score = {
            "F-MOM": c_prev / Cn[i - MOM_W - 1] - 1.0,
            "F-CAR": -Fn[i - 1],                                  # 7d funding through i-1
            "F-REV": -(c_prev / Cn[i - REV_W - 1] - 1.0),
        }
        year = day_year[all_days[i]]
        for f in factors:
            sc = score[f]
            f1 = fwd[1]
            ok = np.isfinite(sc) & np.isfinite(f1) & np.isfinite(c_now) & np.isfinite(c_prev)
            idx = np.where(ok)[0]
            if len(idx) < MIN_COINS:
                continue
            sc_v = sc[idx]
            # IC at k horizons (need finite fwd_k)
            for k in KS:
                fk = fwd[k]
                okk = idx[np.isfinite(fk[idx])]
                if len(okk) >= MIN_COINS:
                    val = spearman(sc[okk], fk[okk])
                    if np.isfinite(val):
                        ic[f][k].append(val)
                        if k == 1:
                            ic_year[f][year].append(val)
            # quintiles by score (ascending order; Q5 = highest long-pref)
            order = idx[np.argsort(sc_v, kind="stable")]
            n = len(order)
            m = n // 5
            if m < 1:
                continue
            groups = [order[:m], order[m:2 * m], order[2 * m:3 * m],
                      order[3 * m:4 * m], order[n - m:]]
            for qi, g in enumerate(groups):
                quint_ret[f][qi].append(float(np.mean(f1[g])))
            long_g, short_g = groups[4], groups[0]
            lr = float(np.mean(f1[long_g]))
            sr = float(np.mean(f1[short_g]))
            gross = lr - sr
            spread_gross[f].append(gross)
            leg_long[f].append(lr); leg_short[f].append(sr)
            btc_ret_series[f].append(float(f1[btc_i]) if np.isfinite(f1[btc_i]) else np.nan)
            # weights for turnover/cost
            w = np.zeros(len(COINS))
            w[long_g] = 1.0 / len(long_g)
            w[short_g] = -1.0 / len(short_g)
            cost = float(np.sum(np.abs(w - prev_w[f]) * fee_arr))
            prev_w[f] = w
            spread_net[f].append(gross - cost)
            cost_tot[f] += cost
            gross_tot[f] += abs(gross)

    # ── assemble results + verdict ──────────────────────────────────────────
    results = {}
    for f in factors:
        ics = np.array(ic[f][1])
        n = len(ics)
        mean_ic, std_ic = float(ics.mean()), float(ics.std(ddof=1))
        t_ic = mean_ic / std_ic * np.sqrt(n) if std_ic > 0 else np.nan
        ir = mean_ic / std_ic if std_ic > 0 else np.nan
        ir_ann = ir * np.sqrt(APY)
        decay = {k: {"mean": float(np.mean(ic[f][k])), "median": float(np.median(ic[f][k])),
                     "n": len(ic[f][k])} for k in KS}
        q_means = [float(np.mean(q)) if q else np.nan for q in quint_ret[f]]
        # monotonic increasing Q1..Q5 (long-pref ascending) violations
        viol = sum(1 for a, b in zip(q_means, q_means[1:]) if b < a)
        sg = np.array(spread_gross[f]); sn = np.array(spread_net[f])
        gross_ann = float(sg.mean() * APY * 100)
        gross_shp = float(sg.mean() / sg.std(ddof=1) * np.sqrt(APY)) if sg.std() > 0 else np.nan
        net_ann = float(sn.mean() * APY * 100)
        net_shp = float(sn.mean() / sn.std(ddof=1) * np.sqrt(APY)) if sn.std() > 0 else np.nan
        cost_ratio = float(cost_tot[f] / gross_tot[f]) if gross_tot[f] > 0 else np.nan
        # beta diagnostic (drop NaN btc days)
        br = np.array(btc_ret_series[f]); ll = np.array(leg_long[f]); ls = np.array(leg_short[f])
        msk = np.isfinite(br) & np.isfinite(sg)
        beta = ols_alpha_beta(sg[msk], br[msk])
        beta_long = ols_alpha_beta(ll[msk], br[msk])
        beta_short = ols_alpha_beta(ls[msk], br[msk])
        # year IC
        yr = {str(y): {"mean": float(np.mean(ic_year[f][y])) if ic_year[f][y] else None,
                       "n": len(ic_year[f][y]),
                       "t": (float(np.mean(ic_year[f][y]) / np.std(ic_year[f][y], ddof=1)
                                   * np.sqrt(len(ic_year[f][y])))
                             if len(ic_year[f][y]) > 1 and np.std(ic_year[f][y]) > 0 else None)}
              for y in YEARS}
        # 2025+ stability
        ic_2025 = ic_year[f][2025] + ic_year[f][2026]
        m25 = float(np.mean(ic_2025)) if ic_2025 else np.nan
        t25 = (float(m25 / np.std(ic_2025, ddof=1) * np.sqrt(len(ic_2025)))
               if len(ic_2025) > 1 and np.std(ic_2025) > 0 else np.nan)

        g1 = abs(t_ic) > 2 and ir_ann > 0.3
        g2 = viol <= 1
        g3 = net_ann > 0
        g4 = beta["alpha_t"] > 2
        g5 = np.isfinite(m25) and (np.sign(m25) == np.sign(mean_ic)) and abs(t25) > 1
        all_pass = g1 and g2 and g3 and g4 and g5
        death = None if all_pass else ",".join(
            x for x, ok in [("G1_IC", g1), ("G2_mono", g2), ("G3_netspread", g3),
                            ("G4_alpha", g4), ("G5_stable", g5)] if not ok)
        verif = ((1.96 / net_shp) ** 2 if all_pass and net_shp and net_shp > 0 else None)

        results[f] = {
            "n_rebalance_days": n,
            "ic_mean": mean_ic, "ic_std": std_ic, "ic_t": t_ic,
            "ic_ir_raw": ir, "ic_ir_annual": ir_ann, "ic_pos_share": float((ics > 0).mean()),
            "ic_decay_kbar": decay,
            "quintile_mean_fwd1": q_means, "quintile_monotonic_violations": viol,
            "gross_annual_pct": gross_ann, "gross_sharpe": gross_shp,
            "net_annual_pct": net_ann, "net_sharpe": net_shp,
            "cost_over_gross_ratio": cost_ratio,
            "beta_spread": beta, "beta_long_leg": beta_long, "beta_short_leg": beta_short,
            "ic_by_year": yr, "ic_2025plus_mean": m25, "ic_2025plus_t": t25,
            "gates": {"G1_ic_sig": g1, "G2_monotonic": g2, "G3_net_positive": g3,
                      "G4_true_alpha": g4, "G5_stable_2025": g5, "ALL": all_pass},
            "death_cause": death,
            "verification_years": verif,
        }
        L(f"\n== {f} ==")
        L(f"  IC mean {mean_ic:+.4f} std {std_ic:.4f} t {t_ic:+.2f} IR_ann {ir_ann:+.2f} "
          f"pos {results[f]['ic_pos_share']:.1%} (n={n})")
        L(f"  IC decay k1/3/5 mean: {decay[1]['mean']:+.4f} / {decay[3]['mean']:+.4f} / {decay[5]['mean']:+.4f}")
        L(f"  quintile fwd1 means: {[round(x,5) for x in q_means]} viol={viol}")
        L(f"  gross ann {gross_ann:+.1f}% Sharpe {gross_shp:+.2f} | net ann {net_ann:+.1f}% "
          f"Sharpe {net_shp:+.2f} | cost/gross {cost_ratio:.1%}")
        L(f"  beta spread={beta['beta']:+.2f} alpha_ann={beta['alpha_annual_pct']:+.1f}% "
          f"alpha_t={beta['alpha_t']:+.2f} | long-leg beta {beta_long['beta']:+.2f} "
          f"short-leg beta {beta_short['beta']:+.2f}")
        L(f"  IC by year: " + " ".join(
            f"{y}:{yr[str(y)]['mean']:+.3f}(n{yr[str(y)]['n']})" if yr[str(y)]['mean'] is not None
            else f"{y}:-" for y in YEARS))
        L(f"  2025+ IC mean {m25:+.4f} t {t25:+.2f}")
        L(f"  GATES G1{int(g1)} G2{int(g2)} G3{int(g3)} G4{int(g4)} G5{int(g5)} -> "
          f"{'GO-CANDIDATE' if all_pass else 'FAIL ['+str(death)+']'}")

    survivors = [f for f in factors if results[f]["gates"]["ALL"]]
    L(f"\n== VERDICT: survivors {survivors or 'NONE'} ==")
    final = ("NOT STARTED (cross-sectional closed on survivor sample)" if not survivors
             else f"GO-CANDIDATE: {survivors} -> survivorship sensitivity stage")
    L(f"  FINAL: {final}")

    OUT.mkdir(parents=True, exist_ok=True)
    for f in factors:
        (OUT / f"{f.replace('-', '_').lower()}.json").write_text(
            json.dumps(results[f], indent=2, default=lambda x: None if x is None else float(x)))
    (OUT / "summary.json").write_text(json.dumps({
        "positioning": "descriptive cross-sectional IC pre-study, go/no-go only; "
                       "survivorship NOT corrected (22 current-listed coins); "
                       "no strategy/params/weights",
        "pre_registered": {"factors": {"F-MOM": "ret30 desc", "F-CAR": "funding7 asc",
                           "F-REV": "ret3 asc"}, "rebalance": "daily UTC00", "min_coins": MIN_COINS,
                           "fee_top5_pct": 0.05, "fee_other_pct": 0.07, "top5": sorted(TOP5),
                           "gates": "G1 |t(IC)|>2 & IR_ann>0.3; G2 monotonic(<=1 viol); "
                                    "G3 net>0; G4 alpha_t>2; G5 2025+ sign kept & |t|>1"},
        "n_coins": len(COINS), "n_days_total": D,
        "results": results, "survivors": survivors, "final_verdict": final,
    }, indent=2, default=lambda x: None if x is None else float(x)))
    L(f"run end (UTC): {datetime.now(timezone.utc).isoformat()}")
    (OUT / "run_log.txt").write_text("\n".join(LOG) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
