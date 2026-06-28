#!/usr/bin/env python3
"""B2_4h multiple-testing DEFLATION re-compute — honest Deflated Sharpe + true
verification horizon for the trend line's chosen survivor (B2_4h, 4h EMA20/100).

Source: 趋势研究方法论评价 2026-06-22 (external review, 3 flaws):
  (1) V1->V1' = post-hoc goal-post moving (procedural stain, NOT self-correction);
  (2) selection bias never deflated — 15 configs, best picked, survivor Sharpe
      未扣选择偏差, "15-year" horizon is an OPTIMISTIC (un-deflated + iid) estimate;
  (3) Sharpe is a bad objective for right-skewed (trend) strategies.

This script addresses (2)+(3) with PRE-REGISTERED, ZERO-CHERRY-PICK arithmetic.
Engines research_trend_baseline (tb) / research_trend_validation (tv) /
research_trend_validation_r2 (r2) are imported VERBATIM — zero modification of
any strategy parameter, gate, or the forward system. Read-only mainnet DB.
The contaminated legacy DB is never touched.

==================  PRE-REGISTERED CHOICES (locked BEFORE results)  ==================
All method / parameter / count conventions below are fixed before looking at any
number, use literature-standard values, and are NOT selected to make B2_4h look
better or worse. The only "reverse = error" is the DSR math identity
(SR*0 >= 0 for N>1  =>  deflated Sharpe <= observed Sharpe); everything else
(haircut size, verification horizon, skew picture) is data-adjudicated and open
to a "mild loosening" reading if the data says so.

[A] Return series for Sharpe / moments / bootstrap
    Daily mark-to-market PnL = to_daily(r2.m2m_pnl(cfg_tf, bars, fund, spans_by)),
    bucket = per-bar M2M PnL summed to the UTC 00:00 grid via index.ceil('D')
    (identical convention to the frozen trend_portfolio sleeve_daily_pnl.csv).
    Each config is measured on ITS OWN live window [first-position-day .. last day]
    (no leading warm-up zeros; long-warm-up configs not penalised). Sharpe is
    scale-invariant in a fixed-notional book, so Sharpe = mean(daily$) / std(daily$),
    std ddof=1; annualised = daily Sharpe x sqrt(365)  (crypto 365 trading days;
    this exactly reproduces the frozen portfolio annualised Sharpe).

[B] Trial set / N (three calibres, all reported)
    nominal N   = 15 baseline configs (the pre-registered classic prototypes that
                  the selection that produced B2_4h ran over).
    extended N  = >= 15 + 4 (family D) + V3 neighbourhood variants + the 3
                  post-closure known-target enhancement studies' variants;
                  reported as a LOWER BOUND only (the universe of things tried on
                  this signal cannot be exactly enumerated). NOT used in main DSR.
    effective N = ENB = (sum lambda)^2 / sum(lambda^2) of the 15x15 daily-return
                  Pearson correlation matrix (common-live window) — the same
                  eigenvalue idea the portfolio used for ENB. Trend configs are
                  highly correlated, so effective N << nominal N. DSR MAIN uses
                  effective N (avoid over-penalising correlated trials); nominal &
                  extended shown as sensitivity (monotone, more deflation).

[C] Deflation method = Deflated Sharpe Ratio, Bailey & Lopez de Prado (2014),
    implemented per the paper (no self-invented variant):
      PSR(SR*) = Phi( (SR_hat - SR*) * sqrt(n-1)
                      / sqrt(1 - skew*SR_hat + ((kurt-1)/4)*SR_hat^2) )
      SR*0 = sqrt(V) * [ (1-g)*Z^{-1}(1 - 1/N) + g*Z^{-1}(1 - 1/(N*e)) ]
      DSR  = PSR(SR*0)
    where SR_hat = DAILY Sharpe, skew/kurt = ACTUAL daily-return higher moments
    (NEVER assume normal — trend is right-skewed), n = #days, V = sample variance
    (ddof=1) of the 15 configs' DAILY Sharpes, g = Euler-Mascheroni 0.5772156649,
    Z^{-1} = scipy.stats.norm.ppf, e = exp(1). All quantities in DAILY units.
    Deflated (haircut) Sharpe := SR_hat - SR*0  (reported annualised & daily).

[D] Honest Sharpe SE = stationary bootstrap (Politis & Romano 1994) of the daily
    return series, mean block length L_days = round(median trade hold_hours / 24)
    = round(196/24) = 8 for B2_4h (captures the 196h-median-hold daily
    autocorrelation); B = 10,000; seed = 20260622. iid SE = same with L=1.
    Inflation = SE_block / SE_iid. Sensitivity L in {4, 8, 16}. This is the
    PRE-DEFINED auxiliary; DSR is the PRE-DEFINED main. We do NOT test multiple
    deflation methods and pick one.

[E] Verification horizon (time-to-significance), from the bootstrap SE — NOT the
    iid closed form (which would contradict [D]). With t-stat ~ sqrt(T):
      T*(t0) = T0 * (1.96 / t0)^2 ,  t0 = SR_annual / SE(SR_annual).
    Reported: T*_iid_closed = (1.96/SR_raw_annual)^2  (the ORIGINAL method, on
    B2_4h); T*_iid_boot; T*_block_raw; T*_block_deflated. Multiplicative decomp:
      autocorrelation factor = (SE_block/SE_iid)^2 ; deflation factor =
      (SR_raw/SR_deflated)^2. Anchored to the frozen portfolio "0.5 -> 15.4y".

[F] Right-skew picture (descriptive, NO gate, parallel to Sharpe — flaw 3):
    Sortino_annual = mean/downside_dev x sqrt(365) (target 0); daily skew & excess
    kurtosis; per-trade PnL skew; payoff ratio; win rate; tail-capture (V1'-c
    ratio total_gross / top5%_gross). Shows B2_4h under a ruler that does not
    punish upside volatility.

[G] Sanity asserts (computation-error guards): effective N > 1 ; SR*0 >= 0 ;
    deflated Sharpe <= observed Sharpe (else DSR math identity violated -> bug).
=====================================================================================
"""

from __future__ import annotations

import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import research_trend_baseline as tb            # noqa: E402  (engines imported verbatim)
import research_trend_validation as tv          # noqa: E402
import research_trend_validation_r2 as r2       # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT = PROJECT_ROOT / "reports" / "trend_methodology_hardening_20260622"
PORT_DIR = PROJECT_ROOT / "reports" / "trend_portfolio_20260611"

TARGET = "B2_4h"
ANN = math.sqrt(365.0)          # crypto 365 trading days/yr (reproduces frozen 0.48)
EULER_GAMMA = 0.5772156649015329
SEED = 20260622
N_BOOT = 10_000
Z = 1.959963984540054           # norm.ppf(0.975)

LOG: list[str] = []


def L(msg=""):
    print(msg, flush=True)
    LOG.append(str(msg))


# ── span generation for the 15 baseline configs (mirrors tb.main, zero new logic)
def baseline_spans(cfg: dict, bars: dict) -> dict:
    spans_by = {}
    for name in tb.SYMBOLS:
        b = bars[(name, cfg["tf"])]
        if cfg["kind"] == "donchian":
            spans = tb.positions_donchian(b, cfg["entry_n"], cfg["exit_n"])
        elif cfg["kind"] == "emax":
            spans = tb.positions_flip(tb.signal_emax(b, cfg["fast"], cfg["slow"]))
        else:
            spans = tb.positions_flip(tb.signal_tsmom(b, cfg["days"], cfg["tf"]))
        spans_by[name] = spans
    return spans_by


def to_daily(perbar: pd.Series) -> pd.Series:
    """Identical to trend_portfolio.to_daily — ceil('D') bucket on UTC grid."""
    return perbar.groupby(perbar.index.ceil("D")).sum()


def first_live_day(cfg: dict, bars: dict, spans_by: dict) -> pd.Timestamp:
    """First UTC day on which ANY symbol holds a position (bar-end basis)."""
    first = None
    for name in tb.SYMBOLS:
        b = bars[(name, cfg["tf"])]
        endm = b["end_min"].to_numpy()
        for ei, xi, _side, _r in spans_by[name]:
            t = pd.Timestamp(int(endm[ei + 1 if ei + 1 < len(endm) else ei]) * 60,
                             unit="s", tz="UTC")
            first = t if first is None else min(first, t)
            break  # spans are time-ordered; first span's start is earliest for this symbol
    return first.ceil("D")


# ── Sharpe / moment helpers (daily units) ────────────────────────────────────
def daily_sharpe(daily: np.ndarray) -> float:
    sd = daily.std(ddof=1)
    return float(daily.mean() / sd) if sd > 0 else float("nan")


def sortino_annual(daily: np.ndarray, target: float = 0.0) -> float:
    downside = np.minimum(daily - target, 0.0)
    dd = math.sqrt(np.mean(downside ** 2))
    return float((daily.mean() - target) / dd * ANN) if dd > 0 else float("inf")


# ── Deflated Sharpe Ratio (Bailey & Lopez de Prado 2014) ─────────────────────
def expected_max_sharpe(var_sr_daily: float, n_trials: float) -> float:
    """SR*0 = sqrt(V) [ (1-g) Z^-1(1-1/N) + g Z^-1(1-1/(N e)) ]  (daily units)."""
    a = stats.norm.ppf(1.0 - 1.0 / n_trials)
    b = stats.norm.ppf(1.0 - 1.0 / (n_trials * math.e))
    return math.sqrt(var_sr_daily) * ((1.0 - EULER_GAMMA) * a + EULER_GAMMA * b)


def psr(sr_daily: float, sr_star_daily: float, n_obs: int,
        skew: float, kurt: float) -> float:
    """Probabilistic Sharpe Ratio at benchmark sr_star (non-excess kurt)."""
    denom = math.sqrt(1.0 - skew * sr_daily + ((kurt - 1.0) / 4.0) * sr_daily ** 2)
    return float(stats.norm.cdf((sr_daily - sr_star_daily) * math.sqrt(n_obs - 1) / denom))


# ── stationary bootstrap (Politis & Romano 1994); L=1 => iid ─────────────────
def stationary_boot_sharpe_se(daily: np.ndarray, L: int, B: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    n = len(daily)
    p = 1.0 / L
    idx = np.empty((B, n), dtype=np.int64)
    idx[:, 0] = rng.integers(0, n, size=B)
    new_block = rng.random((B, n)) < p
    new_starts = rng.integers(0, n, size=(B, n))
    for t in range(1, n):
        cont = ~new_block[:, t]
        idx[:, t] = np.where(cont, (idx[:, t - 1] + 1) % n, new_starts[:, t])
    samp = daily[idx]                         # (B, n)
    mu = samp.mean(axis=1)
    sd = samp.std(axis=1, ddof=1)
    sr_ann = np.where(sd > 0, mu / sd * ANN, np.nan)
    sr_ann = sr_ann[np.isfinite(sr_ann)]
    return {"L_days": L, "B": B, "mean_sharpe_ann": float(sr_ann.mean()),
            "se_sharpe_ann": float(sr_ann.std(ddof=1)),
            "ci95_sharpe_ann": [float(np.percentile(sr_ann, 2.5)),
                                float(np.percentile(sr_ann, 97.5))]}


def years_to_sig(sr_annual: float, se_annual: float, t0_years: float) -> float:
    """t-stat ~ sqrt(T); T* such that SR/SE(T*) = 1.96.  inf if effect<=0."""
    if sr_annual <= 0 or se_annual <= 0:
        return float("inf")
    t0 = sr_annual / se_annual
    return float(t0_years * (Z / t0) ** 2)


def main() -> int:
    L(f"run start (UTC): {datetime.now(timezone.utc).isoformat()}")
    L("DATA: database_mainnet.db (mode=ro) | engines tb/tv/r2 imported VERBATIM, "
      "zero strategy/param/gate/forward modification")
    L("contaminated legacy DB: not touched | forward system / config_frozen.json: not touched")
    OUT.mkdir(parents=True, exist_ok=True)

    # ── load data via the frozen baseline engine ─────────────────────────────
    L("\nloading 1m + funding, aggregating 4h/1d via tb engine ...")
    m1, bars, fund = {}, {}, {}
    for name, (db_sym, inst) in tb.SYMBOLS.items():
        m1[name] = tb.load_1m_utc(db_sym)
        bars[(name, "4h")] = tb.aggregate(m1[name], "4h")
        bars[(name, "1d")] = tb.aggregate(m1[name], "1d")
        fund[name] = tb.load_funding(inst, m1[name])
    L(f"  symbols loaded: {list(tb.SYMBOLS)}")

    # ── reconstruct daily M2M for ALL 15 baseline configs ────────────────────
    L("\n== reconstructing daily M2M PnL for 15 baseline configs (frozen engine) ==")
    daily_full, daily_live, live_start, trade_net, n_trades = {}, {}, {}, {}, {}
    skew_d, kurt_d, hold_med = {}, {}, {}
    for cfg in tb.CONFIGS:
        cid = cfg["id"]
        spans_by = baseline_spans(cfg, bars)
        pb, _ = r2.m2m_pnl(cfg["tf"], bars, fund, spans_by)
        d = to_daily(pb)
        daily_full[cid] = d
        ls = first_live_day(cfg, bars, spans_by)
        live_start[cid] = ls
        dlive = d[d.index >= ls]
        daily_live[cid] = dlive
        # trade-account cross-check + per-trade stats
        trades = []
        for name, (_, inst) in tb.SYMBOLS.items():
            trades.extend(tb.build_trades(name, inst, bars[(name, cfg["tf"])],
                                          fund[name], spans_by[name]))
        trade_net[cid] = float(sum(t["net_pnl_usd"] for t in trades))
        n_trades[cid] = len(trades)
        hold_med[cid] = float(np.median([t["hold_hours"] for t in trades]))
        arr = dlive.to_numpy()
        skew_d[cid] = float(stats.skew(arr))
        kurt_d[cid] = float(stats.kurtosis(arr, fisher=False))  # non-excess
        L(f"  [{cid:6s}] m2m=${float(pb.sum()):>10,.0f} trade=${trade_net[cid]:>10,.0f} "
          f"diff=${float(pb.sum()) - trade_net[cid]:>7.2f} | live {ls.date()} "
          f"days={len(dlive):4d} | SR_ann(live)={daily_sharpe(arr) * ANN:5.3f}")

    # ── FIDELITY CHECK: my full-grid reconstruction vs frozen sleeve_daily_pnl.csv
    L("\n== fidelity check vs frozen trend_portfolio sleeve_daily_pnl.csv ==")
    frozen = pd.read_csv(PORT_DIR / "correlation" / "sleeve_daily_pnl.csv",
                         index_col=0, parse_dates=True)
    fidelity = {}
    for sid in ["B1_4h", "B2_4h", "C2_4h", "C2_1d"]:   # D2 is long/flat, not a baseline config
        mine = daily_full[sid].reindex(frozen.index).fillna(0.0)
        diff = float((mine - frozen[sid]).abs().max())
        fidelity[sid] = diff
        L(f"  {sid:6s}: max |mine - frozen| = ${diff:.6f}")
        assert diff < 1e-6, f"FIDELITY FAIL {sid}: {diff}"
    # reproduce frozen P-A annualised Sharpe (0.48) on the full grid as anchor
    pa = ["B1_4h", "B2_4h", "C2_4h", "C2_1d"]
    pa_daily = sum(frozen[s] * 0.25 for s in pa)
    pa_sr_ann = daily_sharpe(pa_daily.to_numpy()) * ANN
    L(f"  P-A full-grid annualised Sharpe = {pa_sr_ann:.4f}  (frozen report: 0.48)")
    assert abs(pa_sr_ann - 0.48) < 0.01, f"P-A Sharpe anchor fail: {pa_sr_ann}"
    L("  FIDELITY PASS — reconstruction is byte-faithful to frozen engine.")

    # ── per-config Sharpes (own live window) → V and effective N ─────────────
    sr_ann = {cid: daily_sharpe(daily_live[cid].to_numpy()) * ANN for cid in daily_live}
    sr_daily = {cid: daily_sharpe(daily_live[cid].to_numpy()) for cid in daily_live}
    sr_ann_arr = np.array([sr_ann[c["id"]] for c in tb.CONFIGS])
    sr_daily_arr = np.array([sr_daily[c["id"]] for c in tb.CONFIGS])
    var_sr_daily = float(sr_daily_arr.var(ddof=1))
    var_sr_ann = float(sr_ann_arr.var(ddof=1))

    L("\n== 15 baseline configs' annualised Sharpe (own live window) ==")
    for c in tb.CONFIGS:
        cid = c["id"]
        L(f"  {cid:6s} n_tr={n_trades[cid]:4d}  SR_ann={sr_ann[cid]:6.3f}  "
          f"net=${trade_net[cid]:>9,.0f}")
    L(f"  mean SR_ann={sr_ann_arr.mean():.3f}  std(ddof1)={sr_ann_arr.std(ddof=1):.3f}  "
      f"Var(SR_daily)={var_sr_daily:.3e}")

    # effective N via ENB of the 15x15 daily corr on the COMMON-live window
    common_live = max(live_start.values())
    L(f"\ncommon-live window start (latest of 15): {common_live.date()} "
      f"(config: {max(live_start, key=live_start.get)})")
    grid = pd.DataFrame({cid: daily_full[cid] for cid in daily_full}).fillna(0.0)
    cw = grid[grid.index >= common_live]
    corr = cw.corr(method="pearson")
    ev = np.linalg.eigvalsh(corr.to_numpy())[::-1]
    ev = np.clip(ev, 0, None)
    enb = float(ev.sum() ** 2 / (ev ** 2).sum())
    L(f"15x15 daily-corr eigenvalues (top6): {[round(float(x),3) for x in ev[:6]]}")
    L(f"effective N (ENB) = {enb:.3f}   nominal N = 15")
    assert enb > 1.0, f"SANITY FAIL: effective N <= 1 ({enb})"

    # extended N lower bound (documented breakdown; NOT used in main DSR)
    ext = {"baseline": 15, "family_D": 4,
           "V3_neighbourhood_variants_x2_per_gated_config": 2 * 9,  # >=9 configs reached V3, x{0.75,1.25}
           "post_closure_funding_confirm": 4,   # E1-E4
           "post_closure_adx_filter": 3,
           "post_closure_faster_entry": 3}
    extended_lb = sum(ext.values())
    L(f"extended N (LOWER BOUND) >= {extended_lb}  breakdown={ext}")

    # ── DSR at three N calibres ──────────────────────────────────────────────
    tgt = TARGET
    sr_hat_d = sr_daily[tgt]
    sr_hat_a = sr_ann[tgt]
    skew_t, kurt_t = skew_d[tgt], kurt_d[tgt]
    n_obs = len(daily_live[tgt])
    L(f"\n== DSR target {tgt}: SR_daily={sr_hat_d:.5f} SR_ann={sr_hat_a:.4f} "
      f"skew={skew_t:.3f} kurt(non-excess)={kurt_t:.2f} n_days={n_obs} ==")

    dsr_rows = {}
    for label, N in [("effective", enb), ("nominal_15", 15.0),
                     ("extended_LB", float(extended_lb))]:
        sr0_d = expected_max_sharpe(var_sr_daily, N)
        assert sr0_d >= 0, f"SANITY FAIL: SR*0 < 0 ({sr0_d}) for N={N}"
        defl_d = sr_hat_d - sr0_d
        assert defl_d <= sr_hat_d + 1e-12, "SANITY FAIL: deflated > observed (DSR identity)"
        dsr_p = psr(sr_hat_d, sr0_d, n_obs, skew_t, kurt_t)
        dsr_rows[label] = {
            "N": float(N), "SR_star0_daily": sr0_d, "SR_star0_ann": sr0_d * ANN,
            "deflated_sharpe_daily": defl_d, "deflated_sharpe_ann": defl_d * ANN,
            "DSR_prob": dsr_p}
        L(f"  N[{label:11s}]={N:6.2f}  SR*0_ann={sr0_d*ANN:6.3f}  "
          f"deflated_ann={defl_d*ANN:6.3f}  DSR_prob={dsr_p:.3f}")
    psr0 = psr(sr_hat_d, 0.0, n_obs, skew_t, kurt_t)
    L(f"  PSR(0) [no deflation, vs SR*=0] = {psr0:.3f}")

    # ── stationary bootstrap SE of annualised Sharpe (target) ────────────────
    L(f"\n== stationary bootstrap Sharpe SE ({tgt}, median hold {hold_med[tgt]:.0f}h) ==")
    arr = daily_live[tgt].to_numpy()
    L_main = int(round(hold_med[tgt] / 24.0))
    boot = {}
    for Lb in sorted({1, 4, L_main, 16}):
        boot[Lb] = stationary_boot_sharpe_se(arr, Lb, N_BOOT, SEED)
        tag = " (iid)" if Lb == 1 else (" (MAIN)" if Lb == L_main else "")
        L(f"  L={Lb:2d}d{tag:7s}: SE(SR_ann)={boot[Lb]['se_sharpe_ann']:.4f}  "
          f"CI95=[{boot[Lb]['ci95_sharpe_ann'][0]:.3f},{boot[Lb]['ci95_sharpe_ann'][1]:.3f}]")
    se_iid = boot[1]["se_sharpe_ann"]
    se_block = boot[L_main]["se_sharpe_ann"]
    inflation = se_block / se_iid
    L(f"  autocorrelation inflation SE_block/SE_iid = {inflation:.3f}")
    # direct confirmation: lag-1..5 autocorrelation of daily M2M returns
    acf = []
    x0 = arr - arr.mean()
    denom = float((x0 * x0).sum())
    for lag in range(1, 6):
        acf.append(float((x0[:-lag] * x0[lag:]).sum() / denom) if denom > 0 else 0.0)
    L(f"  daily-return ACF lag1..5: {[round(a, 3) for a in acf]}  "
      f"(near-0 => block~=iid, autocorrelation correction immaterial)")

    # ── verification horizon ─────────────────────────────────────────────────
    t0_years = (daily_live[tgt].index[-1] - daily_live[tgt].index[0]).days / 365.25
    defl_ann_eff = dsr_rows["effective"]["deflated_sharpe_ann"]
    T_iid_closed = float((Z / sr_hat_a) ** 2) if sr_hat_a > 0 else float("inf")
    T_iid_boot = years_to_sig(sr_hat_a, se_iid, t0_years)
    T_block_raw = years_to_sig(sr_hat_a, se_block, t0_years)
    T_block_defl = years_to_sig(defl_ann_eff, se_block, t0_years)
    # deflated horizon under all three N calibres (block SE, autocorrelation-honest)
    T_block_defl_byN = {lab: years_to_sig(dsr_rows[lab]["deflated_sharpe_ann"], se_block, t0_years)
                        for lab in ("effective", "nominal_15", "extended_LB")}
    autocorr_factor = (se_block / se_iid) ** 2
    deflation_factor = (sr_hat_a / defl_ann_eff) ** 2 if defl_ann_eff > 0 else float("inf")
    L(f"\n== verification horizon (T0={t0_years:.2f}y) ==")
    L(f"  original method anchor: portfolio Sharpe 0.5 -> (1.96/0.5)^2 = {(Z/0.5)**2:.1f}y")
    L(f"  T*_iid_closed  (orig method on {tgt} raw SR {sr_hat_a:.3f}) = {T_iid_closed:.1f}y")
    L(f"  T*_iid_boot    (raw SR, iid boot SE {se_iid:.3f})           = {T_iid_boot:.1f}y")
    L(f"  T*_block_raw   (raw SR, block boot SE {se_block:.3f})       = {T_block_raw:.1f}y")
    for lab in ("effective", "nominal_15", "extended_LB"):
        tv_ = T_block_defl_byN[lab]
        L(f"  T*_block_defl  [N={lab:11s} deflated SR "
          f"{dsr_rows[lab]['deflated_sharpe_ann']:.3f}] = "
          f"{'inf' if math.isinf(tv_) else f'{tv_:.1f}y'}")
    L(f"  decomposition (effective-N): autocorrelation x{autocorr_factor:.2f} ; "
      f"deflation x{'inf' if math.isinf(deflation_factor) else f'{deflation_factor:.2f}'}")

    # ── right-skew picture (descriptive, parallel to Sharpe) ─────────────────
    L(f"\n== right-skew picture ({tgt}) ==")
    cfg_b = json.loads((PROJECT_ROOT / "reports" / "trend_baseline_20260611" /
                        "configs" / f"{tgt}.json").read_text())
    v1p = json.loads((PROJECT_ROOT / "reports" / "trend_validation_r2_20260611" /
                      "gates" / f"{tgt}_v1p.json").read_text())
    # per-trade skew
    tr_net = []
    for name, (_, inst) in tb.SYMBOLS.items():
        for t in tb.build_trades(name, inst, bars[(name, "4h")], fund[name],
                                 baseline_spans(next(c for c in tb.CONFIGS if c["id"] == tgt), bars)[name]):
            tr_net.append(t["net_pnl_usd"])
    tr_net = np.array(tr_net)
    sortino = sortino_annual(arr)
    skew_picture = {
        "sharpe_ann": sr_hat_a, "sortino_ann": sortino,
        "sortino_over_sharpe": sortino / sr_hat_a if sr_hat_a else None,
        "daily_skew": skew_t, "daily_excess_kurtosis": kurt_t - 3.0,
        "trade_net_skew": float(stats.skew(tr_net)),
        "payoff_ratio": cfg_b["portfolio"]["payoff_ratio"],
        "win_rate_pct": cfg_b["portfolio"]["win_rate_pct"],
        "tail_capture_ratio_v1p_c": v1p["c_tail_efficiency"]["ratio"],
    }
    L(f"  Sharpe_ann={sr_hat_a:.3f}  Sortino_ann={sortino:.3f}  "
      f"Sortino/Sharpe={skew_picture['sortino_over_sharpe']:.2f}")
    L(f"  daily skew={skew_t:.2f} excess-kurt={kurt_t-3:.1f}  trade skew="
      f"{skew_picture['trade_net_skew']:.2f}  payoff={skew_picture['payoff_ratio']:.2f}  "
      f"WR={skew_picture['win_rate_pct']:.1f}%  tail-capture={skew_picture['tail_capture_ratio_v1p_c']:.2f}")
    # Sortino vs Sharpe across the 15 (does the ruler change the ranking?)
    sortino_all = {c["id"]: sortino_annual(daily_live[c["id"]].to_numpy()) for c in tb.CONFIGS}

    # ── dump everything ──────────────────────────────────────────────────────
    out = {
        "meta": {"target": tgt, "seed": SEED, "n_boot": N_BOOT, "ann_factor": "sqrt(365)",
                 "t0_years": t0_years, "run_utc": datetime.now(timezone.utc).isoformat()},
        "fidelity_max_abs_diff_usd": fidelity, "P_A_fullgrid_sharpe_ann": pa_sr_ann,
        "configs_15": [
            {"id": c["id"], "n_trades": n_trades[c["id"]], "trade_net": trade_net[c["id"]],
             "live_start": str(live_start[c["id"]].date()), "n_days": len(daily_live[c["id"]]),
             "median_hold_h": hold_med[c["id"]], "sharpe_daily": sr_daily[c["id"]],
             "sharpe_ann": sr_ann[c["id"]], "sortino_ann": sortino_all[c["id"]],
             "daily_skew": skew_d[c["id"]], "daily_kurt_nonexcess": kurt_d[c["id"]]}
            for c in tb.CONFIGS],
        "N_calibres": {"nominal": 15, "effective_ENB": enb,
                       "extended_lower_bound": extended_lb, "extended_breakdown": ext,
                       "corr_eigenvalues": [float(x) for x in ev]},
        "var_sr": {"daily": var_sr_daily, "annual": var_sr_ann,
                   "mean_sharpe_ann": float(sr_ann_arr.mean()),
                   "std_sharpe_ann": float(sr_ann_arr.std(ddof=1))},
        "dsr": {"target_sr_daily": sr_hat_d, "target_sr_ann": sr_hat_a,
                "target_daily_skew": skew_t, "target_daily_kurt_nonexcess": kurt_t,
                "n_obs_days": n_obs, "PSR_vs_zero": psr0, "rows": dsr_rows},
        "bootstrap": {"median_hold_h": hold_med[tgt], "L_main_days": L_main,
                      "by_L": boot, "se_iid": se_iid, "se_block": se_block,
                      "inflation": inflation, "daily_acf_lag1_5": acf},
        "verification_horizon": {
            "t0_years": t0_years, "portfolio_anchor_0p5_to_15y": float((Z / 0.5) ** 2),
            "T_iid_closed_raw": T_iid_closed, "T_iid_boot_raw": T_iid_boot,
            "T_block_raw": T_block_raw, "T_block_deflated_effN": T_block_defl,
            "T_block_deflated_byN": T_block_defl_byN,
            "autocorrelation_factor": autocorr_factor, "deflation_factor": deflation_factor},
        "right_skew": skew_picture,
        "sortino_vs_sharpe_15": {c["id"]: {"sharpe_ann": sr_ann[c["id"]],
                                           "sortino_ann": sortino_all[c["id"]]}
                                 for c in tb.CONFIGS},
    }
    (OUT / "deflated_sharpe.json").write_text(json.dumps(out, indent=2, default=float))
    with open(OUT / "configs_15_sharpe.jsonl", "w") as f:
        for row in out["configs_15"]:
            f.write(json.dumps(row, default=float) + "\n")
    grid.to_csv(OUT / "daily_m2m_15configs_fullgrid.csv")
    L(f"\nwrote {OUT}/deflated_sharpe.json + configs_15_sharpe.jsonl + "
      f"daily_m2m_15configs_fullgrid.csv")
    L(f"run end (UTC): {datetime.now(timezone.utc).isoformat()}")
    (OUT / "run_log.txt").write_text("\n".join(LOG) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
