#!/usr/bin/env python3
"""Pre-study: mean-reversion strength vs timescale (15m/30m/1h/2h/4h).
PURELY DESCRIPTIVE — no strategy, no backtest, no PnL curve. Output is a
go/no-go topic-selection verdict under the new CLAUDE.md verification-horizon
rule, plus the feasible-scale interval (if any).

POSITIONING (verbatim into report header):
  本研究产出"立项/不立项"判定与可行尺度区间，不产出策略、不产出参数。
  任何后续策略研究须另行开题，其 gate 另行预注册。
  5m MR 终审判决（毛利≈0，reports/regime/mr5m_mainnet_baseline_20260611/）
  不重测、不挑战；本研究最小尺度 15m。

DATA: .vntrader/database_mainnet.db READ-ONLY (Binance cross-validated PASS);
contaminated legacy DB never touched. Cross-confirmation source (only for
scales that pass thickness): data/binance_vision/ 2020-01..2026-05.

PRE-REGISTERED DEFINITIONS (frozen before any run; no variants, no scans):
  Scales T ∈ {15m, 30m, 1h, 2h, 4h}, 1m aggregated on UTC boundaries.
  E1 channel break : close > prior-20-bar high (up) / < prior-20-bar low
                     (down); channel excludes current bar (shift 1); N=20
                     turtle standard, NOT tunable. Reversion baseline =
                     pre-break channel midpoint (eh+el)/2; reversion
                     direction d = toward midpoint (up-break -> down).
  E2 large move    : single-bar return beyond ±2σ, σ = rolling std of
                     1-bar returns over the past 90 days of bars at that
                     scale (window shifted by 1 bar — no lookahead, full
                     window required). Baseline = pre-event close c[i-1];
                     d = back toward it.
  Horizons k ∈ {1, 2, 4, 8, 16} bars. For each event with full horizon:
    proj_k = (c[i+k] - c[i]) / c[i] * d   (signed % move toward baseline)
    reversion rate = share(proj_k > 0); magnitude = median(proj_k).
  Control baseline per (T, k): unconditional share of strictly-down /
  strictly-up k-bar forward moves over all bars; matched to each event
  set by its direction mix. z-score reported with the stated caveat that
  overlapping events violate iid (informational, not a gate).
  Cluster convention (measurement, not tuning): same-direction events of
  the same type within 20 bars of the previous one form ONE cluster;
  cluster count = conservative n for the topic-selection arithmetic.
  Event tables report both raw and cluster counts; rates/magnitudes use
  all events (simplest, pre-registered).

PRE-REGISTERED FEASIBILITY & VERDICT LINES (may not be changed after results):
  Full cost per (T, k) = taker 0.05%×2 = 0.10%  +  (k·T_hours / 8) ×
    median |funding rate| (per-symbol cost uses own median; pooled cost
    uses the 5-symbol pooled settlement median). Funding charged as a
    cost regardless of sign (cannot be predicted at event level).
  THICKNESS-FEASIBLE cell: pooled median(proj_k) ≥ 2 × full cost (2× =
    safety margin, pre-registered).
  Symbol-consistency rule: a pooled feasible cell only counts if ≥3 of 5
  symbols individually clear the same 2×-own-cost line at that cell
  (a pooled result carried by 1-2 symbols may NOT found a research line).
  Isolated-spike rule: a feasible cell with NO feasible neighbor (adjacent
  T, same k; or adjacent k, same T; same event type) is flagged suspect
  and may NOT found a research line.
  Cross-confirmation: qualifying scales recomputed on Binance 2020-2026;
  both samples must be feasible.
  Topic-selection arithmetic per qualifying cell (coarse, order-of-
  magnitude — stated as such): per-event μ = mean(proj_k) − cost,
  σ = std(proj_k); annual Sharpe ≈ (μ/σ)·sqrt(clusters per year, 5 syms);
  cross-symbol event correlation NOT discounted -> estimate is optimistic,
  i.e. lenient toward 立项 (if it still fails, the negative is robust).
  Verification horizon ≈ (1.96 / Sharpe)² years.
  VERDICT: ≤12 months -> eligible; 12-24 months -> marginal (listed,
  flagged); >24 months -> NOT started (CLAUDE.md hard rule).
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
# 2026-07 重构批次5：脚本迁入 research/_closed/crypto_perp/mr_timescale/scripts/；共享依赖真身在
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

import research_trend_baseline as tb  # loaders reused verbatim, read-only

PROJECT_ROOT = Path(__file__).resolve().parents[5]  # 2026-07 重构批次5：迁入 research/_closed/crypto_perp/mr_timescale/scripts/，深度 1→5
OUT = PROJECT_ROOT / "reports" / "mr_timescale_structure_20260612"

SCALES = {"15m": 15, "30m": 30, "1h": 60, "2h": 120, "4h": 240}
KS = (1, 2, 4, 8, 16)
N_CH = 20            # E1 turtle standard, frozen
SIGMA_MULT = 2.0     # E2, frozen
LOOKBACK_DAYS = 90   # E2 sigma window, frozen
FEE_RT = 0.0010      # taker 0.05% x 2, frozen
SAFETY = 2.0         # thickness margin, frozen
MIN_SYMBOLS = 3      # symbol-consistency rule, frozen
VERDICT_GO_Y, VERDICT_MARGINAL_Y = 1.0, 2.0  # 12 / 24 months, frozen

LOG: list[str] = []


def L(msg=""):
    print(msg, flush=True)
    LOG.append(str(msg))


def aggregate_step(df1m: pd.DataFrame, step: int) -> pd.DataFrame:
    """UTC-boundary buckets of `step` minutes (same convention as tb.aggregate)."""
    g = df1m.groupby(df1m["min_utc"] // step)
    out = pd.DataFrame({
        "open": g["open"].first(), "high": g["high"].max(),
        "low": g["low"].min(), "close": g["close"].last(),
    })
    out["start_min"] = out.index.astype("int64") * step
    return out.reset_index(drop=True)


# ── event extraction ─────────────────────────────────────────────────────────
def events_e1(b: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Returns (idx, d) for channel-break events; d = direction toward baseline."""
    c = b["close"].to_numpy()
    eh = b["high"].rolling(N_CH).max().shift(1).to_numpy()
    el = b["low"].rolling(N_CH).min().shift(1).to_numpy()
    up = np.where(c > eh)[0]   # NaN comparisons are False
    dn = np.where(c < el)[0]
    idx = np.concatenate([up, dn])
    d = np.concatenate([-np.ones(len(up)), np.ones(len(dn))])
    o = np.argsort(idx, kind="stable")
    return idx[o], d[o]


def events_e2(b: pd.DataFrame, step: int) -> tuple[np.ndarray, np.ndarray]:
    """Returns (idx, d) for ±2σ single-bar-move events."""
    r = b["close"].pct_change()
    w = int(LOOKBACK_DAYS * 1440 / step)
    sig = r.rolling(w, min_periods=w).std().shift(1)
    up = np.where((r > SIGMA_MULT * sig).to_numpy())[0]
    dn = np.where((r < -SIGMA_MULT * sig).to_numpy())[0]
    idx = np.concatenate([up, dn])
    d = np.concatenate([-np.ones(len(up)), np.ones(len(dn))])
    o = np.argsort(idx, kind="stable")
    return idx[o], d[o]


def cluster_count(idx: np.ndarray, d: np.ndarray) -> int:
    """Same-direction events within N_CH bars of the previous one = one cluster."""
    n = 0
    for dd in (-1.0, 1.0):
        ii = idx[d == dd]
        if len(ii):
            n += 1 + int((np.diff(ii) > N_CH).sum())
    return n


# ── per-(symbol, scale, type) statistics ─────────────────────────────────────
def uncond_baseline(b: pd.DataFrame, k: int) -> tuple[float, float]:
    """Unconditional P(strict down k-bar fwd move), P(strict up)."""
    c = b["close"].to_numpy()
    fwd = c[k:] - c[:-k]
    return float((fwd < 0).mean()), float((fwd > 0).mean())


def event_stats(b: pd.DataFrame, idx: np.ndarray, d: np.ndarray, k: int) -> dict | None:
    c = b["close"].to_numpy()
    m = idx + k < len(c)
    ii, dd = idx[m], d[m]
    if len(ii) == 0:
        return None
    proj = (c[ii + k] - c[ii]) / c[ii] * dd
    p_dn, p_up = uncond_baseline(b, k)
    n_dn, n_up = int((dd == -1).sum()), int((dd == 1).sum())
    base = (n_dn * p_dn + n_up * p_up) / len(ii)
    rate = float((proj > 0).mean())
    z = (rate - base) / np.sqrt(base * (1 - base) / len(ii)) if 0 < base < 1 else None
    return {"n": int(len(ii)), "n_dir": {"down_rev": n_dn, "up_rev": n_up},
            "rate": rate, "baseline": float(base),
            "rate_minus_baseline": float(rate - base),
            "z_iid_caveat": float(z) if z is not None else None,
            "median_proj_pct": float(np.median(proj) * 100),
            "mean_proj_pct": float(proj.mean() * 100),
            "std_proj_pct": float(proj.std(ddof=1) * 100),
            "_proj": proj, "_dirs": (n_dn, n_up), "_pdnup": (p_dn, p_up)}


def cost_pct(step: int, k: int, med_abs_rate: float) -> float:
    return (FEE_RT + (k * step / 60.0 / 8.0) * med_abs_rate) * 100


# ── dataset runner (shared by OKX and Binance) ───────────────────────────────
def run_dataset(tag: str, m1: dict[str, pd.DataFrame], med_fund: dict[str, float],
                med_fund_pooled: float, scales: dict[str, int]) -> dict:
    """Full descriptive grid for one dataset. Returns nested result dict."""
    res = {"tag": tag, "scales": {}}
    for sc, step in scales.items():
        bars = {s: aggregate_step(m1[s], step) for s in m1}
        span_years = {s: (m1[s]["min_utc"].iloc[-1] - m1[s]["min_utc"].iloc[0])
                      / (60 * 24 * 365.25) for s in m1}
        sc_out = {}
        for etype in ("E1", "E2"):
            ev = {s: (events_e1(bars[s]) if etype == "E1"
                      else events_e2(bars[s], step)) for s in m1}
            per_sym, pooled_k = {}, {}
            n_raw = sum(len(ev[s][0]) for s in m1)
            n_clu = sum(cluster_count(*ev[s]) for s in m1)
            for s in m1:
                idx, d = ev[s]
                per_sym[s] = {"n_events": int(len(idx)),
                              "n_clusters": cluster_count(idx, d),
                              "per_k": {}}
                for k in KS:
                    st = event_stats(bars[s], idx, d, k)
                    if st is None:
                        continue
                    cst = cost_pct(step, k, med_fund[s])
                    st["cost_pct"] = cst
                    st["feasible_2x_own_cost"] = st["median_proj_pct"] >= SAFETY * cst
                    per_sym[s]["per_k"][k] = st
            for k in KS:
                projs = [per_sym[s]["per_k"][k]["_proj"] for s in m1
                         if k in per_sym[s]["per_k"]]
                if not projs:
                    continue
                proj = np.concatenate(projs)
                num = den = 0.0
                for s in m1:
                    if k not in per_sym[s]["per_k"]:
                        continue
                    st = per_sym[s]["per_k"][k]
                    (n_dn, n_up), (p_dn, p_up) = st["_dirs"], st["_pdnup"]
                    num += n_dn * p_dn + n_up * p_up
                    den += n_dn + n_up
                base = num / den
                rate = float((proj > 0).mean())
                cst = cost_pct(step, k, med_fund_pooled)
                med = float(np.median(proj) * 100)
                n_support = sum(1 for s in m1
                                if k in per_sym[s]["per_k"]
                                and per_sym[s]["per_k"][k]["feasible_2x_own_cost"])
                pooled_k[k] = {
                    "n": int(len(proj)), "rate": rate, "baseline": float(base),
                    "rate_minus_baseline": float(rate - base),
                    "z_iid_caveat": float((rate - base) /
                                          np.sqrt(base * (1 - base) / len(proj))),
                    "median_proj_pct": med,
                    "mean_proj_pct": float(proj.mean() * 100),
                    "std_proj_pct": float(proj.std(ddof=1) * 100),
                    "cost_pct": cst, "threshold_2x_pct": SAFETY * cst,
                    "feasible_pooled": med >= SAFETY * cst,
                    "n_symbols_supporting": n_support,
                    "feasible_with_support": (med >= SAFETY * cst
                                              and n_support >= MIN_SYMBOLS),
                }
            for s in m1:  # strip numpy intermediates before JSON
                for k in list(per_sym[s]["per_k"]):
                    for key in ("_proj", "_dirs", "_pdnup"):
                        per_sym[s]["per_k"][k].pop(key, None)
            yrs = float(np.mean(list(span_years.values())))
            sc_out[etype] = {"n_events_raw": n_raw, "n_clusters": n_clu,
                             "clusters_per_year_5sym": n_clu / yrs,
                             "span_years": yrs,
                             "pooled": pooled_k, "per_symbol": per_sym}
            L(f"  [{tag} {sc} {etype}] events {n_raw:,} (clusters {n_clu:,}) | " +
              " | ".join(f"k={k}: rate {v['rate']:.3f} (base {v['baseline']:.3f}) "
                         f"med {v['median_proj_pct']:+.3f}% thr {v['threshold_2x_pct']:.3f}%"
                         f"{' FEAS' if v['feasible_pooled'] else ''}"
                         f"{'+SUP' if v['feasible_with_support'] else ''}"
                         for k, v in pooled_k.items()))
        res["scales"][sc] = sc_out
    return res


def isolated_flags(res: dict) -> list[dict]:
    """Apply the isolated-spike rule over the (scale, k) grid per event type."""
    order = list(SCALES)
    out = []
    for etype in ("E1", "E2"):
        feas = {(sc, k): res["scales"][sc][etype]["pooled"].get(k, {})
                .get("feasible_with_support", False)
                for sc in order for k in KS}
        for (sc, k), ok in feas.items():
            if not ok:
                continue
            si, ki = order.index(sc), KS.index(k)
            neigh = []
            if si > 0:
                neigh.append((order[si - 1], k))
            if si < len(order) - 1:
                neigh.append((order[si + 1], k))
            if ki > 0:
                neigh.append((sc, KS[ki - 1]))
            if ki < len(KS) - 1:
                neigh.append((sc, KS[ki + 1]))
            isolated = not any(feas.get(nb, False) for nb in neigh)
            out.append({"type": etype, "scale": sc, "k": k, "isolated": isolated})
    return out


def arithmetic(res: dict, cells: list[dict]) -> list[dict]:
    """Topic-selection arithmetic for qualifying (non-isolated, supported) cells."""
    rows = []
    for cell in cells:
        if cell["isolated"]:
            rows.append({**cell, "verdict": "EXCLUDED (isolated spike)"})
            continue
        sc, k, etype = cell["scale"], cell["k"], cell["type"]
        node = res["scales"][sc][etype]
        p = node["pooled"][k]
        mu = (p["mean_proj_pct"] - p["cost_pct"]) / 100
        sd = p["std_proj_pct"] / 100
        ev_y = node["clusters_per_year_5sym"]
        sharpe = (mu / sd) * np.sqrt(ev_y) if sd > 0 else None
        years = (1.96 / sharpe) ** 2 if sharpe and sharpe > 0 else float("inf")
        verdict = ("ELIGIBLE" if years <= VERDICT_GO_Y else
                   "MARGINAL" if years <= VERDICT_MARGINAL_Y else
                   "NOT STARTED (>24mo)")
        rows.append({**cell, "mean_minus_cost_pct": mu * 100,
                     "std_pct": sd * 100, "clusters_per_year": ev_y,
                     "sharpe_coarse_optimistic": sharpe,
                     "verification_years": years, "verdict": verdict})
    return rows


def main() -> int:
    L(f"run start (UTC): {datetime.now(timezone.utc).isoformat()}")
    L("DATA ENVIRONMENT: .vntrader/database_mainnet.db (mode=ro, cross-validated "
      "PASS); contaminated legacy DB not touched; 5m NOT re-tested (closed verdict)")
    L("PRE-REGISTERED: E1 N=20 / E2 ±2σ(90d) frozen; thickness = median ≥ 2×cost; "
      "support ≥3/5 symbols; isolated spikes excluded; verdict lines 12/24 months")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "per_symbol").mkdir(exist_ok=True)

    L("\n== loading OKX mainnet 1m + funding ==")
    m1, med_fund, all_rates = {}, {}, []
    for name, (db_sym, inst) in tb.SYMBOLS.items():
        m1[name] = tb.load_1m_utc(db_sym)
        f = tb.load_funding(inst, m1[name])
        med_fund[name] = float(np.median(np.abs(f["rate"])))
        all_rates.append(np.abs(f["rate"].to_numpy()))
        L(f"  {name}: 1m {len(m1[name]):,} | median |funding| {med_fund[name]:.6f}")
    med_pooled = float(np.median(np.concatenate(all_rates)))
    L(f"  pooled median |funding| = {med_pooled:.6f} per 8h settlement")
    (OUT / "funding_medians.json").write_text(json.dumps(
        {"per_symbol": med_fund, "pooled": med_pooled}, indent=2))

    L("\n== OKX descriptive grid ==")
    res = run_dataset("OKX", m1, med_fund, med_pooled, SCALES)

    flags = isolated_flags(res)
    arith = arithmetic(res, flags)

    # cross-confirmation only for scales hosting qualifying cells
    qual_scales = sorted({c["scale"] for c in arith
                          if not c.get("isolated", True)})
    bnc = None
    if qual_scales:
        L(f"\n== Binance cross-confirmation for scales {qual_scales} ==")
        from binance_funding import load_funding_binance
        from research_trend_dualcycle import B_SYM, load_1m_bv
        m1b, med_fb, ratesb = {}, {}, []
        for name, bs in B_SYM.items():
            m1b[name] = load_1m_bv(bs)
            fb = load_funding_binance(bs, m1b[name])
            med_fb[name] = float(np.median(np.abs(fb["rate"])))
            ratesb.append(np.abs(fb["rate"].to_numpy()))
            L(f"  {name}: 1m {len(m1b[name]):,} | median |funding| {med_fb[name]:.6f}")
        med_pb = float(np.median(np.concatenate(ratesb)))
        bnc = run_dataset("BNC", m1b, med_fb, med_pb,
                          {sc: SCALES[sc] for sc in qual_scales})
        (OUT / "binance_crosscheck.json").write_text(
            json.dumps(bnc, indent=2, default=float))
    else:
        L("\n== no qualifying cell on OKX -> Binance cross-confirmation skipped "
          "per pre-registration ==")

    for s in m1:
        sym_out = {sc: {et: {"n_events": res["scales"][sc][et]["per_symbol"][s]["n_events"],
                             "n_clusters": res["scales"][sc][et]["per_symbol"][s]["n_clusters"],
                             "per_k": res["scales"][sc][et]["per_symbol"][s]["per_k"]}
                        for et in ("E1", "E2")} for sc in SCALES}
        (OUT / "per_symbol" / f"{s}.json").write_text(
            json.dumps(sym_out, indent=2, default=float))

    summary = {"positioning": "descriptive pre-study; go/no-go verdict only; "
                              "no strategy, no params; 5m not re-tested",
               "pre_registered": {"E1_N": N_CH, "E2_sigma": SIGMA_MULT,
                                  "sigma_lookback_days": LOOKBACK_DAYS,
                                  "fee_roundtrip": FEE_RT, "safety_mult": SAFETY,
                                  "min_symbols": MIN_SYMBOLS,
                                  "verdict_months": [12, 24]},
               "okx": {sc: {et: {"n_events_raw": res["scales"][sc][et]["n_events_raw"],
                                 "n_clusters": res["scales"][sc][et]["n_clusters"],
                                 "clusters_per_year_5sym":
                                     res["scales"][sc][et]["clusters_per_year_5sym"],
                                 "pooled": res["scales"][sc][et]["pooled"]}
                            for et in ("E1", "E2")} for sc in SCALES},
               "feasible_cells": flags,
               "arithmetic": arith,
               "binance_crosscheck_ran": bool(qual_scales)}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=float))

    L("\n== verdict cells ==")
    if not arith:
        L("  NONE — no (scale, k) cell passes thickness + symbol-support")
    for row in arith:
        L(f"  {row}")
    L(f"run end (UTC): {datetime.now(timezone.utc).isoformat()}")
    (OUT / "run_log.txt").write_text("\n".join(LOG) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
