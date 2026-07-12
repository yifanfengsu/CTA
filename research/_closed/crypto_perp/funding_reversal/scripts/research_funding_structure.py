#!/usr/bin/env python3
"""Pre-study: extreme-funding-rate structure (descriptive only, go/no-go).
No strategy, no backtest, no PnL curve. Answers ONE question: can the cash
flow collected on the funding leg cover the price leg's expected loss plus
trading cost, and if so, is the verification horizon acceptable?

KNOWN HEADWIND (verbatim into report header):
  收费方持仓方向 = 逆拥挤方向，而本市场已证为右偏延续市场——价格腿预期
  为负是默认假设；本研究量化它能否被现金流覆盖，而非假设它不存在。

POSITIONING: 产出立项/不立项判定，不产出策略与参数。

DATA: data/funding/okx/ (formally verified) + database_mainnet.db READ-ONLY
(price leg via settlement prices) + data/binance_vision/ (cross sample,
its OWN rate series defines events). Contaminated DB never touched.
OKX funding truncated at the last settlement with a real 1m settle price
(db ends 2026-05-28) — no stale-price settlements enter the price leg.

PRE-REGISTERED DEFINITIONS (frozen before any run; zero variants):
  Percentile basis: per symbol, rolling 180-day funding-rate distribution,
  window = all settlements strictly BEFORE the current one (no lookahead),
  full 180d span required.
  F1 extreme entry : settlement rate first crosses rolling p95 (positive)
                     or p5 (negative); "first" = previous settlement not in
                     the extreme zone (both settlements need full windows).
  F2 persistent    : rate stays > p90 (or < p10) for >=3 consecutive
                     settlements (24h at 8h cadence); unit = maximal run,
                     event time = 3rd settlement of the run (state
                     confirmed, no lookahead).
  Horizons k ∈ {1, 3, 6, 12, 24} settlements (8h-192h at 8h cadence;
  Binance counts settlements, interval episodes recorded as-is).
  Per event (collector side: positive extreme -> SHORT, negative -> LONG):
   (a) funding leg = actual cumulative collected funding over k settlements
       (realized rates summed with collector sign; adverse flips included),
       % of notional;
   (b) rate reversion = settlements until rate crosses its rolling p50
       (toward neutral), censored at 90 settlements;
   (c) price leg = collector-signed k-settlement price return (settle px to
       settle px), MEDIAN and MEAN both reported (frozen lesson from
       mr_timescale_structure: they tell opposite stories under left skew);
   (d) control = unconditional same-length raw price-return distribution.
  Grid cell = (event type, polarity, k) — polarities NOT merged (bull-side
  and bear-side extremes are structurally different).
  Cluster convention (counting only): same (symbol, type, polarity) events
  within 24 settlements of the previous = one cluster.

PRE-REGISTERED VERDICT LINES (may not be changed after results):
  Full cost = taker both sides 0.10% of notional (no funding cost — the
  structure collects funding). MAIN METRIC = two-leg combined MEAN
  (median reported alongside, never substituted).
  THICKNESS  : total_mean − cost > 0 AND ≥ 0.5 × cost, i.e. total_mean ≥
               0.15%. (0.5× margin vs 2× in the price-MR study: the funding
               leg is a deterministic cash flow; uncertainty concentrates
               in the price leg.)
  CONSISTENCY: ≥3/5 symbols with per-symbol total_mean > 0 at that cell.
  NON-ISOLATED: feasible cell needs ≥1 adjacent-k cell (same type &
               polarity) with total_mean − cost > 0 (net-positive).
  DUAL-SAMPLE: OKX-feasible cells must ALSO pass THICKNESS + CONSISTENCY
               on Binance 2020-2026 events from Binance's own rate series.
               (Binance full grid is computed regardless as descriptive
               context for Q3/2021 — stated upfront, not gate shopping;
               gates apply only to OKX-feasible cells.)
  ARITHMETIC : per qualifying cell μ = total_mean − cost, σ = std(total);
               annual Sharpe ≈ (μ/σ)·√(clusters/year, 5 syms) — coarse,
               cross-symbol correlation NOT discounted (optimistic,
               stated); verification ≈ (1.96/Sharpe)² years; ≤12mo start /
               12-24mo marginal / >24mo NOT started (CLAUDE.md hard rule).
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
# 2026-07 重构批次5：脚本迁入 research/_closed/crypto_perp/funding_reversal/scripts/；共享依赖真身在
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

PROJECT_ROOT = Path(__file__).resolve().parents[5]  # 2026-07 重构批次5：迁入 research/_closed/crypto_perp/funding_reversal/scripts/，深度 1→5
OUT = PROJECT_ROOT / "reports" / "funding_structure_20260612"

KS = (1, 3, 6, 12, 24)
WIN_DAYS = 180
MIN_RUN = 3
REVERT_CAP = 90
COST_PCT = 0.10
THICK_MIN = 1.5 * COST_PCT
MIN_SYMBOLS = 3
CLUSTER_GAP = 24
VERDICT_GO_Y, VERDICT_MARGINAL_Y = 1.0, 2.0

MIN_PER_DAY = 1440
CELLS = [(et, pol) for et in ("F1", "F2") for pol in ("pos", "neg")]
LOG: list[str] = []


def L(msg=""):
    print(msg, flush=True)
    LOG.append(str(msg))


# ── event machinery ──────────────────────────────────────────────────────────
def thresholds(slots: np.ndarray, rates: np.ndarray):
    n = len(slots)
    lo = np.searchsorted(slots, slots - WIN_DAYS * MIN_PER_DAY, side="left")
    valid = (slots - slots[0]) >= WIN_DAYS * MIN_PER_DAY
    th = {p: np.full(n, np.nan) for p in (5, 10, 50, 90, 95)}
    for i in range(n):
        if not valid[i] or lo[i] >= i:
            valid[i] = False
            continue
        q = np.percentile(rates[lo[i]:i], [5, 10, 50, 90, 95])
        for p, v in zip((5, 10, 50, 90, 95), q):
            th[p][i] = v
    return th, valid


def events_f1(rates, th, valid) -> dict[str, np.ndarray]:
    ev = {}
    for pol, z in (("pos", (rates > th[95]) & valid),
                   ("neg", (rates < th[5]) & valid)):
        prev_ok, prev_in = np.roll(valid, 1), np.roll(z, 1)
        prev_ok[0] = prev_in[0] = False
        ev[pol] = np.where(z & prev_ok & ~prev_in)[0]
    return ev


def events_f2(rates, th, valid) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    out = {}
    for pol, z in (("pos", (rates > th[90]) & valid),
                   ("neg", (rates < th[10]) & valid)):
        idx, runs, i, n = [], [], 0, len(z)
        while i < n:
            if z[i]:
                j = i
                while j + 1 < n and z[j + 1]:
                    j += 1
                if j - i + 1 >= MIN_RUN:
                    idx.append(i + MIN_RUN - 1)
                    runs.append(j - i + 1)
                i = j + 1
            else:
                i += 1
        out[pol] = (np.array(idx, dtype=int), np.array(runs, dtype=int))
    return out


def cluster_count(idx: np.ndarray) -> int:
    return 0 if len(idx) == 0 else 1 + int((np.diff(idx) > CLUSTER_GAP).sum())


def revert_counts(rates, th, valid, idx, pol) -> np.ndarray:
    n, out = len(rates), []
    for i in idx:
        c = np.nan
        for j in range(i + 1, min(i + REVERT_CAP + 1, n)):
            if valid[j] and ((pol == "pos" and rates[j] < th[50][j]) or
                             (pol == "neg" and rates[j] > th[50][j])):
                c = j - i
                break
        out.append(c)
    return np.array(out, dtype=float)


def legs(rates, px, idx, pol, k):
    """Per-event arrays (% of notional): funding leg, price leg, total."""
    ii = idx[idx + k < len(rates)]
    if len(ii) == 0:
        return None
    s = 1.0 if pol == "pos" else -1.0   # collector sign on funding flows
    d = -1.0 if pol == "pos" else 1.0   # collector position direction
    fund = np.array([s * rates[i + 1:i + k + 1].sum() for i in ii]) * 100
    price = d * (px[ii + k] - px[ii]) / px[ii] * 100
    return fund, price, fund + price, ii


def leg_summary(fund, price, total) -> dict:
    return {"n": int(len(total)),
            "fund_med": float(np.median(fund)), "fund_mean": float(fund.mean()),
            "price_med": float(np.median(price)), "price_mean": float(price.mean()),
            "total_med": float(np.median(total)), "total_mean": float(total.mean()),
            "total_std": float(total.std(ddof=1)) if len(total) > 1 else None}


def uncond(px, k) -> dict:
    r = (px[k:] - px[:-k]) / px[:-k] * 100
    return {"mean": float(r.mean()), "median": float(np.median(r))}


# ── dataset runner ───────────────────────────────────────────────────────────
def run_dataset(tag: str, funds: dict[str, pd.DataFrame]) -> dict:
    sym = {}
    for s, f in funds.items():
        slots = f["slot_min"].to_numpy()
        rates = f["rate"].to_numpy(dtype=float)
        px = f["settle_px"].to_numpy(dtype=float)
        th, valid = thresholds(slots, rates)
        f2 = events_f2(rates, th, valid)
        sym[s] = dict(slots=slots, rates=rates, px=px, th=th, valid=valid,
                      ev={"F1": events_f1(rates, th, valid),
                          "F2": {p: f2[p][0] for p in f2}},
                      f2runs={p: f2[p][1] for p in f2},
                      span_y=((slots[valid][-1] - slots[valid][0])
                              / (MIN_PER_DAY * 365.25)) if valid.any() else 0.0)
    years = float(np.mean([sym[s]["span_y"] for s in funds]))

    res = {"tag": tag, "span_years": years, "structure": {}, "grid": {}}
    for et, pol in CELLS:
        key = f"{et}_{pol}"
        n_ev = sum(len(sym[s]["ev"][et][pol]) for s in funds)
        n_cl = sum(cluster_count(sym[s]["ev"][et][pol]) for s in funds)
        revs = (np.concatenate([revert_counts(sym[s]["rates"], sym[s]["th"],
                                              sym[s]["valid"], sym[s]["ev"][et][pol],
                                              pol) for s in funds])
                if n_ev else np.array([]))
        fin = revs[~np.isnan(revs)]
        st = {"n_events": n_ev, "n_clusters": n_cl,
              "events_per_year_5sym": n_ev / years,
              "clusters_per_year_5sym": n_cl / years,
              "per_symbol_events": {s: int(len(sym[s]["ev"][et][pol])) for s in funds},
              "revert_to_p50_median": float(np.median(fin)) if len(fin) else None,
              "revert_to_p50_p75": float(np.percentile(fin, 75)) if len(fin) else None,
              "revert_censored_n": int(np.isnan(revs).sum())}
        if et == "F2":
            runs = np.concatenate([sym[s]["f2runs"][pol] for s in funds])
            st["seg_len_median"] = float(np.median(runs)) if len(runs) else None
            st["seg_len_p75"] = float(np.percentile(runs, 75)) if len(runs) else None
        res["structure"][key] = st

        grid_k = {}
        for k in KS:
            per_sym, pf, pp, pt = {}, [], [], []
            for s in funds:
                lg = legs(sym[s]["rates"], sym[s]["px"], sym[s]["ev"][et][pol], pol, k)
                if lg is None:
                    continue
                fund, price, total, _ = lg
                d = leg_summary(fund, price, total)
                d["uncond"] = uncond(sym[s]["px"], k)
                per_sym[s] = d
                pf.append(fund); pp.append(price); pt.append(total)
            if not pt:
                continue
            fund, price, total = map(np.concatenate, (pf, pp, pt))
            pooled = leg_summary(fund, price, total)
            pooled["cost_pct"] = COST_PCT
            pooled["thickness_pass"] = pooled["total_mean"] >= THICK_MIN
            pooled["net_positive"] = pooled["total_mean"] > COST_PCT
            pooled["n_symbols_supporting"] = sum(
                1 for s in per_sym if per_sym[s]["total_mean"] > 0)
            pooled["consistency_pass"] = pooled["n_symbols_supporting"] >= MIN_SYMBOLS
            grid_k[k] = {"pooled": pooled, "per_symbol": per_sym}
        res["grid"][key] = grid_k
        L(f"  [{tag} {key}] ev {n_ev} (cl {n_cl}, {n_cl / years:.0f}/y) | " +
          " | ".join(
              f"k={k}: fund {v['pooled']['fund_mean']:+.3f} "
              f"px(med {v['pooled']['price_med']:+.3f}/mean {v['pooled']['price_mean']:+.3f}) "
              f"tot {v['pooled']['total_mean']:+.3f}"
              f"{' THICK' if v['pooled']['thickness_pass'] else ''}"
              f"({v['pooled']['n_symbols_supporting']}/5)"
              for k, v in grid_k.items()))
    return res


def feasible_cells(res: dict) -> list[dict]:
    out = []
    for et, pol in CELLS:
        key = f"{et}_{pol}"
        g = res["grid"][key]
        for k in KS:
            if k not in g:
                continue
            p = g[k]["pooled"]
            if not (p["thickness_pass"] and p["consistency_pass"]):
                continue
            neigh = [kk for kk in KS if abs(KS.index(kk) - KS.index(k)) == 1]
            non_iso = any(g.get(kk, {}).get("pooled", {}).get("net_positive", False)
                          for kk in neigh)
            out.append({"cell": key, "k": k, "isolated": not non_iso,
                        "total_mean": p["total_mean"], "total_std": p["total_std"],
                        "n_symbols_supporting": p["n_symbols_supporting"]})
    return out


def main() -> int:
    L(f"run start (UTC): {datetime.now(timezone.utc).isoformat()}")
    L("DATA ENVIRONMENT: database_mainnet.db (mode=ro) for settle prices + "
      "data/funding/okx (verified) + data/binance_vision (verified); "
      "contaminated DB not touched; closed verdicts (5m MR, timescale MR, "
      "trend line) not re-tested")
    L("PRE-REGISTERED: F1 p95/p5 first-cross, F2 >=3 settlements in p90/p10; "
      "MAIN METRIC = two-leg combined MEAN; thickness total_mean >= 0.15%; "
      ">=3/5 symbols; non-isolated; Binance dual-sample; 12/24mo lines")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "per_symbol").mkdir(exist_ok=True)

    L("\n== loading OKX mainnet 1m + funding ==")
    funds = {}
    for name, (db_sym, inst) in tb.SYMBOLS.items():
        m1 = tb.load_1m_utc(db_sym)
        f = tb.load_funding(inst, m1)
        last_px_min = int(m1["min_utc"].iloc[-1])
        n0 = len(f)
        f = f[f["slot_min"] <= last_px_min + 1].reset_index(drop=True)
        funds[name] = f
        L(f"  {name}: settlements {len(f):,} (truncated {n0 - len(f)} past db end) "
          f"| rate med {f['rate'].median():.6f} | "
          f"[{pd.Timestamp(int(f['slot_min'].iloc[0]) * 60, unit='s', tz='UTC').date()}"
          f" .. {pd.Timestamp(int(f['slot_min'].iloc[-1]) * 60, unit='s', tz='UTC').date()}]")

    L("\n== OKX grid ==")
    okx = run_dataset("OKX", funds)

    L("\n== loading Binance 1m + funding (full grid = descriptive context, "
      "pre-declared) ==")
    from binance_funding import load_funding_binance
    from research_trend_dualcycle import B_SYM, load_1m_bv
    funds_b = {}
    for name, bs in B_SYM.items():
        m1b = load_1m_bv(bs)
        fb = load_funding_binance(bs, m1b)
        funds_b[name] = fb
        L(f"  {name}: settlements {len(fb):,} | intervals "
          f"{sorted(fb['interval_h'].unique().tolist())}")
    L("\n== Binance grid ==")
    bnc = run_dataset("BNC", funds_b)

    # 2021 slice on Binance (Q3 context): events entered in 2021
    y21 = {}
    t0 = int(pd.Timestamp("2021-01-01", tz="UTC").timestamp() // 60)
    t1 = int(pd.Timestamp("2022-01-01", tz="UTC").timestamp() // 60)
    for et, pol in CELLS:
        key = f"{et}_{pol}"
        for k in KS:
            pf, pp, pt = [], [], []
            for name, fb in funds_b.items():
                slots = fb["slot_min"].to_numpy()
                rates = fb["rate"].to_numpy(dtype=float)
                px = fb["settle_px"].to_numpy(dtype=float)
                th, valid = thresholds(slots, rates)
                ev = (events_f1(rates, th, valid)[pol] if et == "F1"
                      else events_f2(rates, th, valid)[pol][0])
                ev = ev[(slots[ev] >= t0) & (slots[ev] < t1)]
                lg = legs(rates, px, ev, pol, k)
                if lg is None:
                    continue
                fund, price, total, _ = lg
                pf.append(fund); pp.append(price); pt.append(total)
            if pt:
                y21.setdefault(key, {})[k] = leg_summary(
                    *map(np.concatenate, (pf, pp, pt)))
    (OUT / "binance_2021_slice.json").write_text(json.dumps(y21, indent=2))

    # ── verdict chain ────────────────────────────────────────────────────────
    feas = feasible_cells(okx)
    L(f"\n== OKX feasible cells (thickness + consistency): "
      f"{[(c['cell'], c['k'], 'ISO' if c['isolated'] else 'ok') for c in feas] or 'NONE'} ==")
    arith = []
    for c in feas:
        if c["isolated"]:
            arith.append({**c, "verdict": "EXCLUDED (isolated spike)"})
            continue
        bp = bnc["grid"][c["cell"]].get(c["k"], {}).get("pooled")
        b_ok = bool(bp and bp["thickness_pass"] and bp["consistency_pass"])
        if not b_ok:
            arith.append({**c, "binance_pass": False,
                          "verdict": "EXCLUDED (fails Binance dual-sample)"})
            continue
        mu = c["total_mean"] - COST_PCT
        sd = c["total_std"]
        ev_y = okx["structure"][c["cell"]]["clusters_per_year_5sym"]
        sharpe = (mu / sd) * np.sqrt(ev_y) if sd else None
        years = (1.96 / sharpe) ** 2 if sharpe and sharpe > 0 else float("inf")
        verdict = ("ELIGIBLE" if years <= VERDICT_GO_Y else
                   "MARGINAL" if years <= VERDICT_MARGINAL_Y else
                   "NOT STARTED (>24mo)")
        arith.append({**c, "binance_pass": True, "mu_pct": mu, "sigma_pct": sd,
                      "clusters_per_year": ev_y,
                      "sharpe_coarse_optimistic": float(sharpe),
                      "verification_years": float(years), "verdict": verdict})
    for row in arith:
        L(f"  {row}")
    if not arith:
        L("  NONE — no cell reaches the arithmetic")

    # ── persistence ──────────────────────────────────────────────────────────
    for s in funds:
        per = {f"{et}_{pol}": {str(k): okx["grid"][f"{et}_{pol}"][k]["per_symbol"].get(s)
                               for k in KS if k in okx["grid"][f"{et}_{pol}"]}
               for et, pol in CELLS}
        (OUT / "per_symbol" / f"{s}.json").write_text(
            json.dumps(per, indent=2, default=float))
    strip = lambda r: {"tag": r["tag"], "span_years": r["span_years"],
                       "structure": r["structure"],
                       "grid": {key: {str(k): {"pooled": g[k]["pooled"]}
                                      for k in g} for key, g in r["grid"].items()}}
    (OUT / "summary.json").write_text(json.dumps(
        {"positioning": "descriptive pre-study; go/no-go only; no strategy; "
                        "main metric = two-leg combined MEAN",
         "pre_registered": {"F1": "first cross p95/p5 (180d rolling, no lookahead)",
                            "F2": ">=3 settlements beyond p90/p10",
                            "ks": list(KS), "cost_pct": COST_PCT,
                            "thickness_min_total_mean_pct": THICK_MIN,
                            "min_symbols": MIN_SYMBOLS,
                            "verdict_months": [12, 24]},
         "okx": strip(okx), "feasible_cells": feas, "arithmetic": arith},
        indent=2, default=float))
    (OUT / "binance_crosscheck.json").write_text(
        json.dumps(strip(bnc), indent=2, default=float))
    L(f"run end (UTC): {datetime.now(timezone.utc).isoformat()}")
    (OUT / "run_log.txt").write_text("\n".join(LOG) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
