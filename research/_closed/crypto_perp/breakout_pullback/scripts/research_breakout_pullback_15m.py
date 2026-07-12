#!/usr/bin/env python3
"""Pre-study: breakout pullback-continuation structure at 15m (descriptive
only, go/no-go). No strategy, no backtest, no PnL curve.

CANDIDATE MECHANISM (user-specified, 15m fixed): after a breakout, profit
taking causes a pullback; near 50% retracement of the breakout segment, if
the trend is real, price resumes the breakout direction — enter with-trend
there.

PRIOR (verbatim into report header, citing archived studies, not opinion):
  既有双样本研究表明：15m 尺度无条件延续期望 ≈0（成本前），该尺度全成本约
  0.10%/回合（taker 双边主导，funding 在 ≤4h 持有期可忽略）；5m 方向性信号
  已终审（毛利≈0）。本研究检验的是条件化（触及 50% 回撤后）能否将期望抬升
  至成本线之上——该条件期望此前未被测量，为真实未决问题。先验为负。

SURVIVORSHIP STATEMENT (verbatim into header):
  本研究以全部突破事件为分母。"50% 处反弹"的印象可能完全来自事后挑选——
  浅回调的强趋势等不到入场、深回调的失败突破击穿位置继续走。分母完整性
  是本研究的存在理由。

POSITIONING: 产出立项/不立项判定，不产出策略与参数。

DATA: database_mainnet.db READ-ONLY (primary) + data/binance_vision/
(dual-sample, its own 15m events). Contaminated DB never touched.

PRE-REGISTERED DEFINITIONS (frozen before any run; zero variants):
  Scale: 15m only (1m aggregated, UTC boundaries). No other scales.
  BO event: close breaks prior 20-bar high/low (N=20, comparability with
  prior studies; 5h channel at 15m). Same-direction events within <=20 bars
  merged into one cluster; CLUSTER HEAD = anchor event. Denominator = ALL
  cluster heads.
  Anchor A = pre-breakout 20-bar channel midpoint (eh+el)/2 (channel
  excludes the event bar).
  Breakout extreme E = farthest point in breakout direction since the event
  bar (bar highs for up / lows for down), updated bar by bar while the
  event is alive.
  Pullback depth R(t) = (E - close_t)/(E - A), direction-normalized
  (close-based; R can enter the band only via a close inside it).
  INTERPRETIVE RULING (fixed pre-run; uses only pre-registered numbers):
  "越过 E 创新极值 = 延续确认" is armed once max R >= 0.4 (the band's lower
  edge — a material pullback exists); before that, new extremes simply
  extend E (the two clauses "E 逐 bar 更新" and "越过 E 终止" are otherwise
  contradictory). Within a bar: continuation check (vs prior E) precedes
  extension update, which precedes R/failure check.
  Tracking ends at first of:
    (1) continuation confirm (armed new extreme)  -> class "cont"
    (2) R(t) > 1.0 (close beyond anchor)          -> class "fail"
    (3) 60 bars (15h) timeout                     -> class "timeout"
  Band touch (for G-50): first t with R(t) in [0.4, 0.6]; touch can occur
  on the event bar; a pullback that jumps across the band without a close
  inside it never touches (recorded as-is).
  Measurements per event:
    - max R and terminal class (histogram 0.1 buckets, 0..>1.0; cross-tab
      class x bucket);
    - if touched: with-trend signed returns from touch close over
      k in {4, 8, 16} bars (1h/2h/4h), MEAN / MEDIAN both (frozen house
      lesson);
    - B1 control: with-trend k-bar returns from the BREAKOUT bar close,
      ALL cluster heads (the wait-for-pullback opportunity cost must be
      in the denominator);
    - B0 control: unconditional k-bar return distribution.

PRE-REGISTERED VERDICT LINES (may not be changed after results):
  Full cost(k) = taker 0.10% + (k*0.25h/8h) * median|funding| (pooled).
  THICKNESS  : G50_mean - cost > 0 AND >= 0.5*cost (i.e. >= 1.5*cost).
  SUPERIORITY: G50_mean > B1_mean (same k, pooled).
  CONSISTENCY: >=3/5 symbols with per-symbol G50_mean > 0.
  DUAL-SAMPLE: all three lines replicate on Binance 2020-2026 (its own
               events). Binance full grid computed regardless as
               descriptive context — declared upfront, gates apply only
               to OKX-passing cells.
  NON-ISOLATION: the set of passing k's must contain >=2 ADJACENT k's
               (a single isolated k passing alone does not qualify).
  ARITHMETIC : mu = G50_mean - cost; sigma = G50_std; annual events =
               band-touch clusters/year (5 syms, as-is — the 15m event
               base is large; if it passes, weeks-to-months verification
               is the genuine advantage of this scale, stated honestly).
               Coarse Sharpe = (mu/sigma)*sqrt(events/year), cross-symbol
               correlation NOT discounted (optimistic, stated).
               Verification = (1.96/Sharpe)^2 years; <=12mo start /
               12-24mo marginal / >24mo NOT started (CLAUDE.md hard rule).
  DEATH-CAUSE TAXONOMY (Q4, pre-registered): (a) no structure near 50%;
  (b) no continuation edge after touch; (c) gross mean positive but
  thinner than cost (-> notes a maker-execution direction, RECORDED NOT
  ACTED ON); (d) thick but rare. GROSS column is mandatory output.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
# 2026-07 重构批次5：脚本迁入 research/_closed/crypto_perp/breakout_pullback/scripts/；共享依赖真身在
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

PROJECT_ROOT = Path(__file__).resolve().parents[5]  # 2026-07 重构批次5：迁入 research/_closed/crypto_perp/breakout_pullback/scripts/，深度 1→5
OUT = PROJECT_ROOT / "reports" / "breakout_pullback_15m_20260613"

STEP = 15
N_CH = 20
CLUSTER_GAP = 20
BAND_LO, BAND_HI = 0.4, 0.6
TIMEOUT = 60
KS = (4, 8, 16)
FEE_RT_PCT = 0.10
THICK_MULT = 1.5
MIN_SYMBOLS = 3
VERDICT_GO_Y, VERDICT_MARGINAL_Y = 1.0, 2.0

LOG: list[str] = []


def L(msg=""):
    print(msg, flush=True)
    LOG.append(str(msg))


def aggregate_step(df1m: pd.DataFrame, step: int) -> pd.DataFrame:
    g = df1m.groupby(df1m["min_utc"] // step)
    out = pd.DataFrame({"open": g["open"].first(), "high": g["high"].max(),
                        "low": g["low"].min(), "close": g["close"].last()})
    out["start_min"] = out.index.astype("int64") * step
    return out.reset_index(drop=True)


def cluster_heads(idx: np.ndarray) -> np.ndarray:
    if len(idx) == 0:
        return idx
    keep = np.ones(len(idx), dtype=bool)
    keep[1:] = np.diff(idx) > CLUSTER_GAP
    return idx[keep]


def bo_events(b: pd.DataFrame):
    """Cluster-head breakout events: arrays (idx, dir, anchor A)."""
    c = b["close"].to_numpy()
    eh = b["high"].rolling(N_CH).max().shift(1).to_numpy()
    el = b["low"].rolling(N_CH).min().shift(1).to_numpy()
    mid = (eh + el) / 2
    up = cluster_heads(np.where(c > eh)[0])
    dn = cluster_heads(np.where(c < el)[0])
    idx = np.concatenate([up, dn])
    d = np.concatenate([np.ones(len(up)), -np.ones(len(dn))])
    a = mid[idx]
    o = np.argsort(idx, kind="stable")
    return idx[o], d[o], a[o]


def track_event(h, l, c, i, dir_, A, n):
    """Returns (maxR, terminal, touch_t or None). Pre-registered rules."""
    E = h[i] if dir_ > 0 else l[i]
    span = dir_ * (E - A)
    if span <= 0:  # degenerate (cannot happen for true breaks; guard)
        return None
    R = dir_ * (E - c[i]) / span
    maxR = R
    touch = i if BAND_LO <= R <= BAND_HI else None
    if R > 1.0:
        return maxR, "fail", touch
    t_end = min(i + TIMEOUT, n - 1)
    for t in range(i + 1, t_end + 1):
        ext = h[t] if dir_ > 0 else l[t]
        if dir_ * (ext - E) > 0:
            if maxR >= BAND_LO:           # armed: pullback existed
                return maxR, "cont", touch
            E = ext                        # extension, event still alive
            span = dir_ * (E - A)
        R = dir_ * (E - c[t]) / span
        if touch is None and BAND_LO <= R <= BAND_HI:
            touch = t
        if R > maxR:
            maxR = R
        if R > 1.0:
            return maxR, "fail", touch
    return maxR, "timeout", touch


def stats(arr: np.ndarray) -> dict:
    return {"n": int(len(arr)), "mean": float(arr.mean()),
            "median": float(np.median(arr)),
            "std": float(arr.std(ddof=1)) if len(arr) > 1 else None}


def run_dataset(tag: str, bars15: dict[str, pd.DataFrame],
                med_fund_pooled: float) -> dict:
    res = {"tag": tag, "per_symbol": {}, "pooled": {}}
    pooled_R = {}
    pooled_g50 = {k: [] for k in KS}
    pooled_b1 = {k: [] for k in KS}
    pooled_b0 = {k: [] for k in KS}
    pooled_term = []
    years_list = []
    touch_total = 0
    for s, b in bars15.items():
        h, l, c = (b[x].to_numpy() for x in ("high", "low", "close"))
        n = len(c)
        years = (b["start_min"].iloc[-1] - b["start_min"].iloc[0]) / (1440 * 365.25)
        years_list.append(years)
        idx, d, a = bo_events(b)
        maxRs, terms, touches, dirs = [], [], [], []
        for i, dir_, A in zip(idx, d, a):
            r = track_event(h, l, c, int(i), float(dir_), float(A), n)
            if r is None:
                continue
            maxRs.append(r[0]); terms.append(r[1]); touches.append(r[2])
            dirs.append(dir_)
        maxRs = np.array(maxRs)
        hist_edges = [round(0.1 * j, 1) for j in range(11)]
        hist = {f"{hist_edges[j]}-{hist_edges[j+1]}":
                int(((maxRs >= hist_edges[j]) & (maxRs < hist_edges[j + 1])).sum())
                for j in range(10)}
        hist[">1.0"] = int((maxRs >= 1.0).sum())
        cross = {}
        for cls in ("cont", "fail", "timeout"):
            m = np.array([t == cls for t in terms])
            cr = {f"{hist_edges[j]}-{hist_edges[j+1]}":
                  int(((maxRs >= hist_edges[j]) & (maxRs < hist_edges[j + 1]) & m).sum())
                  for j in range(10)}
            cr[">1.0"] = int(((maxRs >= 1.0) & m).sum())
            cross[cls] = cr
        sym_out = {"n_events": int(len(maxRs)),
                   "events_per_year": len(maxRs) / years,
                   "terminal_counts": {cls: int(sum(1 for t in terms if t == cls))
                                       for cls in ("cont", "fail", "timeout")},
                   "maxR_hist": hist, "crosstab": cross,
                   "maxR_median": float(np.median(maxRs)),
                   "touched_n": int(sum(1 for t in touches if t is not None)),
                   "g50": {}, "b1": {}, "b0": {}}
        touch_total += sym_out["touched_n"]
        pooled_R[s] = maxRs
        pooled_term.extend(terms)
        for k in KS:
            ti = np.array([t for t in touches if t is not None and t + k < n])
            td = np.array([dd for t, dd in zip(touches, dirs)
                           if t is not None and t + k < n])
            if len(ti):
                g50 = td * (c[ti + k] - c[ti]) / c[ti] * 100
                sym_out["g50"][k] = stats(g50)
                pooled_g50[k].append(g50)
            ei = idx[idx + k < n]
            ed = d[idx + k < n]
            b1 = ed * (c[ei + k] - c[ei]) / c[ei] * 100
            sym_out["b1"][k] = stats(b1)
            pooled_b1[k].append(b1)
            b0 = (c[k:] - c[:-k]) / c[:-k] * 100
            sym_out["b0"][k] = {"mean": float(b0.mean()),
                                "median": float(np.median(b0))}
            pooled_b0[k].append(b0)
        res["per_symbol"][s] = sym_out
    years = float(np.mean(years_list))
    allR = np.concatenate(list(pooled_R.values()))
    hist_edges = [round(0.1 * j, 1) for j in range(11)]
    p_hist = {f"{hist_edges[j]}-{hist_edges[j+1]}":
              int(((allR >= hist_edges[j]) & (allR < hist_edges[j + 1])).sum())
              for j in range(10)}
    p_hist[">1.0"] = int((allR >= 1.0).sum())
    res["pooled"] = {
        "span_years": years, "n_events": int(len(allR)),
        "events_per_year_5sym": len(allR) / years,
        "touched_n": touch_total, "touch_share": touch_total / len(allR),
        "touch_clusters_per_year_5sym": touch_total / years,
        "terminal_counts": {cls: int(sum(1 for t in pooled_term if t == cls))
                            for cls in ("cont", "fail", "timeout")},
        "maxR_hist": p_hist, "maxR_median": float(np.median(allR)),
        "grid": {}}
    for k in KS:
        g50 = np.concatenate(pooled_g50[k]) if pooled_g50[k] else np.array([])
        b1 = np.concatenate(pooled_b1[k])
        b0 = np.concatenate(pooled_b0[k])
        cost = FEE_RT_PCT + (k * STEP / 60 / 8) * med_fund_pooled * 100
        cell = {"g50": stats(g50) if len(g50) else None,
                "b1": stats(b1),
                "b0": {"mean": float(b0.mean()), "median": float(np.median(b0))},
                "cost_pct": cost,
                "g50_gross_mean": float(g50.mean()) if len(g50) else None}
        if len(g50):
            n_sup = sum(1 for s in bars15
                        if res["per_symbol"][s]["g50"].get(k, {}).get("mean", 0) > 0)
            cell.update({
                "thickness_pass": g50.mean() >= THICK_MULT * cost,
                "superiority_pass": g50.mean() > b1.mean(),
                "n_symbols_supporting": n_sup,
                "consistency_pass": n_sup >= MIN_SYMBOLS,
                "all_pass": (g50.mean() >= THICK_MULT * cost
                             and g50.mean() > b1.mean() and n_sup >= MIN_SYMBOLS)})
        res["pooled"]["grid"][k] = cell
        L(f"  [{tag} k={k}] G50 mean {cell['g50']['mean']:+.4f}% med "
          f"{cell['g50']['median']:+.4f}% (n={cell['g50']['n']}) | "
          f"B1 mean {cell['b1']['mean']:+.4f}% | B0 mean {cell['b0']['mean']:+.4f}% | "
          f"cost {cost:.4f}% | thick {cell.get('thickness_pass')} sup "
          f"{cell.get('n_symbols_supporting')}/5 superior {cell.get('superiority_pass')}")
    return res


def main() -> int:
    L(f"run start (UTC): {datetime.now(timezone.utc).isoformat()}")
    L("DATA ENVIRONMENT: database_mainnet.db (mode=ro) + data/binance_vision; "
      "contaminated DB not touched; closed verdicts not re-tested")
    L("PRE-REGISTERED: N=20 / band [0.4,0.6] / timeout 60 / k={4,8,16} frozen; "
      "denominator = ALL breakout cluster heads; gross column mandatory; "
      "thickness 1.5x cost, superiority vs B1, >=3/5 symbols, dual-sample, "
      ">=2 adjacent k, 12/24mo lines")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "per_symbol").mkdir(exist_ok=True)

    L("\n== loading OKX mainnet 1m + funding ==")
    bars, rates = {}, []
    for name, (db_sym, inst) in tb.SYMBOLS.items():
        m1 = tb.load_1m_utc(db_sym)
        bars[name] = aggregate_step(m1, STEP)
        f = tb.load_funding(inst, m1)
        rates.append(np.abs(f["rate"].to_numpy()))
        L(f"  {name}: 15m bars {len(bars[name]):,}")
    med_fund = float(np.median(np.concatenate(rates)))
    L(f"  pooled median |funding| {med_fund:.6f}/settlement "
      f"-> cost(k=16) = {FEE_RT_PCT + 0.5 * med_fund * 100:.4f}%")

    L("\n== OKX 15m grid ==")
    okx = run_dataset("OKX", bars, med_fund)

    L("\n== loading Binance 1m + funding (full grid descriptive, pre-declared) ==")
    from binance_funding import load_funding_binance
    from research_trend_dualcycle import B_SYM, load_1m_bv
    bars_b, rates_b = {}, []
    for name, bs in B_SYM.items():
        m1b = load_1m_bv(bs)
        bars_b[name] = aggregate_step(m1b, STEP)
        fb = load_funding_binance(bs, m1b)
        rates_b.append(np.abs(fb["rate"].to_numpy()))
        L(f"  {name}: 15m bars {len(bars_b[name]):,}")
    med_fund_b = float(np.median(np.concatenate(rates_b)))
    L("\n== Binance 15m grid ==")
    bnc = run_dataset("BNC", bars_b, med_fund_b)

    # ── verdict chain ────────────────────────────────────────────────────────
    passing = [k for k in KS if okx["pooled"]["grid"][k].get("all_pass")]
    adjacent_ok = any(KS.index(b) - KS.index(a) == 1
                      for a in passing for b in passing)
    arith = []
    for k in passing:
        cell = okx["pooled"]["grid"][k]
        bcell = bnc["pooled"]["grid"][k]
        b_ok = bool(bcell.get("all_pass"))
        row = {"k": k, "okx_pass": True, "binance_pass": b_ok,
               "non_isolated": adjacent_ok}
        if not (b_ok and adjacent_ok):
            row["verdict"] = ("EXCLUDED (fails Binance dual-sample)" if not b_ok
                              else "EXCLUDED (isolated k)")
            arith.append(row)
            continue
        mu = cell["g50"]["mean"] - cell["cost_pct"]
        sd = cell["g50"]["std"]
        ev_y = okx["pooled"]["touch_clusters_per_year_5sym"]
        sharpe = (mu / sd) * np.sqrt(ev_y)
        years = (1.96 / sharpe) ** 2 if sharpe > 0 else float("inf")
        row.update({"mu_pct": mu, "sigma_pct": sd, "touch_per_year": ev_y,
                    "sharpe_coarse_optimistic": float(sharpe),
                    "verification_years": float(years),
                    "verdict": ("ELIGIBLE" if years <= VERDICT_GO_Y else
                                "MARGINAL" if years <= VERDICT_MARGINAL_Y else
                                "NOT STARTED (>24mo)")})
        arith.append(row)
    L(f"\n== OKX passing k: {passing or 'NONE'} | arithmetic: ==")
    for r in arith:
        L(f"  {r}")
    if not arith:
        L("  NONE — no k passes thickness+superiority+consistency on OKX")

    for s in bars:
        (OUT / "per_symbol" / f"{s}.json").write_text(
            json.dumps(okx["per_symbol"][s], indent=2, default=float))
    slim = lambda r: {"tag": r["tag"], "pooled": r["pooled"],
                      "per_symbol_g50_means": {
                          s: {str(k): r["per_symbol"][s]["g50"].get(k, {}).get("mean")
                              for k in KS} for s in r["per_symbol"]}}
    (OUT / "summary.json").write_text(json.dumps(
        {"positioning": "descriptive pre-study, go/no-go only; denominator = "
                        "all breakout cluster heads; gross column mandatory",
         "pre_registered": {"N": N_CH, "band": [BAND_LO, BAND_HI],
                            "timeout_bars": TIMEOUT, "ks": list(KS),
                            "fee_rt_pct": FEE_RT_PCT, "thick_mult": THICK_MULT,
                            "min_symbols": MIN_SYMBOLS, "verdict_months": [12, 24]},
         "okx": slim(okx), "passing_k": passing, "arithmetic": arith},
        indent=2, default=float))
    (OUT / "binance_crosscheck.json").write_text(
        json.dumps(slim(bnc), indent=2, default=float))
    L(f"run end (UTC): {datetime.now(timezone.utc).isoformat()}")
    (OUT / "run_log.txt").write_text("\n".join(LOG) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
