#!/usr/bin/env python3
"""Pre-study: volatility-event structure (descriptive only, go/no-go).
No strategy, no backtest, no PnL curve.

NON-DIRECTIONAL PROPOSITION (user-specified): the first five studies closed
every <=4h DIRECTIONAL signal for retail takers (5m final, 15m-4h MR, funding
harvest, breakout pullback — all NOT STARTED). This study tests a non-directional
claim: is VOLATILITY itself ("how much it moves", not "which way") predictable,
and can it be MONETIZED with perpetuals?

PHYSICAL CONSTRAINT (verbatim into report header): this project has NO options
data / NO options trading (implied vol unavailable). We therefore CANNOT test the
vol-premium SELL side. This study can only test "long-vol synthesised from perps"
— i.e. the predictability of vol EXPANSION after a low-vol state and whether that
expansion can be monetised.

CORE TRAP (verbatim into header): capturing a vol burst with perps forces a
breakout/chase entry, which degenerates into a DIRECTIONAL strategy — and the
directional band is already proven empty. Worse, perpetuals are LINEAR: a static
long+short straddle nets to exactly zero gross (no convexity/gamma without
options). The design therefore SEPARATES "vol is predictable" (Part 1, pure
statistics) from "vol is monetisable" (Part 2, direction-stripped). Either fails
=> NOT STARTED.

POSITIONING: produces a go/no-go verdict, not a strategy or parameters.

DATA: database_mainnet.db READ-ONLY (primary) + data/binance_vision/ (dual-sample,
its own low-vol events). Contaminated DB never touched.

================================ PRE-REGISTERED (frozen; zero variants) ============
SCALE: 1h ONLY (1m aggregated, UTC boundaries). Rationale: <1h repeatedly proven
  cost-dominated; >4h events too sparse; 1h is the "low-vol clustering measurable +
  enough events" compromise. NO multi-scale scan.

VOLATILITY MEASURE (zero variants):
  r(t)  = ln(close_t / close_{t-1})            (1h log return)
  RV(t) = std (ddof=1) of the past 24 1h log returns (24h realised vol)
  Rolling 180d distribution = past 4320 1h RV values, window SHIFTED by 1 bar
  (no lookahead, FULL window required, min_periods=4320). p20/p50/p80 thresholds.
  Low-vol state  = RV < p20 threshold.   High-vol state = RV > p80 threshold.
  ENTRY event    = state True AND previous bar state False (transition; the
  denominator is ALL low-vol entries — no event hand-picking).

PART 1 — PREDICTABILITY (pure description; NO direction, NO cost):
  For each low-vol entry at t, k in {6,12,24,48} hours after:
    - RV(t+k) distribution (mean/median) vs the symbol's unconditional RV;
    - P(high vol within (t,t+k]) = P(any bar's RV > its own p80 thr) vs the
      unconditional baseline P(high vol within any k-window).
  Reverse control (full picture, NOT a gate): high-vol entries -> P(RV falls
  below p50 thr within k) and RV(t+k).
  PART-1 PASS LINE (frozen): directional volatility clustering present =
  on BOTH samples AND >=3/5 symbols: mean RV(t+6) after low-vol < unconditional
  mean RV  AND  mean RV(t+6) after high-vol > unconditional mean RV.
  (Clustering is among the most robust facts in finance; expect PASS. If even this
  fails -> terminate, no physical base.)  STRATEGY-RELEVANT SIGN, reported for Q1
  honesty: dP = P(high|low) - P(high|uncond). If dP <= 0, clustering says CALM
  PERSISTS -> low vol does NOT time an imminent expansion (foreshadows Part 2).

PART 2 — MONETISABILITY (direction-stripped; this is the life/death gate):
  Perps are linear: a single-entry/single-exit long+short straddle nets ~0 gross
  (verified numerically). The only perp long-vol proxy is a breakout-entry straddle
  that gives up entry slippage. Two pre-registered direction-NEUTRAL capture
  measures per low-vol entry over (t,t+k], P0 = close[t]:
    CAP_net = |close[t+k]/P0 - 1|          (NET absolute displacement; realistic
              "max(...) - slippage": you exit at the close, not the extreme.) GATE.
    CAP_exc = max( (maxhigh-P0)/P0 , (P0-minlow)/P0 )   (no-slippage CEILING;
              optimistic upper bound, reported not gated.)
  Unconditional control: same CAP_net at ALL valid (random) t.
  DOUBLE-SIDED COST (frozen): synthetic straddle = 2 round trips = taker 4 legs
  * 0.05% = 0.20% of notional. (Far above a single-leg directional策略 — faced head on.)
  MONETISATION GATES (frozen, may not change after results):
    THICKNESS  : CAP_net mean - 0.20% > 0 AND >= 0.5*cost (i.e. CAP_net mean >= 0.30%).
    INCREMENT  : CAP_net mean (post-low-vol) > unconditional CAP_net mean
                 (else "waiting for low vol" carries no information).
    CONSISTENCY: >=3/5 symbols with per-symbol (CAP_net mean - cost) > 0.
    DUAL-SAMPLE: all gates replicate on Binance 2020-2026 (its own low-vol events).
    mean/median both reported (right-tail gamble vs typical profitability).

PART 3 — ARITHMETIC (only if Part 1 AND Part 2 pass):
  For monetisation-passing cells: mu = CAP_net mean - 0.20%; sigma = CAP_net std;
  annual low-vol entries (5 syms, as-is); coarse Sharpe = (mu/sigma)*sqrt(events/yr),
  cross-symbol correlation NOT discounted (optimistic, stated); verification =
  (1.96/Sharpe)^2 years; <=12mo start / 12-24mo marginal / >24mo NOT STARTED.

DEATH-CAUSE TAXONOMY (Q4, pre-registered):
  a) vol unpredictable (extremely unlikely);
  b) predictable but perp-synthesis cost (0.20% double-sided) eats the capturable
     magnitude (MOST LIKELY -> the precise statement of "vol edge is real but
     belongs to options players, perps cannot reach it");
  c) thick but events too rare to verify.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

sys.path.insert(0, str(Path(__file__).resolve().parent))
# 2026-07 重构批次5：脚本迁入 research/_closed/crypto_perp/volatility_event/scripts/；共享依赖真身在
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

PROJECT_ROOT = Path(__file__).resolve().parents[5]  # 2026-07 重构批次5：迁入 research/_closed/crypto_perp/volatility_event/scripts/，深度 1→5
OUT = PROJECT_ROOT / "reports" / "volatility_event_20260613"

# ── frozen pre-registered constants (zero variants) ──────────────────────────
STEP = 60                 # 1h bars from 1m
RV_WIN = 24               # 24h realised vol window
ROLL_WIN = 4320           # 180 days of 1h bars
P_LO, P_MID, P_HI = 0.20, 0.50, 0.80
KS = (6, 12, 24, 48)      # hours after the entry event
COST_PCT = 0.20           # taker 4 legs * 0.05% (double-sided synthetic straddle)
THICK_NET_PCT = 0.30      # CAP_net mean >= cost + 0.5*cost
MIN_SYMBOLS = 3
VERDICT_GO_Y, VERDICT_MARGINAL_Y = 1.0, 2.0

LOG: list[str] = []


def L(msg=""):
    print(msg, flush=True)
    LOG.append(str(msg))


def aggregate_1h(df1m: pd.DataFrame) -> pd.DataFrame:
    g = df1m.groupby(df1m["min_utc"] // STEP)
    out = pd.DataFrame({"open": g["open"].first(), "high": g["high"].max(),
                        "low": g["low"].min(), "close": g["close"].last()})
    out["start_min"] = out.index.astype("int64") * STEP
    return out.reset_index(drop=True)


def forward_any(mask: np.ndarray, k: int) -> np.ndarray:
    """out[t] = mask[t+1 .. t+k].any() ; out[t]=False where t+k >= n."""
    n = len(mask)
    out = np.zeros(n, dtype=bool)
    if n > k:
        w = sliding_window_view(mask, k)          # w[i] = mask[i:i+k]
        wany = w.any(axis=1)                       # length n-k+1
        m = n - k - 1                              # last t with t+k < n
        out[: m + 1] = wany[1: m + 2]
    return out


def stats(arr: np.ndarray) -> dict | None:
    if len(arr) == 0:
        return None
    return {"n": int(len(arr)), "mean": float(arr.mean()),
            "median": float(np.median(arr)),
            "std": float(arr.std(ddof=1)) if len(arr) > 1 else None}


def compute_symbol(b: pd.DataFrame) -> dict:
    c = b["close"].to_numpy(dtype=float)
    h = b["high"].to_numpy(dtype=float)
    lo = b["low"].to_numpy(dtype=float)
    n = len(c)
    span_years = (b["start_min"].iloc[-1] - b["start_min"].iloc[0]) / (60 * 24 * 365.25)

    r = np.concatenate([[np.nan], np.log(c[1:] / c[:-1])])
    rv = pd.Series(r).rolling(RV_WIN).std(ddof=1).to_numpy()
    rv_s = pd.Series(rv)
    p20 = rv_s.rolling(ROLL_WIN, min_periods=ROLL_WIN).quantile(P_LO).shift(1).to_numpy()
    p50 = rv_s.rolling(ROLL_WIN, min_periods=ROLL_WIN).quantile(P_MID).shift(1).to_numpy()
    p80 = rv_s.rolling(ROLL_WIN, min_periods=ROLL_WIN).quantile(P_HI).shift(1).to_numpy()

    valid = ~np.isnan(rv) & ~np.isnan(p20) & ~np.isnan(p80)
    low_state = valid & (rv < p20)
    high_state = valid & (rv > p80)
    above80 = np.where(valid, rv > p80, False)
    below50 = np.where(valid & ~np.isnan(p50), rv < p50, False)

    prev_low = np.concatenate([[False], low_state[:-1]])
    prev_high = np.concatenate([[False], high_state[:-1]])
    low_ev = np.where(low_state & ~prev_low)[0]
    high_ev = np.where(high_state & ~prev_high)[0]

    uncond_rv = rv[valid]
    out = {
        "span_years": float(span_years), "n_1h_bars": int(n),
        "n_low_entries": int(len(low_ev)), "n_high_entries": int(len(high_ev)),
        "low_entries_per_year": float(len(low_ev) / span_years),
        "uncond_rv_mean": float(uncond_rv.mean()),
        "uncond_rv_median": float(np.median(uncond_rv)),
        "part1": {}, "part2": {},
        # raw arrays for pooling
        "_low_ev": low_ev, "_high_ev": high_ev, "_valid": valid,
        "_rv": rv, "_above80": above80, "_below50": below50, "_c": c,
        "_h": h, "_lo": lo, "_uncond_rv": uncond_rv, "_span_years": span_years,
    }

    for k in KS:
        # ---- Part 1: predictability ----
        f_hi = forward_any(above80, k)
        f_rev = forward_any(below50, k)
        base_valid = valid.copy()
        base_valid[n - k:] = False                      # need t+k in range
        rv_kp = np.full(n, np.nan)
        rv_kp[: n - k] = rv[k:]

        lev = low_ev[low_ev + k < n]
        hev = high_ev[high_ev + k < n]
        cond_low_rv = rv_kp[lev]
        cond_high_rv = rv_kp[hev]
        p_hi_low = float(f_hi[lev].mean()) if len(lev) else None
        p_hi_base = float(f_hi[base_valid].mean()) if base_valid.any() else None
        p_rev_high = float(f_rev[hev].mean()) if len(hev) else None

        out["part1"][k] = {
            "low_rv_kplus": stats(cond_low_rv[~np.isnan(cond_low_rv)]),
            "high_rv_kplus": stats(cond_high_rv[~np.isnan(cond_high_rv)]),
            "p_highvol_within_given_low": p_hi_low,
            "p_highvol_within_uncond": p_hi_base,
            "dP_high_lift": (None if p_hi_low is None or p_hi_base is None
                             else p_hi_low - p_hi_base),
            "p_revert_below_p50_given_high": p_rev_high,
        }

        # ---- Part 2: monetisability (direction-neutral) ----
        ev = lev
        cap_net = np.abs(c[ev + k] / c[ev] - 1.0) * 100
        U = np.array([(h[i + 1:i + k + 1].max() - c[i]) / c[i] for i in ev]) * 100
        D = np.array([(c[i] - lo[i + 1:i + k + 1].min()) / c[i] for i in ev]) * 100
        cap_exc = np.maximum(U, D)
        # unconditional CAP_net at all valid random t
        bv = np.where(base_valid)[0]
        unc_net = np.abs(c[bv + k] / c[bv] - 1.0) * 100
        out["part2"][k] = {
            "cap_net": stats(cap_net), "cap_exc": stats(cap_exc),
            "uncond_cap_net": stats(unc_net),
            "cost_pct": COST_PCT,
            "net_after_cost_mean": (float(cap_net.mean() - COST_PCT)
                                    if len(cap_net) else None),
            "increment_vs_uncond": (float(cap_net.mean() - unc_net.mean())
                                    if len(cap_net) and len(unc_net) else None),
            "_cap_net": cap_net,
        }
    return out


def pool(symbols: dict[str, dict]) -> dict:
    syms = list(symbols)
    span_years = float(np.mean([symbols[s]["_span_years"] for s in syms]))
    tot_low = sum(symbols[s]["n_low_entries"] for s in syms)
    pooled = {"span_years": span_years, "n_low_entries_5sym": tot_low,
              "low_entries_per_year_5sym": tot_low / span_years,
              "uncond_rv_mean": float(np.mean(
                  np.concatenate([symbols[s]["_uncond_rv"] for s in syms]))),
              "part1": {}, "part2": {}}
    unc_rv_mean = pooled["uncond_rv_mean"]
    for k in KS:
        # Part 1 pooled
        low_rv = np.concatenate([symbols[s]["_rv"][symbols[s]["_low_ev"]
                  [symbols[s]["_low_ev"] + k < symbols[s]["n_1h_bars"]] + k]
                  for s in syms])
        high_rv = np.concatenate([symbols[s]["_rv"][symbols[s]["_high_ev"]
                   [symbols[s]["_high_ev"] + k < symbols[s]["n_1h_bars"]] + k]
                   for s in syms])
        low_rv = low_rv[~np.isnan(low_rv)]
        high_rv = high_rv[~np.isnan(high_rv)]
        # pooled high-vol probability lift (event-weighted)
        p_low_num = p_low_den = p_base_num = p_base_den = 0
        for s in syms:
            d = symbols[s]
            n = d["n_1h_bars"]
            f_hi = forward_any(d["_above80"], k)
            lev = d["_low_ev"][d["_low_ev"] + k < n]
            bv = d["_valid"].copy(); bv[n - k:] = False
            p_low_num += int(f_hi[lev].sum()); p_low_den += len(lev)
            p_base_num += int(f_hi[bv].sum()); p_base_den += int(bv.sum())
        p_low = p_low_num / p_low_den if p_low_den else None
        p_base = p_base_num / p_base_den if p_base_den else None
        # per-symbol RV(t+6)-style direction check uses k as given; record at all k
        n_low_below = sum(1 for s in syms
                          if symbols[s]["part1"][k]["low_rv_kplus"]
                          and symbols[s]["part1"][k]["low_rv_kplus"]["mean"]
                          < symbols[s]["uncond_rv_mean"])
        n_high_above = sum(1 for s in syms
                           if symbols[s]["part1"][k]["high_rv_kplus"]
                           and symbols[s]["part1"][k]["high_rv_kplus"]["mean"]
                           > symbols[s]["uncond_rv_mean"])
        pooled["part1"][k] = {
            "low_rv_kplus_mean": float(low_rv.mean()),
            "low_rv_kplus_median": float(np.median(low_rv)),
            "high_rv_kplus_mean": float(high_rv.mean()),
            "high_rv_kplus_median": float(np.median(high_rv)),
            "uncond_rv_mean": unc_rv_mean,
            "p_highvol_given_low": p_low, "p_highvol_uncond": p_base,
            "dP_high_lift": (None if p_low is None or p_base is None
                             else p_low - p_base),
            "n_sym_low_below_uncond": n_low_below,
            "n_sym_high_above_uncond": n_high_above,
        }
        # Part 2 pooled
        cap_net = np.concatenate([symbols[s]["part2"][k]["_cap_net"] for s in syms])
        unc = []
        for s in syms:
            d = symbols[s]; n = d["n_1h_bars"]; c = d["_c"]
            bv = np.where((d["_valid"]) & (np.arange(n) + k < n))[0]
            unc.append(np.abs(c[bv + k] / c[bv] - 1.0) * 100)
        unc = np.concatenate(unc)
        n_sym_pos = sum(1 for s in syms
                        if symbols[s]["part2"][k]["cap_net"]
                        and symbols[s]["part2"][k]["cap_net"]["mean"] - COST_PCT > 0)
        st = stats(cap_net)
        pooled["part2"][k] = {
            "cap_net": st,
            "cap_exc": stats(np.concatenate(
                [np.maximum(
                    np.array([(symbols[s]["_h"][i + 1:i + k + 1].max() - symbols[s]["_c"][i]) / symbols[s]["_c"][i]
                              for i in symbols[s]["_low_ev"][symbols[s]["_low_ev"] + k < symbols[s]["n_1h_bars"]]]),
                    np.array([(symbols[s]["_c"][i] - symbols[s]["_lo"][i + 1:i + k + 1].min()) / symbols[s]["_c"][i]
                              for i in symbols[s]["_low_ev"][symbols[s]["_low_ev"] + k < symbols[s]["n_1h_bars"]]]))
                 * 100 for s in syms])),
            "uncond_cap_net": stats(unc),
            "cost_pct": COST_PCT,
            "net_after_cost_mean": float(st["mean"] - COST_PCT),
            "increment_vs_uncond": float(st["mean"] - unc.mean()),
            "thickness_pass": bool(st["mean"] >= THICK_NET_PCT),
            "increment_pass": bool(st["mean"] > unc.mean()),
            "n_symbols_supporting": n_sym_pos,
            "consistency_pass": bool(n_sym_pos >= MIN_SYMBOLS),
        }
        pooled["part2"][k]["all_pass"] = bool(
            pooled["part2"][k]["thickness_pass"]
            and pooled["part2"][k]["increment_pass"]
            and pooled["part2"][k]["consistency_pass"])
    return pooled


def slim_symbol(d: dict) -> dict:
    out = {kk: vv for kk, vv in d.items() if not kk.startswith("_")}
    out["part2"] = {k: {kk: vv for kk, vv in cell.items()
                        if not kk.startswith("_")}
                    for k, cell in out["part2"].items()}
    return out


def run_dataset(tag: str, bars: dict[str, pd.DataFrame]) -> tuple[dict, dict]:
    L(f"\n== {tag}: per-symbol volatility-event computation ==")
    per = {}
    for s, b in bars.items():
        per[s] = compute_symbol(b)
        d = per[s]
        L(f"  {s}: 1h bars {d['n_1h_bars']:,} | low entries {d['n_low_entries']} "
          f"({d['low_entries_per_year']:.0f}/yr) | high entries {d['n_high_entries']} "
          f"| uncond RV {d['uncond_rv_mean']:.5f}")
    pooled = pool(per)
    L(f"  [{tag}] Part1 (clustering):")
    for k in KS:
        p = pooled["part1"][k]
        L(f"    k={k:>2}h | RV(t+k) low {p['low_rv_kplus_mean']:.5f} vs high "
          f"{p['high_rv_kplus_mean']:.5f} vs uncond {p['uncond_rv_mean']:.5f} "
          f"| P(highvol) low {p['p_highvol_given_low']:.3f} uncond "
          f"{p['p_highvol_uncond']:.3f} dP {p['dP_high_lift']:+.3f} "
          f"| sym low<unc {p['n_sym_low_below_uncond']}/5 high>unc {p['n_sym_high_above_uncond']}/5")
    L(f"  [{tag}] Part2 (monetisation, cost {COST_PCT}%):")
    for k in KS:
        p = pooled["part2"][k]
        L(f"    k={k:>2}h | CAP_net mean {p['cap_net']['mean']:.4f}% med "
          f"{p['cap_net']['median']:.4f}% | uncond {p['uncond_cap_net']['mean']:.4f}% "
          f"| incr {p['increment_vs_uncond']:+.4f}% | CAP_exc mean "
          f"{p['cap_exc']['mean']:.4f}% | thick {p['thickness_pass']} incr "
          f"{p['increment_pass']} sup {p['n_symbols_supporting']}/5 | ALL {p['all_pass']}")
    return {s: slim_symbol(per[s]) for s in per}, pooled


def main() -> int:
    L(f"run start (UTC): {datetime.now(timezone.utc).isoformat()}")
    L("DATA ENVIRONMENT: database_mainnet.db (mode=ro) + data/binance_vision; "
      "contaminated DB not touched; closed verdicts not re-tested")
    L("PRE-REGISTERED: 1h / RV win 24 / roll 180d (4320) / p20-p80 / k={6,12,24,48} / "
      "cost 0.20% double-sided / thick CAP_net>=0.30% / increment vs uncond / "
      ">=3/5 symbols / dual-sample / 12-24mo lines — ALL FROZEN, zero variants")
    L("PHYSICAL CONSTRAINT: no options data; only perp-synthesised long-vol testable. "
      "Perps linear -> static straddle gross ~0; only breakout-entry captures vol "
      "(degenerates to directional, already empty band).")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "per_symbol").mkdir(exist_ok=True)

    L("\n== loading OKX mainnet 1m -> 1h ==")
    bars = {}
    for name, (db_sym, inst) in tb.SYMBOLS.items():
        bars[name] = aggregate_1h(tb.load_1m_utc(db_sym))
        L(f"  {name}: 1h bars {len(bars[name]):,}")
    okx_sym, okx_pool = run_dataset("OKX", bars)

    L("\n== loading Binance 1m -> 1h (dual-sample, its own events) ==")
    from research_trend_dualcycle import B_SYM, load_1m_bv
    bars_b = {}
    for name, bs in B_SYM.items():
        bars_b[name] = aggregate_1h(load_1m_bv(bs))
        L(f"  {name}: 1h bars {len(bars_b[name]):,}")
    bnc_sym, bnc_pool = run_dataset("BNC", bars_b)

    # ── verdict chain ──────────────────────────────────────────────────────────
    def part1_pass(pool_) -> bool:
        p = pool_["part1"][6]
        return p["n_sym_low_below_uncond"] >= MIN_SYMBOLS and \
            p["n_sym_high_above_uncond"] >= MIN_SYMBOLS
    p1_okx = part1_pass(okx_pool)
    p1_bnc = part1_pass(bnc_pool)
    predictability = p1_okx and p1_bnc

    passing = [k for k in KS if okx_pool["part2"][k]["all_pass"]
               and bnc_pool["part2"][k]["all_pass"]]
    adjacent_ok = any(KS.index(b) - KS.index(a) == 1
                      for a in passing for b in passing)
    arith = []
    for k in passing:
        cell = okx_pool["part2"][k]
        mu = cell["cap_net"]["mean"] - COST_PCT
        sd = cell["cap_net"]["std"]
        ev_y = okx_pool["low_entries_per_year_5sym"]
        sharpe = (mu / sd) * np.sqrt(ev_y) if sd else float("-inf")
        years = (1.96 / sharpe) ** 2 if sharpe > 0 else float("inf")
        arith.append({"k": k, "mu_pct": mu, "sigma_pct": sd,
                      "events_per_year": ev_y,
                      "sharpe_coarse_optimistic": float(sharpe),
                      "verification_years": float(years),
                      "non_isolated": adjacent_ok,
                      "verdict": ("ELIGIBLE" if years <= VERDICT_GO_Y and adjacent_ok
                                  else "MARGINAL" if years <= VERDICT_MARGINAL_Y and adjacent_ok
                                  else "NOT STARTED")})

    L(f"\n== VERDICT CHAIN ==")
    L(f"  Part 1 predictability: OKX {p1_okx} / Binance {p1_bnc} -> {predictability}")
    L(f"  Part 2 passing k (OKX & Binance): {passing or 'NONE'}")
    for r in arith:
        L(f"    {r}")
    if predictability and not passing:
        death = "b) predictable but perp-synthesis cost (0.20%) eats capturable magnitude"
    elif not predictability:
        death = "a) volatility not predictable (unexpected)"
    elif passing and not arith:
        death = "c) thick but isolated/too rare"
    else:
        death = None
    final = ("NOT STARTED" if not (predictability and passing and
             any(r["verdict"] == "ELIGIBLE" for r in arith)) else "STARTED-CANDIDATE")
    L(f"  DEATH CAUSE: {death}")
    L(f"  FINAL: {final}")

    # ── persist ──────────────────────────────────────────────────────────────
    for s in bars:
        (OUT / "per_symbol" / f"{s}.json").write_text(
            json.dumps(okx_sym[s], indent=2, default=float))
    pre_reg = {"scale": "1h", "rv_window": RV_WIN, "roll_window_bars": ROLL_WIN,
               "percentiles": [P_LO, P_MID, P_HI], "ks_hours": list(KS),
               "cost_pct_double_sided": COST_PCT, "thick_net_pct": THICK_NET_PCT,
               "min_symbols": MIN_SYMBOLS, "verdict_months": [12, 24]}
    (OUT / "summary.json").write_text(json.dumps({
        "positioning": "descriptive vol-event pre-study, go/no-go only; "
                       "denominator = all low-vol entries; mean/median split; "
                       "no options data -> only perp long-vol testable",
        "pre_registered": pre_reg,
        "part1_predictability_pass": {"okx": p1_okx, "binance": p1_bnc,
                                      "overall": predictability},
        "okx_pooled": okx_pool, "passing_k": passing, "arithmetic": arith,
        "death_cause": death, "final_verdict": final,
    }, indent=2, default=float))
    (OUT / "binance_crosscheck.json").write_text(json.dumps({
        "binance_pooled": bnc_pool,
        "part1_pass": p1_bnc,
        "per_symbol": bnc_sym,
    }, indent=2, default=float))
    L(f"run end (UTC): {datetime.now(timezone.utc).isoformat()}")
    (OUT / "run_log.txt").write_text("\n".join(LOG) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
