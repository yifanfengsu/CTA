#!/usr/bin/env python3
"""Portfolio construction stage: correlation structure of the 5 frozen
survivors + 4 pre-registered portfolios through pre-registered gates.

Engines research_trend_baseline (tb) / research_trend_validation (tv) /
research_trend_validation_r2 (r2) imported VERBATIM, zero modification.
Read-only mainnet DB. Same cost convention as all prior stages (verbatim):
  入场与出场均按信号 bar 收盘价 ±1 tick 以 taker 成交（费率 0.05%/边）；
  不假设任何 maker 成交；计入真实 OKX funding（8h 结算）。

POSITIONING STATEMENT (verbatim into report header):
  全部 3.4 年数据已被筛查与验证两轮用尽，幸存者系全样本选择的产物。
  本阶段无论结果如何都无法消除选择偏差；其产出是"进入前向验证的
  最终候选结构"，不是可部署策略。全部结论沿用 VALIDATED* 星号规则。

FROZEN POOL (zero config changes): B1_4h / B2_4h / C2_4h / C2_1d / D2.

ACCOUNTING (fixed before run):
  Each sleeve keeps its original $10k/signal notional. Portfolio = weighted
  sum of sleeve M2M pnl streams with weights summing to 1 (equal = 1/N), so
  a fully-invested portfolio carries ~$50k gross nominal — same capital
  scale as the B&H benchmark, making the absolute P3 gate dimensionally
  comparable. P-D weights = 1/std(full-period sleeve daily pnl), normalized
  once; no rolling, no optimization.
  Daily series = per-bar M2M pnl (r2.m2m_pnl method) bucketed to the UTC
  00:00 grid via index.ceil('D') — 4h and 1d sleeves land on identical
  day boundaries. Monthly = calendar month of those daily buckets.

PRE-REGISTERED PORTFOLIOS (exactly 4, nothing added):
  P-A: B1_4h + B2_4h + C2_4h + C2_1d equal weight
  P-B: all five equal weight (B2/D2 signal-overlap control)
  P-C: B1_4h + D2 + C2_4h + C2_1d equal weight
  P-D: P-A structure, inverse-volatility weights

PRE-REGISTERED GATES (final; may not be modified after results):
  P1 significance: bootstrap of portfolio daily net pnl (10,000 resamples,
     seed=20260611), 95% CI of the mean excludes 0.
  P2 time robustness: no calendar year contributes >60% of net (M2M
     attribution, total>0, all shares <=0.60) AND >=60% of the 30 rolling
     12-month windows (tv.ROLL_STARTS) have positive net.
  P3 drawdown: portfolio maxDD (per-bar M2M equity) < 0.5 x B&H maxDD
     (= $156,571.60, B&H maxDD $313,143.19 from r1 diagnostics).
  3/3 pass = PORTFOLIO-CANDIDATE* (star rule inherited).

CORRELATION DIAGNOSTICS (R1-R3, no gate):
  R1 pairwise Pearson of sleeve daily pnl (window: from first date all 5
     sleeves are live, i.e. max over sleeves of first in-market bar) +
     monthly-frequency cross-check.
  R2 signal overlap: share of (4h bar x symbol) slots where two sleeves
     hold the same symbol in the same direction (1d sleeve mapped onto the
     4h grid of its own symbol; full-period denominator).
  R3 eigen-decomposition of the R1 correlation matrix: #PCs for >=80%
     cumulative variance; effective number of bets (sum λ)^2/Σλ^2 as
     supplementary diagnostic.
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
import research_trend_baseline as tb
import research_trend_validation as tv
import research_trend_validation_r2 as r2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT = PROJECT_ROOT / "reports" / "trend_portfolio_20260611"
R1_DIR = PROJECT_ROOT / "reports" / "trend_validation_20260611"
SEED = 20260611
N_BOOT = 10_000

SLEEVES = [
    {"id": "B1_4h", "tf": "4h", "kind": "emax", "fast": 50, "slow": 200, "mode": "flip"},
    {"id": "B2_4h", "tf": "4h", "kind": "emax", "fast": 20, "slow": 100, "mode": "flip"},
    {"id": "C2_4h", "tf": "4h", "kind": "tsmom", "days": 90, "mode": "flip"},
    {"id": "C2_1d", "tf": "1d", "kind": "tsmom", "days": 90, "mode": "flip"},
    {"id": "D2", "tf": "4h", "kind": "emax", "fast": 20, "slow": 100, "mode": "longflat"},
]
SLEEVE_IDS = [s["id"] for s in SLEEVES]

PORTFOLIOS = {
    "P-A": {"sleeves": ["B1_4h", "B2_4h", "C2_4h", "C2_1d"], "weighting": "equal"},
    "P-B": {"sleeves": SLEEVE_IDS, "weighting": "equal"},
    "P-C": {"sleeves": ["B1_4h", "D2", "C2_4h", "C2_1d"], "weighting": "equal"},
    "P-D": {"sleeves": ["B1_4h", "B2_4h", "C2_4h", "C2_1d"], "weighting": "inverse_vol"},
}

LOG: list[str] = []


def L(msg=""):
    print(msg, flush=True)
    LOG.append(str(msg))


# ── sleeve machinery (outer layer; engines untouched) ────────────────────────
def sleeve_signal(cfg: dict, b: pd.DataFrame) -> np.ndarray:
    if cfg["kind"] == "emax":
        return tb.signal_emax(b, cfg["fast"], cfg["slow"])
    return tb.signal_tsmom(b, cfg["days"], cfg["tf"])


def sleeve_spans(cfg: dict, bars: dict) -> dict:
    spans_by = {}
    for name in tb.SYMBOLS:
        sig = sleeve_signal(cfg, bars[(name, cfg["tf"])])
        spans_by[name] = (r2.positions_longflat(sig) if cfg["mode"] == "longflat"
                          else tb.positions_flip(sig))
    return spans_by


def sleeve_trades(cfg: dict, bars: dict, fund: dict, spans_by: dict) -> list[dict]:
    trades = []
    for name, (_, inst) in tb.SYMBOLS.items():
        trades.extend(tb.build_trades(name, inst, bars[(name, cfg["tf"])],
                                      fund[name], spans_by[name]))
    return trades


def signed_pos(cfg: dict, bars: dict, spans_by: dict) -> dict:
    """Per-symbol signed position (+1/0/-1) indexed by bar END time UTC."""
    out = {}
    for name in tb.SYMBOLS:
        b = bars[(name, cfg["tf"])]
        pos = np.zeros(len(b))
        for ei, xi, side, _ in spans_by[name]:
            pos[ei + 1:xi + 1] = side
        idx = pd.to_datetime(b["end_min"].to_numpy() * 60, unit="s", utc=True)
        out[name] = pd.Series(pos, index=idx)
    return out


def map_to_4h(s: pd.Series, idx4h: pd.DatetimeIndex) -> pd.Series:
    """Map a coarser-grid signed pos onto 4h end-times: each 4h bar takes the
    value of the bar whose holding interval contains it (first end >= t)."""
    j = np.searchsorted(s.index.values, idx4h.values, side="left")
    v = np.zeros(len(idx4h))
    ok = j < len(s)
    v[ok] = s.to_numpy()[j[ok]]
    return pd.Series(v, index=idx4h)


def to_daily(perbar: pd.Series) -> pd.Series:
    return perbar.groupby(perbar.index.ceil("D")).sum()


def boot_ci_daily(x: np.ndarray) -> dict:
    rng = np.random.default_rng(SEED)
    means = rng.choice(x, size=(N_BOOT, len(x)), replace=True).mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {"n_days": int(len(x)), "mean_daily": float(x.mean()),
            "ci95_daily_mean": [float(lo), float(hi)],
            "excludes_zero": bool(lo > 0 or hi < 0)}


def series_stats(s: pd.Series) -> dict:
    q = s.quantile([0.05, 0.5, 0.95])
    return {"mean": float(s.mean()), "p5": float(q[0.05]), "p50": float(q[0.5]),
            "p95": float(q[0.95]), "min": float(s.min()), "max": float(s.max())}


def main() -> int:
    L(f"run start (UTC): {datetime.now(timezone.utc).isoformat()}")
    L("DATA: database_mainnet.db (mode=ro) | engines tb/tv/r2 imported verbatim, unmodified")
    L("cost convention identical to prior stages: taker both sides ±1 tick + real OKX 8h funding")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "correlation").mkdir(exist_ok=True)
    (OUT / "portfolios").mkdir(exist_ok=True)

    bh_ref = json.loads((R1_DIR / "diagnostics" / "buy_hold.json").read_text())
    P3_LIMIT = 0.5 * bh_ref["max_dd_usd"]
    L(f"B&H maxDD ${bh_ref['max_dd_usd']:,.2f} -> P3 limit ${P3_LIMIT:,.2f} (pre-registered $156,572)")

    L("\nloading data via baseline engine ...")
    m1, bars, fund = {}, {}, {}
    for name, (db_sym, inst) in tb.SYMBOLS.items():
        m1[name] = tb.load_1m_utc(db_sym)
        bars[(name, "4h")] = tb.aggregate(m1[name], "4h")
        bars[(name, "1d")] = tb.aggregate(m1[name], "1d")
        fund[name] = tb.load_funding(inst, m1[name])

    bh_eq = r2.bh_equity_4h(bars, fund)
    bh_daily = to_daily(bh_eq.diff().dropna())
    bh_monthly = bh_daily.groupby(bh_daily.index.to_period("M")).sum()

    # ── sleeves: M2M pnl, signed positions, cross-checks ─────────────────────
    L("\n== sleeves (frozen pool, zero config changes) ==")
    perbar, daily, pos4h, sleeve_net, first_live = {}, {}, {}, {}, {}
    for cfg in SLEEVES:
        sid = cfg["id"]
        spans_by = sleeve_spans(cfg, bars)
        trades = sleeve_trades(cfg, bars, fund, spans_by)
        pb, _ = r2.m2m_pnl(cfg["tf"], bars, fund, spans_by)
        perbar[sid] = pb
        daily[sid] = to_daily(pb)
        sp = signed_pos(cfg, bars, spans_by)
        if cfg["tf"] != "4h":
            sp = {n: map_to_4h(s, pd.DatetimeIndex(
                pd.to_datetime(bars[(n, "4h")]["end_min"].to_numpy() * 60,
                               unit="s", utc=True))) for n, s in sp.items()}
        pos4h[sid] = pd.DataFrame(sp).fillna(0.0)
        sleeve_net[sid] = float(pb.sum())
        live = pd.concat(sp.values(), axis=1).fillna(0.0)
        first_live[sid] = live.index[(live != 0).any(axis=1)][0]
        tnet = tv.net_of(trades)
        L(f"[{sid}] m2m net ${sleeve_net[sid]:,.0f} | trade-account net ${tnet:,.0f} "
          f"| diff ${sleeve_net[sid] - tnet:,.2f} | first live {first_live[sid].date()} "
          f"| daily std ${daily[sid].std():,.0f}")

    common_live = max(first_live.values())
    L(f"\ncommon-live date (corr window start): {common_live.date()} "
      f"(latest sleeve: {max(first_live, key=first_live.get)})")

    # align all sleeve dailies on the union daily grid
    dfd = pd.DataFrame(daily).fillna(0.0)
    dfd_corrwin = dfd[dfd.index >= common_live.ceil("D")]
    dfd.to_csv(OUT / "correlation" / "sleeve_daily_pnl.csv")

    # ── R1: correlation matrices ─────────────────────────────────────────────
    corr_d = dfd_corrwin.corr(method="pearson")
    monthly_all = dfd_corrwin.groupby(dfd_corrwin.index.to_period("M")).sum()
    corr_m = monthly_all.corr(method="pearson")
    (OUT / "correlation" / "daily_corr.json").write_text(json.dumps(
        {"window_start": str(common_live.ceil('D')), "n_days": len(dfd_corrwin),
         "pearson_daily": corr_d.round(4).to_dict()}, indent=2))
    (OUT / "correlation" / "monthly_corr.json").write_text(json.dumps(
        {"n_months": len(monthly_all), "pearson_monthly": corr_m.round(4).to_dict()},
        indent=2))
    L("\n== R1 daily Pearson corr (window from common-live) ==")
    L(corr_d.round(3).to_string())
    L("\nmonthly cross-check:")
    L(corr_m.round(3).to_string())

    # ── R2: signal overlap ───────────────────────────────────────────────────
    L("\n== R2 same-symbol same-direction holding overlap (share of 4h bar-symbol slots) ==")
    overlap = {}
    for a, b in combinations(SLEEVE_IDS, 2):
        pa, pb_ = pos4h[a].align(pos4h[b], join="outer", fill_value=0.0)
        prod = pa.to_numpy() * pb_.to_numpy()
        n_slots = prod.size
        same = float((prod > 0).sum() / n_slots)
        opp = float((prod < 0).sum() / n_slots)
        both = float(((pa.to_numpy() != 0) & (pb_.to_numpy() != 0)).sum() / n_slots)
        overlap[f"{a}|{b}"] = {"same_dir": same, "opposite_dir": opp,
                               "both_in_market": both,
                               "same_dir_given_both": same / both if both else None}
        L(f"  {a:6s} vs {b:6s}: same-dir {same:6.1%} | opp {opp:6.1%} | "
          f"both-in {both:6.1%} | same|both {same / both if both else 0:.1%}")
    (OUT / "correlation" / "overlap.json").write_text(json.dumps(overlap, indent=2))

    # ── R3: eigen decomposition ──────────────────────────────────────────────
    ev = np.linalg.eigvalsh(corr_d.to_numpy())[::-1]
    cum = np.cumsum(ev) / ev.sum()
    n80 = int(np.argmax(cum >= 0.80) + 1)
    enb = float(ev.sum() ** 2 / (ev ** 2).sum())
    (OUT / "correlation" / "pca.json").write_text(json.dumps(
        {"eigenvalues": [float(x) for x in ev],
         "cum_var_explained": [float(x) for x in cum],
         "n_pc_for_80pct": n80, "effective_n_bets": enb}, indent=2))
    L(f"\n== R3 == eigenvalues {[round(float(x), 3) for x in ev]} | "
      f"cum var {[round(float(x), 3) for x in cum]} | PCs for 80%: {n80} | ENB {enb:.2f}")

    # ── portfolios ───────────────────────────────────────────────────────────
    L("\n== portfolios (pre-registered, weights fixed before gates) ==")
    full_std = {sid: float(dfd[sid].std()) for sid in SLEEVE_IDS}
    summary = []
    for pid, spec in PORTFOLIOS.items():
        ids = spec["sleeves"]
        if spec["weighting"] == "equal":
            w = {s: 1.0 / len(ids) for s in ids}
        else:  # inverse full-period daily vol, one-shot, no optimization
            inv = {s: 1.0 / full_std[s] for s in ids}
            tot = sum(inv.values())
            w = {s: v / tot for s, v in inv.items()}

        pb_port = None
        for s in ids:
            x = perbar[s] * w[s]
            pb_port = x if pb_port is None else pb_port.add(x, fill_value=0.0)
        pb_port = pb_port.sort_index()
        eq = pb_port.cumsum()
        maxdd = float((eq.cummax() - eq).max())
        d_port = (dfd[ids] * pd.Series(w)).sum(axis=1)
        net = float(d_port.sum())

        yearly = d_port.groupby(d_port.index.year).sum()
        shares = {int(y): float(v / net) if net != 0 else float("inf")
                  for y, v in yearly.items()}
        year_ok = net > 0 and all(s_ <= 0.60 for s_ in shares.values())
        mon = d_port.groupby(d_port.index.to_period("M")).sum()
        wins = [float(mon[(mon.index >= st) & (mon.index < st + 12)].sum())
                for st in tv.ROLL_STARTS]
        roll_frac = sum(1 for x in wins if x > 0) / len(wins)
        p2 = {"year_shares": shares, "year_share_ok": year_ok,
              "rolling12_frac_positive": roll_frac, "rolling12_ok": roll_frac >= 0.60,
              "pass": year_ok and roll_frac >= 0.60}
        p1 = boot_ci_daily(d_port.to_numpy())
        p1["pass"] = p1["excludes_zero"]
        p3 = {"max_dd_usd": maxdd, "limit": P3_LIMIT, "pass": maxdd < P3_LIMIT}
        gates = {"P1": p1, "P2": p2, "P3": p3}
        verdict = ("PORTFOLIO-CANDIDATE*" if all(g["pass"] for g in gates.values())
                   else "FAIL")
        died = [k for k, g in gates.items() if not g["pass"]]

        bm = bh_monthly.reindex(mon.index).fillna(0.0)
        corr_bh = float(np.corrcoef(mon.to_numpy(), bm.to_numpy())[0, 1])
        contrib = {s: {"weight": w[s], "net_contribution": w[s] * sleeve_net[s],
                       "share_of_net": w[s] * sleeve_net[s] / net if net else None}
                   for s in ids}

        # nominal exposure on the 4h grid ($10k per held signal, weighted)
        net_exp, gross_exp = None, None
        for s in ids:
            ne = pos4h[s].sum(axis=1) * tb.NOTIONAL * w[s]
            ge = pos4h[s].abs().sum(axis=1) * tb.NOTIONAL * w[s]
            net_exp = ne if net_exp is None else net_exp.add(ne, fill_value=0.0)
            gross_exp = ge if gross_exp is None else gross_exp.add(ge, fill_value=0.0)
        exposure = {
            "gross": series_stats(gross_exp),
            "net": series_stats(net_exp),
            "share_time_net_long": float((net_exp > 0).mean()),
            "share_time_net_short": float((net_exp < 0).mean()),
            "net_monthly_mean": {str(k): float(v) for k, v in
                                 net_exp.groupby(net_exp.index.to_period("M")).mean().items()},
        }

        out = {"portfolio": pid, "sleeves": ids, "weights": w, "verdict": verdict,
               "died_at": died, "net_pnl": net, "max_dd_usd": maxdd,
               "net_over_dd": net / maxdd if maxdd else float("inf"),
               "yearly_net": {int(y): float(v) for y, v in yearly.items()},
               "gates": gates, "corr_bh_monthly": corr_bh,
               "sleeve_contribution": contrib, "exposure": exposure,
               "rolling12_windows_net": wins,
               "monthly_net": {str(k): float(v) for k, v in mon.items()},
               "m2m_consistency_check": {"perbar_sum": float(pb_port.sum()),
                                         "daily_sum": net,
                                         "diff": float(pb_port.sum()) - net}}
        (OUT / "portfolios" / f"{pid}.json").write_text(json.dumps(out, indent=2, default=float))
        d_port.rename("daily_net_pnl_usd").to_csv(OUT / "portfolios" / f"{pid}_daily.csv")
        summary.append({k: out[k] for k in ("portfolio", "sleeves", "weights", "verdict",
                                            "died_at", "net_pnl", "max_dd_usd",
                                            "net_over_dd", "corr_bh_monthly")}
                       | {"P1_ci": p1["ci95_daily_mean"], "P1_pass": p1["pass"],
                          "P2_pass": p2["pass"], "P3_pass": p3["pass"],
                          "year_shares": shares, "roll_frac": roll_frac})
        L(f"[{pid}] {verdict}{' died:' + ','.join(died) if died else ''} | "
          f"net ${net:,.0f} | maxDD ${maxdd:,.0f} (limit ${P3_LIMIT:,.0f}) | net/DD "
          f"{net / maxdd:.2f} | P1 CI [{p1['ci95_daily_mean'][0]:.2f},"
          f"{p1['ci95_daily_mean'][1]:.2f}] excl0={p1['excludes_zero']} | "
          f"yrs {dict((y, round(s_, 2)) for y, s_ in shares.items())} | roll12 {roll_frac:.0%} | "
          f"corrBH {corr_bh:.3f} | gross-exp mean ${exposure['gross']['mean']:,.0f}")

    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    nc = sum(1 for s in summary if s["verdict"] == "PORTFOLIO-CANDIDATE*")
    L(f"\nPORTFOLIO-CANDIDATE*: {nc} / 4 | effective independent signals (PCs for 80%): {n80}")
    L(f"run end (UTC): {datetime.now(timezone.utc).isoformat()}")
    (OUT / "run_log.txt").write_text("\n".join(LOG) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
