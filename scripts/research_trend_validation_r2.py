#!/usr/bin/env python3
"""Trend validation round 2: V1' concentration-QUALITY re-審 + long/flat family D.

Engines research_trend_baseline / research_trend_validation imported VERBATIM,
zero modification. Same cost convention (taker both sides ±1 tick + real OKX
8h funding). Read-only mainnet DB.

DISCIPLINE STATEMENT (verbatim into report, unchanged):
  V1（剔除 top 5% 后毛利>0）对趋势类策略构成 category error：该类策略的利润
  分布设计上即为尾部收割。本轮以预注册的 V1'（集中度质量检验）替代 V1 重审。
  纪律边界：① 原 V1 判定（0/15 FAIL）保持不变、永久存档；② V1' 数字基于机制
  论证预先写死，未参考本项目 top 交易分布数据；③ 本次是唯一一次 gate 修正机会，
  若 V1' 亦不过，结论即为终局，不允许设计 V1''；④ 任何经 V1' 通过的结论永久
  标注"V1' 系事后设计的修正检验"，证据等级低于一次通过。

V1' (pre-registered, all numbers fixed):
  T = top 5% trades by full-period net PnL (count ceil).
  a 分散性 : T spans >=3 symbols AND >=2 calendar years (entry time).
  b 可重复性: >=50% of the 30 rolling 12m windows (monthly steps, starts
             2023-01..2025-06) contain >=1 ENTRY of a full-period top-10%
             (by net, ceil) trade.
  c 尾部效率: portfolio total GROSS >= 0.3 x (sum of GROSS of T).
  PASS = a AND b AND c.

FAMILY D (pre-registered, 4 configs, zero new params):
  D1 = B1_4h long/flat (EMA50/200: golden-cross hold long, death-cross FLAT)
  D2 = B2_4h long/flat (EMA20/100)
  D3 = C2_4h long/flat (90d momentum >0 hold, <=0 flat)
  D4 = C2_1d long/flat (same, 1d)
  Full gates: V1' + V2 + V3 + V4 + V5 (same operational defs as round 1;
  V3 neighborhood scales mother lookbacks x{0.75,1,1.25}).

D diagnostics (no gate; conclusions must cite):
  d-i  monthly Pearson corr with B&H: both sides MARK-TO-MARKET monthly PnL
       (per-bar position x close-to-close pnl, fees on entry/exit bars,
       funding attributed to its settlement bar). corr > 0.9 -> label
       "化妆的 B&H" verbatim.
  d-ii net/maxDD vs B&H (0.21) and vs mother config.
  d-iii bootstrap 95% CI of per-trade net (10,000, seed 20260611).
  d-iv flat-time share; and B&H deep-drawdown coverage: deep-DD bars :=
       bars where B&H portfolio equity sits below running peak by more than
       0.5 x B&H maxDD; coverage = average flat share over those bars
       (vs overall flat share — higher = timing exits where it matters).
"""

from __future__ import annotations

import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import research_trend_baseline as tb
import research_trend_validation as tv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT = PROJECT_ROOT / "reports" / "trend_validation_r2_20260611"
R1 = PROJECT_ROOT / "reports" / "trend_validation_20260611"
SEED = 20260611

MOTHERS = ["B1_4h", "B2_4h", "C2_4h", "C2_1d"]
D_FAMILY = [
    {"id": "D1", "mother": "B1_4h", "tf": "4h", "kind": "emax", "fast": 50, "slow": 200},
    {"id": "D2", "mother": "B2_4h", "tf": "4h", "kind": "emax", "fast": 20, "slow": 100},
    {"id": "D3", "mother": "C2_4h", "tf": "4h", "kind": "tsmom", "days": 90},
    {"id": "D4", "mother": "C2_1d", "tf": "1d", "kind": "tsmom", "days": 90},
]

LOG: list[str] = []


def L(msg=""):
    print(msg, flush=True)
    LOG.append(str(msg))


# ── long/flat machinery (new outer-layer code; engines untouched) ────────────
def positions_longflat(sig: np.ndarray) -> list[tuple]:
    res, pos, ei = [], 0, -1
    for i in range(len(sig)):
        s = sig[i]
        if np.isnan(s):
            continue
        if pos == 0 and s > 0:
            pos, ei = 1, i
        elif pos == 1 and s <= 0:
            res.append((ei, i, 1, "flat_signal"))
            pos = 0
    if pos == 1:
        res.append((ei, len(sig) - 1, 1, "end_of_data"))
    return res


def d_signal(cfg: dict, b: pd.DataFrame) -> np.ndarray:
    if cfg["kind"] == "emax":
        return tb.signal_emax(b, cfg["fast"], cfg["slow"])
    return tb.signal_tsmom(b, cfg["days"], cfg["tf"])


def run_d(cfg: dict, bars, fund) -> tuple[list[dict], dict]:
    """Returns (trades, spans_by_symbol)."""
    trades, spans_by = [], {}
    for name, (_, inst) in tb.SYMBOLS.items():
        b = bars[(name, cfg["tf"])]
        spans = positions_longflat(d_signal(cfg, b))
        spans_by[name] = spans
        trades.extend(tb.build_trades(name, inst, b, fund[name], spans))
    return trades, spans_by


def d_variants(cfg: dict) -> list[dict]:
    out = []
    for f in tv.SCALES:
        if cfg["kind"] == "emax":
            out.append({**cfg, "fast": max(2, round(cfg["fast"] * f)),
                        "slow": max(3, round(cfg["slow"] * f)), "_scale": (f,)})
        else:
            out.append({**cfg, "days": max(2, round(cfg["days"] * f)), "_scale": (f,)})
    return out


def gate_v3_d(cfg, bars, fund) -> dict:
    pts = []
    for v in d_variants(cfg):
        n = tv.net_of(run_d(v, bars, fund)[0])
        pts.append({"scale": v["_scale"], "net": n})
    frac = sum(1 for p in pts if p["net"] > 0) / len(pts)
    return {"points": pts, "frac_positive": frac, "pass": frac >= 0.75}


# ── V1' (pre-registered) ──────────────────────────────────────────────────────
def gate_v1p(trades: list[dict]) -> dict:
    n = len(trades)
    k5, k10 = math.ceil(0.05 * n), math.ceil(0.10 * n)
    by_net = sorted(trades, key=lambda t: t["net_pnl_usd"], reverse=True)
    T, T10 = by_net[:k5], by_net[:k10]
    syms = sorted({t["symbol"] for t in T})
    years = sorted({pd.Timestamp(t["entry_time"]).year for t in T})
    a_ok = len(syms) >= 3 and len(years) >= 2
    t10_entries = pd.PeriodIndex(
        [pd.Timestamp(t["entry_time"]).tz_convert("UTC").tz_localize(None) for t in T10],
        freq="M") if T10 else pd.PeriodIndex([], freq="M")
    hits = []
    for st in tv.ROLL_STARTS:
        hits.append(bool(((t10_entries >= st) & (t10_entries < st + 12)).any()))
    b_frac = sum(hits) / len(hits)
    total_gross = tv.gross_of(trades)
    t_gross = tv.gross_of(T)
    c_ok = total_gross >= 0.3 * t_gross
    profile = {
        "top5pct_n": k5, "symbols": syms, "years": years,
        "symbol_counts": {s: sum(1 for t in T if t["symbol"] == s) for s in syms},
        "year_counts": {str(y): sum(1 for t in T if pd.Timestamp(t["entry_time"]).year == y)
                        for y in years},
        "entry_months": sorted({str(pd.Timestamp(t["entry_time"]).tz_convert('UTC').strftime('%Y-%m')) for t in T}),
        "T_gross": t_gross, "T_net": tv.net_of(T),
    }
    return {"a_dispersion": {"n_symbols": len(syms), "n_years": len(years), "pass": a_ok},
            "b_repeatability": {"frac_windows_with_top10_entry": b_frac, "pass": b_frac >= 0.50},
            "c_tail_efficiency": {"total_gross": total_gross, "tail_gross": t_gross,
                                  "ratio": total_gross / t_gross if t_gross > 0 else float("inf"),
                                  "pass": c_ok},
            "profile": profile,
            "pass": a_ok and (b_frac >= 0.50) and c_ok}


# ── mark-to-market monthly pnl (for d-i / d-iv) ───────────────────────────────
def m2m_pnl(cfg_tf: str, bars, fund, spans_by) -> tuple[pd.Series, dict]:
    """Portfolio per-bar M2M pnl (Series indexed by bar end UTC) + pos arrays."""
    total, pos_by = None, {}
    for name, (_, inst) in tb.SYMBOLS.items():
        spec = tb.CONTRACT_SPECS[inst]
        b = bars[(name, cfg_tf)]
        c = b["close"].to_numpy()
        endm = b["end_min"].to_numpy()
        pnl = np.zeros(len(b))
        pos = np.zeros(len(b))
        f = fund[name]
        fmins = f["slot_min"].to_numpy()
        fpay = (f["rate"] * f["settle_px"]).to_numpy()
        for ei, xi, side, _ in spans_by[name]:
            n = tb.calc_contracts(inst, c[ei])
            pos[ei + 1:xi + 1] = 1
            pnl[ei + 1:xi + 1] += np.diff(c[ei:xi + 1]) * n * spec["ctVal"] * side
            # fills at close±tick + taker fees, charged on entry/exit bars
            pnl[ei] -= spec["tickSz"] * n * spec["ctVal"] + \
                tb.FEE_TAKER * (c[ei] + spec["tickSz"]) * n * spec["ctVal"]
            pnl[xi] -= spec["tickSz"] * n * spec["ctVal"] + \
                tb.FEE_TAKER * (c[xi] - spec["tickSz"]) * n * spec["ctVal"]
            in_span = (fmins > endm[ei]) & (fmins <= endm[xi])
            bi = np.searchsorted(endm, fmins[in_span], side="left")
            for j, p in zip(bi, fpay[in_span] * n * spec["ctVal"] * side):
                if j < len(b):
                    pnl[j] -= p
        idx = pd.to_datetime(endm * 60, unit="s", utc=True)
        s = pd.Series(pnl, index=idx)
        total = s if total is None else total.add(s, fill_value=0)
        pos_by[name] = pd.Series(pos, index=idx)
    return total, pos_by


def bh_equity_4h(bars, fund) -> pd.Series:
    total = None
    for name, (_, inst) in tb.SYMBOLS.items():
        spec = tb.CONTRACT_SPECS[inst]
        b = bars[(name, "4h")]
        ep_raw = float(b["open"].iloc[0])
        ep = ep_raw + spec["tickSz"]
        n = tb.calc_contracts(inst, ep_raw)
        f = fund[name]
        fmins = f["slot_min"].to_numpy()
        fpay = (f["rate"] * f["settle_px"]).to_numpy() * n * spec["ctVal"]
        endm = b["end_min"].to_numpy()
        acc = np.zeros(len(b))
        bi = np.searchsorted(endm, fmins, side="left")
        for j, p in zip(bi, fpay):
            if 0 <= j < len(b):
                acc[j] += p
        eq = (b["close"].to_numpy() - ep) * n * spec["ctVal"] - np.cumsum(acc) \
            - tb.FEE_TAKER * ep * n * spec["ctVal"]
        idx = pd.to_datetime(endm * 60, unit="s", utc=True)
        s = pd.Series(eq, index=idx)
        total = s if total is None else total.add(s, fill_value=0)
    return total


def main() -> int:
    L(f"run start (UTC): {datetime.now(timezone.utc).isoformat()}")
    L("DATA: database_mainnet.db (mode=ro) | engines tb/tv imported verbatim, unmodified")
    OUT.mkdir(parents=True, exist_ok=True)
    for sub in ("gates", "top_trades_profile", "d_family", "diagnostics"):
        (OUT / sub).mkdir(exist_ok=True)

    m1, bars, fund = {}, {}, {}
    for name, (db_sym, inst) in tb.SYMBOLS.items():
        m1[name] = tb.load_1m_utc(db_sym)
        bars[(name, "4h")] = tb.aggregate(m1[name], "4h")
        bars[(name, "1d")] = tb.aggregate(m1[name], "1d")
        fund[name] = tb.load_funding(inst, m1[name])

    bh_eq = bh_equity_4h(bars, fund)
    bh_peak = bh_eq.cummax()
    bh_dd = bh_peak - bh_eq
    bh_maxdd = float(bh_dd.max())
    bh_deep = bh_dd > 0.5 * bh_maxdd
    bh_monthly = bh_eq.diff().groupby(bh_eq.index.to_period("M")).sum()
    bh_net = float(bh_eq.iloc[-1]) - 0  # exit fee/tick ignored in path; summary uses r1 number
    L(f"B&H: path-net ${bh_net:,.0f} | maxDD ${bh_maxdd:,.0f} | deep-DD bars "
      f"{int(bh_deep.sum())}/{len(bh_deep)} ({bh_deep.mean()*100:.1f}%)")

    rng = np.random.default_rng(SEED)
    results = []

    # ── part 2: mothers V1' re-review (V2-V5 cited from r1, configs unchanged)
    L("\n== mothers V1' ==")
    for cid in MOTHERS:
        cfg = next(c for c in tb.CONFIGS if c["id"] == cid)
        trades = tv.run_config(cfg, bars, fund)
        v1p = gate_v1p(trades)
        r1g = json.loads((R1 / "gates" / f"{cid}.json").read_text())
        others_pass = all(r1g["gates"][k]["pass"] for k in ("V2", "V3", "V4", "V5"))
        verdict = "VALIDATED*" if (v1p["pass"] and others_pass) else "FAIL"
        (OUT / "gates" / f"{cid}_v1p.json").write_text(json.dumps(v1p, indent=2, default=float))
        (OUT / "top_trades_profile" / f"{cid}.json").write_text(
            json.dumps(v1p["profile"], indent=2, default=float))
        results.append({"id": cid, "group": "mother", "verdict": verdict,
                        "v1p": {k: v1p[k]["pass"] if isinstance(v1p[k], dict) and "pass" in v1p[k]
                                else None for k in ("a_dispersion", "b_repeatability", "c_tail_efficiency")},
                        "v1p_pass": v1p["pass"], "r1_v2v5_pass": others_pass,
                        "net": tv.net_of(trades)})
        L(f"[{cid}] V1' a={v1p['a_dispersion']['pass']} "
          f"b={v1p['b_repeatability']['frac_windows_with_top10_entry']:.0%}->{v1p['b_repeatability']['pass']} "
          f"c=ratio {v1p['c_tail_efficiency']['ratio']:.2f}->{v1p['c_tail_efficiency']['pass']} "
          f"=> {verdict} | top5%: syms {v1p['profile']['symbol_counts']} years {v1p['profile']['year_counts']}")

    # ── part 3: family D full gates + diagnostics
    L("\n== family D (long/flat) ==")
    for cfg in D_FAMILY:
        cid = cfg["id"]
        trades, spans_by = run_d(cfg, bars, fund)
        gates = {"V1p": gate_v1p(trades), "V2": tv.gate_v2(trades),
                 "V3": gate_v3_d(cfg, bars, fund), "V4": tv.gate_v4(trades),
                 "V5": tv.gate_v5(trades)}
        verdict = "VALIDATED*" if all(g["pass"] for g in gates.values()) else "FAIL"
        died = [k for k, g in gates.items() if not g["pass"]]
        d3 = tv.diag_d3_bootstrap(trades, rng)
        mm, pos_by = m2m_pnl(cfg["tf"], bars, fund, spans_by)
        m2m_total = float(mm.sum())
        trade_net = tv.net_of(trades)
        monthly = mm.groupby(mm.index.to_period("M")).sum()
        bm = bh_monthly.reindex(monthly.index).fillna(0)
        corr = float(np.corrcoef(monthly.to_numpy(), bm.to_numpy())[0, 1])
        met = tb.metrics(trades)
        ndd = trade_net / met["max_dd_usd"] if met["max_dd_usd"] else float("inf")
        # d-iv flat share + deep-DD coverage
        pos_df = pd.DataFrame(pos_by)
        flat_share = float(1 - pos_df.mean().mean())
        if cfg["tf"] == "4h":
            deep_idx = bh_deep.reindex(pos_df.index).fillna(False)
        else:
            deep_idx = bh_deep.reindex(pos_df.index, method="ffill").fillna(False)
        cov = float(1 - pos_df[deep_idx.to_numpy()].mean().mean()) if deep_idx.any() else float("nan")
        mother_net_dd = next(r["net_over_dd"] for r in json.loads(
            (R1 / "summary.json").read_text()) if r["id"] == cfg["mother"])
        diag = {"d_i_monthly_corr_bh": corr, "d_i_label": "化妆的 B&H" if corr > 0.9 else "独立性可接受",
                "d_ii_net_over_dd": ndd, "d_ii_bh": 0.21, "d_ii_mother": mother_net_dd,
                "d_iii_bootstrap": d3,
                "d_iv_flat_share": flat_share, "d_iv_deepdd_coverage": cov,
                "m2m_vs_trades_check": {"m2m": m2m_total, "trades": trade_net,
                                        "diff": m2m_total - trade_net}}
        (OUT / "d_family" / f"{cid}.json").write_text(json.dumps(
            {"config": cfg, "verdict": verdict, "died_at": died, "gates": gates,
             "metrics": met, "diagnostics": diag}, indent=2, default=float))
        (OUT / "top_trades_profile" / f"{cid}.json").write_text(
            json.dumps(gates["V1p"]["profile"], indent=2, default=float))
        results.append({"id": cid, "group": "D", "mother": cfg["mother"], "verdict": verdict,
                        "died_at": died, "net": trade_net, "n": met["n"],
                        "net_over_dd": ndd, "corr_bh": corr, "flat_share": flat_share,
                        "deepdd_coverage": cov, "d3": d3})
        L(f"[{cid}<-{cfg['mother']}] {verdict}{' died:' + ','.join(died) if died else ''} | "
          f"n={met['n']} net ${trade_net:,.0f} | net/DD {ndd:.2f} (mother {mother_net_dd:.2f}) | "
          f"corrBH {corr:.3f} | flat {flat_share:.0%} cov {cov:.0%} | "
          f"CI [{d3['ci95'][0]:.1f},{d3['ci95'][1]:.1f}] excl0={d3['excludes_zero']} | "
          f"m2m check diff ${diag['m2m_vs_trades_check']['diff']:.0f}")

    (OUT / "summary.json").write_text(json.dumps(results, indent=2, default=float))
    nv = sum(1 for r in results if r["verdict"] == "VALIDATED*")
    L(f"\nVALIDATED*: {nv} / FAIL: {len(results) - nv} (8 candidates)")
    L(f"run end (UTC): {datetime.now(timezone.utc).isoformat()}")
    (OUT / "run_log.txt").write_text("\n".join(LOG) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
