#!/usr/bin/env python3
"""Trend validation stage: beta separation + 5 robustness gates over the 15
screen survivors. Outer-layer extension of research_trend_baseline.py — the
baseline engine (incl. the hand-checked funding module) is imported VERBATIM
and not modified. Same cost convention as the screen (taker both sides
±1 tick + real OKX 8h funding). 15 configs untouched, zero param changes.

PRE-REGISTERED GATES (final; may not be modified after results):
  V1 concentration : drop top ceil(5% n) trades by net PnL ->
                     remaining portfolio GROSS > 0.
  V2 leave-one-out : exclude each of the 5 symbols in turn ->
                     portfolio NET > 0 in all 5 exclusions.
  V3 neighborhood  : core lookbacks x {0.75, 1.0, 1.25}
                     (A: entry/exit scaled independently -> 9 points;
                      B: fast&slow scaled together -> 3; C: days -> 3);
                     >=75% of points NET > 0. Flatness check ONLY —
                     results must NOT be used to change any parameter.
  V4 slippage      : 1 tick -> 3 ticks both sides, NET still > 0
                     (5-tick figure reported as diagnostic only).
                     Note: fees are invariant to symmetric slippage
                     (entry +k·t, exit −k·t cancel in notional sum), so
                     net(k) = net(1) − 2(k−1)·tick·size·ctVal per trade —
                     exact post-processing, no re-run needed.
  V5 time          : (a) no single calendar year contributes >60% of total
                     net (attribution by ENTRY time, share = year_net /
                     total_net); AND (b) >=60% of the 30 rolling 12-month
                     windows (monthly steps, starts 2023-01..2025-06) have
                     positive net. Both required.
  VALIDATED = 5/5 pass; otherwise FAIL (record which gate killed it).

DIAGNOSTICS (no gate, answer the beta question):
  D1 long/short decomposition (counts, gross, net, time-in-market share).
  D2 buy & hold benchmark: equal-weight 5 symbols, $10k each, taker entry at
     first 4h bar open +1 tick, taker exit at last 4h close −1 tick, REAL
     funding every 8h for the full period; net, maxDD (4h mark-to-market
     equity incl. accrued funding), net/maxDD vs each config.
  D3 bootstrap on per-trade net (10,000 resamples, seed 20260611), 95% CI of
     the mean; n<50 -> CI flagged unreliable.
  Beta flags (transparent rule, stated before results): short_net < 0;
  long share of net > 90%; net/maxDD < B&H net/maxDD.
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
import research_trend_baseline as tb  # engine reused verbatim, not modified

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT = PROJECT_ROOT / "reports" / "trend_validation_20260611"
SEED = 20260611
SCALES = (0.75, 1.0, 1.25)
ROLL_STARTS = pd.period_range("2023-01", "2025-06", freq="M")  # 30 windows

LOG: list[str] = []


def L(msg=""):
    print(msg, flush=True)
    LOG.append(str(msg))


# ── engine harness (uses tb functions only) ──────────────────────────────────
def run_config(cfg: dict, bars: dict, fund: dict) -> list[dict]:
    trades = []
    for name, (_, inst) in tb.SYMBOLS.items():
        b = bars[(name, cfg["tf"])]
        if cfg["kind"] == "donchian":
            spans = tb.positions_donchian(b, cfg["entry_n"], cfg["exit_n"])
        elif cfg["kind"] == "emax":
            spans = tb.positions_flip(tb.signal_emax(b, cfg["fast"], cfg["slow"]))
        else:
            spans = tb.positions_flip(tb.signal_tsmom(b, cfg["days"], cfg["tf"]))
        trades.extend(tb.build_trades(name, inst, b, fund[name], spans))
    return trades


def net_of(trades) -> float:
    return float(sum(t["net_pnl_usd"] for t in trades))


def gross_of(trades) -> float:
    return float(sum(t["gross_pnl_usd"] for t in trades))


# ── gates ─────────────────────────────────────────────────────────────────────
def gate_v1(trades) -> dict:
    k = math.ceil(0.05 * len(trades))
    keep = sorted(trades, key=lambda t: t["net_pnl_usd"], reverse=True)[k:]
    g = gross_of(keep)
    return {"dropped_top_n": k, "remaining_gross": g, "pass": g > 0}


def gate_v2(trades) -> dict:
    res = {}
    for sym in tb.SYMBOLS:
        n = net_of([t for t in trades if t["symbol"] != sym])
        res[f"ex_{sym}"] = n
    return {"nets": res, "pass": all(v > 0 for v in res.values())}


def variant_params(cfg: dict) -> list[dict]:
    out = []
    if cfg["kind"] == "donchian":
        for fe in SCALES:
            for fx in SCALES:
                out.append({**cfg, "entry_n": max(2, round(cfg["entry_n"] * fe)),
                            "exit_n": max(2, round(cfg["exit_n"] * fx)),
                            "_scale": (fe, fx)})
    elif cfg["kind"] == "emax":
        for f in SCALES:
            out.append({**cfg, "fast": max(2, round(cfg["fast"] * f)),
                        "slow": max(3, round(cfg["slow"] * f)), "_scale": (f,)})
    else:
        for f in SCALES:
            out.append({**cfg, "days": max(2, round(cfg["days"] * f)), "_scale": (f,)})
    return out


def gate_v3(cfg, bars, fund) -> dict:
    pts = []
    for v in variant_params(cfg):
        n = net_of(run_config(v, bars, fund))
        pts.append({"scale": v["_scale"], "net": n})
    frac = sum(1 for p in pts if p["net"] > 0) / len(pts)
    return {"points": pts, "frac_positive": frac, "pass": frac >= 0.75}


def slip_net(trades, k_ticks: int) -> float:
    extra = 0.0
    for t in trades:
        spec = tb.CONTRACT_SPECS[tb.SYMBOLS[t["symbol"]][1]]
        extra += 2 * (k_ticks - 1) * spec["tickSz"] * t["size"] * spec["ctVal"]
    return net_of(trades) - extra


def gate_v4(trades) -> dict:
    n3, n5 = slip_net(trades, 3), slip_net(trades, 5)
    return {"net_1tick": net_of(trades), "net_3tick": n3, "net_5tick_diag": n5,
            "pass": n3 > 0}


def gate_v5(trades) -> dict:
    df = pd.DataFrame(trades)
    df["entry_dt"] = pd.to_datetime(df["entry_time"])
    total = df["net_pnl_usd"].sum()
    yr = df.groupby(df["entry_dt"].dt.year)["net_pnl_usd"].sum()
    shares = {int(y): (float(v / total) if total != 0 else float("inf"))
              for y, v in yr.items()}
    a_ok = total > 0 and all(s <= 0.60 for s in shares.values())
    per = df["entry_dt"].dt.to_period("M")
    wins = []
    for st in ROLL_STARTS:
        w = df[(per >= st) & (per < st + 12)]["net_pnl_usd"].sum()
        wins.append(float(w))
    frac = sum(1 for w in wins if w > 0) / len(wins)
    return {"year_shares": shares, "year_share_ok": a_ok,
            "rolling12_n": len(wins), "rolling12_frac_positive": frac,
            "rolling12_ok": frac >= 0.60, "windows_net": wins,
            "pass": a_ok and frac >= 0.60}


# ── diagnostics ───────────────────────────────────────────────────────────────
def diag_d1(trades) -> dict:
    out = {}
    th = sum(t["hold_hours"] for t in trades)
    for side in ("long", "short"):
        g = [t for t in trades if t["side"] == side]
        out[side] = {"n": len(g), "gross": gross_of(g), "net": net_of(g),
                     "funding": float(sum(t["funding_usd"] for t in g)),
                     "time_share_pct": float(sum(t["hold_hours"] for t in g) / th * 100) if th else 0}
    return out


def diag_d2_buyhold(bars, fund) -> dict:
    per_sym, eq_frames = {}, []
    for name, (_, inst) in tb.SYMBOLS.items():
        spec = tb.CONTRACT_SPECS[inst]
        b = bars[(name, "4h")]
        ep_raw = float(b["open"].iloc[0])
        ep = ep_raw + spec["tickSz"]
        n = tb.calc_contracts(inst, ep_raw)
        xp = float(b["close"].iloc[-1]) - spec["tickSz"]
        entry_min = int(b["start_min"].iloc[0])   # buy at first bar OPEN moment
        exit_min = int(b["end_min"].iloc[-1])
        fee = -tb.FEE_TAKER * (ep + xp) * n * spec["ctVal"]
        fnd = -tb.funding_cost(fund[name], entry_min, exit_min, 1, n, spec["ctVal"])
        gross = (xp - ep) * n * spec["ctVal"]
        per_sym[name] = {"contracts": n, "entry": ep, "exit": xp, "gross": gross,
                         "fee": fee, "funding": fnd, "net": gross + fee + fnd}
        # 4h mark-to-market equity incl. accrued funding
        f = fund[name]
        cum_f = np.zeros(len(b))
        fmins = f["slot_min"].to_numpy()
        fpay = (f["rate"] * f["settle_px"]).to_numpy() * n * spec["ctVal"]
        idx = np.searchsorted(b["end_min"].to_numpy(), fmins, side="left")
        acc = np.zeros(len(b) + 1)
        for i_, p_ in zip(idx, fpay):
            if 0 <= i_ < len(b):
                acc[i_] += p_
        cum_f = np.cumsum(acc[:-1])
        eq = (b["close"].to_numpy() - ep) * n * spec["ctVal"] - cum_f \
            - tb.FEE_TAKER * ep * n * spec["ctVal"]
        eq_frames.append(eq)
    eq_port = np.sum(eq_frames, axis=0)
    peak = np.maximum.accumulate(eq_port)
    maxdd = float((peak - eq_port).max())
    net = float(sum(v["net"] for v in per_sym.values()))
    return {"per_symbol": per_sym, "net": net, "max_dd_usd": maxdd,
            "net_over_dd": net / maxdd if maxdd > 0 else float("inf"),
            "funding_total": float(sum(v["funding"] for v in per_sym.values()))}


def diag_d3_bootstrap(trades, rng) -> dict:
    x = np.array([t["net_pnl_usd"] for t in trades])
    means = rng.choice(x, size=(10_000, len(x)), replace=True).mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {"n": len(x), "mean": float(x.mean()),
            "ci95": [float(lo), float(hi)], "excludes_zero": bool(lo > 0 or hi < 0),
            "ci_unreliable_n_lt_50": len(x) < 50}


# ── main ──────────────────────────────────────────────────────────────────────
def main() -> int:
    L(f"run start (UTC): {datetime.now(timezone.utc).isoformat()}")
    L("DATA: database_mainnet.db (mode=ro) | engine: research_trend_baseline imported verbatim")
    L("cost convention identical to screen: taker both sides ±1 tick + real OKX 8h funding")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "diagnostics").mkdir(exist_ok=True)
    (OUT / "gates").mkdir(exist_ok=True)

    L("\nloading data via baseline engine ...")
    m1, bars, fund = {}, {}, {}
    for name, (db_sym, inst) in tb.SYMBOLS.items():
        m1[name] = tb.load_1m_utc(db_sym)
        bars[(name, "4h")] = tb.aggregate(m1[name], "4h")
        bars[(name, "1d")] = tb.aggregate(m1[name], "1d")
        fund[name] = tb.load_funding(inst, m1[name])

    bh = diag_d2_buyhold(bars, fund)
    (OUT / "diagnostics" / "buy_hold.json").write_text(json.dumps(bh, indent=2))
    L(f"B&H benchmark: net ${bh['net']:,.0f} | funding ${bh['funding_total']:,.0f} | "
      f"maxDD ${bh['max_dd_usd']:,.0f} | net/DD {bh['net_over_dd']:.2f}")

    rng = np.random.default_rng(SEED)
    results, d1_all, d3_all = [], {}, {}
    for cfg in tb.CONFIGS:
        cid = cfg["id"]
        trades = run_config(cfg, bars, fund)
        gates = {"V1": gate_v1(trades), "V2": gate_v2(trades),
                 "V3": gate_v3(cfg, bars, fund), "V4": gate_v4(trades),
                 "V5": gate_v5(trades)}
        verdict = "VALIDATED" if all(g["pass"] for g in gates.values()) else "FAIL"
        died = [k for k, g in gates.items() if not g["pass"]]
        d1 = diag_d1(trades)
        d3 = diag_d3_bootstrap(trades, rng)
        net = net_of(trades)
        dd_m = tb.metrics(trades)
        beta_flags = {
            "short_net_negative": d1["short"]["net"] < 0,
            "long_share_gt_90pct": (net > 0 and d1["long"]["net"] / net > 0.9),
            "risk_adj_below_bh": (net / dd_m["max_dd_usd"] if dd_m["max_dd_usd"] > 0
                                  else float("inf")) < bh["net_over_dd"],
        }
        d1_all[cid], d3_all[cid] = d1, d3
        (OUT / "gates" / f"{cid}.json").write_text(json.dumps(
            {"config": cfg, "verdict": verdict, "died_at": died,
             "gates": gates, "beta_flags": beta_flags}, indent=2, default=float))
        results.append({"id": cid, "family": cfg["family"], "tf": cfg["tf"],
                        "verdict": verdict, "died_at": died, "net": net,
                        "net_over_dd": net / dd_m["max_dd_usd"] if dd_m["max_dd_usd"] else None,
                        "beta_flags": beta_flags, "d3": d3,
                        "short_net": d1["short"]["net"], "long_net": d1["long"]["net"]})
        L(f"[{cid}] {verdict}{' died:' + ','.join(died) if died else ''} | "
          f"net ${net:,.0f} | L ${d1['long']['net']:,.0f} / S ${d1['short']['net']:,.0f} | "
          f"V3 {gates['V3']['frac_positive']:.0%} | V4(3t) ${gates['V4']['net_3tick']:,.0f} | "
          f"V5 roll {gates['V5']['rolling12_frac_positive']:.0%} yrOK {gates['V5']['year_share_ok']} | "
          f"CI [{d3['ci95'][0]:.1f},{d3['ci95'][1]:.1f}]{'!' if d3['ci_unreliable_n_lt_50'] else ''}")

    (OUT / "diagnostics" / "long_short.json").write_text(json.dumps(d1_all, indent=2))
    (OUT / "diagnostics" / "bootstrap.json").write_text(json.dumps(d3_all, indent=2))
    (OUT / "summary.json").write_text(json.dumps(results, indent=2, default=float))
    nv = sum(1 for r in results if r["verdict"] == "VALIDATED")
    L(f"\nVALIDATED: {nv} | FAIL: {len(results) - nv}")
    L(f"run end (UTC): {datetime.now(timezone.utc).isoformat()}")
    (OUT / "run_log.txt").write_text("\n".join(LOG) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
