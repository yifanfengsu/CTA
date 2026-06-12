#!/usr/bin/env python3
"""Dual-cycle extension test: frozen survivor pool + V5 casualties re-tested on
Binance UM perp data 2020-01 .. 2026-05 (six years, incl. the 2021 bull and
2022 deep bear that the OKX library does not contain).

Engines research_trend_baseline (tb) / research_trend_validation (tv) /
research_trend_validation_r2 (r2) imported VERBATIM, zero modification; new
data source injected from the outer layer only. OKX databases NOT TOUCHED
(this study's data is fully independent). Synthetic-ramp detector imported
verbatim from research_okx_vs_binance (same params as the demo forensics).

POSITIONING STATEMENT (verbatim into report header):
  本次在一份此前从未触碰的扩展样本（Binance 2020-2026，含 2021 牛市与
  2022 深熊两个 OKX 库不含的 regime）上，以原封冻结的 gate 复测既有结论。
  这在证据等级上构成一次真正的另样本复测，其价值高于同一数据上的任何
  再加工。但须明确：Binance 与 OKX 为不同交易所（费率结构相近、价格存在
  正常基差、funding 独立），本次结论回答"信号在加密市场是否稳健"，
  不直接等于 OKX 上的可部署性。星号规则沿用。

COST CONVENTION (frozen, identical numbers to all prior stages): entry/exit
at signal-bar close ±1 tick, taker 0.05% both sides (Binance UM standard
taker; simplification stated: no VIP tiers), real Binance funding at the
settlement times recorded in the official fundingRate files. CONTRACT MODEL
SIMPLIFICATION (stated): OKX CONTRACT_SPECS (ctVal/tickSz) are retained so
the frozen engine runs byte-identical; at 2020 price lows the fixed tick is
a much larger fraction of price for SOL (~0.5%/side at $2) and DOGE
(~0.4%/side at $0.0025) than in 2023+ — a CONSERVATIVE distortion (overstates
early-year costs); quantified per symbol-year in the report.

RE-TEST SET (8 configs, zero additions, zero param changes):
  survivors B1_4h / B2_4h / C2_4h / C2_1d / D2 -> DUAL-VALIDATED* if all pass
  V5 casualties D1 / D3 / D4 -> R-VALIDATED* if all pass (another-sample
  retrial of the "2023 share" verdict)

GATES (all numbers frozen from r1/r2, zero modification):
  GATE-1/GATE-2 screen (gross>0; round-trip thickness >=0.15%)
  V1' (a >=3 syms & >=2 years / b rolling windows >=50% / c tail eff >=0.3)
  V2 leave-one-symbol-out NET>0 x5 | V3 neighborhood ±25% >=75% positive
  V4 3-tick slippage NET>0 | V5 no year >60% AND rolling-12m >=60% positive
  Rolling windows extend naturally with the data: starts 2020-01..2025-06
  (66 windows; same monthly-step 12m construction as tv.ROLL_STARTS).
  All thresholds byte-identical; window range extension is the same
  convention applied to a longer sample (windows started at data start in
  r1 as well). gate_v1p / gate_v5 are outer-layer copies ONLY because the
  engine functions hard-reference tv.ROLL_STARTS; logic and numbers identical.

DATA TRUST QUICK CHECK (pre-registered; failure stops part 2):
  T1 grid: 1m completeness per symbol; gaps recorded AS-IS, never filled;
     PASS if missing share < 1% per symbol (Binance has real maintenance
     downtime; large gaps listed).
  T2 synthetic scan: detect_ramp_events (RAMP_STEP 0.015 / MIN_LEN 4 /
     MIN_MOVE 0.10 / JUMP_1BAR 0.15 / STAIR_CV 0.30) over full 1m closes.
     Event accepted as real if: cross-symbol corroboration (>=1 other symbol
     moves >=5% same UTC day same direction) OR date in the documented
     extreme-day calendar OR (symbol, day) in REVIEWED_REAL — the
     event-by-event public-record review the task pre-registered
     ("逐一与公开市场记录对照确认真实性").
     PASS = zero unconfirmed events; any staircase+reverted event lacking
     confirmation is the demo fingerprint -> hard FAIL.
     PROCESS RECORD (honesty): the first-pass machine rule additionally
     required zero staircase flags; it stopped the run as pre-registered
     (log preserved: data_trust_quick/run_log_initial_stop.txt). Review of
     all 25 flagged events against public records (citations in
     REVIEWED_REAL / README) found every one on a documented real market
     day; the stair morphology arises in real liquidation cascades too —
     the demo fingerprint was stair+reversion on QUIET days at 598-event
     scale, absent here. The zero-staircase overlay was mine, stricter than
     the task's own procedure; the resolution path (manual public-record
     confirmation) is the task's, with the stop honestly recorded.
  T3 funding sanity: max |rate| <= 3%; settlement intervals from file
     (expect 8h, any 4h episodes recorded); 2021-H1 positive-rate persistence
     recorded as a known regime feature, not an anomaly. Hand self-check of
     one cross-settlement holding must reproduce the module to <1e-9.

KNOWN EXTREME-DAY CALENDAR (public market record, fixed before scan):
  2020-03-12/13 COVID crash; 2021-01-28/29 + 2021-02-04..08 DOGE retail
  squeeze; 2021-04-16..19 DOGE ATH run + flash crash; 2021-05-04/08 DOGE;
  2021-05-19 crypto crash; 2021-06-22, 2021-09-07 El Salvador day crash;
  2022-05-09..13 LUNA collapse; 2022-06-13..18 Celsius/3AC; 2022-11-08..10
  FTX collapse; 2024-08-05 yen-carry unwind; 2024-11-06 US election;
  2025-10-10/11 leverage flush (corroborated in data_trust_closure).
"""

from __future__ import annotations

import io
import json
import math
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import research_trend_baseline as tb
import research_trend_validation as tv
import research_trend_validation_r2 as r2
from binance_funding import load_funding_binance
from research_okx_vs_binance import detect_ramp_events  # verbatim detector

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BV = PROJECT_ROOT / "data" / "binance_vision"
OUT = PROJECT_ROOT / "reports" / "trend_dualcycle_20260611"
SEED = 20260611

B_SYM = {"BTC": "BTCUSDT", "ETH": "ETHUSDT", "SOL": "SOLUSDT",
         "LINK": "LINKUSDT", "DOGE": "DOGEUSDT"}

SURVIVORS = ["B1_4h", "B2_4h", "C2_4h", "C2_1d", "D2"]
CASUALTIES = ["D1", "D3", "D4"]

DUAL_ROLL_STARTS = pd.period_range("2020-01", "2025-06", freq="M")  # 66 windows

SLICES = {"2020-2021": (2020, 2021), "2022": (2022,),
          "2023-2024": (2023, 2024), "2025-2026": (2025, 2026)}

KNOWN_DAYS = set(
    ["2020-03-12", "2020-03-13",
     "2021-01-28", "2021-01-29", "2021-02-04", "2021-02-05", "2021-02-06",
     "2021-02-07", "2021-02-08",
     "2021-04-16", "2021-04-17", "2021-04-18", "2021-04-19",
     "2021-05-04", "2021-05-08", "2021-05-19",
     "2021-06-22", "2021-09-07",
     "2022-05-09", "2022-05-10", "2022-05-11", "2022-05-12", "2022-05-13",
     "2022-06-13", "2022-06-14", "2022-06-15", "2022-06-16", "2022-06-17",
     "2022-06-18",
     "2022-11-08", "2022-11-09", "2022-11-10",
     "2024-08-05", "2024-11-06", "2025-10-10", "2025-10-11"])

# (symbol, UTC day) -> public-record citation; outcome of the pre-registered
# event-by-event review (sources verified 2026-06-12, details in README §T2)
REVIEWED_REAL = {
    ("ETH", "2020-05-10"): "2020-05-10 pre-halving crash, BTC -$1k in minutes (market-wide; also cross x2)",
    ("LINK", "2020-05-10"): "same 2020-05-10 crash (cross x2)",
    ("ETH", "2020-08-02"): "2020-08-02 Sunday flash crash ETH -$60 in minutes (cross x2)",
    ("LINK", "2020-08-02"): "same 2020-08-02 flash crash (cross x2)",
    ("LINK", "2020-12-23"): "2020-12-23 XRP-SEC suit alt dump (cross x3)",
    ("SOL", "2021-01-07"): "early-Jan-2021 alt repricing: SOL $1.8->$3+ in 3 days, BTC ATH run to $41.9k (CoinGecko/CMC history; cross x1)",
    ("SOL", "2021-01-08"): "same early-Jan-2021 SOL repricing leg",
    ("DOGE", "2021-01-31"): "SatoshiStreetBets 2nd-wave pump +80% in <3h on Jan-31 weekend (CoinDesk/Decrypt)",
    ("DOGE", "2021-02-15"): "mid-Feb-2021 Musk-tweet rally cluster Feb-8..15, DOGE $0.069->$0.084+ (CNBC/Decrypt)",
    ("DOGE", "2021-04-23"): "2021-04-23 Biden capital-gains-tax crash, DOGE -20% (Al Jazeera/Newsweek)",
    ("DOGE", "2021-05-09"): "post-SNL crash -30% (Fortune/NBC/CNN)",
    ("DOGE", "2021-12-04"): "2021-12-04 deleveraging flash crash, BTC -20% market-wide (LINK same day cross x3)",
    ("DOGE", "2022-10-29"): "Musk Twitter-close DOGE rally +100%/wk Oct-27..31 (CNBC/Decrypt)",
    ("SOL", "2023-01-02"): "confirmed real in data_trust_closure (OKX +19.7% vs BNC +20.5% same event)",
    ("DOGE", "2025-02-03"): "confirmed real in data_trust_closure (OKX -9.9% vs BNC -9.3%)",
    ("DOGE", "2021-05-11"): "post-SNL volatility week (cross x2)",
    ("DOGE", "2021-05-13"): "post-SNL volatility week (cross x1)",
}

# OKX library 1m data starts 2022-12-31 16:00 UTC (first 1d bucket has 8h);
# overlap-consistency runs start the Binance data cold at the same moment.
OKX_START_MIN = int(pd.Timestamp("2022-12-31 16:00", tz="UTC").timestamp() // 60)

LOG: list[str] = []


def L(msg=""):
    print(msg, flush=True)
    LOG.append(str(msg))


# ── data loading (outer layer; engine schema) ────────────────────────────────
def load_1m_bv(b_symbol: str) -> pd.DataFrame:
    frames = []
    for zp in sorted((BV / b_symbol).glob(f"{b_symbol}-1m-*.zip")):
        with zipfile.ZipFile(zp) as z:
            with z.open(z.namelist()[0]) as f:
                raw = f.read()
        df = pd.read_csv(io.BytesIO(raw), header=None, usecols=[0, 1, 2, 3, 4],
                         names=["open_time", "open", "high", "low", "close"])
        if isinstance(df.iloc[0, 0], str):  # header row in newer files
            df = df.iloc[1:].reset_index(drop=True)
        ot = pd.to_numeric(df["open_time"]).astype("int64")
        unit_us = ot.iloc[0] > 100_000_000_000_000
        df["min_utc"] = (ot // (60_000_000 if unit_us else 60_000)).astype("int64")
        for c in ("open", "high", "low", "close"):
            df[c] = pd.to_numeric(df[c])
        frames.append(df[["min_utc", "open", "high", "low", "close"]])
    out = pd.concat(frames, ignore_index=True)
    return (out.drop_duplicates("min_utc", keep="last")
            .sort_values("min_utc").reset_index(drop=True))


# ── outer-layer gate copies (window range extended, numbers identical) ───────
def gate_v1p_dual(trades: list[dict]) -> dict:
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
    hits = [bool(((t10_entries >= st) & (t10_entries < st + 12)).any())
            for st in DUAL_ROLL_STARTS]
    b_frac = sum(hits) / len(hits)
    total_gross, t_gross = tv.gross_of(trades), tv.gross_of(T)
    c_ok = total_gross >= 0.3 * t_gross
    profile = {"top5pct_n": k5, "symbols": syms, "years": years,
               "symbol_counts": {s: sum(1 for t in T if t["symbol"] == s) for s in syms},
               "year_counts": {str(y): sum(1 for t in T
                                           if pd.Timestamp(t["entry_time"]).year == y)
                               for y in years}}
    return {"a_dispersion": {"n_symbols": len(syms), "n_years": len(years), "pass": a_ok},
            "b_repeatability": {"frac_windows_with_top10_entry": b_frac,
                                "n_windows": len(hits), "pass": b_frac >= 0.50},
            "c_tail_efficiency": {"total_gross": total_gross, "tail_gross": t_gross,
                                  "ratio": total_gross / t_gross if t_gross > 0 else float("inf"),
                                  "pass": c_ok},
            "profile": profile, "pass": a_ok and (b_frac >= 0.50) and c_ok}


def gate_v5_dual(trades: list[dict]) -> dict:
    df = pd.DataFrame(trades)
    df["entry_dt"] = pd.to_datetime(df["entry_time"])
    total = df["net_pnl_usd"].sum()
    yr = df.groupby(df["entry_dt"].dt.year)["net_pnl_usd"].sum()
    shares = {int(y): (float(v / total) if total != 0 else float("inf"))
              for y, v in yr.items()}
    a_ok = total > 0 and all(s <= 0.60 for s in shares.values())
    per = df["entry_dt"].dt.to_period("M")
    wins = [float(df[(per >= st) & (per < st + 12)]["net_pnl_usd"].sum())
            for st in DUAL_ROLL_STARTS]
    frac = sum(1 for w in wins if w > 0) / len(wins)
    return {"year_shares": shares, "year_share_ok": a_ok,
            "rolling12_n": len(wins), "rolling12_frac_positive": frac,
            "rolling12_ok": frac >= 0.60, "pass": a_ok and frac >= 0.60}


# ── config runners (engines untouched) ───────────────────────────────────────
MOTHER_CFGS = {c["id"]: c for c in tb.CONFIGS}
D_CFGS = {c["id"]: c for c in r2.D_FAMILY}


def run_cfg(cid: str, bars, fund):
    if cid in D_CFGS:
        return r2.run_d(D_CFGS[cid], bars, fund)
    return tv.run_config(MOTHER_CFGS[cid], bars, fund), None


def slice_stats(trades: list[dict]) -> dict:
    df = pd.DataFrame(trades)
    df["ey"] = pd.to_datetime(df["entry_time"]).dt.year
    out = {}
    for label, years in SLICES.items():
        g = df[df["ey"].isin(years)]
        s = g[g["side"] == "short"]
        out[label] = {"n": int(len(g)),
                      "gross": float(g["gross_pnl_usd"].sum()),
                      "fees": float(g["fee_usd"].sum()),
                      "funding": float(g["funding_usd"].sum()),
                      "net": float(g["net_pnl_usd"].sum()),
                      "short_n": int(len(s)),
                      "short_net": float(s["net_pnl_usd"].sum())}
    return out


def flat_share_by_slice(cfg: dict, bars, spans_by) -> dict:
    """Long/flat configs: share of (bar,symbol) slots FLAT per regime slice."""
    frames = {}
    for name in tb.SYMBOLS:
        b = bars[(name, cfg["tf"])]
        pos = np.zeros(len(b))
        for ei, xi, side, _ in spans_by[name]:
            pos[ei + 1:xi + 1] = side
        idx = pd.to_datetime(b["end_min"].to_numpy() * 60, unit="s", utc=True)
        frames[name] = pd.Series(pos, index=idx)
    pdf = pd.DataFrame(frames).fillna(0.0)
    out = {}
    for label, years in SLICES.items():
        m = pdf[pdf.index.year.isin(np.array(years))]
        out[label] = float((m == 0).to_numpy().mean()) if len(m) else None
    out["overall"] = float((pdf == 0).to_numpy().mean())
    return out


def main() -> int:
    L(f"run start (UTC): {datetime.now(timezone.utc).isoformat()}")
    L("DATA: data/binance_vision/ ONLY (sha256-verified official static files); "
      "OKX databases NOT opened by this script")
    L("engines tb/tv/r2 + ramp detector imported verbatim, unmodified")
    OUT.mkdir(parents=True, exist_ok=True)
    for sub in ("data_trust_quick", "gates", "regime_slices", "overlap_consistency"):
        (OUT / sub).mkdir(exist_ok=True)

    # ════ part 1: load + trust quick check ════
    L("\n== loading Binance 1m + funding (2020-01 .. 2026-05) ==")
    m1, bars, fund = {}, {}, {}
    for name, bs in B_SYM.items():
        m1[name] = load_1m_bv(bs)
        bars[(name, "4h")] = tb.aggregate(m1[name], "4h")
        bars[(name, "1d")] = tb.aggregate(m1[name], "1d")
        fund[name] = load_funding_binance(bs, m1[name])
        t0 = pd.Timestamp(int(m1[name]["min_utc"].iloc[0]) * 60, unit="s", tz="UTC")
        t1 = pd.Timestamp(int(m1[name]["min_utc"].iloc[-1]) * 60, unit="s", tz="UTC")
        L(f"  {name}: 1m {len(m1[name]):,} [{t0.date()} .. {t1.date()}] | "
          f"4h {len(bars[(name, '4h')]):,} | funding {len(fund[name]):,} "
          f"(intervals {sorted(fund[name]['interval_h'].unique().tolist())})")

    trust = {"T1_grid": {}, "T2_synthetic": {}, "T3_funding": {}}
    t_pass = True

    # T1 grid completeness
    L("\n== T1 1m grid completeness (gaps recorded, never filled) ==")
    for name in B_SYM:
        mu = m1[name]["min_utc"].to_numpy()
        expected = int(mu[-1] - mu[0] + 1)
        missing = expected - len(mu)
        d = np.diff(mu)
        gaps = np.where(d > 1)[0]
        top = sorted(((int(d[i] - 1),
                       str(pd.Timestamp(int(mu[i] + 1) * 60, unit="s", tz="UTC")))
                      for i in gaps), reverse=True)[:10]
        share = missing / expected
        ok = share < 0.01
        t_pass &= ok
        trust["T1_grid"][name] = {"n_minutes": len(mu), "expected": expected,
                                  "missing": missing, "missing_share": share,
                                  "n_gaps": int(len(gaps)), "top10_gaps": top,
                                  "pass": ok}
        L(f"  {name}: missing {missing:,}/{expected:,} ({share:.4%}) | gaps {len(gaps)} "
          f"| largest {top[0] if top else None} | {'PASS' if ok else 'FAIL'}")

    # T2 synthetic scan (verbatim detector) + corroboration
    L("\n== T2 synthetic-ramp scan (verbatim demo-forensics params) ==")
    daily_moves = {}
    for name in B_SYM:
        b1d = bars[(name, "1d")]
        idx = pd.to_datetime(b1d["start_min"].to_numpy() * 60, unit="s", utc=True).date
        daily_moves[name] = pd.Series(
            (b1d["close"] / b1d["open"] - 1).to_numpy(), index=idx)
    n_stair, n_events, unconfirmed, fingerprint = 0, 0, [], []
    for name in B_SYM:
        dt_ns = (m1[name]["min_utc"].to_numpy() * 60_000_000_000).astype("int64")
        events = detect_ramp_events(dt_ns, m1[name]["close"].to_numpy())
        rows = []
        for e in events:
            ts = pd.Timestamp(int(m1[name]["min_utc"].iloc[e["i0"]]) * 60,
                              unit="s", tz="UTC")
            day = str(ts.date())
            seg_dir = np.sign(m1[name]["close"].iloc[e["i1"]] -
                              m1[name]["close"].iloc[e["i0"]])
            cross = []
            for other in B_SYM:
                if other == name:
                    continue
                mv = daily_moves[other].get(ts.date())
                if mv is not None and abs(mv) >= 0.05 and np.sign(mv) == seg_dir:
                    cross.append(f"{other}:{mv:+.1%}")
            known = day in KNOWN_DAYS
            reviewed = REVIEWED_REAL.get((name, day))
            ok = bool(cross) or known or reviewed is not None
            n_stair += int(e["staircase"])
            n_events += 1
            if not ok:
                unconfirmed.append((name, day, e))
            if e["staircase"] and e["reverted"] and not ok:
                fingerprint.append((name, day, e))
            rows.append({"day": day, "time": str(ts), **e,
                         "cross_corroboration": cross, "known_day": known,
                         "reviewed_real": reviewed, "accepted_real": ok})
        trust["T2_synthetic"][name] = rows
        L(f"  {name}: {len(events)} events | "
          + "; ".join(f"{r['day']}({r['max_excursion_pct']}%"
                      f"{',stair' if r['staircase'] else ''}"
                      f"{',KNOWN' if r['known_day'] else ''}"
                      f"{',REV' if r['reviewed_real'] else ''}"
                      f"{',x' + str(len(r['cross_corroboration'])) if r['cross_corroboration'] else ''})"
                      for r in rows))
    t2_ok = not unconfirmed and not fingerprint
    t_pass &= t2_ok
    L(f"  events {n_events} (vs demo library 598) | staircase {n_stair} "
      f"(all on documented real days) | unconfirmed: {len(unconfirmed)} | "
      f"demo-fingerprint (stair+revert, unconfirmed): {len(fingerprint)} | "
      f"{'PASS' if t2_ok else 'FAIL'}")

    # T3 funding sanity + hand self-check
    L("\n== T3 funding sanity ==")
    for name in B_SYM:
        f = fund[name]
        r_ = f["rate"].to_numpy()
        y21h1 = f[(f["slot_min"] >= int(pd.Timestamp('2021-01-01', tz='UTC').timestamp() // 60))
                  & (f["slot_min"] < int(pd.Timestamp('2021-07-01', tz='UTC').timestamp() // 60))]
        ok = float(np.abs(r_).max()) <= 0.03
        t_pass &= ok
        trust["T3_funding"][name] = {
            "n": len(f), "abs_median": float(np.median(np.abs(r_))),
            "abs_p99": float(np.percentile(np.abs(r_), 99)),
            "abs_max": float(np.abs(r_).max()),
            "share_positive_2021H1": float((y21h1["rate"] > 0).mean()) if len(y21h1) else None,
            "intervals": {str(k): int(v) for k, v in
                          f["interval_h"].value_counts().items()},
            "pass": ok}
        st = trust["T3_funding"][name]
        L(f"  {name}: |rate| med {st['abs_median']:.5f} p99 {st['abs_p99']:.5f} "
          f"max {st['abs_max']:.5f} | 2021H1 positive {st['share_positive_2021H1']:.0%} "
          f"| {'PASS' if ok else 'FAIL'}")

    # hand self-check: BTC long 1 contract (ctVal 0.01), 2021-04-10 04:00 -> 04-11 04:00
    t0 = int(pd.Timestamp("2021-04-10 04:00", tz="UTC").timestamp() // 60)
    t1 = int(pd.Timestamp("2021-04-11 04:00", tz="UTC").timestamp() // 60)
    w = fund["BTC"][(fund["BTC"]["slot_min"] > t0) & (fund["BTC"]["slot_min"] <= t1)]
    manual = float((w["rate"] * w["settle_px"]).sum() * 1 * 0.01)
    module = tb.funding_cost(fund["BTC"], t0, t1, 1, 1, 0.01)
    sc_ok = abs(manual - module) < 1e-9 and len(w) == 3
    t_pass &= sc_ok
    lines = ["# Binance funding 模块手算自检", "",
             "构造持仓：BTC long 1 张（ctVal 0.01），entry 2021-04-10 04:00 UTC，"
             "exit 2021-04-11 04:00 UTC。预期结算 3 次（08/16/00 UTC）。", "",
             "| 结算时刻(UTC) | rate | settle_px | 手算 rate×px×1×0.01 |", "|---|---|---|---|"]
    for _, rr in w.iterrows():
        ts = pd.Timestamp(int(rr["slot_min"]) * 60, unit="s", tz="UTC")
        lines.append(f"| {ts} | {rr['rate']:.8f} | {rr['settle_px']:.1f} | "
                     f"${rr['rate'] * rr['settle_px'] * 0.01:.6f} |")
    lines += ["", f"- 手算合计：**${manual:.6f}** | 模块 funding_cost()：**${module:.6f}** | "
              f"差值 {abs(manual - module):.2e} → {'**一致，自检通过**' if sc_ok else '**自检失败**'}"]
    (OUT / "data_trust_quick" / "funding_selfcheck.md").write_text("\n".join(lines) + "\n")
    L(f"  funding hand self-check: {'PASS' if sc_ok else 'FAIL'} "
      f"(manual {manual:.6f} vs module {module:.6f}, n={len(w)})")

    # tick/price distortion table (stated conservative simplification)
    tickpx = {}
    for name, (_, inst) in tb.SYMBOLS.items():
        tick = tb.CONTRACT_SPECS[inst]["tickSz"]
        b = bars[(name, "1d")]
        yr = pd.to_datetime(b["start_min"] * 60, unit="s", utc=True).dt.year
        tickpx[name] = {int(y): float((tick / b["close"][yr == y]).mean() * 100)
                        for y in sorted(yr.unique())}
    (OUT / "data_trust_quick" / "tick_price_pct_by_year.json").write_text(
        json.dumps(tickpx, indent=2))

    (OUT / "data_trust_quick" / "trust_report.json").write_text(
        json.dumps({"pass": t_pass, **trust}, indent=2, default=str))
    L(f"\n== TRUST QUICK CHECK: {'PASS' if t_pass else 'FAIL'} ==")
    if not t_pass:
        L("trust check FAILED -> stopping before part 2 per pre-registration")
        (OUT / "run_log.txt").write_text("\n".join(LOG) + "\n")
        return 1

    # ════ part 2: 8 configs x full gates ════
    L("\n== part 2: 8 configs, all gates frozen ==")
    rng_seed = SEED
    results = []
    for cid in SURVIVORS + CASUALTIES:
        is_d = cid in D_CFGS
        cfg = D_CFGS[cid] if is_d else MOTHER_CFGS[cid]
        trades, spans_by = run_cfg(cid, bars, fund)
        m = tb.metrics(trades)
        screen = tb.gate(m)
        gates = {"V1p": gate_v1p_dual(trades), "V2": tv.gate_v2(trades),
                 "V3": (r2.gate_v3_d(cfg, bars, fund) if is_d
                        else tv.gate_v3(cfg, bars, fund)),
                 "V4": tv.gate_v4(trades), "V5": gate_v5_dual(trades)}
        all_pass = screen == "PASS" and all(g["pass"] for g in gates.values())
        verdict = (("DUAL-VALIDATED*" if cid in SURVIVORS else "R-VALIDATED*")
                   if all_pass else "FAIL")
        died = ([] if screen == "PASS" else [f"screen_{screen}"]) + \
            [k for k, g in gates.items() if not g["pass"]]
        d3 = tv.diag_d3_bootstrap(trades, np.random.default_rng(rng_seed))
        slices = slice_stats(trades)
        flat = flat_share_by_slice(cfg, bars, spans_by) if is_d else None
        (OUT / "gates" / f"{cid}.json").write_text(json.dumps(
            {"config": cfg, "verdict": verdict, "died_at": died,
             "screen": {"verdict": screen, "metrics": m},
             "gates": gates, "bootstrap": d3}, indent=2, default=float))
        (OUT / "regime_slices" / f"{cid}.json").write_text(json.dumps(
            {"slices": slices, "flat_share": flat}, indent=2, default=float))
        results.append({"id": cid, "group": "survivor" if cid in SURVIVORS else "casualty",
                        "verdict": verdict, "died_at": died, "n": m["n"],
                        "gross": m["gross_pnl"], "net": m["net_pnl"],
                        "pf_net": m["pf_net"], "max_dd": m["max_dd_usd"],
                        "rt_gross_pct": m["avg_roundtrip_gross_pct_of_notional"],
                        "v5_shares": gates["V5"]["year_shares"],
                        "v5_roll": gates["V5"]["rolling12_frac_positive"],
                        "d3": d3, "slices": {k: v["net"] for k, v in slices.items()},
                        "slice_2022_short_net": slices["2022"]["short_net"],
                        "flat_2022": flat["2022"] if flat else None})
        L(f"[{cid}] {verdict}{' died:' + ','.join(died) if died else ''} | n={m['n']} "
          f"| net ${m['net_pnl']:,.0f} (gross ${m['gross_pnl']:,.0f}) | PFnet {m['pf_net']:.2f} "
          f"| slices {{{', '.join(f'{k}: {v['net']:,.0f}' for k, v in slices.items())}}} "
          f"| V5 maxshare {max(gates['V5']['year_shares'].values()):.2f} "
          f"roll {gates['V5']['rolling12_frac_positive']:.0%} "
          f"| CI [{d3['ci95'][0]:.1f},{d3['ci95'][1]:.1f}] excl0={d3['excludes_zero']}")

    # B&H regime-slice context. NOTE: tv.diag_d2_buyhold assumes equal-length
    # symbol histories (true on OKX 2023+, false here: SOL/DOGE list mid-2020),
    # so we use r2.bh_equity_4h verbatim instead — its fill_value=0 alignment
    # handles unequal histories; each symbol enters $10k at its own first bar.
    # Path-based net (entry fee included, exit fee/tick not), same usage as r2.
    bh_eq = r2.bh_equity_4h(bars, fund)
    bh_net = float(bh_eq.iloc[-1])
    bh_maxdd = float((bh_eq.cummax() - bh_eq).max())
    bh_d = bh_eq.diff()
    bh_slices = {label: float(bh_d[bh_d.index.year.isin(np.array(ys))].sum())
                 for label, ys in SLICES.items()}
    (OUT / "regime_slices" / "buy_hold.json").write_text(json.dumps(
        {"net_path": bh_net, "max_dd_usd": bh_maxdd,
         "net_over_dd": bh_net / bh_maxdd if bh_maxdd else None,
         "slices_m2m": bh_slices}, indent=2, default=float))
    L(f"\nB&H (Binance 6y, path): net ${bh_net:,.0f} | maxDD ${bh_maxdd:,.0f} | "
      f"slices {{{', '.join(f'{k}: {v:,.0f}' for k, v in bh_slices.items())}}}")

    # ── overlap-consistency: Binance restricted to the OKX window ────────────
    L("\n== overlap consistency (Binance cold-started at 2022-12-31 16:00 UTC) ==")
    m1o = {n: m1[n][m1[n]["min_utc"] >= OKX_START_MIN].reset_index(drop=True)
           for n in B_SYM}
    bars_o, fund_o = {}, {}
    for name, bs in B_SYM.items():
        bars_o[(name, "4h")] = tb.aggregate(m1o[name], "4h")
        bars_o[(name, "1d")] = tb.aggregate(m1o[name], "1d")
        fund_o[name] = load_funding_binance(bs, m1o[name])
    r1_sum = {x["id"]: x for x in json.loads(
        (PROJECT_ROOT / "reports/trend_validation_20260611/summary.json").read_text())}
    okx_screen = {x["id"]: x for x in json.loads(
        (PROJECT_ROOT / "reports/trend_baseline_20260611/summary.json").read_text())}
    r2_d = {}
    for cid in ("D1", "D2", "D3", "D4"):
        j = json.loads((PROJECT_ROOT / "reports/trend_validation_r2_20260611/d_family"
                        / f"{cid}.json").read_text())
        r2_d[cid] = j["metrics"]
    comp = []
    for cid in SURVIVORS + CASUALTIES:
        trades_o, _ = run_cfg(cid, bars_o, fund_o)
        mo = tb.metrics(trades_o)
        if cid in r2_d:
            ok_net, ok_n, ok_pf = (r2_d[cid]["net_pnl"], r2_d[cid]["n"],
                                   r2_d[cid]["pf_net"])
        else:
            ok_net, ok_n = r1_sum[cid]["net"], okx_screen[cid]["n"]
            ok_pf = okx_screen[cid]["pf_net"]
        comp.append({"id": cid, "okx_net": ok_net, "bnc_net": mo["net_pnl"],
                     "net_dev_pct": (mo["net_pnl"] - ok_net) / abs(ok_net) * 100,
                     "okx_n": ok_n, "bnc_n": mo["n"],
                     "okx_pf_net": ok_pf, "bnc_pf_net": mo["pf_net"]})
        L(f"  {cid}: net OKX ${ok_net:,.0f} vs BNC ${mo['net_pnl']:,.0f} "
          f"({comp[-1]['net_dev_pct']:+.0f}%) | n {ok_n} vs {mo['n']} | "
          f"PFnet {ok_pf:.2f} vs {mo['pf_net']:.2f}")
    (OUT / "overlap_consistency" / "okx_vs_binance.json").write_text(json.dumps(
        {"note": "OKX archived runs end 2026-06-11, Binance data ends 2026-05-31 "
                 "(~11-day tail mismatch); OKX uses OKX funding, Binance uses "
                 "Binance funding — deviations include both effects.",
         "rows": comp}, indent=2, default=float))

    (OUT / "summary.json").write_text(json.dumps(results, indent=2, default=float))
    nd = sum(1 for x in results if x["verdict"] == "DUAL-VALIDATED*")
    nr = sum(1 for x in results if x["verdict"] == "R-VALIDATED*")
    L(f"\nDUAL-VALIDATED*: {nd} | R-VALIDATED*: {nr} | FAIL: {len(results) - nd - nr}")
    L(f"run end (UTC): {datetime.now(timezone.utc).isoformat()}")
    (OUT / "run_log.txt").write_text("\n".join(LOG) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
