#!/usr/bin/env python3
"""OKX mainnet DB vs Binance vision — full-period external cross-validation.

Read-only on .vntrader/database_mainnet.db (sqlite mode=ro). Binance side from
data/binance_vision/ monthly zips (sha256-verified, see manifest there).
Writes ONLY to reports/regime/data_trust_closure_20260611/.

Alignment convention (explicit, also in report header):
  * OKX DB datetime = naive Asia/Shanghai local time of bar OPEN
    -> UTC epoch-minute = naive_minute - 480.
  * Binance open_time = UTC epoch ms (files >= 2025-01: microseconds);
    normalized to epoch-minute of bar OPEN.
  * Join on UTC epoch-minute (inner). Both are last-price 1m klines.
  * OKX and Binance are different venues: a perp basis is NORMAL. The goal is
    "same real market" (no systematic dislocation, no synthetic patterns),
    not tick equality.

PRE-REGISTERED VERDICT GATES (task 2d — written before any result was seen,
may not be modified after results):
  PASS requires ALL of:
   G1 synthetic-pattern scan on OKX mainnet detects 0 events, OR every event
      is corroborated by Binance moving in the same direction in the same
      window (corroboration rule, fixed here: sign(net move) equal AND
      Binance |max excursion| >= 0.5 x OKX |max excursion|).
   G2 no month shows sustained one-sided dislocation > 0.5%
      (|monthly median signed basis| > 0.5%).
   G3 per-symbol full-period median |close relative deviation| < 0.1%.
  Any gate failed -> FAIL, stop and report; no explanatory waivers.

Synthetic detector: detect_ramp_events ported VERBATIM (params and logic) from
scripts/research_v2b_dd_diagnosis.py step-4 (the detector that found the 598
demo ramps), so mainnet counts are directly comparable to the demo library's.
"""

from __future__ import annotations

import io
import json
import sqlite3
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DB_MAIN = PROJECT_ROOT / ".vntrader" / "database_mainnet.db"
BV_ROOT = PROJECT_ROOT / "data" / "binance_vision"
OUT = PROJECT_ROOT / "reports/regime/data_trust_closure_20260611"

SYMBOLS = {  # name -> (okx db symbol, binance symbol)
    "BTC": ("BTCUSDT_SWAP_OKX", "BTCUSDT"),
    "ETH": ("ETHUSDT_SWAP_OKX", "ETHUSDT"),
    "SOL": ("SOLUSDT_SWAP_OKX", "SOLUSDT"),
    "LINK": ("LINKUSDT_SWAP_OKX", "LINKUSDT"),
    "DOGE": ("DOGEUSDT_SWAP_OKX", "DOGEUSDT"),
}
MONTHS = [f"{y}-{m:02d}" for y in (2023, 2024, 2025) for m in range(1, 13)] + \
         [f"2026-{m:02d}" for m in range(1, 6)]
SH_OFFSET_MIN = 480  # Asia/Shanghai = UTC+8, no DST

# gates (pre-registered, see module docstring)
GATE_BASIS_MONTH_MEDIAN = 0.005   # 0.5%
GATE_MEDIAN_DEV = 0.001           # 0.1%
CORROB_SIGN_SAME = True
CORROB_EXC_RATIO = 0.5

# ── synthetic-ramp detector: VERBATIM port from research_v2b_dd_diagnosis.py ──
RAMP_STEP = 0.015
RAMP_MIN_LEN = 4
RAMP_MIN_MOVE = 0.10
JUMP_1BAR = 0.15
MERGE_GAP_MIN = 60
REVERT_TOL = 0.02
REVERT_WIN_MIN = 240
STAIR_CV = 0.30


def detect_ramp_events(dt_ns, c):
    """dt_ns = int64 epoch-ns array, c = 1m closes. Returns merged events."""
    gap_min = np.diff(dt_ns) / 60_000_000_000
    r = np.diff(c) / c[:-1]
    valid = gap_min <= 2.0
    cands = []
    i = 0
    n = len(r)
    while i < n:
        if not (valid[i] and abs(r[i]) >= RAMP_STEP):
            i += 1
            continue
        s = np.sign(r[i])
        j = i
        while j + 1 < n and valid[j + 1] and abs(r[j + 1]) >= RAMP_STEP and np.sign(r[j + 1]) == s:
            j += 1
        if j - i + 1 >= RAMP_MIN_LEN and abs(c[j + 1] / c[i] - 1) >= RAMP_MIN_MOVE:
            cands.append((i, j + 1))
        i = j + 1
    for k in np.where(valid & (np.abs(r) >= JUMP_1BAR))[0]:
        cands.append((int(k), int(k) + 1))
    if not cands:
        return []
    cands.sort()
    merged = [list(cands[0])]
    for a, b in cands[1:]:
        if (dt_ns[a] - dt_ns[merged[-1][1]]) / 60_000_000_000 <= MERGE_GAP_MIN:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    events = []
    for a, b in merged:
        ref = c[max(0, a - 1)]
        seg = c[a:b + 1]
        exc = max(abs(seg.max() / ref - 1), abs(seg.min() / ref - 1))
        hi = dt_ns[b] + REVERT_WIN_MIN * 60_000_000_000
        k = b + 1
        reverted = False
        while k < len(c) and dt_ns[k] <= hi:
            if abs(c[k] / ref - 1) <= REVERT_TOL:
                reverted = True
                break
            k += 1
        steps = np.diff(c[a:b + 1])
        stair = False
        if len(steps) >= RAMP_MIN_LEN:
            sgn = np.sign(steps)
            i0 = 0
            for i1 in range(1, len(steps) + 1):
                if i1 == len(steps) or sgn[i1] != sgn[i0]:
                    run = steps[i0:i1]
                    if len(run) >= 5 and abs(run.mean()) > 0:
                        if run.std() / abs(run.mean()) <= STAIR_CV:
                            stair = True
                    i0 = i1
        events.append({"i0": int(a), "i1": int(b), "ref_px": float(ref),
                       "max_excursion_pct": round(float(exc) * 100, 2),
                       "reverted": reverted, "staircase": stair})
    return events


# ── loaders ───────────────────────────────────────────────────────────────────
def load_okx(db_symbol: str) -> pd.DataFrame:
    """OKX mainnet 1m: naive Shanghai bar-open -> UTC epoch-minute key."""
    conn = sqlite3.connect(f"file:{DB_MAIN}?mode=ro", uri=True)
    try:
        df = pd.read_sql_query(
            "select datetime, close_price as close from dbbardata "
            "where symbol=? and exchange='GLOBAL' and interval='1m' order by datetime",
            conn, params=(db_symbol,),
        )
    finally:
        conn.close()
    ts = pd.to_datetime(df["datetime"])
    # resolution-independent epoch-minute (pandas>=3.0 may infer non-ns units)
    df["min_utc"] = ((ts - pd.Timestamp("1970-01-01")) // pd.Timedelta(minutes=1)
                     ).astype("int64") - SH_OFFSET_MIN
    df["close"] = pd.to_numeric(df["close"])
    return df[["min_utc", "close"]]


def load_binance(b_symbol: str) -> pd.DataFrame:
    """Binance vision monthly zips -> UTC epoch-minute key (ms or us stamps)."""
    frames = []
    for month in MONTHS:
        zp = BV_ROOT / b_symbol / f"{b_symbol}-1m-{month}.zip"
        with zipfile.ZipFile(zp) as z:
            with z.open(z.namelist()[0]) as f:
                raw = f.read()
        df = pd.read_csv(io.BytesIO(raw), header=None, usecols=[0, 4],
                         names=["open_time", "close"])
        if isinstance(df.iloc[0, 0], str):  # header row present in newer files
            df = df.iloc[1:].reset_index(drop=True)
        ot = pd.to_numeric(df["open_time"]).astype("int64")
        unit_us = ot.iloc[0] > 100_000_000_000_000  # >1e14 -> microseconds
        df["min_utc"] = (ot // (60_000_000 if unit_us else 60_000)).astype("int64")
        df["close"] = pd.to_numeric(df["close"])
        frames.append(df[["min_utc", "close"]])
    out = pd.concat(frames, ignore_index=True)
    return out.drop_duplicates("min_utc", keep="last").sort_values("min_utc").reset_index(drop=True)


# ── stats helpers ─────────────────────────────────────────────────────────────
def dist_stats(x: np.ndarray) -> dict:
    x = x[~np.isnan(x)]
    return {"n": int(len(x)), "median": float(np.median(x)),
            "p90": float(np.percentile(x, 90)), "p99": float(np.percentile(x, 99)),
            "p99_9": float(np.percentile(x, 99.9)), "max": float(np.max(x))}


def exceedance(x: np.ndarray) -> dict:
    x = x[~np.isnan(x)]
    return {f"gt_{t}pct": float(np.mean(x > t / 100) * 100)
            for t in (0.1, 0.5, 1.0, 5.0)}


def lag_check(okx: np.ndarray, bnc: np.ndarray, max_lag: int = 5) -> dict:
    """Cross-correlation of 1m log returns at lags -max_lag..max_lag.
    Positive k = OKX lags Binance by k minutes. Peak must be at k=0."""
    ro = np.diff(np.log(okx))
    rb = np.diff(np.log(bnc))
    out = {}
    for k in range(-max_lag, max_lag + 1):
        if k >= 0:
            a, b = ro[k:], rb[:len(rb) - k] if k else rb
        else:
            a, b = ro[:k], rb[-k:]
        m = min(len(a), len(b))
        out[str(k)] = float(np.corrcoef(a[:m], b[:m])[0, 1])
    peak = max(out, key=lambda kk: out[kk])
    return {"corr_by_lag": out, "peak_lag": int(peak), "peak_corr": out[peak]}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "okx_vs_binance_stats").mkdir(exist_ok=True)
    log_lines = []

    def L(msg=""):
        print(msg, flush=True)
        log_lines.append(str(msg))

    L(f"run start (UTC): {datetime.now(timezone.utc).isoformat()}")
    L("DATA: OKX .vntrader/database_mainnet.db (mode=ro) vs data/binance_vision/ (sha256-verified)")
    L("GATES (pre-registered): G1 synthetic=0/corroborated; "
      "G2 no month |median signed basis|>0.5%; G3 median |dev|<0.1% per symbol")

    gate_fail = {"G1": [], "G2": [], "G3": []}
    scan_out = {}
    for name, (okx_sym, b_sym) in SYMBOLS.items():
        L(f"\n=== {name} ===")
        okx = load_okx(okx_sym)
        bnc = load_binance(b_sym)
        merged = okx.merge(bnc, on="min_utc", how="inner", suffixes=("_okx", "_bnc"))
        n_okx_only = len(okx) - len(merged)
        n_bnc_only = len(bnc) - len(merged)
        L(f"okx bars {len(okx):,} | binance bars {len(bnc):,} | aligned {len(merged):,} "
          f"| okx_only {n_okx_only:,} | binance_only {n_bnc_only:,}")

        co = merged["close_okx"].to_numpy(dtype=float)
        cb = merged["close_bnc"].to_numpy(dtype=float)
        signed = (co - cb) / cb
        absdev = np.abs(signed)

        sym_stats = {
            "okx_bars": int(len(okx)), "binance_bars": int(len(bnc)),
            "aligned_bars": int(len(merged)),
            "okx_only": int(n_okx_only), "binance_only": int(n_bnc_only),
            "abs_close_dev": dist_stats(absdev),
            "exceedance_pct_of_bars": exceedance(absdev),
            "signed_basis": dist_stats(np.abs(signed)) | {
                "mean_signed": float(np.mean(signed)),
                "median_signed": float(np.median(signed))},
            "lag_check": lag_check(co, cb),
        }

        # monthly buckets (UTC month of bar open)
        mts = pd.to_datetime(merged["min_utc"] * 60, unit="s", utc=True)
        mdf = pd.DataFrame({"month": mts.dt.strftime("%Y-%m"),
                            "signed": signed, "absdev": absdev})
        monthly = []
        for month, g in mdf.groupby("month", sort=True):
            monthly.append({
                "month": month, "n": int(len(g)),
                "median_abs_dev_pct": float(np.median(g["absdev"]) * 100),
                "p99_abs_dev_pct": float(np.percentile(g["absdev"], 99) * 100),
                "frac_gt_0_1pct": float(np.mean(g["absdev"] > 0.001) * 100),
                "median_signed_basis_pct": float(np.median(g["signed"]) * 100),
                "mean_signed_basis_pct": float(np.mean(g["signed"]) * 100),
            })
        sym_stats["monthly"] = monthly

        # G2: sustained one-sided dislocation
        bad_months = [m for m in monthly
                      if abs(m["median_signed_basis_pct"]) > GATE_BASIS_MONTH_MEDIAN * 100]
        if bad_months:
            gate_fail["G2"].append({name: [m["month"] for m in bad_months]})
        # G3: full-period median
        if sym_stats["abs_close_dev"]["median"] >= GATE_MEDIAN_DEV:
            gate_fail["G3"].append(name)
        L(f"median |dev| {sym_stats['abs_close_dev']['median']*100:.4f}% | "
          f"p99 {sym_stats['abs_close_dev']['p99']*100:.3f}% | "
          f">0.1%: {sym_stats['exceedance_pct_of_bars']['gt_0.1pct']:.2f}% of bars | "
          f"lag peak k={sym_stats['lag_check']['peak_lag']} "
          f"(corr {sym_stats['lag_check']['peak_corr']:.3f}) | "
          f"G2 bad months: {[m['month'] for m in bad_months] or 'none'}")

        # ── 2b synthetic scan on OKX mainnet, Binance corroboration ──
        dt_ns = (merged["min_utc"].to_numpy(dtype="int64") * 60_000_000_000)
        events = detect_ramp_events(dt_ns, co)
        ev_out = []
        for ev in events:
            a, b = ev["i0"], ev["i1"]
            ref_o = co[max(0, a - 1)]
            ref_b = cb[max(0, a - 1)]
            seg_o = co[a:b + 1]
            seg_b = cb[a:b + 1]
            exc_o = max(abs(seg_o.max() / ref_o - 1), abs(seg_o.min() / ref_o - 1))
            exc_b = max(abs(seg_b.max() / ref_b - 1), abs(seg_b.min() / ref_b - 1))
            net_o = seg_o[-1] / ref_o - 1
            net_b = seg_b[-1] / ref_b - 1
            same_sign = bool(np.sign(net_o) == np.sign(net_b)) if net_o != 0 and net_b != 0 else False
            corroborated = bool(same_sign and exc_b >= CORROB_EXC_RATIO * exc_o)
            t0 = pd.to_datetime(int(merged['min_utc'].iloc[a]) * 60, unit="s", utc=True)
            t1 = pd.to_datetime(int(merged['min_utc'].iloc[b]) * 60, unit="s", utc=True)
            ev_out.append({
                "window_utc": [t0.isoformat(), t1.isoformat()],
                "okx_max_excursion_pct": round(exc_o * 100, 2),
                "binance_max_excursion_pct": round(exc_b * 100, 2),
                "okx_net_move_pct": round(net_o * 100, 2),
                "binance_net_move_pct": round(net_b * 100, 2),
                "reverted_4h": ev["reverted"], "staircase_flag": ev["staircase"],
                "binance_corroborated": corroborated,
            })
            if not corroborated:
                gate_fail["G1"].append({name: ev_out[-1]})
        scan_out[name] = {"n_events": len(ev_out), "events": ev_out}
        L(f"synthetic scan: {len(ev_out)} event(s)"
          + ("" if not ev_out else f", corroborated: {sum(e['binance_corroborated'] for e in ev_out)}"))

        (OUT / "okx_vs_binance_stats" / f"{name}.json").write_text(
            json.dumps(sym_stats, indent=2))
        del okx, bnc, merged, co, cb, signed, absdev, mdf

    # demo-library reference for comparison (from v2b_dd_diagnosis): 598 events
    scan_summary = {
        "detector": "verbatim port of research_v2b_dd_diagnosis.py step-4 "
                    "(RAMP_STEP 0.015 / MIN_LEN 4 / MIN_MOVE 0.10 / JUMP_1BAR 0.15 / STAIR_CV 0.30)",
        "demo_library_reference_counts": {"SOL": 344, "DOGE": 216, "ETH": 23, "LINK": 15, "BTC": 0,
                                          "total": 598},
        "mainnet_counts": {k: v["n_events"] for k, v in scan_out.items()},
        "corroboration_rule": "same net-move sign AND Binance excursion >= 0.5x OKX excursion",
        "per_symbol": scan_out,
    }
    (OUT / "synthetic_scan_mainnet.json").write_text(json.dumps(scan_summary, indent=2))

    verdict = "PASS" if not (gate_fail["G1"] or gate_fail["G2"] or gate_fail["G3"]) else "FAIL"
    vd = {"verdict": verdict, "gate_failures": gate_fail,
          "gates": {"G1": "synthetic events 0 or all Binance-corroborated",
                    "G2": "no month |median signed basis| > 0.5%",
                    "G3": "per-symbol full-period median |close dev| < 0.1%"}}
    (OUT / "verdict.json").write_text(json.dumps(vd, indent=2))
    L(f"\nVERDICT: {verdict}")
    if verdict == "FAIL":
        L(json.dumps(gate_fail, indent=2))
    L(f"run end (UTC): {datetime.now(timezone.utc).isoformat()}")
    (OUT / "run_log.txt").write_text("\n".join(log_lines) + "\n")
    return 0 if verdict == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
