#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
research_basis_feasibility.py
OKX 期现基差套利（cash-and-carry）可行性研究 — 方向第一块地基（stage-1 foundation）。

POSITIONING (also in report README)
-----------------------------------
本研究是期现基差套利方向的第一块地基，性质与此前所有"半天判生死的前置研究"不同：
它是真金工程的起点。期现基差不赌方向，edge 来自交割合约到期对现货/指数的强制收敛
（合约条款，非统计倾向）——这是绕开本市场右偏延续结构的方向。本脚本：
  ① 确认 OKX 季度交割合约历史数据可得性（硬前提，B1）；
  ② 诚实核算历史基差的净收益率与风险（这个套利在 OKX 上到底有多少肉）。
产出立项/不立项判定 + 风险画像，不搭建执行系统（下一阶段）。

DATA AVAILABILITY (verified during recon, 2026-06-16)
----------------------------------------------------
* OKX REST history-candles REJECTS expired delivery contracts (code 50047 "settled");
  mark-price / instrument-metadata for expired contracts are PURGED (code 51001).
* BUT OKX retains expired-contract data on the data-download CDN as DAILY TRADE files,
  free, no API key, arbitrary dates, back to 2019:
    https://www.okx.com/cdn/okex/traderecords/trades/daily/<YYYYMMDD>/<instId>-trades-<YYYY-MM-DD>.zip
  CSV cols (header garbled GBK): trade_id, side, size(contracts), price, created_time(ms).
  -> futures price history is RECONSTRUCTABLE (parse trades -> daily OHLCV). B1 = GREEN.
* Index (the settlement reference) IS available via REST history-index-candles bar=1Dutc,
  back to 2019-08; UTC-aligned -> matches our UTC trade-date aggregation.
* LINEAR (USDT-margined) quarterlies were PHASED OUT by OKX ~late-2025 (no live USDT
  quarter/next_quarter; 251226/260327 absent on CDN). INVERSE (BTC-USD/ETH-USD) is the
  deep + ONGOING vehicle (260925 quarter / 261225 next_quarter live). -> measure on INVERSE.

PRE-REGISTERED (frozen BEFORE results; NOT tunable after — project rule)
------------------------------------------------------------------------
Vehicle           : OKX INVERSE quarterly delivery futures, BTC-USD & ETH-USD.
Spot/anchor leg   : OKX index (BTC-USD / ETH-USD) via REST history-index-candles bar=1Dutc
                    = exactly what the future settles to at expiry.
Universe          : all settled quarterlies 2022Q1..2026Q1 (last Friday of Mar/Jun/Sep/Dec,
                    08:00 UTC), i.e. 17 rounds/coin. (>= 4 required by B1.)
Basis (per day)   : basis = (F_close - Index_close) / Index_close
Annualized basis  : ann = basis * 365 / days_to_expiry
Entry horizon     : ENTRY_DAYS = 60 calendar days before expiry.
Deployable rule   : classic cash-and-carry (LONG spot + SHORT future) is deployable ONLY when
                    entry basis > fee hurdle (contango). Backwardation rounds need costly spot
                    borrow (reverse carry) -> recorded, EXCLUDED from deployable-yield average,
                    NOT counted as losses.
Fees (OKX std Lv1, conservative ALL-TAKER), round trip on notional:
                    spot 0.10%/side x2  +  futures 0.05% entry  +  0.05% exit
                    (delivery settle modeled at TAKER 0.05% though real delivery fee is lower
                     ~0.01-0.02% -> conservative)   =  FEE_ROUNDTRIP = 0.30%.
Opportunity cost  : RISK_FREE = 5.0%/yr (USD T-bill-era proxy for capital occupation).
Risk premium      : RISK_PREMIUM = 3.0%/yr (margin/liquidation/exchange/non-convergence risk).

DECISION GATES (real-money standard, NOT Sharpe)
------------------------------------------------
B1 data available : settled quarterlies with usable data >= 4 rounds.                 [hard prereq]
B2 net positive   : mean(deployable-round net-annualized AFTER fees) > RISK_FREE+RISK_PREMIUM
                    (= 8.0%)  AND  >= 75% of deployable rounds net-positive after fees.
B3 risk control   : worst intra-hold ADVERSE move on the SHORT futures leg (isolated/conservative)
                    survivable at <= 2x futures leverage (>= 50% initial margin, i.e. move < ~49%);
                    basis-anomaly loss quantifiable and not unbounded.
B4 convergence    : >= 75% of rounds converge to |final basis| < 0.30% by last pre-expiry obs.
ALL pass -> 立项候选 (next stage: dual-leg execution + margin mgmt).  ANY fail -> 记名死因.

DATA RULES (project 铁律)
-------------------------
* writes ONLY to data/basis/ ; NEVER touches database_mainnet.db or the contaminated db.
* OKX-native measurement; NEW data cross-validated vs INDEPENDENT source (>=3 random days):
    - futures daily close   vs  Binance vision COIN-M delivery futures (BTCUSD_<exp>);
    - OKX index daily close  vs  Binance spot (local 1m -> daily).
* manifest records source / caliber / coverage / server.

Usage:
  python scripts/research_basis_feasibility.py --all          # download + analyze (default)
  python scripts/research_basis_feasibility.py --download     # fetch + cache only
  python scripts/research_basis_feasibility.py --analyze      # compute from cache + write report
  optional: --force (re-download)  --workers N  --smoke (1 contract/coin)
"""
from __future__ import annotations

import argparse
import datetime as dt
import io
import json
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import requests

# ----------------------------------------------------------------------------- config
ROOT = Path(__file__).resolve().parents[5]  # 2026-07 重构批次5：迁入 research/_closed/basis/basis_arbitrage/scripts/，深度 1→5
DATA_DIR = ROOT / "data" / "basis"
FUT_DIR = DATA_DIR / "futures"
IDX_DIR = DATA_DIR / "index"
REPORT_DIR = ROOT / "reports" / "basis_arbitrage_feasibility_20260615"
BINANCE_LOCAL = ROOT / "data" / "binance_vision"

COINS = ["BTC", "ETH"]
CDN = "https://www.okx.com/cdn/okex/traderecords/trades/daily"
OKX_REST = "https://www.okx.com/api/v5/market/history-index-candles"
BINANCE_CM = "https://data.binance.vision/data/futures/cm/monthly/klines"

WINDOW_DAYS = 65          # dense daily window [expiry-65d, expiry]
SPARSE_EXTRA = [90, 120]  # extra sparse points for term-structure sketch
ENTRY_DAYS = 60           # pre-registered entry horizon (calendar days before expiry)
# OKX data-download CDN publishes daily trade files with a rolling lag: at recon
# (2026-06-16) coverage is COMPLETE only through ~2025-09-07. Contracts expiring after
# that lose their convergence tail (verified: 250926 truncates at 2025-09-07; 250627 is
# complete through its 2025-06-27 expiry). To keep every measured round fully observable
# (entry -> convergence), universe = quarters with expiry <= CDN_CUTOFF. Settled-but-
# -truncated quarters (250926+) are recorded as a B1 boundary, not measured.
CDN_CUTOFF = dt.date(2025, 6, 30)

# --- pre-registered economics (DO NOT EDIT AFTER RESULTS) ---
FEE_ROUNDTRIP = 0.0030    # 0.30% notional, all-taker round trip
RISK_FREE = 0.05          # 5%/yr opportunity cost of capital
RISK_PREMIUM = 0.03       # 3%/yr risk premium
B2_BAR = RISK_FREE + RISK_PREMIUM      # 8.0% net annualized (mean) hurdle
B2_CONSISTENCY = 0.75     # >=75% deployable rounds net-positive
B3_MAX_LEVERAGE = 2.0     # survive worst adverse move at <=2x (IM>=50%)
B3_MMR = 0.005            # maintenance margin rate assumption for liq calc
CONVERGE_THRESH = 0.0030  # |final basis| < 0.30% = converged
B4_CONSISTENCY = 0.75
B1_MIN_ROUNDS = 4

# OKX data-download CDN trade files are bounded by HKT (UTC+8) calendar day: a file
# labeled YYYY-MM-DD spans UTC [D-1 16:00, D 16:00), so its last trade ("close") sits at
# ~16:00 UTC = HKT midnight. We therefore align the INDEX to the SAME boundary (OKX bar=1D,
# HKT) so basis = F(HKT close) - Index(HKT close) is time-matched (verified during recon).
HKT = dt.timezone(dt.timedelta(hours=8))

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "cta-basis-research/1.0"})

LOG_LINES: list[str] = []


def log(msg: str):
    line = f"[{dt.datetime.now(dt.UTC).strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    LOG_LINES.append(line)


# ----------------------------------------------------------------------------- schedule
def last_friday(y: int, m: int) -> dt.date:
    nxt = dt.date(y + 1, 1, 1) if m == 12 else dt.date(y, m + 1, 1)
    d = nxt - dt.timedelta(days=1)
    while d.weekday() != 4:  # Friday
        d -= dt.timedelta(days=1)
    return d


def settled_quarters(today: dt.date) -> list[tuple[str, dt.date]]:
    out = []
    for y in range(2022, 2027):
        for m in (3, 6, 9, 12):
            exp = last_friday(y, m)
            if exp < today:  # only fully settled
                out.append((exp.strftime("%y%m%d"), exp))
    return out


# ----------------------------------------------------------------------------- futures dl
def _parse_trade_zip(content: bytes) -> dict | None:
    """zip(csv: trade_id,side,size,price,created_time) -> daily OHLCV dict (UTC day of file)."""
    try:
        zf = zipfile.ZipFile(io.BytesIO(content))
        name = zf.namelist()[0]
        with zf.open(name) as fh:
            # header row is GBK-garbled -> skip it; data rows are ASCII (latin-1 safe)
            df = pd.read_csv(fh, header=None, skiprows=1, encoding="latin-1")
        df = df.iloc[:, [2, 3, 4]]          # positional: size, price, created_time(ms)
        df.columns = ["size", "price", "ts"]
    except Exception:
        return None
    if df.empty:
        return None
    df = df.dropna()
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["size"] = pd.to_numeric(df["size"], errors="coerce")
    df["ts"] = pd.to_numeric(df["ts"], errors="coerce")
    df = df.dropna()
    if df.empty:
        return None
    df = df.sort_values("ts")
    vol = float(df["size"].sum())
    vwap = float((df["price"] * df["size"]).sum() / vol) if vol > 0 else float(df["price"].mean())
    return {
        "open": float(df["price"].iloc[0]),
        "high": float(df["price"].max()),
        "low": float(df["price"].min()),
        "close": float(df["price"].iloc[-1]),
        "vwap": vwap,
        "vol_contracts": vol,
        "ntrades": int(len(df)),
    }


def _fetch_day(instId: str, day: dt.date) -> tuple[dt.date, dict | None]:
    url = f"{CDN}/{day.strftime('%Y%m%d')}/{instId}-trades-{day.strftime('%Y-%m-%d')}.zip"
    for attempt in range(3):
        try:
            r = SESSION.get(url, timeout=30)
            if r.status_code == 404:
                return day, None
            if r.status_code == 200:
                return day, _parse_trade_zip(r.content)
        except Exception:
            time.sleep(0.5 * (attempt + 1))
    return day, None


def download_futures(quarters, workers: int, force: bool):
    FUT_DIR.mkdir(parents=True, exist_ok=True)
    tasks = []  # (instId, expiry, day)
    contracts = []
    for coin in COINS:
        for yymmdd, exp in quarters:
            instId = f"{coin}-USD-{yymmdd}"
            out = FUT_DIR / f"{instId}.csv"
            contracts.append((instId, exp, out))
            if out.exists() and not force:
                continue
            days = sorted({exp - dt.timedelta(days=k) for k in range(WINDOW_DAYS + 1)}
                          | {exp - dt.timedelta(days=k) for k in SPARSE_EXTRA})
            for d in days:
                tasks.append((instId, exp, d))

    if not tasks:
        log(f"futures: all {len(contracts)} contracts cached, skip download")
    else:
        log(f"futures: downloading {len(tasks)} day-files across "
            f"{len({t[0] for t in tasks})} contracts ({workers} workers)")
        rows: dict[str, list] = {}
        done = 0
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(_fetch_day, inst, day): (inst, exp, day)
                    for inst, exp, day in tasks}
            for fut in as_completed(futs):
                inst, exp, day = futs[fut]
                _, agg = fut.result()
                done += 1
                if done % 250 == 0:
                    log(f"  ... {done}/{len(tasks)} day-files")
                if agg is None:
                    continue
                agg["date"] = day.isoformat()
                agg["days_to_expiry"] = (exp - day).days
                rows.setdefault(inst, []).append(agg)
        for inst, rl in rows.items():
            df = pd.DataFrame(rl).sort_values("date")
            df = df[["date", "days_to_expiry", "open", "high", "low", "close",
                     "vwap", "vol_contracts", "ntrades"]]
            df.to_csv(FUT_DIR / f"{inst}.csv", index=False)
        log(f"futures: wrote {len(rows)} contract CSVs")

    # report empty/missing contracts
    have = []
    for instId, exp, out in contracts:
        if out.exists():
            n = sum(1 for _ in open(out)) - 1
            if n > 0:
                have.append((instId, n))
    log(f"futures: {len(have)}/{len(contracts)} contracts have data")
    return contracts


# ----------------------------------------------------------------------------- index dl
def download_index(force: bool):
    IDX_DIR.mkdir(parents=True, exist_ok=True)
    start = dt.date(2022, 1, 1)
    for coin in COINS:
        inst = f"{coin}-USD"
        out = IDX_DIR / f"{inst}_index_1Dutc.csv"
        if out.exists() and not force:
            log(f"index: {inst} cached")
            continue
        rows = []
        after = None
        oldest = None
        for _ in range(60):
            p = {"instId": inst, "bar": "1D", "limit": "100"}   # HKT day boundary (matches futures files)
            if after:
                p["after"] = str(after)
            r = SESSION.get(OKX_REST, params=p, timeout=20).json()
            d = r.get("data", [])
            if not d:
                break
            for x in d:
                ts = int(x[0])
                rows.append({"date": dt.datetime.fromtimestamp(ts / 1000, HKT).date().isoformat(),
                             "close": float(x[4])})
            oldest = min(int(x[0]) for x in d)
            after = oldest
            if dt.datetime.fromtimestamp(oldest / 1000, dt.UTC).date() <= start:
                break
            time.sleep(0.1)
        df = pd.DataFrame(rows).drop_duplicates("date").sort_values("date")
        df = df[df["date"] >= start.isoformat()]
        df.to_csv(out, index=False)
        log(f"index: {inst} {len(df)} days [{df['date'].iloc[0]}..{df['date'].iloc[-1]}]")


# ----------------------------------------------------------------------------- cross-validate
def _binance_cm_daily(symbol_exp: str, month: str) -> pd.DataFrame | None:
    """symbol_exp like BTCUSD_250328, month 'YYYY-MM' -> daily close df or None."""
    url = f"{BINANCE_CM}/{symbol_exp}/1d/{symbol_exp}-1d-{month}.zip"
    try:
        r = SESSION.get(url, timeout=30)
        if r.status_code != 200:
            return None
        zf = zipfile.ZipFile(io.BytesIO(r.content))
        with zf.open(zf.namelist()[0]) as fh:
            df = pd.read_csv(fh, header=None)
        # binance kline cols: openTime,open,high,low,close,...  (newer files carry a header row)
        df[0] = pd.to_numeric(df[0], errors="coerce")
        df = df.dropna(subset=[0])
        df["date"] = pd.to_datetime(df[0], unit="ms", utc=True).dt.date.astype(str)
        df["close"] = pd.to_numeric(df[4], errors="coerce")
        return df[["date", "close"]].dropna()
    except Exception:
        return None


def _binance_spot_close_at_hkt_midnight(symbol: str, day: dt.date) -> float | None:
    """Binance spot price at HKT-day-`day` close (= 16:00 UTC of `day`), from local 1m zip.
    Matches the OKX 1D(HKT) index close so the cross-check is time-aligned."""
    z = BINANCE_LOCAL / symbol / f"{symbol}-1m-{day.strftime('%Y-%m')}.zip"
    if not z.exists():
        return None
    try:
        zf = zipfile.ZipFile(z)
        with zf.open(zf.namelist()[0]) as fh:
            df = pd.read_csv(fh, header=None)
        df[0] = pd.to_numeric(df[0], errors="coerce")   # drop header row if present
        df = df.dropna(subset=[0])
        target_ms = int(dt.datetime(day.year, day.month, day.day, 16, 0, tzinfo=dt.UTC).timestamp() * 1000)
        df["d"] = (df[0] - target_ms).abs()
        row = df.loc[df["d"].idxmin()]
        if abs(int(row[0]) - target_ms) > 5 * 60 * 1000:   # no 1m bar within 5min of 16:00 UTC
            return None
        return float(pd.to_numeric(row[4]))
    except Exception:
        return None


def cross_validate(quarters) -> dict:
    out = {"futures_vs_binance_cm": [], "index_vs_binance_spot": []}
    # FUTURES: 2 contracts x >=3 days each
    fut_checks = [("BTC", "250328", "BTCUSD_250328", "2025-02"),
                  ("ETH", "250627", "ETHUSD_250627", "2025-05")]
    for coin, yymmdd, bsym, month in fut_checks:
        inst = f"{coin}-USD-{yymmdd}"
        f = FUT_DIR / f"{inst}.csv"
        bdf = _binance_cm_daily(bsym, month)
        if not f.exists() or bdf is None:
            continue
        odf = pd.read_csv(f)
        m = odf.merge(bdf, on="date", suffixes=("_okx", "_bin"))
        m = m.dropna().tail(20).head(6)  # a handful of overlapping days
        for _, row in m.iterrows():
            diff = abs(row["close_okx"] - row["close_bin"]) / row["close_bin"]
            out["futures_vs_binance_cm"].append(
                {"inst": inst, "date": row["date"], "okx": round(float(row["close_okx"]), 2),
                 "binance_cm": round(float(row["close_bin"]), 2), "abs_pct_diff": round(diff * 100, 4)})
    # INDEX: BTC-USD index vs binance spot, >=3 random days (within measured range, locally covered)
    rng = np.random.default_rng(42)
    for coin, bspot in [("BTC", "BTCUSDT"), ("ETH", "ETHUSDT")]:
        idx = pd.read_csv(IDX_DIR / f"{coin}-USD_index_1Dutc.csv")
        idx = idx[(idx["date"] >= "2022-06-01") & (idx["date"] <= CDN_CUTOFF.isoformat())]
        sample = idx.sample(min(4, len(idx)), random_state=int(rng.integers(0, 1e6)))
        for _, row in sample.iterrows():
            day = dt.date.fromisoformat(row["date"])
            bc = _binance_spot_close_at_hkt_midnight(bspot, day)
            if bc is None:
                continue
            diff = abs(row["close"] - bc) / bc
            out["index_vs_binance_spot"].append(
                {"coin": coin, "date": row["date"], "okx_index_hkt_close": round(float(row["close"]), 2),
                 "binance_spot_1600utc": round(bc, 2), "abs_pct_diff": round(diff * 100, 4)})
    # verdicts
    fdiffs = [x["abs_pct_diff"] for x in out["futures_vs_binance_cm"]]
    idiffs = [x["abs_pct_diff"] for x in out["index_vs_binance_spot"]]
    out["futures_median_abs_pct_diff"] = round(float(np.median(fdiffs)), 4) if fdiffs else None
    out["index_median_abs_pct_diff"] = round(float(np.median(idiffs)), 4) if idiffs else None
    # futures is cross-exchange (micro-basis differs) -> level-sanity threshold 1.5%; index 0.5%
    out["futures_pass"] = (out["futures_median_abs_pct_diff"] is not None
                           and out["futures_median_abs_pct_diff"] < 1.5)
    out["index_pass"] = (out["index_median_abs_pct_diff"] is not None
                         and out["index_median_abs_pct_diff"] < 0.5)
    log(f"cross-val: futures med diff={out['futures_median_abs_pct_diff']}% (pass={out['futures_pass']}), "
        f"index med diff={out['index_median_abs_pct_diff']}% (pass={out['index_pass']})")
    return out


# ----------------------------------------------------------------------------- analysis
def build_basis(contracts) -> dict:
    """per contract: merge futures+index -> daily basis series."""
    idx_cache = {c: pd.read_csv(IDX_DIR / f"{c}-USD_index_1Dutc.csv").set_index("date")["close"]
                 for c in COINS}
    curves = {}
    for instId, exp, out in contracts:
        if not out.exists():
            continue
        fdf = pd.read_csv(out)
        if fdf.empty:
            continue
        coin = instId.split("-")[0]
        idx = idx_cache[coin]
        fdf = fdf[fdf["date"].isin(idx.index)].copy()
        if fdf.empty:
            continue
        fdf["index_close"] = fdf["date"].map(idx)
        fdf["basis"] = (fdf["close"] - fdf["index_close"]) / fdf["index_close"]
        fdf["ann_basis"] = fdf["basis"] * 365.0 / fdf["days_to_expiry"].clip(lower=1)
        fdf = fdf.sort_values("days_to_expiry", ascending=False)
        curves[instId] = {
            "coin": coin, "expiry": exp.isoformat(),
            "series": fdf[["date", "days_to_expiry", "close", "high", "low", "index_close",
                           "basis", "ann_basis"]].to_dict("records"),
        }
    log(f"basis: built {len(curves)} contract curves")
    return curves


def accounting_and_risk(curves: dict) -> tuple[list, dict]:
    rounds = []
    for instId, c in curves.items():
        s = pd.DataFrame(c["series"]).sort_values("days_to_expiry", ascending=False)
        if s.empty:
            continue
        # entry = row with days_to_expiry closest to ENTRY_DAYS (always within [0,65d] window)
        s["d_from_entry"] = (s["days_to_expiry"] - ENTRY_DAYS).abs()
        entry = s.loc[s["d_from_entry"].idxmin()]
        entry_basis = float(entry["basis"])
        H = int(entry["days_to_expiry"])
        F_entry = float(entry["close"])
        # holding path = rows at/after entry (days_to_expiry <= entry)
        hold = s[s["days_to_expiry"] <= entry["days_to_expiry"]].sort_values("days_to_expiry",
                                                                             ascending=False)
        # convergence measured at last CLEAN day (d2e>=1): the expiry day (d2e=0) halts at
        # 08:00 UTC settlement, an unavoidable 8h gap vs the HKT-midnight index close.
        hold_clean = hold[hold["days_to_expiry"] >= 1]
        final = hold_clean.iloc[-1] if not hold_clean.empty else hold.iloc[-1]
        final_basis = float(final["basis"])
        final_d2e = int(final["days_to_expiry"])
        converged = abs(final_basis) < CONVERGE_THRESH
        # margin stress on SHORT future leg (isolated, conservative): worst intraday UPWARD move
        worst_up = float((hold["high"].max() - F_entry) / F_entry)  # >=0 if price rose
        # adverse basis excursion AFTER entry (basis widening against convergence)
        if entry_basis >= 0:
            worst_basis_widen = float((hold["basis"] - entry_basis).max())   # basis going more +ve
        else:
            worst_basis_widen = float((entry_basis - hold["basis"]).max())   # basis going more -ve
        # net carry for a deployable (contango) round = gross - fees; annualize
        deployable = entry_basis > FEE_ROUNDTRIP
        net_carry = entry_basis - FEE_ROUNDTRIP  # cash-and-carry captures (basis - fees)
        net_ann = net_carry * 365.0 / H if H > 0 else np.nan
        gross_ann = entry_basis * 365.0 / H if H > 0 else np.nan
        rounds.append({
            "instId": instId, "coin": c["coin"], "expiry": c["expiry"],
            "entry_date": entry["date"], "days_held": H,
            "n_obs_in_hold": int(len(hold)),
            "entry_basis_pct": round(entry_basis * 100, 4),
            "gross_ann_pct": round(gross_ann * 100, 3),
            "net_carry_pct": round(net_carry * 100, 4),
            "net_ann_pct": round(net_ann * 100, 3),
            "regime": "contango" if entry_basis > 0 else "backwardation",
            "deployable": bool(deployable),
            "final_basis_pct": round(final_basis * 100, 4),
            "final_basis_d2e": final_d2e,
            "converged": bool(converged),
            "worst_short_leg_up_move_pct": round(worst_up * 100, 3),
            "worst_basis_widen_pct": round(worst_basis_widen * 100, 4),
        })
    rounds.sort(key=lambda r: (r["coin"], r["expiry"]))

    # ---- aggregate risk ----
    dep = [r for r in rounds if r["deployable"]]
    risk = {
        "n_rounds": len(rounds),
        "n_deployable_contango": len(dep),
        "n_backwardation_at_entry": sum(1 for r in rounds if r["regime"] == "backwardation"),
        "frac_contango_at_entry": round(sum(1 for r in rounds if r["regime"] == "contango") / len(rounds), 3) if rounds else None,
        "converged_frac": round(sum(1 for r in rounds if r["converged"]) / len(rounds), 3) if rounds else None,
        "worst_short_leg_up_move_pct": round(max((r["worst_short_leg_up_move_pct"] for r in rounds), default=0.0), 3),
        "worst_short_leg_event": max(rounds, key=lambda r: r["worst_short_leg_up_move_pct"])["instId"] if rounds else None,
        "worst_basis_widen_pct": round(max((r["worst_basis_widen_pct"] for r in rounds), default=0.0), 4),
        "worst_basis_widen_event": max(rounds, key=lambda r: r["worst_basis_widen_pct"])["instId"] if rounds else None,
    }
    return rounds, risk


def evaluate_gates(rounds, risk, xval) -> dict:
    dep = [r for r in rounds if r["deployable"]]
    dep_net_ann = [r["net_ann_pct"] for r in dep]
    dep_net_carry_pos = [r for r in dep if r["net_carry_pct"] > 0]

    # B1
    b1 = risk["n_rounds"] >= B1_MIN_ROUNDS
    # B2
    mean_net_ann = float(np.mean(dep_net_ann)) if dep_net_ann else float("nan")
    median_net_ann = float(np.median(dep_net_ann)) if dep_net_ann else float("nan")
    frac_pos = (len(dep_net_carry_pos) / len(dep)) if dep else 0.0
    b2 = (len(dep) >= 1) and (mean_net_ann > B2_BAR * 100) and (frac_pos >= B2_CONSISTENCY)
    # B3: worst adverse move survivable at <=2x  (liq move ~ 1/L - mmr = 0.5 - 0.005 = 0.495)
    liq_move_2x = (1.0 / B3_MAX_LEVERAGE - B3_MMR) * 100  # %
    worst = risk["worst_short_leg_up_move_pct"]
    b3 = worst < liq_move_2x
    # implied max safe leverage given worst move: L such that worst < 1/L - mmr -> L < 1/(worst+mmr)
    worst_frac = worst / 100.0
    max_safe_lev = (1.0 / (worst_frac + B3_MMR)) if (worst_frac + B3_MMR) > 0 else float("inf")
    # B4
    b4 = (risk["converged_frac"] or 0) >= B4_CONSISTENCY

    gates = {
        "B1_data_available": {"pass": bool(b1), "n_rounds": risk["n_rounds"], "min_required": B1_MIN_ROUNDS},
        "B2_net_positive": {
            "pass": bool(b2), "bar_net_ann_pct": round(B2_BAR * 100, 2),
            "mean_deployable_net_ann_pct": round(mean_net_ann, 3),
            "median_deployable_net_ann_pct": round(median_net_ann, 3),
            "frac_deployable_net_positive": round(frac_pos, 3),
            "consistency_required": B2_CONSISTENCY,
            "n_deployable": len(dep),
            "also_vs_loose_5pct_bar": bool(len(dep) >= 1 and mean_net_ann > 5.0 and frac_pos >= B2_CONSISTENCY),
        },
        "B3_risk_control": {
            "pass": bool(b3), "worst_short_leg_up_move_pct": worst,
            "liq_move_at_2x_pct": round(liq_move_2x, 2),
            "max_safe_leverage_isolated": round(max_safe_lev, 2),
            "worst_event": risk["worst_short_leg_event"],
            "worst_basis_widen_pct": risk["worst_basis_widen_pct"],
            "note": "isolated-margin conservative; cross-margin (spot collateral offsets) mitigates",
        },
        "B4_convergence": {"pass": bool(b4), "converged_frac": risk["converged_frac"],
                           "consistency_required": B4_CONSISTENCY,
                           "thresh_abs_basis_pct": CONVERGE_THRESH * 100},
        "data_cross_validation": {"futures_pass": xval.get("futures_pass"),
                                  "index_pass": xval.get("index_pass")},
    }
    gates["ALL_PASS"] = bool(b1 and b2 and b3 and b4)
    return gates


# ----------------------------------------------------------------------------- manifest/report
def write_manifest(quarters, contracts, xval):
    have = [(i, o) for i, e, o in contracts if o.exists()]
    man = {
        "positioning": "OKX cash-and-carry (期现基差) feasibility RAW MATERIAL — stage-1 foundation; "
                       "OKX-native measurement, NOT merged into mainnet db; index cross-checked vs Binance",
        "source": "OKX data-download CDN (futures trade files) + OKX REST history-index-candles",
        "server": "OKX-MAINNET-PUBLIC",
        "vehicle": "INVERSE quarterly delivery futures BTC-USD / ETH-USD (USDT quarterly phased out ~late-2025)",
        "futures_url_pattern": f"{CDN}/<YYYYMMDD>/<instId>-trades-<YYYY-MM-DD>.zip",
        "index_endpoint": OKX_REST + " (bar=1Dutc)",
        "caliber": {
            "futures_daily": "trades parsed -> UTC daily OHLCV; close=last trade of UTC day",
            "index_daily": "OKX index history-candles bar=1Dutc (UTC-aligned)",
            "basis": "(F_close - Index_close)/Index_close",
            "window": f"daily [expiry-{WINDOW_DAYS}d, expiry] + sparse {SPARSE_EXTRA}d points",
        },
        "coins": COINS,
        "quarters_settled": [q for q, _ in quarters],
        "n_contracts_with_data": len(have),
        "cross_validation": {
            "futures_vs_binance_cm_median_pct": xval.get("futures_median_abs_pct_diff"),
            "index_vs_binance_spot_median_pct": xval.get("index_median_abs_pct_diff"),
            "futures_pass": xval.get("futures_pass"), "index_pass": xval.get("index_pass"),
        },
        "built_at": dt.datetime.now(dt.UTC).isoformat(),
        "note": "large per-contract futures CSVs gitignored; manifest + report artifacts in git",
    }
    (DATA_DIR / "manifest.json").write_text(json.dumps(man, indent=2))
    log("wrote data/basis/manifest.json")
    return man


def run_analyze(contracts, quarters):
    xval = cross_validate(quarters)
    curves = build_basis(contracts)
    rounds, risk = accounting_and_risk(curves)
    gates = evaluate_gates(rounds, risk, xval)
    man = write_manifest(quarters, contracts, xval)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "basis_curves.json").write_text(json.dumps(curves, indent=2))
    pd.DataFrame(rounds).to_csv(REPORT_DIR / "per_round_accounting.csv", index=False)
    (REPORT_DIR / "per_round_accounting.json").write_text(json.dumps(rounds, indent=2))
    (REPORT_DIR / "risk_profile.json").write_text(json.dumps(risk, indent=2))
    (REPORT_DIR / "decision_gates.json").write_text(json.dumps(gates, indent=2))
    (REPORT_DIR / "cross_validation.json").write_text(json.dumps(xval, indent=2))

    # ---- console summary ----
    log("=" * 70)
    log(f"ROUNDS={risk['n_rounds']}  deployable(contango)={risk['n_deployable_contango']}  "
        f"backwardation={risk['n_backwardation_at_entry']}  converged={risk['converged_frac']}")
    b2 = gates["B2_net_positive"]
    log(f"B1={gates['B1_data_available']['pass']}  "
        f"B2={b2['pass']} (mean net ann {b2['mean_deployable_net_ann_pct']}% vs bar {b2['bar_net_ann_pct']}%, "
        f"pos {b2['frac_deployable_net_positive']})  "
        f"B3={gates['B3_risk_control']['pass']} (worst +{gates['B3_risk_control']['worst_short_leg_up_move_pct']}%, "
        f"safe lev {gates['B3_risk_control']['max_safe_leverage_isolated']}x)  "
        f"B4={gates['B4_convergence']['pass']}")
    log(f"ALL_PASS={gates['ALL_PASS']}")
    log("=" * 70)
    (REPORT_DIR / "run_log.txt").write_text("\n".join(LOG_LINES) + "\n")
    return curves, rounds, risk, gates, xval, man


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="1 contract/coin")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    if not (args.download or args.analyze):
        args.all = True

    log("DATA ENVIRONMENT: OKX-MAINNET-PUBLIC (CDN trade files + REST index). "
        "mainnet/contaminated dbs UNTOUCHED. writes -> data/basis/ only.")
    today = dt.date(2026, 6, 16)
    all_settled = settled_quarters(today)
    quarters = [(q, e) for q, e in all_settled if e <= CDN_CUTOFF]   # CDN-published
    excluded = [q for q, e in all_settled if e > CDN_CUTOFF]          # settled but CDN-lagged
    if excluded:
        log(f"B1 boundary: {len(excluded)} settled quarters NOT yet on CDN (~8-9mo lag), "
            f"excluded: {excluded}")
    if args.smoke:
        quarters = quarters[-1:]  # last CDN-available quarter only (250926)
    log(f"universe: {len(quarters)} CDN-available settled quarters/coin x {len(COINS)} coins "
        f"[{quarters[0][0]}..{quarters[-1][0]}]")

    if args.download or args.all:
        contracts = download_futures(quarters, args.workers, args.force)
        download_index(args.force)
    else:
        contracts = [(f"{c}-USD-{q}", e, FUT_DIR / f"{c}-USD-{q}.csv")
                     for c in COINS for q, e in quarters]

    if args.analyze or args.all:
        run_analyze(contracts, quarters)


if __name__ == "__main__":
    main()
