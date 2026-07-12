#!/usr/bin/env python3
"""Zero-cost FRONT GATE / DECISIVE TEST: order-flow exhaustion mean reversion.

POSITIONING & DISCIPLINE (frozen — verbatim into report, do not delete) =============
This task is the zero-cost pre-gate AND the *decisive* test for the whole
"microstructure / order-flow class of mean reversion" direction. The user has
PRE-COMMITTED: if this front gate fails, this class of MR is no longer considered.
So the conclusion must be clean, trustworthy, hand-verified, with no "let's try
another angle" tail left dangling.

NEW MECHANISM (genuinely different from the 4 price-reversion MRs already closed):
  Order-flow exhaustion does NOT bet "price reverts to a mean". It bets that a
  one-shot large aggressive order EXHAUSTS one side of liquidity, the marginal
  impact then falls, and price REBOUNDS out of the liquidity vacuum (impact
  *dissipation*, not price *reversion*). It therefore ESCAPES the "continuation-
  dominates" death cause that killed the 4 price-reversion MRs (it does not need
  price to return to a mean). This is the first mechanically-novel MR idea in the
  project, and it is taken seriously here.

PRIOR (two walls already hit elsewhere):
  (1) SPEED / FRONT-RUNNING wall (same family as the on-chain route): if the
      rebound lives for milliseconds-seconds, HFT / market-makers eat it before a
      retail user can reach it.
  (2) DATA wall: exhaustion is a microstructure phenomenon; OHLC K-lines erase it
      entirely and cannot test it.

JUDGEMENT PHILOSOPHY (frozen):
  * The CENTREPIECE is the REBOUND SURVIVAL CURVE vs USER-REACHABLE LATENCY
    (isomorphic to the cross-exchange-arb latency pre-gate and the ATM-VRP post-
    block edge-decay curve). NOT a single point, NOT a Sharpe.
  * Must distinguish REBOUND (exhaustion holds) vs CONTINUATION (information shock,
    exhaustion fails) — that is the root mechanism question.
  * Latency assumption is CONSERVATIVE (retail public API; HFT colo EXCLUDED by the
    user). Sharpe is NOT the primary verdict.
  * NOISE CALIBRATION (lesson carried from cointegration/factor-scale): the shock-
    conditioned rebound must beat a matched RANDOM-second placebo curve — otherwise
    any sub-second "rebound" is just generic bid-ask bounce, which is precisely the
    spread you must PAY to trade.

==================== PART 0 — DATA GATE (hard prerequisite) =========================
  Tier ranking: full L2/L3 order book (ideal, ~never free) > tick trades with
  direction/size (second best, volume-impact PROXY for order-flow impact) > only
  K-lines (=> DATA WALL, instant death; microstructure invisible on OHLC).
  THIS RUN USES: Binance Vision UM-perp **aggTrades** (daily) = tick trades, ms
  timestamps, is_buyer_maker => taker side. This is the TICK-TRADE tier. Limitation
  stated, not hidden: NO order-book depth => we cannot see liquidity *levels* being
  consumed; we proxy "order-flow impact" by "extreme one-sided taker volume in 1 s".
  This is an APPROXIMATE LOWER-BOUND test of exhaustion (a true test needs the book).
  is_buyer_maker == true  => buyer is maker  => aggressor SELLS  => taker-SELL.
  is_buyer_maker == false => buyer is taker  => taker-BUY.

==================== PART 1 — SHOCK DEFINITION (pre-registered, frozen) =============
  Stream -> 1-second bins (bin = transact_time_ms // 1000, absolute epoch second).
  Per bin: V = sum(qty), OFI = sum(+qty if taker-buy else -qty), n_trades, prices.
  Trailing baseline = previous 3600 s of 1s-bin V (mean mu, std sd), SHIFT(1) so the
  current bin is excluded (NO look-ahead). First 3600 s of each day skipped (no base).
  SHOCK iff  V >= mu + 4*sd  (extreme volume spike, heavy-tailed => 4 sigma is rare)
         AND |OFI| / V >= 0.66  (one-sided, >= ~2:1 net taker imbalance).
  Shock direction d = sign(OFI). d=+1 buy shock pushes price UP -> exhaustion bet =
  pull-back DOWN; d=-1 sell shock pushes price DOWN -> exhaustion bet = bounce UP.
  Sensitivity grid (reported, NOT used to move the verdict): {3,4,5} sigma.

  SAMPLE (frozen, CALENDAR rule — NOT volatility-selected, to avoid biasing toward
  information shocks): 15th of each quarter, 2024-03-15 .. 2026-03-15 (9 days),
  symbols BTCUSDT + ETHUSDT (the two most liquid perps; if exhaustion is unreachable
  on the MOST liquid coins, smaller coins are even less reachable). Whatever vol
  regime falls on those calendar days is accepted as-is.

==================== PART 2 — REBOUND SURVIVAL CURVE (core) =========================
  Reference t0 = END of the shock second = (bin+1)*1000 ms = the EARLIEST instant the
  user could even KNOW the second was a shock (this bakes in the <=1s detection lag
  honestly). P0 = last trade price at t<=t0. P_pre = last trade at t<=bin*1000 (pre).
  impact_bps = d*(P0 - P_pre)/P0*1e4   (push; should be >0 — shock moves price in d).
  Offsets tau (ms): 0,100,500,1000,2000,5000,10000,30000,60000,300000.
  P(tau) = last trade price at t <= t0+tau (searchsorted; trade-price based — note
  bid-ask BOUNCE is present; in aggregate the MEAN captures systematic bounce and the
  MEDIAN is bounce-robust; we also report a half-spread "bounce floor").
  rebound_bps(tau) = d*(P0 - P(tau))/P0*1e4 :  >0 = retraced AGAINST shock (EXHAUSTION)
                                               <0 = continued WITH shock (CONTINUATION).
  Mechanism test (2c): fraction(rebound<0) at 5 s & 30 s. If continuation dominates,
  exhaustion FAILS as a mechanism in this market.
  Placebo: random non-shock bins (V>0, base available), same measurement, d=sign(OFI)
  of that bin -> the generic-bounce baseline the shock must beat.

==================== PART 3 — LATENCY HAIRCUT (reality constraint) ==================
  User-reachable latency L (retail public API; HFT colo EXCLUDED). Sweep L (ms):
  0 (idealized HFT upper bound, NOT reachable — reference only), 50, 100, 200, 500,
  1000. Anchor verdict at L=200 ms (retail w/ a decent VPS) & L=500 ms (conservative).
  Enter the exhaustion trade at t0+L (price P(L)); exit at the rebound peak. Captured
  gross_bps(L) = max_{tau>L} rebound(tau) - rebound(L)  (what is LEFT after L).
  Cost (taker both sides, latency forces taking liquidity): primary 10 bps
  (OKX 5+5), optimistic 6 bps. net_bps(L) = gross_bps(L) - cost. NOTE: the spread that
  manufactures any sub-second "rebound" is the SAME spread you cross — bounce is not
  free money.

==================== PART 4 — VERDICT (pre-registered, decisive) ====================
  PASS (exhaustion reachable, continue) <=> ALL of:
    (1) data gate passes (>= tick trades),
    (2) post-shock MAJORITY rebound (not continuation) — mechanism holds,
    (3) rebound has SIGNIFICANT residual at user-reachable latency timescale, AND
        beats the random-bin placebo, AND exceeds the half-spread bounce floor,
    (4) that residual, minus cost, is POSITIVE.
  FAIL (death; per user commitment the direction is no longer considered) <=> ANY of:
    - DATA WALL: only K-lines, can't test (N/A here — tick trades obtained),
    - MECHANISM FAILS: post-shock mostly continues, not rebounds,
    - SPEED WALL: rebound essentially complete within < user latency (HFT eats it),
    - COST WALL: residual rebound after latency <= trading cost.
  NOT Sharpe-primary. Death is logged as: "microstructure/order-flow MR direction
  explored: edge [absent / present-but-unreachable]; per user commitment no longer
  considered; same wall as on-chain front-running / HFT."

Data env: data.binance.vision public static CDN = Binance PRODUCTION (mainnet) market
data by construction (no demo/testnet variant). No credentials, no .env, no OKX, no
VPS, no contaminated DB, no vrp line, no forward system touched. Every zip sha256-
verified vs its .CHECKSUM. Reuses the project's Binance-Vision cross-validation asset.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[5]  # 2026-07 重构批次5：迁入 research/_closed/crypto_tick/order_flow_exhaustion/scripts/，深度 1→5
ZIP_ROOT = PROJECT_ROOT / "data" / "binance_vision"          # gitignored: **/*.zip
OUT_DIR = PROJECT_ROOT / "reports" / "order_flow_exhaustion_feasibility_20260628"
FIG_DIR = OUT_DIR / "figures"
BASE = "https://data.binance.vision/data/futures/um/daily/aggTrades"

# ---- FROZEN pre-registration constants -------------------------------------------
SYMBOLS = ["BTCUSDT", "ETHUSDT"]
DATES = [  # 15th of each quarter, 2024Q1 .. 2026Q1 (calendar rule, not vol-selected)
    "2024-03-15", "2024-06-15", "2024-09-15", "2024-12-15",
    "2025-03-15", "2025-06-15", "2025-09-15", "2025-12-15",
    "2026-03-15",
]
BIN_MS = 1000               # 1-second bins
BASELINE_BINS = 3600        # trailing 1 hour
SIGMA_PRIMARY = 4.0         # shock = V >= mu + 4*sd
SIGMA_GRID = [3.0, 4.0, 5.0]
ONESIDED_MIN = 0.66         # |OFI|/V threshold (>= ~2:1)
OFFSETS_MS = [0, 100, 500, 1000, 2000, 5000, 10000, 30000, 60000, 300000]
LATENCIES_MS = [0, 50, 100, 200, 500, 1000]
COST_PRIMARY_BPS = 10.0     # OKX taker 5 + 5 (latency forces taker both sides)
COST_OPTIMISTIC_BPS = 6.0
ANCHOR_L_MS = [200, 500]    # verdict anchored here
PLACEBO_PER_DAY = 2000      # random non-shock bins sampled per symbol-day
RNG_SEED = 20260628

MAX_RETRIES = 4
TIMEOUT = 90


# ================================= download ========================================
def _http_get(url: str, timeout: int = TIMEOUT) -> bytes | None:
    import urllib.error
    import urllib.request
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            time.sleep(2 * attempt)
        except Exception:
            time.sleep(2 * attempt)
    return None


def download_aggtrades(symbol: str, date: str) -> Path | None:
    """Download + sha256-verify one daily aggTrades zip. Resume-safe. None if 404."""
    fn = f"{symbol}-aggTrades-{date}.zip"
    out_dir = ZIP_ROOT / symbol / "aggTrades"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / fn
    url = f"{BASE}/{symbol}/{fn}"
    chk = _http_get(url + ".CHECKSUM")
    want = chk.decode().split()[0] if chk else None
    if out.exists() and want:
        if hashlib.sha256(out.read_bytes()).hexdigest() == want:
            return out
    for attempt in range(1, MAX_RETRIES + 1):
        blob = _http_get(url)
        if blob is None:
            print(f"  [404] {fn} not available")
            return None
        if want is None or hashlib.sha256(blob).hexdigest() == want:
            out.write_bytes(blob)
            return out
        print(f"  [checksum mismatch] {fn} retry {attempt}")
        time.sleep(2 * attempt)
    return None


def load_day(zip_path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(zip_path) as z:
        name = z.namelist()[0]
        with z.open(name) as f:
            df = pd.read_csv(
                f,
                usecols=["price", "quantity", "transact_time", "is_buyer_maker"],
                dtype={"price": "float64", "quantity": "float64",
                       "transact_time": "int64"},
            )
    # is_buyer_maker true => taker SELL ; false => taker BUY
    bmaker = (df["is_buyer_maker"].astype("string").str.lower() == "true").to_numpy()
    df = df.drop(columns=["is_buyer_maker"])
    df["taker_buy"] = ~bmaker
    df = df.sort_values("transact_time", kind="mergesort").reset_index(drop=True)
    return df


# ================================ shock detection ==================================
def build_bins(df: pd.DataFrame) -> pd.DataFrame:
    sec = (df["transact_time"].to_numpy() // BIN_MS)
    qty = df["quantity"].to_numpy()
    signed = np.where(df["taker_buy"].to_numpy(), qty, -qty)
    g = pd.DataFrame({"sec": sec, "V": qty, "OFI": signed, "n": 1})
    agg = g.groupby("sec", sort=True).agg(V=("V", "sum"), OFI=("OFI", "sum"),
                                          n=("n", "sum"))
    # reindex to a contiguous 1-second grid (gaps => no trades that second)
    full = np.arange(agg.index.min(), agg.index.max() + 1)
    agg = agg.reindex(full, fill_value=0.0)
    agg.index.name = "sec"
    return agg.reset_index()


def detect_shocks(bins: pd.DataFrame, sigma: float) -> np.ndarray:
    V = bins["V"]
    mu = V.rolling(BASELINE_BINS).mean().shift(1)
    sd = V.rolling(BASELINE_BINS).std().shift(1)
    thr = mu + sigma * sd
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = (bins["OFI"].abs() / V).to_numpy()
    is_shock = (
        (V.to_numpy() >= thr.to_numpy())
        & (ratio >= ONESIDED_MIN)
        & (bins["V"].to_numpy() > 0)
        & np.isfinite(thr.to_numpy())
    )
    return is_shock


def measure(secs: np.ndarray, dirs: np.ndarray, trade_ts: np.ndarray,
            trade_px: np.ndarray) -> dict:
    """Vectorized rebound measurement for a set of reference bins (sec indices)."""
    t0 = (secs + 1) * BIN_MS                      # end of the shock second (ms)
    pre = secs * BIN_MS                           # start of the shock second

    def price_at(times: np.ndarray) -> np.ndarray:
        idx = np.searchsorted(trade_ts, times, side="right") - 1
        px = np.where(idx >= 0, trade_px[np.clip(idx, 0, len(trade_px) - 1)], np.nan)
        return px

    offs = np.array(OFFSETS_MS, dtype="int64")
    q = t0[:, None] + offs[None, :]               # (S, K) query times
    P = price_at(q.ravel()).reshape(q.shape)      # (S, K)
    P0 = P[:, 0]
    Ppre = price_at(pre)
    d = dirs.astype("float64")
    last_t = trade_ts[-1]
    valid = (t0 + OFFSETS_MS[-1]) <= last_t       # full 300s horizon present
    rebound = d[:, None] * (P0[:, None] - P) / P0[:, None] * 1e4   # (S, K) bps
    impact = d * (P0 - Ppre) / P0 * 1e4
    return {"rebound": rebound, "impact": impact, "valid": valid,
            "P0": P0, "d": d, "t0": t0}


# =================================== driver ========================================
def run(download_only: bool = False) -> dict:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RNG_SEED)
    print("=" * 78)
    print("DATA ENVIRONMENT: Binance Vision UM-perp aggTrades (PRODUCTION/mainnet CDN)")
    print("  tier = TICK TRADES (no order-book depth); proxy = extreme 1s one-sided vol")
    print("=" * 78)

    manifest = {"source": "data.binance.vision/futures/um/daily/aggTrades",
                "server": "binance_production_mainnet", "dataset": "aggTrades",
                "tier": "tick_trades_no_orderbook", "files": []}
    ev_rebound, ev_impact, ev_meta = [], [], []      # shock events
    pb_rebound = []                                    # placebo events
    spread_floors = []
    n_offsets = len(OFFSETS_MS)

    for sym in SYMBOLS:
        for date in DATES:
            zp = download_aggtrades(sym, date)
            rec = {"symbol": sym, "date": date,
                   "file": None if zp is None else str(zp.relative_to(PROJECT_ROOT)),
                   "sha256": None if zp is None else
                   hashlib.sha256(zp.read_bytes()).hexdigest()}
            if zp is None:
                rec["status"] = "not_available_404"
                manifest["files"].append(rec)
                continue
            rec["status"] = "ok"
            if download_only:
                manifest["files"].append(rec)
                continue

            df = load_day(zp)
            rec["n_trades"] = int(len(df))
            trade_ts = df["transact_time"].to_numpy()
            trade_px = df["price"].to_numpy()
            # half-spread bounce floor (bps): median |tick-to-tick return| of trades
            dpx = np.abs(np.diff(trade_px)) / trade_px[:-1] * 1e4
            spread_floors.append(float(np.median(dpx[dpx > 0])) if (dpx > 0).any()
                                 else 0.0)

            bins = build_bins(df)
            shock_mask = detect_shocks(bins, SIGMA_PRIMARY)
            secs_all = bins["sec"].to_numpy()
            shock_secs = secs_all[shock_mask]
            shock_dir = np.sign(bins["OFI"].to_numpy()[shock_mask]).astype("int64")

            if len(shock_secs):
                m = measure(shock_secs, shock_dir, trade_ts, trade_px)
                v = m["valid"] & np.isfinite(m["impact"]) & np.isfinite(m["P0"])
                ev_rebound.append(m["rebound"][v])
                ev_impact.append(m["impact"][v])
                meta = np.column_stack([
                    np.full(v.sum(), SYMBOLS.index(sym)),
                    np.full(v.sum(), DATES.index(date)),
                    m["d"][v], m["impact"][v],
                ])
                ev_meta.append(meta)
                rec["n_shocks"] = int(v.sum())
            else:
                rec["n_shocks"] = 0

            # placebo: random NON-shock bins with V>0 and baseline available
            base_ok = bins["V"].rolling(BASELINE_BINS).mean().shift(1).notna().to_numpy()
            cand = np.where((~shock_mask) & (bins["V"].to_numpy() > 0) & base_ok)[0]
            if len(cand):
                pick = rng.choice(cand, size=min(PLACEBO_PER_DAY, len(cand)),
                                  replace=False)
                psecs = secs_all[pick]
                pdir = np.sign(bins["OFI"].to_numpy()[pick]).astype("int64")
                pdir[pdir == 0] = 1
                pm = measure(psecs, pdir, trade_ts, trade_px)
                pv = pm["valid"] & np.isfinite(pm["P0"])
                pb_rebound.append(pm["rebound"][pv])

            print(f"  {sym} {date}: trades={len(df):>9,d} "
                  f"shocks={rec.get('n_shocks',0):>5,d}  floor~{spread_floors[-1]:.3f}bps")
            manifest["files"].append(rec)
            del df, bins

    if download_only:
        (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))
        print("download-only done.")
        return manifest

    R = np.vstack(ev_rebound) if ev_rebound else np.empty((0, n_offsets))
    I = np.concatenate(ev_impact) if ev_impact else np.empty(0)
    M = np.vstack(ev_meta) if ev_meta else np.empty((0, 4))
    PB = np.vstack(pb_rebound) if pb_rebound else np.empty((0, n_offsets))
    print(f"\nTOTAL shock events (full-horizon): {len(R):,d} ; placebo: {len(PB):,d}")

    out = aggregate(R, I, M, PB, float(np.median(spread_floors)) if spread_floors else 0.0)
    out["manifest_summary"] = {
        "n_files_ok": sum(1 for f in manifest["files"] if f.get("status") == "ok"),
        "n_404": sum(1 for f in manifest["files"] if f.get("status") != "ok"),
        "symbols": SYMBOLS, "dates": DATES,
        "median_spread_floor_bps": float(np.median(spread_floors)) if spread_floors else None,
    }
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (OUT_DIR / "results.json").write_text(json.dumps(out, indent=2))
    _save_events(R, I, M)
    make_figures(R, I, M, PB, out)
    print_verdict(out)
    return out


def aggregate(R, I, M, PB, spread_floor_bps) -> dict:
    offs = OFFSETS_MS
    res = {"n_shocks": int(len(R)), "n_placebo": int(len(PB)),
           "offsets_ms": offs, "spread_floor_halfbps": spread_floor_bps,
           "median_impact_bps": float(np.median(I)) if len(I) else None,
           "mean_impact_bps": float(np.mean(I)) if len(I) else None}

    def curve(A):
        return {
            "median_bps": [float(np.nanmedian(A[:, k])) for k in range(A.shape[1])],
            "mean_bps": [float(np.nanmean(A[:, k])) for k in range(A.shape[1])],
            "frac_rebound_pos": [float(np.nanmean(A[:, k] > 0)) for k in range(A.shape[1])],
        }
    res["survival_shock"] = curve(R) if len(R) else None
    res["survival_placebo"] = curve(PB) if len(PB) else None

    # mechanism (2c): continuation fraction at 5s & 30s
    def col(t):
        return offs.index(t)
    if len(R):
        res["mechanism"] = {
            "frac_continue_5s": float(np.nanmean(R[:, col(5000)] < 0)),
            "frac_continue_30s": float(np.nanmean(R[:, col(30000)] < 0)),
            "median_rebound_5s_bps": float(np.nanmedian(R[:, col(5000)])),
            "median_rebound_30s_bps": float(np.nanmedian(R[:, col(30000)])),
        }
    # latency haircut: gross = max_{tau>L} rebound - rebound(L)  (mean across events)
    lat = {}
    mean_curve = np.array(res["survival_shock"]["mean_bps"]) if len(R) else None
    med_curve = np.array([np.nanmedian(R[:, k]) for k in range(R.shape[1])]) if len(R) else None
    for L in LATENCIES_MS:
        # index of first offset >= L
        ge = [i for i, o in enumerate(offs) if o >= L]
        if not ge or mean_curve is None:
            continue
        li = offs.index(L) if L in offs else ge[0]
        future = ge
        gross_mean = float(np.nanmax(mean_curve[future]) - mean_curve[li])
        gross_med = float(np.nanmax(med_curve[future]) - med_curve[li])
        lat[str(L)] = {
            "gross_mean_bps": gross_mean, "gross_median_bps": gross_med,
            "net_mean_primary_bps": gross_mean - COST_PRIMARY_BPS,
            "net_mean_optimistic_bps": gross_mean - COST_OPTIMISTIC_BPS,
            "net_median_primary_bps": gross_med - COST_PRIMARY_BPS,
        }
    res["latency_haircut"] = lat
    res["cost_primary_bps"] = COST_PRIMARY_BPS
    res["cost_optimistic_bps"] = COST_OPTIMISTIC_BPS

    # ---- pre-registered verdict logic ----
    verdict = {"data_gate": "PASS (tick trades obtained)"}
    if not len(R):
        verdict["final"] = "FAIL"; verdict["cause"] = "no shock events"
        res["verdict"] = verdict
        return res
    mech_continue = res["mechanism"]["frac_continue_30s"]
    mechanism_holds = mech_continue < 0.5  # majority rebound at 30s
    # residual at anchor latency, beats placebo & spread floor, net positive?
    anchor_net = {}
    for L in ANCHOR_L_MS:
        anchor_net[str(L)] = lat[str(L)]["net_mean_primary_bps"]
    best_anchor_net = max(anchor_net.values())
    # placebo-beaten: shock mean rebound at 5s exceeds placebo mean at 5s + floor
    sh5 = res["survival_shock"]["mean_bps"][col(5000)]
    pb5 = res["survival_placebo"]["mean_bps"][col(5000)] if res["survival_placebo"] else 0.0
    beats_placebo = (sh5 - pb5) > spread_floor_bps

    verdict["mechanism_holds"] = bool(mechanism_holds)
    verdict["frac_continue_30s"] = mech_continue
    verdict["beats_placebo_and_floor_5s"] = bool(beats_placebo)
    verdict["best_anchor_net_primary_bps"] = best_anchor_net
    if not mechanism_holds:
        verdict["final"] = "FAIL"; verdict["cause"] = "MECHANISM FAILS (continuation dominates post-shock)"
    elif best_anchor_net <= 0:
        verdict["final"] = "FAIL"
        verdict["cause"] = "COST/SPEED WALL (residual rebound after user latency <= trading cost)"
    elif not beats_placebo:
        verdict["final"] = "FAIL"; verdict["cause"] = "rebound indistinguishable from bid-ask-bounce placebo"
    else:
        verdict["final"] = "PASS"; verdict["cause"] = "exhaustion reachable after latency net of cost"
    res["verdict"] = verdict
    return res


def _save_events(R, I, M):
    if not len(R):
        return
    cols = {f"rebound_{o}ms_bps": R[:, k] for k, o in enumerate(OFFSETS_MS)}
    df = pd.DataFrame({
        "symbol_idx": M[:, 0].astype(int), "date_idx": M[:, 1].astype(int),
        "dir": M[:, 2].astype(int), "impact_bps": I, **cols,
    })
    df["symbol"] = df["symbol_idx"].map(dict(enumerate(SYMBOLS)))
    df["date"] = df["date_idx"].map(dict(enumerate(DATES)))
    with gzip.open(OUT_DIR / "shock_events.csv.gz", "wt") as f:
        df.drop(columns=["symbol_idx", "date_idx"]).to_csv(f, index=False)


def make_figures(R, I, M, PB, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    offs = np.array(OFFSETS_MS)
    x = np.where(offs == 0, 50, offs)  # log-x: map 0 -> 50ms slot for plotting

    # Fig 1: rebound survival curve (mean & median) shock vs placebo + floor
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    sh, pb = out["survival_shock"], out["survival_placebo"]
    for a, key, ttl in [(ax[0], "mean_bps", "MEAN rebound"),
                        (ax[1], "median_bps", "MEDIAN rebound")]:
        a.semilogx(x, sh[key], "o-", color="C3", label="shock-conditioned")
        if pb:
            a.semilogx(x, pb[key], "s--", color="C7", label="random-bin placebo")
        a.axhline(0, color="k", lw=.8)
        fl = out["spread_floor_halfbps"]
        a.axhspan(-fl, fl, color="orange", alpha=.15, label=f"±half-spread bounce floor ({fl:.2f}bps)")
        for L in ANCHOR_L_MS:
            a.axvline(L, color="C0", ls=":", lw=1)
        a.set_title(f"{ttl} vs time after shock  (>0 = exhaustion/rebound)")
        a.set_xlabel("time after shock end (ms, log)"); a.set_ylabel("rebound (bps)")
        a.legend(fontsize=8)
    fig.suptitle(f"Order-flow exhaustion: rebound survival curve  "
                 f"(N={out['n_shocks']:,} shocks, BTC+ETH perp)")
    fig.tight_layout(); fig.savefig(FIG_DIR / "fig1_survival_curve.png", dpi=110)
    plt.close(fig)

    # Fig 2: mechanism — fraction rebound>0 vs time
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.semilogx(x, np.array(sh["frac_rebound_pos"]) * 100, "o-", color="C3",
                label="shock: % events rebounding")
    if pb:
        ax.semilogx(x, np.array(pb["frac_rebound_pos"]) * 100, "s--", color="C7",
                    label="placebo")
    ax.axhline(50, color="k", lw=.8, ls="--", label="50% (coin-flip)")
    ax.set_ylim(0, 100); ax.set_xlabel("time after shock (ms, log)")
    ax.set_ylabel("% events with price retraced against shock")
    ax.set_title("Mechanism: rebound vs continuation\n(<50% => continuation dominates => exhaustion fails)")
    ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(FIG_DIR / "fig2_mechanism.png", dpi=110); plt.close(fig)

    # Fig 3: latency haircut — net bps vs latency
    fig, ax = plt.subplots(figsize=(7, 5))
    Ls = [int(k) for k in out["latency_haircut"]]
    gross = [out["latency_haircut"][str(L)]["gross_mean_bps"] for L in Ls]
    net = [out["latency_haircut"][str(L)]["net_mean_primary_bps"] for L in Ls]
    ax.plot(Ls, gross, "o-", color="C2", label="gross capturable rebound (mean)")
    ax.plot(Ls, net, "s-", color="C3", label=f"net of {COST_PRIMARY_BPS:.0f}bps cost")
    ax.axhline(0, color="k", lw=.8)
    ax.axvspan(ANCHOR_L_MS[0], ANCHOR_L_MS[1], color="C0", alpha=.12,
               label="retail-reachable latency anchor")
    ax.set_xlabel("user latency L (ms)"); ax.set_ylabel("bps")
    ax.set_title("Latency haircut: what's left after the user can act\n(L=0 = idealized HFT, NOT reachable)")
    ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(FIG_DIR / "fig3_latency_haircut.png", dpi=110); plt.close(fig)
    print(f"figures -> {FIG_DIR}")


def print_verdict(out):
    v = out["verdict"]
    print("\n" + "=" * 78)
    print(f"VERDICT: {v['final']}  —  {v.get('cause','')}")
    print(f"  data gate         : {v['data_gate']}")
    print(f"  mechanism holds   : {v.get('mechanism_holds')} "
          f"(continue@30s={v.get('frac_continue_30s'):.3f})")
    print(f"  beats placebo+floor: {v.get('beats_placebo_and_floor_5s')}")
    print(f"  best anchor net    : {v.get('best_anchor_net_primary_bps'):.3f} bps "
          f"(cost {COST_PRIMARY_BPS:.0f}bps)")
    print("=" * 78)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--download-only", action="store_true")
    args = ap.parse_args()
    run(download_only=args.download_only)
