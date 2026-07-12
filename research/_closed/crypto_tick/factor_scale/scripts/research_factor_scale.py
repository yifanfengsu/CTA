#!/usr/bin/env python3
"""Stage A pre-gate: does cross-sectional factor ALPHA improve with UNIVERSE SCALE?

This is the ZERO-COST front gate of the "large-scale factor" big-engineering route.
It is NOT a hunt for "a factor with significant IC" — with hundreds of coins x many
factors under multiple testing you are GUARANTEED to find "significant" factors, and
that is precisely the trap (cf. MR cointegration: 231 "cointegrated" pairs ~= pure
noise, 11.55 false-positive pairs/window). The ONLY question here is:

    Taking the SAME pre-registered factors from 22 coins to ~100 coins, does the
    alpha get STRONGER, stay the SAME, or get WEAKER?
      up   = scale itself brings edge (small-cap inefficiency = real alpha big caps
             lack)  -> worth the big engineering + institutional data spend
      flat = scale contributes nothing (same weak alpha, just measured with smaller SE)
      down / indistinguishable-from-noise = scale only adds more false positives

Two lessons are wired in from the start:
  (1) cross-sectional study (22-coin MOM/CAR/REV all FAILED: alpha too weak + decayed
      + eaten by cost; IC != tradeable alpha).
  (2) MR cointegration study (MISSED noise calibration -> a possible multiple-testing
      false positive was misread as a mechanism). THIS task installs noise calibration
      at step one.

JUDGEMENT PHILOSOPHY (frozen, do not delete from report):
  * Noise calibration BEFORE IC interpretation: a real IC must beat the "shuffled-
    returns false-positive baseline", NOT beat 0 (under multiple testing, noise alone
    produces positive IC).
  * Factor set PRE-REGISTERED & FIXED (no p-hacking in factor space). This fixed set
    is THE difference from the "hundreds of coins, pick the significant factor" trap.
  * No Sharpe as the primary verdict (cross-sectional discipline): look at IC vs noise
    baseline / tradeable spread / liquidity-tier source.

================== PRE-REGISTERED UNIVERSE (frozen before results) ==================
  Free public daily klines, Binance Vision UM-perp (production CDN; no demo variant).
  Target ~100 coins = top by recent (2026-05) USDT quote volume. Nested pools by
  volume rank: pool_22 (anchor, aligned to the cross-sectional study's set) subset of
  pool_50 subset of pool_100.
  Common study window: chosen as the LONGEST window over which >=100 ranked coins have
  FULL daily history (deterministic, availability-driven, NOT result-driven) so the
  scale gradient is apples-to-apples (every day pool_K has exactly K coins).
  SURVIVORSHIP (frozen, verbatim into report): current-listed coins only = survivor
  bias (dead coins e.g. LUNA/FTT excluded), so alpha is SYSTEMATICALLY OPTIMISTIC.
  Direction is known: an optimistic alpha judged DEAD => conclusion is the more robust
  (truth is worse); judged ALIVE => must flag "needs delisted-inclusive institutional
  data to confirm; this is an optimistic upper bound". Point-in-time is ideal but not
  feasible on free data; this limit is stated, not hidden.

================== PRE-REGISTERED FACTORS (frozen; zero variants) ===================
  ONLY the three already-tested cross-sectional factors (caliber copied verbatim from
  scripts/research_cross_sectional_ic.py; parameters NOT re-searched, NONE added):
    F-MOM momentum : 30-day cumulative return, long-pref score = +ret30
    F-CAR carry    : 7-day mean funding rate, long-pref score = -mean_funding
                     (most negative funding = longs paid = long). CAR is computed only
                     if free funding history loads; else CAR is marked PENDING and
                     MOM/REV proceed (CAR does not block).
    F-REV reversal : 3-day cumulative return, long-pref score = -ret3
  IC = Spearman(long-pref score, forward 1-day return); IC>0 = factor adds value in its
  intended direction. Daily rebalance at UTC 00:00 using info through the prior close.
  Day index i: score uses closes <= i-1; fwd1 = close[i]/close[i-1]-1 (return of day i).
  Fixed factor set = NO multiple testing in factor space (the core defence).

================== NOISE CALIBRATION (the gate MR-cointegration missed) =============
  Primary null (NULL-A, per task spec "每个币的收益序列做 block-shuffle"): for each
  coin INDEPENDENTLY, moving-block bootstrap (block=5d) its forward-return series in
  time, keeping the REAL score cross-section fixed. This destroys the score->return
  contemporaneous cross-sectional relationship while preserving each coin's marginal
  distribution + short serial structure. >=200 reshuffles -> null distribution of the
  mean daily IC = the false-positive baseline.
  Secondary null (NULL-B, robustness): SAME block bootstrap applied to whole return
  ROWS (one index for all coins) -> preserves cross-coin contemporaneous correlation,
  breaks only score<->return time alignment.
  A real IC counts as signal ONLY if it beats the null's 95th percentile (one-sided,
  factors are sign-directed) with permutation p<0.05 -- beating 0 does NOT count.

================== SCALE GRADIENT (the core question) ===============================
  For K in {22,50,100}: real mean-IC + IC_IR + that pool's OWN null baseline (NULL run
  once per K). READ THE IC POINT ESTIMATE TREND, NOT the t-stat: t rises mechanically
  with N (more names -> smaller SE) and will fool you. Up=scale edge; flat=no scale
  contribution; down=scale adds noise.

================== LIQUIDITY STRATIFICATION (anti-disguise) =========================
  Split pool_100 by 2026-05 volume into top33/mid33/bottom34. Per-tier IC + long-short
  gross spread + rough tradeable-cost estimate. If alpha lives only in the bottom
  (least-tradeable) tier => liquidity-premium DISGUISE (impact eats it; cost wall is
  higher on small caps) => not monetizable. Alpha present in top/mid => more credible.

================== VERDICT (FROZEN before results; iron rule A) =====================
  PASS (large-scale factor worth the big engineering, proceed) <=> ALL of:
    (1) >=1 factor real mean-IC > its pool-100 NULL 95th pct AND one-sided perm p<0.05
        (beats the noise baseline, not 0).
    (2) that factor's alpha NOT mainly from the bottom liquidity tier: its top-OR-mid
        tier IC is >= 0.5x its bottom-tier IC AND that top/mid IC itself exceeds the
        tier's null 95th pct (edge survives where trading is feasible).
    (3) that factor's real mean-IC POINT ESTIMATE improves with scale: IC(100) exceeds
        IC(22) by more than one null SE, AND IC(50) is not below IC(22) by more than
        one null SE (upward, not U-shaped-down).
  FAIL (not worth it) <=> ANY of:
    - IC indistinguishable from the noise baseline (multiple-testing artefact), or
    - alpha concentrated in the untradeable small-cap tier (liquidity disguise), or
    - expanding the pool does NOT improve (flat) or weakens the IC point estimate.
  No Sharpe as primary verdict. Survivor bias makes alpha optimistic: DEAD => robust;
  ALIVE => "needs delisted-inclusive institutional data recheck; optimistic upper bound".

DATA: Binance Vision public 1d klines + fundingRate (READ-ONLY pulls to
data/binance_vision/). Contaminated DB never touched; vrp line / forward system / VPS
never touched. OKX mainnet DB used READ-ONLY for the 5-coin cross-validation gate only.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sqlite3
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[5]  # 2026-07 重构批次5：迁入 research/_closed/crypto_tick/factor_scale/scripts/，深度 1→5
BV = PROJECT_ROOT / "data" / "binance_vision"
OUT = PROJECT_ROOT / "reports" / "factor_scale_feasibility_20260628"
MAINNET_DB = PROJECT_ROOT / ".vntrader" / "database_mainnet.db"

S3 = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
CDN = "https://data.binance.vision"
KPREFIX = "data/futures/um/monthly/klines/"
BASE = "https://data.binance.vision/data/futures/um/monthly"

# ---- frozen pre-registered constants ----
MOM_W, CAR_W, REV_W = 30, 7, 3
POOLS = (22, 50, 100)
RANK_MONTH = "2026-05"          # latest complete month (today 2026-06-28)
END_MONTH = "2026-05"
N_SHUFFLE = 200
BLOCK = 5
APY = 365.0
FWD_KS = (1, 3, 5)
SEED = 20260628

WORKERS = 24
TIMEOUT = 60
RETRIES = 4

LOG: list[str] = []


def L(msg: str = "") -> None:
    print(msg, flush=True)
    LOG.append(str(msg))


def fetch(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "cta-factor-scale/1.0"})
    last = None
    for attempt in range(1, RETRIES + 1):
        try:
            with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
                return r.read()
        except urllib.error.HTTPError:
            raise
        except Exception as exc:  # noqa: BLE001 transient
            last = exc
            time.sleep(min(2 ** attempt, 10))
    raise last  # type: ignore[misc]


def sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def month_range(start: str, end: str) -> list[str]:
    ys, ms = map(int, start.split("-"))
    ye, me = map(int, end.split("-"))
    out = []
    while (ys, ms) <= (ye, me):
        out.append(f"{ys}-{ms:02d}")
        ms += 1
        if ms == 13:
            ys, ms = ys + 1, 1
    return out


def _findall(xml: str, tag: str) -> list[str]:
    out, i, op, cl = [], 0, f"<{tag}>", f"</{tag}>"
    while True:
        a = xml.find(op, i)
        if a < 0:
            break
        b = xml.find(cl, a)
        out.append(xml[a + len(op):b])
        i = b + len(cl)
    return out


# ───────────────────────── data acquisition layer ─────────────────────────
def list_usdt_perps() -> list[str]:
    syms, marker = [], ""
    while True:
        u = f"{S3}?delimiter=/&prefix={KPREFIX}" + (f"&marker={urllib.parse.quote(marker)}" if marker else "")
        xml = fetch(u).decode()
        pref = _findall(xml, "Prefix")
        for p in pref:
            n = p[len(KPREFIX):].rstrip("/")
            if n:
                syms.append(n)
        if "<IsTruncated>true</IsTruncated>" not in xml:
            break
        nm = _findall(xml, "NextMarker")
        marker = nm[0] if nm else (pref[-1][len(KPREFIX):].rstrip("/") if pref else "")
        if not marker:
            break
    # clean ASCII alphanumeric tickers only (drops 3 joke symbols with CJK chars)
    return sorted(s for s in syms if s.endswith("USDT") and s.isascii()
                  and s[:-4].isalnum())


def quote_volume(symbol: str, month: str) -> float | None:
    url = f"{CDN}/{KPREFIX}{symbol}/1d/{symbol}-1d-{month}.zip"
    try:
        blob = fetch(url)
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        return None
    try:
        with zipfile.ZipFile(io.BytesIO(blob)) as z:
            raw = z.read(z.namelist()[0]).decode().strip().splitlines()
    except Exception:  # noqa: BLE001
        return None
    tot = 0.0
    for line in raw:
        f = line.split(",")
        try:
            tot += float(f[7])
        except (ValueError, IndexError):
            continue
    return tot


def list_1d_months(symbol: str) -> list[str]:
    """All available YYYY-MM for the symbol's 1d klines (S3 listing)."""
    months: list[str] = []
    marker = ""
    pre = f"{KPREFIX}{symbol}/1d/"
    while True:
        u = f"{S3}?prefix={pre}" + (f"&marker={urllib.parse.quote(marker)}" if marker else "")
        try:
            xml = fetch(u).decode()
        except Exception:  # noqa: BLE001
            break
        keys = _findall(xml, "Key")
        for k in keys:
            if "-1d-" in k and k.endswith(".zip"):
                months.append(k.split("-1d-")[1].replace(".zip", ""))
        if "<IsTruncated>true</IsTruncated>" not in xml:
            break
        nm = _findall(xml, "NextMarker")
        marker = nm[0] if nm else (keys[-1] if keys else "")
        if not marker:
            break
    return sorted(set(months))


def download_zip(url: str, dest: Path) -> str:
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        want = fetch(url + ".CHECKSUM").decode().split()[0].strip()
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return "not_available_404"
        return f"FAILED checksum {e}"
    if dest.exists() and sha256(dest.read_bytes()) == want:
        return "cached_verified"
    try:
        blob = fetch(url)
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return "not_available_404"
        return f"FAILED {e}"
    if sha256(blob) != want:
        return "FAILED sha256_mismatch"
    dest.write_bytes(blob)
    return "downloaded_verified"


def kline_1d_url(sym: str, month: str) -> str:
    return f"{BASE}/klines/{sym}/1d/{sym}-1d-{month}.zip"


def kline_1d_dest(sym: str, month: str) -> Path:
    return BV / sym / "1d" / f"{sym}-1d-{month}.zip"


def funding_url(sym: str, month: str) -> str:
    return f"{BASE}/fundingRate/{sym}/{sym}-fundingRate-{month}.zip"


def funding_dest(sym: str, month: str) -> Path:
    return BV / "funding" / sym / f"{sym}-fundingRate-{month}.zip"


# ───────────────────────── loaders (downloaded zips) ─────────────────────────
def _ts_to_day(ts: np.ndarray) -> np.ndarray:
    """epoch ms (or us) -> integer UTC day index (days since 1970-01-01)."""
    ts = ts.astype("float64")
    ms = np.where(ts > 1e14, ts / 1000.0, ts)  # us-> ms for safety (1d is ms in practice)
    return (ms // 86_400_000).astype("int64")


def load_daily_close(sym: str) -> pd.Series:
    frames = []
    for zp in sorted((BV / sym / "1d").glob(f"{sym}-1d-*.zip")):
        with zipfile.ZipFile(zp) as z:
            raw = z.read(z.namelist()[0]).decode().strip().splitlines()
        for line in raw:
            f = line.split(",")
            try:
                ot = float(f[0]); cl = float(f[4])
            except (ValueError, IndexError):
                continue  # header
            frames.append((ot, cl))
    if not frames:
        return pd.Series(dtype=float)
    arr = np.array(frames)
    day = _ts_to_day(arr[:, 0])
    s = pd.Series(arr[:, 1], index=day)
    return s.groupby(level=0).last().sort_index()


def load_daily_funding(sym: str) -> pd.Series:
    frames = []
    for zp in sorted((BV / "funding" / sym).glob(f"{sym}-fundingRate-*.zip")):
        with zipfile.ZipFile(zp) as z:
            raw = z.read(z.namelist()[0]).decode().strip().splitlines()
        for line in raw:
            f = line.split(",")
            try:
                ct = float(f[0]); rate = float(f[2])
            except (ValueError, IndexError):
                continue
            frames.append((ct, rate))
    if not frames:
        return pd.Series(dtype=float)
    arr = np.array(frames)
    day = _ts_to_day(arr[:, 0])
    s = pd.Series(arr[:, 1], index=day)
    return s.groupby(level=0).mean().sort_index()  # daily mean funding


# ───────────────────────── IC / noise machinery ─────────────────────────
def rank_rows(X: np.ndarray) -> np.ndarray:
    """Ordinal rank along axis=1 (continuous data -> ties negligible)."""
    order = X.argsort(axis=1, kind="stable")
    ranks = np.empty_like(order, dtype=np.int64)
    T, K = X.shape
    ranks[np.arange(T)[:, None], order] = np.arange(K)
    return ranks.astype(float)


def rowwise_corr(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    Am = A - A.mean(1, keepdims=True)
    Bm = B - B.mean(1, keepdims=True)
    num = (Am * Bm).sum(1)
    den = np.sqrt((Am * Am).sum(1) * (Bm * Bm).sum(1))
    out = np.full(A.shape[0], np.nan)
    ok = den > 0
    out[ok] = num[ok] / den[ok]
    return out


def block_idx_1d(n: int, block: int, rng: np.random.Generator) -> np.ndarray:
    """Moving-block bootstrap index of length n (vectorised)."""
    nb = int(np.ceil(n / block))
    starts = rng.integers(0, n, size=nb)
    offs = np.arange(block)
    return ((starts[:, None] + offs[None, :]) % n).ravel()[:n]


def block_idx_2d(n: int, k: int, block: int, rng: np.random.Generator) -> np.ndarray:
    """k INDEPENDENT moving-block bootstrap index columns, shape (n, k)."""
    nb = int(np.ceil(n / block))
    starts = rng.integers(0, n, size=(k, nb))
    offs = np.arange(block)
    idx = (starts[:, :, None] + offs[None, None, :]) % n      # (k, nb, block)
    return idx.reshape(k, nb * block)[:, :n].T                # (n, k)


def ic_stats(ic_series: np.ndarray) -> dict:
    ic = ic_series[np.isfinite(ic_series)]
    n = len(ic)
    m = float(ic.mean())
    sd = float(ic.std(ddof=1)) if n > 1 else np.nan
    ir = m / sd if sd and sd > 0 else np.nan
    return {"mean": m, "std": sd, "ir": ir, "ir_ann": ir * np.sqrt(APY) if np.isfinite(ir) else np.nan,
            "t": (m / sd * np.sqrt(n)) if sd and sd > 0 else np.nan,
            "pos_share": float((ic > 0).mean()), "n_days": n}


def real_and_null(score: np.ndarray, ret: np.ndarray, rng: np.random.Generator,
                  n_shuffle: int = N_SHUFFLE) -> dict:
    """score, ret: complete-case [T x K] panels. Returns real IC stats + null dists.

    NULL-A: per-coin independent block bootstrap of the return column (primary).
    NULL-B: shared row block bootstrap (preserves cross-coin corr) (secondary).
    """
    S_rank = rank_rows(score)
    real_ic = rowwise_corr(S_rank, rank_rows(ret))
    real = ic_stats(real_ic)
    T, K = ret.shape

    nullA, nullB = np.empty(n_shuffle), np.empty(n_shuffle)
    for b in range(n_shuffle):
        # NULL-A: independent per coin (vectorised gather)
        idxA = block_idx_2d(T, K, BLOCK, rng)                 # (T, K)
        Rb = np.take_along_axis(ret, idxA, axis=0)
        nullA[b] = np.nanmean(rowwise_corr(S_rank, rank_rows(Rb)))
        # NULL-B: shared rows (preserves cross-coin contemporaneous corr)
        nullB[b] = np.nanmean(rowwise_corr(S_rank, rank_rows(ret[block_idx_1d(T, BLOCK, rng)])))

    def pack(null: np.ndarray) -> dict:
        return {"null_mean": float(null.mean()), "null_std": float(null.std(ddof=1)),
                "p95": float(np.percentile(null, 95)), "p99": float(np.percentile(null, 99)),
                "p_one_sided": float((null >= real["mean"]).mean()),
                "real_percentile": float((null < real["mean"]).mean() * 100.0),
                "exceeds_p95": bool(real["mean"] > np.percentile(null, 95))}

    return {"real": real, "real_ic_series": real_ic, "nullA": pack(nullA), "nullB": pack(nullB),
            "nullA_dist": nullA, "nullB_dist": nullB}


# ───────────────────────── panel construction ─────────────────────────
def build_panels(coins: list[str], vols: dict[str, float], have_funding: set[str]):
    """Load closes/funding, return aligned matrices + score panels over full days."""
    closes, fundings = {}, {}
    for c in coins:
        closes[c] = load_daily_close(c)
        if c in have_funding:
            fundings[c] = load_daily_funding(c)
    days_all = sorted(set().union(*[set(s.index) for s in closes.values()]))
    C = pd.DataFrame(index=days_all, columns=coins, dtype=float)
    F = pd.DataFrame(index=days_all, columns=coins, dtype=float)
    for c in coins:
        C[c] = closes[c].reindex(days_all)
        if c in fundings:
            F[c] = fundings[c].reindex(days_all)
    FundMA = F.rolling(CAR_W, min_periods=CAR_W).mean()
    return C, FundMA, days_all


def factor_scores(C: pd.DataFrame, FundMA: pd.DataFrame):
    """Return dict factor -> score DataFrame and fwd-return DataFrames, day-aligned."""
    logc = C  # use levels
    ret1 = C / C.shift(1) - 1.0           # fwd1 indexed at the realisation day i
    mom = C.shift(1) / C.shift(MOM_W + 1) - 1.0
    rev = -(C.shift(1) / C.shift(REV_W + 1) - 1.0)
    car = -FundMA.shift(1)
    return {"F-MOM": mom, "F-REV": rev, "F-CAR": car}, ret1


def ls_spread(score: np.ndarray, ret: np.ndarray, fee: float) -> dict:
    """Quintile long-short (top/bottom 20%), daily rebalance. gross/cost/net annualised.

    Rough tradeability proxy (Part 4c): IC != tradeable alpha. Cost = daily portfolio
    turnover * per-side taker fee (fee passed in; bottom tiers use a wider fee). Daily
    reversal-type signals turn over ~fully -> cost wall is large by construction.
    """
    T, K = score.shape
    nq = max(1, K // 5)
    gross = np.empty(T); turn = np.empty(T)
    prevw = np.zeros(K)
    for t in range(T):
        order = np.argsort(score[t], kind="stable")
        short, long = order[:nq], order[-nq:]
        w = np.zeros(K); w[long] = 1.0 / nq; w[short] = -1.0 / nq
        gross[t] = ret[t, long].mean() - ret[t, short].mean()
        turn[t] = np.abs(w - prevw).sum()
        prevw = w
    cost = turn * fee
    net = gross - cost
    return {"gross_ann_pct": float(gross.mean() * APY * 100),
            "cost_ann_pct": float(cost.mean() * APY * 100),
            "net_ann_pct": float(net.mean() * APY * 100),
            "avg_daily_turnover": float(turn.mean()),
            "cost_over_gross": float(cost.mean() / gross.mean()) if gross.mean() != 0 else None,
            "net_positive": bool(net.mean() > 0), "fee_per_side_pct": fee * 100, "T": T}


def complete_panel(score: pd.DataFrame, ret1: pd.DataFrame, cols: list[str], days: list[int]):
    s = score.loc[days, cols]
    r = ret1.loc[days, cols]
    mask = s.notna().all(axis=1) & r.notna().all(axis=1)
    return s[mask].to_numpy(float), r[mask].to_numpy(float), int(mask.sum())


# ───────────────────────── cross-validation gate ─────────────────────────
OKX_MAP = {"BTCUSDT": "BTCUSDT_SWAP_OKX", "ETHUSDT": "ETHUSDT_SWAP_OKX",
           "SOLUSDT": "SOLUSDT_SWAP_OKX", "LINKUSDT": "LINKUSDT_SWAP_OKX",
           "DOGEUSDT": "DOGEUSDT_SWAP_OKX"}


def cross_validate(rng: np.random.Generator) -> dict:
    """Binance 1d close vs OKX mainnet DB daily close, random days, overlap coins."""
    if not MAINNET_DB.exists():
        return {"status": "mainnet_db_absent"}
    con = sqlite3.connect(f"file:{MAINNET_DB}?mode=ro", uri=True)
    res = {}
    for bsym, osym in OKX_MAP.items():
        bd = load_daily_close(bsym)
        if bd.empty:
            continue
        rows = con.execute(
            "SELECT datetime, close_price FROM dbbardata WHERE symbol=? AND interval='1m'",
            (osym,)).fetchall()
        if not rows:
            continue
        # OKX DB stores naive Shanghai bar-open -> UTC = -8h (validated: -8h gives
        # 0.035% median dev vs Binance, matching data_trust_closure; +0h gives 0.61%).
        dt = pd.to_datetime([r[0] for r in rows]) - pd.Timedelta(hours=8)
        cp = np.array([r[1] for r in rows], float)
        day = dt.values.astype("datetime64[D]").astype("int64")  # epoch-day, resolution-agnostic
        okx_daily = pd.Series(cp, index=day).groupby(level=0).last()
        common = sorted(set(bd.index) & set(okx_daily.index))
        if len(common) < 5:
            continue
        pick = rng.choice(common, size=min(5, len(common)), replace=False)
        devs = [abs(bd.loc[d] / okx_daily.loc[d] - 1.0) for d in pick]
        res[bsym] = {"sample_days": [int(x) for x in pick],
                     "median_abs_dev_pct": float(np.median(devs) * 100),
                     "max_abs_dev_pct": float(np.max(devs) * 100)}
    con.close()
    allmed = float(np.median([v["median_abs_dev_pct"] for v in res.values()])) if res else None
    return {"status": "ok", "per_symbol": res, "overall_median_abs_dev_pct": allmed,
            "gate_pass": (allmed is not None and allmed < 0.5)}


# ───────────────────────── modes ─────────────────────────
def mode_survey() -> int:
    L(f"run start (UTC): {datetime.now(timezone.utc).isoformat()}")
    L("DATA ENVIRONMENT: BINANCE-VISION-PUBLIC (production; no demo variant)")
    syms = list_usdt_perps()
    L(f"universe: {len(syms)} USDT UM-perp symbols")
    vols: dict[str, float] = {}
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(quote_volume, s, RANK_MONTH): s for s in syms}
        for n, fut in enumerate(as_completed(futs), 1):
            v = fut.result()
            if v:
                vols[futs[fut]] = v
            if n % 150 == 0:
                L(f"  volume probed {n}/{len(syms)}")
    ranked = sorted(vols.items(), key=lambda kv: kv[1], reverse=True)
    L(f"{len(ranked)} symbols have a {RANK_MONTH} 1d file (ranked by quote volume)")
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        months = dict(zip([s for s, _ in ranked], ex.map(list_1d_months, [s for s, _ in ranked])))
    cand = []
    for rank, (s, v) in enumerate(ranked, 1):
        ms = months[s]
        cand.append({"rank": rank, "symbol": s, "quote_volume_2026_05": v,
                     "earliest_1d_month": ms[0] if ms else None,
                     "n_months": len(ms)})
    # FRONTIER: for each window start, how many FULL-HISTORY coins exist (ranked by vol).
    # Pool design = top-K by volume AMONG coins with earliest <= start (balanced panel).
    L("\n== full-history frontier: #coins (of all ranked) with earliest_1d_month <= start ==")
    for start in ("2021-01", "2022-01", "2022-07", "2023-01", "2023-07", "2024-01", "2024-07", "2025-01"):
        full = [c for c in cand if c["earliest_1d_month"] and c["earliest_1d_month"] <= start]
        # volume rank of the 100th full-history coin (how deep we reach)
        rank100 = full[99]["rank"] if len(full) >= 100 else None
        L(f"  start {start}: full-history coins={len(full):>3}  (>=100? {'YES' if len(full)>=100 else 'no'}; "
          f"100th coin volume-rank={rank100})")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "survey_candidates.json").write_text(json.dumps(
        {"generated_at": datetime.now(timezone.utc).isoformat(), "rank_month": RANK_MONTH,
         "n_universe": len(syms), "n_with_rank_file": len(ranked), "candidates": cand}, indent=2))
    (OUT / "survey_log.txt").write_text("\n".join(LOG) + "\n")
    L(f"\nwrote {OUT/'survey_candidates.json'} ({len(cand)} candidates)")
    return 0


def mode_download(window_start: str, n_pool: int) -> int:
    L(f"run start (UTC): {datetime.now(timezone.utc).isoformat()}")
    L("DATA ENVIRONMENT: BINANCE-VISION-PUBLIC (production)")
    cand = json.loads((OUT / "survey_candidates.json").read_text())["candidates"]
    # final pool = top n_pool by volume with full history over the window (earliest<=window_start)
    pool = [c for c in cand if c["earliest_1d_month"] and c["earliest_1d_month"] <= window_start][:n_pool]
    syms = [c["symbol"] for c in pool]
    L(f"pool: {len(syms)} coins, window_start<= {window_start}, klines {window_start}..{END_MONTH}")
    months = month_range(window_start, END_MONTH)
    jobs = [("k", s, m) for s in syms for m in months] + [("f", s, m) for s in syms for m in months]
    results = []
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {}
        for kind, s, m in jobs:
            if kind == "k":
                futs[ex.submit(download_zip, kline_1d_url(s, m), kline_1d_dest(s, m))] = (kind, s, m)
            else:
                futs[ex.submit(download_zip, funding_url(s, m), funding_dest(s, m))] = (kind, s, m)
        for n, fut in enumerate(as_completed(futs), 1):
            kind, s, m = futs[fut]
            st = fut.result()
            results.append({"kind": kind, "symbol": s, "month": m, "status": st})
            if st.startswith("FAILED"):
                L(f"  FAILED {kind} {s} {m}: {st}")
            if n % 400 == 0:
                L(f"  [{n}/{len(jobs)}]")
    nfail = sum(1 for r in results if r["status"].startswith("FAILED"))
    n404 = sum(1 for r in results if r["status"] == "not_available_404")
    nok = len(results) - nfail
    manifest = {
        "source": "binance-vision-public-static", "server": "BINANCE-PRODUCTION",
        "datasets": ["futures/um monthly klines 1d", "futures/um monthly fundingRate"],
        "window": [window_start, END_MONTH], "n_pool": len(syms), "pool_symbols": syms,
        "rank_metric": f"{RANK_MONTH} 1d quote volume (USDT)",
        "survivorship": "current-listed only; delisted coins absent; alpha optimistic",
        "checksum": "sha256 via vision .CHECKSUM, all verified",
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
        "n_files": len(results), "n_ok": nok, "n_failed": nfail, "n_not_available_404": n404,
        "pool": pool,
    }
    BV.mkdir(parents=True, exist_ok=True)
    (BV / "factor_scale_manifest.json").write_text(json.dumps(manifest, indent=2))
    L(f"\ndownload done: {nok}/{len(results)} ok, {nfail} failed, {n404} 404")
    L(f"manifest -> {BV/'factor_scale_manifest.json'}")
    (OUT / "download_log.txt").write_text("\n".join(LOG) + "\n")
    return 1 if nfail else 0


def mode_analyze(window_start: str) -> int:
    rng = np.random.default_rng(SEED)
    L(f"run start (UTC): {datetime.now(timezone.utc).isoformat()}")
    L("DATA ENVIRONMENT: BINANCE-VISION-PUBLIC (production); OKX mainnet DB read-only (xval)")
    man = json.loads((BV / "factor_scale_manifest.json").read_text())
    pool = man["pool"]                       # ranked by volume, full-history over window
    coins = [p["symbol"] for p in pool]
    vols = {p["symbol"]: p["quote_volume_2026_05"] for p in pool}
    L(f"pool {len(coins)} coins; window {window_start}..{END_MONTH}; survivorship=optimistic")

    # cross-validation gate
    L("\n== cross-validation gate (Binance 1d vs OKX mainnet DB) ==")
    xval = cross_validate(rng)
    if xval.get("status") == "ok":
        for s, d in xval["per_symbol"].items():
            L(f"  {s:<10} median|dev| {d['median_abs_dev_pct']:.3f}%  max {d['max_abs_dev_pct']:.3f}%")
        L(f"  overall median |dev| {xval['overall_median_abs_dev_pct']:.3f}%  "
          f"gate(<0.5%)={'PASS' if xval['gate_pass'] else 'FAIL'}")
    else:
        L(f"  xval status: {xval.get('status')}")

    # funding availability
    have_funding = {c for c in coins if (BV / "funding" / c).exists()
                    and any((BV / "funding" / c).glob(f"{c}-fundingRate-*.zip"))}
    L(f"\nfunding available for {len(have_funding)}/{len(coins)} coins "
      f"(CAR {'ENABLED' if len(have_funding) >= 10 else 'PENDING'})")

    C, FundMA, days_all = build_panels(coins, vols, have_funding)
    scores, ret1 = factor_scores(C, FundMA)
    # window day-range (inclusive) by epoch-day (resolution-agnostic)
    ws_day = int(np.datetime64(window_start + "-01", "D").astype("int64"))
    days = [d for d in days_all if d >= ws_day]
    L(f"window epoch-days >= {ws_day} ({window_start}); {len(days)} calendar days in window")

    factors = ["F-MOM", "F-REV"] + (["F-CAR"] if len(have_funding) >= 10 else [])
    rank_by_vol = sorted(coins, key=lambda c: vols[c], reverse=True)

    # ── scale gradient: K in POOLS ──
    grad: dict = {f: {} for f in factors}
    dist_plot: dict = {}                      # K=100 null distributions for histograms
    for K in POOLS:
        cols = rank_by_vol[:K]
        L(f"\n===== POOL K={K} ({cols[0]}..{cols[-1]}) =====")
        for f in factors:
            cols_f = [c for c in cols if c in have_funding] if f == "F-CAR" else cols
            s_np, r_np, T = complete_panel(scores[f], ret1, cols_f, days)
            if T < 60 or s_np.shape[1] < 5:
                L(f"  {f}: insufficient panel (T={T}, K={s_np.shape[1]})")
                continue
            res = real_and_null(s_np, r_np, rng)
            grad[f][K] = {k: v for k, v in res.items() if not k.endswith("_dist") and k != "real_ic_series"}
            grad[f][K]["null_se"] = res["nullA"]["null_std"]
            grad[f][K]["T"] = T
            grad[f][K]["K"] = s_np.shape[1]
            if K == 100:
                dist_plot[f] = {"real_mean_ic": res["real"]["mean"],
                                "nullA": res["nullA_dist"].tolist(),
                                "nullB": res["nullB_dist"].tolist()}
            r = res["real"]; a = res["nullA"]; bb = res["nullB"]
            L(f"  {f}: IC {r['mean']:+.4f} IR {r['ir']:+.3f} t {r['t']:+.2f} (T={T},K={s_np.shape[1]}) | "
              f"NULL-A mean {a['null_mean']:+.4f} p95 {a['p95']:+.4f} -> real pctile {a['real_percentile']:.1f} "
              f"p={a['p_one_sided']:.3f} {'BEATS' if a['exceeds_p95'] else 'in-noise'} | "
              f"NULL-B p95 {bb['p95']:+.4f} p={bb['p_one_sided']:.3f}")

    # ── liquidity stratification on pool_100 ──
    L("\n===== LIQUIDITY STRATA (pool_100 by 2026-05 volume) =====")
    K100 = min(100, len(rank_by_vol))
    p100 = rank_by_vol[:K100]
    third = K100 // 3
    tiers = {"top": p100[:third], "mid": p100[third:2 * third], "bottom": p100[2 * third:]}
    strata: dict = {f: {} for f in factors}
    for f in factors:
        for tname, tcols0 in tiers.items():
            tcols = [c for c in tcols0 if c in have_funding] if f == "F-CAR" else tcols0
            s_np, r_np, T = complete_panel(scores[f], ret1, tcols, days)
            if T < 60 or s_np.shape[1] < 5:
                strata[f][tname] = {"insufficient": True, "T": T, "K": s_np.shape[1]}
                continue
            res = real_and_null(s_np, r_np, rng, n_shuffle=N_SHUFFLE)
            strata[f][tname] = {"real": res["real"], "nullA": res["nullA"],
                                "T": T, "K": len(tcols),
                                "vol_range": [vols[tcols[0]], vols[tcols[-1]]]}
            r = res["real"]; a = res["nullA"]
            L(f"  {f} {tname:<6} ({len(tcols)} coins): IC {r['mean']:+.4f} t {r['t']:+.2f} | "
              f"null p95 {a['p95']:+.4f} p={a['p_one_sided']:.3f} "
              f"{'BEATS' if a['exceeds_p95'] else 'in-noise'}")

    # ── tradeability (Part 4c): rough cost-adjusted LS spread, pool100 + per tier ──
    L("\n===== TRADEABILITY (quintile LS, daily rebal; IC != tradeable alpha) =====")
    TIER_FEE = {"pool100": 0.0006, "top": 0.0005, "mid": 0.0006, "bottom": 0.0008}
    trade: dict = {f: {} for f in factors}
    for f in factors:
        targets = {"pool100": rank_by_vol[:K100], **tiers}
        for tname, tcols0 in targets.items():
            tcols = [c for c in tcols0 if c in have_funding] if f == "F-CAR" else tcols0
            s_np, r_np, T = complete_panel(scores[f], ret1, tcols, days)
            if T < 60 or s_np.shape[1] < 10:
                continue
            sp = ls_spread(s_np, r_np, TIER_FEE[tname])
            trade[f][tname] = sp
            if tname in ("pool100", "bottom"):
                L(f"  {f} {tname:<7}: gross {sp['gross_ann_pct']:+.1f}%/yr  cost {sp['cost_ann_pct']:.1f}%/yr "
                  f"(turn {sp['avg_daily_turnover']:.2f}/d, fee {sp['fee_per_side_pct']:.3f}%) "
                  f"-> NET {sp['net_ann_pct']:+.1f}%/yr  {'POSITIVE' if sp['net_positive'] else 'NEGATIVE'}")

    # ── verdict (frozen gates) ──
    L("\n===== VERDICT (frozen pre-registered gates) =====")
    verdict = {}
    for f in factors:
        g = grad[f]
        if 100 not in g or 22 not in g:
            verdict[f] = {"gate1": False, "gate2": False, "gate3": False, "pass": False,
                          "note": "incomplete pools"}
            continue
        ic22, ic50, ic100 = g[22]["real"]["mean"], g.get(50, g[22])["real"]["mean"], g[100]["real"]["mean"]
        se100 = g[100]["null_se"]
        # gate1: beats pool-100 noise
        gate1 = g[100]["nullA"]["exceeds_p95"] and g[100]["nullA"]["p_one_sided"] < 0.05
        # gate2: not bottom-disguise
        st = strata.get(f, {})
        ic_top = st.get("top", {}).get("real", {}).get("mean", np.nan)
        ic_mid = st.get("mid", {}).get("real", {}).get("mean", np.nan)
        ic_bot = st.get("bottom", {}).get("real", {}).get("mean", np.nan)
        topmid_beats = (st.get("top", {}).get("nullA", {}).get("exceeds_p95", False)
                        or st.get("mid", {}).get("nullA", {}).get("exceeds_p95", False))
        best_tm = np.nanmax([ic_top, ic_mid]) if np.isfinite(ic_top) or np.isfinite(ic_mid) else np.nan
        gate2 = bool(np.isfinite(best_tm) and (not np.isfinite(ic_bot) or ic_bot <= 0
                     or best_tm >= 0.5 * ic_bot) and topmid_beats)
        # gate3: improves with scale
        gate3 = (ic100 - ic22 > se100) and (ic50 - ic22 > -se100)
        passed = gate1 and gate2 and gate3
        verdict[f] = {"ic22": ic22, "ic50": ic50, "ic100": ic100, "null_se_100": se100,
                      "ic_top": ic_top, "ic_mid": ic_mid, "ic_bottom": ic_bot,
                      "gate1_beats_noise": gate1, "gate2_not_bottom_disguise": gate2,
                      "gate3_scale_improves": gate3, "pass": passed}
        L(f"  {f}: IC 22->50->100 = {ic22:+.4f} -> {ic50:+.4f} -> {ic100:+.4f} (null SE {se100:.4f}) | "
          f"G1noise={int(gate1)} G2tradeable={int(gate2)} G3scale-up={int(gate3)} -> "
          f"{'PASS' if passed else 'FAIL'}")
    any_pass = any(v.get("pass") for v in verdict.values())
    final = ("PASS — large-scale factor worth the spend (flag: optimistic; needs delisted-"
             "inclusive institutional data recheck)" if any_pass else
             "FAIL — large-scale factor NOT worth it on free ~100-coin survivor data")
    L(f"\n  FINAL: {final}")

    OUT.mkdir(parents=True, exist_ok=True)
    out = {
        "positioning": "Stage-A zero-cost pre-gate: does cross-sectional alpha improve with scale; "
                       "noise-calibrated; fixed factor set; no Sharpe primary; survivorship optimistic",
        "pre_registered": {"factors": "F-MOM ret30 / F-REV ret3 / F-CAR funding7 (caliber from "
                           "research_cross_sectional_ic.py; not re-searched)",
                           "pools": list(POOLS), "n_shuffle": N_SHUFFLE, "block": BLOCK,
                           "window_start": window_start, "end_month": END_MONTH,
                           "gates": "G1 real IC > pool100 null p95 & perm p<0.05; G2 top/mid tier "
                                    "carries (not bottom-disguise); G3 IC(100)>IC(22)+nullSE & not U-down"},
        "window": [window_start, END_MONTH], "n_coins_pool": len(coins),
        "cross_validation": xval, "funding_coins": len(have_funding),
        "scale_gradient": grad, "liquidity_strata": strata, "tradeability": trade,
        "verdict": verdict, "any_pass": any_pass, "final": final,
    }
    (OUT / "results.json").write_text(json.dumps(out, indent=2, default=lambda x: None if x is None
                                                 else (float(x) if isinstance(x, (np.floating, float)) else x)))
    (OUT / "noise_distributions.json").write_text(json.dumps(
        {"pool": 100, "window": [window_start, END_MONTH], "factors": dist_plot}, indent=2))
    try:
        make_figures(grad, dist_plot, factors, window_start)
        L("wrote figures: noise_calibration_pool100.png, scale_gradient_ic.png")
    except Exception as exc:  # noqa: BLE001 — plotting must not fail the run
        L(f"figure generation skipped: {exc}")
    L(f"run end (UTC): {datetime.now(timezone.utc).isoformat()}")
    (OUT / "analyze_log.txt").write_text("\n".join(LOG) + "\n")
    L(f"wrote {OUT/'results.json'}")
    return 0


def make_figures(grad: dict, dist_plot: dict, factors: list[str], window_start: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Fig 1: noise calibration histograms (pool-100), real IC marked
    n = len(dist_plot)
    if n:
        fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), squeeze=False)
        for ax, (f, d) in zip(axes[0], dist_plot.items()):
            na = np.array(d["nullA"])
            ax.hist(na, bins=40, color="#9ecae1", edgecolor="white", label="NULL-A (block-shuffle)")
            p95 = np.percentile(na, 95)
            ax.axvline(p95, color="#d95f02", ls="--", lw=1.5, label=f"null p95 {p95:+.4f}")
            ax.axvline(d["real_mean_ic"], color="#e7298a", lw=2.5,
                       label=f"real IC {d['real_mean_ic']:+.4f}")
            ax.axvline(0, color="grey", lw=0.8)
            ax.set_title(f"{f} — pool 100"); ax.set_xlabel("mean daily IC"); ax.legend(fontsize=7)
        fig.suptitle(f"Noise calibration: real IC vs block-shuffle null (window {window_start}..{END_MONTH})")
        fig.tight_layout()
        fig.savefig(OUT / "noise_calibration_pool100.png", dpi=110)
        plt.close(fig)

    # Fig 2: scale gradient — IC point estimate vs K, with null p95 band
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = {"F-MOM": "#1b9e77", "F-REV": "#d95f02", "F-CAR": "#7570b3"}
    for f in factors:
        ks = [k for k in POOLS if k in grad.get(f, {})]
        if not ks:
            continue
        ic = [grad[f][k]["real"]["mean"] for k in ks]
        p95 = [grad[f][k]["nullA"]["p95"] for k in ks]
        ax.plot(ks, ic, "o-", color=colors.get(f, "k"), lw=2, label=f"{f} real IC")
        ax.plot(ks, p95, "x:", color=colors.get(f, "k"), alpha=0.5, label=f"{f} null p95")
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_xlabel("universe size K (top-K by volume, full-history balanced panel)")
    ax.set_ylabel("mean daily IC (point estimate)")
    ax.set_title(f"Scale gradient: does IC improve 22->50->100?  (window {window_start}..{END_MONTH})")
    ax.set_xticks(list(POOLS)); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "scale_gradient_ic.png", dpi=110)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["survey", "download", "analyze"], required=True)
    ap.add_argument("--window-start", default="2023-07")
    ap.add_argument("--n-pool", type=int, default=100)
    args = ap.parse_args()
    if args.mode == "survey":
        return mode_survey()
    if args.mode == "download":
        return mode_download(args.window_start, args.n_pool)
    return mode_analyze(args.window_start)


if __name__ == "__main__":
    raise SystemExit(main())
