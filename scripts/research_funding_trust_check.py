#!/usr/bin/env python3
"""Formal trust check of the OKX funding-rate dataset (task part 3).

(3a) Environment forensics, simplified H1/H2:
  H1 code audit  — programmatic scan of both funding downloaders for any
                   demo header / OKX_SERVER usage (expected: zero hits,
                   hardcoded public mainnet URLs only).
  H2 record audit — 10 random historical records (seed fixed, across symbols
                   and years) re-fetched from OKX TODAY and compared exactly:
                   * records within API depth (~3 months): public
                     funding-rate-history endpoint;
                   * older records: fresh re-download of the official OKX
                     historical CDN monthly zip (static.okx.com traderecords).
                   Funding history is an immutable settlement record — values
                   must match exactly.
  Plus magnitude sanity: |rate| distribution, share above 0.3%.

(3b) Post-backfill integrity: 8h-grid continuity of the combined dataset
     (legacy 2023-01-01..2026-03-31 CSVs + new 2026-04-01..2026-06-11 CSVs).

Read-only on all existing data; new evidence files go ONLY to
reports/regime/data_trust_closure_20260611/funding_refetch/.
"""

from __future__ import annotations

import io
import json
import random
import re
import time
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FUNDING_DIR = PROJECT_ROOT / "data" / "funding" / "okx"
OUT = PROJECT_ROOT / "reports/regime/data_trust_closure_20260611"
REFETCH_DIR = OUT / "funding_refetch"

INSTS = ["BTC-USDT-SWAP", "ETH-USDT-SWAP", "SOL-USDT-SWAP", "LINK-USDT-SWAP", "DOGE-USDT-SWAP"]
LEGACY_SUFFIX = "_funding_2023-01-01_2026-03-31.csv"
NEW_SUFFIX = "_funding_2026-04-01_2026-06-11.csv"
API_URL = "https://www.okx.com/api/v5/public/funding-rate-history"
CDN_URL = "https://static.okx.com/cdn/okex/traderecords/swaprates/monthly/{ym}/{inst}-fundingrates-{y}-{m:02d}.zip"
SEED = 20260611
N_SAMPLES = 10
EIGHT_H_MS = 8 * 3600 * 1000
RATE_SANITY = 0.003  # 0.3%


def fetch(url: str, tries: int = 4) -> bytes:
    last = None
    for a in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "cta-funding-trust/1.0"})
            with urllib.request.urlopen(req, timeout=30) as r:
                return r.read()
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(2 ** a)
    raise RuntimeError(f"fetch failed {url}: {last}")


def code_audit() -> dict:
    out = {}
    for script in ("download_okx_funding_history.py", "download_okx_historical_funding_files.py"):
        src = (PROJECT_ROOT / "scripts" / script).read_text()
        out[script] = {
            "simulated_header_hits": len(re.findall(r"x-simulated-trading", src, re.I)),
            "okx_server_env_hits": len(re.findall(r"OKX_SERVER", src)),
            "dotenv_hits": len(re.findall(r"dotenv|load_dotenv", src)),
            "hardcoded_public_urls": sorted(set(re.findall(r"https://[a-z.]*okx\.com[^\"' ]*", src)))[:5],
        }
    return out


def load_all() -> pd.DataFrame:
    frames = []
    for inst in INSTS:
        for suffix in (LEGACY_SUFFIX, NEW_SUFFIX):
            p = FUNDING_DIR / f"{inst}{suffix}"
            df = pd.read_csv(p, usecols=["inst_id", "funding_time", "funding_rate"])
            df["source_file"] = p.name
            frames.append(df)
    allf = pd.concat(frames, ignore_index=True)
    return allf.drop_duplicates(["inst_id", "funding_time"], keep="first").reset_index(drop=True)


def api_lookup(inst: str, ts_ms: int) -> str | None:
    """Return fundingRate string for exact fundingTime, or None if out of depth."""
    blob = json.loads(fetch(f"{API_URL}?instId={inst}&after={ts_ms + 1}&limit=1").decode())
    for rec in blob.get("data", []):
        if int(rec["fundingTime"]) == ts_ms:
            return rec["fundingRate"]
    return None


def cdn_lookup(inst: str, ts_ms: int) -> str | None:
    """Fresh re-download of the official monthly zip; exact fundingTime match."""
    dt = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
    # OKX monthly files are keyed by HK-time month of the settlement; try the
    # UTC month first, then the adjacent month (boundary records).
    for y, m in {(dt.year, dt.month),
                 ((dt.year if dt.month < 12 else dt.year + 1), (dt.month % 12) + 1)}:
        url = CDN_URL.format(ym=f"{y}{m:02d}", inst=inst, y=y, m=m)
        dest = REFETCH_DIR / url.rsplit("/", 1)[-1]
        try:
            blob = dest.read_bytes() if dest.exists() else fetch(url)
        except RuntimeError:
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            dest.write_bytes(blob)
        with zipfile.ZipFile(io.BytesIO(blob)) as z:
            csv = pd.read_csv(io.BytesIO(z.read(z.namelist()[0])))
        hit = csv[csv["funding_time"] == ts_ms]
        if len(hit):
            return str(hit.iloc[0]["funding_rate"])
    return None


def main() -> int:
    print("DATA ENVIRONMENT: OKX public endpoints (www.okx.com / static.okx.com), "
          "no credentials, no .env — MAINNET by construction", flush=True)
    REFETCH_DIR.mkdir(parents=True, exist_ok=True)
    result: dict = {"run_utc": datetime.now(timezone.utc).isoformat(), "seed": SEED}

    # H1 — code audit
    result["h1_code_audit"] = code_audit()
    h1_clean = all(v["simulated_header_hits"] == 0 and v["okx_server_env_hits"] == 0
                   for v in result["h1_code_audit"].values())
    result["h1_clean"] = h1_clean
    print(f"H1 code audit clean (no demo header / no OKX_SERVER): {h1_clean}")

    allf = load_all()
    print(f"dataset: {len(allf):,} records across {allf['inst_id'].nunique()} instruments")

    # H2 — 10-record re-fetch comparison (stratified-ish: random over all rows)
    rng = random.Random(SEED)
    idx = rng.sample(range(len(allf)), N_SAMPLES)
    samples = []
    for i in idx:
        row = allf.iloc[i]
        inst, ts_ms = row["inst_id"], int(row["funding_time"])
        local_rate = float(row["funding_rate"])
        src, ext = "api", api_lookup(inst, ts_ms)
        if ext is None:
            src, ext = "cdn_refetch", cdn_lookup(inst, ts_ms)
        match = ext is not None and abs(float(ext) - local_rate) < 1e-12
        samples.append({
            "inst_id": inst, "funding_time_ms": ts_ms,
            "funding_time_utc": datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).isoformat(),
            "local_rate": f"{local_rate:.16f}".rstrip("0"),
            "external_rate": ext, "external_source": src, "match": bool(match),
        })
        print(f"  sample {inst} {samples[-1]['funding_time_utc']} -> {src}: "
              f"{'MATCH' if match else 'MISMATCH/NOT-FOUND'}")
    result["h2_samples"] = samples
    result["h2_all_match"] = all(s["match"] for s in samples)

    # magnitude sanity
    rates = allf["funding_rate"].astype(float)
    by_sym = {}
    for inst, g in allf.groupby("inst_id"):
        r = g["funding_rate"].astype(float).abs()
        by_sym[inst] = {"n": int(len(g)), "max_abs_rate": float(r.max()),
                        "share_gt_0_3pct": float((r > RATE_SANITY).mean() * 100)}
    result["magnitude_sanity"] = {
        "per_symbol": by_sym,
        "overall_max_abs_rate": float(rates.abs().max()),
        "overall_share_gt_0_3pct_pct": float((rates.abs() > RATE_SANITY).mean() * 100),
    }

    # (3b) 8h-grid continuity of combined dataset.
    # Criterion: a continuity GAP is an interval deviating from 8h by more
    # than 60s (legacy CDN records carry second-level settlement-timestamp
    # jitter, e.g. 16:00:11 — that is jitter, not a missing settlement);
    # additionally the record count must equal the theoretical grid count.
    GAP_TOL_MS = 60_000
    grid = {}
    for inst, g in allf.groupby("inst_id"):
        ts = g["funding_time"].astype("int64").sort_values().to_numpy()
        diffs = (ts[1:] - ts[:-1])
        gaps = [{"after_utc": datetime.fromtimestamp(int(a) / 1000, tz=timezone.utc).isoformat(),
                 "gap_hours": float(d / 3600000)}
                for a, d in zip(ts[:-1], diffs) if abs(int(d) - EIGHT_H_MS) > GAP_TOL_MS]
        expected_n = int(round((ts[-1] - ts[0]) / EIGHT_H_MS)) + 1
        grid[inst] = {
            "n": int(len(ts)), "expected_n_from_span": expected_n,
            "count_complete": bool(len(ts) == expected_n),
            "first_utc": datetime.fromtimestamp(int(ts[0]) / 1000, tz=timezone.utc).isoformat(),
            "last_utc": datetime.fromtimestamp(int(ts[-1]) / 1000, tz=timezone.utc).isoformat(),
            "gaps_gt_60s_off_8h": gaps,
            "timestamp_jitter_max_seconds": float(np.abs(diffs - EIGHT_H_MS).max() / 1000),
            "n_jittered_intervals": int(np.sum((diffs != EIGHT_H_MS)
                                               & (np.abs(diffs - EIGHT_H_MS) <= GAP_TOL_MS))),
            "duplicates": int(g.duplicated("funding_time").sum()),
        }
    result["grid_continuity"] = grid
    grid_ok = all(not v["gaps_gt_60s_off_8h"] and v["duplicates"] == 0 and v["count_complete"]
                  for v in grid.values())
    result["grid_ok"] = grid_ok

    # backfill manifest per iron rules
    result["backfill_manifest"] = {
        "source": "okx-public-rest", "server": "MAINNET",
        "endpoint": API_URL,
        "script": "scripts/download_okx_funding_history.py (existing, audited above)",
        "range": ["2026-04-01", "2026-06-11"],
        "files": [f"data/funding/okx/{i}{NEW_SUFFIX}" for i in INSTS],
    }

    verdict = "通过" if (h1_clean and result["h2_all_match"] and grid_ok) else "异常"
    result["verdict"] = verdict
    (OUT / "funding_verification.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"\nH2 all match: {result['h2_all_match']} | grid ok: {grid_ok} | "
          f"max |rate| {result['magnitude_sanity']['overall_max_abs_rate']:.4%} | "
          f"FUNDING VERDICT: {verdict}")
    return 0 if verdict == "通过" else 2


if __name__ == "__main__":
    raise SystemExit(main())
