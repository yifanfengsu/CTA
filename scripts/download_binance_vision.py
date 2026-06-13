#!/usr/bin/env python3
"""Download Binance vision UM-perp 1m monthly kline zips (external cross-validation source).

Data environment: data.binance.vision public static files — Binance production
(mainnet) market data by construction; there is no demo/testnet variant of this
CDN. No credentials, no .env, no OKX involvement.

- Writes ONLY to data/binance_vision/<SYMBOL>/ (new directory, never touches
  .vntrader/ or data/funding/).
- Every zip is verified against its .CHECKSUM (sha256) file; mismatch -> retry.
- Resume-safe: existing files that pass sha256 are skipped.
- Manifest with explicit source/server fields per CLAUDE.md data iron rules.

2026-06-11 range extension (dual-cycle study): optional CLI args
  --dataset {klines,fundingRate}  --start-month YYYY-MM  --end-month YYYY-MM
extend the month range / dataset; download+verify logic unchanged. HTTP 404 is
recorded as not_available_404 without retries (expected for months before a
perp contract's listing). Non-default invocations write a range-specific
manifest (manifest_<dataset>_<start>_<end>.json); the original manifest.json
of the 2023-2026 klines download is never overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = PROJECT_ROOT / "data" / "binance_vision"
BASE_ROOT = "https://data.binance.vision/data/futures/um/monthly"
BASE = f"{BASE_ROOT}/klines"

SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "LINKUSDT", "DOGEUSDT"]
MONTHS = [f"{y}-{m:02d}" for y in (2023, 2024, 2025) for m in range(1, 13)] + \
         [f"2026-{m:02d}" for m in range(1, 6)]  # 2023-01 .. 2026-05 = 41 months


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

MAX_RETRIES = 4
TIMEOUT = 60
WORKERS = 6


def fetch(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "cta-data-trust-closure/1.0"})
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
        return r.read()


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def one_file(symbol: str, month: str, dataset: str = "klines") -> dict:
    if dataset == "klines":
        name = f"{symbol}-1m-{month}.zip"
        url = f"{BASE_ROOT}/klines/{symbol}/1m/{name}"
        dest = OUT_ROOT / symbol / name
    else:
        name = f"{symbol}-fundingRate-{month}.zip"
        url = f"{BASE_ROOT}/fundingRate/{symbol}/{name}"
        dest = OUT_ROOT / "funding" / symbol / name
    dest.parent.mkdir(parents=True, exist_ok=True)
    rec = {"symbol": symbol, "month": month, "file": str(dest.relative_to(PROJECT_ROOT)),
           "url": url, "status": "", "sha256": "", "bytes": 0}
    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            want = fetch(url + ".CHECKSUM").decode().split()[0].strip()
            if dest.exists():
                got = sha256(dest.read_bytes())
                if got == want:
                    rec.update(status="cached_verified", sha256=got, bytes=dest.stat().st_size)
                    return rec
            blob = fetch(url)
            got = sha256(blob)
            if got != want:
                raise ValueError(f"sha256 mismatch want={want} got={got}")
            dest.write_bytes(blob)
            rec.update(status="downloaded_verified", sha256=got, bytes=len(blob))
            return rec
        except urllib.error.HTTPError as exc:
            if exc.code == 404:  # pre-listing month: expected, no retries
                rec.update(status="not_available_404")
                return rec
            last_err = exc
            time.sleep(min(2 ** attempt, 15))
        except Exception as exc:  # noqa: BLE001 — retry any transient failure
            last_err = exc
            time.sleep(min(2 ** attempt, 15))
    rec.update(status=f"FAILED: {last_err}")
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["klines", "fundingRate"], default="klines")
    ap.add_argument("--start-month")
    ap.add_argument("--end-month")
    ap.add_argument("--symbols", help="comma-separated symbol override (e.g. "
                    "XRPUSDT,BNBUSDT). Default = the original 5 cross-validation coins. "
                    "Download/verify/manifest logic is identical regardless of list.")
    args = ap.parse_args()
    default_run = (args.dataset == "klines" and not args.start_month
                   and not args.end_month and not args.symbols)
    months = MONTHS if (args.dataset == "klines" and not args.start_month
                        and not args.end_month) else month_range(args.start_month, args.end_month)
    symbols = ([s.strip() for s in args.symbols.split(",") if s.strip()]
               if args.symbols else SYMBOLS)

    print("DATA ENVIRONMENT: BINANCE-VISION-PUBLIC (production market data; "
          "no demo variant exists for this CDN)", flush=True)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    jobs = [(s, m) for s in symbols for m in months]
    print(f"{len(jobs)} files ({len(symbols)} symbols x {len(months)} months, "
          f"dataset={args.dataset}) -> {OUT_ROOT}")
    results, failed = [], []
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(one_file, s, m, args.dataset): (s, m) for s, m in jobs}
        for n, fut in enumerate(as_completed(futs), 1):
            rec = fut.result()
            results.append(rec)
            if rec["status"].startswith("FAILED"):
                failed.append(rec)
            if n % 25 == 0 or rec["status"].startswith("FAILED"):
                print(f"[{n}/{len(jobs)}] {rec['symbol']} {rec['month']}: {rec['status']}", flush=True)
    results.sort(key=lambda r: (r["symbol"], r["month"]))
    manifest = {
        "source": "binance-vision-public-static",
        "server": "BINANCE-PRODUCTION",  # explicit environment field per iron rules
        "dataset": ("futures/um monthly klines 1m (last-price klines)"
                    if args.dataset == "klines" else "futures/um monthly fundingRate"),
        "base_url": f"{BASE_ROOT}/{args.dataset}",
        "symbols": symbols,
        "months": [months[0], months[-1]],
        "checksum": "sha256 via vision .CHECKSUM files, all verified",
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
        "script": "scripts/download_binance_vision.py 1.2.0 (2026-06-13 --symbols arg)",
        "n_files": len(results),
        "n_failed": len(failed),
        "n_not_available_404": sum(1 for r in results if r["status"] == "not_available_404"),
        "files": results,
    }
    # custom symbol lists get a distinct manifest name so the original 5-coin
    # cross-validation manifests are never clobbered.
    if default_run:
        mf = "manifest.json"
    elif args.symbols:
        mf = f"manifest_{args.dataset}_{months[0]}_{months[-1]}_xsec.json"
    else:
        mf = f"manifest_{args.dataset}_{months[0]}_{months[-1]}.json"
    (OUT_ROOT / mf).write_text(json.dumps(manifest, indent=2))
    total_mb = sum(r["bytes"] for r in results) / 1048576
    print(f"\ndone: {len(results) - len(failed)}/{len(jobs)} ok, {len(failed)} failed, {total_mb:.0f} MB")
    for r in failed:
        print("  FAILED:", r["symbol"], r["month"], r["status"])
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
