#!/usr/bin/env python3
"""Build the cross-sectional DOWNLOAD LIST by mechanical rules (no research judgment).

This is a data-acquisition helper, NOT analysis: it produces the list of symbols to
download and a watchlist of excluded symbols. It defines NO cross-sectional universe,
handles NO survivorship bias, computes NO returns/IC/ranking-of-returns.

MECHANICAL RULES (pre-registered, no subjective add/remove):
  universe   = all Binance UM-perp symbols on data.binance.vision ending in 'USDT'
               (S3 bucket listing of .../futures/um/monthly/klines/).
  rank metric= total USDT quote volume in 2026-05 (latest complete month in range),
               read from the monthly 1d kline file. This is the "recent ~30d volume"
               proxy. NOTE: fapi.binance.com ticker/24hr is geo-blocked here (HTTP 451),
               so the vision daily-stats path (allowed by spec) is used instead.
               A symbol with no 2026-05 1d file (delisted/never-listed) ranks volume 0
               and falls out — this also enforces "current mainstream".
  select     = top 40 by that volume.
  age rule   = within the top 40, exclude any whose EARLIEST available 1m month on
               vision is later than 2024-11 (i.e. < 18 months before 2026-05). Excluded
               ones go to a watchlist (recorded, NOT downloaded this run).
  reuse      = existing 5 coins (BTC/ETH/SOL/LINK/DOGE) are reused if present; the
               downloader is resume-safe and skips already-verified files.

Output: data/binance_vision/cross_sectional_download_list.json (intermediate;
consumed by the download step and folded into cross_sectional_manifest.json).
"""

from __future__ import annotations

import io
import json
import sys
import time
import urllib.request
import urllib.error
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT = PROJECT_ROOT / "data" / "binance_vision"
S3 = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
CDN = "https://data.binance.vision"
KPREFIX = "data/futures/um/monthly/klines/"
EXISTING = {"BTCUSDT", "ETHUSDT", "SOLUSDT", "LINKUSDT", "DOGEUSDT"}

TOP_N = 40
RANK_MONTH = "2026-05"
AGE_CUTOFF_MONTH = "2024-11"   # earliest 1m month must be <= this (18mo before 2026-05)
WORKERS = 12
TIMEOUT = 60
RETRIES = 4


def fetch(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "cta-cross-sectional-list/1.0"})
    last = None
    for attempt in range(1, RETRIES + 1):
        try:
            with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
                return r.read()
        except urllib.error.HTTPError:
            raise                              # caller handles 404 etc.
        except Exception as exc:               # noqa: BLE001 — retry transient (IncompleteRead, timeout)
            last = exc
            time.sleep(min(2 ** attempt, 10))
    raise last


def list_symbols() -> list[str]:
    """All UM-perp symbol dirs via paginated S3 CommonPrefixes; filter *USDT."""
    syms, marker = [], ""
    while True:
        url = f"{S3}?delimiter=/&prefix={KPREFIX}"
        if marker:
            url += f"&marker={urllib.parse.quote(marker)}"
        xml = fetch(url).decode()
        prefixes = [p[len("<Prefix>"):-len("</Prefix>")]
                    for p in _findall(xml, "Prefix") if p.startswith("<Prefix>" + KPREFIX)]
        for p in prefixes:
            name = p[len(KPREFIX):].rstrip("/")
            if name:
                syms.append(name)
        if "<IsTruncated>true</IsTruncated>" not in xml:
            break
        nm = _findall(xml, "NextMarker")
        marker = nm[0][len("<NextMarker>"):-len("</NextMarker>")] if nm else (prefixes[-1] if prefixes else "")
        if not marker:
            break
    usdt = sorted(s for s in syms if s.endswith("USDT"))
    return usdt


import urllib.parse  # noqa: E402


def _findall(xml: str, tag: str) -> list[str]:
    out, i, op, cl = [], 0, f"<{tag}>", f"</{tag}>"
    while True:
        a = xml.find(op, i)
        if a < 0:
            break
        b = xml.find(cl, a)
        out.append(xml[a:b + len(cl)])
        i = b + len(cl)
    return out


def quote_volume(symbol: str, month: str) -> float | None:
    """Sum USDT quote volume from the monthly 1d kline zip; None if no file (404)."""
    url = f"{CDN}/{KPREFIX}{symbol}/1d/{symbol}-1d-{month}.zip"
    for attempt in range(1, RETRIES + 1):
        try:
            blob = fetch(url)
            with zipfile.ZipFile(io.BytesIO(blob)) as z:
                raw = z.read(z.namelist()[0]).decode().strip().splitlines()
            total = 0.0
            for line in raw:
                f = line.split(",")
                try:
                    total += float(f[7])      # col 7 = quote_asset_volume
                except (ValueError, IndexError):
                    continue                  # header row or malformed -> skip
            return total
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            time.sleep(min(2 ** attempt, 10))
        except Exception:                     # noqa: BLE001
            time.sleep(min(2 ** attempt, 10))
    return None


def earliest_1m_month(symbol: str) -> str | None:
    """First available YYYY-MM in the symbol's monthly 1m dir (S3 listing)."""
    url = f"{S3}?prefix={KPREFIX}{symbol}/1m/{symbol}-1m-"
    try:
        xml = fetch(url).decode()
    except Exception:                         # noqa: BLE001
        return None
    keys = [k[len("<Key>"):-len("</Key>")] for k in _findall(xml, "Key")]
    months = sorted(k.split("-1m-")[1].replace(".zip", "").replace(".CHECKSUM", "")
                    for k in keys if "-1m-" in k and k.endswith(".zip"))
    return months[0] if months else None


def main() -> int:
    t0 = datetime.now(timezone.utc)
    print(f"run start (UTC): {t0.isoformat()}", flush=True)
    print("DATA ENVIRONMENT: BINANCE-VISION-PUBLIC (production; fapi geo-blocked 451 "
          "-> ranking via vision 1d quote-volume, spec-allowed)", flush=True)

    syms = list_symbols()
    print(f"universe: {len(syms)} *USDT UM-perp symbols listed on vision", flush=True)

    vols: dict[str, float | None] = {}
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(quote_volume, s, RANK_MONTH): s for s in syms}
        for n, fut in enumerate(as_completed(futs), 1):
            s = futs[fut]
            vols[s] = fut.result()
            if n % 100 == 0:
                print(f"  volume probed {n}/{len(syms)}", flush=True)

    ranked = sorted(((s, v) for s, v in vols.items() if v),
                    key=lambda kv: kv[1], reverse=True)
    top = ranked[:TOP_N]
    print(f"\ntop {TOP_N} by {RANK_MONTH} quote volume (with {len([v for v in vols.values() if v])} "
          f"symbols having a {RANK_MONTH} 1d file):", flush=True)

    entries, download, watchlist = [], [], []
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        earliest = dict(zip([s for s, _ in top],
                            ex.map(earliest_1m_month, [s for s, _ in top])))
    for rank, (s, v) in enumerate(top, 1):
        em = earliest[s]
        age_ok = em is not None and em <= AGE_CUTOFF_MONTH
        rec = {"rank": rank, "symbol": s, "quote_volume_2026_05_usdt": v,
               "earliest_1m_month": em,
               "age_ge_18mo": bool(age_ok),
               "already_have": s in EXISTING}
        entries.append(rec)
        if age_ok:
            download.append(s)
        else:
            watchlist.append(rec)
        flag = "HAVE" if s in EXISTING else ("DL" if age_ok else "WATCH<18mo")
        print(f"  #{rank:>2} {s:<16} vol={v:,.0f}  first={em}  [{flag}]", flush=True)

    out = {
        "positioning": "mechanical download list; no universe definition, no survivorship "
                       "handling, no analysis — raw-material acquisition only",
        "rules": {"universe": "vision UM-perp *USDT", "rank_metric": f"{RANK_MONTH} 1d quote volume (USDT)",
                  "rank_metric_note": "fapi ticker/24hr geo-blocked (HTTP 451); vision daily-stats path used",
                  "top_n": TOP_N, "age_rule": f"earliest 1m month <= {AGE_CUTOFF_MONTH} (>=18mo before 2026-05)"},
        "generated_at": t0.isoformat(),
        "n_universe_usdt": len(syms),
        "n_with_rank_month_file": len([v for v in vols.values() if v]),
        "top_n_entries": entries,
        "download_symbols": download,
        "watchlist_excluded_lt_18mo": watchlist,
        "existing_reused": sorted(EXISTING & set(download)),
        "new_to_download": sorted(set(download) - EXISTING),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "cross_sectional_download_list.json").write_text(json.dumps(out, indent=2))
    print(f"\ndownload list: {len(download)} symbols ({len(out['new_to_download'])} new, "
          f"{len(out['existing_reused'])} reused) | watchlist {len(watchlist)} | "
          f"-> {OUT / 'cross_sectional_download_list.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
