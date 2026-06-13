#!/usr/bin/env python3
"""Integrity catalog for the cross-sectional raw-material download (descriptive only).

NOT analysis: counts bars, lists gaps, records checksum status. Defines no universe,
estimates no returns, makes no go/no-go judgment. Gaps are recorded as-is, NOT filled
(consistent with existing Binance-vision handling).

Reads:  data/binance_vision/cross_sectional_download_list.json (the mechanical list)
        data/binance_vision/<SYM>/*.zip (1m klines), funding/<SYM>/*.zip
        data/binance_vision/manifest_*_xsec.json (checksum status for new coins)
Writes: data/binance_vision/cross_sectional_manifest.json   (in git)
        data/binance_vision/cross_sectional_README.md        (in git)
        data/binance_vision/failed_downloads.txt             (in git; or 'none')
"""

from __future__ import annotations

import io
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BV = PROJECT_ROOT / "data" / "binance_vision"
EXISTING = {"BTCUSDT", "ETHUSDT", "SOLUSDT", "LINKUSDT", "DOGEUSDT"}


def scan_klines(symbol: str) -> dict:
    """Per-month bar counts + minute-grid continuity (gaps recorded, not filled)."""
    zips = sorted((BV / symbol).glob(f"{symbol}-1m-*.zip"))
    per_month, all_min = {}, []
    total_bytes = 0
    for zp in zips:
        month = zp.stem.split("-1m-")[1]
        total_bytes += zp.stat().st_size
        with zipfile.ZipFile(zp) as z:
            raw = z.read(z.namelist()[0])
        df = pd.read_csv(io.BytesIO(raw), header=None, usecols=[0], names=["ot"])
        if isinstance(df.iloc[0, 0], str):
            df = df.iloc[1:].reset_index(drop=True)
        ot = pd.to_numeric(df["ot"]).astype("int64")
        unit_us = ot.iloc[0] > 100_000_000_000_000
        mu = (ot // (60_000_000 if unit_us else 60_000)).astype("int64").to_numpy()
        per_month[month] = int(len(mu))
        all_min.append(mu)
    if not all_min:
        return {"n_months": 0}
    mins = np.unique(np.concatenate(all_min))
    diffs = np.diff(mins)
    gap_idx = np.where(diffs > 1)[0]
    missing = int((diffs[gap_idx] - 1).sum()) if len(gap_idx) else 0
    top_gaps = sorted(
        ({"after_min_utc": int(mins[i]),
          "gap_minutes": int(diffs[i] - 1),
          "after_utc": pd.Timestamp(int(mins[i]) * 60, unit="s", tz="UTC").isoformat()}
         for i in gap_idx), key=lambda g: g["gap_minutes"], reverse=True)[:10]
    return {
        "n_months": len(per_month),
        "earliest_month": min(per_month), "latest_month": max(per_month),
        "total_bars": int(len(mins)),
        "expected_bars_first_to_last": int(mins[-1] - mins[0] + 1),
        "missing_minutes": missing,
        "n_gap_events": int(len(gap_idx)),
        "continuity_pct": round(100.0 * len(mins) / (mins[-1] - mins[0] + 1), 4),
        "top_gaps": top_gaps,
        "klines_bytes": total_bytes,
        "per_month_bar_counts": per_month,
    }


def count_funding(symbol: str) -> dict:
    zips = sorted((BV / "funding" / symbol).glob(f"{symbol}-fundingRate-*.zip"))
    rows, total_bytes = 0, 0
    for zp in zips:
        total_bytes += zp.stat().st_size
        with zipfile.ZipFile(zp) as z:
            raw = z.read(z.namelist()[0]).decode().strip().splitlines()
        rows += sum(1 for ln in raw if ln and not ln[0].isalpha())  # skip header
    return {"n_funding_files": len(zips), "n_settlements": rows,
            "funding_bytes": total_bytes}


def load_manifest_status() -> tuple[dict, list]:
    """Per-symbol checksum status from the _xsec manifests; collect FAILED list."""
    status, failed = {}, []
    for mf in sorted(BV.glob("manifest_*_xsec.json")):
        m = json.loads(mf.read_text())
        for r in m["files"]:
            s = r["symbol"]
            d = status.setdefault(s, {"verified": 0, "not_available_404": 0, "failed": 0})
            st = r["status"]
            if st in ("downloaded_verified", "cached_verified"):
                d["verified"] += 1
            elif st == "not_available_404":
                d["not_available_404"] += 1
            elif st.startswith("FAILED"):
                d["failed"] += 1
                failed.append({"symbol": s, "month": r["month"],
                               "file": r["file"], "status": st})
    return status, failed


def main() -> int:
    print(f"catalog run (UTC): {datetime.now(timezone.utc).isoformat()}", flush=True)
    dl = json.loads((BV / "cross_sectional_download_list.json").read_text())
    rank_info = {e["symbol"]: e for e in dl["top_n_entries"]}
    symbols = sorted(set(dl["download_symbols"]))
    status, failed = load_manifest_status()

    coins, tot_klines_bytes, tot_funding_bytes = {}, 0, 0
    for s in symbols:
        k = scan_klines(s)
        f = count_funding(s)
        st = status.get(s, {})
        chk = ("reused-prior-dualcycle-verified" if s in EXISTING
               else ("all_pass" if st.get("failed", 0) == 0 and st.get("verified", 0) > 0
                     else "HAS_FAILURES" if st.get("failed", 0) else "unknown"))
        ri = rank_info.get(s, {})
        coins[s] = {
            "rank": ri.get("rank"), "quote_volume_2026_05_usdt": ri.get("quote_volume_2026_05_usdt"),
            "earliest_1m_month_listed": ri.get("earliest_1m_month"),
            "already_have_reused": s in EXISTING,
            "checksum_status": chk,
            "source": "binance_vision", "server": "N/A (public static)",
            **k, **f,
        }
        tot_klines_bytes += k.get("klines_bytes", 0)
        tot_funding_bytes += f.get("funding_bytes", 0)
        print(f"  {s:<14} months={k.get('n_months',0):>3} bars={k.get('total_bars',0):>9,} "
              f"gaps={k.get('n_gap_events',0):>4} miss={k.get('missing_minutes',0):>6} "
              f"cont={k.get('continuity_pct','-')}% fund={f['n_settlements']:>5} [{chk}]", flush=True)

    manifest = {
        "positioning": "cross-sectional RAW MATERIAL catalog — download integrity only; "
                       "NOT a research universe; survivorship NOT handled (see README); "
                       "no returns/IC/analysis; gaps recorded not filled",
        "source": "binance-vision-public-static", "server": "BINANCE-PRODUCTION",
        "rank_metric": dl["rules"]["rank_metric"],
        "rank_metric_note": dl["rules"]["rank_metric_note"],
        "selection_rules": dl["rules"],
        "cataloged_at": datetime.now(timezone.utc).isoformat(),
        "n_symbols": len(symbols),
        "n_new_downloaded": len(set(symbols) - EXISTING),
        "n_existing_reused": len(set(symbols) & EXISTING),
        "n_watchlist_excluded_lt_18mo": len(dl["watchlist_excluded_lt_18mo"]),
        "watchlist_excluded_lt_18mo": dl["watchlist_excluded_lt_18mo"],
        "total_klines_mb": round(tot_klines_bytes / 1048576, 1),
        "total_funding_mb": round(tot_funding_bytes / 1048576, 2),
        "n_failed_downloads": len(failed),
        "symbols": coins,
    }
    (BV / "cross_sectional_manifest.json").write_text(json.dumps(manifest, indent=2))
    ftxt = "none\n" if not failed else "\n".join(
        f"{x['symbol']} {x['month']} {x['status']}" for x in failed) + "\n"
    (BV / "failed_downloads.txt").write_text(ftxt)
    print(f"\ncataloged {len(symbols)} symbols, {manifest['total_klines_mb']} MB klines + "
          f"{manifest['total_funding_mb']} MB funding, {len(failed)} failed downloads", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
