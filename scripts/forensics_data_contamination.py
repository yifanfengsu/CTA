#!/usr/bin/env python3
"""Read-only forensics: trace sampled synthetic-ramp events back to download
manifests to test whether download_okx_history.py (gateway=DEMO path) is the
contamination source.

Strictly read-only: opens database.db with mode=ro, never writes outside
reports/regime/data_contamination_forensics_<date>/.
"""

from __future__ import annotations

import glob
import json
import sqlite3
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / ".vntrader" / "database.db"
MANIFEST_DIR = PROJECT_ROOT / "data" / "history_manifests"
DIAG_JSON = (
    PROJECT_ROOT
    / "reports/regime/v2b_dd_diagnosis_20260610/pattern_replication_check.json"
)
OUT_DIR = PROJECT_ROOT / "reports/regime/data_contamination_forensics_20260610"

SYMBOLS = ["BTC", "ETH", "SOL", "LINK", "DOGE"]
DB_SYMBOL = {s: f"{s}USDT_SWAP_OKX" for s in SYMBOLS}

# Representative samples (symbol, event start UTC) chosen from the 598 events:
# 2 SOL (worst-hit), 1 DOGE, 1 ETH; BTC has zero events (reverse evidence).
SAMPLE_KEYS = [
    ("SOL", "2023-01-02 05:50:00+00:00"),
    ("SOL", "2025-05-29"),  # the known v2B DD day; pick first SOL event that day
    ("DOGE", "2024"),  # pick largest-excursion DOGE event in 2024
    ("ETH", "2026"),  # pick largest-excursion ETH event in 2026
]


def load_events() -> list[dict]:
    return json.load(DIAG_JSON.open())["events"]


def pick_samples(events: list[dict]) -> list[dict]:
    picked: list[dict] = []
    for sym, key in SAMPLE_KEYS:
        cands = [e for e in events if e["symbol"] == sym and str(e["start"]).startswith(key)]
        if not cands and len(key) <= 4:
            cands = [e for e in events if e["symbol"] == sym and str(e["start"])[:4] == key]
        if not cands:
            picked.append({"symbol": sym, "wanted": key, "found": False})
            continue
        best = max(cands, key=lambda e: abs(e.get("max_excursion_pct", 0)))
        if key.count(":") > 0:  # exact-start key: keep the exact event
            best = cands[0]
        best = dict(best)
        best["found"] = True
        picked.append(best)
    return picked


def fetch_window(cur: sqlite3.Cursor, symbol: str, start: datetime, end: datetime) -> list[tuple]:
    return cur.execute(
        "SELECT datetime, open_price, high_price, low_price, close_price, volume "
        "FROM dbbardata WHERE symbol=? AND interval='1m' AND datetime>=? AND datetime<? "
        "ORDER BY datetime",
        (DB_SYMBOL[symbol], start.strftime("%Y-%m-%d %H:%M:%S"), end.strftime("%Y-%m-%d %H:%M:%S")),
    ).fetchall()


def manifest_chunks_covering(window_start: datetime, window_end: datetime, symbol: str) -> list[dict]:
    """Find every manifest chunk whose [start_utc, end_exclusive_utc) covers the window."""
    hits: list[dict] = []
    for path in sorted(glob.glob(str(MANIFEST_DIR / f"{symbol}USDT*.json"))):
        payload = json.loads(Path(path).read_text())
        for chunk in payload.get("chunks", []):
            try:
                cs = datetime.fromisoformat(chunk["start_utc"])
                ce = datetime.fromisoformat(chunk["end_exclusive_utc"])
            except (KeyError, ValueError):
                continue
            if cs <= window_start < ce or cs < window_end <= ce:
                hits.append(
                    {
                        "manifest": Path(path).name,
                        "chunk_index": chunk.get("index"),
                        "chunk_start_utc": chunk["start_utc"],
                        "chunk_end_exclusive_utc": chunk["end_exclusive_utc"],
                        "status": chunk.get("status"),
                        "source_used": chunk.get("source_used"),
                        "bar_count": chunk.get("bar_count"),
                        "saved_at": chunk.get("saved_at"),
                        "verified_at": chunk.get("verified_at"),
                        "last_error": chunk.get("last_error"),
                    }
                )
    return hits


def summarize_ramp(rows: list[tuple]) -> dict:
    if not rows:
        return {"bars": 0}
    closes = [r[4] for r in rows]
    diffs = [b - a for a, b in zip(closes, closes[1:])]
    rng = (max(closes) - min(closes)) / min(closes) * 100 if min(closes) else 0.0
    return {
        "bars": len(rows),
        "first": {"dt": rows[0][0], "close": rows[0][4]},
        "last": {"dt": rows[-1][0], "close": rows[-1][4]},
        "min_close": min(closes),
        "max_close": max(closes),
        "range_pct": round(rng, 2),
        "step_diffs": [round(d, 6) for d in diffs[:20]],
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sample_dir = OUT_DIR / "sample_traceback"
    sample_dir.mkdir(exist_ok=True)

    events = load_events()
    dist = Counter(e["symbol"] for e in events)

    con = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    cur = con.cursor()

    samples = pick_samples(events)
    results = []
    for i, ev in enumerate(samples, 1):
        if not ev.get("found"):
            results.append(ev)
            continue
        start = datetime.fromisoformat(str(ev["start"])).astimezone(timezone.utc).replace(tzinfo=None)
        end = datetime.fromisoformat(str(ev["end"])).astimezone(timezone.utc).replace(tzinfo=None)
        pad_start = start - timedelta(minutes=10)
        pad_end = end + timedelta(minutes=10)

        per_symbol = {}
        for sym in SYMBOLS:
            rows = fetch_window(cur, sym, pad_start, pad_end)
            per_symbol[sym] = summarize_ramp(rows)
            if sym == ev["symbol"]:
                per_symbol[sym]["raw_bars"] = [
                    {"dt": r[0], "o": r[1], "h": r[2], "l": r[3], "c": r[4], "v": r[5]}
                    for r in rows
                ]

        ws = start.replace(tzinfo=timezone.utc)
        we = end.replace(tzinfo=timezone.utc)
        chunks = manifest_chunks_covering(ws, we, ev["symbol"])

        record = {
            "sample_id": i,
            "event": ev,
            "window_utc": [str(ws), str(we)],
            "db_check": per_symbol,
            "manifest_chunks": chunks,
        }
        results.append(record)
        (sample_dir / f"sample_{i}_{ev['symbol']}_{start:%Y%m%d}.json").write_text(
            json.dumps(record, indent=2, ensure_ascii=False, default=str)
        )

    # BTC reverse evidence: 0 events
    btc_note = {
        "symbol": "BTC",
        "events_in_diagnosis": dist.get("BTC", 0),
        "note": "BTC has zero ramp events; cannot sample. Reverse evidence recorded in README.",
    }
    (sample_dir / "sample_0_BTC_none.json").write_text(json.dumps(btc_note, indent=2))

    # Manifest audit (step 3)
    audit = {"per_manifest": [], "source_used_totals": Counter(), "per_symbol_source": {}}
    for path in sorted(glob.glob(str(MANIFEST_DIR / "*.json"))):
        payload = json.loads(Path(path).read_text())
        sym = payload.get("vt_symbol", "?").split("USDT")[0]
        cnt = Counter((c.get("source_used") or "none") for c in payload.get("chunks", []))
        audit["per_manifest"].append(
            {
                "manifest": Path(path).name,
                "vt_symbol": payload.get("vt_symbol"),
                "range": [payload.get("start"), payload.get("end_display")],
                "created_at": payload.get("created_at"),
                "updated_at": payload.get("updated_at"),
                "source_arg": payload.get("source"),
                "chunks_by_source_used": dict(cnt),
            }
        )
        audit["source_used_totals"].update(cnt)
        audit["per_symbol_source"].setdefault(sym, Counter()).update(cnt)
    audit["source_used_totals"] = dict(audit["source_used_totals"])
    audit["per_symbol_source"] = {k: dict(v) for k, v in audit["per_symbol_source"].items()}
    audit["fields_present_in_chunk"] = sorted(
        json.loads(Path(sorted(glob.glob(str(MANIFEST_DIR / "*.json")))[0]).read_text())["chunks"][0].keys()
    )
    audit["server_recorded_in_manifest"] = False  # no DEMO/REAL field anywhere in schema
    (OUT_DIR / "manifest_audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False))

    print(json.dumps({"samples": results, "event_distribution": dict(dist)}, indent=2, default=str)[:6000])
    print("\nWrote:", OUT_DIR)


if __name__ == "__main__":
    main()
