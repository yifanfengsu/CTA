#!/usr/bin/env python3
"""Rebuild 1m kline history from OKX MAINNET public REST into an isolated DB.

Hard rules baked in:
- MAINNET ONLY: plain public endpoint, NO x-simulated-trading header, never
  reads OKX_SERVER or any .env value (no credentials needed at all).
- Writes ONLY to .vntrader/database_mainnet.db — never touches database.db.
- Resume-safe: per-chunk manifest with explicit "server": "MAINNET" field.
- No gap filling / interpolation: missing bars are recorded, never fabricated.

Conventions match the legacy DB exactly so comparison tooling can reuse loaders:
symbol e.g. BTCUSDT_SWAP_OKX, exchange GLOBAL, interval '1m', datetime stored
as naive Asia/Shanghai local time of the bar OPEN, volume=row[5] (contracts),
turnover=row[6] (volCcy), open_interest=0.
"""

from __future__ import annotations

import json
import sqlite3
import sys
import time
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / ".vntrader" / "database_mainnet.db"
LEGACY_DB_PATH = PROJECT_ROOT / ".vntrader" / "database.db"  # never opened here
MANIFEST_DIR = PROJECT_ROOT / "reports/regime/mainnet_rebuild_20260610/download_manifests"
SUMMARY_PATH = PROJECT_ROOT / "reports/regime/mainnet_rebuild_20260610/integrity_summary.json"

OKX_URL = "https://www.okx.com/api/v5/market/history-candles"
SCRIPT_VERSION = "1.0.0 (2026-06-10)"

# Target range: identical to legacy DB coverage.
SHANGHAI = timezone(timedelta(hours=8))
RANGE_START = datetime(2023, 1, 1, 0, 0, tzinfo=SHANGHAI)
RANGE_END_EXCL = datetime(2026, 5, 29, 0, 0, tzinfo=SHANGHAI)  # last bar 2026-05-28 23:59

SYMBOLS = [  # download order: highest expected fidelity first
    ("BTC-USDT-SWAP", "BTCUSDT_SWAP_OKX"),
    ("ETH-USDT-SWAP", "ETHUSDT_SWAP_OKX"),
    ("SOL-USDT-SWAP", "SOLUSDT_SWAP_OKX"),
    ("LINK-USDT-SWAP", "LINKUSDT_SWAP_OKX"),
    ("DOGE-USDT-SWAP", "DOGEUSDT_SWAP_OKX"),
]

CHUNK_DAYS = 5
MIN_REQUEST_INTERVAL = 0.125  # 8 req/s = OKX 20req/2s limit minus 20% margin
MAX_RETRIES = 8
LEGACY_BAR_COUNT = 1_791_360  # per symbol, for the (1d) comparison only

_last_request_ts = 0.0


def throttle() -> None:
    global _last_request_ts
    wait = _last_request_ts + MIN_REQUEST_INTERVAL - time.monotonic()
    if wait > 0:
        time.sleep(wait)
    _last_request_ts = time.monotonic()


def fetch_page(inst_id: str, after_ms: int) -> list[list[str]]:
    """One mainnet request. Retries with backoff on transient failures."""
    params = f"instId={inst_id}&bar=1m&limit=100&after={after_ms}"
    req = urllib.request.Request(
        f"{OKX_URL}?{params}",
        headers={"User-Agent": "cta-mainnet-rebuild/1.0"},  # deliberately nothing else
    )
    last_err: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        throttle()
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                payload = json.loads(resp.read().decode())
            if payload.get("code") == "0" and isinstance(payload.get("data"), list):
                return payload["data"]
            last_err = RuntimeError(f"OKX code={payload.get('code')} msg={payload.get('msg')}")
        except Exception as exc:  # noqa: BLE001 - network layer, retry everything
            last_err = exc
        time.sleep(min(60.0, 2.0 * (2 ** (attempt - 1))))
    raise RuntimeError(f"fetch failed after {MAX_RETRIES} retries: {last_err}")


def ensure_db() -> sqlite3.Connection:
    assert DB_PATH != LEGACY_DB_PATH
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=60)
    conn.execute("PRAGMA journal_mode=WAL")  # concurrent per-symbol writer processes
    conn.execute("PRAGMA busy_timeout=60000")
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS "dbbardata" ("id" INTEGER NOT NULL PRIMARY KEY,
          "symbol" VARCHAR(255) NOT NULL, "exchange" VARCHAR(255) NOT NULL,
          "datetime" DATETIME NOT NULL, "interval" VARCHAR(255) NOT NULL,
          "volume" REAL NOT NULL, "turnover" REAL NOT NULL, "open_interest" REAL NOT NULL,
          "open_price" REAL NOT NULL, "high_price" REAL NOT NULL,
          "low_price" REAL NOT NULL, "close_price" REAL NOT NULL);
        CREATE UNIQUE INDEX IF NOT EXISTS "dbbardata_symbol_exchange_interval_datetime"
          ON "dbbardata" ("symbol", "exchange", "interval", "datetime");
        CREATE TABLE IF NOT EXISTS "dbbaroverview" ("id" INTEGER NOT NULL PRIMARY KEY,
          "symbol" VARCHAR(255) NOT NULL, "exchange" VARCHAR(255) NOT NULL,
          "interval" VARCHAR(255) NOT NULL, "count" INTEGER NOT NULL,
          "start" DATETIME NOT NULL, "end" DATETIME NOT NULL);
        CREATE UNIQUE INDEX IF NOT EXISTS "dbbaroverview_symbol_exchange_interval"
          ON "dbbaroverview" ("symbol", "exchange", "interval");
        CREATE TABLE IF NOT EXISTS "download_meta" ("id" INTEGER NOT NULL PRIMARY KEY,
          "symbol" VARCHAR(255) NOT NULL, "source" VARCHAR(255) NOT NULL,
          "server" VARCHAR(255) NOT NULL, "endpoint" VARCHAR(255) NOT NULL,
          "script" VARCHAR(255) NOT NULL, "started_at" DATETIME, "finished_at" DATETIME,
          "bar_count" INTEGER, "gap_count" INTEGER, "missing_minutes" INTEGER);
        """
    )
    conn.commit()
    return conn


def manifest_path(vt_symbol: str) -> Path:
    return MANIFEST_DIR / f"{vt_symbol}_1m_mainnet.json"


def load_manifest(vt_symbol: str) -> dict:
    path = manifest_path(vt_symbol)
    if path.exists():
        return json.loads(path.read_text())
    chunks = []
    cur = RANGE_START
    idx = 1
    while cur < RANGE_END_EXCL:
        end = min(cur + timedelta(days=CHUNK_DAYS), RANGE_END_EXCL)
        chunks.append(
            {
                "index": idx,
                "start_utc": cur.astimezone(timezone.utc).isoformat(),
                "end_exclusive_utc": end.astimezone(timezone.utc).isoformat(),
                "status": "pending",
                "bar_count": 0,
                "saved_at": None,
            }
        )
        cur = end
        idx += 1
    return {
        "version": 1,
        "server": "MAINNET",  # explicit environment field — the forensics lesson
        "source": "okx-public-rest",
        "endpoint": OKX_URL,
        "demo_header": False,
        "script": f"scripts/download_mainnet_history.py {SCRIPT_VERSION}",
        "vt_symbol": vt_symbol,
        "interval": "1m",
        "range_start": RANGE_START.isoformat(),
        "range_end_exclusive": RANGE_END_EXCL.isoformat(),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "chunks": chunks,
    }


def save_manifest(vt_symbol: str, manifest: dict) -> None:
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
    tmp = manifest_path(vt_symbol).with_suffix(".json.tmp")
    tmp.write_text(json.dumps(manifest, indent=2))
    tmp.replace(manifest_path(vt_symbol))


def download_chunk(inst_id: str, start_utc: datetime, end_excl_utc: datetime) -> dict[int, tuple]:
    """Paginate backwards from end; return {open_ts_ms: row} within window."""
    bars: dict[int, tuple] = {}
    after = int(end_excl_utc.timestamp() * 1000)
    start_ms = int(start_utc.timestamp() * 1000)
    seen: set[int] = set()
    while True:
        page = fetch_page(inst_id, after)
        if not page:
            break
        for row in page:
            ts = int(row[0])
            if str(row[-1]) != "1":  # unconfirmed bar — never store
                continue
            if start_ms <= ts < int(end_excl_utc.timestamp() * 1000):
                bars[ts] = (float(row[1]), float(row[2]), float(row[3]), float(row[4]),
                            float(row[5]), float(row[6]))
        oldest = int(page[-1][0])
        if oldest <= start_ms or oldest in seen:
            break
        seen.add(oldest)
        after = oldest
    return bars


def save_bars(conn: sqlite3.Connection, vt_symbol: str, bars: dict[int, tuple]) -> int:
    rows = []
    for ts in sorted(bars):
        o, h, l, c, vol, to = bars[ts]
        dt_local = datetime.fromtimestamp(ts / 1000, tz=timezone.utc).astimezone(SHANGHAI)
        rows.append((vt_symbol, "GLOBAL", dt_local.strftime("%Y-%m-%d %H:%M:%S"), "1m",
                     vol, to, 0.0, o, h, l, c))
    conn.executemany(
        'INSERT OR REPLACE INTO dbbardata (symbol,exchange,datetime,interval,volume,'
        'turnover,open_interest,open_price,high_price,low_price,close_price) '
        "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        rows,
    )
    conn.commit()
    return len(rows)


def integrity_check(conn: sqlite3.Connection, vt_symbol: str) -> dict:
    """Count bars, list gaps on the minute grid. No filling."""
    rows = conn.execute(
        "SELECT datetime FROM dbbardata WHERE symbol=? AND interval='1m' ORDER BY datetime",
        (vt_symbol,),
    ).fetchall()
    count = len(rows)
    expected = int((RANGE_END_EXCL - RANGE_START).total_seconds() // 60)
    gaps = []
    prev = None
    for (dt_str,) in rows:
        cur = datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S")
        if prev is not None and (cur - prev) > timedelta(minutes=1):
            gaps.append({
                "after": prev.isoformat(), "before": cur.isoformat(),
                "missing_minutes": int((cur - prev).total_seconds() // 60) - 1,
            })
        prev = cur
    missing = expected - count
    return {
        "vt_symbol": vt_symbol,
        "bar_count": count,
        "expected_minute_grid": expected,
        "missing_minutes_total": missing,
        "legacy_bar_count": LEGACY_BAR_COUNT,
        "diff_vs_legacy": count - LEGACY_BAR_COUNT,
        "first_dt": rows[0][0] if rows else None,
        "last_dt": rows[-1][0] if rows else None,
        "gap_count": len(gaps),
        "gaps": gaps,
    }


def update_overview_and_meta(conn: sqlite3.Connection, vt_symbol: str,
                             started_at: str, stats: dict) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO dbbaroverview (symbol,exchange,interval,count,start,end) "
        "VALUES (?,?,?,?,?,?)",
        (vt_symbol, "GLOBAL", "1m", stats["bar_count"], stats["first_dt"], stats["last_dt"]),
    )
    conn.execute(
        "INSERT INTO download_meta (symbol,source,server,endpoint,script,started_at,"
        "finished_at,bar_count,gap_count,missing_minutes) VALUES (?,?,?,?,?,?,?,?,?,?)",
        (vt_symbol, "okx-public-rest", "MAINNET", OKX_URL,
         f"scripts/download_mainnet_history.py {SCRIPT_VERSION}", started_at,
         datetime.now(timezone.utc).isoformat(), stats["bar_count"],
         stats["gap_count"], stats["missing_minutes_total"]),
    )
    conn.commit()


def main() -> int:
    # Optional CLI symbol filter (e.g. `... BTC SOL`) so symbols can run as
    # parallel processes; aggregate request rate stays under the IP limit.
    wanted = {a.upper() for a in sys.argv[1:]}
    symbols = [s for s in SYMBOLS if not wanted or s[1].split("USDT")[0] in wanted]
    if not symbols:
        print(f"no symbol matches {sorted(wanted)}", flush=True)
        return 1
    conn = ensure_db()
    all_stats = []
    for inst_id, vt_symbol in symbols:
        started_at = datetime.now(timezone.utc).isoformat()
        manifest = load_manifest(vt_symbol)
        pending = [c for c in manifest["chunks"] if c["status"] != "done"]
        total = len(manifest["chunks"])
        print(f"[{vt_symbol}] chunks: {total - len(pending)}/{total} done, "
              f"{len(pending)} to download", flush=True)
        t0 = time.monotonic()
        for n, chunk in enumerate(pending, 1):
            start_utc = datetime.fromisoformat(chunk["start_utc"])
            end_utc = datetime.fromisoformat(chunk["end_exclusive_utc"])
            bars = download_chunk(inst_id, start_utc, end_utc)
            saved = save_bars(conn, vt_symbol, bars)
            chunk["status"] = "done"
            chunk["bar_count"] = saved
            chunk["saved_at"] = datetime.now(timezone.utc).isoformat()
            save_manifest(vt_symbol, manifest)
            if n % 10 == 0 or n == len(pending):
                rate = n / max(time.monotonic() - t0, 1e-9)
                eta_min = (len(pending) - n) / max(rate, 1e-9) / 60
                print(f"[{vt_symbol}] chunk {n}/{len(pending)} "
                      f"({chunk['start_utc'][:10]}) saved={saved} ETA {eta_min:.0f} min",
                      flush=True)
        stats = integrity_check(conn, vt_symbol)
        update_overview_and_meta(conn, vt_symbol, started_at, stats)
        all_stats.append(stats)
        per_symbol_path = SUMMARY_PATH.with_name(f"integrity_{vt_symbol}.json")
        per_symbol_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False))
        print(f"[{vt_symbol}] DONE bars={stats['bar_count']} "
              f"(legacy {LEGACY_BAR_COUNT}, diff {stats['diff_vs_legacy']:+d}) "
              f"gaps={stats['gap_count']} missing_min={stats['missing_minutes_total']}",
              flush=True)
    print("ALL SYMBOLS COMPLETE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
