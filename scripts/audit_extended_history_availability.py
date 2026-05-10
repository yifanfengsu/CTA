#!/usr/bin/env python3
"""Audit extended local history availability without downloading data."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from history_time_utils import (
    DEFAULT_TIMEZONE,
    HistoryRange,
    expected_bar_count,
    normalize_bar_datetime,
    parse_history_range,
)
from history_utils import (
    HistoryCoverageSummary,
    analyze_datetime_coverage,
    get_database_timezone,
    parse_interval,
    to_database_query_range,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATABASE_PATH = PROJECT_ROOT / ".vntrader" / "database.db"
DEFAULT_CONFIG_DIR = PROJECT_ROOT / "config" / "instruments"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "reports" / "research" / "extended_history_availability"
DEFAULT_SYMBOLS = [
    "BTCUSDT_SWAP_OKX.GLOBAL",
    "ETHUSDT_SWAP_OKX.GLOBAL",
    "SOLUSDT_SWAP_OKX.GLOBAL",
    "LINKUSDT_SWAP_OKX.GLOBAL",
    "DOGEUSDT_SWAP_OKX.GLOBAL",
]
DEFAULT_WINDOWS = [
    "2025-01-01:2026-03-31",
    "2023-01-01:2026-03-31",
    "2021-01-01:2026-03-31",
]
DEFAULT_INTERVAL = "1m"
OKX_PUBLIC_INSTRUMENTS_URL = "https://www.okx.com/api/v5/public/instruments"
NUMERIC_METADATA_FIELDS = {"size", "pricetick", "min_volume"}


@dataclass(frozen=True, slots=True)
class AuditWindow:
    """One requested history availability window."""

    label: str
    start_arg: str
    end_arg: str
    history_range: HistoryRange


class OkxListingMetadataError(Exception):
    """Raised when OKX public listing metadata is not available."""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(
        description="Audit local sqlite coverage for extended multi-symbol trend research."
    )
    parser.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS))
    parser.add_argument("--windows", default=",".join(DEFAULT_WINDOWS))
    parser.add_argument("--interval", default=DEFAULT_INTERVAL, choices=("1m", "1h", "d", "w"))
    parser.add_argument("--timezone", default=DEFAULT_TIMEZONE)
    parser.add_argument("--config-dir", default=str(DEFAULT_CONFIG_DIR))
    parser.add_argument("--database-path", default=str(DEFAULT_DATABASE_PATH))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--check-okx-listing-metadata",
        action="store_true",
        help="Optionally query OKX public instruments metadata for listTime/state.",
    )
    parser.add_argument("--okx-timeout", type=float, default=10.0)
    parser.add_argument("--json", action="store_true", help="Print the JSON payload after writing outputs.")
    return parser.parse_args(argv)


def resolve_path(path_arg: str | Path, project_root: Path = PROJECT_ROOT) -> Path:
    """Resolve a relative path from project root."""

    path = Path(path_arg)
    if not path.is_absolute():
        path = project_root / path
    return path


def parse_symbol_list(value: str) -> list[str]:
    """Parse comma/space separated vt_symbols while preserving order."""

    seen: set[str] = set()
    symbols: list[str] = []
    for token in re.split(r"[\s,]+", value):
        symbol = token.strip()
        if not symbol or symbol in seen:
            continue
        seen.add(symbol)
        symbols.append(symbol)
    if not symbols:
        raise ValueError("--symbols must contain at least one vt_symbol")
    return symbols


def interval_to_delta(interval: str) -> timedelta:
    """Return the expected bar delta for a supported interval string."""

    mapping = {
        "1m": timedelta(minutes=1),
        "1h": timedelta(hours=1),
        "d": timedelta(days=1),
        "w": timedelta(weeks=1),
    }
    try:
        return mapping[interval]
    except KeyError as exc:
        raise ValueError(f"Unsupported interval: {interval}") from exc


def parse_windows(value: str, interval_delta: timedelta, timezone_name: str) -> list[AuditWindow]:
    """Parse comma/space separated START:END windows."""

    windows: list[AuditWindow] = []
    seen: set[str] = set()
    for token in re.split(r"[\s,]+", value):
        text = token.strip()
        if not text:
            continue
        if ":" not in text:
            raise ValueError(f"Invalid window {text!r}; expected START:END")
        start_arg, end_arg = [part.strip() for part in text.split(":", maxsplit=1)]
        if not start_arg or not end_arg:
            raise ValueError(f"Invalid window {text!r}; expected START:END")
        label = f"{start_arg}:{end_arg}"
        if label in seen:
            continue
        seen.add(label)
        windows.append(
            AuditWindow(
                label=label,
                start_arg=start_arg,
                end_arg=end_arg,
                history_range=parse_history_range(
                    start_arg=start_arg,
                    end_arg=end_arg,
                    interval_delta=interval_delta,
                    timezone_name=timezone_name,
                ),
            )
        )
    if not windows:
        raise ValueError("--windows must contain at least one START:END window")
    return windows


def split_vt_symbol(vt_symbol: str) -> tuple[str, str]:
    """Split a vn.py vt_symbol into symbol and exchange text like verify_okx_history."""

    from vnpy.trader.utility import extract_vt_symbol

    symbol, exchange = extract_vt_symbol(vt_symbol)
    return symbol, exchange.value


def config_path_for_vt_symbol(config_dir: Path, vt_symbol: str) -> Path:
    """Return the expected local instrument config path."""

    symbol, _exchange = split_vt_symbol(vt_symbol)
    return config_dir / f"{symbol.lower()}.json"


def read_json_object(path: Path) -> tuple[dict[str, Any], str | None]:
    """Read one JSON object from disk."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}, "instrument_config_missing"
    except Exception as exc:
        return {}, f"instrument_config_read_failed: {exc!r}"
    if not isinstance(payload, dict):
        return {}, "instrument_config_root_not_object"
    return payload, None


def finite_positive(value: Any) -> bool:
    """Return whether value is a positive finite number."""

    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return bool(math.isfinite(number) and number > 0)


def metadata_field_missing(payload: dict[str, Any], field: str) -> bool:
    """Return whether a required metadata field is absent or invalid."""

    if field in NUMERIC_METADATA_FIELDS:
        return not finite_positive(payload.get(field))
    return not str(payload.get(field) or "").strip()


def inspect_instrument_metadata(config_dir: Path, vt_symbol: str) -> dict[str, Any]:
    """Inspect local config/instruments metadata for one symbol."""

    symbol, exchange = split_vt_symbol(vt_symbol)
    path = config_path_for_vt_symbol(config_dir, vt_symbol)
    payload, warning = read_json_object(path)
    missing_fields = [
        field
        for field in ("okx_inst_id", "product", "size", "pricetick", "min_volume")
        if metadata_field_missing(payload, field)
    ]
    if payload.get("needs_okx_contract_metadata_refresh") is True:
        missing_fields.append("needs_okx_contract_metadata_refresh")
    warnings = []
    if warning:
        warnings.append(warning)
    if missing_fields:
        warnings.append(f"metadata_incomplete: {', '.join(missing_fields)}")

    return {
        "vt_symbol": vt_symbol,
        "symbol": payload.get("symbol") or symbol,
        "exchange": payload.get("exchange") or exchange,
        "config_path": str(path),
        "config_exists": path.exists(),
        "okx_inst_id": str(payload.get("okx_inst_id") or payload.get("name") or "").strip(),
        "product": payload.get("product"),
        "size": payload.get("size"),
        "pricetick": payload.get("pricetick"),
        "min_volume": payload.get("min_volume"),
        "metadata_complete": bool(not warning and not missing_fields),
        "missing_fields": missing_fields,
        "warnings": warnings,
    }


def format_dt(value: datetime | None) -> str | None:
    """Format datetime for reports without timezone suffix."""

    if value is None:
        return None
    return value.replace(tzinfo=None).isoformat(sep=" ", timespec="seconds")


def parse_database_datetime(
    value: Any,
    history_range: HistoryRange,
    database_timezone: Any,
) -> datetime | None:
    """Parse sqlite datetime text using vnpy_sqlite's database timezone."""

    if value is None:
        return None
    if isinstance(value, datetime):
        parsed = value
    else:
        try:
            parsed = datetime.fromisoformat(str(value))
        except ValueError:
            return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=database_timezone)
    return normalize_bar_datetime(
        parsed,
        timezone_name=history_range.timezone_name,
        interval_delta=history_range.interval_delta,
    )


def command_date(value: datetime) -> str:
    """Return a date-only command argument from a missing range endpoint."""

    return value.date().isoformat()


def build_suggested_download_command(
    vt_symbol: str,
    interval: str,
    start_dt: datetime,
    end_dt: datetime,
    timezone_name: str,
) -> str:
    """Build a non-executing command suggestion for one missing range."""

    return (
        "python scripts/download_okx_history.py "
        f"--vt-symbol {vt_symbol} "
        f"--interval {interval} "
        f"--start {command_date(start_dt)} "
        f"--end {command_date(end_dt)} "
        "--chunk-days 3 "
        f"--timezone {timezone_name} "
        "--resume "
        "--repair-missing "
        "--source auto"
    )


def build_missing_ranges(
    vt_symbol: str,
    interval: str,
    actual_times: list[datetime],
    history_range: HistoryRange,
) -> list[dict[str, Any]]:
    """Compress coverage gaps into contiguous missing ranges."""

    expected = expected_bar_count(history_range)
    if expected <= 0:
        return []
    delta = history_range.interval_delta
    ranges: list[dict[str, Any]] = []

    def append_range(start_dt: datetime, end_dt: datetime) -> None:
        missing_count = int((end_dt - start_dt) / delta) + 1
        if missing_count <= 0:
            return
        ranges.append(
            {
                "start": format_dt(start_dt),
                "end": format_dt(end_dt),
                "missing_count": missing_count,
                "suggested_download_command": build_suggested_download_command(
                    vt_symbol=vt_symbol,
                    interval=interval,
                    start_dt=start_dt,
                    end_dt=end_dt,
                    timezone_name=history_range.timezone_name,
                ),
            }
        )

    if not actual_times:
        append_range(history_range.start, history_range.end_exclusive - delta)
        return ranges

    first = actual_times[0]
    if first > history_range.start:
        append_range(history_range.start, first - delta)

    previous = first
    for current in actual_times[1:]:
        if current - previous > delta:
            append_range(previous + delta, current - delta)
        previous = current

    last_expected = history_range.end_exclusive - delta
    if previous < last_expected:
        append_range(previous + delta, last_expected)

    return ranges


def coverage_missing_ranges(
    vt_symbol: str,
    interval: str,
    coverage: HistoryCoverageSummary,
    history_range: HistoryRange,
) -> list[dict[str, Any]]:
    """Convert shared coverage missing ranges into audit report rows."""

    ranges: list[dict[str, Any]] = []
    for item in coverage.missing_ranges:
        ranges.append(
            {
                "start": format_dt(item.start),
                "end": format_dt(item.end),
                "missing_count": item.missing_count,
                "suggested_download_command": build_suggested_download_command(
                    vt_symbol=vt_symbol,
                    interval=interval,
                    start_dt=item.start,
                    end_dt=item.end,
                    timezone_name=history_range.timezone_name,
                ),
            }
        )
    return ranges


def build_query_debug(
    *,
    database_path: Path,
    database_exists: bool,
    table_exists: bool,
    vt_symbol: str,
    symbol: str,
    exchange: str,
    interval: str,
    history_range: HistoryRange,
    rows_found_total: int = 0,
    first_dt_all: Any = None,
    last_dt_all: Any = None,
    rows_found_in_window: int = 0,
    normalized_rows_found_in_window: int = 0,
    warning: str | None = None,
) -> dict[str, Any]:
    """Build the per-symbol query diagnostics required by the audit report."""

    query_start, query_end = to_database_query_range(history_range)
    db_tz = get_database_timezone()
    return {
        "database_path": str(database_path),
        "database_exists": database_exists,
        "dbbardata_table_exists": table_exists,
        "vt_symbol": vt_symbol,
        "db_symbol_used": symbol,
        "db_exchange_used": exchange,
        "interval_used": interval,
        "datetime_storage_assumption": f"dbbardata.datetime is stored as naive {db_tz}",
        "query_start": format_dt(query_start),
        "query_end_exclusive": format_dt(query_end),
        "history_start": format_dt(history_range.start),
        "history_end_exclusive": format_dt(history_range.end_exclusive),
        "history_start_utc": history_range.start_utc.isoformat(),
        "history_end_exclusive_utc": history_range.end_exclusive_utc.isoformat(),
        "rows_found_total": rows_found_total,
        "first_dt_all": str(first_dt_all) if first_dt_all is not None else None,
        "last_dt_all": str(last_dt_all) if last_dt_all is not None else None,
        "rows_found_in_window": rows_found_in_window,
        "normalized_rows_found_in_window": normalized_rows_found_in_window,
        "warning": warning,
    }


def empty_coverage(
    vt_symbol: str,
    interval: str,
    window: AuditWindow,
    warning: str,
    database_path: Path,
    database_exists: bool,
    table_exists: bool,
) -> dict[str, Any]:
    """Build a coverage record when sqlite data cannot be queried."""

    expected = expected_bar_count(window.history_range)
    try:
        symbol, exchange = split_vt_symbol(vt_symbol)
    except ValueError:
        symbol, exchange = "", ""
    missing_ranges = build_missing_ranges(
        vt_symbol=vt_symbol,
        interval=interval,
        actual_times=[],
        history_range=window.history_range,
    )
    return {
        "vt_symbol": vt_symbol,
        "interval": interval,
        "window": window.label,
        "start": window.start_arg,
        "end": window.end_arg,
        "expected_count": expected,
        "total_count": 0,
        "missing_count": expected,
        "gap_count": len(missing_ranges),
        "first_dt": None,
        "last_dt": None,
        "coverage_ratio": 0.0,
        "history_ready": False,
        "missing_ranges": missing_ranges,
        "warning": warning,
        "query_debug": build_query_debug(
            database_path=database_path,
            database_exists=database_exists,
            table_exists=table_exists,
            vt_symbol=vt_symbol,
            symbol=symbol,
            exchange=exchange,
            interval=interval,
            history_range=window.history_range,
            warning=warning,
        ),
    }


def query_window_coverage(
    connection: sqlite3.Connection,
    database_path: Path,
    vt_symbol: str,
    symbol: str,
    exchange: str,
    interval: str,
    window: AuditWindow,
) -> dict[str, Any]:
    """Query local sqlite coverage for one symbol/window."""

    history_range = window.history_range
    query_start_dt, query_end_dt = to_database_query_range(history_range)
    query_start = format_dt(query_start_dt)
    query_end = format_dt(query_end_dt)
    total_row = connection.execute(
        (
            "select count(distinct datetime), min(datetime), max(datetime) "
            "from dbbardata where symbol = ? and exchange = ? and interval = ?"
        ),
        (symbol, exchange, interval),
    ).fetchone()
    rows_found_total = int(total_row[0] or 0)
    first_dt_all = total_row[1]
    last_dt_all = total_row[2]
    rows = connection.execute(
        (
            "select datetime from dbbardata "
            "where symbol = ? and exchange = ? and interval = ? "
            "and datetime >= ? and datetime < ? "
            "order by datetime"
        ),
        (symbol, exchange, interval, query_start, query_end),
    ).fetchall()
    database_timezone = get_database_timezone()
    actual_set = {
        parsed
        for row in rows
        if (parsed := parse_database_datetime(row[0], history_range, database_timezone)) is not None
        and history_range.start <= parsed < history_range.end_exclusive
    }
    actual_times = sorted(actual_set)
    coverage = analyze_datetime_coverage(actual_times, history_range)
    missing_ranges = coverage_missing_ranges(
        vt_symbol=vt_symbol,
        interval=interval,
        coverage=coverage,
        history_range=history_range,
    )
    coverage_ratio = (coverage.total_count / coverage.expected_count) if coverage.expected_count else 0.0
    history_ready = bool(coverage.is_complete and coverage.gap_count == 0)
    return {
        "vt_symbol": vt_symbol,
        "interval": interval,
        "window": window.label,
        "start": window.start_arg,
        "end": window.end_arg,
        "expected_count": coverage.expected_count,
        "total_count": coverage.total_count,
        "missing_count": coverage.missing_count,
        "gap_count": coverage.gap_count,
        "first_dt": format_dt(coverage.first_dt),
        "last_dt": format_dt(coverage.last_dt),
        "coverage_ratio": coverage_ratio,
        "history_ready": history_ready,
        "missing_ranges": missing_ranges,
        "query_debug": build_query_debug(
            database_path=database_path,
            database_exists=True,
            table_exists=True,
            vt_symbol=vt_symbol,
            symbol=symbol,
            exchange=exchange,
            interval=interval,
            history_range=history_range,
            rows_found_total=rows_found_total,
            first_dt_all=first_dt_all,
            last_dt_all=last_dt_all,
            rows_found_in_window=len(rows),
            normalized_rows_found_in_window=coverage.total_count,
        ),
    }


def query_database_coverage(
    database_path: Path,
    symbols: list[str],
    instruments: dict[str, dict[str, Any]],
    windows: list[AuditWindow],
    interval: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Query local sqlite coverage for every symbol/window."""

    coverages: list[dict[str, Any]] = []
    if not database_path.exists():
        warning = f"database_not_found: {database_path}"
        for vt_symbol in symbols:
            for window in windows:
                coverages.append(
                    empty_coverage(
                        vt_symbol,
                        interval,
                        window,
                        warning,
                        database_path=database_path,
                        database_exists=False,
                        table_exists=False,
                    )
                )
        return {
            "path": str(database_path),
            "database_path": str(database_path),
            "exists": False,
            "database_exists": False,
            "dbbardata_table_exists": False,
            "warning": warning,
        }, coverages

    connection: sqlite3.Connection | None = None
    try:
        connection = sqlite3.connect(database_path)
        table_exists = bool(
            connection.execute(
                "select 1 from sqlite_master where type='table' and name='dbbardata'"
            ).fetchone()
        )
        if not table_exists:
            warning = "dbbardata_table_not_found"
            for vt_symbol in symbols:
                for window in windows:
                    coverages.append(
                        empty_coverage(
                            vt_symbol,
                            interval,
                            window,
                            warning,
                            database_path=database_path,
                            database_exists=True,
                            table_exists=False,
                        )
                    )
            return {
                "path": str(database_path),
                "database_path": str(database_path),
                "exists": True,
                "database_exists": True,
                "dbbardata_table_exists": False,
                "warning": warning,
            }, coverages

        interval_value = parse_interval(interval)[0].value
        row_totals: dict[str, dict[str, Any]] = {}
        for vt_symbol in symbols:
            symbol, exchange = split_vt_symbol(vt_symbol)
            total_row = connection.execute(
                (
                    "select count(distinct datetime), min(datetime), max(datetime) "
                    "from dbbardata where symbol = ? and exchange = ? and interval = ?"
                ),
                (symbol, exchange, interval_value),
            ).fetchone()
            row_totals[vt_symbol] = {
                "db_symbol_used": symbol,
                "db_exchange_used": exchange,
                "interval_used": interval_value,
                "rows_found_total": int(total_row[0] or 0),
                "first_dt_all": str(total_row[1]) if total_row[1] is not None else None,
                "last_dt_all": str(total_row[2]) if total_row[2] is not None else None,
            }

        for vt_symbol in symbols:
            symbol, exchange = split_vt_symbol(vt_symbol)
            for window in windows:
                coverages.append(
                    query_window_coverage(
                        connection=connection,
                        database_path=database_path,
                        vt_symbol=vt_symbol,
                        symbol=symbol,
                        exchange=exchange,
                        interval=interval_value,
                        window=window,
                    )
                )
        return {
            "path": str(database_path),
            "database_path": str(database_path),
            "exists": True,
            "database_exists": True,
            "table": "dbbardata",
            "dbbardata_table_exists": True,
            "symbol_totals": row_totals,
        }, coverages
    except Exception as exc:
        warning = f"database_query_failed: {exc!r}"
        for vt_symbol in symbols:
            for window in windows:
                coverages.append(
                    empty_coverage(
                        vt_symbol,
                        interval,
                        window,
                        warning,
                        database_path=database_path,
                        database_exists=database_path.exists(),
                        table_exists=False,
                    )
                )
        return {
            "path": str(database_path),
            "database_path": str(database_path),
            "exists": database_path.exists(),
            "database_exists": database_path.exists(),
            "dbbardata_table_exists": False,
            "warning": warning,
        }, coverages
    finally:
        if connection is not None:
            connection.close()


def build_okx_url(inst_id: str) -> str:
    """Build OKX public instruments URL for one SWAP instrument."""

    return f"{OKX_PUBLIC_INSTRUMENTS_URL}?{urlencode({'instType': 'SWAP', 'instId': inst_id})}"


def fetch_okx_instrument(inst_id: str, timeout: float = 10.0) -> dict[str, Any]:
    """Fetch one OKX public instrument metadata object."""

    request = Request(build_okx_url(inst_id), headers={"User-Agent": "cta-extended-history-audit/1.0"})
    try:
        with urlopen(request, timeout=timeout) as response:
            raw_body = response.read().decode("utf-8")
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace") if exc.fp else ""
        raise OkxListingMetadataError(f"okx_http_error status={exc.code} body={body[:300]}") from exc
    except URLError as exc:
        raise OkxListingMetadataError(f"okx_url_error reason={exc.reason!r}") from exc
    except TimeoutError as exc:
        raise OkxListingMetadataError("okx_request_timeout") from exc

    try:
        payload = json.loads(raw_body)
    except json.JSONDecodeError as exc:
        raise OkxListingMetadataError(f"okx_invalid_json: {exc!r}") from exc
    if not isinstance(payload, dict):
        raise OkxListingMetadataError("okx_response_root_not_object")
    if str(payload.get("code")) != "0":
        raise OkxListingMetadataError(f"okx_error code={payload.get('code')} msg={payload.get('msg')}")
    data = payload.get("data")
    if not isinstance(data, list) or not data:
        raise OkxListingMetadataError("okx_response_data_empty")
    item = data[0]
    if not isinstance(item, dict):
        raise OkxListingMetadataError("okx_response_data_item_not_object")
    return item


def parse_okx_millis(value: Any) -> datetime | None:
    """Parse an OKX millisecond timestamp into UTC datetime."""

    text = str(value or "").strip()
    if not text:
        return None
    try:
        millis = int(text)
    except ValueError:
        return None
    return datetime.fromtimestamp(millis / 1000, tz=timezone.utc)


def build_listing_metadata_unknown(
    vt_symbol: str,
    okx_inst_id: str,
    warning: str | None = None,
) -> dict[str, Any]:
    """Build an unknown listing metadata record."""

    warnings = []
    if warning:
        warnings.append(warning)
    return {
        "vt_symbol": vt_symbol,
        "okx_inst_id": okx_inst_id,
        "okx_list_time": None,
        "okx_exp_time": None,
        "okx_state": None,
        "listing_metadata_available": False,
        "warning": warning,
        "warnings": warnings,
    }


def collect_listing_metadata(
    instruments: dict[str, dict[str, Any]],
    check_okx: bool,
    timeout: float,
    fetcher: Callable[[str, float], dict[str, Any]] = fetch_okx_instrument,
) -> dict[str, dict[str, Any]]:
    """Collect optional OKX listing metadata, failing open on errors."""

    listings: dict[str, dict[str, Any]] = {}
    for vt_symbol, instrument in instruments.items():
        okx_inst_id = str(instrument.get("okx_inst_id") or "").strip()
        if not okx_inst_id:
            listings[vt_symbol] = build_listing_metadata_unknown(
                vt_symbol=vt_symbol,
                okx_inst_id="",
                warning="okx_inst_id_missing_for_listing_metadata",
            )
            continue
        if not check_okx:
            listings[vt_symbol] = build_listing_metadata_unknown(
                vt_symbol=vt_symbol,
                okx_inst_id=okx_inst_id,
                warning="okx_listing_metadata_not_checked",
            )
            continue
        try:
            raw = fetcher(okx_inst_id, timeout)
        except Exception as exc:
            listings[vt_symbol] = build_listing_metadata_unknown(
                vt_symbol=vt_symbol,
                okx_inst_id=okx_inst_id,
                warning=f"okx_listing_metadata_unavailable: {exc}",
            )
            continue

        list_time = parse_okx_millis(raw.get("listTime"))
        exp_time = parse_okx_millis(raw.get("expTime"))
        warning = None if list_time is not None else "okx_list_time_unknown"
        listings[vt_symbol] = {
            "vt_symbol": vt_symbol,
            "okx_inst_id": str(raw.get("instId") or okx_inst_id),
            "okx_list_time": list_time.isoformat() if list_time is not None else None,
            "okx_exp_time": exp_time.isoformat() if exp_time is not None else None,
            "okx_state": raw.get("state"),
            "listing_metadata_available": list_time is not None,
            "warning": warning,
            "warnings": [warning] if warning else [],
        }
    return listings


def listing_before_window_start(listing: dict[str, Any], window: AuditWindow) -> bool | str:
    """Return true/false/unknown for listing before window start."""

    value = listing.get("okx_list_time")
    if not value:
        return "unknown"
    try:
        list_time = datetime.fromisoformat(str(value))
    except ValueError:
        return "unknown"
    if list_time.tzinfo is None:
        list_time = list_time.replace(tzinfo=timezone.utc)
    return bool(list_time <= window.history_range.start_utc)


def build_symbol_window_records(
    symbols: list[str],
    instruments: dict[str, dict[str, Any]],
    listings: dict[str, dict[str, Any]],
    coverages: list[dict[str, Any]],
    windows: list[AuditWindow],
) -> list[dict[str, Any]]:
    """Merge metadata, listing metadata, and coverage for each symbol/window."""

    coverage_by_key = {(item["vt_symbol"], item["window"]): item for item in coverages}
    records: list[dict[str, Any]] = []
    for vt_symbol in symbols:
        instrument = instruments[vt_symbol]
        listing = listings[vt_symbol]
        for window in windows:
            coverage = coverage_by_key[(vt_symbol, window.label)]
            listing_relation = listing_before_window_start(listing, window)
            warnings = []
            warnings.extend(instrument.get("warnings") or [])
            warnings.extend(listing.get("warnings") or [])
            if coverage.get("warning"):
                warnings.append(str(coverage["warning"]))
            if listing_relation == "unknown":
                warnings.append("listing_before_window_start_unknown")
            records.append(
                {
                    **coverage,
                    "instrument_metadata": {
                        "okx_inst_id": instrument.get("okx_inst_id"),
                        "product": instrument.get("product"),
                        "size": instrument.get("size"),
                        "pricetick": instrument.get("pricetick"),
                        "min_volume": instrument.get("min_volume"),
                        "metadata_complete": instrument.get("metadata_complete"),
                    },
                    "listing_metadata": {
                        **listing,
                        "listing_before_window_start": listing_relation,
                    },
                    "metadata_complete": bool(instrument.get("metadata_complete")),
                    "listing_before_window_start": listing_relation,
                    "ready": bool(
                        coverage.get("history_ready")
                        and listing_relation is not False
                    ),
                    "warnings": warnings,
                }
            )
    return records


def readiness_key_for_window(window: AuditWindow) -> str:
    """Return the summary readiness key for a window start year."""

    return f"ready_for_{window.start_arg[:4]}_window"


def build_readiness(records: list[dict[str, Any]], windows: list[AuditWindow]) -> dict[str, Any]:
    """Build high-level extended research readiness decision."""

    readiness: dict[str, Any] = {}
    window_summaries: list[dict[str, Any]] = []
    for window in windows:
        rows = [item for item in records if item["window"] == window.label]
        ready_symbols = [item["vt_symbol"] for item in rows if item["ready"]]
        missing_symbols = [item["vt_symbol"] for item in rows if int(item.get("missing_count") or 0) > 0]
        metadata_incomplete = [item["vt_symbol"] for item in rows if not item.get("metadata_complete")]
        listing_false = [item["vt_symbol"] for item in rows if item.get("listing_before_window_start") is False]
        listing_unknown = [item["vt_symbol"] for item in rows if item.get("listing_before_window_start") == "unknown"]
        ready = bool(len(ready_symbols) == len(rows) and rows)
        readiness[readiness_key_for_window(window)] = ready
        window_summaries.append(
            {
                "window": window.label,
                "start": window.start_arg,
                "end": window.end_arg,
                "ready": ready,
                "ready_symbols": ready_symbols,
                "missing_symbols": missing_symbols,
                "metadata_incomplete_symbols": metadata_incomplete,
                "listing_after_window_start_symbols": listing_false,
                "listing_unknown_symbols": listing_unknown,
                "total_expected_count": sum(int(item.get("expected_count") or 0) for item in rows),
                "total_count": sum(int(item.get("total_count") or 0) for item in rows),
                "total_missing_count": sum(int(item.get("missing_count") or 0) for item in rows),
            }
        )

    summary_by_window = {item["window"]: item for item in window_summaries}
    preferred = "none"
    reason = "no_missing_extended_window"
    window_2023 = next((window for window in windows if window.start_arg.startswith("2023-")), None)
    window_2021 = next((window for window in windows if window.start_arg.startswith("2021-")), None)
    if window_2023 is not None:
        summary_2023 = summary_by_window[window_2023.label]
        if not summary_2023["ready"] and not summary_2023["listing_after_window_start_symbols"]:
            preferred = window_2023.label
            reason = "2023_window_can_be_completed_by_filling_missing_local_history"
    if preferred == "none" and window_2021 is not None:
        summary_2021 = summary_by_window[window_2021.label]
        if not summary_2021["ready"] and not summary_2021["listing_after_window_start_symbols"]:
            preferred = window_2021.label
            reason = "2021_window_is_next_missing_window_after_2023_ready"

    can_enter_extended = False
    blocking_reasons: list[str] = []
    optional_warnings: list[str] = []
    if window_2023 is not None and summary_by_window[window_2023.label]["ready"]:
        can_enter_extended = True
    elif window_2021 is not None and summary_by_window[window_2021.label]["ready"]:
        can_enter_extended = True

    if not can_enter_extended:
        for summary in window_summaries:
            if summary["window"].startswith("2025-"):
                continue
            if summary["total_missing_count"]:
                blocking_reasons.append(
                    f"missing_{summary['start'][:4]}_window_history: "
                    f"missing_count={summary['total_missing_count']} symbols={','.join(summary['missing_symbols'])}"
                )
            if summary["metadata_incomplete_symbols"]:
                optional_warnings.append(
                    f"incomplete_metadata_for_{summary['start'][:4]}_window: "
                    f"{','.join(summary['metadata_incomplete_symbols'])}"
                )
            if summary["listing_after_window_start_symbols"]:
                blocking_reasons.append(
                    f"listing_after_{summary['start'][:4]}_window_start: "
                    f"{','.join(summary['listing_after_window_start_symbols'])}"
                )
    listing_unknown_all = sorted({symbol for summary in window_summaries for symbol in summary["listing_unknown_symbols"]})
    if listing_unknown_all:
        optional_warnings.append(f"listing_time_unknown: {','.join(listing_unknown_all)}")

    return {
        **readiness,
        "windows": window_summaries,
        "recommended_next_download_window": preferred,
        "recommendation_reason": reason,
        "can_enter_extended_trend_research": can_enter_extended and not blocking_reasons,
        "blocking_reasons": blocking_reasons,
        "optional_warnings": optional_warnings,
    }


def flatten_missing_ranges(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return CSV-friendly missing range rows."""

    rows: list[dict[str, Any]] = []
    for record in records:
        for item in record.get("missing_ranges") or []:
            rows.append(
                {
                    "vt_symbol": record["vt_symbol"],
                    "window": record["window"],
                    "interval": record["interval"],
                    "start": item["start"],
                    "end": item["end"],
                    "missing_count": item["missing_count"],
                    "suggested_download_command": item["suggested_download_command"],
                }
            )
    return rows


def build_download_plan(missing_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build a sorted download plan from missing range rows."""

    return [
        {
            "priority": index,
            "vt_symbol": row["vt_symbol"],
            "window": row["window"],
            "interval": row["interval"],
            "start": row["start"],
            "end": row["end"],
            "missing_count": row["missing_count"],
            "command": row["suggested_download_command"],
        }
        for index, row in enumerate(
            sorted(
                missing_rows,
                key=lambda item: (
                    0 if str(item["window"]).startswith("2023-") else 1,
                    str(item["vt_symbol"]),
                    str(item["start"]),
                ),
            ),
            start=1,
        )
    ]


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    """Write CSV rows, including a header for empty outputs."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def bool_text(value: Any) -> str:
    """Format booleans and unknowns for Markdown."""

    if isinstance(value, bool):
        return str(value).lower()
    return str(value)


def render_window_summary_table(readiness: dict[str, Any]) -> str:
    """Render Markdown window readiness summary."""

    lines = [
        "| window | ready_symbols | missing_symbols | missing_count | recommended |",
        "|---|---:|---|---:|---|",
    ]
    recommended = readiness["recommended_next_download_window"]
    for summary in readiness["windows"]:
        lines.append(
            "| {window} | {ready} | {missing} | {missing_count} | {recommended} |".format(
                window=summary["window"],
                ready=len(summary["ready_symbols"]),
                missing=", ".join(summary["missing_symbols"]) or "-",
                missing_count=summary["total_missing_count"],
                recommended=str(summary["window"] == recommended).lower(),
            )
        )
    return "\n".join(lines)


def render_missing_table(missing_rows: list[dict[str, Any]], limit: int = 20) -> str:
    """Render Markdown missing range summary."""

    lines = [
        "| symbol | window | missing_count | suggested_command |",
        "|---|---|---:|---|",
    ]
    if not missing_rows:
        lines.append("| - | - | 0 | - |")
        return "\n".join(lines)
    for row in missing_rows[:limit]:
        lines.append(
            "| {symbol} | {window} | {missing_count} | `{command}` |".format(
                symbol=row["vt_symbol"],
                window=row["window"],
                missing_count=row["missing_count"],
                command=str(row["suggested_download_command"]).replace("|", "/"),
            )
        )
    return "\n".join(lines)


def render_symbol_coverage_table(records: list[dict[str, Any]]) -> str:
    """Render Markdown per-symbol coverage rows."""

    lines = [
        "| symbol | window | total_count | expected_count | missing_count | gap_count | coverage_ratio | history_ready | listing_before_start | warning |",
        "|---|---|---:|---:|---:|---:|---:|---|---|---|",
    ]
    for record in records:
        warnings = "; ".join(dict.fromkeys(record.get("warnings") or [])) or "-"
        lines.append(
            "| {symbol} | {window} | {total} | {expected} | {missing} | {gaps} | {ratio:.6f} | {ready} | {listing} | {warning} |".format(
                symbol=record["vt_symbol"],
                window=record["window"],
                total=record["total_count"],
                expected=record["expected_count"],
                missing=record["missing_count"],
                gaps=record["gap_count"],
                ratio=float(record["coverage_ratio"]),
                ready=str(bool(record["history_ready"])).lower(),
                listing=bool_text(record["listing_before_window_start"]),
                warning=warnings.replace("|", "/"),
            )
        )
    return "\n".join(lines)


def render_report(payload: dict[str, Any]) -> str:
    """Render the required Markdown availability report."""

    readiness = payload["readiness"]
    missing_rows = payload["missing_ranges"]
    window_by_label = {item["window"]: item for item in readiness["windows"]}
    summary_2025 = next((item for item in readiness["windows"] if item["start"].startswith("2025-")), None)
    summary_2023 = next((item for item in readiness["windows"] if item["start"].startswith("2023-")), None)
    summary_2021 = next((item for item in readiness["windows"] if item["start"].startswith("2021-")), None)
    listing_unknown = sorted(
        {
            record["vt_symbol"]
            for record in payload["symbol_windows"]
            if record.get("listing_before_window_start") == "unknown"
        }
    )
    listing_after = sorted(
        {
            record["vt_symbol"]
            for record in payload["symbol_windows"]
            if record.get("listing_before_window_start") is False
        }
    )
    blocking_lines = (
        "\n".join(f"- {item}" for item in readiness["blocking_reasons"])
        if readiness["blocking_reasons"]
        else "- none"
    )
    optional_warning_lines = (
        "\n".join(f"- {item}" for item in readiness.get("optional_warnings") or [])
        if readiness.get("optional_warnings")
        else "- none"
    )
    missing_2023 = (
        ", ".join(summary_2023["missing_symbols"])
        if summary_2023 and summary_2023["missing_symbols"]
        else "none"
    )
    missing_2021 = (
        ", ".join(summary_2021["missing_symbols"])
        if summary_2021 and summary_2021["missing_symbols"]
        else "none"
    )
    return (
        "# Extended History Availability Audit\n\n"
        "## Required Questions\n"
        f"1. Current 2025-01-01 to 2026-03-31 complete: {str(bool(summary_2025 and summary_2025['ready'])).lower()}.\n"
        f"2. 2023-01-01 to 2026-03-31 missing symbols: {missing_2023}.\n"
        f"3. 2021-01-01 to 2026-03-31 feasible with current local data: {str(bool(summary_2021 and summary_2021['ready'])).lower()}.\n"
        f"4. Suspected listing later than 2021/2023: {', '.join(listing_after) if listing_after else 'none confirmed'}; listing_time_unknown={', '.join(listing_unknown) if listing_unknown else 'none'}.\n"
        f"5. Recommended next download window: {readiness['recommended_next_download_window']}.\n"
        f"6. Can enter Extended Trend Research: {str(bool(readiness['can_enter_extended_trend_research'])).lower()}.\n"
        f"7. Blocking reasons: {', '.join(readiness['blocking_reasons']) if readiness['blocking_reasons'] else 'none'}.\n\n"
        "## Summary\n"
        f"- symbols={', '.join(payload['symbols'])}\n"
        f"- interval={payload['interval']}\n"
        f"- timezone={payload['timezone']}\n"
        f"- database_path={payload['database']['path']}\n"
        f"- database_exists={str(bool(payload['database'].get('database_exists', payload['database'].get('exists')))).lower()}\n"
        f"- dbbardata_table_exists={str(bool(payload['database'].get('dbbardata_table_exists'))).lower()}\n"
        f"- matches_verify_default_database_path={str(bool(payload['database'].get('matches_verify_default_database_path'))).lower()}\n"
        f"- check_okx_listing_metadata={str(bool(payload['check_okx_listing_metadata'])).lower()}\n"
        f"- recommended_next_download_window={readiness['recommended_next_download_window']}\n"
        f"- recommendation_reason={readiness['recommendation_reason']}\n"
        f"- can_enter_extended_trend_research={str(bool(readiness['can_enter_extended_trend_research'])).lower()}\n\n"
        "## Window Readiness\n"
        f"{render_window_summary_table(readiness)}\n\n"
        "## Main Missing Ranges\n"
        f"{render_missing_table(missing_rows)}\n\n"
        "## Symbol Coverage\n"
        f"{render_symbol_coverage_table(payload['symbol_windows'])}\n\n"
        "## Blocking Reasons\n"
        f"{blocking_lines}\n\n"
        "## Optional Warnings\n"
        f"{optional_warning_lines}\n\n"
        "## Decision Notes\n"
        "- This audit is data availability only; it is not a strategy return conclusion.\n"
        "- It does not download data, place orders, connect private trading, write API keys, or modify strategy logic.\n"
        "- If OKX listing metadata is not checked or lacks listTime, listing_before_window_start remains unknown as a warning.\n"
        "- Extended Trend Research requires the selected extended window to be fully covered locally before research starts.\n"
    )


def build_payload(
    symbols: list[str],
    windows: list[AuditWindow],
    interval: str,
    timezone_name: str,
    config_dir: Path,
    database_path: Path,
    output_dir: Path,
    check_okx_listing_metadata: bool,
    okx_timeout: float,
    fetcher: Callable[[str, float], dict[str, Any]] = fetch_okx_instrument,
) -> dict[str, Any]:
    """Run the audit and return the structured payload."""

    instruments = {symbol: inspect_instrument_metadata(config_dir, symbol) for symbol in symbols}
    listings = collect_listing_metadata(
        instruments=instruments,
        check_okx=check_okx_listing_metadata,
        timeout=okx_timeout,
        fetcher=fetcher,
    )
    database, coverages = query_database_coverage(
        database_path=database_path,
        symbols=symbols,
        instruments=instruments,
        windows=windows,
        interval=interval,
    )
    database["verify_default_database_path"] = str(DEFAULT_DATABASE_PATH)
    database["matches_verify_default_database_path"] = bool(
        database_path.resolve() == DEFAULT_DATABASE_PATH.resolve()
    )
    symbol_windows = build_symbol_window_records(
        symbols=symbols,
        instruments=instruments,
        listings=listings,
        coverages=coverages,
        windows=windows,
    )
    readiness = build_readiness(symbol_windows, windows)
    missing_rows = flatten_missing_ranges(symbol_windows)
    download_plan = build_download_plan(missing_rows)
    return {
        "output_dir": str(output_dir),
        "symbols": symbols,
        "windows": [
            {
                "window": window.label,
                "start": window.start_arg,
                "end": window.end_arg,
                "expected_count": expected_bar_count(window.history_range),
                "start_datetime": format_dt(window.history_range.start),
                "end_exclusive": format_dt(window.history_range.end_exclusive),
                "end_display": format_dt(window.history_range.end_display),
            }
            for window in windows
        ],
        "interval": interval,
        "timezone": timezone_name,
        "config_dir": str(config_dir),
        "database": database,
        "check_okx_listing_metadata": check_okx_listing_metadata,
        "instruments": instruments,
        "listing_metadata": listings,
        "symbol_windows": symbol_windows,
        "missing_ranges": missing_rows,
        "download_plan": download_plan,
        "readiness": readiness,
    }


def write_outputs(payload: dict[str, Any], output_dir: Path) -> dict[str, str]:
    """Write JSON, Markdown, and CSV outputs."""

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "extended_history_availability.json"
    report_path = output_dir / "extended_history_availability_report.md"
    missing_csv_path = output_dir / "extended_history_missing_ranges.csv"
    plan_csv_path = output_dir / "extended_history_download_plan.csv"
    outputs = {
        "json": str(json_path),
        "report": str(report_path),
        "missing_ranges_csv": str(missing_csv_path),
        "download_plan_csv": str(plan_csv_path),
    }
    payload["outputs"] = outputs
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    report_path.write_text(render_report(payload), encoding="utf-8")
    write_csv(
        missing_csv_path,
        payload["missing_ranges"],
        ["vt_symbol", "window", "interval", "start", "end", "missing_count", "suggested_download_command"],
    )
    write_csv(
        plan_csv_path,
        payload["download_plan"],
        ["priority", "vt_symbol", "window", "interval", "start", "end", "missing_count", "command"],
    )
    return outputs


def run_audit(
    symbols: list[str] | None = None,
    windows: list[AuditWindow] | None = None,
    interval: str = DEFAULT_INTERVAL,
    timezone_name: str = DEFAULT_TIMEZONE,
    config_dir: Path = DEFAULT_CONFIG_DIR,
    database_path: Path = DEFAULT_DATABASE_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    check_okx_listing_metadata: bool = False,
    okx_timeout: float = 10.0,
    fetcher: Callable[[str, float], dict[str, Any]] = fetch_okx_instrument,
) -> dict[str, Any]:
    """Run the extended history audit and write outputs."""

    interval_delta = interval_to_delta(interval)
    effective_symbols = list(DEFAULT_SYMBOLS) if symbols is None else list(symbols)
    effective_windows = (
        parse_windows(",".join(DEFAULT_WINDOWS), interval_delta, timezone_name)
        if windows is None
        else list(windows)
    )
    payload = build_payload(
        symbols=effective_symbols,
        windows=effective_windows,
        interval=interval,
        timezone_name=timezone_name,
        config_dir=config_dir,
        database_path=database_path,
        output_dir=output_dir,
        check_okx_listing_metadata=check_okx_listing_metadata,
        okx_timeout=okx_timeout,
        fetcher=fetcher,
    )
    write_outputs(payload, output_dir)
    return payload


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""

    args = parse_args(argv)
    interval_delta = interval_to_delta(args.interval)
    symbols = parse_symbol_list(args.symbols)
    windows = parse_windows(args.windows, interval_delta, args.timezone)
    payload = run_audit(
        symbols=symbols,
        windows=windows,
        interval=args.interval,
        timezone_name=args.timezone,
        config_dir=resolve_path(args.config_dir),
        database_path=resolve_path(args.database_path),
        output_dir=resolve_path(args.output_dir),
        check_okx_listing_metadata=args.check_okx_listing_metadata,
        okx_timeout=args.okx_timeout,
    )

    print("Extended history availability audit:")
    print(f"- output_json={payload['outputs']['json']}")
    print(f"- output_report={payload['outputs']['report']}")
    print(f"- missing_ranges_csv={payload['outputs']['missing_ranges_csv']}")
    print(f"- download_plan_csv={payload['outputs']['download_plan_csv']}")
    print(f"- recommended_next_download_window={payload['readiness']['recommended_next_download_window']}")
    print(
        "- can_enter_extended_trend_research="
        f"{str(bool(payload['readiness']['can_enter_extended_trend_research'])).lower()}"
    )
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
