#!/usr/bin/env python3
"""Audit multi-symbol data readiness without downloading or trading."""

from __future__ import annotations

import argparse
import json
import math
import re
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from history_time_utils import (
    DEFAULT_TIMEZONE,
    HistoryRange,
    expected_bar_count,
    normalize_bar_datetime,
    parse_history_range,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_DIR = PROJECT_ROOT / "config" / "instruments"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "reports" / "research" / "multisymbol_readiness"
DEFAULT_DATABASE_PATH = PROJECT_ROOT / ".vntrader" / "database.db"
DEFAULT_MAKEFILE_PATH = PROJECT_ROOT / "Makefile"
DEFAULT_README_PATH = PROJECT_ROOT / "README.md"
REQUIRED_METADATA_FIELDS = [
    "vt_symbol",
    "symbol",
    "exchange",
    "okx_inst_id",
    "product",
    "size",
    "pricetick",
    "min_volume",
    "needs_okx_contract_metadata_refresh",
]
NUMERIC_METADATA_FIELDS = {"size", "pricetick", "min_volume"}
REQUIRED_TREND_V3_SYMBOLS = [
    "BTCUSDT_SWAP_OKX.GLOBAL",
    "ETHUSDT_SWAP_OKX.GLOBAL",
    "SOLUSDT_SWAP_OKX.GLOBAL",
    "LINKUSDT_SWAP_OKX.GLOBAL",
    "DOGEUSDT_SWAP_OKX.GLOBAL",
]
OPTIONAL_TREND_V3_SYMBOLS = [
    "BNBUSDT_SWAP_OKX.GLOBAL",
    "XRPUSDT_SWAP_OKX.GLOBAL",
]
DESIRED_TREND_V3_SYMBOLS = REQUIRED_TREND_V3_SYMBOLS + OPTIONAL_TREND_V3_SYMBOLS
MIN_TREND_V3_READY_SYMBOLS = 5
DEFAULT_START = "2025-01-01"
DEFAULT_END = "2026-03-31"
DEFAULT_INTERVAL = "1m"
BATCH_DOWNLOAD_TARGETS = {
    "download-history-all",
    "download-history-batch",
    "download-history-multi",
    "download-history-multisymbol",
    "download-multisymbol",
}
BATCH_VERIFY_TARGETS = {
    "verify-history-all",
    "verify-history-batch",
    "verify-history-multi",
    "verify-history-multisymbol",
    "verify-multisymbol",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description="Audit multi-symbol data readiness.")
    parser.add_argument("--config-dir", default=str(DEFAULT_CONFIG_DIR))
    parser.add_argument("--database-path", default=str(DEFAULT_DATABASE_PATH))
    parser.add_argument("--makefile-path", default=str(DEFAULT_MAKEFILE_PATH))
    parser.add_argument("--readme-path", default=str(DEFAULT_README_PATH))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--end", default=DEFAULT_END)
    parser.add_argument("--interval", default=DEFAULT_INTERVAL)
    parser.add_argument("--timezone", default=DEFAULT_TIMEZONE)
    parser.add_argument("--min-ready-symbols", type=int, default=MIN_TREND_V3_READY_SYMBOLS)
    parser.add_argument(
        "--required-symbols",
        default=",".join(REQUIRED_TREND_V3_SYMBOLS),
        help="Comma-separated vt_symbols that must be ready for Trend V3.",
    )
    parser.add_argument(
        "--optional-symbols",
        default=",".join(OPTIONAL_TREND_V3_SYMBOLS),
        help="Comma-separated vt_symbols that should be audited as optional warnings only.",
    )
    parser.add_argument("--json", action="store_true", help="Print JSON payload after writing outputs.")
    return parser.parse_args(argv)


def resolve_path(path_arg: str | Path, project_root: Path = PROJECT_ROOT) -> Path:
    """Resolve relative paths from the project root."""

    path = Path(path_arg)
    if not path.is_absolute():
        path = project_root / path
    return path


def json_default(value: Any) -> Any:
    """JSON fallback for path-like values."""

    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object is not JSON serializable: {type(value)!r}")


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
    return symbols


def split_vt_symbol_text(vt_symbol: str) -> tuple[str, str]:
    """Split a vt_symbol into database symbol and exchange strings."""

    symbol, separator, exchange = vt_symbol.partition(".")
    if not separator or not symbol or not exchange:
        raise ValueError(f"Invalid vt_symbol: {vt_symbol!r}")
    return symbol, exchange


def role_label(item: dict[str, Any]) -> str:
    """Return the audit role for one instrument."""

    if item.get("is_required"):
        return "required"
    if item.get("is_optional"):
        return "optional"
    return "other"


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


def format_local_datetime(value: datetime | None) -> str | None:
    """Format a local database bar timestamp without timezone noise."""

    if value is None:
        return None
    return value.replace(tzinfo=None).isoformat(sep=" ", timespec="seconds")


def read_json_file(path: Path) -> tuple[dict[str, Any], str | None]:
    """Read one JSON object and return a warning on failure."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {}, f"unable_to_read_json: {exc!r}"
    if not isinstance(payload, dict):
        return {}, "json_root_is_not_object"
    return payload, None


def finite_positive(value: Any) -> bool:
    """Return whether a value is a positive finite number."""

    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return bool(math.isfinite(number) and number > 0)


def required_field_missing(payload: dict[str, Any], field: str) -> bool:
    """Return whether a required metadata field is missing or empty."""

    if field not in payload:
        return True
    value = payload.get(field)
    if field in NUMERIC_METADATA_FIELDS:
        return not finite_positive(value)
    if value is None:
        return True
    return not str(value).strip()


def derive_vt_symbol_from_filename(path: Path) -> str | None:
    """Best-effort vt_symbol fallback for malformed local metadata."""

    stem = path.stem.upper()
    if not stem:
        return None
    return f"{stem}.GLOBAL"


def inspect_instrument_file(path: Path) -> dict[str, Any]:
    """Inspect one config/instruments JSON file."""

    payload, read_warning = read_json_file(path)
    vt_symbol = str(payload.get("vt_symbol") or derive_vt_symbol_from_filename(path) or "")
    okx_inst_id = str(payload.get("okx_inst_id") or "").strip()
    fallback_okx_inst_id = str(payload.get("instId") or payload.get("name") or "").strip()
    if okx_inst_id:
        okx_inst_id_source = "okx_inst_id"
    elif fallback_okx_inst_id:
        okx_inst_id_source = "fallback_available_but_not_canonical"
    else:
        okx_inst_id_source = "missing"

    canonical_missing: list[str] = []
    canonical_invalid: list[str] = []
    for field in ("okx_inst_id", "product"):
        if not str(payload.get(field) or "").strip():
            canonical_missing.append(field)
    needs_refresh_present = "needs_okx_contract_metadata_refresh" in payload
    if not needs_refresh_present:
        canonical_missing.append("needs_okx_contract_metadata_refresh")
    for field in ("size", "pricetick", "min_volume"):
        if not finite_positive(payload.get(field)):
            canonical_invalid.append(field)
    structural_missing = [
        field
        for field in ("vt_symbol", "symbol", "exchange")
        if required_field_missing(payload, field)
    ]
    needs_refresh = bool(payload.get("needs_okx_contract_metadata_refresh"))
    warnings: list[str] = []
    if read_warning:
        warnings.append(read_warning)
    for field in canonical_missing:
        warnings.append(f"missing_canonical_field: {field}")
    for field in canonical_invalid:
        warnings.append(f"invalid_canonical_field: {field}")
    for field in structural_missing:
        warnings.append(f"missing_required_field: {field}")
    if needs_refresh:
        warnings.append("needs_okx_contract_metadata_refresh")

    metadata_complete = bool(
        not read_warning
        and not canonical_missing
        and not canonical_invalid
        and not structural_missing
        and not needs_refresh
    )
    return {
        "path": str(path),
        "filename": path.name,
        "vt_symbol": vt_symbol,
        "symbol": payload.get("symbol") or vt_symbol.split(".", maxsplit=1)[0],
        "exchange": payload.get("exchange"),
        "okx_inst_id": okx_inst_id,
        "okx_inst_id_fallback": fallback_okx_inst_id,
        "okx_inst_id_source": okx_inst_id_source,
        "product": payload.get("product"),
        "size": payload.get("size"),
        "pricetick": payload.get("pricetick"),
        "min_volume": payload.get("min_volume"),
        "needs_okx_contract_metadata_refresh": needs_refresh,
        "missing_fields": canonical_missing + canonical_invalid + structural_missing,
        "missing_canonical_fields": canonical_missing,
        "invalid_canonical_fields": canonical_invalid,
        "metadata_complete": metadata_complete,
        "has_any_history": False,
        "short_history_exists": False,
        "required_coverage_ready": False,
        "history_ready": False,
        "can_backtest": False,
        "can_backtest_for_window": False,
        "is_required": False,
        "is_optional": False,
        "warnings": warnings,
    }


def scan_instruments(config_dir: Path) -> list[dict[str, Any]]:
    """Scan config/instruments/*.json."""

    if not config_dir.exists():
        return []
    return [
        inspect_instrument_file(path)
        for path in sorted(config_dir.glob("*.json"))
        if path.is_file()
    ]


def build_missing_instrument_record(vt_symbol: str) -> dict[str, Any]:
    """Build a non-ready audit row for a missing target instrument file."""

    try:
        symbol, exchange = split_vt_symbol_text(vt_symbol)
    except ValueError:
        symbol, exchange = "", ""
    return {
        "path": None,
        "filename": None,
        "vt_symbol": vt_symbol,
        "symbol": symbol,
        "exchange": exchange,
        "okx_inst_id": "",
        "okx_inst_id_fallback": "",
        "okx_inst_id_source": "missing",
        "product": None,
        "size": None,
        "pricetick": None,
        "min_volume": None,
        "needs_okx_contract_metadata_refresh": True,
        "missing_fields": list(REQUIRED_METADATA_FIELDS),
        "missing_canonical_fields": ["okx_inst_id", "product", "needs_okx_contract_metadata_refresh"],
        "invalid_canonical_fields": ["size", "pricetick", "min_volume"],
        "metadata_complete": False,
        "has_any_history": False,
        "short_history_exists": False,
        "required_coverage_ready": False,
        "history_ready": False,
        "can_backtest": False,
        "can_backtest_for_window": False,
        "is_required": False,
        "is_optional": False,
        "warnings": ["instrument_config_missing"],
    }


def prepare_target_instruments(
    instruments: list[dict[str, Any]],
    required_symbols: list[str],
    optional_symbols: list[str],
) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    """Annotate required/optional roles and add missing target placeholders."""

    required_set = set(required_symbols)
    optional_set = set(optional_symbols)
    by_symbol = {
        str(item.get("vt_symbol")): item
        for item in instruments
        if str(item.get("vt_symbol") or "").strip()
    }
    missing_required = [symbol for symbol in required_symbols if symbol not in by_symbol]
    missing_optional = [symbol for symbol in optional_symbols if symbol not in by_symbol]

    prepared = list(instruments)
    for vt_symbol in missing_required + missing_optional:
        item = build_missing_instrument_record(vt_symbol)
        by_symbol[vt_symbol] = item
        prepared.append(item)

    for item in prepared:
        vt_symbol = str(item.get("vt_symbol") or "")
        item["is_required"] = vt_symbol in required_set
        item["is_optional"] = vt_symbol in optional_set

    order = {symbol: index for index, symbol in enumerate(required_symbols + optional_symbols)}
    prepared.sort(
        key=lambda item: (
            order.get(str(item.get("vt_symbol") or ""), len(order)),
            str(item.get("vt_symbol") or ""),
        )
    )
    return prepared, missing_required, missing_optional


def parse_makefile_targets(makefile_path: Path) -> set[str]:
    """Return simple Makefile target names."""

    if not makefile_path.exists():
        return set()
    text = makefile_path.read_text(encoding="utf-8")
    targets: set[str] = set()
    for line in text.splitlines():
        match = re.match(r"^([A-Za-z0-9_.-]+)\s*:", line)
        if match:
            targets.add(match.group(1))
    return targets


def inspect_makefile(makefile_path: Path) -> dict[str, Any]:
    """Inspect Makefile support for multi-symbol helper targets."""

    targets = parse_makefile_targets(makefile_path)
    download_targets = sorted(targets & BATCH_DOWNLOAD_TARGETS)
    verify_targets = sorted(targets & BATCH_VERIFY_TARGETS)
    return {
        "path": str(makefile_path),
        "exists": makefile_path.exists(),
        "audit_multisymbol_target_exists": "audit-multisymbol" in targets,
        "batch_download_target_exists": bool(download_targets),
        "batch_download_targets": download_targets,
        "batch_verify_target_exists": bool(verify_targets),
        "batch_verify_targets": verify_targets,
    }


def inspect_source_capabilities(project_root: Path) -> dict[str, Any]:
    """Inspect current script surfaces without executing trading or downloads."""

    download_text = (project_root / "scripts" / "download_okx_history.py").read_text(encoding="utf-8")
    verify_text = (project_root / "scripts" / "verify_okx_history.py").read_text(encoding="utf-8")
    trend_text = (project_root / "scripts" / "research_trend_following_v2.py").read_text(encoding="utf-8")
    makefile_text = (project_root / "Makefile").read_text(encoding="utf-8") if (project_root / "Makefile").exists() else ""
    tests_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted((project_root / "tests").glob("test_*.py"))
        if path.is_file()
    ) if (project_root / "tests").exists() else ""
    return {
        "download_supports_vt_symbol": "--vt-symbol" in download_text and "extract_vt_symbol(args.vt_symbol)" in download_text,
        "verify_supports_vt_symbol": "--vt-symbol" in verify_text and "extract_vt_symbol(vt_symbol)" in verify_text,
        "research_trend_v2_single_symbol": "--vt-symbol" in trend_text and "--vt-symbols" not in trend_text,
        "research_trend_v2_filters_by_vt_symbol": "load_bar_data(symbol, exchange" in trend_text,
        "trend_v2_default_output_has_symbol_token": "trend_following_v2/$(VT_SYMBOL)" in makefile_text,
        "tests_reference_non_btc_symbol": bool(re.search(r"(ETH|SOL|LINK|DOGE|BNB|XRP)USDT_SWAP_OKX", tests_text)),
    }


def parse_database_datetime(
    value: Any,
    history_range: HistoryRange,
) -> datetime | None:
    """Parse and normalize one sqlite datetime value into the audit timezone."""

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
        parsed = parsed.replace(tzinfo=history_range.start.tzinfo)
    return normalize_bar_datetime(
        parsed,
        timezone_name=history_range.timezone_name,
        interval_delta=history_range.interval_delta,
    )


def build_missing_ranges_from_actual(
    actual_times: list[datetime],
    history_range: HistoryRange,
) -> list[dict[str, Any]]:
    """Build contiguous missing ranges from normalized actual timestamps."""

    expected = expected_bar_count(history_range)
    if expected <= 0:
        return []
    delta = history_range.interval_delta
    if not actual_times:
        return [
            {
                "start": format_local_datetime(history_range.start),
                "end": format_local_datetime(history_range.end_exclusive - delta),
                "missing_count": expected,
            }
        ]

    ranges: list[dict[str, Any]] = []

    def append_range(start: datetime, end: datetime) -> None:
        missing_count = int((end - start) / delta) + 1
        if missing_count > 0:
            ranges.append(
                {
                    "start": format_local_datetime(start),
                    "end": format_local_datetime(end),
                    "missing_count": missing_count,
                }
            )

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


def apply_coverage_to_instrument(instrument: dict[str, Any], coverage: dict[str, Any]) -> None:
    """Store coverage both nested and in the top-level JSON audit row."""

    instrument["database"] = coverage
    instrument["history_coverage"] = coverage
    instrument["has_any_history"] = bool(coverage.get("has_any_history") or coverage.get("has_history"))
    instrument["short_history_exists"] = bool(coverage.get("short_history_exists"))
    instrument["required_coverage_ready"] = bool(coverage.get("required_coverage_ready"))
    instrument["history_ready"] = bool(coverage.get("required_coverage_ready"))
    instrument["total_count"] = int(coverage.get("total_count") or 0)
    instrument["expected_count"] = int(coverage.get("expected_count") or 0)
    instrument["missing_count"] = int(coverage.get("missing_count") or 0)
    instrument["gap_count"] = int(coverage.get("gap_count") or 0)
    instrument["can_backtest"] = bool(
        instrument.get("metadata_complete") and instrument.get("required_coverage_ready")
    )
    instrument["can_backtest_for_window"] = bool(instrument["can_backtest"])


def query_database_coverage(
    database_path: Path,
    instruments: list[dict[str, Any]],
    interval: str,
    history_range: HistoryRange,
) -> dict[str, Any]:
    """Query local sqlite coverage by symbol/exchange/interval and requested range."""

    connection: sqlite3.Connection | None = None
    expected_count = expected_bar_count(history_range)
    if not database_path.exists():
        warning = f"database_not_found: {database_path}"
        for instrument in instruments:
            coverage = {
                "exists": False,
                "warning": warning,
                "interval": interval,
                "expected_count": expected_count,
                "total_count": 0,
                "missing_count": expected_count,
                "gap_count": 1 if expected_count > 0 else 0,
                "first_dt": None,
                "last_dt": None,
                "has_any_history": False,
                "short_history_exists": False,
                "required_coverage_ready": False,
            }
            apply_coverage_to_instrument(instrument, coverage)
            instrument.setdefault("warnings", []).append(warning)
        return {"path": str(database_path), "exists": False, "warning": warning}

    try:
        connection = sqlite3.connect(database_path)
        table_exists = bool(
            connection.execute(
                "select 1 from sqlite_master where type='table' and name='dbbardata'"
            ).fetchone()
        )
        if not table_exists:
            warning = "dbbardata_table_not_found"
            for instrument in instruments:
                coverage = {
                    "exists": True,
                    "warning": warning,
                    "interval": interval,
                    "expected_count": expected_count,
                    "total_count": 0,
                    "missing_count": expected_count,
                    "gap_count": 1 if expected_count > 0 else 0,
                    "first_dt": None,
                    "last_dt": None,
                    "has_any_history": False,
                    "short_history_exists": False,
                    "required_coverage_ready": False,
                }
                apply_coverage_to_instrument(instrument, coverage)
                instrument.setdefault("warnings", []).append(warning)
            return {"path": str(database_path), "exists": True, "warning": warning}

        query_start = format_local_datetime(history_range.start)
        query_end = format_local_datetime(history_range.end_exclusive)
        for instrument in instruments:
            symbol = str(instrument.get("symbol") or "").strip()
            exchange = str(instrument.get("exchange") or "").strip()
            if not symbol or not exchange:
                warning = "cannot_query_database_without_symbol_and_exchange"
                coverage = {
                    "exists": True,
                    "warning": warning,
                    "interval": interval,
                    "expected_count": expected_count,
                    "total_count": 0,
                    "missing_count": expected_count,
                    "gap_count": 1 if expected_count > 0 else 0,
                    "first_dt": None,
                    "last_dt": None,
                    "has_any_history": False,
                    "short_history_exists": False,
                    "required_coverage_ready": False,
                }
                apply_coverage_to_instrument(instrument, coverage)
                instrument.setdefault("warnings", []).append(warning)
                continue
            any_row = connection.execute(
                (
                    "select count(distinct datetime), min(datetime), max(datetime) "
                    "from dbbardata where symbol = ? and exchange = ? and interval = ?"
                ),
                (symbol, exchange, interval),
            ).fetchone()
            any_count = int(any_row[0] or 0)
            any_first_dt = any_row[1]
            any_last_dt = any_row[2]
            has_any_history = bool(any_count > 0 and any_first_dt and any_last_dt)
            rows = connection.execute(
                (
                    "select datetime from dbbardata "
                    "where symbol = ? and exchange = ? and interval = ? "
                    "and datetime >= ? and datetime < ? "
                    "order by datetime"
                ),
                (symbol, exchange, interval, query_start, query_end),
            ).fetchall()
            actual_set = {
                parsed
                for row in rows
                if (parsed := parse_database_datetime(row[0], history_range)) is not None
                and history_range.start <= parsed < history_range.end_exclusive
            }
            actual_times = sorted(actual_set)
            total_count = len(actual_times)
            missing_count = max(expected_count - total_count, 0)
            missing_ranges = build_missing_ranges_from_actual(actual_times, history_range)
            gap_count = len(missing_ranges)
            largest_gap = max(missing_ranges, key=lambda item: int(item["missing_count"]), default=None)
            required_coverage_ready = bool(
                total_count == expected_count
                and missing_count == 0
                and gap_count == 0
                and expected_count > 0
            )
            coverage = {
                "exists": True,
                "symbol": symbol,
                "exchange": exchange,
                "interval": interval,
                "start": format_local_datetime(history_range.start),
                "end_exclusive": format_local_datetime(history_range.end_exclusive),
                "end_display": format_local_datetime(history_range.end_display),
                "expected_count": expected_count,
                "total_count": total_count,
                "bar_count": total_count,
                "missing_count": missing_count,
                "gap_count": gap_count,
                "first_dt": format_local_datetime(actual_times[0]) if actual_times else None,
                "last_dt": format_local_datetime(actual_times[-1]) if actual_times else None,
                "largest_gap": largest_gap,
                "missing_ranges": missing_ranges[:20],
                "has_history": has_any_history,
                "has_any_history": has_any_history,
                "short_history_exists": bool(has_any_history and not required_coverage_ready),
                "required_coverage_ready": required_coverage_ready,
            }
            if not has_any_history:
                warning = f"no_local_sqlite_history_for_{symbol}.{exchange}_{interval}"
                coverage["warning"] = warning
                instrument.setdefault("warnings", []).append(warning)
            elif not required_coverage_ready:
                warning = (
                    "incomplete_required_history_coverage: "
                    f"expected={expected_count} total={total_count} "
                    f"missing={missing_count} gaps={gap_count}"
                )
                coverage["warning"] = warning
                instrument.setdefault("warnings", []).append(warning)
            apply_coverage_to_instrument(instrument, coverage)
        return {"path": str(database_path), "exists": True, "table": "dbbardata"}
    except Exception as exc:
        warning = f"database_query_failed: {exc!r}"
        for instrument in instruments:
            coverage = {
                "exists": True,
                "warning": warning,
                "interval": interval,
                "expected_count": expected_count,
                "total_count": 0,
                "missing_count": expected_count,
                "gap_count": 1 if expected_count > 0 else 0,
                "first_dt": None,
                "last_dt": None,
                "has_any_history": False,
                "short_history_exists": False,
                "required_coverage_ready": False,
            }
            apply_coverage_to_instrument(instrument, coverage)
            instrument.setdefault("warnings", []).append(warning)
        return {"path": str(database_path), "exists": True, "warning": warning}
    finally:
        if connection is not None:
            connection.close()


def build_readiness(
    instruments: list[dict[str, Any]],
    makefile: dict[str, Any],
    missing_required_symbols: list[str],
    missing_optional_symbols: list[str],
    required_symbols: list[str],
    optional_symbols: list[str],
    min_ready_symbols: int,
) -> dict[str, Any]:
    """Build high-level Trend V3 readiness decision."""

    blocking_reasons: list[str] = []
    required_warnings: list[str] = []
    optional_warnings: list[str] = []
    target_set = set(required_symbols + optional_symbols)
    ready_symbols = [
        item["vt_symbol"]
        for item in instruments
        if item.get("vt_symbol") in target_set
        and item.get("metadata_complete")
        and item.get("required_coverage_ready")
    ]
    by_symbol = {
        str(item.get("vt_symbol")): item
        for item in instruments
        if str(item.get("vt_symbol") or "").strip()
    }
    ready_set = set(ready_symbols)
    missing_required_ready = [symbol for symbol in required_symbols if symbol not in ready_set]

    if missing_required_symbols:
        required_warnings.append(f"missing_required_instrument_files: {', '.join(missing_required_symbols)}")
    if missing_optional_symbols:
        optional_warnings.append(f"missing_optional_instrument_files: {', '.join(missing_optional_symbols)}")

    required_metadata_incomplete: list[str] = []
    required_missing_history: list[str] = []
    required_incomplete_coverage: list[str] = []
    optional_metadata_incomplete: list[str] = []
    optional_missing_history: list[str] = []
    optional_incomplete_coverage: list[str] = []

    for vt_symbol in required_symbols:
        item = by_symbol.get(vt_symbol)
        if not item:
            required_metadata_incomplete.append(vt_symbol)
            required_missing_history.append(vt_symbol)
            continue
        if not item.get("metadata_complete"):
            required_metadata_incomplete.append(vt_symbol)
        if not item.get("has_any_history"):
            required_missing_history.append(vt_symbol)
        elif not item.get("required_coverage_ready"):
            required_incomplete_coverage.append(vt_symbol)

    for vt_symbol in optional_symbols:
        item = by_symbol.get(vt_symbol)
        if not item:
            optional_metadata_incomplete.append(vt_symbol)
            optional_missing_history.append(vt_symbol)
            continue
        if not item.get("metadata_complete"):
            optional_metadata_incomplete.append(vt_symbol)
        if not item.get("has_any_history"):
            optional_missing_history.append(vt_symbol)
        elif not item.get("required_coverage_ready"):
            optional_incomplete_coverage.append(vt_symbol)

    if required_metadata_incomplete:
        required_warnings.append(
            f"incomplete_required_instrument_metadata: {', '.join(required_metadata_incomplete)}"
        )
    if required_missing_history:
        required_warnings.append(f"missing_required_local_sqlite_history: {', '.join(required_missing_history)}")
    if required_incomplete_coverage:
        required_warnings.append(
            f"incomplete_required_history_coverage: {', '.join(required_incomplete_coverage)}"
        )
    if optional_metadata_incomplete:
        optional_warnings.append(
            f"incomplete_optional_instrument_metadata: {', '.join(optional_metadata_incomplete)}"
        )
    if optional_missing_history:
        optional_warnings.append(f"missing_optional_local_sqlite_history: {', '.join(optional_missing_history)}")
    if optional_incomplete_coverage:
        optional_warnings.append(
            f"incomplete_optional_history_coverage: {', '.join(optional_incomplete_coverage)}"
        )

    if not makefile.get("batch_download_target_exists"):
        blocking_reasons.append("makefile_has_no_batch_download_target")
    if not makefile.get("batch_verify_target_exists"):
        blocking_reasons.append("makefile_has_no_batch_verify_target")
    if len(ready_symbols) < min_ready_symbols:
        blocking_reasons.append(
            f"ready_symbols_below_minimum: ready={len(ready_symbols)} minimum={min_ready_symbols}"
        )
    if missing_required_ready:
        blocking_reasons.append(f"required_ready_symbols_missing: {', '.join(missing_required_ready)}")
    if required_metadata_incomplete:
        blocking_reasons.append(
            f"incomplete_required_instrument_metadata: {', '.join(required_metadata_incomplete)}"
        )
    if required_missing_history:
        blocking_reasons.append(
            f"missing_required_local_sqlite_history: {', '.join(required_missing_history)}"
        )
    if required_incomplete_coverage:
        blocking_reasons.append(
            f"incomplete_required_history_coverage: {', '.join(required_incomplete_coverage)}"
        )

    return {
        "ready_symbols": ready_symbols,
        "can_enter_trend_v3": bool(not blocking_reasons),
        "blocking_reasons": blocking_reasons,
        "required_warnings": required_warnings,
        "optional_warnings": optional_warnings,
        "data_warnings": required_warnings + optional_warnings,
        "required_symbols": required_symbols,
        "optional_symbols": optional_symbols,
        "required_ready_symbols": required_symbols,
        "min_ready_symbols": min_ready_symbols,
    }


def render_instrument_table(instruments: list[dict[str, Any]]) -> str:
    """Render instrument status rows."""

    if not instruments:
        return (
            "| vt_symbol | role | okx_inst_id | okx_inst_id_source | product | metadata_complete | "
            "has_any_history | required_coverage_ready | total_count | expected_count | "
            "missing_count | gap_count | can_backtest_for_window | warning |\n"
            "|---|---|---|---|---|---|---|---|---:|---:|---:|---:|---|---|\n"
        )
    lines = [
        "| vt_symbol | role | okx_inst_id | okx_inst_id_source | product | metadata_complete | has_any_history | required_coverage_ready | total_count | expected_count | missing_count | gap_count | can_backtest_for_window | warning |",
        "|---|---|---|---|---|---|---|---|---:|---:|---:|---:|---|---|",
    ]
    for item in instruments:
        warning = "; ".join(item.get("warnings") or []) or "-"
        lines.append(
            "| {vt_symbol} | {role} | {okx_inst_id} | {source} | {product} | {metadata} | {has_any} | {coverage_ready} | {total_count} | {expected_count} | {missing_count} | {gap_count} | {backtest} | {warning} |".format(
                vt_symbol=item.get("vt_symbol") or "",
                role=role_label(item),
                okx_inst_id=item.get("okx_inst_id") or "",
                source=item.get("okx_inst_id_source") or "",
                product=item.get("product") or "",
                metadata=str(bool(item.get("metadata_complete"))).lower(),
                has_any=str(bool(item.get("has_any_history"))).lower(),
                coverage_ready=str(bool(item.get("required_coverage_ready"))).lower(),
                total_count=item.get("total_count", ""),
                expected_count=item.get("expected_count", ""),
                missing_count=item.get("missing_count", ""),
                gap_count=item.get("gap_count", ""),
                backtest=str(bool(item.get("can_backtest_for_window"))).lower(),
                warning=warning.replace("|", "/"),
            )
        )
    return "\n".join(lines)


def render_report(payload: dict[str, Any]) -> str:
    """Render Markdown report."""

    readiness = payload["trend_v3_readiness"]
    window = payload["coverage_window"]
    configured = payload.get("configured_instrument_files") or payload["configured_vt_symbols"]
    ready = readiness["ready_symbols"]
    blocking = readiness["blocking_reasons"]
    optional_warnings = readiness.get("optional_warnings") or []
    required_instruments = [item for item in payload["instruments"] if item.get("is_required")]
    optional_instruments = [item for item in payload["instruments"] if item.get("is_optional")]
    other_instruments = [
        item for item in payload["instruments"] if not item.get("is_required") and not item.get("is_optional")
    ]
    blocking_lines = "\n".join(f"- {item}" for item in blocking) if blocking else "- none"
    optional_warning_lines = (
        "\n".join(f"- {item}" for item in optional_warnings) if optional_warnings else "- none"
    )
    other_section = (
        "\n## Other Instruments\n"
        f"{render_instrument_table(other_instruments)}\n"
        if other_instruments
        else ""
    )
    return (
        "# Multi-symbol Data Readiness Audit\n\n"
        "## Summary\n"
        f"- configured_instruments={len(configured)}\n"
        f"- ready_symbols={len(ready)}\n"
        f"- can_enter_trend_v3={str(bool(readiness['can_enter_trend_v3'])).lower()}\n"
        f"- required_symbols={', '.join(readiness['required_symbols'])}\n"
        f"- optional_symbols={', '.join(readiness['optional_symbols'])}\n"
        f"- min_ready_symbols={readiness['min_ready_symbols']}\n"
        f"- coverage_window={window['start']} to {window['end']} ({window['timezone']}, {window['interval']})\n"
        f"- full_trend_v3_window={str(bool(window['is_default_trend_v3_window'])).lower()}\n"
        f"- database_path={payload['database']['path']}\n\n"
        "## Coverage Window Note\n"
        "The readiness counts below apply to this audit coverage window. A short-window audit validates the download "
        "chain for that window only and does not prove full Trend V3 readiness unless `full_trend_v3_window=true`.\n\n"
        "## Required Instruments\n"
        f"{render_instrument_table(required_instruments)}\n\n"
        "## Optional Instruments\n"
        f"{render_instrument_table(optional_instruments)}\n"
        f"{other_section}\n"
        "## Makefile\n"
        f"- audit_multisymbol_target_exists={str(bool(payload['makefile']['audit_multisymbol_target_exists'])).lower()}\n"
        f"- batch_download_target_exists={str(bool(payload['makefile']['batch_download_target_exists'])).lower()}\n"
        f"- batch_verify_target_exists={str(bool(payload['makefile']['batch_verify_target_exists'])).lower()}\n\n"
        "## Source Capability\n"
        f"- download_supports_vt_symbol={str(bool(payload['capabilities']['download_supports_vt_symbol'])).lower()}\n"
        f"- verify_supports_vt_symbol={str(bool(payload['capabilities']['verify_supports_vt_symbol'])).lower()}\n"
        f"- research_trend_v2_single_symbol={str(bool(payload['capabilities']['research_trend_v2_single_symbol'])).lower()}\n"
        f"- research_trend_v2_filters_by_vt_symbol={str(bool(payload['capabilities']['research_trend_v2_filters_by_vt_symbol'])).lower()}\n"
        f"- trend_v2_default_output_has_symbol_token={str(bool(payload['capabilities']['trend_v2_default_output_has_symbol_token'])).lower()}\n"
        f"- tests_reference_non_btc_symbol={str(bool(payload['capabilities']['tests_reference_non_btc_symbol'])).lower()}\n\n"
        "## Blocking Reasons\n"
        f"{blocking_lines}\n\n"
        "## Optional Warnings\n"
        f"{optional_warning_lines}\n\n"
        "## Decision\n"
        "Do not enter full Trend V3 until metadata is canonical, the default full coverage window is complete for the "
        "required symbols, at least the minimum number of symbols are ready, and batch download plus batch verify "
        "helpers exist. Optional instruments are reported as warnings and do not block first-batch Trend V3 readiness. "
        "The current audit does not download data and does not change strategy trading logic.\n"
    )


def run_audit(
    config_dir: Path = DEFAULT_CONFIG_DIR,
    database_path: Path = DEFAULT_DATABASE_PATH,
    makefile_path: Path = DEFAULT_MAKEFILE_PATH,
    readme_path: Path = DEFAULT_README_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    start: str = DEFAULT_START,
    end: str = DEFAULT_END,
    interval: str = DEFAULT_INTERVAL,
    timezone_name: str = DEFAULT_TIMEZONE,
    min_ready_symbols: int = MIN_TREND_V3_READY_SYMBOLS,
    required_symbols: list[str] | None = None,
    optional_symbols: list[str] | None = None,
    project_root: Path = PROJECT_ROOT,
) -> dict[str, Any]:
    """Run the audit and write JSON/Markdown outputs."""

    interval_delta = interval_to_delta(interval)
    history_range = parse_history_range(
        start_arg=start,
        end_arg=end,
        interval_delta=interval_delta,
        timezone_name=timezone_name,
    )
    required_symbols = list(REQUIRED_TREND_V3_SYMBOLS) if required_symbols is None else list(required_symbols)
    optional_symbols = list(OPTIONAL_TREND_V3_SYMBOLS) if optional_symbols is None else list(optional_symbols)
    instruments, missing_required, missing_optional = prepare_target_instruments(
        scan_instruments(config_dir),
        required_symbols=required_symbols,
        optional_symbols=optional_symbols,
    )
    database = query_database_coverage(database_path, instruments, interval, history_range)
    database["verify_default_database_path"] = str(project_root / ".vntrader" / "database.db")
    database["matches_verify_default_database_path"] = bool(
        database_path.resolve() == (project_root / ".vntrader" / "database.db").resolve()
    )
    audited = sorted(item["vt_symbol"] for item in instruments if item.get("vt_symbol"))
    configured_files = sorted(
        item["vt_symbol"]
        for item in instruments
        if item.get("vt_symbol") and "instrument_config_missing" not in set(item.get("warnings") or [])
    )
    missing_desired = missing_required + missing_optional
    makefile = inspect_makefile(makefile_path)
    capabilities = inspect_source_capabilities(project_root)
    readme = {
        "path": str(readme_path),
        "exists": readme_path.exists(),
        "mentions_multisymbol_readiness": (
            "Multi-symbol Data Readiness" in readme_path.read_text(encoding="utf-8")
            if readme_path.exists()
            else False
        ),
    }
    payload = {
        "config_dir": str(config_dir),
        "output_dir": str(output_dir),
        "coverage_window": {
            "start": start,
            "end": end,
            "interval": interval,
            "timezone": timezone_name,
            "start_datetime": format_local_datetime(history_range.start),
            "end_exclusive": format_local_datetime(history_range.end_exclusive),
            "end_display": format_local_datetime(history_range.end_display),
            "expected_count": expected_bar_count(history_range),
            "is_default_trend_v3_window": bool(
                start == DEFAULT_START
                and end == DEFAULT_END
                and interval == DEFAULT_INTERVAL
                and timezone_name == DEFAULT_TIMEZONE
            ),
        },
        "required_metadata_fields": REQUIRED_METADATA_FIELDS,
        "required_symbols": required_symbols,
        "optional_symbols": optional_symbols,
        "desired_trend_v3_symbols": required_symbols + optional_symbols,
        "configured_vt_symbols": configured_files,
        "audited_vt_symbols": audited,
        "configured_instrument_files": configured_files,
        "missing_desired_instrument_files": missing_desired,
        "missing_required_instrument_files": missing_required,
        "missing_optional_instrument_files": missing_optional,
        "instruments": instruments,
        "database": database,
        "makefile": makefile,
        "readme": readme,
        "capabilities": capabilities,
    }
    payload["trend_v3_readiness"] = build_readiness(
        instruments=instruments,
        makefile=makefile,
        missing_required_symbols=missing_required,
        missing_optional_symbols=missing_optional,
        required_symbols=required_symbols,
        optional_symbols=optional_symbols,
        min_ready_symbols=min_ready_symbols,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "multisymbol_readiness.json"
    report_path = output_dir / "multisymbol_readiness_report.md"
    payload["outputs"] = {"json": str(json_path), "report": str(report_path)}
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=json_default), encoding="utf-8")
    report_path.write_text(render_report(payload), encoding="utf-8")
    return payload


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""

    args = parse_args(argv)
    payload = run_audit(
        config_dir=resolve_path(args.config_dir),
        database_path=resolve_path(args.database_path),
        makefile_path=resolve_path(args.makefile_path),
        readme_path=resolve_path(args.readme_path),
        output_dir=resolve_path(args.output_dir),
        start=args.start,
        end=args.end,
        interval=args.interval,
        timezone_name=args.timezone,
        min_ready_symbols=args.min_ready_symbols,
        required_symbols=parse_symbol_list(args.required_symbols),
        optional_symbols=parse_symbol_list(args.optional_symbols),
    )
    print("Multi-symbol readiness audit:")
    print(f"- output_json={payload['outputs']['json']}")
    print(f"- output_report={payload['outputs']['report']}")
    print(
        "- coverage_window="
        f"{payload['coverage_window']['start']}..{payload['coverage_window']['end']} "
        f"{payload['coverage_window']['interval']} {payload['coverage_window']['timezone']}"
    )
    print(f"- ready_symbols={len(payload['trend_v3_readiness']['ready_symbols'])}")
    print(f"- can_enter_trend_v3={str(bool(payload['trend_v3_readiness']['can_enter_trend_v3'])).lower()}")
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2, default=json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
