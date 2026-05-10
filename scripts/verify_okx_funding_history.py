#!/usr/bin/env python3
"""Verify downloaded OKX funding-rate history CSV coverage."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from common_runtime import PROJECT_ROOT, ensure_headless_runtime, log_event, print_json_block, setup_logging, to_jsonable
from download_okx_funding_history import (
    DEFAULT_END,
    DEFAULT_INST_IDS_ARG,
    DEFAULT_OUTPUT_DIR as DEFAULT_FUNDING_DIR,
    DEFAULT_REPORTS_DIR,
    DEFAULT_START,
    FundingWindow,
    funding_csv_path,
    parse_funding_window,
    parse_inst_ids,
)


class FundingVerificationError(Exception):
    """Raised when funding verification arguments are invalid."""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description="Verify OKX funding history CSV coverage.")
    parser.add_argument("--funding-dir", default=str(DEFAULT_FUNDING_DIR))
    parser.add_argument("--inst-ids", default=DEFAULT_INST_IDS_ARG)
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--end", default=DEFAULT_END)
    parser.add_argument("--timezone", default="Asia/Shanghai")
    parser.add_argument("--output-dir", default=str(DEFAULT_REPORTS_DIR))
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def resolve_path(path_arg: str | Path) -> Path:
    """Resolve a path relative to project root."""

    path = Path(path_arg)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def parse_timestamp(row: dict[str, str]) -> datetime:
    """Parse one CSV row funding timestamp as UTC datetime."""

    raw_ms = str(row.get("funding_time") or "").strip()
    if raw_ms:
        return datetime.fromtimestamp(int(raw_ms) / 1000.0, tz=timezone.utc)
    raw_iso = str(row.get("funding_time_utc") or "").strip()
    if raw_iso.endswith("Z"):
        raw_iso = raw_iso[:-1] + "+00:00"
    parsed = datetime.fromisoformat(raw_iso)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def read_funding_timestamps(path: Path) -> list[datetime]:
    """Read funding timestamps from a CSV file."""

    timestamps: list[datetime] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            timestamps.append(parse_timestamp(row))
    return timestamps


def load_download_summary(output_dir: Path) -> dict[str, dict[str, Any]]:
    """Load downloader coverage metadata keyed by instrument id."""

    path = output_dir / "okx_funding_download_summary.json"
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    rows = payload.get("downloads") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        return {}
    return {
        str(row.get("inst_id")): row
        for row in rows
        if isinstance(row, dict) and row.get("inst_id") is not None
    }


def interval_hours(start: datetime, end: datetime) -> float:
    """Return interval hours between two datetimes."""

    return (end - start).total_seconds() / 3600.0


def large_gap_threshold(median_interval_hours: float | None) -> float:
    """Return a conservative large-gap threshold based on observed timestamps."""

    if median_interval_hours is None or median_interval_hours <= 0:
        return 12.0
    return max(12.0, min(24.0, median_interval_hours * 2.5))


def verify_inst_id(
    inst_id: str,
    funding_dir: Path,
    window: FundingWindow,
    download_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Verify one instrument funding CSV."""

    path = funding_csv_path(funding_dir, inst_id, window)
    metadata = download_metadata or {}
    endpoint_lower_bound_insufficient = bool(
        metadata.get("endpoint_history_limit_suspected")
        or metadata.get("completion_status") == "partial_endpoint_limited"
    )
    requested_start = window.start_utc.isoformat()
    requested_end = window.end_exclusive_utc.isoformat()
    if not path.exists():
        return {
            "inst_id": inst_id,
            "csv_path": str(path),
            "exists": False,
            "row_count": 0,
            "completion_status": "missing_csv",
            "incomplete_reason": ["csv_missing"],
            "requested_start": requested_start,
            "requested_end": requested_end,
            "first_available_time": None,
            "last_available_time": None,
            "missing_before_first_available": True,
            "missing_after_last_available": True,
            "endpoint_lower_bound_insufficient": endpoint_lower_bound_insufficient,
            "timestamp_strictly_increasing": False,
            "duplicate_timestamp_count": 0,
            "first_funding_time": None,
            "last_funding_time": None,
            "median_interval_hours": None,
            "max_interval_hours": None,
            "large_gap_threshold_hours": None,
            "missing_or_large_gap_count": 1,
            "large_gaps": [],
            "warnings": ["missing_funding_csv"],
            "coverage_complete": False,
        }

    timestamps = read_funding_timestamps(path)
    row_count = len(timestamps)
    if row_count == 0:
        return {
            "inst_id": inst_id,
            "csv_path": str(path),
            "exists": True,
            "row_count": 0,
            "completion_status": "empty_csv",
            "incomplete_reason": ["csv_empty"],
            "requested_start": requested_start,
            "requested_end": requested_end,
            "first_available_time": None,
            "last_available_time": None,
            "missing_before_first_available": True,
            "missing_after_last_available": True,
            "endpoint_lower_bound_insufficient": endpoint_lower_bound_insufficient,
            "timestamp_strictly_increasing": False,
            "duplicate_timestamp_count": 0,
            "first_funding_time": None,
            "last_funding_time": None,
            "median_interval_hours": None,
            "max_interval_hours": None,
            "large_gap_threshold_hours": None,
            "missing_or_large_gap_count": 1,
            "large_gaps": [],
            "warnings": ["empty_funding_csv"],
            "coverage_complete": False,
        }

    timestamp_ms = [int(item.timestamp() * 1000) for item in timestamps]
    duplicate_count = row_count - len(set(timestamp_ms))
    strictly_increasing = all(left < right for left, right in zip(timestamp_ms, timestamp_ms[1:]))
    sorted_unique = sorted(set(timestamps))
    intervals = [interval_hours(left, right) for left, right in zip(sorted_unique, sorted_unique[1:])]
    median_interval = float(statistics.median(intervals)) if intervals else None
    max_interval = float(max(intervals)) if intervals else None
    threshold = large_gap_threshold(median_interval)

    large_gaps = [
        {
            "start": left.isoformat(),
            "end": right.isoformat(),
            "interval_hours": interval_hours(left, right),
        }
        for left, right in zip(sorted_unique, sorted_unique[1:])
        if interval_hours(left, right) > threshold
    ]
    warnings: list[str] = []
    incomplete_reason: list[str] = []
    if not strictly_increasing:
        warnings.append("timestamp_not_strictly_increasing")
        incomplete_reason.append("timestamp_not_strictly_increasing")
    if duplicate_count:
        warnings.append(f"duplicate_timestamp_count={duplicate_count}")
        incomplete_reason.append("duplicate_timestamp")
    if large_gaps:
        warnings.append(f"large_gap_count={len(large_gaps)}")
        incomplete_reason.append("large_gap")

    boundary_gap_count = 0
    first_time = sorted_unique[0]
    last_time = sorted_unique[-1]
    start_boundary_hours = interval_hours(window.start_utc, first_time)
    end_boundary_hours = interval_hours(last_time, window.end_exclusive_utc)
    missing_before_first_available = start_boundary_hours > threshold
    missing_after_last_available = end_boundary_hours > threshold
    if start_boundary_hours > threshold:
        boundary_gap_count += 1
        warnings.append(f"start_boundary_gap_hours={start_boundary_hours:.2f}")
        incomplete_reason.append("missing_before_first_available")
    if end_boundary_hours > threshold:
        boundary_gap_count += 1
        warnings.append(f"end_boundary_gap_hours={end_boundary_hours:.2f}")
        incomplete_reason.append("missing_after_last_available")
    if endpoint_lower_bound_insufficient and missing_before_first_available:
        warnings.append("endpoint_lower_bound_insufficient")
        incomplete_reason.append("endpoint_lower_bound_insufficient")

    missing_or_large_gap_count = len(large_gaps) + boundary_gap_count
    coverage_complete = bool(row_count > 0 and strictly_increasing and duplicate_count == 0 and missing_or_large_gap_count == 0)
    completion_status = "complete" if coverage_complete else "incomplete"
    return {
        "inst_id": inst_id,
        "csv_path": str(path),
        "exists": True,
        "row_count": row_count,
        "completion_status": completion_status,
        "incomplete_reason": list(dict.fromkeys(incomplete_reason)),
        "requested_start": requested_start,
        "requested_end": requested_end,
        "first_available_time": first_time.isoformat(),
        "last_available_time": last_time.isoformat(),
        "missing_before_first_available": missing_before_first_available,
        "missing_after_last_available": missing_after_last_available,
        "endpoint_lower_bound_insufficient": endpoint_lower_bound_insufficient and missing_before_first_available,
        "timestamp_strictly_increasing": strictly_increasing,
        "duplicate_timestamp_count": duplicate_count,
        "first_funding_time": first_time.isoformat(),
        "last_funding_time": last_time.isoformat(),
        "median_interval_hours": median_interval,
        "max_interval_hours": max_interval,
        "large_gap_threshold_hours": threshold,
        "missing_or_large_gap_count": missing_or_large_gap_count,
        "large_gaps": large_gaps[:50],
        "warnings": warnings,
        "coverage_complete": coverage_complete,
    }


def render_verify_report(summary: dict[str, Any]) -> str:
    """Render Markdown verification report."""

    table_lines = [
        "| inst_id | status | row_count | requested_start | first_available | last_available | missing_before | gaps | incomplete_reason | warnings |",
        "|---|---|---:|---|---|---|---|---:|---|---|",
    ]
    for row in summary.get("results") or []:
        warnings = "; ".join(row.get("warnings") or [])
        reasons = "; ".join(row.get("incomplete_reason") or [])
        table_lines.append(
            f"| {row.get('inst_id')} | {row.get('completion_status')} | {row.get('row_count')} | "
            f"{row.get('requested_start') or ''} | {row.get('first_available_time') or ''} | "
            f"{row.get('last_available_time') or ''} | {str(bool(row.get('missing_before_first_available'))).lower()} | "
            f"{row.get('missing_or_large_gap_count')} | {reasons} | {warnings} |"
        )

    gap_lines = []
    for row in summary.get("results") or []:
        for gap in row.get("large_gaps") or []:
            gap_lines.append(f"- {row.get('inst_id')}: {gap.get('start')} -> {gap.get('end')} ({gap.get('interval_hours'):.2f}h)")
    gap_text = "\n".join(gap_lines) if gap_lines else "- 无"

    return (
        "# OKX Funding Verification Report\n\n"
        "## Scope\n"
        f"- start={summary.get('start')}\n"
        f"- end={summary.get('end')}\n"
        f"- timezone={summary.get('timezone')}\n"
        "- Funding interval is inferred from returned `funding_time`; fixed 8h intervals are not assumed.\n\n"
        "## Coverage Summary\n"
        f"- funding_data_complete={str(bool(summary.get('funding_data_complete'))).lower()}\n"
        f"- allow_partial={str(bool(summary.get('allow_partial'))).lower()}\n"
        f"- incomplete_reason={summary.get('incomplete_reason')}\n"
        f"- first_available_time={summary.get('first_available_time')}\n"
        f"- last_available_time={summary.get('last_available_time')}\n"
        f"- requested_start={summary.get('requested_start')}\n"
        f"- requested_end={summary.get('requested_end')}\n"
        f"- missing_before_first_available={str(bool(summary.get('missing_before_first_available'))).lower()}\n"
        f"- symbols_with_warnings={summary.get('symbols_with_warnings')}\n\n"
        f"{'**WARNING: partial funding data accepted for audit/reporting only; it is not decision-grade.**' if summary.get('allow_partial') and not summary.get('funding_data_complete') else ''}\n\n"
        f"{chr(10).join(table_lines)}\n\n"
        "## Large Gaps\n"
        f"{gap_text}\n"
    )


def format_optional(value: Any, digits: int = 4) -> str:
    """Format optional numeric values for Markdown."""

    if value is None:
        return ""
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def write_outputs(output_dir: Path, summary: dict[str, Any]) -> None:
    """Write verification JSON and Markdown reports."""

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "okx_funding_verify_summary.json").write_text(
        json.dumps(to_jsonable(summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "okx_funding_verify_report.md").write_text(render_verify_report(summary), encoding="utf-8")


def run_verify(
    inst_ids: list[str],
    funding_dir: Path,
    window: FundingWindow,
    output_dir: Path,
    *,
    allow_partial: bool = False,
) -> dict[str, Any]:
    """Run funding CSV verification."""

    download_metadata = load_download_summary(output_dir)
    results = [verify_inst_id(inst_id, funding_dir, window, download_metadata.get(inst_id)) for inst_id in inst_ids]
    symbols_with_warnings = [str(row["inst_id"]) for row in results if row.get("warnings")]
    first_times = [row.get("first_available_time") for row in results if row.get("first_available_time")]
    last_times = [row.get("last_available_time") for row in results if row.get("last_available_time")]
    incomplete_reasons: list[str] = []
    for row in results:
        incomplete_reasons.extend(str(item) for item in (row.get("incomplete_reason") or []))
    funding_data_complete = bool(results and all(row.get("coverage_complete") for row in results))
    summary = {
        "inst_ids": inst_ids,
        "funding_dir": str(funding_dir),
        "timezone": window.timezone_name,
        "start": window.start_arg,
        "end": window.end_arg,
        "start_utc": window.start_utc.isoformat(),
        "end_exclusive_utc": window.end_exclusive_utc.isoformat(),
        "requested_start": window.start_utc.isoformat(),
        "requested_end": window.end_exclusive_utc.isoformat(),
        "allow_partial": allow_partial,
        "funding_data_complete": funding_data_complete,
        "incomplete_reason": list(dict.fromkeys(incomplete_reasons)),
        "first_available_time": min(first_times) if first_times else None,
        "last_available_time": max(last_times) if last_times else None,
        "missing_before_first_available": bool(any(row.get("missing_before_first_available") for row in results)),
        "endpoint_lower_bound_insufficient": bool(any(row.get("endpoint_lower_bound_insufficient") for row in results)),
        "symbols_with_warnings": symbols_with_warnings,
        "symbols_with_large_gaps": [str(row["inst_id"]) for row in results if row.get("missing_or_large_gap_count", 0) > 0],
        "results": results,
    }
    write_outputs(output_dir, summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""

    args = parse_args(argv)
    ensure_headless_runtime()
    logger = setup_logging("verify_okx_funding_history", verbose=args.verbose)
    try:
        inst_ids = parse_inst_ids(args.inst_ids)
        window = parse_funding_window(args.start, args.end, args.timezone)
        summary = run_verify(
            inst_ids=inst_ids,
            funding_dir=resolve_path(args.funding_dir),
            window=window,
            output_dir=resolve_path(args.output_dir),
            allow_partial=bool(args.allow_partial),
        )
        print_json_block(
            "OKX funding verification summary:",
            {
                "funding_data_complete": summary.get("funding_data_complete"),
                "allow_partial": summary.get("allow_partial"),
                "incomplete_reason": summary.get("incomplete_reason"),
                "first_available_time": summary.get("first_available_time"),
                "last_available_time": summary.get("last_available_time"),
                "missing_before_first_available": summary.get("missing_before_first_available"),
                "symbols_with_warnings": summary.get("symbols_with_warnings"),
                "symbols_with_large_gaps": summary.get("symbols_with_large_gaps"),
                "output_dir": resolve_path(args.output_dir),
            },
        )
        return 0 if summary.get("funding_data_complete") or args.allow_partial else 2
    except Exception as exc:
        log_event(
            logger,
            logging.ERROR,
            "funding.verify.error",
            "Funding verification failed",
            error_class=exc.__class__.__name__,
            error_message=str(exc),
        )
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
