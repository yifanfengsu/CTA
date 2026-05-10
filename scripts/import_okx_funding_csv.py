#!/usr/bin/env python3
"""Import manually downloaded OKX historical funding-rate CSV files."""

from __future__ import annotations

import argparse
import csv
import json
from decimal import Decimal, InvalidOperation
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from common_runtime import PROJECT_ROOT, ensure_headless_runtime, print_json_block, setup_logging, to_jsonable
from download_okx_funding_history import (
    CSV_COLUMNS,
    DEFAULT_END,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_REPORTS_DIR,
    DEFAULT_START,
    FundingDownloadError,
    funding_csv_path,
    parse_funding_window,
    resolve_path as resolve_download_path,
    write_funding_csv,
)
from history_time_utils import DEFAULT_TIMEZONE, resolve_timezone


TIME_COLUMNS = [
    "funding_time",
    "fundingtime",
    "funding_time_utc",
    "funding time",
    "timestamp",
    "time",
    "ts",
    "settlement_time",
]
RATE_COLUMNS = [
    "funding_rate",
    "fundingrate",
    "realized_rate",
    "realizedrate",
    "rate",
    "funding rate",
]
INST_COLUMNS = ["inst_id", "instid", "instrument", "instrument_id", "symbol"]
IMPORT_REPORT_NAME = "okx_funding_import_report.md"
IMPORT_SUMMARY_NAME = "okx_funding_import_summary.json"


class FundingCsvImportError(Exception):
    """Raised when a manual funding CSV cannot be imported."""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description="Import a manually downloaded OKX historical funding CSV.")
    parser.add_argument("--input")
    parser.add_argument("--inputs")
    parser.add_argument("--input-dir")
    parser.add_argument("--inst-id", required=True)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--reports-dir", default=str(DEFAULT_REPORTS_DIR))
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--end", default=DEFAULT_END)
    parser.add_argument("--timezone", default=DEFAULT_TIMEZONE)
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--append", action="store_true")
    mode_group.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def resolve_path(path_arg: str | Path) -> Path:
    """Resolve a path relative to project root."""

    path = Path(path_arg)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def normalize_header(name: str) -> str:
    """Normalize CSV column names for flexible imported files."""

    return str(name or "").strip().lower().replace("-", "_")


def find_column(fieldnames: list[str], candidates: list[str]) -> str | None:
    """Find a source CSV column by normalized aliases."""

    normalized = {normalize_header(name): name for name in fieldnames}
    for candidate in candidates:
        key = normalize_header(candidate)
        if key in normalized:
            return normalized[key]
    return None


def normalize_identifier(value: str) -> str:
    """Normalize an instrument/file identifier for loose filename matching."""

    return "".join(char.lower() for char in str(value or "") if char.isalnum())


def parse_inputs(raw_inputs: str | None) -> list[str]:
    """Parse comma-separated input paths."""

    if not raw_inputs:
        return []
    return [item.strip() for item in raw_inputs.split(",") if item.strip()]


def discover_input_paths(
    *,
    input_path: str | Path | None,
    inputs: str | None,
    input_dir: str | Path | None,
    inst_id: str,
) -> list[Path]:
    """Resolve requested input CSV paths from --input/--inputs/--input-dir."""

    paths: list[Path] = []
    if input_path:
        paths.append(resolve_path(input_path))
    paths.extend(resolve_path(item) for item in parse_inputs(inputs))
    if input_dir:
        directory = resolve_path(input_dir)
        if not directory.exists():
            raise FundingCsvImportError(f"input directory does not exist: {directory}")
        if not directory.is_dir():
            raise FundingCsvImportError(f"input-dir is not a directory: {directory}")
        target = normalize_identifier(inst_id)
        matched = [
            item
            for item in sorted(directory.iterdir())
            if item.is_file()
            and item.suffix.lower() == ".csv"
            and target in normalize_identifier(item.name)
        ]
        paths.extend(matched)
    deduped = list(dict.fromkeys(paths))
    if not deduped:
        raise FundingCsvImportError("at least one of --input, --inputs, or --input-dir is required")
    missing = [str(path) for path in deduped if not path.exists()]
    if missing:
        raise FundingCsvImportError(f"input CSV does not exist: {missing}")
    non_files = [str(path) for path in deduped if not path.is_file()]
    if non_files:
        raise FundingCsvImportError(f"input path is not a file: {non_files}")
    return deduped


def normalize_funding_rate(value: str) -> str:
    """Normalize a funding rate string without changing its value."""

    text = str(value or "").strip()
    if not text:
        return ""
    try:
        decimal_value = Decimal(text)
    except InvalidOperation as exc:
        raise FundingCsvImportError(f"cannot parse funding rate: {value}") from exc
    return format(decimal_value.normalize(), "f")


def parse_time_value(value: str, timezone_name: str) -> datetime:
    """Parse millisecond/second epoch or ISO timestamp into UTC."""

    text = str(value or "").strip()
    if not text:
        raise FundingCsvImportError("empty funding timestamp")
    if text.isdigit():
        number = int(text)
        if number > 10_000_000_000:
            return datetime.fromtimestamp(number / 1000.0, tz=timezone.utc)
        return datetime.fromtimestamp(number, tz=timezone.utc)
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise FundingCsvImportError(f"cannot parse funding timestamp: {value}") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=resolve_timezone(timezone_name))
    return parsed.astimezone(timezone.utc)


def row_inst_id(row: dict[str, str], inst_column: str | None) -> str:
    """Return optional input row instrument id."""

    if inst_column is None:
        return ""
    return str(row.get(inst_column) or "").strip()


def import_rows(input_path: Path, inst_id: str, timezone_name: str) -> list[dict[str, Any]]:
    """Read and normalize funding rows from one manual CSV."""

    with input_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        if not fieldnames:
            raise FundingCsvImportError(f"input CSV has no header: {input_path}")
        time_column = find_column(fieldnames, TIME_COLUMNS)
        rate_column = find_column(fieldnames, RATE_COLUMNS)
        inst_column = find_column(fieldnames, INST_COLUMNS)
        if time_column is None:
            raise FundingCsvImportError(
                f"input CSV missing funding time column: {input_path}; accepted aliases={TIME_COLUMNS}"
            )
        if rate_column is None:
            raise FundingCsvImportError(
                f"input CSV missing funding rate column: {input_path}; accepted aliases={RATE_COLUMNS}"
            )

        tz = resolve_timezone(timezone_name)
        rows: list[dict[str, Any]] = []
        for raw in reader:
            source_inst_id = row_inst_id(raw, inst_column)
            if source_inst_id and source_inst_id != inst_id:
                continue
            funding_dt_utc = parse_time_value(str(raw.get(time_column) or ""), timezone_name)
            funding_ms = int(funding_dt_utc.timestamp() * 1000)
            funding_rate = normalize_funding_rate(str(raw.get(rate_column) or ""))
            if not funding_rate:
                continue
            rows.append(
                {
                    "inst_id": inst_id,
                    "funding_time": str(funding_ms),
                    "funding_time_utc": funding_dt_utc.isoformat(),
                    "funding_time_local": funding_dt_utc.astimezone(tz).isoformat(),
                    "funding_rate": funding_rate,
                    "realized_rate": funding_rate,
                    "raw_funding_rate": funding_rate,
                    "method": str(raw.get("method") or ""),
                    "formula_type": str(raw.get("formulaType") or raw.get("formula_type") or ""),
                    "next_funding_time": str(raw.get("nextFundingTime") or raw.get("next_funding_time") or ""),
                    "raw_json": json.dumps(raw, ensure_ascii=False, sort_keys=True),
                }
            )
    return rows


def filter_window(rows: list[dict[str, Any]], start_ms: int, end_exclusive_ms: int) -> list[dict[str, Any]]:
    """Filter imported rows to the requested half-open window and sort."""

    by_time: dict[int, dict[str, Any]] = {}
    for row in rows:
        funding_ms = int(row["funding_time"])
        if start_ms <= funding_ms < end_exclusive_ms:
            by_time[funding_ms] = row
    return [by_time[key] for key in sorted(by_time)]


def render_import_report(summary: dict[str, Any]) -> str:
    """Render a Markdown import report."""

    source_lines = [
        "| input_csv | imported_rows |",
        "|---|---:|",
    ]
    for item in summary.get("sources") or []:
        source_lines.append(f"| {item.get('input_csv')} | {item.get('imported_rows')} |")

    return (
        "# OKX Funding CSV Import Report\n\n"
        "## Scope\n"
        f"- inst_id={summary.get('inst_id')}\n"
        f"- start={summary.get('start')}\n"
        f"- end={summary.get('end')}\n"
        f"- timezone={summary.get('timezone')}\n"
        f"- mode={summary.get('mode')}\n"
        f"- output_csv={summary.get('output_csv')}\n\n"
        "## Result\n"
        f"- status={summary.get('status')}\n"
        f"- row_count={summary.get('row_count')}\n"
        f"- input_row_count={summary.get('input_row_count')}\n"
        f"- existing_row_count={summary.get('existing_row_count')}\n"
        f"- duplicate_timestamp_count={summary.get('duplicate_timestamp_count')}\n"
        f"- filtered_out_of_window_count={summary.get('filtered_out_of_window_count')}\n"
        f"- first_available_time={summary.get('first_available_time')}\n"
        f"- last_available_time={summary.get('last_available_time')}\n\n"
        "## Sources\n"
        f"{chr(10).join(source_lines)}\n\n"
        "## Notes\n"
        "- Raw source columns are preserved in the standard `raw_json` column.\n"
        "- Output rows are de-duplicated by `funding_time` and sorted ascending.\n"
    )


def write_import_reports(reports_dir: Path, summary: dict[str, Any]) -> None:
    """Write import JSON and Markdown reports."""

    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / IMPORT_SUMMARY_NAME).write_text(
        json.dumps(to_jsonable(summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (reports_dir / IMPORT_REPORT_NAME).write_text(render_import_report(summary), encoding="utf-8")


def run_import(
    *,
    input_path: Path | None = None,
    input_paths: list[Path] | None = None,
    inst_id: str,
    output_dir: Path,
    reports_dir: Path | None = None,
    start: str,
    end: str,
    timezone_name: str,
    append: bool = False,
) -> dict[str, Any]:
    """Import one or more manually downloaded funding CSVs."""

    resolved_input_paths = list(input_paths or ([] if input_path is None else [input_path]))
    if not resolved_input_paths:
        raise FundingCsvImportError("at least one input CSV is required")
    missing = [str(path) for path in resolved_input_paths if not path.exists()]
    if missing:
        raise FundingCsvImportError(f"input CSV does not exist: {missing}")
    window = parse_funding_window(start, end, timezone_name)
    output_csv = funding_csv_path(output_dir, inst_id, window)
    existing_rows = import_rows(output_csv, inst_id, timezone_name) if append and output_csv.exists() else []
    imported_rows: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    for path in resolved_input_paths:
        rows_for_path = import_rows(path, inst_id, timezone_name)
        imported_rows.extend(rows_for_path)
        sources.append({"input_csv": str(path), "imported_rows": len(rows_for_path)})

    combined_rows = existing_rows + imported_rows
    in_window_rows = [
        row
        for row in combined_rows
        if window.start_ms <= int(row["funding_time"]) < window.end_exclusive_ms
    ]
    rows = filter_window(combined_rows, window.start_ms, window.end_exclusive_ms)
    filtered_out_count = len(combined_rows) - len(in_window_rows)
    duplicate_count = len(in_window_rows) - len(rows)
    write_funding_csv(output_csv, rows)
    summary = {
        "inputs": [str(path) for path in resolved_input_paths],
        "inst_id": inst_id,
        "output_csv": str(output_csv),
        "reports_dir": str(reports_dir) if reports_dir else None,
        "sources": sources,
        "mode": "append" if append else "overwrite",
        "start": start,
        "end": end,
        "timezone": timezone_name,
        "input_row_count": len(imported_rows),
        "existing_row_count": len(existing_rows),
        "combined_row_count": len(combined_rows),
        "duplicate_timestamp_count": duplicate_count,
        "filtered_out_of_window_count": filtered_out_count,
        "row_count": len(rows),
        "requested_start": window.start_utc.isoformat(),
        "requested_end": window.end_exclusive_utc.isoformat(),
        "first_available_time": rows[0]["funding_time_utc"] if rows else None,
        "last_available_time": rows[-1]["funding_time_utc"] if rows else None,
        "status": "imported" if rows else "empty",
    }
    if reports_dir is not None:
        write_import_reports(reports_dir, summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""

    args = parse_args(argv)
    ensure_headless_runtime()
    logger = setup_logging("import_okx_funding_csv", verbose=args.verbose)
    try:
        input_paths = discover_input_paths(
            input_path=args.input,
            inputs=args.inputs,
            input_dir=args.input_dir,
            inst_id=str(args.inst_id),
        )
        summary = run_import(
            input_paths=input_paths,
            inst_id=str(args.inst_id),
            output_dir=resolve_download_path(args.output_dir),
            reports_dir=resolve_download_path(args.reports_dir),
            start=str(args.start),
            end=str(args.end),
            timezone_name=str(args.timezone),
            append=bool(args.append),
        )
        print_json_block("OKX funding CSV import summary:", to_jsonable(summary))
        return 0 if summary.get("row_count", 0) > 0 else 2
    except (FundingCsvImportError, FundingDownloadError) as exc:
        logger.error("Funding CSV import failed: %s", exc, extra={"event": "funding.import.error"})
        return 2
    except Exception:
        logger.exception("Unexpected funding CSV import failure", extra={"event": "funding.import.unexpected"})
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
