#!/usr/bin/env python3
"""Download OKX Historical Market Data funding files when the public query endpoint is confirmed.

The OKX changelog says the public "Get historical market data" endpoint was
added on 2025-09-02 for batch historical files, including the Funding rate
module with daily/monthly aggregation. This downloader intentionally relies on
the probe output for the formal endpoint path; if the official path cannot be
confirmed, it writes an explicit fallback report and does not guess API URLs.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import time
import zipfile
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.parse import unquote, urlparse
from urllib.request import Request, urlopen

from common_runtime import PROJECT_ROOT, ensure_headless_runtime, log_event, print_json_block, setup_logging, to_jsonable
from download_okx_funding_history import DEFAULT_END, DEFAULT_INST_IDS_ARG, DEFAULT_START, parse_inst_ids
from history_time_utils import DEFAULT_TIMEZONE
from import_okx_funding_csv import run_import
from probe_okx_historical_market_data import (
    CHANGELOG_FACTS,
    DEFAULT_OUTPUT_DIR as DEFAULT_PROBE_OUTPUT_DIR,
    ENDPOINT_FACTS,
    build_probe_params,
    build_probe_url,
    recursive_find_download_urls,
    run_probe,
)


DEFAULT_RAW_OUTPUT_DIR = PROJECT_ROOT / "data" / "funding" / "okx_historical_raw"
DEFAULT_STANDARD_OUTPUT_DIR = PROJECT_ROOT / "data" / "funding" / "okx"
DEFAULT_REPORTS_DIR = PROJECT_ROOT / "reports" / "research" / "funding_historical_download"
DEFAULT_PROBE_SUMMARY_PATH = DEFAULT_PROBE_OUTPUT_DIR / "okx_historical_market_data_probe.json"
REQUEST_TIMEOUT_SECONDS = 30

FILES_CSV_COLUMNS = [
    "inst_id",
    "aggregation",
    "period_start",
    "period_end",
    "request_url",
    "download_url",
    "local_path",
    "extracted_path",
    "status",
    "error",
]

FALLBACK_STEPS = [
    "manual CSV import with scripts/import_okx_funding_csv.py",
    "free/public external historical funding source audit before use",
    "pause funding-aware research while funding_data_complete=false",
]


class HistoricalFundingFilesError(Exception):
    """Raised when historical funding files cannot be downloaded or imported."""


@dataclass(frozen=True, slots=True)
class Period:
    """One inclusive historical-data request period."""

    start: date
    end: date

    @property
    def start_arg(self) -> str:
        return self.start.isoformat()

    @property
    def end_arg(self) -> str:
        return self.end.isoformat()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description="Download OKX historical funding files and import them to standard CSV.")
    parser.add_argument("--inst-ids", default=DEFAULT_INST_IDS_ARG)
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--end", default=DEFAULT_END)
    parser.add_argument("--aggregation", choices=("daily", "monthly"), default="monthly")
    parser.add_argument("--raw-output-dir", default=str(DEFAULT_RAW_OUTPUT_DIR))
    parser.add_argument("--standard-output-dir", default=str(DEFAULT_STANDARD_OUTPUT_DIR))
    parser.add_argument("--reports-dir", default=str(DEFAULT_REPORTS_DIR))
    parser.add_argument("--probe-summary", default=str(DEFAULT_PROBE_SUMMARY_PATH))
    parser.add_argument("--endpoint-url")
    parser.add_argument("--module")
    parser.add_argument("--timezone", default=DEFAULT_TIMEZONE)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--throttle-seconds", type=float, default=0.5)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def resolve_path(path_arg: str | Path) -> Path:
    """Resolve a path relative to the project root."""

    path = Path(path_arg)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def parse_date_arg(value: str) -> date:
    """Parse a date-only argument."""

    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise HistoricalFundingFilesError(f"invalid date: {value}") from exc


def month_start(value: date) -> date:
    """Return the first day of a date's month."""

    return value.replace(day=1)


def next_month(value: date) -> date:
    """Return the first day of the next month."""

    if value.month == 12:
        return value.replace(year=value.year + 1, month=1, day=1)
    return value.replace(month=value.month + 1, day=1)


def build_periods(start: str, end: str, aggregation: str) -> list[Period]:
    """Build inclusive daily/monthly file request periods."""

    start_date = parse_date_arg(start)
    end_date = parse_date_arg(end)
    if end_date < start_date:
        raise HistoricalFundingFilesError(f"end must be on or after start: start={start}, end={end}")

    periods: list[Period] = []
    if aggregation == "daily":
        current = start_date
        while current <= end_date:
            periods.append(Period(current, current))
            current += timedelta(days=1)
        return periods

    current = start_date
    while current <= end_date:
        period_end = min(next_month(month_start(current)) - timedelta(days=1), end_date)
        periods.append(Period(current, period_end))
        current = period_end + timedelta(days=1)
    return periods


def build_download_plan(
    *,
    inst_ids: list[str],
    start: str,
    end: str,
    aggregation: str,
    endpoint_url: str,
    module: str | None = None,
) -> list[dict[str, Any]]:
    """Build historical funding file request plan rows."""

    plan: list[dict[str, Any]] = []
    for inst_id in inst_ids:
        for period in build_periods(start, end, aggregation):
            params = build_probe_params(inst_id, period.start_arg, period.end_arg, "funding", aggregation, module)
            request_url = build_probe_url(endpoint_url, params)
            plan.append(
                {
                    "inst_id": inst_id,
                    "aggregation": aggregation,
                    "period_start": period.start_arg,
                    "period_end": period.end_arg,
                    "params": params,
                    "request_url": request_url,
                }
            )
    return plan


def request_json(url: str, timeout: int = REQUEST_TIMEOUT_SECONDS) -> dict[str, Any]:
    """Request JSON from a public URL."""

    request = Request(url, headers={"User-Agent": "cta-strategy-funding-research/1.0"})
    with urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8", errors="replace"))
    if not isinstance(payload, dict):
        raise HistoricalFundingFilesError("historical funding endpoint response is not a JSON object")
    return payload


def download_bytes(url: str, timeout: int = REQUEST_TIMEOUT_SECONDS) -> bytes:
    """Download a file URL."""

    request = Request(url, headers={"User-Agent": "cta-strategy-funding-research/1.0"})
    with urlopen(request, timeout=timeout) as response:
        return response.read()


def retry_call(
    func: Callable[[str], Any],
    value: str,
    *,
    max_retries: int,
    throttle_seconds: float,
) -> Any:
    """Call a URL function with bounded retries."""

    attempts = max(1, max_retries)
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return func(value)
        except (HTTPError, URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
            last_error = exc
            if attempt < attempts:
                time.sleep(max(0.0, throttle_seconds))
    assert last_error is not None
    raise last_error


def extract_download_urls(payload: dict[str, Any]) -> list[str]:
    """Extract CSV/ZIP download links from a query response."""

    return recursive_find_download_urls(payload)


def safe_filename_from_url(download_url: str, fallback_name: str) -> str:
    """Return a filesystem-safe filename from a URL path."""

    parsed = urlparse(download_url)
    name = unquote(Path(parsed.path).name)
    if not name:
        name = fallback_name
    name = name.replace(os.sep, "_")
    if os.altsep:
        name = name.replace(os.altsep, "_")
    return name


def is_zip_url_or_path(path_or_url: str | Path) -> bool:
    """Return whether a URL/path points at a ZIP file."""

    return str(path_or_url).lower().split("?", 1)[0].endswith(".zip")


def safe_extract_zip(zip_path: Path, extract_dir: Path) -> list[Path]:
    """Extract a ZIP file without allowing path traversal."""

    extract_dir.mkdir(parents=True, exist_ok=True)
    extracted: list[Path] = []
    base = extract_dir.resolve()
    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.infolist():
            if member.is_dir():
                continue
            target = (extract_dir / member.filename).resolve()
            if os.path.commonpath([str(base), str(target)]) != str(base):
                raise HistoricalFundingFilesError(f"unsafe ZIP member path: {member.filename}")
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as source, target.open("wb") as destination:
                destination.write(source.read())
            extracted.append(target)
    return extracted


def write_binary_file(path: Path, payload: bytes, overwrite: bool) -> None:
    """Write a downloaded file respecting overwrite policy."""

    if path.exists() and not overwrite:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def local_file_paths_for_download(
    *,
    raw_output_dir: Path,
    inst_id: str,
    aggregation: str,
    period_start: str,
    period_end: str,
    download_url: str,
    index: int,
) -> tuple[Path, Path]:
    """Return raw and extraction paths for one file URL."""

    suffix = ".zip" if is_zip_url_or_path(download_url) else ".csv"
    fallback = f"{inst_id}_funding_{period_start}_{period_end}_{index}{suffix}"
    filename = safe_filename_from_url(download_url, fallback)
    if "." not in filename:
        filename = f"{filename}{suffix}"
    period_dir = raw_output_dir / inst_id / aggregation / f"{period_start}_{period_end}"
    return period_dir / filename, period_dir / f"{Path(filename).stem}_extracted"


def download_and_expand_file(
    *,
    download_url: str,
    raw_output_dir: Path,
    inst_id: str,
    aggregation: str,
    period_start: str,
    period_end: str,
    index: int,
    downloader: Callable[[str], bytes],
    max_retries: int,
    throttle_seconds: float,
    overwrite: bool,
) -> tuple[Path, list[Path]]:
    """Download a CSV/ZIP URL and return CSV files available for import."""

    local_path, extract_dir = local_file_paths_for_download(
        raw_output_dir=raw_output_dir,
        inst_id=inst_id,
        aggregation=aggregation,
        period_start=period_start,
        period_end=period_end,
        download_url=download_url,
        index=index,
    )
    if not local_path.exists() or overwrite:
        payload = retry_call(downloader, download_url, max_retries=max_retries, throttle_seconds=throttle_seconds)
        write_binary_file(local_path, payload, overwrite=True)

    if is_zip_url_or_path(local_path):
        extracted = safe_extract_zip(local_path, extract_dir)
        csv_paths = [path for path in extracted if path.suffix.lower() == ".csv"]
        if not csv_paths:
            raise HistoricalFundingFilesError(f"ZIP did not contain CSV files: {local_path}")
        return local_path, csv_paths
    if local_path.suffix.lower() != ".csv":
        raise HistoricalFundingFilesError(f"downloaded file is not CSV or ZIP: {local_path}")
    return local_path, [local_path]


def load_probe_summary(path: Path) -> dict[str, Any] | None:
    """Load an existing probe summary if available."""

    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise HistoricalFundingFilesError(f"probe summary is not a JSON object: {path}")
    return payload


def endpoint_from_probe_or_override(
    *,
    endpoint_url: str | None,
    probe_summary_path: Path,
    inst_ids: list[str],
    start: str,
    end: str,
    aggregation: str,
    reports_dir: Path,
) -> tuple[str | None, dict[str, Any] | None, str | None]:
    """Return a confirmed endpoint URL or a failure reason."""

    if endpoint_url:
        return endpoint_url, {"endpoint_available": True, "can_auto_download": True, "endpoint_url": endpoint_url}, None

    probe_summary = load_probe_summary(probe_summary_path)
    if probe_summary is None:
        probe_summary = run_probe(
            inst_ids=inst_ids,
            start=start,
            end=end,
            data_type="funding",
            aggregation=aggregation,
            output_dir=reports_dir.parent / "funding_endpoint_probe",
            dry_run=True,
        )

    if bool(probe_summary.get("can_auto_download")) and probe_summary.get("endpoint_url"):
        return str(probe_summary["endpoint_url"]), probe_summary, None

    if probe_summary.get("endpoint_discovery_failed"):
        return None, probe_summary, "endpoint_discovery_failed"
    if probe_summary.get("auth_required"):
        return None, probe_summary, "auth_required"
    if not probe_summary.get("funding_module_available"):
        return None, probe_summary, "funding_module_unavailable"
    if not probe_summary.get("aggregation_supported"):
        return None, probe_summary, "aggregation_unavailable"
    return None, probe_summary, "can_auto_download_false"


def standard_csv_stats(path: Path) -> dict[str, Any]:
    """Return row count and time range for a standard funding CSV."""

    if not path.exists():
        return {"row_count": 0, "first_time": None, "last_time": None}
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {
        "row_count": len(rows),
        "first_time": rows[0].get("funding_time_utc") if rows else None,
        "last_time": rows[-1].get("funding_time_utc") if rows else None,
    }


def import_downloaded_files(
    *,
    downloaded_csvs_by_inst_id: dict[str, list[Path]],
    standard_output_dir: Path,
    reports_dir: Path,
    start: str,
    end: str,
    timezone_name: str,
) -> dict[str, dict[str, Any]]:
    """Import downloaded raw CSV files into canonical funding CSVs."""

    imports: dict[str, dict[str, Any]] = {}
    for inst_id, paths in downloaded_csvs_by_inst_id.items():
        if not paths:
            imports[inst_id] = {"status": "no_raw_csv", "row_count": 0}
            continue
        imports[inst_id] = run_import(
            input_paths=paths,
            inst_id=inst_id,
            output_dir=standard_output_dir,
            reports_dir=reports_dir,
            start=start,
            end=end,
            timezone_name=timezone_name,
            append=False,
        )
    return imports


def write_files_csv(reports_dir: Path, rows: list[dict[str, Any]]) -> Path:
    """Write file-level request/download trace CSV."""

    reports_dir.mkdir(parents=True, exist_ok=True)
    path = reports_dir / "okx_historical_funding_files.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FILES_CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in FILES_CSV_COLUMNS})
    return path


def render_report(summary: dict[str, Any]) -> str:
    """Render Markdown download report."""

    lines = [
        "# OKX Historical Funding Download Report",
        "",
        "## Source Audit",
        f"- changelog_2025_09_02={CHANGELOG_FACTS['historical_market_data_query_endpoint']['description']}",
        f"- endpoint_url={summary.get('endpoint_url') or ''}",
        f"- endpoint_available={str(bool(summary.get('endpoint_available'))).lower()}",
        f"- funding_module_available={str(bool(summary.get('funding_module_available'))).lower()}",
        f"- aggregation_supported={str(bool(summary.get('aggregation_supported'))).lower()}",
        f"- auth_required={str(bool(summary.get('auth_required'))).lower()}",
        f"- can_auto_download={str(bool(summary.get('can_auto_download'))).lower()}",
        f"- request_parameters=module, instType, instIdList/instFamilyList, dateAggrType, begin, end",
        f"- funding_module={ENDPOINT_FACTS['funding_module']}",
        f"- response={ENDPOINT_FACTS['response_kind']}",
        f"- timestamp_timezone={ENDPOINT_FACTS['timestamp_timezone']}",
        f"- max_query_range={ENDPOINT_FACTS['max_query_range']}",
        f"- rate_limit={summary.get('rate_limit') or ENDPOINT_FACTS['rate_limit']}",
        "",
        "## Result",
        f"- status={summary.get('status')}",
        f"- dry_run={str(bool(summary.get('dry_run'))).lower()}",
        f"- failure_reason={summary.get('failure_reason') or ''}",
        f"- plan_request_count={summary.get('plan_request_count')}",
        f"- downloaded_file_count={summary.get('downloaded_file_count')}",
        f"- extracted_csv_count={summary.get('extracted_csv_count')}",
        "",
        "## Instrument Coverage",
        "| inst_id | files_downloaded | row_count | first_time | last_time | complete |",
        "|---|---:|---:|---|---|---|",
    ]
    for row in summary.get("inst_results") or []:
        lines.append(
            "| {inst_id} | {files_downloaded} | {row_count} | {first_time} | {last_time} | {complete} |".format(
                inst_id=row.get("inst_id"),
                files_downloaded=row.get("files_downloaded", 0),
                row_count=row.get("row_count", 0),
                first_time=row.get("first_time") or "",
                last_time=row.get("last_time") or "",
                complete=str(bool(row.get("complete"))).lower(),
            )
        )
    lines.extend(
        [
            "",
            "## Fallback",
            f"- recommended_next_step={summary.get('recommended_next_step')}",
            "- fallback=" + "; ".join(summary.get("fallback") or []),
            "",
            "## Gate Note",
            "- Downloader output does not certify completeness; `make verify-funding` is the source of truth for `funding_data_complete`.",
            "- If funding_data_complete=false, V3.1, Strategy V3, demo, and live remain forbidden.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_summary_reports(reports_dir: Path, summary: dict[str, Any], file_rows: list[dict[str, Any]]) -> None:
    """Write JSON, Markdown, and file trace reports."""

    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "okx_historical_funding_download_summary.json").write_text(
        json.dumps(to_jsonable(summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (reports_dir / "okx_historical_funding_download_report.md").write_text(render_report(summary), encoding="utf-8")
    write_files_csv(reports_dir, file_rows)


def build_unavailable_summary(
    *,
    inst_ids: list[str],
    start: str,
    end: str,
    aggregation: str,
    dry_run: bool,
    raw_output_dir: Path,
    standard_output_dir: Path,
    reports_dir: Path,
    probe_summary: dict[str, Any] | None,
    failure_reason: str,
) -> dict[str, Any]:
    """Build a summary for unavailable or unconfirmed endpoints."""

    return {
        "status": "endpoint_unavailable",
        "dry_run": dry_run,
        "inst_ids": inst_ids,
        "start": start,
        "end": end,
        "aggregation": aggregation,
        "raw_output_dir": str(raw_output_dir),
        "standard_output_dir": str(standard_output_dir),
        "reports_dir": str(reports_dir),
        "endpoint_url": (probe_summary or {}).get("endpoint_url"),
        "endpoint_available": bool((probe_summary or {}).get("endpoint_available")),
        "funding_module_available": bool((probe_summary or {}).get("funding_module_available")),
        "aggregation_supported": bool((probe_summary or {}).get("aggregation_supported")),
        "auth_required": bool((probe_summary or {}).get("auth_required")),
        "can_auto_download": bool((probe_summary or {}).get("can_auto_download")),
        "can_cover_2023_2026": bool((probe_summary or {}).get("can_cover_2023_2026")),
        "failure_reason": failure_reason,
        "fallback": FALLBACK_STEPS,
        "recommended_next_step": "manual CSV import or pause funding-aware research",
        "plan_request_count": 0,
        "downloaded_file_count": 0,
        "extracted_csv_count": 0,
        "inst_results": [
            {
                "inst_id": inst_id,
                "files_downloaded": 0,
                "row_count": 0,
                "first_time": None,
                "last_time": None,
                "complete": False,
            }
            for inst_id in inst_ids
        ],
    }


def run_download(
    *,
    inst_ids: list[str],
    start: str,
    end: str,
    aggregation: str,
    raw_output_dir: Path,
    standard_output_dir: Path,
    reports_dir: Path,
    probe_summary_path: Path = DEFAULT_PROBE_SUMMARY_PATH,
    endpoint_url: str | None = None,
    module: str | None = None,
    timezone_name: str = DEFAULT_TIMEZONE,
    dry_run: bool = False,
    throttle_seconds: float = 0.5,
    max_retries: int = 5,
    overwrite: bool = False,
    requester: Callable[[str], dict[str, Any]] = request_json,
    downloader: Callable[[str], bytes] = download_bytes,
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    """Download historical funding files and import them to standard CSVs."""

    endpoint, probe_summary, failure_reason = endpoint_from_probe_or_override(
        endpoint_url=endpoint_url,
        probe_summary_path=probe_summary_path,
        inst_ids=inst_ids,
        start=start,
        end=end,
        aggregation=aggregation,
        reports_dir=reports_dir,
    )
    if endpoint is None:
        summary = build_unavailable_summary(
            inst_ids=inst_ids,
            start=start,
            end=end,
            aggregation=aggregation,
            dry_run=dry_run,
            raw_output_dir=raw_output_dir,
            standard_output_dir=standard_output_dir,
            reports_dir=reports_dir,
            probe_summary=probe_summary,
            failure_reason=failure_reason or "endpoint_unavailable_or_not_confirmed",
        )
        write_summary_reports(reports_dir, summary, [])
        return summary

    plan = build_download_plan(
        inst_ids=inst_ids,
        start=start,
        end=end,
        aggregation=aggregation,
        endpoint_url=endpoint,
        module=module,
    )
    file_rows: list[dict[str, Any]] = []
    if dry_run:
        for item in plan:
            file_rows.append(
                {
                    "inst_id": item["inst_id"],
                    "aggregation": item["aggregation"],
                    "period_start": item["period_start"],
                    "period_end": item["period_end"],
                    "request_url": item["request_url"],
                    "status": "planned",
                }
            )
        summary = {
            "status": "dry_run",
            "dry_run": True,
            "inst_ids": inst_ids,
            "start": start,
            "end": end,
            "aggregation": aggregation,
            "raw_output_dir": str(raw_output_dir),
            "standard_output_dir": str(standard_output_dir),
            "reports_dir": str(reports_dir),
            "endpoint_url": endpoint,
            "endpoint_available": True,
            "funding_module_available": True,
            "aggregation_supported": True,
            "auth_required": False,
            "can_auto_download": True,
            "can_cover_2023_2026": True,
            "failure_reason": None,
            "fallback": [],
            "recommended_next_step": "run make download-funding-historical",
            "plan_request_count": len(plan),
            "downloaded_file_count": 0,
            "extracted_csv_count": 0,
            "inst_results": [
                {
                    "inst_id": inst_id,
                    "files_downloaded": 0,
                    "row_count": 0,
                    "first_time": None,
                    "last_time": None,
                    "complete": False,
                }
                for inst_id in inst_ids
            ],
        }
        write_summary_reports(reports_dir, summary, file_rows)
        return summary

    downloaded_csvs_by_inst_id: dict[str, list[Path]] = {inst_id: [] for inst_id in inst_ids}
    downloaded_file_count = 0
    extracted_csv_count = 0

    for item in plan:
        if throttle_seconds > 0:
            time.sleep(throttle_seconds)
        try:
            payload = retry_call(requester, item["request_url"], max_retries=max_retries, throttle_seconds=throttle_seconds)
            code = str(payload.get("code", ""))
            if code != "0":
                raise HistoricalFundingFilesError(f"historical query returned code={code}, msg={payload.get('msg')}")
            urls = extract_download_urls(payload)
            if not urls:
                file_rows.append({**item, "download_url": "", "status": "no_download_url", "error": ""})
                continue
            for index, download_url in enumerate(urls, start=1):
                local_path, csv_paths = download_and_expand_file(
                    download_url=download_url,
                    raw_output_dir=raw_output_dir,
                    inst_id=str(item["inst_id"]),
                    aggregation=str(item["aggregation"]),
                    period_start=str(item["period_start"]),
                    period_end=str(item["period_end"]),
                    index=index,
                    downloader=downloader,
                    max_retries=max_retries,
                    throttle_seconds=throttle_seconds,
                    overwrite=overwrite,
                )
                downloaded_file_count += 1
                extracted_csv_count += len(csv_paths)
                downloaded_csvs_by_inst_id[str(item["inst_id"])].extend(csv_paths)
                file_rows.append(
                    {
                        **item,
                        "download_url": download_url,
                        "local_path": str(local_path),
                        "extracted_path": ",".join(str(path) for path in csv_paths),
                        "status": "downloaded",
                        "error": "",
                    }
                )
        except Exception as exc:
            error_message = f"{exc.__class__.__name__}: {exc}"
            if logger:
                log_event(
                    logger,
                    logging.WARNING,
                    "funding.historical_download.item_failed",
                    "Historical funding file request failed",
                    inst_id=item["inst_id"],
                    period_start=item["period_start"],
                    period_end=item["period_end"],
                    error=error_message,
                )
            file_rows.append({**item, "download_url": "", "local_path": "", "extracted_path": "", "status": "error", "error": error_message})

    imports = import_downloaded_files(
        downloaded_csvs_by_inst_id=downloaded_csvs_by_inst_id,
        standard_output_dir=standard_output_dir,
        reports_dir=reports_dir,
        start=start,
        end=end,
        timezone_name=timezone_name,
    )
    inst_results: list[dict[str, Any]] = []
    for inst_id in inst_ids:
        import_summary = imports.get(inst_id) or {}
        output_csv = Path(str(import_summary.get("output_csv") or ""))
        stats = standard_csv_stats(output_csv) if output_csv != Path(".") else {"row_count": 0, "first_time": None, "last_time": None}
        inst_results.append(
            {
                "inst_id": inst_id,
                "files_downloaded": len(downloaded_csvs_by_inst_id.get(inst_id, [])),
                "row_count": stats["row_count"],
                "first_time": stats["first_time"],
                "last_time": stats["last_time"],
                "complete": False,
                "complete_decided_by": "make verify-funding",
                "output_csv": str(output_csv) if output_csv != Path(".") else None,
            }
        )

    any_errors = any(row.get("status") == "error" for row in file_rows)
    summary = {
        "status": "downloaded_with_errors" if any_errors else "downloaded",
        "dry_run": False,
        "inst_ids": inst_ids,
        "start": start,
        "end": end,
        "aggregation": aggregation,
        "raw_output_dir": str(raw_output_dir),
        "standard_output_dir": str(standard_output_dir),
        "reports_dir": str(reports_dir),
        "endpoint_url": endpoint,
        "endpoint_available": True,
        "funding_module_available": True,
        "aggregation_supported": True,
        "auth_required": False,
        "can_auto_download": True,
        "can_cover_2023_2026": True,
        "failure_reason": "download_errors" if any_errors else None,
        "fallback": FALLBACK_STEPS if any_errors else [],
        "recommended_next_step": "run make verify-funding",
        "plan_request_count": len(plan),
        "downloaded_file_count": downloaded_file_count,
        "extracted_csv_count": extracted_csv_count,
        "inst_results": inst_results,
        "imports": imports,
    }
    write_summary_reports(reports_dir, summary, file_rows)
    return summary


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""

    args = parse_args(argv)
    ensure_headless_runtime()
    logger = setup_logging("download_okx_historical_funding_files", verbose=args.verbose)
    try:
        summary = run_download(
            inst_ids=parse_inst_ids(args.inst_ids),
            start=str(args.start),
            end=str(args.end),
            aggregation=str(args.aggregation),
            raw_output_dir=resolve_path(args.raw_output_dir),
            standard_output_dir=resolve_path(args.standard_output_dir),
            reports_dir=resolve_path(args.reports_dir),
            probe_summary_path=resolve_path(args.probe_summary),
            endpoint_url=args.endpoint_url,
            module=args.module,
            timezone_name=str(args.timezone),
            dry_run=bool(args.dry_run),
            throttle_seconds=float(args.throttle_seconds),
            max_retries=int(args.max_retries),
            overwrite=bool(args.overwrite),
            logger=logger,
        )
        print_json_block(
            "OKX historical funding download summary:",
            {
                "status": summary.get("status"),
                "endpoint_available": summary.get("endpoint_available"),
                "can_auto_download": summary.get("can_auto_download"),
                "failure_reason": summary.get("failure_reason"),
                "recommended_next_step": summary.get("recommended_next_step"),
            },
        )
        if args.dry_run:
            return 0
        return 0 if summary.get("status") == "downloaded" else 2
    except Exception as exc:
        log_event(
            logger,
            logging.ERROR,
            "funding.historical_download.error",
            "Historical funding download failed",
            error_class=exc.__class__.__name__,
            error_message=str(exc),
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
