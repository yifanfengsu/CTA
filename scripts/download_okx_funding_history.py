#!/usr/bin/env python3
"""Download OKX public funding-rate history for research-only analysis."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import time
from dataclasses import dataclass
from datetime import date, datetime, time as dt_time, timedelta, timezone
from pathlib import Path
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

from common_runtime import PROJECT_ROOT, ensure_headless_runtime, log_event, print_json_block, setup_logging, to_jsonable
from history_time_utils import DEFAULT_TIMEZONE, resolve_timezone


DEFAULT_INST_IDS = [
    "BTC-USDT-SWAP",
    "ETH-USDT-SWAP",
    "SOL-USDT-SWAP",
    "LINK-USDT-SWAP",
    "DOGE-USDT-SWAP",
]
DEFAULT_INST_IDS_ARG = ",".join(DEFAULT_INST_IDS)
DEFAULT_START = "2023-01-01"
DEFAULT_END = "2026-03-31"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "funding" / "okx"
DEFAULT_REPORTS_DIR = PROJECT_ROOT / "reports" / "research" / "funding"
OKX_FUNDING_HISTORY_URL = "https://www.okx.com/api/v5/public/funding-rate-history"
OKX_DOCS_URL = "https://app.okx.com/docs-v5/zh/#public-data-rest-api-get-funding-rate-history"
OKX_HISTORICAL_DATA_URL = "https://www.okx.com/en-ar/historical-data"
OKX_LIMIT = 400
DEFAULT_MAX_PAGES = 500
REQUEST_TIMEOUT_SECONDS = 20

CSV_COLUMNS = [
    "inst_id",
    "funding_time",
    "funding_time_utc",
    "funding_time_local",
    "funding_rate",
    "realized_rate",
    "raw_funding_rate",
    "method",
    "formula_type",
    "next_funding_time",
    "raw_json",
]

REQUEST_TRACE_COLUMNS = [
    "inst_id",
    "request_index",
    "url_or_endpoint",
    "params_json",
    "row_count",
    "newest_funding_time",
    "oldest_funding_time",
    "cursor_after_used",
    "stop_reason",
    "error",
]


class FundingDownloadError(Exception):
    """Raised when OKX funding history cannot be downloaded."""


class OkxFundingRestError(Exception):
    """Raised when OKX public REST returns a non-zero response."""

    def __init__(self, code: str | None, message: str, response_body: str | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.response_body = response_body


@dataclass(frozen=True, slots=True)
class FundingWindow:
    """Half-open funding timestamp window."""

    start: datetime
    end_exclusive: datetime
    timezone_name: str
    start_arg: str
    end_arg: str

    @property
    def start_utc(self) -> datetime:
        return self.start.astimezone(timezone.utc)

    @property
    def end_exclusive_utc(self) -> datetime:
        return self.end_exclusive.astimezone(timezone.utc)

    @property
    def start_ms(self) -> int:
        return datetime_to_ms(self.start_utc)

    @property
    def end_exclusive_ms(self) -> int:
        return datetime_to_ms(self.end_exclusive_utc)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description="Download OKX public funding-rate history.")
    parser.add_argument("--inst-ids", default=DEFAULT_INST_IDS_ARG)
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--end", default=DEFAULT_END)
    parser.add_argument("--timezone", default=DEFAULT_TIMEZONE)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--reports-dir", default=str(DEFAULT_REPORTS_DIR))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--throttle-seconds", type=float, default=0.25)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--limit", type=int, default=OKX_LIMIT)
    parser.add_argument("--max-pages", type=int, default=DEFAULT_MAX_PAGES)
    parser.add_argument("--pagination-mode", choices=("backward",), default="backward")
    parser.add_argument("--stop-on-short-page", type=parse_bool_arg, default=False)
    parser.add_argument("--allow-partial", type=parse_bool_arg, default=True)
    parser.add_argument("--fail-if-incomplete", type=parse_bool_arg, default=False)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def parse_bool_arg(value: str | bool) -> bool:
    """Parse explicit true/false CLI values."""

    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"expected true/false, got {value!r}")


def resolve_path(path_arg: str | Path) -> Path:
    """Resolve a path relative to project root."""

    path = Path(path_arg)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def parse_inst_ids(raw_value: str | list[str] | tuple[str, ...]) -> list[str]:
    """Parse comma/space-separated OKX instrument ids."""

    if isinstance(raw_value, (list, tuple)):
        values = [str(item).strip() for item in raw_value]
    else:
        values = [token.strip() for token in str(raw_value).replace(",", " ").split()]
    inst_ids = [value for value in values if value]
    if not inst_ids:
        raise FundingDownloadError("--inst-ids 不能为空")
    return list(dict.fromkeys(inst_ids))


def parse_date_or_datetime(value: str, tz: ZoneInfo) -> datetime:
    """Parse a date-only or ISO datetime value in the requested timezone."""

    text = value.strip()
    if not text:
        raise FundingDownloadError("start/end 不能为空")
    try:
        if len(text) == 10:
            return datetime.combine(date.fromisoformat(text), dt_time.min).replace(tzinfo=tz)
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise FundingDownloadError(f"无法解析时间: {value}") from exc
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=tz)
    return parsed.astimezone(tz)


def parse_funding_window(start_arg: str, end_arg: str, timezone_name: str) -> FundingWindow:
    """Parse a half-open funding window from CLI arguments."""

    tz = resolve_timezone(timezone_name)
    start = parse_date_or_datetime(start_arg, tz)
    if len(end_arg.strip()) == 10:
        end_exclusive = parse_date_or_datetime(end_arg, tz) + timedelta(days=1)
    else:
        end_exclusive = parse_date_or_datetime(end_arg, tz)
    if end_exclusive <= start:
        raise FundingDownloadError(f"end must be later than start: start={start}, end_exclusive={end_exclusive}")
    return FundingWindow(
        start=start,
        end_exclusive=end_exclusive,
        timezone_name=timezone_name,
        start_arg=start_arg,
        end_arg=end_arg,
    )


def datetime_to_ms(value: datetime) -> int:
    """Convert an aware datetime to Unix milliseconds."""

    return int(value.astimezone(timezone.utc).timestamp() * 1000)


def ms_to_datetime(value: int | str) -> datetime:
    """Convert Unix milliseconds to UTC datetime."""

    return datetime.fromtimestamp(int(value) / 1000.0, tz=timezone.utc)


def funding_csv_path(output_dir: Path, inst_id: str, window: FundingWindow) -> Path:
    """Return the canonical funding CSV path for one instrument."""

    return output_dir / f"{inst_id}_funding_{window.start_arg}_{window.end_arg}.csv"


def validate_limit(limit: int) -> int:
    """Validate OKX funding-rate-history page limit."""

    value = int(limit)
    if value <= 0 or value > OKX_LIMIT:
        raise FundingDownloadError(f"--limit must be between 1 and {OKX_LIMIT}; got {limit}")
    return value


def build_request_params(
    inst_id: str,
    *,
    after_ms: int | None = None,
    before_ms: int | None = None,
    limit: int = OKX_LIMIT,
) -> dict[str, str]:
    """Build OKX funding history query params.

    OKX documents `before` as records newer than the requested `fundingTime`
    and `after` as records older than the requested `fundingTime`; `limit`
    is capped at 400. Backward pagination therefore advances with
    `after=<oldest fundingTime from the previous page>`.
    """

    params: dict[str, str] = {"instId": inst_id, "limit": str(validate_limit(limit))}
    if after_ms is not None:
        params["after"] = str(after_ms)
    if before_ms is not None:
        params["before"] = str(before_ms)
    return params


def build_request_url(inst_id: str, *, after_ms: int | None = None, before_ms: int | None = None, limit: int = OKX_LIMIT) -> str:
    """Build one OKX funding history URL."""

    params = build_request_params(inst_id, after_ms=after_ms, before_ms=before_ms, limit=limit)
    return f"{OKX_FUNDING_HISTORY_URL}?{urlencode(params)}"


def request_json(url: str, timeout: int = REQUEST_TIMEOUT_SECONDS) -> dict[str, Any]:
    """Request and decode JSON from OKX public REST."""

    request = Request(url, headers={"User-Agent": "cta-strategy-funding-research/1.0"})
    with urlopen(request, timeout=timeout) as response:
        body = response.read().decode("utf-8")
    try:
        payload = json.loads(body)
    except json.JSONDecodeError as exc:
        raise FundingDownloadError(f"OKX response is not JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise FundingDownloadError("OKX response JSON is not an object")
    return payload


def parse_okx_funding_response(payload: dict[str, Any], timezone_name: str) -> list[dict[str, Any]]:
    """Parse OKX funding-rate-history response records into CSV rows."""

    code = str(payload.get("code", ""))
    if code != "0":
        raise OkxFundingRestError(code=code, message=str(payload.get("msg") or "OKX funding API error"), response_body=json.dumps(payload))
    data = payload.get("data")
    if data is None:
        return []
    if not isinstance(data, list):
        raise FundingDownloadError("OKX funding response data is not a list")

    tz = resolve_timezone(timezone_name)
    rows: list[dict[str, Any]] = []
    for raw in data:
        if not isinstance(raw, dict):
            continue
        funding_time_raw = raw.get("fundingTime")
        if funding_time_raw in (None, ""):
            continue
        funding_dt_utc = ms_to_datetime(funding_time_raw)
        realized_rate = str(raw.get("realizedRate") or "")
        raw_funding_rate = str(raw.get("fundingRate") or "")
        effective_rate = realized_rate if realized_rate else raw_funding_rate
        rows.append(
            {
                "inst_id": str(raw.get("instId") or ""),
                "funding_time": str(int(funding_time_raw)),
                "funding_time_utc": funding_dt_utc.isoformat(),
                "funding_time_local": funding_dt_utc.astimezone(tz).isoformat(),
                "funding_rate": effective_rate,
                "realized_rate": realized_rate,
                "raw_funding_rate": raw_funding_rate,
                "method": str(raw.get("method") or ""),
                "formula_type": str(raw.get("formulaType") or ""),
                "next_funding_time": str(raw.get("nextFundingTime") or ""),
                "raw_json": json.dumps(raw, ensure_ascii=False, sort_keys=True),
            }
        )
    return rows


def fetch_okx_funding_page(
    inst_id: str,
    *,
    after_ms: int | None,
    before_ms: int | None = None,
    limit: int = OKX_LIMIT,
    timezone_name: str = DEFAULT_TIMEZONE,
    requester: Callable[[str], dict[str, Any]] = request_json,
) -> list[dict[str, Any]]:
    """Fetch and parse one OKX funding-rate-history page."""

    url = build_request_url(inst_id, after_ms=after_ms, before_ms=before_ms, limit=limit)
    payload = requester(url)
    return parse_okx_funding_response(payload, timezone_name)


def fetch_with_retries(
    inst_id: str,
    *,
    after_ms: int | None,
    before_ms: int | None,
    limit: int,
    timezone_name: str,
    max_retries: int,
    logger: logging.Logger,
    requester: Callable[[str], dict[str, Any]] = request_json,
) -> list[dict[str, Any]]:
    """Fetch one page with retries and exponential backoff."""

    attempts = max(1, int(max_retries))
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            url = build_request_url(inst_id, after_ms=after_ms, before_ms=before_ms, limit=limit)
            payload = requester(url)
            return parse_okx_funding_response(payload, timezone_name)
        except (HTTPError, URLError, TimeoutError, OSError, FundingDownloadError, OkxFundingRestError) as exc:
            last_error = exc
            log_event(
                logger,
                logging.WARNING,
                "funding.download.retry",
                "OKX funding page request failed",
                inst_id=inst_id,
                after_ms=after_ms,
                before_ms=before_ms,
                limit=limit,
                attempt=attempt,
                max_retries=attempts,
                error_class=exc.__class__.__name__,
                error_message=str(exc),
            )
            if attempt < attempts:
                time.sleep(min(8.0, 0.5 * (2 ** (attempt - 1))))
    raise FundingDownloadError(f"{inst_id} funding history download failed after {attempts} attempts: {last_error!r}")


def build_download_plan(
    inst_ids: list[str],
    window: FundingWindow,
    output_dir: Path,
    *,
    limit: int = OKX_LIMIT,
    max_pages: int = DEFAULT_MAX_PAGES,
    pagination_mode: str = "backward",
) -> list[dict[str, Any]]:
    """Build the deterministic dry-run/download plan."""

    limit = validate_limit(limit)
    return [
        {
            "inst_id": inst_id,
            "initial_after_ms": window.end_exclusive_ms,
            "initial_after_utc": window.end_exclusive_utc.isoformat(),
            "start_utc": window.start_utc.isoformat(),
            "end_exclusive_utc": window.end_exclusive_utc.isoformat(),
            "endpoint": build_request_url(inst_id, after_ms=window.end_exclusive_ms, limit=limit),
            "output_csv": str(funding_csv_path(output_dir, inst_id, window)),
            "pagination": "backward_by_after_cursor_using_oldest_fundingTime",
            "pagination_mode": pagination_mode,
            "limit": limit,
            "max_pages": max_pages,
        }
        for inst_id in inst_ids
    ]


def dedupe_and_sort_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Deduplicate by funding_time and sort ascending."""

    by_time: dict[int, dict[str, Any]] = {}
    for row in rows:
        try:
            key = int(row["funding_time"])
        except (KeyError, TypeError, ValueError):
            continue
        by_time[key] = row
    return [by_time[key] for key in sorted(by_time)]


def write_funding_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write one funding CSV."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def trace_row(
    *,
    inst_id: str,
    request_index: int,
    params: dict[str, str],
    rows: list[dict[str, Any]],
    cursor_after_used: int | None,
    stop_reason: str = "",
    error: str = "",
) -> dict[str, Any]:
    """Build one request trace CSV row."""

    page_times = sorted(int(row["funding_time"]) for row in rows if row.get("funding_time") not in (None, ""))
    newest = ms_to_datetime(page_times[-1]).isoformat() if page_times else ""
    oldest = ms_to_datetime(page_times[0]).isoformat() if page_times else ""
    return {
        "inst_id": inst_id,
        "request_index": request_index,
        "url_or_endpoint": OKX_FUNDING_HISTORY_URL,
        "params_json": json.dumps(params, ensure_ascii=False, sort_keys=True),
        "row_count": len(rows),
        "newest_funding_time": newest,
        "oldest_funding_time": oldest,
        "cursor_after_used": "" if cursor_after_used is None else str(cursor_after_used),
        "stop_reason": stop_reason,
        "error": error,
    }


def summarize_download_rows(
    inst_id: str,
    rows: list[dict[str, Any]],
    output_csv: Path,
    request_count: int,
    warnings: list[str],
    *,
    window: FundingWindow,
    reached_requested_start: bool,
    stop_reason: str,
    endpoint_history_lower_bound_ms: int | None,
    error: str | None = None,
) -> dict[str, Any]:
    """Build one instrument download summary."""

    first = rows[0] if rows else None
    last = rows[-1] if rows else None
    first_ms = int(first["funding_time"]) if first else None
    completion_status = classify_completion_status(
        rows=rows,
        reached_requested_start=reached_requested_start,
        stop_reason=stop_reason,
        first_ms=first_ms,
        requested_start_ms=window.start_ms,
        error=error,
    )
    endpoint_history_limit_suspected = bool(
        completion_status == "partial_endpoint_limited"
        and first_ms is not None
        and first_ms > window.start_ms
    )
    partial_pagination_failed = completion_status == "partial_pagination_failed"
    summary_warnings = list(warnings)
    if completion_status != "complete":
        summary_warnings.append(
            f"funding data does not fully cover requested_start={window.start_utc.isoformat()}"
        )
    if endpoint_history_limit_suspected:
        summary_warnings.append("REST endpoint history lower bound appears newer than requested start")
    if partial_pagination_failed:
        summary_warnings.append(f"pagination stopped before requested start: {stop_reason}")
    return {
        "inst_id": inst_id,
        "status": completion_status,
        "completion_status": completion_status,
        "row_count": len(rows),
        "request_count": request_count,
        "requested_start": window.start_utc.isoformat(),
        "requested_end": window.end_exclusive_utc.isoformat(),
        "fetched_first_time": first.get("funding_time_utc") if first else None,
        "fetched_last_time": last.get("funding_time_utc") if last else None,
        "first_funding_time": first.get("funding_time_utc") if first else None,
        "last_funding_time": last.get("funding_time_utc") if last else None,
        "reached_requested_start": reached_requested_start,
        "endpoint_history_lower_bound": (
            ms_to_datetime(endpoint_history_lower_bound_ms).isoformat() if endpoint_history_lower_bound_ms is not None else None
        ),
        "endpoint_history_limit_suspected": endpoint_history_limit_suspected,
        "partial_pagination_failed": partial_pagination_failed,
        "stop_reason": stop_reason,
        "output_csv": str(output_csv),
        "warnings": list(dict.fromkeys(summary_warnings)),
    }


def classify_completion_status(
    *,
    rows: list[dict[str, Any]],
    reached_requested_start: bool,
    stop_reason: str,
    first_ms: int | None,
    requested_start_ms: int,
    error: str | None,
) -> str:
    """Classify one instrument's endpoint coverage."""

    if error:
        return "error"
    if not rows:
        return "empty"
    if reached_requested_start:
        return "complete"
    if first_ms is not None and first_ms > requested_start_ms and stop_reason in {
        "empty_page_endpoint_history_lower_bound",
        "no_older_rows_after_short_page",
    }:
        return "partial_endpoint_limited"
    return "partial_pagination_failed"


def download_one_inst_id(
    inst_id: str,
    window: FundingWindow,
    output_dir: Path,
    *,
    throttle_seconds: float,
    max_retries: int,
    limit: int,
    max_pages: int,
    pagination_mode: str,
    stop_on_short_page: bool,
    logger: logging.Logger,
    requester: Callable[[str], dict[str, Any]] = request_json,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Download all funding rows for one OKX instrument."""

    if pagination_mode != "backward":
        raise FundingDownloadError(f"unsupported pagination mode: {pagination_mode}")
    limit = validate_limit(limit)
    cursor_after = window.end_exclusive_ms
    collected: list[dict[str, Any]] = []
    warnings: list[str] = []
    request_count = 0
    seen_page_cursors: set[int] = set()
    traces: list[dict[str, Any]] = []
    reached_requested_start = False
    stop_reason = ""
    endpoint_history_lower_bound_ms: int | None = None
    previous_short_page = False

    while request_count < max_pages:
        if cursor_after in seen_page_cursors:
            stop_reason = "pagination_cursor_repeated"
            warnings.append(f"pagination cursor repeated at {cursor_after}")
            break
        seen_page_cursors.add(cursor_after)
        request_index = request_count + 1
        params = build_request_params(inst_id, after_ms=cursor_after, limit=limit)

        try:
            page = fetch_with_retries(
                inst_id,
                after_ms=cursor_after,
                before_ms=None,
                limit=limit,
                timezone_name=window.timezone_name,
                max_retries=max_retries,
                logger=logger,
                requester=requester,
            )
        except Exception as exc:
            stop_reason = "request_error"
            traces.append(
                trace_row(
                    inst_id=inst_id,
                    request_index=request_index,
                    params=params,
                    rows=[],
                    cursor_after_used=cursor_after,
                    stop_reason=stop_reason,
                    error=f"{exc.__class__.__name__}: {exc}",
                )
            )
            raise
        request_count += 1
        if not page:
            stop_reason = "no_older_rows_after_short_page" if previous_short_page else "empty_page_endpoint_history_lower_bound"
            traces.append(
                trace_row(
                    inst_id=inst_id,
                    request_index=request_index,
                    params=params,
                    rows=[],
                    cursor_after_used=cursor_after,
                    stop_reason=stop_reason,
                )
            )
            break

        page_times = sorted(int(row["funding_time"]) for row in page)
        oldest_ms = page_times[0]
        endpoint_history_lower_bound_ms = (
            oldest_ms if endpoint_history_lower_bound_ms is None else min(endpoint_history_lower_bound_ms, oldest_ms)
        )
        for row in page:
            funding_ms = int(row["funding_time"])
            if window.start_ms <= funding_ms < window.end_exclusive_ms:
                collected.append(row)

        if oldest_ms <= window.start_ms:
            reached_requested_start = True
            stop_reason = "reached_requested_start"
            traces.append(
                trace_row(
                    inst_id=inst_id,
                    request_index=request_index,
                    params=params,
                    rows=page,
                    cursor_after_used=cursor_after,
                    stop_reason=stop_reason,
                )
            )
            break
        if oldest_ms >= cursor_after:
            stop_reason = "pagination_cursor_not_moving"
            warnings.append(f"pagination did not move backward: after={cursor_after}, oldest={oldest_ms}")
            traces.append(
                trace_row(
                    inst_id=inst_id,
                    request_index=request_index,
                    params=params,
                    rows=page,
                    cursor_after_used=cursor_after,
                    stop_reason=stop_reason,
                )
            )
            break
        if len(page) < limit:
            warnings.append(f"OKX returned a short page before start was reached: page_size={len(page)}, oldest_ms={oldest_ms}")
            if stop_on_short_page:
                stop_reason = "short_page_stop_on_short_page"
                traces.append(
                    trace_row(
                        inst_id=inst_id,
                        request_index=request_index,
                        params=params,
                        rows=page,
                        cursor_after_used=cursor_after,
                        stop_reason=stop_reason,
                    )
                )
                break
            previous_short_page = True
        else:
            previous_short_page = False

        traces.append(
            trace_row(
                inst_id=inst_id,
                request_index=request_index,
                params=params,
                rows=page,
                cursor_after_used=cursor_after,
                stop_reason="short_page_continue_confirming_older" if previous_short_page else "continue",
            )
        )

        next_cursor_after = oldest_ms
        if next_cursor_after == cursor_after:
            stop_reason = "pagination_cursor_not_moving"
            warnings.append(f"pagination cursor did not change: after={cursor_after}")
            break
        cursor_after = next_cursor_after
        if throttle_seconds > 0:
            time.sleep(throttle_seconds)
    else:
        stop_reason = "max_pages_reached"
        warnings.append(f"max pages reached before requested start: max_pages={max_pages}")
        if traces:
            traces[-1]["stop_reason"] = stop_reason

    rows = dedupe_and_sort_rows(collected)
    output_csv = funding_csv_path(output_dir, inst_id, window)
    write_funding_csv(output_csv, rows)
    if not stop_reason:
        stop_reason = "unknown_stop"
    return (
        summarize_download_rows(
            inst_id,
            rows,
            output_csv,
            request_count,
            warnings,
            window=window,
            reached_requested_start=reached_requested_start,
            stop_reason=stop_reason,
            endpoint_history_lower_bound_ms=endpoint_history_lower_bound_ms,
        ),
        traces,
    )


def render_download_report(summary: dict[str, Any]) -> str:
    """Render Markdown download report."""

    rows = summary.get("downloads") or []
    table_lines = [
        "| inst_id | status | row_count | requested_start | first_available | last_available | reached_start | endpoint_limit_suspected | requests | warnings |",
        "|---|---|---:|---|---|---|---|---|---:|---|",
    ]
    for row in rows:
        warnings = "; ".join(row.get("warnings") or [])
        table_lines.append(
            f"| {row.get('inst_id')} | {row.get('status')} | {row.get('row_count')} | "
            f"{row.get('requested_start') or ''} | {row.get('fetched_first_time') or ''} | "
            f"{row.get('fetched_last_time') or ''} | {str(bool(row.get('reached_requested_start'))).lower()} | "
            f"{str(bool(row.get('endpoint_history_limit_suspected'))).lower()} | "
            f"{row.get('request_count') or 0} | {warnings} |"
        )

    return (
        "# OKX Funding Download Report\n\n"
        "## Source Audit\n"
        f"- endpoint={OKX_FUNDING_HISTORY_URL}\n"
        f"- docs={OKX_DOCS_URL}\n"
        "- OKX docs semantics: `before` returns records newer than the requested `fundingTime`; "
        "`after` returns records older than the requested `fundingTime`; `limit` maximum is 400.\n"
        "- Endpoint history may be bounded (for example recent three months); the downloader treats it as incomplete unless pagination reaches the requested start.\n"
        f"- fallback={OKX_HISTORICAL_DATA_URL} provides historical perpetual funding rates from March 2022 onwards.\n"
        "- Funding intervals are inferred from returned `fundingTime` differences; fixed 8h intervals are not assumed.\n\n"
        "## Scope\n"
        f"- dry_run={str(bool(summary.get('dry_run'))).lower()}\n"
        f"- endpoint={OKX_FUNDING_HISTORY_URL}\n"
        "- API key required=false\n"
        f"- timezone={summary.get('timezone')}\n"
        f"- start={summary.get('start')}\n"
        f"- end={summary.get('end')}\n"
        f"- pagination_mode={summary.get('pagination_mode')}\n"
        f"- limit={summary.get('limit')}\n"
        f"- max_pages={summary.get('max_pages')}\n"
        f"- stop_on_short_page={str(bool(summary.get('stop_on_short_page'))).lower()}\n"
        "- pagination=backward pages using `after=<oldest fundingTime>`; short pages are confirmed with one more older request unless configured otherwise.\n\n"
        "## Coverage Summary\n"
        f"- funding_data_complete={str(bool(summary.get('funding_data_complete'))).lower()}\n"
        f"- endpoint_history_limit_suspected={str(bool(summary.get('endpoint_history_limit_suspected'))).lower()}\n"
        f"- partial_pagination_failed={str(bool(summary.get('partial_pagination_failed'))).lower()}\n"
        f"- request_trace={summary.get('request_trace_path')}\n\n"
        "## Download Results\n"
        f"{chr(10).join(table_lines)}\n\n"
        "## Notes\n"
        "- Raw funding CSV files are research inputs and are ignored by git.\n"
        "- `funding_rate` uses OKX `realizedRate` when present, otherwise `fundingRate`.\n"
        "- Download success does not imply data completeness; partial endpoint coverage cannot be used for strategy decisions.\n"
        "- If the REST endpoint does not cover the requested start, manually import OKX Historical Data funding CSVs and rerun verification.\n"
    )


def write_request_trace_csv(reports_dir: Path, request_traces: list[dict[str, Any]]) -> Path:
    """Write the per-request pagination trace CSV."""

    reports_dir.mkdir(parents=True, exist_ok=True)
    path = reports_dir / "okx_funding_download_requests.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REQUEST_TRACE_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(request_traces)
    return path


def write_reports(reports_dir: Path, summary: dict[str, Any], request_traces: list[dict[str, Any]] | None = None) -> None:
    """Write JSON and Markdown download reports."""

    reports_dir.mkdir(parents=True, exist_ok=True)
    trace_path = write_request_trace_csv(reports_dir, request_traces or [])
    summary["request_trace_path"] = str(trace_path)
    (reports_dir / "okx_funding_download_summary.json").write_text(
        json.dumps(to_jsonable(summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (reports_dir / "okx_funding_download_report.md").write_text(render_download_report(summary), encoding="utf-8")


def run_download(
    *,
    inst_ids: list[str],
    window: FundingWindow,
    output_dir: Path,
    reports_dir: Path,
    dry_run: bool,
    throttle_seconds: float,
    max_retries: int,
    logger: logging.Logger,
    limit: int = OKX_LIMIT,
    max_pages: int = DEFAULT_MAX_PAGES,
    pagination_mode: str = "backward",
    stop_on_short_page: bool = False,
    allow_partial: bool = True,
    fail_if_incomplete: bool = False,
    requester: Callable[[str], dict[str, Any]] = request_json,
) -> dict[str, Any]:
    """Run funding download or dry-run planning."""

    limit = validate_limit(limit)
    plan = build_download_plan(
        inst_ids,
        window,
        output_dir,
        limit=limit,
        max_pages=max_pages,
        pagination_mode=pagination_mode,
    )
    summary: dict[str, Any] = {
        "dry_run": dry_run,
        "endpoint": OKX_FUNDING_HISTORY_URL,
        "docs_url": OKX_DOCS_URL,
        "historical_data_url": OKX_HISTORICAL_DATA_URL,
        "inst_ids": inst_ids,
        "timezone": window.timezone_name,
        "start": window.start_arg,
        "end": window.end_arg,
        "start_utc": window.start_utc.isoformat(),
        "end_exclusive_utc": window.end_exclusive_utc.isoformat(),
        "requested_start": window.start_utc.isoformat(),
        "requested_end": window.end_exclusive_utc.isoformat(),
        "limit": limit,
        "max_pages": max_pages,
        "pagination_mode": pagination_mode,
        "stop_on_short_page": stop_on_short_page,
        "allow_partial": allow_partial,
        "fail_if_incomplete": fail_if_incomplete,
        "pagination_semantics": {
            "before": "records newer than requested fundingTime",
            "after": "records older than requested fundingTime",
            "limit_max": OKX_LIMIT,
            "backward_cursor": "after=<oldest fundingTime from previous page>",
        },
        "funding_interval_assumption": "inferred_from_fundingTime_differences_not_fixed_8h",
        "page_plan": plan,
        "downloads": [],
        "funding_data_complete": False,
        "endpoint_history_limit_suspected": False,
        "partial_pagination_failed": False,
        "next_step": None,
        "success": True,
    }

    if dry_run:
        summary["downloads"] = [
            {
                "inst_id": item["inst_id"],
                "status": "planned",
                "completion_status": "planned",
                "row_count": 0,
                "request_count": 0,
                "requested_start": window.start_utc.isoformat(),
                "requested_end": window.end_exclusive_utc.isoformat(),
                "fetched_first_time": None,
                "fetched_last_time": None,
                "first_funding_time": None,
                "last_funding_time": None,
                "reached_requested_start": False,
                "endpoint_history_lower_bound": None,
                "endpoint_history_limit_suspected": False,
                "partial_pagination_failed": False,
                "stop_reason": "dry_run",
                "output_csv": item["output_csv"],
                "warnings": ["dry_run_no_network_no_csv_written"],
            }
            for item in plan
        ]
        summary["success"] = True
        write_reports(reports_dir, summary, [])
        return summary

    downloads: list[dict[str, Any]] = []
    request_traces: list[dict[str, Any]] = []
    for inst_id in inst_ids:
        try:
            result, traces = download_one_inst_id(
                inst_id,
                window,
                output_dir,
                throttle_seconds=throttle_seconds,
                max_retries=max_retries,
                limit=limit,
                max_pages=max_pages,
                pagination_mode=pagination_mode,
                stop_on_short_page=stop_on_short_page,
                logger=logger,
                requester=requester,
            )
            downloads.append(result)
            request_traces.extend(traces)
        except Exception as exc:
            downloads.append(
                {
                    "inst_id": inst_id,
                    "status": "error",
                    "completion_status": "error",
                    "row_count": 0,
                    "request_count": 0,
                    "requested_start": window.start_utc.isoformat(),
                    "requested_end": window.end_exclusive_utc.isoformat(),
                    "fetched_first_time": None,
                    "fetched_last_time": None,
                    "first_funding_time": None,
                    "last_funding_time": None,
                    "reached_requested_start": False,
                    "endpoint_history_lower_bound": None,
                    "endpoint_history_limit_suspected": False,
                    "partial_pagination_failed": False,
                    "stop_reason": "request_error",
                    "output_csv": str(funding_csv_path(output_dir, inst_id, window)),
                    "warnings": [f"{exc.__class__.__name__}: {exc}"],
                }
            )
            log_event(
                logger,
                logging.ERROR,
                "funding.download.failed",
                "Funding download failed for instrument",
                inst_id=inst_id,
                error_class=exc.__class__.__name__,
                error_message=str(exc),
            )

    summary["downloads"] = downloads
    summary["funding_data_complete"] = bool(downloads and all(row.get("completion_status") == "complete" for row in downloads))
    summary["endpoint_history_limit_suspected"] = bool(any(row.get("endpoint_history_limit_suspected") for row in downloads))
    summary["partial_pagination_failed"] = bool(any(row.get("partial_pagination_failed") for row in downloads))
    if not summary["funding_data_complete"]:
        summary["next_step"] = "manual historical funding CSV import"
    has_errors = any(row.get("completion_status") == "error" for row in downloads)
    has_rows_or_partial = bool(downloads and all(row.get("row_count", 0) > 0 or row.get("completion_status") == "complete" for row in downloads))
    summary["success"] = bool(
        downloads
        and not has_errors
        and (summary["funding_data_complete"] or (allow_partial and has_rows_or_partial))
        and not (fail_if_incomplete and not summary["funding_data_complete"])
    )
    write_reports(reports_dir, summary, request_traces)
    return summary


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""

    args = parse_args(argv)
    ensure_headless_runtime()
    logger = setup_logging("download_okx_funding_history", verbose=args.verbose)
    try:
        inst_ids = parse_inst_ids(args.inst_ids)
        window = parse_funding_window(args.start, args.end, args.timezone)
        summary = run_download(
            inst_ids=inst_ids,
            window=window,
            output_dir=resolve_path(args.output_dir),
            reports_dir=resolve_path(args.reports_dir),
            dry_run=bool(args.dry_run),
            throttle_seconds=float(args.throttle_seconds),
            max_retries=int(args.max_retries),
            logger=logger,
            limit=int(args.limit),
            max_pages=int(args.max_pages),
            pagination_mode=str(args.pagination_mode),
            stop_on_short_page=bool(args.stop_on_short_page),
            allow_partial=bool(args.allow_partial),
            fail_if_incomplete=bool(args.fail_if_incomplete),
        )
        print_json_block(
            "OKX funding download summary:",
            {
                "dry_run": summary.get("dry_run"),
                "success": summary.get("success"),
                "funding_data_complete": summary.get("funding_data_complete"),
                "endpoint_history_limit_suspected": summary.get("endpoint_history_limit_suspected"),
                "partial_pagination_failed": summary.get("partial_pagination_failed"),
                "next_step": summary.get("next_step"),
                "downloads": summary.get("downloads"),
                "reports_dir": resolve_path(args.reports_dir),
            },
        )
        return 0 if summary.get("success") else 2
    except FundingDownloadError as exc:
        log_event(logger, logging.ERROR, "funding.download.config_error", str(exc))
        return 2
    except Exception:
        logger.exception("Unexpected funding download failure", extra={"event": "funding.download.unexpected"})
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
