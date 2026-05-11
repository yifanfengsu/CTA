#!/usr/bin/env python3
"""Audit OKX public derivatives data readiness for research-only trend confirmation."""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
from dataclasses import dataclass
from datetime import date, datetime, time as dt_time, timedelta, timezone
from pathlib import Path
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from common_runtime import PROJECT_ROOT, ensure_headless_runtime, print_json_block, to_jsonable


DEFAULT_INST_IDS = [
    "BTC-USDT-SWAP",
    "ETH-USDT-SWAP",
    "SOL-USDT-SWAP",
    "LINK-USDT-SWAP",
    "DOGE-USDT-SWAP",
]
DEFAULT_CCYS = ["BTC", "ETH", "SOL", "LINK", "DOGE"]
DEFAULT_START = "2023-01-01"
DEFAULT_END = "2026-03-31"
DEFAULT_TIMEZONE = "Asia/Shanghai"
DEFAULT_PROBE_DATE = "2026-03-01"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "reports" / "research" / "derivatives_data_readiness"
DEFAULT_FUNDING_DIR = PROJECT_ROOT / "data" / "funding" / "okx"
OKX_BASE_URL = "https://www.okx.com"
OKX_DOCS_URL = "https://www.okx.com/docs-v5/en/"
OKX_CHANGELOG_URL = "https://www.okx.com/docs-v5/log_en/"
REQUEST_TIMEOUT_SECONDS = 20

ENDPOINT_CSV_COLUMNS = [
    "endpoint_name",
    "endpoint_path",
    "endpoint_available",
    "auth_required",
    "request_params",
    "response_code",
    "response_ok",
    "row_count",
    "sample_fields",
    "first_timestamp",
    "last_timestamp",
    "period",
    "max_window_limit",
    "can_segment_download",
    "can_cover_2023_2026",
    "usable_for_research",
    "warning",
    "next_step",
    "failure_reason",
]

FEATURE_CSV_COLUMNS = [
    "tier",
    "feature_name",
    "feature_group",
    "data_source",
    "endpoint_name",
    "endpoint_path",
    "coverage_start",
    "coverage_end",
    "can_cover_2023_2026",
    "requires_segment_download",
    "requires_okx_historical_file",
    "requires_private_api_key",
    "usable_for_research",
    "status",
    "notes",
]

DOWNLOAD_PLAN_COLUMNS = [
    "feature_name",
    "endpoint_name",
    "endpoint_path",
    "download_method",
    "segment_unit",
    "suggested_period",
    "instruments_or_ccys",
    "start",
    "end",
    "requires_private_api_key",
    "requires_okx_historical_file",
    "prerequisite",
    "output_target",
    "warning",
]

UNAVAILABLE_COLUMNS = [
    "feature_name",
    "tier",
    "reason",
    "endpoint_name",
    "endpoint_path",
    "requires_private_api_key",
    "next_step",
]

TIMESTAMP_KEYS = (
    "ts",
    "fundingTime",
    "nextFundingTime",
    "begin",
    "end",
    "uTime",
    "cTime",
    "timestamp",
    "time",
)


class DerivativesDataAuditError(Exception):
    """Raised for invalid audit configuration."""


@dataclass(frozen=True, slots=True)
class HttpProbe:
    """One HTTP probe response."""

    status_code: int | str
    payload: dict[str, Any] | None
    raw_text: str
    error: str | None = None


@dataclass(frozen=True, slots=True)
class EndpointSpec:
    """Static metadata and param builders for one OKX endpoint probe."""

    name: str
    path: str
    feature_group: str
    recent_params_builder: Callable[[str, str, str, str, str, str], dict[str, str]]
    period: str
    max_window_limit: str
    can_segment_by_api: bool
    next_step_if_covered: str
    next_step_if_not_covered: str
    coverage_params_builder: Callable[[str, str, str, str, str, str], dict[str, str]] | None = None
    docs_note: str = ""
    docs_added_date: str = ""
    auth_required: bool = False


Requester = Callable[[str, dict[str, str], int], HttpProbe]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description="Audit OKX derivatives data readiness without downloading history.")
    parser.add_argument("--inst-ids", default=",".join(DEFAULT_INST_IDS))
    parser.add_argument("--ccys", default=",".join(DEFAULT_CCYS))
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--end", default=DEFAULT_END)
    parser.add_argument("--timezone", default=DEFAULT_TIMEZONE)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--probe-date", default=DEFAULT_PROBE_DATE)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--throttle-seconds", type=float, default=0.35)
    parser.add_argument("--json", action="store_true", help="Print the full JSON payload.")
    return parser.parse_args(argv)


def resolve_path(path_arg: str | Path) -> Path:
    """Resolve a path relative to project root."""

    path = Path(path_arg)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def parse_csv_list(raw_value: str | list[str] | tuple[str, ...], option_name: str) -> list[str]:
    """Parse comma/space separated CLI values while preserving order."""

    if isinstance(raw_value, (list, tuple)):
        tokens = [str(item).strip() for item in raw_value]
    else:
        tokens = [token.strip() for token in re.split(r"[\s,]+", str(raw_value))]
    values = [token for token in tokens if token]
    if not values:
        raise DerivativesDataAuditError(f"{option_name} must contain at least one value")
    return list(dict.fromkeys(values))


def parse_date_arg(value: str) -> date:
    """Parse a YYYY-MM-DD date."""

    try:
        return date.fromisoformat(value.strip())
    except ValueError as exc:
        raise DerivativesDataAuditError(f"invalid date: {value!r}") from exc


def resolve_timezone(timezone_name: str) -> ZoneInfo:
    """Resolve a timezone name."""

    try:
        return ZoneInfo(timezone_name)
    except ZoneInfoNotFoundError as exc:
        raise DerivativesDataAuditError(f"invalid timezone: {timezone_name}") from exc


def date_to_ms(value: str, timezone_name: str, *, days_offset: int = 0) -> str:
    """Convert a date at local midnight to UTC milliseconds."""

    tz = resolve_timezone(timezone_name)
    parsed = parse_date_arg(value) + timedelta(days=days_offset)
    dt_value = datetime.combine(parsed, dt_time.min).replace(tzinfo=tz)
    return str(int(dt_value.astimezone(timezone.utc).timestamp() * 1000))


def day_window_ms(value: str, timezone_name: str) -> tuple[str, str]:
    """Return begin/end millisecond strings for one local calendar day."""

    return date_to_ms(value, timezone_name), date_to_ms(value, timezone_name, days_offset=1)


def index_inst_id_from_ccy(ccy: str) -> str:
    """Build an OKX USDT index id from a base currency."""

    return f"{ccy}-USDT"


def build_url(path: str, params: dict[str, str]) -> str:
    """Build an OKX public REST URL."""

    return f"{OKX_BASE_URL}{path}?{urlencode(params)}"


def request_okx_json(path: str, params: dict[str, str], timeout: int = REQUEST_TIMEOUT_SECONDS) -> HttpProbe:
    """Request one OKX JSON endpoint without authentication."""

    url = build_url(path, params)
    request = Request(url, headers={"User-Agent": "cta-derivatives-readiness-audit/1.0"})
    response_text = ""
    try:
        with urlopen(request, timeout=timeout) as response:
            response_text = response.read().decode("utf-8", errors="replace")
            status_code = int(getattr(response, "status", response.getcode()))
    except HTTPError as exc:
        response_text = exc.read().decode("utf-8", errors="replace")
        return HttpProbe(status_code=exc.code, payload=parse_json_object(response_text), raw_text=response_text, error=str(exc))
    except (URLError, TimeoutError, OSError) as exc:
        return HttpProbe(status_code=0, payload=None, raw_text=response_text, error=f"{exc.__class__.__name__}: {exc}")

    return HttpProbe(status_code=status_code, payload=parse_json_object(response_text), raw_text=response_text, error=None)


def parse_json_object(text: str) -> dict[str, Any] | None:
    """Parse JSON text into an object if possible."""

    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def funding_recent_params(inst_id: str, _ccy: str, probe_date: str, _start: str, _end: str, timezone_name: str) -> dict[str, str]:
    """Build a small funding history request near the probe date."""

    return {
        "instId": inst_id,
        "after": date_to_ms(probe_date, timezone_name, days_offset=2),
        "limit": "5",
    }


def funding_coverage_params(inst_id: str, _ccy: str, _probe_date: str, start: str, _end: str, timezone_name: str) -> dict[str, str]:
    """Build a small funding history request near the requested start."""

    return {
        "instId": inst_id,
        "after": date_to_ms(start, timezone_name, days_offset=3),
        "limit": "5",
    }


def open_interest_params(inst_id: str, _ccy: str, _probe_date: str, _start: str, _end: str, _timezone_name: str) -> dict[str, str]:
    """Build current open interest params."""

    return {"instType": "SWAP", "instId": inst_id}


def mark_price_params(inst_id: str, _ccy: str, _probe_date: str, _start: str, _end: str, _timezone_name: str) -> dict[str, str]:
    """Build current mark price params."""

    return {"instType": "SWAP", "instId": inst_id}


def mark_price_history_params(inst_id: str, _ccy: str, probe_date: str, _start: str, _end: str, timezone_name: str) -> dict[str, str]:
    """Build a small mark-price candle history request near the probe date."""

    return {
        "instId": inst_id,
        "bar": "1D",
        "after": date_to_ms(probe_date, timezone_name, days_offset=2),
        "limit": "5",
    }


def mark_price_history_coverage_params(
    inst_id: str,
    _ccy: str,
    _probe_date: str,
    start: str,
    _end: str,
    timezone_name: str,
) -> dict[str, str]:
    """Build a small mark-price candle history request near the requested start."""

    return {
        "instId": inst_id,
        "bar": "1D",
        "after": date_to_ms(start, timezone_name, days_offset=5),
        "limit": "5",
    }


def index_ticker_params(_inst_id: str, ccy: str, _probe_date: str, _start: str, _end: str, _timezone_name: str) -> dict[str, str]:
    """Build current index ticker params."""

    return {"instId": index_inst_id_from_ccy(ccy)}


def index_history_params(_inst_id: str, ccy: str, probe_date: str, _start: str, _end: str, timezone_name: str) -> dict[str, str]:
    """Build a small index candle history request near the probe date."""

    return {
        "instId": index_inst_id_from_ccy(ccy),
        "bar": "1D",
        "after": date_to_ms(probe_date, timezone_name, days_offset=2),
        "limit": "5",
    }


def index_history_coverage_params(
    _inst_id: str,
    ccy: str,
    _probe_date: str,
    start: str,
    _end: str,
    timezone_name: str,
) -> dict[str, str]:
    """Build a small index candle history request near the requested start."""

    return {
        "instId": index_inst_id_from_ccy(ccy),
        "bar": "1D",
        "after": date_to_ms(start, timezone_name, days_offset=5),
        "limit": "5",
    }


def rubik_stats_params(_inst_id: str, ccy: str, probe_date: str, _start: str, _end: str, timezone_name: str) -> dict[str, str]:
    """Build one-day Trading Statistics params near the probe date."""

    begin, end = day_window_ms(probe_date, timezone_name)
    return {"ccy": ccy, "instType": "CONTRACTS", "begin": begin, "end": end, "period": "1D"}


def rubik_stats_coverage_params(_inst_id: str, ccy: str, _probe_date: str, start: str, _end: str, timezone_name: str) -> dict[str, str]:
    """Build one-day Trading Statistics params near the requested start."""

    begin, end = day_window_ms(start, timezone_name)
    return {"ccy": ccy, "instType": "CONTRACTS", "begin": begin, "end": end, "period": "1D"}


def rubik_contract_params(_inst_id: str, ccy: str, probe_date: str, _start: str, _end: str, timezone_name: str) -> dict[str, str]:
    """Build one-day contract Trading Statistics params near the probe date."""

    begin, end = day_window_ms(probe_date, timezone_name)
    return {"ccy": ccy, "begin": begin, "end": end, "period": "1D"}


def rubik_contract_coverage_params(_inst_id: str, ccy: str, _probe_date: str, start: str, _end: str, timezone_name: str) -> dict[str, str]:
    """Build one-day contract Trading Statistics params near the requested start."""

    begin, end = day_window_ms(start, timezone_name)
    return {"ccy": ccy, "begin": begin, "end": end, "period": "1D"}


def contract_oi_history_params(inst_id: str, _ccy: str, probe_date: str, _start: str, _end: str, timezone_name: str) -> dict[str, str]:
    """Build one-day contract open interest history params near the probe date."""

    begin, end = day_window_ms(probe_date, timezone_name)
    return {"instId": inst_id, "begin": begin, "end": end, "period": "1D"}


def contract_oi_history_coverage_params(inst_id: str, _ccy: str, _probe_date: str, start: str, _end: str, timezone_name: str) -> dict[str, str]:
    """Build one-day contract open interest history params near the requested start."""

    begin, end = day_window_ms(start, timezone_name)
    return {"instId": inst_id, "begin": begin, "end": end, "period": "1D"}


def premium_history_params(inst_id: str, _ccy: str, probe_date: str, _start: str, _end: str, timezone_name: str) -> dict[str, str]:
    """Build a small premium history request near the probe date."""

    return {
        "instId": inst_id,
        "after": date_to_ms(probe_date, timezone_name, days_offset=2),
        "limit": "5",
    }


def premium_history_coverage_params(
    inst_id: str,
    _ccy: str,
    _probe_date: str,
    start: str,
    _end: str,
    timezone_name: str,
) -> dict[str, str]:
    """Build a small premium history request near the requested start."""

    return {
        "instId": inst_id,
        "after": date_to_ms(start, timezone_name, days_offset=5),
        "limit": "5",
    }


def endpoint_specs() -> list[EndpointSpec]:
    """Return all endpoint probes in audit order."""

    contract_stats_note = (
        "OKX changelog lists contract open interest history, contract taker volume, "
        "and contract long/short ratio as new endpoints on 2024-06-13."
    )
    return [
        EndpointSpec(
            name="Open Interest",
            path="/api/v5/public/open-interest",
            feature_group="open_interest",
            recent_params_builder=open_interest_params,
            period="current",
            max_window_limit="current snapshot only",
            can_segment_by_api=False,
            next_step_if_covered="use only as current sanity check; historical OI needs another source",
            next_step_if_not_covered="use contract open interest history or historical files if available",
            docs_note="Public Data current open interest endpoint.",
        ),
        EndpointSpec(
            name="Funding Rate History",
            path="/api/v5/public/funding-rate-history",
            feature_group="funding",
            recent_params_builder=funding_recent_params,
            coverage_params_builder=funding_coverage_params,
            period="8h",
            max_window_limit="limit <= 400 rows per request",
            can_segment_by_api=True,
            next_step_if_covered="use existing local actual funding CSV; no new funding download needed",
            next_step_if_not_covered="import OKX Historical Market Data funding files",
            docs_note="Existing local actual OKX funding is treated as the source of truth for 2023-2026.",
        ),
        EndpointSpec(
            name="Mark Price",
            path="/api/v5/public/mark-price",
            feature_group="mark_index_basis",
            recent_params_builder=mark_price_params,
            period="current",
            max_window_limit="current snapshot only",
            can_segment_by_api=False,
            next_step_if_covered="use only as current sanity check",
            next_step_if_not_covered="use mark price candlestick history for 2023-2026 research",
            docs_note="Current mark price endpoint does not provide historical bars.",
        ),
        EndpointSpec(
            name="Mark Price Candles History",
            path="/api/v5/market/history-mark-price-candles",
            feature_group="mark_index_basis",
            recent_params_builder=mark_price_history_params,
            coverage_params_builder=mark_price_history_coverage_params,
            period="1D probe",
            max_window_limit="paginated candles; probe uses limit=5",
            can_segment_by_api=True,
            next_step_if_covered="segment download mark price candles",
            next_step_if_not_covered="import OKX Historical Market Data mark price files if available",
            docs_note="Public Data includes mark price candlesticks history.",
        ),
        EndpointSpec(
            name="Index Ticker",
            path="/api/v5/market/index-tickers",
            feature_group="mark_index_basis",
            recent_params_builder=index_ticker_params,
            period="current",
            max_window_limit="current snapshot only",
            can_segment_by_api=False,
            next_step_if_covered="use only as current sanity check",
            next_step_if_not_covered="use index candlestick history for 2023-2026 research",
            docs_note="Current index ticker endpoint does not provide historical bars.",
        ),
        EndpointSpec(
            name="Index Candles History",
            path="/api/v5/market/history-index-candles",
            feature_group="mark_index_basis",
            recent_params_builder=index_history_params,
            coverage_params_builder=index_history_coverage_params,
            period="1D probe",
            max_window_limit="paginated candles; probe uses limit=5",
            can_segment_by_api=True,
            next_step_if_covered="segment download index price candles",
            next_step_if_not_covered="import OKX Historical Market Data index price files if available",
            docs_note="Public Data includes index candlesticks history.",
        ),
        EndpointSpec(
            name="Taker Buy/Sell Volume",
            path="/api/v5/rubik/stat/taker-volume",
            feature_group="taker_flow",
            recent_params_builder=rubik_stats_params,
            coverage_params_builder=rubik_stats_coverage_params,
            period="1D probe",
            max_window_limit="begin/end + period; probe uses one day",
            can_segment_by_api=True,
            next_step_if_covered="segment download taker buy/sell volume",
            next_step_if_not_covered="find historical taker volume files or pause this feature",
            docs_note="Trading Statistics taker volume endpoint.",
        ),
        EndpointSpec(
            name="Contract Long/Short Account Ratio",
            path="/api/v5/rubik/stat/contracts/long-short-account-ratio",
            feature_group="long_short_ratio",
            recent_params_builder=rubik_contract_params,
            coverage_params_builder=rubik_contract_coverage_params,
            period="1D probe",
            max_window_limit="begin/end + period; probe uses one day",
            can_segment_by_api=True,
            next_step_if_covered="segment download contract long/short account ratio",
            next_step_if_not_covered="find historical long/short ratio files or pause this feature",
            docs_note=contract_stats_note,
            docs_added_date="2024-06-13",
        ),
        EndpointSpec(
            name="Contracts Open Interest and Volume",
            path="/api/v5/rubik/stat/contracts/open-interest-volume",
            feature_group="open_interest",
            recent_params_builder=rubik_contract_params,
            coverage_params_builder=rubik_contract_coverage_params,
            period="1D probe",
            max_window_limit="begin/end + period; probe uses one day",
            can_segment_by_api=True,
            next_step_if_covered="segment download contracts open interest and volume",
            next_step_if_not_covered="find historical OI/volume files or pause this feature",
            docs_note="Trading Statistics contracts open interest and volume endpoint.",
        ),
        EndpointSpec(
            name="Contract Open Interest History",
            path="/api/v5/rubik/stat/contracts/open-interest-history",
            feature_group="open_interest",
            recent_params_builder=contract_oi_history_params,
            coverage_params_builder=contract_oi_history_coverage_params,
            period="1D probe",
            max_window_limit="begin/end + period; probe uses one day",
            can_segment_by_api=True,
            next_step_if_covered="segment download contract open interest history",
            next_step_if_not_covered="find historical open interest files or pause this feature",
            docs_note=contract_stats_note,
            docs_added_date="2024-06-13",
        ),
        EndpointSpec(
            name="Premium History",
            path="/api/v5/public/premium-history",
            feature_group="mark_index_basis",
            recent_params_builder=premium_history_params,
            coverage_params_builder=premium_history_coverage_params,
            period="historical probe",
            max_window_limit="paginated history; probe uses limit=5",
            can_segment_by_api=True,
            next_step_if_covered="segment download premium history",
            next_step_if_not_covered="derive basis from mark/index history or import historical files",
            docs_note="Public Data lists premium history; swap coverage must be proven by probe.",
        ),
    ]


def datetime_from_ms(raw_value: Any) -> datetime | None:
    """Parse a millisecond timestamp-like value into UTC datetime."""

    text = str(raw_value or "").strip()
    if not text:
        return None
    if not re.fullmatch(r"-?\d+(\.\d+)?", text):
        return None
    try:
        numeric = int(float(text))
    except (TypeError, ValueError, OverflowError):
        return None
    if abs(numeric) < 10_000_000_000:
        numeric *= 1000
    try:
        return datetime.fromtimestamp(numeric / 1000.0, tz=timezone.utc)
    except (OSError, OverflowError, ValueError):
        return None


def extract_timestamps_from_item(item: Any) -> list[datetime]:
    """Extract timestamp fields from one OKX data row."""

    timestamps: list[datetime] = []
    if isinstance(item, dict):
        for key in TIMESTAMP_KEYS:
            parsed = datetime_from_ms(item.get(key))
            if parsed is not None:
                timestamps.append(parsed)
    elif isinstance(item, list) and item:
        parsed = datetime_from_ms(item[0])
        if parsed is not None:
            timestamps.append(parsed)
    return timestamps


def describe_sample_fields(data: list[Any]) -> str:
    """Describe fields from the first response row."""

    if not data:
        return ""
    sample = data[0]
    if isinstance(sample, dict):
        return ",".join(sorted(str(key) for key in sample.keys()))
    if isinstance(sample, list):
        return f"array_row_length={len(sample)}"
    return type(sample).__name__


def parse_probe_payload(payload: dict[str, Any] | None) -> dict[str, Any]:
    """Parse an OKX response payload into common probe fields."""

    if not isinstance(payload, dict):
        return {
            "okx_code": None,
            "okx_msg": "missing_or_invalid_json_payload",
            "row_count": 0,
            "sample_fields": "",
            "first_timestamp": None,
            "last_timestamp": None,
        }

    data = payload.get("data")
    rows = data if isinstance(data, list) else []
    timestamps: list[datetime] = []
    for row in rows:
        timestamps.extend(extract_timestamps_from_item(row))
    timestamps = sorted(set(timestamps))
    return {
        "okx_code": str(payload.get("code")) if payload.get("code") is not None else None,
        "okx_msg": str(payload.get("msg") or ""),
        "row_count": len(rows),
        "sample_fields": describe_sample_fields(rows),
        "first_timestamp": timestamps[0].isoformat() if timestamps else None,
        "last_timestamp": timestamps[-1].isoformat() if timestamps else None,
    }


def response_ok(status_code: int | str, parsed: dict[str, Any]) -> bool:
    """Return whether an HTTP/OKX response is successful."""

    try:
        status_int = int(status_code)
    except (TypeError, ValueError):
        return False
    code = parsed.get("okx_code")
    return 200 <= status_int < 300 and code == "0"


def looks_auth_required(http_result: HttpProbe, parsed: dict[str, Any]) -> bool:
    """Return whether a failed response looks like an auth problem."""

    try:
        status_int = int(http_result.status_code)
    except (TypeError, ValueError):
        status_int = 0
    if status_int in {401, 403}:
        return True
    text = " ".join(
        [
            str(http_result.error or ""),
            str(parsed.get("okx_msg") or ""),
            json.dumps(http_result.payload or {}, ensure_ascii=False),
        ]
    ).lower()
    return any(token in text for token in ("api key", "apikey", "signature", "login", "auth", "permission", "unauthorized"))


def one_probe_result(
    spec: EndpointSpec,
    params: dict[str, str],
    *,
    requester: Requester,
    dry_run: bool,
) -> tuple[HttpProbe, dict[str, Any], bool]:
    """Run or simulate one endpoint probe."""

    if dry_run:
        http_result = HttpProbe(status_code="dry_run", payload=None, raw_text="", error="dry_run:no_network_probe")
        parsed = parse_probe_payload(None)
        return http_result, parsed, False

    try:
        http_result = requester(spec.path, params, REQUEST_TIMEOUT_SECONDS)
    except Exception as exc:
        http_result = HttpProbe(status_code=0, payload=None, raw_text="", error=f"{exc.__class__.__name__}: {exc}")
    parsed = parse_probe_payload(http_result.payload)
    ok = response_ok(http_result.status_code, parsed)
    return http_result, parsed, ok


def local_funding_timestamp(row: dict[str, str]) -> datetime | None:
    """Parse a timestamp from one local funding CSV row."""

    raw_ms = str(row.get("funding_time") or "").strip()
    parsed_ms = datetime_from_ms(raw_ms)
    if parsed_ms is not None:
        return parsed_ms.replace(tzinfo=None)
    for column in ("funding_time_utc", "funding_time_local"):
        raw_iso = str(row.get(column) or "").strip()
        if not raw_iso:
            continue
        if raw_iso.endswith("Z"):
            raw_iso = raw_iso[:-1] + "+00:00"
        try:
            return datetime.fromisoformat(raw_iso).replace(tzinfo=None)
        except ValueError:
            continue
    return None


def inspect_one_funding_csv(path: Path, start: str, end: str) -> dict[str, Any]:
    """Inspect one canonical local funding CSV."""

    result: dict[str, Any] = {
        "path": str(path),
        "exists": path.exists(),
        "row_count": 0,
        "first_timestamp": None,
        "last_timestamp": None,
        "large_gap_count": 0,
        "coverage_complete": False,
        "warnings": [],
    }
    if not path.exists():
        result["warnings"].append("missing_funding_csv")
        return result

    timestamps: list[datetime] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            parsed = local_funding_timestamp(row)
            if parsed is not None:
                timestamps.append(parsed)

    unique_sorted = sorted(set(timestamps))
    result["row_count"] = len(timestamps)
    if not unique_sorted:
        result["warnings"].append("empty_or_unparseable_funding_csv")
        return result

    gaps = [
        right - left
        for left, right in zip(unique_sorted, unique_sorted[1:])
        if right - left > timedelta(hours=12)
    ]
    start_date = parse_date_arg(start)
    end_date = parse_date_arg(end)
    first_timestamp = unique_sorted[0]
    last_timestamp = unique_sorted[-1]
    covers_dates = first_timestamp.date() <= start_date and last_timestamp.date() >= end_date
    result.update(
        {
            "first_timestamp": first_timestamp.isoformat(sep=" ", timespec="seconds"),
            "last_timestamp": last_timestamp.isoformat(sep=" ", timespec="seconds"),
            "large_gap_count": len(gaps),
            "coverage_complete": bool(covers_dates and not gaps),
        }
    )
    if not covers_dates:
        result["warnings"].append("requested_window_not_covered")
    if gaps:
        result["warnings"].append(f"large_gap_count={len(gaps)}")
    return result


def inspect_local_funding(inst_ids: list[str], start: str, end: str, funding_dir: Path | None = None) -> dict[str, Any]:
    """Inspect already imported OKX Historical Market Data funding CSVs."""

    resolved_funding_dir = funding_dir or DEFAULT_FUNDING_DIR
    rows = []
    for inst_id in inst_ids:
        path = resolved_funding_dir / f"{inst_id}_funding_{start}_{end}.csv"
        row = inspect_one_funding_csv(path, start, end)
        row["inst_id"] = inst_id
        rows.append(row)
    return {
        "funding_source": "OKX Historical Market Data",
        "funding_dir": str(resolved_funding_dir),
        "records": rows,
        "funding_data_available": bool(rows and all(item["exists"] and item["row_count"] > 0 for item in rows)),
        "funding_data_complete": bool(rows and all(item["coverage_complete"] for item in rows)),
    }


def decide_can_cover(
    spec: EndpointSpec,
    *,
    recent_ok: bool,
    coverage_ok: bool,
    coverage_row_count: int,
    local_funding_complete: bool,
) -> bool:
    """Decide whether the data source can cover the requested 2023-2026 window."""

    if spec.feature_group == "funding" and local_funding_complete:
        return True
    if not recent_ok:
        return False
    if not spec.can_segment_by_api:
        return False
    return bool(coverage_ok and coverage_row_count > 0)


def probe_endpoint(
    spec: EndpointSpec,
    *,
    inst_id: str,
    ccy: str,
    start: str,
    end: str,
    timezone_name: str,
    probe_date: str,
    local_funding_complete: bool,
    requester: Requester = request_okx_json,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Probe one endpoint and return a normalized result row."""

    request_params = spec.recent_params_builder(inst_id, ccy, probe_date, start, end, timezone_name)
    http_result, parsed, ok = one_probe_result(spec, request_params, requester=requester, dry_run=dry_run)

    coverage_payload: dict[str, Any] | None = None
    coverage_ok = False
    coverage_row_count = 0
    if spec.coverage_params_builder is not None:
        coverage_params = spec.coverage_params_builder(inst_id, ccy, probe_date, start, end, timezone_name)
        coverage_http, coverage_parsed, coverage_ok = one_probe_result(
            spec,
            coverage_params,
            requester=requester,
            dry_run=dry_run,
        )
        coverage_row_count = int(coverage_parsed.get("row_count") or 0)
        coverage_payload = {
            "request_params": coverage_params,
            "response_code": coverage_http.status_code,
            "response_ok": coverage_ok,
            "row_count": coverage_row_count,
            "sample_fields": coverage_parsed.get("sample_fields") or "",
            "first_timestamp": coverage_parsed.get("first_timestamp"),
            "last_timestamp": coverage_parsed.get("last_timestamp"),
            "okx_code": coverage_parsed.get("okx_code"),
            "okx_msg": coverage_parsed.get("okx_msg"),
            "error": coverage_http.error,
        }

    can_cover = decide_can_cover(
        spec,
        recent_ok=ok,
        coverage_ok=coverage_ok,
        coverage_row_count=coverage_row_count,
        local_funding_complete=local_funding_complete,
    )
    endpoint_available = bool(ok)
    auth_required = bool(spec.auth_required or looks_auth_required(http_result, parsed))
    warnings: list[str] = []
    if dry_run:
        warnings.append("dry_run:no_network_probe")
    if http_result.error:
        warnings.append(str(http_result.error))
    if parsed.get("okx_msg"):
        warnings.append(str(parsed["okx_msg"]))
    if not spec.can_segment_by_api:
        warnings.append("current_snapshot_only")
    if spec.coverage_params_builder is not None and not can_cover and spec.feature_group != "funding":
        warnings.append("start_boundary_probe_failed_or_empty")
    if spec.docs_added_date and parse_date_arg(spec.docs_added_date) > parse_date_arg(start):
        warnings.append(f"docs_added_after_requested_start:{spec.docs_added_date}")
    if spec.feature_group == "funding" and local_funding_complete:
        warnings.append("local_actual_funding_complete_from_okx_historical_market_data")

    failure_reason = ""
    if not endpoint_available:
        failure_parts = [str(http_result.error or ""), str(parsed.get("okx_msg") or "")]
        failure_reason = "; ".join(part for part in failure_parts if part) or "endpoint_probe_not_successful"

    return {
        "endpoint_name": spec.name,
        "endpoint_path": spec.path,
        "endpoint_available": endpoint_available,
        "auth_required": auth_required,
        "request_params": request_params,
        "response_code": http_result.status_code,
        "response_ok": ok,
        "row_count": int(parsed.get("row_count") or 0),
        "sample_fields": parsed.get("sample_fields") or "",
        "first_timestamp": parsed.get("first_timestamp"),
        "last_timestamp": parsed.get("last_timestamp"),
        "period": request_params.get("period") or request_params.get("bar") or spec.period,
        "max_window_limit": spec.max_window_limit,
        "can_segment_download": bool(spec.can_segment_by_api and endpoint_available),
        "can_cover_2023_2026": can_cover,
        "usable_for_research": bool(can_cover and not auth_required),
        "warning": "; ".join(list(dict.fromkeys(warning for warning in warnings if warning))),
        "next_step": spec.next_step_if_covered if can_cover else spec.next_step_if_not_covered,
        "failure_reason": failure_reason,
        "okx_code": parsed.get("okx_code"),
        "okx_msg": parsed.get("okx_msg"),
        "docs_note": spec.docs_note,
        "docs_added_date": spec.docs_added_date,
        "coverage_probe": coverage_payload,
    }


def bool_text(value: Any) -> str:
    """Render bool-ish values as lower-case text."""

    return str(bool(value)).lower()


def endpoint_by_name(endpoint_results: list[dict[str, Any]], name: str) -> dict[str, Any] | None:
    """Find an endpoint result by display name."""

    return next((row for row in endpoint_results if row.get("endpoint_name") == name), None)


def endpoint_covers(endpoint_results: list[dict[str, Any]], *names: str) -> bool:
    """Return whether any named endpoint can cover 2023-2026 and is research usable."""

    for name in names:
        row = endpoint_by_name(endpoint_results, name)
        if row and bool(row.get("usable_for_research")):
            return True
    return False


def endpoint_available(endpoint_results: list[dict[str, Any]], *names: str) -> bool:
    """Return whether any named endpoint responded successfully."""

    for name in names:
        row = endpoint_by_name(endpoint_results, name)
        if row and bool(row.get("endpoint_available")):
            return True
    return False


def endpoint_path(endpoint_results: list[dict[str, Any]], name: str) -> str:
    """Return endpoint path by name."""

    row = endpoint_by_name(endpoint_results, name)
    return str(row.get("endpoint_path") or "") if row else ""


def feature_status_from_endpoint(
    endpoint_results: list[dict[str, Any]],
    endpoint_name: str,
    *,
    local_available: bool = False,
) -> tuple[bool, bool, str]:
    """Return endpoint availability, coverage and a compact status."""

    row = endpoint_by_name(endpoint_results, endpoint_name)
    available = bool(row and row.get("endpoint_available")) or local_available
    covers = bool(row and row.get("can_cover_2023_2026")) or local_available
    if covers:
        status = "available_2023_2026"
    elif available:
        status = "available_but_not_full_window"
    else:
        status = "unavailable_or_unconfirmed"
    return available, covers, status


def build_feature_rows(
    endpoint_results: list[dict[str, Any]],
    *,
    start: str,
    end: str,
    local_funding_complete: bool,
) -> list[dict[str, Any]]:
    """Build proposed feature tiers from probe results."""

    rows: list[dict[str, Any]] = []

    def add_row(
        tier: str,
        feature_name: str,
        feature_group: str,
        data_source: str,
        endpoint_name: str,
        *,
        covers: bool,
        requires_segment_download: bool,
        requires_okx_historical_file: bool,
        usable: bool,
        status: str,
        notes: str,
        private_key: bool = False,
    ) -> None:
        rows.append(
            {
                "tier": tier,
                "feature_name": feature_name,
                "feature_group": feature_group,
                "data_source": data_source,
                "endpoint_name": endpoint_name,
                "endpoint_path": endpoint_path(endpoint_results, endpoint_name) if endpoint_name else "",
                "coverage_start": start if covers else "",
                "coverage_end": end if covers else "",
                "can_cover_2023_2026": covers,
                "requires_segment_download": requires_segment_download,
                "requires_okx_historical_file": requires_okx_historical_file,
                "requires_private_api_key": private_key,
                "usable_for_research": usable,
                "status": status,
                "notes": notes,
            }
        )

    funding_available = bool(local_funding_complete or endpoint_available(endpoint_results, "Funding Rate History"))
    funding_covers = bool(local_funding_complete or endpoint_covers(endpoint_results, "Funding Rate History"))
    funding_status = "available_2023_2026" if funding_covers else "funding_history_incomplete_or_unconfirmed"
    for feature in ["actual funding rate", "funding dispersion", "funding sign breadth", "funding trend"]:
        add_row(
            "Tier 1",
            feature,
            "funding",
            "local OKX Historical Market Data funding CSV",
            "Funding Rate History",
            covers=funding_covers,
            requires_segment_download=False,
            requires_okx_historical_file=not funding_covers,
            usable=funding_covers,
            status=funding_status,
            notes="Existing actual OKX funding data is the source of truth for funding-derived features.",
        )

    mark_current_available, mark_current_covers, mark_status = feature_status_from_endpoint(endpoint_results, "Mark Price")
    mark_history_covers = endpoint_covers(endpoint_results, "Mark Price Candles History")
    add_row(
        "Tier 1",
        "mark price",
        "mark_index_basis",
        "OKX public mark price / mark price candle history",
        "Mark Price",
        covers=mark_history_covers,
        requires_segment_download=mark_current_available and not mark_current_covers,
        requires_okx_historical_file=mark_current_available and not mark_history_covers,
        usable=mark_history_covers,
        status="segment_download_ready" if mark_history_covers else ("available_current_only" if mark_current_available else mark_status),
        notes="Current mark price is directly available when probe succeeds; 2023-2026 research requires mark price candle history.",
    )

    index_current_available, index_current_covers, index_status = feature_status_from_endpoint(endpoint_results, "Index Ticker")
    index_history_covers = endpoint_covers(endpoint_results, "Index Candles History")
    add_row(
        "Tier 1",
        "index price",
        "mark_index_basis",
        "OKX public index ticker / index candle history",
        "Index Ticker",
        covers=index_history_covers,
        requires_segment_download=index_current_available and not index_current_covers,
        requires_okx_historical_file=index_current_available and not index_history_covers,
        usable=index_history_covers,
        status="segment_download_ready" if index_history_covers else ("available_current_only" if index_current_available else index_status),
        notes="Current index price is directly available when probe succeeds; 2023-2026 research requires index candle history.",
    )

    tier2_specs = [
        ("taker buy/sell volume", "taker_flow", "Taker Buy/Sell Volume"),
        ("contract long/short account ratio", "long_short_ratio", "Contract Long/Short Account Ratio"),
        ("contracts open interest and volume", "open_interest", "Contracts Open Interest and Volume"),
        ("open interest history", "open_interest", "Contract Open Interest History"),
        ("mark price candle history", "mark_index_basis", "Mark Price Candles History"),
        ("index price candle history", "mark_index_basis", "Index Candles History"),
    ]
    for feature_name, group, endpoint_name_value in tier2_specs:
        row = endpoint_by_name(endpoint_results, endpoint_name_value)
        available = bool(row and row.get("endpoint_available"))
        covers = bool(row and row.get("can_cover_2023_2026"))
        add_row(
            "Tier 2",
            feature_name,
            group,
            "OKX public API segmented download",
            endpoint_name_value,
            covers=covers,
            requires_segment_download=available,
            requires_okx_historical_file=available and not covers,
            usable=covers,
            status="segment_download_ready" if covers else ("recent_available_full_window_unconfirmed" if available else "unavailable"),
            notes=str(row.get("warning") or "") if row else "endpoint not probed",
        )

    premium_covers = endpoint_covers(endpoint_results, "Premium History")
    basis_from_mark_index = bool(mark_history_covers and index_history_covers)
    add_row(
        "Tier 3",
        "premium/basis historical",
        "mark_index_basis",
        "Premium History endpoint or mark-index derived proxy",
        "Premium History",
        covers=bool(premium_covers or basis_from_mark_index),
        requires_segment_download=endpoint_available(endpoint_results, "Premium History") or basis_from_mark_index,
        requires_okx_historical_file=not (premium_covers or basis_from_mark_index),
        usable=bool(premium_covers or basis_from_mark_index),
        status="derive_from_mark_index" if basis_from_mark_index and not premium_covers else ("segment_download_ready" if premium_covers else "historical_source_required"),
        notes="Use premium history if it covers swaps; otherwise derive a basis proxy from mark price minus index price.",
    )
    add_row(
        "Tier 3",
        "detailed liquidation data",
        "liquidation",
        "not confirmed in this audit",
        "",
        covers=False,
        requires_segment_download=False,
        requires_okx_historical_file=True,
        usable=False,
        status="unavailable_or_out_of_scope",
        notes="No multi-year public historical liquidation endpoint was accepted for this audit.",
    )
    add_row(
        "Tier 3",
        "private account level features",
        "private_account",
        "private account endpoints",
        "",
        covers=False,
        requires_segment_download=False,
        requires_okx_historical_file=False,
        private_key=True,
        usable=False,
        status="forbidden_private_key_required",
        notes="Private account features are excluded by the no-key research scope.",
    )

    return rows


def evaluate_decision(
    feature_rows: list[dict[str, Any]],
    endpoint_results: list[dict[str, Any]],
    *,
    local_funding_complete: bool,
) -> dict[str, Any]:
    """Evaluate the Derivatives-confirmed Trend Research readiness gate."""

    categories = {
        "funding": bool(local_funding_complete),
        "open_interest": endpoint_covers(
            endpoint_results,
            "Contracts Open Interest and Volume",
            "Contract Open Interest History",
        ),
        "taker_flow": endpoint_covers(endpoint_results, "Taker Buy/Sell Volume"),
        "long_short_ratio": endpoint_covers(endpoint_results, "Contract Long/Short Account Ratio"),
        "basis_premium": any(
            bool(row.get("usable_for_research"))
            for row in feature_rows
            if row.get("feature_name") == "premium/basis historical"
        ),
    }
    available_non_price_categories = [
        category
        for category in ["funding", "open_interest", "taker_flow", "long_short_ratio", "basis_premium"]
        if categories[category]
    ]
    at_least_three = len(available_non_price_categories) >= 3
    required_mix = bool(categories["funding"] and categories["open_interest"] and (categories["taker_flow"] or categories["long_short_ratio"]))
    no_private_key = not any(
        bool(row.get("requires_private_api_key")) and bool(row.get("usable_for_research"))
        for row in feature_rows
    )
    data_plan_executable = bool(at_least_three and required_mix and no_private_key)
    strategy_development_allowed = False
    demo_live_allowed = False
    can_enter = bool(
        at_least_three
        and required_mix
        and data_plan_executable
        and no_private_key
        and not strategy_development_allowed
        and not demo_live_allowed
    )

    blockers: list[str] = []
    if not at_least_three:
        blockers.append(
            f"non_price_derivatives_feature_categories_below_3:{len(available_non_price_categories)}"
        )
    if not categories["funding"]:
        blockers.append("funding_feature_missing")
    if not categories["open_interest"]:
        blockers.append("open_interest_feature_missing_or_not_2023_2026")
    if not (categories["taker_flow"] or categories["long_short_ratio"]):
        blockers.append("taker_flow_or_long_short_ratio_missing_or_not_2023_2026")
    if not no_private_key:
        blockers.append("private_api_key_required")
    if not data_plan_executable:
        blockers.append("derivatives_data_download_plan_not_executable")

    if can_enter:
        recommended_next_step = "download derivatives metrics"
    else:
        recommended_next_step = "pause research"

    return {
        "can_enter_derivatives_confirmed_trend_research": can_enter,
        "strategy_development_allowed": strategy_development_allowed,
        "demo_live_allowed": demo_live_allowed,
        "strategy_v3_allowed": False,
        "current_v3_family_failed": True,
        "current_v3_family_failed_after_actual_funding": True,
        "vsvcb_v1_phase1_failed": True,
        "no_policy_can_be_traded": True,
        "non_price_derivatives_feature_categories_available": available_non_price_categories,
        "non_price_derivatives_feature_category_count": len(available_non_price_categories),
        "required_feature_mix_ok": required_mix,
        "funding_available": categories["funding"],
        "open_interest_available": categories["open_interest"],
        "taker_flow_available": categories["taker_flow"],
        "long_short_ratio_available": categories["long_short_ratio"],
        "basis_premium_available": categories["basis_premium"],
        "no_private_api_key_required": no_private_key,
        "data_download_plan_executable": data_plan_executable,
        "blocking_reasons": list(dict.fromkeys(blockers)),
        "recommended_next_step": recommended_next_step,
    }


def build_download_plan(
    feature_rows: list[dict[str, Any]],
    endpoint_results: list[dict[str, Any]],
    *,
    inst_ids: list[str],
    ccys: list[str],
    start: str,
    end: str,
) -> list[dict[str, Any]]:
    """Build a conservative download/import plan for derivatives metrics."""

    rows: list[dict[str, Any]] = []
    for feature in feature_rows:
        if feature["tier"] == "Tier 1" and feature["feature_group"] == "funding":
            continue
        if not bool(feature.get("requires_segment_download")) and not bool(feature.get("requires_okx_historical_file")):
            continue
        endpoint_name_value = str(feature.get("endpoint_name") or "")
        endpoint_row = endpoint_by_name(endpoint_results, endpoint_name_value)
        endpoint_warning = str(endpoint_row.get("warning") or "") if endpoint_row else str(feature.get("notes") or "")
        method = "segmented_public_api" if bool(feature.get("usable_for_research")) else "historical_file_or_manual_source_audit"
        if feature.get("feature_name") in {"mark price", "index price"} and not bool(feature.get("usable_for_research")):
            method = "use_corresponding_candle_history_endpoint"
        rows.append(
            {
                "feature_name": feature["feature_name"],
                "endpoint_name": endpoint_name_value,
                "endpoint_path": feature.get("endpoint_path") or "",
                "download_method": method,
                "segment_unit": "daily_or_monthly_chunks",
                "suggested_period": "1D for audit baseline; refine before modeling",
                "instruments_or_ccys": ",".join(ccys if feature["feature_group"] in {"taker_flow", "long_short_ratio", "open_interest"} else inst_ids),
                "start": start,
                "end": end,
                "requires_private_api_key": bool_text(feature.get("requires_private_api_key")),
                "requires_okx_historical_file": bool_text(feature.get("requires_okx_historical_file")),
                "prerequisite": "endpoint start-boundary probe must pass before bulk download",
                "output_target": "data/derivatives/okx/<feature>/",
                "warning": endpoint_warning,
            }
        )
    return rows


def build_unavailable_features(feature_rows: list[dict[str, Any]], endpoint_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build unavailable feature rows."""

    unavailable: list[dict[str, Any]] = []
    for feature in feature_rows:
        if bool(feature.get("usable_for_research")):
            continue
        reason = str(feature.get("status") or "not_usable_for_research")
        if feature.get("endpoint_name"):
            endpoint_row = endpoint_by_name(endpoint_results, str(feature["endpoint_name"]))
            if endpoint_row and endpoint_row.get("failure_reason"):
                reason = str(endpoint_row["failure_reason"])
        unavailable.append(
            {
                "feature_name": feature["feature_name"],
                "tier": feature["tier"],
                "reason": reason,
                "endpoint_name": feature.get("endpoint_name") or "",
                "endpoint_path": feature.get("endpoint_path") or "",
                "requires_private_api_key": bool_text(feature.get("requires_private_api_key")),
                "next_step": "exclude from current research" if feature.get("requires_private_api_key") else "resolve coverage or keep unavailable",
            }
        )
    return unavailable


def csv_ready_row(row: dict[str, Any], columns: list[str]) -> dict[str, Any]:
    """Convert one row to flat CSV-safe values."""

    output: dict[str, Any] = {}
    for column in columns:
        value = row.get(column, "")
        if isinstance(value, (dict, list, tuple)):
            output[column] = json.dumps(to_jsonable(value), ensure_ascii=False, sort_keys=True)
        elif isinstance(value, bool):
            output[column] = bool_text(value)
        elif value is None:
            output[column] = ""
        else:
            output[column] = value
    return output


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    """Write CSV output."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(csv_ready_row(row, columns))


def endpoint_markdown_table(endpoint_results: list[dict[str, Any]]) -> str:
    """Render endpoint probe table for Markdown."""

    lines = [
        "| endpoint | available | auth_required | period | can_cover_2023_2026 | usable | warning |",
        "|---|---:|---:|---|---:|---:|---|",
    ]
    for row in endpoint_results:
        warning = str(row.get("warning") or "").replace("|", "/")
        lines.append(
            f"| {row['endpoint_name']} | {bool_text(row['endpoint_available'])} | "
            f"{bool_text(row['auth_required'])} | {row['period']} | "
            f"{bool_text(row['can_cover_2023_2026'])} | {bool_text(row['usable_for_research'])} | {warning} |"
        )
    return "\n".join(lines)


def bullet_names(rows: list[dict[str, Any]], *, tier: str | None = None, usable_only: bool | None = None) -> str:
    """Render feature names as bullets."""

    filtered = []
    for row in rows:
        if tier is not None and row.get("tier") != tier:
            continue
        if usable_only is not None and bool(row.get("usable_for_research")) != usable_only:
            continue
        filtered.append(row)
    if not filtered:
        return "- none"
    return "\n".join(
        f"- {row['feature_name']} ({row['status']})"
        for row in filtered
    )


def render_markdown_report(payload: dict[str, Any]) -> str:
    """Render the derivatives readiness report."""

    decision = payload["decision"]
    funding = payload["local_funding"]
    endpoint_results = payload["endpoint_probe_results"]
    feature_rows = payload["proposed_derivatives_features"]
    cannot_cover = [
        row["endpoint_name"]
        for row in endpoint_results
        if row.get("endpoint_available") and not row.get("can_cover_2023_2026")
    ]
    segmentable = [
        row["endpoint_name"]
        for row in endpoint_results
        if row.get("can_segment_download")
    ]
    historical_needed = [
        row["feature_name"]
        for row in feature_rows
        if row.get("requires_okx_historical_file") and not row.get("usable_for_research")
    ]
    fully_unavailable = [
        row["feature_name"]
        for row in feature_rows
        if row.get("status") in {"unavailable_or_out_of_scope", "forbidden_private_key_required"}
    ]

    answers = (
        "## Required Questions\n"
        "1. 当前不扩大币种的前提下，是否有足够衍生品数据继续趋势确认研究？\n"
        f"   - can_enter_derivatives_confirmed_trend_research={bool_text(decision['can_enter_derivatives_confirmed_trend_research'])}。\n"
        "2. 哪些数据可无密钥获取？\n"
        f"   - {', '.join([row['endpoint_name'] for row in endpoint_results if row.get('endpoint_available') and not row.get('auth_required')]) or 'none'}\n"
        "3. 哪些数据只支持近期，不能覆盖 2023-2026？\n"
        f"   - {', '.join(cannot_cover) or 'none'}\n"
        "4. 哪些数据可以通过分段 API 获取？\n"
        f"   - {', '.join(segmentable) or 'none'}\n"
        "5. 哪些数据需要 OKX Historical Market Data 文件？\n"
        f"   - {', '.join(historical_needed) or 'none'}\n"
        "6. 哪些数据完全不可用？\n"
        f"   - {', '.join(fully_unavailable) or 'none'}\n"
        "7. 推荐下一步是 download derivatives metrics / import historical files / pause research？\n"
        f"   - recommended_next_step={decision['recommended_next_step']}\n"
        "8. 是否允许进入 Derivatives-confirmed Trend Research？\n"
        f"   - can_enter_derivatives_confirmed_trend_research={bool_text(decision['can_enter_derivatives_confirmed_trend_research'])}\n"
        "9. 是否允许 Strategy V3 / demo / live？\n"
        "   - strategy_development_allowed=false\n"
        "   - demo_live_allowed=false\n"
    )

    blockers = "\n".join(f"- {item}" for item in decision["blocking_reasons"]) or "- none"
    return (
        "# OKX Derivatives Data Readiness Audit\n\n"
        "## Scope\n"
        f"- inst_ids={', '.join(payload['scope']['inst_ids'])}\n"
        f"- ccys={', '.join(payload['scope']['ccys'])}\n"
        f"- start={payload['scope']['start']}\n"
        f"- end={payload['scope']['end']}\n"
        f"- timezone={payload['scope']['timezone']}\n"
        f"- probe_date={payload['scope']['probe_date']}\n"
        "- audit_only=true\n"
        "- endpoint_probe_result_is_not_strategy_conclusion=true\n"
        "- strategy_development_allowed=false\n"
        "- demo_live_allowed=false\n\n"
        "## Local Funding Status\n"
        f"- funding_source={funding['funding_source']}\n"
        f"- funding_data_available={bool_text(funding['funding_data_available'])}\n"
        f"- funding_data_complete={bool_text(funding['funding_data_complete'])}\n\n"
        f"{answers}\n"
        "## Endpoint Probe Results\n"
        f"{endpoint_markdown_table(endpoint_results)}\n\n"
        "## Proposed Feature Tiers\n"
        "### Tier 1\n"
        f"{bullet_names(feature_rows, tier='Tier 1')}\n\n"
        "### Tier 2\n"
        f"{bullet_names(feature_rows, tier='Tier 2')}\n\n"
        "### Tier 3\n"
        f"{bullet_names(feature_rows, tier='Tier 3')}\n\n"
        "## Decision Gate\n"
        f"- non_price_derivatives_feature_category_count={decision['non_price_derivatives_feature_category_count']}\n"
        f"- non_price_derivatives_feature_categories_available={', '.join(decision['non_price_derivatives_feature_categories_available']) or 'none'}\n"
        f"- required_feature_mix_ok={bool_text(decision['required_feature_mix_ok'])}\n"
        f"- data_download_plan_executable={bool_text(decision['data_download_plan_executable'])}\n"
        f"- no_private_api_key_required={bool_text(decision['no_private_api_key_required'])}\n"
        f"- can_enter_derivatives_confirmed_trend_research={bool_text(decision['can_enter_derivatives_confirmed_trend_research'])}\n"
        "- strategy_development_allowed=false\n"
        "- demo_live_allowed=false\n"
        f"- recommended_next_step={decision['recommended_next_step']}\n\n"
        "## Blockers\n"
        f"{blockers}\n\n"
        "## Source Notes\n"
        f"- OKX API docs: {OKX_DOCS_URL}\n"
        f"- OKX changelog: {OKX_CHANGELOG_URL}\n"
        "- OKX changelog notes contract derivatives statistics endpoints were added on 2024-06-13; start-boundary probes decide whether 2023 coverage is actually usable.\n"
    )


def write_outputs(output_dir: Path, payload: dict[str, Any]) -> dict[str, str]:
    """Write all required report artifacts."""

    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "markdown_report": str(output_dir / "derivatives_data_readiness_report.md"),
        "json_report": str(output_dir / "derivatives_data_readiness.json"),
        "endpoint_probe_results_csv": str(output_dir / "endpoint_probe_results.csv"),
        "proposed_derivatives_features_csv": str(output_dir / "proposed_derivatives_features.csv"),
        "derivatives_data_download_plan_csv": str(output_dir / "derivatives_data_download_plan.csv"),
        "unavailable_derivatives_features_csv": str(output_dir / "unavailable_derivatives_features.csv"),
    }
    Path(paths["json_report"]).write_text(
        json.dumps(to_jsonable(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    Path(paths["markdown_report"]).write_text(render_markdown_report(payload), encoding="utf-8")
    write_csv(Path(paths["endpoint_probe_results_csv"]), payload["endpoint_probe_results"], ENDPOINT_CSV_COLUMNS)
    write_csv(Path(paths["proposed_derivatives_features_csv"]), payload["proposed_derivatives_features"], FEATURE_CSV_COLUMNS)
    write_csv(Path(paths["derivatives_data_download_plan_csv"]), payload["derivatives_data_download_plan"], DOWNLOAD_PLAN_COLUMNS)
    write_csv(Path(paths["unavailable_derivatives_features_csv"]), payload["unavailable_derivatives_features"], UNAVAILABLE_COLUMNS)
    return paths


def build_payload(
    *,
    inst_ids: list[str],
    ccys: list[str],
    start: str,
    end: str,
    timezone_name: str,
    probe_date: str,
    dry_run: bool,
    throttle_seconds: float,
    requester: Requester = request_okx_json,
) -> dict[str, Any]:
    """Build the complete audit payload."""

    if len(inst_ids) != len(ccys):
        raise DerivativesDataAuditError("--inst-ids and --ccys must contain the same number of values")
    if parse_date_arg(end) < parse_date_arg(start):
        raise DerivativesDataAuditError("--end must be on or after --start")
    parse_date_arg(probe_date)
    resolve_timezone(timezone_name)

    local_funding = inspect_local_funding(inst_ids, start, end)
    local_funding_complete = bool(local_funding["funding_data_complete"])
    probe_inst_id = inst_ids[0]
    probe_ccy = ccys[0]
    endpoint_results: list[dict[str, Any]] = []
    specs = endpoint_specs()
    for index, spec in enumerate(specs):
        result = probe_endpoint(
            spec,
            inst_id=probe_inst_id,
            ccy=probe_ccy,
            start=start,
            end=end,
            timezone_name=timezone_name,
            probe_date=probe_date,
            local_funding_complete=local_funding_complete,
            requester=requester,
            dry_run=dry_run,
        )
        endpoint_results.append(result)
        if not dry_run and index < len(specs) - 1:
            time.sleep(max(throttle_seconds, 0.0))

    feature_rows = build_feature_rows(
        endpoint_results,
        start=start,
        end=end,
        local_funding_complete=local_funding_complete,
    )
    download_plan = build_download_plan(
        feature_rows,
        endpoint_results,
        inst_ids=inst_ids,
        ccys=ccys,
        start=start,
        end=end,
    )
    unavailable_features = build_unavailable_features(feature_rows, endpoint_results)
    decision = evaluate_decision(
        feature_rows,
        endpoint_results,
        local_funding_complete=local_funding_complete,
    )
    return {
        "scope": {
            "inst_ids": inst_ids,
            "ccys": ccys,
            "start": start,
            "end": end,
            "timezone": timezone_name,
            "probe_date": probe_date,
            "dry_run": dry_run,
            "throttle_seconds": throttle_seconds,
        },
        "source_notes": {
            "okx_docs_url": OKX_DOCS_URL,
            "okx_changelog_url": OKX_CHANGELOG_URL,
            "audit_only": True,
            "endpoint_probe_result_is_not_strategy_conclusion": True,
        },
        "local_funding": local_funding,
        "endpoint_probe_results": endpoint_results,
        "proposed_derivatives_features": feature_rows,
        "derivatives_data_download_plan": download_plan,
        "unavailable_derivatives_features": unavailable_features,
        "decision": decision,
    }


def run_audit(
    *,
    inst_ids: list[str],
    ccys: list[str],
    start: str,
    end: str,
    timezone_name: str,
    output_dir: Path,
    probe_date: str,
    dry_run: bool = False,
    throttle_seconds: float = 0.35,
    requester: Requester = request_okx_json,
) -> dict[str, Any]:
    """Run the audit and write outputs."""

    payload = build_payload(
        inst_ids=inst_ids,
        ccys=ccys,
        start=start,
        end=end,
        timezone_name=timezone_name,
        probe_date=probe_date,
        dry_run=dry_run,
        throttle_seconds=throttle_seconds,
        requester=requester,
    )
    output_paths = write_outputs(output_dir, payload)
    payload["output_paths"] = output_paths
    return payload


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""

    args = parse_args(argv)
    ensure_headless_runtime()
    inst_ids = parse_csv_list(args.inst_ids, "--inst-ids")
    ccys = parse_csv_list(args.ccys, "--ccys")
    payload = run_audit(
        inst_ids=inst_ids,
        ccys=ccys,
        start=args.start,
        end=args.end,
        timezone_name=args.timezone,
        output_dir=resolve_path(args.output_dir),
        probe_date=args.probe_date,
        dry_run=bool(args.dry_run),
        throttle_seconds=float(args.throttle_seconds),
    )
    summary = {
        "can_enter_derivatives_confirmed_trend_research": payload["decision"][
            "can_enter_derivatives_confirmed_trend_research"
        ],
        "strategy_development_allowed": payload["decision"]["strategy_development_allowed"],
        "demo_live_allowed": payload["decision"]["demo_live_allowed"],
        "recommended_next_step": payload["decision"]["recommended_next_step"],
        "funding_data_complete": payload["local_funding"]["funding_data_complete"],
        "output_dir": str(resolve_path(args.output_dir)),
    }
    print_json_block("OKX derivatives data readiness summary:", payload if args.json else summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
