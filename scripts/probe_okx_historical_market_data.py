#!/usr/bin/env python3
"""Probe OKX Historical Market Data funding file endpoint availability."""

from __future__ import annotations

import argparse
import json
import logging
import re
from datetime import date, datetime, time as dt_time, timedelta, timezone
from pathlib import Path
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from common_runtime import PROJECT_ROOT, ensure_headless_runtime, log_event, print_json_block, setup_logging, to_jsonable
from download_okx_funding_history import DEFAULT_END, DEFAULT_INST_IDS_ARG, DEFAULT_START, parse_inst_ids


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "reports" / "research" / "funding_endpoint_probe"
CHANGELOG_URL = "https://www.okx.com/docs-v5/log_en/"
API_DOC_URLS = [
    "https://app.okx.com/docs-v5/en/#public-data-rest-api-get-historical-market-data",
    "https://www.okx.com/docs-v5/en/#public-data-rest-api-get-historical-market-data",
]
HISTORICAL_DATA_PAGE_URL = "https://www.okx.com/en-ar/historical-data"
REQUEST_TIMEOUT_SECONDS = 20
DOWNLOAD_URL_KEYS = {"fileHref", "fileUrl", "downloadUrl", "url", "href", "link"}
OKX_HISTORICAL_DATA_TIMEZONE = timezone(timedelta(hours=8))
DEFAULT_FUNDING_MODULE = "3"

CHANGELOG_FACTS = {
    "historical_market_data_query_endpoint": {
        "date": "2025-09-02",
        "description": (
            "OKX changelog says a new public endpoint was added for batch retrieving historical market data; "
            "it supports trade history, candlestick, funding rate, and 50/400/5000-level orderbook modules with daily/monthly aggregation."
        ),
        "source": CHANGELOG_URL,
    },
    "borrowing_rate_module": {
        "date": "2026-04-10",
        "description": "OKX changelog says module 11 (Borrowing rate) was added for request parameter module.",
        "source": CHANGELOG_URL,
    },
}

ENDPOINT_FACTS = {
    "path": "/api/v5/public/market-data-history",
    "funding_module": "3",
    "date_aggregation_param": "dateAggrType",
    "instrument_params": "instType plus instFamilyList for non-SPOT instruments",
    "range_params": "begin/end Unix milliseconds, inclusive",
    "timestamp_timezone": "UTC+8 for modules 1, 2, 3, and 11",
    "max_query_range": "20 days for daily, 20 months for monthly",
    "response_kind": "JSON response with groupDetails[].url download links",
    "rate_limit": "5 requests per 2 seconds by IP",
    "auth_required": False,
    "source": "https://www.okx.com/docs-v5/en/#public-data-rest-api-get-historical-market-data",
}


class HistoricalMarketDataProbeError(Exception):
    """Raised for invalid probe configuration."""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description="Probe OKX historical market data endpoint for funding files.")
    parser.add_argument("--inst-ids", default=DEFAULT_INST_IDS_ARG)
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--end", default=DEFAULT_END)
    parser.add_argument("--data-type", default="funding")
    parser.add_argument("--aggregation", choices=("daily", "monthly"), default="monthly")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--endpoint-url")
    parser.add_argument("--module")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def resolve_path(path_arg: str | Path) -> Path:
    """Resolve a path relative to project root."""

    path = Path(path_arg)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def request_text(url: str, timeout: int = REQUEST_TIMEOUT_SECONDS) -> str:
    """Fetch text from a URL."""

    request = Request(url, headers={"User-Agent": "cta-strategy-funding-research/1.0"})
    with urlopen(request, timeout=timeout) as response:
        return response.read().decode("utf-8", errors="replace")


def request_json(url: str, timeout: int = REQUEST_TIMEOUT_SECONDS) -> dict[str, Any]:
    """Fetch JSON from a URL."""

    text = request_text(url, timeout=timeout)
    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise HistoricalMarketDataProbeError("probe endpoint response is not a JSON object")
    return payload


def parse_date_arg(value: str) -> date:
    """Parse YYYY-MM-DD."""

    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise HistoricalMarketDataProbeError(f"invalid date: {value}") from exc


def discover_endpoint_path_from_docs(text: str) -> str | None:
    """Discover the official endpoint path only if docs explicitly show it.

    The changelog confirms the endpoint exists, but the code only treats a path
    as confirmed when a `GET /api/v5/...` line appears near "Get historical
    market data". It intentionally avoids guessing endpoint paths.
    """

    match = re.search(r"Get historical market data(.{0,2500}?)(GET\s+(/api/v5/[A-Za-z0-9/_-]+))", text, flags=re.I | re.S)
    if match:
        return match.group(3)
    match = re.search(r"GET\s+(/api/v5/[A-Za-z0-9/_-]*historical[A-Za-z0-9/_-]*market[A-Za-z0-9/_-]*)", text, flags=re.I)
    if match:
        return match.group(1)
    return None


def discover_endpoint_path(
    *,
    docs_fetcher: Callable[[str], str] = request_text,
) -> tuple[str | None, list[dict[str, Any]]]:
    """Try to discover a formally documented endpoint path."""

    attempts: list[dict[str, Any]] = []
    for url in [CHANGELOG_URL, *API_DOC_URLS]:
        try:
            text = docs_fetcher(url)
            endpoint_path = discover_endpoint_path_from_docs(text)
            attempts.append({"url": url, "fetched": True, "endpoint_path": endpoint_path, "error": None})
            if endpoint_path:
                return endpoint_path, attempts
        except Exception as exc:
            attempts.append({"url": url, "fetched": False, "endpoint_path": None, "error": f"{exc.__class__.__name__}: {exc}"})
    return None, attempts


def recursive_find_download_urls(payload: Any) -> list[str]:
    """Extract download links from a nested JSON response."""

    urls: list[str] = []
    if isinstance(payload, dict):
        for key, value in payload.items():
            if key in DOWNLOAD_URL_KEYS and isinstance(value, str) and value.startswith(("http://", "https://")):
                urls.append(value)
            else:
                urls.extend(recursive_find_download_urls(value))
    elif isinstance(payload, list):
        for item in payload:
            urls.extend(recursive_find_download_urls(item))
    return list(dict.fromkeys(urls))


def looks_auth_required(payload: dict[str, Any] | None, error: Exception | None = None) -> bool:
    """Return whether a response/error appears to require authentication."""

    if isinstance(error, HTTPError) and error.code in {401, 403}:
        return True
    text = json.dumps(payload or {}, ensure_ascii=False).lower()
    return any(token in text for token in ["api key", "apikey", "login", "auth", "permission", "unauthorized"])


def build_probe_url(endpoint_url: str, params: dict[str, str]) -> str:
    """Build one probe URL."""

    separator = "&" if "?" in endpoint_url else "?"
    return f"{endpoint_url}{separator}{urlencode(params)}"


def normalize_module(data_type: str, module: str | None) -> str:
    """Normalize module aliases from CLI/research language to OKX module ids."""

    if module:
        normalized = str(module).strip().lower()
        if normalized == "funding":
            return DEFAULT_FUNDING_MODULE
        return str(module).strip()
    if data_type == "funding":
        return DEFAULT_FUNDING_MODULE
    return str(data_type)


def inst_type_from_inst_id(inst_id: str) -> str:
    """Infer OKX instrument type for the research instruments."""

    if inst_id.endswith("-SWAP"):
        return "SWAP"
    if inst_id.count("-") >= 2:
        return "FUTURES"
    return "SPOT"


def inst_family_from_inst_id(inst_id: str) -> str:
    """Return instrument family for SWAP/FUTURES ids, e.g. BTC-USDT-SWAP -> BTC-USDT."""

    if inst_id.endswith("-SWAP"):
        return inst_id[: -len("-SWAP")]
    parts = inst_id.split("-")
    if len(parts) >= 2:
        return "-".join(parts[:2])
    return inst_id


def okx_date_to_ms(value: str) -> str:
    """Convert a yyyy-mm-dd date to OKX historical-data milliseconds.

    Official docs specify that modules 1, 2, 3, and 11 parse begin/end dates in
    UTC+8. Funding rate is module 3.
    """

    parsed = parse_date_arg(value)
    timestamp = datetime.combine(parsed, dt_time.min, tzinfo=OKX_HISTORICAL_DATA_TIMEZONE)
    return str(int(timestamp.timestamp() * 1000))


def sample_probe_window(start: str, end: str, aggregation: str) -> tuple[str, str]:
    """Use a small in-range sample probe to avoid OKX max range limits."""

    start_date = parse_date_arg(start)
    end_date = parse_date_arg(end)
    if end_date < start_date:
        raise HistoricalMarketDataProbeError(f"end must be on or after start: start={start}, end={end}")
    if aggregation == "monthly":
        probe_start = max(start_date, end_date.replace(day=1))
        return probe_start.isoformat(), end_date.isoformat()
    return end_date.isoformat(), end_date.isoformat()


def build_probe_params(inst_id: str, start: str, end: str, data_type: str, aggregation: str, module: str | None) -> dict[str, str]:
    """Build official OKX historical market data query params.

    Official docs for `GET /api/v5/public/market-data-history` use:
    `module=3` for funding rate, `instType=SWAP`, `instFamilyList=BTC-USDT`,
    `dateAggrType=daily|monthly`, and inclusive `begin`/`end` timestamps in
    milliseconds. Module 3 timestamp dates are interpreted in UTC+8.
    """

    inst_type = inst_type_from_inst_id(inst_id)
    params = {
        "module": normalize_module(data_type, module),
        "instType": inst_type,
        "dateAggrType": aggregation,
        "begin": okx_date_to_ms(start),
        "end": okx_date_to_ms(end),
    }
    if inst_type == "SPOT":
        params["instIdList"] = inst_id
    else:
        params["instFamilyList"] = inst_family_from_inst_id(inst_id)
    return params


def recommended_next_step(summary: dict[str, Any]) -> str:
    """Return recommended next step from probe outcome."""

    if summary.get("can_auto_download"):
        return "run make download-funding-historical"
    if summary.get("endpoint_discovery_failed"):
        return "manual CSV import or pause funding-aware research until OKX publishes the endpoint path"
    if summary.get("auth_required"):
        return "do not write API keys; use manual CSV import or pause funding-aware research"
    return "manual CSV import or pause funding-aware research"


def run_probe(
    *,
    inst_ids: list[str],
    start: str,
    end: str,
    data_type: str,
    aggregation: str,
    output_dir: Path,
    dry_run: bool,
    endpoint_url: str | None = None,
    module: str | None = None,
    docs_fetcher: Callable[[str], str] = request_text,
    requester: Callable[[str], dict[str, Any]] = request_json,
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    """Run endpoint discovery and a small metadata probe."""

    parse_date_arg(start)
    parse_date_arg(end)
    endpoint_path = None
    discovery_attempts: list[dict[str, Any]] = []
    endpoint_discovery_failed = False
    if endpoint_url:
        endpoint_available_url = endpoint_url
        endpoint_path = endpoint_url
    else:
        endpoint_path, discovery_attempts = discover_endpoint_path(docs_fetcher=docs_fetcher)
        endpoint_discovery_failed = endpoint_path is None
        endpoint_available_url = f"https://www.okx.com{endpoint_path}" if endpoint_path else None

    probe_response: dict[str, Any] | None = None
    probe_error: str | None = None
    download_urls: list[str] = []
    auth_required = False
    endpoint_available = False
    funding_module_available = False
    aggregation_supported = False
    request_params: dict[str, str] | None = None
    request_url: str | None = None
    probe_start, probe_end = sample_probe_window(start, end, aggregation)

    if endpoint_available_url and not endpoint_discovery_failed:
        request_params = build_probe_params(inst_ids[0], probe_start, probe_end, data_type, aggregation, module)
        request_url = build_probe_url(endpoint_available_url, request_params)
        try:
            probe_response = requester(request_url)
            code = str(probe_response.get("code", ""))
            endpoint_available = code == "0"
            response_text = json.dumps(probe_response, ensure_ascii=False).lower()
            funding_module_available = endpoint_available and not any(
                token in response_text for token in ["invalid module", "unsupported module", "module not supported"]
            )
            aggregation_supported = endpoint_available and not any(
                token in response_text
                for token in ["invalid aggregation", "unsupported aggregation", "aggregation not supported"]
            )
            download_urls = recursive_find_download_urls(probe_response)
        except Exception as exc:
            probe_error = f"{exc.__class__.__name__}: {exc}"
            auth_required = looks_auth_required(None, exc)
            if logger:
                log_event(logger, logging.WARNING, "funding.probe.request_failed", "Historical market data probe failed", error=probe_error)
    else:
        probe_error = "endpoint path could not be confirmed from official docs"

    if probe_response is not None:
        auth_required = auth_required or looks_auth_required(probe_response)

    can_auto_download = bool(endpoint_available and funding_module_available and aggregation_supported and download_urls and not auth_required)
    can_cover_2023_2026 = bool(can_auto_download and parse_date_arg(start) >= date(2022, 3, 1) and parse_date_arg(end) <= date(2026, 3, 31))
    summary: dict[str, Any] = {
        "dry_run": dry_run,
        "inst_ids": inst_ids,
        "start": start,
        "end": end,
        "data_type": data_type,
        "aggregation": aggregation,
        "changelog_facts": CHANGELOG_FACTS,
        "endpoint_facts": ENDPOINT_FACTS,
        "historical_data_page_url": HISTORICAL_DATA_PAGE_URL,
        "endpoint_path": endpoint_path,
        "endpoint_url": endpoint_available_url,
        "endpoint_discovery_failed": endpoint_discovery_failed,
        "endpoint_available": endpoint_available,
        "funding_module_available": funding_module_available,
        "aggregation_supported": aggregation_supported,
        "auth_required": auth_required,
        "can_auto_download": can_auto_download,
        "can_cover_2023_2026": can_cover_2023_2026,
        "probe_start": probe_start,
        "probe_end": probe_end,
        "request_params": request_params,
        "request_url": request_url,
        "response_kind": "download_link" if download_urls else ("json_no_download_link" if probe_response is not None else "none"),
        "download_urls": download_urls,
        "probe_error": probe_error,
        "discovery_attempts": discovery_attempts,
    }
    summary["recommended_next_step"] = recommended_next_step(summary)
    write_outputs(output_dir, summary)
    return summary


def render_report(summary: dict[str, Any]) -> str:
    """Render Markdown probe report."""

    return (
        "# OKX Historical Market Data Probe\n\n"
        "## Changelog Audit\n"
        f"- 2025-09-02: {CHANGELOG_FACTS['historical_market_data_query_endpoint']['description']}\n"
        f"- 2026-04-10: {CHANGELOG_FACTS['borrowing_rate_module']['description']}\n"
        f"- changelog={CHANGELOG_URL}\n\n"
        "## Official Endpoint Semantics\n"
        f"- endpoint_path={ENDPOINT_FACTS['path']}\n"
        f"- request_parameters=module, instType, instIdList/instFamilyList, dateAggrType, begin, end\n"
        f"- funding_module={ENDPOINT_FACTS['funding_module']}\n"
        f"- aggregation=daily/monthly via dateAggrType; daily has module-specific limitations in OKX docs\n"
        f"- response={ENDPOINT_FACTS['response_kind']}\n"
        f"- auth_required={str(bool(ENDPOINT_FACTS['auth_required'])).lower()}\n"
        f"- rate_limit={ENDPOINT_FACTS['rate_limit']}\n"
        f"- timestamp_timezone={ENDPOINT_FACTS['timestamp_timezone']}\n"
        f"- max_query_range={ENDPOINT_FACTS['max_query_range']}\n\n"
        "## Probe Result\n"
        f"- endpoint_available={str(bool(summary.get('endpoint_available'))).lower()}\n"
        f"- endpoint_discovery_failed={str(bool(summary.get('endpoint_discovery_failed'))).lower()}\n"
        f"- endpoint_path={summary.get('endpoint_path') or ''}\n"
        f"- funding_module_available={str(bool(summary.get('funding_module_available'))).lower()}\n"
        f"- aggregation_supported={str(bool(summary.get('aggregation_supported'))).lower()}\n"
        f"- auth_required={str(bool(summary.get('auth_required'))).lower()}\n"
        f"- can_auto_download={str(bool(summary.get('can_auto_download'))).lower()}\n"
        f"- can_cover_2023_2026={str(bool(summary.get('can_cover_2023_2026'))).lower()}\n"
        f"- probe_start={summary.get('probe_start')}\n"
        f"- probe_end={summary.get('probe_end')}\n"
        f"- response_kind={summary.get('response_kind')}\n"
        f"- recommended_next_step={summary.get('recommended_next_step')}\n\n"
        "## Notes\n"
        "- If the official endpoint path cannot be confirmed, this probe does not guess candidate API URLs.\n"
        "- No API key is written or required by this script.\n"
        "- `funding_data_complete=false` must keep V3.1, Strategy V3, demo, and live gates closed.\n"
    )


def write_outputs(output_dir: Path, summary: dict[str, Any]) -> None:
    """Write probe JSON and Markdown outputs."""

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "okx_historical_market_data_probe.json").write_text(
        json.dumps(to_jsonable(summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "okx_historical_market_data_probe_report.md").write_text(render_report(summary), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""

    args = parse_args(argv)
    ensure_headless_runtime()
    logger = setup_logging("probe_okx_historical_market_data", verbose=args.verbose)
    try:
        summary = run_probe(
            inst_ids=parse_inst_ids(args.inst_ids),
            start=str(args.start),
            end=str(args.end),
            data_type=str(args.data_type),
            aggregation=str(args.aggregation),
            output_dir=resolve_path(args.output_dir),
            dry_run=bool(args.dry_run),
            endpoint_url=args.endpoint_url,
            module=args.module,
            logger=logger,
        )
        print_json_block(
            "OKX historical market data probe summary:",
            {
                "endpoint_available": summary.get("endpoint_available"),
                "funding_module_available": summary.get("funding_module_available"),
                "aggregation_supported": summary.get("aggregation_supported"),
                "auth_required": summary.get("auth_required"),
                "can_auto_download": summary.get("can_auto_download"),
                "can_cover_2023_2026": summary.get("can_cover_2023_2026"),
                "recommended_next_step": summary.get("recommended_next_step"),
            },
        )
        return 0
    except Exception as exc:
        log_event(logger, logging.ERROR, "funding.probe.error", "Probe failed", error_class=exc.__class__.__name__, error_message=str(exc))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
