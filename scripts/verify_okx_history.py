#!/usr/bin/env python3
"""Verify local sqlite history coverage for OKX backtests."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from common_runtime import (
    PROJECT_ROOT,
    ensure_headless_runtime,
    log_event,
    print_json_block,
    setup_logging,
    to_jsonable,
)
from history_time_utils import DEFAULT_TIMEZONE, HistoryRange, parse_history_range
from history_utils import (
    HistoryCoverageSummary,
    build_repair_command,
    parse_interval,
    verify_database_coverage,
)


DEFAULT_VT_SYMBOL: str = "BTCUSDT_SWAP_OKX.GLOBAL"


class ConfigurationError(Exception):
    """Raised when CLI arguments or runtime configuration are invalid."""


@dataclass(frozen=True, slots=True)
class VerificationResult:
    """Structured sqlite coverage verification result."""

    vt_symbol: str
    interval_value: str
    history_range: HistoryRange
    coverage: HistoryCoverageSummary
    repair_command: str

    @property
    def is_complete(self) -> bool:
        """Return True when the requested range is fully covered."""

        return self.coverage.is_complete

    @property
    def is_empty(self) -> bool:
        """Return True when no bars were found."""

        return self.coverage.is_empty


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description="Verify local OKX sqlite history coverage.")
    parser.add_argument("--vt-symbol", default=DEFAULT_VT_SYMBOL)
    parser.add_argument("--interval", default="1m", choices=("1m", "1h", "d", "w"))
    parser.add_argument("--start", default="2025-01-01")
    parser.add_argument("--end", default="2026-03-31")
    parser.add_argument("--timezone", default=DEFAULT_TIMEZONE)
    parser.add_argument("--strict", action="store_true", help="Exit with code 2 when gaps are found.")
    parser.add_argument(
        "--output-json",
        help="Optional JSON output path. Relative paths are resolved from project root.",
    )
    parser.add_argument(
        "--print-repair-command",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Print the recommended repair command. Default: enabled.",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def configure_sqlite_settings(logger: logging.Logger) -> None:
    """Programmatically force vn.py to use the project sqlite database."""

    from vnpy.trader.setting import SETTINGS

    SETTINGS["database.name"] = "sqlite"
    SETTINGS["database.database"] = "database.db"

    log_event(
        logger,
        logging.INFO,
        "verify.db_settings",
        "Configured vn.py database settings for sqlite",
        database_name=SETTINGS["database.name"],
        database_database=SETTINGS["database.database"],
    )


def verify_history_range(
    vt_symbol: str,
    interval_value: str,
    start_arg: str,
    end_arg: str,
    timezone_name: str,
    history_range: HistoryRange | None = None,
) -> VerificationResult:
    """Load sqlite bars and verify one requested range."""

    from vnpy.trader.utility import extract_vt_symbol

    interval, interval_delta = parse_interval(interval_value)
    effective_range = history_range or parse_history_range(
        start_arg=start_arg,
        end_arg=end_arg,
        interval_delta=interval_delta,
        timezone_name=timezone_name,
    )
    symbol, exchange = extract_vt_symbol(vt_symbol)
    coverage = verify_database_coverage(
        symbol=symbol,
        exchange=exchange,
        interval=interval,
        history_range=effective_range,
    )
    return VerificationResult(
        vt_symbol=vt_symbol,
        interval_value=interval_value,
        history_range=effective_range,
        coverage=coverage,
        repair_command=build_repair_command(
            vt_symbol=vt_symbol,
            interval_value=interval_value,
            start=start_arg,
            end=end_arg,
            timezone_name=timezone_name,
            chunk_days=3,
            source="auto",
        ),
    )


def build_payload(result: VerificationResult) -> dict[str, Any]:
    """Convert a verification result into a JSON-serializable payload."""

    history_range = result.history_range
    return {
        "vt_symbol": result.vt_symbol,
        "interval": result.interval_value,
        "timezone": history_range.timezone_name,
        "start": history_range.start,
        "end_exclusive": history_range.end_exclusive,
        "end_display": history_range.end_display,
        "start_utc": history_range.start_utc,
        "end_exclusive_utc": history_range.end_exclusive_utc,
        "total_count": result.coverage.total_count,
        "first_dt": result.coverage.first_dt,
        "last_dt": result.coverage.last_dt,
        "first_dt_utc": (
            result.coverage.first_dt.astimezone(history_range.start_utc.tzinfo)
            if result.coverage.first_dt is not None
            else None
        ),
        "last_dt_utc": (
            result.coverage.last_dt.astimezone(history_range.start_utc.tzinfo)
            if result.coverage.last_dt is not None
            else None
        ),
        "expected_count": result.coverage.expected_count,
        "missing_count": result.coverage.missing_count,
        "gap_count": result.coverage.gap_count,
        "largest_gap": (
            {
                "start": result.coverage.largest_gap.start,
                "end": result.coverage.largest_gap.end,
                "missing_count": result.coverage.largest_gap.missing_count,
            }
            if result.coverage.largest_gap is not None
            else None
        ),
        "missing_ranges": [
            {"start": item.start, "end": item.end, "missing_count": item.missing_count}
            for item in result.coverage.missing_ranges
        ],
        "repair_command": result.repair_command,
    }


def write_json(path_arg: str, payload: dict[str, Any]) -> Path:
    """Write JSON payload to disk."""

    output_path = Path(path_arg)
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(to_jsonable(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output_path


def main() -> int:
    """Run sqlite verification from CLI."""

    args = parse_args()
    ensure_headless_runtime()
    logger = setup_logging("verify_okx_history", verbose=args.verbose)

    try:
        configure_sqlite_settings(logger)
        result = verify_history_range(
            vt_symbol=args.vt_symbol,
            interval_value=args.interval,
            start_arg=args.start,
            end_arg=args.end,
            timezone_name=args.timezone,
        )
        payload = build_payload(result)
        if args.output_json:
            output_path = write_json(args.output_json, payload)
            payload["output_json"] = output_path

        print_json_block("History verification summary:", payload)
        if args.print_repair_command:
            print(result.repair_command)

        if result.is_complete:
            print("History verification passed")
            log_event(
                logger,
                logging.INFO,
                "verify.complete",
                "History verification passed",
                vt_symbol=args.vt_symbol,
                interval=args.interval,
                timezone=args.timezone,
                total_count=result.coverage.total_count,
            )
            return 0

        if result.is_empty:
            log_event(
                logger,
                logging.ERROR,
                "verify.no_data",
                "No local sqlite history found for the requested range",
                vt_symbol=args.vt_symbol,
                interval=args.interval,
                timezone=args.timezone,
            )
            return 3

        if args.strict:
            log_event(
                logger,
                logging.ERROR,
                "verify.incomplete",
                "Requested sqlite history range still has gaps",
                vt_symbol=args.vt_symbol,
                interval=args.interval,
                timezone=args.timezone,
                missing_count=result.coverage.missing_count,
                gap_count=result.coverage.gap_count,
                repair_command=result.repair_command,
            )
            return 2

        log_event(
            logger,
            logging.WARNING,
            "verify.partial",
            "History verification found gaps, but strict mode is disabled",
            vt_symbol=args.vt_symbol,
            interval=args.interval,
            timezone=args.timezone,
            missing_count=result.coverage.missing_count,
            gap_count=result.coverage.gap_count,
            repair_command=result.repair_command,
        )
        return 0
    except ValueError as exc:
        log_event(logger, logging.ERROR, "verify.config_error", str(exc))
        return 1
    except ConfigurationError as exc:
        log_event(logger, logging.ERROR, "verify.config_error", str(exc))
        return 1
    except Exception:
        logger.exception(
            "Unexpected error during sqlite history verification",
            extra={"event": "verify.unexpected_error"},
        )
        return 4


if __name__ == "__main__":
    raise SystemExit(main())
