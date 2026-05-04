#!/usr/bin/env python3
"""Shared helpers for sqlite history coverage and repair planning."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Iterable

from common_runtime import PROJECT_ROOT
from history_time_utils import (
    HistoryRange,
    expected_bar_count,
    iter_expected_datetimes,
    normalize_bar_datetime,
)

if TYPE_CHECKING:
    from datetime import datetime, tzinfo, timedelta

    from vnpy.trader.constant import Exchange, Interval
    from vnpy.trader.database import BaseDatabase
    from vnpy.trader.object import BarData


@dataclass(frozen=True, slots=True)
class MissingRange:
    """One contiguous missing timestamp range, inclusive on both sides."""

    start: datetime
    end: datetime
    missing_count: int


@dataclass(frozen=True, slots=True)
class HistoryCoverageSummary:
    """Coverage summary for one requested history range."""

    total_count: int
    first_dt: datetime | None
    last_dt: datetime | None
    expected_count: int
    missing_count: int
    gap_count: int
    largest_gap: MissingRange | None
    missing_ranges: list[MissingRange]

    @property
    def is_complete(self) -> bool:
        """Return True when all expected timestamps exist."""

        return self.expected_count > 0 and self.missing_count == 0

    @property
    def is_empty(self) -> bool:
        """Return True when no bars were found."""

        return self.total_count == 0


def get_database_timezone() -> "tzinfo":
    """Return vnpy_sqlite's configured database timezone."""

    from vnpy_sqlite.sqlite_database import DB_TZ

    return DB_TZ


def parse_interval(interval_value: str) -> tuple["Interval", "timedelta"]:
    """Map CLI interval text to vn.py Interval and expected delta."""

    from datetime import timedelta

    from vnpy.trader.constant import Interval

    mapping: dict[str, tuple[Interval, timedelta]] = {
        "1m": (Interval.MINUTE, timedelta(minutes=1)),
        "1h": (Interval.HOUR, timedelta(hours=1)),
        "d": (Interval.DAILY, timedelta(days=1)),
        "w": (Interval.WEEKLY, timedelta(weeks=1)),
    }
    try:
        return mapping[interval_value]
    except KeyError as exc:
        raise ValueError(f"Unsupported interval: {interval_value}") from exc


def _build_missing_ranges(
    missing_times: list["datetime"],
    interval_delta: "timedelta",
) -> list[MissingRange]:
    """Compress missing timestamps into contiguous missing ranges."""

    if not missing_times:
        return []

    ranges: list[MissingRange] = []
    range_start = missing_times[0]
    previous = missing_times[0]
    count = 1

    for current in missing_times[1:]:
        if current - previous == interval_delta:
            previous = current
            count += 1
            continue

        ranges.append(MissingRange(start=range_start, end=previous, missing_count=count))
        range_start = current
        previous = current
        count = 1

    ranges.append(MissingRange(start=range_start, end=previous, missing_count=count))
    return ranges


def analyze_history_coverage(
    bars: Iterable["BarData"],
    history_range: HistoryRange,
) -> HistoryCoverageSummary:
    """Analyze database or downloaded bars against a half-open history range."""

    actual_times: set[datetime] = set()
    for bar in bars:
        normalized = normalize_bar_datetime(
            bar.datetime,
            timezone_name=history_range.timezone_name,
            interval_delta=history_range.interval_delta,
        )
        if history_range.start <= normalized < history_range.end_exclusive:
            actual_times.add(normalized)

    actual_sorted = sorted(actual_times)
    expected_times = list(iter_expected_datetimes(history_range))
    missing_times = [timestamp for timestamp in expected_times if timestamp not in actual_times]
    missing_ranges = _build_missing_ranges(missing_times, history_range.interval_delta)

    return HistoryCoverageSummary(
        total_count=len(actual_sorted),
        first_dt=actual_sorted[0] if actual_sorted else None,
        last_dt=actual_sorted[-1] if actual_sorted else None,
        expected_count=expected_bar_count(history_range),
        missing_count=len(missing_times),
        gap_count=len(missing_ranges),
        largest_gap=max(missing_ranges, key=lambda item: item.missing_count, default=None),
        missing_ranges=missing_ranges,
    )


def to_database_query_range(history_range: HistoryRange) -> tuple["datetime", "datetime"]:
    """Convert an aware half-open range into vnpy_sqlite query datetimes."""

    db_tz = get_database_timezone()
    query_start = history_range.start.astimezone(db_tz).replace(tzinfo=None)
    query_end = history_range.end_exclusive.astimezone(db_tz).replace(tzinfo=None)
    return query_start, query_end


def verify_database_coverage(
    symbol: str,
    exchange: "Exchange",
    interval: "Interval",
    history_range: HistoryRange,
    database: "BaseDatabase | None" = None,
) -> HistoryCoverageSummary:
    """Load sqlite bars and analyze them against a half-open history range."""

    from vnpy.trader.database import get_database

    db = database or get_database()
    query_start, query_end = to_database_query_range(history_range)
    bars = db.load_bar_data(symbol, exchange, interval, query_start, query_end)
    return analyze_history_coverage(bars, history_range)


def build_instrument_config_path(vt_symbol: str) -> Path:
    """Return the expected local config/instruments path for a vt_symbol."""

    symbol, separator, _exchange = vt_symbol.partition(".")
    if not separator or not symbol:
        raise ValueError(f"Invalid vt_symbol: {vt_symbol}")
    return PROJECT_ROOT / "config" / "instruments" / f"{symbol.lower()}.json"


def build_repair_command(
    vt_symbol: str,
    interval_value: str,
    start: str,
    end: str,
    timezone_name: str,
    chunk_days: int = 3,
    source: str = "auto",
) -> str:
    """Build the recommended repair command."""

    return (
        "python scripts/download_okx_history.py "
        f"--vt-symbol {vt_symbol} "
        f"--interval {interval_value} "
        f"--start {start} "
        f"--end {end} "
        f"--chunk-days {chunk_days} "
        f"--timezone {timezone_name} "
        "--resume "
        "--repair-missing "
        f"--source {source}"
    )
