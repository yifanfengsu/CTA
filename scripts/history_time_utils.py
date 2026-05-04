#!/usr/bin/env python3
"""Shared time-range helpers for OKX history download, verify, and backtest."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time as dt_time, timedelta, timezone
from typing import Iterator
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


DEFAULT_TIMEZONE: str = "Asia/Shanghai"


@dataclass(frozen=True, slots=True)
class HistoryRange:
    """One normalized half-open history range."""

    start: datetime
    end_exclusive: datetime
    end_display: datetime
    timezone_name: str
    interval_delta: timedelta

    @property
    def start_utc(self) -> datetime:
        """Return the UTC representation of start."""

        return self.start.astimezone(timezone.utc)

    @property
    def end_exclusive_utc(self) -> datetime:
        """Return the UTC representation of end_exclusive."""

        return self.end_exclusive.astimezone(timezone.utc)

    @classmethod
    def from_bounds(
        cls,
        start: datetime,
        end_exclusive: datetime,
        interval_delta: timedelta,
        timezone_name: str,
    ) -> "HistoryRange":
        """Construct a half-open range from normalized aware bounds."""

        tz = resolve_timezone(timezone_name)
        normalized_start = _coerce_datetime(start, tz)
        normalized_end_exclusive = _coerce_datetime(end_exclusive, tz)
        if normalized_end_exclusive <= normalized_start:
            raise ValueError(
                f"end_exclusive must be later than start: start={normalized_start}, "
                f"end_exclusive={normalized_end_exclusive}"
            )
        return cls(
            start=normalize_bar_datetime(normalized_start, timezone_name, interval_delta),
            end_exclusive=normalize_bar_datetime(normalized_end_exclusive, timezone_name, interval_delta),
            end_display=normalize_bar_datetime(normalized_end_exclusive, timezone_name, interval_delta) - interval_delta,
            timezone_name=timezone_name,
            interval_delta=interval_delta,
        )


@dataclass(frozen=True, slots=True)
class ChunkPlan:
    """One half-open chunk within a larger history range."""

    index: int
    start: datetime
    end_exclusive: datetime
    end_display: datetime
    timezone_name: str

    @property
    def start_utc(self) -> datetime:
        """Return chunk start in UTC."""

        return self.start.astimezone(timezone.utc)

    @property
    def end_exclusive_utc(self) -> datetime:
        """Return chunk end_exclusive in UTC."""

        return self.end_exclusive.astimezone(timezone.utc)


def resolve_timezone(timezone_name: str) -> ZoneInfo:
    """Resolve a timezone name into a ZoneInfo object."""

    try:
        return ZoneInfo(timezone_name)
    except ZoneInfoNotFoundError as exc:
        raise ValueError(f"Unknown timezone: {timezone_name}") from exc


def _coerce_datetime(value: datetime, tz: ZoneInfo) -> datetime:
    """Interpret naive datetimes in tz, or convert aware datetimes to tz."""

    if value.tzinfo is None:
        return value.replace(tzinfo=tz)
    return value.astimezone(tz)


def _normalize_floor(value: datetime, interval_delta: timedelta) -> datetime:
    """Floor a timezone-aware datetime to the bar boundary."""

    if interval_delta == timedelta(minutes=1):
        return value.replace(second=0, microsecond=0)
    if interval_delta == timedelta(hours=1):
        return value.replace(minute=0, second=0, microsecond=0)
    if interval_delta == timedelta(days=1):
        return value.replace(hour=0, minute=0, second=0, microsecond=0)
    if interval_delta == timedelta(weeks=1):
        day_floor = value.replace(hour=0, minute=0, second=0, microsecond=0)
        return day_floor - timedelta(days=day_floor.weekday())

    raise ValueError(f"Unsupported interval delta: {interval_delta}")


def normalize_bar_datetime(dt: datetime, timezone_name: str, interval_delta: timedelta) -> datetime:
    """Normalize any datetime into the configured timezone and bar boundary."""

    tz = resolve_timezone(timezone_name)
    localized = _coerce_datetime(dt, tz)
    return _normalize_floor(localized, interval_delta)


def _parse_date_only(value: str) -> date:
    """Parse a date-only CLI value."""

    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"Unable to parse date-only value: {value}") from exc


def _parse_datetime(value: str) -> datetime:
    """Parse an ISO datetime CLI value."""

    try:
        return datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"Unable to parse datetime value: {value}") from exc


def parse_history_range(
    start_arg: str,
    end_arg: str,
    interval_delta: timedelta,
    timezone_name: str,
) -> HistoryRange:
    """Parse CLI start/end arguments into one normalized half-open range."""

    start_text = start_arg.strip()
    end_text = end_arg.strip()
    if not start_text or not end_text:
        raise ValueError("History range start/end cannot be empty")

    tz = resolve_timezone(timezone_name)

    if len(start_text) == 10:
        start_value = datetime.combine(_parse_date_only(start_text), dt_time.min).replace(tzinfo=tz)
    else:
        start_value = _coerce_datetime(_parse_datetime(start_text), tz)
    normalized_start = normalize_bar_datetime(start_value, timezone_name, interval_delta)

    if len(end_text) == 10:
        end_exclusive = datetime.combine(
            _parse_date_only(end_text) + timedelta(days=1),
            dt_time.min,
        ).replace(tzinfo=tz)
    else:
        end_value = _coerce_datetime(_parse_datetime(end_text), tz)
        end_exclusive = normalize_bar_datetime(end_value, timezone_name, interval_delta) + interval_delta

    if end_exclusive <= normalized_start:
        raise ValueError(
            f"History range end must be later than start: start={normalized_start}, "
            f"end_exclusive={end_exclusive}"
        )

    return HistoryRange(
        start=normalized_start,
        end_exclusive=end_exclusive,
        end_display=end_exclusive - interval_delta,
        timezone_name=timezone_name,
        interval_delta=interval_delta,
    )


def build_half_open_chunks(history_range: HistoryRange, chunk_days: int) -> list[ChunkPlan]:
    """Split one history range into day-based half-open chunks."""

    if chunk_days <= 0:
        raise ValueError(f"chunk_days must be positive: {chunk_days}")

    chunk_span = timedelta(days=chunk_days)
    chunks: list[ChunkPlan] = []
    cursor = history_range.start
    index = 1

    while cursor < history_range.end_exclusive:
        chunk_end_exclusive = min(cursor + chunk_span, history_range.end_exclusive)
        chunks.append(
            ChunkPlan(
                index=index,
                start=cursor,
                end_exclusive=chunk_end_exclusive,
                end_display=chunk_end_exclusive - history_range.interval_delta,
                timezone_name=history_range.timezone_name,
            )
        )
        cursor = chunk_end_exclusive
        index += 1

    return chunks


def expected_bar_count(history_range: HistoryRange) -> int:
    """Return the expected number of bars in a history range."""

    return int((history_range.end_exclusive - history_range.start) / history_range.interval_delta)


def iter_expected_datetimes(history_range: HistoryRange) -> Iterator[datetime]:
    """Yield all expected bar timestamps within a half-open range."""

    current = history_range.start
    while current < history_range.end_exclusive:
        yield current
        current += history_range.interval_delta
