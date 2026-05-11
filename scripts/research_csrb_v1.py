#!/usr/bin/env python3
"""Research-only CSRB-v1 event study and fixed-hold benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import re
import sqlite3
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from common_runtime import PROJECT_ROOT, ensure_headless_runtime, log_event, setup_logging, to_jsonable
from history_time_utils import HistoryRange, expected_bar_count, parse_history_range, resolve_timezone
from history_utils import get_database_timezone


DEFAULT_SYMBOLS = [
    "BTCUSDT_SWAP_OKX.GLOBAL",
    "ETHUSDT_SWAP_OKX.GLOBAL",
    "SOLUSDT_SWAP_OKX.GLOBAL",
]
CORE_SYMBOLS = set(DEFAULT_SYMBOLS)
DEFAULT_TIMEFRAMES = ["15m", "30m", "1h"]
TIMEFRAME_MINUTES = {"15m": 15, "30m": 30, "1h": 60}
DEFAULT_START = "2023-01-01"
DEFAULT_END = "2026-03-31"
DEFAULT_SESSION_TIMEZONE = "UTC"
DEFAULT_REPORT_TIMEZONE = "Asia/Shanghai"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "reports" / "research" / "csrb_v1"
DEFAULT_FUNDING_DIR = PROJECT_ROOT / "data" / "funding" / "okx"
DEFAULT_DATABASE_PATH = PROJECT_ROOT / ".vntrader" / "database.db"
DEFAULT_EVENT_HORIZONS = (4, 8, 16, 32)
GROUPS = ["A", "B", "C", "D", "E"]

SESSION_SPECS: dict[str, dict[str, Any]] = {
    "asia_to_europe": {
        "group": "B",
        "range_name": "Asia",
        "breakout_name": "Europe",
        "range_start_minute": 0,
        "range_end_minute": 7 * 60 + 59,
        "breakout_start_minute": 8 * 60,
        "breakout_end_minute": 11 * 60 + 59,
    },
    "europe_to_us": {
        "group": "C",
        "range_name": "Europe",
        "breakout_name": "US",
        "range_start_minute": 8 * 60,
        "range_end_minute": 12 * 60 + 59,
        "breakout_start_minute": 13 * 60,
        "breakout_end_minute": 17 * 60 + 59,
    },
}

BASE_EVENT_COLUMNS = [
    "event_id",
    "timestamp",
    "timestamp_report",
    "symbol",
    "inst_id",
    "timeframe",
    "group",
    "session_type",
    "source_session_type",
    "session_date",
    "direction",
    "bar_index",
    "range_start",
    "range_end",
    "range_high",
    "range_low",
    "range_width",
    "range_bar_count",
    "breakout_window_start",
    "breakout_window_end",
    "close",
    "atr_prev",
    "buffer_atr",
    "breakout_boundary",
    "entry_time",
    "entry_price",
    "control_key",
]

TRADE_COLUMNS = [
    "trade_id",
    "event_id",
    "symbol",
    "inst_id",
    "timeframe",
    "group",
    "session_type",
    "source_session_type",
    "session_date",
    "direction",
    "entry_time",
    "entry_price",
    "exit_time",
    "exit_price",
    "hold_bars",
    "gross_return",
    "no_cost_pnl",
    "fee_cost",
    "slippage_cost",
    "funding_count",
    "funding_pnl",
    "cost_aware_pnl",
    "funding_adjusted_pnl",
    "split",
]


class CsrbResearchError(Exception):
    """Raised when CSRB-v1 research cannot continue."""


@dataclass(frozen=True, slots=True)
class CsrbConfig:
    """CSRB-v1 fixed Phase 1 research parameters."""

    fixed_notional: float = 1000.0
    fee_bps_per_side: float = 5.0
    slippage_bps_per_side: float = 5.0
    buffer_atr: float = 0.25
    range_min_bars: int = 8
    hold_bars: int = 16
    event_horizons: tuple[int, ...] = DEFAULT_EVENT_HORIZONS
    atr_window: int = 14
    random_seed: int = 17


@dataclass(frozen=True, slots=True)
class TimeSplit:
    """One chronological split interval."""

    name: str
    start: pd.Timestamp
    end: pd.Timestamp


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(
        description="CSRB-v1 research-only event study and fixed-hold benchmark."
    )
    parser.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS))
    parser.add_argument("--timeframes", default=",".join(DEFAULT_TIMEFRAMES))
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--end", default=DEFAULT_END)
    parser.add_argument("--session-timezone", default=DEFAULT_SESSION_TIMEZONE)
    parser.add_argument("--report-timezone", default=DEFAULT_REPORT_TIMEZONE)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--funding-dir", default=str(DEFAULT_FUNDING_DIR))
    parser.add_argument("--database-path", default=str(DEFAULT_DATABASE_PATH))
    parser.add_argument("--fixed-notional", type=float, default=1000.0)
    parser.add_argument("--fee-bps-per-side", type=float, default=5.0)
    parser.add_argument("--slippage-bps-per-side", type=float, default=5.0)
    parser.add_argument("--buffer-atr", type=float, default=0.25)
    parser.add_argument("--range-min-bars", type=int, default=8)
    parser.add_argument("--hold-bars", type=int, default=16)
    parser.add_argument("--event-horizons", default="4,8,16,32")
    parser.add_argument("--data-check-strict", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def resolve_path(path_arg: str | Path) -> Path:
    """Resolve relative paths from the project root."""

    path = Path(path_arg)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def parse_csv_list(value: str | Iterable[str]) -> list[str]:
    """Parse comma/space separated text while preserving order."""

    if isinstance(value, str):
        tokens = re.split(r"[\s,]+", value)
    else:
        tokens = [str(item) for item in value]
    parsed: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        item = token.strip()
        if item and item not in seen:
            parsed.append(item)
            seen.add(item)
    return parsed


def parse_timeframes(value: str | Iterable[str]) -> list[str]:
    """Parse and validate requested timeframes."""

    timeframes = parse_csv_list(value)
    unsupported = [item for item in timeframes if item not in TIMEFRAME_MINUTES]
    if unsupported:
        raise CsrbResearchError(f"unsupported timeframes: {unsupported}")
    if not timeframes:
        raise CsrbResearchError("--timeframes must not be empty")
    return timeframes


def parse_event_horizons(value: str | Iterable[int]) -> tuple[int, ...]:
    """Parse positive integer event horizons."""

    if isinstance(value, str):
        raw = parse_csv_list(value)
    else:
        raw = [str(item) for item in value]
    horizons = tuple(sorted({int(item) for item in raw if str(item).strip()}))
    if not horizons or any(item <= 0 for item in horizons):
        raise CsrbResearchError("--event-horizons must contain positive integers")
    return horizons


def build_config(args: argparse.Namespace) -> CsrbConfig:
    """Build research config from parsed CLI args."""

    return CsrbConfig(
        fixed_notional=float(args.fixed_notional),
        fee_bps_per_side=float(args.fee_bps_per_side),
        slippage_bps_per_side=float(args.slippage_bps_per_side),
        buffer_atr=float(args.buffer_atr),
        range_min_bars=int(args.range_min_bars),
        hold_bars=int(args.hold_bars),
        event_horizons=parse_event_horizons(args.event_horizons),
    )


def validate_config(config: CsrbConfig) -> None:
    """Validate fixed research parameters."""

    if config.fixed_notional <= 0:
        raise CsrbResearchError("--fixed-notional must be positive")
    if config.fee_bps_per_side < 0 or config.slippage_bps_per_side < 0:
        raise CsrbResearchError("cost bps values must be non-negative")
    if config.buffer_atr < 0:
        raise CsrbResearchError("--buffer-atr must be non-negative")
    if config.range_min_bars <= 0:
        raise CsrbResearchError("--range-min-bars must be positive")
    if config.hold_bars <= 0:
        raise CsrbResearchError("--hold-bars must be positive")
    if config.atr_window <= 0:
        raise CsrbResearchError("atr_window must be positive")


def event_columns(config: CsrbConfig) -> list[str]:
    """Return stable event columns for the requested horizons."""

    return BASE_EVENT_COLUMNS + [f"future_return_{horizon}" for horizon in config.event_horizons]


def split_vt_symbol(vt_symbol: str) -> tuple[str, str]:
    """Split a vt_symbol into database symbol and exchange."""

    symbol, separator, exchange = str(vt_symbol).partition(".")
    if not separator or not symbol or not exchange:
        raise CsrbResearchError(f"invalid vt_symbol: {vt_symbol}")
    return symbol, exchange


def symbol_to_inst_id(vt_symbol: str) -> str:
    """Map a local vt_symbol to OKX instId."""

    symbol, _exchange = split_vt_symbol(vt_symbol)
    root = symbol.removesuffix("_OKX")
    pair = root[: -len("_SWAP")] if root.endswith("_SWAP") else root
    if pair.endswith("USDT"):
        return f"{pair[:-4]}-USDT-SWAP"
    return root.replace("_", "-")


def format_timestamp(value: Any, timezone_name: str | None = None) -> str | None:
    """Format a timestamp as ISO text for reports."""

    if value is None or pd.isna(value):
        return None
    timestamp = pd.Timestamp(value)
    if timezone_name is not None:
        if timestamp.tzinfo is None:
            timestamp = timestamp.tz_localize(timezone_name)
        else:
            timestamp = timestamp.tz_convert(timezone_name)
    return timestamp.isoformat()


def finite_float(value: Any, default: float = 0.0) -> float:
    """Return a finite float or default."""

    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if not np.isfinite(number):
        return default
    return number


def minute_of_day(value: Any) -> int:
    """Return minute-of-day for a timestamp."""

    timestamp = pd.Timestamp(value)
    return int(timestamp.hour * 60 + timestamp.minute)


def in_minute_window(value: Any, start_minute: int, end_minute: int) -> bool:
    """Return True when timestamp falls inside an inclusive intraday window."""

    minute = minute_of_day(value)
    return bool(start_minute <= minute <= end_minute)


def window_mask(frame: pd.DataFrame, start_minute: int, end_minute: int) -> pd.Series:
    """Build an inclusive intraday window mask using closed-bar timestamps."""

    minutes = frame["datetime"].map(minute_of_day)
    return (minutes >= start_minute) & (minutes <= end_minute)


def load_1m_bars_from_sqlite(
    vt_symbol: str,
    data_range: HistoryRange,
    database_path: Path,
    session_timezone: str,
) -> pd.DataFrame:
    """Load local vn.py sqlite 1m bars for one symbol."""

    if not database_path.exists():
        raise CsrbResearchError(f"database not found: {database_path}")

    db_tz = get_database_timezone()
    symbol, exchange = split_vt_symbol(vt_symbol)
    query_start = data_range.start.astimezone(db_tz).replace(tzinfo=None).isoformat(sep=" ", timespec="seconds")
    query_end = data_range.end_exclusive.astimezone(db_tz).replace(tzinfo=None).isoformat(sep=" ", timespec="seconds")
    sql = """
        select datetime,
               open_price as open,
               high_price as high,
               low_price as low,
               close_price as close,
               volume
          from dbbardata
         where symbol = ?
           and exchange = ?
           and interval = ?
           and datetime >= ?
           and datetime < ?
         order by datetime
    """
    with sqlite3.connect(database_path) as connection:
        frame = pd.read_sql_query(
            sql,
            connection,
            params=(symbol, exchange, "1m", query_start, query_end),
        )
    return normalize_1m_bars(frame, str(db_tz), session_timezone)


def normalize_1m_bars(frame: pd.DataFrame, source_timezone: str, target_timezone: str) -> pd.DataFrame:
    """Normalize injected or sqlite-loaded 1m OHLCV into the session timezone."""

    columns = ["datetime", "open", "high", "low", "close", "volume"]
    if frame.empty:
        return pd.DataFrame(columns=columns)
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise CsrbResearchError(f"1m bars missing columns: {missing}")

    normalized = frame.loc[:, columns].copy()
    timestamps = pd.to_datetime(normalized["datetime"], errors="coerce")
    if timestamps.isna().any():
        raise CsrbResearchError("1m bars contain unparsable datetime values")
    if timestamps.dt.tz is None:
        timestamps = timestamps.dt.tz_localize(source_timezone)
    timestamps = timestamps.dt.tz_convert(target_timezone)
    normalized["datetime"] = timestamps
    for column in ["open", "high", "low", "close", "volume"]:
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce")
    normalized = normalized.dropna(subset=["datetime", "open", "high", "low", "close", "volume"])
    normalized = normalized.sort_values("datetime", kind="stable").reset_index(drop=True)
    return normalized


def missing_ranges_from_index(missing: pd.DatetimeIndex, limit: int = 10) -> list[dict[str, Any]]:
    """Compress missing minute timestamps into a small report sample."""

    if len(missing) == 0:
        return []
    ranges: list[dict[str, Any]] = []
    start = missing[0]
    previous = missing[0]
    count = 1
    for current in missing[1:]:
        if current - previous == pd.Timedelta(minutes=1):
            previous = current
            count += 1
            continue
        ranges.append({"start": format_timestamp(start), "end": format_timestamp(previous), "missing_count": count})
        if len(ranges) >= limit:
            return ranges
        start = current
        previous = current
        count = 1
    ranges.append({"start": format_timestamp(start), "end": format_timestamp(previous), "missing_count": count})
    return ranges[:limit]


def analyze_1m_quality(
    vt_symbol: str,
    bars_1m: pd.DataFrame,
    data_range: HistoryRange,
    session_timezone: str,
) -> dict[str, Any]:
    """Build one-symbol 1m data quality summary."""

    start = pd.Timestamp(data_range.start).tz_convert(session_timezone)
    end_exclusive = pd.Timestamp(data_range.end_exclusive).tz_convert(session_timezone)
    expected = pd.date_range(start, end_exclusive - pd.Timedelta(minutes=1), freq="min")
    actual = (
        pd.DatetimeIndex(bars_1m["datetime"])
        if not bars_1m.empty
        else pd.DatetimeIndex([], tz=session_timezone)
    )
    duplicate_count = int(actual.duplicated().sum())
    unique_actual = pd.DatetimeIndex(actual.drop_duplicates())
    missing = expected.difference(unique_actual)
    unexpected = unique_actual.difference(expected)
    return {
        "symbol": vt_symbol,
        "timeframe": "1m",
        "expected_count": int(len(expected)),
        "row_count": int(len(bars_1m.index)),
        "unique_count": int(len(unique_actual)),
        "duplicate_count": duplicate_count,
        "missing_count": int(len(missing)),
        "unexpected_count": int(len(unexpected)),
        "first_datetime": format_timestamp(actual.min()) if len(actual) else None,
        "last_datetime": format_timestamp(actual.max()) if len(actual) else None,
        "complete": bool(len(missing) == 0 and duplicate_count == 0 and len(unexpected) == 0),
        "missing_ranges_sample": missing_ranges_from_index(missing),
    }


def resample_ohlcv_closed(
    bars_1m: pd.DataFrame,
    timeframe: str,
    data_range: HistoryRange | None = None,
    session_timezone: str = DEFAULT_SESSION_TIMEZONE,
) -> pd.DataFrame:
    """Resample 1m OHLCV into completed closed bars without lookahead."""

    if timeframe not in TIMEFRAME_MINUTES:
        raise CsrbResearchError(f"unsupported timeframe: {timeframe}")
    minutes = TIMEFRAME_MINUTES[timeframe]
    columns = ["open_time", "datetime", "open", "high", "low", "close", "volume"]
    if bars_1m.empty:
        return pd.DataFrame(columns=columns)

    working = bars_1m.sort_values("datetime", kind="stable").drop_duplicates("datetime", keep="last").copy()
    if data_range is None:
        anchor = pd.Timestamp(working["datetime"].iloc[0])
    else:
        anchor = pd.Timestamp(data_range.start).tz_convert(session_timezone)
    if anchor.tzinfo is None:
        anchor = anchor.tz_localize(session_timezone)
    deltas = (working["datetime"] - anchor) / pd.Timedelta(minutes=1)
    working = working.loc[deltas >= 0].copy()
    working["_slot"] = np.floor(deltas.loc[working.index].to_numpy(dtype=float) / minutes).astype(np.int64)
    grouped = working.groupby("_slot", sort=True, dropna=False)
    result = grouped.agg(
        open_time=("datetime", "min"),
        datetime=("datetime", "max"),
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
        minute_count=("datetime", "size"),
    )
    result = result[result["minute_count"] == minutes].copy()
    result = result.drop(columns=["minute_count"]).reset_index(drop=True)
    return result.loc[:, columns]


def analyze_timeframe_quality(
    vt_symbol: str,
    timeframe: str,
    bars: pd.DataFrame,
    one_minute_quality: dict[str, Any],
) -> dict[str, Any]:
    """Build resampled timeframe data quality summary."""

    minutes = TIMEFRAME_MINUTES[timeframe]
    expected_count = int(one_minute_quality["expected_count"] // minutes)
    return {
        "symbol": vt_symbol,
        "timeframe": timeframe,
        "expected_closed_bar_count": expected_count,
        "row_count": int(len(bars.index)),
        "first_datetime": format_timestamp(bars["datetime"].min()) if not bars.empty else None,
        "last_datetime": format_timestamp(bars["datetime"].max()) if not bars.empty else None,
        "complete": bool(one_minute_quality["complete"] and len(bars.index) == expected_count),
    }


def add_csrb_indicators(frame: pd.DataFrame, config: CsrbConfig) -> pd.DataFrame:
    """Add ATR and calendar helpers without lookahead."""

    out = frame.copy().reset_index(drop=True)
    high = pd.to_numeric(out["high"], errors="coerce")
    low = pd.to_numeric(out["low"], errors="coerce")
    close = pd.to_numeric(out["close"], errors="coerce")
    previous_close = close.shift(1)
    true_range = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = true_range.rolling(config.atr_window, min_periods=config.atr_window).mean()
    out["atr"] = atr
    out["atr_prev"] = atr.shift(1)
    out["open_next"] = out["open"].shift(-1)
    out["entry_next_open"] = out["open_next"]
    out["session_date"] = out["datetime"].map(lambda value: pd.Timestamp(value).date().isoformat())
    out["minute_of_day"] = out["datetime"].map(minute_of_day)
    return out


def directional_return(entry_price: float, exit_price: float, direction: str) -> float:
    """Return signed directional return."""

    if not np.isfinite(entry_price) or entry_price == 0 or not np.isfinite(exit_price):
        return np.nan
    sign = 1.0 if direction == "long" else -1.0
    return float(sign * (exit_price / entry_price - 1.0))


def future_returns_for_event(
    frame: pd.DataFrame,
    pos: int,
    direction: str,
    horizons: tuple[int, ...],
) -> dict[str, float]:
    """Calculate future close-to-close directional returns for event study output."""

    close = float(frame.iloc[pos]["close"])
    metrics: dict[str, float] = {}
    for horizon in horizons:
        value = np.nan
        if pos + horizon < len(frame.index):
            value = directional_return(close, float(frame.iloc[pos + horizon]["close"]), direction)
        metrics[f"future_return_{horizon}"] = value
    return metrics


def signed_funding_pnl(notional: float, funding_rate: float, direction: str) -> float:
    """Return signed funding PnL under the standard perpetual sign convention."""

    if direction == "short":
        return float(abs(notional) * funding_rate)
    return float(-abs(notional) * funding_rate)


def load_funding_csv(path: Path) -> pd.DataFrame:
    """Load one funding CSV with normalized UTC timestamps."""

    frame = pd.read_csv(path)
    if frame.empty:
        return pd.DataFrame(columns=["funding_time_utc", "funding_rate"])
    if "funding_time_utc" in frame.columns:
        timestamps = pd.to_datetime(frame["funding_time_utc"], utc=True, errors="coerce")
    elif "funding_time" in frame.columns:
        timestamps = pd.to_datetime(
            pd.to_numeric(frame["funding_time"], errors="coerce"),
            unit="ms",
            utc=True,
            errors="coerce",
        )
    else:
        raise CsrbResearchError(f"funding CSV missing funding_time columns: {path}")
    result = pd.DataFrame(
        {
            "funding_time_utc": timestamps,
            "funding_rate": pd.to_numeric(frame.get("funding_rate"), errors="coerce"),
        }
    )
    result = result.dropna(subset=["funding_time_utc", "funding_rate"])
    result = result.sort_values("funding_time_utc", kind="stable").drop_duplicates("funding_time_utc", keep="last")
    return result.reset_index(drop=True)


def select_funding_csv(funding_dir: Path, inst_id: str, start_arg: str, end_arg: str) -> Path | None:
    """Select the canonical funding CSV or the most recent matching fallback."""

    canonical = funding_dir / f"{inst_id}_funding_{start_arg}_{end_arg}.csv"
    if canonical.exists():
        return canonical
    matches = sorted(funding_dir.glob(f"{inst_id}_funding_*.csv"))
    return matches[-1] if matches else None


def analyze_funding_quality(
    funding_df: pd.DataFrame,
    data_range: HistoryRange,
    path: Path | None,
) -> dict[str, Any]:
    """Build one instrument funding quality record."""

    if path is None:
        return {
            "csv_path": None,
            "exists": False,
            "row_count": 0,
            "first_funding_time": None,
            "last_funding_time": None,
            "large_gap_count": 0,
            "coverage_complete": False,
            "warnings": ["missing_funding_csv"],
        }
    if funding_df.empty:
        return {
            "csv_path": str(path),
            "exists": True,
            "row_count": 0,
            "first_funding_time": None,
            "last_funding_time": None,
            "large_gap_count": 0,
            "coverage_complete": False,
            "warnings": ["empty_funding_csv"],
        }

    times = pd.DatetimeIndex(funding_df["funding_time_utc"])
    first_time = times.min()
    last_time = times.max()
    intervals = times.to_series().diff().dropna().dt.total_seconds() / 3600.0
    large_gap_count = int((intervals > 24.0).sum())
    start_utc = pd.Timestamp(data_range.start).tz_convert("UTC")
    end_utc = pd.Timestamp(data_range.end_exclusive).tz_convert("UTC")
    warnings: list[str] = []
    starts_before_window = bool(first_time <= start_utc + pd.Timedelta(hours=8))
    ends_after_last_required_funding = bool(last_time >= end_utc - pd.Timedelta(hours=8))
    if not starts_before_window:
        warnings.append("funding_starts_after_window_start")
    if not ends_after_last_required_funding:
        warnings.append("funding_ends_before_window_end")
    if large_gap_count:
        warnings.append(f"large_gap_count={large_gap_count}")
    return {
        "csv_path": str(path),
        "exists": True,
        "row_count": int(len(funding_df.index)),
        "first_funding_time": format_timestamp(first_time),
        "last_funding_time": format_timestamp(last_time),
        "large_gap_count": large_gap_count,
        "coverage_complete": bool(starts_before_window and ends_after_last_required_funding and large_gap_count == 0),
        "warnings": warnings,
    }


def load_funding_histories(
    funding_dir: Path,
    symbols: list[str],
    data_range: HistoryRange,
    start_arg: str,
    end_arg: str,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    """Load funding histories and return quality details."""

    histories: dict[str, pd.DataFrame] = {}
    records: dict[str, Any] = {}
    for symbol in symbols:
        inst_id = symbol_to_inst_id(symbol)
        path = select_funding_csv(funding_dir, inst_id, start_arg, end_arg)
        if path is None:
            histories[inst_id] = pd.DataFrame(columns=["funding_time_utc", "funding_rate"])
            records[inst_id] = analyze_funding_quality(histories[inst_id], data_range, None)
            continue
        try:
            histories[inst_id] = load_funding_csv(path)
        except Exception as exc:
            histories[inst_id] = pd.DataFrame(columns=["funding_time_utc", "funding_rate"])
            records[inst_id] = {
                "csv_path": str(path),
                "exists": True,
                "row_count": 0,
                "first_funding_time": None,
                "last_funding_time": None,
                "large_gap_count": 0,
                "coverage_complete": False,
                "warnings": [f"funding_csv_read_failed: {exc!r}"],
            }
            continue
        records[inst_id] = analyze_funding_quality(histories[inst_id], data_range, path)

    complete = bool(records and all(record["coverage_complete"] for record in records.values()))
    quality = {
        "funding_data_complete": complete,
        "records": records,
        "missing_inst_ids": [
            inst_id
            for inst_id, record in records.items()
            if not record.get("coverage_complete")
        ],
    }
    return histories, quality


def funding_pnl_for_interval(
    funding_df: pd.DataFrame,
    entry_time: Any,
    exit_time: Any,
    direction: str,
    notional: float,
) -> tuple[float, int]:
    """Calculate inclusive funding PnL for one holding interval."""

    if funding_df.empty or entry_time is None or exit_time is None or pd.isna(entry_time) or pd.isna(exit_time):
        return 0.0, 0
    entry_utc = pd.Timestamp(entry_time)
    exit_utc = pd.Timestamp(exit_time)
    if entry_utc.tzinfo is None:
        entry_utc = entry_utc.tz_localize(DEFAULT_SESSION_TIMEZONE)
    if exit_utc.tzinfo is None:
        exit_utc = exit_utc.tz_localize(DEFAULT_SESSION_TIMEZONE)
    entry_utc = entry_utc.tz_convert("UTC")
    exit_utc = exit_utc.tz_convert("UTC")
    if exit_utc < entry_utc:
        entry_utc, exit_utc = exit_utc, entry_utc
    mask = (funding_df["funding_time_utc"] >= entry_utc) & (funding_df["funding_time_utc"] <= exit_utc)
    rates = pd.to_numeric(funding_df.loc[mask, "funding_rate"], errors="coerce").dropna()
    pnl = float(sum(signed_funding_pnl(notional, float(rate), direction) for rate in rates))
    return pnl, int(len(rates.index))


def stable_control_key(parts: Iterable[Any]) -> str:
    """Build a deterministic compact key for source/control mapping."""

    text = "|".join(str(part) for part in parts)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def build_event_row(
    frame: pd.DataFrame,
    pos: int,
    symbol: str,
    timeframe: str,
    group: str,
    session_type: str,
    source_session_type: str,
    direction: str,
    config: CsrbConfig,
    report_timezone: str,
    range_context: dict[str, Any] | None = None,
    control_key: str = "",
) -> dict[str, Any]:
    """Build one event row from a closed-bar signal."""

    row = frame.iloc[pos]
    range_context = range_context or {}
    atr_prev = finite_float(row.get("atr_prev"), default=np.nan)
    range_high = finite_float(range_context.get("range_high"), default=np.nan)
    range_low = finite_float(range_context.get("range_low"), default=np.nan)
    if direction == "long":
        boundary = range_high + config.buffer_atr * atr_prev if np.isfinite(range_high) else np.nan
    else:
        boundary = range_low - config.buffer_atr * atr_prev if np.isfinite(range_low) else np.nan
    entry_time = frame.iloc[pos + 1]["open_time"] if pos + 1 < len(frame.index) else None
    entry_price = float(frame.iloc[pos + 1]["open"]) if pos + 1 < len(frame.index) else np.nan
    metrics = future_returns_for_event(frame, pos, direction, config.event_horizons)
    event = {
        "event_id": "",
        "timestamp": row["datetime"],
        "timestamp_report": format_timestamp(row["datetime"], report_timezone),
        "symbol": symbol,
        "inst_id": symbol_to_inst_id(symbol),
        "timeframe": timeframe,
        "group": group,
        "session_type": session_type,
        "source_session_type": source_session_type,
        "session_date": str(range_context.get("session_date") or row.get("session_date")),
        "direction": direction,
        "bar_index": int(pos),
        "range_start": range_context.get("range_start"),
        "range_end": range_context.get("range_end"),
        "range_high": range_high,
        "range_low": range_low,
        "range_width": finite_float(range_context.get("range_width"), default=np.nan),
        "range_bar_count": int(range_context.get("range_bar_count") or 0),
        "breakout_window_start": range_context.get("breakout_window_start"),
        "breakout_window_end": range_context.get("breakout_window_end"),
        "close": float(row["close"]),
        "atr_prev": atr_prev,
        "buffer_atr": float(config.buffer_atr),
        "breakout_boundary": boundary,
        "entry_time": entry_time,
        "entry_price": entry_price,
        "control_key": control_key,
    }
    event.update(metrics)
    return event


def range_context_from_session(
    session_date: str,
    range_frame: pd.DataFrame,
    spec: dict[str, Any],
) -> dict[str, Any]:
    """Build range metadata for a session breakout event."""

    range_high = float(pd.to_numeric(range_frame["high"], errors="coerce").max())
    range_low = float(pd.to_numeric(range_frame["low"], errors="coerce").min())
    return {
        "session_date": session_date,
        "range_start": range_frame["open_time"].min(),
        "range_end": range_frame["datetime"].max(),
        "range_high": range_high,
        "range_low": range_low,
        "range_width": float(range_high - range_low),
        "range_bar_count": int(len(range_frame.index)),
        "breakout_window_start": int(spec["breakout_start_minute"]),
        "breakout_window_end": int(spec["breakout_end_minute"]),
    }


def generate_session_breakouts_for_frame(
    frame: pd.DataFrame,
    symbol: str,
    timeframe: str,
    config: CsrbConfig,
    report_timezone: str = DEFAULT_REPORT_TIMEZONE,
) -> tuple[pd.DataFrame, list[str]]:
    """Generate B and C session breakout events for one symbol/timeframe."""

    rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    working = add_csrb_indicators(frame, config)
    if working.empty:
        return pd.DataFrame(columns=event_columns(config)), []

    for session_type, spec in SESSION_SPECS.items():
        group = str(spec["group"])
        for session_date, day in working.groupby("session_date", sort=True):
            range_frame = day[
                (day["minute_of_day"] >= int(spec["range_start_minute"]))
                & (day["minute_of_day"] <= int(spec["range_end_minute"]))
            ]
            if len(range_frame.index) < config.range_min_bars:
                continue
            context = range_context_from_session(str(session_date), range_frame, spec)
            breakout_frame = day[
                (day["minute_of_day"] >= int(spec["breakout_start_minute"]))
                & (day["minute_of_day"] <= int(spec["breakout_end_minute"]))
            ]
            for pos in breakout_frame.index:
                row = working.iloc[int(pos)]
                atr_prev = finite_float(row.get("atr_prev"), default=np.nan)
                if not np.isfinite(atr_prev) or atr_prev <= 0:
                    continue
                upper = context["range_high"] + config.buffer_atr * atr_prev
                lower = context["range_low"] - config.buffer_atr * atr_prev
                close = float(row["close"])
                long_signal = close > upper
                short_signal = close < lower
                if long_signal and short_signal:
                    warnings.append(f"{symbol} {timeframe} {session_date} {session_type}: same-bar anomaly skipped")
                    continue
                if not (long_signal or short_signal):
                    continue
                direction = "long" if long_signal else "short"
                rows.append(
                    build_event_row(
                        working,
                        int(pos),
                        symbol,
                        timeframe,
                        group,
                        session_type,
                        session_type,
                        direction,
                        config,
                        report_timezone,
                        context,
                    )
                )
                break

    events = pd.DataFrame(rows, columns=event_columns(config))
    if not events.empty:
        events = events.sort_values(["timestamp", "symbol", "timeframe", "group"], kind="stable").reset_index(drop=True)
    return events, warnings


def rolling_range_bars_for_timeframe(timeframe: str, config: CsrbConfig) -> int:
    """Return A-group ordinary breakout lookback bars."""

    minutes = TIMEFRAME_MINUTES[timeframe]
    return max(config.range_min_bars, int(round((8 * 60) / minutes)))


def generate_ordinary_breakouts_for_frame(
    frame: pd.DataFrame,
    symbol: str,
    timeframe: str,
    config: CsrbConfig,
    report_timezone: str = DEFAULT_REPORT_TIMEZONE,
) -> tuple[pd.DataFrame, list[str]]:
    """Generate A-group session-agnostic ordinary rolling range breakouts."""

    rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    working = add_csrb_indicators(frame, config)
    if working.empty:
        return pd.DataFrame(columns=event_columns(config)), []

    lookback = rolling_range_bars_for_timeframe(timeframe, config)
    high = pd.to_numeric(working["high"], errors="coerce")
    low = pd.to_numeric(working["low"], errors="coerce")
    working["ordinary_upper"] = high.rolling(lookback, min_periods=lookback).max().shift(1)
    working["ordinary_lower"] = low.rolling(lookback, min_periods=lookback).min().shift(1)
    for pos, row in working.iterrows():
        atr_prev = finite_float(row.get("atr_prev"), default=np.nan)
        upper_raw = finite_float(row.get("ordinary_upper"), default=np.nan)
        lower_raw = finite_float(row.get("ordinary_lower"), default=np.nan)
        if not (np.isfinite(atr_prev) and np.isfinite(upper_raw) and np.isfinite(lower_raw)):
            continue
        upper = upper_raw + config.buffer_atr * atr_prev
        lower = lower_raw - config.buffer_atr * atr_prev
        close = float(row["close"])
        long_signal = close > upper
        short_signal = close < lower
        if long_signal and short_signal:
            warnings.append(f"{symbol} {timeframe}: ordinary same-bar anomaly skipped at pos={pos}")
            continue
        if not (long_signal or short_signal):
            continue
        start_pos = max(0, int(pos) - lookback)
        range_frame = working.iloc[start_pos:int(pos)]
        context = {
            "session_date": row["session_date"],
            "range_start": range_frame["open_time"].min() if not range_frame.empty else None,
            "range_end": range_frame["datetime"].max() if not range_frame.empty else None,
            "range_high": upper_raw,
            "range_low": lower_raw,
            "range_width": float(upper_raw - lower_raw),
            "range_bar_count": int(len(range_frame.index)),
            "breakout_window_start": None,
            "breakout_window_end": None,
        }
        direction = "long" if long_signal else "short"
        rows.append(
            build_event_row(
                working,
                int(pos),
                symbol,
                timeframe,
                "A",
                "ordinary_rolling",
                "ordinary_rolling",
                direction,
                config,
                report_timezone,
                context,
            )
        )

    events = pd.DataFrame(rows, columns=event_columns(config))
    if not events.empty:
        events = events.sort_values(["timestamp", "symbol", "timeframe", "direction"], kind="stable").reset_index(drop=True)
    return events, warnings


def deterministic_rng(config: CsrbConfig, *parts: Any) -> np.random.Generator:
    """Create a deterministic RNG scoped to one control selection."""

    digest = hashlib.sha256("|".join(str(part) for part in (config.random_seed, *parts)).encode("utf-8")).digest()
    seed = int.from_bytes(digest[:8], "little") % (2**32 - 1)
    return np.random.default_rng(seed)


def generate_random_time_controls_for_frame(
    frame: pd.DataFrame,
    core_events: pd.DataFrame,
    symbol: str,
    timeframe: str,
    config: CsrbConfig,
    report_timezone: str = DEFAULT_REPORT_TIMEZONE,
) -> tuple[pd.DataFrame, list[str]]:
    """Generate D-group randomized same-day non-breakout-window controls."""

    rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    if core_events.empty:
        return pd.DataFrame(columns=event_columns(config)), []
    working = add_csrb_indicators(frame, config)
    used_positions: set[tuple[str, int]] = set()
    for event in core_events.itertuples(index=False):
        spec = SESSION_SPECS.get(str(event.session_type))
        if spec is None:
            continue
        day = working[working["session_date"] == str(event.session_date)]
        if day.empty:
            warnings.append(f"{symbol} {timeframe} {event.session_date}: no day rows for random control")
            continue
        candidates = day[
            ~(
                (day["minute_of_day"] >= int(spec["breakout_start_minute"]))
                & (day["minute_of_day"] <= int(spec["breakout_end_minute"]))
            )
        ].copy()
        candidates = candidates[
            candidates.index.map(lambda pos: int(pos) + config.hold_bars + 1 < len(working.index))
            & candidates["atr_prev"].map(lambda value: np.isfinite(finite_float(value, default=np.nan)))
        ]
        candidates = candidates[
            ~candidates.index.map(lambda pos: (str(event.session_date), int(pos)) in used_positions)
        ]
        if candidates.empty:
            warnings.append(f"{symbol} {timeframe} {event.session_date}: random control candidate missing")
            continue
        rng = deterministic_rng(config, symbol, timeframe, event.session_date, event.session_type, event.direction, event.bar_index)
        chosen_pos = int(rng.choice(candidates.index.to_numpy(dtype=int)))
        used_positions.add((str(event.session_date), chosen_pos))
        context = {
            "session_date": event.session_date,
            "range_start": event.range_start,
            "range_end": event.range_end,
            "range_high": event.range_high,
            "range_low": event.range_low,
            "range_width": event.range_width,
            "range_bar_count": int(event.range_bar_count),
            "breakout_window_start": event.breakout_window_start,
            "breakout_window_end": event.breakout_window_end,
        }
        rows.append(
            build_event_row(
                working,
                chosen_pos,
                symbol,
                timeframe,
                "D",
                "random_time_control",
                str(event.session_type),
                str(event.direction),
                config,
                report_timezone,
                context,
                stable_control_key([symbol, timeframe, event.session_date, event.session_type, event.bar_index, chosen_pos]),
            )
        )

    events = pd.DataFrame(rows, columns=event_columns(config))
    if not events.empty:
        events = events.sort_values(["timestamp", "symbol", "timeframe", "source_session_type"], kind="stable").reset_index(drop=True)
    return events, warnings


def generate_reverse_tests_for_frame(
    frame: pd.DataFrame,
    core_events: pd.DataFrame,
    symbol: str,
    timeframe: str,
    config: CsrbConfig,
    report_timezone: str = DEFAULT_REPORT_TIMEZONE,
) -> pd.DataFrame:
    """Generate E-group reverse-direction events from B/C core events."""

    if core_events.empty:
        return pd.DataFrame(columns=event_columns(config))
    working = add_csrb_indicators(frame, config)
    rows: list[dict[str, Any]] = []
    for event in core_events.itertuples(index=False):
        direction = "short" if str(event.direction) == "long" else "long"
        context = {
            "session_date": event.session_date,
            "range_start": event.range_start,
            "range_end": event.range_end,
            "range_high": event.range_high,
            "range_low": event.range_low,
            "range_width": event.range_width,
            "range_bar_count": int(event.range_bar_count),
            "breakout_window_start": event.breakout_window_start,
            "breakout_window_end": event.breakout_window_end,
        }
        rows.append(
            build_event_row(
                working,
                int(event.bar_index),
                symbol,
                timeframe,
                "E",
                "reverse_test",
                str(event.session_type),
                direction,
                config,
                report_timezone,
                context,
                stable_control_key([symbol, timeframe, event.session_date, event.session_type, event.bar_index, "reverse"]),
            )
        )
    return pd.DataFrame(rows, columns=event_columns(config))


def generate_events_for_frame(
    frame: pd.DataFrame,
    symbol: str,
    timeframe: str,
    config: CsrbConfig,
    report_timezone: str = DEFAULT_REPORT_TIMEZONE,
) -> tuple[pd.DataFrame, list[str]]:
    """Generate A/B/C/D/E events for one symbol/timeframe."""

    ordinary, ordinary_warnings = generate_ordinary_breakouts_for_frame(frame, symbol, timeframe, config, report_timezone)
    core, core_warnings = generate_session_breakouts_for_frame(frame, symbol, timeframe, config, report_timezone)
    reverse = generate_reverse_tests_for_frame(frame, core, symbol, timeframe, config, report_timezone)
    random_controls, control_warnings = generate_random_time_controls_for_frame(
        frame,
        core,
        symbol,
        timeframe,
        config,
        report_timezone,
    )
    frames = [ordinary, core, random_controls, reverse]
    events = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=event_columns(config))
    if not events.empty:
        events = events.sort_values(
            ["timestamp", "symbol", "timeframe", "group", "session_type", "direction"],
            kind="stable",
        ).reset_index(drop=True)
    return events, ordinary_warnings + core_warnings + control_warnings


def assign_event_ids(events: pd.DataFrame) -> pd.DataFrame:
    """Assign stable event ids."""

    if events.empty:
        return events.copy()
    out = events.copy().reset_index(drop=True)
    out["event_id"] = [
        f"csrb_v1_{index:08d}_{row.symbol}_{row.timeframe}_{row.group}_{row.direction}"
        for index, row in enumerate(out.itertuples(index=False), start=1)
    ]
    return out


def build_time_splits(data_range: HistoryRange, session_timezone: str) -> list[TimeSplit]:
    """Build chronological 60/20/20 time splits in session timezone."""

    start = pd.Timestamp(data_range.start).tz_convert(session_timezone)
    end = pd.Timestamp(data_range.end_exclusive).tz_convert(session_timezone)
    span = end - start
    train_end = start + span * 0.6
    validation_end = start + span * 0.8
    return [
        TimeSplit("train", start, train_end),
        TimeSplit("validation", train_end, validation_end),
        TimeSplit("oos", validation_end, end),
    ]


def assign_split_for_time(value: Any, splits: list[TimeSplit]) -> str:
    """Assign one timestamp to train/validation/oos."""

    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize(splits[0].start.tz)
    for split in splits[:-1]:
        if split.start <= timestamp < split.end:
            return split.name
    last = splits[-1]
    if last.start <= timestamp <= last.end:
        return last.name
    if timestamp < splits[0].start:
        return splits[0].name
    return splits[-1].name


def split_dates_payload(splits: list[TimeSplit], report_timezone: str) -> dict[str, Any]:
    """Return JSON-safe split date payload."""

    payload: dict[str, Any] = {}
    for split in splits:
        payload[f"{split.name}_start"] = format_timestamp(split.start)
        payload[f"{split.name}_end"] = format_timestamp(split.end)
        payload[f"{split.name}_start_report"] = format_timestamp(split.start, report_timezone)
        payload[f"{split.name}_end_report"] = format_timestamp(split.end, report_timezone)
    return payload


def simulate_fixed_hold_trades(
    events: pd.DataFrame,
    bars_by_key: dict[tuple[str, str], pd.DataFrame],
    config: CsrbConfig,
    funding_histories: dict[str, pd.DataFrame],
    splits: list[TimeSplit],
) -> tuple[pd.DataFrame, list[str]]:
    """Simulate fixed-hold trades with single-symbol single-group filtering."""

    if events.empty:
        return pd.DataFrame(columns=TRADE_COLUMNS), []

    warnings: list[str] = []
    active_until: dict[tuple[str, str, str], pd.Timestamp] = {}
    trades: list[dict[str, Any]] = []
    skipped_future = 0
    skipped_conflict = 0
    sorted_events = events.sort_values(["timestamp", "symbol", "timeframe", "group", "event_id"], kind="stable")
    for row in sorted_events.itertuples(index=False):
        key = (row.symbol, row.timeframe)
        frame = bars_by_key.get(key)
        if frame is None or frame.empty:
            skipped_future += 1
            continue
        pos = int(row.bar_index)
        entry_pos = pos + 1
        exit_pos = pos + config.hold_bars + 1
        if entry_pos >= len(frame.index) or exit_pos >= len(frame.index):
            skipped_future += 1
            continue
        group_key = (row.symbol, row.timeframe, row.group)
        event_time = pd.Timestamp(row.timestamp)
        existing_exit = active_until.get(group_key)
        if existing_exit is not None and event_time < existing_exit:
            skipped_conflict += 1
            continue

        entry_time = frame.iloc[entry_pos]["open_time"]
        exit_time = frame.iloc[exit_pos]["open_time"]
        entry_price = float(frame.iloc[entry_pos]["open"])
        exit_price = float(frame.iloc[exit_pos]["open"])
        gross_return = directional_return(entry_price, exit_price, str(row.direction))
        no_cost_pnl = config.fixed_notional * gross_return if np.isfinite(gross_return) else np.nan
        fee_cost = config.fixed_notional * (config.fee_bps_per_side / 10000.0) * 2.0
        slippage_cost = config.fixed_notional * (config.slippage_bps_per_side / 10000.0) * 2.0
        funding_df = funding_histories.get(row.inst_id, pd.DataFrame(columns=["funding_time_utc", "funding_rate"]))
        funding_pnl, funding_count = funding_pnl_for_interval(
            funding_df,
            entry_time,
            exit_time,
            str(row.direction),
            config.fixed_notional,
        )
        cost_aware_pnl = no_cost_pnl - fee_cost - slippage_cost
        funding_adjusted_pnl = cost_aware_pnl + funding_pnl
        trades.append(
            {
                "trade_id": f"csrb_trade_{len(trades) + 1:08d}",
                "event_id": row.event_id,
                "symbol": row.symbol,
                "inst_id": row.inst_id,
                "timeframe": row.timeframe,
                "group": row.group,
                "session_type": row.session_type,
                "source_session_type": row.source_session_type,
                "session_date": row.session_date,
                "direction": row.direction,
                "entry_time": entry_time,
                "entry_price": entry_price,
                "exit_time": exit_time,
                "exit_price": exit_price,
                "hold_bars": config.hold_bars,
                "gross_return": gross_return,
                "no_cost_pnl": no_cost_pnl,
                "fee_cost": fee_cost,
                "slippage_cost": slippage_cost,
                "funding_count": funding_count,
                "funding_pnl": funding_pnl,
                "cost_aware_pnl": cost_aware_pnl,
                "funding_adjusted_pnl": funding_adjusted_pnl,
                "split": assign_split_for_time(entry_time, splits),
            }
        )
        active_until[group_key] = pd.Timestamp(exit_time)

    if skipped_future:
        warnings.append(f"skipped_events_with_insufficient_future_bars={skipped_future}")
    if skipped_conflict:
        warnings.append(f"skipped_events_due_to_single_position_filter={skipped_conflict}")
    return pd.DataFrame(trades, columns=TRADE_COLUMNS), warnings


def max_drawdown(pnls: Iterable[float], starting_equity: float) -> tuple[float, float]:
    """Return max drawdown amount and percent."""

    values = pd.Series(list(pnls), dtype=float).fillna(0.0)
    if values.empty:
        return 0.0, 0.0
    equity = starting_equity + values.cumsum()
    peak = equity.cummax()
    drawdown = peak - equity
    max_dd = float(drawdown.max())
    denominator = peak.replace(0.0, np.nan)
    max_dd_pct = float((drawdown / denominator).max()) if not denominator.isna().all() else 0.0
    if not np.isfinite(max_dd_pct):
        max_dd_pct = 0.0
    return max_dd, max_dd_pct


def pnl_sum(frame: pd.DataFrame, column: str) -> float:
    """Return numeric sum for a frame/column."""

    if frame.empty or column not in frame.columns:
        return 0.0
    return float(pd.to_numeric(frame[column], errors="coerce").sum())


def summarize_trades(trades: pd.DataFrame, group_columns: list[str], config: CsrbConfig) -> pd.DataFrame:
    """Summarize trades by requested columns."""

    columns = group_columns + [
        "trade_count",
        "long_count",
        "short_count",
        "no_cost_pnl",
        "cost_aware_pnl",
        "funding_pnl",
        "funding_adjusted_pnl",
        "avg_gross_return",
        "win_rate",
        "max_drawdown",
        "max_drawdown_pct",
    ]
    if trades.empty:
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, Any]] = []
    grouped = trades.groupby(group_columns, dropna=False, sort=True) if group_columns else [((), trades)]
    for keys, group in grouped:
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {column: key for column, key in zip(group_columns, keys)}
        pnl_series = pd.to_numeric(group["funding_adjusted_pnl"], errors="coerce").fillna(0.0)
        max_dd, max_dd_pct = max_drawdown(pnl_series, config.fixed_notional)
        row.update(
            {
                "trade_count": int(len(group.index)),
                "long_count": int((group["direction"] == "long").sum()),
                "short_count": int((group["direction"] == "short").sum()),
                "no_cost_pnl": pnl_sum(group, "no_cost_pnl"),
                "cost_aware_pnl": pnl_sum(group, "cost_aware_pnl"),
                "funding_pnl": pnl_sum(group, "funding_pnl"),
                "funding_adjusted_pnl": float(pnl_series.sum()),
                "avg_gross_return": float(pd.to_numeric(group["gross_return"], errors="coerce").mean()),
                "win_rate": float((pd.to_numeric(group["funding_adjusted_pnl"], errors="coerce") > 0).mean()),
                "max_drawdown": max_dd,
                "max_drawdown_pct": max_dd_pct,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows, columns=columns).sort_values(group_columns, kind="stable").reset_index(drop=True)


def summarize_events(events: pd.DataFrame, config: CsrbConfig) -> pd.DataFrame:
    """Summarize event counts and future-return statistics."""

    columns = [
        "group",
        "session_type",
        "timeframe",
        "event_count",
        "long_count",
        "short_count",
        "avg_range_width",
    ] + [f"mean_future_return_{horizon}" for horizon in config.event_horizons]
    if events.empty:
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, Any]] = []
    for (group, session_type, timeframe), frame in events.groupby(["group", "session_type", "timeframe"], dropna=False, sort=True):
        row = {
            "group": group,
            "session_type": session_type,
            "timeframe": timeframe,
            "event_count": int(len(frame.index)),
            "long_count": int((frame["direction"] == "long").sum()),
            "short_count": int((frame["direction"] == "short").sum()),
            "avg_range_width": float(pd.to_numeric(frame["range_width"], errors="coerce").mean()),
        }
        for horizon in config.event_horizons:
            row[f"mean_future_return_{horizon}"] = float(pd.to_numeric(frame[f"future_return_{horizon}"], errors="coerce").mean())
        rows.append(row)
    return pd.DataFrame(rows, columns=columns)


def largest_symbol_pnl_share(trades: pd.DataFrame, pnl_column: str) -> tuple[float | None, str | None]:
    """Return largest symbol PnL dependency share."""

    if trades.empty:
        return None, None
    symbol_net = pd.to_numeric(trades[pnl_column], errors="coerce").fillna(0.0).groupby(trades["symbol"]).sum()
    if symbol_net.empty:
        return None, None
    positive = symbol_net[symbol_net > 0]
    if not positive.empty and float(positive.sum()) > 0:
        symbol = str(positive.idxmax())
        return float(positive.max() / positive.sum()), symbol
    absolute = symbol_net.abs()
    if float(absolute.sum()) > 0:
        symbol = str(absolute.idxmax())
        return float(absolute.max() / absolute.sum()), symbol
    return None, None


def largest_symbol_trade_share(trades: pd.DataFrame) -> tuple[float | None, str | None]:
    """Return largest symbol trade-count share."""

    if trades.empty:
        return None, None
    counts = trades.groupby("symbol").size()
    if counts.empty or int(counts.sum()) == 0:
        return None, None
    symbol = str(counts.idxmax())
    return float(counts.max() / counts.sum()), symbol


def top_trade_contribution(trades: pd.DataFrame, pnl_column: str) -> tuple[float | None, float, int]:
    """Return top 5 percent trade contribution."""

    if trades.empty:
        return None, 0.0, 0
    pnl = pd.to_numeric(trades[pnl_column], errors="coerce").fillna(0.0).sort_values(ascending=False)
    count = int(len(pnl.index))
    top_count = max(1, int(math.ceil(count * 0.05))) if count else 0
    top_pnl = float(pnl.head(top_count).sum()) if top_count else 0.0
    total = float(pnl.sum())
    return (float(top_pnl / total) if total > 0 else None), top_pnl, top_count


def build_concentration_summary(trades: pd.DataFrame, config: CsrbConfig) -> pd.DataFrame:
    """Build concentration diagnostics by group/timeframe/split."""

    columns = [
        "group",
        "timeframe",
        "split",
        "trade_count",
        "total_funding_adjusted_pnl",
        "largest_symbol_pnl_share",
        "largest_symbol_pnl_symbol",
        "largest_symbol_trade_count_share",
        "largest_symbol_trade_count_symbol",
        "top_5pct_trade_pnl_contribution",
        "top_5pct_trade_pnl",
        "top_5pct_trade_count",
        "max_drawdown",
        "max_drawdown_pct",
    ]
    if trades.empty:
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, Any]] = []
    for (group, timeframe, split), frame in trades.groupby(["group", "timeframe", "split"], dropna=False, sort=True):
        pnl_share, pnl_symbol = largest_symbol_pnl_share(frame, "funding_adjusted_pnl")
        count_share, count_symbol = largest_symbol_trade_share(frame)
        top_share, top_pnl, top_count = top_trade_contribution(frame, "funding_adjusted_pnl")
        max_dd, max_dd_pct = max_drawdown(pd.to_numeric(frame["funding_adjusted_pnl"], errors="coerce").fillna(0.0), config.fixed_notional)
        rows.append(
            {
                "group": group,
                "timeframe": timeframe,
                "split": split,
                "trade_count": int(len(frame.index)),
                "total_funding_adjusted_pnl": float(pd.to_numeric(frame["funding_adjusted_pnl"], errors="coerce").sum()),
                "largest_symbol_pnl_share": pnl_share,
                "largest_symbol_pnl_symbol": pnl_symbol,
                "largest_symbol_trade_count_share": count_share,
                "largest_symbol_trade_count_symbol": count_symbol,
                "top_5pct_trade_pnl_contribution": top_share,
                "top_5pct_trade_pnl": top_pnl,
                "top_5pct_trade_count": top_count,
                "max_drawdown": max_dd,
                "max_drawdown_pct": max_dd_pct,
            }
        )
    return pd.DataFrame(rows, columns=columns)


def core_symbols_from_requested(symbols: list[str]) -> list[str]:
    """Return requested symbols that belong to the Phase 1 core set."""

    core = [symbol for symbol in symbols if symbol in CORE_SYMBOLS]
    return core or symbols


def core_session_trades(trades: pd.DataFrame, symbols: list[str]) -> pd.DataFrame:
    """Return B+C 15m BTC/ETH/SOL core session breakout trades."""

    if trades.empty:
        return pd.DataFrame(columns=TRADE_COLUMNS)
    return trades[
        (trades["group"].isin(["B", "C"]))
        & (trades["timeframe"] == "15m")
        & (trades["symbol"].isin(core_symbols_from_requested(symbols)))
    ].copy()


def baseline_trades(trades: pd.DataFrame, symbols: list[str], group: str) -> pd.DataFrame:
    """Return one 15m core-symbol baseline trade frame."""

    if trades.empty:
        return pd.DataFrame(columns=TRADE_COLUMNS)
    return trades[
        (trades["group"] == group)
        & (trades["timeframe"] == "15m")
        & (trades["symbol"].isin(core_symbols_from_requested(symbols)))
    ].copy()


def build_reverse_test(trades: pd.DataFrame, symbols: list[str]) -> pd.DataFrame:
    """Compare B/C forward trades to E reverse-test trades."""

    columns = [
        "timeframe",
        "source_session_type",
        "split",
        "forward_trade_count",
        "reverse_trade_count",
        "forward_no_cost_pnl",
        "reverse_no_cost_pnl",
        "forward_cost_aware_pnl",
        "reverse_cost_aware_pnl",
        "forward_funding_adjusted_pnl",
        "reverse_funding_adjusted_pnl",
        "reverse_weaker",
    ]
    if trades.empty:
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, Any]] = []
    core_symbols = core_symbols_from_requested(symbols)
    scoped = trades[trades["symbol"].isin(core_symbols)]
    for timeframe in sorted(scoped["timeframe"].dropna().unique()):
        for source_session_type in ["asia_to_europe", "europe_to_us", "core_session_breakout"]:
            for split in ["train", "validation", "oos", "all"]:
                frame = scoped[scoped["timeframe"] == timeframe]
                if source_session_type == "core_session_breakout":
                    forward = frame[frame["group"].isin(["B", "C"])]
                    reverse = frame[frame["group"] == "E"]
                else:
                    forward = frame[(frame["group"].isin(["B", "C"])) & (frame["session_type"] == source_session_type)]
                    reverse = frame[(frame["group"] == "E") & (frame["source_session_type"] == source_session_type)]
                if split != "all":
                    forward = forward[forward["split"] == split]
                    reverse = reverse[reverse["split"] == split]
                forward_no_cost = pnl_sum(forward, "no_cost_pnl")
                reverse_no_cost = pnl_sum(reverse, "no_cost_pnl")
                forward_funding = pnl_sum(forward, "funding_adjusted_pnl")
                reverse_funding = pnl_sum(reverse, "funding_adjusted_pnl")
                rows.append(
                    {
                        "timeframe": timeframe,
                        "source_session_type": source_session_type,
                        "split": split,
                        "forward_trade_count": int(len(forward.index)),
                        "reverse_trade_count": int(len(reverse.index)),
                        "forward_no_cost_pnl": forward_no_cost,
                        "reverse_no_cost_pnl": reverse_no_cost,
                        "forward_cost_aware_pnl": pnl_sum(forward, "cost_aware_pnl"),
                        "reverse_cost_aware_pnl": pnl_sum(reverse, "cost_aware_pnl"),
                        "forward_funding_adjusted_pnl": forward_funding,
                        "reverse_funding_adjusted_pnl": reverse_funding,
                        "reverse_weaker": bool(reverse_no_cost < forward_no_cost and reverse_funding < forward_funding),
                    }
                )
    return pd.DataFrame(rows, columns=columns)


def build_random_time_control(trades: pd.DataFrame, symbols: list[str]) -> pd.DataFrame:
    """Compare B/C forward trades to D random-time controls."""

    columns = [
        "timeframe",
        "source_session_type",
        "split",
        "forward_trade_count",
        "random_trade_count",
        "forward_no_cost_pnl",
        "random_no_cost_pnl",
        "forward_cost_aware_pnl",
        "random_cost_aware_pnl",
        "forward_funding_adjusted_pnl",
        "random_funding_adjusted_pnl",
        "session_better",
    ]
    if trades.empty:
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, Any]] = []
    core_symbols = core_symbols_from_requested(symbols)
    scoped = trades[trades["symbol"].isin(core_symbols)]
    for timeframe in sorted(scoped["timeframe"].dropna().unique()):
        for source_session_type in ["asia_to_europe", "europe_to_us", "core_session_breakout"]:
            for split in ["train", "validation", "oos", "all"]:
                frame = scoped[scoped["timeframe"] == timeframe]
                if source_session_type == "core_session_breakout":
                    forward = frame[frame["group"].isin(["B", "C"])]
                    random = frame[frame["group"] == "D"]
                else:
                    forward = frame[(frame["group"].isin(["B", "C"])) & (frame["session_type"] == source_session_type)]
                    random = frame[(frame["group"] == "D") & (frame["source_session_type"] == source_session_type)]
                if split != "all":
                    forward = forward[forward["split"] == split]
                    random = random[random["split"] == split]
                forward_no_cost = pnl_sum(forward, "no_cost_pnl")
                random_no_cost = pnl_sum(random, "no_cost_pnl")
                rows.append(
                    {
                        "timeframe": timeframe,
                        "source_session_type": source_session_type,
                        "split": split,
                        "forward_trade_count": int(len(forward.index)),
                        "random_trade_count": int(len(random.index)),
                        "forward_no_cost_pnl": forward_no_cost,
                        "random_no_cost_pnl": random_no_cost,
                        "forward_cost_aware_pnl": pnl_sum(forward, "cost_aware_pnl"),
                        "random_cost_aware_pnl": pnl_sum(random, "cost_aware_pnl"),
                        "forward_funding_adjusted_pnl": pnl_sum(forward, "funding_adjusted_pnl"),
                        "random_funding_adjusted_pnl": pnl_sum(random, "funding_adjusted_pnl"),
                        "session_better": bool(forward_no_cost > random_no_cost),
                    }
                )
    return pd.DataFrame(rows, columns=columns)


def build_session_summary(trades: pd.DataFrame, config: CsrbConfig) -> pd.DataFrame:
    """Build a session-type summary for A/B/C/D/E outputs."""

    return summarize_trades(trades, ["group", "session_type", "timeframe"], config)


def build_funding_summary(trades: pd.DataFrame, funding_quality: dict[str, Any]) -> pd.DataFrame:
    """Build funding summary rows."""

    rows: list[dict[str, Any]] = []
    for inst_id, record in (funding_quality.get("records") or {}).items():
        symbol_trades = trades[trades["inst_id"] == inst_id] if not trades.empty else pd.DataFrame()
        rows.append(
            {
                "inst_id": inst_id,
                "funding_data_complete": bool(record.get("coverage_complete")),
                "csv_path": record.get("csv_path"),
                "row_count": int(record.get("row_count") or 0),
                "first_funding_time": record.get("first_funding_time"),
                "last_funding_time": record.get("last_funding_time"),
                "trade_count": int(len(symbol_trades.index)),
                "funding_count": int(pd.to_numeric(symbol_trades.get("funding_count"), errors="coerce").sum()) if not symbol_trades.empty else 0,
                "funding_pnl": pnl_sum(symbol_trades, "funding_pnl") if not symbol_trades.empty else 0.0,
                "warnings": ";".join(record.get("warnings") or []),
            }
        )
    return pd.DataFrame(rows)


def evaluate_phase1_gates(
    trades: pd.DataFrame,
    symbols: list[str],
    funding_data_complete: bool,
    config: CsrbConfig,
) -> dict[str, Any]:
    """Evaluate CSRB-v1 Phase 1 gates."""

    core = core_session_trades(trades, symbols)
    by_split = {split: core[core["split"] == split] for split in ["train", "validation", "oos"]}
    train_no_cost = pnl_sum(by_split["train"], "no_cost_pnl")
    validation_no_cost = pnl_sum(by_split["validation"], "no_cost_pnl")
    oos_no_cost = pnl_sum(by_split["oos"], "no_cost_pnl")
    oos_cost = pnl_sum(by_split["oos"], "cost_aware_pnl")
    oos_funding = pnl_sum(by_split["oos"], "funding_adjusted_pnl")

    train_pass = bool(train_no_cost > 0)
    validation_pass = bool(validation_no_cost > 0)
    oos_pass = bool(oos_no_cost > 0)
    cost_aware_pass = bool(oos_cost >= 0)
    funding_adjusted_pass = bool(funding_data_complete and oos_funding >= 0)
    trade_counts = {split: int(len(frame.index)) for split, frame in by_split.items()}
    trade_count_pass = bool(
        trade_counts["train"] >= 30
        and trade_counts["validation"] >= 10
        and trade_counts["oos"] >= 10
    )

    oos = by_split["oos"]
    pnl_share, pnl_symbol = largest_symbol_pnl_share(oos, "funding_adjusted_pnl")
    count_share, count_symbol = largest_symbol_trade_share(oos)
    top_share, top_pnl, top_count = top_trade_contribution(oos, "funding_adjusted_pnl")
    max_dd, max_dd_pct = (
        max_drawdown(pd.to_numeric(oos.get("funding_adjusted_pnl"), errors="coerce").fillna(0.0), config.fixed_notional)
        if not oos.empty
        else (0.0, 0.0)
    )
    concentration_pass = bool(
        pnl_share is not None
        and count_share is not None
        and top_share is not None
        and pnl_share <= 0.70
        and count_share <= 0.70
        and top_share <= 0.80
        and max_dd_pct <= 0.30
    )

    if trades.empty:
        reverse_test_pass = False
        session_vs_baseline_pass = False
    else:
        scoped = trades[
            (trades["timeframe"] == "15m")
            & (trades["symbol"].isin(core_symbols_from_requested(symbols)))
            & (trades["split"] == "oos")
        ]
        forward = scoped[scoped["group"].isin(["B", "C"])]
        reverse = scoped[scoped["group"] == "E"]
        ordinary = scoped[scoped["group"] == "A"]
        random = scoped[scoped["group"] == "D"]
        forward_no_cost = pnl_sum(forward, "no_cost_pnl")
        reverse_test_pass = bool(
            not forward.empty
            and not reverse.empty
            and forward_no_cost > pnl_sum(reverse, "no_cost_pnl")
            and pnl_sum(forward, "funding_adjusted_pnl") > pnl_sum(reverse, "funding_adjusted_pnl")
        )
        session_vs_baseline_pass = bool(
            not forward.empty
            and not ordinary.empty
            and not random.empty
            and forward_no_cost > pnl_sum(ordinary, "no_cost_pnl")
            and forward_no_cost > pnl_sum(random, "no_cost_pnl")
        )

    baseline_comparison = {
        "session_oos_no_cost_pnl": oos_no_cost,
        "ordinary_oos_no_cost_pnl": pnl_sum(
            baseline_trades(trades, symbols, "A").query("split == 'oos'") if not trades.empty else pd.DataFrame(),
            "no_cost_pnl",
        ),
        "random_oos_no_cost_pnl": pnl_sum(
            baseline_trades(trades, symbols, "D").query("split == 'oos'") if not trades.empty else pd.DataFrame(),
            "no_cost_pnl",
        ),
    }
    gates = {
        "train_pass": train_pass,
        "validation_pass": validation_pass,
        "oos_pass": oos_pass,
        "cost_aware_pass": cost_aware_pass,
        "funding_adjusted_pass": funding_adjusted_pass,
        "trade_count_pass": trade_count_pass,
        "concentration_pass": concentration_pass,
        "reverse_test_pass": reverse_test_pass,
        "session_vs_baseline_pass": session_vs_baseline_pass,
        "train_no_cost_pnl": train_no_cost,
        "validation_no_cost_pnl": validation_no_cost,
        "oos_no_cost_pnl": oos_no_cost,
        "oos_cost_aware_pnl": oos_cost,
        "oos_funding_adjusted_pnl": oos_funding,
        "trade_counts": trade_counts,
        "largest_symbol_pnl_share": pnl_share,
        "largest_symbol_pnl_symbol": pnl_symbol,
        "largest_symbol_trade_count_share": count_share,
        "largest_symbol_trade_count_symbol": count_symbol,
        "top_5pct_trade_pnl_contribution": top_share,
        "top_5pct_trade_pnl": top_pnl,
        "top_5pct_trade_count": top_count,
        "max_drawdown": max_dd,
        "max_drawdown_pct": max_dd_pct,
        "baseline_comparison": baseline_comparison,
    }
    continue_to_phase2 = all(
        bool(gates[key])
        for key in [
            "train_pass",
            "validation_pass",
            "oos_pass",
            "cost_aware_pass",
            "funding_adjusted_pass",
            "trade_count_pass",
            "concentration_pass",
            "reverse_test_pass",
            "session_vs_baseline_pass",
        ]
    )
    gates["continue_to_phase2"] = bool(continue_to_phase2)
    gates["final_decision"] = "continue_to_phase2" if continue_to_phase2 else "postmortem"
    return gates


def dataframe_records(df: pd.DataFrame, limit: int | None = None) -> list[dict[str, Any]]:
    """Convert a dataframe to JSON-safe records."""

    if df.empty:
        return []
    work = df.head(limit).copy() if limit is not None else df.copy()
    work = work.replace({np.nan: None})
    return json.loads(work.to_json(orient="records", force_ascii=False, date_format="iso"))


def build_data_quality_payload(
    *,
    symbols: list[str],
    timeframes: list[str],
    data_range: HistoryRange,
    session_timezone: str,
    report_timezone: str,
    funding_quality: dict[str, Any],
    symbol_quality: dict[str, Any],
    strict: bool,
) -> dict[str, Any]:
    """Build data_quality.json payload."""

    symbol_records = []
    for symbol in symbols:
        one_minute = symbol_quality.get(symbol, {}).get("1m", {})
        timeframe_records = {
            timeframe: symbol_quality.get(symbol, {}).get(timeframe, {})
            for timeframe in timeframes
        }
        symbol_records.append(
            {
                "symbol": symbol,
                "one_minute": one_minute,
                "timeframes": timeframe_records,
            }
        )
    return {
        "data_start_report": data_range.start.isoformat(),
        "data_end_exclusive_report": data_range.end_exclusive.isoformat(),
        "data_start_session": pd.Timestamp(data_range.start).tz_convert(session_timezone).isoformat(),
        "data_end_exclusive_session": pd.Timestamp(data_range.end_exclusive).tz_convert(session_timezone).isoformat(),
        "session_timezone": session_timezone,
        "report_timezone": report_timezone,
        "data_check_strict": strict,
        "expected_1m_count": expected_bar_count(data_range),
        "symbols": symbol_records,
        "funding": funding_quality,
        "all_market_data_complete": bool(
            symbol_records
            and all(record["one_minute"].get("complete") for record in symbol_records)
            and all(
                timeframe_record.get("complete")
                for record in symbol_records
                for timeframe_record in record["timeframes"].values()
            )
        ),
    }


def build_summary_payload(
    *,
    symbols: list[str],
    timeframes: list[str],
    data_range: HistoryRange,
    session_timezone: str,
    report_timezone: str,
    config: CsrbConfig,
    events: pd.DataFrame,
    trades: pd.DataFrame,
    data_quality: dict[str, Any],
    funding_quality: dict[str, Any],
    split_payload: dict[str, Any],
    gates: dict[str, Any],
    warnings: list[str],
) -> dict[str, Any]:
    """Build summary.json payload."""

    event_counts = {
        group: int((events["group"] == group).sum()) if not events.empty else 0
        for group in GROUPS
    }
    trade_counts = {
        group: int((trades["group"] == group).sum()) if not trades.empty else 0
        for group in GROUPS
    }
    return {
        "hypothesis_name": "Crypto Session Range Breakout",
        "version": "CSRB-v1",
        "status": "research_only",
        "phase": "phase1_event_and_fixed_hold",
        "data_start_report": data_range.start.isoformat(),
        "data_end_exclusive_report": data_range.end_exclusive.isoformat(),
        "data_start_session": pd.Timestamp(data_range.start).tz_convert(session_timezone).isoformat(),
        "data_end_exclusive_session": pd.Timestamp(data_range.end_exclusive).tz_convert(session_timezone).isoformat(),
        "session_timezone": session_timezone,
        "report_timezone": report_timezone,
        "symbols": symbols,
        "timeframes": timeframes,
        "parameters": to_jsonable(config),
        "sessions": {
            "asia_range_utc": "00:00-07:59",
            "europe_breakout_utc": "08:00-11:59",
            "europe_range_utc": "08:00-12:59",
            "us_breakout_utc": "13:00-17:59",
        },
        "split_dates": split_payload,
        "event_counts": event_counts,
        "trade_counts": trade_counts,
        "funding_data_complete": bool(funding_quality.get("funding_data_complete")),
        "funding_missing_inst_ids": funding_quality.get("missing_inst_ids") or [],
        "final_decision": gates["final_decision"],
        "train_pass": bool(gates["train_pass"]),
        "validation_pass": bool(gates["validation_pass"]),
        "oos_pass": bool(gates["oos_pass"]),
        "cost_aware_pass": bool(gates["cost_aware_pass"]),
        "funding_adjusted_pass": bool(gates["funding_adjusted_pass"]),
        "trade_count_pass": bool(gates["trade_count_pass"]),
        "concentration_pass": bool(gates["concentration_pass"]),
        "reverse_test_pass": bool(gates["reverse_test_pass"]),
        "session_vs_baseline_pass": bool(gates["session_vs_baseline_pass"]),
        "continue_to_phase2": bool(gates["continue_to_phase2"]),
        "strategy_development_allowed": False,
        "demo_live_allowed": False,
        "official_strategy_modified": False,
        "no_policy_can_be_traded": True,
        "gates": gates,
        "data_quality": data_quality,
        "warnings": warnings,
    }


def format_number(value: Any, digits: int = 4) -> str:
    """Format optional numeric values for Markdown."""

    try:
        number = float(value)
    except (TypeError, ValueError):
        return "N/A"
    if not np.isfinite(number):
        return "N/A"
    return f"{number:.{digits}f}"


def markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    """Render a compact Markdown table."""

    if not rows:
        return "- N/A"
    lines = [
        "| " + " | ".join(columns) + " |",
        "|" + "|".join("---" for _ in columns) + "|",
    ]
    for row in rows:
        values = []
        for column in columns:
            value = row.get(column)
            values.append(format_number(value, 4) if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def render_markdown_summary(
    summary: dict[str, Any],
    event_group_summary: pd.DataFrame,
    trade_group_summary: pd.DataFrame,
    reverse_test: pd.DataFrame,
    random_time_control: pd.DataFrame,
    concentration: pd.DataFrame,
) -> str:
    """Render summary.md with the required answers."""

    split_dates = summary.get("split_dates") or {}
    gates = summary.get("gates") or {}
    event_counts = summary.get("event_counts") or {}
    core_trade_rows = trade_group_summary[
        (trade_group_summary["group"].isin(["B", "C"]))
        & (trade_group_summary["timeframe"] == "15m")
    ] if not trade_group_summary.empty else pd.DataFrame()
    b_rows = core_trade_rows[core_trade_rows["group"] == "B"] if not core_trade_rows.empty else pd.DataFrame()
    c_rows = core_trade_rows[core_trade_rows["group"] == "C"] if not core_trade_rows.empty else pd.DataFrame()
    reverse_oos = reverse_test[
        (reverse_test["timeframe"] == "15m")
        & (reverse_test["source_session_type"] == "core_session_breakout")
        & (reverse_test["split"] == "oos")
    ] if not reverse_test.empty else pd.DataFrame()
    random_oos = random_time_control[
        (random_time_control["timeframe"] == "15m")
        & (random_time_control["source_session_type"] == "core_session_breakout")
        & (random_time_control["split"] == "oos")
    ] if not random_time_control.empty else pd.DataFrame()
    concentration_oos = concentration[
        (concentration["group"].isin(["B", "C"]))
        & (concentration["timeframe"] == "15m")
        & (concentration["split"] == "oos")
    ] if not concentration.empty else pd.DataFrame()
    event_rows = dataframe_records(event_group_summary, limit=30)
    trade_rows = dataframe_records(core_trade_rows, limit=18)

    return (
        "# CSRB-v1 Phase 1 Research\n\n"
        "## 1. CSRB-v1 是什么假设？\n"
        "CSRB-v1 tests whether crypto session transitions create trend-following edge: a clean range in a lower-activity session may break when Europe or US participation arrives, and the breakout may carry directional order flow. This is research-only and uses next-bar-open fixed-hold benchmarks.\n\n"
        "## 2. Session 定义是什么？\n"
        "- Asia range: 00:00-07:59 UTC\n"
        "- Europe breakout window: 08:00-11:59 UTC\n"
        "- Europe range: 08:00-12:59 UTC\n"
        "- US breakout window: 13:00-17:59 UTC\n\n"
        "## 3. 数据范围是什么？\n"
        f"- data_start_report={summary.get('data_start_report')}\n"
        f"- data_end_exclusive_report={summary.get('data_end_exclusive_report')}\n"
        f"- data_start_session={summary.get('data_start_session')}\n"
        f"- data_end_exclusive_session={summary.get('data_end_exclusive_session')}\n"
        f"- session_timezone={summary.get('session_timezone')}\n"
        f"- report_timezone={summary.get('report_timezone')}\n"
        f"- funding_data_complete={str(bool(summary.get('funding_data_complete'))).lower()}\n\n"
        "## 4. Train / Validation / OOS 日期？\n"
        f"- train={split_dates.get('train_start')} ({split_dates.get('train_start_report')}) to {split_dates.get('train_end')} ({split_dates.get('train_end_report')})\n"
        f"- validation={split_dates.get('validation_start')} ({split_dates.get('validation_start_report')}) to {split_dates.get('validation_end')} ({split_dates.get('validation_end_report')})\n"
        f"- oos={split_dates.get('oos_start')} ({split_dates.get('oos_start_report')}) to {split_dates.get('oos_end')} ({split_dates.get('oos_end_report')})\n\n"
        "## 5. A/B/C/D/E 事件数量？\n"
        + "\n".join(f"- {group}={event_counts.get(group, 0)}" for group in GROUPS)
        + "\n\n"
        "## 6. Event Group Summary\n"
        + markdown_table(event_rows, ["group", "session_type", "timeframe", "event_count", "mean_future_return_16"])
        + "\n\n"
        "## 7. Core 15m Trade Result\n"
        + markdown_table(trade_rows, ["group", "session_type", "timeframe", "split", "trade_count", "no_cost_pnl", "cost_aware_pnl", "funding_adjusted_pnl", "max_drawdown_pct"])
        + "\n\n"
        "## 8. Required Answers\n"
        f"1. Asia→Europe 是否有效？{str(bool(not b_rows.empty and pnl_sum(b_rows, 'no_cost_pnl') > 0)).lower()}\n"
        f"2. Europe→US 是否有效？{str(bool(not c_rows.empty and pnl_sum(c_rows, 'no_cost_pnl') > 0)).lower()}\n"
        f"3. Session breakout 是否优于普通 breakout？ordinary_oos_no_cost_pnl={format_number((gates.get('baseline_comparison') or {}).get('ordinary_oos_no_cost_pnl'), 4)}, session_oos_no_cost_pnl={format_number((gates.get('baseline_comparison') or {}).get('session_oos_no_cost_pnl'), 4)}\n"
        f"4. Session breakout 是否优于 random time control？{dataframe_records(random_oos, limit=3)}\n"
        f"5. Reverse test 是否明显更差？{str(bool(summary.get('reverse_test_pass'))).lower()} {dataframe_records(reverse_oos, limit=3)}\n"
        f"6. no-cost 是否通过？train={str(bool(summary.get('train_pass'))).lower()}, validation={str(bool(summary.get('validation_pass'))).lower()}, oos={str(bool(summary.get('oos_pass'))).lower()}\n"
        f"7. cost-aware 是否通过？{str(bool(summary.get('cost_aware_pass'))).lower()}\n"
        f"8. funding-adjusted 是否通过？{str(bool(summary.get('funding_adjusted_pass'))).lower()}\n"
        f"9. 收益是否集中？concentration_pass={str(bool(summary.get('concentration_pass'))).lower()} {dataframe_records(concentration_oos, limit=6)}\n"
        f"10. 是否允许 Phase 2？{str(bool(summary.get('continue_to_phase2'))).lower()}\n"
        "11. 是否允许修改正式策略？false\n"
        "12. 是否允许 demo/live？false\n\n"
        "## 9. Final Decision\n"
        f"- final_decision={summary.get('final_decision')}\n"
        f"- continue_to_phase2={str(bool(summary.get('continue_to_phase2'))).lower()}\n"
        "- strategy_development_allowed=false\n"
        "- demo_live_allowed=false\n"
    )


def render_postmortem(summary: dict[str, Any]) -> str:
    """Render postmortem draft."""

    failed = [
        key
        for key in [
            "train_pass",
            "validation_pass",
            "oos_pass",
            "cost_aware_pass",
            "funding_adjusted_pass",
            "trade_count_pass",
            "concentration_pass",
            "reverse_test_pass",
            "session_vs_baseline_pass",
        ]
        if not bool(summary.get(key))
    ]
    gates = summary.get("gates") or {}
    next_step = (
        "Phase 2 research only. Strategy development and demo/live remain blocked."
        if summary.get("continue_to_phase2")
        else "Postmortem. Do not tune Phase 1 parameters from OOS and do not develop a strategy."
    )
    return (
        "# CSRB-v1 Postmortem Draft\n\n"
        f"- final_decision={summary.get('final_decision')}\n"
        f"- failed_gates={failed}\n"
        f"- train_no_cost_pnl={format_number(gates.get('train_no_cost_pnl'), 6)}\n"
        f"- validation_no_cost_pnl={format_number(gates.get('validation_no_cost_pnl'), 6)}\n"
        f"- oos_no_cost_pnl={format_number(gates.get('oos_no_cost_pnl'), 6)}\n"
        f"- oos_cost_aware_pnl={format_number(gates.get('oos_cost_aware_pnl'), 6)}\n"
        f"- oos_funding_adjusted_pnl={format_number(gates.get('oos_funding_adjusted_pnl'), 6)}\n"
        f"- next_step={next_step}\n"
        "- strategy_development_allowed=false\n"
        "- demo_live_allowed=false\n"
        "- official_strategy_modified=false\n"
    )


def write_dataframe(path: Path, frame: pd.DataFrame, columns: list[str] | None = None) -> None:
    """Write a dataframe with stable empty-file columns."""

    path.parent.mkdir(parents=True, exist_ok=True)
    output = frame.copy()
    if columns is not None:
        for column in columns:
            if column not in output.columns:
                output[column] = np.nan
        output = output.loc[:, columns]
    output.to_csv(path, index=False)


def write_outputs(
    output_dir: Path,
    events: pd.DataFrame,
    trades: pd.DataFrame,
    summary: dict[str, Any],
    event_group_summary: pd.DataFrame,
    trade_group_summary: pd.DataFrame,
    by_symbol: pd.DataFrame,
    by_timeframe: pd.DataFrame,
    by_split: pd.DataFrame,
    session_summary: pd.DataFrame,
    concentration: pd.DataFrame,
    reverse_test: pd.DataFrame,
    random_time_control: pd.DataFrame,
    funding_summary: pd.DataFrame,
    data_quality: dict[str, Any],
    config: CsrbConfig,
) -> None:
    """Write all Phase 1 artifacts."""

    output_dir.mkdir(parents=True, exist_ok=True)
    write_dataframe(output_dir / "events.csv", events, event_columns(config))
    write_dataframe(output_dir / "trades.csv", trades, TRADE_COLUMNS)
    write_dataframe(output_dir / "event_group_summary.csv", event_group_summary)
    write_dataframe(output_dir / "trade_group_summary.csv", trade_group_summary)
    write_dataframe(output_dir / "by_symbol.csv", by_symbol)
    write_dataframe(output_dir / "by_timeframe.csv", by_timeframe)
    write_dataframe(output_dir / "by_split.csv", by_split)
    write_dataframe(output_dir / "session_summary.csv", session_summary)
    write_dataframe(output_dir / "concentration.csv", concentration)
    write_dataframe(output_dir / "reverse_test.csv", reverse_test)
    write_dataframe(output_dir / "random_time_control.csv", random_time_control)
    write_dataframe(output_dir / "funding_summary.csv", funding_summary)
    (output_dir / "summary.json").write_text(
        json.dumps(to_jsonable(summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "data_quality.json").write_text(
        json.dumps(to_jsonable(data_quality), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "summary.md").write_text(
        render_markdown_summary(summary, event_group_summary, trade_group_summary, reverse_test, random_time_control, concentration),
        encoding="utf-8",
    )
    (output_dir / "postmortem_draft.md").write_text(render_postmortem(summary), encoding="utf-8")


def run_research(
    *,
    symbols: list[str],
    timeframes: list[str],
    data_range: HistoryRange,
    session_timezone: str,
    report_timezone: str,
    output_dir: Path,
    funding_dir: Path,
    database_path: Path,
    config: CsrbConfig,
    data_check_strict: bool,
    logger: logging.Logger | None = None,
    bars_by_symbol: dict[str, pd.DataFrame] | None = None,
) -> dict[str, Any]:
    """Run CSRB-v1 Phase 1 research."""

    validate_config(config)
    resolve_timezone(session_timezone)
    resolve_timezone(report_timezone)
    warnings: list[str] = []
    funding_histories, funding_quality = load_funding_histories(
        funding_dir,
        symbols,
        data_range,
        data_range.start.date().isoformat(),
        data_range.end_display.date().isoformat(),
    )
    splits = build_time_splits(data_range, session_timezone)
    split_payload = split_dates_payload(splits, report_timezone)

    bars_by_key: dict[tuple[str, str], pd.DataFrame] = {}
    symbol_quality: dict[str, Any] = {}
    event_frames: list[pd.DataFrame] = []

    for symbol in symbols:
        if logger:
            log_event(logger, logging.INFO, "load_symbol", "Loading symbol history", symbol=symbol)
        if bars_by_symbol is not None and symbol in bars_by_symbol:
            bars_1m = normalize_1m_bars(bars_by_symbol[symbol], data_range.timezone_name, session_timezone)
            start = pd.Timestamp(data_range.start).tz_convert(session_timezone)
            end = pd.Timestamp(data_range.end_exclusive).tz_convert(session_timezone)
            bars_1m = bars_1m[(bars_1m["datetime"] >= start) & (bars_1m["datetime"] < end)].copy()
        else:
            bars_1m = load_1m_bars_from_sqlite(symbol, data_range, database_path, session_timezone)
        one_minute_quality = analyze_1m_quality(symbol, bars_1m, data_range, session_timezone)
        symbol_quality[symbol] = {"1m": one_minute_quality}
        if not one_minute_quality["complete"] and data_check_strict:
            raise CsrbResearchError(f"strict data check failed for {symbol}: {one_minute_quality}")
        if bars_1m.empty:
            raise CsrbResearchError(f"required symbol has no data: {symbol}")

        for timeframe in timeframes:
            resampled = resample_ohlcv_closed(bars_1m, timeframe, data_range, session_timezone)
            timeframe_quality = analyze_timeframe_quality(symbol, timeframe, resampled, one_minute_quality)
            symbol_quality[symbol][timeframe] = timeframe_quality
            if data_check_strict and not timeframe_quality["complete"]:
                raise CsrbResearchError(f"strict resample data check failed for {symbol} {timeframe}: {timeframe_quality}")
            bars_by_key[(symbol, timeframe)] = add_csrb_indicators(resampled, config)
            events, event_warnings = generate_events_for_frame(resampled, symbol, timeframe, config, report_timezone)
            event_frames.append(events)
            warnings.extend(event_warnings)

    events = pd.concat(event_frames, ignore_index=True) if event_frames else pd.DataFrame(columns=event_columns(config))
    events = assign_event_ids(events)

    trades, trade_warnings = simulate_fixed_hold_trades(events, bars_by_key, config, funding_histories, splits)
    warnings.extend(trade_warnings)

    data_quality = build_data_quality_payload(
        symbols=symbols,
        timeframes=timeframes,
        data_range=data_range,
        session_timezone=session_timezone,
        report_timezone=report_timezone,
        funding_quality=funding_quality,
        symbol_quality=symbol_quality,
        strict=data_check_strict,
    )
    event_group_summary = summarize_events(events, config)
    trade_group_summary = summarize_trades(trades, ["group", "session_type", "timeframe", "split"], config)
    by_symbol = summarize_trades(trades, ["group", "session_type", "timeframe", "split", "symbol"], config)
    by_timeframe = summarize_trades(trades, ["group", "session_type", "timeframe", "split"], config)
    by_split = summarize_trades(trades, ["group", "session_type", "split"], config)
    session_summary = build_session_summary(trades, config)
    concentration = build_concentration_summary(trades, config)
    reverse_test = build_reverse_test(trades, symbols)
    random_time_control = build_random_time_control(trades, symbols)
    funding_summary = build_funding_summary(trades, funding_quality)
    gates = evaluate_phase1_gates(trades, symbols, bool(funding_quality.get("funding_data_complete")), config)
    summary = build_summary_payload(
        symbols=symbols,
        timeframes=timeframes,
        data_range=data_range,
        session_timezone=session_timezone,
        report_timezone=report_timezone,
        config=config,
        events=events,
        trades=trades,
        data_quality=data_quality,
        funding_quality=funding_quality,
        split_payload=split_payload,
        gates=gates,
        warnings=warnings,
    )
    write_outputs(
        output_dir=output_dir,
        events=events,
        trades=trades,
        summary=summary,
        event_group_summary=event_group_summary,
        trade_group_summary=trade_group_summary,
        by_symbol=by_symbol,
        by_timeframe=by_timeframe,
        by_split=by_split,
        session_summary=session_summary,
        concentration=concentration,
        reverse_test=reverse_test,
        random_time_control=random_time_control,
        funding_summary=funding_summary,
        data_quality=data_quality,
        config=config,
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""

    args = parse_args(argv)
    ensure_headless_runtime()
    logger = setup_logging("research_csrb_v1", verbose=bool(args.verbose))
    symbols = parse_csv_list(args.symbols)
    if not symbols:
        raise CsrbResearchError("--symbols must not be empty")
    timeframes = parse_timeframes(args.timeframes)
    data_range = parse_history_range(
        start_arg=args.start,
        end_arg=args.end,
        interval_delta=timedelta(minutes=1),
        timezone_name=args.report_timezone,
    )
    config = build_config(args)
    summary = run_research(
        symbols=symbols,
        timeframes=timeframes,
        data_range=data_range,
        session_timezone=args.session_timezone,
        report_timezone=args.report_timezone,
        output_dir=resolve_path(args.output_dir),
        funding_dir=resolve_path(args.funding_dir),
        database_path=resolve_path(args.database_path),
        config=config,
        data_check_strict=bool(args.data_check_strict),
        logger=logger,
    )
    log_event(
        logger,
        logging.INFO,
        "research_complete",
        "CSRB-v1 Phase 1 research complete",
        final_decision=summary.get("final_decision"),
        continue_to_phase2=summary.get("continue_to_phase2"),
        output_dir=str(resolve_path(args.output_dir)),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
