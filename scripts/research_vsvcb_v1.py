#!/usr/bin/env python3
"""Research-only VSVCB-v1 event study and fixed-hold benchmark."""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from common_runtime import PROJECT_ROOT, ensure_headless_runtime, log_event, setup_logging, to_jsonable
from history_time_utils import DEFAULT_TIMEZONE, HistoryRange, expected_bar_count, parse_history_range


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
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "reports" / "research" / "vsvcb_v1"
DEFAULT_FUNDING_DIR = PROJECT_ROOT / "data" / "funding" / "okx"
DEFAULT_DATABASE_PATH = PROJECT_ROOT / ".vntrader" / "database.db"

GROUPS = ["A", "B", "C", "D", "E"]
PLACEHOLDER_FILES = [
    "parameter_plateau.csv",
    "randomization_test.csv",
    "factor_decomposition.csv",
]

EVENT_COLUMNS = [
    "event_id",
    "timestamp",
    "symbol",
    "timeframe",
    "direction",
    "group",
    "close",
    "open_next",
    "breakout_boundary",
    "bb_width",
    "bb_width_percentile",
    "squeeze",
    "volume",
    "volume_ma_prev",
    "volume_ratio",
    "volume_confirm",
    "atr",
    "entry_close_theoretical",
    "entry_next_open",
    "future_return_3",
    "future_return_5",
    "future_return_10",
    "future_return_20",
    "mfe_10",
    "mae_10",
    "mfe_mae_ratio_10",
    "reversal_flag_3",
    "reversal_flag_5",
    "reversal_flag_10",
    "funding_crossed",
    "funding_cost_estimate",
    "bar_index",
    "inst_id",
]

TRADE_COLUMNS = [
    "trade_id",
    "event_id",
    "symbol",
    "timeframe",
    "group",
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
    "funding_pnl",
    "cost_aware_pnl",
    "funding_adjusted_pnl",
    "mfe",
    "mae",
    "mfe_mae_ratio",
    "reversal_flags",
    "split",
]


class VsvcbResearchError(Exception):
    """Raised when VSVCB-v1 research cannot continue."""


@dataclass(frozen=True, slots=True)
class VsvcbConfig:
    """VSVCB-v1 research parameters."""

    bb_length: int = 20
    bb_std: float = 2.0
    bb_width_lookback: int = 200
    squeeze_quantile: float = 0.2
    breakout_window: int = 20
    volume_ma_window: int = 20
    volume_ratio: float = 1.5
    hold_bars: int = 10
    event_horizons: tuple[int, ...] = (3, 5, 10, 20)
    fixed_notional: float = 1000.0
    fee_bps_per_side: float = 5.0
    slippage_bps_per_side: float = 5.0


@dataclass(frozen=True, slots=True)
class TimeSplit:
    """One chronological split interval."""

    name: str
    start: pd.Timestamp
    end: pd.Timestamp


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(
        description="VSVCB-v1 research-only event study and fixed-hold benchmark."
    )
    parser.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS))
    parser.add_argument("--timeframes", default=",".join(DEFAULT_TIMEFRAMES))
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--end", default=DEFAULT_END)
    parser.add_argument("--timezone", default=DEFAULT_TIMEZONE)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--funding-dir", default=str(DEFAULT_FUNDING_DIR))
    parser.add_argument("--database-path", default=str(DEFAULT_DATABASE_PATH))
    parser.add_argument("--fixed-notional", type=float, default=1000.0)
    parser.add_argument("--fee-bps-per-side", type=float, default=5.0)
    parser.add_argument("--slippage-bps-per-side", type=float, default=5.0)
    parser.add_argument("--bb-length", type=int, default=20)
    parser.add_argument("--bb-std", type=float, default=2.0)
    parser.add_argument("--bb-width-lookback", type=int, default=200)
    parser.add_argument("--squeeze-quantile", type=float, default=0.2)
    parser.add_argument("--breakout-window", type=int, default=20)
    parser.add_argument("--volume-ma-window", type=int, default=20)
    parser.add_argument("--volume-ratio", type=float, default=1.5)
    parser.add_argument("--hold-bars", type=int, default=10)
    parser.add_argument("--event-horizons", default="3,5,10,20")
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
            seen.add(item)
            parsed.append(item)
    return parsed


def parse_timeframes(value: str | Iterable[str]) -> list[str]:
    """Parse and validate requested timeframes."""

    timeframes = parse_csv_list(value)
    unsupported = [item for item in timeframes if item not in TIMEFRAME_MINUTES]
    if unsupported:
        raise VsvcbResearchError(f"unsupported timeframes: {unsupported}")
    if not timeframes:
        raise VsvcbResearchError("--timeframes must not be empty")
    return timeframes


def parse_event_horizons(value: str | Iterable[int]) -> tuple[int, ...]:
    """Parse positive integer event horizons."""

    if isinstance(value, str):
        raw = parse_csv_list(value)
    else:
        raw = [str(item) for item in value]
    horizons = tuple(sorted({int(item) for item in raw if str(item).strip()}))
    if not horizons or any(item <= 0 for item in horizons):
        raise VsvcbResearchError("--event-horizons must contain positive integers")
    return horizons


def build_config(args: argparse.Namespace) -> VsvcbConfig:
    """Build research config from parsed CLI args."""

    return VsvcbConfig(
        bb_length=int(args.bb_length),
        bb_std=float(args.bb_std),
        bb_width_lookback=int(args.bb_width_lookback),
        squeeze_quantile=float(args.squeeze_quantile),
        breakout_window=int(args.breakout_window),
        volume_ma_window=int(args.volume_ma_window),
        volume_ratio=float(args.volume_ratio),
        hold_bars=int(args.hold_bars),
        event_horizons=parse_event_horizons(args.event_horizons),
        fixed_notional=float(args.fixed_notional),
        fee_bps_per_side=float(args.fee_bps_per_side),
        slippage_bps_per_side=float(args.slippage_bps_per_side),
    )


def validate_config(config: VsvcbConfig) -> None:
    """Validate parameter ranges without optimizing them."""

    positive_ints = {
        "bb_length": config.bb_length,
        "bb_width_lookback": config.bb_width_lookback,
        "breakout_window": config.breakout_window,
        "volume_ma_window": config.volume_ma_window,
        "hold_bars": config.hold_bars,
    }
    invalid = [name for name, value in positive_ints.items() if int(value) <= 0]
    if invalid:
        raise VsvcbResearchError(f"parameters must be positive: {invalid}")
    if config.fixed_notional <= 0:
        raise VsvcbResearchError("--fixed-notional must be positive")
    if not 0.0 < config.squeeze_quantile < 1.0:
        raise VsvcbResearchError("--squeeze-quantile must be between 0 and 1")
    if config.bb_std <= 0:
        raise VsvcbResearchError("--bb-std must be positive")


def split_vt_symbol(vt_symbol: str) -> tuple[str, str]:
    """Split a vt_symbol into database symbol and exchange."""

    symbol, separator, exchange = str(vt_symbol).partition(".")
    if not separator or not symbol or not exchange:
        raise VsvcbResearchError(f"invalid vt_symbol: {vt_symbol}")
    return symbol, exchange


def symbol_to_inst_id(vt_symbol: str) -> str:
    """Map a local vt_symbol to OKX instId."""

    symbol, _exchange = split_vt_symbol(vt_symbol)
    root = symbol.removesuffix("_OKX")
    if root.endswith("_SWAP"):
        pair = root[: -len("_SWAP")]
    else:
        pair = root
    if pair.endswith("USDT"):
        return f"{pair[:-4]}-USDT-SWAP"
    return root.replace("_", "-")


def format_timestamp(value: Any) -> str | None:
    """Format a timestamp as ISO text for reports."""

    if value is None or pd.isna(value):
        return None
    timestamp = pd.Timestamp(value)
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


def load_1m_bars_from_sqlite(
    vt_symbol: str,
    history_range: HistoryRange,
    database_path: Path,
) -> pd.DataFrame:
    """Load local vn.py sqlite 1m bars for one symbol."""

    if not database_path.exists():
        raise VsvcbResearchError(f"database not found: {database_path}")

    symbol, exchange = split_vt_symbol(vt_symbol)
    query_start = history_range.start.replace(tzinfo=None).isoformat(sep=" ", timespec="seconds")
    query_end = history_range.end_exclusive.replace(tzinfo=None).isoformat(sep=" ", timespec="seconds")
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
    return normalize_1m_bars(frame, history_range.timezone_name)


def normalize_1m_bars(frame: pd.DataFrame, timezone_name: str) -> pd.DataFrame:
    """Normalize an injected or sqlite-loaded 1m OHLCV frame."""

    columns = ["datetime", "open", "high", "low", "close", "volume"]
    if frame.empty:
        return pd.DataFrame(columns=columns)
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise VsvcbResearchError(f"1m bars missing columns: {missing}")

    normalized = frame.loc[:, columns].copy()
    timestamps = pd.to_datetime(normalized["datetime"], errors="coerce")
    if timestamps.isna().any():
        raise VsvcbResearchError("1m bars contain unparsable datetime values")
    if timestamps.dt.tz is None:
        timestamps = timestamps.dt.tz_localize(timezone_name)
    else:
        timestamps = timestamps.dt.tz_convert(timezone_name)
    normalized["datetime"] = timestamps
    for column in ["open", "high", "low", "close", "volume"]:
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce")
    normalized = normalized.dropna(subset=["datetime", "open", "high", "low", "close", "volume"])
    normalized = normalized.sort_values("datetime", kind="stable").reset_index(drop=True)
    return normalized


def missing_ranges_from_index(missing: pd.DatetimeIndex, limit: int = 10) -> list[dict[str, Any]]:
    """Compress missing minute timestamps into a small report list."""

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
        ranges.append(
            {
                "start": format_timestamp(start),
                "end": format_timestamp(previous),
                "missing_count": count,
            }
        )
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
    history_range: HistoryRange,
) -> dict[str, Any]:
    """Build one-symbol 1m data quality summary."""

    expected = pd.date_range(
        history_range.start,
        history_range.end_exclusive - pd.Timedelta(minutes=1),
        freq="min",
    )
    actual = pd.DatetimeIndex(bars_1m["datetime"]) if not bars_1m.empty else pd.DatetimeIndex([], tz=history_range.timezone_name)
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
    history_range: HistoryRange | None = None,
) -> pd.DataFrame:
    """Resample 1m OHLCV into completed closed bars without using future bars."""

    if timeframe not in TIMEFRAME_MINUTES:
        raise VsvcbResearchError(f"unsupported timeframe: {timeframe}")
    minutes = TIMEFRAME_MINUTES[timeframe]
    columns = ["open_time", "datetime", "open", "high", "low", "close", "volume"]
    if bars_1m.empty:
        return pd.DataFrame(columns=columns)

    working = bars_1m.sort_values("datetime", kind="stable").drop_duplicates("datetime", keep="last").copy()
    anchor = pd.Timestamp(history_range.start if history_range is not None else working["datetime"].iloc[0])
    if anchor.tzinfo is None:
        anchor = anchor.tz_localize(working["datetime"].iloc[0].tz)
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


def percentile_last(values: np.ndarray) -> float:
    """Return percentile rank of the last element within the rolling window."""

    if len(values) == 0 or not np.isfinite(values[-1]):
        return np.nan
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return np.nan
    return float(np.sum(finite <= values[-1]) / len(finite))


def add_vsvcb_indicators(frame: pd.DataFrame, config: VsvcbConfig) -> pd.DataFrame:
    """Calculate VSVCB-v1 event-study features without lookahead."""

    out = frame.copy().reset_index(drop=True)
    close = pd.to_numeric(out["close"], errors="coerce")
    high = pd.to_numeric(out["high"], errors="coerce")
    low = pd.to_numeric(out["low"], errors="coerce")
    volume = pd.to_numeric(out["volume"], errors="coerce")

    middle = close.rolling(config.bb_length, min_periods=config.bb_length).mean()
    std = close.rolling(config.bb_length, min_periods=config.bb_length).std(ddof=0)
    upper = middle + config.bb_std * std
    lower = middle - config.bb_std * std
    bb_width_raw = (upper - lower) / middle.replace(0.0, np.nan)
    width_quantile = bb_width_raw.rolling(
        config.bb_width_lookback,
        min_periods=config.bb_width_lookback,
    ).quantile(config.squeeze_quantile)
    width_percentile = bb_width_raw.rolling(
        config.bb_width_lookback,
        min_periods=config.bb_width_lookback,
    ).apply(percentile_last, raw=True)

    volume_ma_prev = volume.rolling(
        config.volume_ma_window,
        min_periods=config.volume_ma_window,
    ).mean().shift(1)
    upper_boundary = high.rolling(
        config.breakout_window,
        min_periods=config.breakout_window,
    ).max().shift(1)
    lower_boundary = low.rolling(
        config.breakout_window,
        min_periods=config.breakout_window,
    ).min().shift(1)

    previous_close = close.shift(1)
    true_range = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr14 = true_range.rolling(14, min_periods=14).mean()

    out["bb_middle"] = middle
    out["bb_upper"] = upper
    out["bb_lower"] = lower
    out["bb_width_raw"] = bb_width_raw
    out["bb_width"] = bb_width_raw.shift(1)
    out["bb_width_percentile"] = width_percentile.shift(1)
    out["squeeze_threshold"] = width_quantile.shift(1)
    out["squeeze"] = (bb_width_raw.shift(1) <= width_quantile.shift(1)).fillna(False)
    out["volume_ma_prev"] = volume_ma_prev
    out["volume_ratio"] = volume / volume_ma_prev.replace(0.0, np.nan)
    out["volume_confirm"] = (out["volume_ratio"] >= config.volume_ratio).fillna(False)
    out["upper_boundary"] = upper_boundary
    out["lower_boundary"] = lower_boundary
    out["long_breakout"] = (close > upper_boundary).fillna(False)
    out["short_breakout"] = (close < lower_boundary).fillna(False)
    out["atr"] = atr14
    out["open_next"] = out["open"].shift(-1)
    out["entry_next_open"] = out["open_next"]
    return out


def directional_return(entry_price: float, exit_price: float, direction: str) -> float:
    """Return signed directional return."""

    if not np.isfinite(entry_price) or entry_price == 0 or not np.isfinite(exit_price):
        return np.nan
    sign = 1.0 if direction == "long" else -1.0
    return float(sign * (exit_price / entry_price - 1.0))


def directional_mfe_mae(
    frame: pd.DataFrame,
    pos: int,
    direction: str,
    reference_price: float,
    horizon: int,
) -> tuple[float, float, float]:
    """Calculate directional MFE, MAE, and ratio over future bars."""

    if not np.isfinite(reference_price) or reference_price == 0 or pos + horizon >= len(frame.index):
        return np.nan, np.nan, np.nan
    future = frame.iloc[pos + 1 : pos + horizon + 1]
    if future.empty:
        return np.nan, np.nan, np.nan
    max_high = float(pd.to_numeric(future["high"], errors="coerce").max())
    min_low = float(pd.to_numeric(future["low"], errors="coerce").min())
    if direction == "long":
        mfe = max_high / reference_price - 1.0
        mae = min_low / reference_price - 1.0
    else:
        mfe = reference_price / min_low - 1.0 if min_low > 0 else np.nan
        mae = reference_price / max_high - 1.0 if max_high > 0 else np.nan
    ratio = mfe / abs(mae) if np.isfinite(mfe) and np.isfinite(mae) and mae != 0 else np.nan
    return float(mfe), float(mae), float(ratio) if np.isfinite(ratio) else np.nan


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
        raise VsvcbResearchError(f"funding CSV missing funding_time columns: {path}")
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
    history_range: HistoryRange,
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
    start_utc = pd.Timestamp(history_range.start_utc)
    end_utc = pd.Timestamp(history_range.end_exclusive_utc)
    warnings: list[str] = []
    starts_before_window = bool(first_time <= start_utc + pd.Timedelta(minutes=1))
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
    history_range: HistoryRange,
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
            records[inst_id] = analyze_funding_quality(histories[inst_id], history_range, None)
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
        records[inst_id] = analyze_funding_quality(histories[inst_id], history_range, path)

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
        entry_utc = entry_utc.tz_localize(DEFAULT_TIMEZONE)
    if exit_utc.tzinfo is None:
        exit_utc = exit_utc.tz_localize(DEFAULT_TIMEZONE)
    entry_utc = entry_utc.tz_convert("UTC")
    exit_utc = exit_utc.tz_convert("UTC")
    if exit_utc < entry_utc:
        entry_utc, exit_utc = exit_utc, entry_utc
    mask = (funding_df["funding_time_utc"] >= entry_utc) & (funding_df["funding_time_utc"] <= exit_utc)
    rates = pd.to_numeric(funding_df.loc[mask, "funding_rate"], errors="coerce").dropna()
    pnl = float(sum(signed_funding_pnl(notional, float(rate), direction) for rate in rates))
    return pnl, int(len(rates.index))


def direction_metrics_for_event(
    frame: pd.DataFrame,
    pos: int,
    direction: str,
    horizons: tuple[int, ...],
) -> dict[str, Any]:
    """Calculate future returns, MFE/MAE, and reversal flags for an event."""

    close = float(frame.iloc[pos]["close"])
    metrics: dict[str, Any] = {}
    for horizon in horizons:
        value = np.nan
        if pos + horizon < len(frame.index):
            value = directional_return(close, float(frame.iloc[pos + horizon]["close"]), direction)
        metrics[f"future_return_{horizon}"] = value
        metrics[f"reversal_flag_{horizon}"] = bool(np.isfinite(value) and value < 0.0)
    mfe, mae, ratio = directional_mfe_mae(frame, pos, direction, close, 10)
    metrics["mfe_10"] = mfe
    metrics["mae_10"] = mae
    metrics["mfe_mae_ratio_10"] = ratio
    return metrics


def groups_for_signal(squeeze: bool, volume_confirm: bool) -> list[str]:
    """Return overlapping A/B/C/D group labels for one breakout."""

    groups = ["A"]
    if squeeze:
        groups.append("B")
    if volume_confirm:
        groups.append("C")
    if squeeze and volume_confirm:
        groups.append("D")
    return groups


def build_event_row(
    frame: pd.DataFrame,
    pos: int,
    symbol: str,
    timeframe: str,
    direction: str,
    group: str,
    config: VsvcbConfig,
    funding_histories: dict[str, pd.DataFrame] | None = None,
) -> dict[str, Any]:
    """Build one event row."""

    row = frame.iloc[pos]
    inst_id = symbol_to_inst_id(symbol)
    boundary = row["upper_boundary"] if direction == "long" else row["lower_boundary"]
    metrics = direction_metrics_for_event(frame, pos, direction, config.event_horizons)
    entry_next_time = frame.iloc[pos + 1]["open_time"] if pos + 1 < len(frame.index) else None
    exit_time = frame.iloc[pos + config.hold_bars + 1]["open_time"] if pos + config.hold_bars + 1 < len(frame.index) else None
    funding_df = (funding_histories or {}).get(inst_id, pd.DataFrame(columns=["funding_time_utc", "funding_rate"]))
    funding_pnl, funding_count = funding_pnl_for_interval(
        funding_df,
        entry_next_time,
        exit_time,
        direction,
        config.fixed_notional,
    )
    event = {
        "event_id": "",
        "timestamp": row["datetime"],
        "symbol": symbol,
        "timeframe": timeframe,
        "direction": direction,
        "group": group,
        "close": float(row["close"]),
        "open_next": finite_float(row.get("open_next"), default=np.nan),
        "breakout_boundary": finite_float(boundary, default=np.nan),
        "bb_width": finite_float(row.get("bb_width"), default=np.nan),
        "bb_width_percentile": finite_float(row.get("bb_width_percentile"), default=np.nan),
        "squeeze": bool(row.get("squeeze")),
        "volume": finite_float(row.get("volume"), default=np.nan),
        "volume_ma_prev": finite_float(row.get("volume_ma_prev"), default=np.nan),
        "volume_ratio": finite_float(row.get("volume_ratio"), default=np.nan),
        "volume_confirm": bool(row.get("volume_confirm")),
        "atr": finite_float(row.get("atr"), default=np.nan),
        "entry_close_theoretical": float(row["close"]),
        "entry_next_open": finite_float(row.get("entry_next_open"), default=np.nan),
        "funding_crossed": bool(funding_count > 0),
        "funding_cost_estimate": float(-funding_pnl),
        "bar_index": int(pos),
        "inst_id": inst_id,
    }
    for horizon in (3, 5, 10, 20):
        event[f"future_return_{horizon}"] = metrics.get(f"future_return_{horizon}", np.nan)
        event[f"reversal_flag_{horizon}"] = bool(metrics.get(f"reversal_flag_{horizon}", False))
    event["mfe_10"] = metrics["mfe_10"]
    event["mae_10"] = metrics["mae_10"]
    event["mfe_mae_ratio_10"] = metrics["mfe_mae_ratio_10"]
    return event


def generate_events_for_frame(
    frame: pd.DataFrame,
    symbol: str,
    timeframe: str,
    config: VsvcbConfig,
    funding_histories: dict[str, pd.DataFrame] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Generate A/B/C/D/E events for one symbol/timeframe."""

    rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    working = add_vsvcb_indicators(frame, config)
    anomaly_count = 0
    signal_mask = (working["long_breakout"] | working["short_breakout"]).fillna(False)
    for pos in np.flatnonzero(signal_mask.to_numpy(dtype=bool)):
        long_signal = bool(working.iloc[pos]["long_breakout"])
        short_signal = bool(working.iloc[pos]["short_breakout"])
        if long_signal and short_signal:
            anomaly_count += 1
            continue
        direction = "long" if long_signal else "short"
        signal_groups = groups_for_signal(
            bool(working.iloc[pos]["squeeze"]),
            bool(working.iloc[pos]["volume_confirm"]),
        )
        original_d: dict[str, Any] | None = None
        for group in signal_groups:
            event = build_event_row(working, pos, symbol, timeframe, direction, group, config, funding_histories)
            rows.append(event)
            if group == "D":
                original_d = event
        if original_d is not None:
            reverse_direction = "short" if direction == "long" else "long"
            reverse = build_event_row(working, pos, symbol, timeframe, reverse_direction, "E", config, funding_histories)
            rows.append(reverse)
    if anomaly_count:
        warnings.append(f"{symbol} {timeframe}: skipped same-bar long/short anomalies={anomaly_count}")
    events = pd.DataFrame(rows, columns=EVENT_COLUMNS)
    if not events.empty:
        events = events.sort_values(["timestamp", "symbol", "timeframe", "group", "direction"], kind="stable").reset_index(drop=True)
    return events, warnings


def assign_event_ids(events: pd.DataFrame) -> pd.DataFrame:
    """Assign stable event ids."""

    if events.empty:
        return pd.DataFrame(columns=EVENT_COLUMNS)
    out = events.copy().reset_index(drop=True)
    out["event_id"] = [
        f"vsvcb_v1_{index:08d}_{row.symbol}_{row.timeframe}_{row.group}_{row.direction}"
        for index, row in enumerate(out.itertuples(index=False), start=1)
    ]
    return out


def build_time_splits(history_range: HistoryRange) -> list[TimeSplit]:
    """Build chronological 60/20/20 time splits."""

    start = pd.Timestamp(history_range.start)
    end = pd.Timestamp(history_range.end_exclusive)
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


def calculate_trade_mfe_mae(
    frame: pd.DataFrame,
    event_pos: int,
    direction: str,
    entry_price: float,
    hold_bars: int,
) -> tuple[float, float, float]:
    """Calculate trade MFE/MAE during the fixed hold interval."""

    if event_pos + hold_bars >= len(frame.index):
        return np.nan, np.nan, np.nan
    future = frame.iloc[event_pos + 1 : event_pos + hold_bars + 1]
    if future.empty or not np.isfinite(entry_price) or entry_price == 0:
        return np.nan, np.nan, np.nan
    max_high = float(pd.to_numeric(future["high"], errors="coerce").max())
    min_low = float(pd.to_numeric(future["low"], errors="coerce").min())
    if direction == "long":
        mfe = max_high / entry_price - 1.0
        mae = min_low / entry_price - 1.0
    else:
        mfe = entry_price / min_low - 1.0 if min_low > 0 else np.nan
        mae = entry_price / max_high - 1.0 if max_high > 0 else np.nan
    ratio = mfe / abs(mae) if np.isfinite(mfe) and np.isfinite(mae) and mae != 0 else np.nan
    return float(mfe), float(mae), float(ratio) if np.isfinite(ratio) else np.nan


def simulate_fixed_hold_trades(
    events: pd.DataFrame,
    bars_by_key: dict[tuple[str, str], pd.DataFrame],
    config: VsvcbConfig,
    funding_histories: dict[str, pd.DataFrame],
    splits: list[TimeSplit],
) -> tuple[pd.DataFrame, list[str]]:
    """Simulate fixed-hold trades with single-symbol single-position filtering."""

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
        exit_pos = pos + config.hold_bars + 1
        entry_pos = pos + 1
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
        gross_return = directional_return(entry_price, exit_price, row.direction)
        no_cost_pnl = config.fixed_notional * gross_return if np.isfinite(gross_return) else np.nan
        fee_cost = config.fixed_notional * (config.fee_bps_per_side / 10000.0) * 2.0
        slippage_cost = config.fixed_notional * (config.slippage_bps_per_side / 10000.0) * 2.0
        funding_df = funding_histories.get(row.inst_id, pd.DataFrame(columns=["funding_time_utc", "funding_rate"]))
        funding_pnl, _funding_count = funding_pnl_for_interval(
            funding_df,
            entry_time,
            exit_time,
            row.direction,
            config.fixed_notional,
        )
        cost_aware_pnl = no_cost_pnl - fee_cost - slippage_cost
        funding_adjusted_pnl = cost_aware_pnl + funding_pnl
        mfe, mae, ratio = calculate_trade_mfe_mae(frame, pos, row.direction, entry_price, config.hold_bars)
        reversal_flags = ";".join(
            f"{horizon}:{bool(getattr(row, f'reversal_flag_{horizon}', False))}"
            for horizon in (3, 5, 10)
        )
        split = assign_split_for_time(entry_time, splits)
        trades.append(
            {
                "trade_id": f"vsvcb_trade_{len(trades) + 1:08d}",
                "event_id": row.event_id,
                "symbol": row.symbol,
                "timeframe": row.timeframe,
                "group": row.group,
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
                "funding_pnl": funding_pnl,
                "cost_aware_pnl": cost_aware_pnl,
                "funding_adjusted_pnl": funding_adjusted_pnl,
                "mfe": mfe,
                "mae": mae,
                "mfe_mae_ratio": ratio,
                "reversal_flags": reversal_flags,
                "split": split,
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


def summarize_trades(trades: pd.DataFrame, group_columns: list[str], config: VsvcbConfig) -> pd.DataFrame:
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
        "mfe_mean",
        "mae_mean",
        "mfe_mae_ratio_mean",
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
                "no_cost_pnl": float(pd.to_numeric(group["no_cost_pnl"], errors="coerce").sum()),
                "cost_aware_pnl": float(pd.to_numeric(group["cost_aware_pnl"], errors="coerce").sum()),
                "funding_pnl": float(pd.to_numeric(group["funding_pnl"], errors="coerce").sum()),
                "funding_adjusted_pnl": float(pnl_series.sum()),
                "avg_gross_return": float(pd.to_numeric(group["gross_return"], errors="coerce").mean()),
                "win_rate": float((pd.to_numeric(group["funding_adjusted_pnl"], errors="coerce") > 0).mean()),
                "mfe_mean": float(pd.to_numeric(group["mfe"], errors="coerce").mean()),
                "mae_mean": float(pd.to_numeric(group["mae"], errors="coerce").mean()),
                "mfe_mae_ratio_mean": float(pd.to_numeric(group["mfe_mae_ratio"], errors="coerce").mean()),
                "max_drawdown": max_dd,
                "max_drawdown_pct": max_dd_pct,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows, columns=columns).sort_values(group_columns, kind="stable").reset_index(drop=True)


def summarize_events(events: pd.DataFrame) -> pd.DataFrame:
    """Summarize event counts and outcome statistics."""

    columns = [
        "group",
        "timeframe",
        "event_count",
        "long_count",
        "short_count",
        "squeeze_count",
        "volume_confirm_count",
        "mean_future_return_3",
        "mean_future_return_5",
        "mean_future_return_10",
        "mean_future_return_20",
        "reversal_rate_3",
        "reversal_rate_5",
        "reversal_rate_10",
    ]
    if events.empty:
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, Any]] = []
    for (group, timeframe), frame in events.groupby(["group", "timeframe"], dropna=False, sort=True):
        rows.append(
            {
                "group": group,
                "timeframe": timeframe,
                "event_count": int(len(frame.index)),
                "long_count": int((frame["direction"] == "long").sum()),
                "short_count": int((frame["direction"] == "short").sum()),
                "squeeze_count": int(frame["squeeze"].astype(bool).sum()),
                "volume_confirm_count": int(frame["volume_confirm"].astype(bool).sum()),
                "mean_future_return_3": float(pd.to_numeric(frame["future_return_3"], errors="coerce").mean()),
                "mean_future_return_5": float(pd.to_numeric(frame["future_return_5"], errors="coerce").mean()),
                "mean_future_return_10": float(pd.to_numeric(frame["future_return_10"], errors="coerce").mean()),
                "mean_future_return_20": float(pd.to_numeric(frame["future_return_20"], errors="coerce").mean()),
                "reversal_rate_3": float(frame["reversal_flag_3"].astype(bool).mean()),
                "reversal_rate_5": float(frame["reversal_flag_5"].astype(bool).mean()),
                "reversal_rate_10": float(frame["reversal_flag_10"].astype(bool).mean()),
            }
        )
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
    return (float(top_pnl / total) if total else None), top_pnl, top_count


def build_concentration_summary(trades: pd.DataFrame, config: VsvcbConfig) -> pd.DataFrame:
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


def build_reverse_test(trades: pd.DataFrame) -> pd.DataFrame:
    """Compare D-group trades to E reverse-test trades."""

    columns = [
        "timeframe",
        "split",
        "d_trade_count",
        "e_trade_count",
        "d_no_cost_pnl",
        "e_no_cost_pnl",
        "d_cost_aware_pnl",
        "e_cost_aware_pnl",
        "d_funding_adjusted_pnl",
        "e_funding_adjusted_pnl",
        "reverse_weaker",
    ]
    if trades.empty:
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, Any]] = []
    for timeframe in sorted(trades["timeframe"].dropna().unique()):
        for split in ["train", "validation", "oos", "all"]:
            frame = trades[trades["timeframe"] == timeframe]
            if split != "all":
                frame = frame[frame["split"] == split]
            d = frame[frame["group"] == "D"]
            e = frame[frame["group"] == "E"]
            d_no_cost = float(pd.to_numeric(d["no_cost_pnl"], errors="coerce").sum())
            e_no_cost = float(pd.to_numeric(e["no_cost_pnl"], errors="coerce").sum())
            d_funding = float(pd.to_numeric(d["funding_adjusted_pnl"], errors="coerce").sum())
            e_funding = float(pd.to_numeric(e["funding_adjusted_pnl"], errors="coerce").sum())
            rows.append(
                {
                    "timeframe": timeframe,
                    "split": split,
                    "d_trade_count": int(len(d.index)),
                    "e_trade_count": int(len(e.index)),
                    "d_no_cost_pnl": d_no_cost,
                    "e_no_cost_pnl": e_no_cost,
                    "d_cost_aware_pnl": float(pd.to_numeric(d["cost_aware_pnl"], errors="coerce").sum()),
                    "e_cost_aware_pnl": float(pd.to_numeric(e["cost_aware_pnl"], errors="coerce").sum()),
                    "d_funding_adjusted_pnl": d_funding,
                    "e_funding_adjusted_pnl": e_funding,
                    "reverse_weaker": bool(e_no_cost < d_no_cost and e_funding < d_funding),
                }
            )
    return pd.DataFrame(rows, columns=columns)


def build_funding_summary(
    trades: pd.DataFrame,
    funding_quality: dict[str, Any],
) -> pd.DataFrame:
    """Build funding summary rows."""

    rows: list[dict[str, Any]] = []
    for inst_id, record in (funding_quality.get("records") or {}).items():
        symbol_trades = trades[trades["symbol"].map(symbol_to_inst_id) == inst_id] if not trades.empty else pd.DataFrame()
        rows.append(
            {
                "inst_id": inst_id,
                "funding_data_complete": bool(record.get("coverage_complete")),
                "csv_path": record.get("csv_path"),
                "row_count": int(record.get("row_count") or 0),
                "first_funding_time": record.get("first_funding_time"),
                "last_funding_time": record.get("last_funding_time"),
                "trade_count": int(len(symbol_trades.index)),
                "funding_pnl": float(pd.to_numeric(symbol_trades.get("funding_pnl"), errors="coerce").sum()) if not symbol_trades.empty else 0.0,
                "warnings": ";".join(record.get("warnings") or []),
            }
        )
    return pd.DataFrame(rows)


def split_dates_payload(splits: list[TimeSplit]) -> dict[str, Any]:
    """Return JSON-safe split date payload."""

    payload: dict[str, Any] = {}
    for split in splits:
        payload[f"{split.name}_start"] = format_timestamp(split.start)
        payload[f"{split.name}_end"] = format_timestamp(split.end)
    return payload


def core_trade_frame(trades: pd.DataFrame, symbols: list[str]) -> pd.DataFrame:
    """Return core D/15m/BTC-ETH-SOL trade frame."""

    core_symbols = [symbol for symbol in symbols if symbol in CORE_SYMBOLS]
    if not core_symbols:
        core_symbols = symbols
    if trades.empty:
        return pd.DataFrame(columns=TRADE_COLUMNS)
    return trades[
        (trades["group"] == "D")
        & (trades["timeframe"] == "15m")
        & (trades["symbol"].isin(core_symbols))
    ].copy()


def pnl_sum(frame: pd.DataFrame, column: str) -> float:
    """Return numeric sum for a frame/column."""

    if frame.empty or column not in frame.columns:
        return 0.0
    return float(pd.to_numeric(frame[column], errors="coerce").sum())


def evaluate_phase1_gates(
    trades: pd.DataFrame,
    symbols: list[str],
    funding_data_complete: bool,
    config: VsvcbConfig,
) -> dict[str, Any]:
    """Evaluate VSVCB-v1 Phase 1 gates."""

    core = core_trade_frame(trades, symbols)
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
    max_dd, max_dd_pct = max_drawdown(pd.to_numeric(oos.get("funding_adjusted_pnl"), errors="coerce").fillna(0.0), config.fixed_notional) if not oos.empty else (0.0, 0.0)
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
    else:
        reverse_core = trades[
            (trades["group"].isin(["D", "E"]))
            & (trades["timeframe"] == "15m")
            & (trades["symbol"].isin([symbol for symbol in symbols if symbol in CORE_SYMBOLS] or symbols))
            & (trades["split"] == "oos")
        ]
        d_reverse = reverse_core[reverse_core["group"] == "D"]
        e_reverse = reverse_core[reverse_core["group"] == "E"]
        reverse_test_pass = bool(
            not d_reverse.empty
            and not e_reverse.empty
            and pnl_sum(d_reverse, "no_cost_pnl") > pnl_sum(e_reverse, "no_cost_pnl")
            and pnl_sum(d_reverse, "funding_adjusted_pnl") > pnl_sum(e_reverse, "funding_adjusted_pnl")
        )

    baseline_wins = 0
    baseline_rows: list[dict[str, Any]] = []
    d_oos = by_split["oos"]
    d_oos_pnl = pnl_sum(d_oos, "no_cost_pnl")
    for baseline in ["A", "B", "C"]:
        base = trades[
            (trades["group"] == baseline)
            & (trades["timeframe"] == "15m")
            & (trades["symbol"].isin([symbol for symbol in symbols if symbol in CORE_SYMBOLS] or symbols))
            & (trades["split"] == "oos")
        ]
        base_pnl = pnl_sum(base, "no_cost_pnl")
        won = bool(d_oos_pnl > base_pnl)
        baseline_wins += int(won)
        baseline_rows.append(
            {
                "baseline_group": baseline,
                "d_oos_no_cost_pnl": d_oos_pnl,
                "baseline_oos_no_cost_pnl": base_pnl,
                "d_better": won,
            }
        )
    ablation_pass = bool(baseline_wins >= 2)

    gates = {
        "train_pass": train_pass,
        "validation_pass": validation_pass,
        "oos_pass": oos_pass,
        "cost_aware_pass": cost_aware_pass,
        "funding_adjusted_pass": funding_adjusted_pass,
        "trade_count_pass": trade_count_pass,
        "concentration_pass": concentration_pass,
        "reverse_test_pass": reverse_test_pass,
        "ablation_pass": ablation_pass,
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
        "baseline_comparison": baseline_rows,
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
            "ablation_pass",
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


def build_summary_payload(
    *,
    symbols: list[str],
    timeframes: list[str],
    history_range: HistoryRange,
    config: VsvcbConfig,
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
        "hypothesis_name": "Volatility Squeeze with Volume Confirmation Breakout",
        "version": "v1.0",
        "status": "research_only",
        "phase": "phase1_event_and_fixed_hold",
        "start": history_range.start.isoformat(),
        "end_exclusive": history_range.end_exclusive.isoformat(),
        "timezone": history_range.timezone_name,
        "symbols": symbols,
        "timeframes": timeframes,
        "parameters": to_jsonable(config),
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
        "ablation_pass": bool(gates["ablation_pass"]),
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


def render_markdown_summary(
    summary: dict[str, Any],
    event_group_summary: pd.DataFrame,
    trade_group_summary: pd.DataFrame,
    reverse_test: pd.DataFrame,
    concentration: pd.DataFrame,
) -> str:
    """Render summary.md with the required answers."""

    split_dates = summary.get("split_dates") or {}
    gates = summary.get("gates") or {}
    event_counts = summary.get("event_counts") or {}
    d_trade_rows = trade_group_summary[
        (trade_group_summary["group"] == "D")
        & (trade_group_summary["timeframe"] == "15m")
    ] if not trade_group_summary.empty else pd.DataFrame()
    d_trade_table = dataframe_records(d_trade_rows, limit=12)

    def table(rows: list[dict[str, Any]], columns: list[str]) -> str:
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
                if isinstance(value, float):
                    values.append(format_number(value, 4))
                else:
                    values.append(str(value))
            lines.append("| " + " | ".join(values) + " |")
        return "\n".join(lines)

    reverse_oos_15m = reverse_test[
        (reverse_test["timeframe"] == "15m") & (reverse_test["split"] == "oos")
    ] if not reverse_test.empty else pd.DataFrame()
    concentration_oos_15m = concentration[
        (concentration["group"] == "D")
        & (concentration["timeframe"] == "15m")
        & (concentration["split"] == "oos")
    ] if not concentration.empty else pd.DataFrame()
    reverse_answer = dataframe_records(reverse_oos_15m, limit=3)
    concentration_answer = dataframe_records(concentration_oos_15m, limit=3)
    event_rows = dataframe_records(event_group_summary, limit=30)

    return (
        "# VSVCB-v1 Phase 1 Research\n\n"
        "## 1. Hypothesis\n"
        "VSVCB-v1 tests whether ordinary breakouts become more persistent when they occur after a low-volatility Bollinger Band Width squeeze and with breakout-bar volume expansion. It is research-only and uses next-bar-open fixed-hold benchmarks.\n\n"
        "## 2. Data Range\n"
        f"- start={summary.get('start')}\n"
        f"- end_exclusive={summary.get('end_exclusive')}\n"
        f"- timezone={summary.get('timezone')}\n"
        f"- funding_data_complete={str(bool(summary.get('funding_data_complete'))).lower()}\n\n"
        "## 3. Symbols And Timeframes\n"
        f"- symbols={summary.get('symbols')}\n"
        f"- timeframes={summary.get('timeframes')}\n\n"
        "## 4. Train / Validation / OOS Dates\n"
        f"- train={split_dates.get('train_start')} to {split_dates.get('train_end')}\n"
        f"- validation={split_dates.get('validation_start')} to {split_dates.get('validation_end')}\n"
        f"- oos={split_dates.get('oos_start')} to {split_dates.get('oos_end')}\n\n"
        "## 5. A/B/C/D/E Event Counts\n"
        + "\n".join(f"- {group}={event_counts.get(group, 0)}" for group in GROUPS)
        + "\n\n"
        "## 6. Event Group Summary\n"
        + table(event_rows, ["group", "timeframe", "event_count", "mean_future_return_10", "reversal_rate_10"])
        + "\n\n"
        "## 7. D Group 15m Trade Result\n"
        + table(d_trade_table, ["group", "timeframe", "split", "trade_count", "no_cost_pnl", "cost_aware_pnl", "funding_adjusted_pnl", "max_drawdown_pct"])
        + "\n\n"
        "## 8. Required Answers\n"
        f"1. D group 是否优于 A/B/C 中至少两个？{str(bool(summary.get('ablation_pass'))).lower()}\n"
        f"2. 反向测试是否明显更差？{str(bool(summary.get('reverse_test_pass'))).lower()} {reverse_answer}\n"
        f"3. no-cost 是否通过？train={str(bool(summary.get('train_pass'))).lower()}, validation={str(bool(summary.get('validation_pass'))).lower()}, oos={str(bool(summary.get('oos_pass'))).lower()}\n"
        f"4. cost-aware 是否通过？{str(bool(summary.get('cost_aware_pass'))).lower()}\n"
        f"5. funding-adjusted 是否通过？{str(bool(summary.get('funding_adjusted_pass'))).lower()}\n"
        f"6. 收益是否集中在单一 symbol？concentration_pass={str(bool(summary.get('concentration_pass'))).lower()} {concentration_answer}\n"
        f"7. 收益是否集中在 top trades？top_5pct_trade_pnl_contribution={format_number(gates.get('top_5pct_trade_pnl_contribution'), 4)}\n"
        f"8. 是否允许进入 Phase 2？{str(bool(summary.get('continue_to_phase2'))).lower()}\n"
        "9. 是否允许修改正式策略？false\n"
        "10. 是否允许 demo/live？false\n\n"
        "## 9. Final Decision\n"
        f"- final_decision={summary.get('final_decision')}\n"
        f"- continue_to_phase2={str(bool(summary.get('continue_to_phase2'))).lower()}\n"
        "- strategy_development_allowed=false\n"
        "- demo_live_allowed=false\n"
    )


def render_postmortem(summary: dict[str, Any]) -> str:
    """Render postmortem draft."""

    gates = summary.get("gates") or {}
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
            "ablation_pass",
        ]
        if not bool(summary.get(key))
    ]
    decision = summary.get("final_decision")
    next_step = (
        "Phase 2 parameter plateau and randomization research only."
        if summary.get("continue_to_phase2")
        else "Postmortem. Do not tune Phase 1 parameters from OOS and do not develop a strategy."
    )
    return (
        "# VSVCB-v1 Postmortem Draft\n\n"
        f"- final_decision={decision}\n"
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
    concentration: pd.DataFrame,
    reverse_test: pd.DataFrame,
    funding_summary: pd.DataFrame,
    data_quality: dict[str, Any],
) -> None:
    """Write all Phase 1 artifacts."""

    output_dir.mkdir(parents=True, exist_ok=True)
    write_dataframe(output_dir / "events.csv", events, EVENT_COLUMNS)
    write_dataframe(output_dir / "trades.csv", trades, TRADE_COLUMNS)
    write_dataframe(output_dir / "event_group_summary.csv", event_group_summary)
    write_dataframe(output_dir / "trade_group_summary.csv", trade_group_summary)
    write_dataframe(output_dir / "by_symbol.csv", by_symbol)
    write_dataframe(output_dir / "by_timeframe.csv", by_timeframe)
    write_dataframe(output_dir / "by_split.csv", by_split)
    write_dataframe(output_dir / "concentration.csv", concentration)
    write_dataframe(output_dir / "reverse_test.csv", reverse_test)
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
        render_markdown_summary(summary, event_group_summary, trade_group_summary, reverse_test, concentration),
        encoding="utf-8",
    )
    (output_dir / "postmortem_draft.md").write_text(render_postmortem(summary), encoding="utf-8")
    for filename in PLACEHOLDER_FILES:
        pd.DataFrame([{"status": "not_run", "reason": "Phase 1 does not execute this analysis"}]).to_csv(
            output_dir / filename,
            index=False,
        )


def build_data_quality_payload(
    *,
    symbols: list[str],
    timeframes: list[str],
    history_range: HistoryRange,
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
        "start": history_range.start.isoformat(),
        "end_exclusive": history_range.end_exclusive.isoformat(),
        "timezone": history_range.timezone_name,
        "data_check_strict": strict,
        "expected_1m_count": expected_bar_count(history_range),
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


def run_research(
    *,
    symbols: list[str],
    timeframes: list[str],
    history_range: HistoryRange,
    output_dir: Path,
    funding_dir: Path,
    database_path: Path,
    config: VsvcbConfig,
    data_check_strict: bool,
    logger: logging.Logger | None = None,
    bars_by_symbol: dict[str, pd.DataFrame] | None = None,
) -> dict[str, Any]:
    """Run VSVCB-v1 Phase 1 research."""

    validate_config(config)
    warnings: list[str] = []
    funding_histories, funding_quality = load_funding_histories(
        funding_dir,
        symbols,
        history_range,
        history_range.start.date().isoformat(),
        history_range.end_display.date().isoformat(),
    )
    splits = build_time_splits(history_range)
    split_payload = split_dates_payload(splits)

    bars_by_key: dict[tuple[str, str], pd.DataFrame] = {}
    symbol_quality: dict[str, Any] = {}
    event_frames: list[pd.DataFrame] = []

    for symbol in symbols:
        if logger:
            log_event(logger, logging.INFO, "load_symbol", "Loading symbol history", symbol=symbol)
        if bars_by_symbol is not None and symbol in bars_by_symbol:
            bars_1m = normalize_1m_bars(bars_by_symbol[symbol], history_range.timezone_name)
            bars_1m = bars_1m[
                (bars_1m["datetime"] >= pd.Timestamp(history_range.start))
                & (bars_1m["datetime"] < pd.Timestamp(history_range.end_exclusive))
            ].copy()
        else:
            bars_1m = load_1m_bars_from_sqlite(symbol, history_range, database_path)
        one_minute_quality = analyze_1m_quality(symbol, bars_1m, history_range)
        symbol_quality[symbol] = {"1m": one_minute_quality}
        if data_check_strict and not one_minute_quality["complete"]:
            raise VsvcbResearchError(f"strict data check failed for {symbol}: {one_minute_quality}")
        if bars_1m.empty:
            if data_check_strict:
                raise VsvcbResearchError(f"required symbol has no data: {symbol}")
            warnings.append(f"{symbol}: no 1m bars loaded")
            continue
        for timeframe in timeframes:
            resampled = resample_ohlcv_closed(bars_1m, timeframe, history_range)
            timeframe_quality = analyze_timeframe_quality(symbol, timeframe, resampled, one_minute_quality)
            symbol_quality[symbol][timeframe] = timeframe_quality
            if data_check_strict and not timeframe_quality["complete"]:
                raise VsvcbResearchError(f"strict resample data check failed for {symbol} {timeframe}: {timeframe_quality}")
            bars_by_key[(symbol, timeframe)] = add_vsvcb_indicators(resampled, config)
            events, event_warnings = generate_events_for_frame(resampled, symbol, timeframe, config, funding_histories)
            event_frames.append(events)
            warnings.extend(event_warnings)

    if not event_frames:
        events = pd.DataFrame(columns=EVENT_COLUMNS)
    else:
        events = pd.concat(event_frames, ignore_index=True)
        events = assign_event_ids(events)
        if not events.empty:
            events["split"] = events["timestamp"].map(lambda value: assign_split_for_time(value, splits))
            events = events.drop(columns=["split"])

    trades, trade_warnings = simulate_fixed_hold_trades(events, bars_by_key, config, funding_histories, splits)
    warnings.extend(trade_warnings)

    data_quality = build_data_quality_payload(
        symbols=symbols,
        timeframes=timeframes,
        history_range=history_range,
        funding_quality=funding_quality,
        symbol_quality=symbol_quality,
        strict=data_check_strict,
    )
    event_group_summary = summarize_events(events)
    trade_group_summary = summarize_trades(trades, ["group", "timeframe", "split"], config)
    by_symbol = summarize_trades(trades, ["group", "timeframe", "split", "symbol"], config)
    by_timeframe = summarize_trades(trades, ["group", "timeframe", "split"], config)
    by_split = summarize_trades(trades, ["group", "split"], config)
    concentration = build_concentration_summary(trades, config)
    reverse_test = build_reverse_test(trades)
    funding_summary = build_funding_summary(trades, funding_quality)
    gates = evaluate_phase1_gates(trades, symbols, bool(funding_quality.get("funding_data_complete")), config)
    summary = build_summary_payload(
        symbols=symbols,
        timeframes=timeframes,
        history_range=history_range,
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
        concentration=concentration,
        reverse_test=reverse_test,
        funding_summary=funding_summary,
        data_quality=data_quality,
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""

    args = parse_args(argv)
    ensure_headless_runtime()
    logger = setup_logging("research_vsvcb_v1", verbose=bool(args.verbose))
    symbols = parse_csv_list(args.symbols)
    if not symbols:
        raise VsvcbResearchError("--symbols must not be empty")
    timeframes = parse_timeframes(args.timeframes)
    history_range = parse_history_range(
        start_arg=args.start,
        end_arg=args.end,
        interval_delta=timedelta(minutes=1),
        timezone_name=args.timezone,
    )
    config = build_config(args)
    summary = run_research(
        symbols=symbols,
        timeframes=timeframes,
        history_range=history_range,
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
        "VSVCB-v1 Phase 1 research complete",
        final_decision=summary.get("final_decision"),
        continue_to_phase2=summary.get("continue_to_phase2"),
        output_dir=str(resolve_path(args.output_dir)),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
