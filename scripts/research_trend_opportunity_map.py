#!/usr/bin/env python3
"""Build an ex-post Trend Opportunity Map for research diagnostics only."""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from common_runtime import PROJECT_ROOT, ensure_headless_runtime, log_event, print_json_block, setup_logging, to_jsonable
from history_time_utils import DEFAULT_TIMEZONE, HistoryRange, expected_bar_count, parse_history_range


DEFAULT_SYMBOLS = [
    "BTCUSDT_SWAP_OKX.GLOBAL",
    "ETHUSDT_SWAP_OKX.GLOBAL",
    "SOLUSDT_SWAP_OKX.GLOBAL",
    "LINKUSDT_SWAP_OKX.GLOBAL",
    "DOGEUSDT_SWAP_OKX.GLOBAL",
]
DEFAULT_START = "2023-01-01"
DEFAULT_END = "2026-03-31"
DEFAULT_TIMEFRAMES = ["4h", "1d"]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "reports" / "research" / "trend_opportunity_map"
DEFAULT_FUNDING_DIR = PROJECT_ROOT / "data" / "funding" / "okx"
DEFAULT_DATABASE_PATH = PROJECT_ROOT / ".vntrader" / "database.db"

TIMEFRAME_MINUTES = {"4h": 240, "1d": 1440}
EFFICIENCY_WINDOWS = (20, 55, 100)
EFFICIENCY_THRESHOLDS = (0.45, 0.60, 0.75)
ATR_TARGETS = (
    (2.0, 1.0, "2atr"),
    (3.0, 1.5, "3atr"),
    (5.0, 2.0, "5atr"),
)
ATR_LOOKAHEAD_BARS = 100
MIN_RUN_LENGTH = 4
PRE_TREND_WINDOWS = (20, 55)
ROLLING_PERCENTILE_WINDOW = 240
REALIZED_VOL_WINDOW = 20
MARKET_CORRELATION_WINDOW = 55

LEGACY_TRADE_FILES = (
    ("trend_v3_extended/train_ext", PROJECT_ROOT / "reports" / "research" / "trend_following_v3_extended" / "train_ext" / "trend_v3_trades.csv"),
    (
        "trend_v3_extended/validation_ext",
        PROJECT_ROOT / "reports" / "research" / "trend_following_v3_extended" / "validation_ext" / "trend_v3_trades.csv",
    ),
    ("trend_v3_extended/oos_ext", PROJECT_ROOT / "reports" / "research" / "trend_following_v3_extended" / "oos_ext" / "trend_v3_trades.csv"),
    ("vsvcb_v1", PROJECT_ROOT / "reports" / "research" / "vsvcb_v1" / "trades.csv"),
    ("csrb_v1", PROJECT_ROOT / "reports" / "research" / "csrb_v1" / "trades.csv"),
)

REQUIRED_OUTPUT_FILES = [
    "trend_opportunity_summary.json",
    "trend_opportunity_report.md",
    "trend_segments.csv",
    "trend_opportunity_by_symbol.csv",
    "trend_opportunity_by_timeframe.csv",
    "trend_opportunity_by_month.csv",
    "trend_opportunity_by_quarter.csv",
    "legacy_strategy_trend_coverage.csv",
    "pre_trend_feature_comparison.csv",
    "data_quality.json",
]

SEGMENT_COLUMNS = [
    "trend_segment_id",
    "symbol",
    "timeframe",
    "direction",
    "start_time",
    "end_time",
    "start_idx",
    "end_idx",
    "duration_bars",
    "duration_days",
    "start_price",
    "end_price",
    "trend_return",
    "mfe",
    "mae",
    "labels",
    "atr_labels",
    "efficiency_labels",
    "has_atr_move",
    "has_efficiency",
    "has_run_length",
    "run_length",
    "max_run_return",
    "max_run_drawdown",
    "run_duration_bars",
    "month",
    "quarter",
    "abs_trend_return",
    "is_major_trend",
]

AGG_COLUMNS = [
    "trend_segment_count",
    "uptrend_count",
    "downtrend_count",
    "avg_duration_bars",
    "median_duration_bars",
    "avg_trend_return",
    "median_trend_return",
    "avg_mfe",
    "avg_mae",
    "trend_opportunity_days_ratio",
    "strongest_month",
    "weakest_month",
]

LEGACY_COVERAGE_COLUMNS = [
    "strategy_source",
    "policy_or_group",
    "symbol",
    "timeframe",
    "trade_id",
    "entry_time",
    "exit_time",
    "trend_segment_id",
    "entered_trend_segment",
    "entry_phase",
    "captured_fraction_of_segment",
    "missed_major_trend",
    "pnl",
    "is_synthetic_missed_segment",
]

PRE_TREND_FEATURES = [
    "pre_trend_atr_percentile",
    "pre_trend_volatility",
    "pre_trend_volume_ratio",
    "pre_trend_funding_rate",
    "pre_trend_funding_percentile",
    "pre_trend_market_breadth",
    "pre_trend_correlation",
    "pre_trend_drawdown_state",
    "pre_trend_range_width",
    "pre_trend_compression_score",
]


class TrendOpportunityMapError(Exception):
    """Raised when the trend opportunity map cannot be built."""


@dataclass(frozen=True, slots=True)
class OpportunityOutputs:
    """Generated outputs for tests and CLI reporting."""

    output_dir: Path
    summary: dict[str, Any]
    data_quality: dict[str, Any]
    trend_segments: pd.DataFrame
    legacy_coverage: pd.DataFrame
    pre_trend_feature_comparison: pd.DataFrame


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description="Build a research-only ex-post Trend Opportunity Map.")
    parser.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS))
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--end", default=DEFAULT_END)
    parser.add_argument("--timezone", default=DEFAULT_TIMEZONE)
    parser.add_argument("--timeframes", default=",".join(DEFAULT_TIMEFRAMES))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--funding-dir", default=str(DEFAULT_FUNDING_DIR))
    parser.add_argument("--database-path", default=str(DEFAULT_DATABASE_PATH))
    parser.add_argument("--data-check-strict", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def resolve_path(path_arg: str | Path) -> Path:
    """Resolve a path relative to the project root."""

    path = Path(path_arg)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def parse_csv_list(value: str | Iterable[str]) -> list[str]:
    """Parse comma or whitespace separated values while preserving order."""

    tokens = re.split(r"[\s,]+", value) if isinstance(value, str) else [str(item) for item in value]
    parsed: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        item = token.strip()
        if item and item not in seen:
            parsed.append(item)
            seen.add(item)
    return parsed


def parse_symbols(value: str | Iterable[str]) -> list[str]:
    """Parse requested vt_symbols."""

    symbols = parse_csv_list(value)
    if not symbols:
        raise TrendOpportunityMapError("--symbols must not be empty")
    return symbols


def parse_timeframes(value: str | Iterable[str]) -> list[str]:
    """Parse and validate requested timeframes."""

    timeframes = parse_csv_list(value)
    unsupported = [item for item in timeframes if item not in TIMEFRAME_MINUTES]
    if unsupported:
        raise TrendOpportunityMapError(f"unsupported timeframes: {unsupported}")
    if not timeframes:
        raise TrendOpportunityMapError("--timeframes must not be empty")
    return timeframes


def resolve_history_range(start: str, end: str, timezone_name: str) -> HistoryRange:
    """Resolve the research date range."""

    try:
        return parse_history_range(start, end, pd.Timedelta(minutes=1).to_pytimedelta(), timezone_name)
    except ValueError as exc:
        raise TrendOpportunityMapError(str(exc)) from exc


def split_vt_symbol(vt_symbol: str) -> tuple[str, str]:
    """Split a vt_symbol into database symbol and exchange."""

    symbol, separator, exchange = str(vt_symbol).partition(".")
    if not separator or not symbol or not exchange:
        raise TrendOpportunityMapError(f"invalid vt_symbol: {vt_symbol}")
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


def safe_symbol(value: str) -> str:
    """Return a stable identifier-safe symbol fragment."""

    return re.sub(r"[^A-Za-z0-9]+", "_", str(value)).strip("_")


def finite_float(value: Any, default: float | None = None) -> float | None:
    """Return a finite float or default."""

    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if not np.isfinite(number):
        return default
    return number


def safe_mean(values: Any) -> float | None:
    """Return the numeric mean or None."""

    series = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    if series.empty:
        return None
    return float(series.mean())


def safe_median(values: Any) -> float | None:
    """Return the numeric median or None."""

    series = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    if series.empty:
        return None
    return float(series.median())


def format_timestamp(value: Any) -> str | None:
    """Format timestamp-like values as ISO strings."""

    if value is None or pd.isna(value):
        return None
    return pd.Timestamp(value).isoformat()


def dataframe_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Convert a frame to JSON-safe records."""

    if df.empty:
        return []
    safe_df = df.replace({np.nan: None})
    return json.loads(safe_df.to_json(orient="records", force_ascii=False, date_format="iso"))


def write_json(path: Path, payload: Any) -> None:
    """Write a JSON file."""

    path.write_text(json.dumps(to_jsonable(payload), ensure_ascii=False, indent=2), encoding="utf-8")


def normalize_1m_bars(frame: pd.DataFrame, timezone_name: str) -> pd.DataFrame:
    """Normalize an injected or sqlite-loaded 1m OHLCV frame."""

    columns = ["datetime", "open", "high", "low", "close", "volume"]
    if frame.empty:
        return pd.DataFrame(columns=columns)
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise TrendOpportunityMapError(f"1m bars missing columns: {missing}")

    normalized = frame.loc[:, columns].copy()
    timestamps = pd.to_datetime(normalized["datetime"], errors="coerce")
    if timestamps.isna().any():
        raise TrendOpportunityMapError("1m bars contain unparsable datetime values")
    if timestamps.dt.tz is None:
        timestamps = timestamps.dt.tz_localize(timezone_name)
    else:
        timestamps = timestamps.dt.tz_convert(timezone_name)
    normalized["datetime"] = timestamps
    for column in ["open", "high", "low", "close", "volume"]:
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce")
    normalized = normalized.dropna(subset=columns)
    normalized = normalized.sort_values("datetime", kind="stable").drop_duplicates("datetime", keep="last")
    return normalized.reset_index(drop=True)


def load_1m_bars_from_sqlite(vt_symbol: str, history_range: HistoryRange, database_path: Path) -> pd.DataFrame:
    """Load local vn.py sqlite 1m bars for one symbol."""

    if not database_path.exists():
        raise TrendOpportunityMapError(f"database not found: {database_path}")

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
        frame = pd.read_sql_query(sql, connection, params=(symbol, exchange, "1m", query_start, query_end))
    return normalize_1m_bars(frame, history_range.timezone_name)


def filter_time_range(df: pd.DataFrame, history_range: HistoryRange) -> pd.DataFrame:
    """Filter bars to the requested research period."""

    if df.empty:
        return df.copy()
    dt = pd.to_datetime(df["datetime"])
    mask = (dt >= pd.Timestamp(history_range.start)) & (dt < pd.Timestamp(history_range.end_exclusive))
    return df.loc[mask].copy().reset_index(drop=True)


def coverage_missing_ranges_from_actual(
    actual_times: list[pd.Timestamp],
    history_range: HistoryRange,
    *,
    limit: int = 20,
) -> tuple[list[dict[str, Any]], int]:
    """Build compact missing ranges without materializing every expected minute."""

    ranges: list[dict[str, Any]] = []
    gap_count = 0
    interval = pd.Timedelta(minutes=1)
    cursor = pd.Timestamp(history_range.start)
    end_exclusive = pd.Timestamp(history_range.end_exclusive)
    for timestamp in actual_times:
        current = pd.Timestamp(timestamp)
        if current < cursor:
            continue
        if current > cursor:
            missing_end = current - interval
            count = int((current - cursor) / interval)
            gap_count += 1
            if len(ranges) < limit:
                ranges.append({"start": cursor.isoformat(), "end": missing_end.isoformat(), "missing_count": count})
        cursor = current + interval
    if cursor < end_exclusive:
        count = int((end_exclusive - cursor) / interval)
        gap_count += 1
        if len(ranges) < limit:
            ranges.append({"start": cursor.isoformat(), "end": (end_exclusive - interval).isoformat(), "missing_count": count})
    return ranges, gap_count


def analyze_1m_quality(vt_symbol: str, bars_1m: pd.DataFrame, history_range: HistoryRange) -> dict[str, Any]:
    """Build strict 1m coverage quality summary for one symbol."""

    target = filter_time_range(bars_1m, history_range)
    actual_times = [
        pd.Timestamp(value)
        for value in pd.to_datetime(target["datetime"]).dropna().drop_duplicates().sort_values(kind="stable")
    ] if not target.empty else []
    expected = int(expected_bar_count(history_range))
    total = int(len(actual_times))
    missing = max(0, expected - total)
    missing_ranges, gap_count = coverage_missing_ranges_from_actual(actual_times, history_range)
    if missing and not missing_ranges:
        missing_ranges = [{"start": history_range.start.isoformat(), "end": history_range.end_display.isoformat(), "missing_count": missing}]
        gap_count = 1
    return {
        "symbol": vt_symbol,
        "timeframe": "1m",
        "expected_count": expected,
        "row_count": int(len(target.index)),
        "unique_count": total,
        "missing_count": int(missing),
        "gap_count": int(gap_count),
        "first_datetime": actual_times[0].isoformat() if actual_times else None,
        "last_datetime": actual_times[-1].isoformat() if actual_times else None,
        "complete": bool(expected > 0 and missing == 0 and total == expected),
        "missing_ranges_sample": missing_ranges,
    }


def resample_ohlcv_closed(bars_1m: pd.DataFrame, timeframe: str, history_range: HistoryRange | None = None) -> pd.DataFrame:
    """Resample 1m OHLCV into completed closed bars timestamped at the final minute."""

    if timeframe not in TIMEFRAME_MINUTES:
        raise TrendOpportunityMapError(f"unsupported timeframe: {timeframe}")
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
    result = result.drop(columns=["minute_count"]).dropna(subset=["open", "high", "low", "close"]).reset_index(drop=True)
    return result.loc[:, columns]


def analyze_timeframe_quality(timeframe: str, bars: pd.DataFrame, one_minute_quality: dict[str, Any]) -> dict[str, Any]:
    """Build quality metadata for one resampled timeframe."""

    minutes = TIMEFRAME_MINUTES[timeframe]
    expected_count = int(one_minute_quality["expected_count"] // minutes)
    return {
        "timeframe": timeframe,
        "minutes": minutes,
        "expected_closed_bar_count": expected_count,
        "row_count": int(len(bars.index)),
        "first_datetime": format_timestamp(bars["datetime"].min()) if not bars.empty else None,
        "last_datetime": format_timestamp(bars["datetime"].max()) if not bars.empty else None,
        "complete": bool(one_minute_quality["complete"] and len(bars.index) == expected_count),
    }


def true_range(df: pd.DataFrame) -> pd.Series:
    """Compute true range."""

    high = pd.to_numeric(df["high"], errors="coerce")
    low = pd.to_numeric(df["low"], errors="coerce")
    close = pd.to_numeric(df["close"], errors="coerce")
    previous_close = close.shift(1)
    ranges = pd.concat([high - low, (high - previous_close).abs(), (low - previous_close).abs()], axis=1)
    return ranges.max(axis=1)


def rolling_percentile(series: pd.Series, window: int = ROLLING_PERCENTILE_WINDOW) -> pd.Series:
    """Return current value percentile versus prior rolling observations only."""

    min_periods = min(max(5, window // 4), window)
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    result = np.full(values.size, np.nan, dtype=float)
    for index, current in enumerate(values):
        if not np.isfinite(current):
            continue
        start = max(0, index - window)
        history = values[start:index]
        clean = history[np.isfinite(history)]
        if clean.size < min_periods:
            continue
        result[index] = float(np.mean(clean <= current))
    return pd.Series(result, index=series.index, dtype=float)


def forward_trend_efficiency(close: pd.Series, window: int) -> pd.Series:
    """Compute ex-post trend efficiency over the next window bars."""

    values = pd.to_numeric(close, errors="coerce").to_numpy(dtype=float)
    result = np.full(values.size, np.nan, dtype=float)
    for index in range(0, max(0, len(values) - window)):
        start = values[index]
        end = values[index + window]
        path_values = values[index : index + window + 1]
        if not np.isfinite(start) or not np.isfinite(end) or np.any(~np.isfinite(path_values)):
            continue
        path = float(np.abs(np.diff(path_values)).sum())
        if path > 0:
            result[index] = float(abs(end - start) / path)
    return pd.Series(result, index=close.index, dtype=float)


def compute_context_indicators(frame: pd.DataFrame) -> pd.DataFrame:
    """Compute label and pre-trend diagnostic indicators on closed bars."""

    out = frame.copy().sort_values("datetime", kind="stable").reset_index(drop=True)
    if out.empty:
        return out
    for column in ["open", "high", "low", "close", "volume"]:
        out[column] = pd.to_numeric(out[column], errors="coerce")

    close = out["close"]
    high = out["high"]
    low = out["low"]
    volume = out["volume"]
    returns = close.pct_change()

    out["atr14"] = true_range(out).rolling(14, min_periods=1).mean()
    out["atr_pct"] = out["atr14"] / close.replace(0.0, np.nan)
    out["atr_percentile"] = rolling_percentile(out["atr_pct"])
    out["realized_volatility"] = returns.rolling(REALIZED_VOL_WINDOW, min_periods=5).std(ddof=0)
    out["volume_ma_prev"] = volume.shift(1).rolling(20, min_periods=5).mean()
    out["volume_ratio"] = volume / out["volume_ma_prev"].replace(0.0, np.nan)
    out["range_width"] = (high.rolling(20, min_periods=5).max() - low.rolling(20, min_periods=5).min()) / close.replace(0.0, np.nan)
    out["range_width_percentile"] = rolling_percentile(out["range_width"])
    out["compression_score"] = 1.0 - pd.concat([out["atr_percentile"], out["range_width_percentile"]], axis=1).mean(axis=1)
    out["compression_score"] = out["compression_score"].clip(0.0, 1.0)
    out["rolling_high_100"] = close.rolling(100, min_periods=5).max()
    out["drawdown_state"] = close / out["rolling_high_100"].replace(0.0, np.nan) - 1.0
    out["sma50"] = close.rolling(50, min_periods=5).mean()
    out["close_above_sma50"] = close > out["sma50"]
    out["return_20"] = close / close.shift(20) - 1.0
    out["bar_return"] = returns

    for window in EFFICIENCY_WINDOWS:
        out[f"trend_efficiency_{window}"] = forward_trend_efficiency(close, window)
    out["trend_efficiency_max"] = out[[f"trend_efficiency_{window}" for window in EFFICIENCY_WINDOWS]].max(axis=1)
    return out


def add_market_context(frames_by_key: dict[tuple[str, str], pd.DataFrame], symbols: list[str], timeframes: list[str]) -> dict[tuple[str, str], pd.DataFrame]:
    """Add market breadth and cross-symbol correlation to each symbol/timeframe frame."""

    result = {key: frame.copy() for key, frame in frames_by_key.items()}
    for timeframe in timeframes:
        combined_parts: list[pd.DataFrame] = []
        close_parts: list[pd.Series] = []
        return_parts: list[pd.Series] = []
        for symbol in symbols:
            frame = result.get((symbol, timeframe))
            if frame is None or frame.empty:
                continue
            part = frame[["datetime", "close_above_sma50", "return_20"]].copy()
            part["symbol"] = symbol
            combined_parts.append(part)
            indexed = frame.set_index("datetime")
            close_parts.append(indexed["close"].rename(symbol))
            return_parts.append(indexed["bar_return"].rename(symbol))
        if not combined_parts:
            continue

        combined = pd.concat(combined_parts, ignore_index=True)
        breadth = combined.groupby("datetime", dropna=False)["close_above_sma50"].mean()
        positive_return_breadth = (combined["return_20"] > 0).groupby(combined["datetime"], dropna=False).mean()
        breadth = pd.concat([breadth, positive_return_breadth], axis=1).mean(axis=1)

        returns = pd.concat(return_parts, axis=1).sort_index() if return_parts else pd.DataFrame()
        correlations: dict[str, pd.Series] = {}
        for symbol in returns.columns:
            others = returns.drop(columns=[symbol])
            if others.empty:
                correlations[symbol] = pd.Series(np.nan, index=returns.index, dtype=float)
            else:
                correlations[symbol] = returns[symbol].rolling(MARKET_CORRELATION_WINDOW, min_periods=10).corr(others.mean(axis=1))

        for symbol in symbols:
            key = (symbol, timeframe)
            frame = result.get(key)
            if frame is None or frame.empty:
                continue
            frame = frame.copy()
            frame["market_breadth"] = frame["datetime"].map(breadth)
            frame["market_correlation"] = frame["datetime"].map(correlations.get(symbol, pd.Series(dtype=float)))
            result[key] = frame
    return result


def atr_move_candidates(frame: pd.DataFrame, symbol: str, timeframe: str, max_lookahead: int = ATR_LOOKAHEAD_BARS) -> list[dict[str, Any]]:
    """Generate ATR move label candidates from future path diagnostics."""

    candidates: list[dict[str, Any]] = []
    if frame.empty:
        return candidates
    close = pd.to_numeric(frame["close"], errors="coerce").to_numpy(dtype=float)
    high = pd.to_numeric(frame["high"], errors="coerce").to_numpy(dtype=float)
    low = pd.to_numeric(frame["low"], errors="coerce").to_numpy(dtype=float)
    atr = pd.to_numeric(frame["atr14"], errors="coerce").to_numpy(dtype=float)
    count = len(frame.index)
    for index in range(count - 1):
        entry = close[index]
        entry_atr = atr[index]
        if not np.isfinite(entry) or not np.isfinite(entry_atr) or entry_atr <= 0:
            continue
        end_limit = min(count - 1, index + max_lookahead)
        for target_mult, adverse_mult, suffix in ATR_TARGETS:
            target = target_mult * entry_atr
            adverse_limit = adverse_mult * entry_atr
            min_low = math.inf
            max_high = -math.inf
            for future_index in range(index + 1, end_limit + 1):
                min_low = min(min_low, low[future_index])
                max_high = max(max_high, high[future_index])
                if np.isfinite(max_high) and max_high - entry >= target:
                    if np.isfinite(min_low) and entry - min_low <= adverse_limit:
                        candidates.append(
                            {
                                "symbol": symbol,
                                "timeframe": timeframe,
                                "direction": "up",
                                "start_idx": index,
                                "end_idx": future_index,
                                "labels": {f"uptrend_{suffix}"},
                                "source": "atr_move",
                            }
                        )
                    break
                if np.isfinite(min_low) and entry - min_low > adverse_limit:
                    break
            min_low = math.inf
            max_high = -math.inf
            for future_index in range(index + 1, end_limit + 1):
                min_low = min(min_low, low[future_index])
                max_high = max(max_high, high[future_index])
                if np.isfinite(min_low) and entry - min_low >= target:
                    if np.isfinite(max_high) and max_high - entry <= adverse_limit:
                        candidates.append(
                            {
                                "symbol": symbol,
                                "timeframe": timeframe,
                                "direction": "down",
                                "start_idx": index,
                                "end_idx": future_index,
                                "labels": {f"downtrend_{suffix}"},
                                "source": "atr_move",
                            }
                        )
                    break
                if np.isfinite(max_high) and max_high - entry > adverse_limit:
                    break
    return candidates


def trend_efficiency_candidates(frame: pd.DataFrame, symbol: str, timeframe: str) -> list[dict[str, Any]]:
    """Generate forward trend-efficiency label candidates."""

    candidates: list[dict[str, Any]] = []
    if frame.empty:
        return candidates
    close = pd.to_numeric(frame["close"], errors="coerce").to_numpy(dtype=float)
    count = len(frame.index)
    for window in EFFICIENCY_WINDOWS:
        column = f"trend_efficiency_{window}"
        if column not in frame.columns:
            continue
        efficiency = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
        for index, value in enumerate(efficiency):
            end_index = index + window
            if end_index >= count or not np.isfinite(value):
                continue
            movement = close[end_index] - close[index]
            if not np.isfinite(movement) or movement == 0:
                continue
            labels = {f"trend_efficiency_{window}_ge_{threshold:.2f}" for threshold in EFFICIENCY_THRESHOLDS if value >= threshold}
            if not labels:
                continue
            candidates.append(
                {
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "direction": "up" if movement > 0 else "down",
                    "start_idx": index,
                    "end_idx": end_index,
                    "labels": labels,
                    "source": "trend_efficiency",
                }
            )
    return candidates


def max_same_direction_close_run(close: pd.Series, direction: str) -> int:
    """Return the maximum consecutive close direction run length."""

    sign_target = 1.0 if direction == "up" else -1.0
    signs = np.sign(pd.to_numeric(close, errors="coerce").diff().fillna(0.0).to_numpy(dtype=float))
    best = 0
    current = 0
    for sign in signs:
        if sign == sign_target:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return int(best)


def run_length_candidates(frame: pd.DataFrame, symbol: str, timeframe: str) -> list[dict[str, Any]]:
    """Generate consecutive directional run candidates."""

    candidates: list[dict[str, Any]] = []
    if frame.empty:
        return candidates
    close = pd.to_numeric(frame["close"], errors="coerce").to_numpy(dtype=float)
    high = pd.to_numeric(frame["high"], errors="coerce").to_numpy(dtype=float)
    low = pd.to_numeric(frame["low"], errors="coerce").to_numpy(dtype=float)
    close_signs = np.sign(np.diff(close, prepend=np.nan))
    structure_signs = np.zeros(len(close), dtype=float)
    for index in range(1, len(close)):
        if high[index] > high[index - 1] and low[index] > low[index - 1]:
            structure_signs[index] = 1.0
        elif high[index] < high[index - 1] and low[index] < low[index - 1]:
            structure_signs[index] = -1.0
    direction_signs = np.where(close_signs != 0, close_signs, structure_signs)

    run_sign = 0.0
    run_start = 0
    run_length = 0
    for index in range(1, len(direction_signs) + 1):
        sign = direction_signs[index] if index < len(direction_signs) else 0.0
        if sign != 0 and sign == run_sign:
            run_length += 1
            continue
        if run_sign != 0 and run_length >= MIN_RUN_LENGTH:
            start_idx = max(0, run_start - 1)
            end_idx = index - 1
            candidates.append(
                {
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "direction": "up" if run_sign > 0 else "down",
                    "start_idx": start_idx,
                    "end_idx": end_idx,
                    "labels": {"run_length"},
                    "source": "run_length",
                }
            )
        run_sign = sign
        run_start = index
        run_length = 1 if sign != 0 else 0
    return candidates


def directional_segment_metrics(frame: pd.DataFrame, start_idx: int, end_idx: int, direction: str) -> dict[str, float | int | None]:
    """Compute return, MFE, MAE, and run metrics for one segment."""

    if frame.empty or start_idx < 0 or end_idx >= len(frame.index) or end_idx < start_idx:
        return {
            "trend_return": None,
            "mfe": None,
            "mae": None,
            "run_length": 0,
            "max_run_return": None,
            "max_run_drawdown": None,
            "run_duration_bars": 0,
        }
    subset = frame.iloc[start_idx : end_idx + 1].copy()
    entry = finite_float(subset.iloc[0]["close"])
    exit_price = finite_float(subset.iloc[-1]["close"])
    if entry is None or exit_price is None or entry == 0:
        trend_return = None
        mfe = None
        mae = None
        path = pd.Series(dtype=float)
    elif direction == "up":
        trend_return = exit_price / entry - 1.0
        mfe = float(pd.to_numeric(subset["high"], errors="coerce").max() / entry - 1.0)
        mae = float(pd.to_numeric(subset["low"], errors="coerce").min() / entry - 1.0)
        path = pd.to_numeric(subset["close"], errors="coerce") / entry - 1.0
    else:
        trend_return = entry / exit_price - 1.0 if exit_price != 0 else None
        min_low = pd.to_numeric(subset["low"], errors="coerce").min()
        max_high = pd.to_numeric(subset["high"], errors="coerce").max()
        mfe = float(entry / min_low - 1.0) if min_low and np.isfinite(min_low) else None
        mae = float(entry / max_high - 1.0) if max_high and np.isfinite(max_high) else None
        path = entry / pd.to_numeric(subset["close"], errors="coerce") - 1.0
    max_run_return = finite_float(path.max(), default=None) if not path.empty else None
    if path.empty or path.dropna().empty:
        max_run_drawdown = None
    else:
        max_run_drawdown = float((path - path.cummax()).min())
    return {
        "trend_return": trend_return,
        "mfe": mfe,
        "mae": mae,
        "run_length": max_same_direction_close_run(subset["close"], direction),
        "max_run_return": max_run_return,
        "max_run_drawdown": max_run_drawdown,
        "run_duration_bars": int(end_idx - start_idx + 1),
    }


def merge_or_dedupe_segments(candidates: list[dict[str, Any]], frames_by_key: dict[tuple[str, str], pd.DataFrame]) -> pd.DataFrame:
    """Merge same-symbol/timeframe/direction overlapping candidates into segments."""

    if not candidates:
        return pd.DataFrame(columns=SEGMENT_COLUMNS)
    sorted_candidates = sorted(
        candidates,
        key=lambda item: (item["symbol"], item["timeframe"], item["direction"], int(item["start_idx"]), int(item["end_idx"])),
    )
    merged: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for candidate in sorted_candidates:
        item = candidate.copy()
        item["labels"] = set(item.get("labels") or [])
        if current is None:
            current = item
            continue
        same_group = (
            item["symbol"] == current["symbol"]
            and item["timeframe"] == current["timeframe"]
            and item["direction"] == current["direction"]
        )
        if same_group and int(item["start_idx"]) <= int(current["end_idx"]) + 1:
            current["end_idx"] = max(int(current["end_idx"]), int(item["end_idx"]))
            current["start_idx"] = min(int(current["start_idx"]), int(item["start_idx"]))
            current["labels"].update(item["labels"])
            continue
        merged.append(current)
        current = item
    if current is not None:
        merged.append(current)

    records: list[dict[str, Any]] = []
    counters: dict[tuple[str, str], int] = {}
    for item in merged:
        symbol = str(item["symbol"])
        timeframe = str(item["timeframe"])
        key = (symbol, timeframe)
        frame = frames_by_key.get(key, pd.DataFrame())
        start_idx = int(item["start_idx"])
        end_idx = int(item["end_idx"])
        if frame.empty or start_idx >= len(frame.index) or end_idx >= len(frame.index):
            continue
        counters[key] = counters.get(key, 0) + 1
        labels = sorted(str(label) for label in item.get("labels", set()))
        atr_labels = [label for label in labels if label.startswith("uptrend_") or label.startswith("downtrend_")]
        efficiency_labels = [label for label in labels if label.startswith("trend_efficiency_")]
        metrics = directional_segment_metrics(frame, start_idx, end_idx, str(item["direction"]))
        start_time = pd.Timestamp(frame.iloc[start_idx]["datetime"])
        end_time = pd.Timestamp(frame.iloc[end_idx]["datetime"])
        naive_start = start_time.tz_localize(None) if start_time.tzinfo is not None else start_time
        segment_id = f"TOM_{safe_symbol(symbol)}_{timeframe}_{counters[key]:05d}"
        records.append(
            {
                "trend_segment_id": segment_id,
                "symbol": symbol,
                "timeframe": timeframe,
                "direction": item["direction"],
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "start_idx": start_idx,
                "end_idx": end_idx,
                "duration_bars": int(end_idx - start_idx + 1),
                "duration_days": float(max((end_time - start_time) / pd.Timedelta(days=1), 0.0)),
                "start_price": finite_float(frame.iloc[start_idx]["close"]),
                "end_price": finite_float(frame.iloc[end_idx]["close"]),
                "trend_return": metrics["trend_return"],
                "mfe": metrics["mfe"],
                "mae": metrics["mae"],
                "labels": ";".join(labels),
                "atr_labels": ";".join(atr_labels),
                "efficiency_labels": ";".join(efficiency_labels),
                "has_atr_move": bool(atr_labels),
                "has_efficiency": bool(efficiency_labels),
                "has_run_length": bool("run_length" in labels),
                "run_length": metrics["run_length"],
                "max_run_return": metrics["max_run_return"],
                "max_run_drawdown": metrics["max_run_drawdown"],
                "run_duration_bars": metrics["run_duration_bars"],
                "month": naive_start.strftime("%Y-%m"),
                "quarter": naive_start.to_period("Q").__str__(),
                "abs_trend_return": abs(float(metrics["trend_return"])) if metrics["trend_return"] is not None else np.nan,
                "is_major_trend": False,
            }
        )

    segments = pd.DataFrame(records, columns=SEGMENT_COLUMNS)
    if segments.empty:
        return segments
    segments["is_major_trend"] = False
    for (_symbol, _timeframe), group in segments.groupby(["symbol", "timeframe"], dropna=False):
        threshold = pd.to_numeric(group["abs_trend_return"], errors="coerce").quantile(0.75)
        duration_threshold = pd.to_numeric(group["duration_bars"], errors="coerce").quantile(0.75)
        mask = (segments.index.isin(group.index)) & (
            (pd.to_numeric(segments["abs_trend_return"], errors="coerce") >= threshold)
            | (pd.to_numeric(segments["duration_bars"], errors="coerce") >= duration_threshold)
        )
        segments.loc[mask, "is_major_trend"] = True
    return segments.sort_values(["symbol", "timeframe", "start_time", "direction"], kind="stable").reset_index(drop=True)


def build_frames_and_segments(
    *,
    symbols: list[str],
    history_range: HistoryRange,
    timeframes: list[str],
    database_path: Path,
    logger: logging.Logger,
    bars_by_symbol: dict[str, pd.DataFrame] | None = None,
) -> tuple[dict[tuple[str, str], pd.DataFrame], pd.DataFrame, dict[str, Any]]:
    """Load bars, build indicator frames, and create trend opportunity segments."""

    frames_by_key: dict[tuple[str, str], pd.DataFrame] = {}
    symbol_quality: dict[str, Any] = {}
    timeframe_quality: dict[str, Any] = {}

    for symbol in symbols:
        log_event(logger, logging.INFO, "trend_opportunity.load_symbol", "Loading symbol history", symbol=symbol)
        if bars_by_symbol is not None and symbol in bars_by_symbol:
            bars_1m = normalize_1m_bars(bars_by_symbol[symbol], history_range.timezone_name)
        else:
            bars_1m = load_1m_bars_from_sqlite(symbol, history_range, database_path)
        bars_1m = filter_time_range(bars_1m, history_range)
        one_minute_quality = analyze_1m_quality(symbol, bars_1m, history_range)
        symbol_quality[symbol] = one_minute_quality
        timeframe_quality[symbol] = {}
        for timeframe in timeframes:
            bars = resample_ohlcv_closed(bars_1m, timeframe, history_range)
            timeframe_quality[symbol][timeframe] = analyze_timeframe_quality(timeframe, bars, one_minute_quality)
            frame = compute_context_indicators(bars)
            frame = filter_time_range(frame, history_range)
            frames_by_key[(symbol, timeframe)] = frame

    frames_by_key = add_market_context(frames_by_key, symbols, timeframes)
    candidates: list[dict[str, Any]] = []
    for (symbol, timeframe), frame in frames_by_key.items():
        before_count = len(candidates)
        candidates.extend(atr_move_candidates(frame, symbol, timeframe))
        candidates.extend(trend_efficiency_candidates(frame, symbol, timeframe))
        candidates.extend(run_length_candidates(frame, symbol, timeframe))
        log_event(
            logger,
            logging.INFO,
            "trend_opportunity.candidates",
            "Generated trend candidates",
            symbol=symbol,
            timeframe=timeframe,
            candidate_count=len(candidates) - before_count,
            bar_count=len(frame.index),
        )
    segments = merge_or_dedupe_segments(candidates, frames_by_key)
    log_event(
        logger,
        logging.INFO,
        "trend_opportunity.segments",
        "Merged trend candidates",
        candidate_count=len(candidates),
        segment_count=len(segments.index),
    )

    data_quality = {
        "mode": "research_only_ex_post_trend_opportunity_map",
        "symbols": symbols,
        "timeframes": timeframes,
        "start": history_range.start.isoformat(),
        "end_display": history_range.end_display.isoformat(),
        "end_exclusive": history_range.end_exclusive.isoformat(),
        "timezone": history_range.timezone_name,
        "database_path": str(database_path),
        "symbol_coverage": symbol_quality,
        "timeframe_coverage": timeframe_quality,
        "all_symbols_complete": bool(symbol_quality and all(item.get("complete") for item in symbol_quality.values())),
        "strategy_development_allowed": False,
        "demo_live_allowed": False,
        "labels_are_ex_post_only": True,
        "silent_skip_symbol": False,
    }
    return frames_by_key, segments, data_quality


def ensure_strict_data_quality(data_quality: dict[str, Any], data_check_strict: bool) -> None:
    """Raise when strict data checking sees incomplete 1m coverage."""

    if not data_check_strict:
        return
    incomplete = [
        symbol
        for symbol, quality in data_quality.get("symbol_coverage", {}).items()
        if not quality.get("complete")
    ]
    if incomplete:
        raise TrendOpportunityMapError(f"Strict data check failed for symbols: {', '.join(incomplete)}")


def covered_days_ratio(group: pd.DataFrame, history_range: HistoryRange) -> float:
    """Return ratio of calendar days covered by at least one segment."""

    total_days = int((pd.Timestamp(history_range.end_display).normalize() - pd.Timestamp(history_range.start).normalize()) / pd.Timedelta(days=1)) + 1
    if total_days <= 0 or group.empty:
        return 0.0
    days: set[str] = set()
    for _, row in group.iterrows():
        start = pd.Timestamp(row["start_time"])
        end = pd.Timestamp(row["end_time"])
        if start.tzinfo is not None:
            start = start.tz_convert(history_range.timezone_name)
        if end.tzinfo is not None:
            end = end.tz_convert(history_range.timezone_name)
        for day in pd.date_range(start.normalize(), end.normalize(), freq="D"):
            days.add(day.strftime("%Y-%m-%d"))
    return float(min(len(days), total_days) / total_days)


def strongest_weakest_month(group: pd.DataFrame) -> tuple[str | None, str | None]:
    """Return strongest and weakest month labels by segment count then return."""

    if group.empty or "month" not in group.columns:
        return None, None
    monthly = (
        group.assign(abs_return=pd.to_numeric(group["abs_trend_return"], errors="coerce").fillna(0.0))
        .groupby("month", dropna=False)
        .agg(trend_segment_count=("trend_segment_id", "size"), avg_abs_return=("abs_return", "mean"))
        .reset_index()
    )
    if monthly.empty:
        return None, None
    strongest = monthly.sort_values(["trend_segment_count", "avg_abs_return", "month"], ascending=[False, False, True], kind="stable").iloc[0]
    weakest = monthly.sort_values(["trend_segment_count", "avg_abs_return", "month"], ascending=[True, True, True], kind="stable").iloc[0]
    return str(strongest["month"]), str(weakest["month"])


def aggregate_segments(segments: pd.DataFrame, group_cols: list[str], history_range: HistoryRange) -> pd.DataFrame:
    """Aggregate trend opportunity statistics by requested grouping columns."""

    if segments.empty:
        return pd.DataFrame(columns=group_cols + AGG_COLUMNS)
    rows: list[dict[str, Any]] = []
    for group_key, group in segments.groupby(group_cols, dropna=False):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        row = {column: group_key[index] for index, column in enumerate(group_cols)}
        strongest_month, weakest_month = strongest_weakest_month(group)
        row.update(
            {
                "trend_segment_count": int(len(group.index)),
                "uptrend_count": int((group["direction"] == "up").sum()),
                "downtrend_count": int((group["direction"] == "down").sum()),
                "avg_duration_bars": safe_mean(group["duration_bars"]),
                "median_duration_bars": safe_median(group["duration_bars"]),
                "avg_trend_return": safe_mean(group["trend_return"]),
                "median_trend_return": safe_median(group["trend_return"]),
                "avg_mfe": safe_mean(group["mfe"]),
                "avg_mae": safe_mean(group["mae"]),
                "trend_opportunity_days_ratio": covered_days_ratio(group, history_range),
                "strongest_month": strongest_month,
                "weakest_month": weakest_month,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(group_cols, kind="stable").reset_index(drop=True)


def load_funding_csv(path: Path) -> pd.DataFrame:
    """Load one funding CSV with normalized UTC timestamps."""

    frame = pd.read_csv(path)
    if frame.empty:
        return pd.DataFrame(columns=["funding_time_utc", "funding_rate"])
    if "funding_time_utc" in frame.columns:
        timestamps = pd.to_datetime(frame["funding_time_utc"], utc=True, errors="coerce")
    elif "funding_time" in frame.columns:
        timestamps = pd.to_datetime(pd.to_numeric(frame["funding_time"], errors="coerce"), unit="ms", utc=True, errors="coerce")
    else:
        raise TrendOpportunityMapError(f"funding CSV missing funding_time columns: {path}")
    result = pd.DataFrame({"funding_time_utc": timestamps, "funding_rate": pd.to_numeric(frame.get("funding_rate"), errors="coerce")})
    result = result.dropna(subset=["funding_time_utc", "funding_rate"])
    result = result.sort_values("funding_time_utc", kind="stable").drop_duplicates("funding_time_utc", keep="last")
    return result.reset_index(drop=True)


def select_funding_csv(funding_dir: Path, inst_id: str, start_arg: str, end_arg: str) -> Path | None:
    """Select the canonical funding CSV or most recent matching fallback."""

    canonical = funding_dir / f"{inst_id}_funding_{start_arg}_{end_arg}.csv"
    if canonical.exists():
        return canonical
    matches = sorted(funding_dir.glob(f"{inst_id}_funding_*.csv"))
    return matches[-1] if matches else None


def load_funding_histories(
    funding_dir: Path,
    symbols: list[str],
    history_range: HistoryRange,
    start_arg: str,
    end_arg: str,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    """Load funding histories for pre-trend diagnostics."""

    histories: dict[str, pd.DataFrame] = {}
    records: dict[str, Any] = {}
    for symbol in symbols:
        inst_id = symbol_to_inst_id(symbol)
        path = select_funding_csv(funding_dir, inst_id, start_arg, end_arg)
        if path is None:
            histories[inst_id] = pd.DataFrame(columns=["funding_time_utc", "funding_rate"])
            records[inst_id] = {"csv_path": None, "exists": False, "row_count": 0, "coverage_complete": False, "warnings": ["missing_funding_csv"]}
            continue
        try:
            histories[inst_id] = load_funding_csv(path)
        except Exception as exc:
            histories[inst_id] = pd.DataFrame(columns=["funding_time_utc", "funding_rate"])
            records[inst_id] = {
                "csv_path": str(path),
                "exists": True,
                "row_count": 0,
                "coverage_complete": False,
                "warnings": [f"funding_csv_read_failed:{exc!r}"],
            }
            continue
        funding = histories[inst_id]
        if funding.empty:
            records[inst_id] = {"csv_path": str(path), "exists": True, "row_count": 0, "coverage_complete": False, "warnings": ["empty_funding_csv"]}
            continue
        first_time = funding["funding_time_utc"].min()
        last_time = funding["funding_time_utc"].max()
        intervals = pd.DatetimeIndex(funding["funding_time_utc"]).to_series().diff().dropna().dt.total_seconds() / 3600.0
        large_gap_count = int((intervals > 24.0).sum())
        start_utc = pd.Timestamp(history_range.start).tz_convert("UTC")
        end_utc = pd.Timestamp(history_range.end_exclusive).tz_convert("UTC")
        coverage_complete = bool(first_time <= start_utc + pd.Timedelta(minutes=1) and last_time >= end_utc - pd.Timedelta(hours=8) and large_gap_count == 0)
        records[inst_id] = {
            "csv_path": str(path),
            "exists": True,
            "row_count": int(len(funding.index)),
            "first_funding_time": first_time.isoformat(),
            "last_funding_time": last_time.isoformat(),
            "large_gap_count": large_gap_count,
            "coverage_complete": coverage_complete,
            "warnings": [] if coverage_complete else ["partial_funding_coverage"],
        }
    quality = {
        "funding_data_complete": bool(records and all(record.get("coverage_complete") for record in records.values())),
        "records": records,
        "missing_inst_ids": [inst_id for inst_id, record in records.items() if not record.get("coverage_complete")],
    }
    return histories, quality


def funding_features_at(funding_df: pd.DataFrame, timestamp: Any) -> tuple[float | None, float | None]:
    """Return last known funding rate and prior percentile at timestamp."""

    if funding_df.empty or timestamp is None or pd.isna(timestamp):
        return None, None
    ts = pd.Timestamp(timestamp)
    if ts.tzinfo is None:
        ts = ts.tz_localize(DEFAULT_TIMEZONE)
    ts = ts.tz_convert("UTC")
    times = pd.DatetimeIndex(funding_df["funding_time_utc"])
    pos = int(times.searchsorted(ts, side="right")) - 1
    if pos < 0:
        return None, None
    rate = finite_float(funding_df.iloc[pos]["funding_rate"], default=None)
    if rate is None:
        return None, None
    prior = pd.to_numeric(funding_df.iloc[: pos + 1]["funding_rate"], errors="coerce").dropna()
    percentile = float((prior <= rate).mean()) if not prior.empty else None
    return rate, percentile


def feature_window_record(
    *,
    frame: pd.DataFrame,
    symbol: str,
    timeframe: str,
    segment_id: str,
    sample_type: str,
    end_idx: int,
    pre_window: int,
    funding_histories: dict[str, pd.DataFrame],
) -> dict[str, Any] | None:
    """Build one pre-window feature sample ending before end_idx."""

    start_idx = max(0, end_idx - pre_window)
    if end_idx <= 0 or start_idx >= end_idx:
        return None
    window = frame.iloc[start_idx:end_idx].copy()
    if window.empty:
        return None
    timestamp = frame.iloc[end_idx]["datetime"] if end_idx < len(frame.index) else window.iloc[-1]["datetime"]
    funding_rate, funding_percentile = funding_features_at(
        funding_histories.get(symbol_to_inst_id(symbol), pd.DataFrame(columns=["funding_time_utc", "funding_rate"])),
        timestamp,
    )
    return {
        "sample_type": sample_type,
        "segment_id": segment_id,
        "symbol": symbol,
        "timeframe": timeframe,
        "pre_window_bars": pre_window,
        "sample_time": format_timestamp(timestamp),
        "pre_trend_atr_percentile": safe_mean(window["atr_percentile"]),
        "pre_trend_volatility": safe_mean(window["realized_volatility"]),
        "pre_trend_volume_ratio": safe_mean(window["volume_ratio"]),
        "pre_trend_funding_rate": funding_rate,
        "pre_trend_funding_percentile": funding_percentile,
        "pre_trend_market_breadth": safe_mean(window.get("market_breadth", pd.Series(dtype=float))),
        "pre_trend_correlation": safe_mean(window.get("market_correlation", pd.Series(dtype=float))),
        "pre_trend_drawdown_state": safe_mean(window["drawdown_state"]),
        "pre_trend_range_width": safe_mean(window["range_width"]),
        "pre_trend_compression_score": safe_mean(window["compression_score"]),
    }


def nontrend_runs(length: int, segments: pd.DataFrame) -> list[tuple[int, int]]:
    """Return consecutive non-trend index runs for one symbol/timeframe."""

    covered = np.zeros(length, dtype=bool)
    for _, segment in segments.iterrows():
        start = max(0, int(segment["start_idx"]))
        end = min(length - 1, int(segment["end_idx"]))
        if end >= start:
            covered[start : end + 1] = True
    runs: list[tuple[int, int]] = []
    run_start: int | None = None
    for index, is_covered in enumerate(covered):
        if not is_covered and run_start is None:
            run_start = index
        elif is_covered and run_start is not None:
            runs.append((run_start, index - 1))
            run_start = None
    if run_start is not None:
        runs.append((run_start, length - 1))
    return runs


def build_pre_trend_feature_comparison(
    frames_by_key: dict[tuple[str, str], pd.DataFrame],
    segments: pd.DataFrame,
    funding_histories: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Compare observable pre-window features before trend and non-trend segments."""

    samples: list[dict[str, Any]] = []
    for _, segment in segments.iterrows():
        symbol = str(segment["symbol"])
        timeframe = str(segment["timeframe"])
        frame = frames_by_key.get((symbol, timeframe), pd.DataFrame())
        if frame.empty:
            continue
        start_idx = int(segment["start_idx"])
        for pre_window in PRE_TREND_WINDOWS:
            record = feature_window_record(
                frame=frame,
                symbol=symbol,
                timeframe=timeframe,
                segment_id=str(segment["trend_segment_id"]),
                sample_type="trend",
                end_idx=start_idx,
                pre_window=pre_window,
                funding_histories=funding_histories,
            )
            if record is not None:
                samples.append(record)

    for (symbol, timeframe), frame in frames_by_key.items():
        symbol_segments = segments[(segments["symbol"] == symbol) & (segments["timeframe"] == timeframe)] if not segments.empty else pd.DataFrame()
        for run_start, run_end in nontrend_runs(len(frame.index), symbol_segments):
            if run_end - run_start + 1 < min(PRE_TREND_WINDOWS):
                continue
            for pre_window in PRE_TREND_WINDOWS:
                if run_start < pre_window:
                    continue
                record = feature_window_record(
                    frame=frame,
                    symbol=symbol,
                    timeframe=timeframe,
                    segment_id=f"non_trend_{safe_symbol(symbol)}_{timeframe}_{run_start}",
                    sample_type="non_trend",
                    end_idx=run_start,
                    pre_window=pre_window,
                    funding_histories=funding_histories,
                )
                if record is not None:
                    samples.append(record)

    sample_frame = pd.DataFrame(samples)
    columns = [
        "feature",
        "pre_window_bars",
        "trend_sample_count",
        "non_trend_sample_count",
        "trend_mean",
        "non_trend_mean",
        "delta",
        "trend_median",
        "non_trend_median",
        "effect_size",
        "abs_effect_size",
    ]
    if sample_frame.empty:
        return pd.DataFrame(columns=columns)

    rows: list[dict[str, Any]] = []
    for pre_window, window_group in sample_frame.groupby("pre_window_bars", dropna=False):
        for feature in PRE_TREND_FEATURES:
            trend = pd.to_numeric(window_group.loc[window_group["sample_type"] == "trend", feature], errors="coerce").dropna()
            nontrend = pd.to_numeric(window_group.loc[window_group["sample_type"] == "non_trend", feature], errors="coerce").dropna()
            trend_mean = float(trend.mean()) if not trend.empty else np.nan
            nontrend_mean = float(nontrend.mean()) if not nontrend.empty else np.nan
            delta = trend_mean - nontrend_mean if np.isfinite(trend_mean) and np.isfinite(nontrend_mean) else np.nan
            pooled_std = float(pd.concat([trend, nontrend]).std(ddof=0)) if len(trend.index) + len(nontrend.index) > 1 else np.nan
            effect_size = delta / pooled_std if np.isfinite(delta) and np.isfinite(pooled_std) and pooled_std > 0 else np.nan
            rows.append(
                {
                    "feature": feature,
                    "pre_window_bars": int(pre_window),
                    "trend_sample_count": int(len(trend.index)),
                    "non_trend_sample_count": int(len(nontrend.index)),
                    "trend_mean": trend_mean,
                    "non_trend_mean": nontrend_mean,
                    "delta": delta,
                    "trend_median": float(trend.median()) if not trend.empty else np.nan,
                    "non_trend_median": float(nontrend.median()) if not nontrend.empty else np.nan,
                    "effect_size": effect_size,
                    "abs_effect_size": abs(effect_size) if np.isfinite(effect_size) else np.nan,
                }
            )
    return pd.DataFrame(rows, columns=columns).sort_values(["pre_window_bars", "abs_effect_size"], ascending=[True, False], kind="stable").reset_index(drop=True)


def select_pnl(row: pd.Series) -> float | None:
    """Select the most complete PnL field available on a legacy trade row."""

    for column in ["funding_adjusted_pnl", "cost_aware_pnl", "net_pnl", "no_cost_pnl", "no_cost_net_pnl", "gross_pnl"]:
        if column in row.index:
            value = finite_float(row.get(column), default=None)
            if value is not None:
                return value
    gross_return = finite_float(row.get("gross_return"), default=None)
    if gross_return is not None:
        return gross_return
    return None


def load_legacy_trades(trade_files: Iterable[tuple[str, Path]] = LEGACY_TRADE_FILES) -> tuple[pd.DataFrame, list[str]]:
    """Load existing legacy strategy trade files when present."""

    rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    for strategy_source, path in trade_files:
        if not path.exists():
            warnings.append(f"missing_trade_file:{strategy_source}:{path}")
            continue
        frame = pd.read_csv(path)
        if frame.empty:
            warnings.append(f"empty_trade_file:{strategy_source}:{path}")
            continue
        for index, row in frame.iterrows():
            trade_id = str(row.get("trade_id") or f"{safe_symbol(strategy_source)}_{index + 1:08d}")
            policy_or_group = row.get("policy_name")
            if policy_or_group is None or pd.isna(policy_or_group):
                policy_or_group = row.get("group")
            if policy_or_group is None or pd.isna(policy_or_group):
                policy_or_group = row.get("session_type", "unknown")
            rows.append(
                {
                    "strategy_source": strategy_source,
                    "policy_or_group": str(policy_or_group),
                    "symbol": row.get("symbol"),
                    "timeframe": row.get("timeframe"),
                    "trade_id": trade_id,
                    "entry_time": row.get("entry_time"),
                    "exit_time": row.get("exit_time"),
                    "direction": row.get("direction"),
                    "pnl": select_pnl(row),
                }
            )
    if not rows:
        return pd.DataFrame(columns=["strategy_source", "policy_or_group", "symbol", "timeframe", "trade_id", "entry_time", "exit_time", "direction", "pnl"]), warnings
    trades = pd.DataFrame(rows)
    trades["pnl"] = pd.to_numeric(trades["pnl"], errors="coerce")
    return trades, warnings


def timestamp_utc(value: Any) -> pd.Timestamp | None:
    """Parse a timestamp as UTC."""

    if value is None or pd.isna(value):
        return None
    parsed = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(parsed):
        return None
    return pd.Timestamp(parsed)


def overlap_fraction(entry: pd.Timestamp, exit_time: pd.Timestamp, segment_start: pd.Timestamp, segment_end: pd.Timestamp) -> float:
    """Return trade/segment overlap as a fraction of segment duration."""

    if segment_end < segment_start:
        return 0.0
    overlap_start = max(entry, segment_start)
    overlap_end = min(exit_time, segment_end)
    if overlap_end < overlap_start:
        return 0.0
    segment_seconds = max((segment_end - segment_start).total_seconds(), 1.0)
    return float(max((overlap_end - overlap_start).total_seconds(), 0.0) / segment_seconds)


def entry_phase(entry: pd.Timestamp, segment_start: pd.Timestamp | None, segment_end: pd.Timestamp | None) -> str:
    """Classify trade entry timing versus a trend segment."""

    if segment_start is None or segment_end is None:
        return "non_trend"
    if entry < segment_start:
        return "pre_segment"
    if entry > segment_end:
        return "post_segment"
    duration = max((segment_end - segment_start).total_seconds(), 1.0)
    ratio = (entry - segment_start).total_seconds() / duration
    if ratio <= 1.0 / 3.0:
        return "early"
    if ratio <= 2.0 / 3.0:
        return "middle"
    return "late"


def align_legacy_trades_to_segments(trades: pd.DataFrame, segments: pd.DataFrame, timeframes: list[str]) -> pd.DataFrame:
    """Align legacy trade holding intervals to trend opportunity segments."""

    if trades.empty:
        return pd.DataFrame(columns=LEGACY_COVERAGE_COLUMNS)
    segments_by_key: dict[tuple[str, str], list[dict[str, Any]]] = {}
    major_segments_by_key: dict[tuple[str, str], list[dict[str, Any]]] = {}
    if not segments.empty:
        segment_frame = segments.copy()
        segment_frame["start_ts"] = pd.to_datetime(segment_frame["start_time"], utc=True, errors="coerce")
        segment_frame["end_ts"] = pd.to_datetime(segment_frame["end_time"], utc=True, errors="coerce")
        segment_frame = segment_frame.dropna(subset=["start_ts", "end_ts"])
        for (symbol, timeframe), group in segment_frame.groupby(["symbol", "timeframe"], dropna=False):
            records: list[dict[str, Any]] = []
            for _, segment_row in group.sort_values("start_ts", kind="stable").iterrows():
                record = segment_row.to_dict()
                record["start_ts"] = pd.Timestamp(segment_row["start_ts"])
                record["end_ts"] = pd.Timestamp(segment_row["end_ts"])
                record["start_ns"] = int(record["start_ts"].value)
                record["end_ns"] = int(record["end_ts"].value)
                records.append(record)
            segments_by_key[(str(symbol), str(timeframe))] = records
            major_segments_by_key[(str(symbol), str(timeframe))] = [record for record in records if bool(record.get("is_major_trend"))]

    expanded_rows: list[dict[str, Any]] = []
    for _, trade in trades.iterrows():
        symbol = str(trade.get("symbol") or "")
        raw_timeframe = str(trade.get("timeframe") or "")
        candidate_timeframes = [raw_timeframe] if raw_timeframe in timeframes else list(timeframes)
        for timeframe in candidate_timeframes:
            expanded_rows.append(
                {
                    "strategy_source": trade.get("strategy_source"),
                    "policy_or_group": trade.get("policy_or_group"),
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "trade_id": trade.get("trade_id"),
                    "entry_time": trade.get("entry_time"),
                    "exit_time": trade.get("exit_time"),
                    "pnl": finite_float(trade.get("pnl"), default=np.nan),
                }
            )
    if not expanded_rows:
        return pd.DataFrame(columns=LEGACY_COVERAGE_COLUMNS)

    expanded = pd.DataFrame(expanded_rows)
    expanded["entry_ts"] = pd.to_datetime(expanded["entry_time"], utc=True, errors="coerce", format="mixed")
    expanded["exit_ts"] = pd.to_datetime(expanded["exit_time"], utc=True, errors="coerce", format="mixed")
    expanded = expanded.dropna(subset=["entry_ts", "exit_ts"]).reset_index(drop=True)
    if expanded.empty:
        return pd.DataFrame(columns=LEGACY_COVERAGE_COLUMNS)
    swapped = expanded["exit_ts"] < expanded["entry_ts"]
    if bool(swapped.any()):
        original_entry = expanded.loc[swapped, "entry_ts"].copy()
        expanded.loc[swapped, "entry_ts"] = expanded.loc[swapped, "exit_ts"]
        expanded.loc[swapped, "exit_ts"] = original_entry
    expanded["entry_ns"] = expanded["entry_ts"].map(lambda value: pd.Timestamp(value).value)
    expanded["exit_ns"] = expanded["exit_ts"].map(lambda value: pd.Timestamp(value).value)

    expanded["trend_segment_id"] = None
    expanded["entered_trend_segment"] = False
    expanded["entry_phase"] = "non_trend"
    expanded["captured_fraction_of_segment"] = 0.0
    expanded["missed_major_trend"] = False
    expanded["is_synthetic_missed_segment"] = False

    for (symbol, timeframe), group_index in expanded.groupby(["symbol", "timeframe"], dropna=False).groups.items():
        symbol_segments = segments_by_key.get((str(symbol), str(timeframe)), [])
        if not symbol_segments:
            continue
        indexes = np.array(list(group_index), dtype=int)
        entries_ns = expanded.loc[indexes, "entry_ns"].to_numpy(dtype=np.int64)
        exits_ns = expanded.loc[indexes, "exit_ns"].to_numpy(dtype=np.int64)
        best_capture = np.zeros(len(indexes), dtype=float)
        best_abs_return = np.full(len(indexes), -np.inf, dtype=float)
        best_duration = np.full(len(indexes), -np.inf, dtype=float)
        best_id = np.full(len(indexes), None, dtype=object)
        best_start = np.zeros(len(indexes), dtype=np.int64)
        best_end = np.zeros(len(indexes), dtype=np.int64)
        for segment in symbol_segments:
            segment_start = int(segment["start_ns"])
            segment_end = int(segment["end_ns"])
            overlap_mask = (entries_ns <= segment_end) & (exits_ns >= segment_start)
            if not bool(overlap_mask.any()):
                continue
            positions = np.flatnonzero(overlap_mask)
            overlap_ns = np.minimum(exits_ns[positions], segment_end) - np.maximum(entries_ns[positions], segment_start)
            duration_ns = max(segment_end - segment_start, 1)
            capture = np.maximum(overlap_ns, 0).astype(float) / float(duration_ns)
            abs_return = finite_float(segment.get("abs_trend_return"), default=0.0) or 0.0
            duration = finite_float(segment.get("duration_bars"), default=0.0) or 0.0
            better = (
                (capture > best_capture[positions])
                | ((capture == best_capture[positions]) & (abs_return > best_abs_return[positions]))
                | ((capture == best_capture[positions]) & (abs_return == best_abs_return[positions]) & (duration > best_duration[positions]))
            )
            if not bool(better.any()):
                continue
            update_positions = positions[better]
            best_capture[update_positions] = capture[better]
            best_abs_return[update_positions] = abs_return
            best_duration[update_positions] = duration
            best_id[update_positions] = segment.get("trend_segment_id")
            best_start[update_positions] = segment_start
            best_end[update_positions] = segment_end
        entered = pd.notna(best_id)
        if not bool(entered.any()):
            continue
        target_indexes = indexes[entered]
        expanded.loc[target_indexes, "trend_segment_id"] = best_id[entered]
        expanded.loc[target_indexes, "entered_trend_segment"] = True
        expanded.loc[target_indexes, "captured_fraction_of_segment"] = best_capture[entered]
        phases: list[str] = []
        for entry_ns, start_ns, end_ns in zip(entries_ns[entered], best_start[entered], best_end[entered], strict=False):
            if entry_ns < start_ns:
                phases.append("pre_segment")
            elif entry_ns > end_ns:
                phases.append("post_segment")
            else:
                ratio = float(entry_ns - start_ns) / float(max(end_ns - start_ns, 1))
                if ratio <= 1.0 / 3.0:
                    phases.append("early")
                elif ratio <= 2.0 / 3.0:
                    phases.append("middle")
                else:
                    phases.append("late")
        expanded.loc[target_indexes, "entry_phase"] = phases

    coverage = expanded.loc[:, LEGACY_COVERAGE_COLUMNS].copy()
    if coverage.empty or not major_segments_by_key:
        return coverage

    for (strategy_source, policy_or_group), group in coverage[~coverage["is_synthetic_missed_segment"]].groupby(["strategy_source", "policy_or_group"], dropna=False):
        traded_symbols = sorted(set(str(value) for value in group["symbol"].dropna()))
        for symbol in traded_symbols:
            for timeframe in timeframes:
                major_segments = major_segments_by_key.get((symbol, timeframe), [])
                policy_rows = group[(group["symbol"] == symbol) & (group["timeframe"] == timeframe)]
                captured_ids = set(policy_rows.loc[policy_rows["entered_trend_segment"], "trend_segment_id"].dropna())
                for segment in major_segments:
                    if segment.get("trend_segment_id") in captured_ids:
                        continue
                    coverage.loc[len(coverage.index)] = {
                        "strategy_source": strategy_source,
                        "policy_or_group": policy_or_group,
                        "symbol": symbol,
                        "timeframe": timeframe,
                        "trade_id": f"missed_{segment.get('trend_segment_id')}",
                        "entry_time": segment.get("start_time"),
                        "exit_time": segment.get("end_time"),
                        "trend_segment_id": segment.get("trend_segment_id"),
                        "entered_trend_segment": False,
                        "entry_phase": "missed",
                        "captured_fraction_of_segment": 0.0,
                        "missed_major_trend": True,
                        "pnl": 0.0,
                        "is_synthetic_missed_segment": True,
                    }
    return coverage.sort_values(["strategy_source", "policy_or_group", "symbol", "timeframe", "entry_time"], kind="stable").reset_index(drop=True)


def legacy_coverage_summary(coverage: pd.DataFrame) -> dict[str, Any]:
    """Summarize legacy strategy coverage behavior."""

    if coverage.empty:
        return {
            "legacy_trade_count": 0,
            "real_trade_count": 0,
            "entered_trend_trade_share": None,
            "avg_captured_fraction": None,
            "missed_major_trend_count": 0,
            "legacy_strategies_failed_to_capture_trends": False,
            "legacy_strategies_trade_too_much_in_nontrend": False,
            "nontrend_loss_share": None,
            "early_exit_share": None,
            "late_entry_share": None,
            "main_failure_mode": "legacy_trade_files_missing",
        }

    real = coverage[~coverage["is_synthetic_missed_segment"]].copy()
    if real.empty:
        real = pd.DataFrame(columns=coverage.columns)
    real_grouped = real.groupby(["strategy_source", "policy_or_group", "trade_id"], dropna=False).agg(
        entered=("entered_trend_segment", "max"),
        pnl=("pnl", "first"),
        captured_fraction=("captured_fraction_of_segment", "max"),
        entry_phase=("entry_phase", lambda values: ";".join(sorted(set(str(value) for value in values)))),
    )
    real_trade_count = int(len(real_grouped.index))
    entered_share = float(real_grouped["entered"].mean()) if real_trade_count else None
    avg_capture = float(pd.to_numeric(real_grouped["captured_fraction"], errors="coerce").mean()) if real_trade_count else None
    losses = real_grouped[pd.to_numeric(real_grouped["pnl"], errors="coerce") < 0]
    nontrend_losses = losses[~losses["entered"].astype(bool)]
    total_loss = float(pd.to_numeric(losses["pnl"], errors="coerce").abs().sum())
    nontrend_loss = float(pd.to_numeric(nontrend_losses["pnl"], errors="coerce").abs().sum())
    nontrend_loss_share = nontrend_loss / total_loss if total_loss > 0 else None
    late_entry_share = float(real["entry_phase"].isin(["middle", "late"]).mean()) if not real.empty else None
    early_exit_share = float((pd.to_numeric(real["captured_fraction_of_segment"], errors="coerce") < 0.50).mean()) if not real.empty else None
    missed_major_count = int(coverage["missed_major_trend"].sum())
    failed_to_capture = bool(missed_major_count > 0 or (avg_capture is not None and avg_capture < 0.35) or (entered_share is not None and entered_share < 0.35))
    too_much_nontrend = bool(nontrend_loss_share is not None and nontrend_loss_share >= 0.50 and len(losses.index) >= 5)

    if too_much_nontrend:
        main_failure_mode = "trade_too_much_in_nontrend"
    elif late_entry_share is not None and late_entry_share >= 0.50:
        main_failure_mode = "entered_middle_or_late"
    elif early_exit_share is not None and early_exit_share >= 0.50:
        main_failure_mode = "exited_too_early"
    elif missed_major_count > 0:
        main_failure_mode = "missed_major_trends"
    elif failed_to_capture:
        main_failure_mode = "low_trend_capture"
    else:
        main_failure_mode = "mixed_or_not_primary_failure"

    return {
        "legacy_trade_count": int(len(coverage.index)),
        "real_trade_count": real_trade_count,
        "entered_trend_trade_share": entered_share,
        "avg_captured_fraction": avg_capture,
        "missed_major_trend_count": missed_major_count,
        "legacy_strategies_failed_to_capture_trends": failed_to_capture,
        "legacy_strategies_trade_too_much_in_nontrend": too_much_nontrend,
        "nontrend_loss_share": nontrend_loss_share,
        "early_exit_share": early_exit_share,
        "late_entry_share": late_entry_share,
        "main_failure_mode": main_failure_mode,
    }


def top_row(frame: pd.DataFrame, sort_columns: list[str], ascending: list[bool]) -> dict[str, Any] | None:
    """Return the first row after sorting, as JSON-safe dict."""

    if frame.empty:
        return None
    sorted_frame = frame.sort_values(sort_columns, ascending=ascending, kind="stable")
    return dataframe_records(sorted_frame.head(1))[0]


def pre_feature_decision(comparison: pd.DataFrame) -> tuple[bool, list[str]]:
    """Decide whether observable pre-trend feature differences exist."""

    if comparison.empty:
        return False, []
    eligible = comparison[
        (pd.to_numeric(comparison["trend_sample_count"], errors="coerce") >= 5)
        & (pd.to_numeric(comparison["non_trend_sample_count"], errors="coerce") >= 5)
        & (pd.to_numeric(comparison["abs_effect_size"], errors="coerce") >= 0.25)
    ].copy()
    if eligible.empty:
        return False, []
    strongest = eligible.sort_values(["abs_effect_size", "feature"], ascending=[False, True], kind="stable").head(5)
    return bool(len(strongest.index) >= 2), [str(row["feature"]) for _, row in strongest.iterrows()]


def build_summary(
    *,
    symbols: list[str],
    timeframes: list[str],
    history_range: HistoryRange,
    segments: pd.DataFrame,
    by_symbol: pd.DataFrame,
    by_timeframe: pd.DataFrame,
    by_month: pd.DataFrame,
    by_quarter: pd.DataFrame,
    legacy_summary: dict[str, Any],
    pre_feature_comparison: pd.DataFrame,
    data_quality: dict[str, Any],
    funding_quality: dict[str, Any],
    legacy_warnings: list[str],
) -> dict[str, Any]:
    """Build the decision JSON summary."""

    total_segments = int(len(segments.index))
    total_up = int((segments["direction"] == "up").sum()) if not segments.empty else 0
    total_down = int((segments["direction"] == "down").sum()) if not segments.empty else 0
    total_days_ratio = covered_days_ratio(segments, history_range) if not segments.empty else 0.0
    active_symbols = int((by_symbol["trend_segment_count"] > 0).sum()) if not by_symbol.empty else 0
    active_timeframes = int((by_timeframe["trend_segment_count"] > 0).sum()) if not by_timeframe.empty else 0
    top_symbol_share = 1.0
    if not by_symbol.empty and by_symbol["trend_segment_count"].sum() > 0:
        top_symbol_share = float(by_symbol["trend_segment_count"].max() / by_symbol["trend_segment_count"].sum())
    enough_trend_opportunities = bool(total_segments >= 20 and total_days_ratio >= 0.10 and active_symbols >= 3)
    diversified = bool(active_symbols >= min(3, len(symbols)) and active_timeframes >= min(2, len(timeframes)) and top_symbol_share <= 0.55)
    pre_features_exist, strongest_features = pre_feature_decision(pre_feature_comparison)

    strongest_symbol = top_row(by_symbol, ["trend_segment_count", "avg_trend_return", "symbol"], [False, False, True]) if not by_symbol.empty else None
    strongest_timeframe = top_row(by_timeframe, ["trend_segment_count", "avg_trend_return", "timeframe"], [False, False, True]) if not by_timeframe.empty else None
    strongest_month = top_row(by_month, ["trend_segment_count", "avg_trend_return", "month"], [False, False, True]) if not by_month.empty else None
    weakest_month = top_row(by_month, ["trend_segment_count", "avg_trend_return", "month"], [True, True, True]) if not by_month.empty else None
    strongest_quarter = top_row(by_quarter, ["trend_segment_count", "avg_trend_return", "quarter"], [False, False, True]) if not by_quarter.empty else None
    weakest_quarter = top_row(by_quarter, ["trend_segment_count", "avg_trend_return", "quarter"], [True, True, True]) if not by_quarter.empty else None

    if not enough_trend_opportunities:
        recommendation = "pause_trend_research"
    elif legacy_summary.get("legacy_strategies_trade_too_much_in_nontrend"):
        recommendation = "trend_entry_timing_research"
    elif legacy_summary.get("early_exit_share") is not None and legacy_summary.get("early_exit_share") >= 0.50:
        recommendation = "trend_exit_convexity_research"
    elif diversified:
        recommendation = "relative_strength_trend_following"
    elif pre_features_exist:
        recommendation = "breakout_retest_continuation"
    else:
        recommendation = "trend_entry_timing_research"

    return {
        "mode": "research_only_trend_opportunity_map",
        "output_files": REQUIRED_OUTPUT_FILES,
        "symbols": symbols,
        "timeframes": timeframes,
        "start": history_range.start.isoformat(),
        "end_display": history_range.end_display.isoformat(),
        "end_exclusive": history_range.end_exclusive.isoformat(),
        "labels_are_ex_post_only": True,
        "labels_must_not_be_used_as_entry_features": True,
        "strategy_development_allowed": False,
        "demo_live_allowed": False,
        "tradable": False,
        "trend_segment_count": total_segments,
        "uptrend_count": total_up,
        "downtrend_count": total_down,
        "trend_opportunity_days_ratio": total_days_ratio,
        "active_symbol_count": active_symbols,
        "active_timeframe_count": active_timeframes,
        "top_symbol_segment_share": top_symbol_share,
        "strongest_symbol": strongest_symbol,
        "strongest_timeframe": strongest_timeframe,
        "strongest_month": strongest_month,
        "weakest_month": weakest_month,
        "strongest_quarter": strongest_quarter,
        "weakest_quarter": weakest_quarter,
        "enough_trend_opportunities": enough_trend_opportunities,
        "trend_opportunities_are_diversified": diversified,
        "legacy_strategies_failed_to_capture_trends": bool(legacy_summary.get("legacy_strategies_failed_to_capture_trends")),
        "legacy_strategies_trade_too_much_in_nontrend": bool(legacy_summary.get("legacy_strategies_trade_too_much_in_nontrend")),
        "pre_trend_features_exist": pre_features_exist,
        "strongest_pre_trend_features": strongest_features,
        "recommended_next_research_direction": recommendation,
        "legacy_analysis": legacy_summary,
        "data_quality": {
            "all_symbols_complete": bool(data_quality.get("all_symbols_complete")),
            "funding_data_complete": bool(funding_quality.get("funding_data_complete")),
            "funding_missing_inst_ids": funding_quality.get("missing_inst_ids") or [],
            "legacy_warnings": legacy_warnings,
        },
        "answers": {
            "enough_trend_opportunities": enough_trend_opportunities,
            "most_opportunities_symbol": (strongest_symbol or {}).get("symbol"),
            "most_opportunities_timeframe": (strongest_timeframe or {}).get("timeframe"),
            "strongest_month": (strongest_month or {}).get("month"),
            "strongest_quarter": (strongest_quarter or {}).get("quarter"),
            "more_up_or_down": "up" if total_up > total_down else "down" if total_down > total_up else "balanced",
            "legacy_captured_trends": not bool(legacy_summary.get("legacy_strategies_failed_to_capture_trends")),
            "legacy_main_failure_mode": legacy_summary.get("main_failure_mode"),
            "legacy_losses_mainly_nontrend": bool(legacy_summary.get("legacy_strategies_trade_too_much_in_nontrend")),
            "observable_pre_trend_features": pre_features_exist,
            "next_research_direction": recommendation,
        },
    }


def pct_text(value: Any) -> str:
    """Format a ratio as percentage text."""

    number = finite_float(value, default=None)
    if number is None:
        return "n/a"
    return f"{number:.2%}"


def number_text(value: Any, digits: int = 4) -> str:
    """Format a number for Markdown."""

    number = finite_float(value, default=None)
    if number is None:
        return "n/a"
    return f"{number:.{digits}f}"


def generate_report(summary: dict[str, Any]) -> str:
    """Render the required research report."""

    answers = summary.get("answers") or {}
    legacy = summary.get("legacy_analysis") or {}
    strongest_symbol = summary.get("strongest_symbol") or {}
    strongest_timeframe = summary.get("strongest_timeframe") or {}
    strongest_month = summary.get("strongest_month") or {}
    weakest_month = summary.get("weakest_month") or {}
    strongest_quarter = summary.get("strongest_quarter") or {}
    weakest_quarter = summary.get("weakest_quarter") or {}
    features = summary.get("strongest_pre_trend_features") or []
    lines = [
        "# Trend Opportunity Map",
        "",
        "This report is a research-only diagnostic. Trend labels are ex-post labels and must not be used as entry signals.",
        "",
        "## Decision Flags",
        f"- enough_trend_opportunities={str(bool(summary.get('enough_trend_opportunities'))).lower()}",
        f"- trend_opportunities_are_diversified={str(bool(summary.get('trend_opportunities_are_diversified'))).lower()}",
        f"- legacy_strategies_failed_to_capture_trends={str(bool(summary.get('legacy_strategies_failed_to_capture_trends'))).lower()}",
        f"- legacy_strategies_trade_too_much_in_nontrend={str(bool(summary.get('legacy_strategies_trade_too_much_in_nontrend'))).lower()}",
        f"- pre_trend_features_exist={str(bool(summary.get('pre_trend_features_exist'))).lower()}",
        f"- recommended_next_research_direction={summary.get('recommended_next_research_direction')}",
        f"- strategy_development_allowed={str(bool(summary.get('strategy_development_allowed'))).lower()}",
        f"- demo_live_allowed={str(bool(summary.get('demo_live_allowed'))).lower()}",
        f"- tradable={str(bool(summary.get('tradable'))).lower()}",
        "",
        "## Opportunity Summary",
        f"- trend_segment_count={summary.get('trend_segment_count')}",
        f"- uptrend_count={summary.get('uptrend_count')}",
        f"- downtrend_count={summary.get('downtrend_count')}",
        f"- trend_opportunity_days_ratio={pct_text(summary.get('trend_opportunity_days_ratio'))}",
        f"- strongest_symbol={strongest_symbol.get('symbol')} count={strongest_symbol.get('trend_segment_count')}",
        f"- strongest_timeframe={strongest_timeframe.get('timeframe')} count={strongest_timeframe.get('trend_segment_count')}",
        f"- strongest_month={strongest_month.get('month')} weakest_month={weakest_month.get('month')}",
        f"- strongest_quarter={strongest_quarter.get('quarter')} weakest_quarter={weakest_quarter.get('quarter')}",
        "",
        "## Legacy Coverage",
        f"- real_trade_count={legacy.get('real_trade_count')}",
        f"- entered_trend_trade_share={pct_text(legacy.get('entered_trend_trade_share'))}",
        f"- avg_captured_fraction={pct_text(legacy.get('avg_captured_fraction'))}",
        f"- missed_major_trend_count={legacy.get('missed_major_trend_count')}",
        f"- nontrend_loss_share={pct_text(legacy.get('nontrend_loss_share'))}",
        f"- late_entry_share={pct_text(legacy.get('late_entry_share'))}",
        f"- early_exit_share={pct_text(legacy.get('early_exit_share'))}",
        f"- main_failure_mode={legacy.get('main_failure_mode')}",
        "",
        "## Required Answers",
        f"1. Did 2023-2026 five-symbol data contain enough trend opportunities? {str(bool(answers.get('enough_trend_opportunities'))).lower()}.",
        f"2. Which symbol had the most opportunities? {answers.get('most_opportunities_symbol')}.",
        f"3. Which timeframe had the most opportunities? {answers.get('most_opportunities_timeframe')}.",
        f"4. Main trend months/quarters: {answers.get('strongest_month')} / {answers.get('strongest_quarter')}.",
        f"5. More uptrends or downtrends? {answers.get('more_up_or_down')}.",
        f"6. Did old V3/VSVCB/CSRB capture these trends? {str(bool(answers.get('legacy_captured_trends'))).lower()}.",
        f"7. Main old-strategy failure mode: {answers.get('legacy_main_failure_mode')}.",
        f"8. Were old-strategy losses mainly in non-trend periods? {str(bool(answers.get('legacy_losses_mainly_nontrend'))).lower()}.",
        f"9. Are there observable common pre-trend features? {str(bool(answers.get('observable_pre_trend_features'))).lower()}; strongest={', '.join(features) or 'none'}.",
        f"10. Recommended next research direction: {answers.get('next_research_direction')}.",
        "",
        "## Output Files",
    ]
    lines.extend([f"- {name}" for name in REQUIRED_OUTPUT_FILES])
    lines.append("")
    return "\n".join(lines)


def write_outputs(
    output_dir: Path,
    *,
    summary: dict[str, Any],
    data_quality: dict[str, Any],
    trend_segments: pd.DataFrame,
    by_symbol: pd.DataFrame,
    by_timeframe: pd.DataFrame,
    by_month: pd.DataFrame,
    by_quarter: pd.DataFrame,
    legacy_coverage: pd.DataFrame,
    pre_trend_feature_comparison: pd.DataFrame,
) -> None:
    """Write all required output artifacts."""

    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "trend_opportunity_summary.json", summary)
    write_json(output_dir / "data_quality.json", data_quality)
    (output_dir / "trend_opportunity_report.md").write_text(generate_report(summary), encoding="utf-8")
    trend_segments.to_csv(output_dir / "trend_segments.csv", index=False)
    by_symbol.to_csv(output_dir / "trend_opportunity_by_symbol.csv", index=False)
    by_timeframe.to_csv(output_dir / "trend_opportunity_by_timeframe.csv", index=False)
    by_month.to_csv(output_dir / "trend_opportunity_by_month.csv", index=False)
    by_quarter.to_csv(output_dir / "trend_opportunity_by_quarter.csv", index=False)
    legacy_coverage.to_csv(output_dir / "legacy_strategy_trend_coverage.csv", index=False)
    pre_trend_feature_comparison.to_csv(output_dir / "pre_trend_feature_comparison.csv", index=False)


def run_research(
    *,
    symbols: list[str],
    start: str,
    end: str,
    timezone_name: str,
    timeframes: list[str],
    output_dir: Path,
    funding_dir: Path,
    database_path: Path,
    data_check_strict: bool,
    logger: logging.Logger,
    bars_by_symbol: dict[str, pd.DataFrame] | None = None,
    legacy_trade_files: Iterable[tuple[str, Path]] = LEGACY_TRADE_FILES,
) -> OpportunityOutputs:
    """Run the full Trend Opportunity Map workflow."""

    history_range = resolve_history_range(start, end, timezone_name)
    frames_by_key, trend_segments, data_quality = build_frames_and_segments(
        symbols=symbols,
        history_range=history_range,
        timeframes=timeframes,
        database_path=database_path,
        logger=logger,
        bars_by_symbol=bars_by_symbol,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "data_quality.json", data_quality)
    ensure_strict_data_quality(data_quality, data_check_strict)

    by_symbol = aggregate_segments(trend_segments, ["symbol"], history_range)
    by_timeframe = aggregate_segments(trend_segments, ["timeframe"], history_range)
    by_month = aggregate_segments(trend_segments, ["month"], history_range)
    by_quarter = aggregate_segments(trend_segments, ["quarter"], history_range)

    funding_histories, funding_quality = load_funding_histories(funding_dir, symbols, history_range, start, end)
    data_quality["funding"] = funding_quality
    log_event(
        logger,
        logging.INFO,
        "trend_opportunity.funding_loaded",
        "Loaded funding histories",
        funding_data_complete=funding_quality.get("funding_data_complete"),
    )

    legacy_trades, legacy_warnings = load_legacy_trades(legacy_trade_files)
    log_event(
        logger,
        logging.INFO,
        "trend_opportunity.legacy_loaded",
        "Loaded legacy trades",
        trade_count=len(legacy_trades.index),
    )
    legacy_coverage = align_legacy_trades_to_segments(legacy_trades, trend_segments, timeframes)
    log_event(
        logger,
        logging.INFO,
        "trend_opportunity.legacy_aligned",
        "Aligned legacy trades to trend segments",
        coverage_rows=len(legacy_coverage.index),
    )
    legacy_summary = legacy_coverage_summary(legacy_coverage)

    pre_trend_feature_comparison = build_pre_trend_feature_comparison(frames_by_key, trend_segments, funding_histories)
    log_event(
        logger,
        logging.INFO,
        "trend_opportunity.pre_features",
        "Built pre-trend feature comparison",
        row_count=len(pre_trend_feature_comparison.index),
    )

    summary = build_summary(
        symbols=symbols,
        timeframes=timeframes,
        history_range=history_range,
        segments=trend_segments,
        by_symbol=by_symbol,
        by_timeframe=by_timeframe,
        by_month=by_month,
        by_quarter=by_quarter,
        legacy_summary=legacy_summary,
        pre_feature_comparison=pre_trend_feature_comparison,
        data_quality=data_quality,
        funding_quality=funding_quality,
        legacy_warnings=legacy_warnings,
    )
    write_outputs(
        output_dir,
        summary=summary,
        data_quality=data_quality,
        trend_segments=trend_segments,
        by_symbol=by_symbol,
        by_timeframe=by_timeframe,
        by_month=by_month,
        by_quarter=by_quarter,
        legacy_coverage=legacy_coverage,
        pre_trend_feature_comparison=pre_trend_feature_comparison,
    )
    return OpportunityOutputs(
        output_dir=output_dir,
        summary=summary,
        data_quality=data_quality,
        trend_segments=trend_segments,
        legacy_coverage=legacy_coverage,
        pre_trend_feature_comparison=pre_trend_feature_comparison,
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""

    ensure_headless_runtime()
    args = parse_args(argv)
    logger = setup_logging("research_trend_opportunity_map", verbose=args.verbose)
    try:
        outputs = run_research(
            symbols=parse_symbols(args.symbols),
            start=args.start,
            end=args.end,
            timezone_name=args.timezone,
            timeframes=parse_timeframes(args.timeframes),
            output_dir=resolve_path(args.output_dir),
            funding_dir=resolve_path(args.funding_dir),
            database_path=resolve_path(args.database_path),
            data_check_strict=bool(args.data_check_strict),
            logger=logger,
        )
        log_event(
            logger,
            logging.INFO,
            "trend_opportunity.completed",
            "Trend Opportunity Map completed",
            output_dir=str(outputs.output_dir),
            trend_segment_count=outputs.summary.get("trend_segment_count"),
            recommended_next_research_direction=outputs.summary.get("recommended_next_research_direction"),
        )
        if args.json:
            print_json_block(outputs.summary)
        return 0
    except TrendOpportunityMapError as exc:
        log_event(logger, logging.ERROR, "trend_opportunity.error", str(exc))
        return 2
    except Exception as exc:
        log_event(logger, logging.ERROR, "trend_opportunity.unexpected_error", str(exc), exc_info=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
