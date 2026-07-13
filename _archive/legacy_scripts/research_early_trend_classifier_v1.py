#!/usr/bin/env python3
"""ETC-v1 early trend classifier feature discovery.

This script is research-only. It builds ex-post early-trend labels from the
Trend Opportunity Map, builds only pre-entry closed-bar features, and runs
bucket, score, event-study, random-control, reverse-control, cost, funding,
and concentration diagnostics. It does not modify strategies or connect to an
exchange.
"""

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

# 2026-07 重构批次6：脚本迁入 _archive/legacy_scripts/；共享依赖真身在
# scripts/（前向冻结区）与 core/data_io/，此处按新深度注入 sys.path。
import sys as _sys
from pathlib import Path as _Path

_REPO_ROOT = _Path(__file__).resolve().parents[2]
for _p in (
    str(_REPO_ROOT / "core" / "data_io"),
    str(_REPO_ROOT / "scripts"),
    str(_REPO_ROOT / "data_engineering" / "scripts"),
    *sorted(str(_q) for _q in (_REPO_ROOT / "research" / "_closed").glob("*/*/scripts")),
):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)

from common_runtime import PROJECT_ROOT, ensure_headless_runtime, log_event, setup_logging, to_jsonable
from history_time_utils import HistoryRange, expected_bar_count, parse_history_range, resolve_timezone


DEFAULT_SYMBOLS = [
    "BTCUSDT_SWAP_OKX.GLOBAL",
    "ETHUSDT_SWAP_OKX.GLOBAL",
    "SOLUSDT_SWAP_OKX.GLOBAL",
    "LINKUSDT_SWAP_OKX.GLOBAL",
    "DOGEUSDT_SWAP_OKX.GLOBAL",
]
DEFAULT_START = "2023-01-01"
DEFAULT_END = "2026-03-31"
DEFAULT_TIMEZONE = "Asia/Shanghai"
DEFAULT_TIMEFRAMES = ["4h", "1d"]
DEFAULT_TREND_MAP_DIR = PROJECT_ROOT / "reports" / "research" / "trend_opportunity_map"
DEFAULT_FUNDING_DIR = PROJECT_ROOT / "data" / "funding" / "okx"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "reports" / "research" / "early_trend_classifier_v1"
DEFAULT_DATABASE_PATH = PROJECT_ROOT / ".vntrader" / "database.db"
TIMEFRAME_MINUTES = {"4h": 240, "1d": 1440}
FUTURE_NON_TREND_WINDOW_BARS = 5
FIXED_NOTIONAL = 1000.0
RANDOM_SEED = 37

LABEL_COLUMNS = [
    "timestamp",
    "symbol",
    "timeframe",
    "label",
    "trend_segment_id",
    "trend_direction",
    "trend_phase_pct",
    "segment_start",
    "segment_end",
    "segment_return",
    "segment_atr_move",
    "segment_efficiency",
]

REQUIRED_FEATURE_COLUMNS = [
    "trend_efficiency_20",
    "trend_efficiency_55",
    "trend_efficiency_100",
    "trend_efficiency_change_20",
    "trend_efficiency_change_55",
    "directional_efficiency_ratio_20",
    "directional_efficiency_ratio_55",
    "positive_return_symbol_count_4h",
    "positive_return_symbol_count_1d",
    "symbols_above_rolling_midline_20",
    "symbols_above_rolling_midline_55",
    "breadth_change_1d",
    "breadth_change_3d",
    "breadth_acceleration",
    "market_return_breadth_20",
    "market_return_breadth_55",
    "return_rank_20",
    "return_rank_55",
    "return_rank_100",
    "vol_adjusted_return_rank_20",
    "vol_adjusted_return_rank_55",
    "trend_efficiency_rank_20",
    "trend_efficiency_rank_55",
    "rank_change_5",
    "rank_change_20",
    "rank_persistence_20",
    "atr_percentile_100",
    "atr_percentile_200",
    "realized_vol_percentile_100",
    "volatility_expansion_rate_20",
    "range_width_percentile_100",
    "efficiency_to_volatility_ratio_20",
    "efficiency_to_volatility_ratio_55",
    "funding_rate",
    "funding_percentile_100",
    "funding_percentile_200",
    "funding_sign",
    "funding_change_7d",
    "funding_change_30d",
    "funding_dispersion",
    "positive_funding_symbol_count",
    "negative_funding_symbol_count",
    "extreme_funding_count",
    "distance_from_20d_high",
    "distance_from_60d_high",
    "drawdown_from_60d_high",
    "rebound_from_recent_low_20",
    "rebound_from_recent_low_60",
    "price_position_in_60d_range",
    "volume_ratio_20",
    "volume_ratio_median_5",
    "volume_ratio_slope_5",
    "volume_stability_20",
    "volume_without_price_exhaustion_score",
]

DERIVED_SCORE_COLUMNS = ["funding_overheat_score", "volatility_noise_score"]
COMPOSITE_TERMS = [
    ("trend_efficiency_change_20", 1.0),
    ("breadth_acceleration", 1.0),
    ("rank_change_20", 1.0),
    ("efficiency_to_volatility_ratio_20", 1.0),
    ("funding_overheat_score", -1.0),
    ("volatility_noise_score", -1.0),
]
SPLIT_NAMES = ["train_ext", "validation_ext", "oos_ext"]
SCORE_BUCKETS = ["top10", "top20", "top30", "bottom30", "random_control"]
EVENT_GROUPS = {
    "A": "top10_score_events",
    "B": "top20_score_events",
    "C": "top30_score_events",
    "D": "random_score_matched_control",
    "E": "reverse_direction_test",
}
HOLD_MINUTES = {
    "hold_4h": 240,
    "hold_8h": 480,
    "hold_1d": 1440,
    "hold_3d": 4320,
}


class EtcResearchError(Exception):
    """Raised when ETC-v1 research cannot continue safely."""


@dataclass(frozen=True, slots=True)
class TimeSplit:
    """One fixed extended split interval."""

    name: str
    start: pd.Timestamp
    end: pd.Timestamp


@dataclass(frozen=True, slots=True)
class EtcConfig:
    """ETC-v1 fixed research parameters."""

    fixed_notional: float = FIXED_NOTIONAL
    fee_bps_per_side: float = 5.0
    slippage_bps_per_side: float = 5.0
    random_seed: int = RANDOM_SEED
    future_nontrend_window_bars: int = FUTURE_NON_TREND_WINDOW_BARS


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description="ETC-v1 early trend classifier feature discovery.")
    parser.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS))
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--end", default=DEFAULT_END)
    parser.add_argument("--timezone", default=DEFAULT_TIMEZONE)
    parser.add_argument("--timeframes", default=",".join(DEFAULT_TIMEFRAMES))
    parser.add_argument("--trend-map-dir", default=str(DEFAULT_TREND_MAP_DIR))
    parser.add_argument("--funding-dir", default=str(DEFAULT_FUNDING_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--database-path", default=str(DEFAULT_DATABASE_PATH))
    parser.add_argument("--fee-bps-per-side", type=float, default=5.0)
    parser.add_argument("--slippage-bps-per-side", type=float, default=5.0)
    parser.add_argument("--data-check-strict", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def resolve_path(path_arg: str | Path) -> Path:
    """Resolve relative paths from the project root."""

    path = Path(path_arg)
    return path if path.is_absolute() else PROJECT_ROOT / path


def parse_csv_list(value: str | Iterable[str]) -> list[str]:
    """Parse comma/space-separated text while preserving order."""

    tokens = re.split(r"[\s,]+", value) if isinstance(value, str) else [str(item) for item in value]
    parsed: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        item = token.strip()
        if item and item not in seen:
            parsed.append(item)
            seen.add(item)
    return parsed


def parse_timeframes(value: str | Iterable[str]) -> list[str]:
    """Parse and validate timeframes."""

    timeframes = parse_csv_list(value)
    unsupported = [item for item in timeframes if item not in TIMEFRAME_MINUTES]
    if unsupported:
        raise EtcResearchError(f"unsupported timeframes: {unsupported}")
    if not timeframes:
        raise EtcResearchError("--timeframes must not be empty")
    return timeframes


def resolve_history_range(start: str, end: str, timezone_name: str) -> HistoryRange:
    """Resolve the research date range."""

    try:
        return parse_history_range(start, end, timedelta(minutes=1), timezone_name)
    except ValueError as exc:
        raise EtcResearchError(str(exc)) from exc


def split_vt_symbol(vt_symbol: str) -> tuple[str, str]:
    """Split a local vt_symbol into database symbol and exchange."""

    symbol, separator, exchange = str(vt_symbol).partition(".")
    if not separator or not symbol or not exchange:
        raise EtcResearchError(f"invalid vt_symbol: {vt_symbol}")
    return symbol, exchange


def symbol_to_inst_id(vt_symbol: str) -> str:
    """Map a local vt_symbol to OKX instId."""

    symbol, _exchange = split_vt_symbol(vt_symbol)
    root = symbol.removesuffix("_OKX")
    pair = root[: -len("_SWAP")] if root.endswith("_SWAP") else root
    if pair.endswith("USDT"):
        return f"{pair[:-4]}-USDT-SWAP"
    return root.replace("_", "-")


def safe_symbol(value: str) -> str:
    """Return an identifier-safe symbol fragment."""

    return re.sub(r"[^A-Za-z0-9]+", "_", str(value)).strip("_")


def finite_float(value: Any, default: float = np.nan) -> float:
    """Return a finite float or default."""

    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if not np.isfinite(number):
        return default
    return number


def format_timestamp(value: Any, timezone_name: str | None = None) -> str | None:
    """Format timestamp-like values as ISO strings."""

    if value is None or pd.isna(value):
        return None
    timestamp = pd.Timestamp(value)
    if timezone_name:
        if timestamp.tzinfo is None:
            timestamp = timestamp.tz_localize(timezone_name)
        else:
            timestamp = timestamp.tz_convert(timezone_name)
    return timestamp.isoformat()


def write_json(path: Path, payload: Any) -> None:
    """Write a JSON file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(payload), ensure_ascii=False, indent=2), encoding="utf-8")


def write_dataframe(path: Path, frame: pd.DataFrame, columns: list[str] | None = None) -> None:
    """Write a dataframe with stable columns for empty outputs."""

    path.parent.mkdir(parents=True, exist_ok=True)
    output = frame.copy()
    if columns is not None:
        for column in columns:
            if column not in output.columns:
                output[column] = np.nan
        output = output.loc[:, columns]
    output.to_csv(path, index=False)


def dataframe_records(frame: pd.DataFrame, limit: int | None = None) -> list[dict[str, Any]]:
    """Convert a dataframe to JSON-safe records."""

    if frame.empty:
        return []
    work = frame.head(limit).copy() if limit is not None else frame.copy()
    work = work.replace({np.nan: None})
    return json.loads(work.to_json(orient="records", force_ascii=False, date_format="iso"))


def format_number(value: Any, digits: int = 6) -> str:
    """Format optional numeric values for Markdown."""

    try:
        number = float(value)
    except (TypeError, ValueError):
        return "N/A"
    if not np.isfinite(number):
        return "N/A"
    return f"{number:.{digits}f}"


def markdown_table(rows: list[dict[str, Any]], columns: list[str], limit: int = 50) -> str:
    """Render a compact Markdown table."""

    if not rows:
        return "- N/A"
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join("---" for _ in columns) + "|"]
    for row in rows[:limit]:
        values: list[str] = []
        for column in columns:
            value = row.get(column)
            values.append(format_number(value, 6) if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def read_json_if_exists(path: Path, warnings: list[str]) -> dict[str, Any]:
    """Read JSON if present, recording warnings instead of crashing."""

    if not path.exists():
        warnings.append(f"missing_json:{path}")
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        warnings.append(f"json_read_failed:{path}:{exc!r}")
        return {}


def read_csv_if_exists(path: Path, warnings: list[str]) -> pd.DataFrame:
    """Read CSV if present, recording warnings instead of crashing."""

    if not path.exists():
        warnings.append(f"missing_csv:{path}")
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception as exc:
        warnings.append(f"csv_read_failed:{path}:{exc!r}")
        return pd.DataFrame()


def normalize_1m_bars(frame: pd.DataFrame, timezone_name: str) -> pd.DataFrame:
    """Normalize injected or sqlite-loaded 1m OHLCV into report timezone."""

    columns = ["datetime", "open", "high", "low", "close", "volume"]
    if frame.empty:
        return pd.DataFrame(columns=columns)
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise EtcResearchError(f"1m bars missing columns: {missing}")
    out = frame.loc[:, columns].copy()
    timestamps = pd.to_datetime(out["datetime"], errors="coerce")
    if timestamps.isna().any():
        raise EtcResearchError("1m bars contain unparsable datetime values")
    if timestamps.dt.tz is None:
        timestamps = timestamps.dt.tz_localize(timezone_name)
    else:
        timestamps = timestamps.dt.tz_convert(timezone_name)
    out["datetime"] = timestamps
    for column in ["open", "high", "low", "close", "volume"]:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    out = out.dropna(subset=columns).sort_values("datetime", kind="stable").drop_duplicates("datetime", keep="last")
    return out.reset_index(drop=True)


def load_1m_bars_from_sqlite(vt_symbol: str, history_range: HistoryRange, database_path: Path) -> pd.DataFrame:
    """Load local vn.py sqlite 1m bars for one symbol."""

    if not database_path.exists():
        raise EtcResearchError(f"database not found: {database_path}")
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


def coverage_missing_ranges_from_actual(
    actual_times: list[pd.Timestamp],
    history_range: HistoryRange,
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
            gap_count += 1
            if len(ranges) < limit:
                ranges.append(
                    {
                        "start": cursor.isoformat(),
                        "end": (current - interval).isoformat(),
                        "missing_count": int((current - cursor) / interval),
                    }
                )
        cursor = current + interval
    if cursor < end_exclusive:
        gap_count += 1
        if len(ranges) < limit:
            ranges.append(
                {
                    "start": cursor.isoformat(),
                    "end": (end_exclusive - interval).isoformat(),
                    "missing_count": int((end_exclusive - cursor) / interval),
                }
            )
    return ranges, gap_count


def analyze_1m_quality(vt_symbol: str, bars_1m: pd.DataFrame, history_range: HistoryRange) -> dict[str, Any]:
    """Build strict 1m coverage quality summary for one symbol."""

    if bars_1m.empty:
        actual_times: list[pd.Timestamp] = []
    else:
        start = pd.Timestamp(history_range.start)
        end = pd.Timestamp(history_range.end_exclusive)
        target = bars_1m[(bars_1m["datetime"] >= start) & (bars_1m["datetime"] < end)]
        actual_times = [
            pd.Timestamp(value)
            for value in pd.to_datetime(target["datetime"]).dropna().drop_duplicates().sort_values(kind="stable")
        ]
    expected = int(expected_bar_count(history_range))
    total = int(len(actual_times))
    missing = max(0, expected - total)
    missing_ranges, gap_count = coverage_missing_ranges_from_actual(actual_times, history_range)
    if missing and not missing_ranges:
        missing_ranges = [
            {
                "start": history_range.start.isoformat(),
                "end": history_range.end_display.isoformat(),
                "missing_count": missing,
            }
        ]
        gap_count = 1
    return {
        "symbol": vt_symbol,
        "timeframe": "1m",
        "expected_count": expected,
        "row_count": int(total),
        "unique_count": total,
        "missing_count": int(missing),
        "gap_count": int(gap_count),
        "first_datetime": actual_times[0].isoformat() if actual_times else None,
        "last_datetime": actual_times[-1].isoformat() if actual_times else None,
        "complete": bool(expected > 0 and missing == 0 and total == expected),
        "missing_ranges_sample": missing_ranges,
    }


def resample_ohlcv_closed(
    bars_1m: pd.DataFrame,
    timeframe: str,
    history_range: HistoryRange | None = None,
) -> pd.DataFrame:
    """Resample 1m OHLCV into completed closed bars without lookahead."""

    if timeframe not in TIMEFRAME_MINUTES:
        raise EtcResearchError(f"unsupported timeframe: {timeframe}")
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


def analyze_timeframe_quality(
    vt_symbol: str,
    timeframe: str,
    bars: pd.DataFrame,
    one_minute_quality: dict[str, Any],
) -> dict[str, Any]:
    """Build quality metadata for one resampled timeframe."""

    minutes = TIMEFRAME_MINUTES[timeframe]
    expected_count = int(one_minute_quality["expected_count"] // minutes)
    return {
        "symbol": vt_symbol,
        "timeframe": timeframe,
        "minutes": minutes,
        "expected_closed_bar_count": expected_count,
        "row_count": int(len(bars.index)),
        "first_datetime": format_timestamp(bars["datetime"].min()) if not bars.empty else None,
        "last_datetime": format_timestamp(bars["datetime"].max()) if not bars.empty else None,
        "complete": bool(one_minute_quality["complete"] and len(bars.index) == expected_count),
    }


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
        raise EtcResearchError(f"funding CSV missing funding_time columns: {path}")
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
    """Select the canonical funding CSV or the latest matching fallback."""

    canonical = funding_dir / f"{inst_id}_funding_{start_arg}_{end_arg}.csv"
    if canonical.exists():
        return canonical
    matches = sorted(funding_dir.glob(f"{inst_id}_funding_*.csv"))
    return matches[-1] if matches else None


def analyze_funding_quality(funding_df: pd.DataFrame, data_range: HistoryRange, path: Path | None) -> dict[str, Any]:
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
    warnings: list[str] | None = None,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    """Load actual OKX funding histories and quality metadata."""

    warnings = warnings if warnings is not None else []
    histories: dict[str, pd.DataFrame] = {}
    records: dict[str, Any] = {}
    for symbol in symbols:
        inst_id = symbol_to_inst_id(symbol)
        path = select_funding_csv(funding_dir, inst_id, start_arg, end_arg)
        if path is None:
            histories[inst_id] = pd.DataFrame(columns=["funding_time_utc", "funding_rate"])
            records[inst_id] = analyze_funding_quality(histories[inst_id], data_range, None)
            warnings.append(f"funding_unavailable:{inst_id}")
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
                "warnings": [f"funding_csv_read_failed:{exc!r}"],
            }
            warnings.append(f"funding_read_failed:{inst_id}:{exc!r}")
            continue
        records[inst_id] = analyze_funding_quality(histories[inst_id], data_range, path)
        if not records[inst_id]["coverage_complete"]:
            warnings.append(f"funding_incomplete:{inst_id}:{records[inst_id].get('warnings')}")
    complete = bool(records and all(record["coverage_complete"] for record in records.values()))
    return histories, {
        "funding_data_complete": complete,
        "records": records,
        "missing_inst_ids": [inst_id for inst_id, record in records.items() if not record.get("coverage_complete")],
    }


def signed_funding_pnl(notional: float, funding_rate: float, direction: str) -> float:
    """Return signed funding PnL using perpetual funding convention."""

    if direction == "short":
        return float(abs(notional) * funding_rate)
    return float(-abs(notional) * funding_rate)


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


def true_range(frame: pd.DataFrame) -> pd.Series:
    """Compute true range."""

    high = pd.to_numeric(frame["high"], errors="coerce")
    low = pd.to_numeric(frame["low"], errors="coerce")
    close = pd.to_numeric(frame["close"], errors="coerce")
    previous_close = close.shift(1)
    ranges = pd.concat([high - low, (high - previous_close).abs(), (low - previous_close).abs()], axis=1)
    return ranges.max(axis=1)


def rolling_percentile(series: pd.Series, window: int, min_periods: int | None = None) -> pd.Series:
    """Return each value percentile inside its closed rolling history."""

    if min_periods is None:
        min_periods = min(max(5, window // 4), window)

    def percentile_last(values: np.ndarray) -> float:
        current = values[-1]
        if not np.isfinite(current):
            return np.nan
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            return np.nan
        return float((finite <= current).sum() / finite.size)

    return pd.to_numeric(series, errors="coerce").rolling(window, min_periods=min_periods).apply(percentile_last, raw=True)


def bars_for_days(timeframe: str, days: int | float) -> int:
    """Convert calendar days into bars for a timeframe."""

    return max(1, int(round(float(days) * 1440.0 / TIMEFRAME_MINUTES[timeframe])))


def add_symbol_features(frame: pd.DataFrame, symbol: str, timeframe: str) -> pd.DataFrame:
    """Build single-symbol closed-bar features without future data."""

    if frame.empty:
        columns = ["timestamp", "symbol", "timeframe", "bar_index", *REQUIRED_FEATURE_COLUMNS, *DERIVED_SCORE_COLUMNS]
        return pd.DataFrame(columns=columns)
    out = frame.copy().reset_index(drop=True)
    out["timestamp"] = pd.to_datetime(out["datetime"])
    out["symbol"] = symbol
    out["inst_id"] = symbol_to_inst_id(symbol)
    out["timeframe"] = timeframe
    out["bar_index"] = np.arange(len(out.index))
    close = pd.to_numeric(out["close"], errors="coerce")
    high = pd.to_numeric(out["high"], errors="coerce")
    low = pd.to_numeric(out["low"], errors="coerce")
    volume = pd.to_numeric(out["volume"], errors="coerce")
    returns = close.pct_change()
    out["return_1"] = returns
    tr = true_range(out)
    out["atr_20"] = tr.rolling(20, min_periods=20).mean()
    out["atr_55"] = tr.rolling(55, min_periods=55).mean()
    out["atr_pct_20"] = out["atr_20"] / close.replace(0.0, np.nan)
    out["realized_vol_20"] = returns.rolling(20, min_periods=20).std()
    for window in [20, 55, 100]:
        path = close.diff().abs().rolling(window, min_periods=window).sum()
        raw_return = close / close.shift(window) - 1.0
        out[f"return_{window}"] = raw_return
        out[f"trend_efficiency_{window}"] = (close - close.shift(window)).abs() / path.replace(0.0, np.nan)
        out[f"directional_efficiency_ratio_{window}"] = (close - close.shift(window)) / path.replace(0.0, np.nan)
    out["trend_efficiency_change_20"] = out["trend_efficiency_20"] - out["trend_efficiency_20"].shift(20)
    out["trend_efficiency_change_55"] = out["trend_efficiency_55"] - out["trend_efficiency_55"].shift(55)
    out["atr_percentile_100"] = rolling_percentile(out["atr_20"], 100)
    out["atr_percentile_200"] = rolling_percentile(out["atr_20"], 200)
    out["realized_vol_percentile_100"] = rolling_percentile(out["realized_vol_20"], 100)
    out["volatility_expansion_rate_20"] = out["atr_20"] / out["atr_20"].shift(20).replace(0.0, np.nan) - 1.0
    range_width = (high.rolling(20, min_periods=20).max() - low.rolling(20, min_periods=20).min()) / close.replace(0.0, np.nan)
    out["range_width_percentile_100"] = rolling_percentile(range_width, 100)
    out["efficiency_to_volatility_ratio_20"] = out["trend_efficiency_20"] / out["realized_vol_20"].replace(0.0, np.nan)
    out["efficiency_to_volatility_ratio_55"] = out["trend_efficiency_55"] / returns.rolling(55, min_periods=55).std().replace(0.0, np.nan)
    for days in [20, 60]:
        bars = bars_for_days(timeframe, days)
        rolling_high = high.rolling(bars, min_periods=min(bars, max(5, bars // 4))).max()
        rolling_low = low.rolling(bars, min_periods=min(bars, max(5, bars // 4))).min()
        out[f"distance_from_{days}d_high"] = close / rolling_high.replace(0.0, np.nan) - 1.0
        out[f"rebound_from_recent_low_{days}"] = close / rolling_low.replace(0.0, np.nan) - 1.0
        if days == 60:
            out["drawdown_from_60d_high"] = out[f"distance_from_{days}d_high"]
            out["price_position_in_60d_range"] = (close - rolling_low) / (rolling_high - rolling_low).replace(0.0, np.nan)
    volume_mean20_prev = volume.rolling(20, min_periods=20).mean().shift(1)
    out["volume_ratio_20"] = volume / volume_mean20_prev.replace(0.0, np.nan)
    out["volume_ratio_median_5"] = out["volume_ratio_20"].rolling(5, min_periods=5).median()
    out["volume_ratio_slope_5"] = out["volume_ratio_median_5"] - out["volume_ratio_median_5"].shift(5)
    volume_cv20 = volume.rolling(20, min_periods=20).std() / volume.rolling(20, min_periods=20).mean().replace(0.0, np.nan)
    out["volume_stability_20"] = 1.0 / (1.0 + volume_cv20)
    price_exhaustion = returns.abs() / out["atr_pct_20"].replace(0.0, np.nan)
    out["volume_without_price_exhaustion_score"] = (
        out["volume_ratio_median_5"] * out["volume_stability_20"] / (1.0 + price_exhaustion.clip(lower=0.0))
    )
    out["volatility_noise_score"] = out["realized_vol_percentile_100"] - out["trend_efficiency_20"]
    return out


def merge_asof_column(base: pd.DataFrame, right: pd.DataFrame, value_column: str) -> pd.Series:
    """Merge one market-wide as-of column by timestamp and restore row order."""

    if right.empty or value_column not in right.columns:
        return pd.Series(np.nan, index=base.index)
    left = base.loc[:, ["timestamp"]].copy()
    left["_row_order"] = np.arange(len(left.index))
    left = left.sort_values("timestamp", kind="stable")
    right_work = right.loc[:, ["timestamp", value_column]].copy().sort_values("timestamp", kind="stable")
    merged = pd.merge_asof(left, right_work, on="timestamp", direction="backward")
    merged = merged.sort_values("_row_order", kind="stable")
    return merged[value_column].reset_index(drop=True)


def add_cross_symbol_features(dataset: pd.DataFrame, symbols: list[str], timeframes: list[str]) -> pd.DataFrame:
    """Add breadth and relative-strength features across the universe."""

    if dataset.empty:
        return dataset.copy()
    out = dataset.copy().sort_values(["timeframe", "timestamp", "symbol"], kind="stable").reset_index(drop=True)
    market_count_frames: dict[str, pd.DataFrame] = {}
    for timeframe in timeframes:
        tf = out[out["timeframe"] == timeframe].copy()
        if tf.empty:
            continue
        for window in [20, 55]:
            tf[f"above_midline_{window}"] = tf["close"] > (
                tf.groupby("symbol")["high"].transform(lambda s: s.rolling(window, min_periods=window).max())
                + tf.groupby("symbol")["low"].transform(lambda s: s.rolling(window, min_periods=window).min())
            ) / 2.0
        positive_counts = (
            tf.assign(positive=tf["return_1"] > 0.0)
            .groupby("timestamp", sort=True)["positive"]
            .sum()
            .reset_index(name=f"positive_return_symbol_count_{timeframe}")
        )
        market_count_frames[timeframe] = positive_counts
        for window in [20, 55]:
            count_by_time = (
                tf.groupby("timestamp", sort=True)[f"above_midline_{window}"]
                .sum()
                .reset_index(name=f"symbols_above_rolling_midline_{window}")
            )
            out.loc[out["timeframe"] == timeframe, f"symbols_above_rolling_midline_{window}"] = merge_asof_column(
                out[out["timeframe"] == timeframe],
                count_by_time,
                f"symbols_above_rolling_midline_{window}",
            ).to_numpy()
        for window in [20, 55]:
            breadth = (
                tf.assign(positive_window=tf[f"return_{window}"] > 0.0)
                .groupby("timestamp", sort=True)["positive_window"]
                .sum()
                .reset_index(name=f"market_return_breadth_{window}")
            )
            breadth[f"market_return_breadth_{window}"] = breadth[f"market_return_breadth_{window}"] / max(1, len(symbols))
            out.loc[out["timeframe"] == timeframe, f"market_return_breadth_{window}"] = merge_asof_column(
                out[out["timeframe"] == timeframe],
                breadth,
                f"market_return_breadth_{window}",
            ).to_numpy()
        count20 = (
            tf.groupby("timestamp", sort=True)["above_midline_20"]
            .sum()
            .reset_index(name="breadth_level")
            .sort_values("timestamp", kind="stable")
        )
        bars_day = bars_for_days(timeframe, 1)
        count20["breadth_change_1d"] = count20["breadth_level"].diff(bars_day)
        count20["breadth_change_3d"] = count20["breadth_level"].diff(bars_day * 3)
        count20["breadth_acceleration"] = count20["breadth_change_1d"] - count20["breadth_change_1d"].shift(bars_day)
        for column in ["breadth_change_1d", "breadth_change_3d", "breadth_acceleration"]:
            out.loc[out["timeframe"] == timeframe, column] = merge_asof_column(
                out[out["timeframe"] == timeframe],
                count20[["timestamp", column]],
                column,
            ).to_numpy()
        tf_indexes = out["timeframe"] == timeframe
        for window in [20, 55, 100]:
            out.loc[tf_indexes, f"return_rank_{window}"] = out.loc[tf_indexes].groupby("timestamp")[f"return_{window}"].rank(
                pct=True,
                method="average",
            )
        for window in [20, 55]:
            vol_adjusted = out.loc[tf_indexes, f"return_{window}"] / out.loc[tf_indexes, "realized_vol_20"].replace(0.0, np.nan)
            out.loc[tf_indexes, f"vol_adjusted_return_{window}"] = vol_adjusted
            out.loc[tf_indexes, f"vol_adjusted_return_rank_{window}"] = vol_adjusted.groupby(out.loc[tf_indexes, "timestamp"]).rank(
                pct=True,
                method="average",
            )
            out.loc[tf_indexes, f"trend_efficiency_rank_{window}"] = out.loc[tf_indexes].groupby("timestamp")[
                f"trend_efficiency_{window}"
            ].rank(pct=True, method="average")
    for timeframe, counts in market_count_frames.items():
        for target_tf in timeframes:
            mask = out["timeframe"] == target_tf
            out.loc[mask, f"positive_return_symbol_count_{timeframe}"] = merge_asof_column(
                out.loc[mask],
                counts,
                f"positive_return_symbol_count_{timeframe}",
            ).to_numpy()
    out = out.sort_values(["symbol", "timeframe", "timestamp"], kind="stable").reset_index(drop=True)
    out["rank_change_5"] = out.groupby(["symbol", "timeframe"])["return_rank_20"].diff(5)
    out["rank_change_20"] = out.groupby(["symbol", "timeframe"])["return_rank_20"].diff(20)
    out["rank_persistence_20"] = (
        out.groupby(["symbol", "timeframe"])["return_rank_20"]
        .rolling(20, min_periods=10)
        .mean()
        .reset_index(level=[0, 1], drop=True)
    )
    return out.sort_values(["timeframe", "timestamp", "symbol"], kind="stable").reset_index(drop=True)


def align_funding_features(
    dataset: pd.DataFrame,
    funding_histories: dict[str, pd.DataFrame],
    symbols: list[str],
    timeframes: list[str],
    funding_data_complete: bool,
    warnings: list[str] | None = None,
) -> pd.DataFrame:
    """Add funding features from actual OKX funding CSVs."""

    warnings = warnings if warnings is not None else []
    if dataset.empty:
        return dataset.copy()
    out = dataset.copy()
    for column in [
        "funding_rate",
        "funding_percentile_100",
        "funding_percentile_200",
        "funding_sign",
        "funding_change_7d",
        "funding_change_30d",
        "funding_dispersion",
        "positive_funding_symbol_count",
        "negative_funding_symbol_count",
        "extreme_funding_count",
        "funding_overheat_score",
    ]:
        if column not in out.columns:
            out[column] = np.nan
    if not funding_data_complete:
        warnings.append("funding_features_unavailable_or_partial")
    aligned_frames: list[pd.DataFrame] = []
    for (symbol, timeframe), frame in out.groupby(["symbol", "timeframe"], sort=False):
        work = frame.copy().sort_values("timestamp", kind="stable")
        inst_id = symbol_to_inst_id(str(symbol))
        funding = funding_histories.get(inst_id, pd.DataFrame(columns=["funding_time_utc", "funding_rate"]))
        if funding.empty:
            aligned_frames.append(work)
            continue
        left = work.loc[:, ["timestamp"]].copy()
        left["_timestamp_utc"] = pd.to_datetime(left["timestamp"], utc=True)
        left["_row_order"] = np.arange(len(left.index))
        right = funding.loc[:, ["funding_time_utc", "funding_rate"]].copy().sort_values("funding_time_utc", kind="stable")
        merged = pd.merge_asof(
            left.sort_values("_timestamp_utc", kind="stable"),
            right,
            left_on="_timestamp_utc",
            right_on="funding_time_utc",
            direction="backward",
        ).sort_values("_row_order", kind="stable")
        work["funding_rate"] = pd.to_numeric(merged["funding_rate"], errors="coerce").to_numpy()
        work["funding_percentile_100"] = rolling_percentile(work["funding_rate"], 100)
        work["funding_percentile_200"] = rolling_percentile(work["funding_rate"], 200)
        work["funding_sign"] = np.sign(work["funding_rate"])
        work["funding_change_7d"] = work["funding_rate"] - work["funding_rate"].shift(bars_for_days(str(timeframe), 7))
        work["funding_change_30d"] = work["funding_rate"] - work["funding_rate"].shift(bars_for_days(str(timeframe), 30))
        work["funding_overheat_score"] = (work["funding_percentile_200"] - 0.5).abs() * 2.0
        aligned_frames.append(work)
    out = pd.concat(aligned_frames, ignore_index=True) if aligned_frames else out
    for timeframe in timeframes:
        mask = out["timeframe"] == timeframe
        tf = out.loc[mask].copy()
        if tf.empty:
            continue
        funding_stats = tf.groupby("timestamp", sort=True).agg(
            funding_dispersion=("funding_rate", "std"),
            positive_funding_symbol_count=("funding_rate", lambda s: int((pd.to_numeric(s, errors="coerce") > 0).sum())),
            negative_funding_symbol_count=("funding_rate", lambda s: int((pd.to_numeric(s, errors="coerce") < 0).sum())),
            extreme_funding_count=("funding_overheat_score", lambda s: int((pd.to_numeric(s, errors="coerce") >= 0.9).sum())),
        ).reset_index()
        for column in [
            "funding_dispersion",
            "positive_funding_symbol_count",
            "negative_funding_symbol_count",
            "extreme_funding_count",
        ]:
            out.loc[mask, column] = merge_asof_column(tf, funding_stats[["timestamp", column]], column).to_numpy()
    return out.sort_values(["timeframe", "timestamp", "symbol"], kind="stable").reset_index(drop=True)


def infer_atr_move(labels: Any) -> float | None:
    """Infer the largest ATR label from Trend Opportunity Map text."""

    if labels is None or pd.isna(labels):
        return None
    values = [float(match) for match in re.findall(r"([0-9]+(?:\.[0-9]+)?)atr", str(labels).lower())]
    return max(values) if values else None


def infer_efficiency(labels: Any) -> float | None:
    """Infer the largest efficiency threshold from Trend Opportunity Map text."""

    if labels is None or pd.isna(labels):
        return None
    values = [float(match) for match in re.findall(r"trend_efficiency_[0-9]+_ge_([0-9]+(?:\.[0-9]+)?)", str(labels).lower())]
    return max(values) if values else None


def normalize_trend_segments(segments: pd.DataFrame, timezone_name: str) -> pd.DataFrame:
    """Normalize Trend Opportunity Map segments for ETC labels."""

    if segments.empty:
        return pd.DataFrame(columns=list(segments.columns) + ["segment_start_ts", "segment_end_ts"])
    out = segments.copy()
    for column in ["start_time", "end_time"]:
        if column not in out.columns:
            out[column] = pd.NaT
    out["segment_start_ts"] = pd.to_datetime(out["start_time"], errors="coerce", utc=True).dt.tz_convert(timezone_name)
    out["segment_end_ts"] = pd.to_datetime(out["end_time"], errors="coerce", utc=True).dt.tz_convert(timezone_name)
    out["segment_atr_move"] = out.get("atr_labels", pd.Series(index=out.index, dtype=object)).map(infer_atr_move)
    out["segment_efficiency"] = out.get("efficiency_labels", pd.Series(index=out.index, dtype=object)).map(infer_efficiency)
    out["trend_return"] = pd.to_numeric(out.get("trend_return"), errors="coerce")
    out["duration_bars"] = pd.to_numeric(out.get("duration_bars"), errors="coerce")
    return out.dropna(subset=["segment_start_ts", "segment_end_ts"]).reset_index(drop=True)


def segment_is_effective_2_or_3atr(segment: pd.Series) -> bool:
    """Return True when a segment is a valid 2ATR/3ATR opportunity."""

    labels = str(segment.get("atr_labels") or segment.get("labels") or "").lower()
    return "2atr" in labels or "3atr" in labels


def build_early_trend_labels_for_frame(
    frame: pd.DataFrame,
    symbol: str,
    timeframe: str,
    trend_segments: pd.DataFrame,
    future_window_bars: int = FUTURE_NON_TREND_WINDOW_BARS,
    boundary_buffer_bars: int = 1,
) -> pd.DataFrame:
    """Build ex-post ETC labels for one symbol/timeframe."""

    columns = LABEL_COLUMNS + ["bar_index"]
    if frame.empty:
        return pd.DataFrame(columns=columns)
    rows = frame.loc[:, ["datetime"]].copy().reset_index(drop=True)
    rows["timestamp"] = pd.to_datetime(rows["datetime"])
    rows["symbol"] = symbol
    rows["timeframe"] = timeframe
    rows["bar_index"] = np.arange(len(rows.index))
    for column in LABEL_COLUMNS:
        if column not in rows.columns:
            rows[column] = np.nan
    rows["label"] = "nontrend"
    rows["trend_segment_id"] = None
    rows["trend_direction"] = None
    rows["trend_phase_pct"] = np.nan
    rows["segment_start"] = None
    rows["segment_end"] = None
    rows["segment_return"] = np.nan
    rows["segment_atr_move"] = np.nan
    rows["segment_efficiency"] = np.nan
    rows["_match_count"] = 0

    segments = trend_segments[
        (trend_segments.get("symbol", "") == symbol) & (trend_segments.get("timeframe", "") == timeframe)
    ].copy()
    effective_starts: list[int] = []
    boundary_positions: set[int] = set()
    timestamps = pd.DatetimeIndex(rows["timestamp"])
    for _, segment in segments.iterrows():
        start = pd.Timestamp(segment["segment_start_ts"])
        end = pd.Timestamp(segment["segment_end_ts"])
        start_pos = int(timestamps.searchsorted(start, side="left"))
        end_pos = int(timestamps.searchsorted(end, side="right")) - 1
        if start_pos < len(rows.index):
            if segment_is_effective_2_or_3atr(segment):
                effective_starts.append(start_pos)
            for offset in range(-boundary_buffer_bars, boundary_buffer_bars + 1):
                if 0 <= start_pos + offset < len(rows.index):
                    boundary_positions.add(start_pos + offset)
                if 0 <= end_pos + offset < len(rows.index):
                    boundary_positions.add(end_pos + offset)
        mask = (rows["timestamp"] >= start) & (rows["timestamp"] <= end)
        if not bool(mask.any()):
            continue
        match_index = rows.index[mask]
        rows.loc[match_index, "_match_count"] = rows.loc[match_index, "_match_count"] + 1
        single_match = match_index[rows.loc[match_index, "_match_count"] == 1]
        if len(single_match) == 0:
            continue
        duration_seconds = max((end - start).total_seconds(), 1.0)
        phase = (rows.loc[single_match, "timestamp"].map(lambda value: (pd.Timestamp(value) - start).total_seconds()) / duration_seconds).clip(
            lower=0.0,
            upper=1.0,
        )
        direction = str(segment.get("direction") or "").lower()
        early_label = "early_uptrend" if direction == "up" else "early_downtrend" if direction == "down" else "excluded_ambiguous"
        labels = np.where(phase < 0.25, early_label, np.where(phase < 0.75, "middle_trend", "late_trend"))
        rows.loc[single_match, "label"] = labels
        rows.loc[single_match, "trend_segment_id"] = segment.get("trend_segment_id")
        rows.loc[single_match, "trend_direction"] = direction
        rows.loc[single_match, "trend_phase_pct"] = phase.to_numpy(dtype=float)
        rows.loc[single_match, "segment_start"] = format_timestamp(start)
        rows.loc[single_match, "segment_end"] = format_timestamp(end)
        rows.loc[single_match, "segment_return"] = finite_float(segment.get("trend_return"))
        rows.loc[single_match, "segment_atr_move"] = finite_float(segment.get("segment_atr_move"))
        rows.loc[single_match, "segment_efficiency"] = finite_float(segment.get("segment_efficiency"))

    overlap = rows["_match_count"] > 1
    rows.loc[overlap, "label"] = "excluded_ambiguous"
    rows.loc[overlap, "trend_segment_id"] = None
    rows.loc[overlap, "trend_direction"] = None
    rows.loc[overlap, "trend_phase_pct"] = np.nan

    effective_starts = sorted(set(effective_starts))
    for index in rows.index[rows["_match_count"] == 0]:
        bar_index = int(rows.at[index, "bar_index"])
        future_is_near_effective_trend = any(bar_index < start <= bar_index + future_window_bars for start in effective_starts)
        insufficient_future = bar_index + future_window_bars >= len(rows.index)
        near_boundary = bar_index in boundary_positions
        if future_is_near_effective_trend or insufficient_future or near_boundary:
            rows.at[index, "label"] = "excluded_ambiguous"
    return rows.loc[:, columns].copy()


def build_time_splits(timezone_name: str) -> list[TimeSplit]:
    """Build fixed extended splits."""

    tz = resolve_timezone(timezone_name)
    return [
        TimeSplit("train_ext", pd.Timestamp("2023-01-01", tz=tz), pd.Timestamp("2024-07-01", tz=tz)),
        TimeSplit("validation_ext", pd.Timestamp("2024-07-01", tz=tz), pd.Timestamp("2025-07-01", tz=tz)),
        TimeSplit("oos_ext", pd.Timestamp("2025-07-01", tz=tz), pd.Timestamp("2026-04-01", tz=tz)),
    ]


def assign_split_for_time(value: Any, splits: list[TimeSplit]) -> str:
    """Assign one timestamp to a fixed split."""

    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize(splits[0].start.tz)
    else:
        timestamp = timestamp.tz_convert(splits[0].start.tz)
    for split in splits:
        if split.start <= timestamp < split.end:
            return split.name
    if timestamp < splits[0].start:
        return splits[0].name
    return splits[-1].name


def direction_from_proxy(row: pd.Series) -> str:
    """Infer event direction from only pre-entry closed-bar features."""

    signal = 0.0
    for column, weight in [
        ("directional_efficiency_ratio_20", 1.0),
        ("return_20", 1.0),
        ("rank_change_20", 0.5),
    ]:
        value = finite_float(row.get(column), 0.0)
        signal += weight * value
    return "long" if signal >= 0.0 else "short"


def add_labels_to_dataset(features: pd.DataFrame, labels: pd.DataFrame, splits: list[TimeSplit]) -> pd.DataFrame:
    """Join labels and split metadata into the feature dataset."""

    if features.empty:
        return features.copy()
    out = features.copy()
    if not labels.empty:
        label_cols = [column for column in LABEL_COLUMNS if column in labels.columns]
        out = out.merge(
            labels.loc[:, label_cols],
            on=["timestamp", "symbol", "timeframe"],
            how="left",
            suffixes=("", "_label"),
        )
    if "label" not in out.columns:
        out["label"] = "excluded_ambiguous"
    out["label"] = out["label"].fillna("excluded_ambiguous")
    out["split"] = out["timestamp"].map(lambda value: assign_split_for_time(value, splits))
    out["direction_proxy"] = out.apply(direction_from_proxy, axis=1)
    out["early_trend"] = out["label"].isin(["early_uptrend", "early_downtrend"])
    out["nontrend"] = out["label"].eq("nontrend")
    out["direction_match"] = np.where(
        out["label"].eq("early_uptrend"),
        out["direction_proxy"].eq("long"),
        np.where(out["label"].eq("early_downtrend"), out["direction_proxy"].eq("short"), np.nan),
    )
    return out


def directional_return(entry_price: float, exit_price: float, direction: str) -> float:
    """Return signed directional return."""

    if not np.isfinite(entry_price) or not np.isfinite(exit_price) or entry_price == 0:
        return np.nan
    sign = 1.0 if direction == "long" else -1.0
    return float(sign * (exit_price / entry_price - 1.0))


def add_forward_outcomes(dataset: pd.DataFrame, bars_by_key: dict[tuple[str, str], pd.DataFrame]) -> pd.DataFrame:
    """Add future outcome columns for research evaluation only."""

    if dataset.empty:
        return dataset.copy()
    out = dataset.copy()
    for column in ["forward_return_4h", "forward_return_1d", "forward_return_3d", "future_2atr_hit", "future_3atr_hit"]:
        out[column] = np.nan
    for index, row in out.iterrows():
        key = (str(row["symbol"]), str(row["timeframe"]))
        bars = bars_by_key.get(key)
        if bars is None or bars.empty:
            continue
        pos = int(row["bar_index"])
        if pos >= len(bars.index):
            continue
        close = finite_float(row.get("close"))
        direction = str(row.get("direction_proxy") or "long")
        for name, minutes in [("forward_return_4h", 240), ("forward_return_1d", 1440), ("forward_return_3d", 4320)]:
            horizon = max(1, int(math.ceil(minutes / TIMEFRAME_MINUTES[str(row["timeframe"])])))
            future_pos = pos + horizon
            if future_pos < len(bars.index):
                out.at[index, name] = directional_return(close, float(bars.iloc[future_pos]["close"]), direction)
        horizon_3d = max(1, int(math.ceil(4320 / TIMEFRAME_MINUTES[str(row["timeframe"])])))
        end_pos = min(len(bars.index) - 1, pos + horizon_3d)
        if end_pos <= pos:
            continue
        atr = finite_float(row.get("atr_20"))
        if not np.isfinite(atr) or atr <= 0 or not np.isfinite(close):
            continue
        future = bars.iloc[pos + 1 : end_pos + 1]
        if direction == "long":
            move = float(pd.to_numeric(future["high"], errors="coerce").max() - close)
        else:
            move = float(close - pd.to_numeric(future["low"], errors="coerce").min())
        out.at[index, "future_2atr_hit"] = 1.0 if move >= 2.0 * atr else 0.0
        out.at[index, "future_3atr_hit"] = 1.0 if move >= 3.0 * atr else 0.0
    return out


def build_feature_dataset(
    bars_by_key: dict[tuple[str, str], pd.DataFrame],
    labels: pd.DataFrame,
    funding_histories: dict[str, pd.DataFrame],
    symbols: list[str],
    timeframes: list[str],
    splits: list[TimeSplit],
    funding_data_complete: bool,
    warnings: list[str] | None = None,
) -> pd.DataFrame:
    """Build the full ETC-v1 feature dataset."""

    warnings = warnings if warnings is not None else []
    frames: list[pd.DataFrame] = []
    for symbol in symbols:
        for timeframe in timeframes:
            bars = bars_by_key.get((symbol, timeframe), pd.DataFrame())
            frames.append(add_symbol_features(bars, symbol, timeframe))
    features = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if features.empty:
        return features
    features = add_cross_symbol_features(features, symbols, timeframes)
    features = align_funding_features(features, funding_histories, symbols, timeframes, funding_data_complete, warnings)
    features = add_labels_to_dataset(features, labels, splits)
    features = add_forward_outcomes(features, bars_by_key)
    selected_columns = [
        "timestamp",
        "symbol",
        "inst_id",
        "timeframe",
        "bar_index",
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "split",
        "label",
        "trend_segment_id",
        "trend_direction",
        "trend_phase_pct",
        "segment_start",
        "segment_end",
        "segment_return",
        "segment_atr_move",
        "segment_efficiency",
        "direction_proxy",
        "early_trend",
        "nontrend",
        "direction_match",
        "return_1",
        "return_20",
        "return_55",
        "return_100",
        "atr_20",
        "atr_55",
        "atr_pct_20",
        "realized_vol_20",
        *REQUIRED_FEATURE_COLUMNS,
        *DERIVED_SCORE_COLUMNS,
        "forward_return_4h",
        "forward_return_1d",
        "forward_return_3d",
        "future_2atr_hit",
        "future_3atr_hit",
    ]
    for column in selected_columns:
        if column not in features.columns:
            features[column] = np.nan
    return features.loc[:, selected_columns].sort_values(["timeframe", "timestamp", "symbol"], kind="stable").reset_index(drop=True)


def valid_research_sample(dataset: pd.DataFrame) -> pd.DataFrame:
    """Return rows allowed in feature/bucket statistics."""

    if dataset.empty:
        return dataset.copy()
    return dataset[dataset["label"].isin(["early_uptrend", "early_downtrend", "middle_trend", "late_trend", "nontrend"])].copy()


def bucket_edges(train_values: pd.Series) -> list[float]:
    """Build q0/q20/.../q100 bucket edges from train only."""

    values = pd.to_numeric(train_values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if values.empty:
        return [np.nan] * 6
    return [float(values.quantile(q)) for q in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]]


def assign_quantile_bucket(value: Any, edges: list[float]) -> str | None:
    """Assign one value to fixed train quantile edges."""

    number = finite_float(value)
    if not np.isfinite(number) or len(edges) != 6 or not np.isfinite(edges[0]):
        return None
    labels = ["q0-q20", "q20-q40", "q40-q60", "q60-q80", "q80-q100"]
    for index, label in enumerate(labels):
        high = edges[index + 1]
        if index == 4:
            if number <= high or np.isclose(number, high):
                return label
        elif number <= high or np.isclose(number, high):
            return label
    return labels[-1] if number > edges[-1] else labels[0]


def metric_summary(frame: pd.DataFrame) -> dict[str, Any]:
    """Summarize label and forward metrics for one sample set."""

    if frame.empty:
        return {
            "sample_count": 0,
            "early_trend_count": 0,
            "early_trend_rate": np.nan,
            "early_uptrend_count": 0,
            "early_downtrend_count": 0,
            "nontrend_count": 0,
            "nontrend_rate": np.nan,
            "direction_match_rate": np.nan,
            "avg_forward_return_4h": np.nan,
            "avg_forward_return_1d": np.nan,
            "avg_forward_return_3d": np.nan,
            "future_2atr_hit_rate": np.nan,
            "future_3atr_hit_rate": np.nan,
        }
    count = int(len(frame.index))
    early_mask = frame["label"].isin(["early_uptrend", "early_downtrend"])
    nontrend_mask = frame["label"].eq("nontrend")
    if "direction_match" in frame.columns:
        direction_matches = pd.to_numeric(frame.loc[early_mask, "direction_match"], errors="coerce")
    else:
        direction_matches = pd.Series(dtype=float)
    forward_4h = frame["forward_return_4h"] if "forward_return_4h" in frame.columns else pd.Series(dtype=float)
    forward_1d = frame["forward_return_1d"] if "forward_return_1d" in frame.columns else pd.Series(dtype=float)
    forward_3d = frame["forward_return_3d"] if "forward_return_3d" in frame.columns else pd.Series(dtype=float)
    hit_2atr = frame["future_2atr_hit"] if "future_2atr_hit" in frame.columns else pd.Series(dtype=float)
    hit_3atr = frame["future_3atr_hit"] if "future_3atr_hit" in frame.columns else pd.Series(dtype=float)
    return {
        "sample_count": count,
        "early_trend_count": int(early_mask.sum()),
        "early_trend_rate": float(early_mask.mean()) if count else np.nan,
        "early_uptrend_count": int(frame["label"].eq("early_uptrend").sum()),
        "early_downtrend_count": int(frame["label"].eq("early_downtrend").sum()),
        "nontrend_count": int(nontrend_mask.sum()),
        "nontrend_rate": float(nontrend_mask.mean()) if count else np.nan,
        "direction_match_rate": float(direction_matches.mean()) if not direction_matches.dropna().empty else np.nan,
        "avg_forward_return_4h": float(pd.to_numeric(forward_4h, errors="coerce").mean()),
        "avg_forward_return_1d": float(pd.to_numeric(forward_1d, errors="coerce").mean()),
        "avg_forward_return_3d": float(pd.to_numeric(forward_3d, errors="coerce").mean()),
        "future_2atr_hit_rate": float(pd.to_numeric(hit_2atr, errors="coerce").mean()),
        "future_3atr_hit_rate": float(pd.to_numeric(hit_3atr, errors="coerce").mean()),
    }


def monotonicity_from_rates(rates: list[float]) -> float:
    """Return absolute bucket-index correlation for early-trend rates."""

    clean = [(index, rate) for index, rate in enumerate(rates) if np.isfinite(rate)]
    if len(clean) < 3:
        return np.nan
    x = np.array([item[0] for item in clean], dtype=float)
    y = np.array([item[1] for item in clean], dtype=float)
    if np.nanstd(y) == 0.0:
        return 0.0
    return float(abs(np.corrcoef(x, y)[0, 1]))


def split_lift(bucket_rows: pd.DataFrame, split: str) -> float:
    """Return q80-q100 minus q0-q20 early trend lift for one split."""

    subset = bucket_rows[bucket_rows["split"] == split]
    if subset.empty:
        return np.nan
    top = subset.loc[subset["bucket"] == "q80-q100", "early_trend_rate"]
    bottom = subset.loc[subset["bucket"] == "q0-q20", "early_trend_rate"]
    if top.empty or bottom.empty:
        return np.nan
    return float(top.iloc[0] - bottom.iloc[0])


def build_feature_bucket_analysis(dataset: pd.DataFrame, feature_columns: list[str] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run single-feature train-defined bucket analysis."""

    samples = valid_research_sample(dataset)
    feature_columns = feature_columns or REQUIRED_FEATURE_COLUMNS
    bucket_labels = ["q0-q20", "q20-q40", "q40-q60", "q60-q80", "q80-q100"]
    rows: list[dict[str, Any]] = []
    score_rows: list[dict[str, Any]] = []
    for feature in feature_columns:
        if feature not in samples.columns:
            continue
        edges = bucket_edges(samples.loc[samples["split"] == "train_ext", feature])
        work = samples.copy()
        work["_bucket"] = work[feature].map(lambda value: assign_quantile_bucket(value, edges))
        feature_rows: list[dict[str, Any]] = []
        for split in SPLIT_NAMES:
            split_frame = work[work["split"] == split]
            for bucket_index, bucket in enumerate(bucket_labels):
                subset = split_frame[split_frame["_bucket"] == bucket]
                metrics = metric_summary(subset)
                row = {
                    "feature": feature,
                    "split": split,
                    "bucket": bucket,
                    **metrics,
                    "monotonicity_score": np.nan,
                    "feature_predictiveness_score": np.nan,
                    "train_validation_consistency": np.nan,
                    "oos_consistency": np.nan,
                    "train_bucket_boundary_low": edges[bucket_index],
                    "train_bucket_boundary_high": edges[bucket_index + 1],
                }
                rows.append(row)
                feature_rows.append(row)
        feature_df = pd.DataFrame(feature_rows)
        train_rates = [
            float(feature_df[(feature_df["split"] == "train_ext") & (feature_df["bucket"] == bucket)]["early_trend_rate"].iloc[0])
            for bucket in bucket_labels
        ]
        finite_rates = [rate for rate in train_rates if np.isfinite(rate)]
        predictiveness = float(max(finite_rates) - min(finite_rates)) if finite_rates else np.nan
        monotonicity = monotonicity_from_rates(train_rates)
        train_lift = split_lift(feature_df, "train_ext")
        validation_lift = split_lift(feature_df, "validation_ext")
        oos_lift = split_lift(feature_df, "oos_ext")
        train_validation_consistency = bool(np.isfinite(train_lift) and np.isfinite(validation_lift) and np.sign(train_lift) == np.sign(validation_lift))
        oos_consistency = bool(np.isfinite(train_lift) and np.isfinite(oos_lift) and np.sign(train_lift) == np.sign(oos_lift))
        for row in rows:
            if row["feature"] == feature:
                row["monotonicity_score"] = monotonicity
                row["feature_predictiveness_score"] = predictiveness
                row["train_validation_consistency"] = train_validation_consistency
                row["oos_consistency"] = oos_consistency
        score_rows.append(
            {
                "feature": feature,
                "feature_predictiveness_score": predictiveness,
                "monotonicity_score": monotonicity,
                "train_validation_consistency": train_validation_consistency,
                "oos_consistency": oos_consistency,
                "train_lift_q80_minus_q0": train_lift,
                "validation_lift_q80_minus_q0": validation_lift,
                "oos_lift_q80_minus_q0": oos_lift,
            }
        )
    bucket_analysis = pd.DataFrame(rows)
    predictiveness = pd.DataFrame(score_rows)
    if not predictiveness.empty:
        predictiveness = predictiveness.sort_values(
            ["feature_predictiveness_score", "monotonicity_score"],
            ascending=[False, False],
            kind="stable",
        ).reset_index(drop=True)
    return bucket_analysis, predictiveness


def compute_composite_score(dataset: pd.DataFrame, warnings: list[str] | None = None) -> pd.DataFrame:
    """Compute pre-registered linear ETC score using train z-scores only."""

    warnings = warnings if warnings is not None else []
    out = dataset.copy()
    out["early_trend_score"] = 0.0
    out["early_trend_score_component_count"] = 0
    out["composite_score_available"] = False
    train = out[out["split"] == "train_ext"]
    used_terms: list[str] = []
    for column, sign in COMPOSITE_TERMS:
        if column not in out.columns:
            warnings.append(f"composite_missing_feature:{column}")
            continue
        train_values = pd.to_numeric(train[column], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        if len(train_values.index) < 10:
            warnings.append(f"composite_unusable_feature:{column}")
            continue
        mean = float(train_values.mean())
        std = float(train_values.std(ddof=0))
        if not np.isfinite(std) or std == 0.0:
            warnings.append(f"composite_zero_std_feature:{column}")
            continue
        z = (pd.to_numeric(out[column], errors="coerce") - mean) / std
        component = sign * z
        out["early_trend_score"] = out["early_trend_score"] + component.fillna(0.0)
        out["early_trend_score_component_count"] = out["early_trend_score_component_count"] + component.notna().astype(int)
        used_terms.append(column)
    out.loc[out["early_trend_score_component_count"] == 0, "early_trend_score"] = np.nan
    out["composite_score_available"] = out["early_trend_score_component_count"] > 0
    out.attrs["composite_terms_used"] = used_terms
    return out


def deterministic_rng(seed: int, *parts: Any) -> np.random.Generator:
    """Create a deterministic RNG scoped to a control selection."""

    text = "|".join(str(part) for part in (seed, *parts))
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "little") % (2**32 - 1))


def score_thresholds_from_train(scored: pd.DataFrame) -> dict[str, float]:
    """Build score bucket boundaries from train only."""

    train_scores = pd.to_numeric(scored.loc[scored["split"] == "train_ext", "early_trend_score"], errors="coerce").dropna()
    if train_scores.empty:
        return {"top10": np.nan, "top20": np.nan, "top30": np.nan, "bottom30": np.nan}
    return {
        "top10": float(train_scores.quantile(0.90)),
        "top20": float(train_scores.quantile(0.80)),
        "top30": float(train_scores.quantile(0.70)),
        "bottom30": float(train_scores.quantile(0.30)),
    }


def build_score_bucket_analysis(scored: pd.DataFrame, config: EtcConfig | None = None) -> pd.DataFrame:
    """Analyze composite score buckets and random controls."""

    config = config or EtcConfig()
    samples = valid_research_sample(scored)
    thresholds = score_thresholds_from_train(samples)
    rows: list[dict[str, Any]] = []
    random_rates: dict[str, float] = {}
    for split in SPLIT_NAMES:
        split_frame = samples[samples["split"] == split].copy()
        if split_frame.empty:
            random_rates[split] = np.nan
            continue
        top20 = split_frame[split_frame["early_trend_score"] >= thresholds["top20"]]
        eligible = split_frame[pd.to_numeric(split_frame["early_trend_score"], errors="coerce").notna()]
        random_count = min(len(top20.index), len(eligible.index))
        if random_count > 0:
            rng = deterministic_rng(config.random_seed, "score_bucket_random", split, random_count)
            chosen = rng.choice(eligible.index.to_numpy(), size=random_count, replace=False)
            random_frame = eligible.loc[chosen]
        else:
            random_frame = eligible.head(0)
        random_metrics = metric_summary(random_frame)
        random_rates[split] = random_metrics["early_trend_rate"]
        rows.append({"split": split, "score_bucket": "random_control", **random_metrics, "random_control_comparison": 0.0})
        for bucket in ["top10", "top20", "top30", "bottom30"]:
            if bucket.startswith("top"):
                subset = split_frame[split_frame["early_trend_score"] >= thresholds[bucket]]
            else:
                subset = split_frame[split_frame["early_trend_score"] <= thresholds[bucket]]
            metrics = metric_summary(subset)
            comparison = (
                metrics["early_trend_rate"] - random_rates[split]
                if np.isfinite(metrics["early_trend_rate"]) and np.isfinite(random_rates[split])
                else np.nan
            )
            rows.append({"split": split, "score_bucket": bucket, **metrics, "random_control_comparison": comparison})
    return pd.DataFrame(rows)


def can_trade_from_event(row: pd.Series, bars_by_key: dict[tuple[str, str], pd.DataFrame], max_hold_minutes: int = 4320) -> bool:
    """Return True when a row has enough next-open and max-hold bars."""

    bars = bars_by_key.get((str(row["symbol"]), str(row["timeframe"])))
    if bars is None or bars.empty:
        return False
    hold_bars = max(1, int(math.ceil(max_hold_minutes / TIMEFRAME_MINUTES[str(row["timeframe"])])))
    return int(row["bar_index"]) + hold_bars + 1 < len(bars.index)


def make_event_row(
    row: pd.Series,
    group: str,
    event_group: str,
    score_bucket: str,
    control_source_event_id: str | None = None,
    reverse: bool = False,
) -> dict[str, Any]:
    """Build one ETC event row."""

    direction = str(row.get("direction_proxy") or "long")
    if reverse:
        direction = "short" if direction == "long" else "long"
    return {
        "event_id": "",
        "timestamp": row["timestamp"],
        "symbol": row["symbol"],
        "inst_id": row["inst_id"],
        "timeframe": row["timeframe"],
        "group": group,
        "event_group": event_group,
        "score_bucket": score_bucket,
        "direction": direction,
        "direction_policy": "long if direction-aware return + trend efficiency direction + rank_change_20 is non-negative, else short",
        "reverse_of_event_id": control_source_event_id if reverse else None,
        "control_source_event_id": control_source_event_id if not reverse else None,
        "bar_index": int(row["bar_index"]),
        "score": finite_float(row.get("early_trend_score")),
        "component_count": int(finite_float(row.get("early_trend_score_component_count"), 0.0)),
        "label": row.get("label"),
        "trend_segment_id": row.get("trend_segment_id"),
        "trend_direction": row.get("trend_direction"),
        "direction_match": row.get("direction_match"),
        "split": row.get("split"),
        "close": finite_float(row.get("close")),
        "atr_20": finite_float(row.get("atr_20")),
        "forward_return_4h": finite_float(row.get("forward_return_4h")),
        "forward_return_1d": finite_float(row.get("forward_return_1d")),
        "forward_return_3d": finite_float(row.get("forward_return_3d")),
        "future_2atr_hit": row.get("future_2atr_hit"),
        "future_3atr_hit": row.get("future_3atr_hit"),
    }


def assign_event_ids(events: pd.DataFrame) -> pd.DataFrame:
    """Assign stable ETC event ids."""

    if events.empty:
        return events.copy()
    out = events.copy().reset_index(drop=True)
    out["event_id"] = [
        f"etc_v1_{index:08d}_{safe_symbol(row.symbol)}_{row.timeframe}_{row.group}_{row.direction}"
        for index, row in enumerate(out.itertuples(index=False), start=1)
    ]
    return out


def generate_top_score_events(
    scored: pd.DataFrame,
    bars_by_key: dict[tuple[str, str], pd.DataFrame],
    config: EtcConfig | None = None,
) -> pd.DataFrame:
    """Generate A/B/C top-score events, D random control, and E reverse test."""

    config = config or EtcConfig()
    samples = valid_research_sample(scored)
    thresholds = score_thresholds_from_train(samples)
    eligible = samples[
        pd.to_numeric(samples["early_trend_score"], errors="coerce").notna()
        & samples.apply(lambda row: can_trade_from_event(row, bars_by_key), axis=1)
    ].copy()
    rows: list[dict[str, Any]] = []
    selections: dict[str, pd.DataFrame] = {}
    for group, bucket, threshold in [("A", "top10", thresholds["top10"]), ("B", "top20", thresholds["top20"]), ("C", "top30", thresholds["top30"])]:
        selected = eligible[eligible["early_trend_score"] >= threshold].copy()
        selections[group] = selected
        for _, row in selected.iterrows():
            rows.append(make_event_row(row, group, EVENT_GROUPS[group], f"{bucket}_score_events"))
    events = assign_event_ids(pd.DataFrame(rows))
    rows = dataframe_records(events) if not events.empty else []
    top20 = selections.get("B", pd.DataFrame())
    random_rows: list[dict[str, Any]] = []
    for split in SPLIT_NAMES:
        split_top = top20[top20["split"] == split]
        split_eligible = eligible[eligible["split"] == split]
        count = min(len(split_top.index), len(split_eligible.index))
        if count <= 0:
            continue
        rng = deterministic_rng(config.random_seed, "events_random", split, count)
        chosen = rng.choice(split_eligible.index.to_numpy(), size=count, replace=False)
        for _, row in split_eligible.loc[chosen].iterrows():
            random_rows.append(make_event_row(row, "D", EVENT_GROUPS["D"], "random_score_matched_control"))
    random_events = assign_event_ids(pd.DataFrame(random_rows))
    if not random_events.empty:
        random_events["event_id"] = [f"etc_v1_random_{index:08d}" for index in range(1, len(random_events.index) + 1)]
    reverse_rows: list[dict[str, Any]] = []
    if not events.empty:
        top20_events = events[events["group"] == "B"]
        event_by_key = {
            (row.symbol, row.timeframe, int(row.bar_index)): row.event_id
            for row in top20_events.itertuples(index=False)
        }
        for _, row in top20.iterrows():
            source_event_id = event_by_key.get((row["symbol"], row["timeframe"], int(row["bar_index"])))
            reverse_rows.append(make_event_row(row, "E", EVENT_GROUPS["E"], "reverse_direction_test", source_event_id, reverse=True))
    reverse_events = assign_event_ids(pd.DataFrame(reverse_rows))
    if not reverse_events.empty:
        reverse_events["event_id"] = [f"etc_v1_reverse_{index:08d}" for index in range(1, len(reverse_events.index) + 1)]
    frames = [events, random_events, reverse_events]
    output = pd.concat([frame for frame in frames if not frame.empty], ignore_index=True) if any(not frame.empty for frame in frames) else pd.DataFrame()
    if output.empty:
        return pd.DataFrame(
            columns=[
                "event_id",
                "timestamp",
                "symbol",
                "inst_id",
                "timeframe",
                "group",
                "event_group",
                "score_bucket",
                "direction",
                "direction_policy",
                "reverse_of_event_id",
                "control_source_event_id",
                "bar_index",
                "score",
                "component_count",
                "label",
                "trend_segment_id",
                "trend_direction",
                "split",
                "close",
                "atr_20",
                "forward_return_4h",
                "forward_return_1d",
                "forward_return_3d",
                "future_2atr_hit",
                "future_3atr_hit",
            ]
        )
    return output.sort_values(["timestamp", "symbol", "timeframe", "group"], kind="stable").reset_index(drop=True)


def simulate_event_trades(
    events: pd.DataFrame,
    bars_by_key: dict[tuple[str, str], pd.DataFrame],
    funding_histories: dict[str, pd.DataFrame],
    config: EtcConfig | None = None,
) -> pd.DataFrame:
    """Simulate fixed-hold no-cost, cost-aware, and funding-adjusted event trades."""

    config = config or EtcConfig()
    columns = [
        "trade_id",
        "event_id",
        "timestamp",
        "symbol",
        "inst_id",
        "timeframe",
        "group",
        "event_group",
        "score_bucket",
        "direction",
        "hold",
        "hold_bars",
        "entry_time",
        "entry_price",
        "exit_time",
        "exit_price",
        "gross_return",
        "no_cost_pnl",
        "fee_cost",
        "slippage_cost",
        "cost_aware_pnl",
        "funding_count",
        "funding_pnl",
        "funding_adjusted_pnl",
        "split",
        "label",
    ]
    if events.empty:
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, Any]] = []
    for event in events.itertuples(index=False):
        bars = bars_by_key.get((str(event.symbol), str(event.timeframe)))
        if bars is None or bars.empty:
            continue
        pos = int(event.bar_index)
        entry_pos = pos + 1
        if entry_pos >= len(bars.index):
            continue
        for hold_name, minutes in HOLD_MINUTES.items():
            hold_bars = max(1, int(math.ceil(minutes / TIMEFRAME_MINUTES[str(event.timeframe)])))
            exit_pos = pos + hold_bars + 1
            if exit_pos >= len(bars.index):
                continue
            entry_time = bars.iloc[entry_pos]["open_time"]
            exit_time = bars.iloc[exit_pos]["open_time"]
            entry_price = float(bars.iloc[entry_pos]["open"])
            exit_price = float(bars.iloc[exit_pos]["open"])
            gross_return = directional_return(entry_price, exit_price, str(event.direction))
            no_cost_pnl = config.fixed_notional * gross_return if np.isfinite(gross_return) else np.nan
            fee_cost = config.fixed_notional * (config.fee_bps_per_side / 10000.0) * 2.0
            slippage_cost = config.fixed_notional * (config.slippage_bps_per_side / 10000.0) * 2.0
            funding_df = funding_histories.get(str(event.inst_id), pd.DataFrame(columns=["funding_time_utc", "funding_rate"]))
            funding_pnl, funding_count = funding_pnl_for_interval(
                funding_df,
                entry_time,
                exit_time,
                str(event.direction),
                config.fixed_notional,
            )
            cost_aware_pnl = no_cost_pnl - fee_cost - slippage_cost
            rows.append(
                {
                    "trade_id": f"etc_v1_trade_{len(rows) + 1:08d}",
                    "event_id": event.event_id,
                    "timestamp": event.timestamp,
                    "symbol": event.symbol,
                    "inst_id": event.inst_id,
                    "timeframe": event.timeframe,
                    "group": event.group,
                    "event_group": event.event_group,
                    "score_bucket": event.score_bucket,
                    "direction": event.direction,
                    "hold": hold_name,
                    "hold_bars": hold_bars,
                    "entry_time": entry_time,
                    "entry_price": entry_price,
                    "exit_time": exit_time,
                    "exit_price": exit_price,
                    "gross_return": gross_return,
                    "no_cost_pnl": no_cost_pnl,
                    "fee_cost": fee_cost,
                    "slippage_cost": slippage_cost,
                    "cost_aware_pnl": cost_aware_pnl,
                    "funding_count": funding_count,
                    "funding_pnl": funding_pnl,
                    "funding_adjusted_pnl": cost_aware_pnl + funding_pnl,
                    "split": event.split,
                    "label": event.label,
                }
            )
    return pd.DataFrame(rows, columns=columns)


def pnl_sum(frame: pd.DataFrame, column: str) -> float:
    """Return numeric sum for a PnL column."""

    if frame.empty or column not in frame.columns:
        return 0.0
    return float(pd.to_numeric(frame[column], errors="coerce").sum())


def largest_symbol_pnl_share(trades: pd.DataFrame, pnl_column: str) -> tuple[float | None, str | None]:
    """Return largest symbol PnL share."""

    if trades.empty or pnl_column not in trades.columns:
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


def largest_symbol_event_share(frame: pd.DataFrame) -> tuple[float | None, str | None]:
    """Return largest symbol event/trade-count share."""

    if frame.empty:
        return None, None
    counts = frame.groupby("symbol").size()
    if counts.empty or int(counts.sum()) == 0:
        return None, None
    symbol = str(counts.idxmax())
    return float(counts.max() / counts.sum()), symbol


def top_trade_contribution(trades: pd.DataFrame, pnl_column: str) -> tuple[float | None, float, int]:
    """Return top 5 percent trade contribution."""

    if trades.empty or pnl_column not in trades.columns:
        return None, 0.0, 0
    pnl = pd.to_numeric(trades[pnl_column], errors="coerce").fillna(0.0).sort_values(ascending=False)
    count = int(len(pnl.index))
    top_count = max(1, int(math.ceil(count * 0.05))) if count else 0
    top_pnl = float(pnl.head(top_count).sum()) if top_count else 0.0
    total = float(pnl.sum())
    return (float(top_pnl / total) if total > 0 else None), top_pnl, top_count


def summarize_events(events: pd.DataFrame) -> pd.DataFrame:
    """Summarize event labels by group/split/timeframe."""

    columns = [
        "group",
        "event_group",
        "score_bucket",
        "split",
        "timeframe",
        "event_count",
        "early_trend_rate",
        "early_uptrend_rate",
        "early_downtrend_rate",
        "nontrend_rate",
        "direction_match_rate",
        "future_2atr_hit_rate",
        "future_3atr_hit_rate",
        "avg_forward_return_3d",
    ]
    if events.empty:
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, Any]] = []
    for keys, group in events.groupby(["group", "event_group", "score_bucket", "split", "timeframe"], dropna=False, sort=True):
        group_label, event_group, score_bucket, split, timeframe = keys
        metrics = metric_summary(group.rename(columns={"event_count": "_event_count"}))
        rows.append(
            {
                "group": group_label,
                "event_group": event_group,
                "score_bucket": score_bucket,
                "split": split,
                "timeframe": timeframe,
                "event_count": int(len(group.index)),
                "early_trend_rate": metrics["early_trend_rate"],
                "early_uptrend_rate": float(group["label"].eq("early_uptrend").mean()),
                "early_downtrend_rate": float(group["label"].eq("early_downtrend").mean()),
                "nontrend_rate": metrics["nontrend_rate"],
                "direction_match_rate": metrics["direction_match_rate"],
                "future_2atr_hit_rate": metrics["future_2atr_hit_rate"],
                "future_3atr_hit_rate": metrics["future_3atr_hit_rate"],
                "avg_forward_return_3d": metrics["avg_forward_return_3d"],
            }
        )
    return pd.DataFrame(rows, columns=columns)


def summarize_trades(trades: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    """Summarize trades by requested columns."""

    columns = group_columns + [
        "trade_count",
        "no_cost_pnl",
        "cost_aware_pnl",
        "funding_pnl",
        "funding_adjusted_pnl",
        "avg_gross_return",
        "win_rate",
    ]
    if trades.empty:
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, Any]] = []
    for keys, group in trades.groupby(group_columns, dropna=False, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {column: key for column, key in zip(group_columns, keys)}
        row.update(
            {
                "trade_count": int(len(group.index)),
                "no_cost_pnl": pnl_sum(group, "no_cost_pnl"),
                "cost_aware_pnl": pnl_sum(group, "cost_aware_pnl"),
                "funding_pnl": pnl_sum(group, "funding_pnl"),
                "funding_adjusted_pnl": pnl_sum(group, "funding_adjusted_pnl"),
                "avg_gross_return": float(pd.to_numeric(group["gross_return"], errors="coerce").mean()),
                "win_rate": float((pd.to_numeric(group["funding_adjusted_pnl"], errors="coerce") > 0).mean()),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows, columns=columns).sort_values(group_columns, kind="stable").reset_index(drop=True)


def build_event_summary(events: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
    """Build event-study summary by group/hold/split."""

    columns = [
        "group",
        "event_group",
        "hold",
        "split",
        "event_count",
        "trade_count",
        "early_trend_rate",
        "early_uptrend_rate",
        "early_downtrend_rate",
        "nontrend_rate",
        "direction_match_rate",
        "future_2atr_hit_rate",
        "future_3atr_hit_rate",
        "avg_forward_return_3d",
        "no_cost_pnl",
        "cost_aware_pnl",
        "funding_pnl",
        "funding_adjusted_pnl",
    ]
    if events.empty:
        return pd.DataFrame(columns=columns)
    trade_summary = summarize_trades(trades, ["group", "event_group", "hold", "split"])
    rows: list[dict[str, Any]] = []
    for (group, event_group, split), event_frame in events.groupby(["group", "event_group", "split"], dropna=False, sort=True):
        metrics = metric_summary(event_frame)
        for hold in HOLD_MINUTES:
            trades_row = trade_summary[
                (trade_summary["group"] == group)
                & (trade_summary["event_group"] == event_group)
                & (trade_summary["hold"] == hold)
                & (trade_summary["split"] == split)
            ]
            pnl_values = trades_row.iloc[0].to_dict() if not trades_row.empty else {}
            rows.append(
                {
                    "group": group,
                    "event_group": event_group,
                    "hold": hold,
                    "split": split,
                    "event_count": int(len(event_frame.index)),
                    "trade_count": int(pnl_values.get("trade_count", 0) or 0),
                    "early_trend_rate": metrics["early_trend_rate"],
                    "early_uptrend_rate": float(event_frame["label"].eq("early_uptrend").mean()),
                    "early_downtrend_rate": float(event_frame["label"].eq("early_downtrend").mean()),
                    "nontrend_rate": metrics["nontrend_rate"],
                    "direction_match_rate": metrics["direction_match_rate"],
                    "future_2atr_hit_rate": metrics["future_2atr_hit_rate"],
                    "future_3atr_hit_rate": metrics["future_3atr_hit_rate"],
                    "avg_forward_return_3d": metrics["avg_forward_return_3d"],
                    "no_cost_pnl": finite_float(pnl_values.get("no_cost_pnl"), 0.0),
                    "cost_aware_pnl": finite_float(pnl_values.get("cost_aware_pnl"), 0.0),
                    "funding_pnl": finite_float(pnl_values.get("funding_pnl"), 0.0),
                    "funding_adjusted_pnl": finite_float(pnl_values.get("funding_adjusted_pnl"), 0.0),
                }
            )
    return pd.DataFrame(rows, columns=columns)


def build_concentration_summary(trades: pd.DataFrame) -> pd.DataFrame:
    """Build concentration diagnostics by group/hold/split."""

    columns = [
        "group",
        "event_group",
        "hold",
        "split",
        "trade_count",
        "total_funding_adjusted_pnl",
        "largest_symbol_pnl_share",
        "largest_symbol_pnl_symbol",
        "largest_symbol_event_share",
        "largest_symbol_event_symbol",
        "top_5pct_trade_pnl_contribution",
        "top_5pct_trade_pnl",
        "top_5pct_trade_count",
        "concentration_pass",
    ]
    if trades.empty:
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, Any]] = []
    for (group, event_group, hold, split), frame in trades.groupby(["group", "event_group", "hold", "split"], dropna=False, sort=True):
        pnl_share, pnl_symbol = largest_symbol_pnl_share(frame, "funding_adjusted_pnl")
        event_share, event_symbol = largest_symbol_event_share(frame)
        top_share, top_pnl, top_count = top_trade_contribution(frame, "funding_adjusted_pnl")
        concentration_pass = bool(
            (pnl_share is None or pnl_share <= 0.70)
            and (event_share is None or event_share <= 0.70)
            and (top_share is None or top_share <= 0.80)
        )
        rows.append(
            {
                "group": group,
                "event_group": event_group,
                "hold": hold,
                "split": split,
                "trade_count": int(len(frame.index)),
                "total_funding_adjusted_pnl": pnl_sum(frame, "funding_adjusted_pnl"),
                "largest_symbol_pnl_share": pnl_share,
                "largest_symbol_pnl_symbol": pnl_symbol,
                "largest_symbol_event_share": event_share,
                "largest_symbol_event_symbol": event_symbol,
                "top_5pct_trade_pnl_contribution": top_share,
                "top_5pct_trade_pnl": top_pnl,
                "top_5pct_trade_count": top_count,
                "concentration_pass": concentration_pass,
            }
        )
    return pd.DataFrame(rows, columns=columns)


def build_random_control(event_summary: pd.DataFrame) -> pd.DataFrame:
    """Compare top-score groups to random score-matched control."""

    columns = [
        "group",
        "event_group",
        "hold",
        "split",
        "positive_early_trend_rate",
        "random_early_trend_rate",
        "positive_no_cost_pnl",
        "random_no_cost_pnl",
        "positive_funding_adjusted_pnl",
        "random_funding_adjusted_pnl",
        "random_weaker",
    ]
    if event_summary.empty:
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, Any]] = []
    random_rows = event_summary[event_summary["group"] == "D"]
    for row in event_summary[event_summary["group"].isin(["A", "B", "C"])].itertuples(index=False):
        control = random_rows[(random_rows["hold"] == row.hold) & (random_rows["split"] == row.split)]
        if control.empty:
            random_rate = np.nan
            random_no_cost = np.nan
            random_funding = np.nan
        else:
            item = control.iloc[0]
            random_rate = finite_float(item["early_trend_rate"])
            random_no_cost = finite_float(item["no_cost_pnl"])
            random_funding = finite_float(item["funding_adjusted_pnl"])
        rows.append(
            {
                "group": row.group,
                "event_group": row.event_group,
                "hold": row.hold,
                "split": row.split,
                "positive_early_trend_rate": row.early_trend_rate,
                "random_early_trend_rate": random_rate,
                "positive_no_cost_pnl": row.no_cost_pnl,
                "random_no_cost_pnl": random_no_cost,
                "positive_funding_adjusted_pnl": row.funding_adjusted_pnl,
                "random_funding_adjusted_pnl": random_funding,
                "random_weaker": bool(
                    np.isfinite(random_rate)
                    and row.early_trend_rate > random_rate
                    and row.no_cost_pnl > random_no_cost
                    and row.funding_adjusted_pnl > random_funding
                ),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def build_reverse_test(event_summary: pd.DataFrame) -> pd.DataFrame:
    """Compare top20 score events against reverse direction test."""

    columns = [
        "hold",
        "split",
        "forward_early_trend_rate",
        "reverse_early_trend_rate",
        "forward_no_cost_pnl",
        "reverse_no_cost_pnl",
        "forward_funding_adjusted_pnl",
        "reverse_funding_adjusted_pnl",
        "reverse_weaker",
    ]
    if event_summary.empty:
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, Any]] = []
    forward = event_summary[event_summary["group"] == "B"]
    reverse = event_summary[event_summary["group"] == "E"]
    for hold in HOLD_MINUTES:
        for split in SPLIT_NAMES:
            f = forward[(forward["hold"] == hold) & (forward["split"] == split)]
            r = reverse[(reverse["hold"] == hold) & (reverse["split"] == split)]
            if f.empty or r.empty:
                rows.append(
                    {
                        "hold": hold,
                        "split": split,
                        "forward_early_trend_rate": np.nan,
                        "reverse_early_trend_rate": np.nan,
                        "forward_no_cost_pnl": np.nan,
                        "reverse_no_cost_pnl": np.nan,
                        "forward_funding_adjusted_pnl": np.nan,
                        "reverse_funding_adjusted_pnl": np.nan,
                        "reverse_weaker": False,
                    }
                )
                continue
            f_row = f.iloc[0]
            r_row = r.iloc[0]
            rows.append(
                {
                    "hold": hold,
                    "split": split,
                    "forward_early_trend_rate": f_row["early_trend_rate"],
                    "reverse_early_trend_rate": r_row["early_trend_rate"],
                    "forward_no_cost_pnl": f_row["no_cost_pnl"],
                    "reverse_no_cost_pnl": r_row["no_cost_pnl"],
                    "forward_funding_adjusted_pnl": f_row["funding_adjusted_pnl"],
                    "reverse_funding_adjusted_pnl": r_row["funding_adjusted_pnl"],
                    "reverse_weaker": bool(
                        f_row["no_cost_pnl"] > r_row["no_cost_pnl"]
                        and f_row["funding_adjusted_pnl"] > r_row["funding_adjusted_pnl"]
                    ),
                }
            )
    return pd.DataFrame(rows, columns=columns)


def evaluate_phase_gates(
    event_summary: pd.DataFrame,
    random_control: pd.DataFrame,
    reverse_test: pd.DataFrame,
    concentration: pd.DataFrame,
    funding_data_complete: bool,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Evaluate ETC-v1 Phase 2 gate criteria."""

    candidate_rows: list[dict[str, Any]] = []
    if event_summary.empty:
        gates = {
            "top_score_event_pass": False,
            "random_control_pass": False,
            "reverse_test_pass": False,
            "cost_aware_pass": False,
            "funding_adjusted_pass": False,
            "concentration_pass": False,
            "can_enter_phase2": False,
            "final_decision": "postmortem_or_pause",
            "recommended_next_step": "postmortem_or_pause",
        }
        return gates, pd.DataFrame(columns=["group", "hold", "passed", "rejected_reasons"])
    for group in ["A", "B", "C"]:
        for hold in HOLD_MINUTES:
            reasons: list[str] = []
            subset = event_summary[(event_summary["group"] == group) & (event_summary["hold"] == hold)]
            if subset.empty:
                reasons.append("missing_event_summary")
            by_split = {split: subset[subset["split"] == split].iloc[0] for split in SPLIT_NAMES if not subset[subset["split"] == split].empty}
            random_by_split = {
                split: random_control[
                    (random_control["group"] == group) & (random_control["hold"] == hold) & (random_control["split"] == split)
                ].iloc[0]
                for split in SPLIT_NAMES
                if not random_control[
                    (random_control["group"] == group) & (random_control["hold"] == hold) & (random_control["split"] == split)
                ].empty
            }
            for split, multiplier in [("train_ext", 1.20), ("validation_ext", 1.10), ("oos_ext", 1.10)]:
                row = by_split.get(split)
                control = random_by_split.get(split)
                if row is None or control is None:
                    reasons.append(f"{split}_missing_control_or_group")
                    continue
                random_rate = finite_float(control["random_early_trend_rate"])
                if not (np.isfinite(random_rate) and row["early_trend_rate"] > random_rate * multiplier):
                    reasons.append(f"{split}_early_trend_rate_not_above_random_{multiplier}")
                if split in ["train_ext", "validation_ext", "oos_ext"] and row["no_cost_pnl"] <= 0:
                    reasons.append(f"{split}_no_cost_not_positive")
                if split == "oos_ext":
                    if row["cost_aware_pnl"] < 0:
                        reasons.append("oos_cost_aware_negative")
                    if (not funding_data_complete) or row["funding_adjusted_pnl"] < 0:
                        reasons.append("oos_funding_adjusted_negative_or_unavailable")
                if row["event_count"] < 30:
                    reasons.append(f"{split}_event_count_lt_30")
            reverse_rows = reverse_test[(reverse_test["hold"] == hold) & (reverse_test["split"] == "oos_ext")]
            if reverse_rows.empty or not bool(reverse_rows.iloc[0].get("reverse_weaker")):
                reasons.append("reverse_direction_not_weaker")
            random_rows = random_control[
                (random_control["group"] == group) & (random_control["hold"] == hold) & (random_control["split"] == "oos_ext")
            ]
            if random_rows.empty or not bool(random_rows.iloc[0].get("random_weaker")):
                reasons.append("random_control_not_weaker")
            concentration_rows = concentration[(concentration["group"] == group) & (concentration["hold"] == hold)]
            if concentration_rows.empty or not bool(concentration_rows["concentration_pass"].all()):
                reasons.append("concentration_failed")
            candidate_rows.append(
                {
                    "group": group,
                    "event_group": EVENT_GROUPS[group],
                    "hold": hold,
                    "passed": len(reasons) == 0,
                    "rejected_reasons": ";".join(reasons) if reasons else "",
                }
            )
    rejected = pd.DataFrame(candidate_rows)
    can_enter = bool((rejected["passed"] == True).any()) if not rejected.empty else False
    gates = {
        "top_score_event_pass": can_enter,
        "random_control_pass": bool(not random_control.empty and random_control["random_weaker"].fillna(False).any()),
        "reverse_test_pass": bool(not reverse_test.empty and reverse_test["reverse_weaker"].fillna(False).any()),
        "cost_aware_pass": bool((event_summary[(event_summary["group"].isin(["A", "B", "C"])) & (event_summary["split"] == "oos_ext")]["cost_aware_pnl"] >= 0).any()),
        "funding_adjusted_pass": bool(
            funding_data_complete
            and (event_summary[(event_summary["group"].isin(["A", "B", "C"])) & (event_summary["split"] == "oos_ext")]["funding_adjusted_pnl"] >= 0).any()
        ),
        "concentration_pass": bool(not concentration.empty and concentration["concentration_pass"].fillna(False).any()),
        "can_enter_phase2": can_enter,
        "final_decision": "phase2_research_asset" if can_enter else "postmortem_or_pause",
        "recommended_next_step": "etc_v1_phase2_research_only" if can_enter else "postmortem_or_pause",
    }
    return gates, rejected


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
                "large_gap_count": int(record.get("large_gap_count") or 0),
                "trade_count": int(len(symbol_trades.index)) if not symbol_trades.empty else 0,
                "funding_count": int(pd.to_numeric(symbol_trades.get("funding_count"), errors="coerce").sum()) if not symbol_trades.empty else 0,
                "funding_pnl": pnl_sum(symbol_trades, "funding_pnl") if not symbol_trades.empty else 0.0,
                "warnings": ";".join(record.get("warnings") or []),
            }
        )
    return pd.DataFrame(rows)


def build_summary_tables(trades: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build by-symbol, by-timeframe, and by-split outputs."""

    by_symbol = summarize_trades(trades, ["group", "event_group", "hold", "split", "symbol"])
    by_timeframe = summarize_trades(trades, ["group", "event_group", "hold", "split", "timeframe"])
    by_split = summarize_trades(trades, ["group", "event_group", "hold", "split"])
    return by_symbol, by_timeframe, by_split


def build_data_quality_payload(
    symbols: list[str],
    timeframes: list[str],
    data_range: HistoryRange,
    symbol_quality: dict[str, Any],
    funding_quality: dict[str, Any],
    trend_map_quality: dict[str, Any],
    strict: bool,
) -> dict[str, Any]:
    """Build data_quality.json payload."""

    symbol_records = []
    for symbol in symbols:
        one_minute = symbol_quality.get(symbol, {}).get("1m", {})
        timeframe_records = {timeframe: symbol_quality.get(symbol, {}).get(timeframe, {}) for timeframe in timeframes}
        symbol_records.append({"symbol": symbol, "one_minute": one_minute, "timeframes": timeframe_records})
    all_complete = bool(
        symbol_records
        and all(record["one_minute"].get("complete") for record in symbol_records)
        and all(
            timeframe_record.get("complete")
            for record in symbol_records
            for timeframe_record in record["timeframes"].values()
        )
    )
    return {
        "data_start": data_range.start.isoformat(),
        "data_end_exclusive": data_range.end_exclusive.isoformat(),
        "timezone": data_range.timezone_name,
        "data_check_strict": strict,
        "expected_1m_count": expected_bar_count(data_range),
        "market_data_complete": all_complete,
        "all_market_data_complete": all_complete,
        "symbols": symbol_records,
        "funding": funding_quality,
        "funding_data_complete": bool(funding_quality.get("funding_data_complete")),
        "trend_map_data_quality": trend_map_quality,
    }


def build_summary_payload(
    *,
    symbols: list[str],
    timeframes: list[str],
    data_range: HistoryRange,
    data_quality: dict[str, Any],
    labels: pd.DataFrame,
    feature_dataset: pd.DataFrame,
    feature_predictiveness: pd.DataFrame,
    scored: pd.DataFrame,
    gates: dict[str, Any],
    warnings: list[str],
    trend_summary: dict[str, Any],
) -> dict[str, Any]:
    """Build etc_v1_summary.json payload."""

    best_features = dataframe_records(feature_predictiveness, limit=10)
    label_counts = labels["label"].value_counts().to_dict() if not labels.empty else {}
    return {
        "hypothesis_name": "Early Trend Classifier",
        "version": "v1.0",
        "status": "research_only",
        "phase": "feature_discovery_and_event_study",
        "data_start": data_range.start.isoformat(),
        "data_end_exclusive": data_range.end_exclusive.isoformat(),
        "timezone": data_range.timezone_name,
        "symbols": symbols,
        "timeframes": timeframes,
        "data_ready": bool(data_quality.get("market_data_complete")),
        "funding_data_complete": bool(data_quality.get("funding_data_complete")),
        "labels_generated": bool(not labels.empty),
        "label_count": int(len(labels.index)),
        "label_counts": label_counts,
        "early_trend_count": int(labels["label"].isin(["early_uptrend", "early_downtrend"]).sum()) if not labels.empty else 0,
        "nontrend_count": int(labels["label"].eq("nontrend").sum()) if not labels.empty else 0,
        "feature_dataset_generated": bool(not feature_dataset.empty),
        "feature_count": int(len(REQUIRED_FEATURE_COLUMNS)),
        "best_single_features": best_features,
        "composite_score_available": bool(not scored.empty and scored["composite_score_available"].fillna(False).any()),
        "composite_terms_used": scored.attrs.get("composite_terms_used", []) if hasattr(scored, "attrs") else [],
        "top_score_event_pass": bool(gates.get("top_score_event_pass")),
        "random_control_pass": bool(gates.get("random_control_pass")),
        "reverse_test_pass": bool(gates.get("reverse_test_pass")),
        "cost_aware_pass": bool(gates.get("cost_aware_pass")),
        "funding_adjusted_pass": bool(gates.get("funding_adjusted_pass")),
        "concentration_pass": bool(gates.get("concentration_pass")),
        "can_enter_phase2": bool(gates.get("can_enter_phase2")),
        "strategy_development_allowed": False,
        "demo_live_allowed": False,
        "no_policy_can_be_traded": True,
        "final_decision": gates.get("final_decision", "postmortem_or_pause"),
        "recommended_next_step": gates.get("recommended_next_step", "postmortem_or_pause"),
        "trend_opportunity_summary": trend_summary,
        "data_quality": data_quality,
        "warnings": warnings,
    }


def render_report(
    summary: dict[str, Any],
    feature_predictiveness: pd.DataFrame,
    score_bucket_analysis: pd.DataFrame,
    event_summary: pd.DataFrame,
    reverse_test: pd.DataFrame,
    random_control: pd.DataFrame,
    concentration: pd.DataFrame,
) -> str:
    """Render ETC-v1 Markdown report with required answers."""

    best_feature_rows = dataframe_records(feature_predictiveness, limit=12)
    score_rows = dataframe_records(score_bucket_analysis, limit=30)
    top_events = event_summary[event_summary["group"].isin(["A", "B", "C"])] if not event_summary.empty else pd.DataFrame()
    event_rows = dataframe_records(top_events, limit=30)
    return (
        "# ETC-v1 Early Trend Classifier Feature Discovery\n\n"
        "## 1. ETC-v1 是什么研究？\n"
        "ETC-v1 是 research-only 的趋势早期识别特征发现研究。它使用 Trend Opportunity Map 的趋势段生成 ex-post label，"
        "再检验入场前可见 closed-bar 特征是否能提高未来进入 early trend 的概率。\n\n"
        "## 2. 趋势早期标签如何定义？\n"
        "趋势段前 25% 标记为 early_uptrend 或 early_downtrend，25%-75% 标记为 middle_trend，75% 之后标记为 late_trend。"
        "不在趋势段且未来窗口内没有有效 2ATR/3ATR 机会的样本标记为 nontrend；重叠、方向冲突、边界、数据不足和未来窗口不足标记为 excluded_ambiguous。\n\n"
        "## 3. 哪些特征完全使用入场前信息？\n"
        "所有 feature columns 均来自当前 timestamp 已完成 closed bar 及其历史 rolling 窗口，包括效率、广度、相对强弱、波动、funding、回撤位置和成交结构。"
        "未来收益、future MFE/MAE、trend_segment_end 和趋势标签没有进入特征计算。\n\n"
        "## 4. 哪些特征最能区分 early trend 和 nontrend？\n"
        + markdown_table(
            best_feature_rows,
            [
                "feature",
                "feature_predictiveness_score",
                "monotonicity_score",
                "train_validation_consistency",
                "oos_consistency",
            ],
        )
        + "\n\n"
        "## 5. 是否存在 train / validation / oos 一致的单特征？\n"
        f"- consistent_feature_count={int(sum(bool(row.get('train_validation_consistency')) and bool(row.get('oos_consistency')) for row in best_feature_rows))}\n\n"
        "## 6. composite score 是否优于 random control？\n"
        + markdown_table(
            score_rows,
            [
                "split",
                "score_bucket",
                "sample_count",
                "early_trend_rate",
                "future_3atr_hit_rate",
                "random_control_comparison",
            ],
        )
        + "\n\n"
        "## 7. top-score events 是否能在 train / validation / oos 中产生正收益？\n"
        + markdown_table(
            event_rows,
            [
                "group",
                "event_group",
                "hold",
                "split",
                "event_count",
                "early_trend_rate",
                "no_cost_pnl",
                "cost_aware_pnl",
                "funding_adjusted_pnl",
            ],
        )
        + "\n\n"
        "## 8. cost-aware 和 funding-adjusted 是否通过？\n"
        f"- cost_aware_pass={str(bool(summary.get('cost_aware_pass'))).lower()}\n"
        f"- funding_adjusted_pass={str(bool(summary.get('funding_adjusted_pass'))).lower()}\n\n"
        "## 9. reverse test 是否弱于正向？\n"
        + markdown_table(dataframe_records(reverse_test, limit=16), ["hold", "split", "forward_no_cost_pnl", "reverse_no_cost_pnl", "reverse_weaker"])
        + "\n\n"
        "## 10. 收益是否集中在单一 symbol 或 top trades？\n"
        + markdown_table(
            dataframe_records(concentration, limit=20),
            [
                "group",
                "hold",
                "split",
                "trade_count",
                "largest_symbol_pnl_share",
                "largest_symbol_event_share",
                "top_5pct_trade_pnl_contribution",
                "concentration_pass",
            ],
        )
        + "\n\n"
        "## 11. 是否允许进入 Phase 2？\n"
        f"- can_enter_phase2={str(bool(summary.get('can_enter_phase2'))).lower()}\n"
        f"- final_decision={summary.get('final_decision')}\n"
        f"- recommended_next_step={summary.get('recommended_next_step')}\n\n"
        "## 12. 是否允许修改正式策略？\n"
        "- strategy_development_allowed=false\n\n"
        "## 13. 是否允许 demo/live？\n"
        "- demo_live_allowed=false\n\n"
        "## Controls\n"
        + markdown_table(
            dataframe_records(random_control, limit=20),
            [
                "group",
                "hold",
                "split",
                "positive_early_trend_rate",
                "random_early_trend_rate",
                "positive_no_cost_pnl",
                "random_no_cost_pnl",
                "random_weaker",
            ],
        )
    )


def write_outputs(
    output_dir: Path,
    *,
    data_quality: dict[str, Any],
    labels: pd.DataFrame,
    feature_dataset: pd.DataFrame,
    feature_bucket_analysis: pd.DataFrame,
    feature_predictiveness: pd.DataFrame,
    scored: pd.DataFrame,
    score_bucket_analysis: pd.DataFrame,
    events: pd.DataFrame,
    trades: pd.DataFrame,
    event_summary: pd.DataFrame,
    by_symbol: pd.DataFrame,
    by_timeframe: pd.DataFrame,
    by_split: pd.DataFrame,
    concentration: pd.DataFrame,
    reverse_test: pd.DataFrame,
    random_control: pd.DataFrame,
    funding_summary: pd.DataFrame,
    rejected_reasons: pd.DataFrame,
    summary: dict[str, Any],
) -> None:
    """Write all ETC-v1 artifacts."""

    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "etc_v1_summary.json", summary)
    write_json(output_dir / "data_quality.json", data_quality)
    write_dataframe(output_dir / "early_trend_labels.csv", labels, LABEL_COLUMNS)
    write_dataframe(output_dir / "early_trend_feature_dataset.csv", feature_dataset)
    write_dataframe(output_dir / "feature_bucket_analysis.csv", feature_bucket_analysis)
    write_dataframe(output_dir / "feature_predictiveness.csv", feature_predictiveness)
    write_dataframe(output_dir / "composite_score_dataset.csv", scored)
    write_dataframe(output_dir / "score_bucket_analysis.csv", score_bucket_analysis)
    write_dataframe(output_dir / "early_trend_events.csv", events)
    write_dataframe(output_dir / "early_trend_event_trades.csv", trades)
    write_dataframe(output_dir / "early_trend_event_summary.csv", event_summary)
    write_dataframe(output_dir / "by_symbol.csv", by_symbol)
    write_dataframe(output_dir / "by_timeframe.csv", by_timeframe)
    write_dataframe(output_dir / "by_split.csv", by_split)
    write_dataframe(output_dir / "concentration.csv", concentration)
    write_dataframe(output_dir / "reverse_test.csv", reverse_test)
    write_dataframe(output_dir / "random_control.csv", random_control)
    write_dataframe(output_dir / "funding_summary.csv", funding_summary)
    write_dataframe(output_dir / "rejected_reasons.csv", rejected_reasons)
    report = render_report(summary, feature_predictiveness, score_bucket_analysis, event_summary, reverse_test, random_control, concentration)
    (output_dir / "etc_v1_report.md").write_text(report, encoding="utf-8")


def load_reference_inputs(trend_map_dir: Path, warnings: list[str]) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    """Read required and optional upstream research artifacts."""

    segments = read_csv_if_exists(trend_map_dir / "trend_segments.csv", warnings)
    trend_summary = read_json_if_exists(trend_map_dir / "trend_opportunity_summary.json", warnings)
    trend_quality = read_json_if_exists(trend_map_dir / "data_quality.json", warnings)
    optional_paths = [
        PROJECT_ROOT / "reports" / "research" / "trend_entry_timing" / "candidate_entry_events.csv",
        PROJECT_ROOT / "reports" / "research" / "trend_entry_timing" / "candidate_entry_trade_tests.csv",
        PROJECT_ROOT / "reports" / "research" / "trend_capture_exit_convexity" / "trend_capture_diagnostics.csv",
        PROJECT_ROOT / "reports" / "research" / "research_decision_dossier" / "research_decision_dossier.json",
    ]
    for path in optional_paths:
        if path.suffix == ".json":
            read_json_if_exists(path, warnings)
        else:
            read_csv_if_exists(path, warnings)
    return segments, trend_summary, trend_quality


def run_research(
    *,
    symbols: list[str],
    timeframes: list[str],
    data_range: HistoryRange,
    trend_map_dir: Path,
    funding_dir: Path,
    output_dir: Path,
    database_path: Path,
    config: EtcConfig,
    data_check_strict: bool,
    logger: logging.Logger | None = None,
    bars_by_symbol: dict[str, pd.DataFrame] | None = None,
) -> dict[str, Any]:
    """Run ETC-v1 research end to end."""

    warnings: list[str] = []
    raw_segments, trend_summary, trend_map_quality = load_reference_inputs(trend_map_dir, warnings)
    segments = normalize_trend_segments(raw_segments, data_range.timezone_name)
    funding_histories, funding_quality = load_funding_histories(
        funding_dir,
        symbols,
        data_range,
        data_range.start.date().isoformat(),
        data_range.end_display.date().isoformat(),
        warnings,
    )
    splits = build_time_splits(data_range.timezone_name)
    bars_by_key: dict[tuple[str, str], pd.DataFrame] = {}
    symbol_quality: dict[str, Any] = {}
    label_frames: list[pd.DataFrame] = []
    for symbol in symbols:
        if logger:
            log_event(logger, logging.INFO, "load_symbol", "Loading symbol market data", symbol=symbol)
        if bars_by_symbol is not None and symbol in bars_by_symbol:
            bars_1m = normalize_1m_bars(bars_by_symbol[symbol], data_range.timezone_name)
            bars_1m = bars_1m[(bars_1m["datetime"] >= pd.Timestamp(data_range.start)) & (bars_1m["datetime"] < pd.Timestamp(data_range.end_exclusive))]
        else:
            bars_1m = load_1m_bars_from_sqlite(symbol, data_range, database_path)
        one_minute_quality = analyze_1m_quality(symbol, bars_1m, data_range)
        symbol_quality[symbol] = {"1m": one_minute_quality}
        if bars_1m.empty:
            raise EtcResearchError(f"required symbol has no 1m data: {symbol}")
        if data_check_strict and not one_minute_quality["complete"]:
            raise EtcResearchError(f"strict data check failed for {symbol}: {one_minute_quality}")
        for timeframe in timeframes:
            resampled = resample_ohlcv_closed(bars_1m, timeframe, data_range)
            quality = analyze_timeframe_quality(symbol, timeframe, resampled, one_minute_quality)
            symbol_quality[symbol][timeframe] = quality
            if data_check_strict and not quality["complete"]:
                raise EtcResearchError(f"strict resample data check failed for {symbol} {timeframe}: {quality}")
            bars_by_key[(symbol, timeframe)] = resampled
            label_frames.append(
                build_early_trend_labels_for_frame(
                    resampled,
                    symbol,
                    timeframe,
                    segments,
                    future_window_bars=config.future_nontrend_window_bars,
                )
            )
    labels = pd.concat(label_frames, ignore_index=True) if label_frames else pd.DataFrame(columns=LABEL_COLUMNS)
    feature_dataset = build_feature_dataset(
        bars_by_key,
        labels,
        funding_histories,
        symbols,
        timeframes,
        splits,
        bool(funding_quality.get("funding_data_complete")),
        warnings,
    )
    feature_bucket_analysis, feature_predictiveness = build_feature_bucket_analysis(feature_dataset)
    scored = compute_composite_score(feature_dataset, warnings)
    score_bucket_analysis = build_score_bucket_analysis(scored, config)
    events = generate_top_score_events(scored, bars_by_key, config)
    trades = simulate_event_trades(events, bars_by_key, funding_histories, config)
    event_summary = build_event_summary(events, trades)
    concentration = build_concentration_summary(trades)
    random_control = build_random_control(event_summary)
    reverse_test = build_reverse_test(event_summary)
    gates, rejected_reasons = evaluate_phase_gates(
        event_summary,
        random_control,
        reverse_test,
        concentration,
        bool(funding_quality.get("funding_data_complete")),
    )
    by_symbol, by_timeframe, by_split = build_summary_tables(trades)
    funding_summary = build_funding_summary(trades, funding_quality)
    data_quality = build_data_quality_payload(
        symbols,
        timeframes,
        data_range,
        symbol_quality,
        funding_quality,
        trend_map_quality,
        data_check_strict,
    )
    summary = build_summary_payload(
        symbols=symbols,
        timeframes=timeframes,
        data_range=data_range,
        data_quality=data_quality,
        labels=labels,
        feature_dataset=feature_dataset,
        feature_predictiveness=feature_predictiveness,
        scored=scored,
        gates=gates,
        warnings=warnings,
        trend_summary=trend_summary,
    )
    write_outputs(
        output_dir,
        data_quality=data_quality,
        labels=labels,
        feature_dataset=feature_dataset,
        feature_bucket_analysis=feature_bucket_analysis,
        feature_predictiveness=feature_predictiveness,
        scored=scored,
        score_bucket_analysis=score_bucket_analysis,
        events=events,
        trades=trades,
        event_summary=event_summary,
        by_symbol=by_symbol,
        by_timeframe=by_timeframe,
        by_split=by_split,
        concentration=concentration,
        reverse_test=reverse_test,
        random_control=random_control,
        funding_summary=funding_summary,
        rejected_reasons=rejected_reasons,
        summary=summary,
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""

    args = parse_args(argv)
    ensure_headless_runtime()
    logger = setup_logging("research_early_trend_classifier_v1", verbose=bool(args.verbose))
    symbols = parse_csv_list(args.symbols)
    if not symbols:
        raise EtcResearchError("--symbols must not be empty")
    timeframes = parse_timeframes(args.timeframes)
    data_range = resolve_history_range(args.start, args.end, args.timezone)
    config = EtcConfig(
        fee_bps_per_side=float(args.fee_bps_per_side),
        slippage_bps_per_side=float(args.slippage_bps_per_side),
    )
    summary = run_research(
        symbols=symbols,
        timeframes=timeframes,
        data_range=data_range,
        trend_map_dir=resolve_path(args.trend_map_dir),
        funding_dir=resolve_path(args.funding_dir),
        output_dir=resolve_path(args.output_dir),
        database_path=resolve_path(args.database_path),
        config=config,
        data_check_strict=bool(args.data_check_strict),
        logger=logger,
    )
    log_event(
        logger,
        logging.INFO,
        "research_complete",
        "ETC-v1 research complete",
        final_decision=summary.get("final_decision"),
        can_enter_phase2=summary.get("can_enter_phase2"),
        strategy_development_allowed=False,
        demo_live_allowed=False,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
