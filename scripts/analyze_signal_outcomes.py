#!/usr/bin/env python3
"""Analyze post-entry signal outcomes with MFE/MAE diagnostics."""

from __future__ import annotations

import argparse
import json
import logging
from datetime import timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from common_runtime import PROJECT_ROOT, ensure_headless_runtime, log_event, print_json_block, setup_logging, to_jsonable
from history_time_utils import DEFAULT_TIMEZONE
from history_utils import get_database_timezone


DEFAULT_HORIZONS: str = "5,15,30,60,120"
SIGNAL_TRACE_COLUMNS: list[str] = [
    "signal_id",
    "datetime",
    "vt_symbol",
    "direction",
    "action",
    "price",
    "close_1m",
    "donchian_high",
    "donchian_low",
    "breakout_distance",
    "breakout_distance_atr",
    "atr_1m",
    "atr_pct",
    "rsi",
    "fast_ema_5m",
    "slow_ema_5m",
    "ema_spread",
    "ema_spread_pct",
    "regime",
    "regime_persistence_count",
    "hour",
    "weekday",
    "is_weekend",
    "filter_reject_reason",
    "position_before",
    "volume",
    "stop_price",
    "take_profit_price",
    "trail_stop_price",
]
BREAKOUT_BUCKETS: list[tuple[float, str]] = [
    (0.0, "<=0"),
    (0.25, "0-0.25"),
    (0.5, "0.25-0.5"),
    (1.0, "0.5-1"),
    (2.0, "1-2"),
    (float("inf"), ">2"),
]


class SignalOutcomeError(Exception):
    """Raised when signal outcome analysis cannot continue."""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description="Analyze signal-level future returns, MFE, and MAE.")
    parser.add_argument(
        "--report-dir",
        required=True,
        help="Backtest report directory containing signal_trace.csv.",
    )
    parser.add_argument(
        "--signal-trace",
        help="Signal trace CSV. Default: <report-dir>/signal_trace.csv.",
    )
    parser.add_argument(
        "--bars-from-db",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Load 1m bars from vn.py sqlite. Default: enabled.",
    )
    parser.add_argument(
        "--horizons",
        default=DEFAULT_HORIZONS,
        help=f"Comma-separated horizons in minutes. Default: {DEFAULT_HORIZONS}.",
    )
    parser.add_argument(
        "--timezone",
        default=DEFAULT_TIMEZONE,
        help=f"Timezone used to align signal and bar times. Default: {DEFAULT_TIMEZONE}.",
    )
    parser.add_argument(
        "--output-dir",
        help="Output directory. Default: <report-dir>/signal_outcomes.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print outcome_summary.json payload to stdout after generation.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose structured logs.",
    )
    return parser.parse_args(argv)


def resolve_path(path_arg: str | None, default_path: Path | None = None) -> Path:
    """Resolve CLI paths relative to project root."""

    if path_arg:
        path = Path(path_arg)
    elif default_path is not None:
        path = default_path
    else:
        raise SignalOutcomeError("缺少路径参数且没有默认值")

    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def parse_horizons(raw_value: str) -> list[int]:
    """Parse comma-separated positive minute horizons."""

    horizons: set[int] = set()
    for token in str(raw_value or "").split(","):
        text = token.strip()
        if not text:
            continue
        try:
            value = int(text)
        except ValueError as exc:
            raise SignalOutcomeError(f"--horizons 包含非法整数: {text!r}") from exc
        if value <= 0:
            raise SignalOutcomeError(f"--horizons 必须为正整数分钟: {value}")
        horizons.add(value)

    if not horizons:
        raise SignalOutcomeError("--horizons 不能为空")
    return sorted(horizons)


def number_or_none(value: Any) -> float | None:
    """Return a finite float or None."""

    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number):
        return None
    return number


def normalize_direction(value: Any) -> str:
    """Normalize signal direction."""

    text = str(value).strip().lower()
    if text in {"long", "多", "buy"}:
        return "long"
    if text in {"short", "空", "sell"}:
        return "short"
    return "unknown"


def normalize_bool(value: Any) -> bool | None:
    """Normalize bool-ish CSV values."""

    if pd.isna(value):
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    return None


def normalize_datetime_series(series: pd.Series, timezone_name: str) -> pd.Series:
    """Parse datetimes and localize/convert to the configured timezone."""

    timezone = ZoneInfo(timezone_name)
    parsed = pd.to_datetime(series, errors="coerce")
    if parsed.empty:
        return parsed

    try:
        if parsed.dt.tz is None:
            return parsed.dt.tz_localize(timezone)
        return parsed.dt.tz_convert(timezone)
    except Exception:
        parsed_utc = pd.to_datetime(series, errors="coerce", utc=True)
        return parsed_utc.dt.tz_convert(timezone)


def read_signal_trace(path: Path, timezone_name: str) -> pd.DataFrame:
    """Read and normalize signal_trace.csv."""

    if not path.exists():
        raise SignalOutcomeError(f"signal_trace.csv 不存在: {path}")
    if not path.is_file():
        raise SignalOutcomeError(f"signal_trace 路径不是文件: {path}")

    try:
        trace_df = pd.read_csv(path)
    except Exception as exc:
        raise SignalOutcomeError(f"读取 signal_trace.csv 失败: {path} | {exc!r}") from exc

    for column in SIGNAL_TRACE_COLUMNS:
        if column not in trace_df.columns:
            trace_df[column] = None

    if trace_df.empty:
        return trace_df

    trace_df = trace_df.copy()
    trace_df["action"] = trace_df["action"].fillna("entry").astype(str).str.strip().str.lower()
    trace_df["direction"] = trace_df["direction"].map(normalize_direction)
    trace_df["_signal_dt"] = normalize_datetime_series(trace_df["datetime"], timezone_name)

    numeric_columns = [
        "price",
        "close_1m",
        "donchian_high",
        "donchian_low",
        "breakout_distance",
        "breakout_distance_atr",
        "atr_1m",
        "atr_pct",
        "rsi",
        "fast_ema_5m",
        "slow_ema_5m",
        "ema_spread",
        "ema_spread_pct",
        "regime_persistence_count",
        "hour",
        "weekday",
        "position_before",
        "volume",
        "stop_price",
        "take_profit_price",
        "trail_stop_price",
    ]
    for column in numeric_columns:
        trace_df[column] = pd.to_numeric(trace_df[column], errors="coerce")

    trace_df["is_weekend"] = trace_df["is_weekend"].map(normalize_bool)
    missing_weekend = trace_df["is_weekend"].isna() & trace_df["_signal_dt"].notna()
    trace_df.loc[missing_weekend, "is_weekend"] = trace_df.loc[missing_weekend, "_signal_dt"].dt.weekday >= 5
    missing_hour = trace_df["hour"].isna() & trace_df["_signal_dt"].notna()
    trace_df.loc[missing_hour, "hour"] = trace_df.loc[missing_hour, "_signal_dt"].dt.hour
    missing_weekday = trace_df["weekday"].isna() & trace_df["_signal_dt"].notna()
    trace_df.loc[missing_weekday, "weekday"] = trace_df.loc[missing_weekday, "_signal_dt"].dt.weekday
    return trace_df


def prepare_entry_signals(trace_df: pd.DataFrame, warnings: list[str]) -> pd.DataFrame:
    """Return valid entry rows from the signal trace."""

    if trace_df.empty:
        warnings.append("signal_trace.csv 为空，没有 entry signal 可分析")
        return trace_df.copy()

    entry_df = trace_df[trace_df["action"] == "entry"].copy()
    if entry_df.empty:
        warnings.append("signal_trace.csv 没有 action=entry 的记录")
        return entry_df

    invalid_dt = int(entry_df["_signal_dt"].isna().sum())
    if invalid_dt:
        warnings.append(f"忽略 datetime 无法解析的 entry signal 数: {invalid_dt}")
    entry_df = entry_df.dropna(subset=["_signal_dt"]).copy()

    entry_df["price"] = pd.to_numeric(entry_df["price"], errors="coerce")
    missing_price = entry_df["price"].isna() | (entry_df["price"] <= 0)
    if missing_price.any():
        entry_df.loc[missing_price, "price"] = pd.to_numeric(
            entry_df.loc[missing_price, "close_1m"],
            errors="coerce",
        )
    invalid_price = int((entry_df["price"].isna() | (entry_df["price"] <= 0)).sum())
    if invalid_price:
        warnings.append(f"忽略 price/close_1m 无效的 entry signal 数: {invalid_price}")
    entry_df = entry_df[entry_df["price"].notna() & (entry_df["price"] > 0)].copy()

    invalid_direction = int((~entry_df["direction"].isin(["long", "short"])).sum())
    if invalid_direction:
        warnings.append(f"忽略 direction 无法识别的 entry signal 数: {invalid_direction}")
    entry_df = entry_df[entry_df["direction"].isin(["long", "short"])].copy()
    return entry_df.sort_values("_signal_dt", kind="stable").reset_index(drop=True)


def configure_sqlite_settings(logger: logging.Logger) -> None:
    """Force vn.py to use the project-local sqlite database."""

    from vnpy.trader.setting import SETTINGS

    SETTINGS["database.name"] = "sqlite"
    SETTINGS["database.database"] = "database.db"
    log_event(
        logger,
        logging.INFO,
        "signal_outcomes.db_settings",
        "Configured vn.py database settings for sqlite",
        database_name=SETTINGS["database.name"],
        database_database=SETTINGS["database.database"],
    )


def split_vt_symbol(vt_symbol: str) -> tuple[str, str]:
    """Split vt_symbol into vn.py symbol/exchange strings."""

    symbol, separator, exchange = str(vt_symbol).partition(".")
    if not separator or not symbol or not exchange:
        raise SignalOutcomeError(f"无法解析 vt_symbol: {vt_symbol!r}")
    return symbol, exchange


def resolve_exchange(exchange_value: str) -> Any:
    """Resolve vn.py Exchange from either enum value or name."""

    from vnpy.trader.constant import Exchange

    try:
        return Exchange(exchange_value)
    except ValueError:
        try:
            return Exchange[exchange_value]
        except KeyError as exc:
            raise SignalOutcomeError(f"无法解析 exchange: {exchange_value!r}") from exc


def normalize_bar_timestamp(value: Any, timezone_name: str) -> pd.Timestamp:
    """Normalize one bar timestamp into the analysis timezone."""

    timestamp = pd.Timestamp(value)
    timezone = ZoneInfo(timezone_name)
    if timestamp.tzinfo is None:
        return timestamp.tz_localize(timezone)
    return timestamp.tz_convert(timezone)


def bars_to_dataframe(bars: list[Any], timezone_name: str) -> pd.DataFrame:
    """Convert vn.py BarData objects into an OHLC DataFrame."""

    records: list[dict[str, Any]] = []
    for bar in bars:
        records.append(
            {
                "datetime": normalize_bar_timestamp(getattr(bar, "datetime", None), timezone_name),
                "open": number_or_none(getattr(bar, "open_price", None)),
                "high": number_or_none(getattr(bar, "high_price", None)),
                "low": number_or_none(getattr(bar, "low_price", None)),
                "close": number_or_none(getattr(bar, "close_price", None)),
                "volume": number_or_none(getattr(bar, "volume", None)),
            }
        )

    bar_df = pd.DataFrame(records, columns=["datetime", "open", "high", "low", "close", "volume"])
    if bar_df.empty:
        return bar_df
    bar_df = bar_df.dropna(subset=["datetime", "high", "low", "close"]).copy()
    bar_df = bar_df.sort_values("datetime", kind="stable").drop_duplicates("datetime", keep="last")
    return bar_df.reset_index(drop=True)


def dataframe_bars_to_ohlc(bars_df: pd.DataFrame, timezone_name: str) -> pd.DataFrame:
    """Normalize a test/user-provided bars DataFrame."""

    required = ["datetime", "high", "low", "close"]
    missing = [column for column in required if column not in bars_df.columns]
    if missing:
        raise SignalOutcomeError(f"bars DataFrame 缺少列: {', '.join(missing)}")

    result_df = bars_df.copy()
    result_df["datetime"] = normalize_datetime_series(result_df["datetime"], timezone_name)
    for column in ["open", "high", "low", "close", "volume"]:
        if column in result_df.columns:
            result_df[column] = pd.to_numeric(result_df[column], errors="coerce")
        else:
            result_df[column] = np.nan
    result_df = result_df.dropna(subset=["datetime", "high", "low", "close"]).copy()
    result_df = result_df.sort_values("datetime", kind="stable").drop_duplicates("datetime", keep="last")
    return result_df.reset_index(drop=True)[["datetime", "open", "high", "low", "close", "volume"]]


def load_bars_from_db(
    entry_df: pd.DataFrame,
    horizons: list[int],
    timezone_name: str,
    warnings: list[str],
    logger: logging.Logger,
) -> pd.DataFrame:
    """Load future 1m bars for all entry signals from vn.py sqlite."""

    if entry_df.empty:
        return pd.DataFrame(columns=["datetime", "open", "high", "low", "close", "volume"])

    vt_symbols = [str(value).strip() for value in entry_df["vt_symbol"].dropna().unique() if str(value).strip()]
    if not vt_symbols:
        raise SignalOutcomeError("entry signal 缺少 vt_symbol，无法从数据库读取 1m bar")
    if len(vt_symbols) > 1:
        warnings.append(f"signal_trace 包含多个 vt_symbol，仅使用第一个读取 bars: {vt_symbols[0]}")

    symbol, exchange_value = split_vt_symbol(vt_symbols[0])
    exchange = resolve_exchange(exchange_value)
    from vnpy.trader.constant import Interval
    from vnpy.trader.database import get_database

    configure_sqlite_settings(logger)
    db_tz = get_database_timezone()
    start = entry_df["_signal_dt"].min().to_pydatetime().astimezone(db_tz).replace(tzinfo=None)
    end = (
        entry_df["_signal_dt"].max()
        + pd.Timedelta(minutes=max(horizons) + 1)
    ).to_pydatetime().astimezone(db_tz).replace(tzinfo=None)

    bars = get_database().load_bar_data(symbol, exchange, Interval.MINUTE, start, end)
    if not bars:
        warnings.append(
            f"数据库没有返回 1m bars: vt_symbol={vt_symbols[0]}, start={start}, end={end}"
        )
    return bars_to_dataframe(list(bars), timezone_name)


def compute_excursions(
    window_df: pd.DataFrame,
    direction: str,
    entry_price: float,
) -> tuple[float | None, float | None, float | None, float | None]:
    """Compute MFE/MAE as fractional returns and raw price moves."""

    if window_df.empty or entry_price <= 0:
        return None, None, None, None

    high = pd.to_numeric(window_df["high"], errors="coerce")
    low = pd.to_numeric(window_df["low"], errors="coerce")
    if direction == "long":
        favorable_move = float((high - entry_price).max())
        adverse_move = float((entry_price - low).max())
    else:
        favorable_move = float((entry_price - low).max())
        adverse_move = float((high - entry_price).max())

    favorable_move = max(favorable_move, 0.0)
    adverse_move = max(adverse_move, 0.0)
    return (
        favorable_move / entry_price,
        adverse_move / entry_price,
        favorable_move,
        adverse_move,
    )


def compute_future_return(
    window_df: pd.DataFrame,
    direction: str,
    entry_price: float,
) -> float | None:
    """Compute direction-adjusted future return from the last close in a window."""

    if window_df.empty or entry_price <= 0:
        return None

    future_close = number_or_none(window_df["close"].iloc[-1])
    if future_close is None:
        return None
    if direction == "long":
        return (future_close - entry_price) / entry_price
    return (entry_price - future_close) / entry_price


def compute_extreme_times(
    window_df: pd.DataFrame,
    direction: str,
    entry_dt: pd.Timestamp,
    entry_price: float,
) -> tuple[float | None, float | None]:
    """Return minutes to max favorable and max adverse excursion."""

    if window_df.empty:
        return None, None

    working_df = window_df.copy()
    if direction == "long":
        working_df["_favorable"] = pd.to_numeric(working_df["high"], errors="coerce") - entry_price
        working_df["_adverse"] = entry_price - pd.to_numeric(working_df["low"], errors="coerce")
    else:
        working_df["_favorable"] = entry_price - pd.to_numeric(working_df["low"], errors="coerce")
        working_df["_adverse"] = pd.to_numeric(working_df["high"], errors="coerce") - entry_price

    favorable_time = None
    adverse_time = None
    if not working_df["_favorable"].dropna().empty:
        fav_idx = working_df["_favorable"].idxmax()
        favorable_time = (
            pd.Timestamp(working_df.loc[fav_idx, "datetime"]) - entry_dt
        ).total_seconds() / 60.0
    if not working_df["_adverse"].dropna().empty:
        adv_idx = working_df["_adverse"].idxmax()
        adverse_time = (
            pd.Timestamp(working_df.loc[adv_idx, "datetime"]) - entry_dt
        ).total_seconds() / 60.0
    return favorable_time, adverse_time


def compute_stop_take_profit_order(
    window_df: pd.DataFrame,
    direction: str,
    stop_price: float | None,
    take_profit_price: float | None,
    warnings: list[str],
) -> tuple[bool | None, bool | None]:
    """Determine whether TP or stop is hit first within the max horizon window."""

    if window_df.empty or stop_price is None or take_profit_price is None:
        return None, None

    for row in window_df.itertuples(index=False):
        high = number_or_none(getattr(row, "high", None))
        low = number_or_none(getattr(row, "low", None))
        if high is None or low is None:
            continue

        if direction == "long":
            stop_hit = low <= stop_price
            tp_hit = high >= take_profit_price
        else:
            stop_hit = high >= stop_price
            tp_hit = low <= take_profit_price

        if stop_hit and tp_hit:
            warning = "同一根 1m bar 同时触发 stop/take-profit，保守按 stop first 处理"
            if warning not in warnings:
                warnings.append(warning)
            return False, True
        if tp_hit:
            return True, False
        if stop_hit:
            return False, True

    return False, False


def base_outcome_columns(horizons: list[int]) -> list[str]:
    """Build stable signal_outcomes.csv columns."""

    columns = list(SIGNAL_TRACE_COLUMNS)
    for horizon in horizons:
        columns.extend([f"future_return_{horizon}m", f"mfe_{horizon}m", f"mae_{horizon}m"])
    columns.extend(
        [
            "mfe_atr",
            "mae_atr",
            "hit_take_profit_before_stop",
            "hit_stop_before_take_profit",
            "max_favorable_time",
            "max_adverse_time",
            "breakout_distance_bucket",
        ]
    )
    return columns


def compute_signal_outcomes(
    entry_df: pd.DataFrame,
    bars_df: pd.DataFrame,
    horizons: list[int],
    warnings: list[str],
) -> pd.DataFrame:
    """Compute future returns and MFE/MAE for each entry signal."""

    output_columns = base_outcome_columns(horizons)
    if entry_df.empty:
        return pd.DataFrame(columns=output_columns)
    if bars_df.empty:
        warnings.append("1m bars 为空，无法计算任何 signal outcome")
        return pd.DataFrame(columns=output_columns)

    bars = bars_df.sort_values("datetime", kind="stable").reset_index(drop=True)
    bar_times = pd.Series(bars["datetime"])
    max_bar_time = pd.Timestamp(bar_times.iloc[-1])
    max_horizon = max(horizons)
    horizon_exceeded_counts: dict[int, int] = {horizon: 0 for horizon in horizons}
    records: list[dict[str, Any]] = []

    for row in entry_df.to_dict(orient="records"):
        entry_dt = pd.Timestamp(row["_signal_dt"])
        direction = str(row.get("direction"))
        entry_price = float(row.get("price"))
        start_index = int(bar_times.searchsorted(entry_dt, side="right"))

        record = {column: row.get(column) for column in SIGNAL_TRACE_COLUMNS}
        record["datetime"] = entry_dt.isoformat()

        for horizon in horizons:
            target_dt = entry_dt + pd.Timedelta(minutes=horizon)
            if target_dt > max_bar_time:
                horizon_exceeded_counts[horizon] += 1
            end_index = int(bar_times.searchsorted(target_dt, side="right"))
            window = bars.iloc[start_index:end_index].copy()
            future_return = compute_future_return(window, direction, entry_price)
            mfe, mae, _mfe_move, _mae_move = compute_excursions(window, direction, entry_price)
            record[f"future_return_{horizon}m"] = future_return
            record[f"mfe_{horizon}m"] = mfe
            record[f"mae_{horizon}m"] = mae

        max_target_dt = entry_dt + pd.Timedelta(minutes=max_horizon)
        max_end_index = int(bar_times.searchsorted(max_target_dt, side="right"))
        max_window = bars.iloc[start_index:max_end_index].copy()
        _mfe, _mae, mfe_move, mae_move = compute_excursions(max_window, direction, entry_price)
        atr_1m = number_or_none(row.get("atr_1m"))
        record["mfe_atr"] = mfe_move / atr_1m if atr_1m not in (None, 0.0) and mfe_move is not None else None
        record["mae_atr"] = mae_move / atr_1m if atr_1m not in (None, 0.0) and mae_move is not None else None

        tp_first, stop_first = compute_stop_take_profit_order(
            max_window,
            direction,
            number_or_none(row.get("stop_price")),
            number_or_none(row.get("take_profit_price")),
            warnings,
        )
        record["hit_take_profit_before_stop"] = tp_first
        record["hit_stop_before_take_profit"] = stop_first
        fav_time, adv_time = compute_extreme_times(max_window, direction, entry_dt, entry_price)
        record["max_favorable_time"] = fav_time
        record["max_adverse_time"] = adv_time
        record["breakout_distance_bucket"] = breakout_distance_bucket(row.get("breakout_distance_atr"))
        records.append(record)

    for horizon, count in horizon_exceeded_counts.items():
        if count:
            warnings.append(f"horizon {horizon}m 超过可用 bar 数据范围的 entry 数: {count}")

    outcome_df = pd.DataFrame(records)
    for column in output_columns:
        if column not in outcome_df.columns:
            outcome_df[column] = None
    return outcome_df[output_columns]


def breakout_distance_bucket(value: Any) -> str:
    """Bucket breakout_distance_atr."""

    number = number_or_none(value)
    if number is None:
        return "unknown"
    for upper, label in BREAKOUT_BUCKETS:
        if number <= upper:
            return label
    return ">2"


def true_rate(series: pd.Series) -> float | None:
    """Return true-rate for bool-ish values."""

    clean = series.dropna()
    if clean.empty:
        return None
    return float(clean.astype(bool).mean())


def summarize_outcome_frame(df: pd.DataFrame, horizons: list[int]) -> dict[str, Any]:
    """Summarize one outcome slice."""

    summary: dict[str, Any] = {"entry_count": int(len(df.index))}
    if df.empty:
        for horizon in horizons:
            for metric in ["future_return", "mfe", "mae"]:
                summary[f"{metric}_{horizon}m_mean"] = None
                summary[f"{metric}_{horizon}m_median"] = None
            summary[f"future_return_{horizon}m_positive_rate"] = None
        summary["mfe_atr_median"] = None
        summary["mae_atr_median"] = None
        summary["take_profit_first_rate"] = None
        summary["stop_first_rate"] = None
        return summary

    for horizon in horizons:
        future_col = f"future_return_{horizon}m"
        mfe_col = f"mfe_{horizon}m"
        mae_col = f"mae_{horizon}m"
        for metric, column in [("future_return", future_col), ("mfe", mfe_col), ("mae", mae_col)]:
            series = pd.to_numeric(df[column], errors="coerce").dropna()
            summary[f"{metric}_{horizon}m_mean"] = float(series.mean()) if not series.empty else None
            summary[f"{metric}_{horizon}m_median"] = float(series.median()) if not series.empty else None
        future_series = pd.to_numeric(df[future_col], errors="coerce").dropna()
        summary[f"future_return_{horizon}m_positive_rate"] = (
            float((future_series > 0).mean()) if not future_series.empty else None
        )

    for column in ["mfe_atr", "mae_atr"]:
        series = pd.to_numeric(df[column], errors="coerce").dropna()
        summary[f"{column}_median"] = float(series.median()) if not series.empty else None
        summary[f"{column}_mean"] = float(series.mean()) if not series.empty else None

    summary["take_profit_first_rate"] = true_rate(df["hit_take_profit_before_stop"])
    summary["stop_first_rate"] = true_rate(df["hit_stop_before_take_profit"])
    return summary


def build_group_summary(
    outcome_df: pd.DataFrame,
    group_column: str,
    horizons: list[int],
    base_values: list[Any] | None = None,
) -> pd.DataFrame:
    """Build a summary DataFrame by one column."""

    rows: list[dict[str, Any]] = []
    if base_values is not None:
        iterable = [(value, outcome_df[outcome_df[group_column] == value]) for value in base_values]
    elif outcome_df.empty or group_column not in outcome_df.columns:
        iterable = []
    else:
        iterable = list(outcome_df.groupby(group_column, dropna=False))

    for value, group_df in iterable:
        row = {group_column: value}
        row.update(summarize_outcome_frame(group_df, horizons))
        rows.append(row)

    if not rows:
        return pd.DataFrame(columns=[group_column, "entry_count"])
    return pd.DataFrame(rows)


def build_weekday_summary(outcome_df: pd.DataFrame, horizons: list[int]) -> pd.DataFrame:
    """Build weekday summary with stable weekday rows."""

    weekday_df = build_group_summary(outcome_df, "weekday", horizons, base_values=list(range(7)))
    weekday_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    weekday_df.insert(1, "weekday_name", weekday_names[: len(weekday_df.index)])
    return weekday_df


def build_breakout_bucket_summary(outcome_df: pd.DataFrame, horizons: list[int]) -> pd.DataFrame:
    """Build breakout-distance bucket summary."""

    base_values = [label for _upper, label in BREAKOUT_BUCKETS] + ["unknown"]
    return build_group_summary(outcome_df, "breakout_distance_bucket", horizons, base_values=base_values)


def dataframe_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Convert a DataFrame to JSON-safe records."""

    if df.empty:
        return []
    safe_df = df.replace({np.nan: None})
    return json.loads(safe_df.to_json(orient="records", force_ascii=False))


def median_value(df: pd.DataFrame, column: str) -> float | None:
    """Return a finite median value from a DataFrame column."""

    if df.empty or column not in df.columns:
        return None
    series = pd.to_numeric(df[column], errors="coerce").dropna()
    if series.empty:
        return None
    return float(series.median())


def infer_trace_atr_multiple(outcome_df: pd.DataFrame, target_column: str) -> float | None:
    """Infer stop/take-profit ATR multiple from trace prices."""

    if outcome_df.empty or target_column not in outcome_df.columns:
        return None
    price = pd.to_numeric(outcome_df["price"], errors="coerce")
    target = pd.to_numeric(outcome_df[target_column], errors="coerce")
    atr = pd.to_numeric(outcome_df["atr_1m"], errors="coerce")
    multiple = (target - price).abs() / atr.replace(0, np.nan)
    multiple = multiple.replace([np.inf, -np.inf], np.nan).dropna()
    if multiple.empty:
        return None
    return float(multiple.median())


def compare_side_mfe(side_df: pd.DataFrame, horizon: int) -> dict[str, Any]:
    """Answer which side has worse MFE."""

    column = f"mfe_{horizon}m_median"
    values: dict[str, float] = {}
    for side in ["long", "short"]:
        row = side_df[side_df["direction"] == side]
        if not row.empty:
            value = number_or_none(row.iloc[0].get(column))
            if value is not None:
                values[side] = value

    if len(values) < 2:
        return {"worse_side": None, "basis": column, "values": values, "answer": "样本不足，无法比较多空 MFE"}

    worse_side = "long" if values["long"] < values["short"] else "short"
    return {
        "worse_side": worse_side,
        "basis": column,
        "values": values,
        "answer": f"{worse_side} 的 {horizon}m MFE 中位数更低",
    }


def continuation_by_horizon(outcome_df: pd.DataFrame, horizons: list[int]) -> dict[str, Any]:
    """Summarize forward continuation by horizon."""

    result: dict[str, Any] = {}
    for horizon in horizons:
        column = f"future_return_{horizon}m"
        median_return = median_value(outcome_df, column)
        positive_rate = None
        if column in outcome_df.columns:
            series = pd.to_numeric(outcome_df[column], errors="coerce").dropna()
            if not series.empty:
                positive_rate = float((series > 0).mean())
        result[f"{horizon}m"] = {
            "median_future_return": median_return,
            "positive_rate": positive_rate,
            "has_positive_continuation": bool(median_return is not None and median_return > 0),
        }
    return result


def assess_stop_take_profit(outcome_df: pd.DataFrame) -> dict[str, Any]:
    """Assess whether current stop/take-profit distance looks reachable."""

    stop_atr = infer_trace_atr_multiple(outcome_df, "stop_price")
    take_profit_atr = infer_trace_atr_multiple(outcome_df, "take_profit_price")
    median_mfe_atr = median_value(outcome_df, "mfe_atr")
    median_mae_atr = median_value(outcome_df, "mae_atr")
    tp_rate = true_rate(outcome_df["hit_take_profit_before_stop"]) if not outcome_df.empty else None
    stop_rate = true_rate(outcome_df["hit_stop_before_take_profit"]) if not outcome_df.empty else None

    if median_mfe_atr is None or take_profit_atr is None:
        answer = "缺少 ATR 或 take-profit trace，无法评估止盈距离"
    elif median_mfe_atr < take_profit_atr:
        answer = "当前 take_profit_atr 大于信号后 MFE 中位数，止盈对典型信号偏远"
    else:
        answer = "当前 take_profit_atr 在 MFE 中位数范围内，止盈距离并非主要障碍"

    return {
        "inferred_stop_atr": stop_atr,
        "inferred_take_profit_atr": take_profit_atr,
        "median_mfe_atr": median_mfe_atr,
        "median_mae_atr": median_mae_atr,
        "take_profit_first_rate": tp_rate,
        "stop_first_rate": stop_rate,
        "answer": answer,
    }


def assess_stop_vs_followthrough(outcome_df: pd.DataFrame, horizons: list[int]) -> dict[str, Any]:
    """Diagnose whether losses are more likely stop distance or entry follow-through."""

    stop_atr = infer_trace_atr_multiple(outcome_df, "stop_price")
    take_profit_atr = infer_trace_atr_multiple(outcome_df, "take_profit_price")
    median_mfe_atr = median_value(outcome_df, "mfe_atr")
    median_mae_atr = median_value(outcome_df, "mae_atr")
    horizon_60 = 60 if 60 in horizons else max(horizons)
    median_return = median_value(outcome_df, f"future_return_{horizon_60}m")
    stop_rate = true_rate(outcome_df["hit_stop_before_take_profit"]) if not outcome_df.empty else None

    no_followthrough = bool(
        median_mfe_atr is not None
        and take_profit_atr is not None
        and median_mfe_atr < take_profit_atr
        and (median_return is None or median_return <= 0)
    )
    stop_too_close = bool(
        stop_atr is not None
        and median_mae_atr is not None
        and median_mae_atr >= stop_atr
        and median_mfe_atr is not None
        and median_mfe_atr >= stop_atr
        and stop_rate is not None
        and stop_rate > 0.5
    )

    if no_followthrough:
        answer = "更像是入场本身没有 follow-through，而不是单纯止损太近"
    elif stop_too_close:
        answer = "信号存在一定 MFE，但 stop-first 比例偏高，止损距离可能过近"
    else:
        answer = "证据不足以单独归因于止损距离或入场质量"

    return {
        "basis_horizon": horizon_60,
        "median_future_return": median_return,
        "inferred_stop_atr": stop_atr,
        "median_mfe_atr": median_mfe_atr,
        "median_mae_atr": median_mae_atr,
        "stop_first_rate": stop_rate,
        "no_followthrough": no_followthrough,
        "stop_too_close": stop_too_close,
        "answer": answer,
    }


def assess_breakout_distance_effect(outcome_df: pd.DataFrame, horizons: list[int]) -> dict[str, Any]:
    """Assess whether larger breakout_distance_atr improves outcomes."""

    horizon = 60 if 60 in horizons else max(horizons)
    return_column = f"future_return_{horizon}m"
    working_df = outcome_df[["breakout_distance_atr", return_column, f"mfe_{horizon}m"]].copy() if not outcome_df.empty else pd.DataFrame()
    if working_df.empty:
        return {"basis_horizon": horizon, "spearman_future_return": None, "spearman_mfe": None, "answer": "样本不足"}

    working_df["breakout_distance_atr"] = pd.to_numeric(working_df["breakout_distance_atr"], errors="coerce")
    working_df[return_column] = pd.to_numeric(working_df[return_column], errors="coerce")
    working_df[f"mfe_{horizon}m"] = pd.to_numeric(working_df[f"mfe_{horizon}m"], errors="coerce")
    clean_return = working_df.dropna(subset=["breakout_distance_atr", return_column])
    clean_mfe = working_df.dropna(subset=["breakout_distance_atr", f"mfe_{horizon}m"])
    return_corr = rank_correlation(clean_return["breakout_distance_atr"], clean_return[return_column])
    mfe_corr = rank_correlation(clean_mfe["breakout_distance_atr"], clean_mfe[f"mfe_{horizon}m"])

    if return_corr is None and mfe_corr is None:
        answer = "样本不足，无法判断突破距离是否越大越好"
    elif (return_corr or 0.0) > 0.1 and (mfe_corr or 0.0) > 0.1:
        answer = "突破距离变大与更好 outcome 有弱正相关"
    else:
        answer = "没有证据表明 breakout_distance_atr 越大越好"

    return {
        "basis_horizon": horizon,
        "spearman_future_return": return_corr,
        "spearman_mfe": mfe_corr,
        "answer": answer,
    }


def rank_correlation(left: pd.Series, right: pd.Series) -> float | None:
    """Compute Spearman-like rank correlation without scipy."""

    paired_df = pd.DataFrame({"left": left, "right": right}).dropna()
    if len(paired_df.index) < 3:
        return None

    left_rank = paired_df["left"].rank(method="average")
    right_rank = paired_df["right"].rank(method="average")
    correlation = left_rank.corr(right_rank, method="pearson")
    if correlation is None or not np.isfinite(correlation):
        return None
    return float(correlation)


def assess_weekend(outcome_df: pd.DataFrame, horizons: list[int]) -> dict[str, Any]:
    """Assess weekend MFE/MAE."""

    horizon = 60 if 60 in horizons else max(horizons)
    if outcome_df.empty or "is_weekend" not in outcome_df.columns:
        return {"basis_horizon": horizon, "answer": "样本不足"}

    weekend_df = outcome_df[outcome_df["is_weekend"] == True]  # noqa: E712
    weekday_df = outcome_df[outcome_df["is_weekend"] == False]  # noqa: E712
    weekend_mfe = median_value(weekend_df, f"mfe_{horizon}m")
    weekday_mfe = median_value(weekday_df, f"mfe_{horizon}m")
    weekend_mae = median_value(weekend_df, f"mae_{horizon}m")
    weekday_mae = median_value(weekday_df, f"mae_{horizon}m")

    clearly_worse = bool(
        weekend_mfe is not None
        and weekday_mfe is not None
        and weekend_mae is not None
        and weekday_mae is not None
        and weekend_mfe < weekday_mfe
        and weekend_mae > weekday_mae
    )
    answer = "周末信号 MFE 更低且 MAE 更高" if clearly_worse else "周末信号没有同时呈现 MFE 更低和 MAE 更高"
    return {
        "basis_horizon": horizon,
        "weekend_count": int(len(weekend_df.index)),
        "weekday_count": int(len(weekday_df.index)),
        "weekend_mfe_median": weekend_mfe,
        "weekday_mfe_median": weekday_mfe,
        "weekend_mae_median": weekend_mae,
        "weekday_mae_median": weekday_mae,
        "clearly_worse": clearly_worse,
        "answer": answer,
    }


def assess_worst_hour(hour_df: pd.DataFrame, outcome_df: pd.DataFrame, horizons: list[int]) -> dict[str, Any]:
    """Assess whether the worst hour is genuinely worse on future returns."""

    horizon = 60 if 60 in horizons else max(horizons)
    column = f"future_return_{horizon}m_median"
    if hour_df.empty or column not in hour_df.columns:
        return {"basis_horizon": horizon, "answer": "样本不足"}

    working_df = hour_df.copy()
    working_df[column] = pd.to_numeric(working_df[column], errors="coerce")
    working_df = working_df[pd.to_numeric(working_df["entry_count"], errors="coerce").fillna(0) > 0]
    if working_df.empty:
        return {"basis_horizon": horizon, "answer": "样本不足"}

    worst_row = working_df.sort_values(column, ascending=True).iloc[0]
    overall_median = median_value(outcome_df, f"future_return_{horizon}m")
    worst_median = number_or_none(worst_row.get(column))
    genuinely_worse = bool(
        overall_median is not None and worst_median is not None and worst_median < overall_median
    )
    return {
        "basis_horizon": horizon,
        "worst_hour": int(worst_row.get("hour")),
        "worst_hour_median_future_return": worst_median,
        "overall_median_future_return": overall_median,
        "genuinely_worse": genuinely_worse,
        "answer": "最差小时的未来收益确实低于总体中位数" if genuinely_worse else "最差小时未明显差于总体中位数",
    }


def build_summary(
    report_dir: Path,
    signal_trace_path: Path,
    output_dir: Path,
    outcome_df: pd.DataFrame,
    horizons: list[int],
    warnings: list[str],
    side_df: pd.DataFrame,
    hour_df: pd.DataFrame,
    weekday_df: pd.DataFrame,
    regime_df: pd.DataFrame,
    breakout_df: pd.DataFrame,
    timezone_name: str,
) -> dict[str, Any]:
    """Build outcome_summary.json payload."""

    max_horizon = max(horizons)
    overall = summarize_outcome_frame(outcome_df, horizons)
    continuation = continuation_by_horizon(outcome_df, horizons)
    median_60 = continuation.get("60m", {}).get("median_future_return") if "60m" in continuation else None
    hypothesis_failed = bool(median_60 is not None and median_60 < 0)

    diagnostic_answers = {
        "worse_mfe_side": compare_side_mfe(side_df.rename(columns={"direction": "direction"}), max_horizon),
        "continuation_by_horizon": continuation,
        "stop_take_profit_reasonableness": assess_stop_take_profit(outcome_df),
        "stop_vs_followthrough": assess_stop_vs_followthrough(outcome_df, horizons),
        "breakout_distance_effect": assess_breakout_distance_effect(outcome_df, horizons),
        "weekend_effect": assess_weekend(outcome_df, horizons),
        "worst_hour": assess_worst_hour(hour_df, outcome_df, horizons),
        "breakout_continuation_hypothesis_failed": hypothesis_failed,
    }
    if hypothesis_failed:
        warnings.append("breakout continuation hypothesis failed")

    return {
        "report_dir": str(report_dir),
        "signal_trace_path": str(signal_trace_path),
        "output_dir": str(output_dir),
        "timezone": timezone_name,
        "horizons": horizons,
        "entry_count": int(len(outcome_df.index)),
        "warnings": warnings,
        "overall": overall,
        "by_side": dataframe_records(side_df),
        "by_hour": dataframe_records(hour_df),
        "by_weekday": dataframe_records(weekday_df),
        "by_regime": dataframe_records(regime_df),
        "by_breakout_distance_bucket": dataframe_records(breakout_df),
        "diagnostic_answers": diagnostic_answers,
    }


def format_number(value: Any, digits: int = 6) -> str:
    """Format numeric values for markdown."""

    number = number_or_none(value)
    if number is None:
        return "N/A"
    return f"{number:.{digits}f}"


def render_markdown(summary: dict[str, Any]) -> str:
    """Render a human-readable signal outcome report."""

    horizons = summary.get("horizons") or []
    answers = summary.get("diagnostic_answers") or {}
    continuation = answers.get("continuation_by_horizon") or {}
    continuation_lines = []
    for horizon in horizons:
        item = continuation.get(f"{horizon}m") or {}
        continuation_lines.append(
            f"- {horizon}m: median_future_return={format_number(item.get('median_future_return'))}, "
            f"positive_rate={format_number(item.get('positive_rate'), 4)}, "
            f"has_positive_continuation={item.get('has_positive_continuation')}"
        )

    warnings = summary.get("warnings") or []
    warning_lines = "\n".join(f"- {item}" for item in warnings) if warnings else "- 无"
    failed_line = ""
    if answers.get("breakout_continuation_hypothesis_failed"):
        failed_line = "\n- breakout continuation hypothesis failed\n"

    return (
        "# 信号级 MFE/MAE 与突破延续性诊断\n\n"
        "## 核心结论\n"
        f"- entry signal 数：{summary.get('entry_count')}\n"
        f"- 多空 MFE：{(answers.get('worse_mfe_side') or {}).get('answer')}\n"
        f"- stop/take-profit：{(answers.get('stop_take_profit_reasonableness') or {}).get('answer')}\n"
        f"- 止损 vs 入场：{(answers.get('stop_vs_followthrough') or {}).get('answer')}\n"
        f"- 突破距离：{(answers.get('breakout_distance_effect') or {}).get('answer')}\n"
        f"- 周末：{(answers.get('weekend_effect') or {}).get('answer')}\n"
        f"- 最差小时：{(answers.get('worst_hour') or {}).get('answer')}\n"
        f"{failed_line}\n"
        "## 5/15/30/60/120m 延续性\n"
        f"{chr(10).join(continuation_lines)}\n\n"
        "## 当前 stop_atr / take_profit_atr 诊断\n"
        f"- inferred_stop_atr={format_number((answers.get('stop_take_profit_reasonableness') or {}).get('inferred_stop_atr'))}\n"
        f"- inferred_take_profit_atr={format_number((answers.get('stop_take_profit_reasonableness') or {}).get('inferred_take_profit_atr'))}\n"
        f"- median_mfe_atr={format_number((answers.get('stop_take_profit_reasonableness') or {}).get('median_mfe_atr'))}\n"
        f"- median_mae_atr={format_number((answers.get('stop_take_profit_reasonableness') or {}).get('median_mae_atr'))}\n"
        f"- take_profit_first_rate={format_number((answers.get('stop_take_profit_reasonableness') or {}).get('take_profit_first_rate'), 4)}\n"
        f"- stop_first_rate={format_number((answers.get('stop_take_profit_reasonableness') or {}).get('stop_first_rate'), 4)}\n\n"
        "## 输出文件\n"
        "- signal_outcomes.csv\n"
        "- outcome_summary.json\n"
        "- outcome_by_side.csv\n"
        "- outcome_by_hour.csv\n"
        "- outcome_by_weekday.csv\n"
        "- outcome_by_regime.csv\n"
        "- outcome_by_breakout_distance_bucket.csv\n"
        "- outcome_report.md\n\n"
        "## Warning\n"
        f"{warning_lines}\n"
    )


def write_json(path: Path, payload: Any) -> None:
    """Write JSON with UTF-8 encoding."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(payload), ensure_ascii=False, indent=2), encoding="utf-8")


def write_dataframe(path: Path, df: pd.DataFrame) -> None:
    """Write CSV with UTF-8 encoding."""

    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")


def write_outputs(
    output_dir: Path,
    outcome_df: pd.DataFrame,
    summary: dict[str, Any],
    side_df: pd.DataFrame,
    hour_df: pd.DataFrame,
    weekday_df: pd.DataFrame,
    regime_df: pd.DataFrame,
    breakout_df: pd.DataFrame,
    markdown: str,
) -> None:
    """Write all signal outcome artifacts."""

    output_dir.mkdir(parents=True, exist_ok=True)
    write_dataframe(output_dir / "signal_outcomes.csv", outcome_df)
    write_json(output_dir / "outcome_summary.json", summary)
    write_dataframe(output_dir / "outcome_by_side.csv", side_df)
    write_dataframe(output_dir / "outcome_by_hour.csv", hour_df)
    write_dataframe(output_dir / "outcome_by_weekday.csv", weekday_df)
    write_dataframe(output_dir / "outcome_by_regime.csv", regime_df)
    write_dataframe(output_dir / "outcome_by_breakout_distance_bucket.csv", breakout_df)
    (output_dir / "outcome_report.md").write_text(markdown, encoding="utf-8")


def run_analysis(
    report_dir: Path,
    signal_trace_path: Path,
    output_dir: Path,
    horizons: list[int],
    timezone_name: str,
    bars_from_db: bool,
    logger: logging.Logger,
    bars_df: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Run the full signal outcome workflow."""

    ZoneInfo(timezone_name)
    warnings: list[str] = []
    trace_df = read_signal_trace(signal_trace_path, timezone_name)
    entry_df = prepare_entry_signals(trace_df, warnings)

    if bars_df is None:
        if not bars_from_db:
            raise SignalOutcomeError("--no-bars-from-db 已设置，但当前 CLI 未提供替代 bars 输入")
        bar_df = load_bars_from_db(entry_df, horizons, timezone_name, warnings, logger)
    else:
        bar_df = dataframe_bars_to_ohlc(bars_df, timezone_name)

    outcome_df = compute_signal_outcomes(entry_df, bar_df, horizons, warnings)
    side_df = build_group_summary(outcome_df.rename(columns={"direction": "direction"}), "direction", horizons, base_values=["long", "short"])
    hour_df = build_group_summary(outcome_df, "hour", horizons, base_values=list(range(24)))
    weekday_df = build_weekday_summary(outcome_df, horizons)
    regime_df = build_group_summary(outcome_df, "regime", horizons)
    breakout_df = build_breakout_bucket_summary(outcome_df, horizons)

    summary = build_summary(
        report_dir=report_dir,
        signal_trace_path=signal_trace_path,
        output_dir=output_dir,
        outcome_df=outcome_df,
        horizons=horizons,
        warnings=warnings,
        side_df=side_df,
        hour_df=hour_df,
        weekday_df=weekday_df,
        regime_df=regime_df,
        breakout_df=breakout_df,
        timezone_name=timezone_name,
    )
    markdown = render_markdown(summary)
    write_outputs(output_dir, outcome_df, summary, side_df, hour_df, weekday_df, regime_df, breakout_df, markdown)
    return summary


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""

    args = parse_args(argv)
    ensure_headless_runtime()
    logger = setup_logging("analyze_signal_outcomes", verbose=args.verbose)

    try:
        report_dir = resolve_path(args.report_dir)
        signal_trace_path = resolve_path(args.signal_trace, report_dir / "signal_trace.csv")
        output_dir = resolve_path(args.output_dir, report_dir / "signal_outcomes")
        horizons = parse_horizons(args.horizons)
        summary = run_analysis(
            report_dir=report_dir,
            signal_trace_path=signal_trace_path,
            output_dir=output_dir,
            horizons=horizons,
            timezone_name=args.timezone,
            bars_from_db=bool(args.bars_from_db),
            logger=logger,
        )
        print_json_block(
            "Signal outcome summary:",
            {
                "output_dir": output_dir,
                "entry_count": summary.get("entry_count"),
                "warnings": summary.get("warnings"),
                "diagnostic_answers": summary.get("diagnostic_answers"),
            },
        )
        if args.json:
            print(json.dumps(to_jsonable(summary), ensure_ascii=False, indent=2))
        return 0
    except SignalOutcomeError as exc:
        log_event(logger, logging.ERROR, "signal_outcomes.error", str(exc))
        return 2
    except Exception:
        logger.exception(
            "Unexpected error during signal outcome analysis",
            extra={"event": "signal_outcomes.unexpected_error"},
        )
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
