#!/usr/bin/env python3
"""Research signal feature predictive power against future outcomes."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from analyze_signal_outcomes import (
    SignalOutcomeError,
    bars_to_dataframe,
    compute_signal_outcomes,
    configure_sqlite_settings,
    dataframe_bars_to_ohlc,
    normalize_bool,
    number_or_none,
    prepare_entry_signals,
    rank_correlation,
    read_signal_trace,
    resolve_exchange,
    split_vt_symbol,
)
from common_runtime import PROJECT_ROOT, ensure_headless_runtime, log_event, print_json_block, setup_logging, to_jsonable
from history_time_utils import DEFAULT_TIMEZONE
from history_utils import get_database_timezone


DEFAULT_HORIZONS = "15,30,60,120"
REQUIRED_LABEL_HORIZONS = [15, 30, 60, 120]
LOOKBACK_MINUTES = 30
PREDICTIVE_IC_THRESHOLD = 0.10
STOP_IC_THRESHOLD = 0.10
NUMERIC_FEATURES = [
    "breakout_distance_atr",
    "atr_pct",
    "ema_spread_pct",
    "rsi",
    "donchian_width_atr",
    "close_location_in_donchian",
    "recent_return_5m",
    "recent_return_15m",
    "recent_return_30m",
    "recent_volatility_30m",
    "volume_zscore_30m",
    "upper_wick_ratio",
    "lower_wick_ratio",
    "body_ratio",
    "range_atr",
]
CATEGORICAL_FEATURES = ["hour", "weekday", "is_weekend", "direction", "regime"]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES
BASE_COLUMNS = ["signal_id", "datetime", "vt_symbol", "price", "close_1m", "atr_1m"]
CORE_LABEL_COLUMNS = [
    "mfe_60m",
    "mae_60m",
    "mfe_atr",
    "mae_atr",
    "stop_first",
    "tp_first",
    "good_signal_60m",
    "bad_signal_60m",
]


class SignalFeatureResearchError(Exception):
    """Raised when signal feature research cannot continue."""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description="Research signal features against future returns, MFE, and MAE.")
    parser.add_argument("--report-dir", required=True, help="Backtest report directory containing signal_trace.csv.")
    parser.add_argument("--signal-trace", help="Signal trace CSV. Default: <report-dir>/signal_trace.csv.")
    parser.add_argument(
        "--timezone",
        default=DEFAULT_TIMEZONE,
        help=f"Timezone used to align signal and bar times. Default: {DEFAULT_TIMEZONE}.",
    )
    parser.add_argument("--output-dir", help="Output directory. Default: <report-dir>/signal_feature_research.")
    parser.add_argument(
        "--horizons",
        default=DEFAULT_HORIZONS,
        help=f"Comma-separated future-return horizons in minutes. Default: {DEFAULT_HORIZONS}.",
    )
    parser.add_argument("--bins", type=int, default=5, help="Quantile bin count for numeric features. Default: 5.")
    parser.add_argument("--min-count", type=int, default=50, help="Minimum count for robust evidence flags. Default: 50.")
    parser.add_argument(
        "--feature-list",
        help="Comma-separated feature names to analyze. Default: all Signal Lab features.",
    )
    parser.add_argument(
        "--data-check-strict",
        action="store_true",
        help="Fail when critical dataset/label checks are not satisfied.",
    )
    parser.add_argument(
        "--bars-from-db",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Load 1m bars from vn.py sqlite. Default: enabled.",
    )
    parser.add_argument("--json", action="store_true", help="Print feature_summary.json payload to stdout.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose structured logs.")
    return parser.parse_args(argv)


def resolve_path(path_arg: str | None, default_path: Path | None = None) -> Path:
    """Resolve CLI paths relative to project root."""

    if path_arg:
        path = Path(path_arg)
    elif default_path is not None:
        path = default_path
    else:
        raise SignalFeatureResearchError("缺少路径参数且没有默认值")
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
            raise SignalFeatureResearchError(f"--horizons 包含非法整数: {text!r}") from exc
        if value <= 0:
            raise SignalFeatureResearchError(f"--horizons 必须为正整数分钟: {value}")
        horizons.add(value)
    if not horizons:
        raise SignalFeatureResearchError("--horizons 不能为空")
    return sorted(horizons)


def label_horizons(requested_horizons: list[int]) -> list[int]:
    """Return the horizons required for the feature dataset labels."""

    return sorted(set(requested_horizons) | set(REQUIRED_LABEL_HORIZONS))


def parse_feature_list(raw_value: str | None) -> list[str]:
    """Parse requested feature names."""

    if raw_value is None or not str(raw_value).strip():
        return list(FEATURE_COLUMNS)

    requested = [token.strip() for token in str(raw_value).split(",") if token.strip()]
    if not requested or any(token.lower() == "all" for token in requested):
        return list(FEATURE_COLUMNS)

    unknown = [feature for feature in requested if feature not in FEATURE_COLUMNS]
    if unknown:
        raise SignalFeatureResearchError(f"--feature-list 包含未知特征: {', '.join(unknown)}")
    return requested


def future_return_columns(horizons: list[int]) -> list[str]:
    """Build future-return label column names."""

    return [f"future_return_{horizon}m" for horizon in horizons]


def label_columns(horizons: list[int]) -> list[str]:
    """Build all label columns used in feature_dataset.csv."""

    return future_return_columns(horizons) + list(CORE_LABEL_COLUMNS)


def append_warning(warnings: list[str], message: str) -> None:
    """Append a warning once."""

    if message not in warnings:
        warnings.append(message)


def finite_or_none(value: Any) -> float | None:
    """Return a finite float or None."""

    number = number_or_none(value)
    if number is None or not np.isfinite(number):
        return None
    return number


def bool_series(series: pd.Series) -> pd.Series:
    """Normalize a Series into bool-or-null values."""

    return series.map(normalize_bool)


def bool_rate(series: pd.Series) -> float | None:
    """Return true rate for bool-like values."""

    if series.empty:
        return None
    normalized = bool_series(series).dropna()
    if normalized.empty:
        return None
    return float(normalized.astype(bool).mean())


def numeric_series(frame: pd.DataFrame, column: str) -> pd.Series:
    """Return a numeric column, or a null series when missing."""

    if column not in frame.columns:
        return pd.Series([np.nan] * len(frame.index), index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def safe_ratio(numerator: float | None, denominator: float | None) -> float | None:
    """Return numerator / denominator when both are usable."""

    if numerator is None or denominator is None or denominator == 0:
        return None
    value = numerator / denominator
    if not np.isfinite(value):
        return None
    return float(value)


def normalize_direction_for_feature(value: Any) -> str:
    """Normalize signal direction for feature rows."""

    text = str(value).strip().lower()
    if text in {"long", "多", "buy"}:
        return "long"
    if text in {"short", "空", "sell"}:
        return "short"
    return "unknown"


def ensure_signal_ids(entry_df: pd.DataFrame) -> pd.DataFrame:
    """Ensure every entry has a stable signal_id for merging labels."""

    result_df = entry_df.copy().reset_index(drop=True)
    if "signal_id" not in result_df.columns:
        result_df["signal_id"] = None
    signal_id = result_df["signal_id"].astype(str)
    missing = result_df["signal_id"].isna() | signal_id.str.strip().isin(["", "nan", "None"])
    result_df.loc[missing, "signal_id"] = [f"row-{index}" for index in result_df.index[missing]]
    return result_df


def load_feature_bars_from_db(
    entry_df: pd.DataFrame,
    horizons: list[int],
    timezone_name: str,
    warnings: list[str],
    logger: logging.Logger,
) -> pd.DataFrame:
    """Load enough 1m bars for lookback features and future labels."""

    if entry_df.empty:
        return pd.DataFrame(columns=["datetime", "open", "high", "low", "close", "volume"])

    vt_symbols = [str(value).strip() for value in entry_df["vt_symbol"].dropna().unique() if str(value).strip()]
    if not vt_symbols:
        raise SignalFeatureResearchError("entry signal 缺少 vt_symbol，无法从数据库读取 1m bar")
    if len(vt_symbols) > 1:
        append_warning(warnings, f"signal_trace 包含多个 vt_symbol，仅使用第一个读取 bars: {vt_symbols[0]}")

    symbol, exchange_value = split_vt_symbol(vt_symbols[0])
    exchange = resolve_exchange(exchange_value)

    from vnpy.trader.constant import Interval
    from vnpy.trader.database import get_database

    configure_sqlite_settings(logger)
    db_tz = get_database_timezone()
    start = (
        entry_df["_signal_dt"].min() - pd.Timedelta(minutes=LOOKBACK_MINUTES + 1)
    ).to_pydatetime().astimezone(db_tz).replace(tzinfo=None)
    end = (
        entry_df["_signal_dt"].max() + pd.Timedelta(minutes=max(horizons) + 1)
    ).to_pydatetime().astimezone(db_tz).replace(tzinfo=None)

    bars = get_database().load_bar_data(symbol, exchange, Interval.MINUTE, start, end)
    if not bars:
        append_warning(
            warnings,
            f"数据库没有返回 1m bars: vt_symbol={vt_symbols[0]}, start={start}, end={end}",
        )
    return bars_to_dataframe(list(bars), timezone_name)


def warn_missing_source_columns(
    signal_trace_path: Path,
    selected_features: list[str],
    warnings: list[str],
) -> None:
    """Warn about source fields that are absent from signal_trace.csv."""

    try:
        header_columns = set(pd.read_csv(signal_trace_path, nrows=0).columns)
    except Exception:
        return

    direct_columns = {
        "breakout_distance_atr": ["breakout_distance_atr", "breakout_distance", "atr_1m"],
        "atr_pct": ["atr_pct", "atr_1m", "close_1m"],
        "ema_spread_pct": ["ema_spread_pct", "ema_spread", "fast_ema_5m", "slow_ema_5m"],
        "rsi": ["rsi"],
        "hour": ["hour", "datetime"],
        "weekday": ["weekday", "datetime"],
        "is_weekend": ["is_weekend", "datetime"],
        "direction": ["direction"],
        "regime": ["regime"],
        "donchian_width_atr": ["donchian_width", "donchian_high", "donchian_low", "atr_1m"],
        "close_location_in_donchian": ["donchian_high", "donchian_low", "close_1m"],
    }
    for feature in selected_features:
        candidates = direct_columns.get(feature)
        if not candidates:
            continue
        if not any(column in header_columns for column in candidates):
            append_warning(warnings, f"signal_trace 缺少生成 {feature} 所需字段，结果将为 null")


def bar_index_at_or_before(bar_times: pd.Series, signal_dt: pd.Timestamp) -> int:
    """Return the index of the last bar at or before signal_dt."""

    return int(bar_times.searchsorted(signal_dt, side="right")) - 1


def bar_index_after(bar_times: pd.Series, signal_dt: pd.Timestamp) -> int:
    """Return the index of the first bar after signal_dt."""

    return int(bar_times.searchsorted(signal_dt, side="right"))


def get_bar_value(bar: pd.Series | None, column: str) -> float | None:
    """Return a finite bar value."""

    if bar is None:
        return None
    return finite_or_none(bar.get(column))


def direction_adjusted_return(direction: str, start_close: float, end_close: float) -> float | None:
    """Return direction-adjusted recent return."""

    if start_close <= 0:
        return None
    raw_return = (end_close - start_close) / start_close
    if direction == "short":
        raw_return = -raw_return
    if not np.isfinite(raw_return):
        return None
    return float(raw_return)


def compute_recent_return(
    bars: pd.DataFrame,
    bar_times: pd.Series,
    signal_dt: pd.Timestamp,
    direction: str,
    minutes: int,
) -> float | None:
    """Compute direction-adjusted close return over a pre-signal lookback window."""

    current_index = bar_index_at_or_before(bar_times, signal_dt)
    past_index = bar_index_at_or_before(bar_times, signal_dt - pd.Timedelta(minutes=minutes))
    if current_index < 0 or past_index < 0 or current_index >= len(bars.index):
        return None

    start_close = finite_or_none(bars.iloc[past_index].get("close"))
    end_close = finite_or_none(bars.iloc[current_index].get("close"))
    if start_close is None or end_close is None:
        return None
    return direction_adjusted_return(direction, start_close, end_close)


def compute_recent_volatility(
    bars: pd.DataFrame,
    bar_times: pd.Series,
    signal_dt: pd.Timestamp,
    minutes: int = 30,
) -> float | None:
    """Compute pre-signal close-to-close volatility."""

    current_index = bar_index_at_or_before(bar_times, signal_dt)
    start_index = bar_index_after(bar_times, signal_dt - pd.Timedelta(minutes=minutes))
    if current_index < 1 or start_index < 0 or start_index >= current_index:
        return None

    closes = pd.to_numeric(bars.iloc[start_index : current_index + 1]["close"], errors="coerce")
    returns = closes.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    if len(returns.index) < 2:
        return None
    value = float(returns.std(ddof=0))
    return value if np.isfinite(value) else None


def compute_volume_zscore(
    bars: pd.DataFrame,
    bar_times: pd.Series,
    signal_dt: pd.Timestamp,
    minutes: int = 30,
) -> float | None:
    """Compute current-volume z-score versus the pre-signal lookback."""

    if "volume" not in bars.columns:
        return None

    current_index = bar_index_at_or_before(bar_times, signal_dt)
    start_index = bar_index_after(bar_times, signal_dt - pd.Timedelta(minutes=minutes))
    if current_index < 0 or start_index < 0 or start_index >= current_index:
        return None

    current_volume = finite_or_none(bars.iloc[current_index].get("volume"))
    if current_volume is None:
        return None
    history = pd.to_numeric(bars.iloc[start_index:current_index]["volume"], errors="coerce").dropna()
    if len(history.index) < 2:
        return None
    std = float(history.std(ddof=0))
    if std == 0 or not np.isfinite(std):
        return None
    mean = float(history.mean())
    return float((current_volume - mean) / std)


def compute_wick_body_features(bar: pd.Series | None, atr: float | None) -> dict[str, float | None]:
    """Compute signal-bar wick/body/range features."""

    open_price = get_bar_value(bar, "open")
    high = get_bar_value(bar, "high")
    low = get_bar_value(bar, "low")
    close = get_bar_value(bar, "close")
    if open_price is None or high is None or low is None or close is None:
        return {
            "upper_wick_ratio": None,
            "lower_wick_ratio": None,
            "body_ratio": None,
            "range_atr": None,
        }

    bar_range = high - low
    if bar_range <= 0:
        return {
            "upper_wick_ratio": None,
            "lower_wick_ratio": None,
            "body_ratio": None,
            "range_atr": safe_ratio(bar_range, atr),
        }

    upper_wick = max(0.0, high - max(open_price, close))
    lower_wick = max(0.0, min(open_price, close) - low)
    body = abs(close - open_price)
    return {
        "upper_wick_ratio": float(upper_wick / bar_range),
        "lower_wick_ratio": float(lower_wick / bar_range),
        "body_ratio": float(body / bar_range),
        "range_atr": safe_ratio(bar_range, atr),
    }


def compute_close_location(
    direction: str,
    close_price: float | None,
    donchian_high: float | None,
    donchian_low: float | None,
) -> float | None:
    """Compute directional Donchian close location in the 0-1 range."""

    if close_price is None or donchian_high is None or donchian_low is None:
        return None
    width = donchian_high - donchian_low
    if width <= 0:
        return None
    if direction == "short":
        value = (donchian_high - close_price) / width
    else:
        value = (close_price - donchian_low) / width
    if not np.isfinite(value):
        return None
    return float(min(max(value, 0.0), 1.0))


def build_feature_dataset(
    entry_df: pd.DataFrame,
    bars_df: pd.DataFrame,
    selected_features: list[str],
    warnings: list[str],
) -> pd.DataFrame:
    """Build one feature row per entry signal."""

    feature_columns = list(dict.fromkeys(BASE_COLUMNS + selected_features))
    if entry_df.empty:
        return pd.DataFrame(columns=feature_columns)

    bars = bars_df.sort_values("datetime", kind="stable").reset_index(drop=True)
    bar_times = pd.Series(bars["datetime"]) if not bars.empty else pd.Series([], dtype="datetime64[ns]")
    missing_bar_rows = 0
    records: list[dict[str, Any]] = []

    for row in entry_df.to_dict(orient="records"):
        signal_dt = pd.Timestamp(row["_signal_dt"])
        direction = normalize_direction_for_feature(row.get("direction"))
        entry_price = finite_or_none(row.get("price"))
        close_1m = finite_or_none(row.get("close_1m")) or entry_price
        atr_1m = finite_or_none(row.get("atr_1m"))
        donchian_high = finite_or_none(row.get("donchian_high"))
        donchian_low = finite_or_none(row.get("donchian_low"))

        current_bar: pd.Series | None = None
        if not bars.empty:
            current_index = bar_index_at_or_before(bar_times, signal_dt)
            if 0 <= current_index < len(bars.index):
                current_bar = bars.iloc[current_index]
        if current_bar is None:
            missing_bar_rows += 1

        if close_1m is None:
            close_1m = get_bar_value(current_bar, "close")

        breakout_distance = finite_or_none(row.get("breakout_distance"))
        breakout_distance_atr = finite_or_none(row.get("breakout_distance_atr"))
        if breakout_distance_atr is None:
            breakout_distance_atr = safe_ratio(breakout_distance, atr_1m)

        atr_pct = finite_or_none(row.get("atr_pct"))
        if atr_pct is None:
            atr_pct = safe_ratio(atr_1m, close_1m)

        ema_spread = finite_or_none(row.get("ema_spread"))
        if ema_spread is None:
            fast_ema = finite_or_none(row.get("fast_ema_5m"))
            slow_ema = finite_or_none(row.get("slow_ema_5m"))
            if fast_ema is not None and slow_ema is not None:
                ema_spread = fast_ema - slow_ema
        ema_spread_pct = finite_or_none(row.get("ema_spread_pct"))
        if ema_spread_pct is None:
            ema_spread_pct = safe_ratio(ema_spread, close_1m)

        donchian_width = finite_or_none(row.get("donchian_width"))
        if donchian_width is None and donchian_high is not None and donchian_low is not None:
            donchian_width = donchian_high - donchian_low
        donchian_width_atr = safe_ratio(donchian_width, atr_1m)

        wick_body = compute_wick_body_features(current_bar, atr_1m)
        record: dict[str, Any] = {
            "signal_id": row.get("signal_id"),
            "datetime": signal_dt.isoformat(),
            "vt_symbol": row.get("vt_symbol"),
            "price": entry_price,
            "close_1m": close_1m,
            "atr_1m": atr_1m,
            "breakout_distance_atr": breakout_distance_atr,
            "atr_pct": atr_pct,
            "ema_spread_pct": ema_spread_pct,
            "rsi": finite_or_none(row.get("rsi")),
            "hour": int(row.get("hour")) if finite_or_none(row.get("hour")) is not None else signal_dt.hour,
            "weekday": int(row.get("weekday")) if finite_or_none(row.get("weekday")) is not None else signal_dt.weekday(),
            "is_weekend": normalize_bool(row.get("is_weekend"))
            if normalize_bool(row.get("is_weekend")) is not None
            else bool(signal_dt.weekday() >= 5),
            "direction": direction,
            "regime": row.get("regime"),
            "donchian_width_atr": donchian_width_atr,
            "close_location_in_donchian": compute_close_location(
                direction,
                close_1m,
                donchian_high,
                donchian_low,
            ),
            "recent_return_5m": compute_recent_return(bars, bar_times, signal_dt, direction, 5)
            if not bars.empty
            else None,
            "recent_return_15m": compute_recent_return(bars, bar_times, signal_dt, direction, 15)
            if not bars.empty
            else None,
            "recent_return_30m": compute_recent_return(bars, bar_times, signal_dt, direction, 30)
            if not bars.empty
            else None,
            "recent_volatility_30m": compute_recent_volatility(bars, bar_times, signal_dt, 30)
            if not bars.empty
            else None,
            "volume_zscore_30m": compute_volume_zscore(bars, bar_times, signal_dt, 30) if not bars.empty else None,
            **wick_body,
        }
        records.append({column: record.get(column) for column in feature_columns})

    if missing_bar_rows:
        append_warning(warnings, f"有 {missing_bar_rows} 条 signal 找不到 signal time 之前的 1m bar，bar 派生特征将为 null")
    if "volume_zscore_30m" in selected_features and (bars.empty or "volume" not in bars.columns or numeric_series(bars, "volume").dropna().empty):
        append_warning(warnings, "1m bar volume 缺失，volume_zscore_30m 将为 null")

    feature_df = pd.DataFrame(records, columns=feature_columns)
    for feature in selected_features:
        if feature in feature_df.columns:
            missing_count = int(feature_df[feature].isna().sum())
            if missing_count:
                append_warning(warnings, f"feature {feature} 缺失值数量: {missing_count}/{len(feature_df.index)}")
    return feature_df


def read_existing_outcomes(report_dir: Path, warnings: list[str]) -> pd.DataFrame | None:
    """Read signal_outcomes/signal_outcomes.csv when present."""

    outcome_path = report_dir / "signal_outcomes" / "signal_outcomes.csv"
    if not outcome_path.exists():
        return None
    try:
        outcome_df = pd.read_csv(outcome_path)
    except Exception as exc:
        append_warning(warnings, f"读取已有 signal_outcomes.csv 失败，将重新计算: {exc!r}")
        return None
    append_warning(warnings, f"已优先使用已有 outcome 文件: {outcome_path}")
    return ensure_signal_ids(outcome_df)


def outcome_has_required_columns(outcome_df: pd.DataFrame, horizons: list[int]) -> bool:
    """Return whether an outcome frame can provide all required labels."""

    required = future_return_columns(horizons) + ["mfe_60m", "mae_60m"]
    return all(column in outcome_df.columns for column in required)


def load_or_compute_outcomes(
    report_dir: Path,
    entry_df: pd.DataFrame,
    bars_df: pd.DataFrame,
    horizons: list[int],
    warnings: list[str],
) -> tuple[pd.DataFrame, str]:
    """Load existing outcomes or compute labels from 1m bars."""

    existing_df = read_existing_outcomes(report_dir, warnings)
    if existing_df is not None and outcome_has_required_columns(existing_df, horizons):
        return existing_df, "existing_signal_outcomes"

    if existing_df is not None:
        append_warning(warnings, "已有 signal_outcomes.csv 缺少必要 label 列，将用 1m bar 重新计算 labels")
    return compute_signal_outcomes(entry_df, bars_df, horizons, warnings), "computed_from_1m_bars"


def build_label_frame(outcome_df: pd.DataFrame, horizons: list[int], warnings: list[str]) -> pd.DataFrame:
    """Build normalized labels from signal outcomes."""

    labels = pd.DataFrame()
    labels["signal_id"] = outcome_df["signal_id"].astype(str) if "signal_id" in outcome_df.columns else []
    for horizon in horizons:
        column = f"future_return_{horizon}m"
        if column in outcome_df.columns:
            labels[column] = pd.to_numeric(outcome_df[column], errors="coerce")
        else:
            labels[column] = np.nan
            append_warning(warnings, f"label {column} 缺失，将为 null")

    for column in ["mfe_60m", "mae_60m"]:
        if column in outcome_df.columns:
            labels[column] = pd.to_numeric(outcome_df[column], errors="coerce")
        else:
            labels[column] = np.nan
            append_warning(warnings, f"label {column} 缺失，将为 null")

    price = numeric_series(outcome_df, "price")
    atr = numeric_series(outcome_df, "atr_1m").replace(0, np.nan)
    labels["mfe_atr"] = (labels["mfe_60m"] * price / atr).replace([np.inf, -np.inf], np.nan)
    labels["mae_atr"] = (labels["mae_60m"] * price / atr).replace([np.inf, -np.inf], np.nan)
    if "mfe_atr" in outcome_df.columns:
        labels["mfe_atr"] = labels["mfe_atr"].combine_first(pd.to_numeric(outcome_df["mfe_atr"], errors="coerce"))
    if "mae_atr" in outcome_df.columns:
        labels["mae_atr"] = labels["mae_atr"].combine_first(pd.to_numeric(outcome_df["mae_atr"], errors="coerce"))

    stop_col = "stop_first" if "stop_first" in outcome_df.columns else "hit_stop_before_take_profit"
    tp_col = "tp_first" if "tp_first" in outcome_df.columns else "hit_take_profit_before_stop"
    labels["stop_first"] = bool_series(outcome_df[stop_col]) if stop_col in outcome_df.columns else pd.NA
    labels["tp_first"] = bool_series(outcome_df[tp_col]) if tp_col in outcome_df.columns else pd.NA

    future_60 = pd.to_numeric(labels["future_return_60m"], errors="coerce")
    mfe_atr = pd.to_numeric(labels["mfe_atr"], errors="coerce")
    mae_atr = pd.to_numeric(labels["mae_atr"], errors="coerce")
    good_mask = future_60.notna() & mfe_atr.notna() & mae_atr.notna() & (future_60 > 0) & (mfe_atr > mae_atr)
    bad_mask = future_60.notna() & mfe_atr.notna() & mae_atr.notna() & (future_60 < 0) & (mae_atr > mfe_atr)
    known_mask = future_60.notna() & mfe_atr.notna() & mae_atr.notna()
    labels["good_signal_60m"] = pd.Series(pd.NA, index=labels.index, dtype="object")
    labels.loc[known_mask, "good_signal_60m"] = False
    labels.loc[good_mask, "good_signal_60m"] = True
    labels["bad_signal_60m"] = pd.Series(pd.NA, index=labels.index, dtype="object")
    labels.loc[known_mask, "bad_signal_60m"] = False
    labels.loc[bad_mask, "bad_signal_60m"] = True
    return labels[["signal_id"] + label_columns(horizons)]


def merge_features_and_labels(feature_df: pd.DataFrame, label_df: pd.DataFrame) -> pd.DataFrame:
    """Merge feature rows with labels."""

    if feature_df.empty:
        for column in label_df.columns:
            if column != "signal_id":
                feature_df[column] = None
        return feature_df

    labels = label_df.copy()
    labels["signal_id"] = labels["signal_id"].astype(str)
    labels = labels.drop_duplicates("signal_id", keep="last")
    merged = feature_df.copy()
    merged["signal_id"] = merged["signal_id"].astype(str)
    return merged.merge(labels, on="signal_id", how="left")


def dataframe_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Convert a DataFrame into JSON-safe records."""

    if df.empty:
        return []
    safe_df = df.replace({np.nan: None})
    return json.loads(safe_df.to_json(orient="records", force_ascii=False))


def format_number(value: Any, digits: int = 6) -> str:
    """Format numeric values for reports."""

    number = finite_or_none(value)
    if number is None:
        return "N/A"
    return f"{number:.{digits}f}"


def compute_feature_ic(
    dataset_df: pd.DataFrame,
    selected_features: list[str],
    horizons: list[int],
) -> pd.DataFrame:
    """Compute Spearman IC for numeric features against target labels."""

    target_columns = future_return_columns(horizons) + ["mfe_atr", "mae_atr"]
    rows: list[dict[str, Any]] = []
    for feature in selected_features:
        if feature not in NUMERIC_FEATURES or feature not in dataset_df.columns:
            continue
        feature_values = pd.to_numeric(dataset_df[feature], errors="coerce")
        for target in target_columns:
            if target not in dataset_df.columns:
                continue
            target_values = pd.to_numeric(dataset_df[target], errors="coerce")
            paired = pd.DataFrame({"feature": feature_values, "target": target_values}).dropna()
            spearman = spearman_or_none(paired["feature"], paired["target"])
            rows.append(
                {
                    "feature": feature,
                    "target": target,
                    "count": int(len(paired.index)),
                    "spearman": spearman,
                    "abs_spearman": abs(spearman) if spearman is not None else None,
                }
            )
    return pd.DataFrame(rows, columns=["feature", "target", "count", "spearman", "abs_spearman"])


def spearman_or_none(left: pd.Series, right: pd.Series) -> float | None:
    """Return Spearman rank correlation when both sides have enough variation."""

    paired = pd.DataFrame({"left": left, "right": right}).dropna()
    if len(paired.index) < 3:
        return None
    if paired["left"].nunique(dropna=True) < 2 or paired["right"].nunique(dropna=True) < 2:
        return None
    return rank_correlation(paired["left"], paired["right"])


def compute_stop_first_ic(dataset_df: pd.DataFrame, selected_features: list[str]) -> pd.DataFrame:
    """Compute numeric-feature IC versus stop_first."""

    if "stop_first" not in dataset_df.columns:
        return pd.DataFrame(columns=["feature", "target", "count", "spearman", "abs_spearman"])

    stop_values = bool_series(dataset_df["stop_first"]).map(lambda value: float(value) if value is not None else np.nan)
    rows: list[dict[str, Any]] = []
    for feature in selected_features:
        if feature not in NUMERIC_FEATURES or feature not in dataset_df.columns:
            continue
        feature_values = pd.to_numeric(dataset_df[feature], errors="coerce")
        paired = pd.DataFrame({"feature": feature_values, "target": stop_values}).dropna()
        spearman = spearman_or_none(paired["feature"], paired["target"])
        rows.append(
            {
                "feature": feature,
                "target": "stop_first",
                "count": int(len(paired.index)),
                "spearman": spearman,
                "abs_spearman": abs(spearman) if spearman is not None else None,
            }
        )
    return pd.DataFrame(rows, columns=["feature", "target", "count", "spearman", "abs_spearman"])


def summarize_slice(group_df: pd.DataFrame) -> dict[str, Any]:
    """Summarize one bin/category slice."""

    future_60 = pd.to_numeric(group_df.get("future_return_60m"), errors="coerce")
    mfe_atr = pd.to_numeric(group_df.get("mfe_atr"), errors="coerce")
    mae_atr = pd.to_numeric(group_df.get("mae_atr"), errors="coerce")
    clean_future = future_60.dropna()
    return {
        "median_future_return_60m": float(clean_future.median()) if not clean_future.empty else None,
        "mean_future_return_60m": float(clean_future.mean()) if not clean_future.empty else None,
        "positive_rate_60m": float((clean_future > 0).mean()) if not clean_future.empty else None,
        "median_mfe_atr": float(mfe_atr.dropna().median()) if not mfe_atr.dropna().empty else None,
        "median_mae_atr": float(mae_atr.dropna().median()) if not mae_atr.dropna().empty else None,
        "stop_first_rate": bool_rate(group_df["stop_first"]) if "stop_first" in group_df.columns else None,
        "tp_first_rate": bool_rate(group_df["tp_first"]) if "tp_first" in group_df.columns else None,
    }


def build_quantile_bins(
    dataset_df: pd.DataFrame,
    selected_features: list[str],
    bins: int,
    min_count: int,
) -> pd.DataFrame:
    """Build quantile bins for numeric features."""

    rows: list[dict[str, Any]] = []
    for feature in selected_features:
        if feature not in NUMERIC_FEATURES or feature not in dataset_df.columns:
            continue
        working_df = dataset_df.copy()
        working_df[feature] = pd.to_numeric(working_df[feature], errors="coerce").replace([np.inf, -np.inf], np.nan)
        working_df = working_df.dropna(subset=[feature]).copy()
        if working_df.empty:
            continue

        unique_count = int(working_df[feature].nunique(dropna=True))
        if unique_count <= 1:
            working_df["_bin_code"] = 1
        else:
            bin_count = max(1, min(int(bins), unique_count))
            try:
                working_df["_bin_code"] = pd.qcut(
                    working_df[feature],
                    q=bin_count,
                    labels=False,
                    duplicates="drop",
                )
                working_df["_bin_code"] = working_df["_bin_code"].astype(int) + 1
            except ValueError:
                working_df["_bin_code"] = 1

        for bin_code, group_df in working_df.groupby("_bin_code", dropna=False):
            feature_values = pd.to_numeric(group_df[feature], errors="coerce").dropna()
            row = {
                "feature": feature,
                "bin": int(bin_code) if pd.notna(bin_code) else None,
                "count": int(len(group_df.index)),
                "min": float(feature_values.min()) if not feature_values.empty else None,
                "max": float(feature_values.max()) if not feature_values.empty else None,
                "meets_min_count": bool(len(group_df.index) >= min_count),
            }
            row.update(summarize_slice(group_df))
            rows.append(row)

    columns = [
        "feature",
        "bin",
        "count",
        "min",
        "max",
        "median_future_return_60m",
        "mean_future_return_60m",
        "positive_rate_60m",
        "median_mfe_atr",
        "median_mae_atr",
        "stop_first_rate",
        "tp_first_rate",
        "meets_min_count",
    ]
    return pd.DataFrame(rows, columns=columns)


def build_categorical_bins(
    dataset_df: pd.DataFrame,
    selected_features: list[str],
    min_count: int,
) -> pd.DataFrame:
    """Build grouped summaries for categorical features."""

    rows: list[dict[str, Any]] = []
    for feature in CATEGORICAL_FEATURES:
        if feature not in selected_features or feature not in dataset_df.columns:
            continue
        working_df = dataset_df.copy()
        working_df["_category"] = working_df[feature].where(working_df[feature].notna(), "unknown").astype(str)
        for category, group_df in working_df.groupby("_category", dropna=False):
            row = {
                "feature": feature,
                "category": category,
                "count": int(len(group_df.index)),
                "meets_min_count": bool(len(group_df.index) >= min_count),
            }
            row.update(summarize_slice(group_df))
            rows.append(row)

    columns = [
        "feature",
        "category",
        "count",
        "median_future_return_60m",
        "mean_future_return_60m",
        "positive_rate_60m",
        "median_mfe_atr",
        "median_mae_atr",
        "stop_first_rate",
        "tp_first_rate",
        "meets_min_count",
    ]
    return pd.DataFrame(rows, columns=columns)


def top_ic_rows(ic_df: pd.DataFrame, target: str, min_count: int, limit: int = 5) -> pd.DataFrame:
    """Return top IC rows for a target."""

    if ic_df.empty:
        return ic_df
    subset = ic_df[ic_df["target"] == target].copy()
    subset["count"] = pd.to_numeric(subset["count"], errors="coerce").fillna(0)
    robust = subset[subset["count"] >= min_count].copy()
    if robust.empty:
        robust = subset
    robust["abs_spearman"] = pd.to_numeric(robust["abs_spearman"], errors="coerce")
    return robust.sort_values(["abs_spearman", "count"], ascending=[False, False]).head(limit)


def ic_value(ic_df: pd.DataFrame, feature: str, target: str) -> float | None:
    """Return one IC value."""

    if ic_df.empty:
        return None
    row = ic_df[(ic_df["feature"] == feature) & (ic_df["target"] == target)]
    if row.empty:
        return None
    return finite_or_none(row.iloc[0].get("spearman"))


def bin_edge_medians(feature_bins_df: pd.DataFrame, feature: str) -> tuple[float | None, float | None]:
    """Return first/last quantile median future returns for a feature."""

    if feature_bins_df.empty:
        return None, None
    subset = feature_bins_df[feature_bins_df["feature"] == feature].sort_values("bin")
    if subset.empty:
        return None, None
    first = finite_or_none(subset.iloc[0].get("median_future_return_60m"))
    last = finite_or_none(subset.iloc[-1].get("median_future_return_60m"))
    return first, last


def feature_screen_answer(
    feature: str,
    ic_df: pd.DataFrame,
    feature_bins_df: pd.DataFrame,
    min_count: int,
) -> dict[str, Any]:
    """Assess whether one feature has screening value."""

    spearman = ic_value(ic_df, feature, "future_return_60m")
    first_median, last_median = bin_edge_medians(feature_bins_df, feature)
    has_value = bool(spearman is not None and abs(spearman) >= PREDICTIVE_IC_THRESHOLD)
    if has_value:
        answer = f"{feature} 与 future_return_60m 的 IC 达到筛选阈值"
    elif spearman is None:
        answer = f"{feature} 样本不足或缺失，暂不能判断筛选价值"
    else:
        answer = f"{feature} 没有达到稳定筛选阈值"
    return {
        "feature": feature,
        "basis_target": "future_return_60m",
        "spearman": spearman,
        "first_bin_median_future_return_60m": first_median,
        "last_bin_median_future_return_60m": last_median,
        "min_count": min_count,
        "has_screening_value": has_value,
        "answer": answer,
    }


def assess_atr_risk(feature_bins_df: pd.DataFrame, ic_df: pd.DataFrame) -> dict[str, Any]:
    """Assess low/high ATR risk zones."""

    subset = feature_bins_df[feature_bins_df["feature"] == "atr_pct"].sort_values("bin") if not feature_bins_df.empty else pd.DataFrame()
    if subset.empty:
        return {"answer": "atr_pct 缺失或样本不足，无法判断过低/过高风险区间"}
    first = finite_or_none(subset.iloc[0].get("median_future_return_60m"))
    last = finite_or_none(subset.iloc[-1].get("median_future_return_60m"))
    middle = subset.iloc[1:-1] if len(subset.index) > 2 else pd.DataFrame()
    middle_median = (
        float(pd.to_numeric(middle["median_future_return_60m"], errors="coerce").median())
        if not middle.empty
        else None
    )
    risk_low = bool(first is not None and middle_median is not None and first < middle_median)
    risk_high = bool(last is not None and middle_median is not None and last < middle_median)
    if risk_low and risk_high:
        answer = "atr_pct 低分位和高分位都弱于中间区间，存在两端风险"
    elif risk_low:
        answer = "atr_pct 低分位弱于中间区间，低波动可能是风险区"
    elif risk_high:
        answer = "atr_pct 高分位弱于中间区间，高波动可能是风险区"
    else:
        answer = "atr_pct 分箱没有显示明确的过低/过高风险区间"
    return {
        "spearman": ic_value(ic_df, "atr_pct", "future_return_60m"),
        "low_bin_median_future_return_60m": first,
        "middle_bin_median_future_return_60m": middle_median,
        "high_bin_median_future_return_60m": last,
        "low_volatility_risk": risk_low,
        "high_volatility_risk": risk_high,
        "answer": answer,
    }


def assess_breakout_distance(ic_df: pd.DataFrame, feature_bins_df: pd.DataFrame) -> dict[str, Any]:
    """Assess whether larger breakout distance is worse."""

    spearman = ic_value(ic_df, "breakout_distance_atr", "future_return_60m")
    first, last = bin_edge_medians(feature_bins_df, "breakout_distance_atr")
    larger_worse = bool(
        spearman is not None
        and spearman < -0.05
        and first is not None
        and last is not None
        and last < first
    )
    if larger_worse:
        answer = "breakout_distance_atr 越大，future_return_60m 越差的证据成立"
    elif spearman is None:
        answer = "breakout_distance_atr 样本不足或缺失，无法判断"
    else:
        answer = "没有足够证据证明 breakout_distance_atr 越大越差"
    return {
        "spearman_future_return_60m": spearman,
        "low_bin_median_future_return_60m": first,
        "high_bin_median_future_return_60m": last,
        "larger_is_worse": larger_worse,
        "answer": answer,
    }


def assess_wick_body(stop_ic_df: pd.DataFrame, ic_df: pd.DataFrame) -> dict[str, Any]:
    """Assess wick/body fake-breakout detection value."""

    features = ["upper_wick_ratio", "lower_wick_ratio", "body_ratio", "range_atr"]
    rows = []
    for feature in features:
        rows.append(
            {
                "feature": feature,
                "future_return_60m_ic": ic_value(ic_df, feature, "future_return_60m"),
                "stop_first_ic": ic_value(stop_ic_df, feature, "stop_first"),
            }
        )
    strong = [
        row
        for row in rows
        if row["stop_first_ic"] is not None and abs(row["stop_first_ic"]) >= STOP_IC_THRESHOLD
    ]
    if strong:
        answer = "wick/body 结构与 stop_first 有可观察关系，可作为假突破候选特征继续验证"
    else:
        answer = "wick/body 结构未达到假突破识别阈值"
    return {"features": rows, "has_fake_breakout_signal": bool(strong), "answer": answer}


def build_summary(
    report_dir: Path,
    signal_trace_path: Path,
    output_dir: Path,
    dataset_df: pd.DataFrame,
    feature_ic_df: pd.DataFrame,
    stop_ic_df: pd.DataFrame,
    feature_bins_df: pd.DataFrame,
    categorical_bins_df: pd.DataFrame,
    selected_features: list[str],
    horizons: list[int],
    bins: int,
    min_count: int,
    warnings: list[str],
    timezone_name: str,
    outcome_source: str,
) -> dict[str, Any]:
    """Build feature_summary.json payload."""

    missing_rates = {
        column: float(dataset_df[column].isna().mean()) if column in dataset_df.columns and len(dataset_df.index) else None
        for column in selected_features + label_columns(horizons)
    }
    future_60 = pd.to_numeric(dataset_df.get("future_return_60m"), errors="coerce") if not dataset_df.empty else pd.Series(dtype=float)
    top_future = top_ic_rows(feature_ic_df, "future_return_60m", min_count)
    top_stop = top_ic_rows(stop_ic_df, "stop_first", min_count)
    robust_future = top_future[
        pd.to_numeric(top_future.get("count", pd.Series(dtype=float)), errors="coerce").fillna(0) >= min_count
    ].copy()
    robust_future["abs_spearman"] = pd.to_numeric(robust_future.get("abs_spearman"), errors="coerce")
    predictive_features = robust_future[robust_future["abs_spearman"] >= PREDICTIVE_IC_THRESHOLD]
    signal_feature_hypothesis_failed = bool(predictive_features.empty)
    if signal_feature_hypothesis_failed:
        append_warning(warnings, "signal_feature_hypothesis_failed=true")

    diagnostic_answers = {
        "strongest_future_return_60m_features": dataframe_records(top_future),
        "strongest_stop_first_features": dataframe_records(top_stop),
        "breakout_distance_atr_larger_is_worse": assess_breakout_distance(feature_ic_df, feature_bins_df),
        "rsi_screening_value": feature_screen_answer("rsi", feature_ic_df, feature_bins_df, min_count),
        "ema_spread_pct_screening_value": feature_screen_answer("ema_spread_pct", feature_ic_df, feature_bins_df, min_count),
        "atr_pct_risk_zones": assess_atr_risk(feature_bins_df, feature_ic_df),
        "volume_zscore_screening_value": feature_screen_answer("volume_zscore_30m", feature_ic_df, feature_bins_df, min_count),
        "wick_body_fake_breakout": assess_wick_body(stop_ic_df, feature_ic_df),
        "signal_feature_hypothesis_failed": signal_feature_hypothesis_failed,
    }

    return {
        "report_dir": str(report_dir),
        "signal_trace_path": str(signal_trace_path),
        "output_dir": str(output_dir),
        "timezone": timezone_name,
        "horizons": horizons,
        "bins": int(bins),
        "min_count": int(min_count),
        "selected_features": selected_features,
        "outcome_source": outcome_source,
        "entry_count": int(len(dataset_df.index)),
        "field_missing_rate": missing_rates,
        "label_distribution": {
            "future_return_60m_non_null_count": int(future_60.dropna().count()),
            "future_return_60m_positive_rate": float((future_60.dropna() > 0).mean()) if not future_60.dropna().empty else None,
            "future_return_60m_median": float(future_60.dropna().median()) if not future_60.dropna().empty else None,
            "good_signal_60m_rate": bool_rate(dataset_df["good_signal_60m"]) if "good_signal_60m" in dataset_df.columns else None,
            "bad_signal_60m_rate": bool_rate(dataset_df["bad_signal_60m"]) if "bad_signal_60m" in dataset_df.columns else None,
            "stop_first_rate": bool_rate(dataset_df["stop_first"]) if "stop_first" in dataset_df.columns else None,
            "tp_first_rate": bool_rate(dataset_df["tp_first"]) if "tp_first" in dataset_df.columns else None,
        },
        "warnings": warnings,
        "diagnostic_answers": diagnostic_answers,
        "categorical_features_analyzed": sorted(categorical_bins_df["feature"].dropna().unique().tolist())
        if not categorical_bins_df.empty
        else [],
    }


def render_markdown(summary: dict[str, Any]) -> str:
    """Render a human-readable feature research report."""

    answers = summary.get("diagnostic_answers") or {}
    top_future = answers.get("strongest_future_return_60m_features") or []
    top_stop = answers.get("strongest_stop_first_features") or []
    warnings = summary.get("warnings") or []

    def rows_to_lines(rows: list[dict[str, Any]], target_name: str) -> str:
        if not rows:
            return "- 样本不足或没有可用数值特征"
        lines = []
        for row in rows:
            lines.append(
                f"- {row.get('feature')}: {target_name} IC={format_number(row.get('spearman'), 4)}, "
                f"count={row.get('count')}"
            )
        return "\n".join(lines)

    warning_lines = "\n".join(f"- {item}" for item in warnings) if warnings else "- 无"
    failed = bool(answers.get("signal_feature_hypothesis_failed"))
    return (
        "# Signal Lab 特征研究报告\n\n"
        "## 核心结论\n"
        f"- entry signal 数：{summary.get('entry_count')}\n"
        f"- outcome 来源：{summary.get('outcome_source')}\n"
        f"- signal_feature_hypothesis_failed={str(failed).lower()}\n\n"
        "## 哪些特征和 future_return_60m 关系最强？\n"
        f"{rows_to_lines(top_future, 'future_return_60m')}\n\n"
        "## 哪些特征和 stop_first 最相关？\n"
        f"{rows_to_lines(top_stop, 'stop_first')}\n\n"
        "## 必答诊断\n"
        f"- breakout_distance_atr 是否越大越差：{(answers.get('breakout_distance_atr_larger_is_worse') or {}).get('answer')}\n"
        f"- rsi 是否有筛选价值：{(answers.get('rsi_screening_value') or {}).get('answer')}\n"
        f"- ema_spread_pct 是否有筛选价值：{(answers.get('ema_spread_pct_screening_value') or {}).get('answer')}\n"
        f"- atr_pct 是否存在过低/过高风险区间：{(answers.get('atr_pct_risk_zones') or {}).get('answer')}\n"
        f"- volume_zscore 是否能改善信号：{(answers.get('volume_zscore_screening_value') or {}).get('answer')}\n"
        f"- wick/body 结构是否能识别假突破：{(answers.get('wick_body_fake_breakout') or {}).get('answer')}\n\n"
        "## 输出文件\n"
        "- feature_dataset.csv\n"
        "- feature_summary.json\n"
        "- feature_ic.csv\n"
        "- feature_bins.csv\n"
        "- categorical_feature_bins.csv\n"
        "- feature_report.md\n\n"
        "## Warning\n"
        f"{warning_lines}\n"
    )


def write_json(path: Path, payload: Any) -> None:
    """Write JSON with UTF-8 encoding."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(payload), ensure_ascii=False, indent=2), encoding="utf-8")


def write_dataframe(path: Path, df: pd.DataFrame) -> None:
    """Write a DataFrame as UTF-8 CSV."""

    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")


def write_outputs(
    output_dir: Path,
    dataset_df: pd.DataFrame,
    summary: dict[str, Any],
    feature_ic_df: pd.DataFrame,
    feature_bins_df: pd.DataFrame,
    categorical_bins_df: pd.DataFrame,
    markdown: str,
) -> None:
    """Write all Signal Lab artifacts."""

    output_dir.mkdir(parents=True, exist_ok=True)
    write_dataframe(output_dir / "feature_dataset.csv", dataset_df)
    write_json(output_dir / "feature_summary.json", summary)
    write_dataframe(output_dir / "feature_ic.csv", feature_ic_df)
    write_dataframe(output_dir / "feature_bins.csv", feature_bins_df)
    write_dataframe(output_dir / "categorical_feature_bins.csv", categorical_bins_df)
    (output_dir / "feature_report.md").write_text(markdown, encoding="utf-8")


def run_research(
    report_dir: Path,
    signal_trace_path: Path,
    output_dir: Path,
    horizons: list[int],
    bins: int,
    min_count: int,
    selected_features: list[str],
    timezone_name: str,
    bars_from_db: bool,
    data_check_strict: bool,
    logger: logging.Logger,
    bars_df: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Run the full Signal Lab feature workflow."""

    ZoneInfo(timezone_name)
    if bins <= 0:
        raise SignalFeatureResearchError("--bins 必须为正整数")
    if min_count <= 0:
        raise SignalFeatureResearchError("--min-count 必须为正整数")

    warnings: list[str] = []
    label_horizon_values = label_horizons(horizons)
    warn_missing_source_columns(signal_trace_path, selected_features, warnings)
    trace_df = read_signal_trace(signal_trace_path, timezone_name)
    entry_df = ensure_signal_ids(prepare_entry_signals(trace_df, warnings))

    if bars_df is None:
        if not bars_from_db:
            raise SignalFeatureResearchError("--no-bars-from-db 已设置，但当前 CLI 未提供替代 bars 输入")
        bar_df = load_feature_bars_from_db(entry_df, label_horizon_values, timezone_name, warnings, logger)
    else:
        bar_df = dataframe_bars_to_ohlc(bars_df, timezone_name)

    feature_df = build_feature_dataset(entry_df, bar_df, selected_features, warnings)
    outcome_df, outcome_source = load_or_compute_outcomes(report_dir, entry_df, bar_df, label_horizon_values, warnings)
    label_df = build_label_frame(outcome_df, label_horizon_values, warnings)
    dataset_df = merge_features_and_labels(feature_df, label_df)

    output_columns = BASE_COLUMNS + selected_features + label_columns(label_horizon_values)
    for column in output_columns:
        if column not in dataset_df.columns:
            dataset_df[column] = None
    dataset_df = dataset_df[output_columns]

    feature_ic_df = compute_feature_ic(dataset_df, selected_features, label_horizon_values)
    stop_ic_df = compute_stop_first_ic(dataset_df, selected_features)
    feature_bins_df = build_quantile_bins(dataset_df, selected_features, bins, min_count)
    categorical_bins_df = build_categorical_bins(dataset_df, selected_features, min_count)
    summary = build_summary(
        report_dir=report_dir,
        signal_trace_path=signal_trace_path,
        output_dir=output_dir,
        dataset_df=dataset_df,
        feature_ic_df=feature_ic_df,
        stop_ic_df=stop_ic_df,
        feature_bins_df=feature_bins_df,
        categorical_bins_df=categorical_bins_df,
        selected_features=selected_features,
        horizons=label_horizon_values,
        bins=bins,
        min_count=min_count,
        warnings=warnings,
        timezone_name=timezone_name,
        outcome_source=outcome_source,
    )
    markdown = render_markdown(summary)

    if data_check_strict:
        future_60 = pd.to_numeric(dataset_df.get("future_return_60m"), errors="coerce") if not dataset_df.empty else pd.Series(dtype=float)
        if dataset_df.empty:
            raise SignalFeatureResearchError("--data-check-strict: feature_dataset 为空")
        if future_60.dropna().empty:
            raise SignalFeatureResearchError("--data-check-strict: future_return_60m 全为空")

    write_outputs(output_dir, dataset_df, summary, feature_ic_df, feature_bins_df, categorical_bins_df, markdown)
    return summary


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""

    args = parse_args(argv)
    ensure_headless_runtime()
    logger = setup_logging("research_signal_features", verbose=args.verbose)

    try:
        report_dir = resolve_path(args.report_dir)
        signal_trace_path = resolve_path(args.signal_trace, report_dir / "signal_trace.csv")
        output_dir = resolve_path(args.output_dir, report_dir / "signal_feature_research")
        horizons = parse_horizons(args.horizons)
        selected_features = parse_feature_list(args.feature_list)
        summary = run_research(
            report_dir=report_dir,
            signal_trace_path=signal_trace_path,
            output_dir=output_dir,
            horizons=horizons,
            bins=int(args.bins),
            min_count=int(args.min_count),
            selected_features=selected_features,
            timezone_name=args.timezone,
            bars_from_db=bool(args.bars_from_db),
            data_check_strict=bool(args.data_check_strict),
            logger=logger,
        )
        print_json_block(
            "Signal feature research summary:",
            {
                "output_dir": output_dir,
                "entry_count": summary.get("entry_count"),
                "outcome_source": summary.get("outcome_source"),
                "signal_feature_hypothesis_failed": (
                    summary.get("diagnostic_answers") or {}
                ).get("signal_feature_hypothesis_failed"),
                "warnings": summary.get("warnings"),
            },
        )
        if args.json:
            print(json.dumps(to_jsonable(summary), ensure_ascii=False, indent=2))
        return 0
    except (SignalFeatureResearchError, SignalOutcomeError) as exc:
        log_event(logger, logging.ERROR, "signal_features.error", str(exc))
        return 2
    except Exception:
        logger.exception(
            "Unexpected error during signal feature research",
            extra={"event": "signal_features.unexpected_error"},
        )
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
