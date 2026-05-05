#!/usr/bin/env python3
"""Offline HTF signal research for 1h regime, 15m structure, and 5m reclaim."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from analyze_signal_outcomes import (
    configure_sqlite_settings,
    dataframe_bars_to_ohlc,
    resolve_exchange,
    split_vt_symbol,
)
from common_runtime import PROJECT_ROOT, ensure_headless_runtime, log_event, print_json_block, setup_logging, to_jsonable
from history_time_utils import DEFAULT_TIMEZONE, HistoryRange, parse_history_range
from history_utils import get_database_timezone


DEFAULT_VT_SYMBOL = "BTCUSDT_SWAP_OKX.GLOBAL"
DEFAULT_HORIZONS = "60,120,240,480"
DEFAULT_STOP_ATR_GRID = "1.5,2.0,2.5,3.0,4.0"
DEFAULT_TP_ATR_GRID = "2.0,3.0,4.0,5.0,6.0"
DEFAULT_DONCHIAN_WINDOW = 20
DEFAULT_VWAP_WINDOW = 48
DEFAULT_COOLDOWN_BARS_5M = 6
ROLLING_PERCENTILE_WINDOW = 240
PULLBACK_LOOKBACK_BARS_5M = 6
TIMEFRAME_MINUTES = {"1m": 1, "5m": 5, "15m": 15, "1h": 60}
SPLIT_RANGES = {
    "train": ("2025-01-01", "2025-09-30"),
    "validation": ("2025-10-01", "2025-12-31"),
    "oos": ("2026-01-01", "2026-03-31"),
    "full": ("2025-01-01", "2026-03-31"),
}
POLICY_NAMES = [
    "htf_1h_ema_regime_only",
    "htf_1h_ema_15m_ema_structure",
    "htf_1h_ema_15m_vwap_structure",
    "htf_1h_ema_15m_donchian_structure",
    "htf_1h_15m_structure_with_vol_cap",
    "htf_1h_15m_structure_strict_vol_cap",
    "htf_1h_15m_structure_no_overextension",
    "htf_1h_15m_structure_5m_pullback_reclaim",
    "htf_1h_15m_structure_5m_pullback_reclaim_vol_cap",
    "htf_1h_15m_structure_5m_pullback_reclaim_strict",
]
PULLBACK_RECLAIM_POLICY_NAMES = {
    "htf_1h_15m_structure_5m_pullback_reclaim",
    "htf_1h_15m_structure_5m_pullback_reclaim_vol_cap",
    "htf_1h_15m_structure_5m_pullback_reclaim_strict",
}
POLICY_DESCRIPTIONS = {
    "htf_1h_ema_regime_only": "1h close/EMA50/EMA200 regime baseline.",
    "htf_1h_ema_15m_ema_structure": "1h regime plus 15m EMA21/EMA55 structure.",
    "htf_1h_ema_15m_vwap_structure": "1h regime plus 15m VWAP and EMA structure.",
    "htf_1h_ema_15m_donchian_structure": "1h regime plus 15m Donchian mid/slope structure.",
    "htf_1h_15m_structure_with_vol_cap": "15m EMA structure with loose volatility cap.",
    "htf_1h_15m_structure_strict_vol_cap": "15m EMA structure with strict volatility cap.",
    "htf_1h_15m_structure_no_overextension": "15m EMA structure with overextension filters.",
    "htf_1h_15m_structure_5m_pullback_reclaim": "1h/15m trend with 5m pullback then EMA21 reclaim.",
    "htf_1h_15m_structure_5m_pullback_reclaim_vol_cap": "5m pullback reclaim with loose volatility cap.",
    "htf_1h_15m_structure_5m_pullback_reclaim_strict": "5m pullback reclaim with volatility and overextension filters.",
}


class HtfSignalResearchError(Exception):
    """Raised when HTF signal research cannot continue."""


@dataclass(frozen=True, slots=True)
class BracketResult:
    """One virtual bracket simulation result."""

    exit_type: str
    r_multiple: float


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description="Research HTF offline signal candidates.")
    parser.add_argument("--vt-symbol", default=DEFAULT_VT_SYMBOL, help=f"Default: {DEFAULT_VT_SYMBOL}.")
    parser.add_argument("--start", help="Start date/datetime. Defaults to the selected split preset.")
    parser.add_argument("--end", help="End date/datetime. Defaults to the selected split preset.")
    parser.add_argument("--split", choices=sorted(SPLIT_RANGES), default="train", help="Sample split preset.")
    parser.add_argument(
        "--timezone",
        default=DEFAULT_TIMEZONE,
        help=f"Timezone used for bars and reports. Default: {DEFAULT_TIMEZONE}.",
    )
    parser.add_argument("--output-dir", help="Default: reports/research/htf_signals/<split>.")
    parser.add_argument("--horizons", default=DEFAULT_HORIZONS, help=f"Comma-separated minutes. Default: {DEFAULT_HORIZONS}.")
    parser.add_argument("--stop-atr-grid", default=DEFAULT_STOP_ATR_GRID, help=f"Default: {DEFAULT_STOP_ATR_GRID}.")
    parser.add_argument("--tp-atr-grid", default=DEFAULT_TP_ATR_GRID, help=f"Default: {DEFAULT_TP_ATR_GRID}.")
    parser.add_argument(
        "--cooldown-bars-5m",
        type=int,
        default=DEFAULT_COOLDOWN_BARS_5M,
        help=f"Cooldown per policy/direction in 5m bars. Default: {DEFAULT_COOLDOWN_BARS_5M}.",
    )
    parser.add_argument("--max-signals", type=int, help="Optional global signal cap after cooldown.")
    parser.add_argument("--data-check-strict", action="store_true", help="Fail when requested 1m data has gaps.")
    parser.add_argument(
        "--bars-from-db",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Load 1m bars from vn.py sqlite. Default: enabled.",
    )
    parser.add_argument("--json", action="store_true", help="Print htf_policy_summary.json payload to stdout.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose structured logs.")
    return parser.parse_args(argv)


def resolve_path(path_arg: str | Path | None, default_path: Path) -> Path:
    """Resolve paths relative to the project root."""

    path = Path(path_arg) if path_arg else default_path
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def parse_number_list(raw_value: str, option_name: str, *, integer: bool = False) -> list[Any]:
    """Parse comma-separated positive numbers."""

    values: list[Any] = []
    for token in str(raw_value or "").split(","):
        text = token.strip()
        if not text:
            continue
        try:
            value = int(text) if integer else float(text)
        except ValueError as exc:
            raise HtfSignalResearchError(f"{option_name} 包含非法数字: {text!r}") from exc
        if value <= 0:
            raise HtfSignalResearchError(f"{option_name} 必须为正数: {value}")
        values.append(value)
    if not values:
        raise HtfSignalResearchError(f"{option_name} 不能为空")
    return sorted(set(values))


def parse_horizons(raw_value: str) -> list[int]:
    """Parse horizon minutes."""

    return [int(value) for value in parse_number_list(raw_value, "--horizons", integer=True)]


def resolve_split_range(split: str, start_arg: str | None, end_arg: str | None, timezone_name: str) -> HistoryRange:
    """Resolve CLI start/end using split defaults when needed."""

    default_start, default_end = SPLIT_RANGES[split]
    start = start_arg or default_start
    end = end_arg or default_end
    try:
        return parse_history_range(start, end, pd.Timedelta(minutes=1).to_pytimedelta(), timezone_name)
    except ValueError as exc:
        raise HtfSignalResearchError(str(exc)) from exc


def finite_or_none(value: Any) -> float | None:
    """Return a finite float or None."""

    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number):
        return None
    return number


def datetime_series_to_ns(series: pd.Series) -> np.ndarray:
    """Convert a datetime Series to UTC nanoseconds."""

    return pd.to_datetime(series, utc=True).dt.as_unit("ns").astype("int64").to_numpy()


def load_bars_from_db(
    vt_symbol: str,
    history_range: HistoryRange,
    max_horizon_minutes: int,
    timezone_name: str,
    logger: logging.Logger,
) -> pd.DataFrame:
    """Load requested 1m bars plus enough future bars for outcome labels."""

    symbol, exchange_value = split_vt_symbol(vt_symbol)
    exchange = resolve_exchange(exchange_value)

    from vnpy.trader.constant import Interval
    from vnpy.trader.database import get_database

    configure_sqlite_settings(logger)
    db_tz = get_database_timezone()
    query_start = history_range.start.astimezone(db_tz).replace(tzinfo=None)
    query_end = (
        history_range.end_exclusive + pd.Timedelta(minutes=max_horizon_minutes + 5).to_pytimedelta()
    ).astimezone(db_tz).replace(tzinfo=None)
    bars = get_database().load_bar_data(symbol, exchange, Interval.MINUTE, query_start, query_end)
    records = [bar_to_record(bar) for bar in bars]
    if not records:
        return pd.DataFrame(columns=["datetime", "open", "high", "low", "close", "volume"])
    return dataframe_bars_to_ohlc(pd.DataFrame(records), timezone_name)


def bar_to_record(bar: Any) -> dict[str, Any]:
    """Convert one vn.py BarData-like object into a plain record."""

    return {
        "datetime": getattr(bar, "datetime", None),
        "open": getattr(bar, "open_price", None),
        "high": getattr(bar, "high_price", None),
        "low": getattr(bar, "low_price", None),
        "close": getattr(bar, "close_price", None),
        "volume": getattr(bar, "volume", None),
    }


def normalize_1m_bars(bars_df: pd.DataFrame, timezone_name: str) -> pd.DataFrame:
    """Normalize a 1m OHLCV DataFrame."""

    result = dataframe_bars_to_ohlc(bars_df, timezone_name)
    result["datetime"] = pd.to_datetime(result["datetime"])
    for column in ["open", "high", "low", "close", "volume"]:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    result = result.dropna(subset=["datetime", "open", "high", "low", "close"])
    result = result.sort_values("datetime", kind="stable").drop_duplicates("datetime", keep="last")
    return result.reset_index(drop=True)


def filter_time_range(df: pd.DataFrame, history_range: HistoryRange) -> pd.DataFrame:
    """Filter bars to the signal research period."""

    if df.empty:
        return df.copy()
    dt = pd.to_datetime(df["datetime"])
    mask = (dt >= pd.Timestamp(history_range.start)) & (dt < pd.Timestamp(history_range.end_exclusive))
    return df.loc[mask].copy().reset_index(drop=True)


def resample_ohlcv(bars_1m: pd.DataFrame, minutes: int) -> pd.DataFrame:
    """Resample 1m bars into OHLCV bars timestamped at the last included 1m bar close."""

    if minutes <= 0:
        raise HtfSignalResearchError("resample minutes must be positive")
    if bars_1m.empty:
        return pd.DataFrame(columns=["datetime", "open", "high", "low", "close", "volume"])

    working = bars_1m.copy()
    working["datetime"] = pd.to_datetime(working["datetime"])
    working = working.sort_values("datetime", kind="stable").set_index("datetime")
    rule = f"{minutes}min"
    # Intervals are left-closed and labeled at the last constituent 1m close
    # below. A 15m bar built from 00:00..00:14 is timestamped 00:14, so a
    # 5m signal at 00:09 cannot asof-join that not-yet-completed 15m bar.
    grouped = working.resample(rule, label="left", closed="left")
    result = grouped.agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
    volume_count = grouped["volume"].count()
    last_timestamp = working.index.to_series().resample(rule, label="left", closed="left").max()
    expected_close_timestamp = last_timestamp.index + pd.Timedelta(minutes=minutes - 1)
    complete_group = last_timestamp == expected_close_timestamp
    result.loc[volume_count == 0, "volume"] = np.nan
    result = result.loc[complete_group.reindex(result.index).fillna(False)]
    result = result.dropna(subset=["open", "high", "low", "close"]).copy()
    result.index = result.index + pd.Timedelta(minutes=minutes - 1)
    return result.reset_index().rename(columns={"index": "datetime"})


def true_range(df: pd.DataFrame) -> pd.Series:
    """Compute true range."""

    high = pd.to_numeric(df["high"], errors="coerce")
    low = pd.to_numeric(df["low"], errors="coerce")
    close = pd.to_numeric(df["close"], errors="coerce")
    previous_close = close.shift(1)
    ranges = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    )
    return ranges.max(axis=1)


def rolling_percentile(series: pd.Series, window: int = ROLLING_PERCENTILE_WINDOW) -> pd.Series:
    """Return the current value percentile against prior rolling observations only."""

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


def compute_recent_return(close: pd.Series, timeframe_minutes: int, lookback_minutes: int) -> pd.Series:
    """Compute raw close-to-close return for a lookback supported by the timeframe."""

    if lookback_minutes < timeframe_minutes:
        return pd.Series(np.nan, index=close.index, dtype=float)
    periods = max(1, lookback_minutes // timeframe_minutes)
    return close / close.shift(periods) - 1.0


def compute_indicators(
    bars: pd.DataFrame,
    timeframe_minutes: int,
    donchian_window: int = DEFAULT_DONCHIAN_WINDOW,
    vwap_window: int = DEFAULT_VWAP_WINDOW,
    compute_percentiles: bool = False,
) -> pd.DataFrame:
    """Compute HTF research indicators for one timeframe."""

    df = bars.copy().sort_values("datetime", kind="stable").reset_index(drop=True)
    if df.empty:
        return df

    for column in ["open", "high", "low", "close", "volume"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    for span in [21, 50, 55, 200]:
        df[f"ema{span}"] = close.ewm(span=span, adjust=False, min_periods=1).mean()
    df["ema50_slope"] = df["ema50"] - df["ema50"].shift(1)

    df["atr14"] = true_range(df).rolling(14, min_periods=1).mean()
    df["atr_pct"] = df["atr14"] / close.replace(0, np.nan)

    df["donchian_high"] = high.rolling(donchian_window, min_periods=1).max()
    df["donchian_low"] = low.rolling(donchian_window, min_periods=1).min()
    df["donchian_mid"] = (df["donchian_high"] + df["donchian_low"]) / 2.0
    df["donchian_high_slope"] = df["donchian_high"] - df["donchian_high"].shift(1)
    df["donchian_low_slope"] = df["donchian_low"] - df["donchian_low"].shift(1)

    typical_price = (high + low + close) / 3.0
    volume_sum = volume.rolling(vwap_window, min_periods=1).sum()
    price_volume_sum = (typical_price * volume).rolling(vwap_window, min_periods=1).sum()
    rolling_typical = typical_price.rolling(vwap_window, min_periods=1).mean()
    df["rolling_vwap"] = (price_volume_sum / volume_sum.replace(0, np.nan)).where(volume_sum > 0, rolling_typical)

    for minutes in [5, 15, 30, 60]:
        df[f"recent_return_{minutes}m"] = compute_recent_return(close, timeframe_minutes, minutes)

    volatility_periods = 30 // timeframe_minutes if 30 >= timeframe_minutes else 0
    if volatility_periods >= 2:
        df["recent_volatility_30m"] = close.pct_change().rolling(volatility_periods, min_periods=2).std(ddof=0)
        previous_volume = volume.shift(1)
        volume_mean = previous_volume.rolling(volatility_periods, min_periods=2).mean()
        volume_std = previous_volume.rolling(volatility_periods, min_periods=2).std(ddof=0)
        df["volume_zscore_30m"] = (volume - volume_mean) / volume_std.replace(0, np.nan)
    else:
        df["recent_volatility_30m"] = np.nan
        df["volume_zscore_30m"] = np.nan

    bar_range = (high - low).replace(0, np.nan)
    df["body_ratio"] = (close - df["open"]).abs() / bar_range
    df["range_atr"] = (high - low) / df["atr14"].replace(0, np.nan)

    if compute_percentiles:
        for column in ["atr_pct", "recent_volatility_30m", "volume_zscore_30m", "body_ratio"]:
            df[f"{column}_percentile"] = rolling_percentile(df[column])
        df["directional_recent_return_30m_percentile_long"] = rolling_percentile(df["recent_return_30m"])
        df["directional_recent_return_30m_percentile_short"] = rolling_percentile(-df["recent_return_30m"])
    return df


def prefix_columns(df: pd.DataFrame, suffix: str) -> pd.DataFrame:
    """Append a timeframe suffix to all columns except datetime."""

    return df.rename(columns={column: f"{column}_{suffix}" for column in df.columns if column != "datetime"})


def build_timeframes(bars_1m: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Build 5m, 15m, and 1h bars from 1m data."""

    return {
        "1m": bars_1m.copy(),
        "5m": resample_ohlcv(bars_1m, 5),
        "15m": resample_ohlcv(bars_1m, 15),
        "1h": resample_ohlcv(bars_1m, 60),
    }


def build_indicator_frames(timeframes: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Compute indicators for the HTF research timeframes."""

    return {
        "5m": compute_indicators(timeframes["5m"], 5, compute_percentiles=False),
        "15m": compute_indicators(timeframes["15m"], 15, compute_percentiles=True),
        "1h": compute_indicators(timeframes["1h"], 60, compute_percentiles=False),
    }


def add_policy_conditions(aligned_df: pd.DataFrame) -> pd.DataFrame:
    """Add regime, structure, risk-filter, and pullback condition columns."""

    df = aligned_df.copy().sort_values("datetime", kind="stable").reset_index(drop=True)
    required_numeric_columns = [
        "close_1h",
        "ema50_1h",
        "ema200_1h",
        "ema50_slope_1h",
        "close_15m",
        "ema21_15m",
        "ema55_15m",
        "rolling_vwap_15m",
        "donchian_mid_15m",
        "donchian_high_slope_15m",
        "donchian_low_slope_15m",
        "atr_pct_percentile_15m",
        "recent_volatility_30m_percentile_15m",
        "directional_recent_return_30m_percentile_long_15m",
        "directional_recent_return_30m_percentile_short_15m",
        "volume_zscore_30m_percentile_15m",
        "body_ratio_percentile_15m",
        "low_5m",
        "high_5m",
        "ema21_5m",
        "close_5m",
        "atr14_15m",
    ]
    for column in required_numeric_columns:
        if column not in df.columns:
            df[column] = np.nan
    df["regime_long_1h"] = (
        (df["close_1h"] > df["ema50_1h"])
        & (df["ema50_1h"] > df["ema200_1h"])
        & (df["ema50_slope_1h"] > 0)
    )
    df["regime_short_1h"] = (
        (df["close_1h"] < df["ema50_1h"])
        & (df["ema50_1h"] < df["ema200_1h"])
        & (df["ema50_slope_1h"] < 0)
    )
    df["ema_structure_long_15m"] = (df["close_15m"] > df["ema21_15m"]) & (df["ema21_15m"] > df["ema55_15m"])
    df["ema_structure_short_15m"] = (df["close_15m"] < df["ema21_15m"]) & (df["ema21_15m"] < df["ema55_15m"])
    df["vwap_structure_long_15m"] = (df["close_15m"] > df["rolling_vwap_15m"]) & (df["ema21_15m"] > df["ema55_15m"])
    df["vwap_structure_short_15m"] = (df["close_15m"] < df["rolling_vwap_15m"]) & (df["ema21_15m"] < df["ema55_15m"])
    df["donchian_structure_long_15m"] = (df["close_15m"] > df["donchian_mid_15m"]) & (df["donchian_high_slope_15m"] >= 0)
    df["donchian_structure_short_15m"] = (df["close_15m"] < df["donchian_mid_15m"]) & (df["donchian_low_slope_15m"] <= 0)

    df["vol_cap_80_15m"] = (df["atr_pct_percentile_15m"] <= 0.8) & (df["recent_volatility_30m_percentile_15m"] <= 0.8)
    df["vol_cap_60_15m"] = (df["atr_pct_percentile_15m"] <= 0.6) & (df["recent_volatility_30m_percentile_15m"] <= 0.6)
    df["no_overextension_long_15m"] = (
        (df["directional_recent_return_30m_percentile_long_15m"] <= 0.8)
        & (df["volume_zscore_30m_percentile_15m"] <= 0.8)
        & (df["body_ratio_percentile_15m"] <= 0.8)
    )
    df["no_overextension_short_15m"] = (
        (df["directional_recent_return_30m_percentile_short_15m"] <= 0.8)
        & (df["volume_zscore_30m_percentile_15m"] <= 0.8)
        & (df["body_ratio_percentile_15m"] <= 0.8)
    )

    long_distance = pd.concat(
        [
            (df["low_5m"] - df["ema21_15m"]).abs(),
            (df["low_5m"] - df["rolling_vwap_15m"]).abs(),
        ],
        axis=1,
    ).min(axis=1)
    short_distance = pd.concat(
        [
            (df["high_5m"] - df["ema21_15m"]).abs(),
            (df["high_5m"] - df["rolling_vwap_15m"]).abs(),
        ],
        axis=1,
    ).min(axis=1)
    atr_15m = df["atr14_15m"].replace(0, np.nan)
    df["pullback_distance_atr_long_5m"] = long_distance / atr_15m
    df["pullback_distance_atr_short_5m"] = short_distance / atr_15m
    df["pullback_recent_long_5m"] = (
        df["pullback_distance_atr_long_5m"].rolling(PULLBACK_LOOKBACK_BARS_5M, min_periods=1).min() <= 0.5
    )
    df["pullback_recent_short_5m"] = (
        df["pullback_distance_atr_short_5m"].rolling(PULLBACK_LOOKBACK_BARS_5M, min_periods=1).min() <= 0.5
    )
    previous_close = df["close_5m"].shift(1)
    previous_ema21 = df["ema21_5m"].shift(1)
    df["reclaim_long_5m"] = (df["close_5m"] > df["ema21_5m"]) & (previous_close <= previous_ema21)
    df["reclaim_short_5m"] = (df["close_5m"] < df["ema21_5m"]) & (previous_close >= previous_ema21)
    df["next_5m_bar_time"] = df["datetime"].shift(-1)
    df["next_close_5m"] = df["close_5m"].shift(-1)
    return df


def align_indicator_frames(indicators: dict[str, pd.DataFrame], history_range: HistoryRange) -> pd.DataFrame:
    """Align latest completed 15m and 1h context onto every 5m bar."""

    base = prefix_columns(indicators["5m"], "5m")
    base["used_5m_bar_time"] = base["datetime"]
    base = base[(base["datetime"] >= pd.Timestamp(history_range.start)) & (base["datetime"] < pd.Timestamp(history_range.end_exclusive))]
    base = base.sort_values("datetime", kind="stable").reset_index(drop=True)
    if base.empty:
        return add_policy_conditions(base)

    context_15m = prefix_columns(indicators["15m"], "15m").sort_values("datetime", kind="stable")
    context_15m["used_15m_bar_time"] = context_15m["datetime"]
    context_1h = prefix_columns(indicators["1h"], "1h").sort_values("datetime", kind="stable")
    context_1h["used_1h_bar_time"] = context_1h["datetime"]
    # merge_asof(backward) is the closed-bar gate: only HTF bars whose close
    # timestamp is <= the current 5m close are visible to the signal.
    aligned = pd.merge_asof(
        base,
        context_15m,
        on="datetime",
        direction="backward",
    )
    aligned = pd.merge_asof(
        aligned,
        context_1h,
        on="datetime",
        direction="backward",
    )
    return add_policy_conditions(aligned)


def policy_condition(row: pd.Series, policy_name: str, direction: str) -> bool:
    """Return whether one policy is true for a row and direction."""

    side = "long" if direction == "long" else "short"
    regime = bool(row.get(f"regime_{side}_1h", False))
    ema_structure = bool(row.get(f"ema_structure_{side}_15m", False))
    vwap_structure = bool(row.get(f"vwap_structure_{side}_15m", False))
    donchian_structure = bool(row.get(f"donchian_structure_{side}_15m", False))
    no_overextension = bool(row.get(f"no_overextension_{side}_15m", False))
    pullback = bool(row.get(f"pullback_recent_{side}_5m", False)) and bool(row.get(f"reclaim_{side}_5m", False))

    if policy_name == "htf_1h_ema_regime_only":
        return regime
    if policy_name == "htf_1h_ema_15m_ema_structure":
        return regime and ema_structure
    if policy_name == "htf_1h_ema_15m_vwap_structure":
        return regime and vwap_structure
    if policy_name == "htf_1h_ema_15m_donchian_structure":
        return regime and donchian_structure
    if policy_name == "htf_1h_15m_structure_with_vol_cap":
        return regime and ema_structure and bool(row.get("vol_cap_80_15m", False))
    if policy_name == "htf_1h_15m_structure_strict_vol_cap":
        return regime and ema_structure and bool(row.get("vol_cap_60_15m", False))
    if policy_name == "htf_1h_15m_structure_no_overextension":
        return regime and ema_structure and no_overextension
    if policy_name == "htf_1h_15m_structure_5m_pullback_reclaim":
        return regime and ema_structure and pullback
    if policy_name == "htf_1h_15m_structure_5m_pullback_reclaim_vol_cap":
        return regime and ema_structure and pullback and bool(row.get("vol_cap_80_15m", False))
    if policy_name == "htf_1h_15m_structure_5m_pullback_reclaim_strict":
        return (
            regime
            and ema_structure
            and pullback
            and bool(row.get("vol_cap_60_15m", False))
            and no_overextension
        )
    raise HtfSignalResearchError(f"Unknown policy: {policy_name}")


def feature_value(row: pd.Series, column: str) -> float | None:
    """Return one numeric row feature."""

    return finite_or_none(row.get(column))


def is_pullback_reclaim_policy(policy_name: str) -> bool:
    """Return whether a policy requires a conservative next-5m-bar entry."""

    return policy_name in PULLBACK_RECLAIM_POLICY_NAMES


def build_signal_record(vt_symbol: str, policy_name: str, direction: str, row: pd.Series, signal_index: int) -> dict[str, Any]:
    """Build one signal dataset record before outcome labels."""

    signal_dt = pd.Timestamp(row["datetime"])
    delayed_entry = is_pullback_reclaim_policy(policy_name)
    raw_entry_dt = row.get("next_5m_bar_time") if delayed_entry else signal_dt
    entry_dt = None if raw_entry_dt is None or pd.isna(raw_entry_dt) else pd.Timestamp(raw_entry_dt)
    entry_price = feature_value(row, "next_close_5m" if delayed_entry else "close_5m")
    direction_sign = 1 if direction == "long" else -1
    directional_return_30m = feature_value(row, "recent_return_30m_15m")
    if directional_return_30m is not None:
        directional_return_30m *= direction_sign
    directional_percentile = feature_value(row, f"directional_recent_return_30m_percentile_{direction}_15m")
    pullback_distance_col = f"pullback_distance_atr_{direction}_5m"
    return {
        "signal_id": f"HTF-{signal_index:08d}",
        "_signal_dt": signal_dt,
        "_entry_dt": entry_dt,
        "signal_time": signal_dt.isoformat(),
        "entry_time": entry_dt.isoformat() if entry_dt is not None else None,
        "vt_symbol": vt_symbol,
        "policy_name": policy_name,
        "policy_description": POLICY_DESCRIPTIONS.get(policy_name),
        "direction": direction,
        "direction_sign": direction_sign,
        "entry_price": entry_price,
        "entry_delay_bars_5m": 1 if delayed_entry else 0,
        "used_1h_bar_time": iso_or_none(row.get("used_1h_bar_time")),
        "used_15m_bar_time": iso_or_none(row.get("used_15m_bar_time")),
        "used_5m_bar_time": iso_or_none(row.get("used_5m_bar_time")),
        "hour": int(signal_dt.hour),
        "weekday": int(signal_dt.weekday()),
        "close_1h": feature_value(row, "close_1h"),
        "ema50_1h": feature_value(row, "ema50_1h"),
        "ema200_1h": feature_value(row, "ema200_1h"),
        "ema50_slope_1h": feature_value(row, "ema50_slope_1h"),
        "regime_long_1h": bool(row.get("regime_long_1h", False)),
        "regime_short_1h": bool(row.get("regime_short_1h", False)),
        "close_15m": feature_value(row, "close_15m"),
        "ema21_15m": feature_value(row, "ema21_15m"),
        "ema55_15m": feature_value(row, "ema55_15m"),
        "rolling_vwap_15m": feature_value(row, "rolling_vwap_15m"),
        "donchian_high_15m": feature_value(row, "donchian_high_15m"),
        "donchian_low_15m": feature_value(row, "donchian_low_15m"),
        "donchian_mid_15m": feature_value(row, "donchian_mid_15m"),
        "donchian_high_slope_15m": feature_value(row, "donchian_high_slope_15m"),
        "donchian_low_slope_15m": feature_value(row, "donchian_low_slope_15m"),
        "ema_structure_long_15m": bool(row.get("ema_structure_long_15m", False)),
        "ema_structure_short_15m": bool(row.get("ema_structure_short_15m", False)),
        "vwap_structure_long_15m": bool(row.get("vwap_structure_long_15m", False)),
        "vwap_structure_short_15m": bool(row.get("vwap_structure_short_15m", False)),
        "donchian_structure_long_15m": bool(row.get("donchian_structure_long_15m", False)),
        "donchian_structure_short_15m": bool(row.get("donchian_structure_short_15m", False)),
        "close_5m": feature_value(row, "close_5m"),
        "ema21_5m": feature_value(row, "ema21_5m"),
        "pullback_recent_long_5m": bool(row.get("pullback_recent_long_5m", False)),
        "pullback_recent_short_5m": bool(row.get("pullback_recent_short_5m", False)),
        "reclaim_long_5m": bool(row.get("reclaim_long_5m", False)),
        "reclaim_short_5m": bool(row.get("reclaim_short_5m", False)),
        "pullback_distance_atr_5m": feature_value(row, pullback_distance_col),
        "atr_reference": feature_value(row, "atr14_15m"),
        "atr_pct": feature_value(row, "atr_pct_15m"),
        "atr_pct_percentile": feature_value(row, "atr_pct_percentile_15m"),
        "recent_volatility_30m": feature_value(row, "recent_volatility_30m_15m"),
        "recent_volatility_30m_percentile": feature_value(row, "recent_volatility_30m_percentile_15m"),
        "directional_recent_return_30m": directional_return_30m,
        "directional_recent_return_30m_percentile": directional_percentile,
        "volume_zscore_30m": feature_value(row, "volume_zscore_30m_15m"),
        "volume_zscore_30m_percentile": feature_value(row, "volume_zscore_30m_percentile_15m"),
        "body_ratio": feature_value(row, "body_ratio_15m"),
        "body_ratio_percentile": feature_value(row, "body_ratio_percentile_15m"),
        "range_atr": feature_value(row, "range_atr_15m"),
    }


def apply_cooldown_and_build_signals(
    aligned_df: pd.DataFrame,
    vt_symbol: str,
    cooldown_bars_5m: int,
    max_signals: int | None,
) -> pd.DataFrame:
    """Generate one signal row per policy/direction after cooldown de-duplication."""

    if cooldown_bars_5m < 0:
        raise HtfSignalResearchError("--cooldown-bars-5m 不能为负数")
    if max_signals is not None and max_signals <= 0:
        raise HtfSignalResearchError("--max-signals 必须为正整数")

    last_trigger: dict[tuple[str, str], int] = {}
    records: list[dict[str, Any]] = []
    signal_index = 1
    for bar_index, row in aligned_df.iterrows():
        for policy_name in POLICY_NAMES:
            chosen_direction: str | None = None
            if policy_condition(row, policy_name, "long"):
                chosen_direction = "long"
            elif policy_condition(row, policy_name, "short"):
                chosen_direction = "short"
            if chosen_direction is None:
                continue

            key = (policy_name, chosen_direction)
            previous_index = last_trigger.get(key)
            if previous_index is not None and int(bar_index) - previous_index < cooldown_bars_5m:
                continue

            record = build_signal_record(vt_symbol, policy_name, chosen_direction, row, signal_index)
            if record["entry_price"] is None or record.get("_entry_dt") is None:
                continue
            records.append(record)
            last_trigger[key] = int(bar_index)
            signal_index += 1
            if max_signals is not None and len(records) >= max_signals:
                return add_overlap_counts(pd.DataFrame(records), max_horizon_minutes=0)

    return pd.DataFrame(records)


def add_overlap_counts(signal_df: pd.DataFrame, max_horizon_minutes: int) -> pd.DataFrame:
    """Count prior same-policy/same-direction signals still inside the max horizon."""

    if signal_df.empty:
        signal_df["overlap_count"] = []
        return signal_df

    result = signal_df.copy()
    result["overlap_count"] = 0
    horizon_delta = pd.Timedelta(minutes=max_horizon_minutes)
    for (_policy, _direction), group in result.groupby(["policy_name", "direction"], sort=False):
        active: list[pd.Timestamp] = []
        for index, row in group.sort_values("_signal_dt", kind="stable").iterrows():
            current = pd.Timestamp(row["_signal_dt"])
            active = [timestamp for timestamp in active if timestamp + horizon_delta > current]
            result.loc[index, "overlap_count"] = len(active)
            active.append(current)
    return result


def direction_future_return_from_close(direction: str, entry_price: float, future_close: float) -> float | None:
    """Compute direction-adjusted return from entry to a future close."""

    if entry_price <= 0 or future_close <= 0:
        return None
    if direction == "long":
        return future_close / entry_price - 1.0
    if direction == "short":
        return entry_price / future_close - 1.0
    return None


def direction_excursions_from_extremes(
    direction: str,
    entry_price: float,
    max_high: float,
    min_low: float,
) -> tuple[float | None, float | None]:
    """Compute MFE/MAE using the same direction-aware return convention."""

    if entry_price <= 0 or max_high <= 0 or min_low <= 0:
        return None, None
    if direction == "long":
        return max(max_high / entry_price - 1.0, 0.0), max(1.0 - min_low / entry_price, 0.0)
    if direction == "short":
        return max(entry_price / min_low - 1.0, 0.0), max(1.0 - entry_price / max_high, 0.0)
    return None, None


def finite_min_max(values: np.ndarray) -> tuple[float | None, float | None]:
    """Return finite min/max for a numeric array."""

    clean = values[np.isfinite(values)]
    if clean.size == 0:
        return None, None
    return float(np.min(clean)), float(np.max(clean))


def direction_future_return(window_df: pd.DataFrame, direction: str, entry_price: float) -> float | None:
    """Compute direction-adjusted future return to the last close."""

    if window_df.empty or entry_price <= 0:
        return None
    future_close = finite_or_none(window_df["close"].iloc[-1])
    if future_close is None:
        return None
    return direction_future_return_from_close(direction, entry_price, future_close)


def direction_excursions(window_df: pd.DataFrame, direction: str, entry_price: float) -> tuple[float | None, float | None]:
    """Compute fractional MFE and MAE for one future window."""

    if window_df.empty or entry_price <= 0:
        return None, None
    high = pd.to_numeric(window_df["high"], errors="coerce")
    low = pd.to_numeric(window_df["low"], errors="coerce")
    min_low, _max_low = finite_min_max(low.to_numpy(dtype=float))
    _min_high, max_high = finite_min_max(high.to_numpy(dtype=float))
    if min_low is None or max_high is None:
        return None, None
    return direction_excursions_from_extremes(direction, entry_price, max_high, min_low)


def compute_signal_outcomes(
    signal_df: pd.DataFrame,
    bars_1m: pd.DataFrame,
    horizons: list[int],
    warnings: list[str],
) -> pd.DataFrame:
    """Add future return, MFE, and MAE labels to the signal dataset."""

    result = signal_df.copy()
    for horizon in horizons:
        for prefix in ["future_return", "mfe", "mae"]:
            result[f"{prefix}_{horizon}m"] = np.nan
    if result.empty:
        return result
    if bars_1m.empty:
        warnings.append("1m bars 为空，无法计算 HTF signal outcomes")
        return result

    bars = bars_1m.copy().sort_values("datetime", kind="stable").reset_index(drop=True)
    bars["datetime"] = pd.to_datetime(bars["datetime"])
    bar_ns = datetime_series_to_ns(bars["datetime"])
    high_values = pd.to_numeric(bars["high"], errors="coerce").to_numpy(dtype=float)
    low_values = pd.to_numeric(bars["low"], errors="coerce").to_numpy(dtype=float)
    close_values = pd.to_numeric(bars["close"], errors="coerce").to_numpy(dtype=float)
    max_bar_ns = int(bar_ns[-1])
    horizon_exceeded: dict[int, int] = {horizon: 0 for horizon in horizons}
    label_values: dict[str, np.ndarray] = {}
    for horizon in horizons:
        for prefix in ["future_return", "mfe", "mae"]:
            label_values[f"{prefix}_{horizon}m"] = np.full(len(result.index), np.nan, dtype=float)

    entry_time_column = "_entry_dt" if "_entry_dt" in result.columns else "_signal_dt"
    entry_times = pd.to_datetime(result[entry_time_column], utc=True, errors="coerce")
    entry_ns_values = entry_times.dt.as_unit("ns").astype("int64").to_numpy()
    entry_time_valid = entry_times.notna().to_numpy()
    entry_values = pd.to_numeric(result["entry_price"], errors="coerce").to_numpy(dtype=float)
    direction_values = result["direction"].astype(str).to_numpy()
    one_minute_ns = int(pd.Timedelta(minutes=1).value)

    for row_pos, entry_ns in enumerate(entry_ns_values):
        entry_price = float(entry_values[row_pos])
        direction = direction_values[row_pos]
        if not entry_time_valid[row_pos] or not np.isfinite(entry_price) or entry_price <= 0:
            continue
        start_index = int(np.searchsorted(bar_ns, entry_ns, side="right"))
        for horizon in horizons:
            target_ns = int(entry_ns + horizon * one_minute_ns)
            if target_ns > max_bar_ns:
                horizon_exceeded[horizon] += 1
            end_index = int(np.searchsorted(bar_ns, target_ns, side="right"))
            if end_index <= start_index:
                continue
            high_window = high_values[start_index:end_index]
            low_window = low_values[start_index:end_index]
            close_window = close_values[start_index:end_index]
            if close_window.size == 0 or not np.isfinite(close_window[-1]):
                continue
            min_low, _max_low = finite_min_max(low_window)
            _min_high, max_high = finite_min_max(high_window)
            if min_low is None or max_high is None:
                continue
            future_return = direction_future_return_from_close(direction, entry_price, float(close_window[-1]))
            mfe, mae = direction_excursions_from_extremes(direction, entry_price, max_high, min_low)
            if future_return is not None:
                label_values[f"future_return_{horizon}m"][row_pos] = future_return
            if mfe is not None:
                label_values[f"mfe_{horizon}m"][row_pos] = mfe
            if mae is not None:
                label_values[f"mae_{horizon}m"][row_pos] = mae

    for column, values in label_values.items():
        result[column] = values

    for horizon, count in horizon_exceeded.items():
        if count:
            warnings.append(f"horizon {horizon}m 超过可用 1m bar 数据范围的 HTF signal 数: {count}")
    return result


def simulate_bracket_for_signal(
    signal: pd.Series,
    bars: pd.DataFrame,
    bar_times: pd.Series,
    horizon_minutes: int,
    stop_atr: float,
    tp_atr: float,
) -> BracketResult | None:
    """Run a no-cost virtual bracket; same-bar stop/tp is stop first."""

    entry_price = finite_or_none(signal.get("entry_price"))
    atr = finite_or_none(signal.get("atr_reference"))
    if entry_price is None or atr is None or entry_price <= 0 or atr <= 0:
        return None
    direction = str(signal.get("direction"))
    raw_entry_dt = signal.get("_entry_dt", signal.get("_signal_dt"))
    if raw_entry_dt is None or pd.isna(raw_entry_dt):
        return None
    entry_dt = pd.Timestamp(raw_entry_dt)
    start_index = int(bar_times.searchsorted(entry_dt, side="right"))
    end_index = int(bar_times.searchsorted(entry_dt + pd.Timedelta(minutes=horizon_minutes), side="right"))
    window = bars.iloc[start_index:end_index].copy()
    if window.empty:
        return None

    stop_distance = stop_atr * atr
    if direction == "long":
        stop_price = entry_price - stop_distance
        tp_price = entry_price + tp_atr * atr
    else:
        stop_price = entry_price + stop_distance
        tp_price = entry_price - tp_atr * atr

    for bar in window.itertuples(index=False):
        high = finite_or_none(getattr(bar, "high", None))
        low = finite_or_none(getattr(bar, "low", None))
        if high is None or low is None:
            continue
        if direction == "long":
            stop_hit = low <= stop_price
            tp_hit = high >= tp_price
        else:
            stop_hit = high >= stop_price
            tp_hit = low <= tp_price
        if stop_hit and tp_hit:
            return BracketResult(exit_type="stop", r_multiple=-1.0)
        if stop_hit:
            return BracketResult(exit_type="stop", r_multiple=-1.0)
        if tp_hit:
            return BracketResult(exit_type="tp", r_multiple=tp_atr / stop_atr)

    close = finite_or_none(window["close"].iloc[-1])
    if close is None:
        return None
    if direction == "long":
        r_multiple = (close - entry_price) / stop_distance
    else:
        r_multiple = (entry_price - close) / stop_distance
    return BracketResult(exit_type="horizon", r_multiple=float(r_multiple))


def simulate_bracket_arrays(
    direction: str,
    entry_price: float,
    atr: float,
    high_window: np.ndarray,
    low_window: np.ndarray,
    close_window: np.ndarray,
    stop_atr: float,
    tp_atr: float,
) -> BracketResult | None:
    """Run one bracket simulation against numpy OHLC windows."""

    if entry_price <= 0 or atr <= 0 or high_window.size == 0 or low_window.size == 0 or close_window.size == 0:
        return None
    stop_distance = stop_atr * atr
    if direction == "long":
        stop_hits = low_window <= entry_price - stop_distance
        tp_hits = high_window >= entry_price + tp_atr * atr
    else:
        stop_hits = high_window >= entry_price + stop_distance
        tp_hits = low_window <= entry_price - tp_atr * atr

    stop_indices = np.flatnonzero(stop_hits)
    tp_indices = np.flatnonzero(tp_hits)
    first_stop = int(stop_indices[0]) if stop_indices.size else None
    first_tp = int(tp_indices[0]) if tp_indices.size else None
    if first_stop is not None and (first_tp is None or first_stop <= first_tp):
        return BracketResult(exit_type="stop", r_multiple=-1.0)
    if first_tp is not None:
        return BracketResult(exit_type="tp", r_multiple=tp_atr / stop_atr)

    close = close_window[-1]
    if not np.isfinite(close):
        return None
    if direction == "long":
        r_multiple = (close - entry_price) / stop_distance
    else:
        r_multiple = (entry_price - close) / stop_distance
    return BracketResult(exit_type="horizon", r_multiple=float(r_multiple))


def compute_bracket_grid(
    signal_df: pd.DataFrame,
    future_bars: pd.DataFrame,
    stop_grid: list[float],
    tp_grid: list[float],
    bracket_horizon: int,
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    """Compute policy-level bracket grid and best row per policy."""

    rows: list[dict[str, Any]] = []
    bars = future_bars.copy().sort_values("datetime", kind="stable").reset_index(drop=True)
    bars["datetime"] = pd.to_datetime(bars["datetime"])
    if bars.empty:
        bar_ns = np.array([], dtype=np.int64)
        high_values = np.array([], dtype=float)
        low_values = np.array([], dtype=float)
        close_values = np.array([], dtype=float)
    else:
        bar_ns = datetime_series_to_ns(bars["datetime"])
        high_values = pd.to_numeric(bars["high"], errors="coerce").to_numpy(dtype=float)
        low_values = pd.to_numeric(bars["low"], errors="coerce").to_numpy(dtype=float)
        close_values = pd.to_numeric(bars["close"], errors="coerce").to_numpy(dtype=float)

    for policy_name in POLICY_NAMES:
        policy_df = signal_df[signal_df["policy_name"] == policy_name].copy() if not signal_df.empty else pd.DataFrame()
        signal_inputs: list[tuple[str, float, float, int, int]] = []
        if not policy_df.empty and bar_ns.size:
            for _index, signal in policy_df.iterrows():
                entry_price = finite_or_none(signal.get("entry_price"))
                atr = finite_or_none(signal.get("atr_reference"))
                if entry_price is None or atr is None or entry_price <= 0 or atr <= 0:
                    continue
                raw_entry_dt = signal.get("_entry_dt", signal.get("_signal_dt"))
                if raw_entry_dt is None or pd.isna(raw_entry_dt):
                    continue
                entry_dt = pd.Timestamp(raw_entry_dt)
                start_index = int(np.searchsorted(bar_ns, entry_dt.value, side="right"))
                end_index = int(
                    np.searchsorted(
                        bar_ns,
                        (entry_dt + pd.Timedelta(minutes=bracket_horizon)).value,
                        side="right",
                    )
                )
                if end_index <= start_index:
                    continue
                signal_inputs.append((str(signal.get("direction")), float(entry_price), float(atr), start_index, end_index))
        for stop_atr in stop_grid:
            for tp_atr in tp_grid:
                results: list[BracketResult] = []
                if signal_inputs:
                    for direction, entry_price, atr, start_index, end_index in signal_inputs:
                        result = simulate_bracket_arrays(
                            direction,
                            entry_price,
                            atr,
                            high_values[start_index:end_index],
                            low_values[start_index:end_index],
                            close_values[start_index:end_index],
                            stop_atr,
                            tp_atr,
                        )
                        if result is not None:
                            results.append(result)
                r_values = [item.r_multiple for item in results]
                exit_types = [item.exit_type for item in results]
                count = len(results)
                rows.append(
                    {
                        "policy_name": policy_name,
                        "horizon_minutes": int(bracket_horizon),
                        "stop_atr": float(stop_atr),
                        "tp_atr": float(tp_atr),
                        "signal_count": int(count),
                        "expectancy_r": float(np.mean(r_values)) if r_values else None,
                        "median_r": float(np.median(r_values)) if r_values else None,
                        "stop_first_rate": float(exit_types.count("stop") / count) if count else None,
                        "tp_first_rate": float(exit_types.count("tp") / count) if count else None,
                        "horizon_exit_rate": float(exit_types.count("horizon") / count) if count else None,
                    }
                )

    grid_df = pd.DataFrame(rows)
    best_by_policy: dict[str, dict[str, Any]] = {}
    for policy_name in POLICY_NAMES:
        subset = grid_df[(grid_df["policy_name"] == policy_name) & grid_df["expectancy_r"].notna()].copy()
        if subset.empty:
            best_by_policy[policy_name] = {
                "best_expectancy_r": None,
                "best_stop_atr": None,
                "best_tp_atr": None,
                "stop_first_rate": None,
                "tp_first_rate": None,
                "horizon_exit_rate": None,
            }
            continue
        best = subset.sort_values(["expectancy_r", "signal_count"], ascending=[False, False]).iloc[0]
        best_by_policy[policy_name] = {
            "best_expectancy_r": finite_or_none(best.get("expectancy_r")),
            "best_stop_atr": finite_or_none(best.get("stop_atr")),
            "best_tp_atr": finite_or_none(best.get("tp_atr")),
            "stop_first_rate": finite_or_none(best.get("stop_first_rate")),
            "tp_first_rate": finite_or_none(best.get("tp_first_rate")),
            "horizon_exit_rate": finite_or_none(best.get("horizon_exit_rate")),
        }
    return grid_df, best_by_policy


def series_summary(series: pd.Series) -> dict[str, float | None]:
    """Return median, mean, and positive rate for a numeric series."""

    clean = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        return {"median": None, "mean": None, "positive_rate": None}
    return {
        "median": float(clean.median()),
        "mean": float(clean.mean()),
        "positive_rate": float((clean > 0).mean()),
    }


def summarize_signal_slice(df: pd.DataFrame, horizons: list[int]) -> dict[str, Any]:
    """Summarize one policy/group signal slice."""

    summary: dict[str, Any] = {
        "signal_count": int(len(df.index)),
        "long_count": int((df.get("direction") == "long").sum()) if "direction" in df.columns else 0,
        "short_count": int((df.get("direction") == "short").sum()) if "direction" in df.columns else 0,
        "overlap_count": int(pd.to_numeric(df.get("overlap_count", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
        if not df.empty
        else 0,
    }
    for horizon in horizons:
        future = series_summary(df.get(f"future_return_{horizon}m", pd.Series(dtype=float)))
        mfe = series_summary(df.get(f"mfe_{horizon}m", pd.Series(dtype=float)))
        mae = series_summary(df.get(f"mae_{horizon}m", pd.Series(dtype=float)))
        summary[f"median_future_return_{horizon}m"] = future["median"]
        summary[f"mean_future_return_{horizon}m"] = future["mean"]
        summary[f"positive_rate_{horizon}m"] = future["positive_rate"]
        summary[f"median_mfe_{horizon}m"] = mfe["median"]
        summary[f"median_mae_{horizon}m"] = mae["median"]
    return summary


def policy_notes(row: dict[str, Any], primary_horizon: int) -> str:
    """Build compact leaderboard notes."""

    notes: list[str] = []
    if int(row.get("signal_count") or 0) < 30:
        notes.append("low_sample")
    median_return = finite_or_none(row.get(f"median_future_return_{primary_horizon}m"))
    expectancy = finite_or_none(row.get("best_expectancy_r"))
    if median_return is not None and median_return > 0:
        notes.append(f"median_{primary_horizon}m_positive")
    if expectancy is not None and expectancy > 0:
        notes.append("positive_bracket_expectancy")
    if not notes:
        notes.append("no_positive_edge_flag")
    return "; ".join(notes)


def build_policy_leaderboard(
    signal_df: pd.DataFrame,
    horizons: list[int],
    best_brackets: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    """Build policy leaderboard with all requested policies."""

    primary_horizon = 120 if 120 in horizons else max(horizons)
    rows: list[dict[str, Any]] = []
    for policy_name in POLICY_NAMES:
        policy_df = signal_df[signal_df["policy_name"] == policy_name].copy() if not signal_df.empty else pd.DataFrame()
        row: dict[str, Any] = {
            "policy_name": policy_name,
            "policy_description": POLICY_DESCRIPTIONS.get(policy_name),
        }
        row.update(summarize_signal_slice(policy_df, horizons))
        row.update(best_brackets.get(policy_name, {}))
        row["notes"] = policy_notes(row, primary_horizon)
        rows.append(row)

    preferred_columns = [
        "policy_name",
        "signal_count",
        "long_count",
        "short_count",
        "median_future_return_60m",
        "median_future_return_120m",
        "median_future_return_240m",
        "median_future_return_480m",
        "positive_rate_120m",
        "median_mfe_120m",
        "median_mae_120m",
        "best_expectancy_r",
        "best_stop_atr",
        "best_tp_atr",
        "stop_first_rate",
        "tp_first_rate",
        "horizon_exit_rate",
        "notes",
        "policy_description",
    ]
    leaderboard = pd.DataFrame(rows)
    for column in preferred_columns:
        if column not in leaderboard.columns:
            leaderboard[column] = None
    extra_columns = [column for column in leaderboard.columns if column not in preferred_columns]
    return leaderboard[preferred_columns + extra_columns]


def build_group_summary(signal_df: pd.DataFrame, group_columns: list[str], horizons: list[int]) -> pd.DataFrame:
    """Build summaries by side/hour/weekday."""

    rows: list[dict[str, Any]] = []
    if signal_df.empty:
        return pd.DataFrame(columns=group_columns + ["signal_count"])
    for keys, group_df in signal_df.groupby(group_columns, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {column: key for column, key in zip(group_columns, keys)}
        row.update(summarize_signal_slice(group_df, horizons))
        rows.append(row)
    return pd.DataFrame(rows)


def iso_or_none(value: Any) -> str | None:
    """Return ISO timestamp text."""

    if value is None or pd.isna(value):
        return None
    return pd.Timestamp(value).isoformat()


def data_quality_for_frame(df: pd.DataFrame, timeframe_name: str, expected_minutes: int) -> dict[str, Any]:
    """Build a brief missing/gap summary for one timeframe."""

    if df.empty:
        return {
            "timeframe": timeframe_name,
            "bar_count": 0,
            "first_dt": None,
            "last_dt": None,
            "missing_count": None,
            "gap_count": None,
            "largest_gap_minutes": None,
        }
    times = pd.to_datetime(df["datetime"]).sort_values(kind="stable").drop_duplicates()
    diffs = times.diff().dropna()
    expected_delta = pd.Timedelta(minutes=expected_minutes)
    gap_diffs = diffs[diffs > expected_delta]
    missing_count = int(sum(max(0, int(diff / expected_delta) - 1) for diff in gap_diffs))
    largest_gap = float(gap_diffs.max() / pd.Timedelta(minutes=1)) if not gap_diffs.empty else None
    return {
        "timeframe": timeframe_name,
        "bar_count": int(len(times.index)),
        "first_dt": iso_or_none(times.iloc[0]),
        "last_dt": iso_or_none(times.iloc[-1]),
        "missing_count": missing_count,
        "gap_count": int(len(gap_diffs.index)),
        "largest_gap_minutes": largest_gap,
    }


def build_data_quality(timeframes: dict[str, pd.DataFrame], history_range: HistoryRange) -> dict[str, Any]:
    """Build data_quality.json payload."""

    frames = {
        name: filter_time_range(frame, history_range)
        for name, frame in timeframes.items()
    }
    return {
        "requested_start": history_range.start.isoformat(),
        "requested_end_exclusive": history_range.end_exclusive.isoformat(),
        "timeframes": {
            name: data_quality_for_frame(frames[name], name, minutes)
            for name, minutes in TIMEFRAME_MINUTES.items()
            if name in frames
        },
    }


def dataframe_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Convert a DataFrame to JSON-safe records."""

    if df.empty:
        return []
    safe_df = df.replace({np.nan: None})
    return json.loads(safe_df.to_json(orient="records", force_ascii=False))


def future_time_violation_count(df: pd.DataFrame, used_column: str, reference_column: str) -> int:
    """Count rows where a used timestamp is after the reference timestamp."""

    if df.empty or used_column not in df.columns or reference_column not in df.columns:
        return 0
    used = pd.to_datetime(df[used_column], utc=True, errors="coerce")
    reference = pd.to_datetime(df[reference_column], utc=True, errors="coerce")
    mask = used.notna() & reference.notna() & (used > reference)
    return int(mask.sum())


def build_signal_time_alignment_sample(signal_df: pd.DataFrame, limit: int = 20) -> list[dict[str, Any]]:
    """Build the audit sample of signal/context/entry timestamps."""

    if signal_df.empty:
        return []
    records: list[dict[str, Any]] = []
    for row in signal_df.head(limit).to_dict(orient="records"):
        records.append(
            {
                "signal_time": row.get("signal_time"),
                "used_1h_bar_time": row.get("used_1h_bar_time"),
                "used_15m_bar_time": row.get("used_15m_bar_time"),
                "used_5m_bar_time": row.get("used_5m_bar_time"),
                "entry_time": row.get("entry_time"),
                "entry_price": finite_or_none(row.get("entry_price")),
                "direction": row.get("direction"),
            }
        )
    return records


def build_research_audit(signal_df: pd.DataFrame) -> dict[str, Any]:
    """Build htf_research_audit.json with explicit correctness policies."""

    pullback_df = (
        signal_df[signal_df["policy_name"].isin(PULLBACK_RECLAIM_POLICY_NAMES)].copy()
        if not signal_df.empty and "policy_name" in signal_df.columns
        else pd.DataFrame()
    )
    pullback_entry_delay_values = (
        sorted(pd.to_numeric(pullback_df.get("entry_delay_bars_5m"), errors="coerce").dropna().unique().tolist())
        if not pullback_df.empty
        else []
    )
    return {
        "direction_return_formula": {
            "long": "future_return = future_close / entry_price - 1",
            "short": "future_return = entry_price / future_close - 1",
            "long_mfe": "max(high / entry_price - 1, 0)",
            "long_mae": "max(1 - low / entry_price, 0)",
            "short_mfe": "max(entry_price / low - 1, 0)",
            "short_mae": "max(1 - entry_price / high, 0)",
        },
        "resample_closed_bar_policy": (
            "1m bars are resampled into left-closed intervals and timestamped at the last included 1m close; "
            "a resampled bar is kept only when the actual last 1m timestamp equals that bar close; "
            "15m/1h context is joined to 5m signal bars with merge_asof(direction='backward'), so a signal can "
            "only use HTF bars whose close timestamp is <= the 5m signal close."
        ),
        "rolling_percentile_policy": (
            "atr_pct, recent_volatility_30m, directional_recent_return_30m, volume_zscore_30m, and body_ratio "
            "percentiles compare the current completed 15m value against a rolling window of prior values only; "
            "the current value and all future values are excluded from the percentile distribution."
        ),
        "no_lookahead_checks": {
            "used_15m_after_signal_count": future_time_violation_count(signal_df, "used_15m_bar_time", "signal_time"),
            "used_1h_after_signal_count": future_time_violation_count(signal_df, "used_1h_bar_time", "signal_time"),
            "used_5m_after_signal_count": future_time_violation_count(signal_df, "used_5m_bar_time", "signal_time"),
            "entry_before_signal_count": future_time_violation_count(signal_df, "signal_time", "entry_time"),
            "pullback_reclaim_entry_delay_bars_5m_values": pullback_entry_delay_values,
            "pullback_reclaim_uses_next_5m_close_entry": bool(
                pullback_df.empty or set(pullback_entry_delay_values) == {1}
            ),
            "outcome_and_bracket_windows_start_after_entry_time": True,
        },
        "bracket_policy": {
            "long": "stop = entry - stop_atr * atr; tp = entry + tp_atr * atr",
            "short": "stop = entry + stop_atr * atr; tp = entry - tp_atr * atr",
            "same_bar_stop_tp": "stop first",
            "horizon_exit": "exit at horizon close when neither stop nor tp is hit",
            "r_multiple": "tp uses tp_atr / stop_atr; stop is -1R; horizon R is direction-adjusted close move / stop distance",
        },
        "signal_time_alignment_sample": build_signal_time_alignment_sample(signal_df),
    }


def metric_value(leaderboard: pd.DataFrame, policy_name: str, column: str) -> float | None:
    """Return one leaderboard numeric value."""

    if leaderboard.empty or column not in leaderboard.columns:
        return None
    row = leaderboard[leaderboard["policy_name"] == policy_name]
    if row.empty:
        return None
    return finite_or_none(row.iloc[0].get(column))


def compare_metric_answer(
    leaderboard: pd.DataFrame,
    candidate: str,
    baseline: str,
    metric: str,
    question: str,
) -> dict[str, Any]:
    """Answer an improvement question with one metric."""

    candidate_value = metric_value(leaderboard, candidate, metric)
    baseline_value = metric_value(leaderboard, baseline, metric)
    improved = bool(candidate_value is not None and baseline_value is not None and candidate_value > baseline_value)
    if candidate_value is None or baseline_value is None:
        answer = f"{question}: 样本不足，无法判断"
    elif improved:
        answer = f"{question}: 是，{candidate} 的 {metric} 高于 {baseline}"
    else:
        answer = f"{question}: 否，{candidate} 的 {metric} 未高于 {baseline}"
    return {
        "question": question,
        "candidate_policy": candidate,
        "baseline_policy": baseline,
        "metric": metric,
        "candidate_value": candidate_value,
        "baseline_value": baseline_value,
        "improved": improved,
        "answer": answer,
    }


def cross_split_snapshot(output_dir: Path) -> dict[str, Any]:
    """Try to infer cross-split stability when sibling split outputs already exist."""

    parent = output_dir.parent
    paths = {split: parent / split / "htf_policy_leaderboard.csv" for split in ["train", "validation", "oos"]}
    if not all(path.exists() for path in paths.values()):
        return {
            "available": False,
            "stable_positive_policies": [],
            "single_split_only_policies": [],
            "overfit_risk": None,
            "answer": "当前只看到单个 split，请运行 make compare-htf 判断 train/validation/oos 稳定性",
        }

    frames = {}
    for split, path in paths.items():
        frame = pd.read_csv(path)
        frame["split"] = split
        frames[split] = frame

    stable: list[str] = []
    single_only: list[str] = []
    for policy_name in POLICY_NAMES:
        returns: list[float] = []
        expectancies: list[float] = []
        effective_splits: list[str] = []
        for split, frame in frames.items():
            row = frame[frame["policy_name"] == policy_name]
            median_return = finite_or_none(row.iloc[0].get("median_future_return_120m")) if not row.empty else None
            expectancy = finite_or_none(row.iloc[0].get("best_expectancy_r")) if not row.empty else None
            if median_return is not None:
                returns.append(median_return)
            if expectancy is not None:
                expectancies.append(expectancy)
            if median_return is not None and median_return > 0 and expectancy is not None and expectancy > 0:
                effective_splits.append(split)
        if len(returns) == 3 and len(expectancies) == 3 and all(value > 0 for value in returns) and all(value > 0 for value in expectancies):
            stable.append(policy_name)
        elif len(effective_splits) == 1:
            single_only.append(policy_name)

    return {
        "available": True,
        "stable_positive_policies": stable,
        "single_split_only_policies": single_only,
        "overfit_risk": bool(single_only),
        "answer": "存在跨三段稳定为正 policy" if stable else "没有发现 train/validation/oos 都稳定为正的 policy",
    }


def build_diagnostic_answers(leaderboard: pd.DataFrame, output_dir: Path) -> dict[str, Any]:
    """Build report-required diagnostic answers."""

    primary_metric = "median_future_return_120m"
    baseline = "htf_1h_ema_regime_only"
    ema_policy = "htf_1h_ema_15m_ema_structure"
    cross_split = cross_split_snapshot(output_dir)
    viable = leaderboard[
        (pd.to_numeric(leaderboard.get(primary_metric), errors="coerce") > 0)
        & (pd.to_numeric(leaderboard.get("best_expectancy_r"), errors="coerce") > 0)
    ].copy()
    failed = bool(viable.empty)
    if cross_split.get("available"):
        failed = bool(not cross_split.get("stable_positive_policies"))

    return {
        "one_hour_regime_only_vs_original_1m_breakout": {
            "answer": (
                "本脚本不读取原 1m breakout 基准报告，不能单独证明比原策略更好；"
                f"当前 1h regime only 的 {primary_metric}="
                f"{metric_value(leaderboard, baseline, primary_metric)}，best_expectancy_r="
                f"{metric_value(leaderboard, baseline, 'best_expectancy_r')}。"
            ),
            "requires_external_baseline": True,
        },
        "ema_structure_improves": compare_metric_answer(
            leaderboard,
            ema_policy,
            baseline,
            primary_metric,
            "15m EMA structure 是否改善",
        ),
        "vwap_structure_improves": compare_metric_answer(
            leaderboard,
            "htf_1h_ema_15m_vwap_structure",
            baseline,
            primary_metric,
            "15m VWAP structure 是否改善",
        ),
        "donchian_structure_improves": compare_metric_answer(
            leaderboard,
            "htf_1h_ema_15m_donchian_structure",
            baseline,
            primary_metric,
            "15m Donchian structure 是否改善",
        ),
        "vol_cap_improves": compare_metric_answer(
            leaderboard,
            "htf_1h_15m_structure_with_vol_cap",
            ema_policy,
            primary_metric,
            "vol cap 是否改善",
        ),
        "strict_vol_cap_improves": compare_metric_answer(
            leaderboard,
            "htf_1h_15m_structure_strict_vol_cap",
            ema_policy,
            primary_metric,
            "strict vol cap 是否改善",
        ),
        "no_overextension_improves": compare_metric_answer(
            leaderboard,
            "htf_1h_15m_structure_no_overextension",
            ema_policy,
            primary_metric,
            "no overextension 是否改善",
        ),
        "pullback_reclaim_improves": compare_metric_answer(
            leaderboard,
            "htf_1h_15m_structure_5m_pullback_reclaim",
            ema_policy,
            primary_metric,
            "5m pullback reclaim 是否改善",
        ),
        "cross_split_stability": cross_split,
        "htf_signal_hypothesis_failed": failed,
        "overfit_risk": cross_split.get("overfit_risk"),
    }


def build_summary(
    vt_symbol: str,
    split: str,
    history_range: HistoryRange,
    output_dir: Path,
    horizons: list[int],
    signal_df: pd.DataFrame,
    leaderboard_df: pd.DataFrame,
    bracket_grid_df: pd.DataFrame,
    data_quality: dict[str, Any],
    warnings: list[str],
) -> dict[str, Any]:
    """Build htf_policy_summary.json payload."""

    answers = build_diagnostic_answers(leaderboard_df, output_dir)
    return {
        "vt_symbol": vt_symbol,
        "split": split,
        "start": history_range.start.isoformat(),
        "end_exclusive": history_range.end_exclusive.isoformat(),
        "timezone": history_range.timezone_name,
        "output_dir": str(output_dir),
        "horizons": horizons,
        "policy_count": len(POLICY_NAMES),
        "signal_count": int(len(signal_df.index)),
        "bracket_grid_rows": int(len(bracket_grid_df.index)),
        "warnings": warnings,
        "data_quality": data_quality,
        "leaderboard": dataframe_records(leaderboard_df),
        "diagnostic_answers": answers,
    }


def format_number(value: Any, digits: int = 6) -> str:
    """Format a number for Markdown."""

    number = finite_or_none(value)
    if number is None:
        return "N/A"
    return f"{number:.{digits}f}"


def format_leaderboard_lines(leaderboard_df: pd.DataFrame, limit: int = 10) -> str:
    """Format leaderboard rows for the report."""

    if leaderboard_df.empty:
        return "- 无"
    working = leaderboard_df.copy()
    working["_sort_expectancy"] = pd.to_numeric(working.get("best_expectancy_r"), errors="coerce")
    working["_sort_median"] = pd.to_numeric(working.get("median_future_return_120m"), errors="coerce")
    working = working.sort_values(["_sort_expectancy", "_sort_median", "signal_count"], ascending=[False, False, False])
    lines = []
    for row in working.head(limit).to_dict(orient="records"):
        lines.append(
            f"- {row.get('policy_name')}: signals={row.get('signal_count')}, "
            f"median_120m={format_number(row.get('median_future_return_120m'))}, "
            f"best_expectancy_r={format_number(row.get('best_expectancy_r'))}, notes={row.get('notes')}"
        )
    return "\n".join(lines)


def render_markdown(summary: dict[str, Any], leaderboard_df: pd.DataFrame) -> str:
    """Render htf_research_report.md."""

    answers = summary.get("diagnostic_answers") or {}
    cross_split = answers.get("cross_split_stability") or {}
    warnings = summary.get("warnings") or []
    warning_lines = "\n".join(f"- {item}" for item in warnings) if warnings else "- 无"
    return (
        "# HTF Signal Research 报告\n\n"
        "## 核心结论\n"
        f"- split={summary.get('split')}, signal_count={summary.get('signal_count')}\n"
        f"- htf_signal_hypothesis_failed={str(bool(answers.get('htf_signal_hypothesis_failed'))).lower()}\n"
        f"- overfit_risk={str(answers.get('overfit_risk')).lower()}\n\n"
        "## Policy Leaderboard\n"
        f"{format_leaderboard_lines(leaderboard_df)}\n\n"
        "## 必答问题\n"
        f"1. 1h regime only 是否比原 1m breakout 更好？{(answers.get('one_hour_regime_only_vs_original_1m_breakout') or {}).get('answer')}\n"
        f"2. 15m EMA structure 是否改善？{(answers.get('ema_structure_improves') or {}).get('answer')}\n"
        f"3. 15m VWAP structure 是否改善？{(answers.get('vwap_structure_improves') or {}).get('answer')}\n"
        f"4. 15m Donchian structure 是否改善？{(answers.get('donchian_structure_improves') or {}).get('answer')}\n"
        f"5. vol cap 是否改善？{(answers.get('vol_cap_improves') or {}).get('answer')}；"
        f"{(answers.get('strict_vol_cap_improves') or {}).get('answer')}\n"
        f"6. no overextension 是否改善？{(answers.get('no_overextension_improves') or {}).get('answer')}\n"
        f"7. 5m pullback reclaim 是否改善？{(answers.get('pullback_reclaim_improves') or {}).get('answer')}\n"
        f"8. 是否存在 train / validation / oos 都稳定为正的 policy？{cross_split.get('answer')}\n"
        f"9. htf_signal_hypothesis_failed={str(bool(answers.get('htf_signal_hypothesis_failed'))).lower()}\n"
        f"10. overfit_risk={str(answers.get('overfit_risk')).lower()}；"
        f"single_split_only={cross_split.get('single_split_only_policies')}\n\n"
        "## 输出文件\n"
        "- htf_signal_dataset.csv\n"
        "- htf_policy_summary.json\n"
        "- htf_research_audit.json\n"
        "- htf_policy_leaderboard.csv\n"
        "- htf_bracket_grid.csv\n"
        "- htf_policy_by_side.csv\n"
        "- htf_policy_by_hour.csv\n"
        "- htf_policy_by_weekday.csv\n"
        "- htf_research_report.md\n"
        "- data_quality.json\n\n"
        "## Warning\n"
        f"{warning_lines}\n"
    )


def write_json(path: Path, payload: Any) -> None:
    """Write JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(payload), ensure_ascii=False, indent=2), encoding="utf-8")


def write_dataframe(path: Path, df: pd.DataFrame) -> None:
    """Write CSV."""

    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")


def write_outputs(
    output_dir: Path,
    dataset_df: pd.DataFrame,
    summary: dict[str, Any],
    leaderboard_df: pd.DataFrame,
    bracket_grid_df: pd.DataFrame,
    by_side_df: pd.DataFrame,
    by_hour_df: pd.DataFrame,
    by_weekday_df: pd.DataFrame,
    markdown: str,
    data_quality: dict[str, Any],
    audit: dict[str, Any],
) -> None:
    """Write all HTF research artifacts."""

    output_dir.mkdir(parents=True, exist_ok=True)
    output_dataset = dataset_df.drop(columns=["_signal_dt", "_entry_dt"], errors="ignore")
    write_dataframe(output_dir / "htf_signal_dataset.csv", output_dataset)
    write_json(output_dir / "htf_policy_summary.json", summary)
    write_json(output_dir / "htf_research_audit.json", audit)
    write_dataframe(output_dir / "htf_policy_leaderboard.csv", leaderboard_df)
    write_dataframe(output_dir / "htf_bracket_grid.csv", bracket_grid_df)
    write_dataframe(output_dir / "htf_policy_by_side.csv", by_side_df)
    write_dataframe(output_dir / "htf_policy_by_hour.csv", by_hour_df)
    write_dataframe(output_dir / "htf_policy_by_weekday.csv", by_weekday_df)
    (output_dir / "htf_research_report.md").write_text(markdown, encoding="utf-8")
    write_json(output_dir / "data_quality.json", data_quality)


def run_research(
    vt_symbol: str,
    split: str,
    history_range: HistoryRange,
    output_dir: Path,
    horizons: list[int],
    stop_grid: list[float],
    tp_grid: list[float],
    timezone_name: str,
    cooldown_bars_5m: int,
    data_check_strict: bool,
    logger: logging.Logger,
    max_signals: int | None = None,
    bars_from_db: bool = True,
    bars_df: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Run the complete HTF signal research workflow."""

    ZoneInfo(timezone_name)
    warnings: list[str] = []
    if bars_df is None:
        if not bars_from_db:
            raise HtfSignalResearchError("--no-bars-from-db 已设置，但未提供 bars_df")
        bars_1m = load_bars_from_db(vt_symbol, history_range, max(horizons), timezone_name, logger)
    else:
        bars_1m = normalize_1m_bars(bars_df, timezone_name)

    if bars_1m.empty:
        warnings.append("没有可用 1m bars")

    timeframes = build_timeframes(bars_1m)
    data_quality = build_data_quality(timeframes, history_range)
    one_minute_quality = (data_quality.get("timeframes") or {}).get("1m") or {}
    if data_check_strict:
        if int(one_minute_quality.get("bar_count") or 0) <= 0:
            raise HtfSignalResearchError("--data-check-strict: requested 1m bars 为空")
        if int(one_minute_quality.get("missing_count") or 0) > 0:
            raise HtfSignalResearchError("--data-check-strict: requested 1m bars 存在缺口")

    indicators = build_indicator_frames(timeframes)
    aligned_df = align_indicator_frames(indicators, history_range)
    signal_df = apply_cooldown_and_build_signals(aligned_df, vt_symbol, cooldown_bars_5m, max_signals)
    signal_df = add_overlap_counts(signal_df, max(horizons))
    signal_df = compute_signal_outcomes(signal_df, bars_1m, horizons, warnings)

    bracket_horizon = 120 if 120 in horizons else max(horizons)
    bracket_grid_df, best_brackets = compute_bracket_grid(
        signal_df,
        timeframes["5m"],
        stop_grid,
        tp_grid,
        bracket_horizon,
    )
    leaderboard_df = build_policy_leaderboard(signal_df, horizons, best_brackets)
    by_side_df = build_group_summary(signal_df, ["policy_name", "direction"], horizons)
    by_hour_df = build_group_summary(signal_df, ["policy_name", "hour"], horizons)
    by_weekday_df = build_group_summary(signal_df, ["policy_name", "weekday"], horizons)
    audit = build_research_audit(signal_df)
    summary = build_summary(
        vt_symbol=vt_symbol,
        split=split,
        history_range=history_range,
        output_dir=output_dir,
        horizons=horizons,
        signal_df=signal_df,
        leaderboard_df=leaderboard_df,
        bracket_grid_df=bracket_grid_df,
        data_quality=data_quality,
        warnings=warnings,
    )
    markdown = render_markdown(summary, leaderboard_df)
    write_outputs(
        output_dir=output_dir,
        dataset_df=signal_df,
        summary=summary,
        leaderboard_df=leaderboard_df,
        bracket_grid_df=bracket_grid_df,
        by_side_df=by_side_df,
        by_hour_df=by_hour_df,
        by_weekday_df=by_weekday_df,
        markdown=markdown,
        data_quality=data_quality,
        audit=audit,
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""

    args = parse_args(argv)
    ensure_headless_runtime()
    logger = setup_logging("research_htf_signals", verbose=args.verbose)

    try:
        history_range = resolve_split_range(args.split, args.start, args.end, args.timezone)
        output_dir = resolve_path(args.output_dir, PROJECT_ROOT / "reports" / "research" / "htf_signals" / args.split)
        horizons = parse_horizons(args.horizons)
        stop_grid = [float(value) for value in parse_number_list(args.stop_atr_grid, "--stop-atr-grid")]
        tp_grid = [float(value) for value in parse_number_list(args.tp_atr_grid, "--tp-atr-grid")]
        summary = run_research(
            vt_symbol=args.vt_symbol,
            split=args.split,
            history_range=history_range,
            output_dir=output_dir,
            horizons=horizons,
            stop_grid=stop_grid,
            tp_grid=tp_grid,
            timezone_name=args.timezone,
            cooldown_bars_5m=int(args.cooldown_bars_5m),
            data_check_strict=bool(args.data_check_strict),
            logger=logger,
            max_signals=args.max_signals,
            bars_from_db=bool(args.bars_from_db),
        )
        answers = summary.get("diagnostic_answers") or {}
        print_json_block(
            "HTF signal research summary:",
            {
                "output_dir": output_dir,
                "split": args.split,
                "signal_count": summary.get("signal_count"),
                "htf_signal_hypothesis_failed": answers.get("htf_signal_hypothesis_failed"),
                "overfit_risk": answers.get("overfit_risk"),
                "warnings": summary.get("warnings"),
            },
        )
        if args.json:
            print(json.dumps(to_jsonable(summary), ensure_ascii=False, indent=2))
        return 0
    except HtfSignalResearchError as exc:
        log_event(logger, logging.ERROR, "htf_signal_research.error", str(exc))
        return 2
    except Exception:
        logger.exception(
            "Unexpected error during HTF signal research",
            extra={"event": "htf_signal_research.unexpected_error"},
        )
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
