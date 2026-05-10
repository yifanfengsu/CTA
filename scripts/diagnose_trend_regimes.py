#!/usr/bin/env python3
"""Diagnose multi-symbol trend regimes for Trend Following V3 postmortem work."""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from analyze_signal_outcomes import configure_sqlite_settings, resolve_exchange, split_vt_symbol
from common_runtime import PROJECT_ROOT, ensure_headless_runtime, log_event, print_json_block, setup_logging, to_jsonable
from history_time_utils import DEFAULT_TIMEZONE, HistoryRange, expected_bar_count, parse_history_range
from history_utils import get_database_timezone
from research_trend_following_v2 import (
    dataframe_records,
    filter_time_range,
    finite_or_none,
    normalize_1m_bars,
    resample_ohlcv,
    rolling_percentile,
    safe_mean,
    safe_sum,
    true_range,
)


DEFAULT_SYMBOLS = [
    "BTCUSDT_SWAP_OKX.GLOBAL",
    "ETHUSDT_SWAP_OKX.GLOBAL",
    "SOLUSDT_SWAP_OKX.GLOBAL",
    "LINKUSDT_SWAP_OKX.GLOBAL",
    "DOGEUSDT_SWAP_OKX.GLOBAL",
]
DEFAULT_TIMEFRAMES = ["4h", "1d"]
DEFAULT_TREND_WINDOWS = [20, 55, 100]
DEFAULT_START = "2023-01-01"
DEFAULT_END = "2026-03-31"
DEFAULT_INTERVAL = "1m"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "reports" / "research" / "trend_regime_diagnostics"
DEFAULT_V3_TRADE_FILES = {
    "train_ext": PROJECT_ROOT / "reports" / "research" / "trend_following_v3_extended" / "train_ext" / "trend_v3_trades.csv",
    "validation_ext": PROJECT_ROOT
    / "reports"
    / "research"
    / "trend_following_v3_extended"
    / "validation_ext"
    / "trend_v3_trades.csv",
    "oos_ext": PROJECT_ROOT / "reports" / "research" / "trend_following_v3_extended" / "oos_ext" / "trend_v3_trades.csv",
}
DEFAULT_COMPARE_SUMMARY = (
    PROJECT_ROOT
    / "reports"
    / "research"
    / "trend_following_v3_extended_compare"
    / "trend_v3_extended_compare_summary.json"
)
DEFAULT_FUNDING_STRESS = (
    PROJECT_ROOT
    / "reports"
    / "research"
    / "trend_following_v3_extended_compare"
    / "trend_v3_extended_compare_funding_stress.csv"
)

TIMEFRAME_MINUTES = {"4h": 240, "1d": 1440}
REGIME_LABELS = [
    "strong_uptrend",
    "weak_uptrend",
    "strong_downtrend",
    "weak_downtrend",
    "choppy",
    "high_vol_choppy",
    "compression",
]
STRONG_REGIMES = {"strong_uptrend", "strong_downtrend"}
TREND_REGIMES = {"strong_uptrend", "weak_uptrend", "strong_downtrend", "weak_downtrend"}
CHOPPY_REGIMES = {"choppy", "high_vol_choppy"}
UPTREND_REGIMES = {"strong_uptrend", "weak_uptrend"}
DOWNTREND_REGIMES = {"strong_downtrend", "weak_downtrend"}
ROLLING_PERCENTILE_WINDOW = 240
REALIZED_VOL_WINDOW = 20
PERSISTENCE_WINDOW = 20
ADX_PERIOD = 14
REGIME_RULES = {
    "4h": {
        "weak_ema_spread_pct": 0.004,
        "strong_ema_spread_pct": 0.012,
        "trend_score_spread_cap": 0.025,
    },
    "1d": {
        "weak_ema_spread_pct": 0.010,
        "strong_ema_spread_pct": 0.030,
        "trend_score_spread_cap": 0.060,
    },
}
REQUIRED_OUTPUT_FILES = [
    "trend_regime_summary.json",
    "trend_regime_report.md",
    "data_quality.json",
    "regime_dataset.csv",
    "regime_by_symbol.csv",
    "regime_by_month.csv",
    "regime_by_quarter.csv",
    "regime_by_timeframe.csv",
    "trend_score_by_symbol.csv",
    "trend_score_by_month.csv",
    "trade_regime_attribution.csv",
    "policy_regime_performance.csv",
    "v3_1_regime_recommendations.json",
]


class TrendRegimeDiagnosticsError(Exception):
    """Raised when trend regime diagnostics cannot complete."""


@dataclass(frozen=True, slots=True)
class DiagnosticOutputs:
    """Generated diagnostics and output paths."""

    output_dir: Path
    summary: dict[str, Any]
    data_quality: dict[str, Any]
    regime_dataset: pd.DataFrame
    trade_attribution: pd.DataFrame
    policy_regime_performance: pd.DataFrame
    recommendations: dict[str, Any]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description="Diagnose trend regimes for Trend Following V3 research.")
    parser.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS))
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--end", default=DEFAULT_END)
    parser.add_argument("--timezone", default=DEFAULT_TIMEZONE)
    parser.add_argument("--interval", default=DEFAULT_INTERVAL, choices=("1m",))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--timeframes", default=",".join(DEFAULT_TIMEFRAMES))
    parser.add_argument("--trend-score-window-bars", default=",".join(str(value) for value in DEFAULT_TREND_WINDOWS))
    parser.add_argument("--data-check-strict", action="store_true")
    parser.add_argument("--json", action="store_true", help="Print trend_regime_summary.json after writing outputs.")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def resolve_path(path_arg: str | Path) -> Path:
    """Resolve paths relative to the project root."""

    path = Path(path_arg)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def parse_symbol_list(raw_value: str) -> list[str]:
    """Parse comma or whitespace separated vt_symbols."""

    symbols: list[str] = []
    seen: set[str] = set()
    for token in re.split(r"[\s,]+", str(raw_value or "")):
        symbol = token.strip()
        if not symbol or symbol in seen:
            continue
        symbols.append(symbol)
        seen.add(symbol)
    if not symbols:
        raise TrendRegimeDiagnosticsError("--symbols must contain at least one vt_symbol")
    return symbols


def parse_timeframes(raw_value: str) -> list[str]:
    """Parse requested diagnostic timeframes."""

    timeframes: list[str] = []
    seen: set[str] = set()
    for token in re.split(r"[\s,]+", str(raw_value or "")):
        timeframe = token.strip()
        if not timeframe or timeframe in seen:
            continue
        if timeframe not in TIMEFRAME_MINUTES:
            raise TrendRegimeDiagnosticsError(f"Unsupported timeframe: {timeframe}")
        timeframes.append(timeframe)
        seen.add(timeframe)
    if not timeframes:
        raise TrendRegimeDiagnosticsError("--timeframes must contain at least one supported timeframe")
    return timeframes


def parse_positive_int_list(raw_value: str, option_name: str) -> list[int]:
    """Parse comma or whitespace separated positive integers."""

    values: list[int] = []
    seen: set[int] = set()
    for token in re.split(r"[\s,]+", str(raw_value or "")):
        text = token.strip()
        if not text:
            continue
        try:
            value = int(text)
        except ValueError as exc:
            raise TrendRegimeDiagnosticsError(f"{option_name} contains invalid integer: {text!r}") from exc
        if value <= 0:
            raise TrendRegimeDiagnosticsError(f"{option_name} values must be positive: {value}")
        if value not in seen:
            values.append(value)
            seen.add(value)
    if not values:
        raise TrendRegimeDiagnosticsError(f"{option_name} must not be empty")
    return sorted(values)


def resolve_history_range(start: str, end: str, timezone_name: str) -> HistoryRange:
    """Resolve the requested 1m history range."""

    try:
        return parse_history_range(start, end, pd.Timedelta(minutes=1).to_pytimedelta(), timezone_name)
    except ValueError as exc:
        raise TrendRegimeDiagnosticsError(str(exc)) from exc


def bar_to_record(bar: Any) -> dict[str, Any]:
    """Convert one vn.py BarData-like object to a plain OHLCV record."""

    return {
        "datetime": getattr(bar, "datetime", None),
        "open": getattr(bar, "open_price", None),
        "high": getattr(bar, "high_price", None),
        "low": getattr(bar, "low_price", None),
        "close": getattr(bar, "close_price", None),
        "volume": getattr(bar, "volume", None),
    }


def load_bars_from_db(vt_symbol: str, history_range: HistoryRange, timezone_name: str, logger: logging.Logger) -> pd.DataFrame:
    """Load exact 1m bars from vn.py sqlite for one vt_symbol."""

    symbol, exchange_value = split_vt_symbol(vt_symbol)
    exchange = resolve_exchange(exchange_value)

    from vnpy.trader.constant import Interval
    from vnpy.trader.database import get_database

    configure_sqlite_settings(logger)
    db_tz = get_database_timezone()
    query_start = history_range.start.astimezone(db_tz).replace(tzinfo=None)
    query_end = history_range.end_exclusive.astimezone(db_tz).replace(tzinfo=None)
    bars = get_database().load_bar_data(symbol, exchange, Interval.MINUTE, query_start, query_end)
    records = [bar_to_record(bar) for bar in bars]
    if not records:
        return pd.DataFrame(columns=["datetime", "open", "high", "low", "close", "volume"])
    return normalize_1m_bars(pd.DataFrame(records), timezone_name)


def normalize_input_bars(vt_symbol: str, bars_df: pd.DataFrame, timezone_name: str) -> pd.DataFrame:
    """Filter by vt_symbol when available and normalize a provided 1m OHLCV DataFrame."""

    working = bars_df.copy()
    if "vt_symbol" in working.columns:
        working = working[working["vt_symbol"] == vt_symbol].copy()
    return normalize_1m_bars(working, timezone_name)


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
                ranges.append(
                    {
                        "start": cursor.isoformat(),
                        "end": missing_end.isoformat(),
                        "missing_count": count,
                    }
                )
        cursor = current + interval
    if cursor < end_exclusive:
        count = int((end_exclusive - cursor) / interval)
        gap_count += 1
        if len(ranges) < limit:
            ranges.append(
                {
                    "start": cursor.isoformat(),
                    "end": (end_exclusive - interval).isoformat(),
                    "missing_count": count,
                    }
            )
    return ranges, gap_count


def build_1m_coverage_summary(bars_1m: pd.DataFrame, history_range: HistoryRange) -> dict[str, Any]:
    """Build strict 1m coverage summary for one symbol."""

    target = filter_time_range(bars_1m, history_range)
    if target.empty:
        actual_times: list[pd.Timestamp] = []
    else:
        actual_times = [
            pd.Timestamp(value)
            for value in pd.to_datetime(target["datetime"]).dropna().drop_duplicates().sort_values(kind="stable")
        ]
    expected = int(expected_bar_count(history_range))
    total = int(len(actual_times))
    missing = max(0, expected - total)
    missing_ranges, gap_count = coverage_missing_ranges_from_actual(actual_times, history_range)
    if missing and not missing_ranges:
        missing_ranges = [{"start": history_range.start.isoformat(), "end": history_range.end_display.isoformat(), "missing_count": missing}]
        gap_count = 1
    return {
        "expected_count": expected,
        "total_count": total,
        "missing_count": int(missing),
        "gap_count": int(gap_count),
        "first_dt": actual_times[0].isoformat() if actual_times else None,
        "last_dt": actual_times[-1].isoformat() if actual_times else None,
        "missing_ranges": missing_ranges,
        "required_coverage_ready": bool(expected > 0 and total == expected and missing == 0),
    }


def data_quality_for_timeframe(frame: pd.DataFrame, timeframe: str, history_range: HistoryRange) -> dict[str, Any]:
    """Build quality metadata for a resampled timeframe."""

    filtered = filter_time_range(frame, history_range)
    if filtered.empty:
        return {
            "timeframe": timeframe,
            "minutes": TIMEFRAME_MINUTES[timeframe],
            "bar_count": 0,
            "first_dt": None,
            "last_dt": None,
        }
    dt = pd.to_datetime(filtered["datetime"]).dropna().sort_values(kind="stable")
    return {
        "timeframe": timeframe,
        "minutes": TIMEFRAME_MINUTES[timeframe],
        "bar_count": int(len(dt.index)),
        "first_dt": dt.iloc[0].isoformat() if len(dt.index) else None,
        "last_dt": dt.iloc[-1].isoformat() if len(dt.index) else None,
    }


def finite_float(value: Any) -> float | None:
    """Return a finite float or None."""

    return finite_or_none(value)


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Divide while treating zero denominators as missing."""

    return numerator / denominator.replace(0, np.nan)


def compute_trend_efficiency(close: pd.Series, window: int) -> pd.Series:
    """Compute abs(net movement) divided by path length over a rolling window."""

    numeric_close = pd.to_numeric(close, errors="coerce")
    net_move = (numeric_close - numeric_close.shift(window)).abs()
    path = numeric_close.diff().abs().rolling(window, min_periods=window).sum()
    return net_move / path.replace(0, np.nan)


def signed_run_lengths(values: pd.Series) -> pd.Series:
    """Return signed consecutive positive/negative bar run lengths."""

    signs = np.sign(pd.to_numeric(values, errors="coerce").fillna(0.0).to_numpy(dtype=float))
    runs = np.zeros(signs.size, dtype=float)
    previous = 0.0
    count = 0
    for index, sign in enumerate(signs):
        if sign == 0:
            previous = 0.0
            count = 0
            runs[index] = 0.0
            continue
        if sign == previous:
            count += 1
        else:
            previous = sign
            count = 1
        runs[index] = sign * count
    return pd.Series(runs, index=values.index, dtype=float)


def compute_adx(df: pd.DataFrame, period: int = ADX_PERIOD) -> pd.DataFrame:
    """Compute a simple ADX/+DI/-DI implementation with Wilder-style smoothing."""

    high = pd.to_numeric(df["high"], errors="coerce")
    low = pd.to_numeric(df["low"], errors="coerce")
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = pd.Series(
        np.where((up_move > down_move) & (up_move > 0), up_move, 0.0),
        index=df.index,
        dtype=float,
    )
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0), down_move, 0.0),
        index=df.index,
        dtype=float,
    )
    atr = true_range(df).ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    plus_di = 100.0 * plus_dm.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean() / atr.replace(0, np.nan)
    minus_di = 100.0 * minus_dm.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean() / atr.replace(0, np.nan)
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    return pd.DataFrame({"adx14": adx, "plus_di": plus_di, "minus_di": minus_di}, index=df.index)


def primary_window(windows: list[int]) -> int:
    """Return the primary trend scoring window."""

    if 55 in windows:
        return 55
    return sorted(windows)[len(windows) // 2]


def max_available_efficiency(row: pd.Series, windows: list[int]) -> float | None:
    """Return the maximum available trend efficiency across configured windows."""

    values = [finite_float(row.get(f"trend_efficiency_{window}")) for window in windows]
    clean = [value for value in values if value is not None]
    if not clean:
        return None
    return float(max(clean))


def compute_regime_indicators(frame: pd.DataFrame, timeframe: str, windows: list[int]) -> pd.DataFrame:
    """Compute trend, Donchian, volatility, persistence, and ADX diagnostics."""

    df = frame.copy().sort_values("datetime", kind="stable").reset_index(drop=True)
    if df.empty:
        return df

    for column in ["open", "high", "low", "close", "volume"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    close = df["close"]
    high = df["high"]
    low = df["low"]
    returns = close.pct_change()

    df["ema50"] = close.ewm(span=50, adjust=False, min_periods=1).mean()
    df["ema200"] = close.ewm(span=200, adjust=False, min_periods=1).mean()
    df["ema_spread_pct"] = (df["ema50"] - df["ema200"]) / close.replace(0, np.nan)
    df["ema50_slope"] = df["ema50"].diff()
    df["ema200_slope"] = df["ema200"].diff()
    df["ema50_slope_pct"] = df["ema50_slope"] / close.replace(0, np.nan)
    df["ema200_slope_pct"] = df["ema200_slope"] / close.replace(0, np.nan)
    df["close_distance_to_ema50"] = close / df["ema50"].replace(0, np.nan) - 1.0
    df["close_distance_to_ema200"] = close / df["ema200"].replace(0, np.nan) - 1.0

    primary = primary_window(windows)
    for window in windows:
        channel_high = high.rolling(window, min_periods=1).max()
        channel_low = low.rolling(window, min_periods=1).min()
        channel_width = channel_high - channel_low
        previous_high = high.rolling(window, min_periods=window).max().shift(1)
        previous_low = low.rolling(window, min_periods=window).min().shift(1)
        breakout_direction = np.select([close > previous_high, close < previous_low], [1, -1], default=0)
        close_position = (close - channel_low) / channel_width.replace(0, np.nan)

        df[f"donchian_high_{window}"] = channel_high
        df[f"donchian_low_{window}"] = channel_low
        df[f"donchian_mid_{window}"] = (channel_high + channel_low) / 2.0
        df[f"donchian_width_{window}"] = channel_width
        df[f"donchian_width_pct_{window}"] = channel_width / close.replace(0, np.nan)
        df[f"donchian_breakout_direction_{window}"] = breakout_direction
        df[f"donchian_breakout_count_{window}"] = (
            pd.Series(np.abs(breakout_direction), index=df.index, dtype=float).rolling(window, min_periods=1).sum()
        )
        df[f"donchian_channel_direction_{window}"] = np.sign(df[f"donchian_mid_{window}"].diff()).astype(float)
        df[f"close_position_in_donchian_{window}"] = close_position
        df[f"trend_efficiency_{window}"] = compute_trend_efficiency(close, window)

    df["donchian_width_pct"] = df[f"donchian_width_pct_{primary}"]
    df["donchian_width_percentile"] = rolling_percentile(df["donchian_width_pct"], ROLLING_PERCENTILE_WINDOW)
    df["trend_efficiency"] = df[f"trend_efficiency_{primary}"]

    df["atr14"] = true_range(df).rolling(14, min_periods=1).mean()
    df["atr_pct"] = df["atr14"] / close.replace(0, np.nan)
    df["realized_volatility"] = returns.rolling(REALIZED_VOL_WINDOW, min_periods=5).std(ddof=0)
    df["atr_percentile"] = rolling_percentile(df["atr_pct"], ROLLING_PERCENTILE_WINDOW)
    df["realized_volatility_percentile"] = rolling_percentile(df["realized_volatility"], ROLLING_PERCENTILE_WINDOW)
    prior_atr_median = df["atr_pct"].shift(1).rolling(REALIZED_VOL_WINDOW, min_periods=5).median()
    prior_rv_median = df["realized_volatility"].shift(1).rolling(REALIZED_VOL_WINDOW, min_periods=5).median()
    df["volatility_expansion"] = df["atr_pct"] / prior_atr_median.replace(0, np.nan) - 1.0
    df["realized_volatility_expansion"] = df["realized_volatility"] / prior_rv_median.replace(0, np.nan) - 1.0

    df["return_autocorrelation"] = returns.rolling(PERSISTENCE_WINDOW, min_periods=5).corr(returns.shift(1))
    run_lengths = signed_run_lengths(close.diff())
    df["consecutive_directional_bars"] = run_lengths.abs()
    df["positive_trend_run_length"] = run_lengths.clip(lower=0.0)
    df["negative_trend_run_length"] = (-run_lengths).clip(lower=0.0)
    signed_returns = np.sign(returns.fillna(0.0))
    df["directional_persistence_score"] = (
        pd.Series(signed_returns, index=df.index, dtype=float).rolling(PERSISTENCE_WINDOW, min_periods=5).mean()
    )
    rolling_return_mean = returns.rolling(PERSISTENCE_WINDOW, min_periods=5).mean()
    rolling_return_std = returns.rolling(PERSISTENCE_WINDOW, min_periods=5).std(ddof=0)
    df["rolling_sharpe_like_directional_score"] = (
        rolling_return_mean / rolling_return_std.replace(0, np.nan) * math.sqrt(PERSISTENCE_WINDOW)
    )

    adx = compute_adx(df, ADX_PERIOD)
    df["adx14"] = adx["adx14"]
    df["plus_di"] = adx["plus_di"]
    df["minus_di"] = adx["minus_di"]
    df["adx_or_proxy"] = df["adx14"].where(df["adx14"].notna(), df["trend_efficiency"] * 50.0)
    df = add_regime_labels(df, timeframe, windows)
    df["symbol_month_placeholder"] = ""
    return df


def classify_regime_row(row: pd.Series, timeframe: str, windows: list[int] | None = None) -> str:
    """Classify one closed bar into a trend regime label."""

    rules = REGIME_RULES.get(timeframe, REGIME_RULES["4h"])
    windows = windows or DEFAULT_TREND_WINDOWS
    spread = finite_float(row.get("ema_spread_pct")) or 0.0
    abs_spread = abs(spread)
    ema50_slope = finite_float(row.get("ema50_slope")) or 0.0
    ema200_slope = finite_float(row.get("ema200_slope")) or 0.0
    close_to_ema50 = finite_float(row.get("close_distance_to_ema50")) or 0.0
    close_to_ema200 = finite_float(row.get("close_distance_to_ema200")) or 0.0
    trend_efficiency = finite_float(row.get("trend_efficiency"))
    if trend_efficiency is None:
        trend_efficiency = max_available_efficiency(row, windows) or 0.0
    adx_or_proxy = finite_float(row.get("adx_or_proxy"))
    if adx_or_proxy is None:
        adx = finite_float(row.get("adx14"))
        adx_or_proxy = adx if adx is not None else trend_efficiency * 50.0
    atr_percentile = finite_float(row.get("atr_percentile"))
    rv_percentile = finite_float(row.get("realized_volatility_percentile"))
    width_percentile = finite_float(row.get("donchian_width_percentile"))
    high_vol = any(value is not None and value >= 0.80 for value in [atr_percentile, rv_percentile])
    low_vol = all(value is not None and value <= 0.25 for value in [atr_percentile, rv_percentile, width_percentile])
    weak_spread = rules["weak_ema_spread_pct"]
    strong_spread = rules["strong_ema_spread_pct"]

    up_structure = (
        spread >= weak_spread
        and ema50_slope > 0
        and ema200_slope >= 0
        and close_to_ema50 > -0.015
        and close_to_ema200 > -0.030
    )
    down_structure = (
        spread <= -weak_spread
        and ema50_slope < 0
        and ema200_slope <= 0
        and close_to_ema50 < 0.015
        and close_to_ema200 < 0.030
    )

    if up_structure:
        if trend_efficiency >= 0.35 and (adx_or_proxy >= 20.0 or abs_spread >= strong_spread):
            return "strong_uptrend"
        if trend_efficiency >= 0.18 or abs_spread >= strong_spread:
            return "weak_uptrend"

    if down_structure:
        if trend_efficiency >= 0.35 and (adx_or_proxy >= 20.0 or abs_spread >= strong_spread):
            return "strong_downtrend"
        if trend_efficiency >= 0.18 or abs_spread >= strong_spread:
            return "weak_downtrend"

    if low_vol and trend_efficiency < 0.25 and abs_spread < strong_spread:
        return "compression"
    if high_vol:
        return "high_vol_choppy"
    return "choppy"


def add_regime_labels(df: pd.DataFrame, timeframe: str, windows: list[int]) -> pd.DataFrame:
    """Add regime labels and trend score to an indicator frame."""

    result = df.copy()
    labels = [classify_regime_row(row, timeframe, windows) for _, row in result.iterrows()]
    result["regime_label"] = labels
    cap = REGIME_RULES.get(timeframe, REGIME_RULES["4h"])["trend_score_spread_cap"]
    spread_score = (pd.to_numeric(result["ema_spread_pct"], errors="coerce").abs() / cap).clip(0.0, 1.0)
    efficiency_score = pd.to_numeric(result["trend_efficiency"], errors="coerce").clip(0.0, 1.0)
    adx_score = (pd.to_numeric(result["adx_or_proxy"], errors="coerce") / 50.0).clip(0.0, 1.0)
    persistence_score = pd.to_numeric(result["directional_persistence_score"], errors="coerce").abs().clip(0.0, 1.0)
    result["trend_score"] = pd.concat([spread_score, efficiency_score, adx_score, persistence_score], axis=1).mean(axis=1)
    return result


def build_symbol_regime_dataset(
    vt_symbol: str,
    bars_1m: pd.DataFrame,
    history_range: HistoryRange,
    timeframes: list[str],
    windows: list[int],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build all requested timeframe regime rows for one symbol."""

    symbol_frames: list[pd.DataFrame] = []
    timeframe_quality: dict[str, Any] = {}
    for timeframe in timeframes:
        frame = resample_ohlcv(bars_1m, TIMEFRAME_MINUTES[timeframe])
        timeframe_quality[timeframe] = data_quality_for_timeframe(frame, timeframe, history_range)
        indicators = compute_regime_indicators(frame, timeframe, windows)
        indicators = filter_time_range(indicators, history_range)
        if indicators.empty:
            continue
        indicators.insert(0, "symbol", vt_symbol)
        indicators.insert(1, "timeframe", timeframe)
        symbol_frames.append(indicators)

    dataset = pd.concat(symbol_frames, ignore_index=True) if symbol_frames else empty_regime_dataset(windows)
    if not dataset.empty:
        dt = pd.to_datetime(dataset["datetime"])
        dataset["month"] = dt.dt.strftime("%Y-%m")
        quarter_dt = dt.dt.tz_localize(None) if dt.dt.tz is not None else dt
        dataset["quarter"] = quarter_dt.dt.to_period("Q").astype(str)
        dataset = dataset.drop(columns=["symbol_month_placeholder"], errors="ignore")
    return dataset, timeframe_quality


def empty_regime_dataset(windows: list[int]) -> pd.DataFrame:
    """Return an empty regime dataset with stable core columns."""

    columns = [
        "symbol",
        "timeframe",
        "datetime",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "ema50",
        "ema200",
        "ema_spread_pct",
        "ema50_slope",
        "ema200_slope",
        "close_distance_to_ema50",
        "close_distance_to_ema200",
        "atr14",
        "atr_pct",
        "realized_volatility",
        "atr_percentile",
        "realized_volatility_percentile",
        "donchian_width_percentile",
        "trend_efficiency",
        "return_autocorrelation",
        "consecutive_directional_bars",
        "directional_persistence_score",
        "rolling_sharpe_like_directional_score",
        "positive_trend_run_length",
        "negative_trend_run_length",
        "adx14",
        "plus_di",
        "minus_di",
        "adx_or_proxy",
        "regime_label",
        "trend_score",
        "month",
        "quarter",
    ]
    for window in windows:
        columns.extend(
            [
                f"donchian_high_{window}",
                f"donchian_low_{window}",
                f"donchian_mid_{window}",
                f"donchian_width_{window}",
                f"donchian_width_pct_{window}",
                f"donchian_breakout_direction_{window}",
                f"donchian_breakout_count_{window}",
                f"donchian_channel_direction_{window}",
                f"close_position_in_donchian_{window}",
                f"trend_efficiency_{window}",
            ]
        )
    return pd.DataFrame(columns=columns)


def build_regime_datasets(
    symbols: list[str],
    history_range: HistoryRange,
    timezone_name: str,
    timeframes: list[str],
    windows: list[int],
    logger: logging.Logger,
    *,
    bars_by_symbol: dict[str, pd.DataFrame] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load bars, check data quality, and build the full regime dataset."""

    symbol_frames: list[pd.DataFrame] = []
    symbol_coverage: dict[str, Any] = {}
    symbol_timeframes: dict[str, Any] = {}

    for vt_symbol in symbols:
        log_event(logger, logging.INFO, "trend_regime.load_symbol", "Loading symbol history", symbol=vt_symbol)
        if bars_by_symbol is not None and vt_symbol in bars_by_symbol:
            bars_1m = normalize_input_bars(vt_symbol, bars_by_symbol[vt_symbol], timezone_name)
        else:
            bars_1m = load_bars_from_db(vt_symbol, history_range, timezone_name, logger)

        coverage = build_1m_coverage_summary(bars_1m, history_range)
        symbol_coverage[vt_symbol] = coverage
        dataset, timeframe_quality = build_symbol_regime_dataset(vt_symbol, bars_1m, history_range, timeframes, windows)
        symbol_timeframes[vt_symbol] = timeframe_quality
        if not dataset.empty:
            symbol_frames.append(dataset)

    regime_dataset = pd.concat(symbol_frames, ignore_index=True) if symbol_frames else empty_regime_dataset(windows)
    data_quality = {
        "symbols": symbols,
        "start": history_range.start.isoformat(),
        "end_display": history_range.end_display.isoformat(),
        "end_exclusive": history_range.end_exclusive.isoformat(),
        "interval": DEFAULT_INTERVAL,
        "timezone": timezone_name,
        "symbol_coverage": symbol_coverage,
        "timeframes": symbol_timeframes,
        "all_symbols_complete": bool(symbol_coverage and all(item.get("required_coverage_ready") for item in symbol_coverage.values())),
        "silent_skip_symbol": False,
    }
    return regime_dataset, data_quality


def ensure_strict_data_quality(data_quality: dict[str, Any], data_check_strict: bool) -> None:
    """Raise when strict mode sees incomplete data or skipped symbols."""

    if not data_check_strict:
        return
    incomplete = [
        symbol
        for symbol, coverage in data_quality.get("symbol_coverage", {}).items()
        if not coverage.get("required_coverage_ready")
    ]
    if incomplete:
        raise TrendRegimeDiagnosticsError(f"Strict data check failed for symbols: {', '.join(incomplete)}")


def regime_count_columns() -> list[str]:
    """Return count and pct columns for all regime labels."""

    columns: list[str] = []
    for label in REGIME_LABELS:
        columns.append(f"{label}_count")
        columns.append(f"{label}_pct")
    return columns


def empty_aggregation(group_cols: list[str]) -> pd.DataFrame:
    """Return an empty aggregation with stable columns."""

    return pd.DataFrame(
        columns=group_cols
        + [
            "bar_count",
            "strong_trend_count",
            "trend_regime_count",
            "choppy_high_vol_count",
            "strong_trend_pct",
            "trend_regime_pct",
            "choppy_high_vol_pct",
            "avg_trend_score",
            "avg_trend_efficiency",
            "avg_abs_ema_spread_pct",
            "avg_atr_pct",
            "avg_adx_or_proxy",
        ]
        + regime_count_columns()
    )


def aggregate_regimes(regime_dataset: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    """Aggregate regime distribution and trend score by grouping columns."""

    if regime_dataset.empty:
        return empty_aggregation(group_cols)

    rows: list[dict[str, Any]] = []
    for group_key, group in regime_dataset.groupby(group_cols, dropna=False):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        row = {column: group_key[index] for index, column in enumerate(group_cols)}
        count = int(len(group.index))
        labels = group["regime_label"].fillna("unknown")
        row["bar_count"] = count
        for label in REGIME_LABELS:
            label_count = int((labels == label).sum())
            row[f"{label}_count"] = label_count
            row[f"{label}_pct"] = float(label_count / count) if count else 0.0
        strong_count = int(labels.isin(STRONG_REGIMES).sum())
        trend_count = int(labels.isin(TREND_REGIMES).sum())
        choppy_high_vol_count = int(labels.isin(CHOPPY_REGIMES).sum())
        row["strong_trend_count"] = strong_count
        row["trend_regime_count"] = trend_count
        row["choppy_high_vol_count"] = choppy_high_vol_count
        row["strong_trend_pct"] = float(strong_count / count) if count else 0.0
        row["trend_regime_pct"] = float(trend_count / count) if count else 0.0
        row["choppy_high_vol_pct"] = float(choppy_high_vol_count / count) if count else 0.0
        row["avg_trend_score"] = safe_mean(group["trend_score"])
        row["avg_trend_efficiency"] = safe_mean(group["trend_efficiency"])
        row["avg_abs_ema_spread_pct"] = safe_mean(pd.to_numeric(group["ema_spread_pct"], errors="coerce").abs())
        row["avg_atr_pct"] = safe_mean(group["atr_pct"])
        row["avg_adx_or_proxy"] = safe_mean(group["adx_or_proxy"])
        rows.append(row)
    return pd.DataFrame(rows).sort_values(group_cols, kind="stable").reset_index(drop=True)


def trend_score_table(regime_dataset: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    """Build compact trend score ranking table."""

    if regime_dataset.empty:
        return pd.DataFrame(
            columns=group_cols
            + [
                "bar_count",
                "avg_trend_score",
                "median_trend_score",
                "strong_trend_pct",
                "trend_regime_pct",
                "choppy_high_vol_pct",
                "avg_trend_efficiency",
                "avg_abs_ema_spread_pct",
                "avg_adx_or_proxy",
            ]
        )

    rows: list[dict[str, Any]] = []
    for group_key, group in regime_dataset.groupby(group_cols, dropna=False):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        row = {column: group_key[index] for index, column in enumerate(group_cols)}
        count = int(len(group.index))
        labels = group["regime_label"].fillna("unknown")
        row["bar_count"] = count
        row["avg_trend_score"] = safe_mean(group["trend_score"])
        row["median_trend_score"] = float(pd.to_numeric(group["trend_score"], errors="coerce").median())
        row["strong_trend_pct"] = float(labels.isin(STRONG_REGIMES).sum() / count) if count else 0.0
        row["trend_regime_pct"] = float(labels.isin(TREND_REGIMES).sum() / count) if count else 0.0
        row["choppy_high_vol_pct"] = float(labels.isin(CHOPPY_REGIMES).sum() / count) if count else 0.0
        row["avg_trend_efficiency"] = safe_mean(group["trend_efficiency"])
        row["avg_abs_ema_spread_pct"] = safe_mean(pd.to_numeric(group["ema_spread_pct"], errors="coerce").abs())
        row["avg_adx_or_proxy"] = safe_mean(group["adx_or_proxy"])
        rows.append(row)
    return pd.DataFrame(rows).sort_values("avg_trend_score", ascending=False, kind="stable").reset_index(drop=True)


def load_v3_trade_files(paths_by_split: dict[str, Path] | None = None) -> tuple[pd.DataFrame, list[str]]:
    """Load existing V3 extended trade CSVs when present."""

    if paths_by_split is None:
        paths_by_split = DEFAULT_V3_TRADE_FILES
    frames: list[pd.DataFrame] = []
    warnings: list[str] = []
    for split, path in paths_by_split.items():
        if not path.exists():
            warnings.append(f"trade_file_missing:{split}:{path}")
            continue
        frame = pd.read_csv(path)
        frame.insert(0, "split", split)
        frames.append(frame)
    if not frames:
        return empty_trade_frame(), warnings
    trades = pd.concat(frames, ignore_index=True)
    return trades, warnings


def empty_trade_frame() -> pd.DataFrame:
    """Return an empty trade frame with core V3 columns."""

    return pd.DataFrame(
        columns=[
            "split",
            "policy_name",
            "symbol",
            "direction",
            "entry_time",
            "exit_time",
            "net_pnl",
            "no_cost_pnl",
            "timeframe",
        ]
    )


def empty_trade_attribution() -> pd.DataFrame:
    """Return an empty trade regime attribution frame."""

    return pd.DataFrame(
        columns=[
            "split",
            "policy_name",
            "symbol",
            "direction",
            "entry_time",
            "exit_time",
            "net_pnl",
            "no_cost_pnl",
            "timeframe",
            "regime_at_entry",
            "trend_efficiency",
            "ema_spread_pct",
            "atr_pct",
            "adx_or_proxy",
            "is_regime_aligned",
        ]
    )


def expand_trades_for_timeframes(trades: pd.DataFrame, timeframes: list[str]) -> pd.DataFrame:
    """Duplicate mixed-timeframe trades so attribution can be evaluated on each requested timeframe."""

    if trades.empty:
        return empty_trade_attribution().drop(columns=["regime_at_entry", "trend_efficiency", "ema_spread_pct", "atr_pct", "adx_or_proxy", "is_regime_aligned"])

    records: list[dict[str, Any]] = []
    for _, row in trades.iterrows():
        raw_timeframe = str(row.get("timeframe") or "").strip()
        candidate_timeframes = [raw_timeframe] if raw_timeframe in timeframes else list(timeframes)
        for timeframe in candidate_timeframes:
            records.append(
                {
                    "split": row.get("split"),
                    "policy_name": row.get("policy_name"),
                    "symbol": row.get("symbol"),
                    "direction": row.get("direction"),
                    "entry_time": row.get("entry_time"),
                    "exit_time": row.get("exit_time"),
                    "net_pnl": row.get("net_pnl"),
                    "no_cost_pnl": row.get("no_cost_pnl", row.get("no_cost_net_pnl")),
                    "timeframe": timeframe,
                }
            )
    expanded = pd.DataFrame(records)
    expanded["entry_timestamp"] = pd.to_datetime(expanded["entry_time"], errors="coerce", utc=True)
    return expanded


def align_trades_to_regimes(trades: pd.DataFrame, regime_dataset: pd.DataFrame, timeframes: list[str]) -> pd.DataFrame:
    """Align each trade entry_time to the latest completed regime row for its symbol/timeframe."""

    if trades.empty or regime_dataset.empty:
        return empty_trade_attribution()

    expanded = expand_trades_for_timeframes(trades, timeframes)
    if expanded.empty:
        return empty_trade_attribution()

    output_frames: list[pd.DataFrame] = []
    regime_columns = [
        "datetime",
        "symbol",
        "timeframe",
        "regime_label",
        "trend_efficiency",
        "ema_spread_pct",
        "atr_pct",
        "adx_or_proxy",
    ]
    regimes = regime_dataset[regime_columns].copy()
    regimes["datetime"] = pd.to_datetime(regimes["datetime"], utc=True)
    for (symbol, timeframe), group in expanded.groupby(["symbol", "timeframe"], dropna=False):
        left = group.copy().sort_values("entry_timestamp", kind="stable")
        valid_left = left[left["entry_timestamp"].notna()].copy()
        invalid_left = left[left["entry_timestamp"].isna()].copy()
        right = regimes[(regimes["symbol"] == symbol) & (regimes["timeframe"] == timeframe)].copy()
        if valid_left.empty or right.empty:
            left["regime_at_entry"] = None
            left["trend_efficiency"] = np.nan
            left["ema_spread_pct"] = np.nan
            left["atr_pct"] = np.nan
            left["adx_or_proxy"] = np.nan
            output_frames.append(left)
            continue
        right = right.sort_values("datetime", kind="stable").rename(
            columns={"datetime": "regime_datetime", "regime_label": "regime_at_entry"}
        )
        merged = pd.merge_asof(
            valid_left,
            right[["regime_datetime", "regime_at_entry", "trend_efficiency", "ema_spread_pct", "atr_pct", "adx_or_proxy"]],
            left_on="entry_timestamp",
            right_on="regime_datetime",
            direction="backward",
        )
        if not invalid_left.empty:
            invalid_left["regime_at_entry"] = None
            invalid_left["trend_efficiency"] = np.nan
            invalid_left["ema_spread_pct"] = np.nan
            invalid_left["atr_pct"] = np.nan
            invalid_left["adx_or_proxy"] = np.nan
            merged = pd.concat([merged, invalid_left], ignore_index=True)
        output_frames.append(merged)

    if not output_frames:
        return empty_trade_attribution()
    attributed = pd.concat(output_frames, ignore_index=True)
    attributed["net_pnl"] = pd.to_numeric(attributed["net_pnl"], errors="coerce")
    attributed["no_cost_pnl"] = pd.to_numeric(attributed["no_cost_pnl"], errors="coerce")
    attributed["is_regime_aligned"] = [
        is_trade_regime_aligned(direction, regime)
        for direction, regime in zip(attributed["direction"], attributed["regime_at_entry"], strict=False)
    ]
    columns = empty_trade_attribution().columns.tolist()
    return attributed[columns].sort_values(["split", "policy_name", "entry_time", "symbol", "timeframe"], kind="stable").reset_index(drop=True)


def is_trade_regime_aligned(direction: Any, regime: Any) -> bool:
    """Return whether a trade direction matches the entry regime direction."""

    text = str(direction or "").lower()
    label = str(regime or "")
    if text == "long":
        return label in UPTREND_REGIMES
    if text == "short":
        return label in DOWNTREND_REGIMES
    return False


def policy_family(policy_name: Any) -> str:
    """Map V3 policy names to requested families."""

    name = str(policy_name or "")
    if "1d_ema" in name:
        return "1d_ema"
    if "1d_donchian" in name:
        return "1d_donchian"
    if "4h_ema" in name:
        return "4h_ema"
    if "4h_donchian" in name or "4h_vol_compression_donchian" in name:
        return "4h_donchian"
    if "ensemble" in name:
        return "ensemble"
    return "other"


def build_policy_regime_performance(trade_attribution: pd.DataFrame) -> pd.DataFrame:
    """Aggregate policy performance by entry regime."""

    columns = [
        "split",
        "policy_name",
        "policy_family",
        "timeframe",
        "regime_at_entry",
        "trade_count",
        "aligned_trade_count",
        "net_pnl",
        "no_cost_pnl",
        "avg_net_pnl",
        "avg_no_cost_pnl",
        "positive_trade_count",
        "negative_trade_count",
        "win_rate",
        "positive_pnl_sum",
        "loss_abs_sum",
    ]
    if trade_attribution.empty:
        return pd.DataFrame(columns=columns)

    working = trade_attribution.copy()
    working["policy_family"] = working["policy_name"].map(policy_family)
    rows: list[dict[str, Any]] = []
    for keys, group in working.groupby(["split", "policy_name", "policy_family", "timeframe", "regime_at_entry"], dropna=False):
        split, policy_name, family, timeframe, regime = keys
        net = pd.to_numeric(group["net_pnl"], errors="coerce").fillna(0.0)
        no_cost = pd.to_numeric(group["no_cost_pnl"], errors="coerce").fillna(0.0)
        positive = net[net > 0]
        negative = net[net < 0]
        trade_count = int(len(group.index))
        rows.append(
            {
                "split": split,
                "policy_name": policy_name,
                "policy_family": family,
                "timeframe": timeframe,
                "regime_at_entry": regime,
                "trade_count": trade_count,
                "aligned_trade_count": int(group["is_regime_aligned"].fillna(False).sum()),
                "net_pnl": float(net.sum()),
                "no_cost_pnl": float(no_cost.sum()),
                "avg_net_pnl": float(net.mean()) if trade_count else 0.0,
                "avg_no_cost_pnl": float(no_cost.mean()) if trade_count else 0.0,
                "positive_trade_count": int((net > 0).sum()),
                "negative_trade_count": int((net < 0).sum()),
                "win_rate": float((net > 0).sum() / trade_count) if trade_count else 0.0,
                "positive_pnl_sum": float(positive.sum()),
                "loss_abs_sum": float(negative.abs().sum()),
            }
        )
    return pd.DataFrame(rows, columns=columns).sort_values(
        ["split", "policy_name", "timeframe", "regime_at_entry"], kind="stable"
    )


def distribution_from_labels(labels: pd.Series) -> dict[str, Any]:
    """Build count and pct distribution for regime labels."""

    total = int(len(labels.index))
    counts = labels.fillna("unknown").value_counts().to_dict()
    return {
        "total": total,
        "counts": {label: int(counts.get(label, 0)) for label in REGIME_LABELS},
        "pct": {label: (float(counts.get(label, 0) / total) if total else 0.0) for label in REGIME_LABELS},
    }


def top_label_by_positive_pnl(trade_attribution: pd.DataFrame) -> dict[str, Any]:
    """Return positive-pnl regime concentration."""

    if trade_attribution.empty:
        return {"regime": None, "positive_pnl_sum": 0.0, "share": 0.0}
    net = pd.to_numeric(trade_attribution["net_pnl"], errors="coerce").fillna(0.0)
    positive = trade_attribution.loc[net > 0].copy()
    if positive.empty:
        return {"regime": None, "positive_pnl_sum": 0.0, "share": 0.0}
    positive["positive_pnl"] = pd.to_numeric(positive["net_pnl"], errors="coerce").fillna(0.0)
    grouped = positive.groupby("regime_at_entry", dropna=False)["positive_pnl"].sum().sort_values(ascending=False)
    total = float(grouped.sum())
    regime = str(grouped.index[0]) if len(grouped.index) else None
    value = float(grouped.iloc[0]) if len(grouped.index) else 0.0
    return {"regime": regime, "positive_pnl_sum": value, "share": float(value / total) if total else 0.0}


def top_label_by_loss(trade_attribution: pd.DataFrame) -> dict[str, Any]:
    """Return loss regime concentration."""

    if trade_attribution.empty:
        return {"regime": None, "loss_abs_sum": 0.0, "share": 0.0}
    net = pd.to_numeric(trade_attribution["net_pnl"], errors="coerce").fillna(0.0)
    losses = trade_attribution.loc[net < 0].copy()
    if losses.empty:
        return {"regime": None, "loss_abs_sum": 0.0, "share": 0.0}
    losses["loss_abs"] = net.loc[losses.index].abs()
    grouped = losses.groupby("regime_at_entry", dropna=False)["loss_abs"].sum().sort_values(ascending=False)
    total = float(grouped.sum())
    regime = str(grouped.index[0]) if len(grouped.index) else None
    value = float(grouped.iloc[0]) if len(grouped.index) else 0.0
    return {"regime": regime, "loss_abs_sum": value, "share": float(value / total) if total else 0.0}


def pnl_share_for_regimes(trade_attribution: pd.DataFrame, regimes: set[str], *, positive: bool) -> float:
    """Return positive pnl or absolute loss share for selected regimes."""

    if trade_attribution.empty:
        return 0.0
    net = pd.to_numeric(trade_attribution["net_pnl"], errors="coerce").fillna(0.0)
    if positive:
        subset = trade_attribution.loc[net > 0].copy()
        if subset.empty:
            return 0.0
        values = pd.to_numeric(subset["net_pnl"], errors="coerce").fillna(0.0)
        total = float(values.sum())
        selected = float(values[subset["regime_at_entry"].isin(regimes)].sum())
    else:
        subset = trade_attribution.loc[net < 0].copy()
        if subset.empty:
            return 0.0
        values = pd.to_numeric(subset["net_pnl"], errors="coerce").fillna(0.0).abs()
        total = float(values.sum())
        selected = float(values[subset["regime_at_entry"].isin(regimes)].sum())
    return float(selected / total) if total else 0.0


def build_family_diagnostics(trade_attribution: pd.DataFrame) -> dict[str, Any]:
    """Build family-level diagnostics used in recommendations and report answers."""

    diagnostics: dict[str, Any] = {}
    if trade_attribution.empty:
        return diagnostics
    working = trade_attribution.copy()
    working["policy_family"] = working["policy_name"].map(policy_family)
    for family, group in working.groupby("policy_family", dropna=False):
        net = pd.to_numeric(group["net_pnl"], errors="coerce").fillna(0.0)
        no_cost = pd.to_numeric(group["no_cost_pnl"], errors="coerce").fillna(0.0)
        strong = group["regime_at_entry"].isin(STRONG_REGIMES)
        choppy = group["regime_at_entry"].isin(CHOPPY_REGIMES)
        nonstrong = ~strong
        diagnostics[str(family)] = {
            "trade_count": int(len(group.index)),
            "net_pnl": float(net.sum()),
            "no_cost_pnl": float(no_cost.sum()),
            "strong_trade_count": int(strong.sum()),
            "strong_trade_share": float(strong.sum() / len(group.index)) if len(group.index) else 0.0,
            "strong_net_pnl": float(net[strong].sum()),
            "strong_no_cost_pnl": float(no_cost[strong].sum()),
            "nonstrong_net_pnl": float(net[nonstrong].sum()),
            "nonstrong_no_cost_pnl": float(no_cost[nonstrong].sum()),
            "strong_avg_no_cost_pnl": float(no_cost[strong].mean()) if int(strong.sum()) else None,
            "overall_avg_no_cost_pnl": float(no_cost.mean()) if len(group.index) else None,
            "choppy_high_vol_trade_share": float(choppy.sum() / len(group.index)) if len(group.index) else 0.0,
            "choppy_high_vol_loss_share": pnl_share_for_regimes(group, CHOPPY_REGIMES, positive=False),
            "strong_positive_pnl_share": pnl_share_for_regimes(group, STRONG_REGIMES, positive=True),
        }
    return diagnostics


def build_trade_diagnostics(trade_attribution: pd.DataFrame, regime_distribution: dict[str, Any]) -> dict[str, Any]:
    """Summarize V3 trade attribution against regime labels."""

    total_trades = int(len(trade_attribution.index))
    if trade_attribution.empty:
        return {
            "trade_count": 0,
            "profitable_trade_main_regime": None,
            "losing_trade_main_regime": None,
            "strong_positive_pnl_share": 0.0,
            "choppy_high_vol_loss_share": 0.0,
            "choppy_high_vol_trade_share": 0.0,
            "regime_choppy_high_vol_pct": float(
                regime_distribution.get("pct", {}).get("choppy", 0.0)
                + regime_distribution.get("pct", {}).get("high_vol_choppy", 0.0)
            ),
            "v3_trades_too_much_in_choppy": False,
            "family_diagnostics": {},
            "one_day_ema_only_strong_effective": False,
            "one_day_ema_strong_outperforms_full": False,
            "donchian_losses_mainly_choppy_high_vol": False,
        }

    labels = trade_attribution["regime_at_entry"]
    choppy_trade_share = float(labels.isin(CHOPPY_REGIMES).sum() / total_trades) if total_trades else 0.0
    regime_choppy_pct = float(
        regime_distribution.get("pct", {}).get("choppy", 0.0)
        + regime_distribution.get("pct", {}).get("high_vol_choppy", 0.0)
    )
    family_diagnostics = build_family_diagnostics(trade_attribution)
    one_day_ema = family_diagnostics.get("1d_ema", {})
    strong_avg = one_day_ema.get("strong_avg_no_cost_pnl")
    overall_avg = one_day_ema.get("overall_avg_no_cost_pnl")
    one_day_ema_only_strong_effective = bool(
        one_day_ema
        and (one_day_ema.get("strong_no_cost_pnl") or 0.0) > 0
        and (one_day_ema.get("nonstrong_no_cost_pnl") or 0.0) <= 0
    )
    one_day_ema_strong_outperforms_full = bool(
        strong_avg is not None and overall_avg is not None and strong_avg > overall_avg and (one_day_ema.get("strong_no_cost_pnl") or 0.0) > 0
    )
    donchian_loss_shares = [
        family_diagnostics.get(family, {}).get("choppy_high_vol_loss_share", 0.0)
        for family in ["1d_donchian", "4h_donchian"]
    ]
    donchian_losses_mainly_choppy_high_vol = bool(donchian_loss_shares and max(donchian_loss_shares) >= 0.50)
    return {
        "trade_count": total_trades,
        "profitable_trade_main_regime": top_label_by_positive_pnl(trade_attribution),
        "losing_trade_main_regime": top_label_by_loss(trade_attribution),
        "strong_positive_pnl_share": pnl_share_for_regimes(trade_attribution, STRONG_REGIMES, positive=True),
        "choppy_high_vol_loss_share": pnl_share_for_regimes(trade_attribution, CHOPPY_REGIMES, positive=False),
        "choppy_high_vol_trade_share": choppy_trade_share,
        "regime_choppy_high_vol_pct": regime_choppy_pct,
        "v3_trades_too_much_in_choppy": bool(choppy_trade_share > max(0.40, regime_choppy_pct)),
        "family_diagnostics": family_diagnostics,
        "one_day_ema_only_strong_effective": one_day_ema_only_strong_effective,
        "one_day_ema_strong_outperforms_full": one_day_ema_strong_outperforms_full,
        "donchian_losses_mainly_choppy_high_vol": donchian_losses_mainly_choppy_high_vol,
    }


def read_external_v3_constraints() -> dict[str, Any]:
    """Read V3 extended comparison and funding stress notes when available."""

    constraints: dict[str, Any] = {
        "compare_summary_found": DEFAULT_COMPARE_SUMMARY.exists(),
        "funding_stress_found": DEFAULT_FUNDING_STRESS.exists(),
        "stable_candidate_exists": None,
        "stable_candidates": [],
        "trend_following_v3_extended_failed": None,
        "can_enter_v3_1_research_prior": None,
        "high_top_5pct_trade_pnl_contribution_policies": [],
        "high_largest_symbol_pnl_share_policies": [],
        "funding_stress_negative_at_3bps_or_more": None,
        "funding_stress_notice": None,
    }
    if DEFAULT_COMPARE_SUMMARY.exists():
        try:
            summary = json.loads(DEFAULT_COMPARE_SUMMARY.read_text(encoding="utf-8"))
            constraints.update(
                {
                    "stable_candidate_exists": summary.get("stable_candidate_exists"),
                    "stable_candidates": summary.get("stable_candidates") or [],
                    "trend_following_v3_extended_failed": summary.get("trend_following_v3_extended_failed"),
                    "can_enter_v3_1_research_prior": summary.get("can_enter_v3_1_research"),
                    "high_top_5pct_trade_pnl_contribution_policies": summary.get("high_top_5pct_trade_pnl_contribution_policies") or [],
                    "high_largest_symbol_pnl_share_policies": summary.get("high_largest_symbol_pnl_share_policies") or [],
                    "funding_stress_notice": summary.get("funding_stress_notice"),
                }
            )
        except Exception as exc:
            constraints["compare_summary_warning"] = f"read_failed:{exc!r}"
    if DEFAULT_FUNDING_STRESS.exists():
        try:
            funding = pd.read_csv(DEFAULT_FUNDING_STRESS)
            if not funding.empty and {"funding_bps_per_8h", "remains_positive_after_funding"}.issubset(funding.columns):
                bps = pd.to_numeric(funding["funding_bps_per_8h"], errors="coerce")
                remains = funding["remains_positive_after_funding"].astype(str).str.lower().isin(["true", "1", "yes"])
                constraints["funding_stress_negative_at_3bps_or_more"] = bool((bps >= 3.0).any() and (~remains[bps >= 3.0]).any())
        except Exception as exc:
            constraints["funding_stress_warning"] = f"read_failed:{exc!r}"
    return constraints


def build_v3_1_recommendations(
    regime_distribution: dict[str, Any],
    trade_diagnostics: dict[str, Any],
    external_constraints: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build research-only V3.1 recommendation from regime and trade evidence."""

    external_constraints = external_constraints or {}
    strong_trend_exists = float(
        regime_distribution.get("pct", {}).get("strong_uptrend", 0.0)
        + regime_distribution.get("pct", {}).get("strong_downtrend", 0.0)
    ) >= 0.08
    profits_mainly_strong = float(trade_diagnostics.get("strong_positive_pnl_share", 0.0)) >= 0.50
    losses_mainly_choppy = float(trade_diagnostics.get("choppy_high_vol_loss_share", 0.0)) >= 0.50
    one_day_ema_better = bool(trade_diagnostics.get("one_day_ema_strong_outperforms_full"))
    proceed = bool(strong_trend_exists and profits_mainly_strong and losses_mainly_choppy and one_day_ema_better)

    recommended_policy_families: list[str] = []
    rejected_policy_families = ["1d_donchian", "4h_ema", "4h_donchian", "ensemble"]
    family_diagnostics = trade_diagnostics.get("family_diagnostics", {})
    if proceed and family_diagnostics.get("1d_ema", {}).get("strong_no_cost_pnl", 0.0) > 0:
        recommended_policy_families.append("1d_ema")
        if "1d_ema" in rejected_policy_families:
            rejected_policy_families.remove("1d_ema")
    else:
        rejected_policy_families.insert(0, "1d_ema")
    if not proceed:
        recommended_policy_families = []
        rejected_policy_families = ["1d_ema", "1d_donchian", "4h_ema", "4h_donchian", "ensemble"]

    rationale = [
        f"strong_trend_exists={str(strong_trend_exists).lower()}",
        f"profits_mainly_strong={str(profits_mainly_strong).lower()}",
        f"losses_mainly_choppy_high_vol={str(losses_mainly_choppy).lower()}",
        f"one_day_ema_strong_outperforms_full={str(one_day_ema_better).lower()}",
        "strategy_development_allowed=false and demo_live_allowed=false by design.",
    ]
    if external_constraints.get("funding_stress_negative_at_3bps_or_more"):
        rationale.append("Existing V3 best-policy synthetic funding stress turns negative at or above 3 bps / 8h.")
    if external_constraints.get("high_top_5pct_trade_pnl_contribution_policies"):
        rationale.append("Existing V3 extended comparison flags top-trade contribution concentration.")
    if external_constraints.get("stable_candidate_exists") is False:
        rationale.append("Existing V3 extended comparison has stable_candidate_exists=false.")

    return {
        "proceed_to_v3_1_research": proceed,
        "allowed_research_only": proceed,
        "strategy_development_allowed": False,
        "demo_live_allowed": False,
        "recommended_filters": {
            "min_trend_efficiency": 0.35,
            "min_adx_or_proxy": 20.0,
            "allowed_regimes": ["strong_uptrend", "strong_downtrend"],
            "blocked_regimes": ["choppy", "high_vol_choppy", "compression"],
            "max_atr_percentile": 0.80,
            "min_ema_spread_pct": {
                "4h": REGIME_RULES["4h"]["weak_ema_spread_pct"],
                "1d": REGIME_RULES["1d"]["weak_ema_spread_pct"],
            },
        },
        "recommended_policy_families": recommended_policy_families,
        "rejected_policy_families": rejected_policy_families,
        "rationale": rationale,
    }


def first_row_value(frame: pd.DataFrame, column: str) -> Any:
    """Return first-row column value or None."""

    if frame.empty or column not in frame.columns:
        return None
    return frame.iloc[0][column]


def strongest_and_weakest(score_table: pd.DataFrame, name_column: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return strongest and weakest group rows by avg_trend_score."""

    if score_table.empty:
        return {}, {}
    sorted_table = score_table.sort_values("avg_trend_score", ascending=False, kind="stable")
    return (
        dataframe_records(sorted_table.head(1))[0],
        dataframe_records(sorted_table.tail(1))[0],
    )


def build_summary(
    *,
    symbols: list[str],
    timeframes: list[str],
    windows: list[int],
    history_range: HistoryRange,
    data_quality: dict[str, Any],
    regime_dataset: pd.DataFrame,
    regime_by_symbol: pd.DataFrame,
    regime_by_month: pd.DataFrame,
    regime_by_quarter: pd.DataFrame,
    regime_by_timeframe: pd.DataFrame,
    trend_score_by_symbol: pd.DataFrame,
    trend_score_by_month: pd.DataFrame,
    trade_attribution: pd.DataFrame,
    policy_regime_performance: pd.DataFrame,
    trade_warnings: list[str],
    external_constraints: dict[str, Any],
) -> dict[str, Any]:
    """Build the main JSON summary payload."""

    regime_distribution = distribution_from_labels(regime_dataset["regime_label"] if not regime_dataset.empty else pd.Series(dtype=object))
    trade_diagnostics = build_trade_diagnostics(trade_attribution, regime_distribution)
    recommendations = build_v3_1_recommendations(regime_distribution, trade_diagnostics, external_constraints)

    strongest_symbol, weakest_symbol = strongest_and_weakest(trend_score_by_symbol, "symbol")
    strongest_month, weakest_month = strongest_and_weakest(trend_score_by_month, "month")
    trend_score_by_quarter = trend_score_table(regime_dataset, ["quarter"])
    strongest_quarter, weakest_quarter = strongest_and_weakest(trend_score_by_quarter, "quarter")

    timeframe_rows = {str(row.get("timeframe")): row for row in dataframe_records(regime_by_timeframe)}
    four_hour_score = finite_float(timeframe_rows.get("4h", {}).get("avg_trend_score"))
    one_day_score = finite_float(timeframe_rows.get("1d", {}).get("avg_trend_score"))
    one_day_better = bool(one_day_score is not None and four_hour_score is not None and one_day_score > four_hour_score)

    answers = {
        "sufficient_trend_regime_exists": bool(recommendations["proceed_to_v3_1_research"] or regime_distribution["pct"].get("strong_uptrend", 0.0) + regime_distribution["pct"].get("strong_downtrend", 0.0) >= 0.08),
        "best_trending_symbol": strongest_symbol.get("symbol"),
        "most_choppy_symbol": weakest_symbol.get("symbol"),
        "strongest_month": strongest_month.get("month"),
        "weakest_month": weakest_month.get("month"),
        "strongest_quarter": strongest_quarter.get("quarter"),
        "weakest_quarter": weakest_quarter.get("quarter"),
        "one_day_timeframe_better_than_4h": one_day_better,
        "ema_regime_better_matches_structure_than_donchian": bool(
            trade_diagnostics.get("one_day_ema_strong_outperforms_full")
            and trade_diagnostics.get("donchian_losses_mainly_choppy_high_vol")
        ),
        "v3_losses_mainly_choppy_high_vol": bool(trade_diagnostics.get("choppy_high_vol_loss_share", 0.0) >= 0.50),
        "oos_best_policy_only_effective_in_strong_trend": bool(trade_diagnostics.get("one_day_ema_only_strong_effective")),
        "recommend_v3_1_research": bool(recommendations["proceed_to_v3_1_research"]),
    }

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "symbols": symbols,
            "start": history_range.start.isoformat(),
            "end_display": history_range.end_display.isoformat(),
            "end_exclusive": history_range.end_exclusive.isoformat(),
            "timeframes": timeframes,
            "trend_score_window_bars": windows,
            "interval": DEFAULT_INTERVAL,
        },
        "data_quality_passed": bool(data_quality.get("all_symbols_complete")),
        "regime_distribution": regime_distribution,
        "strong_trend_pct": float(regime_distribution["pct"].get("strong_uptrend", 0.0) + regime_distribution["pct"].get("strong_downtrend", 0.0)),
        "choppy_high_vol_pct": float(regime_distribution["pct"].get("choppy", 0.0) + regime_distribution["pct"].get("high_vol_choppy", 0.0)),
        "strongest_symbol": strongest_symbol,
        "weakest_symbol": weakest_symbol,
        "strongest_month": strongest_month,
        "weakest_month": weakest_month,
        "strongest_quarter": strongest_quarter,
        "weakest_quarter": weakest_quarter,
        "timeframe_comparison": {
            "1d_better_than_4h_by_avg_trend_score": one_day_better,
            "4h": timeframe_rows.get("4h", {}),
            "1d": timeframe_rows.get("1d", {}),
        },
        "trade_warnings": trade_warnings,
        "trade_diagnostics": trade_diagnostics,
        "answers": answers,
        "external_v3_constraints": external_constraints,
        "recommendations": recommendations,
        "output_files": REQUIRED_OUTPUT_FILES,
        "policy_regime_performance_rows": int(len(policy_regime_performance.index)),
    }


def pct_text(value: Any) -> str:
    """Format a decimal pct value."""

    number = finite_float(value)
    if number is None:
        return "n/a"
    return f"{number * 100:.2f}%"


def number_text(value: Any, digits: int = 4) -> str:
    """Format a number for markdown."""

    number = finite_float(value)
    if number is None:
        return "n/a"
    return f"{number:.{digits}f}"


def generate_report(summary: dict[str, Any]) -> str:
    """Generate the markdown trend regime diagnostics report."""

    regime_distribution = summary.get("regime_distribution", {})
    trade = summary.get("trade_diagnostics", {})
    rec = summary.get("recommendations", {})
    answers = summary.get("answers", {})
    external = summary.get("external_v3_constraints", {})
    family = trade.get("family_diagnostics", {})
    one_day_ema = family.get("1d_ema", {})
    four_hour_donchian = family.get("4h_donchian", {})
    one_day_donchian = family.get("1d_donchian", {})

    lines = [
        "# Trend Regime Diagnostics",
        "",
        "## 诊断边界",
        "- 本报告是趋势环境诊断，不是策略开发、参数搜索、demo runner 或 live runner。",
        "- Regime 只使用已完成的 4h/1d bar；V3 trade 归因用 entry_time 向后对齐到最近一个已完成 regime bar。",
        "- Funding fee 未进入 V3 trade PnL；下方只引用已有 synthetic funding stress 作为风险约束。",
        "",
        "## Regime 分类规则",
        "- strong_uptrend/strong_downtrend：EMA50/EMA200 spread 达到对应 timeframe 阈值，EMA50 与 EMA200 slope 同向，trend_efficiency >= 0.35，且 ADX14 或 proxy >= 20。",
        "- weak_uptrend/weak_downtrend：EMA 结构同向，但 trend_efficiency 或 EMA spread 未达到 strong 条件。",
        "- compression：ATR percentile、realized volatility percentile、Donchian width percentile 均处低位，且 trend_efficiency < 0.25。",
        "- high_vol_choppy：未形成趋势结构，但 ATR 或 realized volatility percentile >= 0.80。",
        "- choppy：不满足趋势、压缩或高波动震荡条件的剩余样本。",
        "",
        "## Regime 分布",
        f"- strong trend 占比：{pct_text(summary.get('strong_trend_pct'))}",
        f"- choppy/high_vol_choppy 占比：{pct_text(summary.get('choppy_high_vol_pct'))}",
        f"- strong_uptrend：{pct_text((regime_distribution.get('pct') or {}).get('strong_uptrend'))}",
        f"- strong_downtrend：{pct_text((regime_distribution.get('pct') or {}).get('strong_downtrend'))}",
        f"- compression：{pct_text((regime_distribution.get('pct') or {}).get('compression'))}",
        "",
        "## 核心排序",
        f"- 趋势性最好 symbol：{(summary.get('strongest_symbol') or {}).get('symbol')}，avg_trend_score={number_text((summary.get('strongest_symbol') or {}).get('avg_trend_score'))}",
        f"- 最震荡 symbol：{(summary.get('weakest_symbol') or {}).get('symbol')}，avg_trend_score={number_text((summary.get('weakest_symbol') or {}).get('avg_trend_score'))}",
        f"- 趋势性最强月份：{(summary.get('strongest_month') or {}).get('month')}，avg_trend_score={number_text((summary.get('strongest_month') or {}).get('avg_trend_score'))}",
        f"- 最不适合趋势跟踪月份：{(summary.get('weakest_month') or {}).get('month')}，avg_trend_score={number_text((summary.get('weakest_month') or {}).get('avg_trend_score'))}",
        f"- 趋势性最强季度：{(summary.get('strongest_quarter') or {}).get('quarter')}，avg_trend_score={number_text((summary.get('strongest_quarter') or {}).get('avg_trend_score'))}",
        f"- 最不适合趋势跟踪季度：{(summary.get('weakest_quarter') or {}).get('quarter')}，avg_trend_score={number_text((summary.get('weakest_quarter') or {}).get('avg_trend_score'))}",
        f"- 1d 是否优于 4h：{str(bool((summary.get('timeframe_comparison') or {}).get('1d_better_than_4h_by_avg_trend_score'))).lower()}",
        "",
        "## V3.0 Trade Regime 归因",
        f"- 盈利交易主要 regime：{(trade.get('profitable_trade_main_regime') or {}).get('regime')}，share={pct_text((trade.get('profitable_trade_main_regime') or {}).get('share'))}",
        f"- 亏损交易主要 regime：{(trade.get('losing_trade_main_regime') or {}).get('regime')}，share={pct_text((trade.get('losing_trade_main_regime') or {}).get('share'))}",
        f"- 盈利 PnL 来自 strong trend 的占比：{pct_text(trade.get('strong_positive_pnl_share'))}",
        f"- 亏损来自 choppy/high_vol_choppy 的占比：{pct_text(trade.get('choppy_high_vol_loss_share'))}",
        f"- V3.0 是否在 choppy/high_vol_choppy 中交易过多：{str(bool(trade.get('v3_trades_too_much_in_choppy'))).lower()}",
        f"- 1d EMA strong_no_cost_pnl={number_text(one_day_ema.get('strong_no_cost_pnl'))}，nonstrong_no_cost_pnl={number_text(one_day_ema.get('nonstrong_no_cost_pnl'))}",
        f"- 4h Donchian choppy/high_vol loss share={pct_text(four_hour_donchian.get('choppy_high_vol_loss_share'))}",
        f"- 1d Donchian choppy/high_vol loss share={pct_text(one_day_donchian.get('choppy_high_vol_loss_share'))}",
        "",
        "## 必答问题",
        f"1. 2023-2026 是否存在足够趋势 regime？{str(bool(answers.get('sufficient_trend_regime_exists'))).lower()}。",
        f"2. 哪些 symbol 趋势性最好？{answers.get('best_trending_symbol')}。",
        f"3. 哪些 symbol 最震荡？{answers.get('most_choppy_symbol')}。",
        f"4. 哪些月份/季度趋势性最强？{answers.get('strongest_month')} / {answers.get('strongest_quarter')}。",
        f"5. 哪些月份/季度最不适合趋势跟踪？{answers.get('weakest_month')} / {answers.get('weakest_quarter')}。",
        f"6. 1d timeframe 是否优于 4h timeframe？{str(bool(answers.get('one_day_timeframe_better_than_4h'))).lower()}。",
        f"7. EMA regime 是否比 Donchian breakout 更符合趋势结构？{str(bool(answers.get('ema_regime_better_matches_structure_than_donchian'))).lower()}。",
        f"8. V3.0 亏损是否主要来自 choppy/high_vol_choppy？{str(bool(answers.get('v3_losses_mainly_choppy_high_vol'))).lower()}。",
        f"9. OOS best policy 是否只是在某些 strong trend regime 中有效？{str(bool(answers.get('oos_best_policy_only_effective_in_strong_trend'))).lower()}。",
        f"10. 是否建议进入 V3.1 research？{str(bool(answers.get('recommend_v3_1_research'))).lower()}。",
        f"11. 如果建议 V3.1，应保留哪些方向？{', '.join(rec.get('recommended_policy_families') or []) or 'none'}；filters={rec.get('recommended_filters')}",
        f"12. 如果不建议 V3.1，应停止哪些方向？{', '.join(rec.get('rejected_policy_families') or []) or 'none'}。",
        "",
        "## Funding 与集中度约束",
        f"- stable_candidate_exists={str(external.get('stable_candidate_exists')).lower() if external.get('stable_candidate_exists') is not None else 'unknown'}",
        f"- high_top_5pct_trade_pnl_contribution_policies={external.get('high_top_5pct_trade_pnl_contribution_policies')}",
        f"- high_largest_symbol_pnl_share_policies={external.get('high_largest_symbol_pnl_share_policies')}",
        f"- funding_stress_negative_at_3bps_or_more={str(external.get('funding_stress_negative_at_3bps_or_more')).lower() if external.get('funding_stress_negative_at_3bps_or_more') is not None else 'unknown'}",
        "",
        "## V3.1 Research-only 建议",
        f"- proceed_to_v3_1_research={str(bool(rec.get('proceed_to_v3_1_research'))).lower()}",
        f"- allowed_research_only={str(bool(rec.get('allowed_research_only'))).lower()}",
        f"- strategy_development_allowed={str(bool(rec.get('strategy_development_allowed'))).lower()}",
        f"- demo_live_allowed={str(bool(rec.get('demo_live_allowed'))).lower()}",
        f"- recommended_filters={json.dumps(rec.get('recommended_filters'), ensure_ascii=False)}",
        f"- recommended_policy_families={rec.get('recommended_policy_families')}",
        f"- rejected_policy_families={rec.get('rejected_policy_families')}",
        "",
        "## 输出文件",
    ]
    lines.extend([f"- {name}" for name in REQUIRED_OUTPUT_FILES])
    lines.append("")
    return "\n".join(lines)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write a JSON file."""

    path.write_text(json.dumps(to_jsonable(payload), ensure_ascii=False, indent=2), encoding="utf-8")


def write_outputs(
    output_dir: Path,
    *,
    summary: dict[str, Any],
    data_quality: dict[str, Any],
    regime_dataset: pd.DataFrame,
    regime_by_symbol: pd.DataFrame,
    regime_by_month: pd.DataFrame,
    regime_by_quarter: pd.DataFrame,
    regime_by_timeframe: pd.DataFrame,
    trend_score_by_symbol: pd.DataFrame,
    trend_score_by_month: pd.DataFrame,
    trade_attribution: pd.DataFrame,
    policy_regime_performance: pd.DataFrame,
    recommendations: dict[str, Any],
) -> None:
    """Write all required output artifacts."""

    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "trend_regime_summary.json", summary)
    write_json(output_dir / "data_quality.json", data_quality)
    write_json(output_dir / "v3_1_regime_recommendations.json", recommendations)
    (output_dir / "trend_regime_report.md").write_text(generate_report(summary), encoding="utf-8")

    regime_dataset.to_csv(output_dir / "regime_dataset.csv", index=False)
    regime_by_symbol.to_csv(output_dir / "regime_by_symbol.csv", index=False)
    regime_by_month.to_csv(output_dir / "regime_by_month.csv", index=False)
    regime_by_quarter.to_csv(output_dir / "regime_by_quarter.csv", index=False)
    regime_by_timeframe.to_csv(output_dir / "regime_by_timeframe.csv", index=False)
    trend_score_by_symbol.to_csv(output_dir / "trend_score_by_symbol.csv", index=False)
    trend_score_by_month.to_csv(output_dir / "trend_score_by_month.csv", index=False)
    trade_attribution.to_csv(output_dir / "trade_regime_attribution.csv", index=False)
    policy_regime_performance.to_csv(output_dir / "policy_regime_performance.csv", index=False)


def run_diagnostics(
    *,
    symbols: list[str],
    start: str,
    end: str,
    timezone_name: str,
    output_dir: Path,
    timeframes: list[str],
    windows: list[int],
    data_check_strict: bool,
    logger: logging.Logger,
    bars_by_symbol: dict[str, pd.DataFrame] | None = None,
    trade_paths: dict[str, Path] | None = None,
) -> DiagnosticOutputs:
    """Run the full trend regime diagnostics workflow."""

    history_range = resolve_history_range(start, end, timezone_name)
    regime_dataset, data_quality = build_regime_datasets(
        symbols,
        history_range,
        timezone_name,
        timeframes,
        windows,
        logger,
        bars_by_symbol=bars_by_symbol,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "data_quality.json", data_quality)
    ensure_strict_data_quality(data_quality, data_check_strict)

    regime_by_symbol = aggregate_regimes(regime_dataset, ["symbol"])
    regime_by_month = aggregate_regimes(regime_dataset, ["month"])
    regime_by_quarter = aggregate_regimes(regime_dataset, ["quarter"])
    regime_by_timeframe = aggregate_regimes(regime_dataset, ["timeframe"])
    trend_score_by_symbol = trend_score_table(regime_dataset, ["symbol"])
    trend_score_by_month = trend_score_table(regime_dataset, ["month"])

    trades, trade_warnings = load_v3_trade_files(trade_paths)
    trade_attribution = align_trades_to_regimes(trades, regime_dataset, timeframes)
    policy_regime_performance = build_policy_regime_performance(trade_attribution)
    external_constraints = read_external_v3_constraints()
    summary = build_summary(
        symbols=symbols,
        timeframes=timeframes,
        windows=windows,
        history_range=history_range,
        data_quality=data_quality,
        regime_dataset=regime_dataset,
        regime_by_symbol=regime_by_symbol,
        regime_by_month=regime_by_month,
        regime_by_quarter=regime_by_quarter,
        regime_by_timeframe=regime_by_timeframe,
        trend_score_by_symbol=trend_score_by_symbol,
        trend_score_by_month=trend_score_by_month,
        trade_attribution=trade_attribution,
        policy_regime_performance=policy_regime_performance,
        trade_warnings=trade_warnings,
        external_constraints=external_constraints,
    )
    recommendations = summary["recommendations"]
    write_outputs(
        output_dir,
        summary=summary,
        data_quality=data_quality,
        regime_dataset=regime_dataset,
        regime_by_symbol=regime_by_symbol,
        regime_by_month=regime_by_month,
        regime_by_quarter=regime_by_quarter,
        regime_by_timeframe=regime_by_timeframe,
        trend_score_by_symbol=trend_score_by_symbol,
        trend_score_by_month=trend_score_by_month,
        trade_attribution=trade_attribution,
        policy_regime_performance=policy_regime_performance,
        recommendations=recommendations,
    )
    return DiagnosticOutputs(
        output_dir=output_dir,
        summary=summary,
        data_quality=data_quality,
        regime_dataset=regime_dataset,
        trade_attribution=trade_attribution,
        policy_regime_performance=policy_regime_performance,
        recommendations=recommendations,
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""

    ensure_headless_runtime()
    args = parse_args(argv)
    logger = setup_logging("diagnose_trend_regimes", verbose=args.verbose)
    try:
        symbols = parse_symbol_list(args.symbols)
        timeframes = parse_timeframes(args.timeframes)
        windows = parse_positive_int_list(args.trend_score_window_bars, "--trend-score-window-bars")
        output_dir = resolve_path(args.output_dir)
        outputs = run_diagnostics(
            symbols=symbols,
            start=args.start,
            end=args.end,
            timezone_name=args.timezone,
            output_dir=output_dir,
            timeframes=timeframes,
            windows=windows,
            data_check_strict=bool(args.data_check_strict),
            logger=logger,
        )
        log_event(
            logger,
            logging.INFO,
            "trend_regime.completed",
            "Trend regime diagnostics completed",
            output_dir=outputs.output_dir,
            proceed_to_v3_1_research=outputs.recommendations.get("proceed_to_v3_1_research"),
        )
        if args.json:
            print_json_block(outputs.summary)
        return 0
    except TrendRegimeDiagnosticsError as exc:
        log_event(logger, logging.ERROR, "trend_regime.error", str(exc))
        return 2
    except Exception as exc:
        log_event(logger, logging.ERROR, "trend_regime.unexpected_error", str(exc), exc_info=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
