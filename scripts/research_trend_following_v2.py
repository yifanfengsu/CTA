#!/usr/bin/env python3
"""Offline Trend Following V2 research with complete trade simulation."""

from __future__ import annotations

import argparse
import json
import logging
import math
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
from history_utils import build_instrument_config_path, get_database_timezone


DEFAULT_VT_SYMBOL = "BTCUSDT_SWAP_OKX.GLOBAL"
DEFAULT_ATR_MULTS = "3.0,4.0,5.0"
DEFAULT_FIXED_SIZE = 0.01
DEFAULT_RISK_PER_TRADE = 0.001
ROLLING_PERCENTILE_WINDOW = 240
WARMUP_DAYS = 45
TIMEFRAME_MINUTES = {"1m": 1, "15m": 15, "1h": 60, "4h": 240}
COMPARE_MIN_TRADE_COUNT = 10
COMPARE_MAX_DDPERCENT_THRESHOLD = 30.0
COMPARE_LOW_FREQUENCY_TRADE_COUNT = 20
COMPARE_MAX_EXPLAINABLE_OOS_COST_LOSS_TO_NO_COST = 0.5
SPLIT_RANGES = {
    "train": ("2025-01-01", "2025-09-30"),
    "validation": ("2025-10-01", "2025-12-31"),
    "oos": ("2026-01-01", "2026-03-31"),
    "full": ("2025-01-01", "2026-03-31"),
}
BASE_POLICY_ORDER = [
    "tf_1h_donchian_20_10",
    "tf_1h_donchian_55_20",
    "tf_4h_donchian_20_10",
    "tf_1h_ema_cross_atr_trail",
    "tf_4h_ema_cross_atr_trail",
    "tf_1h_vol_compression_breakout",
    "tf_1h_donchian_55_with_risk_filters",
    "tf_4h_donchian_20_with_risk_filters",
]
POLICY_DESCRIPTIONS = {
    "tf_1h_donchian_20_10": "1h close breaks previous Donchian 20; exit on previous Donchian 10 or ATR trail.",
    "tf_1h_donchian_55_20": "1h close breaks previous Donchian 55; exit on previous Donchian 20 or ATR trail.",
    "tf_4h_donchian_20_10": "4h close breaks previous Donchian 20; 15m bars approximate execution.",
    "tf_1h_ema_cross_atr_trail": "1h EMA50/EMA200 trend with close on the EMA50 side; exit on EMA50 loss or ATR trail.",
    "tf_4h_ema_cross_atr_trail": "4h EMA50/EMA200 trend with ATR trailing exit.",
    "tf_1h_vol_compression_breakout": "1h low-volatility and narrow-channel compression followed by Donchian breakout.",
    "tf_1h_donchian_55_with_risk_filters": "1h Donchian 55/20 with Signal Lab risk percentiles capped at 0.8.",
    "tf_4h_donchian_20_with_risk_filters": "4h Donchian 20/10 with latest completed 15m Signal Lab risk filters.",
}
LEADERBOARD_COLUMNS = [
    "policy_name",
    "base_policy_name",
    "atr_mult",
    "timeframe",
    "trade_count",
    "long_count",
    "short_count",
    "gross_pnl",
    "net_pnl",
    "no_cost_net_pnl",
    "fee_total",
    "slippage_total",
    "win_rate",
    "profit_factor",
    "avg_win",
    "avg_loss",
    "avg_trade_net_pnl",
    "median_trade_net_pnl",
    "max_drawdown",
    "max_ddpercent",
    "return_drawdown_ratio",
    "sharpe_like",
    "avg_holding_minutes",
    "median_holding_minutes",
    "top_5pct_trade_pnl_contribution",
    "best_trade",
    "worst_trade",
    "cost_drag",
]
TRADE_COLUMNS = [
    "policy_name",
    "base_policy_name",
    "atr_mult",
    "timeframe",
    "direction",
    "volume",
    "contract_size",
    "entry_time",
    "entry_price",
    "exit_time",
    "exit_price",
    "exit_reason",
    "holding_minutes",
    "gross_pnl",
    "fee",
    "slippage",
    "net_pnl",
    "no_cost_net_pnl",
    "r_multiple",
    "mfe",
    "mae",
    "size",
    "entry_signal_time",
    "exit_signal_time",
    "used_htf_bar_time",
    "entry_atr",
    "initial_risk",
]


class TrendFollowingV2Error(Exception):
    """Raised when Trend Following V2 research cannot continue."""


@dataclass(frozen=True, slots=True)
class PolicyRun:
    """One concrete policy run, including ATR-trailing multiplier."""

    policy_name: str
    base_policy_name: str
    timeframe: str
    entry_type: str
    entry_window: int | None
    exit_window: int | None
    atr_mult: float
    use_risk_filters: bool = False
    use_donchian_exit: bool = False
    use_ema_exit: bool = False
    use_atr_trailing: bool = True
    max_hold_bars: int | None = None
    description: str = ""


@dataclass(slots=True)
class OpenPosition:
    """Mutable state for one simulated position."""

    direction: str
    entry_signal_time: pd.Timestamp
    entry_time: pd.Timestamp
    entry_price: float
    size: float
    contract_size: float
    entry_atr: float | None
    initial_risk: float | None
    highest_close: float
    lowest_close: float
    trailing_stop: float | None
    entry_htf_index: int


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description="Research low-frequency Trend Following V2 policies.")
    parser.add_argument("--vt-symbol", default=DEFAULT_VT_SYMBOL, help=f"Default: {DEFAULT_VT_SYMBOL}.")
    parser.add_argument("--split", choices=sorted(SPLIT_RANGES), default="train", help="Sample split preset.")
    parser.add_argument("--start", help="Start date/datetime. Defaults to the selected split preset.")
    parser.add_argument("--end", help="End date/datetime. Defaults to the selected split preset.")
    parser.add_argument(
        "--timezone",
        default=DEFAULT_TIMEZONE,
        help=f"Timezone used for bars and reports. Default: {DEFAULT_TIMEZONE}.",
    )
    parser.add_argument("--output-dir", help="Default: reports/research/trend_following_v2/<split>.")
    parser.add_argument("--data-check-strict", action="store_true", help="Fail when requested 1m data has gaps.")
    parser.add_argument("--capital", type=float, default=5000.0, help="Starting capital for equity metrics.")
    parser.add_argument("--rate", type=float, default=0.0005, help="Per-side fee rate.")
    parser.add_argument(
        "--slippage-mode",
        choices=["ticks", "absolute"],
        default="ticks",
        help="Interpret --slippage as ticks or absolute price. Default: ticks.",
    )
    parser.add_argument("--slippage", type=float, default=2.0, help="Per-side slippage input.")
    parser.add_argument("--max-runs", type=int, help="Optional cap on policy x ATR-mult runs.")
    parser.add_argument(
        "--atr-mults",
        default=DEFAULT_ATR_MULTS,
        help=f"Comma-separated ATR trailing multipliers. Default: {DEFAULT_ATR_MULTS}.",
    )
    parser.add_argument(
        "--sizing-mode",
        choices=["fixed", "atr"],
        default="fixed",
        help="Position sizing mode. Default: fixed.",
    )
    parser.add_argument("--fixed-size", type=float, default=DEFAULT_FIXED_SIZE, help="Default fixed position size.")
    parser.add_argument(
        "--risk-per-trade",
        type=float,
        default=DEFAULT_RISK_PER_TRADE,
        help="Capital fraction used only when --sizing-mode=atr.",
    )
    parser.add_argument(
        "--bars-from-db",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Load 1m bars from vn.py sqlite. Default: enabled.",
    )
    parser.add_argument("--json", action="store_true", help="Print trend_policy_summary.json payload to stdout.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose structured logs.")
    return parser.parse_args(argv)


def resolve_path(path_arg: str | Path | None, default_path: Path) -> Path:
    """Resolve a path relative to the project root."""

    path = Path(path_arg) if path_arg else default_path
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def parse_number_list(raw_value: str, option_name: str) -> list[float]:
    """Parse comma-separated positive floats."""

    values: list[float] = []
    for token in str(raw_value or "").split(","):
        text = token.strip()
        if not text:
            continue
        try:
            value = float(text)
        except ValueError as exc:
            raise TrendFollowingV2Error(f"{option_name} 包含非法数字: {text!r}") from exc
        if value <= 0:
            raise TrendFollowingV2Error(f"{option_name} 必须为正数: {value}")
        values.append(value)
    if not values:
        raise TrendFollowingV2Error(f"{option_name} 不能为空")
    return sorted(set(values))


def resolve_split_range(split: str, start_arg: str | None, end_arg: str | None, timezone_name: str) -> HistoryRange:
    """Resolve CLI start/end using split defaults when needed."""

    default_start, default_end = SPLIT_RANGES[split]
    start = start_arg or default_start
    end = end_arg or default_end
    try:
        return parse_history_range(start, end, pd.Timedelta(minutes=1).to_pytimedelta(), timezone_name)
    except ValueError as exc:
        raise TrendFollowingV2Error(str(exc)) from exc


def finite_or_none(value: Any) -> float | None:
    """Return a finite float or None."""

    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number):
        return None
    return number


def iso_or_none(value: Any) -> str | None:
    """Return ISO timestamp text."""

    if value is None or pd.isna(value):
        return None
    return pd.Timestamp(value).isoformat()


def dataframe_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Convert a DataFrame to JSON-safe records."""

    if df.empty:
        return []
    safe_df = df.replace({np.nan: None})
    return json.loads(safe_df.to_json(orient="records", force_ascii=False))


def bar_to_record(bar: Any) -> dict[str, Any]:
    """Convert one vn.py BarData-like object to a plain record."""

    return {
        "datetime": getattr(bar, "datetime", None),
        "open": getattr(bar, "open_price", None),
        "high": getattr(bar, "high_price", None),
        "low": getattr(bar, "low_price", None),
        "close": getattr(bar, "close_price", None),
        "volume": getattr(bar, "volume", None),
    }


def normalize_1m_bars(bars_df: pd.DataFrame, timezone_name: str) -> pd.DataFrame:
    """Normalize an OHLCV DataFrame into sorted 1m bars."""

    result = dataframe_bars_to_ohlc(bars_df, timezone_name)
    result["datetime"] = pd.to_datetime(result["datetime"])
    for column in ["open", "high", "low", "close", "volume"]:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    result = result.dropna(subset=["datetime", "open", "high", "low", "close"])
    result = result.sort_values("datetime", kind="stable").drop_duplicates("datetime", keep="last")
    return result.reset_index(drop=True)


def load_bars_from_db(
    vt_symbol: str,
    history_range: HistoryRange,
    timezone_name: str,
    logger: logging.Logger,
) -> pd.DataFrame:
    """Load 1m bars from vn.py sqlite, including warmup before the research split."""

    symbol, exchange_value = split_vt_symbol(vt_symbol)
    exchange = resolve_exchange(exchange_value)

    from vnpy.trader.constant import Interval
    from vnpy.trader.database import get_database

    configure_sqlite_settings(logger)
    db_tz = get_database_timezone()
    query_start = (
        history_range.start - pd.Timedelta(days=WARMUP_DAYS).to_pytimedelta()
    ).astimezone(db_tz).replace(tzinfo=None)
    query_end = history_range.end_exclusive.astimezone(db_tz).replace(tzinfo=None)
    bars = get_database().load_bar_data(symbol, exchange, Interval.MINUTE, query_start, query_end)
    records = [bar_to_record(bar) for bar in bars]
    if not records:
        return pd.DataFrame(columns=["datetime", "open", "high", "low", "close", "volume"])
    return normalize_1m_bars(pd.DataFrame(records), timezone_name)


def read_instrument_meta(vt_symbol: str) -> dict[str, Any]:
    """Read local instrument metadata when available."""

    path = build_instrument_config_path(vt_symbol)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise TrendFollowingV2Error(f"读取 instrument config 失败: {path} | {exc!r}") from exc


def resolve_absolute_slippage(vt_symbol: str, slippage_mode: str, slippage: float) -> float:
    """Resolve per-side absolute slippage."""

    if slippage < 0:
        raise TrendFollowingV2Error(f"slippage 不能小于 0: {slippage}")
    if slippage_mode == "absolute":
        return float(slippage)
    meta = read_instrument_meta(vt_symbol)
    pricetick = finite_or_none(meta.get("pricetick"))
    if pricetick is None or pricetick <= 0:
        raise TrendFollowingV2Error("slippage-mode=ticks 时需要有效 pricetick instrument config")
    return float(slippage) * pricetick


def resolve_contract_size(vt_symbol: str) -> float:
    """Resolve contract size from local instrument metadata."""

    meta = read_instrument_meta(vt_symbol)
    contract_size = finite_or_none(meta.get("size", meta.get("contract_size")))
    if contract_size is None or contract_size <= 0:
        raise TrendFollowingV2Error("Trend V2 研究需要有效 instrument size/contract_size")
    return float(contract_size)


def filter_time_range(df: pd.DataFrame, history_range: HistoryRange) -> pd.DataFrame:
    """Filter bars to the requested research period."""

    if df.empty:
        return df.copy()
    dt = pd.to_datetime(df["datetime"])
    mask = (dt >= pd.Timestamp(history_range.start)) & (dt < pd.Timestamp(history_range.end_exclusive))
    return df.loc[mask].copy().reset_index(drop=True)


def resample_ohlcv(bars_1m: pd.DataFrame, minutes: int) -> pd.DataFrame:
    """Resample 1m bars into closed OHLCV bars timestamped at the final included minute."""

    if minutes <= 0:
        raise TrendFollowingV2Error("resample minutes must be positive")
    if bars_1m.empty:
        return pd.DataFrame(columns=["datetime", "open", "high", "low", "close", "volume"])

    working = bars_1m.copy()
    working["datetime"] = pd.to_datetime(working["datetime"])
    working = working.sort_values("datetime", kind="stable").set_index("datetime")
    rule = f"{minutes}min"
    grouped = working.resample(rule, label="left", closed="left")
    result = grouped.agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
    volume_count = grouped["volume"].count()
    last_timestamp = working.index.to_series().resample(rule, label="left", closed="left").max()
    expected_close_timestamp = last_timestamp.index + pd.Timedelta(minutes=minutes - 1)
    complete_group = (last_timestamp == expected_close_timestamp) & (volume_count == minutes)
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


def compute_recent_return(close: pd.Series, timeframe_minutes: int, lookback_minutes: int) -> pd.Series:
    """Compute close-to-close return for a lookback supported by the timeframe."""

    if lookback_minutes < timeframe_minutes:
        return pd.Series(np.nan, index=close.index, dtype=float)
    periods = max(1, lookback_minutes // timeframe_minutes)
    return close / close.shift(periods) - 1.0


def add_donchian_channels(df: pd.DataFrame, windows: list[int] | None = None) -> pd.DataFrame:
    """Add previous-bar Donchian channels, excluding the current bar."""

    result = df.copy()
    high = pd.to_numeric(result["high"], errors="coerce")
    low = pd.to_numeric(result["low"], errors="coerce")
    for window in windows or [10, 20, 55]:
        result[f"donchian_high_{window}_prev"] = high.rolling(window, min_periods=window).max().shift(1)
        result[f"donchian_low_{window}_prev"] = low.rolling(window, min_periods=window).min().shift(1)
    return result


def compute_indicators(bars: pd.DataFrame, timeframe_minutes: int) -> pd.DataFrame:
    """Compute indicators needed by Trend Following V2 policies."""

    df = bars.copy().sort_values("datetime", kind="stable").reset_index(drop=True)
    if df.empty:
        return df

    for column in ["open", "high", "low", "close", "volume"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    for span in [50, 200]:
        df[f"ema{span}"] = close.ewm(span=span, adjust=False, min_periods=1).mean()

    df["atr14"] = true_range(df).rolling(14, min_periods=1).mean()
    df["atr_pct"] = df["atr14"] / close.replace(0, np.nan)
    df = add_donchian_channels(df, [10, 20, 55])

    width_20 = (df["donchian_high_20_prev"] - df["donchian_low_20_prev"]) / close.replace(0, np.nan)
    df["donchian_width_20_pct"] = width_20
    df["donchian_width_20_percentile"] = rolling_percentile(width_20)
    df["atr_pct_percentile"] = rolling_percentile(df["atr_pct"])
    df["atr_pct_percentile_prev"] = df["atr_pct_percentile"].shift(1)
    df["donchian_width_20_percentile_prev"] = df["donchian_width_20_percentile"].shift(1)

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
    for column in ["recent_volatility_30m", "volume_zscore_30m", "body_ratio"]:
        df[f"{column}_percentile"] = rolling_percentile(df[column])
    df["directional_recent_return_30m_percentile_long"] = rolling_percentile(df["recent_return_30m"])
    df["directional_recent_return_30m_percentile_short"] = rolling_percentile(-df["recent_return_30m"])
    return df


def build_timeframes(bars_1m: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Build 15m, 1h, and 4h bars from 1m data."""

    return {
        "1m": bars_1m.copy(),
        "15m": resample_ohlcv(bars_1m, 15),
        "1h": resample_ohlcv(bars_1m, 60),
        "4h": resample_ohlcv(bars_1m, 240),
    }


def build_indicator_frames(timeframes: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Compute indicators for all research timeframes."""

    return {
        "15m": compute_indicators(timeframes["15m"], 15),
        "1h": compute_indicators(timeframes["1h"], 60),
        "4h": compute_indicators(timeframes["4h"], 240),
    }


def risk_filter_columns() -> list[str]:
    """Return 15m risk columns used by Signal Lab-style filters."""

    return [
        "atr_pct_percentile",
        "recent_volatility_30m_percentile",
        "directional_recent_return_30m_percentile_long",
        "directional_recent_return_30m_percentile_short",
        "body_ratio_percentile",
        "volume_zscore_30m_percentile",
    ]


def prepare_policy_frame(indicators: dict[str, pd.DataFrame], timeframe: str, history_range: HistoryRange) -> pd.DataFrame:
    """Attach latest completed 15m risk context to one HTF indicator frame."""

    htf = indicators[timeframe].copy().sort_values("datetime", kind="stable").reset_index(drop=True)
    if htf.empty:
        return htf
    risk_15m = indicators["15m"][["datetime"] + risk_filter_columns()].copy().sort_values("datetime", kind="stable")
    risk_15m["used_risk_15m_bar_time"] = risk_15m["datetime"]
    risk_15m = risk_15m.rename(
        columns={column: f"risk_{column}" for column in risk_filter_columns()}
    )
    merged = pd.merge_asof(htf, risk_15m, on="datetime", direction="backward")
    return filter_time_range(merged, history_range)


def format_atr_mult(value: float) -> str:
    """Format ATR multiplier for policy names."""

    return f"{value:g}".replace(".", "p")


def build_policy_runs(atr_mults: list[float], max_runs: int | None = None) -> list[PolicyRun]:
    """Build all concrete policy x ATR-mult runs."""

    templates: list[dict[str, Any]] = [
        {
            "base_policy_name": "tf_1h_donchian_20_10",
            "timeframe": "1h",
            "entry_type": "donchian",
            "entry_window": 20,
            "exit_window": 10,
            "use_donchian_exit": True,
        },
        {
            "base_policy_name": "tf_1h_donchian_55_20",
            "timeframe": "1h",
            "entry_type": "donchian",
            "entry_window": 55,
            "exit_window": 20,
            "use_donchian_exit": True,
        },
        {
            "base_policy_name": "tf_4h_donchian_20_10",
            "timeframe": "4h",
            "entry_type": "donchian",
            "entry_window": 20,
            "exit_window": 10,
            "use_donchian_exit": True,
        },
        {
            "base_policy_name": "tf_1h_ema_cross_atr_trail",
            "timeframe": "1h",
            "entry_type": "ema",
            "entry_window": None,
            "exit_window": None,
            "use_ema_exit": True,
        },
        {
            "base_policy_name": "tf_4h_ema_cross_atr_trail",
            "timeframe": "4h",
            "entry_type": "ema",
            "entry_window": None,
            "exit_window": None,
            "use_ema_exit": True,
        },
        {
            "base_policy_name": "tf_1h_vol_compression_breakout",
            "timeframe": "1h",
            "entry_type": "vol_compression",
            "entry_window": 20,
            "exit_window": 10,
            "use_donchian_exit": True,
        },
        {
            "base_policy_name": "tf_1h_donchian_55_with_risk_filters",
            "timeframe": "1h",
            "entry_type": "donchian",
            "entry_window": 55,
            "exit_window": 20,
            "use_donchian_exit": True,
            "use_risk_filters": True,
        },
        {
            "base_policy_name": "tf_4h_donchian_20_with_risk_filters",
            "timeframe": "4h",
            "entry_type": "donchian",
            "entry_window": 20,
            "exit_window": 10,
            "use_donchian_exit": True,
            "use_risk_filters": True,
        },
    ]
    runs: list[PolicyRun] = []
    for template in templates:
        base_name = template["base_policy_name"]
        for atr_mult in atr_mults:
            runs.append(
                PolicyRun(
                    policy_name=f"{base_name}_atr{format_atr_mult(atr_mult)}",
                    base_policy_name=base_name,
                    timeframe=template["timeframe"],
                    entry_type=template["entry_type"],
                    entry_window=template["entry_window"],
                    exit_window=template["exit_window"],
                    atr_mult=float(atr_mult),
                    use_risk_filters=bool(template.get("use_risk_filters", False)),
                    use_donchian_exit=bool(template.get("use_donchian_exit", False)),
                    use_ema_exit=bool(template.get("use_ema_exit", False)),
                    description=POLICY_DESCRIPTIONS.get(base_name, ""),
                )
            )
    if max_runs is not None:
        if max_runs <= 0:
            raise TrendFollowingV2Error("--max-runs 必须为正数")
        runs = runs[:max_runs]
    return runs


def numeric_row_value(row: pd.Series, column: str) -> float | None:
    """Return a finite numeric row value."""

    return finite_or_none(row.get(column))


def risk_filters_pass(row: pd.Series, direction: str) -> bool:
    """Return whether Signal Lab risk percentiles allow a trade."""

    directional_column = f"risk_directional_recent_return_30m_percentile_{direction}"
    required = [
        "risk_recent_volatility_30m_percentile",
        "risk_atr_pct_percentile",
        directional_column,
        "risk_body_ratio_percentile",
        "risk_volume_zscore_30m_percentile",
    ]
    for column in required:
        value = numeric_row_value(row, column)
        if value is None or value > 0.8:
            return False
    return True


def policy_entry_signal(row: pd.Series, policy: PolicyRun, direction: str) -> bool:
    """Return whether one completed HTF bar triggers an entry."""

    if direction not in {"long", "short"}:
        return False
    if policy.use_risk_filters and not risk_filters_pass(row, direction):
        return False

    close = numeric_row_value(row, "close")
    if close is None:
        return False

    if policy.entry_type == "donchian":
        if policy.entry_window is None:
            return False
        high = numeric_row_value(row, f"donchian_high_{policy.entry_window}_prev")
        low = numeric_row_value(row, f"donchian_low_{policy.entry_window}_prev")
        if direction == "long":
            return high is not None and close > high
        return low is not None and close < low

    if policy.entry_type == "ema":
        ema50 = numeric_row_value(row, "ema50")
        ema200 = numeric_row_value(row, "ema200")
        if ema50 is None or ema200 is None:
            return False
        if direction == "long":
            return ema50 > ema200 and close > ema50
        return ema50 < ema200 and close < ema50

    if policy.entry_type == "vol_compression":
        if policy.entry_window is None:
            return False
        atr_percentile = numeric_row_value(row, "atr_pct_percentile_prev")
        width_percentile = numeric_row_value(row, "donchian_width_20_percentile_prev")
        if atr_percentile is None or width_percentile is None or atr_percentile > 0.4 or width_percentile > 0.4:
            return False
        high = numeric_row_value(row, f"donchian_high_{policy.entry_window}_prev")
        low = numeric_row_value(row, f"donchian_low_{policy.entry_window}_prev")
        if direction == "long":
            return high is not None and close > high
        return low is not None and close < low

    raise TrendFollowingV2Error(f"Unknown entry_type: {policy.entry_type}")


def check_atr_trailing_stop(
    direction: str,
    close: float,
    atr: float | None,
    highest_close: float,
    lowest_close: float,
    atr_mult: float,
    current_trailing_stop: float | None = None,
) -> tuple[bool, float | None]:
    """Check closed-bar ATR trailing stop without loosening the existing stop."""

    if atr is None or atr <= 0:
        return False, current_trailing_stop
    previous_stop = finite_or_none(current_trailing_stop)
    if direction == "long":
        candidate = highest_close - atr_mult * atr
        trail = candidate if previous_stop is None else max(previous_stop, candidate)
        return bool(close < trail), float(trail)
    candidate = lowest_close + atr_mult * atr
    trail = candidate if previous_stop is None else min(previous_stop, candidate)
    return bool(close > trail), float(trail)


def check_donchian_exit(row: pd.Series, policy: PolicyRun, direction: str) -> bool:
    """Return whether the previous-channel Donchian exit is hit."""

    if policy.exit_window is None:
        return False
    close = numeric_row_value(row, "close")
    if close is None:
        return False
    if direction == "long":
        low = numeric_row_value(row, f"donchian_low_{policy.exit_window}_prev")
        return bool(low is not None and close < low)
    high = numeric_row_value(row, f"donchian_high_{policy.exit_window}_prev")
    return bool(high is not None and close > high)


def check_ema_exit(row: pd.Series, direction: str) -> bool:
    """Return whether close crosses the EMA50 exit threshold."""

    close = numeric_row_value(row, "close")
    ema50 = numeric_row_value(row, "ema50")
    if close is None or ema50 is None:
        return False
    if direction == "long":
        return bool(close < ema50)
    return bool(close > ema50)


def find_next_execution_row(
    execution_df: pd.DataFrame,
    signal_time: pd.Timestamp,
    end_exclusive: pd.Timestamp,
) -> pd.Series | None:
    """Find the first 15m execution bar strictly after a signal time."""

    if execution_df.empty:
        return None
    times = pd.to_datetime(execution_df["datetime"])
    index = times.searchsorted(pd.Timestamp(signal_time), side="right")
    if index >= len(execution_df.index):
        return None
    row = execution_df.iloc[int(index)]
    if pd.Timestamp(row["datetime"]) >= end_exclusive:
        return None
    return row


def final_execution_row(execution_df: pd.DataFrame, end_exclusive: pd.Timestamp) -> pd.Series | None:
    """Return the final available execution bar inside the research split."""

    if execution_df.empty:
        return None
    times = pd.to_datetime(execution_df["datetime"])
    eligible = execution_df.loc[times < end_exclusive]
    if eligible.empty:
        return None
    return eligible.iloc[-1]


def resolve_trade_size(
    sizing_mode: str,
    fixed_size: float,
    capital: float,
    risk_per_trade: float,
    atr: float | None,
    atr_mult: float,
    contract_size: float = 1.0,
) -> float:
    """Resolve simulated position volume in contracts."""

    if fixed_size <= 0:
        raise TrendFollowingV2Error("--fixed-size 必须为正数")
    if contract_size <= 0:
        raise TrendFollowingV2Error("contract_size 必须为正数")
    if sizing_mode == "fixed":
        return float(fixed_size)
    if risk_per_trade <= 0:
        raise TrendFollowingV2Error("--risk-per-trade 必须为正数")
    if atr is None or atr <= 0 or atr_mult <= 0:
        return float(fixed_size)
    risk_budget = capital * risk_per_trade
    risk_per_unit = atr * atr_mult * contract_size
    return float(max(fixed_size, risk_budget / risk_per_unit))


def calculate_trade_costs(
    entry_price: float,
    exit_price: float,
    volume: float,
    contract_size: float,
    direction: str,
    rate: float,
    absolute_slippage: float,
) -> dict[str, float]:
    """Calculate gross, no-cost, and cost-aware trade PnL."""

    if rate < 0:
        raise TrendFollowingV2Error("--rate 不能为负数")
    if absolute_slippage < 0:
        raise TrendFollowingV2Error("absolute_slippage 不能为负数")
    if volume < 0:
        raise TrendFollowingV2Error("volume 不能为负数")
    if contract_size <= 0:
        raise TrendFollowingV2Error("contract_size 必须为正数")
    sign = 1.0 if direction == "long" else -1.0
    notional_units = volume * contract_size
    gross_pnl = (exit_price - entry_price) * sign * notional_units
    fee = rate * (abs(entry_price * notional_units) + abs(exit_price * notional_units))
    slippage_cost = absolute_slippage * notional_units * 2.0
    net_pnl = gross_pnl - fee - slippage_cost
    return {
        "gross_pnl": float(gross_pnl),
        "no_cost_net_pnl": float(gross_pnl),
        "fee": float(fee),
        "slippage": float(slippage_cost),
        "net_pnl": float(net_pnl),
    }


def compute_mfe_mae(
    execution_df: pd.DataFrame,
    direction: str,
    entry_time: pd.Timestamp,
    exit_time: pd.Timestamp,
    entry_price: float,
    volume: float,
    contract_size: float,
) -> tuple[float, float]:
    """Compute MFE and MAE in PnL units from 15m bars over the holding window."""

    if execution_df.empty:
        return 0.0, 0.0
    times = pd.to_datetime(execution_df["datetime"])
    window = execution_df.loc[(times >= entry_time) & (times <= exit_time)]
    if window.empty:
        return 0.0, 0.0
    high = pd.to_numeric(window["high"], errors="coerce")
    low = pd.to_numeric(window["low"], errors="coerce")
    if direction == "long":
        favorable = max(float((high - entry_price).max()), 0.0)
        adverse = max(float((entry_price - low).max()), 0.0)
    else:
        favorable = max(float((entry_price - low).max()), 0.0)
        adverse = max(float((high - entry_price).max()), 0.0)
    notional_units = volume * contract_size
    return float(favorable * notional_units), float(adverse * notional_units)


def build_trade_record(
    policy: PolicyRun,
    position: OpenPosition,
    exit_row: pd.Series,
    exit_signal_time: pd.Timestamp,
    exit_reason: str,
    execution_df: pd.DataFrame,
    rate: float,
    absolute_slippage: float,
) -> dict[str, Any] | None:
    """Build one closed trade record."""

    exit_time = pd.Timestamp(exit_row["datetime"])
    if exit_time <= position.entry_time:
        return None
    exit_price = float(exit_row["close"])
    costs = calculate_trade_costs(
        entry_price=position.entry_price,
        exit_price=exit_price,
        volume=position.size,
        contract_size=position.contract_size,
        direction=position.direction,
        rate=rate,
        absolute_slippage=absolute_slippage,
    )
    holding_minutes = (exit_time - position.entry_time).total_seconds() / 60.0
    mfe, mae = compute_mfe_mae(
        execution_df=execution_df,
        direction=position.direction,
        entry_time=position.entry_time,
        exit_time=exit_time,
        entry_price=position.entry_price,
        volume=position.size,
        contract_size=position.contract_size,
    )
    r_multiple = None
    if position.initial_risk is not None and position.initial_risk > 0:
        r_multiple = costs["net_pnl"] / position.initial_risk
    return {
        "policy_name": policy.policy_name,
        "base_policy_name": policy.base_policy_name,
        "atr_mult": policy.atr_mult,
        "timeframe": policy.timeframe,
        "direction": position.direction,
        "volume": position.size,
        "contract_size": position.contract_size,
        "entry_time": position.entry_time.isoformat(),
        "entry_price": position.entry_price,
        "exit_time": exit_time.isoformat(),
        "exit_price": exit_price,
        "exit_reason": exit_reason,
        "holding_minutes": float(holding_minutes),
        "gross_pnl": costs["gross_pnl"],
        "fee": costs["fee"],
        "slippage": costs["slippage"],
        "net_pnl": costs["net_pnl"],
        "no_cost_net_pnl": costs["no_cost_net_pnl"],
        "r_multiple": r_multiple,
        "mfe": mfe,
        "mae": mae,
        "size": position.size,
        "entry_signal_time": position.entry_signal_time.isoformat(),
        "exit_signal_time": exit_signal_time.isoformat(),
        "used_htf_bar_time": position.entry_signal_time.isoformat(),
        "entry_atr": position.entry_atr,
        "initial_risk": position.initial_risk,
    }


def determine_exit_reason(row: pd.Series, policy: PolicyRun, position: OpenPosition, htf_index: int) -> str | None:
    """Determine whether a position exits on this completed HTF bar."""

    close = numeric_row_value(row, "close")
    if close is None:
        return None
    if position.direction == "long":
        position.highest_close = max(position.highest_close, close)
    else:
        position.lowest_close = min(position.lowest_close, close)

    if policy.use_atr_trailing:
        hit, _trail = check_atr_trailing_stop(
            direction=position.direction,
            close=close,
            atr=numeric_row_value(row, "atr14"),
            highest_close=position.highest_close,
            lowest_close=position.lowest_close,
            atr_mult=policy.atr_mult,
            current_trailing_stop=position.trailing_stop,
        )
        position.trailing_stop = _trail
        if hit:
            return "atr_trailing_stop"
    if policy.use_donchian_exit and check_donchian_exit(row, policy, position.direction):
        return "donchian_exit"
    if policy.use_ema_exit and check_ema_exit(row, position.direction):
        return "ema_exit"
    if policy.max_hold_bars is not None and htf_index - position.entry_htf_index >= policy.max_hold_bars:
        return "max_hold_bars"
    return None


def simulate_policy(
    policy: PolicyRun,
    htf_df: pd.DataFrame,
    execution_df: pd.DataFrame,
    history_range: HistoryRange,
    *,
    capital: float = 5000.0,
    fixed_size: float = DEFAULT_FIXED_SIZE,
    sizing_mode: str = "fixed",
    risk_per_trade: float = DEFAULT_RISK_PER_TRADE,
    rate: float = 0.0005,
    absolute_slippage: float = 0.0,
    contract_size: float = 1.0,
) -> pd.DataFrame:
    """Simulate one policy with at most one position at a time."""

    end_exclusive = pd.Timestamp(history_range.end_exclusive)
    frame = htf_df.copy().sort_values("datetime", kind="stable").reset_index(drop=True)
    execution = filter_time_range(execution_df, history_range).sort_values("datetime", kind="stable").reset_index(drop=True)
    records: list[dict[str, Any]] = []
    position: OpenPosition | None = None

    for htf_index, row in frame.iterrows():
        signal_time = pd.Timestamp(row["datetime"])
        if signal_time < pd.Timestamp(history_range.start) or signal_time >= end_exclusive:
            continue

        if position is not None and signal_time > position.entry_signal_time:
            exit_reason = determine_exit_reason(row, policy, position, htf_index)
            if exit_reason:
                exit_row = find_next_execution_row(execution, signal_time, end_exclusive)
                if exit_row is None:
                    exit_row = final_execution_row(execution, end_exclusive)
                if exit_row is not None:
                    record = build_trade_record(
                        policy=policy,
                        position=position,
                        exit_row=exit_row,
                        exit_signal_time=signal_time,
                        exit_reason=exit_reason,
                        execution_df=execution,
                        rate=rate,
                        absolute_slippage=absolute_slippage,
                    )
                    if record is not None:
                        records.append(record)
                        position = None
                        continue

        if position is None:
            for direction in ["long", "short"]:
                if not policy_entry_signal(row, policy, direction):
                    continue
                entry_row = find_next_execution_row(execution, signal_time, end_exclusive)
                if entry_row is None:
                    break
                entry_atr = numeric_row_value(row, "atr14")
                size = resolve_trade_size(
                    sizing_mode=sizing_mode,
                    fixed_size=fixed_size,
                    capital=capital,
                    risk_per_trade=risk_per_trade,
                    atr=entry_atr,
                    atr_mult=policy.atr_mult,
                    contract_size=contract_size,
                )
                entry_price = float(entry_row["close"])
                initial_risk = None
                if entry_atr is not None and entry_atr > 0:
                    initial_risk = float(entry_atr * policy.atr_mult * size * contract_size)
                close = numeric_row_value(row, "close") or entry_price
                trailing_stop = None
                if entry_atr is not None and entry_atr > 0:
                    if direction == "long":
                        trailing_stop = float(close - policy.atr_mult * entry_atr)
                    else:
                        trailing_stop = float(close + policy.atr_mult * entry_atr)
                position = OpenPosition(
                    direction=direction,
                    entry_signal_time=signal_time,
                    entry_time=pd.Timestamp(entry_row["datetime"]),
                    entry_price=entry_price,
                    size=size,
                    contract_size=contract_size,
                    entry_atr=entry_atr,
                    initial_risk=initial_risk,
                    highest_close=close,
                    lowest_close=close,
                    trailing_stop=trailing_stop,
                    entry_htf_index=htf_index,
                )
                break

    if position is not None:
        exit_row = final_execution_row(execution, end_exclusive)
        if exit_row is not None:
            record = build_trade_record(
                policy=policy,
                position=position,
                exit_row=exit_row,
                exit_signal_time=pd.Timestamp(exit_row["datetime"]),
                exit_reason="end_of_sample",
                execution_df=execution,
                rate=rate,
                absolute_slippage=absolute_slippage,
            )
            if record is not None:
                records.append(record)

    if not records:
        return pd.DataFrame(columns=TRADE_COLUMNS)
    return pd.DataFrame(records, columns=TRADE_COLUMNS)


def simulate_all_policies(
    policies: list[PolicyRun],
    indicators: dict[str, pd.DataFrame],
    timeframes: dict[str, pd.DataFrame],
    history_range: HistoryRange,
    *,
    capital: float,
    fixed_size: float,
    sizing_mode: str,
    risk_per_trade: float,
    rate: float,
    absolute_slippage: float,
    contract_size: float,
) -> pd.DataFrame:
    """Simulate all policy runs."""

    frames_by_timeframe = {
        "1h": prepare_policy_frame(indicators, "1h", history_range),
        "4h": prepare_policy_frame(indicators, "4h", history_range),
    }
    trade_frames: list[pd.DataFrame] = []
    for policy in policies:
        trade_frames.append(
            simulate_policy(
                policy=policy,
                htf_df=frames_by_timeframe[policy.timeframe],
                execution_df=timeframes["15m"],
                history_range=history_range,
                capital=capital,
                fixed_size=fixed_size,
                sizing_mode=sizing_mode,
                risk_per_trade=risk_per_trade,
                rate=rate,
                absolute_slippage=absolute_slippage,
                contract_size=contract_size,
            )
        )
    if not trade_frames:
        return pd.DataFrame(columns=TRADE_COLUMNS)
    result = pd.concat(trade_frames, ignore_index=True)
    if result.empty:
        return pd.DataFrame(columns=TRADE_COLUMNS)
    return result.sort_values(["policy_name", "entry_time"], kind="stable").reset_index(drop=True)


def safe_sum(series: pd.Series) -> float:
    """Return numeric sum."""

    return float(pd.to_numeric(series, errors="coerce").fillna(0.0).sum())


def safe_mean(series: pd.Series) -> float | None:
    """Return numeric mean or None."""

    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return None
    return float(clean.mean())


def safe_median(series: pd.Series) -> float | None:
    """Return numeric median or None."""

    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return None
    return float(clean.median())


def max_drawdown_from_trades(trade_df: pd.DataFrame, capital: float) -> tuple[float, float]:
    """Compute trade-level max drawdown and percent."""

    if trade_df.empty:
        return 0.0, 0.0
    pnl = pd.to_numeric(trade_df["net_pnl"], errors="coerce").fillna(0.0)
    equity = capital + pnl.cumsum()
    equity_reset = equity.reset_index(drop=True)
    peak = pd.concat([pd.Series([capital], dtype=float), equity_reset], ignore_index=True).cummax().iloc[1:]
    peak = peak.reset_index(drop=True)
    drawdown = peak - equity_reset
    ddpercent = (drawdown / peak.replace(0, np.nan) * 100.0).replace([np.inf, -np.inf], np.nan)
    return float(drawdown.max()), float(ddpercent.max()) if ddpercent.notna().any() else 0.0


def summarize_trade_slice(trade_df: pd.DataFrame, capital: float) -> dict[str, Any]:
    """Summarize a trade slice."""

    if trade_df.empty:
        return {
            "trade_count": 0,
            "long_count": 0,
            "short_count": 0,
            "gross_pnl": 0.0,
            "net_pnl": 0.0,
            "no_cost_net_pnl": 0.0,
            "fee_total": 0.0,
            "slippage_total": 0.0,
            "win_rate": None,
            "profit_factor": None,
            "avg_win": None,
            "avg_loss": None,
            "avg_trade_net_pnl": None,
            "median_trade_net_pnl": None,
            "max_drawdown": 0.0,
            "max_ddpercent": 0.0,
            "return_drawdown_ratio": None,
            "sharpe_like": None,
            "avg_holding_minutes": None,
            "median_holding_minutes": None,
            "top_5pct_trade_pnl_contribution": None,
            "best_trade": None,
            "worst_trade": None,
            "cost_drag": 0.0,
        }

    working = trade_df.copy().sort_values("exit_time", kind="stable")
    net = pd.to_numeric(working["net_pnl"], errors="coerce").fillna(0.0)
    wins = net[net > 0]
    losses = net[net < 0]
    gross = safe_sum(working["gross_pnl"])
    no_cost = safe_sum(working["no_cost_net_pnl"])
    fee_total = safe_sum(working["fee"])
    slippage_total = safe_sum(working["slippage"])
    net_total = safe_sum(working["net_pnl"])
    max_dd, max_ddpercent = max_drawdown_from_trades(working, capital)
    profit_factor = None
    if not losses.empty:
        profit_factor = float(wins.sum() / abs(losses.sum())) if not wins.empty else 0.0
    elif not wins.empty:
        profit_factor = None
    return_drawdown_ratio = net_total / max_dd if max_dd > 0 else None
    clean_net = net.dropna()
    sharpe_like = None
    if len(clean_net.index) >= 2:
        std = float(clean_net.std(ddof=0))
        if std > 0:
            sharpe_like = float(clean_net.mean() / std * math.sqrt(len(clean_net.index)))
    top_contribution = None
    if len(clean_net.index) > 0 and net_total != 0:
        top_n = max(1, int(math.ceil(len(clean_net.index) * 0.05)))
        top_sum = float(clean_net.sort_values(ascending=False).head(top_n).sum())
        top_contribution = top_sum / net_total
    return {
        "trade_count": int(len(working.index)),
        "long_count": int((working["direction"] == "long").sum()),
        "short_count": int((working["direction"] == "short").sum()),
        "gross_pnl": gross,
        "net_pnl": net_total,
        "no_cost_net_pnl": no_cost,
        "fee_total": fee_total,
        "slippage_total": slippage_total,
        "win_rate": float((net > 0).mean()) if len(net.index) else None,
        "profit_factor": profit_factor,
        "avg_win": float(wins.mean()) if not wins.empty else None,
        "avg_loss": float(losses.mean()) if not losses.empty else None,
        "avg_trade_net_pnl": safe_mean(working["net_pnl"]),
        "median_trade_net_pnl": safe_median(working["net_pnl"]),
        "max_drawdown": max_dd,
        "max_ddpercent": max_ddpercent,
        "return_drawdown_ratio": return_drawdown_ratio,
        "sharpe_like": sharpe_like,
        "avg_holding_minutes": safe_mean(working["holding_minutes"]),
        "median_holding_minutes": safe_median(working["holding_minutes"]),
        "top_5pct_trade_pnl_contribution": top_contribution,
        "best_trade": float(net.max()) if len(net.index) else None,
        "worst_trade": float(net.min()) if len(net.index) else None,
        "cost_drag": no_cost - net_total,
    }


def build_policy_leaderboard(trades_df: pd.DataFrame, policies: list[PolicyRun], capital: float) -> pd.DataFrame:
    """Build one leaderboard row per concrete policy run."""

    rows: list[dict[str, Any]] = []
    for policy in policies:
        group = trades_df[trades_df["policy_name"] == policy.policy_name].copy() if not trades_df.empty else pd.DataFrame()
        row: dict[str, Any] = {
            "policy_name": policy.policy_name,
            "base_policy_name": policy.base_policy_name,
            "atr_mult": policy.atr_mult,
            "timeframe": policy.timeframe,
        }
        row.update(summarize_trade_slice(group, capital))
        rows.append(row)
    leaderboard = pd.DataFrame(rows)
    if leaderboard.empty:
        return pd.DataFrame(columns=LEADERBOARD_COLUMNS)
    return leaderboard[LEADERBOARD_COLUMNS].sort_values(
        ["net_pnl", "no_cost_net_pnl", "trade_count"],
        ascending=[False, False, False],
        kind="stable",
    ).reset_index(drop=True)


def build_group_summary(trades_df: pd.DataFrame, group_columns: list[str], capital: float) -> pd.DataFrame:
    """Build grouped trade summaries."""

    if trades_df.empty:
        return pd.DataFrame(columns=group_columns + ["trade_count"])
    rows: list[dict[str, Any]] = []
    for keys, group in trades_df.groupby(group_columns, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {column: key for column, key in zip(group_columns, keys)}
        row.update(summarize_trade_slice(group, capital))
        rows.append(row)
    return pd.DataFrame(rows)


def build_daily_pnl(trades_df: pd.DataFrame, capital: float) -> pd.DataFrame:
    """Build daily PnL by policy."""

    if trades_df.empty:
        return pd.DataFrame(columns=["policy_name", "date", "trade_count", "net_pnl", "no_cost_net_pnl"])
    working = trades_df.copy()
    working["exit_dt"] = pd.to_datetime(working["exit_time"])
    working["date"] = working["exit_dt"].dt.strftime("%Y-%m-%d")
    grouped = working.groupby(["policy_name", "date"], dropna=False).agg(
        trade_count=("net_pnl", "size"),
        gross_pnl=("gross_pnl", "sum"),
        net_pnl=("net_pnl", "sum"),
        no_cost_net_pnl=("no_cost_net_pnl", "sum"),
        fee_total=("fee", "sum"),
        slippage_total=("slippage", "sum"),
    )
    result = grouped.reset_index().sort_values(["policy_name", "date"], kind="stable")
    result["cumulative_net_pnl"] = result.groupby("policy_name")["net_pnl"].cumsum()
    result["equity"] = capital + result["cumulative_net_pnl"]
    return result.reset_index(drop=True)


def build_equity_curve(trades_df: pd.DataFrame, capital: float) -> pd.DataFrame:
    """Build per-policy trade-level equity curves."""

    if trades_df.empty:
        return pd.DataFrame(columns=["policy_name", "time", "net_pnl", "equity", "drawdown", "ddpercent"])
    rows: list[dict[str, Any]] = []
    for policy_name, group in trades_df.groupby("policy_name", dropna=False):
        working = group.copy().sort_values("exit_time", kind="stable").reset_index(drop=True)
        net = pd.to_numeric(working["net_pnl"], errors="coerce").fillna(0.0)
        no_cost = pd.to_numeric(working["no_cost_net_pnl"], errors="coerce").fillna(0.0)
        equity = capital + net.cumsum()
        no_cost_equity = capital + no_cost.cumsum()
        peak = pd.concat([pd.Series([capital], dtype=float), equity], ignore_index=True).cummax().iloc[1:]
        peak = peak.reset_index(drop=True)
        equity = equity.reset_index(drop=True)
        no_cost_equity = no_cost_equity.reset_index(drop=True)
        drawdown = peak - equity
        ddpercent = drawdown / peak.replace(0, np.nan) * 100.0
        rows.append(
            {
                "policy_name": policy_name,
                "time": working.iloc[0]["entry_time"],
                "net_pnl": 0.0,
                "no_cost_net_pnl": 0.0,
                "cumulative_net_pnl": 0.0,
                "cumulative_no_cost_net_pnl": 0.0,
                "equity": float(capital),
                "no_cost_equity": float(capital),
                "drawdown": 0.0,
                "ddpercent": 0.0,
            }
        )
        for index, trade in working.iterrows():
            rows.append(
                {
                    "policy_name": policy_name,
                    "time": trade["exit_time"],
                    "net_pnl": float(net.iloc[index]),
                    "no_cost_net_pnl": float(no_cost.iloc[index]),
                    "cumulative_net_pnl": float(equity.iloc[index] - capital),
                    "cumulative_no_cost_net_pnl": float(no_cost_equity.iloc[index] - capital),
                    "equity": float(equity.iloc[index]),
                    "no_cost_equity": float(no_cost_equity.iloc[index]),
                    "drawdown": float(drawdown.iloc[index]),
                    "ddpercent": float(ddpercent.iloc[index]) if pd.notna(ddpercent.iloc[index]) else 0.0,
                }
            )
    return pd.DataFrame(rows)


def data_quality_for_frame(df: pd.DataFrame, timeframe_name: str, expected_minutes: int) -> dict[str, Any]:
    """Build a gap summary for one timeframe."""

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

    frames = {name: filter_time_range(frame, history_range) for name, frame in timeframes.items()}
    return {
        "requested_start": history_range.start.isoformat(),
        "requested_end_exclusive": history_range.end_exclusive.isoformat(),
        "closed_bar_policy": (
            "Resampled bars are left-closed and timestamped at the final included 1m close. "
            "A 15m/1h/4h bar is kept only when every constituent 1m timestamp exists."
        ),
        "timeframes": {
            name: data_quality_for_frame(frames[name], name, minutes)
            for name, minutes in TIMEFRAME_MINUTES.items()
            if name in frames
        },
    }


def count_incomplete_resampled_bars(bars_1m: pd.DataFrame, resampled_bars: pd.DataFrame, minutes: int) -> int:
    """Count resampled bars whose constituent 1m bars are not all present."""

    if bars_1m.empty or resampled_bars.empty:
        return 0
    one_minute_times = pd.to_datetime(bars_1m["datetime"]).dropna()
    one_minute_ns = {int(timestamp.value) for timestamp in one_minute_times}
    one_minute = pd.Timedelta(minutes=1)
    count = 0
    for bar_time in pd.to_datetime(resampled_bars["datetime"]).dropna():
        start = pd.Timestamp(bar_time) - pd.Timedelta(minutes=minutes - 1)
        complete = True
        for offset in range(minutes):
            if int((start + offset * one_minute).value) not in one_minute_ns:
                complete = False
                break
        if not complete:
            count += 1
    return int(count)


def channel_current_bar_use_audit(
    indicators: dict[str, pd.DataFrame],
    history_range: HistoryRange,
    windows_by_timeframe: dict[str, set[int]],
) -> dict[str, Any]:
    """Audit whether stored Donchian channels look current-bar inclusive."""

    total_current_bar_uses = 0
    total_previous_mismatches = 0
    details: dict[str, dict[str, int]] = {}
    for timeframe, windows in windows_by_timeframe.items():
        frame = indicators.get(timeframe, pd.DataFrame()).copy()
        if not frame.empty:
            frame = frame.sort_values("datetime", kind="stable").reset_index(drop=True)
        in_range = pd.Series(False, index=frame.index)
        if not frame.empty:
            dt = pd.to_datetime(frame["datetime"])
            in_range = (dt >= pd.Timestamp(history_range.start)) & (dt < pd.Timestamp(history_range.end_exclusive))
        timeframe_current_uses = 0
        timeframe_mismatches = 0
        if not frame.empty:
            high = pd.to_numeric(frame["high"], errors="coerce")
            low = pd.to_numeric(frame["low"], errors="coerce")
            for window in sorted(windows):
                for side, source, reducer in [
                    ("high", high, "max"),
                    ("low", low, "min"),
                ]:
                    column = f"donchian_{side}_{window}_prev"
                    if column not in frame.columns:
                        continue
                    stored = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
                    rolling = source.rolling(window, min_periods=window)
                    current = getattr(rolling, reducer)().to_numpy(dtype=float)
                    previous = getattr(rolling, reducer)().shift(1).to_numpy(dtype=float)
                    stored_finite = np.isfinite(stored)
                    previous_finite = np.isfinite(previous)
                    current_finite = np.isfinite(current)
                    both_previous = stored_finite & previous_finite
                    mismatch = (stored_finite != previous_finite) | (
                        both_previous & ~np.isclose(stored, previous, rtol=1e-12, atol=1e-12)
                    )
                    current_use = (
                        stored_finite
                        & current_finite
                        & previous_finite
                        & np.isclose(stored, current, rtol=1e-12, atol=1e-12)
                        & ~np.isclose(stored, previous, rtol=1e-12, atol=1e-12)
                    )
                    in_range_array = in_range.to_numpy(dtype=bool)
                    timeframe_mismatches += int((mismatch & in_range_array).sum())
                    timeframe_current_uses += int((current_use & in_range_array).sum())
        details[timeframe] = {
            "current_bar_use_count": timeframe_current_uses,
            "previous_channel_mismatch_count": timeframe_mismatches,
        }
        total_current_bar_uses += timeframe_current_uses
        total_previous_mismatches += timeframe_mismatches
    return {
        "current_bar_use_count": int(total_current_bar_uses),
        "previous_channel_mismatch_count": int(total_previous_mismatches),
        "details": details,
    }


def build_no_lookahead_checks(
    bars_1m: pd.DataFrame,
    timeframes: dict[str, pd.DataFrame],
    indicators: dict[str, pd.DataFrame],
    trades_df: pd.DataFrame,
    policies: list[PolicyRun],
    history_range: HistoryRange,
) -> dict[str, Any]:
    """Build automated no-lookahead audit counters."""

    filtered_timeframes = {name: filter_time_range(frame, history_range) for name, frame in timeframes.items()}
    incomplete_counts = {
        name: count_incomplete_resampled_bars(bars_1m, filtered_timeframes[name], minutes)
        for name, minutes in TIMEFRAME_MINUTES.items()
        if name != "1m" and name in filtered_timeframes
    }
    entry_windows: dict[str, set[int]] = {}
    exit_windows: dict[str, set[int]] = {}
    for policy in policies:
        if policy.entry_type in {"donchian", "vol_compression"} and policy.entry_window is not None:
            entry_windows.setdefault(policy.timeframe, set()).add(policy.entry_window)
        if policy.use_donchian_exit and policy.exit_window is not None:
            exit_windows.setdefault(policy.timeframe, set()).add(policy.exit_window)

    entry_channel_audit = channel_current_bar_use_audit(indicators, history_range, entry_windows)
    exit_channel_audit = channel_current_bar_use_audit(indicators, history_range, exit_windows)

    if trades_df.empty:
        entry_before_signal_count = 0
        used_htf_bar_after_entry_count = 0
        trade_exit_before_entry_count = 0
    else:
        entry_time = pd.to_datetime(trades_df["entry_time"])
        entry_signal_time = pd.to_datetime(trades_df["entry_signal_time"])
        exit_time = pd.to_datetime(trades_df["exit_time"])
        entry_before_signal_count = int((entry_time <= entry_signal_time).sum())
        used_htf_bar_after_entry_count = int((entry_signal_time > entry_time).sum())
        trade_exit_before_entry_count = int((exit_time <= entry_time).sum())

    return {
        "used_incomplete_htf_bar_count": int(incomplete_counts.get("1h", 0) + incomplete_counts.get("4h", 0)),
        "incomplete_resampled_bar_counts": incomplete_counts,
        "entry_channel_uses_current_bar_count": entry_channel_audit["current_bar_use_count"],
        "entry_channel_previous_mismatch_count": entry_channel_audit["previous_channel_mismatch_count"],
        "entry_channel_details": entry_channel_audit["details"],
        "exit_channel_uses_current_bar_count": exit_channel_audit["current_bar_use_count"],
        "exit_channel_previous_mismatch_count": exit_channel_audit["previous_channel_mismatch_count"],
        "exit_channel_details": exit_channel_audit["details"],
        "entry_before_signal_count": entry_before_signal_count,
        "used_htf_bar_after_entry_count": used_htf_bar_after_entry_count,
        "trade_exit_before_entry_count": trade_exit_before_entry_count,
    }


def build_audit_sample_trades(trades_df: pd.DataFrame) -> list[dict[str, Any]]:
    """Return the first 20 trades in the compact audit schema."""

    if trades_df.empty:
        return []
    fields = [
        "policy_name",
        "direction",
        "used_htf_bar_time",
        "entry_signal_time",
        "entry_time",
        "entry_price",
        "exit_time",
        "exit_price",
        "exit_reason",
        "volume",
        "contract_size",
        "gross_pnl",
        "fee",
        "slippage",
        "net_pnl",
        "r_multiple",
    ]
    records = dataframe_records(trades_df.sort_values(["entry_time", "policy_name"], kind="stable").head(20))
    samples: list[dict[str, Any]] = []
    for record in records:
        sample = {field: record.get(field) for field in fields}
        if sample.get("volume") is None:
            sample["volume"] = record.get("size")
        samples.append(sample)
    return samples


def build_trend_research_audit(
    vt_symbol: str,
    history_range: HistoryRange,
    policies: list[PolicyRun],
    trades_df: pd.DataFrame,
    timeframes: dict[str, pd.DataFrame],
    indicators: dict[str, pd.DataFrame],
    bars_1m: pd.DataFrame,
    *,
    rate: float,
    slippage_mode: str,
    slippage: float,
    absolute_slippage: float,
    contract_size: float,
) -> dict[str, Any]:
    """Build trend_research_audit.json payload."""

    no_lookahead_checks = build_no_lookahead_checks(
        bars_1m=bars_1m,
        timeframes=timeframes,
        indicators=indicators,
        trades_df=trades_df,
        policies=policies,
        history_range=history_range,
    )
    return {
        "vt_symbol": vt_symbol,
        "start": history_range.start.isoformat(),
        "end_exclusive": history_range.end_exclusive.isoformat(),
        "donchian_previous_channel_policy": {
            "uses_previous_channel": True,
            "entry_channel_shift": "donchian_high/low_<window>_prev = rolling(high/low, window).max/min().shift(1)",
            "exit_channel_shift": "Donchian exits read the same *_prev fields, so the exit bar does not update its own threshold.",
            "entry_breakout_rule": "long close > previous Donchian high; short close < previous Donchian low; equality does not trigger.",
            "exit_breakout_rule": "long close < previous Donchian low; short close > previous Donchian high.",
        },
        "resample_closed_bar_policy": {
            "resample_timeframes": ["15m", "1h", "4h"],
            "timestamp_represents_close_time": True,
            "timestamp_policy": "HTF timestamps are the final included 1m bar time, e.g. 08:00-11:59 is timestamped 11:59.",
            "incomplete_htf_bars_filtered": True,
            "context_join_policy": "15m risk context is joined with pd.merge_asof(..., direction='backward').",
            "execution_policy": "Signals execute on the first completed 15m bar strictly after signal_time.",
        },
        "pnl_formula": {
            "volume_definition": "volume is simulated contract quantity; size is kept as a backward-compatible alias.",
            "contract_size": contract_size,
            "long_gross_pnl": "(exit_price - entry_price) * volume * contract_size",
            "short_gross_pnl": "(entry_price - exit_price) * volume * contract_size",
            "r_multiple": "cost-aware net_pnl / (entry ATR * atr_mult * volume * contract_size)",
        },
        "trailing_stop_policy": {
            "long_trailing": "candidate = highest_close_since_entry - atr_mult * ATR; trailing_stop = max(previous_stop, candidate)",
            "short_trailing": "candidate = lowest_close_since_entry + atr_mult * ATR; trailing_stop = min(previous_stop, candidate)",
            "only_moves_in_favorable_direction": True,
        },
        "cost_policy": {
            "fee": "fee = rate * (entry_price * volume * contract_size + exit_price * volume * contract_size)",
            "slippage_mode": slippage_mode,
            "slippage_input": slippage,
            "absolute_slippage": absolute_slippage,
            "slippage": "simplified additive round-trip cost = absolute_slippage * volume * contract_size * 2",
            "fill_price_note": (
                "entry_price and exit_price remain the raw next-15m close approximation; slippage is reported "
                "as a separate non-negative cost equivalent to moving both fills against the trade direction."
            ),
            "no_cost_cost_aware_split": "no_cost_net_pnl equals gross_pnl; net_pnl subtracts fee and slippage.",
        },
        "drawdown_policy": {
            "equity_initial_capital_included_as_peak": True,
            "daily_pnl_date": "exit_time date",
            "max_drawdown_sign_convention": "positive magnitude, computed as equity_peak - equity_trough",
            "max_ddpercent": "positive magnitude percent = max_drawdown / peak * 100",
        },
        "compare_policy": {
            "stable_candidate_requires": [
                "train no_cost_net_pnl > 0",
                "validation no_cost_net_pnl > 0",
                "oos no_cost_net_pnl > 0",
                f"each split trade_count >= {COMPARE_MIN_TRADE_COUNT}",
                "OOS cost-aware net_pnl >= 0, or low-frequency cost-drag exception is explicitly marked",
                f"each split max_ddpercent <= {COMPARE_MAX_DDPERCENT_THRESHOLD}",
            ],
            "min_trade_count": COMPARE_MIN_TRADE_COUNT,
            "max_ddpercent_threshold": COMPARE_MAX_DDPERCENT_THRESHOLD,
            "low_frequency_trade_count": COMPARE_LOW_FREQUENCY_TRADE_COUNT,
            "low_frequency_cost_exception": (
                "OOS no-cost must be positive, OOS net must be slightly negative, trade_count must be low, "
                f"cost_drag must be positive, and abs(oos_net_pnl) <= "
                f"{COMPARE_MAX_EXPLAINABLE_OOS_COST_LOSS_TO_NO_COST} * oos_no_cost_net_pnl."
            ),
        },
        "no_lookahead_checks": no_lookahead_checks,
        "sample_trades": build_audit_sample_trades(trades_df),
    }


def build_diagnostic_answers(
    split: str,
    leaderboard_df: pd.DataFrame,
    output_dir: Path,
) -> dict[str, Any]:
    """Build report diagnostics against trend-following acceptance criteria."""

    if leaderboard_df.empty:
        return {
            "trend_following_v2_failed": split == "oos",
            "notes": "leaderboard 为空",
        }
    no_cost_positive = leaderboard_df[pd.to_numeric(leaderboard_df["no_cost_net_pnl"], errors="coerce") > 0]
    cost_positive = leaderboard_df[pd.to_numeric(leaderboard_df["net_pnl"], errors="coerce") > 0]
    concentrated = leaderboard_df[
        pd.to_numeric(leaderboard_df["top_5pct_trade_pnl_contribution"], errors="coerce").abs() >= 0.8
    ]
    high_dd = leaderboard_df[pd.to_numeric(leaderboard_df["max_ddpercent"], errors="coerce") > 30.0]
    oos_all_negative = False
    if split == "oos":
        no_cost = pd.to_numeric(leaderboard_df["no_cost_net_pnl"], errors="coerce").fillna(0.0)
        cost = pd.to_numeric(leaderboard_df["net_pnl"], errors="coerce").fillna(0.0)
        oos_all_negative = bool((no_cost < 0).all() and (cost < 0).all())
    return {
        "no_cost_positive_policy_count": int(len(no_cost_positive.index)),
        "cost_aware_positive_policy_count": int(len(cost_positive.index)),
        "no_cost_positive_policies": no_cost_positive["policy_name"].head(20).tolist(),
        "cost_aware_positive_policies": cost_positive["policy_name"].head(20).tolist(),
        "oos_all_policies_negative": oos_all_negative,
        "trend_following_v2_failed": bool(oos_all_negative),
        "trade_count_range": {
            "min": int(pd.to_numeric(leaderboard_df["trade_count"], errors="coerce").min()),
            "max": int(pd.to_numeric(leaderboard_df["trade_count"], errors="coerce").max()),
        },
        "concentrated_profit_policies": concentrated["policy_name"].head(20).tolist(),
        "high_drawdown_policies_over_30pct": high_dd["policy_name"].head(20).tolist(),
        "cross_split_stability_note": (
            "运行 make compare-trend-v2 后，用 trend_compare_report.md 判定 train/validation/oos 稳定性。"
            if not (output_dir.parent / "train" / "trend_policy_leaderboard.csv").exists()
            else "同级 split 文件已存在；建议运行 make compare-trend-v2 生成正式跨样本结论。"
        ),
    }


def format_number(value: Any, digits: int = 6) -> str:
    """Format a number for Markdown."""

    number = finite_or_none(value)
    if number is None:
        return "N/A"
    return f"{number:.{digits}f}"


def format_leaderboard_lines(leaderboard_df: pd.DataFrame, limit: int = 12) -> str:
    """Format top leaderboard rows."""

    if leaderboard_df.empty:
        return "- 无"
    lines = []
    working = leaderboard_df.copy().sort_values(["net_pnl", "no_cost_net_pnl"], ascending=[False, False])
    for row in working.head(limit).to_dict(orient="records"):
        lines.append(
            f"- {row.get('policy_name')}: trades={row.get('trade_count')}, "
            f"no_cost={format_number(row.get('no_cost_net_pnl'), 4)}, "
            f"net={format_number(row.get('net_pnl'), 4)}, "
            f"max_dd%={format_number(row.get('max_ddpercent'), 2)}, "
            f"top5pct_contrib={format_number(row.get('top_5pct_trade_pnl_contribution'), 3)}"
        )
    return "\n".join(lines)


def render_policy_definitions() -> str:
    """Render policy definitions for the Markdown report."""

    lines = []
    for name in BASE_POLICY_ORDER:
        lines.append(f"- `{name}`: {POLICY_DESCRIPTIONS[name]}")
    return "\n".join(lines)


def render_markdown(summary: dict[str, Any], leaderboard_df: pd.DataFrame) -> str:
    """Render trend_report.md."""

    answers = summary.get("diagnostic_answers") or {}
    warnings = summary.get("warnings") or []
    warning_lines = "\n".join(f"- {item}" for item in warnings) if warnings else "- 无"
    return (
        "# Trend Following V2 研究报告\n\n"
        "## 核心结论\n"
        f"- split={summary.get('split')}, trade_count={summary.get('trade_count')}, policy_runs={summary.get('policy_run_count')}\n"
        f"- no_cost_positive_policy_count={answers.get('no_cost_positive_policy_count')}\n"
        f"- cost_aware_positive_policy_count={answers.get('cost_aware_positive_policy_count')}\n"
        f"- trend_following_v2_failed={str(bool(answers.get('trend_following_v2_failed'))).lower()}\n"
        "- 该研究未计入 perpetual funding fee，成本后结论仍需在进入 Strategy V2 前复核 funding 敏感性。\n\n"
        "## Policy Leaderboard\n"
        f"{format_leaderboard_lines(leaderboard_df)}\n\n"
        "## 趋势跟踪判定\n"
        f"1. no-cost 是否为正：{answers.get('no_cost_positive_policy_count')} 个 policy run 为正。\n"
        f"2. cost-aware 是否仍为正：{answers.get('cost_aware_positive_policy_count')} 个 policy run 为正。\n"
        f"3. OOS 是否为正：仅在 split=oos 或 compare 报告中做正式判断；当前 split={summary.get('split')}。\n"
        f"4. 交易次数是否显著低于旧 1m 策略：当前 trade_count_range={answers.get('trade_count_range')}；旧策略基准未作为输入读取。\n"
        f"5. 是否靠极少数交易贡献收益：concentrated_profit_policies={answers.get('concentrated_profit_policies')}。\n"
        f"6. 最大回撤是否可接受：high_drawdown_policies_over_30pct={answers.get('high_drawdown_policies_over_30pct')}。\n"
        f"7. train/validation/oos 是否稳定：{answers.get('cross_split_stability_note')}\n"
        "8. no-cost 为正但 cost 为负时，说明成本拖累仍不可接受，需要优先看 cost_drag。\n"
        "9. train 正但 validation/oos 负时，按过拟合处理。\n"
        f"10. 若所有 OOS no-cost 和 cost-aware 都为负，trend_following_v2_failed=true；当前值={str(bool(answers.get('trend_following_v2_failed'))).lower()}。\n\n"
        "## Policy 定义\n"
        f"{render_policy_definitions()}\n\n"
        "## 执行与成本假设\n"
        f"- sizing_mode={summary.get('sizing_mode')}, fixed_size={summary.get('fixed_size')}, capital={summary.get('capital')}\n"
        f"- contract_size={summary.get('contract_size')}；fixed_size/volume 按合约张数解释，PnL/fee/slippage 均乘以 contract_size。\n"
        f"- rate={summary.get('rate')}, slippage_mode={summary.get('slippage_mode')}, absolute_slippage={summary.get('absolute_slippage')}\n"
        "- 滑点采用独立成本扣减口径：entry/exit price 保留 15m close 近似，slippage 单独记录为双边不利成交成本。\n"
        "- HTF 信号在 1h/4h bar 收盘后才可见，下一根 15m bar close 作为执行价格近似。\n\n"
        "## 输出文件\n"
        "- trend_policy_summary.json\n"
        "- trend_policy_leaderboard.csv\n"
        "- trend_trades.csv\n"
        "- trend_daily_pnl.csv\n"
        "- trend_equity_curve.csv\n"
        "- trend_policy_by_side.csv\n"
        "- trend_policy_by_month.csv\n"
        "- trend_report.md\n"
        "- data_quality.json\n"
        "- trend_research_audit.json\n\n"
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
    summary: dict[str, Any],
    leaderboard_df: pd.DataFrame,
    trades_df: pd.DataFrame,
    daily_pnl_df: pd.DataFrame,
    equity_curve_df: pd.DataFrame,
    by_side_df: pd.DataFrame,
    by_month_df: pd.DataFrame,
    markdown: str,
    data_quality: dict[str, Any],
    audit: dict[str, Any],
) -> None:
    """Write all Trend Following V2 research artifacts."""

    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "trend_policy_summary.json", summary)
    write_dataframe(output_dir / "trend_policy_leaderboard.csv", leaderboard_df)
    write_dataframe(output_dir / "trend_trades.csv", trades_df)
    write_dataframe(output_dir / "trend_daily_pnl.csv", daily_pnl_df)
    write_dataframe(output_dir / "trend_equity_curve.csv", equity_curve_df)
    write_dataframe(output_dir / "trend_policy_by_side.csv", by_side_df)
    write_dataframe(output_dir / "trend_policy_by_month.csv", by_month_df)
    (output_dir / "trend_report.md").write_text(markdown, encoding="utf-8")
    write_json(output_dir / "data_quality.json", data_quality)
    write_json(output_dir / "trend_research_audit.json", audit)


def build_summary(
    vt_symbol: str,
    split: str,
    history_range: HistoryRange,
    output_dir: Path,
    policies: list[PolicyRun],
    trades_df: pd.DataFrame,
    leaderboard_df: pd.DataFrame,
    data_quality: dict[str, Any],
    warnings: list[str],
    *,
    capital: float,
    rate: float,
    slippage_mode: str,
    slippage: float,
    absolute_slippage: float,
    contract_size: float,
    sizing_mode: str,
    fixed_size: float,
    risk_per_trade: float,
) -> dict[str, Any]:
    """Build trend_policy_summary.json payload."""

    answers = build_diagnostic_answers(split, leaderboard_df, output_dir)
    return {
        "vt_symbol": vt_symbol,
        "split": split,
        "start": history_range.start.isoformat(),
        "end_exclusive": history_range.end_exclusive.isoformat(),
        "timezone": history_range.timezone_name,
        "output_dir": str(output_dir),
        "capital": capital,
        "rate": rate,
        "slippage_mode": slippage_mode,
        "slippage_input": slippage,
        "absolute_slippage": absolute_slippage,
        "contract_size": contract_size,
        "audit_path": str(output_dir / "trend_research_audit.json"),
        "sizing_mode": sizing_mode,
        "fixed_size": fixed_size,
        "risk_per_trade": risk_per_trade,
        "policy_run_count": len(policies),
        "base_policy_count": len(BASE_POLICY_ORDER),
        "trade_count": int(len(trades_df.index)),
        "warnings": warnings,
        "data_quality": data_quality,
        "policy_definitions": [
            {
                "base_policy_name": policy_name,
                "description": POLICY_DESCRIPTIONS[policy_name],
            }
            for policy_name in BASE_POLICY_ORDER
        ],
        "leaderboard": dataframe_records(leaderboard_df),
        "diagnostic_answers": answers,
    }


def run_research(
    vt_symbol: str,
    split: str,
    history_range: HistoryRange,
    output_dir: Path,
    timezone_name: str,
    *,
    capital: float = 5000.0,
    rate: float = 0.0005,
    slippage_mode: str = "ticks",
    slippage: float = 2.0,
    fixed_size: float = DEFAULT_FIXED_SIZE,
    sizing_mode: str = "fixed",
    risk_per_trade: float = DEFAULT_RISK_PER_TRADE,
    atr_mults: list[float] | None = None,
    max_runs: int | None = None,
    data_check_strict: bool = False,
    logger: logging.Logger | None = None,
    bars_from_db: bool = True,
    bars_df: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Run the complete Trend Following V2 research workflow."""

    ZoneInfo(timezone_name)
    logger = logger or logging.getLogger("research_trend_following_v2")
    warnings: list[str] = ["未计入 perpetual funding fee"]
    absolute_slippage = resolve_absolute_slippage(vt_symbol, slippage_mode, slippage)
    contract_size = resolve_contract_size(vt_symbol)
    policies = build_policy_runs(atr_mults or parse_number_list(DEFAULT_ATR_MULTS, "--atr-mults"), max_runs)

    if bars_df is None:
        if not bars_from_db:
            raise TrendFollowingV2Error("--no-bars-from-db 已设置，但未提供 bars_df")
        bars_1m = load_bars_from_db(vt_symbol, history_range, timezone_name, logger)
    else:
        bars_1m = normalize_1m_bars(bars_df, timezone_name)

    if bars_1m.empty:
        warnings.append("没有可用 1m bars")

    timeframes = build_timeframes(bars_1m)
    data_quality = build_data_quality(timeframes, history_range)
    one_minute_quality = (data_quality.get("timeframes") or {}).get("1m") or {}
    if data_check_strict:
        if int(one_minute_quality.get("bar_count") or 0) <= 0:
            raise TrendFollowingV2Error("--data-check-strict: requested 1m bars 为空")
        if int(one_minute_quality.get("missing_count") or 0) > 0:
            raise TrendFollowingV2Error("--data-check-strict: requested 1m bars 存在缺口")

    indicators = build_indicator_frames(timeframes)
    trades_df = simulate_all_policies(
        policies=policies,
        indicators=indicators,
        timeframes=timeframes,
        history_range=history_range,
        capital=capital,
        fixed_size=fixed_size,
        sizing_mode=sizing_mode,
        risk_per_trade=risk_per_trade,
        rate=rate,
        absolute_slippage=absolute_slippage,
        contract_size=contract_size,
    )
    leaderboard_df = build_policy_leaderboard(trades_df, policies, capital)
    daily_pnl_df = build_daily_pnl(trades_df, capital)
    equity_curve_df = build_equity_curve(trades_df, capital)
    by_side_df = build_group_summary(trades_df, ["policy_name", "direction"], capital)
    if trades_df.empty:
        by_month_df = pd.DataFrame(columns=["policy_name", "month", "trade_count"])
    else:
        month_df = trades_df.copy()
        month_df["month"] = pd.to_datetime(month_df["exit_time"]).dt.strftime("%Y-%m")
        by_month_df = build_group_summary(month_df, ["policy_name", "month"], capital)

    summary = build_summary(
        vt_symbol=vt_symbol,
        split=split,
        history_range=history_range,
        output_dir=output_dir,
        policies=policies,
        trades_df=trades_df,
        leaderboard_df=leaderboard_df,
        data_quality=data_quality,
        warnings=warnings,
        capital=capital,
        rate=rate,
        slippage_mode=slippage_mode,
        slippage=slippage,
        absolute_slippage=absolute_slippage,
        contract_size=contract_size,
        sizing_mode=sizing_mode,
        fixed_size=fixed_size,
        risk_per_trade=risk_per_trade,
    )
    audit = build_trend_research_audit(
        vt_symbol=vt_symbol,
        history_range=history_range,
        policies=policies,
        trades_df=trades_df,
        timeframes=timeframes,
        indicators=indicators,
        bars_1m=bars_1m,
        rate=rate,
        slippage_mode=slippage_mode,
        slippage=slippage,
        absolute_slippage=absolute_slippage,
        contract_size=contract_size,
    )
    markdown = render_markdown(summary, leaderboard_df)
    write_outputs(
        output_dir=output_dir,
        summary=summary,
        leaderboard_df=leaderboard_df,
        trades_df=trades_df,
        daily_pnl_df=daily_pnl_df,
        equity_curve_df=equity_curve_df,
        by_side_df=by_side_df,
        by_month_df=by_month_df,
        markdown=markdown,
        data_quality=data_quality,
        audit=audit,
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""

    args = parse_args(argv)
    ensure_headless_runtime()
    logger = setup_logging("research_trend_following_v2", verbose=args.verbose)
    try:
        history_range = resolve_split_range(args.split, args.start, args.end, args.timezone)
        output_dir = resolve_path(
            args.output_dir,
            PROJECT_ROOT / "reports" / "research" / "trend_following_v2" / args.split,
        )
        atr_mults = parse_number_list(args.atr_mults, "--atr-mults")
        summary = run_research(
            vt_symbol=args.vt_symbol,
            split=args.split,
            history_range=history_range,
            output_dir=output_dir,
            timezone_name=args.timezone,
            capital=args.capital,
            rate=args.rate,
            slippage_mode=args.slippage_mode,
            slippage=args.slippage,
            fixed_size=args.fixed_size,
            sizing_mode=args.sizing_mode,
            risk_per_trade=args.risk_per_trade,
            atr_mults=atr_mults,
            max_runs=args.max_runs,
            data_check_strict=args.data_check_strict,
            logger=logger,
            bars_from_db=args.bars_from_db,
        )
        print_json_block(
            "Trend Following V2 summary:",
            {
                "output_dir": output_dir,
                "split": args.split,
                "trade_count": summary.get("trade_count"),
                "trend_following_v2_failed": (summary.get("diagnostic_answers") or {}).get("trend_following_v2_failed"),
                "warnings": summary.get("warnings"),
            },
        )
        if args.json:
            print(json.dumps(to_jsonable(summary), ensure_ascii=False, indent=2))
        return 0
    except TrendFollowingV2Error as exc:
        log_event(logger, logging.ERROR, "trend_following_v2.error", str(exc))
        return 2
    except Exception:
        logger.exception(
            "Unexpected error during Trend Following V2 research",
            extra={"event": "trend_following_v2.unexpected_error"},
        )
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
