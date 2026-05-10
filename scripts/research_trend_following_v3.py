#!/usr/bin/env python3
"""Portfolio-level multi-symbol Trend Following V3 research."""

from __future__ import annotations

import argparse
import json
import logging
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from analyze_signal_outcomes import configure_sqlite_settings, resolve_exchange, split_vt_symbol
from common_runtime import PROJECT_ROOT, ensure_headless_runtime, log_event, print_json_block, setup_logging, to_jsonable
from history_time_utils import DEFAULT_TIMEZONE, HistoryRange, expected_bar_count, iter_expected_datetimes, parse_history_range
from history_utils import build_instrument_config_path, get_database_timezone
from research_trend_following_v2 import (
    check_atr_trailing_stop,
    dataframe_records,
    filter_time_range,
    finite_or_none,
    normalize_1m_bars,
    resample_ohlcv,
    rolling_percentile,
    safe_mean,
    safe_median,
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
DEFAULT_SYMBOLS_ARG = ",".join(DEFAULT_SYMBOLS)
DEFAULT_FIXED_SIZE = 0.01
DEFAULT_CAPITAL = 5000.0
DEFAULT_RATE = 0.0005
DEFAULT_SLIPPAGE = 2.0
DEFAULT_MAX_SYMBOL_WEIGHT = 0.35
DEFAULT_MAX_PORTFOLIO_POSITIONS = 3
ROLLING_PERCENTILE_WINDOW = 240
V3_WARMUP_DAYS = 420
TIMEFRAME_MINUTES = {"1m": 1, "1h": 60, "4h": 240, "1d": 1440}
DONCHIAN_WINDOWS = [10, 20, 30, 55, 100]
SPLIT_RANGES = {
    "train": ("2025-01-01", "2025-09-30"),
    "validation": ("2025-10-01", "2025-12-31"),
    "oos": ("2026-01-01", "2026-03-31"),
    "full": ("2025-01-01", "2026-03-31"),
}
EXTENDED_SPLIT_RANGES = {
    "train_ext": ("2023-01-01", "2024-06-30"),
    "validation_ext": ("2024-07-01", "2025-06-30"),
    "oos_ext": ("2025-07-01", "2026-03-31"),
    "full_ext": ("2023-01-01", "2026-03-31"),
}
SPLIT_SCHEMES = {
    "default": SPLIT_RANGES,
    "extended": EXTENDED_SPLIT_RANGES,
}
ALL_SPLIT_NAMES = sorted({split for ranges in SPLIT_SCHEMES.values() for split in ranges})
FUNDING_WARNING = "OKX perpetual funding fee is not included in Trend Following V3 research outputs."

TRADE_COLUMNS = [
    "policy_name",
    "symbol",
    "direction",
    "entry_time",
    "entry_price",
    "exit_time",
    "exit_price",
    "exit_reason",
    "holding_minutes",
    "volume",
    "contract_size",
    "gross_pnl",
    "fee",
    "slippage",
    "net_pnl",
    "no_cost_pnl",
    "no_cost_net_pnl",
    "r_multiple",
    "mfe",
    "mae",
    "timeframe",
    "entry_source",
    "entry_signal_time",
    "exit_signal_time",
    "entry_atr",
    "initial_risk",
    "turnover",
]

REJECTED_SIGNAL_COLUMNS = [
    "policy_name",
    "symbol",
    "direction",
    "signal_time",
    "source_policy_name",
    "reason",
    "signal_strength",
    "open_position_count",
    "max_portfolio_positions",
]

LEADERBOARD_COLUMNS = [
    "policy_name",
    "symbol_count",
    "trade_count",
    "long_count",
    "short_count",
    "active_symbol_count",
    "no_cost_net_pnl",
    "net_pnl",
    "fee_total",
    "slippage_total",
    "cost_drag",
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
    "largest_symbol_pnl_share",
    "portfolio_turnover",
    "max_concurrent_positions",
]


class TrendFollowingV3Error(Exception):
    """Raised when Trend Following V3 research cannot continue."""


@dataclass(frozen=True, slots=True)
class PolicyRun:
    """One Trend Following V3 portfolio policy."""

    policy_name: str
    timeframe: str
    entry_type: str
    entry_window: int | None
    exit_window: int | None
    atr_mult: float
    use_donchian_exit: bool = False
    use_ema_exit: bool = False
    use_risk_filters: bool = False
    description: str = ""


@dataclass(slots=True)
class SymbolContext:
    """All research data and metadata for one symbol."""

    vt_symbol: str
    bars_1m: pd.DataFrame
    timeframes: dict[str, pd.DataFrame]
    indicators: dict[str, pd.DataFrame]
    contract_size: float
    pricetick: float
    absolute_slippage: float
    metadata: dict[str, Any]
    coverage: dict[str, Any]


@dataclass(slots=True)
class PortfolioPosition:
    """Mutable portfolio position state for one policy and symbol."""

    policy_name: str
    symbol: str
    direction: str
    timeframe: str
    entry_source: str
    entry_signal_time: pd.Timestamp
    entry_time: pd.Timestamp
    entry_price: float
    volume: float
    contract_size: float
    entry_atr: float | None
    initial_risk: float | None
    highest_close: float
    lowest_close: float
    trailing_stop: float | None
    atr_mult: float


@dataclass(slots=True)
class SimulationBundle:
    """Trades plus signal-level audit records from one or more policy simulations."""

    trades: pd.DataFrame
    rejected_signals: pd.DataFrame


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments."""

    parser = argparse.ArgumentParser(description="Research Trend Following V3 multi-symbol portfolio policies.")
    parser.add_argument("--symbols", default=DEFAULT_SYMBOLS_ARG, help=f"Comma or space separated symbols. Default: {DEFAULT_SYMBOLS_ARG}.")
    parser.add_argument("--split-scheme", choices=sorted(SPLIT_SCHEMES), default="default")
    parser.add_argument("--split", choices=ALL_SPLIT_NAMES, default="train")
    parser.add_argument("--split-name", choices=ALL_SPLIT_NAMES, help="Explicit split name; overrides --split.")
    parser.add_argument("--start")
    parser.add_argument("--end")
    parser.add_argument("--timezone", default=DEFAULT_TIMEZONE)
    parser.add_argument("--interval", default="1m", help="Only 1m is currently supported. Default: 1m.")
    parser.add_argument("--output-dir", help="Default: reports/research/trend_following_v3/<split>.")
    parser.add_argument("--capital", type=float, default=DEFAULT_CAPITAL)
    parser.add_argument("--capital-mode", choices=["portfolio_fixed", "per_symbol_fixed"], default="portfolio_fixed")
    parser.add_argument("--position-sizing", choices=["fixed_contract", "volatility_target"], default="fixed_contract")
    parser.add_argument("--fixed-size", type=float, default=DEFAULT_FIXED_SIZE)
    parser.add_argument("--rate", type=float, default=DEFAULT_RATE)
    parser.add_argument("--slippage-mode", choices=["ticks", "absolute"], default="ticks")
    parser.add_argument("--slippage", type=float, default=DEFAULT_SLIPPAGE)
    parser.add_argument("--max-symbol-weight", type=float, default=DEFAULT_MAX_SYMBOL_WEIGHT)
    parser.add_argument("--max-portfolio-positions", type=int, default=DEFAULT_MAX_PORTFOLIO_POSITIONS)
    parser.add_argument("--data-check-strict", action="store_true")
    parser.add_argument("--max-runs", type=int)
    parser.add_argument("--bars-from-db", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def parse_symbols(raw_value: str | list[str]) -> list[str]:
    """Parse comma or whitespace separated vt_symbols."""

    if isinstance(raw_value, list):
        tokens = raw_value
    else:
        tokens = str(raw_value or "").replace(",", " ").split()
    symbols = [token.strip() for token in tokens if token.strip()]
    if not symbols:
        raise TrendFollowingV3Error("--symbols 不能为空")
    return list(dict.fromkeys(symbols))


def resolve_path(path_arg: str | Path | None, default_path: Path) -> Path:
    """Resolve a path relative to project root."""

    path = Path(path_arg) if path_arg else default_path
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def resolve_split_name(split_scheme: str, split_arg: str, split_name_arg: str | None = None) -> str:
    """Resolve and validate a split name within one split scheme."""

    if split_scheme not in SPLIT_SCHEMES:
        raise TrendFollowingV3Error(f"unknown split scheme: {split_scheme}")
    split = split_name_arg or split_arg
    scheme_ranges = SPLIT_SCHEMES[split_scheme]
    if split not in scheme_ranges:
        allowed = ", ".join(sorted(scheme_ranges))
        raise TrendFollowingV3Error(f"--split {split!r} 不属于 --split-scheme {split_scheme!r}; allowed: {allowed}")
    return split


def resolve_split_range(
    split: str,
    start_arg: str | None,
    end_arg: str | None,
    timezone_name: str,
    split_scheme: str = "default",
) -> HistoryRange:
    """Resolve split preset and optional explicit bounds."""

    split = resolve_split_name(split_scheme, split)
    default_start, default_end = SPLIT_SCHEMES[split_scheme][split]
    try:
        return parse_history_range(start_arg or default_start, end_arg or default_end, pd.Timedelta(minutes=1).to_pytimedelta(), timezone_name)
    except ValueError as exc:
        raise TrendFollowingV3Error(str(exc)) from exc


def read_instrument_meta(vt_symbol: str, overrides: dict[str, dict[str, Any]] | None = None) -> dict[str, Any]:
    """Read local instrument metadata, with optional test overrides."""

    if overrides and vt_symbol in overrides:
        return dict(overrides[vt_symbol])
    path = build_instrument_config_path(vt_symbol)
    if not path.exists():
        raise TrendFollowingV3Error(f"missing instrument metadata: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise TrendFollowingV3Error(f"读取 instrument metadata 失败: {path} | {exc!r}") from exc


def metadata_complete(meta: dict[str, Any]) -> bool:
    """Return whether canonical contract metadata is usable."""

    contract_size = finite_or_none(meta.get("size", meta.get("contract_size")))
    pricetick = finite_or_none(meta.get("pricetick"))
    min_volume = finite_or_none(meta.get("min_volume"))
    return bool(
        meta.get("okx_inst_id")
        and meta.get("product")
        and contract_size is not None
        and contract_size > 0
        and pricetick is not None
        and pricetick > 0
        and min_volume is not None
        and min_volume > 0
        and not bool(meta.get("needs_okx_contract_metadata_refresh", False))
    )


def resolve_contract_size_from_meta(meta: dict[str, Any], vt_symbol: str) -> float:
    """Resolve contract size from metadata."""

    contract_size = finite_or_none(meta.get("size", meta.get("contract_size")))
    if contract_size is None or contract_size <= 0:
        raise TrendFollowingV3Error(f"{vt_symbol} 缺少有效 contract size")
    return float(contract_size)


def resolve_pricetick_from_meta(meta: dict[str, Any], vt_symbol: str) -> float:
    """Resolve price tick from metadata."""

    pricetick = finite_or_none(meta.get("pricetick"))
    if pricetick is None or pricetick <= 0:
        raise TrendFollowingV3Error(f"{vt_symbol} 缺少有效 pricetick")
    return float(pricetick)


def resolve_absolute_slippage_from_meta(meta: dict[str, Any], vt_symbol: str, slippage_mode: str, slippage: float) -> float:
    """Resolve per-side slippage in absolute price units."""

    if slippage < 0:
        raise TrendFollowingV3Error("--slippage 不能小于 0")
    if slippage_mode == "absolute":
        return float(slippage)
    return float(slippage) * resolve_pricetick_from_meta(meta, vt_symbol)


def bar_to_record(bar: Any) -> dict[str, Any]:
    """Convert vn.py BarData-like object into a plain record."""

    return {
        "datetime": getattr(bar, "datetime", None),
        "open": getattr(bar, "open_price", None),
        "high": getattr(bar, "high_price", None),
        "low": getattr(bar, "low_price", None),
        "close": getattr(bar, "close_price", None),
        "volume": getattr(bar, "volume", None),
    }


def load_bars_from_db(
    vt_symbol: str,
    history_range: HistoryRange,
    timezone_name: str,
    logger: logging.Logger,
    *,
    warmup_days: int = V3_WARMUP_DAYS,
) -> pd.DataFrame:
    """Load one symbol's 1m bars from vn.py sqlite with warmup."""

    symbol, exchange_value = split_vt_symbol(vt_symbol)
    exchange = resolve_exchange(exchange_value)

    from vnpy.trader.constant import Interval
    from vnpy.trader.database import get_database

    configure_sqlite_settings(logger)
    db_tz = get_database_timezone()
    query_start = (history_range.start - pd.Timedelta(days=warmup_days).to_pytimedelta()).astimezone(db_tz).replace(tzinfo=None)
    query_end = history_range.end_exclusive.astimezone(db_tz).replace(tzinfo=None)
    bars = get_database().load_bar_data(symbol, exchange, Interval.MINUTE, query_start, query_end)
    records = [bar_to_record(bar) for bar in bars]
    if not records:
        return pd.DataFrame(columns=["datetime", "open", "high", "low", "close", "volume"])
    return normalize_1m_bars(pd.DataFrame(records), timezone_name)


def compress_missing_ranges(missing_times: list[pd.Timestamp]) -> list[dict[str, Any]]:
    """Compress missing 1m timestamps into contiguous ranges."""

    if not missing_times:
        return []
    ranges: list[dict[str, Any]] = []
    start = missing_times[0]
    previous = missing_times[0]
    count = 1
    for current in missing_times[1:]:
        if current - previous == pd.Timedelta(minutes=1):
            previous = current
            count += 1
            continue
        ranges.append({"start": start.isoformat(), "end": previous.isoformat(), "missing_count": count})
        start = current
        previous = current
        count = 1
    ranges.append({"start": start.isoformat(), "end": previous.isoformat(), "missing_count": count})
    return ranges


def build_1m_coverage_summary(bars_1m: pd.DataFrame, history_range: HistoryRange) -> dict[str, Any]:
    """Build strict target split coverage summary for one symbol."""

    expected_times = [pd.Timestamp(value) for value in iter_expected_datetimes(history_range)]
    expected_ns = {int(value.value) for value in expected_times}
    target = filter_time_range(bars_1m, history_range)
    if target.empty:
        actual_times = pd.Series([], dtype="datetime64[ns]")
    else:
        actual_times = pd.to_datetime(target["datetime"]).dropna().drop_duplicates().sort_values(kind="stable")
    actual_ns = {int(pd.Timestamp(value).value) for value in actual_times}
    missing_times = [value for value in expected_times if int(value.value) not in actual_ns]
    missing_ranges = compress_missing_ranges(missing_times)
    return {
        "expected_count": int(expected_bar_count(history_range)),
        "total_count": int(len(actual_ns & expected_ns)),
        "missing_count": int(len(missing_times)),
        "gap_count": int(len(missing_ranges)),
        "first_dt": actual_times.iloc[0].isoformat() if len(actual_times.index) else None,
        "last_dt": actual_times.iloc[-1].isoformat() if len(actual_times.index) else None,
        "missing_ranges": missing_ranges[:20],
        "required_coverage_ready": bool(len(expected_times) > 0 and len(missing_times) == 0 and len(actual_ns & expected_ns) == len(expected_times)),
    }


def data_quality_for_timeframe(df: pd.DataFrame, timeframe: str, minutes: int, history_range: HistoryRange) -> dict[str, Any]:
    """Build simple resampled timeframe quality summary."""

    frame = filter_time_range(df, history_range)
    if frame.empty:
        return {"timeframe": timeframe, "bar_count": 0, "first_dt": None, "last_dt": None}
    times = pd.to_datetime(frame["datetime"]).dropna().sort_values(kind="stable")
    return {
        "timeframe": timeframe,
        "bar_count": int(len(times.index)),
        "first_dt": times.iloc[0].isoformat() if len(times.index) else None,
        "last_dt": times.iloc[-1].isoformat() if len(times.index) else None,
        "minutes": minutes,
    }


def build_timeframes(bars_1m: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Resample 1m bars into closed 1h/4h/1d bars."""

    return {
        "1m": bars_1m.copy(),
        "1h": resample_ohlcv(bars_1m, 60),
        "4h": resample_ohlcv(bars_1m, 240),
        "1d": resample_ohlcv(bars_1m, 1440),
    }


def add_donchian_metrics(df: pd.DataFrame, windows: list[int] | None = None) -> pd.DataFrame:
    """Add previous-bar Donchian channels and prior-only width percentiles."""

    result = df.copy()
    high = pd.to_numeric(result["high"], errors="coerce")
    low = pd.to_numeric(result["low"], errors="coerce")
    close = pd.to_numeric(result["close"], errors="coerce").replace(0, np.nan)
    for window in windows or DONCHIAN_WINDOWS:
        channel_high = high.rolling(window, min_periods=window).max().shift(1)
        channel_low = low.rolling(window, min_periods=window).min().shift(1)
        width = channel_high - channel_low
        result[f"donchian_high_{window}_prev"] = channel_high
        result[f"donchian_low_{window}_prev"] = channel_low
        result[f"donchian_mid_{window}_prev"] = (channel_high + channel_low) / 2.0
        result[f"donchian_width_{window}"] = width
        result[f"donchian_width_{window}_pct"] = width / close
        result[f"donchian_width_{window}_percentile"] = rolling_percentile(result[f"donchian_width_{window}_pct"], ROLLING_PERCENTILE_WINDOW)
        result[f"donchian_width_{window}_percentile_prev"] = result[f"donchian_width_{window}_percentile"].shift(1)
    return result


def compute_indicators(bars: pd.DataFrame, timeframe_minutes: int) -> pd.DataFrame:
    """Compute Trend V3 indicators for one timeframe."""

    df = bars.copy().sort_values("datetime", kind="stable").reset_index(drop=True)
    if df.empty:
        return df
    for column in ["open", "high", "low", "close", "volume"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    close = df["close"]
    volume = df["volume"]
    for span in [20, 50, 100, 200]:
        df[f"ema{span}"] = close.ewm(span=span, adjust=False, min_periods=1).mean()
        df[f"close_to_ema{span}_distance"] = close / df[f"ema{span}"].replace(0, np.nan) - 1.0
    df["atr14"] = true_range(df).rolling(14, min_periods=1).mean()
    df["atr_pct"] = df["atr14"] / close.replace(0, np.nan)
    df["atr_pct_percentile"] = rolling_percentile(df["atr_pct"], ROLLING_PERCENTILE_WINDOW)
    df["atr_pct_percentile_prev"] = df["atr_pct_percentile"].shift(1)
    df = add_donchian_metrics(df, DONCHIAN_WINDOWS)
    returns = close.pct_change()
    df["recent_return"] = returns
    for periods in [1, 3, 6, 12, 24]:
        df[f"recent_return_{periods}bar"] = close / close.shift(periods) - 1.0
    volatility_window = max(5, min(30, int(round(1440 / max(timeframe_minutes, 1)))))
    df["realized_volatility"] = returns.rolling(volatility_window, min_periods=min(5, volatility_window)).std(ddof=0)
    df["recent_volatility"] = df["realized_volatility"]
    previous_volume = volume.shift(1)
    volume_mean = previous_volume.rolling(volatility_window, min_periods=min(5, volatility_window)).mean()
    volume_std = previous_volume.rolling(volatility_window, min_periods=min(5, volatility_window)).std(ddof=0)
    df["volume_zscore"] = (volume - volume_mean) / volume_std.replace(0, np.nan)
    bar_range = (df["high"] - df["low"]).replace(0, np.nan)
    df["body_ratio"] = (close - df["open"]).abs() / bar_range
    for column in ["recent_return", "realized_volatility", "recent_volatility", "volume_zscore", "body_ratio"]:
        df[f"{column}_percentile"] = rolling_percentile(df[column], ROLLING_PERCENTILE_WINDOW)
        df[f"{column}_percentile_prev"] = df[f"{column}_percentile"].shift(1)
    df["directional_recent_return_percentile_long"] = rolling_percentile(df["recent_return"], ROLLING_PERCENTILE_WINDOW)
    df["directional_recent_return_percentile_short"] = rolling_percentile(-df["recent_return"], ROLLING_PERCENTILE_WINDOW)
    return df


def build_indicator_frames(timeframes: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Compute indicators for all Trend V3 timeframes."""

    return {
        "1h": compute_indicators(timeframes["1h"], 60),
        "4h": compute_indicators(timeframes["4h"], 240),
        "1d": compute_indicators(timeframes["1d"], 1440),
    }


def risk_context_columns() -> list[str]:
    """Return 1h risk columns joined into higher timeframe rows."""

    return [
        "recent_volatility_percentile",
        "directional_recent_return_percentile_long",
        "directional_recent_return_percentile_short",
        "volume_zscore_percentile",
        "body_ratio_percentile",
    ]


def prepare_policy_frame(context: SymbolContext, timeframe: str, history_range: HistoryRange) -> pd.DataFrame:
    """Attach latest completed 1h risk context to a policy timeframe."""

    htf = context.indicators[timeframe].copy().sort_values("datetime", kind="stable").reset_index(drop=True)
    if htf.empty:
        return htf
    risk_1h = context.indicators["1h"][["datetime"] + risk_context_columns()].copy().sort_values("datetime", kind="stable")
    risk_1h = risk_1h.rename(columns={column: f"risk_1h_{column}" for column in risk_context_columns()})
    merged = pd.merge_asof(htf, risk_1h, on="datetime", direction="backward")
    return filter_time_range(merged, history_range)


def build_symbol_context(
    vt_symbol: str,
    history_range: HistoryRange,
    timezone_name: str,
    logger: logging.Logger,
    *,
    slippage_mode: str,
    slippage: float,
    bars_from_db: bool = True,
    bars_df: pd.DataFrame | None = None,
    instrument_meta_by_symbol: dict[str, dict[str, Any]] | None = None,
) -> SymbolContext:
    """Build one symbol context from sqlite or a provided DataFrame."""

    meta = read_instrument_meta(vt_symbol, instrument_meta_by_symbol)
    contract_size = resolve_contract_size_from_meta(meta, vt_symbol)
    pricetick = resolve_pricetick_from_meta(meta, vt_symbol)
    absolute_slippage = resolve_absolute_slippage_from_meta(meta, vt_symbol, slippage_mode, slippage)
    if bars_df is None:
        if not bars_from_db:
            raise TrendFollowingV3Error("--no-bars-from-db 已设置，但未提供 bars_df")
        bars_1m = load_bars_from_db(vt_symbol, history_range, timezone_name, logger)
    else:
        working = bars_df.copy()
        if "vt_symbol" in working.columns:
            working = working[working["vt_symbol"] == vt_symbol].copy()
        bars_1m = normalize_1m_bars(working, timezone_name)
    coverage = build_1m_coverage_summary(bars_1m, history_range)
    coverage["metadata_complete"] = metadata_complete(meta)
    timeframes = build_timeframes(bars_1m)
    indicators = build_indicator_frames(timeframes)
    coverage["timeframes"] = {
        name: data_quality_for_timeframe(frame, name, TIMEFRAME_MINUTES[name], history_range)
        for name, frame in timeframes.items()
        if name in TIMEFRAME_MINUTES
    }
    return SymbolContext(
        vt_symbol=vt_symbol,
        bars_1m=bars_1m,
        timeframes=timeframes,
        indicators=indicators,
        contract_size=contract_size,
        pricetick=pricetick,
        absolute_slippage=absolute_slippage,
        metadata=meta,
        coverage=coverage,
    )


def build_symbol_contexts(
    symbols: list[str],
    history_range: HistoryRange,
    timezone_name: str,
    logger: logging.Logger,
    *,
    slippage_mode: str,
    slippage: float,
    data_check_strict: bool,
    bars_from_db: bool = True,
    bars_by_symbol: dict[str, pd.DataFrame] | None = None,
    instrument_meta_by_symbol: dict[str, dict[str, Any]] | None = None,
) -> tuple[dict[str, SymbolContext], dict[str, Any], list[str]]:
    """Load all symbol contexts with strict or warning-based coverage handling."""

    contexts: dict[str, SymbolContext] = {}
    data_quality: dict[str, Any] = {
        "requested_symbols": symbols,
        "ready_symbols": [],
        "skipped_symbols": [],
        "symbol_coverage": {},
        "funding_fee_warning": FUNDING_WARNING,
        "closed_bar_policy": "1h/4h/1d bars are left-closed and timestamped at the final included 1m close.",
    }
    warnings: list[str] = []
    for vt_symbol in symbols:
        bars_df = (bars_by_symbol or {}).get(vt_symbol)
        context = build_symbol_context(
            vt_symbol=vt_symbol,
            history_range=history_range,
            timezone_name=timezone_name,
            logger=logger,
            slippage_mode=slippage_mode,
            slippage=slippage,
            bars_from_db=bars_from_db,
            bars_df=bars_df,
            instrument_meta_by_symbol=instrument_meta_by_symbol,
        )
        coverage = dict(context.coverage)
        data_quality["symbol_coverage"][vt_symbol] = coverage
        complete = bool(coverage.get("required_coverage_ready")) and bool(coverage.get("metadata_complete"))
        if not complete:
            reason = (
                f"{vt_symbol} incomplete: metadata_complete={coverage.get('metadata_complete')}, "
                f"expected_count={coverage.get('expected_count')}, total_count={coverage.get('total_count')}, "
                f"missing_count={coverage.get('missing_count')}, gap_count={coverage.get('gap_count')}"
            )
            if data_check_strict:
                raise TrendFollowingV3Error(f"--data-check-strict: {reason}")
            warnings.append(reason)
            data_quality["skipped_symbols"].append(vt_symbol)
            continue
        contexts[vt_symbol] = context
        data_quality["ready_symbols"].append(vt_symbol)
    data_quality["ready_symbol_count"] = int(len(contexts))
    data_quality["all_required_symbols_ready"] = bool(len(contexts) == len(symbols))
    if not contexts:
        raise TrendFollowingV3Error("没有任何 symbol 通过数据质量检查")
    return contexts, data_quality, warnings


def build_policy_runs(max_runs: int | None = None) -> list[PolicyRun]:
    """Build the fixed Trend V3 policy set."""

    policies = [
        PolicyRun("v3_4h_donchian_20_10_atr4", "4h", "donchian", 20, 10, 4.0, use_donchian_exit=True, description="4h Donchian 20/10 with ATR4 trailing stop."),
        PolicyRun("v3_4h_donchian_55_20_atr4", "4h", "donchian", 55, 20, 4.0, use_donchian_exit=True, description="4h Donchian 55/20 with ATR4 trailing stop."),
        PolicyRun("v3_4h_donchian_100_30_atr5", "4h", "donchian", 100, 30, 5.0, use_donchian_exit=True, description="4h Donchian 100/30 with ATR5 trailing stop."),
        PolicyRun("v3_1d_donchian_20_10_atr4", "1d", "donchian", 20, 10, 4.0, use_donchian_exit=True, description="1d Donchian 20/10 with ATR4 trailing stop."),
        PolicyRun("v3_1d_donchian_55_20_atr5", "1d", "donchian", 55, 20, 5.0, use_donchian_exit=True, description="1d Donchian 55/20 with ATR5 trailing stop."),
        PolicyRun("v3_4h_ema_50_200_atr4", "4h", "ema", None, None, 4.0, use_ema_exit=True, description="4h EMA50/EMA200 trend with ATR4 trailing stop."),
        PolicyRun("v3_1d_ema_50_200_atr5", "1d", "ema", None, None, 5.0, use_ema_exit=True, description="1d EMA50/EMA200 trend with ATR5 trailing stop."),
        PolicyRun("v3_4h_vol_compression_donchian_breakout", "4h", "vol_compression", 20, 10, 4.0, use_donchian_exit=True, description="4h low ATR/width percentile compression followed by Donchian breakout."),
        PolicyRun("v3_4h_donchian_55_with_risk_filters", "4h", "donchian", 55, 20, 4.0, use_donchian_exit=True, use_risk_filters=True, description="4h Donchian 55/20 with Signal Lab risk filters."),
        PolicyRun("v3_ensemble_core", "mixed", "ensemble", None, None, 4.0, description="Portfolio ensemble of 4h Donchian 55/20, 1d Donchian 20/10, and 4h EMA50/200."),
    ]
    if max_runs is not None:
        if max_runs <= 0:
            raise TrendFollowingV3Error("--max-runs 必须为正数")
        policies = policies[:max_runs]
    return policies


def ensemble_components() -> list[PolicyRun]:
    """Return concrete signal components used by v3_ensemble_core."""

    return [
        PolicyRun("ensemble_4h_donchian_55_20", "4h", "donchian", 55, 20, 4.0, use_donchian_exit=True),
        PolicyRun("ensemble_1d_donchian_20_10", "1d", "donchian", 20, 10, 4.0, use_donchian_exit=True),
        PolicyRun("ensemble_4h_ema_50_200", "4h", "ema", None, None, 4.0, use_ema_exit=True),
    ]


def policy_components(policy: PolicyRun) -> list[PolicyRun]:
    """Return components for a policy."""

    if policy.entry_type == "ensemble":
        return ensemble_components()
    return [policy]


def numeric_row_value(row: pd.Series, column: str) -> float | None:
    """Return a finite numeric row value."""

    return finite_or_none(row.get(column))


def risk_filters_pass(row: pd.Series, direction: str) -> bool:
    """Apply Trend V3 Signal Lab risk filters."""

    directional = f"risk_1h_directional_recent_return_percentile_{direction}"
    checks = [
        "atr_pct_percentile",
        "risk_1h_recent_volatility_percentile",
        directional,
        "risk_1h_volume_zscore_percentile",
        "risk_1h_body_ratio_percentile",
    ]
    for column in checks:
        value = numeric_row_value(row, column)
        if value is None or value > 0.8:
            return False
    return True


def risk_overheated(row: pd.Series, direction: str) -> bool:
    """Return whether risk percentiles should lower signal score."""

    directional = f"risk_1h_directional_recent_return_percentile_{direction}"
    checks = [
        "atr_pct_percentile",
        "risk_1h_recent_volatility_percentile",
        directional,
        "risk_1h_volume_zscore_percentile",
        "risk_1h_body_ratio_percentile",
    ]
    for column in checks:
        value = numeric_row_value(row, column)
        if value is not None and value > 0.8:
            return True
    return False


def policy_entry_signal(row: pd.Series, policy: PolicyRun, direction: str) -> bool:
    """Return whether one completed HTF row triggers entry."""

    if direction not in {"long", "short"}:
        return False
    if policy.use_risk_filters and not risk_filters_pass(row, direction):
        return False
    close = numeric_row_value(row, "close")
    if close is None:
        return False
    if policy.entry_type == "donchian":
        high = numeric_row_value(row, f"donchian_high_{policy.entry_window}_prev")
        low = numeric_row_value(row, f"donchian_low_{policy.entry_window}_prev")
        if direction == "long":
            return bool(high is not None and close > high)
        return bool(low is not None and close < low)
    if policy.entry_type == "ema":
        ema50 = numeric_row_value(row, "ema50")
        ema200 = numeric_row_value(row, "ema200")
        if ema50 is None or ema200 is None:
            return False
        if direction == "long":
            return bool(ema50 > ema200 and close > ema50)
        return bool(ema50 < ema200 and close < ema50)
    if policy.entry_type == "vol_compression":
        atr_percentile = numeric_row_value(row, "atr_pct_percentile_prev")
        width_percentile = numeric_row_value(row, "donchian_width_20_percentile_prev")
        if atr_percentile is None or width_percentile is None or atr_percentile > 0.4 or width_percentile > 0.4:
            return False
        high = numeric_row_value(row, "donchian_high_20_prev")
        low = numeric_row_value(row, "donchian_low_20_prev")
        if direction == "long":
            return bool(high is not None and close > high)
        return bool(low is not None and close < low)
    return False


def signal_strength(row: pd.Series, policy: PolicyRun, direction: str) -> float:
    """Calculate portfolio signal ranking strength."""

    close = numeric_row_value(row, "close")
    atr = numeric_row_value(row, "atr14")
    if close is None or atr is None or atr <= 0:
        return 0.0
    strength = 0.0
    if policy.entry_type in {"donchian", "vol_compression"}:
        window = policy.entry_window or 20
        high = numeric_row_value(row, f"donchian_high_{window}_prev")
        low = numeric_row_value(row, f"donchian_low_{window}_prev")
        if direction == "long" and high is not None:
            strength = (close - high) / atr
        elif direction == "short" and low is not None:
            strength = (low - close) / atr
    elif policy.entry_type == "ema":
        ema50 = numeric_row_value(row, "ema50")
        ema200 = numeric_row_value(row, "ema200")
        if ema50 is not None and ema200 is not None:
            strength = abs(ema50 - ema200) / atr
    if risk_overheated(row, direction):
        strength *= 0.5
    return float(max(strength, 0.0))


def check_donchian_exit(row: pd.Series, exit_window: int | None, direction: str) -> bool:
    """Return whether a previous-channel Donchian exit is hit."""

    if exit_window is None:
        return False
    close = numeric_row_value(row, "close")
    if close is None:
        return False
    if direction == "long":
        low = numeric_row_value(row, f"donchian_low_{exit_window}_prev")
        return bool(low is not None and close < low)
    high = numeric_row_value(row, f"donchian_high_{exit_window}_prev")
    return bool(high is not None and close > high)


def check_ema_exit(row: pd.Series, direction: str) -> bool:
    """Return whether close loses EMA50 in the opposite direction."""

    close = numeric_row_value(row, "close")
    ema50 = numeric_row_value(row, "ema50")
    if close is None or ema50 is None:
        return False
    if direction == "long":
        return bool(close < ema50)
    return bool(close > ema50)


def find_next_execution_row(execution_df: pd.DataFrame, signal_time: pd.Timestamp, end_exclusive: pd.Timestamp) -> pd.Series | None:
    """Find first 1m execution row strictly after the completed signal bar."""

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
    """Return final available 1m row inside the research split."""

    if execution_df.empty:
        return None
    times = pd.to_datetime(execution_df["datetime"])
    eligible = execution_df.loc[times < end_exclusive]
    if eligible.empty:
        return None
    return eligible.iloc[-1]


def calculate_trade_costs(
    entry_price: float,
    exit_price: float,
    volume: float,
    contract_size: float,
    direction: str,
    rate: float,
    absolute_slippage: float,
) -> dict[str, float]:
    """Calculate gross and cost-aware trade PnL."""

    if rate < 0:
        raise TrendFollowingV3Error("--rate 不能小于 0")
    sign = 1.0 if direction == "long" else -1.0
    notional_units = volume * contract_size
    gross_pnl = (exit_price - entry_price) * sign * notional_units
    turnover = abs(entry_price * notional_units) + abs(exit_price * notional_units)
    fee = rate * turnover
    slippage_cost = absolute_slippage * notional_units * 2.0
    net_pnl = gross_pnl - fee - slippage_cost
    return {
        "gross_pnl": float(gross_pnl),
        "no_cost_pnl": float(gross_pnl),
        "no_cost_net_pnl": float(gross_pnl),
        "fee": float(fee),
        "slippage": float(slippage_cost),
        "net_pnl": float(net_pnl),
        "turnover": float(turnover),
    }


def compute_mfe_mae_1m(
    execution_df: pd.DataFrame,
    direction: str,
    entry_time: pd.Timestamp,
    exit_time: pd.Timestamp,
    entry_price: float,
    volume: float,
    contract_size: float,
) -> tuple[float, float]:
    """Compute MFE/MAE from 1m highs/lows."""

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


def resolve_position_volume(
    *,
    position_sizing: str,
    fixed_size: float,
    portfolio_capital: float,
    max_symbol_weight: float,
    max_portfolio_positions: int,
    price: float,
    atr: float | None,
    atr_mult: float,
    contract_size: float,
) -> float:
    """Resolve contract volume and reject entries that exceed the symbol weight cap."""

    if fixed_size <= 0:
        raise TrendFollowingV3Error("--fixed-size 必须为正数")
    if portfolio_capital <= 0:
        raise TrendFollowingV3Error("--capital 必须为正数")
    if max_symbol_weight <= 0:
        raise TrendFollowingV3Error("--max-symbol-weight 必须为正数")
    if max_portfolio_positions <= 0:
        raise TrendFollowingV3Error("--max-portfolio-positions 必须为正数")
    raw_volume = float(fixed_size)
    if position_sizing == "volatility_target":
        if atr is not None and atr > 0 and atr_mult > 0 and contract_size > 0:
            risk_budget = portfolio_capital * 0.005 / max_portfolio_positions
            raw_volume = risk_budget / (atr * atr_mult * contract_size)
        raw_volume = min(raw_volume, fixed_size * 10.0)
    max_notional = portfolio_capital * max_symbol_weight
    desired_notional = raw_volume * price * contract_size
    if desired_notional > max_notional:
        return 0.0
    return float(max(0.0, raw_volume))


def build_policy_events(policy: PolicyRun, contexts: dict[str, SymbolContext], history_range: HistoryRange) -> pd.DataFrame:
    """Build all HTF bar events for one portfolio policy."""

    rows: list[dict[str, Any]] = []
    for component in policy_components(policy):
        for vt_symbol, context in contexts.items():
            frame = prepare_policy_frame(context, component.timeframe, history_range)
            if frame.empty:
                continue
            for _, source_row in frame.iterrows():
                row = source_row.to_dict()
                long_entry = policy_entry_signal(source_row, component, "long")
                short_entry = policy_entry_signal(source_row, component, "short")
                row.update(
                    {
                        "event_time": pd.Timestamp(source_row["datetime"]),
                        "policy_name": policy.policy_name,
                        "symbol": vt_symbol,
                        "source_policy_name": component.policy_name,
                        "timeframe": component.timeframe,
                        "entry_type": component.entry_type,
                        "entry_window": component.entry_window,
                        "exit_window": component.exit_window,
                        "atr_mult": component.atr_mult,
                        "use_donchian_exit": component.use_donchian_exit,
                        "use_ema_exit": component.use_ema_exit,
                        "entry_long": long_entry,
                        "entry_short": short_entry,
                        "strength_long": signal_strength(source_row, component, "long") if long_entry else 0.0,
                        "strength_short": signal_strength(source_row, component, "short") if short_entry else 0.0,
                    }
                )
                rows.append(row)
    if not rows:
        return pd.DataFrame()
    events = pd.DataFrame(rows)
    events["event_time"] = pd.to_datetime(events["event_time"])
    return events.sort_values(["event_time", "symbol", "source_policy_name"], kind="stable").reset_index(drop=True)


def determine_exit_reason(event: pd.Series, position: PortfolioPosition) -> str | None:
    """Determine whether one event exits an open position."""

    close = numeric_row_value(event, "close")
    if close is None:
        return None
    if position.direction == "long":
        position.highest_close = max(position.highest_close, close)
    else:
        position.lowest_close = min(position.lowest_close, close)
    hit, trail = check_atr_trailing_stop(
        direction=position.direction,
        close=close,
        atr=numeric_row_value(event, "atr14"),
        highest_close=position.highest_close,
        lowest_close=position.lowest_close,
        atr_mult=finite_or_none(event.get("atr_mult")) or position.atr_mult,
        current_trailing_stop=position.trailing_stop,
    )
    position.trailing_stop = trail
    source = str(event.get("source_policy_name") or event.get("policy_name"))
    if hit:
        return f"{source}:atr_trailing_stop"
    if bool(event.get("use_donchian_exit")) and check_donchian_exit(event, int(event["exit_window"]) if pd.notna(event.get("exit_window")) else None, position.direction):
        return f"{source}:donchian_exit"
    if bool(event.get("use_ema_exit")) and check_ema_exit(event, position.direction):
        return f"{source}:ema_exit"
    return None


def build_trade_record(
    position: PortfolioPosition,
    exit_event: pd.Series,
    exit_row: pd.Series,
    exit_reason: str,
    context: SymbolContext,
    rate: float,
) -> dict[str, Any] | None:
    """Build one closed trade record."""

    exit_time = pd.Timestamp(exit_row["datetime"])
    if exit_time <= position.entry_time:
        return None
    exit_price = float(exit_row["close"])
    costs = calculate_trade_costs(
        entry_price=position.entry_price,
        exit_price=exit_price,
        volume=position.volume,
        contract_size=position.contract_size,
        direction=position.direction,
        rate=rate,
        absolute_slippage=context.absolute_slippage,
    )
    holding_minutes = (exit_time - position.entry_time).total_seconds() / 60.0
    mfe, mae = compute_mfe_mae_1m(
        execution_df=context.bars_1m,
        direction=position.direction,
        entry_time=position.entry_time,
        exit_time=exit_time,
        entry_price=position.entry_price,
        volume=position.volume,
        contract_size=position.contract_size,
    )
    r_multiple = None
    if position.initial_risk is not None and position.initial_risk > 0:
        r_multiple = costs["net_pnl"] / position.initial_risk
    return {
        "policy_name": position.policy_name,
        "symbol": position.symbol,
        "direction": position.direction,
        "entry_time": position.entry_time.isoformat(),
        "entry_price": position.entry_price,
        "exit_time": exit_time.isoformat(),
        "exit_price": exit_price,
        "exit_reason": exit_reason,
        "holding_minutes": float(holding_minutes),
        "volume": position.volume,
        "contract_size": position.contract_size,
        "gross_pnl": costs["gross_pnl"],
        "fee": costs["fee"],
        "slippage": costs["slippage"],
        "net_pnl": costs["net_pnl"],
        "no_cost_pnl": costs["no_cost_pnl"],
        "no_cost_net_pnl": costs["no_cost_net_pnl"],
        "r_multiple": r_multiple,
        "mfe": mfe,
        "mae": mae,
        "timeframe": position.timeframe,
        "entry_source": position.entry_source,
        "entry_signal_time": position.entry_signal_time.isoformat(),
        "exit_signal_time": pd.Timestamp(exit_event["event_time"]).isoformat(),
        "entry_atr": position.entry_atr,
        "initial_risk": position.initial_risk,
        "turnover": costs["turnover"],
    }


def merge_entry_candidates(group: pd.DataFrame) -> list[dict[str, Any]]:
    """Merge simultaneous same-symbol same-direction signals."""

    candidates: dict[tuple[str, str], dict[str, Any]] = {}
    for _, event in group.iterrows():
        for direction, flag_column, strength_column in [
            ("long", "entry_long", "strength_long"),
            ("short", "entry_short", "strength_short"),
        ]:
            if not bool(event.get(flag_column)):
                continue
            key = (str(event["symbol"]), direction)
            strength = finite_or_none(event.get(strength_column)) or 0.0
            existing = candidates.get(key)
            source_name = str(event.get("source_policy_name") or event.get("policy_name"))
            if existing is None or strength > existing["strength"]:
                duplicate_sources = []
                candidate_count = 1
                if existing is not None:
                    duplicate_sources = existing["sources"] + existing["duplicate_sources"]
                    candidate_count = int(existing["candidate_count"]) + 1
                candidates[key] = {
                    "symbol": str(event["symbol"]),
                    "direction": direction,
                    "strength": strength,
                    "event": event,
                    "sources": [source_name],
                    "duplicate_sources": duplicate_sources,
                    "candidate_count": candidate_count,
                }
            elif existing is not None and strength == existing["strength"]:
                existing["sources"].append(source_name)
                existing["candidate_count"] = int(existing["candidate_count"]) + 1
            elif existing is not None:
                existing["duplicate_sources"].append(source_name)
                existing["candidate_count"] = int(existing["candidate_count"]) + 1
    return sorted(candidates.values(), key=lambda item: item["strength"], reverse=True)


def rejected_signal_record(
    *,
    policy_name: str,
    symbol: str,
    direction: str,
    signal_time: pd.Timestamp,
    source_policy_name: str,
    reason: str,
    signal_strength_value: float,
    open_position_count: int,
    max_portfolio_positions: int,
) -> dict[str, Any]:
    """Build one rejected or merged signal audit record."""

    return {
        "policy_name": policy_name,
        "symbol": symbol,
        "direction": direction,
        "signal_time": pd.Timestamp(signal_time).isoformat(),
        "source_policy_name": source_policy_name,
        "reason": reason,
        "signal_strength": float(signal_strength_value),
        "open_position_count": int(open_position_count),
        "max_portfolio_positions": int(max_portfolio_positions),
    }


def simulate_portfolio_policy(
    policy: PolicyRun,
    contexts: dict[str, SymbolContext],
    history_range: HistoryRange,
    *,
    portfolio_capital: float,
    position_sizing: str,
    fixed_size: float,
    rate: float,
    max_symbol_weight: float,
    max_portfolio_positions: int,
) -> pd.DataFrame:
    """Simulate one policy at portfolio level across all symbols."""

    bundle = simulate_portfolio_policy_with_audit(
        policy=policy,
        contexts=contexts,
        history_range=history_range,
        portfolio_capital=portfolio_capital,
        position_sizing=position_sizing,
        fixed_size=fixed_size,
        rate=rate,
        max_symbol_weight=max_symbol_weight,
        max_portfolio_positions=max_portfolio_positions,
    )
    return bundle.trades


def simulate_portfolio_policy_with_audit(
    policy: PolicyRun,
    contexts: dict[str, SymbolContext],
    history_range: HistoryRange,
    *,
    portfolio_capital: float,
    position_sizing: str,
    fixed_size: float,
    rate: float,
    max_symbol_weight: float,
    max_portfolio_positions: int,
) -> SimulationBundle:
    """Simulate one policy and return trade plus rejected-signal audit records."""

    events = build_policy_events(policy, contexts, history_range)
    if events.empty:
        return SimulationBundle(
            trades=pd.DataFrame(columns=TRADE_COLUMNS),
            rejected_signals=pd.DataFrame(columns=REJECTED_SIGNAL_COLUMNS),
        )
    end_exclusive = pd.Timestamp(history_range.end_exclusive)
    open_positions: dict[str, PortfolioPosition] = {}
    records: list[dict[str, Any]] = []
    rejected_records: list[dict[str, Any]] = []

    for event_time, group in events.groupby("event_time", sort=True):
        event_timestamp = pd.Timestamp(event_time)

        for symbol, position in list(open_positions.items()):
            if event_timestamp <= position.entry_signal_time:
                continue
            symbol_events = group[group["symbol"] == symbol]
            if symbol_events.empty:
                continue
            context = contexts[symbol]
            for _, event in symbol_events.iterrows():
                reason = determine_exit_reason(event, position)
                if not reason:
                    continue
                exit_row = find_next_execution_row(context.bars_1m, event_timestamp, end_exclusive)
                if exit_row is None:
                    exit_row = final_execution_row(context.bars_1m, end_exclusive)
                if exit_row is not None:
                    record = build_trade_record(
                        position=position,
                        exit_event=event,
                        exit_row=exit_row,
                        exit_reason=reason,
                        context=context,
                        rate=rate,
                    )
                    if record is not None:
                        records.append(record)
                open_positions.pop(symbol, None)
                break

        capacity = max_portfolio_positions - len(open_positions)
        candidates = merge_entry_candidates(group)
        for candidate in candidates:
            for duplicate_source in candidate.get("duplicate_sources", []):
                rejected_records.append(
                    rejected_signal_record(
                        policy_name=policy.policy_name,
                        symbol=candidate["symbol"],
                        direction=candidate["direction"],
                        signal_time=event_timestamp,
                        source_policy_name=str(duplicate_source),
                        reason="duplicate_signal_merged",
                        signal_strength_value=candidate["strength"],
                        open_position_count=len(open_positions),
                        max_portfolio_positions=max_portfolio_positions,
                    )
                )

        if capacity <= 0:
            for candidate in candidates:
                if candidate["symbol"] not in open_positions:
                    rejected_records.append(
                        rejected_signal_record(
                            policy_name=policy.policy_name,
                            symbol=candidate["symbol"],
                            direction=candidate["direction"],
                            signal_time=event_timestamp,
                            source_policy_name="+".join(dict.fromkeys(candidate["sources"])),
                            reason="max_portfolio_positions",
                            signal_strength_value=candidate["strength"],
                            open_position_count=len(open_positions),
                            max_portfolio_positions=max_portfolio_positions,
                        )
                    )
            continue
        for candidate in candidates:
            if capacity <= 0:
                rejected_records.append(
                    rejected_signal_record(
                        policy_name=policy.policy_name,
                        symbol=candidate["symbol"],
                        direction=candidate["direction"],
                        signal_time=event_timestamp,
                        source_policy_name="+".join(dict.fromkeys(candidate["sources"])),
                        reason="max_portfolio_positions",
                        signal_strength_value=candidate["strength"],
                        open_position_count=len(open_positions),
                        max_portfolio_positions=max_portfolio_positions,
                    )
                )
                continue
            symbol = candidate["symbol"]
            if symbol in open_positions:
                rejected_records.append(
                    rejected_signal_record(
                        policy_name=policy.policy_name,
                        symbol=symbol,
                        direction=candidate["direction"],
                        signal_time=event_timestamp,
                        source_policy_name="+".join(dict.fromkeys(candidate["sources"])),
                        reason="duplicate_symbol_position",
                        signal_strength_value=candidate["strength"],
                        open_position_count=len(open_positions),
                        max_portfolio_positions=max_portfolio_positions,
                    )
                )
                continue
            context = contexts[symbol]
            event = candidate["event"]
            entry_row = find_next_execution_row(context.bars_1m, event_timestamp, end_exclusive)
            if entry_row is None:
                rejected_records.append(
                    rejected_signal_record(
                        policy_name=policy.policy_name,
                        symbol=symbol,
                        direction=candidate["direction"],
                        signal_time=event_timestamp,
                        source_policy_name="+".join(dict.fromkeys(candidate["sources"])),
                        reason="no_execution_row",
                        signal_strength_value=candidate["strength"],
                        open_position_count=len(open_positions),
                        max_portfolio_positions=max_portfolio_positions,
                    )
                )
                continue
            entry_price = float(entry_row["close"])
            entry_atr = numeric_row_value(event, "atr14")
            atr_mult = finite_or_none(event.get("atr_mult")) or policy.atr_mult
            volume = resolve_position_volume(
                position_sizing=position_sizing,
                fixed_size=fixed_size,
                portfolio_capital=portfolio_capital,
                max_symbol_weight=max_symbol_weight,
                max_portfolio_positions=max_portfolio_positions,
                price=entry_price,
                atr=entry_atr,
                atr_mult=atr_mult,
                contract_size=context.contract_size,
            )
            if volume <= 0:
                rejected_records.append(
                    rejected_signal_record(
                        policy_name=policy.policy_name,
                        symbol=symbol,
                        direction=candidate["direction"],
                        signal_time=event_timestamp,
                        source_policy_name="+".join(dict.fromkeys(candidate["sources"])),
                        reason="max_symbol_weight",
                        signal_strength_value=candidate["strength"],
                        open_position_count=len(open_positions),
                        max_portfolio_positions=max_portfolio_positions,
                    )
                )
                continue
            initial_risk = None
            if entry_atr is not None and entry_atr > 0:
                initial_risk = float(entry_atr * atr_mult * volume * context.contract_size)
            close = numeric_row_value(event, "close") or entry_price
            trailing_stop = None
            if entry_atr is not None and entry_atr > 0:
                trailing_stop = close - atr_mult * entry_atr if candidate["direction"] == "long" else close + atr_mult * entry_atr
            open_positions[symbol] = PortfolioPosition(
                policy_name=policy.policy_name,
                symbol=symbol,
                direction=candidate["direction"],
                timeframe=str(event.get("timeframe") or policy.timeframe),
                entry_source="+".join(dict.fromkeys(candidate["sources"])),
                entry_signal_time=event_timestamp,
                entry_time=pd.Timestamp(entry_row["datetime"]),
                entry_price=entry_price,
                volume=volume,
                contract_size=context.contract_size,
                entry_atr=entry_atr,
                initial_risk=initial_risk,
                highest_close=close,
                lowest_close=close,
                trailing_stop=trailing_stop,
                atr_mult=atr_mult,
            )
            capacity -= 1

    for symbol, position in list(open_positions.items()):
        context = contexts[symbol]
        exit_row = final_execution_row(context.bars_1m, end_exclusive)
        if exit_row is None:
            continue
        synthetic_event = pd.Series({"event_time": pd.Timestamp(exit_row["datetime"])})
        record = build_trade_record(
            position=position,
            exit_event=synthetic_event,
            exit_row=exit_row,
            exit_reason="end_of_sample",
            context=context,
            rate=rate,
        )
        if record is not None:
            records.append(record)

    trades_df = pd.DataFrame(records, columns=TRADE_COLUMNS) if records else pd.DataFrame(columns=TRADE_COLUMNS)
    if not trades_df.empty:
        trades_df = trades_df.sort_values(["policy_name", "entry_time", "symbol"], kind="stable").reset_index(drop=True)
    rejected_df = (
        pd.DataFrame(rejected_records, columns=REJECTED_SIGNAL_COLUMNS)
        if rejected_records
        else pd.DataFrame(columns=REJECTED_SIGNAL_COLUMNS)
    )
    return SimulationBundle(trades=trades_df, rejected_signals=rejected_df)


def simulate_all_policies(
    policies: list[PolicyRun],
    contexts: dict[str, SymbolContext],
    history_range: HistoryRange,
    *,
    portfolio_capital: float,
    position_sizing: str,
    fixed_size: float,
    rate: float,
    max_symbol_weight: float,
    max_portfolio_positions: int,
) -> pd.DataFrame:
    """Simulate all V3 policies."""

    bundle = simulate_all_policies_with_audit(
        policies=policies,
        contexts=contexts,
        history_range=history_range,
        portfolio_capital=portfolio_capital,
        position_sizing=position_sizing,
        fixed_size=fixed_size,
        rate=rate,
        max_symbol_weight=max_symbol_weight,
        max_portfolio_positions=max_portfolio_positions,
    )
    return bundle.trades


def simulate_all_policies_with_audit(
    policies: list[PolicyRun],
    contexts: dict[str, SymbolContext],
    history_range: HistoryRange,
    *,
    portfolio_capital: float,
    position_sizing: str,
    fixed_size: float,
    rate: float,
    max_symbol_weight: float,
    max_portfolio_positions: int,
) -> SimulationBundle:
    """Simulate all V3 policies and collect signal rejection audit records."""

    bundles = [
        simulate_portfolio_policy_with_audit(
            policy=policy,
            contexts=contexts,
            history_range=history_range,
            portfolio_capital=portfolio_capital,
            position_sizing=position_sizing,
            fixed_size=fixed_size,
            rate=rate,
            max_symbol_weight=max_symbol_weight,
            max_portfolio_positions=max_portfolio_positions,
        )
        for policy in policies
    ]
    trade_frames = [bundle.trades for bundle in bundles if not bundle.trades.empty]
    if trade_frames:
        trades = pd.concat(trade_frames, ignore_index=True).sort_values(
            ["policy_name", "entry_time", "symbol"],
            kind="stable",
        ).reset_index(drop=True)
    else:
        trades = pd.DataFrame(columns=TRADE_COLUMNS)
    rejected_frames = [bundle.rejected_signals for bundle in bundles if not bundle.rejected_signals.empty]
    if rejected_frames:
        rejected = pd.concat(rejected_frames, ignore_index=True).sort_values(
            ["policy_name", "signal_time", "symbol", "reason"],
            kind="stable",
        ).reset_index(drop=True)
    else:
        rejected = pd.DataFrame(columns=REJECTED_SIGNAL_COLUMNS)
    return SimulationBundle(trades=trades, rejected_signals=rejected)


def max_drawdown_from_trades(trade_df: pd.DataFrame, capital: float) -> tuple[float, float]:
    """Compute trade-level max drawdown and percent."""

    if trade_df.empty:
        return 0.0, 0.0
    pnl = pd.to_numeric(trade_df["net_pnl"], errors="coerce").fillna(0.0)
    equity = capital + pnl.cumsum()
    peak = pd.concat([pd.Series([capital], dtype=float), equity.reset_index(drop=True)], ignore_index=True).cummax().iloc[1:].reset_index(drop=True)
    drawdown = peak - equity.reset_index(drop=True)
    ddpercent = drawdown / peak.replace(0, np.nan) * 100.0
    return float(drawdown.max()), float(ddpercent.max()) if ddpercent.notna().any() else 0.0


def calculate_top_5pct_trade_pnl_contribution(trade_df: pd.DataFrame) -> float | None:
    """Calculate top 5 percent trade contribution to total net PnL."""

    if trade_df.empty:
        return None
    net = pd.to_numeric(trade_df["net_pnl"], errors="coerce").fillna(0.0)
    total = float(net.sum())
    if total == 0:
        return None
    top_n = max(1, int(math.ceil(len(net.index) * 0.05)))
    top_sum = float(net.sort_values(ascending=False).head(top_n).sum())
    return float(top_sum / total)


def calculate_largest_symbol_pnl_share(trade_df: pd.DataFrame) -> float | None:
    """Calculate largest single-symbol positive PnL share."""

    if trade_df.empty or "symbol" not in trade_df.columns:
        return None
    grouped = trade_df.groupby("symbol", dropna=False)["net_pnl"].sum()
    positive = grouped[grouped > 0]
    if not positive.empty and positive.sum() > 0:
        return float(positive.max() / positive.sum())
    absolute = grouped.abs()
    if absolute.sum() > 0:
        return float(absolute.max() / absolute.sum())
    return None


def compute_max_concurrent_positions(trade_df: pd.DataFrame) -> int:
    """Compute max concurrent portfolio positions from trade intervals."""

    if trade_df.empty:
        return 0
    events: list[tuple[pd.Timestamp, int]] = []
    for _, trade in trade_df.iterrows():
        events.append((pd.Timestamp(trade["entry_time"]), 1))
        events.append((pd.Timestamp(trade["exit_time"]), -1))
    events.sort(key=lambda item: (item[0], item[1]))
    current = 0
    maximum = 0
    for _, delta in events:
        current += delta
        maximum = max(maximum, current)
    return int(maximum)


def summarize_trade_slice(trade_df: pd.DataFrame, capital: float) -> dict[str, Any]:
    """Summarize one policy/symbol/month trade slice."""

    if trade_df.empty:
        return {
            "trade_count": 0,
            "long_count": 0,
            "short_count": 0,
            "active_symbol_count": 0,
            "no_cost_net_pnl": 0.0,
            "net_pnl": 0.0,
            "fee_total": 0.0,
            "slippage_total": 0.0,
            "cost_drag": 0.0,
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
            "largest_symbol_pnl_share": None,
            "portfolio_turnover": 0.0,
            "max_concurrent_positions": 0,
        }
    working = trade_df.copy().sort_values("exit_time", kind="stable")
    net = pd.to_numeric(working["net_pnl"], errors="coerce").fillna(0.0)
    wins = net[net > 0]
    losses = net[net < 0]
    no_cost = safe_sum(working["no_cost_net_pnl"])
    net_total = safe_sum(working["net_pnl"])
    fee_total = safe_sum(working["fee"])
    slippage_total = safe_sum(working["slippage"])
    max_dd, max_ddpercent = max_drawdown_from_trades(working, capital)
    profit_factor = None
    if not losses.empty:
        profit_factor = float(wins.sum() / abs(losses.sum())) if not wins.empty else 0.0
    return_drawdown_ratio = net_total / max_dd if max_dd > 0 else None
    sharpe_like = None
    if len(net.index) >= 2:
        std = float(net.std(ddof=0))
        if std > 0:
            sharpe_like = float(net.mean() / std * math.sqrt(len(net.index)))
    return {
        "trade_count": int(len(working.index)),
        "long_count": int((working["direction"] == "long").sum()),
        "short_count": int((working["direction"] == "short").sum()),
        "active_symbol_count": int(working["symbol"].nunique()),
        "no_cost_net_pnl": no_cost,
        "net_pnl": net_total,
        "fee_total": fee_total,
        "slippage_total": slippage_total,
        "cost_drag": no_cost - net_total,
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
        "top_5pct_trade_pnl_contribution": calculate_top_5pct_trade_pnl_contribution(working),
        "best_trade": float(net.max()) if len(net.index) else None,
        "worst_trade": float(net.min()) if len(net.index) else None,
        "largest_symbol_pnl_share": calculate_largest_symbol_pnl_share(working),
        "portfolio_turnover": safe_sum(working["turnover"]),
        "max_concurrent_positions": compute_max_concurrent_positions(working),
    }


def build_policy_leaderboard(trades_df: pd.DataFrame, policies: list[PolicyRun], symbol_count: int, capital: float) -> pd.DataFrame:
    """Build V3 policy leaderboard."""

    rows: list[dict[str, Any]] = []
    for policy in policies:
        group = trades_df[trades_df["policy_name"] == policy.policy_name].copy() if not trades_df.empty else pd.DataFrame(columns=TRADE_COLUMNS)
        row: dict[str, Any] = {"policy_name": policy.policy_name, "symbol_count": int(symbol_count)}
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
    """Build portfolio daily PnL by policy."""

    if trades_df.empty:
        return pd.DataFrame(columns=["policy_name", "date", "trade_count", "net_pnl", "no_cost_net_pnl"])
    working = trades_df.copy()
    working["date"] = pd.to_datetime(working["exit_time"]).dt.strftime("%Y-%m-%d")
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
    """Build portfolio trade-level equity curve per policy."""

    if trades_df.empty:
        return pd.DataFrame(columns=["policy_name", "time", "equity", "drawdown", "ddpercent"])
    rows: list[dict[str, Any]] = []
    for policy_name, group in trades_df.groupby("policy_name", dropna=False):
        working = group.copy().sort_values("exit_time", kind="stable").reset_index(drop=True)
        net = pd.to_numeric(working["net_pnl"], errors="coerce").fillna(0.0)
        no_cost = pd.to_numeric(working["no_cost_net_pnl"], errors="coerce").fillna(0.0)
        equity = capital + net.cumsum()
        no_cost_equity = capital + no_cost.cumsum()
        peak = pd.concat([pd.Series([capital], dtype=float), equity.reset_index(drop=True)], ignore_index=True).cummax().iloc[1:].reset_index(drop=True)
        drawdown = peak - equity.reset_index(drop=True)
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


def build_symbol_contribution(trades_df: pd.DataFrame, capital: float) -> pd.DataFrame:
    """Build per-policy per-symbol contribution table."""

    if trades_df.empty:
        return pd.DataFrame(columns=["policy_name", "symbol", "trade_count", "net_pnl", "pnl_share"])
    rows: list[dict[str, Any]] = []
    for policy_name, policy_group in trades_df.groupby("policy_name", dropna=False):
        symbol_net = policy_group.groupby("symbol")["net_pnl"].sum()
        positive_sum = float(symbol_net[symbol_net > 0].sum())
        abs_sum = float(symbol_net.abs().sum())
        for symbol, group in policy_group.groupby("symbol", dropna=False):
            row = {"policy_name": policy_name, "symbol": symbol}
            row.update(summarize_trade_slice(group, capital))
            net = float(symbol_net.loc[symbol])
            if positive_sum > 0 and net > 0:
                share = net / positive_sum
            elif abs_sum > 0:
                share = abs(net) / abs_sum
            else:
                share = None
            row["pnl_share"] = share
            rows.append(row)
    return pd.DataFrame(rows)


def count_incomplete_resampled_bars(bars_1m: pd.DataFrame, resampled_bars: pd.DataFrame, minutes: int) -> int:
    """Count resampled bars that do not have every constituent 1m timestamp."""

    if bars_1m.empty or resampled_bars.empty:
        return 0
    one_minute_times = pd.to_datetime(bars_1m["datetime"]).dropna()
    one_minute_ns = {int(timestamp.value) for timestamp in one_minute_times}
    one_minute = pd.Timedelta(minutes=1)
    count = 0
    for bar_time in pd.to_datetime(resampled_bars["datetime"]).dropna():
        start = pd.Timestamp(bar_time) - pd.Timedelta(minutes=minutes - 1)
        if any(int((start + offset * one_minute).value) not in one_minute_ns for offset in range(minutes)):
            count += 1
    return int(count)


def channel_current_bar_use_count(frame: pd.DataFrame, windows: list[int], history_range: HistoryRange) -> int:
    """Count Donchian previous-channel fields that look current-bar inclusive."""

    if frame.empty:
        return 0
    working = frame.copy().sort_values("datetime", kind="stable").reset_index(drop=True)
    dt = pd.to_datetime(working["datetime"])
    in_range = ((dt >= pd.Timestamp(history_range.start)) & (dt < pd.Timestamp(history_range.end_exclusive))).to_numpy(dtype=bool)
    high = pd.to_numeric(working["high"], errors="coerce")
    low = pd.to_numeric(working["low"], errors="coerce")
    current_uses = 0
    for window in windows:
        for side, source, reducer in [("high", high, "max"), ("low", low, "min")]:
            column = f"donchian_{side}_{window}_prev"
            if column not in working.columns:
                continue
            stored = pd.to_numeric(working[column], errors="coerce").to_numpy(dtype=float)
            rolling = source.rolling(window, min_periods=window)
            current = getattr(rolling, reducer)().to_numpy(dtype=float)
            previous = getattr(rolling, reducer)().shift(1).to_numpy(dtype=float)
            mask = (
                np.isfinite(stored)
                & np.isfinite(current)
                & np.isfinite(previous)
                & np.isclose(stored, current, rtol=1e-12, atol=1e-12)
                & ~np.isclose(stored, previous, rtol=1e-12, atol=1e-12)
                & in_range
            )
            current_uses += int(mask.sum())
    return int(current_uses)


def count_duplicate_symbol_positions(trades_df: pd.DataFrame) -> int:
    """Count overlapping positions for the same policy and symbol."""

    if trades_df.empty:
        return 0
    count = 0
    for (_policy, _symbol), group in trades_df.groupby(["policy_name", "symbol"], dropna=False):
        intervals = sorted(
            (pd.Timestamp(row["entry_time"]), pd.Timestamp(row["exit_time"]))
            for _, row in group.iterrows()
        )
        previous_exit: pd.Timestamp | None = None
        for entry_time, exit_time in intervals:
            if previous_exit is not None and entry_time < previous_exit:
                count += 1
            previous_exit = max(previous_exit, exit_time) if previous_exit is not None else exit_time
    return int(count)


def count_max_symbol_weight_violations(trades_df: pd.DataFrame, capital: float, max_symbol_weight: float) -> int:
    """Count trades whose entry notional exceeds the configured symbol cap."""

    if trades_df.empty:
        return 0
    max_notional = capital * max_symbol_weight
    notional = (
        pd.to_numeric(trades_df["entry_price"], errors="coerce")
        * pd.to_numeric(trades_df["volume"], errors="coerce")
        * pd.to_numeric(trades_df["contract_size"], errors="coerce")
    )
    return int((notional > max_notional + 1e-9).sum())


def build_contribution_warnings(leaderboard_df: pd.DataFrame) -> list[str]:
    """Build warnings for concentrated or hard-to-interpret contribution metrics."""

    warnings: list[str] = []
    if leaderboard_df.empty:
        return warnings
    for row in leaderboard_df.to_dict(orient="records"):
        policy_name = row.get("policy_name")
        trade_count = finite_or_none(row.get("trade_count")) or 0.0
        net_pnl = finite_or_none(row.get("net_pnl")) or 0.0
        largest_share = finite_or_none(row.get("largest_symbol_pnl_share"))
        top_share = finite_or_none(row.get("top_5pct_trade_pnl_contribution"))
        if net_pnl <= 0 and largest_share is not None:
            warnings.append(f"{policy_name}: largest_symbol_pnl_share uses absolute contribution because total net_pnl is non-positive.")
        if trade_count < 20 and top_share is not None:
            warnings.append(f"{policy_name}: top_5pct_trade_pnl_contribution is based on a small trade sample.")
        if top_share is not None and (top_share < 0 or top_share > 1):
            warnings.append(f"{policy_name}: top_5pct_trade_pnl_contribution is hard to interpret because total net_pnl is small or negative.")
    return warnings


def build_no_lookahead_checks(
    contexts: dict[str, SymbolContext],
    trades_df: pd.DataFrame,
    rejected_signals_df: pd.DataFrame,
    history_range: HistoryRange,
    *,
    max_portfolio_positions: int,
    portfolio_capital: float,
    max_symbol_weight: float,
) -> dict[str, Any]:
    """Build automated audit counters for V3 research mechanics."""

    incomplete_counts: dict[str, int] = {}
    current_bar_channel_used_count = 0
    for symbol, context in contexts.items():
        symbol_incomplete = 0
        for timeframe, minutes in [("1h", 60), ("4h", 240), ("1d", 1440)]:
            count = count_incomplete_resampled_bars(context.bars_1m, context.timeframes[timeframe], minutes)
            incomplete_counts[f"{symbol}:{timeframe}"] = count
            symbol_incomplete += count
            current_bar_channel_used_count += channel_current_bar_use_count(context.indicators[timeframe], DONCHIAN_WINDOWS, history_range)
        incomplete_counts[f"{symbol}:total"] = symbol_incomplete

    if trades_df.empty:
        signal_before_bar_close_count = 0
        exit_before_entry_count = 0
        max_position_violation_count = 0
    else:
        entry_time = pd.to_datetime(trades_df["entry_time"])
        entry_signal_time = pd.to_datetime(trades_df["entry_signal_time"])
        exit_time = pd.to_datetime(trades_df["exit_time"])
        signal_before_bar_close_count = int((entry_time <= entry_signal_time).sum())
        exit_before_entry_count = int((exit_time <= entry_time).sum())
        max_position_violation_count = int(
            sum(
                1
                for _policy_name, group in trades_df.groupby("policy_name", dropna=False)
                if compute_max_concurrent_positions(group) > max_portfolio_positions
            )
        )

    rejected_counts = (
        rejected_signals_df["reason"].value_counts().to_dict()
        if not rejected_signals_df.empty and "reason" in rejected_signals_df.columns
        else {}
    )
    return {
        "incomplete_htf_bar_used_count": int(sum(value for key, value in incomplete_counts.items() if not key.endswith(":total"))),
        "incomplete_resampled_bar_counts": incomplete_counts,
        "current_bar_channel_used_count": int(current_bar_channel_used_count),
        "signal_before_bar_close_count": signal_before_bar_close_count,
        "duplicate_symbol_position_count": count_duplicate_symbol_positions(trades_df),
        "max_position_violation_count": max_position_violation_count,
        "max_symbol_weight_violation_count": count_max_symbol_weight_violations(trades_df, portfolio_capital, max_symbol_weight),
        "exit_before_entry_count": exit_before_entry_count,
        "rejected_signal_counts": {str(key): int(value) for key, value in rejected_counts.items()},
    }


def build_audit_sample_trades(trades_df: pd.DataFrame) -> list[dict[str, Any]]:
    """Return the first 20 trades in the required audit schema."""

    if trades_df.empty:
        return []
    fields = [
        "policy_name",
        "symbol",
        "direction",
        "entry_time",
        "exit_time",
        "entry_price",
        "exit_price",
        "exit_reason",
        "net_pnl",
        "no_cost_pnl",
    ]
    records = dataframe_records(trades_df.sort_values(["entry_time", "policy_name", "symbol"], kind="stable").head(20))
    return [{field: record.get(field) for field in fields} for record in records]


def build_research_audit(
    *,
    symbols: list[str],
    contexts: dict[str, SymbolContext],
    policies: list[PolicyRun],
    trades_df: pd.DataFrame,
    rejected_signals_df: pd.DataFrame,
    leaderboard_df: pd.DataFrame,
    data_quality: dict[str, Any],
    history_range: HistoryRange,
    data_check_strict: bool,
    portfolio_capital: float,
    capital_mode: str,
    position_sizing: str,
    fixed_size: float,
    rate: float,
    slippage_mode: str,
    slippage: float,
    max_symbol_weight: float,
    max_portfolio_positions: int,
) -> dict[str, Any]:
    """Build trend_v3_research_audit.json payload."""

    return {
        "data_policy": {
            "symbols": symbols,
            "strict_mode": bool(data_check_strict),
            "coverage_summary": data_quality,
            "vt_symbol_filtering": "Each symbol context filters provided bars by vt_symbol before normalization; sqlite loads use split_vt_symbol(symbol, exchange).",
        },
        "resample_policy": {
            "timeframes": ["1h", "4h", "1d"],
            "closed_bar_only": True,
            "timestamp_meaning": "Resampled timestamps are the final included 1m close time; incomplete HTF bars are dropped.",
            "execution_policy": "Signals execute on the first 1m bar strictly after the completed HTF signal timestamp.",
        },
        "signal_policy": {
            "donchian_previous_channel": "Entry and exit use donchian_high/low_<window>_prev = rolling(...).shift(1); strict >/< comparisons are required.",
            "ema_policy": "EMA50/EMA200 are computed on closed HTF bars; long requires EMA50 > EMA200 and close > EMA50; short is symmetric.",
            "vol_compression_policy": "ATR percentile and Donchian width percentile must be <= 0.4 before breakout.",
            "risk_filter_policy": "Risk-filter policy requires ATR, 1h volatility, directional return, volume z-score, and body-ratio percentiles <= 0.8.",
        },
        "position_policy": {
            "one_position_per_symbol": True,
            "max_portfolio_positions": max_portfolio_positions,
            "signal_strength_ranking": "Simultaneous candidates are sorted by breakout distance / ATR or EMA spread / ATR, with overheated risk scores halved.",
            "duplicate_signal_handling": "Same-symbol same-direction simultaneous signals are merged into one candidate; lower-strength duplicates are audited as duplicate_signal_merged.",
            "max_symbol_weight": max_symbol_weight,
            "max_symbol_weight_policy": "Fixed-contract and volatility-target entries are rejected when entry notional would exceed capital * max_symbol_weight.",
            "position_sizing": position_sizing,
            "fixed_size": fixed_size,
            "capital_mode": capital_mode,
            "portfolio_capital": portfolio_capital,
        },
        "cost_policy": {
            "fee_formula": "fee = turnover * rate, where turnover is entry notional + exit notional.",
            "rate": rate,
            "slippage_mode": slippage_mode,
            "slippage_input": slippage,
            "slippage_formula": "round-trip slippage cost = absolute_slippage * volume * contract_size * 2; ticks use instrument pricetick.",
            "contract_size_source": "config/instruments/<symbol>.json size/contract_size.",
            "funding_fee_warning": FUNDING_WARNING,
        },
        "equity_policy": {
            "equity_curve_type": "closed_trade_equity",
            "mark_to_market": False,
            "daily_pnl_aggregation": "Daily PnL is grouped by exit_time date and policy.",
            "drawdown_formula": "max drawdown is computed from closed-trade cumulative cost-aware net_pnl versus initial capital peak.",
        },
        "contribution_policy": {
            "largest_symbol_pnl_share_formula": "If positive symbol PnL exists, largest positive symbol PnL / sum positive symbol PnL; otherwise largest absolute symbol PnL / sum absolute symbol PnL.",
            "top_5pct_trade_pnl_contribution_formula": "sum of top ceil(5% * trade_count) net_pnl divided by total net_pnl.",
            "negative_total_pnl_handling": "For non-positive total net_pnl, concentration metrics are retained for diagnostics and warnings mark them as hard to interpret.",
            "warnings": build_contribution_warnings(leaderboard_df),
        },
        "no_lookahead_checks": build_no_lookahead_checks(
            contexts=contexts,
            trades_df=trades_df,
            rejected_signals_df=rejected_signals_df,
            history_range=history_range,
            max_portfolio_positions=max_portfolio_positions,
            portfolio_capital=portfolio_capital,
            max_symbol_weight=max_symbol_weight,
        ),
        "rejected_signal_sample": dataframe_records(rejected_signals_df.head(50)),
        "sample_trades": build_audit_sample_trades(trades_df),
        "policy_definitions": [asdict(policy) for policy in policies],
    }


def load_btc_v2_reference(split: str) -> dict[str, Any]:
    """Load BTC Trend V2 reference if available."""

    path = PROJECT_ROOT / "reports" / "research" / "trend_following_v2" / split / "trend_policy_leaderboard.csv"
    if not path.exists():
        return {"available": False, "path": str(path)}
    try:
        frame = pd.read_csv(path)
    except Exception:
        return {"available": False, "path": str(path)}
    if frame.empty:
        return {"available": False, "path": str(path)}
    no_cost = pd.to_numeric(frame.get("no_cost_net_pnl"), errors="coerce")
    net = pd.to_numeric(frame.get("net_pnl"), errors="coerce")
    return {
        "available": True,
        "path": str(path),
        "best_no_cost_net_pnl": float(no_cost.max()) if no_cost.notna().any() else None,
        "best_cost_aware_net_pnl": float(net.max()) if net.notna().any() else None,
    }


def build_diagnostic_answers(split: str, leaderboard_df: pd.DataFrame, btc_v2_reference: dict[str, Any]) -> dict[str, Any]:
    """Build report answers for V3 acceptance questions."""

    is_oos_split = split in {"oos", "oos_ext"}
    if leaderboard_df.empty:
        return {"trend_following_v3_failed": is_oos_split, "notes": "leaderboard empty"}
    no_cost_positive = leaderboard_df[pd.to_numeric(leaderboard_df["no_cost_net_pnl"], errors="coerce") > 0]
    cost_positive = leaderboard_df[pd.to_numeric(leaderboard_df["net_pnl"], errors="coerce") > 0]
    concentrated_symbol = leaderboard_df[pd.to_numeric(leaderboard_df["largest_symbol_pnl_share"], errors="coerce") > 0.7]
    concentrated_trades = leaderboard_df[pd.to_numeric(leaderboard_df["top_5pct_trade_pnl_contribution"], errors="coerce") > 0.8]
    high_dd = leaderboard_df[pd.to_numeric(leaderboard_df["max_ddpercent"], errors="coerce") > 30.0]
    trade_count_low = leaderboard_df[pd.to_numeric(leaderboard_df["trade_count"], errors="coerce") < 10]
    best_v3_no_cost = finite_or_none(pd.to_numeric(leaderboard_df["no_cost_net_pnl"], errors="coerce").max())
    better_than_v2 = None
    if btc_v2_reference.get("available") and best_v3_no_cost is not None and btc_v2_reference.get("best_no_cost_net_pnl") is not None:
        better_than_v2 = bool(best_v3_no_cost > float(btc_v2_reference["best_no_cost_net_pnl"]))
    return {
        "portfolio_vs_btc_v2": {
            "btc_v2_reference": btc_v2_reference,
            "best_v3_no_cost_net_pnl": best_v3_no_cost,
            "multi_symbol_no_cost_better_than_btc_v2": better_than_v2,
        },
        "no_cost_positive_policy_count": int(len(no_cost_positive.index)),
        "cost_aware_positive_policy_count": int(len(cost_positive.index)),
        "no_cost_positive_policies": no_cost_positive["policy_name"].tolist(),
        "cost_aware_positive_policies": cost_positive["policy_name"].tolist(),
        "single_symbol_concentration_policies": concentrated_symbol["policy_name"].tolist(),
        "top_trade_concentration_policies": concentrated_trades["policy_name"].tolist(),
        "high_drawdown_policies_over_30pct": high_dd["policy_name"].tolist(),
        "low_trade_count_policies_under_10": trade_count_low["policy_name"].tolist(),
        "oos_stability_note": "Formal OOS stability is decided by scripts/compare_trend_following_v3.py across train/validation/oos or train_ext/validation_ext/oos_ext.",
        "strategy_v3_prototype_note": "Only stable_candidate_exists=true in the compare summary can enter further research audit; extended research does not directly allow Strategy V3 or demo/live.",
        "trend_following_v3_failed": bool(is_oos_split and cost_positive.empty),
    }


def format_number(value: Any, digits: int = 6) -> str:
    """Format numeric values for Markdown."""

    number = finite_or_none(value)
    if number is None:
        return "N/A"
    return f"{number:.{digits}f}"


def format_leaderboard_lines(leaderboard_df: pd.DataFrame, limit: int = 12) -> str:
    """Format top leaderboard rows."""

    if leaderboard_df.empty:
        return "- 无"
    lines = []
    for row in leaderboard_df.head(limit).to_dict(orient="records"):
        lines.append(
            f"- {row.get('policy_name')}: trades={row.get('trade_count')}, "
            f"symbols={row.get('active_symbol_count')}, no_cost={format_number(row.get('no_cost_net_pnl'), 4)}, "
            f"net={format_number(row.get('net_pnl'), 4)}, max_dd%={format_number(row.get('max_ddpercent'), 2)}, "
            f"largest_symbol_share={format_number(row.get('largest_symbol_pnl_share'), 3)}, "
            f"top5pct_contrib={format_number(row.get('top_5pct_trade_pnl_contribution'), 3)}"
        )
    return "\n".join(lines)


def render_policy_table(policies: list[PolicyRun]) -> str:
    """Render policy definitions table."""

    policy_rows = {
        "v3_4h_donchian_20_10_atr4": ("4h", "close > previous Donchian high 20 / close < previous low 20", "Donchian 10 or ATR4 trail", "portfolio caps"),
        "v3_4h_donchian_55_20_atr4": ("4h", "Donchian 55 breakout", "Donchian 20 or ATR4 trail", "portfolio caps"),
        "v3_4h_donchian_100_30_atr5": ("4h", "Donchian 100 breakout", "Donchian 30 or ATR5 trail", "portfolio caps"),
        "v3_1d_donchian_20_10_atr4": ("1d", "Donchian 20 breakout", "Donchian 10 or ATR4 trail", "portfolio caps"),
        "v3_1d_donchian_55_20_atr5": ("1d", "Donchian 55 breakout", "Donchian 20 or ATR5 trail", "portfolio caps"),
        "v3_4h_ema_50_200_atr4": ("4h", "EMA50/EMA200 trend and close on EMA50 side", "EMA50 loss or ATR4 trail", "portfolio caps"),
        "v3_1d_ema_50_200_atr5": ("1d", "EMA50/EMA200 trend and close on EMA50 side", "EMA50 loss or ATR5 trail", "portfolio caps"),
        "v3_4h_vol_compression_donchian_breakout": ("4h", "ATR and Donchian width percentile <= 0.4 then breakout", "Donchian 10 or ATR4 trail", "compression only, no mean reversion"),
        "v3_4h_donchian_55_with_risk_filters": ("4h", "Donchian 55 breakout", "Donchian 20 or ATR4 trail", "Signal Lab risk percentiles <= 0.8"),
        "v3_ensemble_core": ("mixed", "4h Donchian 55/20 + 1d Donchian 20/10 + 4h EMA50/200", "component exits or ATR trail", "same symbol/direction merged"),
    }
    lines = ["| policy_name | timeframe | entry | exit | risk/filter |", "|---|---|---|---|---|"]
    for policy in policies:
        timeframe, entry, exit_rule, risk = policy_rows[policy.policy_name]
        lines.append(f"| {policy.policy_name} | {timeframe} | {entry} | {exit_rule} | {risk} |")
    return "\n".join(lines)


def render_markdown(summary: dict[str, Any], leaderboard_df: pd.DataFrame, policies: list[PolicyRun]) -> str:
    """Render trend_v3_report.md."""

    answers = summary.get("diagnostic_answers") or {}
    warnings = summary.get("warnings") or []
    warning_lines = "\n".join(f"- {item}" for item in warnings) if warnings else "- 无"
    return (
        "# Trend Following V3 多品种组合级趋势跟踪研究\n\n"
        "## 核心结论\n"
        f"- split_scheme={summary.get('split_scheme')}, split={summary.get('split')}, symbols={summary.get('ready_symbols')}, trade_count={summary.get('trade_count')}\n"
        f"- start={summary.get('start')}, end={summary.get('end')}, end_exclusive={summary.get('end_exclusive')}\n"
        f"- no_cost_positive_policy_count={answers.get('no_cost_positive_policy_count')}\n"
        f"- cost_aware_positive_policy_count={answers.get('cost_aware_positive_policy_count')}\n"
        f"- trend_following_v3_failed={str(bool(answers.get('trend_following_v3_failed'))).lower()}\n"
        f"- funding_fee_warning={FUNDING_WARNING}\n\n"
        "## Policy Leaderboard\n"
        f"{format_leaderboard_lines(leaderboard_df)}\n\n"
        "## 必答问题\n"
        f"1. 多品种组合是否优于 BTC 单品种 Trend V2：{answers.get('portfolio_vs_btc_v2')}。\n"
        f"2. 组合层面 no-cost 为正的 policy：{answers.get('no_cost_positive_policies')}。\n"
        f"3. 成本后仍为正的 policy：{answers.get('cost_aware_positive_policies')}。\n"
        f"4. 是否存在单一 symbol 贡献过度：{answers.get('single_symbol_concentration_policies')}。\n"
        f"5. top 5% trades 是否贡献过度：{answers.get('top_trade_concentration_policies')}。\n"
        f"6. OOS 是否稳定：{answers.get('oos_stability_note')}。\n"
        f"7. 回撤是否可接受：max_ddpercent>30% policies={answers.get('high_drawdown_policies_over_30pct')}。\n"
        f"8. 交易次数是否足够：trade_count<10 policies={answers.get('low_trade_count_policies_under_10')}。\n"
        f"9. 是否可能进入 Strategy V3 原型开发：{answers.get('strategy_v3_prototype_note')}。\n"
        f"10. 如果没有稳定候选，trend_following_v3_failed={str(bool(answers.get('trend_following_v3_failed'))).lower()}。\n\n"
        "## V3 Policy 定义\n"
        f"{render_policy_table(policies)}\n\n"
        "## 组合和成本假设\n"
        f"- capital={summary.get('capital')}, capital_mode={summary.get('capital_mode')}, portfolio_capital={summary.get('portfolio_capital')}\n"
        f"- position_sizing={summary.get('position_sizing')}, fixed_size={summary.get('fixed_size')}\n"
        f"- max_symbol_weight={summary.get('max_symbol_weight')}, max_portfolio_positions={summary.get('max_portfolio_positions')}\n"
        f"- rate={summary.get('rate')}, slippage_mode={summary.get('slippage_mode')}, slippage={summary.get('slippage')}\n"
        "- cost-aware net_pnl subtracts fee and slippage; no_cost_net_pnl equals gross price PnL.\n"
        f"- {FUNDING_WARNING}\n\n"
        "## 输出文件\n"
        "- trend_v3_summary.json\n"
        "- trend_v3_policy_leaderboard.csv\n"
        "- trend_v3_portfolio_equity_curve.csv\n"
        "- trend_v3_portfolio_daily_pnl.csv\n"
        "- trend_v3_trades.csv\n"
        "- trend_v3_policy_by_symbol.csv\n"
        "- trend_v3_policy_by_month.csv\n"
        "- trend_v3_symbol_contribution.csv\n"
        "- trend_v3_drawdown.csv\n"
        "- trend_v3_report.md\n"
        "- trend_v3_research_audit.json\n"
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


def build_summary(
    *,
    split: str,
    split_scheme: str,
    history_range: HistoryRange,
    output_dir: Path,
    symbols: list[str],
    ready_symbols: list[str],
    policies: list[PolicyRun],
    trades_df: pd.DataFrame,
    leaderboard_df: pd.DataFrame,
    data_quality: dict[str, Any],
    warnings: list[str],
    capital: float,
    capital_mode: str,
    portfolio_capital: float,
    position_sizing: str,
    fixed_size: float,
    rate: float,
    slippage_mode: str,
    slippage: float,
    max_symbol_weight: float,
    max_portfolio_positions: int,
    audit_path: Path,
) -> dict[str, Any]:
    """Build trend_v3_summary.json payload."""

    all_warnings = [FUNDING_WARNING] + warnings + build_contribution_warnings(leaderboard_df)
    btc_v2_reference = load_btc_v2_reference(split)
    answers = build_diagnostic_answers(split, leaderboard_df, btc_v2_reference)
    return {
        "split_scheme": split_scheme,
        "split": split,
        "start": history_range.start.isoformat(),
        "end": history_range.end_display.isoformat(),
        "end_exclusive": history_range.end_exclusive.isoformat(),
        "timezone": history_range.timezone_name,
        "output_dir": str(output_dir),
        "requested_symbols": symbols,
        "ready_symbols": ready_symbols,
        "capital": capital,
        "capital_mode": capital_mode,
        "portfolio_capital": portfolio_capital,
        "position_sizing": position_sizing,
        "fixed_size": fixed_size,
        "rate": rate,
        "slippage_mode": slippage_mode,
        "slippage": slippage,
        "max_symbol_weight": max_symbol_weight,
        "max_portfolio_positions": max_portfolio_positions,
        "audit_path": str(audit_path),
        "policy_run_count": len(policies),
        "trade_count": int(len(trades_df.index)),
        "warnings": all_warnings,
        "funding_fee_warning": FUNDING_WARNING,
        "data_quality": data_quality,
        "policy_definitions": [asdict(policy) for policy in policies],
        "leaderboard": dataframe_records(leaderboard_df),
        "diagnostic_answers": answers,
        "trend_following_v3_failed": bool(answers.get("trend_following_v3_failed")),
    }


def write_outputs(
    output_dir: Path,
    summary: dict[str, Any],
    leaderboard_df: pd.DataFrame,
    equity_curve_df: pd.DataFrame,
    daily_pnl_df: pd.DataFrame,
    trades_df: pd.DataFrame,
    by_symbol_df: pd.DataFrame,
    by_month_df: pd.DataFrame,
    symbol_contribution_df: pd.DataFrame,
    drawdown_df: pd.DataFrame,
    markdown: str,
    data_quality: dict[str, Any],
    audit: dict[str, Any],
) -> None:
    """Write all V3 research artifacts."""

    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "trend_v3_summary.json", summary)
    write_dataframe(output_dir / "trend_v3_policy_leaderboard.csv", leaderboard_df)
    write_dataframe(output_dir / "trend_v3_portfolio_equity_curve.csv", equity_curve_df)
    write_dataframe(output_dir / "trend_v3_portfolio_daily_pnl.csv", daily_pnl_df)
    write_dataframe(output_dir / "trend_v3_trades.csv", trades_df)
    write_dataframe(output_dir / "trend_v3_policy_by_symbol.csv", by_symbol_df)
    write_dataframe(output_dir / "trend_v3_policy_by_month.csv", by_month_df)
    write_dataframe(output_dir / "trend_v3_symbol_contribution.csv", symbol_contribution_df)
    write_dataframe(output_dir / "trend_v3_drawdown.csv", drawdown_df)
    (output_dir / "trend_v3_report.md").write_text(markdown, encoding="utf-8")
    write_json(output_dir / "data_quality.json", data_quality)
    write_json(output_dir / "trend_v3_research_audit.json", audit)


def run_research(
    symbols: list[str],
    split: str,
    history_range: HistoryRange,
    output_dir: Path,
    timezone_name: str,
    *,
    split_scheme: str = "default",
    interval: str = "1m",
    capital: float = DEFAULT_CAPITAL,
    capital_mode: str = "portfolio_fixed",
    position_sizing: str = "fixed_contract",
    fixed_size: float = DEFAULT_FIXED_SIZE,
    rate: float = DEFAULT_RATE,
    slippage_mode: str = "ticks",
    slippage: float = DEFAULT_SLIPPAGE,
    max_symbol_weight: float = DEFAULT_MAX_SYMBOL_WEIGHT,
    max_portfolio_positions: int = DEFAULT_MAX_PORTFOLIO_POSITIONS,
    data_check_strict: bool = False,
    max_runs: int | None = None,
    logger: logging.Logger | None = None,
    bars_from_db: bool = True,
    bars_by_symbol: dict[str, pd.DataFrame] | None = None,
    instrument_meta_by_symbol: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Run the complete Trend Following V3 research workflow."""

    ZoneInfo(timezone_name)
    if interval != "1m":
        raise TrendFollowingV3Error("Trend V3 当前仅支持 --interval 1m")
    logger = logger or logging.getLogger("research_trend_following_v3")
    contexts, data_quality, warnings = build_symbol_contexts(
        symbols=symbols,
        history_range=history_range,
        timezone_name=timezone_name,
        logger=logger,
        slippage_mode=slippage_mode,
        slippage=slippage,
        data_check_strict=data_check_strict,
        bars_from_db=bars_from_db,
        bars_by_symbol=bars_by_symbol,
        instrument_meta_by_symbol=instrument_meta_by_symbol,
    )
    data_quality["split_scheme"] = split_scheme
    data_quality["split"] = split
    data_quality["start"] = history_range.start.isoformat()
    data_quality["end"] = history_range.end_display.isoformat()
    data_quality["end_exclusive"] = history_range.end_exclusive.isoformat()
    ready_symbols = list(contexts)
    portfolio_capital = float(capital if capital_mode == "portfolio_fixed" else capital * len(ready_symbols))
    policies = build_policy_runs(max_runs)
    simulation = simulate_all_policies_with_audit(
        policies=policies,
        contexts=contexts,
        history_range=history_range,
        portfolio_capital=portfolio_capital,
        position_sizing=position_sizing,
        fixed_size=fixed_size,
        rate=rate,
        max_symbol_weight=max_symbol_weight,
        max_portfolio_positions=max_portfolio_positions,
    )
    trades_df = simulation.trades
    rejected_signals_df = simulation.rejected_signals
    leaderboard_df = build_policy_leaderboard(trades_df, policies, len(ready_symbols), portfolio_capital)
    equity_curve_df = build_equity_curve(trades_df, portfolio_capital)
    daily_pnl_df = build_daily_pnl(trades_df, portfolio_capital)
    by_symbol_df = build_group_summary(trades_df, ["policy_name", "symbol"], portfolio_capital)
    if trades_df.empty:
        by_month_df = pd.DataFrame(columns=["policy_name", "month", "trade_count"])
    else:
        month_df = trades_df.copy()
        month_df["month"] = pd.to_datetime(month_df["exit_time"]).dt.strftime("%Y-%m")
        by_month_df = build_group_summary(month_df, ["policy_name", "month"], portfolio_capital)
    symbol_contribution_df = build_symbol_contribution(trades_df, portfolio_capital)
    drawdown_df = equity_curve_df[["policy_name", "time", "equity", "drawdown", "ddpercent"]].copy() if not equity_curve_df.empty else pd.DataFrame(columns=["policy_name", "time", "equity", "drawdown", "ddpercent"])
    summary = build_summary(
        split=split,
        split_scheme=split_scheme,
        history_range=history_range,
        output_dir=output_dir,
        symbols=symbols,
        ready_symbols=ready_symbols,
        policies=policies,
        trades_df=trades_df,
        leaderboard_df=leaderboard_df,
        data_quality=data_quality,
        warnings=warnings,
        capital=capital,
        capital_mode=capital_mode,
        portfolio_capital=portfolio_capital,
        position_sizing=position_sizing,
        fixed_size=fixed_size,
        rate=rate,
        slippage_mode=slippage_mode,
        slippage=slippage,
        max_symbol_weight=max_symbol_weight,
        max_portfolio_positions=max_portfolio_positions,
        audit_path=output_dir / "trend_v3_research_audit.json",
    )
    audit = build_research_audit(
        symbols=symbols,
        contexts=contexts,
        policies=policies,
        trades_df=trades_df,
        rejected_signals_df=rejected_signals_df,
        leaderboard_df=leaderboard_df,
        data_quality=data_quality,
        history_range=history_range,
        data_check_strict=data_check_strict,
        portfolio_capital=portfolio_capital,
        capital_mode=capital_mode,
        position_sizing=position_sizing,
        fixed_size=fixed_size,
        rate=rate,
        slippage_mode=slippage_mode,
        slippage=slippage,
        max_symbol_weight=max_symbol_weight,
        max_portfolio_positions=max_portfolio_positions,
    )
    markdown = render_markdown(summary, leaderboard_df, policies)
    write_outputs(
        output_dir=output_dir,
        summary=summary,
        leaderboard_df=leaderboard_df,
        equity_curve_df=equity_curve_df,
        daily_pnl_df=daily_pnl_df,
        trades_df=trades_df,
        by_symbol_df=by_symbol_df,
        by_month_df=by_month_df,
        symbol_contribution_df=symbol_contribution_df,
        drawdown_df=drawdown_df,
        markdown=markdown,
        data_quality=data_quality,
        audit=audit,
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""

    args = parse_args(argv)
    ensure_headless_runtime()
    logger = setup_logging("research_trend_following_v3", verbose=args.verbose)
    try:
        symbols = parse_symbols(args.symbols)
        split = resolve_split_name(args.split_scheme, args.split, args.split_name)
        history_range = resolve_split_range(split, args.start, args.end, args.timezone, args.split_scheme)
        default_root = "trend_following_v3_extended" if args.split_scheme == "extended" else "trend_following_v3"
        output_dir = resolve_path(
            args.output_dir,
            PROJECT_ROOT / "reports" / "research" / default_root / split,
        )
        summary = run_research(
            symbols=symbols,
            split=split,
            history_range=history_range,
            output_dir=output_dir,
            timezone_name=args.timezone,
            split_scheme=args.split_scheme,
            interval=args.interval,
            capital=args.capital,
            capital_mode=args.capital_mode,
            position_sizing=args.position_sizing,
            fixed_size=args.fixed_size,
            rate=args.rate,
            slippage_mode=args.slippage_mode,
            slippage=args.slippage,
            max_symbol_weight=args.max_symbol_weight,
            max_portfolio_positions=args.max_portfolio_positions,
            data_check_strict=args.data_check_strict,
            max_runs=args.max_runs,
            logger=logger,
            bars_from_db=args.bars_from_db,
        )
        print_json_block(
            "Trend Following V3 summary:",
            {
                "output_dir": output_dir,
                "split_scheme": args.split_scheme,
                "split": split,
                "start": summary.get("start"),
                "end": summary.get("end"),
                "end_exclusive": summary.get("end_exclusive"),
                "ready_symbols": summary.get("ready_symbols"),
                "trade_count": summary.get("trade_count"),
                "trend_following_v3_failed": summary.get("trend_following_v3_failed"),
                "warnings": summary.get("warnings"),
            },
        )
        if args.json:
            print(json.dumps(to_jsonable(summary), ensure_ascii=False, indent=2))
        return 0
    except TrendFollowingV3Error as exc:
        log_event(logger, logging.ERROR, "trend_following_v3.error", str(exc))
        return 2
    except Exception:
        logger.exception(
            "Unexpected error during Trend Following V3 research",
            extra={"event": "trend_following_v3.unexpected_error"},
        )
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
