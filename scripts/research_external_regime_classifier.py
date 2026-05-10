#!/usr/bin/env python3
"""Research-only external regime classifier for Trend V3 attribution."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from common_runtime import PROJECT_ROOT, ensure_headless_runtime, print_json_block, to_jsonable
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
DEFAULT_MARKET_INTERVAL = "1m"
DEFAULT_DATABASE_PATH = PROJECT_ROOT / ".vntrader" / "database.db"
DEFAULT_FUNDING_DIR = PROJECT_ROOT / "data" / "funding" / "okx"
DEFAULT_TREND_V3_DIR = PROJECT_ROOT / "reports" / "research" / "trend_following_v3_extended"
DEFAULT_ACTUAL_FUNDING_TRADES = (
    PROJECT_ROOT / "reports" / "research" / "trend_following_v3_actual_funding" / "actual_funding_trade_adjustments.csv"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "reports" / "research" / "external_regime_classifier"

SPLIT_WINDOWS = {
    "train_ext": ("2023-01-01", "2024-06-30"),
    "validation_ext": ("2024-07-01", "2025-06-30"),
    "oos_ext": ("2025-07-01", "2026-03-31"),
}
LABEL_COLUMNS = [
    "is_trend_friendly",
    "is_trend_hostile",
    "is_high_vol_chop",
    "is_broad_uptrend",
    "is_broad_downtrend",
    "is_narrow_single_symbol_trend",
    "is_funding_overheated",
    "is_funding_supportive",
    "is_compression",
]
MARKET_FEATURE_COLUMNS = [
    "number_of_symbols_above_ema50_1d",
    "number_of_symbols_above_ema200_1d",
    "ema50_ema200_breadth",
    "average_ema_spread_pct",
    "median_ema_spread_pct",
    "strong_trend_symbol_count",
    "trend_efficiency_mean",
    "trend_efficiency_median",
    "trend_efficiency_dispersion",
    "average_pairwise_correlation_20d",
    "average_pairwise_correlation_60d",
    "return_dispersion_20d",
    "return_dispersion_60d",
    "market_breadth_return_20d",
    "market_breadth_return_60d",
    "average_atr_pct",
    "median_atr_pct",
    "atr_pct_dispersion",
    "realized_volatility_mean",
    "realized_volatility_dispersion",
    "high_vol_symbol_count",
    "low_vol_symbol_count",
    "average_drawdown_from_60d_high",
    "max_symbol_drawdown_from_60d_high",
    "rebound_breadth",
    "symbols_near_60d_high_count",
]
FUNDING_FEATURE_COLUMNS = [
    "average_funding_rate",
    "median_funding_rate",
    "funding_dispersion",
    "positive_funding_symbol_count",
    "negative_funding_symbol_count",
    "extreme_positive_funding_count",
    "extreme_negative_funding_count",
    "funding_trend_7d",
    "funding_trend_30d",
]
REGIME_FEATURE_COLUMNS = MARKET_FEATURE_COLUMNS + FUNDING_FEATURE_COLUMNS
FORBIDDEN_FEATURE_PATTERNS = ("pnl", "profit", "loss", "trade", "policy", "future")
FILTER_DEFINITIONS = {
    "original_all": "All V3 extended trades.",
    "keep_trend_friendly": "Keep trades whose completed daily regime is trend_friendly.",
    "exclude_hostile_chop_overheated": "Drop trend_hostile, high_vol_chop, and funding_overheated trades.",
    "keep_trend_friendly_exclude_funding_overheated": "Keep trend_friendly trades and drop funding_overheated.",
    "exclude_funding_overheated": "Drop funding_overheated trades only.",
}
REQUIRED_OUTPUT_FILES = [
    "external_regime_feature_dataset.csv",
    "external_regime_thresholds.json",
    "external_regime_labels.csv",
    "regime_label_distribution.csv",
    "trade_regime_classifier_attribution.csv",
    "policy_performance_by_regime.csv",
    "split_performance_by_regime.csv",
    "classifier_filter_experiment.csv",
    "external_regime_classifier_report.md",
    "external_regime_classifier_summary.json",
]


class ExternalRegimeClassifierError(Exception):
    """Raised when external regime classifier research cannot complete."""


@dataclass(frozen=True, slots=True)
class ResearchOutputs:
    """Generated research outputs."""

    output_dir: Path
    feature_dataset: pd.DataFrame
    thresholds: dict[str, Any]
    labels: pd.DataFrame
    trade_attribution: pd.DataFrame
    filter_experiment: pd.DataFrame
    summary: dict[str, Any]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description="Research external regime classifier feasibility for Trend V3.")
    parser.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS))
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--end", default=DEFAULT_END)
    parser.add_argument("--timezone", default=DEFAULT_TIMEZONE)
    parser.add_argument("--market-interval", default=DEFAULT_MARKET_INTERVAL, choices=("1m",))
    parser.add_argument("--funding-dir", default=str(DEFAULT_FUNDING_DIR))
    parser.add_argument("--trend-v3-dir", default=str(DEFAULT_TREND_V3_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--database-path", default=str(DEFAULT_DATABASE_PATH))
    parser.add_argument("--actual-funding-trades", default=str(DEFAULT_ACTUAL_FUNDING_TRADES))
    parser.add_argument("--data-check-strict", action="store_true")
    parser.add_argument("--json", action="store_true", help="Print summary JSON after writing outputs.")
    return parser.parse_args(argv)


def resolve_path(path_arg: str | Path) -> Path:
    """Resolve a project-relative path."""

    path = Path(path_arg)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def parse_symbol_list(value: str) -> list[str]:
    """Parse comma/space separated vt_symbols while preserving order."""

    symbols: list[str] = []
    seen: set[str] = set()
    for token in re.split(r"[\s,]+", str(value or "")):
        symbol = token.strip()
        if not symbol or symbol in seen:
            continue
        symbols.append(symbol)
        seen.add(symbol)
    if not symbols:
        raise ExternalRegimeClassifierError("--symbols must contain at least one vt_symbol")
    return symbols


def split_vt_symbol(vt_symbol: str) -> tuple[str, str]:
    """Split a local vt_symbol into symbol/exchange strings."""

    symbol, separator, exchange = vt_symbol.partition(".")
    if not separator or not symbol or not exchange:
        raise ExternalRegimeClassifierError(f"Invalid vt_symbol: {vt_symbol!r}")
    return symbol, exchange


def vt_symbol_to_okx_inst_id(vt_symbol: str) -> str:
    """Best-effort conversion from local vt_symbol to OKX swap instrument id."""

    symbol, _exchange = split_vt_symbol(vt_symbol)
    if symbol.endswith("_SWAP_OKX"):
        base_quote = symbol[: -len("_SWAP_OKX")]
        if base_quote.endswith("USDT"):
            return f"{base_quote[:-4]}-USDT-SWAP"
    return symbol.replace("_", "-")


def parse_research_range(start: str, end: str, timezone_name: str) -> HistoryRange:
    """Parse the requested 1m research range."""

    return parse_history_range(start, end, timedelta(minutes=1), timezone_name)


def split_for_timestamp(timestamp: pd.Timestamp) -> str:
    """Return the extended split for a timestamp."""

    day = pd.Timestamp(timestamp).date().isoformat()
    for split, (start, end) in SPLIT_WINDOWS.items():
        if start <= day <= end:
            return split
    return "outside"


def finite_float(value: Any) -> float | None:
    """Return a finite float or None."""

    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def safe_sum(values: pd.Series) -> float:
    """Return numeric sum with missing values treated as zero."""

    return float(pd.to_numeric(values, errors="coerce").fillna(0.0).sum())


def true_range(df: pd.DataFrame) -> pd.Series:
    """Compute true range for an OHLCV frame."""

    high = pd.to_numeric(df["high"], errors="coerce")
    low = pd.to_numeric(df["low"], errors="coerce")
    close = pd.to_numeric(df["close"], errors="coerce")
    previous_close = close.shift(1)
    return pd.concat([high - low, (high - previous_close).abs(), (low - previous_close).abs()], axis=1).max(axis=1)


def rolling_prior_percentile(series: pd.Series, window: int = 240) -> pd.Series:
    """Current-value percentile versus prior observations only."""

    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    output = np.full(values.size, np.nan, dtype=float)
    min_periods = min(max(20, window // 4), window)
    for index, current in enumerate(values):
        if not np.isfinite(current):
            continue
        history = values[max(0, index - window):index]
        clean = history[np.isfinite(history)]
        if clean.size < min_periods:
            continue
        output[index] = float(np.mean(clean <= current))
    return pd.Series(output, index=series.index, dtype=float)


def normalize_bars(bars: pd.DataFrame, timezone_name: str) -> pd.DataFrame:
    """Normalize raw bar rows into timezone-aware 1m OHLCV records."""

    if bars.empty:
        return pd.DataFrame(columns=["datetime", "open", "high", "low", "close", "volume"])
    tz = ZoneInfo(timezone_name)
    result = bars.copy()
    result["datetime"] = pd.to_datetime(result["datetime"], errors="coerce")
    if getattr(result["datetime"].dt, "tz", None) is None:
        result["datetime"] = result["datetime"].dt.tz_localize(tz)
    else:
        result["datetime"] = result["datetime"].dt.tz_convert(tz)
    for column in ["open", "high", "low", "close", "volume"]:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    result = result.dropna(subset=["datetime", "open", "high", "low", "close"])
    return result.sort_values("datetime", kind="stable").drop_duplicates("datetime", keep="last").reset_index(drop=True)


def load_symbol_1m_bars(
    vt_symbol: str,
    database_path: Path,
    history_range: HistoryRange,
    timezone_name: str,
) -> pd.DataFrame:
    """Load one symbol's 1m OHLCV bars from local sqlite."""

    if not database_path.exists():
        raise ExternalRegimeClassifierError(f"database missing: {database_path}")
    db_symbol, db_exchange = split_vt_symbol(vt_symbol)
    query_start = history_range.start.replace(tzinfo=None).isoformat(sep=" ", timespec="seconds")
    query_end = history_range.end_exclusive.replace(tzinfo=None).isoformat(sep=" ", timespec="seconds")
    connection = sqlite3.connect(database_path)
    try:
        query = (
            "select datetime, open_price as open, high_price as high, low_price as low, "
            "close_price as close, volume "
            "from dbbardata "
            "where symbol = ? and exchange = ? and interval = ? and datetime >= ? and datetime < ? "
            "order by datetime"
        )
        bars = pd.read_sql_query(query, connection, params=(db_symbol, db_exchange, "1m", query_start, query_end))
    finally:
        connection.close()
    return normalize_bars(bars, timezone_name)


def validate_1m_coverage(vt_symbol: str, bars_1m: pd.DataFrame, history_range: HistoryRange) -> dict[str, Any]:
    """Build strict 1m coverage metadata."""

    expected = expected_bar_count(history_range)
    actual = int(bars_1m["datetime"].nunique()) if not bars_1m.empty else 0
    return {
        "vt_symbol": vt_symbol,
        "expected_count": int(expected),
        "actual_count": int(actual),
        "coverage_complete": bool(expected > 0 and actual == expected),
        "first_datetime": bars_1m["datetime"].min().isoformat() if not bars_1m.empty else None,
        "last_datetime": bars_1m["datetime"].max().isoformat() if not bars_1m.empty else None,
    }


def resample_daily_ohlcv(bars_1m: pd.DataFrame) -> pd.DataFrame:
    """Resample complete 1m bars into daily closed bars timestamped at 23:59."""

    if bars_1m.empty:
        return pd.DataFrame(columns=["datetime", "open", "high", "low", "close", "volume"])
    working = bars_1m.copy().sort_values("datetime", kind="stable").set_index("datetime")
    grouped = working.resample("1D", label="left", closed="left")
    daily = grouped.agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
    counts = grouped["close"].count()
    daily = daily.loc[counts == 1440].dropna(subset=["open", "high", "low", "close"]).copy()
    daily.index = daily.index + pd.Timedelta(hours=23, minutes=59)
    return daily.reset_index().rename(columns={"index": "datetime"})


def compute_symbol_daily_indicators(vt_symbol: str, daily: pd.DataFrame) -> pd.DataFrame:
    """Compute independent daily market-regime indicators for one symbol."""

    if daily.empty:
        return pd.DataFrame()
    df = daily.copy().sort_values("datetime", kind="stable").reset_index(drop=True)
    df["vt_symbol"] = vt_symbol
    for column in ["open", "high", "low", "close", "volume"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    close = df["close"]
    returns = close.pct_change()
    df["return_1d"] = returns
    df["return_20d"] = close / close.shift(20) - 1.0
    df["return_60d"] = close / close.shift(60) - 1.0
    df["ema50"] = close.ewm(span=50, adjust=False, min_periods=1).mean()
    df["ema200"] = close.ewm(span=200, adjust=False, min_periods=1).mean()
    df["above_ema50"] = close > df["ema50"]
    df["above_ema200"] = close > df["ema200"]
    df["ema_spread_pct"] = (df["ema50"] - df["ema200"]) / close.replace(0, np.nan)
    path_20d = close.diff().abs().rolling(20, min_periods=5).sum()
    df["trend_efficiency"] = (close - close.shift(20)).abs() / path_20d.replace(0, np.nan)
    df["atr14"] = true_range(df).rolling(14, min_periods=1).mean()
    df["atr_pct"] = df["atr14"] / close.replace(0, np.nan)
    df["atr_pct_percentile"] = rolling_prior_percentile(df["atr_pct"], 240)
    df["realized_volatility"] = returns.rolling(20, min_periods=5).std(ddof=0)
    df["drawdown_from_60d_high"] = close / close.rolling(60, min_periods=1).max().replace(0, np.nan) - 1.0
    df["rebound_from_20d_low"] = close / close.rolling(20, min_periods=1).min().replace(0, np.nan) - 1.0
    df["near_60d_high"] = df["drawdown_from_60d_high"] >= -0.05
    df["rebound_state"] = df["rebound_from_20d_low"] >= 0.05
    df["high_vol_state"] = df["atr_pct_percentile"] >= 0.80
    df["low_vol_state"] = df["atr_pct_percentile"] <= 0.20
    df["strong_trend_state"] = (df["ema_spread_pct"].abs() >= 0.03) & (df["trend_efficiency"] >= 0.35)
    return df


def average_pairwise_correlation(returns: pd.DataFrame, window: int) -> pd.Series:
    """Compute rolling average pairwise correlation across symbols."""

    values: list[float] = []
    index = list(returns.index)
    min_periods = min(5, window)
    for end in range(len(index)):
        frame = returns.iloc[max(0, end - window + 1): end + 1]
        frame = frame.dropna(axis=1, how="all")
        if len(frame.index) < min_periods or len(frame.columns) < 2:
            values.append(np.nan)
            continue
        corr = frame.corr(min_periods=min_periods).to_numpy(dtype=float)
        upper = corr[np.triu_indices_from(corr, k=1)]
        clean = upper[np.isfinite(upper)]
        values.append(float(np.mean(clean)) if clean.size else np.nan)
    return pd.Series(values, index=returns.index, dtype=float)


def aggregate_market_features(symbol_indicators: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Aggregate per-symbol indicators into market-wide daily features."""

    frames = [frame.copy() for frame in symbol_indicators.values() if not frame.empty]
    if not frames:
        raise ExternalRegimeClassifierError("No symbol daily indicators available")
    combined = pd.concat(frames, ignore_index=True)
    combined["datetime"] = pd.to_datetime(combined["datetime"])
    by_day = combined.groupby("datetime", sort=True)
    features = pd.DataFrame(index=sorted(combined["datetime"].dropna().unique()))
    features.index.name = "datetime"

    features["number_of_symbols_above_ema50_1d"] = by_day["above_ema50"].sum()
    features["number_of_symbols_above_ema200_1d"] = by_day["above_ema200"].sum()
    features["ema50_ema200_breadth"] = by_day.apply(lambda group: int((pd.to_numeric(group["ema_spread_pct"], errors="coerce") > 0).sum()))
    features["average_ema_spread_pct"] = by_day["ema_spread_pct"].mean()
    features["median_ema_spread_pct"] = by_day["ema_spread_pct"].median()
    features["strong_trend_symbol_count"] = by_day["strong_trend_state"].sum()
    features["trend_efficiency_mean"] = by_day["trend_efficiency"].mean()
    features["trend_efficiency_median"] = by_day["trend_efficiency"].median()
    features["trend_efficiency_dispersion"] = by_day["trend_efficiency"].std(ddof=0)
    features["market_breadth_return_20d"] = by_day.apply(lambda group: int((pd.to_numeric(group["return_20d"], errors="coerce") > 0).sum()))
    features["market_breadth_return_60d"] = by_day.apply(lambda group: int((pd.to_numeric(group["return_60d"], errors="coerce") > 0).sum()))
    features["average_atr_pct"] = by_day["atr_pct"].mean()
    features["median_atr_pct"] = by_day["atr_pct"].median()
    features["atr_pct_dispersion"] = by_day["atr_pct"].std(ddof=0)
    features["realized_volatility_mean"] = by_day["realized_volatility"].mean()
    features["realized_volatility_dispersion"] = by_day["realized_volatility"].std(ddof=0)
    features["high_vol_symbol_count"] = by_day["high_vol_state"].sum()
    features["low_vol_symbol_count"] = by_day["low_vol_state"].sum()
    features["average_drawdown_from_60d_high"] = by_day["drawdown_from_60d_high"].mean()
    features["max_symbol_drawdown_from_60d_high"] = by_day["drawdown_from_60d_high"].min()
    features["rebound_breadth"] = by_day["rebound_state"].sum()
    features["symbols_near_60d_high_count"] = by_day["near_60d_high"].sum()

    returns = combined.pivot(index="datetime", columns="vt_symbol", values="return_1d").sort_index()
    ret20 = combined.pivot(index="datetime", columns="vt_symbol", values="return_20d").sort_index()
    ret60 = combined.pivot(index="datetime", columns="vt_symbol", values="return_60d").sort_index()
    features["average_pairwise_correlation_20d"] = average_pairwise_correlation(returns, 20)
    features["average_pairwise_correlation_60d"] = average_pairwise_correlation(returns, 60)
    features["return_dispersion_20d"] = ret20.std(axis=1, ddof=0)
    features["return_dispersion_60d"] = ret60.std(axis=1, ddof=0)

    return features.reset_index()


def parse_funding_csv(path: Path, vt_symbol: str, timezone_name: str) -> pd.DataFrame:
    """Read one local OKX funding CSV and return timestamp/rate rows."""

    if not path.exists():
        return pd.DataFrame(columns=["datetime", "date", "vt_symbol", "funding_rate"])
    raw = pd.read_csv(path)
    if raw.empty:
        return pd.DataFrame(columns=["datetime", "date", "vt_symbol", "funding_rate"])
    tz = ZoneInfo(timezone_name)
    if "funding_time" in raw.columns and raw["funding_time"].notna().any():
        dt = pd.to_datetime(pd.to_numeric(raw["funding_time"], errors="coerce"), unit="ms", utc=True, errors="coerce")
        dt = dt.dt.tz_convert(tz)
    elif "funding_time_local" in raw.columns:
        dt = pd.to_datetime(raw["funding_time_local"], errors="coerce")
        if getattr(dt.dt, "tz", None) is None:
            dt = dt.dt.tz_localize(tz)
        else:
            dt = dt.dt.tz_convert(tz)
    else:
        dt = pd.to_datetime(raw.get("funding_time_utc"), errors="coerce", utc=True).dt.tz_convert(tz)
    frame = pd.DataFrame(
        {
            "datetime": dt,
            "vt_symbol": vt_symbol,
            "funding_rate": pd.to_numeric(raw.get("funding_rate"), errors="coerce"),
        }
    ).dropna(subset=["datetime", "funding_rate"])
    frame["date"] = frame["datetime"].dt.floor("1D")
    return frame


def load_funding_features(symbols: list[str], funding_dir: Path, start: str, end: str, timezone_name: str) -> pd.DataFrame:
    """Aggregate local funding CSVs into daily market-wide features."""

    frames = []
    for vt_symbol in symbols:
        inst_id = vt_symbol_to_okx_inst_id(vt_symbol)
        path = funding_dir / f"{inst_id}_funding_{start}_{end}.csv"
        frames.append(parse_funding_csv(path, vt_symbol, timezone_name))
    if not frames:
        return pd.DataFrame(columns=["datetime", *FUNDING_FEATURE_COLUMNS])
    raw = pd.concat(frames, ignore_index=True)
    if raw.empty:
        return pd.DataFrame(columns=["datetime", *FUNDING_FEATURE_COLUMNS])
    daily_symbol = raw.groupby(["date", "vt_symbol"], sort=True)["funding_rate"].mean().reset_index()
    by_day = daily_symbol.groupby("date", sort=True)
    features = pd.DataFrame(index=sorted(daily_symbol["date"].dropna().unique()))
    features.index.name = "datetime"
    features["average_funding_rate"] = by_day["funding_rate"].mean()
    features["median_funding_rate"] = by_day["funding_rate"].median()
    features["funding_dispersion"] = by_day["funding_rate"].std(ddof=0)
    features["positive_funding_symbol_count"] = by_day.apply(lambda group: int((pd.to_numeric(group["funding_rate"], errors="coerce") > 0).sum()))
    features["negative_funding_symbol_count"] = by_day.apply(lambda group: int((pd.to_numeric(group["funding_rate"], errors="coerce") < 0).sum()))
    features["extreme_positive_funding_count"] = by_day.apply(lambda group: int((pd.to_numeric(group["funding_rate"], errors="coerce") > 0.0005).sum()))
    features["extreme_negative_funding_count"] = by_day.apply(lambda group: int((pd.to_numeric(group["funding_rate"], errors="coerce") < -0.0005).sum()))
    features["funding_trend_7d"] = features["average_funding_rate"] - features["average_funding_rate"].shift(7)
    features["funding_trend_30d"] = features["average_funding_rate"] - features["average_funding_rate"].shift(30)
    return features.reset_index()


def build_external_regime_feature_dataset(
    symbol_daily: dict[str, pd.DataFrame],
    funding_features: pd.DataFrame,
) -> pd.DataFrame:
    """Build the daily external-regime feature dataset."""

    symbol_indicators = {
        vt_symbol: compute_symbol_daily_indicators(vt_symbol, daily)
        for vt_symbol, daily in symbol_daily.items()
    }
    market = aggregate_market_features(symbol_indicators)
    market["datetime"] = pd.to_datetime(market["datetime"])
    if getattr(market["datetime"].dt, "tz", None) is not None:
        market["funding_join_date"] = market["datetime"].dt.floor("1D")
    else:
        market["funding_join_date"] = market["datetime"].dt.tz_localize(None).dt.floor("1D")

    funding = funding_features.copy()
    if funding.empty:
        for column in FUNDING_FEATURE_COLUMNS:
            funding[column] = np.nan
        funding["datetime"] = market["funding_join_date"].drop_duplicates()
    funding["datetime"] = pd.to_datetime(funding["datetime"])
    if getattr(funding["datetime"].dt, "tz", None) is not None:
        funding["funding_join_date"] = funding["datetime"].dt.floor("1D")
    else:
        funding["funding_join_date"] = funding["datetime"].dt.tz_localize(None).dt.floor("1D")
    merged = market.merge(
        funding[["funding_join_date", *FUNDING_FEATURE_COLUMNS]],
        on="funding_join_date",
        how="left",
    )
    merged = merged.drop(columns=["funding_join_date"])
    for column in [
        "positive_funding_symbol_count",
        "negative_funding_symbol_count",
        "extreme_positive_funding_count",
        "extreme_negative_funding_count",
    ]:
        merged[column] = pd.to_numeric(merged[column], errors="coerce").fillna(0.0)
    merged["split"] = merged["datetime"].map(split_for_timestamp)
    return merged[["datetime", "split", *REGIME_FEATURE_COLUMNS]].sort_values("datetime", kind="stable").reset_index(drop=True)


def assert_no_forbidden_feature_columns(feature_columns: list[str]) -> None:
    """Guard against policy PnL or future information entering classifier features."""

    forbidden = [
        column
        for column in feature_columns
        if any(pattern in column.lower() for pattern in FORBIDDEN_FEATURE_PATTERNS)
    ]
    if forbidden:
        raise ExternalRegimeClassifierError(f"Forbidden classifier feature columns: {forbidden}")


def train_quantile(train: pd.DataFrame, column: str, quantile: float) -> float | None:
    """Compute one train-only quantile."""

    values = pd.to_numeric(train.get(column), errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if values.empty:
        return None
    return float(values.quantile(quantile))


def compute_train_thresholds(feature_dataset: pd.DataFrame) -> dict[str, Any]:
    """Compute all classifier thresholds from train_ext only."""

    assert_no_forbidden_feature_columns(REGIME_FEATURE_COLUMNS)
    train = feature_dataset[feature_dataset["split"] == "train_ext"].copy()
    if train.empty:
        raise ExternalRegimeClassifierError("train_ext feature dataset is empty; cannot compute thresholds")
    q: dict[str, Any] = {}
    for column in [
        "trend_efficiency_mean",
        "average_pairwise_correlation_20d",
        "average_atr_pct",
        "return_dispersion_20d",
        "average_funding_rate",
        "funding_dispersion",
    ]:
        q[column] = {
            "q20": train_quantile(train, column, 0.20),
            "q40": train_quantile(train, column, 0.40),
            "q50": train_quantile(train, column, 0.50),
            "q70": train_quantile(train, column, 0.70),
            "q80": train_quantile(train, column, 0.80),
        }
    return {
        "source_split": "train_ext",
        "thresholds_use_validation_ext": False,
        "thresholds_use_oos_ext": False,
        "train_row_count": int(len(train.index)),
        "quantiles": q,
        "fixed_thresholds": {
            "ema50_ema200_breadth_min": 3,
            "high_vol_symbol_count_min": 3,
            "extreme_positive_funding_count_min": 2,
            "strong_trend_symbol_count_narrow_max": 1,
        },
        "forbidden_feature_patterns": list(FORBIDDEN_FEATURE_PATTERNS),
        "feature_columns": list(REGIME_FEATURE_COLUMNS),
    }


def threshold_value(thresholds: dict[str, Any], column: str, name: str, default: float) -> float:
    """Return a threshold value with a fallback."""

    value = (((thresholds.get("quantiles") or {}).get(column) or {}).get(name))
    finite = finite_float(value)
    return default if finite is None else finite


def generate_regime_labels(feature_dataset: pd.DataFrame, thresholds: dict[str, Any]) -> pd.DataFrame:
    """Generate rule-based classifier labels for every completed daily bar."""

    df = feature_dataset.copy()
    te_q70 = threshold_value(thresholds, "trend_efficiency_mean", "q70", 0.35)
    te_q50 = threshold_value(thresholds, "trend_efficiency_mean", "q50", 0.25)
    te_q40 = threshold_value(thresholds, "trend_efficiency_mean", "q40", 0.20)
    corr_median = threshold_value(thresholds, "average_pairwise_correlation_20d", "q50", 0.30)
    corr_low = threshold_value(thresholds, "average_pairwise_correlation_20d", "q40", 0.20)
    atr_q20 = threshold_value(thresholds, "average_atr_pct", "q20", 0.03)
    atr_q80 = threshold_value(thresholds, "average_atr_pct", "q80", 0.08)
    dispersion_q40 = threshold_value(thresholds, "return_dispersion_20d", "q40", 0.05)
    dispersion_q70 = threshold_value(thresholds, "return_dispersion_20d", "q70", 0.10)
    funding_q20 = threshold_value(thresholds, "average_funding_rate", "q20", -0.0001)
    funding_q70 = threshold_value(thresholds, "average_funding_rate", "q70", 0.0002)
    funding_q80 = threshold_value(thresholds, "average_funding_rate", "q80", 0.0003)
    funding_disp_q70 = threshold_value(thresholds, "funding_dispersion", "q70", 0.0003)

    df["is_trend_friendly"] = (
        (df["trend_efficiency_mean"] > te_q70)
        & (df["ema50_ema200_breadth"] >= 3)
        & (df["average_pairwise_correlation_20d"] > corr_median)
        & (df["average_atr_pct"] < atr_q80)
    )
    df["is_trend_hostile"] = (
        (df["trend_efficiency_mean"] < te_q40)
        & (df["high_vol_symbol_count"] >= 3)
        & (df["return_dispersion_20d"] > dispersion_q70)
        & (df["average_pairwise_correlation_20d"] < corr_low)
    )
    df["is_high_vol_chop"] = (df["high_vol_symbol_count"] >= 3) & (df["trend_efficiency_mean"] < te_q50)
    df["is_broad_uptrend"] = (
        (df["ema50_ema200_breadth"] >= 3)
        & (df["market_breadth_return_20d"] >= 3)
        & (df["average_ema_spread_pct"] > 0)
    )
    df["is_broad_downtrend"] = (
        (df["ema50_ema200_breadth"] <= 2)
        & (df["market_breadth_return_20d"] <= 2)
        & (df["average_ema_spread_pct"] < 0)
    )
    df["is_narrow_single_symbol_trend"] = (
        (df["strong_trend_symbol_count"] <= 1)
        & (df["return_dispersion_20d"] > dispersion_q70)
    )
    df["is_funding_overheated"] = (
        (df["average_funding_rate"] >= funding_q80)
        & (df["extreme_positive_funding_count"] >= 2)
    )
    df["is_funding_supportive"] = (
        (df["average_funding_rate"] >= funding_q20)
        & (df["average_funding_rate"] <= funding_q70)
        & (df["funding_dispersion"] <= funding_disp_q70)
        & (~df["is_funding_overheated"])
    )
    df["is_compression"] = (
        (df["average_atr_pct"] < atr_q20)
        & (df["return_dispersion_20d"] < dispersion_q40)
        & (df["trend_efficiency_mean"] < te_q50)
    )

    priority = [
        ("trend_friendly", "is_trend_friendly"),
        ("trend_hostile", "is_trend_hostile"),
        ("high_vol_chop", "is_high_vol_chop"),
        ("funding_overheated", "is_funding_overheated"),
        ("broad_uptrend", "is_broad_uptrend"),
        ("broad_downtrend", "is_broad_downtrend"),
        ("narrow_single_symbol_trend", "is_narrow_single_symbol_trend"),
        ("funding_supportive", "is_funding_supportive"),
        ("compression", "is_compression"),
    ]
    labels = []
    for _, row in df.iterrows():
        label = "neutral"
        for name, column in priority:
            if bool(row.get(column)):
                label = name
                break
        labels.append(label)
    df["regime_label"] = labels
    for column in LABEL_COLUMNS:
        df[column] = df[column].fillna(False).astype(bool)
    return df[["datetime", "split", "regime_label", *LABEL_COLUMNS, *REGIME_FEATURE_COLUMNS]]


def build_regime_label_distribution(labels: pd.DataFrame) -> pd.DataFrame:
    """Build distribution table for each boolean regime label by split."""

    rows: list[dict[str, Any]] = []
    for split, group in labels.groupby("split", sort=True):
        total = int(len(group.index))
        if split == "outside":
            continue
        for column in LABEL_COLUMNS:
            count = int(group[column].sum())
            rows.append(
                {
                    "split": split,
                    "label": column.replace("is_", ""),
                    "true_count": count,
                    "total_count": total,
                    "pct": float(count / total) if total else 0.0,
                }
            )
    return pd.DataFrame(rows)


def read_trade_file(path: Path, split: str) -> pd.DataFrame:
    """Read one Trend V3 extended trade file."""

    if not path.exists():
        return pd.DataFrame()
    trades = pd.read_csv(path)
    if trades.empty:
        return trades
    trades.insert(0, "split", split)
    return trades


def load_v3_extended_trades(trend_v3_dir: Path) -> pd.DataFrame:
    """Load all split trade files."""

    frames = []
    for split in ("train_ext", "validation_ext", "oos_ext"):
        frames.append(read_trade_file(trend_v3_dir / split / "trend_v3_trades.csv", split))
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        raise ExternalRegimeClassifierError(f"No Trend V3 extended trade files found under {trend_v3_dir}")
    trades = pd.concat(frames, ignore_index=True)
    for column in ["entry_time", "exit_time"]:
        trades[column] = pd.to_datetime(trades[column], errors="coerce")
    trades["net_pnl"] = pd.to_numeric(trades.get("net_pnl"), errors="coerce")
    if "no_cost_net_pnl" not in trades.columns and "no_cost_pnl" in trades.columns:
        trades["no_cost_net_pnl"] = trades["no_cost_pnl"]
    trades["no_cost_net_pnl"] = pd.to_numeric(trades.get("no_cost_net_pnl"), errors="coerce")
    return trades


def load_actual_funding_adjustments(path: Path) -> pd.DataFrame:
    """Load optional per-trade actual funding adjustment file."""

    if not path.exists():
        return pd.DataFrame()
    adjustments = pd.read_csv(path)
    if adjustments.empty:
        return adjustments
    for column in ["entry_time", "exit_time"]:
        adjustments[column] = pd.to_datetime(adjustments[column], errors="coerce")
    rename = {
        "funding_adjusted_net_pnl_conservative": "funding_adjusted_net_pnl",
        "original_net_pnl": "funding_original_net_pnl",
    }
    adjustments = adjustments.rename(columns=rename)
    keep = [
        column
        for column in [
            "split",
            "policy_name",
            "symbol",
            "direction",
            "entry_time",
            "exit_time",
            "funding_adjusted_net_pnl",
            "funding_adjusted_net_pnl_signed",
            "funding_data_available",
        ]
        if column in adjustments.columns
    ]
    return adjustments[keep].copy()


def merge_funding_adjustments(trades: pd.DataFrame, adjustments: pd.DataFrame) -> pd.DataFrame:
    """Merge optional actual funding-adjusted PnL onto trade rows."""

    result = trades.copy()
    result["funding_adjusted_net_pnl"] = np.nan
    if adjustments.empty:
        return result
    keys = ["split", "policy_name", "symbol", "direction", "entry_time", "exit_time"]
    merged = result.merge(adjustments, on=keys, how="left", suffixes=("", "_funding"))
    if "funding_adjusted_net_pnl_funding" in merged.columns:
        merged["funding_adjusted_net_pnl"] = merged["funding_adjusted_net_pnl_funding"]
        merged = merged.drop(columns=["funding_adjusted_net_pnl_funding"])
    return merged


def build_feature_snapshot(row: pd.Series) -> str:
    """Serialize regime feature values for one trade attribution row."""

    snapshot = {}
    for column in REGIME_FEATURE_COLUMNS:
        value = finite_float(row.get(column))
        snapshot[column] = value
    return json.dumps(snapshot, ensure_ascii=False, sort_keys=True)


def align_trades_to_regime(trades: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """Align each trade entry_time to the most recent completed daily regime row."""

    if trades.empty:
        return pd.DataFrame()
    drop_existing = ["regime_label", "regime_features_snapshot", *LABEL_COLUMNS, *REGIME_FEATURE_COLUMNS]
    left = trades.copy().drop(columns=drop_existing, errors="ignore").sort_values("entry_time", kind="stable").reset_index(drop=True)
    right = labels.copy().drop(columns=["split"], errors="ignore").sort_values("datetime", kind="stable").reset_index(drop=True)
    left["entry_time"] = pd.to_datetime(left["entry_time"])
    right["datetime"] = pd.to_datetime(right["datetime"])
    left["_entry_merge_key"] = pd.to_datetime(left["entry_time"], utc=True)
    right["_regime_merge_key"] = pd.to_datetime(right["datetime"], utc=True)
    left = left.sort_values("_entry_merge_key", kind="stable")
    right = right.sort_values("_regime_merge_key", kind="stable")
    aligned = pd.merge_asof(left, right, left_on="_entry_merge_key", right_on="_regime_merge_key", direction="backward")
    aligned = aligned.drop(columns=["_entry_merge_key", "_regime_merge_key"], errors="ignore")
    aligned["regime_label"] = aligned["regime_label"].fillna("missing_regime")
    for column in LABEL_COLUMNS:
        aligned[column] = aligned[column].fillna(False).astype(bool)
    aligned["regime_features_snapshot"] = aligned.apply(build_feature_snapshot, axis=1)
    if "funding_adjusted_net_pnl" not in aligned.columns:
        aligned["funding_adjusted_net_pnl"] = np.nan
    output_columns = [
        "split",
        "policy_name",
        "symbol",
        "direction",
        "entry_time",
        "exit_time",
        "net_pnl",
        "no_cost_net_pnl",
        "funding_adjusted_net_pnl",
        "regime_label",
        "regime_features_snapshot",
        "is_trend_friendly",
        "is_trend_hostile",
        "is_funding_overheated",
        "is_high_vol_chop",
        "is_broad_uptrend",
        "is_broad_downtrend",
        "is_narrow_single_symbol_trend",
        "is_funding_supportive",
        "is_compression",
    ]
    return aligned[output_columns].copy()


def aggregate_performance(group: pd.DataFrame, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """Aggregate trade performance for one group."""

    payload: dict[str, Any] = dict(extra or {})
    payload["trade_count"] = int(len(group.index))
    payload["net_pnl"] = safe_sum(group["net_pnl"]) if "net_pnl" in group else 0.0
    payload["no_cost_net_pnl"] = safe_sum(group["no_cost_net_pnl"]) if "no_cost_net_pnl" in group else 0.0
    if "funding_adjusted_net_pnl" in group and group["funding_adjusted_net_pnl"].notna().any():
        payload["funding_adjusted_net_pnl"] = safe_sum(group["funding_adjusted_net_pnl"])
    else:
        payload["funding_adjusted_net_pnl"] = None
    pnl = pd.to_numeric(group.get("net_pnl"), errors="coerce")
    payload["win_rate"] = float((pnl > 0).mean()) if len(pnl.index) else None
    return payload


def build_policy_performance_by_regime(attribution: pd.DataFrame) -> pd.DataFrame:
    """Aggregate policy performance by split and regime label."""

    rows = []
    if attribution.empty:
        return pd.DataFrame()
    for (split, policy, regime), group in attribution.groupby(["split", "policy_name", "regime_label"], dropna=False):
        rows.append(aggregate_performance(group, {"split": split, "policy_name": policy, "regime_label": regime}))
    return pd.DataFrame(rows)


def build_split_performance_by_regime(attribution: pd.DataFrame) -> pd.DataFrame:
    """Aggregate split-level performance by regime label."""

    rows = []
    if attribution.empty:
        return pd.DataFrame()
    for (split, regime), group in attribution.groupby(["split", "regime_label"], dropna=False):
        rows.append(aggregate_performance(group, {"split": split, "regime_label": regime}))
    return pd.DataFrame(rows)


def apply_filter(frame: pd.DataFrame, filter_name: str) -> pd.DataFrame:
    """Apply one offline classifier filter to already executed trades."""

    if filter_name == "original_all":
        return frame.copy()
    if filter_name == "keep_trend_friendly":
        return frame[frame["is_trend_friendly"]].copy()
    if filter_name == "exclude_hostile_chop_overheated":
        return frame[
            ~(frame["is_trend_hostile"] | frame["is_high_vol_chop"] | frame["is_funding_overheated"])
        ].copy()
    if filter_name == "keep_trend_friendly_exclude_funding_overheated":
        return frame[frame["is_trend_friendly"] & ~frame["is_funding_overheated"]].copy()
    if filter_name == "exclude_funding_overheated":
        return frame[~frame["is_funding_overheated"]].copy()
    raise ExternalRegimeClassifierError(f"unknown filter_name: {filter_name}")


def concentration_metrics(trades: pd.DataFrame) -> dict[str, Any]:
    """Compute symbol and top-trade concentration on a trade subset."""

    if trades.empty:
        return {
            "largest_symbol_pnl_share": None,
            "top_5pct_trade_pnl_contribution": None,
            "active_symbol_count": 0,
        }
    pnl = pd.to_numeric(trades["net_pnl"], errors="coerce").fillna(0.0)
    by_symbol = trades.assign(_pnl=pnl).groupby("symbol")["_pnl"].sum()
    active_symbol_count = int(trades["symbol"].dropna().astype(str).nunique()) if "symbol" in trades.columns else 0
    total_positive = float(by_symbol[by_symbol > 0].sum())
    if total_positive > 0:
        largest_symbol = float(by_symbol.max() / total_positive)
    else:
        denom = float(by_symbol.abs().sum())
        largest_symbol = float(by_symbol.abs().max() / denom) if denom else None

    sorted_pnl = pnl.sort_values(ascending=False)
    total_pnl = float(pnl.sum())
    top_n = max(1, int(math.ceil(len(sorted_pnl.index) * 0.05)))
    top_sum = float(sorted_pnl.head(top_n).sum())
    top_share = float(top_sum / total_pnl) if total_pnl != 0 else None
    return {
        "largest_symbol_pnl_share": largest_symbol,
        "top_5pct_trade_pnl_contribution": top_share,
        "active_symbol_count": active_symbol_count,
    }


def funding_adjusted_gate(oos_funding: float | None, oos_no_cost: float) -> bool:
    """Return whether funding-adjusted OOS PnL is non-negative when available."""

    if oos_funding is None:
        return True
    return bool(oos_funding >= 0)


def strict_gate_rejected_reasons(row: dict[str, Any]) -> list[str]:
    """Return strict stable-like rejection reasons for one filter experiment row."""

    reasons: list[str] = []
    if not (finite_float(row.get("train_ext_no_cost_net_pnl")) is not None and float(row["train_ext_no_cost_net_pnl"]) > 0):
        reasons.append("train_no_cost_net_pnl_not_positive")
    if not (
        finite_float(row.get("validation_ext_no_cost_net_pnl")) is not None
        and float(row["validation_ext_no_cost_net_pnl"]) > 0
    ):
        reasons.append("validation_no_cost_net_pnl_not_positive")
    if not (finite_float(row.get("oos_ext_no_cost_net_pnl")) is not None and float(row["oos_ext_no_cost_net_pnl"]) > 0):
        reasons.append("oos_no_cost_net_pnl_not_positive")
    if not (finite_float(row.get("oos_ext_net_pnl")) is not None and float(row["oos_ext_net_pnl"]) >= 0):
        reasons.append("oos_cost_aware_net_pnl_negative")
    oos_funding = finite_float(row.get("oos_ext_funding_adjusted_net_pnl"))
    if oos_funding is not None and oos_funding < 0:
        reasons.append("oos_funding_adjusted_net_pnl_negative")
    for split in ("train_ext", "validation_ext", "oos_ext"):
        count = finite_float(row.get(f"{split}_trade_count"))
        if count is None or count < 10:
            reasons.append(f"{split}_trade_count_under_10")
    largest_share = finite_float(row.get("largest_symbol_pnl_share"))
    if largest_share is None or largest_share > 0.70:
        reasons.append("oos_largest_symbol_pnl_share_over_0p7")
    top_share = finite_float(row.get("top_5pct_trade_pnl_contribution"))
    if top_share is None or top_share > 0.80:
        reasons.append("oos_top_5pct_trade_pnl_contribution_over_0p8")
    active_symbol_count = finite_float(row.get("active_symbol_count"))
    if active_symbol_count is None or active_symbol_count < 2:
        reasons.append("oos_not_enough_active_symbols")
    return reasons


def build_classifier_filter_experiment(attribution: pd.DataFrame) -> pd.DataFrame:
    """Run offline post-trade classifier filter experiment for each V3 policy."""

    rows: list[dict[str, Any]] = []
    policies = sorted(str(policy) for policy in attribution["policy_name"].dropna().unique())
    for policy in policies:
        policy_frame = attribution[attribution["policy_name"] == policy].copy()
        for filter_name in FILTER_DEFINITIONS:
            filtered = apply_filter(policy_frame, filter_name)
            by_split = {split: filtered[filtered["split"] == split] for split in ("train_ext", "validation_ext", "oos_ext")}
            row: dict[str, Any] = {
                "filter_name": filter_name,
                "policy_name": policy,
                "research_only": True,
                "requires_pretrade_implementation_audit": True,
                "not_tradable": True,
                "filter_description": FILTER_DEFINITIONS[filter_name],
            }
            for split, frame in by_split.items():
                row[f"{split}_trade_count"] = int(len(frame.index))
                row[f"{split}_no_cost_net_pnl"] = safe_sum(frame["no_cost_net_pnl"]) if not frame.empty else 0.0
                row[f"{split}_net_pnl"] = safe_sum(frame["net_pnl"]) if not frame.empty else 0.0
                if frame["funding_adjusted_net_pnl"].notna().any():
                    row[f"{split}_funding_adjusted_net_pnl"] = safe_sum(frame["funding_adjusted_net_pnl"])
                else:
                    row[f"{split}_funding_adjusted_net_pnl"] = None
            concentrations = concentration_metrics(by_split["oos_ext"])
            row.update(concentrations)
            funding_ok = funding_adjusted_gate(finite_float(row.get("oos_ext_funding_adjusted_net_pnl")), float(row["oos_ext_no_cost_net_pnl"]))
            split_counts_ok = all(int(row[f"{split}_trade_count"]) >= 10 for split in ("train_ext", "validation_ext", "oos_ext"))
            rejected_reasons = strict_gate_rejected_reasons(row)
            row["stable_candidate_like"] = bool(not rejected_reasons)
            row["funding_adjusted_gate_passed"] = funding_ok
            row["min_split_trade_count_passed"] = split_counts_ok
            row["strict_rejected_reasons"] = ";".join(rejected_reasons)
            rows.append(row)
    return pd.DataFrame(rows)


def distribution_stability(distribution: pd.DataFrame, label: str) -> dict[str, Any]:
    """Return simple split-distribution stability metadata for one label."""

    subset = distribution[distribution["label"] == label]
    by_split = {str(row["split"]): float(row["pct"]) for _, row in subset.iterrows()}
    values = list(by_split.values())
    max_range = float(max(values) - min(values)) if values else None
    return {
        "label": label,
        "pct_by_split": by_split,
        "max_pct_range": max_range,
        "stable_distribution": bool(max_range is not None and max_range <= 0.25),
    }


def build_research_summary(
    labels: pd.DataFrame,
    distribution: pd.DataFrame,
    attribution: pd.DataFrame,
    filter_experiment: pd.DataFrame,
    thresholds: dict[str, Any],
) -> dict[str, Any]:
    """Build decision summary and report facts."""

    stable_rows = filter_experiment[
        (filter_experiment["stable_candidate_like"] == True)
        & (filter_experiment["filter_name"] != "original_all")
    ] if not filter_experiment.empty else pd.DataFrame()
    best_candidate = None
    if not stable_rows.empty:
        ranked = stable_rows.sort_values("oos_ext_net_pnl", ascending=False, kind="stable")
        best_candidate = ranked.iloc[0].to_dict()

    hostile_mask = attribution["is_trend_hostile"] | attribution["is_high_vol_chop"] if not attribution.empty else pd.Series(dtype=bool)
    total_losses = pd.to_numeric(attribution.get("net_pnl"), errors="coerce").clip(upper=0.0).abs().sum() if not attribution.empty else 0.0
    hostile_losses = (
        pd.to_numeric(attribution.loc[hostile_mask, "net_pnl"], errors="coerce").clip(upper=0.0).abs().sum()
        if not attribution.empty
        else 0.0
    )
    hostile_loss_share = float(hostile_losses / total_losses) if total_losses else None

    ema_policy = "v3_1d_ema_50_200_atr5"
    ema_rows = filter_experiment[filter_experiment["policy_name"] == ema_policy] if not filter_experiment.empty else pd.DataFrame()
    ema_original = ema_rows[ema_rows["filter_name"] == "original_all"].iloc[0].to_dict() if not ema_rows[ema_rows["filter_name"] == "original_all"].empty else None
    ema_trend = ema_rows[ema_rows["filter_name"] == "keep_trend_friendly"].iloc[0].to_dict() if not ema_rows[ema_rows["filter_name"] == "keep_trend_friendly"].empty else None
    funding_exclude_rows = filter_experiment[filter_experiment["filter_name"] == "exclude_funding_overheated"] if not filter_experiment.empty else pd.DataFrame()
    funding_filter_improved_count = 0
    if not funding_exclude_rows.empty:
        originals = filter_experiment[filter_experiment["filter_name"] == "original_all"].set_index("policy_name")
        for _, row in funding_exclude_rows.iterrows():
            original = originals.loc[row["policy_name"]] if row["policy_name"] in originals.index else None
            if original is not None and row["oos_ext_net_pnl"] > original["oos_ext_net_pnl"]:
                funding_filter_improved_count += 1

    can_enter = bool(best_candidate is not None)
    reason = (
        "At least one research-only filter passed all train/validation/oos no-cost, OOS cost, funding, count, and concentration gates."
        if can_enter
        else "No filtered policy passed all research-only gates across train_ext, validation_ext, and oos_ext."
    )
    return {
        "can_enter_research_only_v3_1_classifier_experiment": can_enter,
        "strategy_development_allowed": False,
        "demo_live_allowed": False,
        "research_only": True,
        "not_tradable": True,
        "requires_pretrade_implementation_audit": True,
        "reason": reason,
        "best_candidate": to_jsonable(best_candidate),
        "threshold_source_split": thresholds.get("source_split"),
        "thresholds_use_validation_ext": bool(thresholds.get("thresholds_use_validation_ext")),
        "thresholds_use_oos_ext": bool(thresholds.get("thresholds_use_oos_ext")),
        "feature_columns": REGIME_FEATURE_COLUMNS,
        "forbidden_feature_patterns": list(FORBIDDEN_FEATURE_PATTERNS),
        "distribution_stability": {
            "trend_friendly": distribution_stability(distribution, "trend_friendly"),
            "trend_hostile": distribution_stability(distribution, "trend_hostile"),
            "funding_overheated": distribution_stability(distribution, "funding_overheated"),
        },
        "hostile_or_high_vol_chop_loss_share": hostile_loss_share,
        "losses_mainly_in_hostile_or_high_vol_chop": bool(hostile_loss_share is not None and hostile_loss_share >= 0.50),
        "v3_1d_ema_50_200_atr5_original": to_jsonable(ema_original),
        "v3_1d_ema_50_200_atr5_trend_friendly": to_jsonable(ema_trend),
        "funding_overheated_filter_improved_policy_count": int(funding_filter_improved_count),
        "stable_candidate_like_count": int(len(stable_rows.index)),
    }


def format_number(value: Any, digits: int = 4) -> str:
    """Format optional numeric report values."""

    number = finite_float(value)
    if number is None:
        return ""
    return f"{number:.{digits}f}"


def render_report(
    summary: dict[str, Any],
    distribution: pd.DataFrame,
    filter_experiment: pd.DataFrame,
) -> str:
    """Render Markdown report."""

    def distribution_line(label: str) -> str:
        stability = summary["distribution_stability"][label]
        by_split = stability["pct_by_split"]
        return (
            f"- {label}: train_ext={format_number(by_split.get('train_ext'), 3)}, "
            f"validation_ext={format_number(by_split.get('validation_ext'), 3)}, "
            f"oos_ext={format_number(by_split.get('oos_ext'), 3)}, "
            f"stable={str(bool(stability['stable_distribution'])).lower()}"
        )

    stable = filter_experiment[filter_experiment["stable_candidate_like"] == True] if not filter_experiment.empty else pd.DataFrame()
    stable_text = "true" if not stable.empty else "false"
    ema_original = summary.get("v3_1d_ema_50_200_atr5_original") or {}
    ema_trend = summary.get("v3_1d_ema_50_200_atr5_trend_friendly") or {}
    ema_improved = (
        finite_float(ema_trend.get("oos_ext_net_pnl")) is not None
        and finite_float(ema_original.get("oos_ext_net_pnl")) is not None
        and float(ema_trend["oos_ext_net_pnl"]) > float(ema_original["oos_ext_net_pnl"])
    )
    concentration_problem = True
    if not stable.empty:
        concentration_problem = bool(
            (stable["largest_symbol_pnl_share"] > 0.70).any()
            or (stable["top_5pct_trade_pnl_contribution"] > 0.80).any()
        )

    top_rows = filter_experiment.sort_values("oos_ext_net_pnl", ascending=False, kind="stable").head(10)
    table = [
        "| filter_name | policy_name | train no-cost | validation no-cost | oos no-cost | oos cost | oos funding-adjusted | stable_candidate_like |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for _, row in top_rows.iterrows():
        table.append(
            f"| {row['filter_name']} | {row['policy_name']} | "
            f"{format_number(row['train_ext_no_cost_net_pnl'])} | "
            f"{format_number(row['validation_ext_no_cost_net_pnl'])} | "
            f"{format_number(row['oos_ext_no_cost_net_pnl'])} | "
            f"{format_number(row['oos_ext_net_pnl'])} | "
            f"{format_number(row.get('oos_ext_funding_adjusted_net_pnl'))} | "
            f"{str(bool(row['stable_candidate_like'])).lower()} |"
        )

    return (
        "# External Regime Classifier Research\n\n"
        "## Guardrails\n"
        "- research_only=true\n"
        "- not_tradable=true\n"
        "- strategy_development_allowed=false\n"
        "- demo_live_allowed=false\n"
        "- classifier features exclude policy PnL, future returns, and future regime information.\n"
        "- thresholds_source_split=train_ext\n"
        "- validation_ext and oos_ext are validation only.\n\n"
        "## Regime Distribution\n"
        f"{distribution_line('trend_friendly')}\n"
        f"{distribution_line('trend_hostile')}\n"
        f"{distribution_line('funding_overheated')}\n\n"
        "## Required Questions\n"
        "1. 是否存在独立于 V3 PnL 的 trend-friendly regime？\n"
        "   - 是。label 由 market/funding 特征和 train_ext 分位数生成，不使用 V3 PnL。\n"
        "2. trend-friendly regime 在 train / validation / oos 的分布是否稳定？\n"
        f"   - {str(bool(summary['distribution_stability']['trend_friendly']['stable_distribution'])).lower()}。\n"
        "3. 当前 V3 trades 是否主要亏在 trend_hostile / high_vol_chop？\n"
        f"   - {str(bool(summary['losses_mainly_in_hostile_or_high_vol_chop'])).lower()}，loss_share={format_number(summary['hostile_or_high_vol_chop_loss_share'], 3)}。\n"
        "4. v3_1d_ema_50_200_atr5 在 trend-friendly regime 下是否改善？\n"
        f"   - {str(bool(ema_improved)).lower()}。\n"
        "5. 过滤 funding_overheated 后是否改善？\n"
        f"   - improved_policy_count={summary['funding_overheated_filter_improved_policy_count']}。\n"
        "6. 是否有任何 filtered result 在 train/validation/oos 都为正？\n"
        f"   - {stable_text}。\n"
        "7. 过滤后是否仍有 top trade concentration？\n"
        f"   - {str(bool(concentration_problem)).lower()}。\n"
        "8. 过滤后是否仍有 symbol concentration？\n"
        f"   - {str(bool(concentration_problem)).lower()}。\n"
        "9. 是否允许进入 research-only V3.1 classifier-filtered experiment？\n"
        f"   - can_enter_research_only_v3_1_classifier_experiment={str(bool(summary['can_enter_research_only_v3_1_classifier_experiment'])).lower()}。\n"
        "10. 是否允许 Strategy V3 / demo / live？\n"
        "   - strategy_development_allowed=false\n"
        "   - demo_live_allowed=false\n\n"
        "## Filter Experiment Top Rows\n"
        f"{chr(10).join(table)}\n\n"
        "## Decision\n"
        f"- can_enter_research_only_v3_1_classifier_experiment={str(bool(summary['can_enter_research_only_v3_1_classifier_experiment'])).lower()}\n"
        "- strategy_development_allowed=false\n"
        "- demo_live_allowed=false\n"
        f"- reason={summary['reason']}\n"
    )


def write_json(path: Path, payload: Any) -> None:
    """Write JSON payload."""

    path.write_text(json.dumps(to_jsonable(payload), ensure_ascii=False, indent=2), encoding="utf-8")


def write_outputs(
    output_dir: Path,
    feature_dataset: pd.DataFrame,
    thresholds: dict[str, Any],
    labels: pd.DataFrame,
    distribution: pd.DataFrame,
    attribution: pd.DataFrame,
    policy_perf: pd.DataFrame,
    split_perf: pd.DataFrame,
    filter_experiment: pd.DataFrame,
    summary: dict[str, Any],
) -> None:
    """Write all required research artifacts."""

    output_dir.mkdir(parents=True, exist_ok=True)
    feature_dataset.to_csv(output_dir / "external_regime_feature_dataset.csv", index=False)
    write_json(output_dir / "external_regime_thresholds.json", thresholds)
    labels.to_csv(output_dir / "external_regime_labels.csv", index=False)
    distribution.to_csv(output_dir / "regime_label_distribution.csv", index=False)
    attribution.to_csv(output_dir / "trade_regime_classifier_attribution.csv", index=False)
    policy_perf.to_csv(output_dir / "policy_performance_by_regime.csv", index=False)
    split_perf.to_csv(output_dir / "split_performance_by_regime.csv", index=False)
    filter_experiment.to_csv(output_dir / "classifier_filter_experiment.csv", index=False)
    write_json(output_dir / "external_regime_classifier_summary.json", summary)
    (output_dir / "external_regime_classifier_report.md").write_text(
        render_report(summary, distribution, filter_experiment),
        encoding="utf-8",
    )


def load_market_features_from_db(
    symbols: list[str],
    database_path: Path,
    history_range: HistoryRange,
    timezone_name: str,
    data_check_strict: bool,
) -> tuple[dict[str, pd.DataFrame], list[dict[str, Any]]]:
    """Load market data and return daily OHLCV frames plus coverage metadata."""

    daily_by_symbol: dict[str, pd.DataFrame] = {}
    coverage_rows: list[dict[str, Any]] = []
    for vt_symbol in symbols:
        bars = load_symbol_1m_bars(vt_symbol, database_path, history_range, timezone_name)
        coverage = validate_1m_coverage(vt_symbol, bars, history_range)
        coverage_rows.append(coverage)
        if data_check_strict and not coverage["coverage_complete"]:
            raise ExternalRegimeClassifierError(f"market data incomplete for {vt_symbol}: {coverage}")
        daily_by_symbol[vt_symbol] = resample_daily_ohlcv(bars)
    return daily_by_symbol, coverage_rows


def run_research(
    *,
    symbols: list[str],
    start: str,
    end: str,
    timezone_name: str,
    database_path: Path,
    funding_dir: Path,
    trend_v3_dir: Path,
    actual_funding_trades: Path,
    output_dir: Path,
    data_check_strict: bool = False,
) -> ResearchOutputs:
    """Run the full research-only classifier workflow."""

    history_range = parse_research_range(start, end, timezone_name)
    symbol_daily, coverage_rows = load_market_features_from_db(
        symbols,
        database_path,
        history_range,
        timezone_name,
        data_check_strict,
    )
    funding_features = load_funding_features(symbols, funding_dir, start, end, timezone_name)
    feature_dataset = build_external_regime_feature_dataset(symbol_daily, funding_features)
    thresholds = compute_train_thresholds(feature_dataset)
    labels = generate_regime_labels(feature_dataset, thresholds)
    distribution = build_regime_label_distribution(labels)
    trades = load_v3_extended_trades(trend_v3_dir)
    adjustments = load_actual_funding_adjustments(actual_funding_trades)
    trades = merge_funding_adjustments(trades, adjustments)
    attribution = align_trades_to_regime(trades, labels)
    policy_perf = build_policy_performance_by_regime(attribution)
    split_perf = build_split_performance_by_regime(attribution)
    filter_experiment = build_classifier_filter_experiment(attribution)
    summary = build_research_summary(labels, distribution, attribution, filter_experiment, thresholds)
    summary.update(
        {
            "symbols": symbols,
            "start": start,
            "end": end,
            "timezone": timezone_name,
            "market_data_coverage": coverage_rows,
            "output_dir": str(output_dir),
            "required_output_files": REQUIRED_OUTPUT_FILES,
        }
    )
    write_outputs(
        output_dir,
        feature_dataset,
        thresholds,
        labels,
        distribution,
        attribution,
        policy_perf,
        split_perf,
        filter_experiment,
        summary,
    )
    return ResearchOutputs(
        output_dir=output_dir,
        feature_dataset=feature_dataset,
        thresholds=thresholds,
        labels=labels,
        trade_attribution=attribution,
        filter_experiment=filter_experiment,
        summary=summary,
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""

    args = parse_args(argv)
    ensure_headless_runtime()
    outputs = run_research(
        symbols=parse_symbol_list(args.symbols),
        start=args.start,
        end=args.end,
        timezone_name=args.timezone,
        database_path=resolve_path(args.database_path),
        funding_dir=resolve_path(args.funding_dir),
        trend_v3_dir=resolve_path(args.trend_v3_dir),
        actual_funding_trades=resolve_path(args.actual_funding_trades),
        output_dir=resolve_path(args.output_dir),
        data_check_strict=bool(args.data_check_strict),
    )
    compact = {
        "can_enter_research_only_v3_1_classifier_experiment": outputs.summary[
            "can_enter_research_only_v3_1_classifier_experiment"
        ],
        "strategy_development_allowed": outputs.summary["strategy_development_allowed"],
        "demo_live_allowed": outputs.summary["demo_live_allowed"],
        "stable_candidate_like_count": outputs.summary["stable_candidate_like_count"],
        "output_dir": str(outputs.output_dir),
    }
    print_json_block("External regime classifier research summary:", outputs.summary if args.json else compact)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
