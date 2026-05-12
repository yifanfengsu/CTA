#!/usr/bin/env python3
"""Research-only Trend Health State Exit diagnostics."""

from __future__ import annotations

import argparse
import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

import research_trend_capture_exit_convexity as tce
from common_runtime import PROJECT_ROOT, ensure_headless_runtime, log_event, print_json_block, setup_logging, to_jsonable
from history_time_utils import DEFAULT_TIMEZONE


DEFAULT_TREND_MAP_DIR = PROJECT_ROOT / "reports" / "research" / "trend_opportunity_map"
DEFAULT_TREND_V3_DIR = PROJECT_ROOT / "reports" / "research" / "trend_following_v3_extended"
DEFAULT_CAPTURE_EXIT_DIR = PROJECT_ROOT / "reports" / "research" / "trend_capture_exit_convexity"
DEFAULT_FUNDING_DIR = PROJECT_ROOT / "data" / "funding" / "okx"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "reports" / "research" / "trend_health_state_exit"
DEFAULT_DATABASE_PATH = PROJECT_ROOT / ".vntrader" / "database.db"
DEFAULT_SYMBOLS = [
    "BTCUSDT_SWAP_OKX.GLOBAL",
    "ETHUSDT_SWAP_OKX.GLOBAL",
    "SOLUSDT_SWAP_OKX.GLOBAL",
    "LINKUSDT_SWAP_OKX.GLOBAL",
    "DOGEUSDT_SWAP_OKX.GLOBAL",
]
DEFAULT_TIMEFRAMES = ["4h", "1d"]
DEFAULT_START = "2023-01-01"
DEFAULT_END = "2026-03-31"

SPLITS = tce.SPLITS
TIMEFRAME_MINUTES = tce.TIMEFRAME_MINUTES
NON_ORACLE_VARIANTS = [
    "original_exit",
    "health_ema20_core",
    "health_ema50_core",
    "health_no_energy",
    "health_drawdown_only",
    "health_energy_confirmed",
]
EXIT_VARIANTS = NON_ORACLE_VARIANTS + ["oracle_hold_to_trend_end"]
ORACLE_VARIANT = "oracle_hold_to_trend_end"

REQUIRED_OUTPUT_FILES = [
    "health_exit_summary.json",
    "health_exit_report.md",
    "health_exit_variant_trades.csv",
    "health_exit_variant_summary.csv",
    "health_exit_by_policy.csv",
    "health_exit_by_symbol.csv",
    "health_exit_by_split.csv",
    "health_exit_concentration.csv",
    "health_exit_vs_original.csv",
    "health_exit_rejected_variants.csv",
    "health_exit_funding_summary.csv",
    "data_quality.json",
]

TRADE_COLUMNS = [
    "trade_id",
    "source_policy",
    "symbol",
    "timeframe",
    "direction",
    "split",
    "entry_time",
    "entry_price",
    "original_exit_time",
    "variant_exit_time",
    "variant_exit_price",
    "exit_variant",
    "exit_reason",
    "hold_bars",
    "original_hold_bars",
    "no_cost_pnl",
    "cost_aware_pnl",
    "funding_adjusted_pnl",
    "funding_events_count",
    "efficiency_at_exit",
    "energy_at_exit",
    "drawdown_at_exit",
    "health_score_at_exit",
    "max_health_score",
    "min_health_score",
    "avg_health_score",
    "captured_fraction",
    "early_exit_flag",
    "late_entry_flag",
    "oracle",
]


class TrendHealthExitResearchError(Exception):
    """Raised when health-state exit research cannot continue."""


@dataclass(frozen=True, slots=True)
class HealthVariantConfig:
    """Health-state exit variant configuration."""

    name: str
    ema_column: str = "ema20"
    efficiency_threshold: float = 0.5
    energy_threshold: float = 0.8
    drawdown_soft_atr: float = 3.0
    drawdown_hard_atr: float = 4.0
    patience_bars: int = 2
    max_hold_bars: int = 50
    mode: str = "core"


@dataclass(frozen=True, slots=True)
class HealthExitSimulation:
    """One simulated exit result."""

    exit_time: pd.Timestamp
    exit_price: float
    exit_reason: str
    hold_bars: int
    efficiency_at_exit: float | None
    energy_at_exit: float | None
    drawdown_at_exit: float | None
    health_score_at_exit: int | None
    max_health_score: int | None
    min_health_score: int | None
    avg_health_score: float | None


@dataclass(frozen=True, slots=True)
class ResearchOutputs:
    """Generated outputs for tests and CLI reporting."""

    output_dir: Path
    summary: dict[str, Any]
    variant_trades: pd.DataFrame
    variant_summary: pd.DataFrame


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description="Research-only Trend Health State Exit diagnostics.")
    parser.add_argument("--trend-map-dir", default=str(DEFAULT_TREND_MAP_DIR))
    parser.add_argument("--trend-v3-dir", default=str(DEFAULT_TREND_V3_DIR))
    parser.add_argument("--funding-dir", default=str(DEFAULT_FUNDING_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS))
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--end", default=DEFAULT_END)
    parser.add_argument("--timezone", default=DEFAULT_TIMEZONE)
    parser.add_argument("--timeframes", default=",".join(DEFAULT_TIMEFRAMES))
    parser.add_argument("--data-check-strict", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def resolve_path(path_arg: str | Path) -> Path:
    """Resolve a path relative to the project root."""

    return tce.resolve_path(path_arg)


def parse_csv_list(value: str | Iterable[str]) -> list[str]:
    """Parse comma/space-separated values."""

    if isinstance(value, str):
        tokens = value.replace(",", " ").split()
    else:
        tokens = [str(item) for item in value]
    result: list[str] = []
    for token in tokens:
        item = token.strip()
        if item and item not in result:
            result.append(item)
    return result


def parse_date_start(value: str, timezone_name: str) -> pd.Timestamp:
    """Parse inclusive start date in local timezone."""

    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize(timezone_name)
    else:
        timestamp = timestamp.tz_convert(timezone_name)
    return pd.Timestamp(timestamp)


def parse_end_exclusive(value: str, timezone_name: str) -> pd.Timestamp:
    """Parse user end date as inclusive calendar day and return exclusive bound."""

    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize(timezone_name)
    else:
        timestamp = timestamp.tz_convert(timezone_name)
    if timestamp.hour == 0 and timestamp.minute == 0 and timestamp.second == 0 and timestamp.microsecond == 0:
        timestamp = timestamp + pd.Timedelta(days=1)
    return pd.Timestamp(timestamp)


def finite_number(value: Any, default: float | None = None) -> float | None:
    """Return finite float, or default."""

    return tce.finite_float(value, default=default)


def variant_configs() -> dict[str, HealthVariantConfig]:
    """Return all non-original health variant configurations."""

    return {
        "health_ema20_core": HealthVariantConfig(
            name="health_ema20_core",
            ema_column="ema20",
            efficiency_threshold=0.5,
            mode="core",
        ),
        "health_ema50_core": HealthVariantConfig(
            name="health_ema50_core",
            ema_column="ema50",
            efficiency_threshold=0.25,
            mode="core",
        ),
        "health_no_energy": HealthVariantConfig(
            name="health_no_energy",
            ema_column="ema20",
            efficiency_threshold=0.5,
            mode="no_energy",
        ),
        "health_drawdown_only": HealthVariantConfig(
            name="health_drawdown_only",
            ema_column="ema20",
            efficiency_threshold=0.5,
            mode="drawdown_only",
        ),
        "health_energy_confirmed": HealthVariantConfig(
            name="health_energy_confirmed",
            ema_column="ema20",
            efficiency_threshold=0.5,
            mode="energy_confirmed",
        ),
    }


def add_health_indicators(frame: pd.DataFrame) -> pd.DataFrame:
    """Add EMA, ATR, shifted volume baseline, and energy ratio to closed bars."""

    result = frame.copy().sort_values("datetime", kind="stable").reset_index(drop=True)
    if result.empty:
        for column in ["ema20", "ema50", "atr14", "volume_sma20", "volume_ratio", "_time_ns"]:
            result[column] = pd.Series(dtype=float)
        return result
    for column in ["open", "high", "low", "close", "volume"]:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    result["ema20"] = result["close"].ewm(span=20, adjust=False, min_periods=1).mean()
    result["ema50"] = result["close"].ewm(span=50, adjust=False, min_periods=1).mean()
    result["atr14"] = tce.true_range(result).rolling(14, min_periods=1).mean()
    result["volume_sma20"] = result["volume"].rolling(20, min_periods=1).mean()
    denominator = result["volume_sma20"].shift(1).replace(0.0, np.nan)
    result["volume_ratio"] = result["volume"] / denominator
    result["volume_ratio"] = result["volume_ratio"].replace([np.inf, -np.inf], np.nan)
    return tce.add_time_ns(result)


def direction_aware_efficiency(row: pd.Series, direction: str, ema_column: str) -> float | None:
    """Return direction-aware distance from EMA in ATR units."""

    close = finite_number(row.get("close"), None)
    ema = finite_number(row.get(ema_column), None)
    atr = finite_number(row.get("atr14"), None)
    if close is None or ema is None or atr is None or atr <= 0:
        return None
    if str(direction).lower() == "short":
        return float((ema - close) / atr)
    return float((close - ema) / atr)


def drawdown_atr_from_state(direction: str, close: float, best_close: float, entry_atr: float) -> float:
    """Return entry-ATR-normalized drawdown from the best close since entry."""

    if entry_atr <= 0:
        return 0.0
    if str(direction).lower() == "short":
        return float((close - best_close) / entry_atr)
    return float((best_close - close) / entry_atr)


def health_state_for_bar(
    row: pd.Series,
    *,
    direction: str,
    config: HealthVariantConfig,
    best_close: float,
    entry_atr: float,
) -> dict[str, Any]:
    """Calculate one closed-bar health state."""

    close = finite_number(row.get("close"), None)
    efficiency = direction_aware_efficiency(row, direction, config.ema_column)
    energy = finite_number(row.get("volume_ratio"), None)
    drawdown = drawdown_atr_from_state(direction, float(close or 0.0), best_close, entry_atr) if close is not None else None
    efficiency_ok = bool(efficiency is not None and efficiency >= config.efficiency_threshold)
    energy_ok = bool(energy is not None and energy >= config.energy_threshold)
    drawdown_ok = bool(drawdown is not None and drawdown <= config.drawdown_soft_atr)
    if config.mode == "no_energy":
        health_score = int(efficiency_ok) + int(drawdown_ok)
    elif config.mode == "drawdown_only":
        health_score = int(drawdown_ok)
    else:
        health_score = int(efficiency_ok) + int(energy_ok) + int(drawdown_ok)
    return {
        "efficiency_atr": efficiency,
        "energy_ratio": energy,
        "drawdown_atr": drawdown,
        "efficiency_ok": efficiency_ok,
        "energy_ok": energy_ok,
        "drawdown_ok": drawdown_ok,
        "health_score": health_score,
    }


def health_state_is_unhealthy(state: dict[str, Any], config: HealthVariantConfig) -> bool:
    """Return whether a health state counts toward patience-based exit."""

    if config.mode == "drawdown_only":
        return not bool(state["drawdown_ok"])
    if config.mode == "energy_confirmed":
        return bool(not state["efficiency_ok"] and not state["energy_ok"])
    return int(state["health_score"]) <= 1


def execution_bar_after_observation(bars: pd.DataFrame, observed_index: int) -> tuple[pd.Timestamp, float]:
    """Return next-bar-open execution after a closed-bar observation."""

    if bars.empty:
        raise TrendHealthExitResearchError("cannot execute without bars")
    exec_index = min(observed_index + 1, len(bars.index) - 1)
    bar = bars.iloc[exec_index]
    timestamp = pd.Timestamp(bar.get("open_time", bar.get("datetime")))
    price = finite_number(bar.get("open"), None)
    if price is None or price <= 0:
        price = finite_number(bar.get("close"), 0.0) or 0.0
        timestamp = pd.Timestamp(bar.get("datetime"))
    return timestamp, float(price)


def observation_index_at_or_after_time(bars: pd.DataFrame, target_time: pd.Timestamp) -> int | None:
    """Return first closed bar at or after target time."""

    if bars.empty:
        return None
    times = bars["_time_ns"].to_numpy(dtype=np.int64) if "_time_ns" in bars.columns else pd.to_datetime(bars["datetime"]).map(lambda value: pd.Timestamp(value).value).to_numpy(dtype=np.int64)
    position = int(np.searchsorted(times, pd.Timestamp(target_time).value, side="left"))
    if position >= len(bars.index):
        return None
    return position


def hold_bars_between(entry_time: pd.Timestamp, exit_time: pd.Timestamp, timeframe: str) -> int:
    """Return approximate hold bars for a trade."""

    minutes = TIMEFRAME_MINUTES.get(str(timeframe), 1)
    delta_minutes = max((pd.Timestamp(exit_time) - pd.Timestamp(entry_time)).total_seconds() / 60.0, 0.0)
    return int(math.ceil(delta_minutes / max(minutes, 1)))


def simulate_health_exit(trade: pd.Series, bars: pd.DataFrame, config: HealthVariantConfig) -> HealthExitSimulation:
    """Simulate one health-state exit with next-bar-open execution."""

    if bars.empty:
        return HealthExitSimulation(
            exit_time=pd.Timestamp(trade["exit_ts"]),
            exit_price=float(trade["exit_price"]),
            exit_reason="missing_bars_original_exit",
            hold_bars=0,
            efficiency_at_exit=None,
            energy_at_exit=None,
            drawdown_at_exit=None,
            health_score_at_exit=None,
            max_health_score=None,
            min_health_score=None,
            avg_health_score=None,
        )
    entry_time = pd.Timestamp(trade["entry_ts"])
    direction = str(trade.get("direction") or "long").lower()
    entry_idx = tce.last_completed_bar_index(bars, entry_time)
    observe_idx = observation_index_at_or_after_time(bars, entry_time)
    if observe_idx is None:
        observe_idx = entry_idx if entry_idx is not None else 0
    entry_atr = tce.entry_atr(trade, bars, entry_idx)
    entry_price = float(trade["entry_price"])
    best_close = entry_price
    consecutive_bad = 0
    scores: list[int] = []
    last_state: dict[str, Any] | None = None
    last_observed_index = int(observe_idx)
    end_index = min(len(bars.index) - 1, int(observe_idx) + max(config.max_hold_bars, 1))

    for index in range(int(observe_idx), end_index + 1):
        bar = bars.iloc[index]
        close = finite_number(bar.get("close"), entry_price) or entry_price
        if direction == "short":
            best_close = min(best_close, float(close))
        else:
            best_close = max(best_close, float(close))
        state = health_state_for_bar(
            bar,
            direction=direction,
            config=config,
            best_close=best_close,
            entry_atr=entry_atr,
        )
        last_state = state
        last_observed_index = index
        scores.append(int(state["health_score"]))
        drawdown = finite_number(state.get("drawdown_atr"), 0.0) or 0.0
        if drawdown > config.drawdown_hard_atr:
            exit_time, exit_price = execution_bar_after_observation(bars, index)
            return HealthExitSimulation(
                exit_time=exit_time,
                exit_price=exit_price,
                exit_reason="hard_drawdown",
                hold_bars=hold_bars_between(entry_time, exit_time, str(trade.get("trade_timeframe") or "4h")),
                efficiency_at_exit=finite_number(state.get("efficiency_atr"), None),
                energy_at_exit=finite_number(state.get("energy_ratio"), None),
                drawdown_at_exit=drawdown,
                health_score_at_exit=int(state["health_score"]),
                max_health_score=max(scores) if scores else None,
                min_health_score=min(scores) if scores else None,
                avg_health_score=float(np.mean(scores)) if scores else None,
            )
        if index - int(observe_idx) + 1 >= config.max_hold_bars:
            exit_time, exit_price = execution_bar_after_observation(bars, index)
            return HealthExitSimulation(
                exit_time=exit_time,
                exit_price=exit_price,
                exit_reason="max_hold_bars",
                hold_bars=hold_bars_between(entry_time, exit_time, str(trade.get("trade_timeframe") or "4h")),
                efficiency_at_exit=finite_number(state.get("efficiency_atr"), None),
                energy_at_exit=finite_number(state.get("energy_ratio"), None),
                drawdown_at_exit=finite_number(state.get("drawdown_atr"), None),
                health_score_at_exit=int(state["health_score"]),
                max_health_score=max(scores) if scores else None,
                min_health_score=min(scores) if scores else None,
                avg_health_score=float(np.mean(scores)) if scores else None,
            )
        if health_state_is_unhealthy(state, config):
            consecutive_bad += 1
        else:
            consecutive_bad = 0
        if consecutive_bad >= config.patience_bars:
            exit_time, exit_price = execution_bar_after_observation(bars, index)
            return HealthExitSimulation(
                exit_time=exit_time,
                exit_price=exit_price,
                exit_reason="health_deterioration",
                hold_bars=hold_bars_between(entry_time, exit_time, str(trade.get("trade_timeframe") or "4h")),
                efficiency_at_exit=finite_number(state.get("efficiency_atr"), None),
                energy_at_exit=finite_number(state.get("energy_ratio"), None),
                drawdown_at_exit=finite_number(state.get("drawdown_atr"), None),
                health_score_at_exit=int(state["health_score"]),
                max_health_score=max(scores) if scores else None,
                min_health_score=min(scores) if scores else None,
                avg_health_score=float(np.mean(scores)) if scores else None,
            )

    fallback_bar = bars.iloc[last_observed_index]
    return HealthExitSimulation(
        exit_time=pd.Timestamp(fallback_bar.get("datetime")),
        exit_price=float(finite_number(fallback_bar.get("close"), trade["entry_price"]) or trade["entry_price"]),
        exit_reason="data_end",
        hold_bars=hold_bars_between(entry_time, pd.Timestamp(fallback_bar.get("datetime")), str(trade.get("trade_timeframe") or "4h")),
        efficiency_at_exit=finite_number((last_state or {}).get("efficiency_atr"), None),
        energy_at_exit=finite_number((last_state or {}).get("energy_ratio"), None),
        drawdown_at_exit=finite_number((last_state or {}).get("drawdown_atr"), None),
        health_score_at_exit=int((last_state or {}).get("health_score")) if last_state is not None else None,
        max_health_score=max(scores) if scores else None,
        min_health_score=min(scores) if scores else None,
        avg_health_score=float(np.mean(scores)) if scores else None,
    )


def simulate_original_exit(trade: pd.Series) -> HealthExitSimulation:
    """Return original legacy exit as a simulation row."""

    entry_time = pd.Timestamp(trade["entry_ts"])
    exit_time = pd.Timestamp(trade["exit_ts"])
    return HealthExitSimulation(
        exit_time=exit_time,
        exit_price=float(trade["exit_price"]),
        exit_reason="original_exit",
        hold_bars=hold_bars_between(entry_time, exit_time, str(trade.get("trade_timeframe") or "4h")),
        efficiency_at_exit=None,
        energy_at_exit=None,
        drawdown_at_exit=None,
        health_score_at_exit=None,
        max_health_score=None,
        min_health_score=None,
        avg_health_score=None,
    )


def simulate_oracle_exit(
    trade: pd.Series,
    bars: pd.DataFrame,
    selected_segment: pd.Series | None,
) -> HealthExitSimulation:
    """Return oracle hold-to-trend-end as upper-bound diagnostic."""

    if selected_segment is None or bars.empty:
        return simulate_original_exit(trade)
    segment_end = pd.Timestamp(selected_segment["end_ts"])
    bar = tce.first_bar_at_or_after(bars, segment_end)
    if bar is None:
        return simulate_original_exit(trade)
    exit_time = pd.Timestamp(bar.get("datetime"))
    exit_price = float(finite_number(bar.get("close"), trade["exit_price"]) or trade["exit_price"])
    return HealthExitSimulation(
        exit_time=exit_time,
        exit_price=exit_price,
        exit_reason="oracle_hold_to_trend_end",
        hold_bars=hold_bars_between(pd.Timestamp(trade["entry_ts"]), exit_time, str(trade.get("trade_timeframe") or "4h")),
        efficiency_at_exit=None,
        energy_at_exit=None,
        drawdown_at_exit=None,
        health_score_at_exit=None,
        max_health_score=None,
        min_health_score=None,
        avg_health_score=None,
    )


def load_v3_trades(trend_v3_dir: Path, timezone_name: str, warnings: list[str]) -> pd.DataFrame:
    """Load required V3 extended trades only."""

    frames: list[pd.DataFrame] = []
    for split in SPLITS:
        path = trend_v3_dir / split / "trend_v3_trades.csv"
        raw = tce.read_csv_if_exists(path, warnings, required=True)
        normalized = tce.normalize_trade_frame(
            raw,
            strategy_source="trend_v3_extended",
            source_file=path,
            split_hint=split,
            timezone_name=timezone_name,
        )
        if not normalized.empty:
            frames.append(normalized)
    if not frames:
        return pd.DataFrame()
    trades = pd.concat(frames, ignore_index=True)
    return trades.sort_values(["symbol", "trade_timeframe", "entry_ts", "trade_id"], kind="stable").reset_index(drop=True)


def load_health_bars(
    *,
    symbols: list[str],
    timeframes: list[str],
    start: pd.Timestamp,
    end_exclusive: pd.Timestamp,
    timezone_name: str,
    database_path: Path,
    warnings: list[str],
    bars_by_symbol: dict[str, pd.DataFrame] | None = None,
) -> dict[tuple[str, str], pd.DataFrame]:
    """Load 1m data once per symbol and build closed-bar health frames."""

    frames: dict[tuple[str, str], pd.DataFrame] = {}
    anchor = pd.Timestamp("2023-01-01T00:00:00", tz=timezone_name)
    for symbol in symbols:
        try:
            if bars_by_symbol and symbol in bars_by_symbol:
                bars_1m = tce.normalize_1m_bars(bars_by_symbol[symbol], timezone_name, timezone_name)
            else:
                query_end = end_exclusive - pd.Timedelta(minutes=1)
                bars_1m = tce.load_1m_bars_from_sqlite(symbol, database_path, timezone_name, start, query_end)
        except Exception as exc:
            warnings.append(f"failed_to_load_1m_bars:{symbol}:{exc}")
            bars_1m = pd.DataFrame()
        for timeframe in timeframes:
            try:
                closed = tce.resample_ohlcv_closed(bars_1m, timeframe, anchor=anchor)
                frames[(symbol, timeframe)] = add_health_indicators(closed)
            except Exception as exc:
                warnings.append(f"failed_to_resample_health_bars:{symbol}:{timeframe}:{exc}")
                frames[(symbol, timeframe)] = add_health_indicators(pd.DataFrame())
    return frames


def build_segments_by_key(segments: pd.DataFrame, timeframes: list[str]) -> dict[tuple[str, str], pd.DataFrame]:
    """Build trend segment frames by symbol/timeframe."""

    output: dict[tuple[str, str], pd.DataFrame] = {}
    if segments.empty:
        return output
    for (symbol, timeframe), group in segments[segments["timeframe"].astype(str).isin(timeframes)].groupby(["symbol", "timeframe"], dropna=False):
        output[(str(symbol), str(timeframe))] = group.reset_index(drop=True)
    return output


def find_selected_segment_for_trade(
    trade: pd.Series,
    exit_time: pd.Timestamp,
    segments_by_key: dict[tuple[str, str], pd.DataFrame],
    fallback_timeframes: list[str],
) -> pd.Series | None:
    """Select original overlapping segment for one trade using trade timeframe first."""

    symbol = str(trade.get("symbol") or "")
    timeframe = str(trade.get("trade_timeframe") or "")
    candidates = [timeframe] + [tf for tf in fallback_timeframes if tf != timeframe]
    for candidate_timeframe in candidates:
        segments = segments_by_key.get((symbol, candidate_timeframe))
        if segments is None or segments.empty:
            continue
        selected = tce.find_selected_segment(symbol, pd.Timestamp(trade["entry_ts"]), exit_time, {symbol: segments})
        if selected is not None:
            return selected
    return None


def diagnostic_for_trade(
    trade: pd.Series,
    exit_time: pd.Timestamp,
    selected_segment: pd.Series | None,
    timeframe: str,
) -> dict[str, Any]:
    """Compute capture diagnostics using a pre-selected segment."""

    return tce.compute_trade_diagnostic(
        trade,
        exit_time,
        {},
        timeframe,
        selected_segment=selected_segment,
    )


def funding_for_variant(
    trade: pd.Series,
    exit_time: pd.Timestamp,
    funding_indexes: dict[str, tce.FundingIndex],
) -> dict[str, Any]:
    """Return actual funding metrics for a variant trade."""

    inst_id = tce.symbol_to_inst_id(str(trade["symbol"]))
    notional = float(finite_number(trade.get("notional"), 0.0) or 0.0)
    return tce.funding_for_trade(
        inst_id,
        str(trade.get("direction")),
        pd.Timestamp(trade["entry_ts"]),
        exit_time,
        notional,
        funding_indexes,
    )


def build_trade_output_row(
    *,
    trade: pd.Series,
    variant: str,
    simulation: HealthExitSimulation,
    selected_segment: pd.Series | None,
    funding_indexes: dict[str, tce.FundingIndex],
    oracle: bool,
) -> dict[str, Any]:
    """Build one health exit variant trade row."""

    direction = str(trade.get("direction") or "long").lower()
    multiplier = float(finite_number(trade.get("pnl_multiplier"), 0.0) or 0.0)
    no_cost = tce.directional_price_pnl(direction, float(trade["entry_price"]), simulation.exit_price, multiplier)
    cost_drag = float(finite_number(trade.get("cost_drag"), 0.0) or 0.0)
    cost_aware = no_cost - cost_drag
    funding = funding_for_variant(trade, simulation.exit_time, funding_indexes)
    funding_pnl = float(funding["funding_pnl"])
    timeframe = str(trade.get("trade_timeframe") or "4h")
    diagnostic = diagnostic_for_trade(trade, simulation.exit_time, selected_segment, timeframe)
    original_hold_bars = hold_bars_between(pd.Timestamp(trade["entry_ts"]), pd.Timestamp(trade["exit_ts"]), timeframe)
    return {
        "trade_id": trade.get("trade_id"),
        "source_policy": trade.get("policy_or_group"),
        "symbol": trade.get("symbol"),
        "timeframe": timeframe,
        "direction": direction,
        "split": trade.get("split"),
        "entry_time": tce.format_timestamp(pd.Timestamp(trade["entry_ts"])),
        "entry_price": float(trade["entry_price"]),
        "original_exit_time": tce.format_timestamp(pd.Timestamp(trade["exit_ts"])),
        "variant_exit_time": tce.format_timestamp(simulation.exit_time),
        "variant_exit_price": float(simulation.exit_price),
        "exit_variant": variant,
        "exit_reason": simulation.exit_reason,
        "hold_bars": int(simulation.hold_bars),
        "original_hold_bars": int(original_hold_bars),
        "no_cost_pnl": float(no_cost),
        "cost_aware_pnl": float(cost_aware),
        "funding_adjusted_pnl": float(cost_aware + funding_pnl),
        "funding_events_count": int(funding["funding_events_count"]),
        "efficiency_at_exit": simulation.efficiency_at_exit,
        "energy_at_exit": simulation.energy_at_exit,
        "drawdown_at_exit": simulation.drawdown_at_exit,
        "health_score_at_exit": simulation.health_score_at_exit,
        "max_health_score": simulation.max_health_score,
        "min_health_score": simulation.min_health_score,
        "avg_health_score": simulation.avg_health_score,
        "captured_fraction": float(diagnostic["captured_fraction_of_segment"]),
        "early_exit_flag": bool(diagnostic["early_exit_flag"]),
        "late_entry_flag": bool(diagnostic["late_entry_flag"]),
        "oracle": bool(oracle),
        "funding_pnl": funding_pnl,
        "funding_data_available": bool(funding["funding_data_available"]),
        "funding_interval_covered": bool(funding["funding_interval_covered"]),
        "cost_drag": cost_drag,
        "notional": float(finite_number(trade.get("notional"), 0.0) or 0.0),
        "pnl_multiplier": multiplier,
    }


def build_health_exit_variant_trades(
    trades: pd.DataFrame,
    bars: dict[tuple[str, str], pd.DataFrame],
    segments_by_key: dict[tuple[str, str], pd.DataFrame],
    funding_indexes: dict[str, tce.FundingIndex],
    timeframes: list[str],
    warnings: list[str],
) -> pd.DataFrame:
    """Build all health-exit counterfactual trades."""

    rows: list[dict[str, Any]] = []
    configs = variant_configs()
    if trades.empty:
        return pd.DataFrame(columns=TRADE_COLUMNS)
    for _, trade in trades.iterrows():
        symbol = str(trade.get("symbol") or "")
        timeframe = str(trade.get("trade_timeframe") or "")
        if timeframe not in timeframes:
            timeframe = timeframes[0]
        frame = bars.get((symbol, timeframe), pd.DataFrame())
        selected_segment = find_selected_segment_for_trade(trade, pd.Timestamp(trade["exit_ts"]), segments_by_key, timeframes)
        simulations: dict[str, tuple[HealthExitSimulation, bool]] = {
            "original_exit": (simulate_original_exit(trade), False),
            ORACLE_VARIANT: (simulate_oracle_exit(trade, frame, selected_segment), True),
        }
        for name, config in configs.items():
            simulations[name] = (simulate_health_exit(trade, frame, config), False)
        for variant in EXIT_VARIANTS:
            simulation, oracle = simulations[variant]
            try:
                rows.append(
                    build_trade_output_row(
                        trade=trade,
                        variant=variant,
                        simulation=simulation,
                        selected_segment=selected_segment,
                        funding_indexes=funding_indexes,
                        oracle=oracle,
                    )
                )
            except Exception as exc:
                warnings.append(f"failed_to_build_variant_trade:{trade.get('trade_id')}:{variant}:{exc}")
    return pd.DataFrame(rows, columns=list(dict.fromkeys(TRADE_COLUMNS + [
        "funding_pnl",
        "funding_data_available",
        "funding_interval_covered",
        "cost_drag",
        "notional",
        "pnl_multiplier",
    ])))


def max_drawdown_from_pnl(pnl: pd.Series) -> float:
    """Return max drawdown from cumulative PnL."""

    return tce.max_drawdown_from_pnl(pnl)


def top_5pct_trade_pnl_contribution(trades: pd.DataFrame, pnl_column: str = "funding_adjusted_pnl") -> float | None:
    """Return top 5 percent trade contribution."""

    return tce.top_5pct_trade_pnl_contribution(trades, pnl_column)


def largest_symbol_pnl_share(trades: pd.DataFrame, pnl_column: str = "funding_adjusted_pnl") -> float | None:
    """Return largest symbol PnL share."""

    return tce.largest_symbol_pnl_share(trades, pnl_column)


def summarize_trade_slice(trades: pd.DataFrame) -> dict[str, Any]:
    """Summarize health exit trades."""

    if trades.empty:
        return {
            "trade_count": 0,
            "no_cost_pnl": 0.0,
            "cost_aware_pnl": 0.0,
            "funding_adjusted_pnl": 0.0,
            "funding_events_count": 0,
            "funding_data_complete": False,
            "win_rate": None,
            "profit_factor": None,
            "max_drawdown": 0.0,
            "largest_symbol_pnl_share": None,
            "top_5pct_trade_pnl_contribution": None,
            "avg_captured_fraction": None,
            "early_exit_share": None,
            "late_entry_share": None,
            "avg_hold_bars": None,
            "max_hold_bars_used": 0,
            "exit_reason_distribution": "{}",
        }
    working = trades.copy().sort_values("variant_exit_time", kind="stable")
    pnl = pd.to_numeric(working["funding_adjusted_pnl"], errors="coerce").fillna(0.0)
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    profit_factor = None
    if not losses.empty:
        profit_factor = float(wins.sum() / abs(losses.sum())) if not wins.empty else 0.0
    reason_counts = working["exit_reason"].astype(str).value_counts().sort_index().to_dict() if "exit_reason" in working.columns else {}
    return {
        "trade_count": int(len(working.index)),
        "no_cost_pnl": float(pd.to_numeric(working["no_cost_pnl"], errors="coerce").fillna(0.0).sum()),
        "cost_aware_pnl": float(pd.to_numeric(working["cost_aware_pnl"], errors="coerce").fillna(0.0).sum()),
        "funding_adjusted_pnl": float(pnl.sum()),
        "funding_events_count": int(pd.to_numeric(working["funding_events_count"], errors="coerce").fillna(0).sum()),
        "funding_data_complete": bool(working["funding_data_available"].astype(bool).all() and working["funding_interval_covered"].astype(bool).all()) if "funding_data_available" in working.columns else False,
        "win_rate": float((pnl > 0).mean()) if len(pnl.index) else None,
        "profit_factor": profit_factor,
        "max_drawdown": max_drawdown_from_pnl(pnl),
        "largest_symbol_pnl_share": largest_symbol_pnl_share(working),
        "top_5pct_trade_pnl_contribution": top_5pct_trade_pnl_contribution(working),
        "avg_captured_fraction": float(pd.to_numeric(working["captured_fraction"], errors="coerce").fillna(0.0).mean()),
        "early_exit_share": float(working["early_exit_flag"].astype(bool).mean()),
        "late_entry_share": float(working["late_entry_flag"].astype(bool).mean()),
        "avg_hold_bars": float(pd.to_numeric(working["hold_bars"], errors="coerce").fillna(0.0).mean()),
        "max_hold_bars_used": int(pd.to_numeric(working["hold_bars"], errors="coerce").fillna(0.0).max()),
        "exit_reason_distribution": json.dumps({str(key): int(value) for key, value in reason_counts.items()}, sort_keys=True),
    }


def build_group_summary(trades: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    """Build grouped summaries."""

    metric_columns = [
        "trade_count",
        "no_cost_pnl",
        "cost_aware_pnl",
        "funding_adjusted_pnl",
        "funding_events_count",
        "funding_data_complete",
        "win_rate",
        "profit_factor",
        "max_drawdown",
        "largest_symbol_pnl_share",
        "top_5pct_trade_pnl_contribution",
        "avg_captured_fraction",
        "early_exit_share",
        "late_entry_share",
        "avg_hold_bars",
        "max_hold_bars_used",
        "exit_reason_distribution",
    ]
    if trades.empty:
        return pd.DataFrame(columns=group_columns + metric_columns)
    rows: list[dict[str, Any]] = []
    for keys, group in trades.groupby(group_columns, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_columns, keys, strict=False))
        row.update(summarize_trade_slice(group))
        rows.append(row)
    return pd.DataFrame(rows, columns=group_columns + metric_columns).sort_values(group_columns, kind="stable").reset_index(drop=True)


def build_variant_summary(trades: pd.DataFrame) -> pd.DataFrame:
    """Build one summary row per exit variant."""

    by_variant = build_group_summary(trades, ["exit_variant", "oracle"])
    by_split = build_group_summary(trades, ["exit_variant", "oracle", "split"])
    if by_variant.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for _, row in by_variant.iterrows():
        variant = str(row["exit_variant"])
        oracle = bool(row["oracle"])
        out = row.to_dict()
        split_rows = by_split[(by_split["exit_variant"] == variant) & (by_split["oracle"] == oracle)]
        for split in SPLITS:
            split_row = split_rows[split_rows["split"] == split]
            metrics = summarize_trade_slice(pd.DataFrame()) if split_row.empty else split_row.iloc[0].to_dict()
            out[f"{split}_trade_count"] = int(metrics.get("trade_count") or 0)
            out[f"{split}_no_cost_pnl"] = float(metrics.get("no_cost_pnl") or 0.0)
            out[f"{split}_cost_aware_pnl"] = float(metrics.get("cost_aware_pnl") or 0.0)
            out[f"{split}_funding_adjusted_pnl"] = float(metrics.get("funding_adjusted_pnl") or 0.0)
        rows.append(out)
    return pd.DataFrame(rows).sort_values(["oracle", "exit_variant"], kind="stable").reset_index(drop=True)


def build_concentration_summary(trades: pd.DataFrame) -> pd.DataFrame:
    """Build concentration metrics by variant and split/all."""

    columns = [
        "exit_variant",
        "oracle",
        "scope",
        "trade_count",
        "largest_symbol_pnl_share",
        "top_5pct_trade_pnl_contribution",
        "top_5pct_trade_pnl",
        "top_5pct_trade_count",
    ]
    if trades.empty:
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, Any]] = []
    for (variant, oracle), group in trades.groupby(["exit_variant", "oracle"], dropna=False):
        scopes = [("all_splits", group)]
        scopes.extend((str(split), split_group) for split, split_group in group.groupby("split", dropna=False))
        for scope, scope_group in scopes:
            pnl = pd.to_numeric(scope_group["funding_adjusted_pnl"], errors="coerce").fillna(0.0)
            top_count = max(1, int(math.ceil(len(pnl.index) * 0.05))) if len(pnl.index) else 0
            top_pnl = float(pnl.sort_values(ascending=False).head(top_count).sum()) if top_count else 0.0
            rows.append(
                {
                    "exit_variant": variant,
                    "oracle": bool(oracle),
                    "scope": scope,
                    "trade_count": int(len(scope_group.index)),
                    "largest_symbol_pnl_share": largest_symbol_pnl_share(scope_group),
                    "top_5pct_trade_pnl_contribution": top_5pct_trade_pnl_contribution(scope_group),
                    "top_5pct_trade_pnl": top_pnl,
                    "top_5pct_trade_count": int(top_count),
                }
            )
    return pd.DataFrame(rows, columns=columns).sort_values(["exit_variant", "scope"], kind="stable").reset_index(drop=True)


def original_metrics(variant_summary: pd.DataFrame) -> dict[str, float]:
    """Return original-exit benchmark metrics."""

    if variant_summary.empty:
        return {"avg_captured_fraction": 0.0, "early_exit_share": 1.0}
    row = variant_summary[variant_summary["exit_variant"] == "original_exit"]
    if row.empty:
        return {"avg_captured_fraction": 0.0, "early_exit_share": 1.0}
    first = row.iloc[0]
    return {
        "avg_captured_fraction": float(finite_number(first.get("avg_captured_fraction"), 0.0) or 0.0),
        "early_exit_share": float(finite_number(first.get("early_exit_share"), 1.0) or 1.0),
    }


def evaluate_stable_gates(
    variant_summary: pd.DataFrame,
    concentration: pd.DataFrame,
    funding_data_complete: bool,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Evaluate health exit stable-like gates."""

    rows: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    if variant_summary.empty:
        return pd.DataFrame(), []
    original = original_metrics(variant_summary)
    for _, row in variant_summary.iterrows():
        variant = str(row["exit_variant"])
        oracle = bool(row["oracle"])
        reasons: list[str] = []
        if oracle:
            reasons.append("oracle_variant_excluded_from_stable_gate")
        if variant == "original_exit":
            reasons.append("original_exit_reference_excluded_from_health_gate")
        for split in SPLITS:
            if int(row.get(f"{split}_trade_count") or 0) < 10:
                reasons.append(f"{split}:trade_count_lt_10")
            if float(finite_number(row.get(f"{split}_no_cost_pnl"), 0.0) or 0.0) <= 0:
                reasons.append(f"{split}:no_cost_pnl_not_positive")
        if float(finite_number(row.get("oos_ext_cost_aware_pnl"), 0.0) or 0.0) < 0:
            reasons.append("oos_ext:cost_aware_pnl_negative")
        if float(finite_number(row.get("oos_ext_funding_adjusted_pnl"), 0.0) or 0.0) < 0:
            reasons.append("oos_ext:funding_adjusted_pnl_negative")
        concentration_row = concentration[
            (concentration["exit_variant"].astype(str) == variant)
            & (concentration["oracle"].astype(bool) == oracle)
            & (concentration["scope"].astype(str) == "all_splits")
        ] if not concentration.empty else pd.DataFrame()
        largest_share = finite_number(concentration_row.iloc[0].get("largest_symbol_pnl_share"), None) if not concentration_row.empty else finite_number(row.get("largest_symbol_pnl_share"), None)
        top_share = finite_number(concentration_row.iloc[0].get("top_5pct_trade_pnl_contribution"), None) if not concentration_row.empty else finite_number(row.get("top_5pct_trade_pnl_contribution"), None)
        if largest_share is None or largest_share > 0.7:
            reasons.append("largest_symbol_pnl_share_gt_0.7")
        if top_share is None or top_share > 0.8:
            reasons.append("top_5pct_trade_pnl_contribution_gt_0.8")
        avg_capture = finite_number(row.get("avg_captured_fraction"), None)
        early_exit = finite_number(row.get("early_exit_share"), None)
        if avg_capture is None or avg_capture < original["avg_captured_fraction"] + 0.15:
            reasons.append("avg_captured_fraction_lt_original_plus_0.15")
        if early_exit is None or early_exit > original["early_exit_share"] - 0.20:
            reasons.append("early_exit_share_not_reduced_by_0.20")
        if not funding_data_complete:
            reasons.append("funding_data_incomplete")
        stable = bool(not reasons)
        record = {
            "exit_variant": variant,
            "oracle": oracle,
            "stable_like": stable,
            "rejected_reasons": ";".join(reasons),
            "avg_captured_fraction": avg_capture,
            "early_exit_share": early_exit,
            "original_avg_captured_fraction": original["avg_captured_fraction"],
            "original_early_exit_share": original["early_exit_share"],
        }
        rows.append(record)
        if stable:
            candidates.append(record)
    return pd.DataFrame(rows), candidates


def build_vs_original(variant_summary: pd.DataFrame, previous_exit_summary: pd.DataFrame) -> pd.DataFrame:
    """Compare health exits to original and previous tested exit variants."""

    if variant_summary.empty:
        return pd.DataFrame()
    original = original_metrics(variant_summary)
    previous_non_oracle = previous_exit_summary[previous_exit_summary["oracle"].astype(str).str.lower() != "true"].copy() if not previous_exit_summary.empty and "oracle" in previous_exit_summary.columns else previous_exit_summary.copy()
    previous_best_capture = None
    previous_best_variant = None
    if not previous_non_oracle.empty and "avg_captured_fraction" in previous_non_oracle.columns:
        ordered = previous_non_oracle.copy()
        ordered["avg_captured_fraction"] = pd.to_numeric(ordered["avg_captured_fraction"], errors="coerce")
        ordered = ordered.sort_values("avg_captured_fraction", ascending=False, kind="stable")
        if not ordered.empty:
            previous_best_capture = finite_number(ordered.iloc[0].get("avg_captured_fraction"), None)
            previous_best_variant = ordered.iloc[0].get("exit_variant")
    rows: list[dict[str, Any]] = []
    for _, row in variant_summary.iterrows():
        capture = finite_number(row.get("avg_captured_fraction"), None)
        early_exit = finite_number(row.get("early_exit_share"), None)
        rows.append(
            {
                "exit_variant": row.get("exit_variant"),
                "oracle": bool(row.get("oracle")),
                "avg_captured_fraction": capture,
                "capture_improvement_vs_original": None if capture is None else capture - original["avg_captured_fraction"],
                "early_exit_share": early_exit,
                "early_exit_reduction_vs_original": None if early_exit is None else original["early_exit_share"] - early_exit,
                "previous_best_non_oracle_variant": previous_best_variant,
                "previous_best_non_oracle_avg_capture": previous_best_capture,
                "beats_previous_best_capture": bool(capture is not None and previous_best_capture is not None and capture > previous_best_capture),
            }
        )
    return pd.DataFrame(rows)


def build_funding_summary(trades: pd.DataFrame) -> pd.DataFrame:
    """Summarize funding by variant."""

    columns = [
        "exit_variant",
        "oracle",
        "trade_count",
        "funding_pnl",
        "funding_events_count",
        "funding_adjusted_pnl",
        "funding_data_complete",
    ]
    if trades.empty:
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, Any]] = []
    for (variant, oracle), group in trades.groupby(["exit_variant", "oracle"], dropna=False):
        rows.append(
            {
                "exit_variant": variant,
                "oracle": bool(oracle),
                "trade_count": int(len(group.index)),
                "funding_pnl": float(pd.to_numeric(group["funding_pnl"], errors="coerce").fillna(0.0).sum()),
                "funding_events_count": int(pd.to_numeric(group["funding_events_count"], errors="coerce").fillna(0).sum()),
                "funding_adjusted_pnl": float(pd.to_numeric(group["funding_adjusted_pnl"], errors="coerce").fillna(0.0).sum()),
                "funding_data_complete": bool(group["funding_data_available"].astype(bool).all() and group["funding_interval_covered"].astype(bool).all()),
            }
        )
    return pd.DataFrame(rows, columns=columns).sort_values(["oracle", "exit_variant"], kind="stable").reset_index(drop=True)


def build_data_quality(
    *,
    trend_map_quality: dict[str, Any],
    trend_map_summary: dict[str, Any],
    symbols: list[str],
    timeframes: list[str],
    start: pd.Timestamp,
    end_exclusive: pd.Timestamp,
    warnings: list[str],
    missing_funding: list[str],
    data_check_strict: bool,
) -> dict[str, Any]:
    """Build data quality payload."""

    trend_map_dq = trend_map_quality or {}
    market_complete = bool(trend_map_dq.get("all_symbols_complete", trend_map_dq.get("market_data_complete", True)))
    funding_complete = bool(not missing_funding)
    if data_check_strict and not market_complete:
        warnings.append("data_check_strict:market_data_not_complete")
    if data_check_strict and not funding_complete:
        warnings.append("data_check_strict:funding_data_not_complete")
    return {
        "mode": "research_only_trend_health_state_exit",
        "symbols": symbols,
        "timeframes": timeframes,
        "start": tce.format_timestamp(start),
        "end_exclusive": tce.format_timestamp(end_exclusive),
        "market_data_complete": market_complete,
        "funding_data_complete": funding_complete,
        "funding_missing_inst_ids": missing_funding,
        "trend_opportunity_map": {
            "enough_trend_opportunities": trend_map_summary.get("enough_trend_opportunities"),
            "trend_opportunities_are_diversified": trend_map_summary.get("trend_opportunities_are_diversified"),
            "legacy_main_failure_mode": (trend_map_summary.get("legacy_analysis") or {}).get("main_failure_mode"),
        },
        "trend_map_data_quality": trend_map_quality,
        "warnings": sorted(dict.fromkeys(warnings)),
        "data_check_strict": bool(data_check_strict),
        "strategy_development_allowed": False,
        "demo_live_allowed": False,
    }


def build_summary_payload(
    *,
    output_dir: Path,
    warnings: list[str],
    data_quality: dict[str, Any],
    variant_summary: pd.DataFrame,
    rejected: pd.DataFrame,
    candidates: list[dict[str, Any]],
    vs_original: pd.DataFrame,
) -> dict[str, Any]:
    """Build JSON summary."""

    non_oracle = variant_summary[
        (variant_summary["oracle"].astype(bool) == False)  # noqa: E712
        & (variant_summary["exit_variant"].astype(str) != "original_exit")
    ].copy() if not variant_summary.empty else pd.DataFrame()
    best = pd.DataFrame()
    if not non_oracle.empty:
        best = non_oracle.sort_values(["avg_captured_fraction", "early_exit_share"], ascending=[False, True], kind="stable").head(1)
    original = original_metrics(variant_summary)
    best_variant = tce.dataframe_records(best)[0] if not best.empty else None
    health_phase2 = bool(candidates)
    return {
        "mode": "research_only_trend_health_state_exit",
        "output_dir": str(output_dir),
        "output_files": REQUIRED_OUTPUT_FILES,
        "warnings": sorted(dict.fromkeys(warnings)),
        "old_entries_unchanged": True,
        "no_entry_filter_added": True,
        "labels_are_ex_post_only_for_evaluation": True,
        "oracle_excluded_from_stable_gate": True,
        "data_quality": data_quality,
        "variant_summary": tce.dataframe_records(variant_summary),
        "best_non_oracle_variant": best_variant,
        "original_avg_captured_fraction": original["avg_captured_fraction"],
        "original_early_exit_share": original["early_exit_share"],
        "health_exit_phase2_candidates": candidates,
        "stable_like_candidate_exists": bool(candidates),
        "can_enter_health_exit_phase2": health_phase2,
        "strategy_development_allowed": False,
        "demo_live_allowed": False,
        "no_policy_can_be_traded": True,
        "recommended_next_step": "health_exit_phase2_research_only" if health_phase2 else "entry_timing_research_or_pause",
        "vs_original": tce.dataframe_records(vs_original),
    }


def format_number(value: Any, digits: int = 6) -> str:
    """Format optional number for Markdown."""

    number = finite_number(value, default=np.nan)
    if number is None or not np.isfinite(number):
        return "N/A"
    return f"{number:.{digits}f}"


def markdown_table(rows: list[dict[str, Any]], columns: list[str], limit: int = 50) -> str:
    """Render Markdown table."""

    if not rows:
        return "- none"
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join("---" for _ in columns) + "|"]
    for row in rows[:limit]:
        values: list[str] = []
        for column in columns:
            value = row.get(column)
            if isinstance(value, bool):
                values.append(str(value).lower())
            elif isinstance(value, (int, float)):
                values.append(format_number(value))
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def render_report(summary: dict[str, Any], rejected: pd.DataFrame) -> str:
    """Render Markdown report with required answers."""

    rows = summary.get("variant_summary") or []
    rejected_rows = tce.dataframe_records(rejected)
    best = summary.get("best_non_oracle_variant") or {}
    stable_exists = bool(summary.get("stable_like_candidate_exists"))
    original_capture = float(summary.get("original_avg_captured_fraction") or 0.0)
    original_early = float(summary.get("original_early_exit_share") or 0.0)
    best_capture = finite_number(best.get("avg_captured_fraction"), None)
    best_early = finite_number(best.get("early_exit_share"), None)
    capture_improved = bool(best_capture is not None and best_capture > original_capture)
    early_reduced = bool(best_early is not None and best_early < original_early)
    capture_significant = bool(best_capture is not None and best_capture >= original_capture + 0.15)
    early_significant = bool(best_early is not None and best_early <= original_early - 0.20)
    return (
        "# Trend Health State Exit Research\n\n"
        "## Scope\n"
        "- This is offline counterfactual exit research. Legacy entries remain unchanged.\n"
        "- Trend segment labels are used only for ex-post evaluation and oracle upper-bound diagnostics.\n"
        "- No formal strategy modification, demo, or live permission is granted.\n\n"
        "## Required Answers\n"
        "1. What is the hypothesis? Hold only while trend health remains confirmed by efficiency, energy, drawdown, and time limits.\n"
        f"2. Did it materially improve captured_fraction? best_variant={best.get('exit_variant', 'N/A')}, "
        f"best_avg_capture={format_number(best_capture)}, original_avg_capture={format_number(original_capture)}, "
        f"raw_improved={str(capture_improved).lower()}, gate_material={str(capture_significant).lower()}.\n"
        f"3. Did it materially reduce early_exit_share? best_early_exit_share={format_number(best_early)}, "
        f"original_early_exit_share={format_number(original_early)}, raw_reduced={str(early_reduced).lower()}, "
        f"gate_material={str(early_significant).lower()}.\n"
        f"4. Is it better than original_exit? raw={str(capture_improved and early_reduced).lower()}, "
        f"gate_material={str(capture_significant and early_significant).lower()}.\n"
        "5. Is it better than previously tested exits? See health_exit_vs_original.csv; previous non-oracle best is included when available.\n"
        f"6. Did any non-oracle variant pass train/validation/oos, cost, funding, and concentration gates? {str(stable_exists).lower()}.\n"
        "7. Which health dimension worked best? Compare health_no_energy, health_drawdown_only, and health_energy_confirmed in the summary table.\n"
        "8. Did volume energy help? If health_no_energy or health_energy_confirmed beats health_ema20_core, strict energy likely caused premature exits.\n"
        "9. Is drawdown alone enough? See health_drawdown_only capture, early_exit_share, and gate rejection reasons.\n"
        f"10. If health exit failed, should research turn to entry timing? {str(not stable_exists).lower()}.\n"
        "11. Formal strategy modification allowed? false.\n"
        "12. Demo/live allowed? false.\n\n"
        "## Exit Variant Summary\n"
        f"{markdown_table(rows, ['exit_variant', 'oracle', 'trade_count', 'train_ext_no_cost_pnl', 'validation_ext_no_cost_pnl', 'oos_ext_no_cost_pnl', 'oos_ext_cost_aware_pnl', 'oos_ext_funding_adjusted_pnl', 'avg_captured_fraction', 'early_exit_share', 'stable_like'])}\n\n"
        "## Gate Rejections\n"
        f"{markdown_table(rejected_rows, ['exit_variant', 'oracle', 'stable_like', 'rejected_reasons', 'avg_captured_fraction', 'early_exit_share'], limit=100)}\n\n"
        "## Final Gates\n"
        f"- can_enter_health_exit_phase2={str(bool(summary.get('can_enter_health_exit_phase2'))).lower()}\n"
        "- strategy_development_allowed=false\n"
        "- demo_live_allowed=false\n"
        f"- recommended_next_step={summary.get('recommended_next_step')}\n"
    )


def write_outputs(
    output_dir: Path,
    summary: dict[str, Any],
    data_quality: dict[str, Any],
    variant_trades: pd.DataFrame,
    variant_summary: pd.DataFrame,
    by_policy: pd.DataFrame,
    by_symbol: pd.DataFrame,
    by_split: pd.DataFrame,
    concentration: pd.DataFrame,
    vs_original: pd.DataFrame,
    rejected: pd.DataFrame,
    funding_summary: pd.DataFrame,
) -> None:
    """Write all required outputs."""

    output_dir.mkdir(parents=True, exist_ok=True)
    variant_trades.to_csv(output_dir / "health_exit_variant_trades.csv", index=False, encoding="utf-8")
    variant_summary.to_csv(output_dir / "health_exit_variant_summary.csv", index=False, encoding="utf-8")
    by_policy.to_csv(output_dir / "health_exit_by_policy.csv", index=False, encoding="utf-8")
    by_symbol.to_csv(output_dir / "health_exit_by_symbol.csv", index=False, encoding="utf-8")
    by_split.to_csv(output_dir / "health_exit_by_split.csv", index=False, encoding="utf-8")
    concentration.to_csv(output_dir / "health_exit_concentration.csv", index=False, encoding="utf-8")
    vs_original.to_csv(output_dir / "health_exit_vs_original.csv", index=False, encoding="utf-8")
    rejected.to_csv(output_dir / "health_exit_rejected_variants.csv", index=False, encoding="utf-8")
    funding_summary.to_csv(output_dir / "health_exit_funding_summary.csv", index=False, encoding="utf-8")
    (output_dir / "data_quality.json").write_text(json.dumps(to_jsonable(data_quality), ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "health_exit_summary.json").write_text(json.dumps(to_jsonable(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "health_exit_report.md").write_text(render_report(summary, rejected), encoding="utf-8")


def run_research(
    *,
    trend_map_dir: Path,
    trend_v3_dir: Path,
    funding_dir: Path,
    output_dir: Path,
    symbols: list[str],
    start: str,
    end: str,
    timezone_name: str,
    timeframes: list[str],
    data_check_strict: bool,
    logger: logging.Logger | None = None,
    database_path: Path = DEFAULT_DATABASE_PATH,
    capture_exit_dir: Path = DEFAULT_CAPTURE_EXIT_DIR,
    bars_by_symbol: dict[str, pd.DataFrame] | None = None,
) -> ResearchOutputs:
    """Run Trend Health State Exit research."""

    warnings: list[str] = []
    trend_summary = tce.read_json_if_exists(trend_map_dir / "trend_opportunity_summary.json", warnings)
    trend_quality = tce.read_json_if_exists(trend_map_dir / "data_quality.json", warnings)
    raw_segments = tce.read_csv_if_exists(trend_map_dir / "trend_segments.csv", warnings, required=True)
    capture_diagnostics = tce.read_csv_if_exists(capture_exit_dir / "trend_capture_diagnostics.csv", warnings, required=True)
    previous_exit_summary = tce.read_csv_if_exists(capture_exit_dir / "exit_variant_summary.csv", warnings, required=False)
    _previous_exit_trades = tce.read_csv_if_exists(capture_exit_dir / "exit_variant_trades.csv", warnings, required=False)
    trades = load_v3_trades(trend_v3_dir, timezone_name, warnings)

    if capture_diagnostics.empty:
        warnings.append("trend_capture_diagnostics_unavailable_for_cross_check")

    segments = tce.normalize_segments(raw_segments, timezone_name)
    segments_by_key = build_segments_by_key(segments, timeframes)
    start_ts = parse_date_start(start, timezone_name)
    end_exclusive = parse_end_exclusive(end, timezone_name)
    bars = load_health_bars(
        symbols=symbols,
        timeframes=timeframes,
        start=start_ts,
        end_exclusive=end_exclusive,
        timezone_name=timezone_name,
        database_path=database_path,
        warnings=warnings,
        bars_by_symbol=bars_by_symbol,
    )
    inst_ids = [tce.symbol_to_inst_id(symbol) for symbol in symbols]
    funding_histories, missing_funding = tce.load_funding_histories(funding_dir, inst_ids, warnings)
    funding_indexes = tce.build_funding_indexes(funding_histories)
    data_quality = build_data_quality(
        trend_map_quality=trend_quality,
        trend_map_summary=trend_summary,
        symbols=symbols,
        timeframes=timeframes,
        start=start_ts,
        end_exclusive=end_exclusive,
        warnings=warnings,
        missing_funding=missing_funding,
        data_check_strict=data_check_strict,
    )

    variant_trades = build_health_exit_variant_trades(
        trades,
        bars,
        segments_by_key,
        funding_indexes,
        timeframes,
        warnings,
    )
    variant_summary = build_variant_summary(variant_trades)
    by_policy = build_group_summary(variant_trades, ["exit_variant", "oracle", "source_policy"])
    by_symbol = build_group_summary(variant_trades, ["exit_variant", "oracle", "symbol"])
    by_split = build_group_summary(variant_trades, ["exit_variant", "oracle", "split"])
    concentration = build_concentration_summary(variant_trades)
    rejected, candidates = evaluate_stable_gates(variant_summary, concentration, bool(data_quality["funding_data_complete"]))
    if not variant_summary.empty and not rejected.empty:
        variant_summary = variant_summary.merge(rejected[["exit_variant", "oracle", "stable_like", "rejected_reasons"]], on=["exit_variant", "oracle"], how="left")
    vs_original = build_vs_original(variant_summary, previous_exit_summary)
    funding_summary = build_funding_summary(variant_trades)
    summary = build_summary_payload(
        output_dir=output_dir,
        warnings=warnings,
        data_quality=data_quality,
        variant_summary=variant_summary,
        rejected=rejected,
        candidates=candidates,
        vs_original=vs_original,
    )

    write_outputs(
        output_dir,
        summary,
        data_quality,
        variant_trades,
        variant_summary,
        by_policy,
        by_symbol,
        by_split,
        concentration,
        vs_original,
        rejected,
        funding_summary,
    )
    if logger is not None:
        log_event(
            logger,
            logging.INFO,
            "trend_health_exit.complete",
            "Trend Health State Exit research complete",
            output_dir=str(output_dir),
            can_enter_health_exit_phase2=summary["can_enter_health_exit_phase2"],
            recommended_next_step=summary["recommended_next_step"],
        )
    return ResearchOutputs(output_dir=output_dir, summary=summary, variant_trades=variant_trades, variant_summary=variant_summary)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""

    args = parse_args(argv)
    ensure_headless_runtime()
    logger = setup_logging("research_trend_health_state_exit", verbose=args.verbose)
    outputs = run_research(
        trend_map_dir=resolve_path(args.trend_map_dir),
        trend_v3_dir=resolve_path(args.trend_v3_dir),
        funding_dir=resolve_path(args.funding_dir),
        output_dir=resolve_path(args.output_dir),
        symbols=parse_csv_list(args.symbols),
        start=str(args.start),
        end=str(args.end),
        timezone_name=str(args.timezone),
        timeframes=parse_csv_list(args.timeframes),
        data_check_strict=bool(args.data_check_strict),
        logger=logger,
    )
    print_json_block(
        "Trend Health State Exit summary:",
        {
            "output_dir": str(outputs.output_dir),
            "can_enter_health_exit_phase2": outputs.summary.get("can_enter_health_exit_phase2"),
            "strategy_development_allowed": outputs.summary.get("strategy_development_allowed"),
            "demo_live_allowed": outputs.summary.get("demo_live_allowed"),
            "recommended_next_step": outputs.summary.get("recommended_next_step"),
            "best_non_oracle_variant": outputs.summary.get("best_non_oracle_variant"),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
