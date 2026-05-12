#!/usr/bin/env python3
"""Research-only Trend Entry Timing v1 diagnostics."""

from __future__ import annotations

import argparse
import hashlib
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
DEFAULT_TREND_MAP_DIR = PROJECT_ROOT / "reports" / "research" / "trend_opportunity_map"
DEFAULT_TREND_V3_DIR = PROJECT_ROOT / "reports" / "research" / "trend_following_v3_extended"
DEFAULT_FUNDING_DIR = PROJECT_ROOT / "data" / "funding" / "okx"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "reports" / "research" / "trend_entry_timing"
DEFAULT_CAPTURE_EXIT_DIR = PROJECT_ROOT / "reports" / "research" / "trend_capture_exit_convexity"
DEFAULT_DATABASE_PATH = PROJECT_ROOT / ".vntrader" / "database.db"

SPLITS = tce.SPLITS
TIMEFRAME_MINUTES = {"4h": 240, "1d": 1440}
ENTRY_FAMILIES = [
    "pre_breakout_momentum_acceleration",
    "breakout_retest_reclaim",
    "cross_symbol_breadth_acceleration",
    "funding_neutral_momentum",
    "relative_strength_leader",
]
FIXED_HOLD_BARS = {
    "4h": {
        "fixed_hold_4h": 1,
        "fixed_hold_8h": 2,
        "fixed_hold_1d": 6,
        "fixed_hold_3d": 18,
    },
    "1d": {
        "fixed_hold_1d": 1,
        "fixed_hold_3d": 3,
        "fixed_hold_10d": 10,
    },
}
FIXED_NOTIONAL = 1000.0
ROUND_TRIP_COST_RATE = 0.001
EVENT_COOLDOWN_BARS = {"4h": 3, "1d": 1}

REQUIRED_OUTPUT_FILES = [
    "trend_entry_timing_summary.json",
    "trend_entry_timing_report.md",
    "legacy_entry_timing_diagnostics.csv",
    "candidate_entry_events.csv",
    "candidate_entry_family_summary.csv",
    "candidate_entry_trade_tests.csv",
    "candidate_entry_by_symbol.csv",
    "candidate_entry_by_timeframe.csv",
    "candidate_entry_by_split.csv",
    "candidate_entry_concentration.csv",
    "candidate_entry_reverse_test.csv",
    "candidate_entry_random_control.csv",
    "rejected_candidate_entry_families.csv",
    "data_quality.json",
]


class TrendEntryTimingResearchError(Exception):
    """Raised when entry timing research cannot continue."""


@dataclass(frozen=True, slots=True)
class EntryTimingOutputs:
    """Generated outputs for tests and CLI reporting."""

    output_dir: Path
    summary: dict[str, Any]
    legacy_diagnostics: pd.DataFrame
    events: pd.DataFrame
    family_summary: pd.DataFrame


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description="Research-only Trend Entry Timing v1.")
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
    """Resolve a path relative to project root."""

    return tce.resolve_path(path_arg)


def parse_csv_list(value: str | Iterable[str]) -> list[str]:
    """Parse comma/space separated values."""

    if isinstance(value, str):
        tokens = value.replace(",", " ").split()
    else:
        tokens = [str(item) for item in value]
    result: list[str] = []
    for token in tokens:
        if token and token not in result:
            result.append(token)
    return result


def split_for_time(timestamp: pd.Timestamp, timezone_name: str) -> str:
    """Return extended split label."""

    return tce.infer_split(timestamp, timezone_name)


def split_boundaries(timezone_name: str) -> dict[str, tuple[pd.Timestamp, pd.Timestamp]]:
    """Return split boundaries in local timezone."""

    return {
        split: (pd.Timestamp(start, tz=timezone_name), pd.Timestamp(end, tz=timezone_name))
        for split, (start, end) in tce.SPLIT_RANGES.items()
    }


def split_mask(times: pd.Series, split: str, timezone_name: str) -> pd.Series:
    """Return mask for one split."""

    start, end = split_boundaries(timezone_name)[split]
    parsed = pd.to_datetime(times)
    return (parsed >= start) & (parsed < end)


def safe_divide(numerator: float, denominator: float) -> float | None:
    """Return numerator / denominator when meaningful."""

    if denominator == 0 or not np.isfinite(denominator):
        return None
    return float(numerator / denominator)


def direction_matches(event_direction: str, segment_direction: Any) -> bool:
    """Return whether long/short matches up/down segment direction."""

    direction = str(event_direction or "").lower()
    segment = str(segment_direction or "").lower()
    return bool((direction == "long" and segment == "up") or (direction == "short" and segment == "down"))


def classify_entry_phase(entry_lag_pct: float | None, matched: bool) -> str:
    """Classify entry phase for entry timing research."""

    if not matched or entry_lag_pct is None or not np.isfinite(entry_lag_pct):
        return "nontrend"
    if entry_lag_pct < 0:
        return "pre_trend"
    if entry_lag_pct <= 0.10:
        return "first_10pct"
    if entry_lag_pct <= 0.25:
        return "first_25pct"
    if entry_lag_pct <= 0.75:
        return "middle_25_75pct"
    if entry_lag_pct <= 1.0:
        return "late_75pct_plus"
    return "after_trend"


def add_basic_indicators(frame: pd.DataFrame) -> pd.DataFrame:
    """Add prior-known indicator columns on closed bars."""

    result = frame.copy().sort_values("datetime", kind="stable").reset_index(drop=True)
    if result.empty:
        return result
    for column in ["open", "high", "low", "close", "volume"]:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    close = result["close"]
    result["ret_1"] = close.pct_change(1)
    result["ret_3"] = close / close.shift(3) - 1.0
    result["ret_6"] = close / close.shift(6) - 1.0
    result["ret_20"] = close / close.shift(20) - 1.0
    result["ret_55"] = close / close.shift(55) - 1.0
    diff_abs = close.diff().abs()
    for window in [20, 55]:
        result[f"eff_{window}"] = (close - close.shift(window)).abs() / diff_abs.rolling(window, min_periods=window).sum().replace(0, np.nan)
        result[f"range_high_{window}_prev"] = result["high"].rolling(window, min_periods=window).max().shift(1)
        result[f"range_low_{window}_prev"] = result["low"].rolling(window, min_periods=window).min().shift(1)
        result[f"mid_{window}_prev"] = close.rolling(window, min_periods=window).median().shift(1)
    result["eff_20_rising"] = result["eff_20"] > result["eff_20"].shift(3)
    result["vol_20"] = result["ret_1"].rolling(20, min_periods=10).std(ddof=0)
    result["atr14"] = result.get("atr14", tce.true_range(result).rolling(14, min_periods=1).mean())
    if "_time_ns" not in result.columns:
        result = tce.add_time_ns(result)
    return result


def attach_funding_to_frame(frame: pd.DataFrame, funding: pd.DataFrame | None) -> pd.DataFrame:
    """Attach latest known funding rate to closed bars."""

    result = frame.copy()
    if result.empty:
        result["funding_rate"] = pd.Series(dtype=float)
        return result
    if funding is None or funding.empty:
        result["funding_rate"] = 0.0
        return result
    left = result[["datetime"]].copy()
    left["datetime_utc"] = pd.to_datetime(left["datetime"]).dt.tz_convert("UTC")
    right = funding[["funding_time_utc", "funding_rate"]].copy().sort_values("funding_time_utc", kind="stable")
    merged = pd.merge_asof(
        left.sort_values("datetime_utc", kind="stable"),
        right,
        left_on="datetime_utc",
        right_on="funding_time_utc",
        direction="backward",
    )
    result["funding_rate"] = pd.to_numeric(merged.sort_index()["funding_rate"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    return result


def build_market_features(frames: dict[tuple[str, str], pd.DataFrame], symbols: list[str], timeframes: list[str]) -> dict[str, pd.DataFrame]:
    """Build market-wide breadth features by timeframe."""

    output: dict[str, pd.DataFrame] = {}
    for timeframe in timeframes:
        parts: list[pd.DataFrame] = []
        for symbol in symbols:
            frame = frames.get((symbol, timeframe), pd.DataFrame())
            if frame.empty:
                continue
            parts.append(
                frame[["datetime", "symbol", "close", "ret_3", "ret_20", "funding_rate"]].copy()
            )
        if not parts:
            output[timeframe] = pd.DataFrame()
            continue
        combined = pd.concat(parts, ignore_index=True)
        pivot_ret3 = combined.pivot(index="datetime", columns="symbol", values="ret_3").sort_index()
        pivot_ret20 = combined.pivot(index="datetime", columns="symbol", values="ret_20").sort_index()
        pivot_close = combined.pivot(index="datetime", columns="symbol", values="close").sort_index()
        pivot_funding = combined.pivot(index="datetime", columns="symbol", values="funding_rate").sort_index()
        market = pd.DataFrame(index=pivot_ret3.index)
        median_20 = pivot_close.rolling(20, min_periods=10).median()
        market["positive_ret3_count"] = (pivot_ret3 > 0).sum(axis=1)
        market["negative_ret3_count"] = (pivot_ret3 < 0).sum(axis=1)
        market["above_median_count"] = (pivot_close > median_20).sum(axis=1)
        market["below_median_count"] = (pivot_close < median_20).sum(axis=1)
        market["market_ret3_mean"] = pivot_ret3.mean(axis=1)
        market["dispersion_ret3"] = pivot_ret3.std(axis=1, ddof=0)
        market["funding_dispersion"] = pivot_funding.std(axis=1, ddof=0).fillna(0.0)
        market["positive_funding_count"] = (pivot_funding > 0).sum(axis=1)
        market["negative_funding_count"] = (pivot_funding < 0).sum(axis=1)
        rolling_corr = pivot_ret3.rolling(20, min_periods=10).corr()
        corr_values: list[float] = []
        for timestamp in market.index:
            try:
                corr_matrix = rolling_corr.loc[timestamp]
                values = corr_matrix.where(~np.eye(len(corr_matrix), dtype=bool)).stack()
                corr_values.append(float(values.mean()) if len(values.index) else np.nan)
            except Exception:
                corr_values.append(np.nan)
        market["avg_pairwise_corr_20"] = corr_values
        market["avg_pairwise_corr_20_rising"] = market["avg_pairwise_corr_20"] > market["avg_pairwise_corr_20"].shift(3)
        output[timeframe] = market.reset_index()
    return output


def add_relative_strength(frames: dict[tuple[str, str], pd.DataFrame], symbols: list[str], timeframes: list[str]) -> None:
    """Attach cross-sectional return ranks in-place."""

    for timeframe in timeframes:
        for horizon in [20, 55]:
            parts = []
            for symbol in symbols:
                frame = frames.get((symbol, timeframe), pd.DataFrame())
                if not frame.empty:
                    parts.append(frame[["datetime", "symbol", f"ret_{horizon}"]].copy())
            if not parts:
                continue
            combined = pd.concat(parts, ignore_index=True)
            pivot = combined.pivot(index="datetime", columns="symbol", values=f"ret_{horizon}").sort_index()
            long_rank = pivot.rank(axis=1, ascending=False, method="min")
            short_rank = pivot.rank(axis=1, ascending=True, method="min")
            for symbol in symbols:
                frame = frames.get((symbol, timeframe))
                if frame is None or frame.empty or symbol not in long_rank.columns:
                    continue
                mapped_long = long_rank[symbol].reindex(pd.to_datetime(frame["datetime"])).to_numpy(dtype=float)
                mapped_short = short_rank[symbol].reindex(pd.to_datetime(frame["datetime"])).to_numpy(dtype=float)
                frame[f"rs_rank_long_{horizon}"] = mapped_long
                frame[f"rs_rank_short_{horizon}"] = mapped_short


def train_thresholds(frames: dict[tuple[str, str], pd.DataFrame], market_features: dict[str, pd.DataFrame], timezone_name: str) -> dict[str, dict[str, float]]:
    """Define all event thresholds from train_ext only."""

    thresholds: dict[str, dict[str, float]] = {}

    def finite_quantile(series: pd.Series, q: float, default: float) -> float:
        value = tce.finite_float(pd.to_numeric(series, errors="coerce").quantile(q), default=np.nan)
        return float(value) if value is not None and np.isfinite(value) else float(default)

    def finite_ceil_quantile(series: pd.Series, q: float, default: float) -> float:
        value = finite_quantile(series, q, default)
        return float(math.ceil(value)) if np.isfinite(value) else float(default)

    for timeframe in sorted({key[1] for key in frames}):
        train_frames = []
        for (_, tf), frame in frames.items():
            if tf != timeframe or frame.empty:
                continue
            train_frames.append(frame.loc[split_mask(frame["datetime"], "train_ext", timezone_name)])
        train = pd.concat(train_frames, ignore_index=True) if train_frames else pd.DataFrame()
        market = market_features.get(timeframe, pd.DataFrame())
        market_train = market.loc[split_mask(market["datetime"], "train_ext", timezone_name)] if not market.empty else pd.DataFrame()
        thresholds[timeframe] = {
            "ret3_long": finite_quantile(train["ret_3"], 0.70, 0.0) if "ret_3" in train else 0.0,
            "ret3_short": finite_quantile(train["ret_3"], 0.30, 0.0) if "ret_3" in train else 0.0,
            "ret6_long": finite_quantile(train["ret_6"], 0.65, 0.0) if "ret_6" in train else 0.0,
            "ret6_short": finite_quantile(train["ret_6"], 0.35, 0.0) if "ret_6" in train else 0.0,
            "eff_delta": finite_quantile(train["eff_20"] - train["eff_20"].shift(3), 0.60, 0.0) if "eff_20" in train else 0.0,
            "vol_max": finite_quantile(train["vol_20"], 0.80, np.inf) if "vol_20" in train else np.inf,
            "funding_abs_max": finite_quantile(train["funding_rate"].abs(), 0.80, 1.0) if "funding_rate" in train else 1.0,
            "breadth_long_min": float(max(2, finite_ceil_quantile(market_train["positive_ret3_count"], 0.60, 2.0))) if "positive_ret3_count" in market_train else 2.0,
            "breadth_short_min": float(max(2, finite_ceil_quantile(market_train["negative_ret3_count"], 0.60, 2.0))) if "negative_ret3_count" in market_train else 2.0,
            "dispersion_max": finite_quantile(market_train["dispersion_ret3"], 0.80, 1.0) if "dispersion_ret3" in market_train else 1.0,
            "funding_dispersion_max": finite_quantile(market_train["funding_dispersion"], 0.80, 1.0) if "funding_dispersion" in market_train else 1.0,
            "funding_sign_crowd_max": float(max(3, math.ceil(len(DEFAULT_SYMBOLS) * 0.8))),
        }
    return thresholds


def merge_market(frame: pd.DataFrame, market: pd.DataFrame) -> pd.DataFrame:
    """Attach same-time market features."""

    if frame.empty or market.empty:
        return frame.copy()
    merged = frame.merge(market, on="datetime", how="left", suffixes=("", "_market"))
    return merged


def append_event(rows: list[dict[str, Any]], frame: pd.DataFrame, idx: int, family: str, direction: str, timeframe: str, symbol: str, timezone_name: str, extra: dict[str, Any] | None = None) -> None:
    """Append one candidate event row."""

    row = frame.iloc[idx]
    event_time = pd.Timestamp(row["datetime"])
    event_id = f"{family}_{symbol}_{timeframe}_{direction}_{event_time.isoformat()}"
    payload = {
        "event_id": hashlib.sha1(event_id.encode("utf-8")).hexdigest()[:20],
        "family": family,
        "symbol": symbol,
        "timeframe": timeframe,
        "direction": direction,
        "event_time": event_time.isoformat(),
        "event_price": float(row["close"]),
        "split": split_for_time(event_time, timezone_name),
        "ret_3": tce.finite_float(row.get("ret_3"), default=np.nan),
        "ret_6": tce.finite_float(row.get("ret_6"), default=np.nan),
        "ret_20": tce.finite_float(row.get("ret_20"), default=np.nan),
        "funding_rate": tce.finite_float(row.get("funding_rate"), default=0.0),
    }
    if extra:
        payload.update(extra)
    rows.append(payload)


def apply_cooldown(events: pd.DataFrame) -> pd.DataFrame:
    """Deduplicate clustered events with a fixed per-timeframe cooldown."""

    if events.empty:
        return events
    rows: list[pd.Series] = []
    for _, group in events.sort_values(["family", "symbol", "timeframe", "direction", "event_time"], kind="stable").groupby(
        ["family", "symbol", "timeframe", "direction"],
        dropna=False,
    ):
        last_time: pd.Timestamp | None = None
        timeframe = str(group.iloc[0]["timeframe"])
        cooldown = pd.Timedelta(minutes=TIMEFRAME_MINUTES[timeframe] * EVENT_COOLDOWN_BARS[timeframe])
        for _, row in group.iterrows():
            current = pd.Timestamp(row["event_time"])
            if last_time is None or current - last_time >= cooldown:
                rows.append(row)
                last_time = current
    return pd.DataFrame(rows).reset_index(drop=True) if rows else pd.DataFrame(columns=events.columns)


def generate_breakout_retest_events(frame: pd.DataFrame, family: str, symbol: str, timeframe: str, timezone_name: str) -> list[dict[str, Any]]:
    """Generate breakout-retest-reclaim events."""

    rows: list[dict[str, Any]] = []
    if frame.empty:
        return rows
    highs = frame["high"].to_numpy(dtype=float)
    lows = frame["low"].to_numpy(dtype=float)
    closes = frame["close"].to_numpy(dtype=float)
    high20 = frame["range_high_20_prev"].to_numpy(dtype=float)
    low20 = frame["range_low_20_prev"].to_numpy(dtype=float)
    atr = frame["atr14"].to_numpy(dtype=float)
    lookahead = 6 if timeframe == "4h" else 3
    for idx in range(len(frame.index) - 1):
        if np.isfinite(high20[idx]) and closes[idx] > high20[idx]:
            level = high20[idx]
            invalid = low20[idx] if np.isfinite(low20[idx]) else level - atr[idx]
            for future in range(idx + 1, min(len(frame.index), idx + lookahead + 1)):
                near_level = lows[future] <= level * 1.01
                not_invalid = lows[future] >= invalid
                reclaim = closes[future] > level
                if near_level and not_invalid and reclaim:
                    append_event(rows, frame, future, family, "long", timeframe, symbol, timezone_name, {"breakout_level": float(level)})
                    break
        if np.isfinite(low20[idx]) and closes[idx] < low20[idx]:
            level = low20[idx]
            invalid = high20[idx] if np.isfinite(high20[idx]) else level + atr[idx]
            for future in range(idx + 1, min(len(frame.index), idx + lookahead + 1)):
                near_level = highs[future] >= level * 0.99
                not_invalid = highs[future] <= invalid
                reclaim = closes[future] < level
                if near_level and not_invalid and reclaim:
                    append_event(rows, frame, future, family, "short", timeframe, symbol, timezone_name, {"breakout_level": float(level)})
                    break
    return rows


def generate_candidate_events(
    frames: dict[tuple[str, str], pd.DataFrame],
    market_features: dict[str, pd.DataFrame],
    thresholds: dict[str, dict[str, float]],
    symbols: list[str],
    timeframes: list[str],
    timezone_name: str,
) -> pd.DataFrame:
    """Generate all research-only candidate entry events."""

    rows: list[dict[str, Any]] = []
    for symbol in symbols:
        for timeframe in timeframes:
            base = frames.get((symbol, timeframe), pd.DataFrame())
            if base.empty:
                continue
            frame = merge_market(base, market_features.get(timeframe, pd.DataFrame()))
            th = thresholds[timeframe]
            eff_delta = frame["eff_20"] - frame["eff_20"].shift(3)
            vol_ok = frame["vol_20"] <= th["vol_max"]
            funding_ok = frame["funding_rate"].abs() <= th["funding_abs_max"]

            long_pre = (
                (frame["ret_3"] > th["ret3_long"])
                & (frame["ret_6"] > th["ret6_long"])
                & (eff_delta > th["eff_delta"])
                & (frame["close"] > frame["mid_20_prev"])
                & (frame["close"] <= frame["range_high_55_prev"])
                & vol_ok
            )
            short_pre = (
                (frame["ret_3"] < th["ret3_short"])
                & (frame["ret_6"] < th["ret6_short"])
                & (eff_delta > th["eff_delta"])
                & (frame["close"] < frame["mid_20_prev"])
                & (frame["close"] >= frame["range_low_55_prev"])
                & vol_ok
            )
            for idx in np.flatnonzero(long_pre.fillna(False).to_numpy()):
                append_event(rows, frame, int(idx), "pre_breakout_momentum_acceleration", "long", timeframe, symbol, timezone_name)
            for idx in np.flatnonzero(short_pre.fillna(False).to_numpy()):
                append_event(rows, frame, int(idx), "pre_breakout_momentum_acceleration", "short", timeframe, symbol, timezone_name)

            rows.extend(generate_breakout_retest_events(frame, "breakout_retest_reclaim", symbol, timeframe, timezone_name))

            long_breadth = (
                (frame["ret_3"] > 0)
                & (frame["positive_ret3_count"] >= th["breadth_long_min"])
                & (frame["positive_ret3_count"] > frame["positive_ret3_count"].shift(2))
                & (frame["above_median_count"] >= th["breadth_long_min"])
                & (frame["dispersion_ret3"] <= th["dispersion_max"])
            )
            short_breadth = (
                (frame["ret_3"] < 0)
                & (frame["negative_ret3_count"] >= th["breadth_short_min"])
                & (frame["negative_ret3_count"] > frame["negative_ret3_count"].shift(2))
                & (frame["below_median_count"] >= th["breadth_short_min"])
                & (frame["dispersion_ret3"] <= th["dispersion_max"])
            )
            for idx in np.flatnonzero(long_breadth.fillna(False).to_numpy()):
                append_event(rows, frame, int(idx), "cross_symbol_breadth_acceleration", "long", timeframe, symbol, timezone_name)
            for idx in np.flatnonzero(short_breadth.fillna(False).to_numpy()):
                append_event(rows, frame, int(idx), "cross_symbol_breadth_acceleration", "short", timeframe, symbol, timezone_name)

            long_funding = (
                (frame["ret_3"] > th["ret3_long"])
                & (frame["ret_6"] > 0)
                & funding_ok
                & (frame["funding_dispersion"] <= th["funding_dispersion_max"])
                & (frame["positive_funding_count"] <= th["funding_sign_crowd_max"])
            )
            short_funding = (
                (frame["ret_3"] < th["ret3_short"])
                & (frame["ret_6"] < 0)
                & funding_ok
                & (frame["funding_dispersion"] <= th["funding_dispersion_max"])
                & (frame["negative_funding_count"] <= th["funding_sign_crowd_max"])
            )
            for idx in np.flatnonzero(long_funding.fillna(False).to_numpy()):
                append_event(rows, frame, int(idx), "funding_neutral_momentum", "long", timeframe, symbol, timezone_name)
            for idx in np.flatnonzero(short_funding.fillna(False).to_numpy()):
                append_event(rows, frame, int(idx), "funding_neutral_momentum", "short", timeframe, symbol, timezone_name)

            long_rs = (
                ((frame["rs_rank_long_20"] <= 2) | (frame["rs_rank_long_55"] <= 2))
                & ((frame["ret_20"] > frame["ret_20"].shift(3)) | (frame["rs_rank_long_20"] < frame["rs_rank_long_20"].shift(3)))
                & (frame["positive_ret3_count"] >= 2)
                & funding_ok
            )
            short_rs = (
                ((frame["rs_rank_short_20"] <= 2) | (frame["rs_rank_short_55"] <= 2))
                & ((frame["ret_20"] < frame["ret_20"].shift(3)) | (frame["rs_rank_short_20"] < frame["rs_rank_short_20"].shift(3)))
                & (frame["negative_ret3_count"] >= 2)
                & funding_ok
            )
            for idx in np.flatnonzero(long_rs.fillna(False).to_numpy()):
                append_event(rows, frame, int(idx), "relative_strength_leader", "long", timeframe, symbol, timezone_name)
            for idx in np.flatnonzero(short_rs.fillna(False).to_numpy()):
                append_event(rows, frame, int(idx), "relative_strength_leader", "short", timeframe, symbol, timezone_name)
    if not rows:
        return pd.DataFrame()
    events = pd.DataFrame(rows).sort_values(["family", "symbol", "timeframe", "event_time"], kind="stable").reset_index(drop=True)
    return apply_cooldown(events)


def build_segments_by_key(segments: pd.DataFrame, timeframes: list[str]) -> dict[tuple[str, str], pd.DataFrame]:
    """Build normalized segment frames by symbol/timeframe."""

    if segments.empty:
        return {}
    selected = segments[segments["timeframe"].astype(str).isin(timeframes)].copy()
    start_ns = np.array([pd.Timestamp(value).value for value in selected["start_ts"]], dtype=np.int64)
    end_ns = np.array([pd.Timestamp(value).value for value in selected["end_ts"]], dtype=np.int64)
    duration_ns = np.maximum(end_ns - start_ns, 1)
    timeframe_minutes = selected["timeframe"].astype(str).map(TIMEFRAME_MINUTES).fillna(0).astype(float)
    min_pre_ns = (timeframe_minutes * 2.0 * 60.0 * 1_000_000_000.0).astype("int64")
    pre_ns = np.maximum(min_pre_ns.to_numpy(dtype=np.int64), (duration_ns * 0.25).astype(np.int64))
    selected["_start_ns"] = start_ns
    selected["_end_ns"] = end_ns
    selected["_pre_start_ns"] = start_ns - pre_ns
    selected["_direction_norm"] = selected["direction"].astype(str).str.lower()
    return {
        (str(symbol), str(timeframe)): group.sort_values("start_ts", kind="stable").reset_index(drop=True)
        for (symbol, timeframe), group in selected.groupby(["symbol", "timeframe"], dropna=False)
    }


def select_segment_for_entry(symbol: str, timeframe: str, direction: str, entry_time: pd.Timestamp, segments_by_key: dict[tuple[str, str], pd.DataFrame]) -> pd.Series | None:
    """Select a nearby segment for entry timing diagnostics/evaluation."""

    segments = segments_by_key.get((symbol, timeframe))
    if segments is None or segments.empty:
        return None
    entry_ns = pd.Timestamp(entry_time).value
    if {"_start_ns", "_end_ns", "_pre_start_ns", "_direction_norm"}.issubset(segments.columns):
        pre_start = segments["_pre_start_ns"].to_numpy(dtype=np.int64)
        ends = segments["_end_ns"].to_numpy(dtype=np.int64)
        candidate_positions = np.flatnonzero((entry_ns >= pre_start) & (entry_ns <= ends))
        if len(candidate_positions) == 0:
            return None
        starts = segments["_start_ns"].to_numpy(dtype=np.int64)[candidate_positions]
        direction_norm = str(direction or "").lower()
        directions = segments["_direction_norm"].to_numpy(dtype=object)[candidate_positions]
        direction_bonus = (
            ((direction_norm == "long") & (directions == "up"))
            | ((direction_norm == "short") & (directions == "down"))
        ).astype(np.int8)
        lag_abs = np.abs(entry_ns - starts)
        rank = np.lexsort((lag_abs, -direction_bonus))
        return segments.iloc[int(candidate_positions[int(rank[0])])]

    best: tuple[float, float] | None = None
    best_row: pd.Series | None = None
    minutes = TIMEFRAME_MINUTES[timeframe]
    for _, segment in segments.iterrows():
        start = pd.Timestamp(segment["start_ts"])
        end = pd.Timestamp(segment["end_ts"])
        duration_seconds = max((end - start).total_seconds(), 1.0)
        pre_window = max(pd.Timedelta(minutes=minutes * 2), pd.Timedelta(seconds=duration_seconds * 0.25))
        if entry_time < start - pre_window or entry_time > end:
            continue
        direction_bonus = 1.0 if direction_matches(direction, segment.get("direction")) else 0.0
        lag_abs = abs((entry_time - start).total_seconds() / 60.0 / minutes)
        key = (direction_bonus, -lag_abs)
        if best is None or key > best:
            best = key
            best_row = segment
    return best_row


def segment_path_metrics(frame: pd.DataFrame, segment: pd.Series | None, direction: str, entry_time: pd.Timestamp, entry_price: float) -> tuple[float | None, float | None]:
    """Compute missed MFE before entry and remaining MFE after entry using ex-post labels."""

    if frame.empty or segment is None:
        return None, None
    start = pd.Timestamp(segment["start_ts"])
    end = pd.Timestamp(segment["end_ts"])
    start_price = tce.finite_float(segment.get("start_price"), default=None)
    if start_price is None or start_price <= 0:
        start_price = float(frame.loc[pd.to_datetime(frame["datetime"]) >= start, "close"].iloc[0])
    segment_frame = frame[(pd.to_datetime(frame["datetime"]) >= start) & (pd.to_datetime(frame["datetime"]) <= end)].copy()
    if segment_frame.empty:
        return None, None
    before = segment_frame[pd.to_datetime(segment_frame["datetime"]) <= entry_time]
    after = segment_frame[pd.to_datetime(segment_frame["datetime"]) >= max(entry_time, start)]
    if str(direction).lower() == "short":
        missed = 0.0 if before.empty else max(float(start_price) / float(before["low"].min()) - 1.0, 0.0)
        remaining = 0.0 if after.empty else max(float(entry_price) / float(after["low"].min()) - 1.0, 0.0)
    else:
        missed = 0.0 if before.empty else max(float(before["high"].max()) / float(start_price) - 1.0, 0.0)
        remaining = 0.0 if after.empty else max(float(after["high"].max()) / float(entry_price) - 1.0, 0.0)
    return float(missed), float(remaining)


def entry_timing_row(
    *,
    symbol: str,
    timeframe: str,
    direction: str,
    entry_time: pd.Timestamp,
    entry_price: float,
    exit_time: pd.Timestamp | None,
    segment: pd.Series | None,
    frame: pd.DataFrame,
    compute_path_metrics: bool = True,
) -> dict[str, Any]:
    """Build common entry timing diagnostic fields."""

    matched = segment is not None
    start = pd.Timestamp(segment["start_ts"]) if matched else None
    end = pd.Timestamp(segment["end_ts"]) if matched else None
    minutes = TIMEFRAME_MINUTES[timeframe]
    entry_lag_bars = None
    entry_lag_pct = None
    captured = 0.0
    entry_price_vs_start = None
    if matched and start is not None and end is not None:
        duration_seconds = max((end - start).total_seconds(), 1.0)
        entry_lag_bars = float((entry_time - start).total_seconds() / 60.0 / minutes)
        entry_lag_pct = float((entry_time - start).total_seconds() / duration_seconds)
        if exit_time is not None:
            captured = tce.overlap_fraction(entry_time, exit_time, start, end)
        start_price = tce.finite_float(segment.get("start_price"), default=None)
        if start_price and start_price > 0:
            if direction == "short":
                entry_price_vs_start = float(start_price / entry_price - 1.0)
            else:
                entry_price_vs_start = float(entry_price / start_price - 1.0)
    phase = classify_entry_phase(entry_lag_pct, matched)
    if compute_path_metrics:
        missed, remaining = segment_path_metrics(frame, segment, direction, entry_time, entry_price)
    elif matched:
        missed = max(tce.finite_float(entry_price_vs_start, default=0.0) or 0.0, 0.0)
        remaining = tce.finite_float(segment.get("abs_trend_return"), default=None)
    else:
        missed, remaining = None, None
    return {
        "trend_segment_id": segment.get("trend_segment_id") if matched else None,
        "trend_direction": segment.get("direction") if matched else None,
        "entry_lag_bars": entry_lag_bars,
        "entry_lag_pct_of_segment": entry_lag_pct,
        "entry_phase": phase,
        "direction_matches_segment": bool(matched and direction_matches(direction, segment.get("direction"))),
        "entry_price_vs_segment_start_price": entry_price_vs_start,
        "missed_mfe_before_entry": missed,
        "remaining_mfe_after_entry": remaining,
        "captured_fraction": captured,
        "early_exit_flag": bool(matched and captured < 0.50) if exit_time is not None else None,
        "late_entry_flag": phase in {"middle_25_75pct", "late_75pct_plus", "after_trend"},
    }


def build_legacy_entry_diagnostics(
    trades: pd.DataFrame,
    segments_by_key: dict[tuple[str, str], pd.DataFrame],
    frames: dict[tuple[str, str], pd.DataFrame],
) -> pd.DataFrame:
    """Build legacy V3 entry timing diagnostics."""

    rows: list[dict[str, Any]] = []
    for _, trade in trades.iterrows():
        symbol = str(trade["symbol"])
        timeframe = str(trade.get("trade_timeframe") or "")
        if timeframe not in TIMEFRAME_MINUTES:
            continue
        direction = str(trade["direction"])
        entry_time = pd.Timestamp(trade["entry_ts"])
        exit_time = pd.Timestamp(trade["exit_ts"])
        frame = frames.get((symbol, timeframe), pd.DataFrame())
        segment = select_segment_for_entry(symbol, timeframe, direction, entry_time, segments_by_key)
        timing = entry_timing_row(
            symbol=symbol,
            timeframe=timeframe,
            direction=direction,
            entry_time=entry_time,
            entry_price=float(trade["entry_price"]),
            exit_time=exit_time,
            segment=segment,
            frame=frame,
            compute_path_metrics=False,
        )
        rows.append(
            {
                "strategy_source": trade.get("strategy_source"),
                "policy_or_group": trade.get("policy_or_group"),
                "symbol": symbol,
                "timeframe": timeframe,
                "trade_id": trade.get("trade_id"),
                "split": trade.get("split"),
                "direction": direction,
                "entry_time": trade.get("entry_time"),
                "entry_price": trade.get("entry_price"),
                "exit_time": trade.get("exit_time"),
                **timing,
            }
        )
    return pd.DataFrame(rows)


def annotate_events(
    events: pd.DataFrame,
    segments_by_key: dict[tuple[str, str], pd.DataFrame],
    frames: dict[tuple[str, str], pd.DataFrame],
) -> pd.DataFrame:
    """Annotate candidate events with ex-post trend label diagnostics."""

    if events.empty:
        return events
    rows: list[dict[str, Any]] = []
    for _, event in events.iterrows():
        symbol = str(event["symbol"])
        timeframe = str(event["timeframe"])
        direction = str(event["direction"])
        event_time = pd.Timestamp(event["event_time"])
        event_price = float(event["event_price"])
        frame = frames.get((symbol, timeframe), pd.DataFrame())
        segment = select_segment_for_entry(symbol, timeframe, direction, event_time, segments_by_key)
        timing = entry_timing_row(
            symbol=symbol,
            timeframe=timeframe,
            direction=direction,
            entry_time=event_time,
            entry_price=event_price,
            exit_time=None,
            segment=segment,
            frame=frame,
            compute_path_metrics=False,
        )
        rows.append({**event.to_dict(), **timing})
    return pd.DataFrame(rows)


def fixed_hold_exit(frame: pd.DataFrame, event_time: pd.Timestamp, hold_bars: int) -> pd.Series | None:
    """Return exit bar by fixed number of closed bars."""

    if frame.empty:
        return None
    entry_idx = tce.last_completed_bar_index(frame, event_time)
    if entry_idx is None:
        return None
    exit_idx = min(entry_idx + hold_bars, len(frame.index) - 1)
    return frame.iloc[exit_idx]


def event_trade_row(
    event: pd.Series,
    frame: pd.DataFrame,
    hold_label: str,
    hold_bars: int,
    funding_indexes: dict[str, tce.FundingIndex],
    *,
    reverse: bool = False,
) -> dict[str, Any] | None:
    """Build one fixed-hold trade test row."""

    exit_bar = fixed_hold_exit(frame, pd.Timestamp(event["event_time"]), hold_bars)
    if exit_bar is None:
        return None
    entry_price = float(event["event_price"])
    exit_price = float(exit_bar["close"])
    direction = str(event["direction"])
    test_direction = "short" if direction == "long" else "long" if reverse else direction
    gross_return = exit_price / entry_price - 1.0
    if test_direction == "short":
        gross_return = entry_price / exit_price - 1.0
    no_cost = float(gross_return * FIXED_NOTIONAL)
    cost = FIXED_NOTIONAL * ROUND_TRIP_COST_RATE
    cost_aware = no_cost - cost
    inst_id = tce.symbol_to_inst_id(str(event["symbol"]))
    funding = tce.funding_for_trade(
        inst_id,
        test_direction,
        pd.Timestamp(event["event_time"]),
        pd.Timestamp(exit_bar["datetime"]),
        FIXED_NOTIONAL,
        funding_indexes,
    )
    return {
        "event_id": event["event_id"],
        "family": event["family"],
        "symbol": event["symbol"],
        "timeframe": event["timeframe"],
        "split": event["split"],
        "direction": direction,
        "test_direction": test_direction,
        "hold_label": hold_label,
        "hold_bars": hold_bars,
        "entry_time": event["event_time"],
        "entry_price": entry_price,
        "exit_time": pd.Timestamp(exit_bar["datetime"]).isoformat(),
        "exit_price": exit_price,
        "no_cost_pnl": no_cost,
        "cost_aware_pnl": cost_aware,
        "funding_pnl": float(funding["funding_pnl"]),
        "funding_adjusted_pnl": cost_aware + float(funding["funding_pnl"]),
        "funding_events_count": int(funding["funding_events_count"]),
        "funding_data_available": bool(funding["funding_data_available"]),
        "funding_interval_covered": bool(funding["funding_interval_covered"]),
        "trend_segment_id": event.get("trend_segment_id"),
        "direction_matches_segment": bool(event.get("direction_matches_segment")),
        "entry_phase": event.get("entry_phase"),
        "entry_lag_bars": event.get("entry_lag_bars"),
        "entry_lag_pct_of_segment": event.get("entry_lag_pct_of_segment"),
        "missed_mfe_before_entry": event.get("missed_mfe_before_entry"),
        "remaining_mfe_after_entry": event.get("remaining_mfe_after_entry"),
        "reverse": reverse,
    }


def build_trade_tests(events: pd.DataFrame, frames: dict[tuple[str, str], pd.DataFrame], funding_indexes: dict[str, tce.FundingIndex]) -> pd.DataFrame:
    """Build fixed-hold event trade tests."""

    rows: list[dict[str, Any]] = []
    if events.empty:
        return pd.DataFrame()
    for _, event in events.iterrows():
        frame = frames.get((str(event["symbol"]), str(event["timeframe"])), pd.DataFrame())
        for hold_label, hold_bars in FIXED_HOLD_BARS[str(event["timeframe"])].items():
            row = event_trade_row(event, frame, hold_label, hold_bars, funding_indexes, reverse=False)
            if row is not None:
                rows.append(row)
    return pd.DataFrame(rows)


def build_reverse_tests(events: pd.DataFrame, frames: dict[tuple[str, str], pd.DataFrame], funding_indexes: dict[str, tce.FundingIndex], selected_holds: dict[str, str]) -> pd.DataFrame:
    """Build reverse-direction tests using train-selected hold labels."""

    rows: list[dict[str, Any]] = []
    if events.empty:
        return pd.DataFrame()
    for _, event in events.iterrows():
        hold_label = selected_holds.get(str(event["family"]))
        if not hold_label:
            continue
        hold_bars = FIXED_HOLD_BARS[str(event["timeframe"])].get(hold_label)
        if hold_bars is None:
            continue
        frame = frames.get((str(event["symbol"]), str(event["timeframe"])), pd.DataFrame())
        row = event_trade_row(event, frame, hold_label, hold_bars, funding_indexes, reverse=True)
        if row is not None:
            rows.append(row)
    return pd.DataFrame(rows)


def deterministic_seed(text: str) -> int:
    """Stable deterministic seed."""

    return int(hashlib.sha1(text.encode("utf-8")).hexdigest()[:8], 16)


def build_random_controls(events: pd.DataFrame, frames: dict[tuple[str, str], pd.DataFrame], funding_indexes: dict[str, tce.FundingIndex], selected_holds: dict[str, str], timezone_name: str) -> pd.DataFrame:
    """Build random-time controls matching family/symbol/timeframe counts."""

    if events.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for (family, symbol, timeframe, direction, split), group in events.groupby(["family", "symbol", "timeframe", "direction", "split"], dropna=False):
        hold_label = selected_holds.get(str(family))
        hold_bars = FIXED_HOLD_BARS[str(timeframe)].get(hold_label or "")
        frame = frames.get((str(symbol), str(timeframe)), pd.DataFrame())
        if hold_bars is None or frame.empty:
            continue
        mask = split_mask(frame["datetime"], str(split), timezone_name)
        eligible = frame.loc[mask].reset_index(drop=True)
        if len(eligible.index) <= hold_bars + 60:
            continue
        rng = np.random.default_rng(deterministic_seed(f"{family}|{symbol}|{timeframe}|{direction}|{split}"))
        sample_size = min(len(group.index), len(eligible.index) - hold_bars)
        positions = rng.choice(np.arange(60, len(eligible.index) - hold_bars), size=sample_size, replace=False)
        for pos in positions:
            event_like = pd.Series(
                {
                    "event_id": f"random_{family}_{symbol}_{timeframe}_{direction}_{split}_{pos}",
                    "family": family,
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "split": split,
                    "direction": direction,
                    "event_time": pd.Timestamp(eligible.iloc[pos]["datetime"]).isoformat(),
                    "event_price": float(eligible.iloc[pos]["close"]),
                    "trend_segment_id": None,
                    "direction_matches_segment": False,
                    "entry_phase": "random_control",
                    "entry_lag_bars": None,
                    "entry_lag_pct_of_segment": None,
                    "missed_mfe_before_entry": None,
                    "remaining_mfe_after_entry": None,
                }
            )
            row = event_trade_row(event_like, frame, hold_label, hold_bars, funding_indexes, reverse=False)
            if row is not None:
                rows.append(row)
    return pd.DataFrame(rows)


def max_drawdown_from_pnl(pnl: pd.Series) -> float:
    """Return closed-trade max drawdown."""

    return tce.max_drawdown_from_pnl(pnl)


def largest_symbol_pnl_share(trades: pd.DataFrame) -> float | None:
    """Return largest symbol PnL share."""

    return tce.largest_symbol_pnl_share(trades, "funding_adjusted_pnl")


def top_5pct_trade_pnl_contribution(trades: pd.DataFrame) -> float | None:
    """Return top 5% contribution."""

    return tce.top_5pct_trade_pnl_contribution(trades, "funding_adjusted_pnl")


def summarize_trade_slice(trades: pd.DataFrame) -> dict[str, Any]:
    """Summarize candidate trade tests."""

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
        }
    pnl = pd.to_numeric(trades["funding_adjusted_pnl"], errors="coerce").fillna(0.0)
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    return {
        "trade_count": int(len(trades.index)),
        "no_cost_pnl": float(pd.to_numeric(trades["no_cost_pnl"], errors="coerce").fillna(0.0).sum()),
        "cost_aware_pnl": float(pd.to_numeric(trades["cost_aware_pnl"], errors="coerce").fillna(0.0).sum()),
        "funding_adjusted_pnl": float(pnl.sum()),
        "funding_events_count": int(pd.to_numeric(trades["funding_events_count"], errors="coerce").fillna(0).sum()),
        "funding_data_complete": bool(trades["funding_data_available"].astype(bool).all() and trades["funding_interval_covered"].astype(bool).all()),
        "win_rate": float((pnl > 0).mean()) if len(pnl.index) else None,
        "profit_factor": float(wins.sum() / abs(losses.sum())) if not losses.empty and not wins.empty else (0.0 if not losses.empty else None),
        "max_drawdown": max_drawdown_from_pnl(pnl),
        "largest_symbol_pnl_share": largest_symbol_pnl_share(trades),
        "top_5pct_trade_pnl_contribution": top_5pct_trade_pnl_contribution(trades),
    }


def group_summary(trades: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    """Summarize trade tests by groups."""

    if trades.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for keys, group in trades.groupby(group_columns, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_columns, keys, strict=False))
        row.update(summarize_trade_slice(group))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(group_columns, kind="stable").reset_index(drop=True)


def select_train_holds(trade_tests: pd.DataFrame) -> dict[str, str]:
    """Select fixed hold per family using train_ext no-cost only."""

    selected: dict[str, str] = {}
    if trade_tests.empty:
        return selected
    train = trade_tests[trade_tests["split"] == "train_ext"]
    grouped = group_summary(train, ["family", "hold_label"])
    if grouped.empty:
        return selected
    for family, group in grouped.groupby("family", dropna=False):
        group = group.sort_values(["no_cost_pnl", "trade_count", "hold_label"], ascending=[False, False, True], kind="stable")
        selected[str(family)] = str(group.iloc[0]["hold_label"])
    return selected


def segment_recall(events: pd.DataFrame, segments: pd.DataFrame, family: str) -> float | None:
    """Compute direction-matching early trend segment recall."""

    if segments.empty:
        return None
    fam = events[(events["family"] == family) & (events["direction_matches_segment"] == True)].copy()  # noqa: E712
    if fam.empty:
        return 0.0
    early = fam[fam["entry_phase"].isin(["pre_trend", "first_10pct", "first_25pct"])]
    recalled = set(early["trend_segment_id"].dropna().astype(str))
    total = int(len(segments.index))
    return float(len(recalled) / total) if total else None


def event_quality_summary(events: pd.DataFrame, segments: pd.DataFrame) -> pd.DataFrame:
    """Summarize ex-post label quality by family."""

    rows: list[dict[str, Any]] = []
    if events.empty:
        return pd.DataFrame()
    for family, group in events.groupby("family", dropna=False):
        direction_match = group["direction_matches_segment"].astype(bool)
        early = group["entry_phase"].isin(["pre_trend", "first_10pct", "first_25pct"])
        rows.append(
            {
                "family": family,
                "event_count": int(len(group.index)),
                "trend_segment_recall": segment_recall(events, segments, str(family)),
                "early_entry_rate": float((early & direction_match).mean()) if len(group.index) else None,
                "direction_match_rate": float(direction_match.mean()) if len(group.index) else None,
                "median_entry_lag_bars": tce.finite_float(pd.to_numeric(group["entry_lag_bars"], errors="coerce").median(), default=None),
                "median_entry_lag_pct": tce.finite_float(pd.to_numeric(group["entry_lag_pct_of_segment"], errors="coerce").median(), default=None),
                "average_remaining_mfe": tce.finite_float(pd.to_numeric(group["remaining_mfe_after_entry"], errors="coerce").mean(), default=None),
                "average_missed_mfe_before_entry": tce.finite_float(pd.to_numeric(group["missed_mfe_before_entry"], errors="coerce").mean(), default=None),
            }
        )
    return pd.DataFrame(rows)


def reverse_random_summary(frame: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """Summarize reverse or random control frame."""

    if frame.empty:
        return pd.DataFrame(columns=["family", f"{prefix}_no_cost_pnl", f"{prefix}_cost_aware_pnl", f"{prefix}_funding_adjusted_pnl", f"{prefix}_trade_count"])
    summary = group_summary(frame, ["family"])
    return summary.rename(
        columns={
            "trade_count": f"{prefix}_trade_count",
            "no_cost_pnl": f"{prefix}_no_cost_pnl",
            "cost_aware_pnl": f"{prefix}_cost_aware_pnl",
            "funding_adjusted_pnl": f"{prefix}_funding_adjusted_pnl",
        }
    )[["family", f"{prefix}_trade_count", f"{prefix}_no_cost_pnl", f"{prefix}_cost_aware_pnl", f"{prefix}_funding_adjusted_pnl"]]


def build_family_summary(events: pd.DataFrame, trade_tests: pd.DataFrame, segments: pd.DataFrame, reverse: pd.DataFrame, random_control: pd.DataFrame, selected_holds: dict[str, str]) -> pd.DataFrame:
    """Build one selected-hold summary row per event family."""

    quality = event_quality_summary(events, segments)
    rows: list[dict[str, Any]] = []
    reverse_summary = reverse_random_summary(reverse, "reverse")
    random_summary = reverse_random_summary(random_control, "random")
    for family in ENTRY_FAMILIES:
        hold = selected_holds.get(family)
        selected = trade_tests[(trade_tests["family"] == family) & (trade_tests["hold_label"] == hold)] if hold else pd.DataFrame()
        all_summary = summarize_trade_slice(selected)
        split_summary = group_summary(selected, ["split"]) if not selected.empty else pd.DataFrame()
        row: dict[str, Any] = {"family": family, "selected_hold_label": hold}
        q = quality[quality["family"] == family]
        if not q.empty:
            row.update(q.iloc[0].to_dict())
        else:
            row.update(
                {
                    "event_count": 0,
                    "trend_segment_recall": None,
                    "early_entry_rate": None,
                    "direction_match_rate": None,
                    "median_entry_lag_bars": None,
                    "median_entry_lag_pct": None,
                    "average_remaining_mfe": None,
                    "average_missed_mfe_before_entry": None,
                }
            )
        row.update(all_summary)
        for split in SPLITS:
            split_row = split_summary[split_summary["split"] == split] if not split_summary.empty else pd.DataFrame()
            metrics = summarize_trade_slice(pd.DataFrame()) if split_row.empty else split_row.iloc[0].to_dict()
            row[f"{split}_trade_count"] = int(metrics.get("trade_count") or 0)
            row[f"{split}_no_cost_pnl"] = float(metrics.get("no_cost_pnl") or 0.0)
            row[f"{split}_cost_aware_pnl"] = float(metrics.get("cost_aware_pnl") or 0.0)
            row[f"{split}_funding_adjusted_pnl"] = float(metrics.get("funding_adjusted_pnl") or 0.0)
        rev = reverse_summary[reverse_summary["family"] == family]
        rnd = random_summary[random_summary["family"] == family]
        row["reverse_test_result"] = float(rev.iloc[0]["reverse_funding_adjusted_pnl"]) if not rev.empty else None
        row["random_time_control_result"] = float(rnd.iloc[0]["random_funding_adjusted_pnl"]) if not rnd.empty else None
        row["reverse_no_cost_pnl"] = float(rev.iloc[0]["reverse_no_cost_pnl"]) if not rev.empty else None
        row["random_no_cost_pnl"] = float(rnd.iloc[0]["random_no_cost_pnl"]) if not rnd.empty else None
        rows.append(row)
    return pd.DataFrame(rows)


def evaluate_stable_gates(family_summary: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Evaluate entry timing stable-like gates."""

    rows: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    for _, row in family_summary.iterrows():
        reasons: list[str] = []
        family = str(row["family"])
        for split in SPLITS:
            if int(row.get(f"{split}_trade_count") or 0) < 10:
                reasons.append(f"{split}:trade_count_lt_10")
            if float(row.get(f"{split}_no_cost_pnl") or 0.0) <= 0:
                reasons.append(f"{split}:no_cost_pnl_not_positive")
        if float(row.get("oos_ext_cost_aware_pnl") or 0.0) < 0:
            reasons.append("oos_ext:cost_aware_pnl_negative")
        if float(row.get("oos_ext_funding_adjusted_pnl") or 0.0) < 0:
            reasons.append("oos_ext:funding_adjusted_pnl_negative")
        if (row.get("trend_segment_recall") is None) or float(row.get("trend_segment_recall") or 0.0) < 0.20:
            reasons.append("trend_segment_recall_lt_0.20")
        if (row.get("early_entry_rate") is None) or float(row.get("early_entry_rate") or 0.0) < 0.40:
            reasons.append("early_entry_rate_lt_0.40")
        if (row.get("direction_match_rate") is None) or float(row.get("direction_match_rate") or 0.0) < 0.55:
            reasons.append("direction_match_rate_lt_0.55")
        forward = float(row.get("funding_adjusted_pnl") or 0.0)
        reverse_result = row.get("reverse_test_result")
        random_result = row.get("random_time_control_result")
        if reverse_result is None or not np.isfinite(float(reverse_result)) or float(reverse_result) >= forward * 0.5:
            reasons.append("reverse_test_not_clearly_weaker")
        if random_result is None or not np.isfinite(float(random_result)) or float(random_result) >= forward * 0.5:
            reasons.append("random_time_control_not_clearly_weaker")
        largest = row.get("largest_symbol_pnl_share")
        top = row.get("top_5pct_trade_pnl_contribution")
        if largest is None or not np.isfinite(float(largest)) or float(largest) > 0.7:
            reasons.append("largest_symbol_pnl_share_gt_0.7")
        if top is None or not np.isfinite(float(top)) or float(top) > 0.8:
            reasons.append("top_5pct_trade_pnl_contribution_gt_0.8")
        stable = not reasons
        record = {
            "family": family,
            "stable_like": stable,
            "rejected_reasons": ";".join(reasons),
            "selected_hold_label": row.get("selected_hold_label"),
        }
        rows.append(record)
        if stable:
            candidates.append(record)
    return pd.DataFrame(rows), candidates


def legacy_summary_payload(legacy: pd.DataFrame) -> dict[str, Any]:
    """Summarize legacy entry timing diagnostics."""

    if legacy.empty:
        return {
            "late_entry_share": None,
            "avg_entry_lag_pct": None,
            "avg_missed_mfe_before_entry": None,
            "worst_policy": None,
            "worst_symbol": None,
            "late_entry_by_split": {},
        }
    late_entry_share = float(legacy["late_entry_flag"].astype(bool).mean())
    avg_lag = tce.finite_float(pd.to_numeric(legacy["entry_lag_pct_of_segment"], errors="coerce").mean(), default=None)
    avg_missed = tce.finite_float(pd.to_numeric(legacy["missed_mfe_before_entry"], errors="coerce").mean(), default=None)
    by_policy = (
        legacy.groupby("policy_or_group", dropna=False)
        .agg(late_entry_share=("late_entry_flag", "mean"), avg_entry_lag_pct=("entry_lag_pct_of_segment", "mean"), trade_count=("trade_id", "size"))
        .reset_index()
        .sort_values(["late_entry_share", "avg_entry_lag_pct", "trade_count"], ascending=[False, False, False], kind="stable")
    )
    by_symbol = (
        legacy.groupby("symbol", dropna=False)
        .agg(late_entry_share=("late_entry_flag", "mean"), avg_entry_lag_pct=("entry_lag_pct_of_segment", "mean"), trade_count=("trade_id", "size"))
        .reset_index()
        .sort_values(["late_entry_share", "avg_entry_lag_pct", "trade_count"], ascending=[False, False, False], kind="stable")
    )
    by_split = legacy.groupby("split", dropna=False)["late_entry_flag"].mean().to_dict()
    return {
        "late_entry_share": late_entry_share,
        "avg_entry_lag_pct": avg_lag,
        "avg_missed_mfe_before_entry": avg_missed,
        "worst_policy": tce.dataframe_records(by_policy.head(1))[0] if not by_policy.empty else None,
        "worst_symbol": tce.dataframe_records(by_symbol.head(1))[0] if not by_symbol.empty else None,
        "late_entry_by_split": {str(key): float(value) for key, value in by_split.items()},
        "late_entry_consistent_across_splits": bool(all(float(value) >= 0.50 for value in by_split.values())) if by_split else None,
    }


def build_summary_payload(
    *,
    output_dir: Path,
    warnings: list[str],
    trend_summary: dict[str, Any],
    data_quality: dict[str, Any],
    legacy_summary: dict[str, Any],
    family_summary: pd.DataFrame,
    rejected: pd.DataFrame,
    candidates: list[dict[str, Any]],
    thresholds: dict[str, dict[str, float]],
) -> dict[str, Any]:
    """Build JSON summary."""

    can_phase2 = bool(candidates)
    best_recall = family_summary.sort_values("trend_segment_recall", ascending=False, kind="stable").head(1) if not family_summary.empty else pd.DataFrame()
    best_early = family_summary.sort_values("early_entry_rate", ascending=False, kind="stable").head(1) if not family_summary.empty else pd.DataFrame()
    return {
        "mode": "research_only_trend_entry_timing_v1",
        "output_dir": str(output_dir),
        "output_files": REQUIRED_OUTPUT_FILES,
        "warnings": sorted(dict.fromkeys(warnings)),
        "labels_are_ex_post_only": True,
        "labels_must_not_be_used_as_entry_features": True,
        "thresholds_defined_from_train_ext_only": True,
        "thresholds": thresholds,
        "trend_opportunity_map": {
            "enough_trend_opportunities": trend_summary.get("enough_trend_opportunities"),
            "trend_opportunities_are_diversified": trend_summary.get("trend_opportunities_are_diversified"),
            "legacy_main_failure_mode": (trend_summary.get("legacy_analysis") or {}).get("main_failure_mode"),
        },
        "data_quality": data_quality,
        "legacy_entry_timing": legacy_summary,
        "candidate_family_summary": tce.dataframe_records(family_summary),
        "best_recall_family": tce.dataframe_records(best_recall)[0] if not best_recall.empty else None,
        "best_early_entry_family": tce.dataframe_records(best_early)[0] if not best_early.empty else None,
        "stable_like_candidates": candidates,
        "stable_like_candidate_exists": bool(candidates),
        "rejected_candidate_entry_families": tce.dataframe_records(rejected),
        "can_enter_entry_timing_phase2": can_phase2,
        "strategy_development_allowed": False,
        "demo_live_allowed": False,
        "recommended_next_step": "entry_timing_phase2_research" if can_phase2 else "pause_or_new_hypothesis",
    }


def format_number(value: Any, digits: int = 6) -> str:
    """Format optional number."""

    number = tce.finite_float(value, default=np.nan)
    if number is None or not np.isfinite(number):
        return "N/A"
    return f"{number:.{digits}f}"


def markdown_table(rows: list[dict[str, Any]], columns: list[str], limit: int = 50) -> str:
    """Render markdown table."""

    if not rows:
        return "- none"
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join("---" for _ in columns) + "|"]
    for row in rows[:limit]:
        values = []
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


def render_report(summary: dict[str, Any]) -> str:
    """Render Markdown report with required answers."""

    legacy = summary.get("legacy_entry_timing") or {}
    rows = summary.get("candidate_family_summary") or []
    rejected = summary.get("rejected_candidate_entry_families") or []
    best_recall = summary.get("best_recall_family") or {}
    best_early = summary.get("best_early_entry_family") or {}
    stable_exists = bool(summary.get("stable_like_candidate_exists"))
    return (
        "# Trend Entry Timing Research v1\n\n"
        "## Scope\n"
        "- This is research-only. Trend segment labels are ex-post labels and are not entry features.\n"
        "- Candidate events use only closed-bar data available at event time.\n"
        "- No Strategy V3, demo, or live permission is granted.\n\n"
        "## Required Answers\n"
        f"1. Did legacy strategies enter too late? {str(bool((legacy.get('late_entry_share') or 0) >= 0.5)).lower()} "
        f"(late_entry_share={format_number(legacy.get('late_entry_share'))}).\n"
        f"2. Worst legacy policy / symbol / timeframe: policy={legacy.get('worst_policy')}, symbol={legacy.get('worst_symbol')}.\n"
        f"3. Do candidate early-entry events enter trends earlier? best_early_entry_family={best_early.get('family', 'N/A')} "
        f"(early_entry_rate={format_number(best_early.get('early_entry_rate'))}).\n"
        f"4. Highest recall family: {best_recall.get('family', 'N/A')} "
        f"(trend_segment_recall={format_number(best_recall.get('trend_segment_recall'))}).\n"
        f"5. Highest early_entry_rate family: {best_early.get('family', 'N/A')}.\n"
        f"6. Any family passed train/validation/oos, cost, and funding gates? {str(stable_exists).lower()}.\n"
        "7. Reverse test weaker than forward? See rejected_candidate_entry_families.csv; required for gate.\n"
        "8. Random time control weaker than forward? See rejected_candidate_entry_families.csv; required for gate.\n"
        f"9. If no family passes, current candidate features are insufficient; entry timing may still require a new hypothesis. passed={str(stable_exists).lower()}.\n"
        f"10. Entry Timing Phase 2 allowed? {str(bool(summary.get('can_enter_entry_timing_phase2'))).lower()}.\n"
        "11. Formal strategy modification allowed? false.\n"
        "12. Demo/live allowed? false.\n\n"
        "## Candidate Families\n"
        f"{markdown_table(rows, ['family', 'selected_hold_label', 'event_count', 'trend_segment_recall', 'early_entry_rate', 'direction_match_rate', 'train_ext_no_cost_pnl', 'validation_ext_no_cost_pnl', 'oos_ext_no_cost_pnl'])}\n\n"
        "## Gate Rejections\n"
        f"{markdown_table(rejected, ['family', 'stable_like', 'selected_hold_label', 'rejected_reasons'], limit=100)}\n\n"
        "## Final Gates\n"
        f"- can_enter_entry_timing_phase2={str(bool(summary.get('can_enter_entry_timing_phase2'))).lower()}\n"
        "- strategy_development_allowed=false\n"
        "- demo_live_allowed=false\n"
    )


def write_outputs(
    output_dir: Path,
    summary: dict[str, Any],
    data_quality: dict[str, Any],
    legacy: pd.DataFrame,
    events: pd.DataFrame,
    family_summary: pd.DataFrame,
    trade_tests: pd.DataFrame,
    by_symbol: pd.DataFrame,
    by_timeframe: pd.DataFrame,
    by_split: pd.DataFrame,
    concentration: pd.DataFrame,
    reverse: pd.DataFrame,
    random_control: pd.DataFrame,
    rejected: pd.DataFrame,
) -> None:
    """Write all required outputs."""

    output_dir.mkdir(parents=True, exist_ok=True)
    legacy.to_csv(output_dir / "legacy_entry_timing_diagnostics.csv", index=False, encoding="utf-8")
    events.to_csv(output_dir / "candidate_entry_events.csv", index=False, encoding="utf-8")
    family_summary.to_csv(output_dir / "candidate_entry_family_summary.csv", index=False, encoding="utf-8")
    trade_tests.to_csv(output_dir / "candidate_entry_trade_tests.csv", index=False, encoding="utf-8")
    by_symbol.to_csv(output_dir / "candidate_entry_by_symbol.csv", index=False, encoding="utf-8")
    by_timeframe.to_csv(output_dir / "candidate_entry_by_timeframe.csv", index=False, encoding="utf-8")
    by_split.to_csv(output_dir / "candidate_entry_by_split.csv", index=False, encoding="utf-8")
    concentration.to_csv(output_dir / "candidate_entry_concentration.csv", index=False, encoding="utf-8")
    reverse.to_csv(output_dir / "candidate_entry_reverse_test.csv", index=False, encoding="utf-8")
    random_control.to_csv(output_dir / "candidate_entry_random_control.csv", index=False, encoding="utf-8")
    rejected.to_csv(output_dir / "rejected_candidate_entry_families.csv", index=False, encoding="utf-8")
    (output_dir / "data_quality.json").write_text(json.dumps(to_jsonable(data_quality), ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "trend_entry_timing_summary.json").write_text(json.dumps(to_jsonable(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "trend_entry_timing_report.md").write_text(render_report(summary), encoding="utf-8")


def load_v3_trades(trend_v3_dir: Path, timezone_name: str, warnings: list[str]) -> pd.DataFrame:
    """Load required V3 extended trades."""

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
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def build_bar_frames(
    symbols: list[str],
    timeframes: list[str],
    start: str,
    end: str,
    timezone_name: str,
    funding_histories: dict[str, pd.DataFrame],
    *,
    database_path: Path,
    bars_by_symbol: dict[str, pd.DataFrame] | None = None,
    logger: logging.Logger | None = None,
) -> dict[tuple[str, str], pd.DataFrame]:
    """Load 1m data and build timeframe frames with indicators."""

    frames: dict[tuple[str, str], pd.DataFrame] = {}
    start_ts = pd.Timestamp(start, tz=timezone_name) - pd.Timedelta(days=120)
    end_ts = pd.Timestamp(end, tz=timezone_name) + pd.Timedelta(days=15)
    anchor = pd.Timestamp(start, tz=timezone_name)
    for symbol in symbols:
        if logger is not None:
            log_event(logger, logging.INFO, "trend_entry_timing.load_symbol", "Loading symbol bars", symbol=symbol)
        if bars_by_symbol and symbol in bars_by_symbol:
            bars_1m = tce.normalize_1m_bars(bars_by_symbol[symbol], timezone_name, timezone_name)
        else:
            bars_1m = tce.load_1m_bars_from_sqlite(symbol, database_path, timezone_name, start_ts, end_ts)
        funding = funding_histories.get(tce.symbol_to_inst_id(symbol))
        for timeframe in timeframes:
            closed = tce.resample_ohlcv_closed(bars_1m, timeframe, anchor=anchor)
            closed = add_basic_indicators(closed)
            closed = attach_funding_to_frame(closed, funding)
            closed["symbol"] = symbol
            closed["timeframe"] = timeframe
            frames[(symbol, timeframe)] = closed
    add_relative_strength(frames, symbols, timeframes)
    return frames


def build_concentration(trade_tests: pd.DataFrame) -> pd.DataFrame:
    """Build concentration by selected family/hold/split."""

    if trade_tests.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for (family, hold_label, split), group in trade_tests.groupby(["family", "hold_label", "split"], dropna=False):
        top = top_5pct_trade_pnl_contribution(group)
        rows.append(
            {
                "family": family,
                "hold_label": hold_label,
                "split": split,
                "trade_count": int(len(group.index)),
                "largest_symbol_pnl_share": largest_symbol_pnl_share(group),
                "top_5pct_trade_pnl_contribution": top,
            }
        )
    return pd.DataFrame(rows).sort_values(["family", "hold_label", "split"], kind="stable").reset_index(drop=True)


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
    bars_by_symbol: dict[str, pd.DataFrame] | None = None,
) -> EntryTimingOutputs:
    """Run Trend Entry Timing v1 research."""

    warnings: list[str] = []
    trend_summary = tce.read_json_if_exists(trend_map_dir / "trend_opportunity_summary.json", warnings)
    trend_quality = tce.read_json_if_exists(trend_map_dir / "data_quality.json", warnings)
    segments = tce.normalize_segments(tce.read_csv_if_exists(trend_map_dir / "trend_segments.csv", warnings, required=True), timezone_name)
    capture_diag_path = DEFAULT_CAPTURE_EXIT_DIR / "trend_capture_diagnostics.csv"
    _capture_diag = tce.read_csv_if_exists(capture_diag_path, warnings, required=True)
    trades = load_v3_trades(trend_v3_dir, timezone_name, warnings)
    inst_ids = [tce.symbol_to_inst_id(symbol) for symbol in symbols]
    funding_histories, missing_funding = tce.load_funding_histories(funding_dir, inst_ids, warnings)
    funding_indexes = tce.build_funding_indexes(funding_histories)
    frames = build_bar_frames(
        symbols,
        timeframes,
        start,
        end,
        timezone_name,
        funding_histories,
        database_path=database_path,
        bars_by_symbol=bars_by_symbol,
        logger=logger,
    )
    market_features = build_market_features(frames, symbols, timeframes)
    thresholds = train_thresholds(frames, market_features, timezone_name)
    segments_by_key = build_segments_by_key(segments, timeframes)
    legacy = build_legacy_entry_diagnostics(trades, segments_by_key, frames)
    if logger is not None:
        log_event(logger, logging.INFO, "trend_entry_timing.legacy_done", "Built legacy entry diagnostics", row_count=len(legacy.index))
    events = generate_candidate_events(frames, market_features, thresholds, symbols, timeframes, timezone_name)
    if logger is not None:
        log_event(logger, logging.INFO, "trend_entry_timing.events_generated", "Generated candidate entry events", event_count=len(events.index))
    events = annotate_events(events, segments_by_key, frames) if not events.empty else pd.DataFrame()
    if logger is not None:
        log_event(logger, logging.INFO, "trend_entry_timing.events_annotated", "Annotated candidate entry events", event_count=len(events.index))
    trade_tests = build_trade_tests(events, frames, funding_indexes)
    if logger is not None:
        log_event(logger, logging.INFO, "trend_entry_timing.trade_tests_done", "Built fixed-hold trade tests", row_count=len(trade_tests.index))
    selected_holds = select_train_holds(trade_tests)
    reverse = build_reverse_tests(events, frames, funding_indexes, selected_holds)
    random_control = build_random_controls(events, frames, funding_indexes, selected_holds, timezone_name)
    family_summary = build_family_summary(events, trade_tests, segments[segments["timeframe"].isin(timeframes)], reverse, random_control, selected_holds)
    rejected, candidates = evaluate_stable_gates(family_summary)
    by_symbol = group_summary(trade_tests, ["family", "hold_label", "symbol"]) if not trade_tests.empty else pd.DataFrame()
    by_timeframe = group_summary(trade_tests, ["family", "hold_label", "timeframe"]) if not trade_tests.empty else pd.DataFrame()
    by_split = group_summary(trade_tests, ["family", "hold_label", "split"]) if not trade_tests.empty else pd.DataFrame()
    concentration = build_concentration(trade_tests)
    data_quality = {
        "trend_map_data_quality": trend_quality,
        "market_data_complete": bool(trend_quality.get("all_symbols_complete") or (trend_summary.get("data_quality") or {}).get("all_symbols_complete")),
        "funding_data_complete": bool(not missing_funding and (trend_summary.get("data_quality") or {}).get("funding_data_complete", True)),
        "funding_missing_inst_ids": missing_funding,
        "warnings": warnings,
        "data_check_strict": bool(data_check_strict),
    }
    if data_check_strict and not data_quality["market_data_complete"]:
        warnings.append("data_check_strict:market_data_not_complete")
    if data_check_strict and not data_quality["funding_data_complete"]:
        warnings.append("data_check_strict:funding_data_not_complete")
    legacy_payload = legacy_summary_payload(legacy)
    summary = build_summary_payload(
        output_dir=output_dir,
        warnings=warnings,
        trend_summary=trend_summary,
        data_quality=data_quality,
        legacy_summary=legacy_payload,
        family_summary=family_summary,
        rejected=rejected,
        candidates=candidates,
        thresholds=thresholds,
    )
    write_outputs(
        output_dir,
        summary,
        data_quality,
        legacy,
        events,
        family_summary,
        trade_tests,
        by_symbol,
        by_timeframe,
        by_split,
        concentration,
        reverse,
        random_control,
        rejected,
    )
    return EntryTimingOutputs(output_dir=output_dir, summary=summary, legacy_diagnostics=legacy, events=events, family_summary=family_summary)


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""

    args = parse_args(argv)
    ensure_headless_runtime()
    logger = setup_logging("research_trend_entry_timing", verbose=args.verbose)
    try:
        outputs = run_research(
            trend_map_dir=resolve_path(args.trend_map_dir),
            trend_v3_dir=resolve_path(args.trend_v3_dir),
            funding_dir=resolve_path(args.funding_dir),
            output_dir=resolve_path(args.output_dir),
            symbols=parse_csv_list(args.symbols),
            start=args.start,
            end=args.end,
            timezone_name=args.timezone,
            timeframes=parse_csv_list(args.timeframes),
            data_check_strict=bool(args.data_check_strict),
            logger=logger,
        )
        print_json_block(
            "Trend entry timing summary:",
            {
                "late_entry_share": outputs.summary.get("legacy_entry_timing", {}).get("late_entry_share"),
                "can_enter_entry_timing_phase2": outputs.summary.get("can_enter_entry_timing_phase2"),
                "strategy_development_allowed": outputs.summary.get("strategy_development_allowed"),
                "demo_live_allowed": outputs.summary.get("demo_live_allowed"),
                "recommended_next_step": outputs.summary.get("recommended_next_step"),
                "output_dir": outputs.summary.get("output_dir"),
            },
        )
        return 0
    except TrendEntryTimingResearchError as exc:
        log_event(logger, logging.ERROR, "trend_entry_timing.error", str(exc))
        return 2
    except Exception:
        logger.exception("Unexpected Trend Entry Timing failure", extra={"event": "trend_entry_timing.unexpected"})
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
