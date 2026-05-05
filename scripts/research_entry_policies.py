#!/usr/bin/env python3
"""Research offline virtual entry policies with no-cost bracket replay."""

from __future__ import annotations

import argparse
import json
import logging
from array import array
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from analyze_signal_outcomes import (
    SignalOutcomeError,
    dataframe_bars_to_ohlc,
    load_bars_from_db,
    normalize_bool,
    number_or_none,
    prepare_entry_signals,
    read_signal_trace,
)
from common_runtime import PROJECT_ROOT, ensure_headless_runtime, log_event, print_json_block, setup_logging, to_jsonable
from history_time_utils import DEFAULT_TIMEZONE


DEFAULT_HORIZONS = "15,30,60,120"
DEFAULT_STOP_ATR_GRID = "1.0,1.5,2.0,2.5,3.0,4.0"
DEFAULT_TP_ATR_GRID = "1.5,2.0,2.5,3.0,4.0,5.0"
DEFAULT_MAX_WAIT_BARS = 10
POLICY_NAMES = [
    "immediate_baseline",
    "skip_large_breakout_gt_1atr",
    "skip_large_breakout_gt_2atr",
    "small_to_mid_breakout_0_25_to_1atr",
    "delayed_confirm_1bar",
    "delayed_confirm_3bar",
    "pullback_to_breakout_level_5bar",
    "pullback_to_breakout_level_10bar",
    "momentum_followthrough_3bar",
    "avoid_stop_first_profile",
]
POLICY_NOTES = {
    "immediate_baseline": "Signal time immediate entry at trace price.",
    "skip_large_breakout_gt_1atr": "Immediate entry, but skip breakout_distance_atr > 1.",
    "skip_large_breakout_gt_2atr": "Immediate entry, but skip breakout_distance_atr > 2.",
    "small_to_mid_breakout_0_25_to_1atr": "Immediate entry only when 0.25 <= breakout_distance_atr <= 1.",
    "delayed_confirm_1bar": "Wait one 1m bar; enter at that bar close only if close still confirms the breakout side.",
    "delayed_confirm_3bar": "Wait three 1m bars; enter at the third bar close only if close still confirms the breakout side.",
    "pullback_to_breakout_level_5bar": "Wait up to five 1m bars for a pullback touch of the original breakout level.",
    "pullback_to_breakout_level_10bar": "Wait up to ten 1m bars for a pullback touch of the original breakout level.",
    "momentum_followthrough_3bar": "Wait three 1m bars; enter only after at least 0.25 ATR favorable follow-through.",
    "avoid_stop_first_profile": "Use the first three 1m bars as a gate; skip early 1 ATR adverse-first profiles.",
}
LEADERBOARD_COLUMNS = [
    "policy_name",
    "entry_count",
    "skipped_count",
    "avg_return",
    "median_return",
    "win_rate",
    "avg_r",
    "median_r",
    "expectancy_r",
    "max_drawdown_r_est",
    "best_horizon",
    "best_stop_atr",
    "best_tp_atr",
    "stop_first_rate",
    "tp_first_rate",
    "horizon_exit_rate",
    "notes",
]


class EntryPolicyResearchError(Exception):
    """Raised when entry policy research cannot continue."""


@dataclass(frozen=True, slots=True)
class VirtualEntry:
    """One policy decision for one original entry signal."""

    policy_name: str
    row_number: int
    signal_id: str
    signal_dt: pd.Timestamp
    direction: str
    side: str
    hour: int | None
    weekday: int | None
    is_weekend: bool | None
    breakout_distance_atr: float | None
    atr_1m: float | None
    original_price: float
    entry_dt: pd.Timestamp | None
    entry_price: float | None
    bracket_start_index: int | None
    skipped: bool
    skip_reason: str


@dataclass(frozen=True, slots=True)
class BracketResult:
    """One virtual bracket replay result."""

    policy_name: str
    row_number: int
    signal_id: str
    signal_dt: pd.Timestamp
    entry_dt: pd.Timestamp
    direction: str
    side: str
    hour: int | None
    horizon_m: int
    stop_atr: float
    tp_atr: float
    entry_price: float
    stop_price: float
    take_profit_price: float
    exit_dt: pd.Timestamp
    exit_price: float
    exit_reason: str
    return_pct: float
    r_multiple: float


@dataclass(frozen=True, slots=True)
class BarArrays:
    """Fast array view over normalized 1m OHLC bars."""

    datetimes: list[pd.Timestamp]
    timestamp_ns: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray


@dataclass(slots=True)
class ComboAccumulator:
    """Compact metric accumulator for one policy/grid combination."""

    returns: array = field(default_factory=lambda: array("d"))
    r_values: array = field(default_factory=lambda: array("d"))
    stop_first_count: int = 0
    tp_first_count: int = 0
    horizon_exit_count: int = 0

    def add(self, return_pct: float, r_multiple: float, exit_reason: str) -> None:
        """Add one replay outcome."""

        self.returns.append(float(return_pct))
        self.r_values.append(float(r_multiple))
        if exit_reason == "stop_first":
            self.stop_first_count += 1
        elif exit_reason == "tp_first":
            self.tp_first_count += 1
        elif exit_reason == "horizon_exit":
            self.horizon_exit_count += 1


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description="Research offline virtual entry policies from signal_trace.csv.")
    parser.add_argument("--report-dir", required=True, help="Backtest report directory containing signal_trace.csv.")
    parser.add_argument("--signal-trace", help="Signal trace CSV. Default: <report-dir>/signal_trace.csv.")
    parser.add_argument(
        "--timezone",
        default=DEFAULT_TIMEZONE,
        help=f"Timezone used to align signal and bar times. Default: {DEFAULT_TIMEZONE}.",
    )
    parser.add_argument("--output-dir", help="Output directory. Default: <report-dir>/entry_policy_research.")
    parser.add_argument(
        "--horizons",
        default=DEFAULT_HORIZONS,
        help=f"Comma-separated bracket horizons in minutes. Default: {DEFAULT_HORIZONS}.",
    )
    parser.add_argument(
        "--max-wait-bars",
        type=int,
        default=DEFAULT_MAX_WAIT_BARS,
        help=f"Maximum 1m bars an entry policy may wait. Default: {DEFAULT_MAX_WAIT_BARS}.",
    )
    parser.add_argument(
        "--stop-atr-grid",
        default=DEFAULT_STOP_ATR_GRID,
        help=f"Comma-separated stop ATR multiples. Default: {DEFAULT_STOP_ATR_GRID}.",
    )
    parser.add_argument(
        "--tp-atr-grid",
        default=DEFAULT_TP_ATR_GRID,
        help=f"Comma-separated take-profit ATR multiples. Default: {DEFAULT_TP_ATR_GRID}.",
    )
    parser.add_argument(
        "--bars-from-db",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Load 1m bars from vn.py sqlite. Default: enabled.",
    )
    parser.add_argument("--json", action="store_true", help="Print entry_policy_summary.json payload to stdout.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose structured logs.")
    return parser.parse_args(argv)


def resolve_path(path_arg: str | None, default_path: Path | None = None) -> Path:
    """Resolve a path relative to the project root."""

    if path_arg:
        path = Path(path_arg)
    elif default_path is not None:
        path = default_path
    else:
        raise EntryPolicyResearchError("缺少路径参数且没有默认值")
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def parse_positive_int_list(raw_value: str, option_name: str) -> list[int]:
    """Parse comma-separated positive integers."""

    values: set[int] = set()
    for token in str(raw_value or "").split(","):
        text = token.strip()
        if not text:
            continue
        try:
            value = int(text)
        except ValueError as exc:
            raise EntryPolicyResearchError(f"{option_name} 包含非法整数: {text!r}") from exc
        if value <= 0:
            raise EntryPolicyResearchError(f"{option_name} 必须为正整数: {value}")
        values.add(value)
    if not values:
        raise EntryPolicyResearchError(f"{option_name} 不能为空")
    return sorted(values)


def parse_positive_float_list(raw_value: str, option_name: str) -> list[float]:
    """Parse comma-separated positive finite floats."""

    values: set[float] = set()
    for token in str(raw_value or "").split(","):
        text = token.strip()
        if not text:
            continue
        try:
            value = float(text)
        except ValueError as exc:
            raise EntryPolicyResearchError(f"{option_name} 包含非法数字: {text!r}") from exc
        if value <= 0 or not np.isfinite(value):
            raise EntryPolicyResearchError(f"{option_name} 必须为正有限数字: {value}")
        values.add(value)
    if not values:
        raise EntryPolicyResearchError(f"{option_name} 不能为空")
    return sorted(values)


def dataframe_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Convert a DataFrame to JSON-safe records."""

    if df.empty:
        return []
    safe_df = df.replace({np.nan: None})
    return json.loads(safe_df.to_json(orient="records", force_ascii=False))


def write_json(path: Path, payload: Any) -> None:
    """Write JSON with UTF-8 encoding."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(payload), ensure_ascii=False, indent=2), encoding="utf-8")


def write_dataframe(path: Path, df: pd.DataFrame) -> None:
    """Write a DataFrame as UTF-8 CSV."""

    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")


def finite_or_none(value: Any) -> float | None:
    """Return a finite float or None."""

    number = number_or_none(value)
    if number is None or not np.isfinite(number):
        return None
    return number


def bool_or_none(value: Any) -> bool | None:
    """Normalize bool-like values."""

    return normalize_bool(value)


def directional_return(direction: str, entry_price: float, exit_price: float) -> float:
    """Compute direction-adjusted fractional return."""

    if direction == "long":
        return (exit_price - entry_price) / entry_price
    return (entry_price - exit_price) / entry_price


def directional_price_move(direction: str, entry_price: float, exit_price: float) -> float:
    """Compute direction-adjusted raw price move."""

    if direction == "long":
        return exit_price - entry_price
    return entry_price - exit_price


def timestamp_to_utc_ns(value: Any) -> int:
    """Convert a timestamp to UTC nanoseconds for fast searchsorted."""

    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    return int(timestamp.value)


def build_bar_arrays(bars: pd.DataFrame) -> BarArrays:
    """Build a fast array view over normalized bars."""

    datetimes = [pd.Timestamp(value) for value in bars["datetime"]]
    timestamp_ns = np.array([timestamp_to_utc_ns(value) for value in datetimes], dtype=np.int64)
    return BarArrays(
        datetimes=datetimes,
        timestamp_ns=timestamp_ns,
        high=pd.to_numeric(bars["high"], errors="coerce").to_numpy(dtype=float),
        low=pd.to_numeric(bars["low"], errors="coerce").to_numpy(dtype=float),
        close=pd.to_numeric(bars["close"], errors="coerce").to_numpy(dtype=float),
    )


def fallback_int(value: Any) -> int | None:
    """Convert a number-like value to int."""

    number = finite_or_none(value)
    if number is None:
        return None
    return int(number)


def breakout_level(row: dict[str, Any], direction: str, entry_price: float) -> float:
    """Return the best available original breakout level."""

    if direction == "long":
        level = finite_or_none(row.get("donchian_high"))
        if level is not None and level > 0:
            return level
        distance = finite_or_none(row.get("breakout_distance"))
        if distance is not None:
            return entry_price - abs(distance)
        return entry_price

    level = finite_or_none(row.get("donchian_low"))
    if level is not None and level > 0:
        return level
    distance = finite_or_none(row.get("breakout_distance"))
    if distance is not None:
        return entry_price + abs(distance)
    return entry_price


def bar_close_confirms(direction: str, close_price: float, level: float, original_price: float) -> bool:
    """Check whether a delayed close still confirms the original breakout side."""

    threshold = max(level, original_price) if direction == "long" else min(level, original_price)
    if direction == "long":
        return close_price >= threshold
    return close_price <= threshold


def base_virtual_entry(
    policy_name: str,
    row_number: int,
    row: dict[str, Any],
    signal_dt: pd.Timestamp,
    start_index: int,
    skipped: bool,
    skip_reason: str,
    entry_dt: pd.Timestamp | None = None,
    entry_price: float | None = None,
    bracket_start_index: int | None = None,
) -> VirtualEntry:
    """Build a virtual entry decision."""

    direction = str(row.get("direction"))
    original_price = float(row.get("price"))
    hour = fallback_int(row.get("hour"))
    weekday = fallback_int(row.get("weekday"))
    is_weekend = bool_or_none(row.get("is_weekend"))
    return VirtualEntry(
        policy_name=policy_name,
        row_number=row_number,
        signal_id=str(row.get("signal_id") or row_number),
        signal_dt=signal_dt,
        direction=direction,
        side=direction,
        hour=hour,
        weekday=weekday,
        is_weekend=is_weekend,
        breakout_distance_atr=finite_or_none(row.get("breakout_distance_atr")),
        atr_1m=finite_or_none(row.get("atr_1m")),
        original_price=original_price,
        entry_dt=entry_dt,
        entry_price=entry_price,
        bracket_start_index=bracket_start_index,
        skipped=skipped,
        skip_reason=skip_reason,
    )


def immediate_entry(
    policy_name: str,
    row_number: int,
    row: dict[str, Any],
    signal_dt: pd.Timestamp,
    start_index: int,
) -> VirtualEntry:
    """Enter immediately at the trace price."""

    return base_virtual_entry(
        policy_name=policy_name,
        row_number=row_number,
        row=row,
        signal_dt=signal_dt,
        start_index=start_index,
        skipped=False,
        skip_reason="",
        entry_dt=signal_dt,
        entry_price=float(row.get("price")),
        bracket_start_index=start_index,
    )


def skipped_entry(
    policy_name: str,
    row_number: int,
    row: dict[str, Any],
    signal_dt: pd.Timestamp,
    start_index: int,
    reason: str,
) -> VirtualEntry:
    """Build a skipped policy decision."""

    return base_virtual_entry(
        policy_name=policy_name,
        row_number=row_number,
        row=row,
        signal_dt=signal_dt,
        start_index=start_index,
        skipped=True,
        skip_reason=reason,
    )


def delayed_confirm_entry(
    policy_name: str,
    confirm_bars: int,
    row_number: int,
    row: dict[str, Any],
    bars: pd.DataFrame,
    signal_dt: pd.Timestamp,
    start_index: int,
) -> VirtualEntry:
    """Wait for N bars and enter at the confirmation bar close if still valid."""

    confirm_index = start_index + confirm_bars - 1
    if confirm_bars <= 0 or confirm_index >= len(bars.index):
        return skipped_entry(policy_name, row_number, row, signal_dt, start_index, "insufficient_confirm_bars")

    direction = str(row.get("direction"))
    original_price = float(row.get("price"))
    close_price = finite_or_none(bars.iloc[confirm_index].get("close"))
    if close_price is None or close_price <= 0:
        return skipped_entry(policy_name, row_number, row, signal_dt, start_index, "invalid_confirm_close")

    level = breakout_level(row, direction, original_price)
    if not bar_close_confirms(direction, close_price, level, original_price):
        return skipped_entry(policy_name, row_number, row, signal_dt, start_index, "confirm_close_failed")

    return base_virtual_entry(
        policy_name=policy_name,
        row_number=row_number,
        row=row,
        signal_dt=signal_dt,
        start_index=start_index,
        skipped=False,
        skip_reason="",
        entry_dt=pd.Timestamp(bars.iloc[confirm_index]["datetime"]),
        entry_price=close_price,
        bracket_start_index=confirm_index + 1,
    )


def pullback_entry(
    policy_name: str,
    wait_bars: int,
    row_number: int,
    row: dict[str, Any],
    bars: pd.DataFrame,
    signal_dt: pd.Timestamp,
    start_index: int,
    max_wait_bars: int,
) -> VirtualEntry:
    """Wait for a pullback touch to the original breakout level."""

    direction = str(row.get("direction"))
    original_price = float(row.get("price"))
    level = breakout_level(row, direction, original_price)
    if level <= 0:
        return skipped_entry(policy_name, row_number, row, signal_dt, start_index, "invalid_breakout_level")

    max_index = min(len(bars.index), start_index + min(wait_bars, max_wait_bars))
    for index in range(start_index, max_index):
        bar = bars.iloc[index]
        high = finite_or_none(bar.get("high"))
        low = finite_or_none(bar.get("low"))
        if high is None or low is None:
            continue
        touched = low <= level if direction == "long" else high >= level
        if touched:
            return base_virtual_entry(
                policy_name=policy_name,
                row_number=row_number,
                row=row,
                signal_dt=signal_dt,
                start_index=start_index,
                skipped=False,
                skip_reason="",
                entry_dt=pd.Timestamp(bar["datetime"]),
                entry_price=level,
                bracket_start_index=index + 1,
            )

    return skipped_entry(policy_name, row_number, row, signal_dt, start_index, "pullback_not_touched")


def momentum_followthrough_entry(
    policy_name: str,
    follow_bars: int,
    row_number: int,
    row: dict[str, Any],
    bars: pd.DataFrame,
    signal_dt: pd.Timestamp,
    start_index: int,
) -> VirtualEntry:
    """Enter after a small favorable follow-through over the next N bars."""

    follow_index = start_index + follow_bars - 1
    if follow_bars <= 0 or follow_index >= len(bars.index):
        return skipped_entry(policy_name, row_number, row, signal_dt, start_index, "insufficient_followthrough_bars")

    direction = str(row.get("direction"))
    original_price = float(row.get("price"))
    atr = finite_or_none(row.get("atr_1m")) or 0.0
    threshold_move = max(0.0, 0.25 * atr)
    close_price = finite_or_none(bars.iloc[follow_index].get("close"))
    if close_price is None or close_price <= 0:
        return skipped_entry(policy_name, row_number, row, signal_dt, start_index, "invalid_followthrough_close")

    move = directional_price_move(direction, original_price, close_price)
    if move < threshold_move:
        return skipped_entry(policy_name, row_number, row, signal_dt, start_index, "followthrough_too_weak")

    return base_virtual_entry(
        policy_name=policy_name,
        row_number=row_number,
        row=row,
        signal_dt=signal_dt,
        start_index=start_index,
        skipped=False,
        skip_reason="",
        entry_dt=pd.Timestamp(bars.iloc[follow_index]["datetime"]),
        entry_price=close_price,
        bracket_start_index=follow_index + 1,
    )


def avoid_stop_first_profile_entry(
    policy_name: str,
    row_number: int,
    row: dict[str, Any],
    bars: pd.DataFrame,
    signal_dt: pd.Timestamp,
    start_index: int,
    max_wait_bars: int,
) -> VirtualEntry:
    """Skip signals whose first bars show an adverse-first 1 ATR profile."""

    direction = str(row.get("direction"))
    original_price = float(row.get("price"))
    atr = finite_or_none(row.get("atr_1m"))
    if atr is None or atr <= 0:
        return skipped_entry(policy_name, row_number, row, signal_dt, start_index, "invalid_atr")

    wait_bars = min(3, max_wait_bars)
    if wait_bars <= 0 or start_index + wait_bars > len(bars.index):
        return skipped_entry(policy_name, row_number, row, signal_dt, start_index, "insufficient_profile_bars")

    last_index = start_index + wait_bars - 1
    for index in range(start_index, start_index + wait_bars):
        bar = bars.iloc[index]
        high = finite_or_none(bar.get("high"))
        low = finite_or_none(bar.get("low"))
        close = finite_or_none(bar.get("close"))
        if high is None or low is None or close is None:
            continue

        if direction == "long":
            adverse_hit = low <= original_price - atr
            favorable_hit = high >= original_price + atr
        else:
            adverse_hit = high >= original_price + atr
            favorable_hit = low <= original_price - atr

        if adverse_hit and favorable_hit:
            return skipped_entry(policy_name, row_number, row, signal_dt, start_index, "early_same_bar_stop_profile")
        if adverse_hit:
            return skipped_entry(policy_name, row_number, row, signal_dt, start_index, "early_adverse_profile")
        if favorable_hit:
            return base_virtual_entry(
                policy_name=policy_name,
                row_number=row_number,
                row=row,
                signal_dt=signal_dt,
                start_index=start_index,
                skipped=False,
                skip_reason="",
                entry_dt=pd.Timestamp(bar["datetime"]),
                entry_price=close,
                bracket_start_index=index + 1,
            )

    close_price = finite_or_none(bars.iloc[last_index].get("close"))
    if close_price is None or close_price <= 0:
        return skipped_entry(policy_name, row_number, row, signal_dt, start_index, "invalid_profile_close")
    if directional_price_move(direction, original_price, close_price) < 0:
        return skipped_entry(policy_name, row_number, row, signal_dt, start_index, "profile_close_adverse")

    return base_virtual_entry(
        policy_name=policy_name,
        row_number=row_number,
        row=row,
        signal_dt=signal_dt,
        start_index=start_index,
        skipped=False,
        skip_reason="",
        entry_dt=pd.Timestamp(bars.iloc[last_index]["datetime"]),
        entry_price=close_price,
        bracket_start_index=last_index + 1,
    )


def build_virtual_entry_for_policy(
    policy_name: str,
    row_number: int,
    row: dict[str, Any],
    bars: pd.DataFrame,
    bar_times: pd.Series,
    max_wait_bars: int,
) -> VirtualEntry:
    """Apply one entry policy to one signal row."""

    signal_dt = pd.Timestamp(row["_signal_dt"])
    start_index = int(bar_times.searchsorted(signal_dt, side="right"))
    breakout_atr = finite_or_none(row.get("breakout_distance_atr"))

    if start_index >= len(bars.index):
        return skipped_entry(policy_name, row_number, row, signal_dt, start_index, "no_future_bars")

    if policy_name == "immediate_baseline":
        return immediate_entry(policy_name, row_number, row, signal_dt, start_index)
    if policy_name == "skip_large_breakout_gt_1atr":
        if breakout_atr is not None and breakout_atr > 1.0:
            return skipped_entry(policy_name, row_number, row, signal_dt, start_index, "breakout_distance_atr_gt_1")
        return immediate_entry(policy_name, row_number, row, signal_dt, start_index)
    if policy_name == "skip_large_breakout_gt_2atr":
        if breakout_atr is not None and breakout_atr > 2.0:
            return skipped_entry(policy_name, row_number, row, signal_dt, start_index, "breakout_distance_atr_gt_2")
        return immediate_entry(policy_name, row_number, row, signal_dt, start_index)
    if policy_name == "small_to_mid_breakout_0_25_to_1atr":
        if breakout_atr is None or breakout_atr < 0.25 or breakout_atr > 1.0:
            return skipped_entry(policy_name, row_number, row, signal_dt, start_index, "breakout_distance_atr_outside_0_25_to_1")
        return immediate_entry(policy_name, row_number, row, signal_dt, start_index)
    if policy_name == "delayed_confirm_1bar":
        return delayed_confirm_entry(policy_name, 1, row_number, row, bars, signal_dt, start_index)
    if policy_name == "delayed_confirm_3bar":
        return delayed_confirm_entry(policy_name, 3, row_number, row, bars, signal_dt, start_index)
    if policy_name == "pullback_to_breakout_level_5bar":
        return pullback_entry(policy_name, 5, row_number, row, bars, signal_dt, start_index, max_wait_bars)
    if policy_name == "pullback_to_breakout_level_10bar":
        return pullback_entry(policy_name, 10, row_number, row, bars, signal_dt, start_index, max_wait_bars)
    if policy_name == "momentum_followthrough_3bar":
        return momentum_followthrough_entry(policy_name, 3, row_number, row, bars, signal_dt, start_index)
    if policy_name == "avoid_stop_first_profile":
        return avoid_stop_first_profile_entry(policy_name, row_number, row, bars, signal_dt, start_index, max_wait_bars)

    raise EntryPolicyResearchError(f"未知 entry policy: {policy_name}")


def build_virtual_entries(
    entry_df: pd.DataFrame,
    bars: pd.DataFrame,
    max_wait_bars: int,
) -> list[VirtualEntry]:
    """Build all virtual entry decisions for every policy."""

    bar_times = pd.Series(bars["datetime"])
    decisions: list[VirtualEntry] = []
    for row_number, row in enumerate(entry_df.to_dict(orient="records")):
        for policy_name in POLICY_NAMES:
            decisions.append(build_virtual_entry_for_policy(policy_name, row_number, row, bars, bar_times, max_wait_bars))
    return decisions


def simulate_bracket(
    bars: pd.DataFrame,
    bar_times: pd.Series,
    virtual_entry: VirtualEntry,
    horizon_m: int,
    stop_atr: float,
    tp_atr: float,
    warnings: list[str],
) -> BracketResult | None:
    """Replay a no-cost bracket; same-bar stop/tp ties are stop-first."""

    if virtual_entry.skipped:
        return None
    if virtual_entry.entry_dt is None or virtual_entry.entry_price is None or virtual_entry.bracket_start_index is None:
        return None
    atr = virtual_entry.atr_1m
    if atr is None or atr <= 0:
        return None
    entry_price = float(virtual_entry.entry_price)
    if entry_price <= 0:
        return None

    direction = virtual_entry.direction
    if direction == "long":
        stop_price = entry_price - stop_atr * atr
        take_profit_price = entry_price + tp_atr * atr
    else:
        stop_price = entry_price + stop_atr * atr
        take_profit_price = entry_price - tp_atr * atr

    target_dt = pd.Timestamp(virtual_entry.entry_dt) + pd.Timedelta(minutes=horizon_m)
    end_index = int(bar_times.searchsorted(target_dt, side="right"))
    start_index = int(virtual_entry.bracket_start_index)
    if start_index >= len(bars.index):
        return None

    window = bars.iloc[start_index:end_index].copy()
    if window.empty:
        return None

    for row in window.itertuples(index=False):
        high = finite_or_none(getattr(row, "high", None))
        low = finite_or_none(getattr(row, "low", None))
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
            exit_price = stop_price
            exit_reason = "stop_first"
        elif stop_hit:
            exit_price = stop_price
            exit_reason = "stop_first"
        elif tp_hit:
            exit_price = take_profit_price
            exit_reason = "tp_first"
        else:
            continue

        return build_bracket_result(
            virtual_entry=virtual_entry,
            horizon_m=horizon_m,
            stop_atr=stop_atr,
            tp_atr=tp_atr,
            entry_price=entry_price,
            stop_price=stop_price,
            take_profit_price=take_profit_price,
            exit_dt=pd.Timestamp(getattr(row, "datetime")),
            exit_price=exit_price,
            exit_reason=exit_reason,
            atr=atr,
        )

    last_row = window.iloc[-1]
    close_price = finite_or_none(last_row.get("close"))
    if close_price is None or close_price <= 0:
        return None
    return build_bracket_result(
        virtual_entry=virtual_entry,
        horizon_m=horizon_m,
        stop_atr=stop_atr,
        tp_atr=tp_atr,
        entry_price=entry_price,
        stop_price=stop_price,
        take_profit_price=take_profit_price,
        exit_dt=pd.Timestamp(last_row["datetime"]),
        exit_price=close_price,
        exit_reason="horizon_exit",
        atr=atr,
    )


def simulate_bracket_fast(
    arrays: BarArrays,
    virtual_entry: VirtualEntry,
    horizon_m: int,
    stop_atr: float,
    tp_atr: float,
    warnings: list[str],
) -> BracketResult | None:
    """Fast no-cost bracket replay using NumPy arrays."""

    if virtual_entry.skipped:
        return None
    if virtual_entry.entry_dt is None or virtual_entry.entry_price is None or virtual_entry.bracket_start_index is None:
        return None
    atr = virtual_entry.atr_1m
    if atr is None or atr <= 0:
        return None
    entry_price = float(virtual_entry.entry_price)
    if entry_price <= 0:
        return None

    direction = virtual_entry.direction
    if direction == "long":
        stop_price = entry_price - stop_atr * atr
        take_profit_price = entry_price + tp_atr * atr
    else:
        stop_price = entry_price + stop_atr * atr
        take_profit_price = entry_price - tp_atr * atr

    start_index = int(virtual_entry.bracket_start_index)
    if start_index >= len(arrays.timestamp_ns):
        return None

    target_dt = pd.Timestamp(virtual_entry.entry_dt) + pd.Timedelta(minutes=horizon_m)
    target_ns = timestamp_to_utc_ns(target_dt)
    end_index = int(np.searchsorted(arrays.timestamp_ns, target_ns, side="right"))
    if end_index <= start_index:
        return None

    high = arrays.high[start_index:end_index]
    low = arrays.low[start_index:end_index]
    if direction == "long":
        stop_hits = low <= stop_price
        tp_hits = high >= take_profit_price
    else:
        stop_hits = high >= stop_price
        tp_hits = low <= take_profit_price

    hit_mask = stop_hits | tp_hits
    if np.any(hit_mask):
        offset = int(np.argmax(hit_mask))
        absolute_index = start_index + offset
        if bool(stop_hits[offset] and tp_hits[offset]):
            warning = "同一根 1m bar 同时触发 stop/take-profit，保守按 stop first 处理"
            if warning not in warnings:
                warnings.append(warning)
            exit_price = stop_price
            exit_reason = "stop_first"
        elif bool(stop_hits[offset]):
            exit_price = stop_price
            exit_reason = "stop_first"
        else:
            exit_price = take_profit_price
            exit_reason = "tp_first"
        return build_bracket_result(
            virtual_entry=virtual_entry,
            horizon_m=horizon_m,
            stop_atr=stop_atr,
            tp_atr=tp_atr,
            entry_price=entry_price,
            stop_price=stop_price,
            take_profit_price=take_profit_price,
            exit_dt=arrays.datetimes[absolute_index],
            exit_price=exit_price,
            exit_reason=exit_reason,
            atr=atr,
        )

    last_index = end_index - 1
    close_price = arrays.close[last_index]
    if not np.isfinite(close_price) or close_price <= 0:
        return None
    return build_bracket_result(
        virtual_entry=virtual_entry,
        horizon_m=horizon_m,
        stop_atr=stop_atr,
        tp_atr=tp_atr,
        entry_price=entry_price,
        stop_price=stop_price,
        take_profit_price=take_profit_price,
        exit_dt=arrays.datetimes[last_index],
        exit_price=float(close_price),
        exit_reason="horizon_exit",
        atr=atr,
    )


def build_bracket_result(
    virtual_entry: VirtualEntry,
    horizon_m: int,
    stop_atr: float,
    tp_atr: float,
    entry_price: float,
    stop_price: float,
    take_profit_price: float,
    exit_dt: pd.Timestamp,
    exit_price: float,
    exit_reason: str,
    atr: float,
) -> BracketResult:
    """Build a bracket result row."""

    return_pct = directional_return(virtual_entry.direction, entry_price, exit_price)
    risk_per_unit = stop_atr * atr
    r_multiple = directional_price_move(virtual_entry.direction, entry_price, exit_price) / risk_per_unit
    return BracketResult(
        policy_name=virtual_entry.policy_name,
        row_number=virtual_entry.row_number,
        signal_id=virtual_entry.signal_id,
        signal_dt=virtual_entry.signal_dt,
        entry_dt=pd.Timestamp(virtual_entry.entry_dt),
        direction=virtual_entry.direction,
        side=virtual_entry.side,
        hour=virtual_entry.hour,
        horizon_m=horizon_m,
        stop_atr=stop_atr,
        tp_atr=tp_atr,
        entry_price=entry_price,
        stop_price=stop_price,
        take_profit_price=take_profit_price,
        exit_dt=exit_dt,
        exit_price=exit_price,
        exit_reason=exit_reason,
        return_pct=float(return_pct),
        r_multiple=float(r_multiple),
    )


def bracket_result_to_record(result: BracketResult) -> dict[str, Any]:
    """Convert a bracket result to a flat record."""

    return {
        "policy_name": result.policy_name,
        "row_number": result.row_number,
        "signal_id": result.signal_id,
        "signal_dt": result.signal_dt.isoformat(),
        "entry_dt": result.entry_dt.isoformat(),
        "direction": result.direction,
        "side": result.side,
        "hour": result.hour,
        "horizon_m": result.horizon_m,
        "stop_atr": result.stop_atr,
        "tp_atr": result.tp_atr,
        "entry_price": result.entry_price,
        "stop_price": result.stop_price,
        "take_profit_price": result.take_profit_price,
        "exit_dt": result.exit_dt.isoformat(),
        "exit_price": result.exit_price,
        "exit_reason": result.exit_reason,
        "return_pct": result.return_pct,
        "r_multiple": result.r_multiple,
    }


def evaluate_bracket_grid(
    decisions: list[VirtualEntry],
    bars: pd.DataFrame,
    horizons: list[int],
    stop_atr_grid: list[float],
    tp_atr_grid: list[float],
    warnings: list[str],
) -> pd.DataFrame:
    """Evaluate every policy/horizon/stop/tp combination."""

    if bars.empty:
        return pd.DataFrame()

    arrays = build_bar_arrays(bars)
    result_records: list[dict[str, Any]] = []
    active_decisions = [decision for decision in decisions if not decision.skipped]
    for horizon in horizons:
        for stop_atr in stop_atr_grid:
            for tp_atr in tp_atr_grid:
                for decision in active_decisions:
                    result = simulate_bracket_fast(arrays, decision, horizon, stop_atr, tp_atr, warnings)
                    if result is not None:
                        result_records.append(bracket_result_to_record(result))

    if not result_records:
        return pd.DataFrame(
            columns=[
                "policy_name",
                "row_number",
                "signal_id",
                "signal_dt",
                "entry_dt",
                "direction",
                "side",
                "hour",
                "horizon_m",
                "stop_atr",
                "tp_atr",
                "entry_price",
                "stop_price",
                "take_profit_price",
                "exit_dt",
                "exit_price",
                "exit_reason",
                "return_pct",
                "r_multiple",
            ]
        )
    return pd.DataFrame(result_records)


def max_drawdown_r_estimate(result_df: pd.DataFrame) -> float | None:
    """Estimate max drawdown from cumulative R in signal-time order."""

    if result_df.empty or "r_multiple" not in result_df.columns:
        return None
    working = result_df.copy()
    working["r_multiple"] = pd.to_numeric(working["r_multiple"], errors="coerce")
    working = working.dropna(subset=["r_multiple"]).sort_values(["signal_dt", "row_number"], kind="stable")
    if working.empty:
        return None
    cumulative = working["r_multiple"].cumsum()
    peaks = cumulative.cummax()
    drawdowns = peaks - cumulative
    return float(drawdowns.max())


def max_drawdown_r_estimate_from_results(results: list[BracketResult]) -> float | None:
    """Estimate max drawdown from ordered bracket results."""

    if not results:
        return None
    ordered = sorted(results, key=lambda item: (item.signal_dt, item.row_number))
    r_values = np.array([item.r_multiple for item in ordered], dtype=float)
    if r_values.size == 0:
        return None
    cumulative = np.cumsum(r_values)
    peaks = np.maximum.accumulate(cumulative)
    drawdowns = peaks - cumulative
    return float(np.max(drawdowns))


def summarize_bracket_results(results: list[BracketResult], total_count: int) -> dict[str, Any]:
    """Summarize a list of bracket results without materializing every grid row."""

    entry_count = int(len(results))
    skipped_count = max(int(total_count) - entry_count, 0)
    summary: dict[str, Any] = {
        "entry_count": entry_count,
        "skipped_count": skipped_count,
        "avg_return": None,
        "median_return": None,
        "win_rate": None,
        "avg_r": None,
        "median_r": None,
        "expectancy_r": None,
        "max_drawdown_r_est": None,
        "stop_first_rate": None,
        "tp_first_rate": None,
        "horizon_exit_rate": None,
    }
    if not results:
        return summary

    returns = np.array([item.return_pct for item in results], dtype=float)
    r_values = np.array([item.r_multiple for item in results], dtype=float)
    exit_reasons = np.array([item.exit_reason for item in results], dtype=object)
    summary["avg_return"] = float(np.mean(returns))
    summary["median_return"] = float(np.median(returns))
    summary["win_rate"] = float(np.mean(returns > 0))
    summary["avg_r"] = float(np.mean(r_values))
    summary["median_r"] = float(np.median(r_values))
    summary["expectancy_r"] = summary["avg_r"]
    summary["max_drawdown_r_est"] = max_drawdown_r_estimate_from_results(results)
    summary["stop_first_rate"] = float(np.mean(exit_reasons == "stop_first"))
    summary["tp_first_rate"] = float(np.mean(exit_reasons == "tp_first"))
    summary["horizon_exit_rate"] = float(np.mean(exit_reasons == "horizon_exit"))
    return summary


def summarize_accumulator(accumulator: ComboAccumulator, total_count: int) -> dict[str, Any]:
    """Summarize one compact accumulator."""

    entry_count = len(accumulator.r_values)
    skipped_count = max(int(total_count) - entry_count, 0)
    summary: dict[str, Any] = {
        "entry_count": int(entry_count),
        "skipped_count": skipped_count,
        "avg_return": None,
        "median_return": None,
        "win_rate": None,
        "avg_r": None,
        "median_r": None,
        "expectancy_r": None,
        "max_drawdown_r_est": None,
        "stop_first_rate": None,
        "tp_first_rate": None,
        "horizon_exit_rate": None,
    }
    if entry_count <= 0:
        return summary

    returns = np.frombuffer(accumulator.returns, dtype=np.float64)
    r_values = np.frombuffer(accumulator.r_values, dtype=np.float64)
    cumulative = np.cumsum(r_values)
    peaks = np.maximum.accumulate(cumulative)
    drawdowns = peaks - cumulative
    summary["avg_return"] = float(np.mean(returns))
    summary["median_return"] = float(np.median(returns))
    summary["win_rate"] = float(np.mean(returns > 0))
    summary["avg_r"] = float(np.mean(r_values))
    summary["median_r"] = float(np.median(r_values))
    summary["expectancy_r"] = summary["avg_r"]
    summary["max_drawdown_r_est"] = float(np.max(drawdowns))
    summary["stop_first_rate"] = float(accumulator.stop_first_count / entry_count)
    summary["tp_first_rate"] = float(accumulator.tp_first_count / entry_count)
    summary["horizon_exit_rate"] = float(accumulator.horizon_exit_count / entry_count)
    return summary


def summarize_result_slice(result_df: pd.DataFrame, total_count: int) -> dict[str, Any]:
    """Summarize one result slice for the leaderboard/grid/group files."""

    entry_count = int(len(result_df.index))
    skipped_count = max(int(total_count) - entry_count, 0)
    summary: dict[str, Any] = {
        "entry_count": entry_count,
        "skipped_count": skipped_count,
        "avg_return": None,
        "median_return": None,
        "win_rate": None,
        "avg_r": None,
        "median_r": None,
        "expectancy_r": None,
        "max_drawdown_r_est": None,
        "stop_first_rate": None,
        "tp_first_rate": None,
        "horizon_exit_rate": None,
    }
    if result_df.empty:
        return summary

    returns = pd.to_numeric(result_df["return_pct"], errors="coerce").dropna()
    r_values = pd.to_numeric(result_df["r_multiple"], errors="coerce").dropna()
    summary["avg_return"] = float(returns.mean()) if not returns.empty else None
    summary["median_return"] = float(returns.median()) if not returns.empty else None
    summary["win_rate"] = float((returns > 0).mean()) if not returns.empty else None
    summary["avg_r"] = float(r_values.mean()) if not r_values.empty else None
    summary["median_r"] = float(r_values.median()) if not r_values.empty else None
    summary["expectancy_r"] = summary["avg_r"]
    summary["max_drawdown_r_est"] = max_drawdown_r_estimate(result_df)

    exit_reason = result_df["exit_reason"].astype(str)
    summary["stop_first_rate"] = float((exit_reason == "stop_first").mean())
    summary["tp_first_rate"] = float((exit_reason == "tp_first").mean())
    summary["horizon_exit_rate"] = float((exit_reason == "horizon_exit").mean())
    return summary


def evaluate_policy_combo(
    policy_decisions: list[VirtualEntry],
    arrays: BarArrays,
    horizon: int,
    stop_atr: float,
    tp_atr: float,
    warnings: list[str],
) -> list[BracketResult]:
    """Replay one policy/grid combination."""

    results: list[BracketResult] = []
    for decision in policy_decisions:
        result = simulate_bracket_fast(arrays, decision, horizon, stop_atr, tp_atr, warnings)
        if result is not None:
            results.append(result)
    return results


def first_hit_offset(values: np.ndarray, threshold: float) -> int | None:
    """Return the first offset where values reach a threshold."""

    hits = values >= threshold
    if not np.any(hits):
        return None
    return int(np.argmax(hits))


def accumulate_decision_horizon_grid(
    decision: VirtualEntry,
    arrays: BarArrays,
    horizon: int,
    stop_atr_grid: list[float],
    tp_atr_grid: list[float],
    accumulators: dict[tuple[int, float, float], ComboAccumulator],
    warnings: list[str],
) -> None:
    """Accumulate all stop/tp grid outcomes for one decision and horizon."""

    if decision.skipped or decision.entry_dt is None or decision.entry_price is None or decision.bracket_start_index is None:
        return
    atr = decision.atr_1m
    if atr is None or atr <= 0:
        return
    entry_price = float(decision.entry_price)
    if entry_price <= 0:
        return

    start_index = int(decision.bracket_start_index)
    if start_index >= len(arrays.timestamp_ns):
        return
    target_dt = pd.Timestamp(decision.entry_dt) + pd.Timedelta(minutes=horizon)
    target_ns = timestamp_to_utc_ns(target_dt)
    end_index = int(np.searchsorted(arrays.timestamp_ns, target_ns, side="right"))
    if end_index <= start_index:
        return

    high = arrays.high[start_index:end_index]
    low = arrays.low[start_index:end_index]
    close = arrays.close[start_index:end_index]
    if high.size == 0 or low.size == 0 or close.size == 0:
        return

    if decision.direction == "long":
        favorable = high - entry_price
        adverse = entry_price - low
    else:
        favorable = entry_price - low
        adverse = high - entry_price

    stop_offsets = [first_hit_offset(adverse, stop_atr * atr) for stop_atr in stop_atr_grid]
    tp_offsets = [first_hit_offset(favorable, tp_atr * atr) for tp_atr in tp_atr_grid]
    horizon_close = close[-1]
    horizon_close_valid = bool(np.isfinite(horizon_close) and horizon_close > 0)

    for stop_index, stop_atr in enumerate(stop_atr_grid):
        stop_offset = stop_offsets[stop_index]
        for tp_index, tp_atr in enumerate(tp_atr_grid):
            tp_offset = tp_offsets[tp_index]
            stop_price_move = -stop_atr * atr
            tp_price_move = tp_atr * atr

            if stop_offset is None and tp_offset is None:
                if not horizon_close_valid:
                    continue
                exit_reason = "horizon_exit"
                price_move = directional_price_move(decision.direction, entry_price, float(horizon_close))
            elif stop_offset is not None and (tp_offset is None or stop_offset <= tp_offset):
                if tp_offset is not None and stop_offset == tp_offset:
                    warning = "同一根 1m bar 同时触发 stop/take-profit，保守按 stop first 处理"
                    if warning not in warnings:
                        warnings.append(warning)
                exit_reason = "stop_first"
                price_move = stop_price_move
            else:
                exit_reason = "tp_first"
                price_move = tp_price_move

            return_pct = price_move / entry_price
            r_multiple = price_move / (stop_atr * atr)
            accumulators[(horizon, stop_atr, tp_atr)].add(return_pct, r_multiple, exit_reason)


def build_bracket_grid(
    decisions: list[VirtualEntry],
    bars: pd.DataFrame,
    base_signal_count: int,
    horizons: list[int],
    stop_atr_grid: list[float],
    tp_atr_grid: list[float],
    warnings: list[str],
) -> pd.DataFrame:
    """Build summarized metrics for every grid combination."""

    rows: list[dict[str, Any]] = []
    if bars.empty:
        arrays = None
    else:
        arrays = build_bar_arrays(bars)

    decisions_by_policy = {
        policy_name: [decision for decision in decisions if decision.policy_name == policy_name and not decision.skipped]
        for policy_name in POLICY_NAMES
    }
    for policy_name in POLICY_NAMES:
        policy_decisions = decisions_by_policy.get(policy_name, [])
        accumulators = {
            (horizon, stop_atr, tp_atr): ComboAccumulator()
            for horizon in horizons
            for stop_atr in stop_atr_grid
            for tp_atr in tp_atr_grid
        }
        if arrays is not None and policy_decisions:
            for decision in policy_decisions:
                for horizon in horizons:
                    accumulate_decision_horizon_grid(
                        decision,
                        arrays,
                        horizon,
                        stop_atr_grid,
                        tp_atr_grid,
                        accumulators,
                        warnings,
                    )
        for horizon in horizons:
            for stop_atr in stop_atr_grid:
                for tp_atr in tp_atr_grid:
                    row = {
                        "policy_name": policy_name,
                        "horizon_m": horizon,
                        "stop_atr": stop_atr,
                        "tp_atr": tp_atr,
                    }
                    row.update(summarize_accumulator(accumulators[(horizon, stop_atr, tp_atr)], base_signal_count))
                    row["notes"] = POLICY_NOTES.get(policy_name, "")
                    rows.append(row)
    return pd.DataFrame(rows)


def build_best_result_details(
    decisions: list[VirtualEntry],
    bars: pd.DataFrame,
    leaderboard_df: pd.DataFrame,
    warnings: list[str],
) -> pd.DataFrame:
    """Replay only each policy's best combo for side/hour breakdowns."""

    if bars.empty or leaderboard_df.empty:
        return pd.DataFrame()

    arrays = build_bar_arrays(bars)
    decisions_by_policy = {
        policy_name: [decision for decision in decisions if decision.policy_name == policy_name and not decision.skipped]
        for policy_name in POLICY_NAMES
    }
    records: list[dict[str, Any]] = []
    for _, row in leaderboard_df.iterrows():
        policy_name = str(row.get("policy_name"))
        horizon = finite_or_none(row.get("best_horizon"))
        stop_atr = finite_or_none(row.get("best_stop_atr"))
        tp_atr = finite_or_none(row.get("best_tp_atr"))
        if horizon is None or stop_atr is None or tp_atr is None:
            continue
        results = evaluate_policy_combo(
            decisions_by_policy.get(policy_name, []),
            arrays,
            int(horizon),
            float(stop_atr),
            float(tp_atr),
            warnings,
        )
        records.extend(bracket_result_to_record(result) for result in results)

    if not records:
        return pd.DataFrame()
    return pd.DataFrame(records)


def best_grid_row_for_policy(grid_df: pd.DataFrame, policy_name: str) -> dict[str, Any]:
    """Return the best grid row for one policy."""

    policy_df = grid_df[grid_df["policy_name"] == policy_name].copy()
    if policy_df.empty:
        row = {column: None for column in LEADERBOARD_COLUMNS}
        row["policy_name"] = policy_name
        row["notes"] = POLICY_NOTES.get(policy_name, "")
        return row

    policy_df["_expectancy_sort"] = pd.to_numeric(policy_df["expectancy_r"], errors="coerce")
    policy_df["_entry_sort"] = pd.to_numeric(policy_df["entry_count"], errors="coerce").fillna(0)
    policy_df["_drawdown_sort"] = pd.to_numeric(policy_df["max_drawdown_r_est"], errors="coerce").fillna(float("inf"))
    policy_df = policy_df.sort_values(
        ["_expectancy_sort", "_entry_sort", "_drawdown_sort"],
        ascending=[False, False, True],
        na_position="last",
        kind="stable",
    )
    best = dict(policy_df.iloc[0].drop(labels=["_expectancy_sort", "_entry_sort", "_drawdown_sort"], errors="ignore"))
    best["best_horizon"] = best.pop("horizon_m", None)
    best["best_stop_atr"] = best.pop("stop_atr", None)
    best["best_tp_atr"] = best.pop("tp_atr", None)
    best["notes"] = POLICY_NOTES.get(policy_name, "")
    for column in LEADERBOARD_COLUMNS:
        best.setdefault(column, None)
    return {column: best.get(column) for column in LEADERBOARD_COLUMNS}


def build_leaderboard(grid_df: pd.DataFrame) -> pd.DataFrame:
    """Build one best row per policy."""

    rows = [best_grid_row_for_policy(grid_df, policy_name) for policy_name in POLICY_NAMES]
    leaderboard = pd.DataFrame(rows, columns=LEADERBOARD_COLUMNS)
    leaderboard["_expectancy_sort"] = pd.to_numeric(leaderboard["expectancy_r"], errors="coerce")
    leaderboard["_entry_sort"] = pd.to_numeric(leaderboard["entry_count"], errors="coerce").fillna(0)
    leaderboard = leaderboard.sort_values(
        ["_expectancy_sort", "_entry_sort"],
        ascending=[False, False],
        na_position="last",
        kind="stable",
    ).drop(columns=["_expectancy_sort", "_entry_sort"])
    return leaderboard.reset_index(drop=True)


def filter_result_df_to_best(result_df: pd.DataFrame, best_row: pd.Series) -> pd.DataFrame:
    """Filter detailed results to one policy's best grid combo."""

    if result_df.empty:
        return result_df.copy()
    policy_name = str(best_row.get("policy_name"))
    horizon = finite_or_none(best_row.get("best_horizon"))
    stop_atr = finite_or_none(best_row.get("best_stop_atr"))
    tp_atr = finite_or_none(best_row.get("best_tp_atr"))
    if horizon is None or stop_atr is None or tp_atr is None:
        return pd.DataFrame(columns=result_df.columns)
    mask = (
        (result_df["policy_name"] == policy_name)
        & (pd.to_numeric(result_df["horizon_m"], errors="coerce") == int(horizon))
        & (pd.to_numeric(result_df["stop_atr"], errors="coerce") == stop_atr)
        & (pd.to_numeric(result_df["tp_atr"], errors="coerce") == tp_atr)
    )
    return result_df[mask].copy()


def build_policy_by_side(entry_df: pd.DataFrame, result_df: pd.DataFrame, leaderboard_df: pd.DataFrame) -> pd.DataFrame:
    """Build policy metrics by long/short side using each policy's best combo."""

    rows: list[dict[str, Any]] = []
    total_by_side = entry_df.groupby("direction").size().to_dict() if not entry_df.empty else {}
    for _, best_row in leaderboard_df.iterrows():
        policy_name = str(best_row.get("policy_name"))
        best_results = filter_result_df_to_best(result_df, best_row)
        for side in ["long", "short"]:
            side_results = best_results[best_results["direction"] == side] if not best_results.empty else pd.DataFrame()
            row = {"policy_name": policy_name, "side": side}
            row.update(summarize_result_slice(side_results, int(total_by_side.get(side, 0))))
            row["best_horizon"] = best_row.get("best_horizon")
            row["best_stop_atr"] = best_row.get("best_stop_atr")
            row["best_tp_atr"] = best_row.get("best_tp_atr")
            rows.append(row)
    return pd.DataFrame(rows)


def build_policy_by_hour(entry_df: pd.DataFrame, result_df: pd.DataFrame, leaderboard_df: pd.DataFrame) -> pd.DataFrame:
    """Build policy metrics by signal hour using each policy's best combo."""

    rows: list[dict[str, Any]] = []
    total_by_hour = entry_df.groupby("hour").size().to_dict() if not entry_df.empty and "hour" in entry_df.columns else {}
    for _, best_row in leaderboard_df.iterrows():
        policy_name = str(best_row.get("policy_name"))
        best_results = filter_result_df_to_best(result_df, best_row)
        for hour in range(24):
            hour_results = best_results[best_results["hour"] == hour] if not best_results.empty else pd.DataFrame()
            row = {"policy_name": policy_name, "hour": hour}
            row.update(summarize_result_slice(hour_results, int(total_by_hour.get(hour, 0))))
            row["best_horizon"] = best_row.get("best_horizon")
            row["best_stop_atr"] = best_row.get("best_stop_atr")
            row["best_tp_atr"] = best_row.get("best_tp_atr")
            rows.append(row)
    return pd.DataFrame(rows)


def find_policy_row(leaderboard_df: pd.DataFrame, policy_name: str) -> pd.Series | None:
    """Return a leaderboard row by policy name."""

    if leaderboard_df.empty:
        return None
    matches = leaderboard_df[leaderboard_df["policy_name"] == policy_name]
    if matches.empty:
        return None
    return matches.iloc[0]


def row_expectancy(row: pd.Series | None) -> float | None:
    """Extract expectancy_r from a leaderboard row."""

    if row is None:
        return None
    return finite_or_none(row.get("expectancy_r"))


def compare_policy_to_baseline(
    leaderboard_df: pd.DataFrame,
    policy_name: str,
    baseline_expectancy: float | None,
) -> dict[str, Any]:
    """Compare one policy to immediate_baseline by expectancy_r."""

    row = find_policy_row(leaderboard_df, policy_name)
    expectancy = row_expectancy(row)
    delta = expectancy - baseline_expectancy if expectancy is not None and baseline_expectancy is not None else None
    improved = bool(delta is not None and delta > 0)
    if expectancy is None or baseline_expectancy is None:
        answer = "样本不足"
    elif improved:
        answer = "是"
    else:
        answer = "否"
    return {
        "policy_name": policy_name,
        "baseline_expectancy_r": baseline_expectancy,
        "policy_expectancy_r": expectancy,
        "delta_expectancy_r": delta,
        "improved": improved,
        "answer": answer,
    }


def compare_policy_group_to_baseline(
    leaderboard_df: pd.DataFrame,
    policy_names: list[str],
    baseline_expectancy: float | None,
) -> dict[str, Any]:
    """Compare a group of policies to immediate_baseline."""

    comparisons = [compare_policy_to_baseline(leaderboard_df, policy_name, baseline_expectancy) for policy_name in policy_names]
    best = None
    for item in comparisons:
        expectancy = finite_or_none(item.get("policy_expectancy_r"))
        if expectancy is None:
            continue
        if best is None or expectancy > float(best["policy_expectancy_r"]):
            best = item
    improved = bool(best is not None and best.get("improved"))
    answer = "样本不足" if best is None else ("是" if improved else "否")
    return {
        "policy_names": policy_names,
        "best_policy": best,
        "comparisons": comparisons,
        "improved": improved,
        "answer": answer,
    }


def infer_split_name(report_dir: Path) -> str | None:
    """Infer train/validation/oos from the report directory name."""

    name = report_dir.name.lower()
    if "train" in name:
        return "train"
    if "validation" in name or "valid" in name:
        return "validation"
    if "oos" in name:
        return "oos"
    return None


def load_sibling_leaderboard(report_dir: Path, split_name: str) -> pd.DataFrame | None:
    """Load a sibling split leaderboard if it exists."""

    candidate_dirs = [
        report_dir.parent / f"trace_{split_name}",
        report_dir.parent / split_name,
    ]
    for candidate_dir in candidate_dirs:
        path = candidate_dir / "entry_policy_research" / "entry_policy_leaderboard.csv"
        if path.exists() and path.is_file():
            return pd.read_csv(path)
    return None


def assess_cross_split_positive(report_dir: Path, leaderboard_df: pd.DataFrame) -> dict[str, Any]:
    """Check whether any policy is positive across train/validation/oos if sibling outputs exist."""

    split_name = infer_split_name(report_dir)
    split_frames: dict[str, pd.DataFrame] = {}
    for split in ["train", "validation", "oos"]:
        if split == split_name:
            split_frames[split] = leaderboard_df
            continue
        sibling = load_sibling_leaderboard(report_dir, split)
        if sibling is not None:
            split_frames[split] = sibling

    missing = [split for split in ["train", "validation", "oos"] if split not in split_frames]
    positive_by_split: dict[str, list[str]] = {}
    for split, frame in split_frames.items():
        if frame.empty or "policy_name" not in frame.columns or "expectancy_r" not in frame.columns:
            positive_by_split[split] = []
            continue
        working = frame.copy()
        working["expectancy_r"] = pd.to_numeric(working["expectancy_r"], errors="coerce")
        positive_by_split[split] = list(working[working["expectancy_r"] > 0]["policy_name"].astype(str))

    if missing:
        return {
            "available_splits": sorted(split_frames),
            "missing_splits": missing,
            "positive_by_split": positive_by_split,
            "positive_all_splits": [],
            "has_positive_all_splits": None,
            "answer": f"缺少 {', '.join(missing)} leaderboard，无法单次确认 train/validation/oos 全部为正",
        }

    common_positive = sorted(set(positive_by_split["train"]) & set(positive_by_split["validation"]) & set(positive_by_split["oos"]))
    return {
        "available_splits": ["train", "validation", "oos"],
        "missing_splits": [],
        "positive_by_split": positive_by_split,
        "positive_all_splits": common_positive,
        "has_positive_all_splits": bool(common_positive),
        "answer": "是" if common_positive else "否",
    }


def build_diagnostic_answers(report_dir: Path, leaderboard_df: pd.DataFrame) -> dict[str, Any]:
    """Build the required report answers."""

    baseline = row_expectancy(find_policy_row(leaderboard_df, "immediate_baseline"))
    current_positive = []
    if not leaderboard_df.empty:
        working = leaderboard_df.copy()
        working["expectancy_r"] = pd.to_numeric(working["expectancy_r"], errors="coerce")
        current_positive = list(working[working["expectancy_r"] > 0]["policy_name"].astype(str))

    cross_split = assess_cross_split_positive(report_dir, leaderboard_df)
    cross_has_positive = cross_split.get("has_positive_all_splits")
    if cross_has_positive is None:
        hypothesis_failed = not bool(current_positive)
    else:
        hypothesis_failed = not bool(cross_has_positive)

    return {
        "skip_breakout_gt_1_improves": compare_policy_to_baseline(
            leaderboard_df,
            "skip_large_breakout_gt_1atr",
            baseline,
        ),
        "skip_breakout_gt_2_improves": compare_policy_to_baseline(
            leaderboard_df,
            "skip_large_breakout_gt_2atr",
            baseline,
        ),
        "delayed_confirm_beats_immediate": compare_policy_group_to_baseline(
            leaderboard_df,
            ["delayed_confirm_1bar", "delayed_confirm_3bar"],
            baseline,
        ),
        "pullback_beats_immediate": compare_policy_group_to_baseline(
            leaderboard_df,
            ["pullback_to_breakout_level_5bar", "pullback_to_breakout_level_10bar"],
            baseline,
        ),
        "momentum_followthrough_beats_immediate": compare_policy_to_baseline(
            leaderboard_df,
            "momentum_followthrough_3bar",
            baseline,
        ),
        "current_positive_expectancy_policies": current_positive,
        "cross_split_positive_expectancy": cross_split,
        "entry_policy_hypothesis_failed": hypothesis_failed,
    }


def format_number(value: Any, digits: int = 6) -> str:
    """Format a number for markdown."""

    number = finite_or_none(value)
    if number is None:
        return "N/A"
    return f"{number:.{digits}f}"


def render_leaderboard_table(leaderboard_df: pd.DataFrame, limit: int = 10) -> str:
    """Render the top leaderboard rows as a markdown table."""

    columns = [
        "policy_name",
        "entry_count",
        "expectancy_r",
        "median_r",
        "win_rate",
        "best_horizon",
        "best_stop_atr",
        "best_tp_atr",
        "stop_first_rate",
        "horizon_exit_rate",
    ]
    lines = [
        "| policy | entries | exp_r | median_r | win_rate | horizon | stop | tp | stop_first | horizon_exit |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in leaderboard_df.head(limit).iterrows():
        entry_count = finite_or_none(row.get("entry_count"))
        lines.append(
            "| "
            f"{row.get('policy_name')} | "
            f"{int(entry_count or 0)} | "
            f"{format_number(row.get('expectancy_r'), 4)} | "
            f"{format_number(row.get('median_r'), 4)} | "
            f"{format_number(row.get('win_rate'), 4)} | "
            f"{row.get('best_horizon')} | "
            f"{format_number(row.get('best_stop_atr'), 2)} | "
            f"{format_number(row.get('best_tp_atr'), 2)} | "
            f"{format_number(row.get('stop_first_rate'), 4)} | "
            f"{format_number(row.get('horizon_exit_rate'), 4)} |"
        )
    return "\n".join(lines)


def render_policy_report(summary: dict[str, Any], leaderboard_df: pd.DataFrame) -> str:
    """Render the markdown policy report."""

    answers = summary.get("diagnostic_answers") or {}
    cross = answers.get("cross_split_positive_expectancy") or {}
    warnings = summary.get("warnings") or []
    warning_lines = "\n".join(f"- {item}" for item in warnings) if warnings else "- 无"
    policy_lines = "\n".join(f"- `{name}`: {POLICY_NOTES[name]}" for name in POLICY_NAMES)

    return (
        "# 入场政策离线研究报告\n\n"
        "## 结论问题\n"
        f"- 跳过 `breakout_distance_atr > 1` 是否改善：{(answers.get('skip_breakout_gt_1_improves') or {}).get('answer')}\n"
        f"- 跳过 `breakout_distance_atr > 2` 是否改善：{(answers.get('skip_breakout_gt_2_improves') or {}).get('answer')}\n"
        f"- delayed confirm 是否优于 immediate entry：{(answers.get('delayed_confirm_beats_immediate') or {}).get('answer')}\n"
        f"- pullback entry 是否优于 immediate entry：{(answers.get('pullback_beats_immediate') or {}).get('answer')}\n"
        f"- momentum followthrough 是否优于 immediate entry：{(answers.get('momentum_followthrough_beats_immediate') or {}).get('answer')}\n"
        f"- 是否有 policy 在 train / validation / oos 都可能为正 expectancy：{cross.get('answer')}\n"
        f"- entry_policy_hypothesis_failed={answers.get('entry_policy_hypothesis_failed')}\n\n"
        "## 当前样本\n"
        f"- report_dir: `{summary.get('report_dir')}`\n"
        f"- signal_trace: `{summary.get('signal_trace_path')}`\n"
        f"- entry signal 数: {summary.get('entry_signal_count')}\n"
        f"- horizons: {summary.get('horizons')}\n"
        f"- stop_atr_grid: {summary.get('stop_atr_grid')}\n"
        f"- tp_atr_grid: {summary.get('tp_atr_grid')}\n"
        f"- 当前样本正 expectancy policies: {answers.get('current_positive_expectancy_policies')}\n"
        f"- 跨样本正 expectancy policies: {cross.get('positive_all_splits')}\n\n"
        "## Leaderboard Top 10\n"
        f"{render_leaderboard_table(leaderboard_df, limit=10)}\n\n"
        "## Policy 定义\n"
        f"{policy_lines}\n\n"
        "## 输出文件\n"
        "- entry_policy_summary.json\n"
        "- entry_policy_leaderboard.csv\n"
        "- bracket_grid.csv\n"
        "- policy_by_side.csv\n"
        "- policy_by_hour.csv\n"
        "- policy_report.md\n\n"
        "## Warning\n"
        f"{warning_lines}\n"
    )


def build_summary(
    report_dir: Path,
    signal_trace_path: Path,
    output_dir: Path,
    entry_df: pd.DataFrame,
    decisions: list[VirtualEntry],
    result_df: pd.DataFrame,
    grid_df: pd.DataFrame,
    leaderboard_df: pd.DataFrame,
    policy_by_side_df: pd.DataFrame,
    policy_by_hour_df: pd.DataFrame,
    horizons: list[int],
    stop_atr_grid: list[float],
    tp_atr_grid: list[float],
    max_wait_bars: int,
    timezone_name: str,
    warnings: list[str],
) -> dict[str, Any]:
    """Build summary JSON."""

    skipped_by_policy: dict[str, dict[str, int]] = {}
    for policy_name in POLICY_NAMES:
        policy_decisions = [decision for decision in decisions if decision.policy_name == policy_name]
        skipped_by_policy[policy_name] = {
            "entered_decisions": int(sum(not decision.skipped for decision in policy_decisions)),
            "skipped_decisions": int(sum(decision.skipped for decision in policy_decisions)),
        }

    summary = {
        "report_dir": str(report_dir),
        "signal_trace_path": str(signal_trace_path),
        "output_dir": str(output_dir),
        "timezone": timezone_name,
        "horizons": horizons,
        "max_wait_bars": max_wait_bars,
        "stop_atr_grid": stop_atr_grid,
        "tp_atr_grid": tp_atr_grid,
        "entry_signal_count": int(len(entry_df.index)),
        "virtual_decision_count": int(len(decisions)),
        "bracket_result_count": int(len(result_df.index)),
        "warnings": warnings,
        "policies": [{"policy_name": name, "notes": POLICY_NOTES[name]} for name in POLICY_NAMES],
        "skipped_by_policy": skipped_by_policy,
        "leaderboard": dataframe_records(leaderboard_df),
        "bracket_grid_top": dataframe_records(grid_df.head(30)),
        "policy_by_side": dataframe_records(policy_by_side_df),
        "policy_by_hour_top": dataframe_records(policy_by_hour_df.head(60)),
    }
    summary["diagnostic_answers"] = build_diagnostic_answers(report_dir, leaderboard_df)
    if summary["diagnostic_answers"].get("entry_policy_hypothesis_failed"):
        warning = "entry policy hypothesis failed"
        if warning not in warnings:
            warnings.append(warning)
    return summary


def write_outputs(
    output_dir: Path,
    summary: dict[str, Any],
    leaderboard_df: pd.DataFrame,
    grid_df: pd.DataFrame,
    policy_by_side_df: pd.DataFrame,
    policy_by_hour_df: pd.DataFrame,
    markdown: str,
) -> None:
    """Write all required artifacts."""

    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "entry_policy_summary.json", summary)
    write_dataframe(output_dir / "entry_policy_leaderboard.csv", leaderboard_df)
    write_dataframe(output_dir / "bracket_grid.csv", grid_df)
    write_dataframe(output_dir / "policy_by_side.csv", policy_by_side_df)
    write_dataframe(output_dir / "policy_by_hour.csv", policy_by_hour_df)
    (output_dir / "policy_report.md").write_text(markdown, encoding="utf-8")


def run_research(
    report_dir: Path,
    signal_trace_path: Path,
    output_dir: Path,
    horizons: list[int],
    stop_atr_grid: list[float],
    tp_atr_grid: list[float],
    max_wait_bars: int,
    timezone_name: str,
    bars_from_db: bool,
    logger: logging.Logger,
    bars_df: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Run the full offline entry policy research workflow."""

    ZoneInfo(timezone_name)
    if max_wait_bars <= 0:
        raise EntryPolicyResearchError("--max-wait-bars 必须为正整数")

    warnings: list[str] = []
    trace_df = read_signal_trace(signal_trace_path, timezone_name)
    entry_df = prepare_entry_signals(trace_df, warnings)
    if "hour" in entry_df.columns:
        entry_df["hour"] = pd.to_numeric(entry_df["hour"], errors="coerce")

    if bars_df is None:
        if not bars_from_db:
            raise EntryPolicyResearchError("--no-bars-from-db 已设置，但当前 CLI 未提供替代 bars 输入")
        load_horizon = [max(horizons) + max_wait_bars + 1]
        bars = load_bars_from_db(entry_df, load_horizon, timezone_name, warnings, logger)
    else:
        bars = dataframe_bars_to_ohlc(bars_df, timezone_name)

    if bars.empty:
        warnings.append("1m bars 为空，无法计算 entry policy bracket")

    decisions = build_virtual_entries(entry_df, bars, max_wait_bars) if not entry_df.empty and not bars.empty else []
    grid_df = build_bracket_grid(
        decisions,
        bars,
        int(len(entry_df.index)),
        horizons,
        stop_atr_grid,
        tp_atr_grid,
        warnings,
    )
    leaderboard_df = build_leaderboard(grid_df)
    result_df = build_best_result_details(decisions, bars, leaderboard_df, warnings)
    policy_by_side_df = build_policy_by_side(entry_df, result_df, leaderboard_df)
    policy_by_hour_df = build_policy_by_hour(entry_df, result_df, leaderboard_df)

    summary = build_summary(
        report_dir=report_dir,
        signal_trace_path=signal_trace_path,
        output_dir=output_dir,
        entry_df=entry_df,
        decisions=decisions,
        result_df=result_df,
        grid_df=grid_df,
        leaderboard_df=leaderboard_df,
        policy_by_side_df=policy_by_side_df,
        policy_by_hour_df=policy_by_hour_df,
        horizons=horizons,
        stop_atr_grid=stop_atr_grid,
        tp_atr_grid=tp_atr_grid,
        max_wait_bars=max_wait_bars,
        timezone_name=timezone_name,
        warnings=warnings,
    )
    markdown = render_policy_report(summary, leaderboard_df)
    write_outputs(output_dir, summary, leaderboard_df, grid_df, policy_by_side_df, policy_by_hour_df, markdown)
    return summary


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""

    args = parse_args(argv)
    ensure_headless_runtime()
    logger = setup_logging("research_entry_policies", verbose=args.verbose)

    try:
        report_dir = resolve_path(args.report_dir)
        signal_trace_path = resolve_path(args.signal_trace, report_dir / "signal_trace.csv")
        output_dir = resolve_path(args.output_dir, report_dir / "entry_policy_research")
        horizons = parse_positive_int_list(args.horizons, "--horizons")
        stop_atr_grid = parse_positive_float_list(args.stop_atr_grid, "--stop-atr-grid")
        tp_atr_grid = parse_positive_float_list(args.tp_atr_grid, "--tp-atr-grid")
        summary = run_research(
            report_dir=report_dir,
            signal_trace_path=signal_trace_path,
            output_dir=output_dir,
            horizons=horizons,
            stop_atr_grid=stop_atr_grid,
            tp_atr_grid=tp_atr_grid,
            max_wait_bars=int(args.max_wait_bars),
            timezone_name=args.timezone,
            bars_from_db=bool(args.bars_from_db),
            logger=logger,
        )
        print_json_block(
            "Entry policy research summary:",
            {
                "output_dir": output_dir,
                "entry_signal_count": summary.get("entry_signal_count"),
                "entry_policy_hypothesis_failed": (summary.get("diagnostic_answers") or {}).get(
                    "entry_policy_hypothesis_failed"
                ),
                "warnings": summary.get("warnings"),
            },
        )
        if args.json:
            print(json.dumps(to_jsonable(summary), ensure_ascii=False, indent=2))
        return 0
    except (EntryPolicyResearchError, SignalOutcomeError) as exc:
        log_event(logger, logging.ERROR, "entry_policy_research.error", str(exc))
        return 2
    except Exception:
        logger.exception(
            "Unexpected error during entry policy research",
            extra={"event": "entry_policy_research.unexpected_error"},
        )
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
