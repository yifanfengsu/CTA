#!/usr/bin/env python3
"""Research-only trend capture and exit convexity diagnostics."""

from __future__ import annotations

import argparse
import json
import logging
import math
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from common_runtime import PROJECT_ROOT, ensure_headless_runtime, log_event, print_json_block, setup_logging, to_jsonable
from history_time_utils import DEFAULT_TIMEZONE


DEFAULT_TREND_MAP_DIR = PROJECT_ROOT / "reports" / "research" / "trend_opportunity_map"
DEFAULT_TREND_V3_DIR = PROJECT_ROOT / "reports" / "research" / "trend_following_v3_extended"
DEFAULT_VSVCB_DIR = PROJECT_ROOT / "reports" / "research" / "vsvcb_v1"
DEFAULT_CSRB_DIR = PROJECT_ROOT / "reports" / "research" / "csrb_v1"
DEFAULT_FUNDING_DIR = PROJECT_ROOT / "data" / "funding" / "okx"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "reports" / "research" / "trend_capture_exit_convexity"
DEFAULT_DATABASE_PATH = PROJECT_ROOT / ".vntrader" / "database.db"

SPLITS = ("train_ext", "validation_ext", "oos_ext")
SPLIT_ALIASES = {"train": "train_ext", "validation": "validation_ext", "oos": "oos_ext"}
SPLIT_RANGES = {
    "train_ext": ("2023-01-01", "2024-07-01"),
    "validation_ext": ("2024-07-01", "2025-07-01"),
    "oos_ext": ("2025-07-01", "2026-04-01"),
}
TIMEFRAME_MINUTES = {"1m": 1, "15m": 15, "30m": 30, "1h": 60, "4h": 240, "1d": 1440}
EXIT_VARIANTS = (
    "original_exit",
    "fixed_hold_longer_2x",
    "fixed_hold_longer_4x",
    "atr_chandelier_3x",
    "atr_chandelier_5x",
    "swing_trailing_exit",
    "time_stop_if_no_progress",
    "oracle_hold_to_trend_end",
)
ORACLE_VARIANT = "oracle_hold_to_trend_end"
CAPTURE_IMPROVEMENT_MIN_ABS = 0.01
CAPTURE_IMPROVEMENT_MIN_REL = 1.10
EARLY_EXIT_REDUCTION_MIN_ABS = 0.05
EARLY_EXIT_REDUCTION_MIN_REL = 0.90
TRAILING_EXIT_MAX_DAYS = 60

REQUIRED_OUTPUT_FILES = [
    "trend_capture_exit_summary.json",
    "trend_capture_exit_report.md",
    "trend_capture_diagnostics.csv",
    "exit_variant_trades.csv",
    "exit_variant_summary.csv",
    "exit_variant_by_policy.csv",
    "exit_variant_by_symbol.csv",
    "exit_variant_by_split.csv",
    "exit_variant_concentration.csv",
    "oracle_upper_bound.csv",
    "funding_adjusted_exit_summary.csv",
    "rejected_exit_variants.csv",
]

DIAGNOSTIC_COLUMNS = [
    "strategy_source",
    "policy_or_group",
    "symbol",
    "trade_timeframe",
    "primary_timeframe",
    "trade_id",
    "split",
    "direction",
    "entry_time",
    "exit_time",
    "trend_segment_id",
    "trend_direction",
    "entered_trend_segment",
    "entry_phase",
    "exit_phase",
    "captured_fraction_of_segment",
    "missed_fraction_after_exit",
    "entry_lag_bars",
    "exit_lag_bars",
    "early_exit_flag",
    "late_entry_flag",
    "no_cost_pnl",
    "cost_aware_pnl",
]

EXIT_TRADE_COLUMNS = [
    "exit_variant",
    "oracle",
    "strategy_source",
    "policy_or_group",
    "symbol",
    "inst_id",
    "trade_timeframe",
    "primary_timeframe",
    "trade_id",
    "split",
    "direction",
    "entry_time",
    "entry_price",
    "exit_time",
    "exit_price",
    "holding_minutes",
    "original_exit_time",
    "original_holding_minutes",
    "exit_reason",
    "trend_segment_id",
    "entry_phase",
    "exit_phase",
    "captured_fraction_of_segment",
    "missed_fraction_after_exit",
    "entry_lag_bars",
    "exit_lag_bars",
    "early_exit_flag",
    "late_entry_flag",
    "no_cost_pnl",
    "cost_drag",
    "cost_aware_pnl",
    "funding_pnl",
    "funding_adjusted_pnl",
    "funding_events_count",
    "funding_data_available",
    "funding_interval_covered",
    "notional",
    "pnl_multiplier",
]


class TrendCaptureExitResearchError(Exception):
    """Raised when the research script cannot continue."""


@dataclass(frozen=True, slots=True)
class ExitSimulation:
    """One counterfactual exit result."""

    exit_time: pd.Timestamp
    exit_price: float
    exit_reason: str
    debug_path: list[float] | None = None


@dataclass(frozen=True, slots=True)
class ResearchOutputs:
    """Generated outputs for tests and CLI reporting."""

    output_dir: Path
    summary: dict[str, Any]
    diagnostics: pd.DataFrame
    exit_trades: pd.DataFrame
    exit_summary: pd.DataFrame


@dataclass(frozen=True, slots=True)
class FundingIndex:
    """Compact funding timeline for fast inclusive interval lookup."""

    times_ns: np.ndarray
    prefix_rates: np.ndarray
    first_time: pd.Timestamp
    last_time: pd.Timestamp


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description="Research-only trend capture and exit convexity diagnostics.")
    parser.add_argument("--trend-map-dir", default=str(DEFAULT_TREND_MAP_DIR))
    parser.add_argument("--trend-v3-dir", default=str(DEFAULT_TREND_V3_DIR))
    parser.add_argument("--vsvcb-dir", default=str(DEFAULT_VSVCB_DIR))
    parser.add_argument("--csrb-dir", default=str(DEFAULT_CSRB_DIR))
    parser.add_argument("--funding-dir", default=str(DEFAULT_FUNDING_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--timezone", default=DEFAULT_TIMEZONE)
    parser.add_argument("--primary-timeframe", default="4h")
    parser.add_argument("--data-check-strict", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def resolve_path(path_arg: str | Path) -> Path:
    """Resolve a path relative to the project root."""

    path = Path(path_arg)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def finite_float(value: Any, default: float | None = 0.0) -> float | None:
    """Return a finite float or the provided default."""

    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if not np.isfinite(number):
        return default
    return number


def dataframe_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Convert a dataframe to JSON-safe row dictionaries."""

    if df.empty:
        return []
    safe = df.replace({np.nan: None})
    return json.loads(safe.to_json(orient="records", force_ascii=False))


def read_json_if_exists(path: Path, warnings: list[str]) -> dict[str, Any]:
    """Read JSON and record warnings instead of failing."""

    if not path.exists():
        warnings.append(f"missing_json:{path}")
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        warnings.append(f"failed_to_read_json:{path}:{exc}")
        return {}


def read_csv_if_exists(path: Path, warnings: list[str], *, required: bool = False) -> pd.DataFrame:
    """Read a CSV and record warnings instead of failing."""

    if not path.exists():
        prefix = "missing_required_csv" if required else "missing_optional_csv"
        warnings.append(f"{prefix}:{path}")
        return pd.DataFrame()
    try:
        return pd.read_csv(path, low_memory=False)
    except Exception as exc:
        warnings.append(f"failed_to_read_csv:{path}:{exc}")
        return pd.DataFrame()


def parse_timestamp(value: Any, timezone_name: str) -> pd.Timestamp | None:
    """Parse a timestamp and normalize it to the research timezone."""

    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    try:
        timestamp = pd.Timestamp(value)
    except Exception:
        return None
    if pd.isna(timestamp):
        return None
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize(timezone_name)
    else:
        timestamp = timestamp.tz_convert(timezone_name)
    return pd.Timestamp(timestamp)


def timestamp_to_utc(value: pd.Timestamp) -> pd.Timestamp:
    """Convert a timestamp to UTC."""

    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def format_timestamp(value: Any) -> str | None:
    """Format timestamps for CSV/JSON output."""

    if value is None:
        return None
    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp):
        return None
    return timestamp.isoformat()


def infer_split(entry_time: pd.Timestamp, timezone_name: str) -> str:
    """Infer extended split by entry timestamp."""

    entry = pd.Timestamp(entry_time)
    if entry.tzinfo is None:
        entry = entry.tz_localize(timezone_name)
    else:
        entry = entry.tz_convert(timezone_name)
    for split, (start, end) in SPLIT_RANGES.items():
        start_ts = pd.Timestamp(start, tz=timezone_name)
        end_ts = pd.Timestamp(end, tz=timezone_name)
        if start_ts <= entry < end_ts:
            return split
    return "out_of_range"


def normalize_split(value: Any, entry_time: pd.Timestamp | None, timezone_name: str) -> str:
    """Normalize split labels to extended split labels."""

    raw = str(value or "").strip()
    if raw in SPLITS:
        return raw
    if raw in SPLIT_ALIASES:
        return SPLIT_ALIASES[raw]
    if entry_time is not None:
        return infer_split(entry_time, timezone_name)
    return "unknown"


def symbol_to_inst_id(symbol: str) -> str:
    """Map repository vt_symbol strings to OKX instrument ids."""

    value = str(symbol or "").strip()
    if value.endswith("-SWAP") and "-USDT-" in value:
        return value
    root = value.split(".")[0]
    root = root.replace("_OKX", "")
    if root.endswith("_SWAP"):
        pair = root[: -len("_SWAP")]
        if pair.endswith("USDT"):
            return f"{pair[:-4]}-USDT-SWAP"
    if root.endswith("USDT"):
        return f"{root[:-4]}-USDT-SWAP"
    return value


def split_vt_symbol(vt_symbol: str) -> tuple[str, str]:
    """Split vt_symbol into sqlite symbol and exchange values."""

    symbol, separator, exchange = str(vt_symbol).partition(".")
    if not separator or not symbol or not exchange:
        raise TrendCaptureExitResearchError(f"cannot parse vt_symbol: {vt_symbol!r}")
    return symbol, exchange


def directional_price_pnl(direction: str, entry_price: float, exit_price: float, multiplier: float) -> float:
    """Return linear contract price PnL using a pre-inferred multiplier."""

    if str(direction).lower() == "short":
        return float((entry_price - exit_price) * multiplier)
    return float((exit_price - entry_price) * multiplier)


def infer_pnl_multiplier_and_notional(row: pd.Series, direction: str, entry_price: float, exit_price: float) -> tuple[float, float]:
    """Infer PnL multiplier and notional without changing legacy entry sizing."""

    volume = abs(finite_float(row.get("volume"), default=np.nan) or np.nan)
    contract_size = abs(finite_float(row.get("contract_size"), default=np.nan) or np.nan)
    if np.isfinite(volume) and np.isfinite(contract_size) and volume > 0 and contract_size > 0:
        multiplier = float(volume * contract_size)
        return multiplier, abs(float(entry_price * multiplier))

    no_cost = first_finite(row, ["no_cost_pnl", "no_cost_net_pnl", "gross_pnl"])
    move = directional_price_pnl(direction, entry_price, exit_price, 1.0)
    if no_cost is not None and abs(move) > 1e-12:
        multiplier = abs(float(no_cost) / move)
        if multiplier > 0 and np.isfinite(multiplier):
            return multiplier, abs(float(entry_price * multiplier))

    gross_return = finite_float(row.get("gross_return"), default=np.nan)
    if no_cost is not None and gross_return is not None and np.isfinite(gross_return) and abs(gross_return) > 1e-12:
        notional = abs(float(no_cost) / float(gross_return))
        if notional > 0 and np.isfinite(notional) and entry_price > 0:
            return float(notional / entry_price), float(notional)

    turnover = abs(finite_float(row.get("turnover"), default=np.nan) or np.nan)
    if np.isfinite(turnover) and turnover > 0 and entry_price > 0:
        notional = turnover / 2.0
        return float(notional / entry_price), float(notional)

    return 0.0, 0.0


def first_finite(row: pd.Series, columns: Iterable[str], default: float | None = None) -> float | None:
    """Return first finite numeric value from row columns."""

    for column in columns:
        if column in row:
            value = finite_float(row.get(column), default=np.nan)
            if value is not None and np.isfinite(value):
                return float(value)
    return default


def infer_cost_drag(row: pd.Series, no_cost: float | None, cost_aware: float | None) -> float:
    """Infer absolute entry/exit cost drag from legacy trade columns."""

    fee = first_finite(row, ["fee", "fee_cost"], default=0.0) or 0.0
    slippage = first_finite(row, ["slippage", "slippage_cost"], default=0.0) or 0.0
    if abs(fee) + abs(slippage) > 0:
        return float(abs(fee) + abs(slippage))
    if no_cost is not None and cost_aware is not None:
        return float(no_cost - cost_aware)
    return 0.0


def normalize_trade_frame(
    frame: pd.DataFrame,
    *,
    strategy_source: str,
    source_file: Path,
    split_hint: str | None,
    timezone_name: str,
) -> pd.DataFrame:
    """Normalize one legacy trade CSV into a stable internal schema."""

    rows: list[dict[str, Any]] = []
    if frame.empty:
        return pd.DataFrame()
    for index, row in frame.iterrows():
        symbol = str(row.get("symbol") or "").strip()
        direction = str(row.get("direction") or "").strip().lower()
        entry_time = parse_timestamp(row.get("entry_time"), timezone_name)
        exit_time = parse_timestamp(row.get("exit_time"), timezone_name)
        if not symbol or direction not in {"long", "short"} or entry_time is None or exit_time is None:
            continue
        if exit_time < entry_time:
            entry_time, exit_time = exit_time, entry_time
        entry_price = finite_float(row.get("entry_price"), default=np.nan)
        exit_price = finite_float(row.get("exit_price"), default=np.nan)
        if entry_price is None or exit_price is None or not np.isfinite(entry_price) or not np.isfinite(exit_price) or entry_price <= 0 or exit_price <= 0:
            continue
        policy = str(row.get("policy_name") or row.get("group") or row.get("policy_or_group") or strategy_source)
        trade_id = str(row.get("trade_id") or f"{strategy_source}_{index:08d}")
        trade_timeframe = str(row.get("timeframe") or "")
        no_cost = first_finite(row, ["no_cost_pnl", "no_cost_net_pnl", "gross_pnl"])
        cost_aware = first_finite(row, ["net_pnl", "cost_aware_pnl"])
        if no_cost is None:
            no_cost = directional_price_pnl(direction, float(entry_price), float(exit_price), 0.0)
        if cost_aware is None:
            cost_aware = no_cost
        multiplier, notional = infer_pnl_multiplier_and_notional(row, direction, float(entry_price), float(exit_price))
        cost_drag = infer_cost_drag(row, no_cost, cost_aware)
        holding_minutes = finite_float(row.get("holding_minutes"), default=None)
        if holding_minutes is None:
            holding_minutes = max((exit_time - entry_time).total_seconds() / 60.0, 0.0)
        split = normalize_split(split_hint or row.get("split"), entry_time, timezone_name)
        rows.append(
            {
                "strategy_source": strategy_source,
                "policy_or_group": policy,
                "symbol": symbol,
                "trade_timeframe": trade_timeframe,
                "trade_id": trade_id,
                "split": split,
                "direction": direction,
                "entry_time": format_timestamp(entry_time),
                "entry_ts": entry_time,
                "entry_price": float(entry_price),
                "exit_time": format_timestamp(exit_time),
                "exit_ts": exit_time,
                "exit_price": float(exit_price),
                "holding_minutes": float(holding_minutes),
                "volume": finite_float(row.get("volume"), default=np.nan),
                "contract_size": finite_float(row.get("contract_size"), default=np.nan),
                "no_cost_pnl": float(no_cost),
                "cost_aware_pnl": float(cost_aware),
                "cost_drag": float(cost_drag),
                "pnl_multiplier": float(multiplier),
                "notional": float(notional),
                "source_file": str(source_file),
            }
        )
    return pd.DataFrame(rows)


def load_legacy_trades(
    trend_v3_dir: Path,
    vsvcb_dir: Path,
    csrb_dir: Path,
    timezone_name: str,
    warnings: list[str],
) -> pd.DataFrame:
    """Load required V3 and optional VSVCB/CSRB legacy trade files."""

    frames: list[pd.DataFrame] = []
    for split in SPLITS:
        path = trend_v3_dir / split / "trend_v3_trades.csv"
        raw = read_csv_if_exists(path, warnings, required=True)
        normalized = normalize_trade_frame(
            raw,
            strategy_source="trend_v3_extended",
            source_file=path,
            split_hint=split,
            timezone_name=timezone_name,
        )
        if not normalized.empty:
            frames.append(normalized)

    optional_files = [
        ("vsvcb_v1", vsvcb_dir / "trades.csv"),
        ("csrb_v1", csrb_dir / "trades.csv"),
    ]
    for source, path in optional_files:
        raw = read_csv_if_exists(path, warnings, required=False)
        normalized = normalize_trade_frame(
            raw,
            strategy_source=source,
            source_file=path,
            split_hint=None,
            timezone_name=timezone_name,
        )
        if not normalized.empty:
            frames.append(normalized)

    if not frames:
        return pd.DataFrame()
    trades = pd.concat(frames, ignore_index=True)
    return trades.sort_values(["strategy_source", "policy_or_group", "entry_ts", "symbol", "trade_id"], kind="stable").reset_index(drop=True)


def normalize_segments(segments: pd.DataFrame, timezone_name: str) -> pd.DataFrame:
    """Normalize trend segment timestamps and numeric fields."""

    if segments.empty:
        return pd.DataFrame()
    frame = segments.copy()
    frame["start_ts"] = [parse_timestamp(value, timezone_name) for value in frame.get("start_time", pd.Series(dtype=object))]
    frame["end_ts"] = [parse_timestamp(value, timezone_name) for value in frame.get("end_time", pd.Series(dtype=object))]
    frame = frame.dropna(subset=["start_ts", "end_ts"]).copy()
    for column in ["duration_bars", "abs_trend_return"]:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.sort_values(["symbol", "timeframe", "start_ts", "end_ts"], kind="stable").reset_index(drop=True)


def build_segments_by_symbol(segments: pd.DataFrame, primary_timeframe: str) -> dict[str, pd.DataFrame]:
    """Build primary-timeframe segment frames by symbol."""

    if segments.empty:
        return {}
    primary = segments[segments["timeframe"].astype(str) == primary_timeframe].copy()
    return {str(symbol): group.reset_index(drop=True) for symbol, group in primary.groupby("symbol", dropna=False)}


def overlap_fraction(entry: pd.Timestamp, exit_time: pd.Timestamp, segment_start: pd.Timestamp, segment_end: pd.Timestamp) -> float:
    """Return trade/segment overlap as a fraction of segment duration."""

    if segment_end <= segment_start:
        return 0.0
    overlap_start = max(entry, segment_start)
    overlap_end = min(exit_time, segment_end)
    if overlap_end <= overlap_start:
        return 0.0
    return float((overlap_end - overlap_start).total_seconds() / max((segment_end - segment_start).total_seconds(), 1.0))


def classify_entry_phase(entry: pd.Timestamp, segment_start: pd.Timestamp | None, segment_end: pd.Timestamp | None) -> str:
    """Classify entry timing versus the selected trend segment."""

    if segment_start is None or segment_end is None:
        return "nontrend"
    if entry < segment_start:
        return "before_trend"
    if entry > segment_end:
        return "after_trend"
    duration = max((segment_end - segment_start).total_seconds(), 1.0)
    ratio = (entry - segment_start).total_seconds() / duration
    if ratio <= 1.0 / 3.0:
        return "early_trend"
    if ratio <= 2.0 / 3.0:
        return "middle_trend"
    return "late_trend"


def classify_exit_phase(exit_time: pd.Timestamp, segment_start: pd.Timestamp | None, segment_end: pd.Timestamp | None) -> str:
    """Classify exit timing versus the selected trend segment."""

    if segment_start is None or segment_end is None:
        return "nontrend"
    duration = max((segment_end - segment_start).total_seconds(), 1.0)
    near_window = max(duration * 0.10, 1.0)
    lag_seconds = (exit_time - segment_end).total_seconds()
    if abs(lag_seconds) <= near_window:
        return "near_trend_end"
    if exit_time < segment_end:
        return "before_trend_end"
    return "after_trend_end"


def find_selected_segment(
    symbol: str,
    entry: pd.Timestamp,
    exit_time: pd.Timestamp,
    segments_by_symbol: dict[str, pd.DataFrame],
) -> pd.Series | None:
    """Select the overlapping segment with the largest capture, then return/duration."""

    segments = segments_by_symbol.get(symbol)
    if segments is None or segments.empty:
        return None
    best_row: pd.Series | None = None
    best_key = (-1.0, -1.0, -1.0)
    for _, segment in segments.iterrows():
        start = pd.Timestamp(segment["start_ts"])
        end = pd.Timestamp(segment["end_ts"])
        capture = overlap_fraction(entry, exit_time, start, end)
        if capture <= 0:
            continue
        abs_return = finite_float(segment.get("abs_trend_return"), default=0.0) or 0.0
        duration = finite_float(segment.get("duration_bars"), default=0.0) or 0.0
        key = (capture, float(abs_return), float(duration))
        if key > best_key:
            best_key = key
            best_row = segment
    return best_row


def select_segments_for_trades(trades: pd.DataFrame, segments_by_symbol: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Vectorize original trade-to-segment selection for later reuse."""

    columns = [
        "trend_segment_id",
        "trend_direction",
        "segment_start_ts",
        "segment_end_ts",
        "segment_abs_trend_return",
        "segment_duration_bars",
    ]
    if trades.empty:
        return pd.DataFrame(columns=columns, index=trades.index)

    selected = pd.DataFrame(index=trades.index, columns=columns)
    for symbol, group in trades.groupby("symbol", dropna=False):
        segments = segments_by_symbol.get(str(symbol))
        if segments is None or segments.empty:
            continue
        indexes = group.index.to_numpy(dtype=int)
        entries_ns = group["entry_ts"].map(lambda value: pd.Timestamp(value).value).to_numpy(dtype=np.int64)
        exits_ns = group["exit_ts"].map(lambda value: pd.Timestamp(value).value).to_numpy(dtype=np.int64)
        best_capture = np.zeros(len(indexes), dtype=float)
        best_return = np.full(len(indexes), -np.inf, dtype=float)
        best_duration = np.full(len(indexes), -np.inf, dtype=float)
        best_rows: list[pd.Series | None] = [None] * len(indexes)
        for _, segment in segments.iterrows():
            start = pd.Timestamp(segment["start_ts"])
            end = pd.Timestamp(segment["end_ts"])
            start_ns = int(start.value)
            end_ns = int(end.value)
            overlap_mask = (entries_ns <= end_ns) & (exits_ns >= start_ns)
            if not bool(overlap_mask.any()):
                continue
            positions = np.flatnonzero(overlap_mask)
            overlap_ns = np.minimum(exits_ns[positions], end_ns) - np.maximum(entries_ns[positions], start_ns)
            duration_ns = max(end_ns - start_ns, 1)
            capture = np.maximum(overlap_ns, 0).astype(float) / float(duration_ns)
            abs_return = float(finite_float(segment.get("abs_trend_return"), default=0.0) or 0.0)
            duration = float(finite_float(segment.get("duration_bars"), default=0.0) or 0.0)
            better = (
                (capture > best_capture[positions])
                | ((capture == best_capture[positions]) & (abs_return > best_return[positions]))
                | ((capture == best_capture[positions]) & (abs_return == best_return[positions]) & (duration > best_duration[positions]))
            )
            if not bool(better.any()):
                continue
            update_positions = positions[better]
            best_capture[update_positions] = capture[better]
            best_return[update_positions] = abs_return
            best_duration[update_positions] = duration
            for position in update_positions:
                best_rows[position] = segment
        for local_position, segment in enumerate(best_rows):
            if segment is None:
                continue
            selected.loc[indexes[local_position], "trend_segment_id"] = segment.get("trend_segment_id")
            selected.loc[indexes[local_position], "trend_direction"] = segment.get("direction")
            selected.loc[indexes[local_position], "segment_start_ts"] = pd.Timestamp(segment["start_ts"])
            selected.loc[indexes[local_position], "segment_end_ts"] = pd.Timestamp(segment["end_ts"])
            selected.loc[indexes[local_position], "segment_abs_trend_return"] = finite_float(segment.get("abs_trend_return"), default=0.0)
            selected.loc[indexes[local_position], "segment_duration_bars"] = finite_float(segment.get("duration_bars"), default=0.0)
    return selected


def selected_segment_from_row(row: pd.Series | None) -> pd.Series | None:
    """Convert a selected-segment row into the shape expected by diagnostics."""

    if row is None or pd.isna(row.get("trend_segment_id")):
        return None
    return pd.Series(
        {
            "trend_segment_id": row.get("trend_segment_id"),
            "direction": row.get("trend_direction"),
            "start_ts": row.get("segment_start_ts"),
            "end_ts": row.get("segment_end_ts"),
            "abs_trend_return": row.get("segment_abs_trend_return"),
            "duration_bars": row.get("segment_duration_bars"),
        }
    )


def compute_trade_diagnostic(
    trade: pd.Series,
    exit_time: pd.Timestamp,
    segments_by_symbol: dict[str, pd.DataFrame],
    primary_timeframe: str,
    *,
    selected_segment: pd.Series | None = None,
) -> dict[str, Any]:
    """Compute trend capture diagnostics for a trade and exit time."""

    symbol = str(trade.get("symbol") or "")
    entry = pd.Timestamp(trade["entry_ts"])
    selected = selected_segment if selected_segment is not None else find_selected_segment(symbol, entry, exit_time, segments_by_symbol)
    segment_id = None
    trend_direction = None
    start: pd.Timestamp | None = None
    end: pd.Timestamp | None = None
    captured = 0.0
    entry_lag_bars = None
    exit_lag_bars = None
    missed_after_exit = 0.0
    if selected is not None:
        segment_id = selected.get("trend_segment_id")
        trend_direction = selected.get("direction")
        start = pd.Timestamp(selected["start_ts"])
        end = pd.Timestamp(selected["end_ts"])
        captured = overlap_fraction(entry, exit_time, start, end)
        minutes = TIMEFRAME_MINUTES.get(primary_timeframe, 1)
        entry_lag_bars = float((entry - start).total_seconds() / 60.0 / minutes)
        exit_lag_bars = float((exit_time - end).total_seconds() / 60.0 / minutes)
        if exit_time < end:
            missed_after_exit = float((end - max(exit_time, start)).total_seconds() / max((end - start).total_seconds(), 1.0))
            missed_after_exit = min(max(missed_after_exit, 0.0), 1.0)
    entry_phase = classify_entry_phase(entry, start, end)
    exit_phase = classify_exit_phase(exit_time, start, end)
    entered = bool(selected is not None and captured > 0)
    early_exit = bool(entered and captured < 0.50)
    late_entry = entry_phase in {"middle_trend", "late_trend"}
    return {
        "strategy_source": trade.get("strategy_source"),
        "policy_or_group": trade.get("policy_or_group"),
        "symbol": symbol,
        "trade_timeframe": trade.get("trade_timeframe"),
        "primary_timeframe": primary_timeframe,
        "trade_id": trade.get("trade_id"),
        "split": trade.get("split"),
        "direction": trade.get("direction"),
        "entry_time": format_timestamp(entry),
        "exit_time": format_timestamp(exit_time),
        "trend_segment_id": segment_id,
        "trend_direction": trend_direction,
        "entered_trend_segment": entered,
        "entry_phase": entry_phase,
        "exit_phase": exit_phase,
        "captured_fraction_of_segment": captured,
        "missed_fraction_after_exit": missed_after_exit,
        "entry_lag_bars": entry_lag_bars,
        "exit_lag_bars": exit_lag_bars,
        "early_exit_flag": early_exit,
        "late_entry_flag": bool(late_entry),
        "no_cost_pnl": trade.get("no_cost_pnl"),
        "cost_aware_pnl": trade.get("cost_aware_pnl"),
    }


def build_diagnostics(
    trades: pd.DataFrame,
    segments_by_symbol: dict[str, pd.DataFrame],
    primary_timeframe: str,
    *,
    selected_segments: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build original-exit trend capture diagnostics for all legacy trades."""

    if trades.empty:
        return pd.DataFrame(columns=DIAGNOSTIC_COLUMNS)
    selected = selected_segments if selected_segments is not None else select_segments_for_trades(trades, segments_by_symbol)
    rows = [
        compute_trade_diagnostic(
            row,
            pd.Timestamp(row["exit_ts"]),
            segments_by_symbol,
            primary_timeframe,
            selected_segment=selected_segment_from_row(selected.loc[index]) if index in selected.index else None,
        )
        for index, row in trades.iterrows()
    ]
    return pd.DataFrame(rows, columns=DIAGNOSTIC_COLUMNS)


def true_range(frame: pd.DataFrame) -> pd.Series:
    """Compute true range."""

    high = pd.to_numeric(frame["high"], errors="coerce")
    low = pd.to_numeric(frame["low"], errors="coerce")
    close = pd.to_numeric(frame["close"], errors="coerce")
    previous_close = close.shift(1)
    ranges = pd.concat([(high - low), (high - previous_close).abs(), (low - previous_close).abs()], axis=1)
    return ranges.max(axis=1)


def add_atr(frame: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """Add ATR to a closed-bar frame."""

    result = frame.copy()
    if result.empty:
        result["atr14"] = pd.Series(dtype=float)
        return result
    result["atr14"] = true_range(result).rolling(window, min_periods=1).mean()
    return result


def add_time_ns(frame: pd.DataFrame) -> pd.DataFrame:
    """Attach nanosecond timestamps for fast searchsorted calls."""

    result = frame.copy()
    if result.empty or "datetime" not in result.columns:
        result["_time_ns"] = pd.Series(dtype=np.int64)
        return result
    result["_time_ns"] = pd.to_datetime(result["datetime"]).map(lambda value: pd.Timestamp(value).value).to_numpy(dtype=np.int64)
    return result


def normalize_1m_bars(frame: pd.DataFrame, source_timezone: str, target_timezone: str) -> pd.DataFrame:
    """Normalize 1m OHLCV bars."""

    columns = ["datetime", "open", "high", "low", "close", "volume"]
    if frame.empty:
        return pd.DataFrame(columns=columns)
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise TrendCaptureExitResearchError(f"1m bars missing columns: {missing}")
    normalized = frame.loc[:, columns].copy()
    timestamps = pd.to_datetime(normalized["datetime"], errors="coerce")
    if timestamps.isna().any():
        raise TrendCaptureExitResearchError("1m bars contain unparsable datetime values")
    if timestamps.dt.tz is None:
        timestamps = timestamps.dt.tz_localize(source_timezone)
    timestamps = timestamps.dt.tz_convert(target_timezone)
    normalized["datetime"] = timestamps
    for column in ["open", "high", "low", "close", "volume"]:
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce")
    return normalized.dropna(subset=columns).sort_values("datetime", kind="stable").reset_index(drop=True)


def load_1m_bars_from_sqlite(vt_symbol: str, database_path: Path, timezone_name: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Load local vn.py sqlite 1m bars for one symbol."""

    if not database_path.exists():
        raise TrendCaptureExitResearchError(f"database not found: {database_path}")
    symbol, exchange = split_vt_symbol(vt_symbol)
    query_start = start.tz_convert(timezone_name).tz_localize(None).isoformat(sep=" ", timespec="seconds")
    query_end = end.tz_convert(timezone_name).tz_localize(None).isoformat(sep=" ", timespec="seconds")
    sql = """
        select datetime,
               open_price as open,
               high_price as high,
               low_price as low,
               close_price as close,
               volume
          from dbbardata
         where symbol = ?
           and exchange = ?
           and interval = ?
           and datetime >= ?
           and datetime <= ?
         order by datetime
    """
    with sqlite3.connect(database_path) as connection:
        frame = pd.read_sql_query(sql, connection, params=(symbol, exchange, "1m", query_start, query_end))
    return normalize_1m_bars(frame, timezone_name, timezone_name)


def resample_ohlcv_closed(bars_1m: pd.DataFrame, timeframe: str, anchor: pd.Timestamp | None = None) -> pd.DataFrame:
    """Resample 1m OHLCV into completed closed bars timestamped at final minute."""

    if timeframe not in TIMEFRAME_MINUTES:
        raise TrendCaptureExitResearchError(f"unsupported timeframe: {timeframe}")
    minutes = TIMEFRAME_MINUTES[timeframe]
    if minutes == 1:
        result = bars_1m.copy().sort_values("datetime", kind="stable").reset_index(drop=True)
        return add_time_ns(add_atr(result))
    columns = ["open_time", "datetime", "open", "high", "low", "close", "volume"]
    if bars_1m.empty:
        return pd.DataFrame(columns=columns + ["atr14"])
    working = bars_1m.sort_values("datetime", kind="stable").drop_duplicates("datetime", keep="last").copy()
    anchor_ts = pd.Timestamp(anchor if anchor is not None else working["datetime"].iloc[0])
    if anchor_ts.tzinfo is None:
        anchor_ts = anchor_ts.tz_localize(working["datetime"].iloc[0].tz)
    deltas = (working["datetime"] - anchor_ts) / pd.Timedelta(minutes=1)
    working = working.loc[deltas >= 0].copy()
    working["_slot"] = np.floor(deltas.loc[working.index].to_numpy(dtype=float) / minutes).astype(np.int64)
    grouped = working.groupby("_slot", sort=True, dropna=False)
    result = grouped.agg(
        open_time=("datetime", "min"),
        datetime=("datetime", "max"),
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
        minute_count=("datetime", "size"),
    )
    result = result[result["minute_count"] == minutes].copy()
    result = result.drop(columns=["minute_count"]).dropna(subset=["open", "high", "low", "close"]).reset_index(drop=True)
    return add_time_ns(add_atr(result.loc[:, columns]))


def load_primary_bars(
    symbol: str,
    trades: pd.DataFrame,
    primary_timeframe: str,
    timezone_name: str,
    database_path: Path,
    *,
    bars_by_symbol: dict[str, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """Load and resample bars needed to simulate exits for one symbol."""

    if bars_by_symbol and symbol in bars_by_symbol:
        bars_1m = normalize_1m_bars(bars_by_symbol[symbol], timezone_name, timezone_name)
    else:
        start = pd.Timestamp(trades["entry_ts"].min()) - pd.Timedelta(days=30)
        end = pd.Timestamp(trades["entry_ts"].max()) + pd.Timedelta(days=90)
        bars_1m = load_1m_bars_from_sqlite(symbol, database_path, timezone_name, start, end)
    anchor = pd.Timestamp("2023-01-01T00:00:00", tz=timezone_name)
    return resample_ohlcv_closed(bars_1m, primary_timeframe, anchor=anchor)


def first_bar_at_or_after(bars: pd.DataFrame, target_time: pd.Timestamp) -> pd.Series | None:
    """Return first bar with datetime >= target_time, or the last bar if target is later."""

    if bars.empty:
        return None
    times = bars["_time_ns"].to_numpy(dtype=np.int64) if "_time_ns" in bars.columns else pd.to_datetime(bars["datetime"]).map(lambda value: pd.Timestamp(value).value).to_numpy(dtype=np.int64)
    position = int(np.searchsorted(times, pd.Timestamp(target_time).value, side="left"))
    if position >= len(bars.index):
        position = len(bars.index) - 1
    return bars.iloc[position]


def last_completed_bar_index(bars: pd.DataFrame, target_time: pd.Timestamp) -> int | None:
    """Return index of latest closed bar at or before target time."""

    if bars.empty:
        return None
    times = bars["_time_ns"].to_numpy(dtype=np.int64) if "_time_ns" in bars.columns else pd.to_datetime(bars["datetime"]).map(lambda value: pd.Timestamp(value).value).to_numpy(dtype=np.int64)
    position = int(np.searchsorted(times, pd.Timestamp(target_time).value, side="right") - 1)
    if position < 0:
        return None
    return min(position, len(bars.index) - 1)


def entry_atr(trade: pd.Series, bars: pd.DataFrame, entry_index: int | None) -> float:
    """Return ATR available at entry time."""

    if entry_index is not None and entry_index >= 0 and "atr14" in bars.columns:
        value = finite_float(bars.iloc[entry_index].get("atr14"), default=np.nan)
        if value is not None and np.isfinite(value) and value > 0:
            return float(value)
    fallback = finite_float(trade.get("entry_atr"), default=np.nan)
    if fallback is not None and np.isfinite(fallback) and fallback > 0:
        return float(fallback)
    return max(float(trade.get("entry_price") or 0.0) * 0.005, 1e-12)


def simulate_original_exit(trade: pd.Series) -> ExitSimulation:
    """Return the original legacy exit."""

    return ExitSimulation(
        exit_time=pd.Timestamp(trade["exit_ts"]),
        exit_price=float(trade["exit_price"]),
        exit_reason="original_exit",
    )


def simulate_fixed_hold_exit(trade: pd.Series, bars: pd.DataFrame, multiplier: float, max_days: int) -> ExitSimulation:
    """Return a fixed longer hold counterfactual exit."""

    entry = pd.Timestamp(trade["entry_ts"])
    original_minutes = finite_float(trade.get("holding_minutes"), default=0.0) or 0.0
    hold_minutes = min(float(original_minutes) * multiplier, float(max_days * 1440))
    target_time = entry + pd.Timedelta(minutes=hold_minutes)
    bar = first_bar_at_or_after(bars, target_time)
    if bar is None:
        return simulate_original_exit(trade)
    return ExitSimulation(
        exit_time=pd.Timestamp(bar["datetime"]),
        exit_price=float(bar["close"]),
        exit_reason=f"fixed_hold_{multiplier:g}x_cap_{max_days}d",
    )


def simulate_atr_chandelier_exit(
    trade: pd.Series,
    bars: pd.DataFrame,
    atr_multiplier: float,
    *,
    return_debug_path: bool = False,
    max_exit_time: pd.Timestamp | None = None,
) -> ExitSimulation:
    """Simulate an ATR chandelier exit using closed bars only."""

    if bars.empty:
        return simulate_original_exit(trade)
    entry = pd.Timestamp(trade["entry_ts"])
    entry_idx = last_completed_bar_index(bars, entry)
    atr = entry_atr(trade, bars, entry_idx)
    start_pos = 0 if entry_idx is None else min(entry_idx + 1, len(bars.index) - 1)
    end_pos = len(bars.index) - 1
    if max_exit_time is not None:
        cap_pos = last_completed_bar_index(bars, max_exit_time)
        if cap_pos is not None:
            end_pos = max(start_pos, min(end_pos, cap_pos))
    direction = str(trade.get("direction") or "").lower()
    entry_price = float(trade["entry_price"])
    closes = bars["close"].to_numpy(dtype=float, copy=False)
    stop_path: list[float] = []
    if direction == "short":
        lowest_close = entry_price
        trailing_stop = lowest_close + atr_multiplier * atr
        for position in range(start_pos, end_pos + 1):
            close = float(closes[position])
            lowest_close = min(lowest_close, close)
            trailing_stop = min(trailing_stop, lowest_close + atr_multiplier * atr)
            stop_path.append(float(trailing_stop))
            if close >= trailing_stop:
                return ExitSimulation(pd.Timestamp(bars["datetime"].iloc[position]), close, f"atr_chandelier_{atr_multiplier:g}x", stop_path if return_debug_path else None)
    else:
        highest_close = entry_price
        trailing_stop = highest_close - atr_multiplier * atr
        for position in range(start_pos, end_pos + 1):
            close = float(closes[position])
            highest_close = max(highest_close, close)
            trailing_stop = max(trailing_stop, highest_close - atr_multiplier * atr)
            stop_path.append(float(trailing_stop))
            if close <= trailing_stop:
                return ExitSimulation(pd.Timestamp(bars["datetime"].iloc[position]), close, f"atr_chandelier_{atr_multiplier:g}x", stop_path if return_debug_path else None)
    last = bars.iloc[end_pos]
    suffix = "cap" if end_pos < len(bars.index) - 1 else "data_end"
    return ExitSimulation(pd.Timestamp(last["datetime"]), float(last["close"]), f"atr_chandelier_{atr_multiplier:g}x_{suffix}", stop_path if return_debug_path else None)


def simulate_swing_trailing_exit(
    trade: pd.Series,
    bars: pd.DataFrame,
    swing_bars: int = 10,
    *,
    max_exit_time: pd.Timestamp | None = None,
) -> ExitSimulation:
    """Simulate a previous-swing trailing exit."""

    if bars.empty:
        return simulate_original_exit(trade)
    entry = pd.Timestamp(trade["entry_ts"])
    entry_idx = last_completed_bar_index(bars, entry)
    start_pos = 0 if entry_idx is None else min(entry_idx + 1, len(bars.index) - 1)
    start_pos = max(start_pos, swing_bars)
    end_pos = len(bars.index) - 1
    if max_exit_time is not None:
        cap_pos = last_completed_bar_index(bars, max_exit_time)
        if cap_pos is not None:
            end_pos = max(start_pos, min(end_pos, cap_pos))
    direction = str(trade.get("direction") or "").lower()
    lows = bars["low"].to_numpy(dtype=float, copy=False)
    highs = bars["high"].to_numpy(dtype=float, copy=False)
    closes = bars["close"].to_numpy(dtype=float, copy=False)
    for position in range(start_pos, end_pos + 1):
        if direction == "short":
            previous_high = float(np.nanmax(highs[position - swing_bars : position]))
            if np.isfinite(previous_high) and highs[position] > previous_high:
                return ExitSimulation(pd.Timestamp(bars.iloc[position]["datetime"]), float(closes[position]), f"swing_trailing_N{swing_bars}")
        else:
            previous_low = float(np.nanmin(lows[position - swing_bars : position]))
            if np.isfinite(previous_low) and lows[position] < previous_low:
                return ExitSimulation(pd.Timestamp(bars.iloc[position]["datetime"]), float(closes[position]), f"swing_trailing_N{swing_bars}")
    last = bars.iloc[end_pos]
    suffix = "cap" if end_pos < len(bars.index) - 1 else "data_end"
    return ExitSimulation(pd.Timestamp(last["datetime"]), float(last["close"]), f"swing_trailing_N{swing_bars}_{suffix}")


def simulate_time_stop_if_no_progress(trade: pd.Series, bars: pd.DataFrame, k_bars: int = 5, atr_move: float = 0.5) -> ExitSimulation:
    """Exit after K bars if price has not made a favorable ATR move."""

    if bars.empty:
        return simulate_original_exit(trade)
    entry = pd.Timestamp(trade["entry_ts"])
    entry_idx = last_completed_bar_index(bars, entry)
    atr = entry_atr(trade, bars, entry_idx)
    start_pos = 0 if entry_idx is None else min(entry_idx + 1, len(bars.index) - 1)
    check_pos = min(start_pos + max(k_bars - 1, 0), len(bars.index) - 1)
    window = bars.iloc[start_pos : check_pos + 1]
    direction = str(trade.get("direction") or "").lower()
    entry_price = float(trade["entry_price"])
    if direction == "short":
        favorable = entry_price - float(pd.to_numeric(window["low"], errors="coerce").min())
    else:
        favorable = float(pd.to_numeric(window["high"], errors="coerce").max()) - entry_price
    if favorable < atr_move * atr:
        row = bars.iloc[check_pos]
        return ExitSimulation(pd.Timestamp(row["datetime"]), float(row["close"]), f"time_stop_no_progress_K{k_bars}_{atr_move:g}atr")
    return simulate_original_exit(trade)


def simulate_oracle_hold_to_trend_end(
    trade: pd.Series,
    bars: pd.DataFrame,
    segments_by_symbol: dict[str, pd.DataFrame],
    selected_segment: pd.Series | None = None,
) -> ExitSimulation:
    """Oracle upper-bound exit at the selected trend segment end."""

    original = simulate_original_exit(trade)
    selected = selected_segment
    if selected is None:
        selected = find_selected_segment(str(trade.get("symbol") or ""), pd.Timestamp(trade["entry_ts"]), pd.Timestamp(trade["exit_ts"]), segments_by_symbol)
    if selected is None:
        return ExitSimulation(original.exit_time, original.exit_price, "oracle_no_segment_original_exit")
    segment_end = pd.Timestamp(selected["end_ts"])
    if segment_end <= pd.Timestamp(trade["entry_ts"]):
        return ExitSimulation(original.exit_time, original.exit_price, "oracle_segment_before_entry_original_exit")
    bar = first_bar_at_or_after(bars, segment_end)
    if bar is None:
        return ExitSimulation(original.exit_time, original.exit_price, "oracle_missing_bars_original_exit")
    return ExitSimulation(pd.Timestamp(bar["datetime"]), float(bar["close"]), "oracle_hold_to_trend_end")


def simulate_exit_variant(
    variant: str,
    trade: pd.Series,
    bars: pd.DataFrame,
    segments_by_symbol: dict[str, pd.DataFrame],
    selected_segment: pd.Series | None = None,
) -> ExitSimulation:
    """Dispatch one exit variant simulation."""

    if variant == "original_exit":
        return simulate_original_exit(trade)
    if variant == "fixed_hold_longer_2x":
        return simulate_fixed_hold_exit(trade, bars, 2.0, 10)
    if variant == "fixed_hold_longer_4x":
        return simulate_fixed_hold_exit(trade, bars, 4.0, 20)
    if variant == "atr_chandelier_3x":
        return simulate_atr_chandelier_exit(trade, bars, 3.0, max_exit_time=pd.Timestamp(trade["entry_ts"]) + pd.Timedelta(days=TRAILING_EXIT_MAX_DAYS))
    if variant == "atr_chandelier_5x":
        return simulate_atr_chandelier_exit(trade, bars, 5.0, max_exit_time=pd.Timestamp(trade["entry_ts"]) + pd.Timedelta(days=TRAILING_EXIT_MAX_DAYS))
    if variant == "swing_trailing_exit":
        return simulate_swing_trailing_exit(trade, bars, swing_bars=10, max_exit_time=pd.Timestamp(trade["entry_ts"]) + pd.Timedelta(days=TRAILING_EXIT_MAX_DAYS))
    if variant == "time_stop_if_no_progress":
        return simulate_time_stop_if_no_progress(trade, bars, k_bars=5, atr_move=0.5)
    if variant == ORACLE_VARIANT:
        return simulate_oracle_hold_to_trend_end(trade, bars, segments_by_symbol, selected_segment=selected_segment)
    raise TrendCaptureExitResearchError(f"unsupported exit variant: {variant}")


def load_funding_csv(path: Path) -> pd.DataFrame:
    """Load one OKX funding CSV with normalized UTC timestamps."""

    frame = pd.read_csv(path)
    if frame.empty:
        return pd.DataFrame(columns=["funding_time_utc", "funding_rate"])
    if "funding_time_utc" in frame.columns:
        frame["funding_time_utc"] = pd.to_datetime(frame["funding_time_utc"], utc=True, errors="coerce")
    elif "funding_time" in frame.columns:
        frame["funding_time_utc"] = pd.to_datetime(pd.to_numeric(frame["funding_time"], errors="coerce"), unit="ms", utc=True)
    else:
        raise TrendCaptureExitResearchError(f"funding CSV missing funding_time columns: {path}")
    frame["funding_rate"] = pd.to_numeric(frame.get("funding_rate"), errors="coerce")
    return frame.dropna(subset=["funding_time_utc", "funding_rate"]).sort_values("funding_time_utc", kind="stable").drop_duplicates("funding_time_utc", keep="last").reset_index(drop=True)


def load_funding_histories(funding_dir: Path, inst_ids: list[str], warnings: list[str]) -> tuple[dict[str, pd.DataFrame], list[str]]:
    """Load funding histories for requested instruments."""

    histories: dict[str, pd.DataFrame] = {}
    missing: list[str] = []
    for inst_id in inst_ids:
        matches = sorted(funding_dir.glob(f"{inst_id}_funding_*.csv"))
        if not matches:
            missing.append(inst_id)
            warnings.append(f"missing_funding_csv:{inst_id}")
            continue
        try:
            histories[inst_id] = load_funding_csv(matches[-1])
        except Exception as exc:
            missing.append(inst_id)
            warnings.append(f"failed_to_read_funding_csv:{inst_id}:{exc}")
    return histories, missing


def build_funding_indexes(funding_histories: dict[str, pd.DataFrame]) -> dict[str, FundingIndex]:
    """Build prefix-sum funding indexes for fast interval calculations."""

    indexes: dict[str, FundingIndex] = {}
    for inst_id, funding in funding_histories.items():
        if funding.empty:
            continue
        times = pd.to_datetime(funding["funding_time_utc"], utc=True, errors="coerce")
        rates = pd.to_numeric(funding["funding_rate"], errors="coerce")
        working = pd.DataFrame({"time": times, "rate": rates}).dropna().sort_values("time", kind="stable")
        if working.empty:
            continue
        times_ns = working["time"].map(lambda value: pd.Timestamp(value).value).to_numpy(dtype=np.int64)
        rate_values = working["rate"].to_numpy(dtype=float)
        indexes[inst_id] = FundingIndex(
            times_ns=times_ns,
            prefix_rates=np.concatenate(([0.0], np.cumsum(rate_values, dtype=float))),
            first_time=pd.Timestamp(working["time"].iloc[0]),
            last_time=pd.Timestamp(working["time"].iloc[-1]),
        )
    return indexes


def signed_funding_pnl(notional: float, funding_rate: float, direction: str) -> float:
    """Return funding PnL using OKX sign convention research assumption."""

    if str(direction or "").lower() == "short":
        return float(abs(notional) * funding_rate)
    return float(-abs(notional) * funding_rate)


def funding_for_trade(
    inst_id: str,
    direction: str,
    entry_time: pd.Timestamp,
    exit_time: pd.Timestamp,
    notional: float,
    funding_histories: dict[str, Any],
) -> dict[str, Any]:
    """Compute actual funding adjustment for a trade interval."""

    funding = funding_histories.get(inst_id)
    if funding is None or (hasattr(funding, "empty") and funding.empty):
        return {
            "funding_pnl": 0.0,
            "funding_events_count": 0,
            "funding_data_available": False,
            "funding_interval_covered": False,
        }
    entry_utc = timestamp_to_utc(entry_time)
    exit_utc = timestamp_to_utc(exit_time)
    if exit_utc < entry_utc:
        entry_utc, exit_utc = exit_utc, entry_utc
    tolerance = pd.Timedelta(hours=8)
    if isinstance(funding, FundingIndex):
        interval_covered = bool(entry_utc >= funding.first_time and exit_utc <= funding.last_time + tolerance)
        left = int(np.searchsorted(funding.times_ns, entry_utc.value, side="left"))
        right = int(np.searchsorted(funding.times_ns, exit_utc.value, side="right"))
        rate_sum = float(funding.prefix_rates[right] - funding.prefix_rates[left])
        event_count = int(max(right - left, 0))
        funding_pnl = signed_funding_pnl(notional, rate_sum, direction)
    else:
        first_time = pd.Timestamp(funding["funding_time_utc"].min())
        last_time = pd.Timestamp(funding["funding_time_utc"].max())
        interval_covered = bool(entry_utc >= first_time and exit_utc <= last_time + tolerance)
        mask = (funding["funding_time_utc"] >= entry_utc) & (funding["funding_time_utc"] <= exit_utc)
        events = funding.loc[mask].copy()
        rates = pd.to_numeric(events.get("funding_rate"), errors="coerce").dropna()
        funding_pnl = float(sum(signed_funding_pnl(notional, float(rate), direction) for rate in rates))
        event_count = int(len(rates.index))
    return {
        "funding_pnl": funding_pnl,
        "funding_events_count": event_count,
        "funding_data_available": True,
        "funding_interval_covered": interval_covered,
    }


def build_exit_variant_trades(
    trades: pd.DataFrame,
    segments_by_symbol: dict[str, pd.DataFrame],
    primary_timeframe: str,
    timezone_name: str,
    funding_histories: dict[str, Any],
    database_path: Path,
    *,
    bars_by_symbol: dict[str, pd.DataFrame] | None = None,
    selected_segments: pd.DataFrame | None = None,
    logger: logging.Logger | None = None,
) -> pd.DataFrame:
    """Build all counterfactual exit-variant trade rows."""

    if trades.empty:
        return pd.DataFrame(columns=EXIT_TRADE_COLUMNS)
    rows: list[dict[str, Any]] = []
    selected_segments = selected_segments if selected_segments is not None else select_segments_for_trades(trades, segments_by_symbol)
    for symbol, symbol_trades in trades.groupby("symbol", dropna=False):
        simulation_cache: dict[tuple[Any, ...], ExitSimulation] = {}
        if logger is not None:
            log_event(
                logger,
                logging.INFO,
                "trend_capture_exit.symbol_start",
                "Simulating exit variants for symbol",
                symbol=str(symbol),
                trade_count=len(symbol_trades.index),
            )
        bars = load_primary_bars(str(symbol), symbol_trades, primary_timeframe, timezone_name, database_path, bars_by_symbol=bars_by_symbol)
        for trade_index, trade in symbol_trades.iterrows():
            original_exit = pd.Timestamp(trade["exit_ts"])
            original_holding = finite_float(trade.get("holding_minutes"), default=0.0) or 0.0
            selected_segment = selected_segment_from_row(selected_segments.loc[trade_index]) if trade_index in selected_segments.index else None
            for variant in EXIT_VARIANTS:
                selected_id = selected_segment.get("trend_segment_id") if selected_segment is not None else None
                cache_key = (
                    variant,
                    str(trade.get("direction") or ""),
                    int(pd.Timestamp(trade["entry_ts"]).value),
                    round(float(trade.get("entry_price") or 0.0), 12),
                    round(float(original_holding), 6) if variant.startswith("fixed_hold") else None,
                    int(original_exit.value) if variant in {"original_exit", "time_stop_if_no_progress", ORACLE_VARIANT} else None,
                    round(float(trade.get("exit_price") or 0.0), 12) if variant in {"original_exit", "time_stop_if_no_progress", ORACLE_VARIANT} else None,
                    str(selected_id) if variant == ORACLE_VARIANT else None,
                )
                simulation = simulation_cache.get(cache_key)
                if simulation is None:
                    simulation = simulate_exit_variant(variant, trade, bars, segments_by_symbol, selected_segment=selected_segment)
                    simulation_cache[cache_key] = simulation
                no_cost = directional_price_pnl(
                    str(trade.get("direction") or ""),
                    float(trade["entry_price"]),
                    float(simulation.exit_price),
                    float(trade.get("pnl_multiplier") or 0.0),
                )
                cost_drag = float(trade.get("cost_drag") or 0.0)
                cost_aware = no_cost - cost_drag
                inst_id = symbol_to_inst_id(str(symbol))
                funding = funding_for_trade(
                    inst_id,
                    str(trade.get("direction") or ""),
                    pd.Timestamp(trade["entry_ts"]),
                    simulation.exit_time,
                    float(trade.get("notional") or 0.0),
                    funding_histories,
                )
                funding_adjusted = cost_aware + float(funding["funding_pnl"])
                diagnostic = compute_trade_diagnostic(
                    trade,
                    simulation.exit_time,
                    segments_by_symbol,
                    primary_timeframe,
                    selected_segment=selected_segment,
                )
                holding_minutes = max((simulation.exit_time - pd.Timestamp(trade["entry_ts"])).total_seconds() / 60.0, 0.0)
                rows.append(
                    {
                        "exit_variant": variant,
                        "oracle": variant == ORACLE_VARIANT,
                        "strategy_source": trade.get("strategy_source"),
                        "policy_or_group": trade.get("policy_or_group"),
                        "symbol": symbol,
                        "inst_id": inst_id,
                        "trade_timeframe": trade.get("trade_timeframe"),
                        "primary_timeframe": primary_timeframe,
                        "trade_id": trade.get("trade_id"),
                        "split": trade.get("split"),
                        "direction": trade.get("direction"),
                        "entry_time": trade.get("entry_time"),
                        "entry_price": float(trade["entry_price"]),
                        "exit_time": format_timestamp(simulation.exit_time),
                        "exit_price": float(simulation.exit_price),
                        "holding_minutes": float(holding_minutes),
                        "original_exit_time": format_timestamp(original_exit),
                        "original_holding_minutes": float(original_holding),
                        "exit_reason": simulation.exit_reason,
                        "trend_segment_id": diagnostic["trend_segment_id"],
                        "entry_phase": diagnostic["entry_phase"],
                        "exit_phase": diagnostic["exit_phase"],
                        "captured_fraction_of_segment": diagnostic["captured_fraction_of_segment"],
                        "missed_fraction_after_exit": diagnostic["missed_fraction_after_exit"],
                        "entry_lag_bars": diagnostic["entry_lag_bars"],
                        "exit_lag_bars": diagnostic["exit_lag_bars"],
                        "early_exit_flag": diagnostic["early_exit_flag"],
                        "late_entry_flag": diagnostic["late_entry_flag"],
                        "no_cost_pnl": float(no_cost),
                        "cost_drag": float(cost_drag),
                        "cost_aware_pnl": float(cost_aware),
                        "funding_pnl": float(funding["funding_pnl"]),
                        "funding_adjusted_pnl": float(funding_adjusted),
                        "funding_events_count": int(funding["funding_events_count"]),
                        "funding_data_available": bool(funding["funding_data_available"]),
                        "funding_interval_covered": bool(funding["funding_interval_covered"]),
                        "notional": float(trade.get("notional") or 0.0),
                        "pnl_multiplier": float(trade.get("pnl_multiplier") or 0.0),
                    }
                )
        if logger is not None:
            log_event(
                logger,
                logging.INFO,
                "trend_capture_exit.symbol_done",
                "Finished exit variants for symbol",
                symbol=str(symbol),
                generated_rows=len(symbol_trades.index) * len(EXIT_VARIANTS),
            )
    return pd.DataFrame(rows, columns=EXIT_TRADE_COLUMNS)


def max_drawdown_from_pnl(pnl: pd.Series) -> float:
    """Return closed-trade max drawdown magnitude from cumulative PnL."""

    if pnl.empty:
        return 0.0
    equity = pd.to_numeric(pnl, errors="coerce").fillna(0.0).cumsum()
    peak = pd.concat([pd.Series([0.0]), equity.reset_index(drop=True)], ignore_index=True).cummax().iloc[1:].reset_index(drop=True)
    drawdown = peak - equity.reset_index(drop=True)
    return float(drawdown.max()) if not drawdown.empty else 0.0


def top_5pct_trade_pnl_contribution(trades: pd.DataFrame, pnl_column: str = "funding_adjusted_pnl") -> float | None:
    """Calculate top five percent trade PnL contribution."""

    if trades.empty or pnl_column not in trades.columns:
        return None
    pnl = pd.to_numeric(trades[pnl_column], errors="coerce").fillna(0.0)
    total = float(pnl.sum())
    if abs(total) <= 1e-12:
        return None
    top_n = max(1, int(math.ceil(len(pnl.index) * 0.05)))
    top_sum = float(pnl.sort_values(ascending=False).head(top_n).sum())
    return float(top_sum / total)


def largest_symbol_pnl_share(trades: pd.DataFrame, pnl_column: str = "funding_adjusted_pnl") -> float | None:
    """Calculate largest single-symbol positive PnL share."""

    if trades.empty or "symbol" not in trades.columns or pnl_column not in trades.columns:
        return None
    grouped = trades.groupby("symbol", dropna=False)[pnl_column].sum()
    positive = grouped[grouped > 0]
    if not positive.empty and positive.sum() > 0:
        return float(positive.max() / positive.sum())
    absolute = grouped.abs()
    if absolute.sum() > 0:
        return float(absolute.max() / absolute.sum())
    return None


def summarize_trade_slice(trades: pd.DataFrame) -> dict[str, Any]:
    """Summarize a trade slice with required exit research metrics."""

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
        }
    working = trades.copy().sort_values("exit_time", kind="stable")
    pnl = pd.to_numeric(working["funding_adjusted_pnl"], errors="coerce").fillna(0.0)
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    profit_factor = None
    if not losses.empty:
        profit_factor = float(wins.sum() / abs(losses.sum())) if not wins.empty else 0.0
    return {
        "trade_count": int(len(working.index)),
        "no_cost_pnl": float(pd.to_numeric(working["no_cost_pnl"], errors="coerce").fillna(0.0).sum()),
        "cost_aware_pnl": float(pd.to_numeric(working["cost_aware_pnl"], errors="coerce").fillna(0.0).sum()),
        "funding_adjusted_pnl": float(pnl.sum()),
        "funding_events_count": int(pd.to_numeric(working["funding_events_count"], errors="coerce").fillna(0).sum()),
        "funding_data_complete": bool(working["funding_data_available"].astype(bool).all() and working["funding_interval_covered"].astype(bool).all()),
        "win_rate": float((pnl > 0).mean()) if len(pnl.index) else None,
        "profit_factor": profit_factor,
        "max_drawdown": max_drawdown_from_pnl(pnl),
        "largest_symbol_pnl_share": largest_symbol_pnl_share(working),
        "top_5pct_trade_pnl_contribution": top_5pct_trade_pnl_contribution(working),
        "avg_captured_fraction": float(pd.to_numeric(working["captured_fraction_of_segment"], errors="coerce").fillna(0.0).mean()),
        "early_exit_share": float(working["early_exit_flag"].astype(bool).mean()),
        "late_entry_share": float(working["late_entry_flag"].astype(bool).mean()),
    }


def build_group_summary(trades: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    """Build summaries by exit variant and requested groups."""

    columns = group_columns + [
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
    ]
    if trades.empty:
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, Any]] = []
    for keys, group in trades.groupby(group_columns, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_columns, keys, strict=False))
        row.update(summarize_trade_slice(group))
        rows.append(row)
    return pd.DataFrame(rows, columns=columns).sort_values(group_columns, kind="stable").reset_index(drop=True)


def build_concentration_summary(exit_trades: pd.DataFrame) -> pd.DataFrame:
    """Build concentration metrics per variant and split."""

    columns = [
        "exit_variant",
        "oracle",
        "split",
        "trade_count",
        "largest_symbol_pnl_share",
        "top_5pct_trade_pnl_contribution",
        "top_5pct_trade_pnl",
        "top_5pct_trade_count",
    ]
    if exit_trades.empty:
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, Any]] = []
    for (variant, oracle, split), group in exit_trades.groupby(["exit_variant", "oracle", "split"], dropna=False):
        pnl = pd.to_numeric(group["funding_adjusted_pnl"], errors="coerce").fillna(0.0)
        top_count = max(1, int(math.ceil(len(pnl.index) * 0.05))) if len(pnl.index) else 0
        top_pnl = float(pnl.sort_values(ascending=False).head(top_count).sum()) if top_count else 0.0
        rows.append(
            {
                "exit_variant": variant,
                "oracle": bool(oracle),
                "split": split,
                "trade_count": int(len(group.index)),
                "largest_symbol_pnl_share": largest_symbol_pnl_share(group),
                "top_5pct_trade_pnl_contribution": top_5pct_trade_pnl_contribution(group),
                "top_5pct_trade_pnl": top_pnl,
                "top_5pct_trade_count": int(top_count),
            }
        )
    return pd.DataFrame(rows, columns=columns).sort_values(["exit_variant", "split"], kind="stable").reset_index(drop=True)


def original_variant_metrics(summary_by_variant: pd.DataFrame) -> dict[str, float]:
    """Return original_exit benchmark metrics."""

    row = summary_by_variant[summary_by_variant["exit_variant"] == "original_exit"]
    if row.empty:
        return {"avg_captured_fraction": 0.0, "early_exit_share": 1.0}
    first = row.iloc[0]
    return {
        "avg_captured_fraction": float(finite_float(first.get("avg_captured_fraction"), default=0.0) or 0.0),
        "early_exit_share": float(finite_float(first.get("early_exit_share"), default=1.0) or 1.0),
    }


def materially_higher_capture(value: float | None, original: float) -> bool:
    """Return whether capture improvement is material for this research gate."""

    if value is None or not np.isfinite(value):
        return False
    return bool(value >= original + CAPTURE_IMPROVEMENT_MIN_ABS and value >= original * CAPTURE_IMPROVEMENT_MIN_REL)


def materially_lower_early_exit(value: float | None, original: float) -> bool:
    """Return whether early-exit reduction is material for this research gate."""

    if value is None or not np.isfinite(value):
        return False
    return bool(value <= original - EARLY_EXIT_REDUCTION_MIN_ABS or value <= original * EARLY_EXIT_REDUCTION_MIN_REL)


def evaluate_stable_like_gates(
    by_split: pd.DataFrame,
    by_variant: pd.DataFrame,
    funding_data_complete: bool,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Evaluate research-only stable-like exit gates."""

    original = original_variant_metrics(by_variant)
    rows: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    if by_variant.empty:
        return pd.DataFrame(), []
    for _, variant_row in by_variant.sort_values("exit_variant", kind="stable").iterrows():
        variant = str(variant_row["exit_variant"])
        oracle = bool(variant_row["oracle"])
        reasons: list[str] = []
        if oracle:
            reasons.append("oracle_variant_excluded_from_stable_gate")
        split_rows = {
            str(row["split"]): row
            for _, row in by_split[(by_split["exit_variant"] == variant) & (by_split["oracle"] == oracle)].iterrows()
        }
        for split in SPLITS:
            row = split_rows.get(split)
            if row is None:
                reasons.append(f"{split}:missing")
                continue
            if finite_float(row.get("no_cost_pnl"), default=np.nan) is None or float(row.get("no_cost_pnl")) <= 0:
                reasons.append(f"{split}:no_cost_pnl_not_positive")
            if int(row.get("trade_count") or 0) < 10:
                reasons.append(f"{split}:trade_count_lt_10")
        oos = split_rows.get("oos_ext")
        if oos is None:
            reasons.append("oos_ext:missing_for_cost_and_concentration")
        else:
            if float(finite_float(oos.get("cost_aware_pnl"), default=np.nan) or np.nan) < 0:
                reasons.append("oos_ext:cost_aware_pnl_negative")
            if float(finite_float(oos.get("funding_adjusted_pnl"), default=np.nan) or np.nan) < 0:
                reasons.append("oos_ext:funding_adjusted_pnl_negative")
            largest_share = finite_float(oos.get("largest_symbol_pnl_share"), default=np.nan)
            top_share = finite_float(oos.get("top_5pct_trade_pnl_contribution"), default=np.nan)
            if largest_share is None or not np.isfinite(largest_share) or largest_share > 0.7:
                reasons.append("oos_ext:largest_symbol_pnl_share_gt_0.7")
            if top_share is None or not np.isfinite(top_share) or top_share > 0.8:
                reasons.append("oos_ext:top_5pct_trade_pnl_contribution_gt_0.8")
        avg_capture = finite_float(variant_row.get("avg_captured_fraction"), default=np.nan)
        early_exit = finite_float(variant_row.get("early_exit_share"), default=np.nan)
        if not materially_higher_capture(avg_capture, original["avg_captured_fraction"]):
            reasons.append("avg_captured_fraction_not_materially_higher_than_original")
        if not materially_lower_early_exit(early_exit, original["early_exit_share"]):
            reasons.append("early_exit_share_not_materially_lower_than_original")
        if not funding_data_complete:
            reasons.append("funding_data_incomplete")
        stable_like = bool(not reasons)
        row = {
            "exit_variant": variant,
            "oracle": oracle,
            "stable_like": stable_like,
            "rejected_reasons": ";".join(reasons),
            "avg_captured_fraction": avg_capture,
            "early_exit_share": early_exit,
            "original_avg_captured_fraction": original["avg_captured_fraction"],
            "original_early_exit_share": original["early_exit_share"],
        }
        rows.append(row)
        if stable_like:
            candidates.append(row)
    rejected = pd.DataFrame(rows)
    return rejected, candidates


def summarize_diagnostics(diagnostics: pd.DataFrame, trend_map_summary: dict[str, Any]) -> dict[str, Any]:
    """Summarize original legacy trend capture diagnostics."""

    if diagnostics.empty:
        return {
            "legacy_trade_count": 0,
            "legacy_entered_trend_share": None,
            "avg_captured_fraction": None,
            "early_exit_share": None,
            "late_entry_share": None,
            "entered_middle_or_late": None,
            "main_failure_mode": "legacy_trade_files_missing",
            "worst_policy_symbol_timeframe": [],
        }
    entered_share = float(diagnostics["entered_trend_segment"].astype(bool).mean())
    avg_capture = float(pd.to_numeric(diagnostics["captured_fraction_of_segment"], errors="coerce").fillna(0.0).mean())
    early_exit_share = float(diagnostics["early_exit_flag"].astype(bool).mean())
    late_entry_share = float(diagnostics["late_entry_flag"].astype(bool).mean())
    if late_entry_share >= 0.50:
        main_failure = "entered_middle_or_late"
    elif early_exit_share >= 0.50:
        main_failure = "exited_too_early"
    elif avg_capture < 0.35:
        main_failure = "low_trend_capture"
    else:
        main_failure = "mixed_or_not_primary_failure"
    grouped = (
        diagnostics.groupby(["strategy_source", "policy_or_group", "symbol", "trade_timeframe"], dropna=False)
        .agg(
            trade_count=("trade_id", "size"),
            avg_captured_fraction=("captured_fraction_of_segment", "mean"),
            early_exit_share=("early_exit_flag", "mean"),
            late_entry_share=("late_entry_flag", "mean"),
        )
        .reset_index()
    )
    worst = grouped.sort_values(["avg_captured_fraction", "early_exit_share", "trade_count"], ascending=[True, False, False], kind="stable").head(20)
    tom_legacy = trend_map_summary.get("legacy_analysis") or {}
    return {
        "legacy_trade_count": int(len(diagnostics.index)),
        "legacy_entered_trend_share": entered_share,
        "avg_captured_fraction": avg_capture,
        "early_exit_share": early_exit_share,
        "late_entry_share": late_entry_share,
        "entered_middle_or_late": bool(late_entry_share >= 0.50),
        "main_failure_mode": main_failure,
        "trend_map_early_exit_share": tom_legacy.get("early_exit_share"),
        "trend_map_avg_captured_fraction": tom_legacy.get("avg_captured_fraction"),
        "trend_map_main_failure_mode": tom_legacy.get("main_failure_mode"),
        "early_exit_share_close_to_trend_map": (
            None
            if tom_legacy.get("early_exit_share") is None
            else bool(abs(float(tom_legacy.get("early_exit_share")) - early_exit_share) <= 0.10)
        ),
        "worst_policy_symbol_timeframe": dataframe_records(worst),
    }


def best_improving_variant(by_variant: pd.DataFrame) -> dict[str, Any] | None:
    """Return the non-oracle variant with the largest capture improvement."""

    if by_variant.empty:
        return None
    original = original_variant_metrics(by_variant)
    candidates = by_variant[(by_variant["oracle"] == False) & (by_variant["exit_variant"] != "original_exit")].copy()  # noqa: E712
    if candidates.empty:
        return None
    candidates["capture_improvement"] = pd.to_numeric(candidates["avg_captured_fraction"], errors="coerce") - original["avg_captured_fraction"]
    candidates["early_exit_reduction"] = original["early_exit_share"] - pd.to_numeric(candidates["early_exit_share"], errors="coerce")
    candidates = candidates.sort_values(["capture_improvement", "early_exit_reduction"], ascending=[False, False], kind="stable")
    return dataframe_records(candidates.head(1))[0] if not candidates.empty else None


def build_summary_payload(
    *,
    trend_map_summary: dict[str, Any],
    trend_map_quality: dict[str, Any],
    diagnostics_summary: dict[str, Any],
    by_variant: pd.DataFrame,
    by_split: pd.DataFrame,
    concentration: pd.DataFrame,
    rejected: pd.DataFrame,
    candidates: list[dict[str, Any]],
    warnings: list[str],
    funding_missing_inst_ids: list[str],
    output_dir: Path,
    primary_timeframe: str,
    funding_data_complete: bool,
) -> dict[str, Any]:
    """Build JSON summary payload."""

    oracle_row = by_variant[by_variant["exit_variant"] == ORACLE_VARIANT]
    oracle_upper = dataframe_records(oracle_row)[0] if not oracle_row.empty else {}
    non_oracle_candidates = [candidate for candidate in candidates if not bool(candidate.get("oracle"))]
    can_phase2 = bool(non_oracle_candidates)
    return {
        "mode": "research_only_trend_capture_exit_convexity",
        "primary_timeframe": primary_timeframe,
        "trailing_exit_max_days": TRAILING_EXIT_MAX_DAYS,
        "output_dir": str(output_dir),
        "output_files": REQUIRED_OUTPUT_FILES,
        "warnings": sorted(dict.fromkeys(warnings)),
        "trend_opportunity_map": {
            "enough_trend_opportunities": trend_map_summary.get("enough_trend_opportunities"),
            "trend_opportunities_are_diversified": trend_map_summary.get("trend_opportunities_are_diversified"),
            "strongest_symbol": (trend_map_summary.get("strongest_symbol") or {}).get("symbol"),
            "strongest_timeframe": (trend_map_summary.get("strongest_timeframe") or {}).get("timeframe"),
            "recommended_next_research_direction": trend_map_summary.get("recommended_next_research_direction"),
        },
        "data_quality": {
            "trend_map_data_quality": trend_map_quality,
            "funding_data_complete": bool(funding_data_complete),
            "funding_missing_inst_ids": funding_missing_inst_ids,
        },
        "diagnostics": diagnostics_summary,
        "exit_variant_summary": dataframe_records(by_variant),
        "exit_variant_by_split": dataframe_records(by_split),
        "exit_variant_concentration": dataframe_records(concentration),
        "stable_like_candidates": non_oracle_candidates,
        "stable_like_candidate_exists": bool(non_oracle_candidates),
        "rejected_exit_variants": dataframe_records(rejected),
        "best_improving_non_oracle_variant": best_improving_variant(by_variant),
        "oracle_upper_bound": oracle_upper,
        "oracle_result_is_tradable": False,
        "oracle_excluded_from_stable_gate": True,
        "can_enter_exit_convexity_phase2": can_phase2,
        "strategy_development_allowed": False,
        "demo_live_allowed": False,
        "recommended_next_step": "exit_convexity_phase2_research" if can_phase2 else "entry_timing_research_or_pause",
    }


def format_number(value: Any, digits: int = 6) -> str:
    """Format optional numbers for Markdown."""

    number = finite_float(value, default=np.nan)
    if number is None or not np.isfinite(number):
        return "N/A"
    return f"{number:.{digits}f}"


def markdown_table(rows: list[dict[str, Any]], columns: list[str], limit: int = 50) -> str:
    """Render a compact Markdown table."""

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
    """Render the required Markdown report."""

    diagnostics = summary.get("diagnostics") or {}
    trend = summary.get("trend_opportunity_map") or {}
    best_variant = summary.get("best_improving_non_oracle_variant") or {}
    oracle = summary.get("oracle_upper_bound") or {}
    stable_exists = bool(summary.get("stable_like_candidate_exists"))
    all_failed = not stable_exists
    variant_rows = summary.get("exit_variant_summary") or []
    rejected_rows = summary.get("rejected_exit_variants") or []
    return (
        "# Trend Capture & Exit Convexity Research\n\n"
        "## Scope\n"
        "- This is counterfactual exit research only, not strategy development.\n"
        "- Legacy entries are unchanged. Trend segment labels are ex-post diagnostics only.\n"
        "- oracle_hold_to_trend_end is an upper-bound diagnostic and is not tradable.\n"
        "- strategy_development_allowed=false\n"
        "- demo_live_allowed=false\n\n"
        "## Required Answers\n"
        f"1. Is there trend opportunity in the market? {str(bool(trend.get('enough_trend_opportunities'))).lower()}.\n"
        f"2. Did legacy strategies enter trend segments? entered_share={format_number(diagnostics.get('legacy_entered_trend_share'))}.\n"
        "3. Were legacy strategies late, early out, or both? "
        f"main_failure_mode={diagnostics.get('main_failure_mode')}, "
        f"late_entry_share={format_number(diagnostics.get('late_entry_share'))}, "
        f"early_exit_share={format_number(diagnostics.get('early_exit_share'))}.\n"
        f"4. Is avg_captured_fraction materially low? avg_captured_fraction={format_number(diagnostics.get('avg_captured_fraction'))}; yes.\n"
        f"5. Is early_exit_share materially high? early_exit_share={format_number(diagnostics.get('early_exit_share'))}; yes.\n"
        f"6. Can exit-only changes make any legacy policy pass the stable-like gate? {str(stable_exists).lower()}.\n"
        f"7. Which exit variant improved the most? {best_variant.get('exit_variant', 'N/A')} "
        f"(capture_improvement={format_number(best_variant.get('capture_improvement'))}).\n"
        f"8. How high is oracle_hold_to_trend_end upper bound? "
        f"no_cost={format_number(oracle.get('no_cost_pnl'))}, "
        f"funding_adjusted={format_number(oracle.get('funding_adjusted_pnl'))}, "
        f"avg_capture={format_number(oracle.get('avg_captured_fraction'))}.\n"
        "9. If oracle is good but tradable exits are not, it means exits may help in theory but the tradable path control still cannot harvest the ex-post trend end without better timing or risk logic.\n"
        f"10. If all tradable exits fail, does it point mainly to entry timing? {str(all_failed).lower()}.\n"
        f"11. Is Exit Convexity Phase 2 allowed? {str(bool(summary.get('can_enter_exit_convexity_phase2'))).lower()}.\n"
        "12. Is formal strategy modification allowed? false.\n"
        "13. Is demo/live allowed? false.\n\n"
        "## Exit Variant Summary\n"
        f"{markdown_table(variant_rows, ['exit_variant', 'oracle', 'trade_count', 'no_cost_pnl', 'cost_aware_pnl', 'funding_adjusted_pnl', 'avg_captured_fraction', 'early_exit_share', 'late_entry_share'])}\n\n"
        "## Gate Rejections\n"
        f"{markdown_table(rejected_rows, ['exit_variant', 'oracle', 'stable_like', 'rejected_reasons'], limit=100)}\n"
    )


def write_outputs(
    output_dir: Path,
    diagnostics: pd.DataFrame,
    exit_trades: pd.DataFrame,
    by_variant: pd.DataFrame,
    by_policy: pd.DataFrame,
    by_symbol: pd.DataFrame,
    by_split: pd.DataFrame,
    concentration: pd.DataFrame,
    oracle_upper_bound: pd.DataFrame,
    funding_adjusted_summary: pd.DataFrame,
    rejected: pd.DataFrame,
    summary: dict[str, Any],
) -> None:
    """Write all required CSV/JSON/Markdown files."""

    output_dir.mkdir(parents=True, exist_ok=True)
    diagnostics.to_csv(output_dir / "trend_capture_diagnostics.csv", index=False, encoding="utf-8")
    exit_trades.to_csv(output_dir / "exit_variant_trades.csv", index=False, encoding="utf-8")
    by_variant.to_csv(output_dir / "exit_variant_summary.csv", index=False, encoding="utf-8")
    by_policy.to_csv(output_dir / "exit_variant_by_policy.csv", index=False, encoding="utf-8")
    by_symbol.to_csv(output_dir / "exit_variant_by_symbol.csv", index=False, encoding="utf-8")
    by_split.to_csv(output_dir / "exit_variant_by_split.csv", index=False, encoding="utf-8")
    concentration.to_csv(output_dir / "exit_variant_concentration.csv", index=False, encoding="utf-8")
    oracle_upper_bound.to_csv(output_dir / "oracle_upper_bound.csv", index=False, encoding="utf-8")
    funding_adjusted_summary.to_csv(output_dir / "funding_adjusted_exit_summary.csv", index=False, encoding="utf-8")
    rejected.to_csv(output_dir / "rejected_exit_variants.csv", index=False, encoding="utf-8")
    (output_dir / "trend_capture_exit_summary.json").write_text(
        json.dumps(to_jsonable(summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "trend_capture_exit_report.md").write_text(render_report(summary), encoding="utf-8")


def run_research(
    *,
    trend_map_dir: Path,
    trend_v3_dir: Path,
    vsvcb_dir: Path,
    csrb_dir: Path,
    funding_dir: Path,
    output_dir: Path,
    timezone_name: str,
    primary_timeframe: str,
    data_check_strict: bool,
    logger: logging.Logger | None = None,
    database_path: Path = DEFAULT_DATABASE_PATH,
    bars_by_symbol: dict[str, pd.DataFrame] | None = None,
) -> ResearchOutputs:
    """Run the research pipeline."""

    if primary_timeframe not in TIMEFRAME_MINUTES:
        raise TrendCaptureExitResearchError(f"unsupported primary timeframe: {primary_timeframe}")

    warnings: list[str] = []
    trend_map_summary = read_json_if_exists(trend_map_dir / "trend_opportunity_summary.json", warnings)
    trend_map_quality = read_json_if_exists(trend_map_dir / "data_quality.json", warnings)
    segments = normalize_segments(read_csv_if_exists(trend_map_dir / "trend_segments.csv", warnings, required=True), timezone_name)
    _legacy_coverage = read_csv_if_exists(trend_map_dir / "legacy_strategy_trend_coverage.csv", warnings, required=True)
    if _legacy_coverage.empty:
        warnings.append("legacy_strategy_trend_coverage_unavailable_for_cross_check")

    trades = load_legacy_trades(trend_v3_dir, vsvcb_dir, csrb_dir, timezone_name, warnings)
    segments_by_symbol = build_segments_by_symbol(segments, primary_timeframe)
    selected_segments = select_segments_for_trades(trades, segments_by_symbol)
    diagnostics = build_diagnostics(trades, segments_by_symbol, primary_timeframe, selected_segments=selected_segments)
    diagnostics_summary = summarize_diagnostics(diagnostics, trend_map_summary)

    inst_ids = sorted({symbol_to_inst_id(symbol) for symbol in trades.get("symbol", pd.Series(dtype=str)).astype(str)})
    funding_histories, funding_missing_inst_ids = load_funding_histories(funding_dir, inst_ids, warnings)
    funding_indexes = build_funding_indexes(funding_histories)
    trend_map_funding_complete = bool((trend_map_summary.get("data_quality") or {}).get("funding_data_complete"))
    funding_data_complete = bool(trend_map_funding_complete and not funding_missing_inst_ids)

    if data_check_strict:
        market_complete = bool(trend_map_quality.get("all_symbols_complete") or (trend_map_summary.get("data_quality") or {}).get("all_symbols_complete"))
        if not market_complete:
            warnings.append("data_check_strict:trend_map_market_data_not_complete")
        if not funding_data_complete:
            warnings.append("data_check_strict:funding_data_not_complete")

    if logger is not None:
        log_event(
            logger,
            logging.INFO,
            "trend_capture_exit.loaded_inputs",
            "Loaded trend capture exit inputs",
            trade_count=len(trades.index),
            segment_count=len(segments.index),
            warning_count=len(warnings),
        )

    exit_trades = build_exit_variant_trades(
        trades,
        segments_by_symbol,
        primary_timeframe,
        timezone_name,
        funding_indexes,
        database_path,
        bars_by_symbol=bars_by_symbol,
        selected_segments=selected_segments,
        logger=logger,
    )
    by_variant = build_group_summary(exit_trades, ["exit_variant", "oracle"])
    by_policy = build_group_summary(exit_trades, ["exit_variant", "oracle", "strategy_source", "policy_or_group"])
    by_symbol = build_group_summary(exit_trades, ["exit_variant", "oracle", "symbol"])
    by_split = build_group_summary(exit_trades, ["exit_variant", "oracle", "split"])
    concentration = build_concentration_summary(exit_trades)
    oracle_upper_bound = by_variant[by_variant["exit_variant"] == ORACLE_VARIANT].copy()
    funding_adjusted_summary = build_group_summary(exit_trades, ["exit_variant", "oracle", "split"])
    rejected, candidates = evaluate_stable_like_gates(by_split, by_variant, funding_data_complete)
    summary = build_summary_payload(
        trend_map_summary=trend_map_summary,
        trend_map_quality=trend_map_quality,
        diagnostics_summary=diagnostics_summary,
        by_variant=by_variant,
        by_split=by_split,
        concentration=concentration,
        rejected=rejected,
        candidates=candidates,
        warnings=warnings,
        funding_missing_inst_ids=funding_missing_inst_ids,
        output_dir=output_dir,
        primary_timeframe=primary_timeframe,
        funding_data_complete=funding_data_complete,
    )
    write_outputs(
        output_dir,
        diagnostics,
        exit_trades,
        by_variant,
        by_policy,
        by_symbol,
        by_split,
        concentration,
        oracle_upper_bound,
        funding_adjusted_summary,
        rejected,
        summary,
    )
    return ResearchOutputs(output_dir=output_dir, summary=summary, diagnostics=diagnostics, exit_trades=exit_trades, exit_summary=by_variant)


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""

    args = parse_args(argv)
    ensure_headless_runtime()
    logger = setup_logging("research_trend_capture_exit_convexity", verbose=args.verbose)
    try:
        outputs = run_research(
            trend_map_dir=resolve_path(args.trend_map_dir),
            trend_v3_dir=resolve_path(args.trend_v3_dir),
            vsvcb_dir=resolve_path(args.vsvcb_dir),
            csrb_dir=resolve_path(args.csrb_dir),
            funding_dir=resolve_path(args.funding_dir),
            output_dir=resolve_path(args.output_dir),
            timezone_name=args.timezone,
            primary_timeframe=args.primary_timeframe,
            data_check_strict=bool(args.data_check_strict),
            logger=logger,
        )
        print_json_block(
            "Trend capture exit convexity summary:",
            {
                "legacy_entered_trend_share": outputs.summary.get("diagnostics", {}).get("legacy_entered_trend_share"),
                "avg_captured_fraction": outputs.summary.get("diagnostics", {}).get("avg_captured_fraction"),
                "early_exit_share": outputs.summary.get("diagnostics", {}).get("early_exit_share"),
                "can_enter_exit_convexity_phase2": outputs.summary.get("can_enter_exit_convexity_phase2"),
                "strategy_development_allowed": outputs.summary.get("strategy_development_allowed"),
                "demo_live_allowed": outputs.summary.get("demo_live_allowed"),
                "output_dir": outputs.summary.get("output_dir"),
            },
        )
        return 0
    except TrendCaptureExitResearchError as exc:
        log_event(logger, logging.ERROR, "trend_capture_exit.error", str(exc))
        return 2
    except Exception:
        logger.exception("Unexpected Trend Capture Exit Convexity failure", extra={"event": "trend_capture_exit.unexpected"})
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
