#!/usr/bin/env python3
"""Postmortem and control audit for CSRB-v1 Phase 1 outputs."""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from common_runtime import PROJECT_ROOT, ensure_headless_runtime, log_event, setup_logging


DEFAULT_RESEARCH_DIR = PROJECT_ROOT / "reports" / "research" / "csrb_v1"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "reports" / "research" / "csrb_v1_postmortem"
DEFAULT_PRIMARY_TIMEFRAME = "15m"
DEFAULT_CORE_GROUPS = ["B", "C"]
DEFAULT_REVERSE_GROUP = "E"
DEFAULT_RANDOM_CONTROL_GROUP = "D"
CORE_SESSION_TYPES = ["asia_to_europe", "europe_to_us"]
SPLITS = ["train", "validation", "oos"]
TIMEFRAME_MINUTES = {"15m": 15, "30m": 30, "1h": 60}
HORIZONS = [4, 8, 16, 32]
PNL_COLUMNS = ["no_cost_pnl", "cost_aware_pnl", "funding_adjusted_pnl"]
SESSION_BREAKOUT_WINDOWS = {
    "asia_to_europe": (8 * 60, 11 * 60 + 59),
    "europe_to_us": (13 * 60, 17 * 60 + 59),
}

REQUIRED_INPUT_FILES = [
    "events.csv",
    "trades.csv",
    "summary.json",
    "summary.md",
    "event_group_summary.csv",
    "trade_group_summary.csv",
    "by_symbol.csv",
    "by_timeframe.csv",
    "by_split.csv",
    "session_summary.csv",
    "concentration.csv",
    "reverse_test.csv",
    "random_time_control.csv",
    "funding_summary.csv",
    "data_quality.json",
]

OUTPUT_FILES = [
    "csrb_v1_postmortem_report.md",
    "csrb_v1_postmortem_summary.json",
    "implementation_sanity.csv",
    "session_failure_decomposition.csv",
    "random_control_audit.csv",
    "random_control_seed_robustness.csv",
    "reverse_directionality_postmortem.csv",
    "postmortem_by_symbol.csv",
    "postmortem_by_direction.csv",
    "postmortem_by_timeframe.csv",
    "postmortem_by_session.csv",
    "postmortem_by_symbol_direction.csv",
    "postmortem_by_symbol_session.csv",
    "horizon_path_postmortem.csv",
    "feature_bin_postmortem.csv",
    "conflict_filter_impact.csv",
]


class CsrbPostmortemError(Exception):
    """Raised when CSRB-v1 postmortem cannot continue."""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(
        description="Analyze CSRB-v1 Phase 1 failure without tuning or strategy development."
    )
    parser.add_argument("--research-dir", default=str(DEFAULT_RESEARCH_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--primary-timeframe", default=DEFAULT_PRIMARY_TIMEFRAME)
    parser.add_argument("--core-groups", default="B,C")
    parser.add_argument("--reverse-group", default=DEFAULT_REVERSE_GROUP)
    parser.add_argument("--random-control-group", default=DEFAULT_RANDOM_CONTROL_GROUP)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def resolve_path(path_arg: str | Path) -> Path:
    """Resolve relative paths from the project root."""

    path = Path(path_arg)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def parse_csv_list(value: str | Iterable[str]) -> list[str]:
    """Parse comma/space separated values while preserving order."""

    if isinstance(value, str):
        raw = re.split(r"[\s,]+", value)
    else:
        raw = [str(item) for item in value]
    result: list[str] = []
    seen: set[str] = set()
    for token in raw:
        item = token.strip()
        if item and item not in seen:
            result.append(item)
            seen.add(item)
    return result


def finite_number(value: Any, default: float | None = 0.0) -> float | None:
    """Return a finite float or default."""

    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(number):
        return default
    return number


def numeric_series(frame: pd.DataFrame, column: str) -> pd.Series:
    """Return a numeric series, or an empty float series when unavailable."""

    if frame.empty or column not in frame.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def safe_sum(frame: pd.DataFrame, column: str) -> float:
    """Sum a numeric column if available."""

    series = numeric_series(frame, column)
    if series.empty:
        return 0.0
    return float(series.fillna(0.0).sum())


def safe_mean(frame: pd.DataFrame, column: str) -> float | None:
    """Mean a numeric column if available."""

    series = numeric_series(frame, column).dropna()
    if series.empty:
        return None
    return float(series.mean())


def parse_bool_or_none(value: Any) -> bool | None:
    """Parse common boolean encodings."""

    if value is None or pd.isna(value):
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    return None


def format_number(value: Any, digits: int = 4) -> str:
    """Format optional numeric values for reports."""

    number = finite_number(value, default=None)
    if number is None:
        return "N/A"
    return f"{number:.{digits}f}"


def dataframe_records(df: pd.DataFrame, limit: int | None = None) -> list[dict[str, Any]]:
    """Convert a DataFrame into JSON-safe records."""

    if df.empty:
        return []
    work = df.head(limit).copy() if limit is not None else df.copy()
    work = work.astype(object).where(pd.notna(work), None)
    return json.loads(work.to_json(orient="records", force_ascii=False, date_format="iso"))


def clean_json(value: Any) -> Any:
    """Convert nested values into strict JSON-compatible objects."""

    if isinstance(value, dict):
        return {str(key): clean_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [clean_json(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.generic):
        return clean_json(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def read_csv_optional(directory: Path, filename: str, warnings: list[str]) -> pd.DataFrame:
    """Read one optional CSV, recording warnings instead of failing."""

    path = directory / filename
    if not path.exists():
        warnings.append(f"missing input file: {path}")
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception as exc:  # pragma: no cover - defensive branch
        warnings.append(f"failed to read CSV {path}: {exc!r}")
        return pd.DataFrame()


def read_json_optional(directory: Path, filename: str, warnings: list[str]) -> dict[str, Any]:
    """Read one optional JSON file, recording warnings instead of failing."""

    path = directory / filename
    if not path.exists():
        warnings.append(f"missing input file: {path}")
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - defensive branch
        warnings.append(f"failed to read JSON {path}: {exc!r}")
        return {}


def read_text_optional(directory: Path, filename: str, warnings: list[str]) -> str:
    """Read one optional text file, recording warnings instead of failing."""

    path = directory / filename
    if not path.exists():
        warnings.append(f"missing input file: {path}")
        return ""
    try:
        return path.read_text(encoding="utf-8")
    except Exception as exc:  # pragma: no cover - defensive branch
        warnings.append(f"failed to read text {path}: {exc!r}")
        return ""


def load_artifacts(research_dir: Path) -> tuple[dict[str, Any], list[str]]:
    """Load all Phase 1 artifacts, keeping missing files as warnings."""

    warnings: list[str] = []
    artifacts: dict[str, Any] = {}
    for filename in REQUIRED_INPUT_FILES:
        key = "summary_md" if filename == "summary.md" else filename.rsplit(".", 1)[0]
        if filename.endswith(".csv"):
            artifacts[key] = read_csv_optional(research_dir, filename, warnings)
        elif filename.endswith(".json"):
            artifacts[key] = read_json_optional(research_dir, filename, warnings)
        else:
            artifacts[key] = read_text_optional(research_dir, filename, warnings)
    return artifacts, warnings


def write_dataframe(path: Path, frame: pd.DataFrame, columns: list[str] | None = None) -> None:
    """Write a DataFrame with stable empty-file columns."""

    path.parent.mkdir(parents=True, exist_ok=True)
    output = frame.copy()
    if columns is not None:
        for column in columns:
            if column not in output.columns:
                output[column] = np.nan
        output = output.loc[:, columns]
    output.to_csv(path, index=False)


def minute_of_day(value: Any) -> int | None:
    """Return minute of day for a timestamp-like value."""

    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp):
        return None
    return int(timestamp.hour * 60 + timestamp.minute)


def in_breakout_window(value: Any, source_session_type: str) -> bool:
    """Return True if a timestamp is inside a source session breakout window."""

    window = SESSION_BREAKOUT_WINDOWS.get(str(source_session_type))
    minute = minute_of_day(value)
    if window is None or minute is None:
        return False
    return bool(window[0] <= minute <= window[1])


def get_column(frame: pd.DataFrame, column: str) -> pd.Series:
    """Return a column or an aligned empty object series."""

    if column in frame.columns:
        return frame[column]
    return pd.Series([np.nan] * len(frame.index), index=frame.index, dtype=object)


def collect_warnings(summary: dict[str, Any], data_quality: dict[str, Any], load_warnings: list[str]) -> list[str]:
    """Collect warnings from all loaded inputs."""

    warnings = list(load_warnings)
    for item in summary.get("warnings") or []:
        warnings.append(str(item))
    for record in (data_quality.get("funding", {}).get("records") or {}).values():
        for item in record.get("warnings") or []:
            warnings.append(str(item))
    return warnings


def parse_warning_count(warnings: list[str], key: str) -> int:
    """Extract integer warning counts like skipped_events_due_to_single_position_filter=123."""

    pattern = re.compile(rf"{re.escape(key)}\s*=\s*(\d+)")
    total = 0
    for warning in warnings:
        match = pattern.search(str(warning))
        if match:
            total += int(match.group(1))
    return total


def market_data_complete(summary: dict[str, Any], data_quality: dict[str, Any]) -> bool | None:
    """Return market-data completeness when known."""

    for source in [data_quality, summary.get("data_quality") or {}]:
        if "all_market_data_complete" in source:
            return bool(source.get("all_market_data_complete"))
    return None


def funding_data_complete(summary: dict[str, Any], data_quality: dict[str, Any], funding_summary: pd.DataFrame) -> bool | None:
    """Return funding completeness when known."""

    if "funding_data_complete" in summary:
        return bool(summary.get("funding_data_complete"))
    funding = data_quality.get("funding") or {}
    if "funding_data_complete" in funding:
        return bool(funding.get("funding_data_complete"))
    if not funding_summary.empty and "funding_data_complete" in funding_summary.columns:
        parsed = funding_summary["funding_data_complete"].map(parse_bool_or_none).dropna()
        if not parsed.empty:
            return bool(parsed.astype(bool).all())
    return None


def add_event_splits(events: pd.DataFrame, summary: dict[str, Any]) -> pd.DataFrame:
    """Add split labels to events from summary split dates."""

    if events.empty:
        return events.copy()
    out = events.copy()
    if "split" in out.columns:
        return out
    split_dates = summary.get("split_dates") or {}
    boundaries: list[tuple[str, pd.Timestamp, pd.Timestamp]] = []
    for split in SPLITS:
        start = split_dates.get(f"{split}_start")
        end = split_dates.get(f"{split}_end")
        if start and end:
            boundaries.append((split, pd.Timestamp(start), pd.Timestamp(end)))
    if not boundaries or "timestamp" not in out.columns:
        out["split"] = "unknown"
        return out
    timestamps = pd.to_datetime(out["timestamp"], errors="coerce")

    def assign(timestamp: pd.Timestamp) -> str:
        if pd.isna(timestamp):
            return "unknown"
        for index, (split, start, end) in enumerate(boundaries):
            if index == len(boundaries) - 1:
                if start <= timestamp <= end:
                    return split
            elif start <= timestamp < end:
                return split
        if timestamp < boundaries[0][1]:
            return boundaries[0][0]
        return boundaries[-1][0]

    out["split"] = [assign(timestamp) for timestamp in timestamps]
    return out


def core_events(events: pd.DataFrame, core_groups: list[str], primary_timeframe: str | None = None) -> pd.DataFrame:
    """Return B/C core events."""

    if events.empty or "group" not in events.columns:
        return pd.DataFrame()
    out = events[events["group"].astype(str).isin(core_groups)].copy()
    if primary_timeframe and "timeframe" in out.columns:
        out = out[out["timeframe"].astype(str) == primary_timeframe]
    return out


def core_trades(trades: pd.DataFrame, core_groups: list[str], primary_timeframe: str | None = None) -> pd.DataFrame:
    """Return B/C core trades."""

    if trades.empty or "group" not in trades.columns:
        return pd.DataFrame()
    out = trades[trades["group"].astype(str).isin(core_groups)].copy()
    if primary_timeframe and "timeframe" in out.columns:
        out = out[out["timeframe"].astype(str) == primary_timeframe]
    return out


def trade_metric_row(frame: pd.DataFrame) -> dict[str, Any]:
    """Build common trade metrics."""

    count = int(len(frame.index))
    no_cost = safe_sum(frame, "no_cost_pnl")
    cost = safe_sum(frame, "cost_aware_pnl")
    funding = safe_sum(frame, "funding_adjusted_pnl")
    return {
        "trade_count": count,
        "long_count": int((get_column(frame, "direction").astype(str) == "long").sum()) if count else 0,
        "short_count": int((get_column(frame, "direction").astype(str) == "short").sum()) if count else 0,
        "no_cost_pnl": no_cost,
        "cost_aware_pnl": cost,
        "funding_adjusted_pnl": funding,
        "avg_no_cost_pnl": float(no_cost / count) if count else None,
        "avg_funding_adjusted_pnl": float(funding / count) if count else None,
        "win_rate": float((numeric_series(frame, "funding_adjusted_pnl") > 0).mean()) if count and "funding_adjusted_pnl" in frame.columns else None,
        "sample_sufficient": bool(count >= 30),
    }


def max_drawdown(pnls: pd.Series, starting_equity: float = 1000.0) -> tuple[float, float]:
    """Return max drawdown amount and percent."""

    values = pd.to_numeric(pnls, errors="coerce").fillna(0.0)
    if values.empty:
        return 0.0, 0.0
    equity = starting_equity + values.cumsum()
    peak = equity.cummax()
    drawdown = peak - equity
    max_dd = float(drawdown.max())
    denominator = peak.replace(0.0, np.nan)
    max_dd_pct = float((drawdown / denominator).max()) if not denominator.isna().all() else 0.0
    if not np.isfinite(max_dd_pct):
        max_dd_pct = 0.0
    return max_dd, max_dd_pct


def opposite_direction(direction: Any) -> str:
    """Return the opposite long/short direction."""

    text = str(direction).lower()
    if text == "long":
        return "short"
    if text == "short":
        return "long"
    return ""


def check_reverse_event_pairing(events: pd.DataFrame, core_groups: list[str], reverse_group: str) -> dict[str, Any]:
    """Check whether E-group events are strict B/C reversals."""

    required = {"timestamp", "symbol", "timeframe", "direction", "group", "session_type", "source_session_type"}
    if events.empty or not required.issubset(events.columns):
        return {
            "status": "unknown",
            "core_event_count": 0,
            "reverse_event_count": 0,
            "paired_count": 0,
            "opposite_direction_count": 0,
            "strict_reverse": False,
        }
    core = events[events["group"].astype(str).isin(core_groups)].copy()
    reverse = events[events["group"].astype(str) == reverse_group].copy()
    core["pair_source_session_type"] = core["session_type"].astype(str)
    reverse["pair_source_session_type"] = reverse["source_session_type"].astype(str)
    key_columns = ["timestamp", "symbol", "timeframe", "pair_source_session_type"]
    paired = core.loc[:, key_columns + ["direction"]].merge(
        reverse.loc[:, key_columns + ["direction"]],
        on=key_columns,
        how="inner",
        suffixes=("_core", "_reverse"),
    )
    opposite = paired.apply(
        lambda row: str(row["direction_reverse"]).lower() == opposite_direction(row["direction_core"]),
        axis=1,
    ) if not paired.empty else pd.Series(dtype=bool)
    strict = bool(
        len(core.index) > 0
        and len(core.index) == len(reverse.index) == len(paired.index)
        and bool(opposite.all())
    )
    return {
        "status": "pass" if strict else "fail",
        "core_event_count": int(len(core.index)),
        "reverse_event_count": int(len(reverse.index)),
        "paired_count": int(len(paired.index)),
        "opposite_direction_count": int(opposite.sum()) if not opposite.empty else 0,
        "strict_reverse": strict,
    }


def count_base_same_bar_direction_conflicts(events: pd.DataFrame, reverse_group: str) -> int:
    """Count same-bar long/short conflicts within the same non-reverse group."""

    required = {"timestamp", "symbol", "timeframe", "direction", "group"}
    if events.empty or not required.issubset(events.columns):
        return 0
    base = events[events["group"].astype(str) != reverse_group]
    if base.empty:
        return 0
    direction_counts = base.groupby(["timestamp", "symbol", "timeframe", "group"], dropna=False)["direction"].nunique()
    return int((direction_counts > 1).sum())


def timeframe_delta_ok(trades: pd.DataFrame) -> tuple[str, int, int]:
    """Check whether exit_time-entry_time equals hold_bars*timeframe minutes."""

    required = {"entry_time", "exit_time", "hold_bars", "timeframe"}
    if trades.empty or not required.issubset(trades.columns):
        return "unknown", 0, 0
    checked = 0
    mismatch = 0
    for row in trades.itertuples(index=False):
        timeframe = str(getattr(row, "timeframe"))
        minutes = TIMEFRAME_MINUTES.get(timeframe)
        if minutes is None:
            continue
        entry = pd.Timestamp(getattr(row, "entry_time"))
        exit_time = pd.Timestamp(getattr(row, "exit_time"))
        hold_bars = int(getattr(row, "hold_bars"))
        expected = pd.Timedelta(minutes=minutes * hold_bars)
        checked += 1
        if exit_time - entry != expected:
            mismatch += 1
    status = "pass" if checked > 0 and mismatch == 0 else "fail"
    return status, checked, mismatch


def build_implementation_sanity(
    *,
    events: pd.DataFrame,
    trades: pd.DataFrame,
    summary: dict[str, Any],
    data_quality: dict[str, Any],
    funding_summary: pd.DataFrame,
    warnings: list[str],
    core_groups: list[str],
    reverse_group: str,
    random_control_group: str,
    primary_timeframe: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build data and implementation sanity diagnostics."""

    market_complete = market_data_complete(summary, data_quality)
    funding_complete = funding_data_complete(summary, data_quality, funding_summary)
    primary_core_events = core_events(events, core_groups, primary_timeframe)
    primary_core_trades = core_trades(trades, core_groups, primary_timeframe)
    trade_counts = {
        split: int((get_column(primary_core_trades, "split").astype(str) == split).sum()) if not primary_core_trades.empty else 0
        for split in SPLITS
    }
    trade_count_enough = bool(trade_counts["train"] >= 30 and trade_counts["validation"] >= 10 and trade_counts["oos"] >= 10)

    entry_status = "unknown"
    entry_checked = 0
    entry_mismatch = 0
    if not events.empty and not trades.empty and {"event_id", "entry_price"}.issubset(events.columns) and {"event_id", "entry_price"}.issubset(trades.columns):
        joined = primary_core_trades.merge(events[["event_id", "entry_price"]].rename(columns={"entry_price": "event_entry_price"}), on="event_id", how="left")
        prices = pd.to_numeric(joined["entry_price"], errors="coerce")
        event_prices = pd.to_numeric(joined["event_entry_price"], errors="coerce")
        valid = prices.notna() & event_prices.notna()
        entry_checked = int(valid.sum())
        tolerance = np.maximum(1e-9, prices[valid].abs() * 1e-9)
        entry_mismatch = int(((prices[valid] - event_prices[valid]).abs() > tolerance).sum())
        entry_status = "pass" if entry_checked > 0 and entry_mismatch == 0 else "fail"

    exit_status, exit_checked, exit_mismatch = timeframe_delta_ok(primary_core_trades)
    funding_adjusted_available = bool(
        not trades.empty
        and "funding_adjusted_pnl" in trades.columns
        and numeric_series(trades, "funding_adjusted_pnl").notna().any()
    )
    reverse_pairing = check_reverse_event_pairing(events, core_groups, reverse_group)
    base_conflict_count = count_base_same_bar_direction_conflicts(events, reverse_group)
    skipped_conflict = parse_warning_count(warnings, "skipped_events_due_to_single_position_filter")
    skipped_future = parse_warning_count(warnings, "skipped_events_with_insufficient_future_bars")
    total_events = int(len(events.index)) if not events.empty else int(sum((summary.get("event_counts") or {}).values()) or 0)
    skipped_conflict_rate = float(skipped_conflict / total_events) if total_events else None

    random_events = events[get_column(events, "group").astype(str) == random_control_group].copy() if not events.empty else pd.DataFrame()
    d_breakout_violation_count = 0
    if not random_events.empty and {"timestamp", "source_session_type"}.issubset(random_events.columns):
        d_breakout_violation_count = int(
            random_events.apply(lambda row: in_breakout_window(row["timestamp"], str(row["source_session_type"])), axis=1).sum()
        )
    d_control_key_available = bool(not random_events.empty and "control_key" in random_events.columns and random_events["control_key"].notna().any())

    possible_issue = bool(
        market_complete is False
        or funding_complete is False
        or len(primary_core_events.index) == 0
        or len(primary_core_trades.index) == 0
        or not trade_count_enough
        or entry_status == "fail"
        or exit_status == "fail"
        or not funding_adjusted_available
        or reverse_pairing.get("status") == "fail"
        or d_breakout_violation_count > 0
        or base_conflict_count > 0
    )

    def status_from_bool(value: bool | None) -> str:
        if value is None:
            return "unknown"
        return "pass" if value else "fail"

    rows = [
        ("market_data_complete", status_from_bool(market_complete), market_complete, "all_market_data_complete from data_quality.json"),
        ("funding_data_complete", status_from_bool(funding_complete), funding_complete, "actual OKX funding coverage available"),
        ("core_event_count_nonzero", "pass" if len(primary_core_events.index) > 0 else "fail", len(primary_core_events.index), "B/C events in primary timeframe"),
        ("core_trade_count_enough", "pass" if trade_count_enough else "fail", json.dumps(trade_counts, sort_keys=True), "train>=30 validation>=10 oos>=10"),
        ("entry_uses_next_open", entry_status, entry_mismatch, f"checked={entry_checked}; trade.entry_price must equal event.entry_price"),
        ("exit_uses_open_t_plus_hold_plus_one", exit_status, exit_mismatch, f"checked={exit_checked}; exit-entry must equal hold_bars*timeframe"),
        ("funding_adjusted_available", "pass" if funding_adjusted_available else "fail", funding_adjusted_available, "trades.csv funding_adjusted_pnl populated"),
        ("reverse_E_strictly_from_BC", reverse_pairing["status"], reverse_pairing.get("strict_reverse"), json.dumps(reverse_pairing, sort_keys=True)),
        ("random_D_avoids_breakout_window", "pass" if d_breakout_violation_count == 0 else "fail", d_breakout_violation_count, "D timestamps must be outside source breakout windows"),
        ("random_D_control_key_available", "pass" if d_control_key_available else "warning", d_control_key_available, "control_key supports deterministic random-control audit"),
        ("same_bar_long_short_base_anomaly", "pass" if base_conflict_count == 0 else "fail", base_conflict_count, "non-reverse same symbol/timeframe/bar long-short conflicts"),
        ("skipped_events_due_to_single_position_filter", "warning" if skipped_conflict else "pass", skipped_conflict, f"rate={format_number(skipped_conflict_rate, 6)}; skipped_future={skipped_future}"),
        ("possible_implementation_issue", "fail" if possible_issue else "pass", possible_issue, "true only when data completeness or mechanical checks fail"),
    ]
    frame = pd.DataFrame(
        [
            {
                "check_name": name,
                "status": status,
                "value": value,
                "details": details,
                "possible_implementation_issue": possible_issue,
            }
            for name, status, value, details in rows
        ]
    )
    facts = {
        "market_data_complete": market_complete,
        "funding_data_complete": funding_complete,
        "primary_core_event_count": int(len(primary_core_events.index)),
        "primary_core_trade_count": int(len(primary_core_trades.index)),
        "primary_trade_counts": trade_counts,
        "entry_mismatch_count": entry_mismatch,
        "entry_checked_count": entry_checked,
        "exit_mismatch_count": exit_mismatch,
        "exit_checked_count": exit_checked,
        "funding_adjusted_available": funding_adjusted_available,
        "reverse_pairing": reverse_pairing,
        "random_D_breakout_window_violation_count": d_breakout_violation_count,
        "base_same_bar_conflict_count": base_conflict_count,
        "skipped_events_due_to_single_position_filter": skipped_conflict,
        "skipped_events_due_to_single_position_filter_rate": skipped_conflict_rate,
        "skipped_events_with_insufficient_future_bars": skipped_future,
        "possible_implementation_issue": possible_issue,
    }
    return frame, facts


def split_values(frame: pd.DataFrame) -> list[str]:
    """Return split labels plus all."""

    values = []
    if not frame.empty and "split" in frame.columns:
        values = [split for split in SPLITS if split in set(frame["split"].astype(str))]
        values.extend(sorted(set(frame["split"].astype(str)) - set(SPLITS)))
    return values + ["all"]


def filter_split(frame: pd.DataFrame, split: str) -> pd.DataFrame:
    """Filter a frame for one split, preserving all rows when split=all."""

    if split == "all" or frame.empty or "split" not in frame.columns:
        return frame.copy()
    return frame[frame["split"].astype(str) == split].copy()


def summarize_by(trades: pd.DataFrame, group_columns: list[str], core_groups: list[str]) -> pd.DataFrame:
    """Summarize core B/C trades by requested columns."""

    columns = group_columns + [
        "trade_count",
        "long_count",
        "short_count",
        "no_cost_pnl",
        "cost_aware_pnl",
        "funding_adjusted_pnl",
        "avg_no_cost_pnl",
        "avg_funding_adjusted_pnl",
        "win_rate",
        "max_drawdown",
        "max_drawdown_pct",
        "sample_sufficient",
        "drag_rank",
        "largest_drag",
    ]
    focus = core_trades(trades, core_groups)
    if focus.empty:
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, Any]] = []
    grouped = focus.groupby(group_columns, dropna=False, sort=True) if group_columns else [((), focus)]
    for keys, frame in grouped:
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {column: key for column, key in zip(group_columns, keys)}
        row.update(trade_metric_row(frame))
        dd, dd_pct = max_drawdown(numeric_series(frame, "funding_adjusted_pnl"))
        row["max_drawdown"] = dd
        row["max_drawdown_pct"] = dd_pct
        rows.append(row)
    result = pd.DataFrame(rows)
    if result.empty:
        return pd.DataFrame(columns=columns)
    result = result.sort_values("funding_adjusted_pnl", kind="stable").reset_index(drop=True)
    result["drag_rank"] = np.arange(1, len(result.index) + 1)
    result["largest_drag"] = result["drag_rank"] == 1
    return result.loc[:, columns]


def build_session_failure_decomposition(
    trades: pd.DataFrame,
    core_groups: list[str],
    primary_timeframe: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build B/C/core session failure decomposition."""

    columns = [
        "scope",
        "group",
        "session_type",
        "timeframe",
        "split",
        "dimension",
        "dimension_value",
        "trade_count",
        "no_cost_pnl",
        "cost_aware_pnl",
        "funding_adjusted_pnl",
        "avg_no_cost_pnl",
        "all_splits_negative",
        "less_negative_than_A",
        "has_positive_edge",
        "sample_sufficient",
    ]
    if trades.empty:
        return pd.DataFrame(columns=columns), {}
    rows: list[dict[str, Any]] = []
    primary = trades[get_column(trades, "timeframe").astype(str) == primary_timeframe].copy()
    a_primary = primary[get_column(primary, "group").astype(str) == "A"]

    scopes: list[tuple[str, pd.DataFrame, str, str]] = []
    for group in core_groups:
        frame = primary[get_column(primary, "group").astype(str) == group]
        stype = str(frame["session_type"].dropna().iloc[0]) if not frame.empty and "session_type" in frame.columns else group
        scopes.append((group, frame, group, stype))
    scopes.append(("core_session_breakout", primary[get_column(primary, "group").astype(str).isin(core_groups)], "B+C", "core_session_breakout"))

    for scope, frame_scope, group_label, session_type in scopes:
        split_pnls = []
        for split in SPLITS + ["all"]:
            frame = filter_split(frame_scope, split)
            metrics = trade_metric_row(frame)
            split_pnls.append(metrics["no_cost_pnl"]) if split in SPLITS else None
            a_frame = filter_split(a_primary, split)
            rows.append(
                {
                    "scope": scope,
                    "group": group_label,
                    "session_type": session_type,
                    "timeframe": primary_timeframe,
                    "split": split,
                    "dimension": "split",
                    "dimension_value": split,
                    **metrics,
                    "all_splits_negative": None,
                    "less_negative_than_A": bool(metrics["no_cost_pnl"] > safe_sum(a_frame, "no_cost_pnl")),
                    "has_positive_edge": bool(metrics["no_cost_pnl"] > 0),
                }
            )
        all_negative = bool(split_pnls and all(value < 0 for value in split_pnls))
        for index in range(len(rows) - 4, len(rows)):
            rows[index]["all_splits_negative"] = all_negative

        for dimension in ["symbol", "direction"]:
            if dimension not in frame_scope.columns:
                continue
            for value, frame in frame_scope.groupby(dimension, dropna=False, sort=True):
                metrics = trade_metric_row(frame)
                rows.append(
                    {
                        "scope": scope,
                        "group": group_label,
                        "session_type": session_type,
                        "timeframe": primary_timeframe,
                        "split": "all",
                        "dimension": dimension,
                        "dimension_value": value,
                        **metrics,
                        "all_splits_negative": all_negative,
                        "less_negative_than_A": None,
                        "has_positive_edge": bool(metrics["no_cost_pnl"] > 0),
                    }
                )
        if "entry_time" in frame_scope.columns:
            work = frame_scope.copy()
            work["_month"] = pd.to_datetime(work["entry_time"], errors="coerce", utc=True).dt.tz_convert(None).dt.to_period("M").astype(str)
            for value, frame in work.groupby("_month", dropna=False, sort=True):
                metrics = trade_metric_row(frame)
                rows.append(
                    {
                        "scope": scope,
                        "group": group_label,
                        "session_type": session_type,
                        "timeframe": primary_timeframe,
                        "split": "all",
                        "dimension": "month",
                        "dimension_value": value,
                        **metrics,
                        "all_splits_negative": all_negative,
                        "less_negative_than_A": None,
                        "has_positive_edge": bool(metrics["no_cost_pnl"] > 0),
                    }
                )
    result = pd.DataFrame(rows, columns=columns)
    core_all = result[(result["scope"] == "core_session_breakout") & (result["dimension"] == "split") & (result["split"] == "all")]
    b_all = result[(result["scope"] == "B") & (result["dimension"] == "split") & (result["split"] == "all")]
    c_all = result[(result["scope"] == "C") & (result["dimension"] == "split") & (result["split"] == "all")]
    b_pnl = float(b_all["no_cost_pnl"].iloc[0]) if not b_all.empty else 0.0
    c_pnl = float(c_all["no_cost_pnl"].iloc[0]) if not c_all.empty else 0.0
    worst_session = "asia_to_europe" if b_pnl < c_pnl else "europe_to_us"
    facts = {
        "b_all_splits_negative": bool(result[(result["scope"] == "B") & (result["dimension"] == "split")]["all_splits_negative"].dropna().any()),
        "c_all_splits_negative": bool(result[(result["scope"] == "C") & (result["dimension"] == "split")]["all_splits_negative"].dropna().any()),
        "core_no_cost_pnl": float(core_all["no_cost_pnl"].iloc[0]) if not core_all.empty else 0.0,
        "b_no_cost_pnl": b_pnl,
        "c_no_cost_pnl": c_pnl,
        "worse_session_type": worst_session,
        "session_breakout_failure": bool(float(core_all["no_cost_pnl"].iloc[0]) < 0) if not core_all.empty else True,
        "session_breakout_less_negative_than_A": bool(core_all["less_negative_than_A"].iloc[0]) if not core_all.empty else False,
    }
    return result, facts


def distribution_table(frame: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    """Return counts and long ratio by grouped dimensions."""

    if frame.empty:
        return pd.DataFrame(columns=group_columns + ["event_count", "long_count", "short_count", "long_ratio"])
    rows = []
    grouped = frame.groupby(group_columns, dropna=False, sort=True) if group_columns else [((), frame)]
    for keys, group in grouped:
        if not isinstance(keys, tuple):
            keys = (keys,)
        count = int(len(group.index))
        long_count = int((get_column(group, "direction").astype(str) == "long").sum())
        short_count = int((get_column(group, "direction").astype(str) == "short").sum())
        row = {column: key for column, key in zip(group_columns, keys)}
        row.update(
            {
                "event_count": count,
                "long_count": long_count,
                "short_count": short_count,
                "long_ratio": float(long_count / count) if count else None,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def build_random_control_audit(
    events: pd.DataFrame,
    trades: pd.DataFrame,
    core_groups: list[str],
    random_control_group: str,
    primary_timeframe: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build D random-time control audit rows."""

    columns = [
        "check_name",
        "scope",
        "status",
        "core_value",
        "random_value",
        "difference",
        "details",
        "random_control_requires_review",
    ]
    primary_events = events[get_column(events, "timeframe").astype(str) == primary_timeframe].copy() if not events.empty else pd.DataFrame()
    core = primary_events[get_column(primary_events, "group").astype(str).isin(core_groups)].copy()
    random = primary_events[get_column(primary_events, "group").astype(str) == random_control_group].copy()
    rows: list[dict[str, Any]] = []

    def add(check: str, scope: str, status: str, core_value: Any, random_value: Any, details: str) -> None:
        diff = None
        if isinstance(core_value, (int, float, np.integer, np.floating)) and isinstance(random_value, (int, float, np.integer, np.floating)):
            diff = float(random_value) - float(core_value)
        rows.append(
            {
                "check_name": check,
                "scope": scope,
                "status": status,
                "core_value": core_value,
                "random_value": random_value,
                "difference": diff,
                "details": details,
            }
        )

    add("event_count_match", "primary_timeframe", "pass" if len(core.index) == len(random.index) else "warning", len(core.index), len(random.index), "D should preserve B+C primary event count")
    core_long = float((get_column(core, "direction").astype(str) == "long").mean()) if len(core.index) else None
    random_long = float((get_column(random, "direction").astype(str) == "long").mean()) if len(random.index) else None
    add("long_ratio_match", "primary_timeframe", "pass" if core_long == random_long else "warning", core_long, random_long, "D should preserve aggregate direction ratio")

    for dims, name in [
        (["symbol"], "symbol_distribution"),
        (["timeframe"], "timeframe_distribution"),
        (["split"], "split_distribution"),
        (["symbol", "timeframe", "direction"], "symbol_timeframe_direction_distribution"),
    ]:
        if not set(dims).issubset(core.columns) or not set(dims).issubset(random.columns):
            add(name, ",".join(dims), "unknown", None, None, "required columns missing")
            continue
        core_counts = distribution_table(core, dims).rename(columns={"event_count": "core_event_count"})
        random_counts = distribution_table(random, dims).rename(columns={"event_count": "random_event_count"})
        merged = core_counts.merge(random_counts, on=dims, how="outer").fillna(0)
        max_abs_diff = float((merged["random_event_count"] - merged["core_event_count"]).abs().max()) if not merged.empty else 0.0
        add(name, ",".join(dims), "pass" if max_abs_diff == 0 else "warning", 0, max_abs_diff, "max absolute bucket event-count difference")

    breakout_violations = 0
    if not random.empty and {"timestamp", "source_session_type"}.issubset(random.columns):
        breakout_violations = int(random.apply(lambda row: in_breakout_window(row["timestamp"], str(row["source_session_type"])), axis=1).sum())
    add("random_avoids_breakout_window", "primary_timeframe", "pass" if breakout_violations == 0 else "fail", 0, breakout_violations, "D timestamps must be outside source breakout windows")

    d_trades = trades[(get_column(trades, "group").astype(str) == random_control_group) & (get_column(trades, "timeframe").astype(str) == primary_timeframe)] if not trades.empty else pd.DataFrame()
    core_tr = core_trades(trades, core_groups, primary_timeframe)
    d_no_cost = safe_sum(d_trades, "no_cost_pnl")
    core_no_cost = safe_sum(core_tr, "no_cost_pnl")
    add("random_control_no_cost_positive", "primary_timeframe", "warning" if d_no_cost > 0 else "pass", core_no_cost, d_no_cost, "positive random control requires robustness review")

    requires_review = bool(d_no_cost > 0 or breakout_violations > 0 or len(core.index) != len(random.index))
    result = pd.DataFrame(rows, columns=[column for column in columns if column != "random_control_requires_review"])
    result["random_control_requires_review"] = requires_review
    facts = {
        "random_control_requires_review": requires_review,
        "random_control_primary_no_cost_pnl": d_no_cost,
        "core_primary_no_cost_pnl": core_no_cost,
        "random_control_breakout_violation_count": breakout_violations,
        "random_control_event_count_matches_core": bool(len(core.index) == len(random.index)),
    }
    return result.loc[:, columns], facts


def enrich_trades_with_events(trades: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """Attach event timestamp and original event metadata to trades."""

    if trades.empty:
        return trades.copy()
    out = trades.copy()
    if events.empty or "event_id" not in events.columns or "event_id" not in trades.columns:
        return out
    columns = [
        column
        for column in [
            "event_id",
            "timestamp",
            "symbol",
            "timeframe",
            "direction",
            "group",
            "session_type",
            "source_session_type",
            "session_date",
        ]
        if column in events.columns
    ]
    meta = events.loc[:, columns].rename(
        columns={
            "timestamp": "event_timestamp",
            "direction": "event_direction",
            "group": "event_group",
            "session_type": "event_session_type",
            "source_session_type": "event_source_session_type",
        }
    )
    return out.merge(meta, on="event_id", how="left", suffixes=("", "_event"))


def build_random_control_seed_robustness(
    events: pd.DataFrame,
    trades: pd.DataFrame,
    core_groups: list[str],
    random_control_group: str,
    primary_timeframe: str,
    seed_count: int = 100,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build lightweight multi-seed random-time robustness from available non-window event pool."""

    columns = [
        "seed",
        "sample_count",
        "candidate_missing_count",
        "no_cost_pnl",
        "cost_aware_pnl",
        "funding_adjusted_pnl",
        "positive_no_cost",
        "positive_funding_adjusted",
        "candidate_pool_method",
        "no_parameter_tuning_allowed",
    ]
    if events.empty or trades.empty:
        return pd.DataFrame(columns=columns), {
            "seed_count": 0,
            "mean_no_cost_pnl": None,
            "median_no_cost_pnl": None,
            "positive_seed_share": None,
            "multi_seed_random_control_mean_positive": False,
            "current_seed_likely_偶然": None,
        }

    enriched = enrich_trades_with_events(trades, events)
    source = enriched[
        get_column(enriched, "group").astype(str).isin(core_groups)
        & (get_column(enriched, "timeframe").astype(str) == primary_timeframe)
    ].copy()
    candidate_pool = enriched[
        ~get_column(enriched, "group").astype(str).isin(core_groups + [DEFAULT_REVERSE_GROUP])
        & (get_column(enriched, "timeframe").astype(str) == primary_timeframe)
    ].copy()
    if source.empty or candidate_pool.empty:
        return pd.DataFrame(columns=columns), {
            "seed_count": 0,
            "mean_no_cost_pnl": None,
            "median_no_cost_pnl": None,
            "positive_seed_share": None,
            "multi_seed_random_control_mean_positive": False,
            "current_seed_likely_偶然": None,
        }

    candidate_pool = candidate_pool.reset_index(drop=True)
    candidate_pool["_event_minute"] = pd.to_datetime(candidate_pool.get("event_timestamp"), errors="coerce").map(
        lambda value: None if pd.isna(value) else int(value.hour * 60 + value.minute)
    )
    grouped_candidates: dict[tuple[str, str, str, str, str], np.ndarray] = {}
    fallback_candidates: dict[tuple[str, str, str, str], np.ndarray] = {}
    for source_session_type, window in SESSION_BREAKOUT_WINDOWS.items():
        outside = candidate_pool[
            ~candidate_pool["_event_minute"].map(
                lambda minute: bool(minute is not None and window[0] <= minute <= window[1])
            )
        ].copy()
        if outside.empty:
            continue
        for keys, group in outside.groupby(["symbol", "timeframe", "direction", "split"], dropna=False, sort=False):
            symbol, timeframe, direction, split = (str(item) for item in keys)
            grouped_candidates[(source_session_type, symbol, timeframe, direction, split)] = group.index.to_numpy(dtype=int)
        for keys, group in outside.groupby(["symbol", "timeframe", "direction"], dropna=False, sort=False):
            symbol, timeframe, direction = (str(item) for item in keys)
            fallback_candidates[(source_session_type, symbol, timeframe, direction)] = group.index.to_numpy(dtype=int)

    source_specs: list[tuple[str, str, str, str, str]] = []
    for source_row in source.itertuples(index=False):
        source_specs.append(
            (
                str(getattr(source_row, "source_session_type", getattr(source_row, "event_source_session_type", ""))),
                str(getattr(source_row, "symbol")),
                str(getattr(source_row, "timeframe")),
                str(getattr(source_row, "direction")),
                str(getattr(source_row, "split", "")),
            )
        )
    pnl_arrays = {
        column: pd.to_numeric(candidate_pool.get(column), errors="coerce").fillna(0.0).to_numpy(dtype=float)
        for column in PNL_COLUMNS
    }
    rows: list[dict[str, Any]] = []
    for seed in range(seed_count):
        rng = np.random.default_rng(seed)
        sampled_indices: list[int] = []
        missing = 0
        for source_session_type, symbol, timeframe, direction, split in source_specs:
            candidate_indices = grouped_candidates.get((source_session_type, symbol, timeframe, direction, split))
            if candidate_indices is None or len(candidate_indices) == 0:
                candidate_indices = fallback_candidates.get((source_session_type, symbol, timeframe, direction))
            if candidate_indices is None or len(candidate_indices) == 0:
                missing += 1
                continue
            sampled_indices.append(int(candidate_indices[int(rng.integers(0, len(candidate_indices)))]))
        no_cost = float(pnl_arrays["no_cost_pnl"][sampled_indices].sum()) if sampled_indices else 0.0
        cost = float(pnl_arrays["cost_aware_pnl"][sampled_indices].sum()) if sampled_indices else 0.0
        funding = float(pnl_arrays["funding_adjusted_pnl"][sampled_indices].sum()) if sampled_indices else 0.0
        rows.append(
            {
                "seed": seed,
                "sample_count": int(len(sampled_indices)),
                "candidate_missing_count": missing,
                "no_cost_pnl": no_cost,
                "cost_aware_pnl": cost,
                "funding_adjusted_pnl": funding,
                "positive_no_cost": bool(no_cost > 0),
                "positive_funding_adjusted": bool(funding > 0),
                "candidate_pool_method": "available_traded_non_breakout_event_pool_same_symbol_timeframe_direction_split",
                "no_parameter_tuning_allowed": True,
            }
        )
    result = pd.DataFrame(rows, columns=columns)
    mean_no = safe_mean(result, "no_cost_pnl")
    median_no = float(pd.to_numeric(result["no_cost_pnl"], errors="coerce").median()) if not result.empty else None
    positive_share = float(result["positive_no_cost"].mean()) if not result.empty else None
    facts = {
        "seed_count": int(len(result.index)),
        "mean_no_cost_pnl": mean_no,
        "median_no_cost_pnl": median_no,
        "positive_seed_share": positive_share,
        "mean_funding_adjusted_pnl": safe_mean(result, "funding_adjusted_pnl"),
        "multi_seed_random_control_mean_positive": bool(mean_no is not None and mean_no > 0),
        "current_seed_likely_偶然": bool(mean_no is not None and mean_no <= 0 and positive_share is not None and positive_share < 0.5),
    }
    return result, facts


def reverse_comparison_row(
    *,
    dimension: str,
    timeframe: str,
    split: str,
    direction: str,
    session_type: str,
    forward: pd.DataFrame,
    reverse: pd.DataFrame,
) -> dict[str, Any]:
    """Build one reverse comparison row."""

    forward_no = safe_sum(forward, "no_cost_pnl")
    reverse_no = safe_sum(reverse, "no_cost_pnl")
    forward_cost = safe_sum(forward, "cost_aware_pnl")
    reverse_cost = safe_sum(reverse, "cost_aware_pnl")
    forward_funding = safe_sum(forward, "funding_adjusted_pnl")
    reverse_funding = safe_sum(reverse, "funding_adjusted_pnl")
    return {
        "dimension": dimension,
        "timeframe": timeframe,
        "split": split,
        "direction": direction,
        "session_type": session_type,
        "forward_trade_count": int(len(forward.index)),
        "reverse_trade_count": int(len(reverse.index)),
        "forward_no_cost_pnl": forward_no,
        "reverse_no_cost_pnl": reverse_no,
        "forward_cost_aware_pnl": forward_cost,
        "reverse_cost_aware_pnl": reverse_cost,
        "forward_funding_adjusted_pnl": forward_funding,
        "reverse_funding_adjusted_pnl": reverse_funding,
        "reverse_better_no_cost": bool(reverse_no > forward_no),
        "reverse_better_cost_aware": bool(reverse_cost > forward_cost),
        "reverse_better_funding_adjusted": bool(reverse_funding > forward_funding),
        "reverse_cost_after_cost_positive": bool(reverse_cost > 0),
        "reverse_funding_adjusted_positive": bool(reverse_funding > 0),
        "mechanical_no_cost_inverse": bool(abs(forward_no + reverse_no) <= 1e-6 * max(1.0, abs(forward_no), abs(reverse_no))),
    }


def build_reverse_directionality_postmortem(
    trades: pd.DataFrame,
    core_groups: list[str],
    reverse_group: str,
    primary_timeframe: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build E reverse-directionality diagnostics."""

    columns = [
        "dimension",
        "timeframe",
        "split",
        "direction",
        "session_type",
        "forward_trade_count",
        "reverse_trade_count",
        "forward_no_cost_pnl",
        "reverse_no_cost_pnl",
        "forward_cost_aware_pnl",
        "reverse_cost_aware_pnl",
        "forward_funding_adjusted_pnl",
        "reverse_funding_adjusted_pnl",
        "reverse_better_no_cost",
        "reverse_better_cost_aware",
        "reverse_better_funding_adjusted",
        "reverse_cost_after_cost_positive",
        "reverse_funding_adjusted_positive",
        "mechanical_no_cost_inverse",
        "possible_false_breakout_research_hypothesis",
        "csrb_trend_following_hypothesis_failed",
    ]
    if trades.empty:
        return pd.DataFrame(columns=columns), {}
    rows: list[dict[str, Any]] = []
    for timeframe in sorted(get_column(trades, "timeframe").dropna().astype(str).unique()):
        frame_tf = trades[get_column(trades, "timeframe").astype(str) == timeframe]
        for split in split_values(frame_tf):
            frame = filter_split(frame_tf, split)
            rows.append(
                reverse_comparison_row(
                    dimension="timeframe_split",
                    timeframe=timeframe,
                    split=split,
                    direction="all",
                    session_type="core_session_breakout",
                    forward=frame[get_column(frame, "group").astype(str).isin(core_groups)],
                    reverse=frame[get_column(frame, "group").astype(str) == reverse_group],
                )
            )
        for session_type in CORE_SESSION_TYPES:
            frame_s = frame_tf[
                (
                    (get_column(frame_tf, "group").astype(str).isin(core_groups))
                    & (get_column(frame_tf, "session_type").astype(str) == session_type)
                )
                | (
                    (get_column(frame_tf, "group").astype(str) == reverse_group)
                    & (get_column(frame_tf, "source_session_type").astype(str) == session_type)
                )
            ]
            rows.append(
                reverse_comparison_row(
                    dimension="timeframe_session",
                    timeframe=timeframe,
                    split="all",
                    direction="all",
                    session_type=session_type,
                    forward=frame_s[get_column(frame_s, "group").astype(str).isin(core_groups)],
                    reverse=frame_s[get_column(frame_s, "group").astype(str) == reverse_group],
                )
            )
    for direction in ["long", "short"]:
        forward = trades[get_column(trades, "group").astype(str).isin(core_groups) & (get_column(trades, "direction").astype(str) == direction)]
        reverse = trades[(get_column(trades, "group").astype(str) == reverse_group) & (get_column(trades, "direction").astype(str) == opposite_direction(direction))]
        rows.append(
            reverse_comparison_row(
                dimension="original_forward_direction",
                timeframe="all",
                split="all",
                direction=direction,
                session_type="core_session_breakout",
                forward=forward,
                reverse=reverse,
            )
        )
    result = pd.DataFrame(rows, columns=[column for column in columns if column not in {"possible_false_breakout_research_hypothesis", "csrb_trend_following_hypothesis_failed"}])
    primary_rows = result[
        (result["dimension"] == "timeframe_split")
        & (result["timeframe"] == primary_timeframe)
        & (result["split"].isin(SPLITS))
    ] if not result.empty else pd.DataFrame()
    oos_primary = result[
        (result["dimension"] == "timeframe_split")
        & (result["timeframe"] == primary_timeframe)
        & (result["split"] == "oos")
    ] if not result.empty else pd.DataFrame()
    all_timeframes = result[(result["dimension"] == "timeframe_split") & (result["split"] == "all")] if not result.empty else pd.DataFrame()
    all_sessions = result[(result["dimension"] == "timeframe_session")] if not result.empty else pd.DataFrame()
    all_directions = result[result["dimension"] == "original_forward_direction"] if not result.empty else pd.DataFrame()
    reverse_all_primary_splits = bool(not primary_rows.empty and primary_rows["reverse_better_funding_adjusted"].astype(bool).all())
    reverse_all_timeframes = bool(not all_timeframes.empty and all_timeframes["reverse_better_funding_adjusted"].astype(bool).all())
    reverse_all_sessions = bool(not all_sessions.empty and all_sessions["reverse_better_funding_adjusted"].astype(bool).all())
    reverse_all_directions = bool(not all_directions.empty and all_directions["reverse_better_funding_adjusted"].astype(bool).all())
    reverse_oos_positive = bool(not oos_primary.empty and bool(oos_primary.iloc[0]["reverse_funding_adjusted_positive"]))
    reverse_failure = bool(not oos_primary.empty and bool(oos_primary.iloc[0]["reverse_better_funding_adjusted"]))
    possible_false_breakout = bool(reverse_failure and reverse_oos_positive and (reverse_all_primary_splits or reverse_all_timeframes or reverse_all_sessions))
    if result.empty:
        result = pd.DataFrame(columns=columns)
    else:
        result["possible_false_breakout_research_hypothesis"] = possible_false_breakout
        result["csrb_trend_following_hypothesis_failed"] = reverse_failure
    facts = {
        "reverse_E_better_than_forward_oos": reverse_failure,
        "reverse_E_better_all_primary_splits": reverse_all_primary_splits,
        "reverse_E_better_all_timeframes": reverse_all_timeframes,
        "reverse_E_better_all_sessions": reverse_all_sessions,
        "reverse_E_better_all_directions": reverse_all_directions,
        "reverse_E_oos_funding_adjusted_positive": reverse_oos_positive,
        "possible_false_breakout_research_hypothesis": possible_false_breakout,
        "csrb_trend_following_hypothesis_failed": reverse_failure,
    }
    return result.loc[:, columns], facts


def build_breakdown_outputs(trades: pd.DataFrame, core_groups: list[str]) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    """Build symbol/direction/timeframe/session decomposition outputs."""

    outputs = {
        "postmortem_by_symbol": summarize_by(trades, ["symbol"], core_groups),
        "postmortem_by_direction": summarize_by(trades, ["direction"], core_groups),
        "postmortem_by_timeframe": summarize_by(trades, ["timeframe"], core_groups),
        "postmortem_by_session": summarize_by(trades, ["session_type"], core_groups),
        "postmortem_by_symbol_direction": summarize_by(trades, ["symbol", "direction"], core_groups),
        "postmortem_by_symbol_session": summarize_by(trades, ["symbol", "session_type"], core_groups),
    }
    by_symbol = outputs["postmortem_by_symbol"]
    by_direction = outputs["postmortem_by_direction"]
    by_timeframe = outputs["postmortem_by_timeframe"]
    by_session = outputs["postmortem_by_session"]
    by_symbol_session = outputs["postmortem_by_symbol_session"]
    facts = {
        "worst_symbol": by_symbol.iloc[0]["symbol"] if not by_symbol.empty else None,
        "worst_direction": by_direction.iloc[0]["direction"] if not by_direction.empty else None,
        "positive_timeframes": by_timeframe[by_timeframe["funding_adjusted_pnl"] > 0]["timeframe"].tolist() if not by_timeframe.empty else [],
        "worse_session_type": by_session.iloc[0]["session_type"] if not by_session.empty else None,
        "positive_symbol_session_rows": dataframe_records(by_symbol_session[by_symbol_session["funding_adjusted_pnl"] > 0]) if not by_symbol_session.empty else [],
    }
    return outputs, facts


def build_horizon_path_postmortem(events: pd.DataFrame, core_groups: list[str]) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build horizon path diagnostics from events.csv."""

    columns = [
        "scope",
        "group",
        "session_type",
        "timeframe",
        "direction",
        "horizon",
        "event_count",
        "mean_future_return",
        "median_future_return",
        "reversal_rate",
        "mean_mfe",
        "mean_mae",
        "mean_mfe_mae_ratio",
        "horizon_path_note",
    ]
    if events.empty or "group" not in events.columns:
        return pd.DataFrame(columns=columns), {}
    focus = events[events["group"].astype(str).isin(core_groups)].copy()
    if focus.empty:
        return pd.DataFrame(columns=columns), {}
    rows: list[dict[str, Any]] = []
    scopes: list[tuple[str, pd.DataFrame, str, str]] = []
    for group in core_groups:
        frame = focus[focus["group"].astype(str) == group]
        stype = str(frame["session_type"].dropna().iloc[0]) if not frame.empty and "session_type" in frame.columns else group
        scopes.append((group, frame, group, stype))
    scopes.append(("core_session_breakout", focus, "B+C", "core_session_breakout"))
    for scope, scope_frame, group_label, session_type in scopes:
        group_cols = [column for column in ["timeframe", "direction"] if column in scope_frame.columns]
        grouped = scope_frame.groupby(group_cols, dropna=False, sort=True) if group_cols else [((), scope_frame)]
        for keys, frame in grouped:
            if not isinstance(keys, tuple):
                keys = (keys,)
            base = {"timeframe": "all", "direction": "all"}
            base.update({column: key for column, key in zip(group_cols, keys)})
            for horizon in HORIZONS:
                column = f"future_return_{horizon}"
                returns = numeric_series(frame, column).dropna()
                mean_ret = float(returns.mean()) if not returns.empty else None
                median_ret = float(returns.median()) if not returns.empty else None
                reversal = float((returns < 0).mean()) if not returns.empty else None
                note = "missing_horizon_column" if column not in frame.columns else ""
                rows.append(
                    {
                        "scope": scope,
                        "group": group_label,
                        "session_type": session_type,
                        **base,
                        "horizon": horizon,
                        "event_count": int(len(returns.index)),
                        "mean_future_return": mean_ret,
                        "median_future_return": median_ret,
                        "reversal_rate": reversal,
                        "mean_mfe": safe_mean(frame, "mfe_10") if "mfe_10" in frame.columns else None,
                        "mean_mae": safe_mean(frame, "mae_10") if "mae_10" in frame.columns else None,
                        "mean_mfe_mae_ratio": safe_mean(frame, "mfe_mae_ratio_10") if "mfe_mae_ratio_10" in frame.columns else None,
                        "horizon_path_note": note,
                    }
                )
    result = pd.DataFrame(rows, columns=columns)
    core = result[(result["scope"] == "core_session_breakout") & (result["timeframe"] == "15m") & (result["direction"] == "all")]
    if core.empty:
        core = result[result["scope"] == "core_session_breakout"]
    means = {int(row.horizon): row.mean_future_return for row in core.itertuples(index=False) if row.mean_future_return is not None}
    best_horizon = max(means, key=means.get) if means else None
    facts = {
        "best_horizon": best_horizon,
        "horizon_4_better_than_16": bool(4 in means and 16 in means and means[4] > means[16]),
        "horizon_8_better_than_16": bool(8 in means and 16 in means and means[8] > means[16]),
        "horizon_32_worse_than_16": bool(32 in means and 16 in means and means[32] < means[16]),
        "early_reversal_likely": bool(4 in means and means[4] < 0),
        "mfe_mae_available": bool("mfe_10" in events.columns or "mae_10" in events.columns),
    }
    return result, facts


def bin_numeric_feature(series: pd.Series, bins: int = 5) -> pd.Series:
    """Bin a numeric feature for diagnostics."""

    numeric = pd.to_numeric(series, errors="coerce")
    valid = numeric.dropna()
    if valid.empty:
        return pd.Series(["missing"] * len(series.index), index=series.index, dtype=object)
    unique_count = int(valid.nunique())
    if unique_count < 2:
        return numeric.map(lambda value: "missing" if pd.isna(value) else f"value={format_number(value, 6)}").astype(object)
    try:
        return pd.qcut(numeric, q=min(bins, unique_count), duplicates="drop").astype(str).replace("nan", "missing")
    except ValueError:
        return pd.cut(numeric, bins=min(bins, unique_count), duplicates="drop").astype(str).replace("nan", "missing")


def add_diagnostic_features(events: pd.DataFrame) -> pd.DataFrame:
    """Add CSRB diagnostic feature columns when base columns are available."""

    out = events.copy()
    atr = pd.to_numeric(get_column(out, "atr_prev"), errors="coerce").replace(0.0, np.nan)
    range_width = pd.to_numeric(get_column(out, "range_width"), errors="coerce")
    out["range_width_atr"] = range_width / atr
    out["session_range_width"] = range_width
    close = pd.to_numeric(get_column(out, "close"), errors="coerce")
    boundary = pd.to_numeric(get_column(out, "breakout_boundary"), errors="coerce")
    direction = get_column(out, "direction").astype(str)
    out["breakout_distance_atr"] = np.where(direction == "long", (close - boundary) / atr, (boundary - close) / atr)
    range_high = pd.to_numeric(get_column(out, "range_high"), errors="coerce")
    range_low = pd.to_numeric(get_column(out, "range_low"), errors="coerce")
    out["buffer_distance_atr"] = np.where(direction == "long", (boundary - range_high) / atr, (range_low - boundary) / atr)
    if "entry_time" in out.columns:
        out["entry_hour_utc"] = pd.to_datetime(out["entry_time"], errors="coerce").dt.hour
    else:
        out["entry_hour_utc"] = np.nan
    return out


def build_feature_bin_postmortem(events: pd.DataFrame, trades: pd.DataFrame, core_groups: list[str]) -> pd.DataFrame:
    """Build feature-bin diagnostics for B/C/core events without tuning decisions."""

    columns = [
        "feature",
        "bin",
        "event_count",
        "trade_count",
        "no_cost_pnl",
        "cost_aware_pnl",
        "funding_adjusted_pnl",
        "mean_future_return",
        "reversal_rate",
        "mfe_mae_ratio",
        "warning",
        "no_parameter_tuning_allowed",
    ]
    if events.empty or "group" not in events.columns:
        return pd.DataFrame(columns=columns)
    focus = add_diagnostic_features(events[events["group"].astype(str).isin(core_groups)].copy())
    if focus.empty:
        return pd.DataFrame(columns=columns)
    trade_cols = [column for column in ["event_id", "trade_id", *PNL_COLUMNS] if column in trades.columns] if not trades.empty else []
    if trade_cols and "event_id" in focus.columns:
        merged = focus.merge(trades.loc[:, trade_cols], on="event_id", how="left")
    else:
        merged = focus.copy()
        for column in ["trade_id", *PNL_COLUMNS]:
            merged[column] = np.nan
    rows: list[dict[str, Any]] = []
    features = [
        "range_width_atr",
        "buffer_distance_atr",
        "breakout_distance_atr",
        "session_range_width",
        "entry_hour_utc",
        "direction",
        "symbol",
        "timeframe",
    ]
    future_column = "future_return_16" if "future_return_16" in merged.columns else next((f"future_return_{h}" for h in HORIZONS if f"future_return_{h}" in merged.columns), "")
    for feature in features:
        warning = ""
        if feature not in merged.columns:
            warning = f"missing feature column: {feature}"
            rows.append(
                {
                    "feature": feature,
                    "bin": "missing",
                    "event_count": 0,
                    "trade_count": 0,
                    "no_cost_pnl": 0.0,
                    "cost_aware_pnl": 0.0,
                    "funding_adjusted_pnl": 0.0,
                    "mean_future_return": None,
                    "reversal_rate": None,
                    "mfe_mae_ratio": None,
                    "warning": warning,
                    "no_parameter_tuning_allowed": True,
                }
            )
            continue
        work = merged.copy()
        if feature in {"direction", "symbol", "timeframe"}:
            work["_bin"] = work[feature].fillna("missing").astype(str)
        else:
            work["_bin"] = bin_numeric_feature(work[feature])
        for bin_value, frame in work.groupby("_bin", dropna=False, sort=True):
            returns = numeric_series(frame, future_column).dropna() if future_column else pd.Series(dtype=float)
            rows.append(
                {
                    "feature": feature,
                    "bin": str(bin_value),
                    "event_count": int(len(frame.index)),
                    "trade_count": int(frame["trade_id"].notna().sum()) if "trade_id" in frame.columns else 0,
                    "no_cost_pnl": safe_sum(frame, "no_cost_pnl"),
                    "cost_aware_pnl": safe_sum(frame, "cost_aware_pnl"),
                    "funding_adjusted_pnl": safe_sum(frame, "funding_adjusted_pnl"),
                    "mean_future_return": float(returns.mean()) if not returns.empty else None,
                    "reversal_rate": float((returns < 0).mean()) if not returns.empty else None,
                    "mfe_mae_ratio": safe_mean(frame, "mfe_mae_ratio_10") if "mfe_mae_ratio_10" in frame.columns else None,
                    "warning": warning,
                    "no_parameter_tuning_allowed": True,
                }
            )
    return pd.DataFrame(rows, columns=columns)


def build_conflict_filter_impact(
    events: pd.DataFrame,
    trades: pd.DataFrame,
    warnings: list[str],
    core_groups: list[str],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build single-position conflict-filter impact diagnostics."""

    columns = [
        "scope",
        "timeframe",
        "symbol",
        "session_type",
        "core_event_count",
        "core_trade_count",
        "untraded_event_count",
        "untraded_event_rate",
        "global_skipped_events_due_to_single_position_filter",
        "global_skipped_events_with_insufficient_future_bars",
        "traded_mean_future_return_16",
        "untraded_mean_future_return_16",
        "traded_reversal_rate_16",
        "untraded_reversal_rate_16",
        "actual_no_cost_pnl",
        "actual_cost_aware_pnl",
        "actual_funding_adjusted_pnl",
        "filtered_events_theoretically_better",
        "single_position_filter_may_distort_result",
    ]
    if events.empty or "group" not in events.columns:
        return pd.DataFrame(columns=columns), {}
    core_ev = events[events["group"].astype(str).isin(core_groups)].copy()
    core_tr = core_trades(trades, core_groups)
    traded_ids = set(core_tr["event_id"].astype(str)) if not core_tr.empty and "event_id" in core_tr.columns else set()
    core_ev["_traded"] = get_column(core_ev, "event_id").astype(str).isin(traded_ids)
    skipped_conflict = parse_warning_count(warnings, "skipped_events_due_to_single_position_filter")
    skipped_future = parse_warning_count(warnings, "skipped_events_with_insufficient_future_bars")
    rows: list[dict[str, Any]] = []

    def add_scope(scope: str, frame: pd.DataFrame, trade_frame: pd.DataFrame, timeframe: str = "all", symbol: str = "all", session_type: str = "all") -> None:
        traded_events = frame[frame["_traded"]]
        untraded_events = frame[~frame["_traded"]]
        event_count = int(len(frame.index))
        untraded_rate = float(len(untraded_events.index) / event_count) if event_count else None
        traded_ret = numeric_series(traded_events, "future_return_16").dropna()
        untraded_ret = numeric_series(untraded_events, "future_return_16").dropna()
        traded_mean = float(traded_ret.mean()) if not traded_ret.empty else None
        untraded_mean = float(untraded_ret.mean()) if not untraded_ret.empty else None
        traded_rev = float((traded_ret < 0).mean()) if not traded_ret.empty else None
        untraded_rev = float((untraded_ret < 0).mean()) if not untraded_ret.empty else None
        filtered_better = bool(untraded_mean is not None and traded_mean is not None and untraded_mean > traded_mean)
        distortion = bool(
            untraded_rate is not None
            and untraded_rate > 0.10
            and (
                filtered_better
                or (
                    untraded_rev is not None
                    and traded_rev is not None
                    and abs(untraded_rev - traded_rev) > 0.05
                )
            )
        )
        rows.append(
            {
                "scope": scope,
                "timeframe": timeframe,
                "symbol": symbol,
                "session_type": session_type,
                "core_event_count": event_count,
                "core_trade_count": int(len(trade_frame.index)),
                "untraded_event_count": int(len(untraded_events.index)),
                "untraded_event_rate": untraded_rate,
                "global_skipped_events_due_to_single_position_filter": skipped_conflict,
                "global_skipped_events_with_insufficient_future_bars": skipped_future,
                "traded_mean_future_return_16": traded_mean,
                "untraded_mean_future_return_16": untraded_mean,
                "traded_reversal_rate_16": traded_rev,
                "untraded_reversal_rate_16": untraded_rev,
                "actual_no_cost_pnl": safe_sum(trade_frame, "no_cost_pnl"),
                "actual_cost_aware_pnl": safe_sum(trade_frame, "cost_aware_pnl"),
                "actual_funding_adjusted_pnl": safe_sum(trade_frame, "funding_adjusted_pnl"),
                "filtered_events_theoretically_better": filtered_better,
                "single_position_filter_may_distort_result": distortion,
            }
        )

    add_scope("all_core", core_ev, core_tr)
    for column, scope_name in [("timeframe", "timeframe"), ("symbol", "symbol"), ("session_type", "session")]:
        if column not in core_ev.columns:
            continue
        for value, frame in core_ev.groupby(column, dropna=False, sort=True):
            trade_frame = core_tr[get_column(core_tr, column).astype(str) == str(value)] if not core_tr.empty and column in core_tr.columns else pd.DataFrame()
            add_scope(scope_name, frame, trade_frame, timeframe=str(value) if column == "timeframe" else "all", symbol=str(value) if column == "symbol" else "all", session_type=str(value) if column == "session_type" else "all")
    result = pd.DataFrame(rows, columns=columns)
    facts = dataframe_records(result[result["scope"] == "all_core"], limit=1)[0] if not result.empty else {}
    return result, facts


def build_final_decision(
    *,
    summary: dict[str, Any],
    implementation_facts: dict[str, Any],
    session_facts: dict[str, Any],
    random_facts: dict[str, Any],
    seed_facts: dict[str, Any],
    reverse_facts: dict[str, Any],
    breakdown_facts: dict[str, Any],
    horizon_facts: dict[str, Any],
    conflict_facts: dict[str, Any],
    warnings: list[str],
) -> dict[str, Any]:
    """Build final postmortem decision payload."""

    gates = summary.get("gates") or {}
    phase1_failed = bool(
        summary.get("final_decision") == "postmortem"
        or not bool(summary.get("continue_to_phase2"))
        or not all(
            bool(summary.get(key, gates.get(key, False)))
            for key in [
                "train_pass",
                "validation_pass",
                "oos_pass",
                "cost_aware_pass",
                "funding_adjusted_pass",
                "trade_count_pass",
                "concentration_pass",
                "reverse_test_pass",
                "session_vs_baseline_pass",
            ]
        )
    )
    possible_issue = bool(implementation_facts.get("possible_implementation_issue"))
    false_breakout = bool(reverse_facts.get("possible_false_breakout_research_hypothesis"))
    random_review = bool(random_facts.get("random_control_requires_review"))
    if possible_issue:
        recommended = "Review implementation warnings first; do not tune CSRB-v1, do not enter Phase 2, and do not develop a strategy."
    elif false_breakout:
        recommended = "Archive CSRB-v1 trend-following. A separate pre-registered false-breakout research hypothesis may be proposed, but E is not a tradable strategy and cannot rescue CSRB-v1."
    elif random_review:
        recommended = "Archive CSRB-v1. Random control requires review, but B/C/core remain negative and Phase 2 stays blocked."
    else:
        recommended = "Archive CSRB-v1. Do not run Phase 2, parameter plateau, randomization optimization, strategy development, demo, or live."
    return {
        "hypothesis_name": "Crypto Session Range Breakout",
        "version": "CSRB-v1-postmortem",
        "research_only": True,
        "csrb_v1_failed": True,
        "phase1_failed": phase1_failed,
        "csrb_trend_following_hypothesis_failed": True,
        "session_breakout_failure": bool(session_facts.get("session_breakout_failure", True)),
        "possible_false_breakout_research_hypothesis": false_breakout,
        "random_control_requires_review": random_review,
        "multi_seed_random_control_mean_positive": bool(seed_facts.get("multi_seed_random_control_mean_positive")),
        "continue_to_phase2": False,
        "parameter_plateau_allowed": False,
        "randomization_allowed": False,
        "strategy_development_allowed": False,
        "demo_live_allowed": False,
        "official_strategy_modification_allowed": False,
        "no_parameter_tuning_allowed": True,
        "data_or_implementation_issue": possible_issue,
        "possible_implementation_issue": possible_issue,
        "market_data_complete": implementation_facts.get("market_data_complete"),
        "funding_data_complete": implementation_facts.get("funding_data_complete"),
        "gates": {
            "train_pass": False,
            "validation_pass": False,
            "oos_pass": False,
            "cost_aware_pass": False,
            "funding_adjusted_pass": False,
            "trade_count_pass": bool(summary.get("trade_count_pass", gates.get("trade_count_pass", False))),
            "concentration_pass": False,
            "reverse_test_pass": False,
            "session_vs_baseline_pass": False,
            "continue_to_phase2": False,
        },
        "phase1_gates_original": gates,
        "implementation": implementation_facts,
        "session_failure_decomposition": session_facts,
        "random_control": random_facts,
        "random_control_seed_robustness": seed_facts,
        "reverse_E": reverse_facts,
        "breakdown": breakdown_facts,
        "horizon_path": horizon_facts,
        "conflict_filter": conflict_facts,
        "warnings": warnings,
        "recommended_next_step": recommended,
    }


def markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    """Render a compact Markdown table."""

    if not rows:
        return "- N/A"
    lines = [
        "| " + " | ".join(columns) + " |",
        "|" + "|".join("---" for _ in columns) + "|",
    ]
    for row in rows:
        values = []
        for column in columns:
            value = row.get(column)
            if isinstance(value, float):
                values.append(format_number(value, 4))
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def render_report(
    final: dict[str, Any],
    implementation_sanity: pd.DataFrame,
    session_decomposition: pd.DataFrame,
    random_audit: pd.DataFrame,
    seed_robustness: pd.DataFrame,
    reverse: pd.DataFrame,
    by_symbol: pd.DataFrame,
    by_direction: pd.DataFrame,
    by_timeframe: pd.DataFrame,
    by_session: pd.DataFrame,
    conflict_filter: pd.DataFrame,
    primary_timeframe: str,
) -> str:
    """Render the CSRB-v1 postmortem Markdown report."""

    session_facts = final.get("session_failure_decomposition") or {}
    random_facts = final.get("random_control") or {}
    seed_facts = final.get("random_control_seed_robustness") or {}
    reverse_facts = final.get("reverse_E") or {}
    breakdown = final.get("breakdown") or {}
    conflict_facts = final.get("conflict_filter") or {}
    sanity_rows = dataframe_records(
        implementation_sanity[["check_name", "status", "value"]]
        if not implementation_sanity.empty
        else pd.DataFrame(),
        limit=30,
    )
    core_split_rows = session_decomposition[
        (session_decomposition["scope"] == "core_session_breakout")
        & (session_decomposition["dimension"] == "split")
    ] if not session_decomposition.empty else pd.DataFrame()
    reverse_primary = reverse[
        (reverse["dimension"] == "timeframe_split")
        & (reverse["timeframe"] == primary_timeframe)
        & (reverse["split"].isin(SPLITS + ["all"]))
    ] if not reverse.empty else pd.DataFrame()
    seed_summary_rows = dataframe_records(seed_robustness.describe(include="all").reset_index(), limit=12) if not seed_robustness.empty else []
    return (
        "# CSRB-v1 Postmortem Report\n\n"
        "## Executive Decision\n"
        f"- csrb_v1_failed={str(bool(final.get('csrb_v1_failed'))).lower()}\n"
        f"- csrb_trend_following_hypothesis_failed={str(bool(final.get('csrb_trend_following_hypothesis_failed'))).lower()}\n"
        f"- possible_false_breakout_research_hypothesis={str(bool(final.get('possible_false_breakout_research_hypothesis'))).lower()}\n"
        f"- random_control_requires_review={str(bool(final.get('random_control_requires_review'))).lower()}\n"
        f"- continue_to_phase2={str(bool(final.get('continue_to_phase2'))).lower()}\n"
        f"- parameter_plateau_allowed={str(bool(final.get('parameter_plateau_allowed'))).lower()}\n"
        f"- randomization_allowed={str(bool(final.get('randomization_allowed'))).lower()}\n"
        f"- strategy_development_allowed={str(bool(final.get('strategy_development_allowed'))).lower()}\n"
        f"- demo_live_allowed={str(bool(final.get('demo_live_allowed'))).lower()}\n\n"
        "## Required Answers\n"
        f"1. CSRB-v1 Phase 1 是否失败？是，csrb_v1_failed={str(bool(final.get('csrb_v1_failed'))).lower()}。\n"
        f"2. 是否发现实现或数据问题？possible_implementation_issue={str(bool(final.get('possible_implementation_issue'))).lower()}，market_data_complete={final.get('market_data_complete')}，funding_data_complete={final.get('funding_data_complete')}。\n"
        f"3. B/C/core 为什么失败？B、C、core 在 primary {primary_timeframe} 的 train/validation/oos 均为负；core_no_cost_pnl={format_number(session_facts.get('core_no_cost_pnl'), 4)}。\n"
        f"4. Session breakout 是否只是比 ordinary breakout 少亏，而不是有 edge？{str(bool(session_facts.get('session_breakout_less_negative_than_A'))).lower()}；即使少亏，core 仍为负，不能构成 edge。\n"
        f"5. Random time control 为什么为正？当前 D primary no-cost={format_number(random_facts.get('random_control_primary_no_cost_pnl'), 4)}，说明本次 session timing 不优于同结构随机时间，且需要 robustness 审计。\n"
        f"6. 多 seed random control 是否仍为正？mean_no_cost={format_number(seed_facts.get('mean_no_cost_pnl'), 4)}，positive_seed_share={format_number(seed_facts.get('positive_seed_share'), 4)}，multi_seed_mean_positive={str(bool(seed_facts.get('multi_seed_random_control_mean_positive'))).lower()}。\n"
        f"7. Reverse E 是否明显优于正向？{str(bool(reverse_facts.get('reverse_E_better_than_forward_oos'))).lower()}，reverse_E_oos_funding_adjusted_positive={str(bool(reverse_facts.get('reverse_E_oos_funding_adjusted_positive'))).lower()}。\n"
        f"8. 这是否说明 CSRB 趋势延续假设失败？是，csrb_trend_following_hypothesis_failed=true。\n"
        f"9. 是否存在 false-breakout research 线索？possible_false_breakout_research_hypothesis={str(bool(final.get('possible_false_breakout_research_hypothesis'))).lower()}，但 E 不能作为趋势跟踪通过或可交易策略。\n"
        "10. 是否允许 Phase 2？否，continue_to_phase2=false。\n"
        "11. 是否允许修改正式策略？否，strategy_development_allowed=false。\n"
        "12. 是否允许 demo/live？否，demo_live_allowed=false。\n"
        f"13. 下一步建议是什么？{final.get('recommended_next_step')}\n\n"
        "## Data / Implementation Sanity\n"
        + markdown_table(sanity_rows, ["check_name", "status", "value"])
        + "\n\n"
        "## Session Failure Decomposition\n"
        + markdown_table(
            dataframe_records(core_split_rows, limit=10),
            ["scope", "split", "trade_count", "no_cost_pnl", "cost_aware_pnl", "funding_adjusted_pnl", "has_positive_edge"],
        )
        + "\n\n"
        "## Random Control Audit\n"
        + markdown_table(
            dataframe_records(random_audit, limit=12),
            ["check_name", "status", "core_value", "random_value", "details"],
        )
        + "\n\n"
        "## Random Control Seed Robustness\n"
        + markdown_table(seed_summary_rows, ["index", "seed", "sample_count", "no_cost_pnl", "funding_adjusted_pnl"])
        + "\n\n"
        "## Reverse Directionality\n"
        + markdown_table(
            dataframe_records(reverse_primary, limit=8),
            ["timeframe", "split", "forward_no_cost_pnl", "reverse_no_cost_pnl", "forward_funding_adjusted_pnl", "reverse_funding_adjusted_pnl", "reverse_better_funding_adjusted"],
        )
        + "\n\n"
        "## Symbol / Direction / Timeframe / Session Notes\n"
        f"- BTC/ETH/SOL 拖累最大：{breakdown.get('worst_symbol')}\n"
        f"- long/short 拖累最大：{breakdown.get('worst_direction')}\n"
        f"- 正收益 timeframe：{breakdown.get('positive_timeframes') or []}\n"
        f"- Asia→Europe / Europe→US 哪个更差：{breakdown.get('worse_session_type')}\n"
        f"- 局部正收益 symbol+session：{breakdown.get('positive_symbol_session_rows') or []}\n\n"
        "## Conflict Filter\n"
        f"- core_event_count={conflict_facts.get('core_event_count')}\n"
        f"- core_trade_count={conflict_facts.get('core_trade_count')}\n"
        f"- untraded_event_count={conflict_facts.get('untraded_event_count')}\n"
        f"- untraded_event_rate={format_number(conflict_facts.get('untraded_event_rate'), 4)}\n"
        f"- single_position_filter_may_distort_result={conflict_facts.get('single_position_filter_may_distort_result')}\n\n"
        "## Guardrails\n"
        "- no_parameter_tuning_allowed=true\n"
        "- parameter_plateau_allowed=false\n"
        "- randomization_allowed=false\n"
        "- strategy_development_allowed=false\n"
        "- demo_live_allowed=false\n"
        "- D random control is not tradable from this postmortem.\n"
        "- E reverse test is not a trend-following edge and is not tradable from this postmortem.\n"
    )


def run_postmortem(
    *,
    research_dir: Path,
    output_dir: Path,
    primary_timeframe: str = DEFAULT_PRIMARY_TIMEFRAME,
    core_groups: list[str] | None = None,
    reverse_group: str = DEFAULT_REVERSE_GROUP,
    random_control_group: str = DEFAULT_RANDOM_CONTROL_GROUP,
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    """Run the CSRB-v1 postmortem analysis."""

    core_groups = core_groups or DEFAULT_CORE_GROUPS
    artifacts, load_warnings = load_artifacts(research_dir)
    events = artifacts.get("events", pd.DataFrame())
    trades = artifacts.get("trades", pd.DataFrame())
    summary = artifacts.get("summary", {})
    data_quality = artifacts.get("data_quality", {})
    funding_summary = artifacts.get("funding_summary", pd.DataFrame())
    warnings = collect_warnings(summary, data_quality, load_warnings)
    events = add_event_splits(events, summary)

    implementation_sanity, implementation_facts = build_implementation_sanity(
        events=events,
        trades=trades,
        summary=summary,
        data_quality=data_quality,
        funding_summary=funding_summary,
        warnings=warnings,
        core_groups=core_groups,
        reverse_group=reverse_group,
        random_control_group=random_control_group,
        primary_timeframe=primary_timeframe,
    )
    session_decomposition, session_facts = build_session_failure_decomposition(trades, core_groups, primary_timeframe)
    random_audit, random_facts = build_random_control_audit(events, trades, core_groups, random_control_group, primary_timeframe)
    seed_robustness, seed_facts = build_random_control_seed_robustness(events, trades, core_groups, random_control_group, primary_timeframe)
    reverse, reverse_facts = build_reverse_directionality_postmortem(trades, core_groups, reverse_group, primary_timeframe)
    breakdown_outputs, breakdown_facts = build_breakdown_outputs(trades, core_groups)
    horizon, horizon_facts = build_horizon_path_postmortem(events, core_groups)
    feature_bins = build_feature_bin_postmortem(events, trades, core_groups)
    conflict_filter, conflict_facts = build_conflict_filter_impact(events, trades, warnings, core_groups)
    final = build_final_decision(
        summary=summary,
        implementation_facts=implementation_facts,
        session_facts=session_facts,
        random_facts=random_facts,
        seed_facts=seed_facts,
        reverse_facts=reverse_facts,
        breakdown_facts=breakdown_facts,
        horizon_facts=horizon_facts,
        conflict_facts=conflict_facts,
        warnings=warnings,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    write_dataframe(output_dir / "implementation_sanity.csv", implementation_sanity)
    write_dataframe(output_dir / "session_failure_decomposition.csv", session_decomposition)
    write_dataframe(output_dir / "random_control_audit.csv", random_audit)
    write_dataframe(output_dir / "random_control_seed_robustness.csv", seed_robustness)
    write_dataframe(output_dir / "reverse_directionality_postmortem.csv", reverse)
    write_dataframe(output_dir / "postmortem_by_symbol.csv", breakdown_outputs["postmortem_by_symbol"])
    write_dataframe(output_dir / "postmortem_by_direction.csv", breakdown_outputs["postmortem_by_direction"])
    write_dataframe(output_dir / "postmortem_by_timeframe.csv", breakdown_outputs["postmortem_by_timeframe"])
    write_dataframe(output_dir / "postmortem_by_session.csv", breakdown_outputs["postmortem_by_session"])
    write_dataframe(output_dir / "postmortem_by_symbol_direction.csv", breakdown_outputs["postmortem_by_symbol_direction"])
    write_dataframe(output_dir / "postmortem_by_symbol_session.csv", breakdown_outputs["postmortem_by_symbol_session"])
    write_dataframe(output_dir / "horizon_path_postmortem.csv", horizon)
    write_dataframe(output_dir / "feature_bin_postmortem.csv", feature_bins)
    write_dataframe(output_dir / "conflict_filter_impact.csv", conflict_filter)
    (output_dir / "csrb_v1_postmortem_summary.json").write_text(
        json.dumps(clean_json(final), ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    (output_dir / "csrb_v1_postmortem_report.md").write_text(
        render_report(
            final,
            implementation_sanity,
            session_decomposition,
            random_audit,
            seed_robustness,
            reverse,
            breakdown_outputs["postmortem_by_symbol"],
            breakdown_outputs["postmortem_by_direction"],
            breakdown_outputs["postmortem_by_timeframe"],
            breakdown_outputs["postmortem_by_session"],
            conflict_filter,
            primary_timeframe,
        ),
        encoding="utf-8",
    )
    if logger:
        log_event(
            logger,
            logging.INFO,
            "postmortem_complete",
            "CSRB-v1 postmortem complete",
            output_dir=str(output_dir),
            csrb_v1_failed=final.get("csrb_v1_failed"),
            possible_false_breakout_research_hypothesis=final.get("possible_false_breakout_research_hypothesis"),
            random_control_requires_review=final.get("random_control_requires_review"),
            possible_implementation_issue=final.get("possible_implementation_issue"),
        )
    return final


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""

    args = parse_args(argv)
    ensure_headless_runtime()
    logger = setup_logging("postmortem_csrb_v1", verbose=bool(args.verbose))
    run_postmortem(
        research_dir=resolve_path(args.research_dir),
        output_dir=resolve_path(args.output_dir),
        primary_timeframe=str(args.primary_timeframe),
        core_groups=parse_csv_list(args.core_groups),
        reverse_group=str(args.reverse_group),
        random_control_group=str(args.random_control_group),
        logger=logger,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
