#!/usr/bin/env python3
"""Postmortem diagnostics for VSVCB-v1 Phase 1 outputs."""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from common_runtime import PROJECT_ROOT, ensure_headless_runtime, log_event, setup_logging


DEFAULT_RESEARCH_DIR = PROJECT_ROOT / "reports" / "research" / "vsvcb_v1"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "reports" / "research" / "vsvcb_v1_postmortem"
DEFAULT_PRIMARY_SYMBOLS = [
    "BTCUSDT_SWAP_OKX.GLOBAL",
    "ETHUSDT_SWAP_OKX.GLOBAL",
    "SOLUSDT_SWAP_OKX.GLOBAL",
]
DEFAULT_PRIMARY_TIMEFRAME = "15m"
DEFAULT_FOCUS_GROUP = "D"
DEFAULT_REVERSE_GROUP = "E"
SPLITS = ["train", "validation", "oos"]
HORIZONS = [3, 5, 10, 20]
PNL_COLUMNS = ["no_cost_pnl", "cost_aware_pnl", "funding_adjusted_pnl"]
REQUIRED_INPUT_FILES = [
    "events.csv",
    "trades.csv",
    "summary.json",
    "event_group_summary.csv",
    "trade_group_summary.csv",
    "by_symbol.csv",
    "by_timeframe.csv",
    "by_split.csv",
    "concentration.csv",
    "reverse_test.csv",
    "funding_summary.csv",
    "data_quality.json",
]
OUTPUT_CSV_FILES = {
    "implementation_sanity": "implementation_sanity.csv",
    "ablation": "ablation_postmortem.csv",
    "reverse": "reverse_directionality_postmortem.csv",
    "by_symbol": "postmortem_by_symbol.csv",
    "by_direction": "postmortem_by_direction.csv",
    "by_timeframe": "postmortem_by_timeframe.csv",
    "by_symbol_direction": "postmortem_by_symbol_direction.csv",
    "horizon": "horizon_path_postmortem.csv",
    "feature_bins": "feature_bin_postmortem.csv",
    "conflict_filter": "conflict_filter_impact.csv",
}


class VsvcbPostmortemError(Exception):
    """Raised when VSVCB-v1 postmortem cannot continue."""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(
        description="Analyze VSVCB-v1 Phase 1 failure without tuning or strategy development."
    )
    parser.add_argument("--research-dir", default=str(DEFAULT_RESEARCH_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--focus-group", default=DEFAULT_FOCUS_GROUP)
    parser.add_argument("--reverse-group", default=DEFAULT_REVERSE_GROUP)
    parser.add_argument("--primary-timeframe", default=DEFAULT_PRIMARY_TIMEFRAME)
    parser.add_argument("--primary-symbols", default=",".join(DEFAULT_PRIMARY_SYMBOLS))
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def resolve_path(path_arg: str | Path) -> Path:
    """Resolve relative paths from the project root."""

    path = Path(path_arg)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def parse_csv_list(value: str | list[str] | tuple[str, ...]) -> list[str]:
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
            seen.add(item)
            result.append(item)
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


def safe_rate(series: pd.Series) -> float | None:
    """Return mean of a boolean-like series."""

    if series.empty:
        return None
    if series.dtype == bool:
        return float(series.mean())
    normalized = series.map(parse_bool_or_none).dropna()
    if normalized.empty:
        return None
    return float(normalized.astype(bool).mean())


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


def format_number(value: Any, digits: int = 4) -> str:
    """Format optional numeric values for reports."""

    number = finite_number(value, default=None)
    if number is None:
        return "N/A"
    return f"{number:.{digits}f}"


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


def load_artifacts(research_dir: Path) -> tuple[dict[str, Any], list[str]]:
    """Load all Phase 1 artifacts, keeping missing files as warnings."""

    warnings: list[str] = []
    artifacts: dict[str, Any] = {}
    for filename in REQUIRED_INPUT_FILES:
        key = filename.rsplit(".", 1)[0]
        if filename.endswith(".csv"):
            artifacts[key] = read_csv_optional(research_dir, filename, warnings)
        else:
            artifacts[key] = read_json_optional(research_dir, filename, warnings)
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


def primary_frame(trades: pd.DataFrame, focus_group: str, primary_timeframe: str, primary_symbols: list[str]) -> pd.DataFrame:
    """Return the primary D/15m/BTC-ETH-SOL trade frame."""

    if trades.empty:
        return trades.copy()
    out = trades[trades.get("group").astype(str) == focus_group].copy()
    if "timeframe" in out.columns:
        out = out[out["timeframe"].astype(str) == primary_timeframe]
    if primary_symbols and "symbol" in out.columns:
        out = out[out["symbol"].isin(primary_symbols)]
    return out


def opposite_direction(direction: Any) -> str:
    """Return the opposite long/short direction."""

    text = str(direction).lower()
    if text == "long":
        return "short"
    if text == "short":
        return "long"
    return ""


def check_reverse_event_pairing(events: pd.DataFrame, focus_group: str, reverse_group: str) -> dict[str, Any]:
    """Check whether reverse group events are strict D-group reversals."""

    if events.empty or "group" not in events.columns:
        return {
            "status": "unknown",
            "d_event_count": 0,
            "e_event_count": 0,
            "paired_count": 0,
            "opposite_direction_count": 0,
            "strict_reverse": False,
        }
    required = {"timestamp", "symbol", "timeframe", "direction", "group"}
    if not required.issubset(set(events.columns)):
        return {
            "status": "unknown",
            "d_event_count": int((events.get("group", pd.Series(dtype=str)) == focus_group).sum()),
            "e_event_count": int((events.get("group", pd.Series(dtype=str)) == reverse_group).sum()),
            "paired_count": 0,
            "opposite_direction_count": 0,
            "strict_reverse": False,
        }
    key_columns = ["timestamp", "symbol", "timeframe"]
    d_events = events[events["group"].astype(str) == focus_group].loc[:, key_columns + ["direction"]]
    e_events = events[events["group"].astype(str) == reverse_group].loc[:, key_columns + ["direction"]]
    paired = d_events.merge(e_events, on=key_columns, how="inner", suffixes=("_d", "_e"))
    opposite = paired.apply(
        lambda row: str(row["direction_e"]).lower() == opposite_direction(row["direction_d"]),
        axis=1,
    ) if not paired.empty else pd.Series(dtype=bool)
    strict = bool(
        len(d_events.index) > 0
        and len(d_events.index) == len(e_events.index) == len(paired.index)
        and bool(opposite.all())
    )
    return {
        "status": "pass" if strict else "fail",
        "d_event_count": int(len(d_events.index)),
        "e_event_count": int(len(e_events.index)),
        "paired_count": int(len(paired.index)),
        "opposite_direction_count": int(opposite.sum()) if not opposite.empty else 0,
        "strict_reverse": strict,
    }


def count_base_same_bar_direction_conflicts(events: pd.DataFrame, reverse_group: str) -> int:
    """Count same-bar long/short conflicts in non-reverse events."""

    required = {"timestamp", "symbol", "timeframe", "direction", "group"}
    if events.empty or not required.issubset(set(events.columns)):
        return 0
    base = events[events["group"].astype(str) != reverse_group]
    if base.empty:
        return 0
    direction_counts = base.groupby(["timestamp", "symbol", "timeframe"], dropna=False)["direction"].nunique()
    return int((direction_counts > 1).sum())


def build_implementation_sanity(
    *,
    events: pd.DataFrame,
    trades: pd.DataFrame,
    summary: dict[str, Any],
    data_quality: dict[str, Any],
    funding_summary: pd.DataFrame,
    warnings: list[str],
    focus_group: str,
    reverse_group: str,
    primary_timeframe: str,
    primary_symbols: list[str],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build data and implementation sanity diagnostics."""

    market_complete = market_data_complete(summary, data_quality)
    funding_complete = funding_data_complete(summary, data_quality, funding_summary)
    d_events = events[events.get("group", pd.Series(dtype=str)).astype(str) == focus_group] if not events.empty and "group" in events.columns else pd.DataFrame()
    d_trades = trades[trades.get("group", pd.Series(dtype=str)).astype(str) == focus_group] if not trades.empty and "group" in trades.columns else pd.DataFrame()
    primary_trades = primary_frame(trades, focus_group, primary_timeframe, primary_symbols)
    trade_counts = {
        split: int((primary_trades.get("split", pd.Series(dtype=str)).astype(str) == split).sum()) if not primary_trades.empty and "split" in primary_trades.columns else 0
        for split in SPLITS
    }
    trade_count_enough = bool(trade_counts["train"] >= 30 and trade_counts["validation"] >= 10 and trade_counts["oos"] >= 10)

    entry_check_status = "unknown"
    entry_mismatch_count = 0
    entry_checked_count = 0
    if not events.empty and not trades.empty and {"event_id", "entry_next_open"}.issubset(events.columns) and {"event_id", "entry_price"}.issubset(trades.columns):
        joined = d_trades.merge(events[["event_id", "entry_next_open"]], on="event_id", how="left")
        prices = pd.to_numeric(joined["entry_price"], errors="coerce")
        next_open = pd.to_numeric(joined["entry_next_open"], errors="coerce")
        valid = prices.notna() & next_open.notna()
        entry_checked_count = int(valid.sum())
        tolerance = np.maximum(1e-9, prices[valid].abs() * 1e-9)
        entry_mismatch_count = int(((prices[valid] - next_open[valid]).abs() > tolerance).sum())
        entry_check_status = "pass" if entry_checked_count > 0 and entry_mismatch_count == 0 else "fail"

    funding_adjusted_available = bool(
        not trades.empty
        and "funding_adjusted_pnl" in trades.columns
        and numeric_series(trades, "funding_adjusted_pnl").notna().any()
    )
    reverse_pairing = check_reverse_event_pairing(events, focus_group, reverse_group)
    base_conflict_count = count_base_same_bar_direction_conflicts(events, reverse_group)
    skipped_conflict_count = parse_warning_count(warnings, "skipped_events_due_to_single_position_filter")
    skipped_future_count = parse_warning_count(warnings, "skipped_events_with_insufficient_future_bars")
    total_events = int(len(events.index)) if not events.empty else int(sum((summary.get("event_counts") or {}).values()) or 0)
    skipped_conflict_rate = float(skipped_conflict_count / total_events) if total_events else None

    possible_implementation_issue = bool(
        market_complete is False
        or funding_complete is False
        or len(d_events.index) == 0
        or len(d_trades.index) == 0
        or not trade_count_enough
        or entry_check_status == "fail"
        or not funding_adjusted_available
        or reverse_pairing.get("status") == "fail"
        or base_conflict_count > 0
    )

    def status_from_bool(value: bool | None) -> str:
        if value is None:
            return "unknown"
        return "pass" if value else "fail"

    rows = [
        {
            "check_name": "market_data_complete",
            "status": status_from_bool(market_complete),
            "value": market_complete,
            "details": "all_market_data_complete from data_quality.json",
        },
        {
            "check_name": "funding_data_complete",
            "status": status_from_bool(funding_complete),
            "value": funding_complete,
            "details": "actual OKX funding coverage available",
        },
        {
            "check_name": f"{focus_group}_event_count_nonzero",
            "status": "pass" if len(d_events.index) > 0 else "fail",
            "value": int(len(d_events.index)),
            "details": "focus-group events in events.csv",
        },
        {
            "check_name": f"{focus_group}_trade_count_enough",
            "status": "pass" if trade_count_enough else "fail",
            "value": json.dumps(trade_counts, sort_keys=True),
            "details": f"primary {primary_timeframe} trade count gate train>=30 validation>=10 oos>=10",
        },
        {
            "check_name": "entry_uses_next_open",
            "status": entry_check_status,
            "value": entry_mismatch_count,
            "details": f"checked={entry_checked_count}; entry_price must equal event.entry_next_open",
        },
        {
            "check_name": "funding_adjusted_available",
            "status": "pass" if funding_adjusted_available else "fail",
            "value": funding_adjusted_available,
            "details": "trades.csv funding_adjusted_pnl populated",
        },
        {
            "check_name": f"reverse_{reverse_group}_strictly_from_{focus_group}",
            "status": reverse_pairing["status"],
            "value": reverse_pairing.get("strict_reverse"),
            "details": json.dumps(reverse_pairing, sort_keys=True),
        },
        {
            "check_name": "same_bar_long_short_base_anomaly",
            "status": "pass" if base_conflict_count == 0 else "fail",
            "value": base_conflict_count,
            "details": "non-reverse events with both long and short on the same symbol/timeframe/bar",
        },
        {
            "check_name": "skipped_events_due_to_single_position_filter",
            "status": "warning" if skipped_conflict_count else "pass",
            "value": skipped_conflict_count,
            "details": f"rate={format_number(skipped_conflict_rate, 6)}; skipped_future={skipped_future_count}",
        },
        {
            "check_name": "possible_implementation_issue",
            "status": "fail" if possible_implementation_issue else "pass",
            "value": possible_implementation_issue,
            "details": "true only when data completeness or mechanical implementation checks fail",
        },
    ]
    for row in rows:
        row["possible_implementation_issue"] = possible_implementation_issue
    facts = {
        "market_data_complete": market_complete,
        "funding_data_complete": funding_complete,
        "d_event_count": int(len(d_events.index)),
        "d_trade_count": int(len(d_trades.index)),
        "primary_trade_counts": trade_counts,
        "entry_mismatch_count": entry_mismatch_count,
        "entry_checked_count": entry_checked_count,
        "funding_adjusted_available": funding_adjusted_available,
        "reverse_pairing": reverse_pairing,
        "base_same_bar_conflict_count": base_conflict_count,
        "skipped_events_due_to_single_position_filter": skipped_conflict_count,
        "skipped_events_due_to_single_position_filter_rate": skipped_conflict_rate,
        "skipped_events_with_insufficient_future_bars": skipped_future_count,
        "possible_implementation_issue": possible_implementation_issue,
    }
    return pd.DataFrame(rows), facts


def trade_metric_row(frame: pd.DataFrame) -> dict[str, Any]:
    """Build common trade metrics."""

    count = int(len(frame.index))
    no_cost = safe_sum(frame, "no_cost_pnl")
    cost = safe_sum(frame, "cost_aware_pnl")
    funding = safe_sum(frame, "funding_adjusted_pnl")
    return {
        "trade_count": count,
        "long_count": int((frame.get("direction", pd.Series(dtype=str)).astype(str) == "long").sum()) if not frame.empty and "direction" in frame.columns else 0,
        "short_count": int((frame.get("direction", pd.Series(dtype=str)).astype(str) == "short").sum()) if not frame.empty and "direction" in frame.columns else 0,
        "no_cost_pnl": no_cost,
        "cost_aware_pnl": cost,
        "funding_adjusted_pnl": funding,
        "avg_no_cost_pnl": float(no_cost / count) if count else None,
        "avg_funding_adjusted_pnl": float(funding / count) if count else None,
        "win_rate": float((numeric_series(frame, "funding_adjusted_pnl") > 0).mean()) if count and "funding_adjusted_pnl" in frame.columns else None,
        "sample_sufficient": bool(count >= 30),
    }


def split_values(trades: pd.DataFrame) -> list[str]:
    """Return split labels plus all."""

    values = []
    if not trades.empty and "split" in trades.columns:
        values = [split for split in SPLITS if split in set(trades["split"].astype(str))]
        extras = sorted(set(trades["split"].astype(str)) - set(SPLITS))
        values.extend(extras)
    return values + ["all"]


def filter_split(frame: pd.DataFrame, split: str) -> pd.DataFrame:
    """Filter a frame for one split, preserving all rows when split=all."""

    if split == "all" or frame.empty or "split" not in frame.columns:
        return frame.copy()
    return frame[frame["split"].astype(str) == split].copy()


def build_ablation_postmortem(
    trades: pd.DataFrame,
    focus_group: str,
    primary_timeframe: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build D versus A/B/C ablation postmortem rows."""

    columns = [
        "timeframe",
        "split",
        "baseline_group",
        "d_trade_count",
        "baseline_trade_count",
        "d_no_cost_pnl",
        "baseline_no_cost_pnl",
        "d_cost_aware_pnl",
        "baseline_cost_aware_pnl",
        "d_funding_adjusted_pnl",
        "baseline_funding_adjusted_pnl",
        "d_avg_no_cost_pnl",
        "baseline_avg_no_cost_pnl",
        "d_positive",
        "d_better_than_baseline",
        "d_less_negative_but_still_negative",
        "improvement_source",
        "nearest_to_d_among_abc",
        "squeeze_incremental_no_cost_pnl",
        "volume_incremental_no_cost_pnl",
        "stronger_incremental_component",
        "ablation_pass_but_no_edge",
    ]
    if trades.empty or "group" not in trades.columns:
        return pd.DataFrame(columns=columns), {
            "d_any_positive_split_timeframe": False,
            "ablation_pass_but_no_edge": False,
            "nearest_to_d_primary": None,
        }

    rows: list[dict[str, Any]] = []
    timeframes = sorted(trades["timeframe"].dropna().astype(str).unique()) if "timeframe" in trades.columns else ["all"]
    for timeframe in timeframes:
        frame_tf = trades[trades["timeframe"].astype(str) == timeframe] if "timeframe" in trades.columns else trades
        for split in split_values(frame_tf):
            frame = filter_split(frame_tf, split)
            by_group = {group: frame[frame["group"].astype(str) == group] for group in ["A", "B", "C", focus_group]}
            d = by_group[focus_group]
            d_metrics = trade_metric_row(d)
            group_pnls = {
                group: safe_sum(group_frame, "no_cost_pnl")
                for group, group_frame in by_group.items()
                if group != focus_group
            }
            nearest = min(group_pnls, key=lambda group: abs(d_metrics["no_cost_pnl"] - group_pnls[group])) if group_pnls else None
            squeeze_increment = d_metrics["no_cost_pnl"] - group_pnls.get("C", 0.0)
            volume_increment = d_metrics["no_cost_pnl"] - group_pnls.get("B", 0.0)
            if squeeze_increment > volume_increment:
                stronger_component = "squeeze_conditional_on_volume"
            elif volume_increment > squeeze_increment:
                stronger_component = "volume_conditional_on_squeeze"
            else:
                stronger_component = "tie"
            for baseline in ["A", "B", "C"]:
                baseline_metrics = trade_metric_row(by_group[baseline])
                d_better = bool(d_metrics["no_cost_pnl"] > baseline_metrics["no_cost_pnl"])
                d_positive = bool(d_metrics["no_cost_pnl"] > 0)
                if d_better and d_metrics["avg_no_cost_pnl"] is not None and baseline_metrics["avg_no_cost_pnl"] is not None and d_metrics["avg_no_cost_pnl"] > baseline_metrics["avg_no_cost_pnl"]:
                    source = "signal_quality_or_payoff_improvement"
                elif d_better and d_metrics["trade_count"] < baseline_metrics["trade_count"]:
                    source = "fewer_trades_less_loss"
                elif d_better:
                    source = "mixed_or_ambiguous"
                else:
                    source = "no_improvement"
                rows.append(
                    {
                        "timeframe": timeframe,
                        "split": split,
                        "baseline_group": baseline,
                        "d_trade_count": d_metrics["trade_count"],
                        "baseline_trade_count": baseline_metrics["trade_count"],
                        "d_no_cost_pnl": d_metrics["no_cost_pnl"],
                        "baseline_no_cost_pnl": baseline_metrics["no_cost_pnl"],
                        "d_cost_aware_pnl": d_metrics["cost_aware_pnl"],
                        "baseline_cost_aware_pnl": baseline_metrics["cost_aware_pnl"],
                        "d_funding_adjusted_pnl": d_metrics["funding_adjusted_pnl"],
                        "baseline_funding_adjusted_pnl": baseline_metrics["funding_adjusted_pnl"],
                        "d_avg_no_cost_pnl": d_metrics["avg_no_cost_pnl"],
                        "baseline_avg_no_cost_pnl": baseline_metrics["avg_no_cost_pnl"],
                        "d_positive": d_positive,
                        "d_better_than_baseline": d_better,
                        "d_less_negative_but_still_negative": bool(d_better and not d_positive),
                        "improvement_source": source,
                        "nearest_to_d_among_abc": nearest,
                        "squeeze_incremental_no_cost_pnl": squeeze_increment,
                        "volume_incremental_no_cost_pnl": volume_increment,
                        "stronger_incremental_component": stronger_component,
                        "ablation_pass_but_no_edge": bool(d_better and not d_positive),
                    }
                )
    result = pd.DataFrame(rows, columns=columns)
    d_rows = result.drop_duplicates(["timeframe", "split"]) if not result.empty else pd.DataFrame()
    facts = {
        "d_any_positive_split_timeframe": bool((d_rows.get("d_no_cost_pnl", pd.Series(dtype=float)) > 0).any()) if not d_rows.empty else False,
        "ablation_pass_but_no_edge": bool((result["d_less_negative_but_still_negative"]).any()) if not result.empty else False,
        "nearest_to_d_primary": (
            result[(result["timeframe"] == primary_timeframe) & (result["split"] == "oos")]["nearest_to_d_among_abc"].dropna().iloc[0]
            if not result[(result["timeframe"] == primary_timeframe) & (result["split"] == "oos")].empty
            else None
        ) if not result.empty else None,
    }
    return result, facts


def enrich_trades_with_events(trades: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """Attach event timestamp and original event metadata to trades."""

    if trades.empty:
        return trades.copy()
    out = trades.copy()
    if events.empty or "event_id" not in events.columns or "event_id" not in trades.columns:
        return out
    columns = [
        column
        for column in ["event_id", "timestamp", "symbol", "timeframe", "direction", "group"]
        if column in events.columns
    ]
    event_meta = events.loc[:, columns].rename(
        columns={
            "timestamp": "event_timestamp",
            "direction": "event_direction",
            "group": "event_group",
        }
    )
    return out.merge(event_meta, on="event_id", how="left", suffixes=("", "_event"))


def pair_reverse_trades(trades: pd.DataFrame, events: pd.DataFrame, focus_group: str, reverse_group: str) -> pd.DataFrame:
    """Pair D and E trades by event timestamp/symbol/timeframe/split."""

    enriched = enrich_trades_with_events(trades, events)
    required = {"group", "symbol", "timeframe", "split", "event_timestamp"}
    if enriched.empty or not required.issubset(set(enriched.columns)):
        return pd.DataFrame()
    key_columns = ["event_timestamp", "symbol", "timeframe", "split"]
    d = enriched[enriched["group"].astype(str) == focus_group].copy()
    e = enriched[enriched["group"].astype(str) == reverse_group].copy()
    if d.empty or e.empty:
        return pd.DataFrame()
    return d.merge(e, on=key_columns, how="inner", suffixes=("_d", "_e"))


def reverse_comparison_row(
    *,
    dimension: str,
    timeframe: str,
    split: str,
    direction: str,
    d: pd.DataFrame,
    e: pd.DataFrame,
) -> dict[str, Any]:
    """Build one reverse comparison row."""

    d_no = safe_sum(d, "no_cost_pnl")
    e_no = safe_sum(e, "no_cost_pnl")
    d_cost = safe_sum(d, "cost_aware_pnl")
    e_cost = safe_sum(e, "cost_aware_pnl")
    d_funding = safe_sum(d, "funding_adjusted_pnl")
    e_funding = safe_sum(e, "funding_adjusted_pnl")
    return {
        "dimension": dimension,
        "timeframe": timeframe,
        "split": split,
        "direction": direction,
        "d_trade_count": int(len(d.index)),
        "e_trade_count": int(len(e.index)),
        "d_no_cost_pnl": d_no,
        "e_no_cost_pnl": e_no,
        "d_cost_aware_pnl": d_cost,
        "e_cost_aware_pnl": e_cost,
        "d_funding_adjusted_pnl": d_funding,
        "e_funding_adjusted_pnl": e_funding,
        "e_better_no_cost": bool(e_no > d_no),
        "e_better_cost_aware": bool(e_cost > d_cost),
        "e_better_funding_adjusted": bool(e_funding > d_funding),
        "e_cost_after_cost_positive": bool(e_cost > 0),
        "e_funding_adjusted_positive": bool(e_funding > 0),
        "mechanical_no_cost_inverse": bool(abs(d_no + e_no) <= 1e-6 * max(1.0, abs(d_no), abs(e_no))),
        "reverse_test_failure": bool(e_no > d_no or e_funding > d_funding),
    }


def build_reverse_directionality_postmortem(
    trades: pd.DataFrame,
    events: pd.DataFrame,
    reverse_test: pd.DataFrame,
    focus_group: str,
    reverse_group: str,
    primary_timeframe: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build reverse E directionality diagnostics."""

    columns = [
        "dimension",
        "timeframe",
        "split",
        "direction",
        "d_trade_count",
        "e_trade_count",
        "d_no_cost_pnl",
        "e_no_cost_pnl",
        "d_cost_aware_pnl",
        "e_cost_aware_pnl",
        "d_funding_adjusted_pnl",
        "e_funding_adjusted_pnl",
        "e_better_no_cost",
        "e_better_cost_aware",
        "e_better_funding_adjusted",
        "e_cost_after_cost_positive",
        "e_funding_adjusted_positive",
        "mechanical_no_cost_inverse",
        "reverse_test_failure",
        "possible_false_breakout_research_hypothesis",
        "trend_following_hypothesis_failed",
    ]
    rows: list[dict[str, Any]] = []
    if not reverse_test.empty and {"timeframe", "split"}.issubset(reverse_test.columns):
        for row in reverse_test.itertuples(index=False):
            item = {
                "dimension": "timeframe_split",
                "timeframe": getattr(row, "timeframe", "all"),
                "split": getattr(row, "split", "all"),
                "direction": "all",
                "d_trade_count": int(getattr(row, "d_trade_count", 0) or 0),
                "e_trade_count": int(getattr(row, "e_trade_count", 0) or 0),
                "d_no_cost_pnl": finite_number(getattr(row, "d_no_cost_pnl", 0.0), 0.0),
                "e_no_cost_pnl": finite_number(getattr(row, "e_no_cost_pnl", 0.0), 0.0),
                "d_cost_aware_pnl": finite_number(getattr(row, "d_cost_aware_pnl", 0.0), 0.0),
                "e_cost_aware_pnl": finite_number(getattr(row, "e_cost_aware_pnl", 0.0), 0.0),
                "d_funding_adjusted_pnl": finite_number(getattr(row, "d_funding_adjusted_pnl", 0.0), 0.0),
                "e_funding_adjusted_pnl": finite_number(getattr(row, "e_funding_adjusted_pnl", 0.0), 0.0),
            }
            item["e_better_no_cost"] = bool(item["e_no_cost_pnl"] > item["d_no_cost_pnl"])
            item["e_better_cost_aware"] = bool(item["e_cost_aware_pnl"] > item["d_cost_aware_pnl"])
            item["e_better_funding_adjusted"] = bool(item["e_funding_adjusted_pnl"] > item["d_funding_adjusted_pnl"])
            item["e_cost_after_cost_positive"] = bool(item["e_cost_aware_pnl"] > 0)
            item["e_funding_adjusted_positive"] = bool(item["e_funding_adjusted_pnl"] > 0)
            item["mechanical_no_cost_inverse"] = bool(abs(item["d_no_cost_pnl"] + item["e_no_cost_pnl"]) <= 1e-6 * max(1.0, abs(item["d_no_cost_pnl"]), abs(item["e_no_cost_pnl"])))
            item["reverse_test_failure"] = bool(item["e_better_no_cost"] or item["e_better_funding_adjusted"])
            rows.append(item)
    elif not trades.empty and "group" in trades.columns:
        timeframes = sorted(trades["timeframe"].dropna().astype(str).unique()) if "timeframe" in trades.columns else ["all"]
        for timeframe in timeframes:
            frame_tf = trades[trades["timeframe"].astype(str) == timeframe] if "timeframe" in trades.columns else trades
            for split in split_values(frame_tf):
                frame = filter_split(frame_tf, split)
                rows.append(
                    reverse_comparison_row(
                        dimension="timeframe_split",
                        timeframe=timeframe,
                        split=split,
                        direction="all",
                        d=frame[frame["group"].astype(str) == focus_group],
                        e=frame[frame["group"].astype(str) == reverse_group],
                    )
                )

    paired = pair_reverse_trades(trades, events, focus_group, reverse_group)
    if not paired.empty:
        for direction in sorted(paired["direction_d"].dropna().astype(str).unique()):
            frame = paired[paired["direction_d"].astype(str) == direction]
            d = frame.rename(
                columns={
                    "no_cost_pnl_d": "no_cost_pnl",
                    "cost_aware_pnl_d": "cost_aware_pnl",
                    "funding_adjusted_pnl_d": "funding_adjusted_pnl",
                }
            )
            e = frame.rename(
                columns={
                    "no_cost_pnl_e": "no_cost_pnl",
                    "cost_aware_pnl_e": "cost_aware_pnl",
                    "funding_adjusted_pnl_e": "funding_adjusted_pnl",
                }
            )
            rows.append(
                reverse_comparison_row(
                    dimension="original_d_direction",
                    timeframe="all",
                    split="all",
                    direction=direction,
                    d=d,
                    e=e,
                )
            )
    elif not trades.empty and {"group", "direction"}.issubset(trades.columns):
        for direction in ["long", "short"]:
            d = trades[(trades["group"].astype(str) == focus_group) & (trades["direction"].astype(str) == direction)]
            e = trades[(trades["group"].astype(str) == reverse_group) & (trades["direction"].astype(str) == opposite_direction(direction))]
            rows.append(
                reverse_comparison_row(
                    dimension="original_d_direction",
                    timeframe="all",
                    split="all",
                    direction=direction,
                    d=d,
                    e=e,
                )
            )

    result = pd.DataFrame(rows, columns=[column for column in columns if column not in {"possible_false_breakout_research_hypothesis", "trend_following_hypothesis_failed"}])
    primary_rows = result[
        (result["dimension"] == "timeframe_split")
        & (result["timeframe"].astype(str) == primary_timeframe)
        & (result["split"].astype(str).isin(SPLITS))
    ] if not result.empty else pd.DataFrame()
    oos_primary = result[
        (result["dimension"] == "timeframe_split")
        & (result["timeframe"].astype(str) == primary_timeframe)
        & (result["split"].astype(str) == "oos")
    ] if not result.empty else pd.DataFrame()
    e_better_all_primary_splits = bool(not primary_rows.empty and primary_rows["e_better_funding_adjusted"].astype(bool).all())
    e_better_all_timeframes = False
    timeframe_all = result[
        (result["dimension"] == "timeframe_split")
        & (result["split"].astype(str) == "all")
    ] if not result.empty else pd.DataFrame()
    if not timeframe_all.empty:
        e_better_all_timeframes = bool(timeframe_all["e_better_funding_adjusted"].astype(bool).all())
    direction_rows = result[result["dimension"] == "original_d_direction"] if not result.empty else pd.DataFrame()
    e_better_all_directions = bool(not direction_rows.empty and direction_rows["e_better_funding_adjusted"].astype(bool).all())
    e_oos_cost_positive = bool(not oos_primary.empty and bool(oos_primary.iloc[0]["e_cost_after_cost_positive"]))
    reverse_test_failure = bool(not oos_primary.empty and bool(oos_primary.iloc[0]["reverse_test_failure"]))
    possible_false_breakout = bool(
        reverse_test_failure
        and e_oos_cost_positive
        and (e_better_all_primary_splits or e_better_all_timeframes or e_better_all_directions)
    )
    trend_failed = bool(reverse_test_failure)
    if not result.empty:
        result["possible_false_breakout_research_hypothesis"] = possible_false_breakout
        result["trend_following_hypothesis_failed"] = trend_failed
    else:
        result = pd.DataFrame(columns=columns)
    facts = {
        "reverse_test_failure": reverse_test_failure,
        "e_better_all_primary_splits": e_better_all_primary_splits,
        "e_better_all_timeframes": e_better_all_timeframes,
        "e_better_all_directions": e_better_all_directions,
        "e_oos_cost_after_cost_positive": e_oos_cost_positive,
        "possible_false_breakout_research_hypothesis": possible_false_breakout,
        "trend_following_hypothesis_failed": trend_failed,
    }
    return result.loc[:, columns], facts


def summarize_focus_trades(trades: pd.DataFrame, group_columns: list[str], focus_group: str) -> pd.DataFrame:
    """Summarize focus-group trades by requested columns."""

    base_columns = group_columns + [
        "trade_count",
        "long_count",
        "short_count",
        "no_cost_pnl",
        "cost_aware_pnl",
        "funding_adjusted_pnl",
        "avg_no_cost_pnl",
        "avg_funding_adjusted_pnl",
        "win_rate",
        "sample_sufficient",
        "drag_rank",
        "largest_drag",
    ]
    if trades.empty or "group" not in trades.columns:
        return pd.DataFrame(columns=base_columns)
    focus = trades[trades["group"].astype(str) == focus_group].copy()
    if focus.empty:
        return pd.DataFrame(columns=base_columns)
    rows: list[dict[str, Any]] = []
    grouped = focus.groupby(group_columns, dropna=False, sort=True) if group_columns else [((), focus)]
    for keys, frame in grouped:
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {column: key for column, key in zip(group_columns, keys)}
        row.update(trade_metric_row(frame))
        rows.append(row)
    result = pd.DataFrame(rows)
    if result.empty:
        return pd.DataFrame(columns=base_columns)
    result = result.sort_values("funding_adjusted_pnl", kind="stable").reset_index(drop=True)
    result["drag_rank"] = np.arange(1, len(result.index) + 1)
    result["largest_drag"] = result["drag_rank"] == 1
    return result.loc[:, base_columns]


def build_symbol_direction_timeframe_postmortems(
    trades: pd.DataFrame,
    focus_group: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Build symbol/direction/timeframe decomposition outputs."""

    by_symbol = summarize_focus_trades(trades, ["symbol"], focus_group)
    by_direction = summarize_focus_trades(trades, ["direction"], focus_group)
    by_timeframe = summarize_focus_trades(trades, ["timeframe"], focus_group)
    by_symbol_direction = summarize_focus_trades(trades, ["symbol", "direction", "timeframe"], focus_group)

    worst_symbol = by_symbol.iloc[0]["symbol"] if not by_symbol.empty else None
    worst_direction = by_direction.iloc[0]["direction"] if not by_direction.empty else None
    positive_timeframes = by_timeframe[by_timeframe["funding_adjusted_pnl"] > 0]["timeframe"].tolist() if not by_timeframe.empty else []
    facts = {
        "worst_symbol": worst_symbol,
        "worst_direction": worst_direction,
        "positive_timeframes": positive_timeframes,
        "any_positive_timeframe": bool(positive_timeframes),
        "local_positive_symbol_timeframe_rows": dataframe_records(
            by_symbol_direction[by_symbol_direction["funding_adjusted_pnl"] > 0]
            if not by_symbol_direction.empty
            else pd.DataFrame()
        ),
    }
    return by_symbol, by_direction, by_timeframe, by_symbol_direction, facts


def event_metric_row(frame: pd.DataFrame) -> dict[str, Any]:
    """Build common event path metrics."""

    row: dict[str, Any] = {"event_count": int(len(frame.index))}
    returns: dict[int, float | None] = {}
    for horizon in HORIZONS:
        mean_return = safe_mean(frame, f"future_return_{horizon}")
        returns[horizon] = mean_return
        row[f"mean_future_return_{horizon}"] = mean_return
        reversal_col = f"reversal_flag_{horizon}"
        row[f"reversal_rate_{horizon}"] = safe_rate(frame[reversal_col]) if reversal_col in frame.columns else None
    row["mean_mfe_10"] = safe_mean(frame, "mfe_10")
    mae = numeric_series(frame, "mae_10").dropna()
    row["mean_abs_mae_10"] = float(mae.abs().mean()) if not mae.empty else None
    row["mean_mfe_mae_ratio_10"] = safe_mean(frame, "mfe_mae_ratio_10")
    valid_returns = {horizon: value for horizon, value in returns.items() if value is not None}
    row["best_horizon"] = max(valid_returns, key=valid_returns.get) if valid_returns else None
    r3 = returns.get(3)
    r5 = returns.get(5)
    r10 = returns.get(10)
    r20 = returns.get(20)
    row["horizon_3_better_than_10"] = bool(r3 is not None and r10 is not None and r3 > r10)
    row["horizon_5_better_than_10"] = bool(r5 is not None and r10 is not None and r5 > r10)
    row["horizon_20_worse_than_10"] = bool(r20 is not None and r10 is not None and r20 < r10)
    row["early_reversal_likely"] = bool((r3 is not None and r3 < 0) and (row.get("reversal_rate_3") or 0) > 0.5)
    row["fixed_hold_decay_likely"] = bool(r3 is not None and r10 is not None and r3 > r10)
    mean_mfe = row.get("mean_mfe_10")
    mean_abs_mae = row.get("mean_abs_mae_10")
    row["mfe_exists_but_fixed_hold_wasted"] = bool(mean_mfe is not None and r10 is not None and mean_mfe > 0 and r10 < 0)
    row["mae_dominates_mfe"] = bool(mean_mfe is not None and mean_abs_mae is not None and mean_abs_mae > mean_mfe)
    return row


def build_horizon_path_postmortem(events: pd.DataFrame, focus_group: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build horizon path diagnostics from events.csv."""

    columns = [
        "timeframe",
        "split",
        "direction",
        "event_count",
        "mean_future_return_3",
        "mean_future_return_5",
        "mean_future_return_10",
        "mean_future_return_20",
        "reversal_rate_3",
        "reversal_rate_5",
        "reversal_rate_10",
        "mean_mfe_10",
        "mean_abs_mae_10",
        "mean_mfe_mae_ratio_10",
        "best_horizon",
        "horizon_3_better_than_10",
        "horizon_5_better_than_10",
        "horizon_20_worse_than_10",
        "early_reversal_likely",
        "fixed_hold_decay_likely",
        "mfe_exists_but_fixed_hold_wasted",
        "mae_dominates_mfe",
    ]
    if events.empty or "group" not in events.columns:
        return pd.DataFrame(columns=columns), {}
    focus = events[events["group"].astype(str) == focus_group].copy()
    if focus.empty:
        return pd.DataFrame(columns=columns), {}
    rows: list[dict[str, Any]] = []
    group_cols = [column for column in ["timeframe", "split", "direction"] if column in focus.columns]
    grouped = focus.groupby(group_cols, dropna=False, sort=True) if group_cols else [((), focus)]
    for keys, frame in grouped:
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {"timeframe": "all", "split": "all", "direction": "all"}
        row.update({column: key for column, key in zip(group_cols, keys)})
        row.update(event_metric_row(frame))
        rows.append(row)
    all_row = {"timeframe": "all", "split": "all", "direction": "all"}
    all_row.update(event_metric_row(focus))
    rows.append(all_row)
    result = pd.DataFrame(rows, columns=columns)
    overall = result[(result["timeframe"] == "all") & (result["split"] == "all") & (result["direction"] == "all")]
    facts = dataframe_records(overall, limit=1)[0] if not overall.empty else {}
    return result, facts


def bin_numeric_feature(series: pd.Series, bins: int = 5) -> pd.Series:
    """Bin a numeric feature for diagnostics."""

    numeric = pd.to_numeric(series, errors="coerce")
    valid = numeric.dropna()
    if valid.empty:
        return pd.Series(["missing"] * len(series.index), index=series.index, dtype=object)
    unique_count = int(valid.nunique())
    if unique_count < 2:
        labels = numeric.map(lambda value: "missing" if pd.isna(value) else f"value={format_number(value, 6)}")
        return labels.astype(object)
    try:
        binned = pd.qcut(numeric, q=min(bins, unique_count), duplicates="drop")
    except ValueError:
        binned = pd.cut(numeric, bins=min(bins, unique_count), duplicates="drop")
    return binned.astype(str).replace("nan", "missing")


def build_feature_bin_postmortem(events: pd.DataFrame, trades: pd.DataFrame, focus_group: str) -> pd.DataFrame:
    """Build feature-bin diagnostics for D events without tuning decisions."""

    columns = [
        "feature",
        "bin",
        "event_count",
        "trade_count",
        "no_cost_pnl",
        "cost_aware_pnl",
        "funding_adjusted_pnl",
        "mean_future_return_10",
        "reversal_rate_10",
        "mfe_mae_ratio_10",
        "no_parameter_tuning_allowed",
    ]
    if events.empty or "group" not in events.columns:
        return pd.DataFrame(columns=columns)
    focus = events[events["group"].astype(str) == focus_group].copy()
    if focus.empty:
        return pd.DataFrame(columns=columns)
    trade_cols = [
        column
        for column in ["event_id", "trade_id", "no_cost_pnl", "cost_aware_pnl", "funding_adjusted_pnl"]
        if column in trades.columns
    ] if not trades.empty else []
    if trade_cols and "event_id" in focus.columns:
        merged = focus.merge(trades.loc[:, trade_cols], on="event_id", how="left")
    else:
        merged = focus.copy()
        for column in ["trade_id", "no_cost_pnl", "cost_aware_pnl", "funding_adjusted_pnl"]:
            merged[column] = np.nan

    rows: list[dict[str, Any]] = []
    features = ["bb_width_percentile", "volume_ratio", "atr", "timeframe", "direction", "symbol"]
    for feature in features:
        if feature not in merged.columns:
            continue
        working = merged.copy()
        if feature in {"bb_width_percentile", "volume_ratio", "atr"}:
            working["_bin"] = bin_numeric_feature(working[feature])
        else:
            working["_bin"] = working[feature].fillna("missing").astype(str)
        for bin_value, frame in working.groupby("_bin", dropna=False, sort=True):
            rows.append(
                {
                    "feature": feature,
                    "bin": str(bin_value),
                    "event_count": int(len(frame.index)),
                    "trade_count": int(frame["trade_id"].notna().sum()) if "trade_id" in frame.columns else 0,
                    "no_cost_pnl": safe_sum(frame, "no_cost_pnl"),
                    "cost_aware_pnl": safe_sum(frame, "cost_aware_pnl"),
                    "funding_adjusted_pnl": safe_sum(frame, "funding_adjusted_pnl"),
                    "mean_future_return_10": safe_mean(frame, "future_return_10"),
                    "reversal_rate_10": safe_rate(frame["reversal_flag_10"]) if "reversal_flag_10" in frame.columns else None,
                    "mfe_mae_ratio_10": safe_mean(frame, "mfe_mae_ratio_10"),
                    "no_parameter_tuning_allowed": True,
                }
            )
    return pd.DataFrame(rows, columns=columns)


def build_conflict_filter_impact(
    events: pd.DataFrame,
    trades: pd.DataFrame,
    warnings: list[str],
    focus_group: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build single-position conflict-filter impact diagnostics."""

    columns = [
        "scope",
        "timeframe",
        "symbol",
        "d_event_count",
        "d_trade_count",
        "untraded_event_count",
        "untraded_event_rate",
        "global_skipped_events_due_to_single_position_filter",
        "global_skipped_events_with_insufficient_future_bars",
        "traded_mean_future_return_10",
        "untraded_mean_future_return_10",
        "traded_reversal_rate_10",
        "untraded_reversal_rate_10",
        "actual_no_cost_pnl",
        "actual_cost_aware_pnl",
        "actual_funding_adjusted_pnl",
        "filtered_events_theoretically_better",
        "single_position_filter_may_distort_result",
    ]
    if events.empty or "group" not in events.columns:
        return pd.DataFrame(columns=columns), {}
    d_events = events[events["group"].astype(str) == focus_group].copy()
    d_trades = trades[trades["group"].astype(str) == focus_group].copy() if not trades.empty and "group" in trades.columns else pd.DataFrame()
    traded_ids = set(d_trades["event_id"].astype(str)) if not d_trades.empty and "event_id" in d_trades.columns else set()
    if "event_id" in d_events.columns:
        d_events["_traded"] = d_events["event_id"].astype(str).isin(traded_ids)
    else:
        d_events["_traded"] = False
    skipped_conflict = parse_warning_count(warnings, "skipped_events_due_to_single_position_filter")
    skipped_future = parse_warning_count(warnings, "skipped_events_with_insufficient_future_bars")

    rows: list[dict[str, Any]] = []

    def add_scope(scope: str, frame: pd.DataFrame, trade_frame: pd.DataFrame, timeframe: str = "all", symbol: str = "all") -> None:
        traded_events = frame[frame["_traded"]]
        untraded_events = frame[~frame["_traded"]]
        event_count = int(len(frame.index))
        untraded_rate = float(len(untraded_events.index) / event_count) if event_count else None
        traded_mean = safe_mean(traded_events, "future_return_10")
        untraded_mean = safe_mean(untraded_events, "future_return_10")
        traded_rev = safe_rate(traded_events["reversal_flag_10"]) if "reversal_flag_10" in traded_events.columns else None
        untraded_rev = safe_rate(untraded_events["reversal_flag_10"]) if "reversal_flag_10" in untraded_events.columns else None
        filtered_better = bool(
            untraded_mean is not None
            and traded_mean is not None
            and untraded_mean > traded_mean
        )
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
                "d_event_count": event_count,
                "d_trade_count": int(len(trade_frame.index)),
                "untraded_event_count": int(len(untraded_events.index)),
                "untraded_event_rate": untraded_rate,
                "global_skipped_events_due_to_single_position_filter": skipped_conflict,
                "global_skipped_events_with_insufficient_future_bars": skipped_future,
                "traded_mean_future_return_10": traded_mean,
                "untraded_mean_future_return_10": untraded_mean,
                "traded_reversal_rate_10": traded_rev,
                "untraded_reversal_rate_10": untraded_rev,
                "actual_no_cost_pnl": safe_sum(trade_frame, "no_cost_pnl"),
                "actual_cost_aware_pnl": safe_sum(trade_frame, "cost_aware_pnl"),
                "actual_funding_adjusted_pnl": safe_sum(trade_frame, "funding_adjusted_pnl"),
                "filtered_events_theoretically_better": filtered_better,
                "single_position_filter_may_distort_result": distortion,
            }
        )

    add_scope("all_d", d_events, d_trades)
    if "timeframe" in d_events.columns:
        for timeframe, frame in d_events.groupby("timeframe", dropna=False, sort=True):
            trade_frame = d_trades[d_trades["timeframe"].astype(str) == str(timeframe)] if not d_trades.empty and "timeframe" in d_trades.columns else pd.DataFrame()
            add_scope("timeframe", frame, trade_frame, timeframe=str(timeframe))
    if "symbol" in d_events.columns:
        for symbol, frame in d_events.groupby("symbol", dropna=False, sort=True):
            trade_frame = d_trades[d_trades["symbol"].astype(str) == str(symbol)] if not d_trades.empty and "symbol" in d_trades.columns else pd.DataFrame()
            add_scope("symbol", frame, trade_frame, symbol=str(symbol))
    result = pd.DataFrame(rows, columns=columns)
    facts = dataframe_records(result[result["scope"] == "all_d"], limit=1)[0] if not result.empty else {}
    return result, facts


def build_final_decision(
    *,
    summary: dict[str, Any],
    implementation_facts: dict[str, Any],
    ablation_facts: dict[str, Any],
    reverse_facts: dict[str, Any],
    decomposition_facts: dict[str, Any],
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
                "ablation_pass",
            ]
        )
    )
    trend_failed = bool(phase1_failed and (reverse_facts.get("trend_following_hypothesis_failed") or not summary.get("oos_pass", False)))
    possible_false_breakout = bool(reverse_facts.get("possible_false_breakout_research_hypothesis"))
    possible_issue = bool(implementation_facts.get("possible_implementation_issue"))
    data_issue = bool(
        implementation_facts.get("market_data_complete") is False
        or implementation_facts.get("funding_data_complete") is False
    )
    if possible_issue:
        recommended = (
            "Resolve data or implementation warnings first; do not tune VSVCB-v1 and do not enter Phase 2."
        )
    elif possible_false_breakout:
        recommended = (
            "Archive VSVCB-v1 trend-following. A separate pre-registered false-breakout research hypothesis may be proposed, but E is not a tradable strategy and cannot rescue Phase 1."
        )
    else:
        recommended = (
            "Archive VSVCB-v1. Do not run Phase 2, parameter plateau, randomization, strategy development, demo, or live."
        )
    return {
        "hypothesis_name": "Volatility Squeeze with Volume Confirmation Breakout",
        "version": "v1-postmortem",
        "research_only": True,
        "vsvcb_v1_failed": True,
        "phase1_failed": phase1_failed,
        "trend_following_hypothesis_failed": trend_failed,
        "possible_false_breakout_research_hypothesis": possible_false_breakout,
        "separate_false_breakout_research_allowed": bool(possible_false_breakout and not possible_issue),
        "continue_to_phase2": False,
        "parameter_plateau_allowed": False,
        "randomization_allowed": False,
        "strategy_development_allowed": False,
        "demo_live_allowed": False,
        "official_strategy_modification_allowed": False,
        "no_parameter_tuning_allowed": True,
        "reverse_test_failure": bool(reverse_facts.get("reverse_test_failure")),
        "possible_implementation_issue": possible_issue,
        "data_or_implementation_issue": bool(possible_issue or data_issue),
        "market_data_complete": implementation_facts.get("market_data_complete"),
        "funding_data_complete": implementation_facts.get("funding_data_complete"),
        "d_group": {
            "event_count": implementation_facts.get("d_event_count"),
            "trade_count": implementation_facts.get("d_trade_count"),
            "primary_trade_counts": implementation_facts.get("primary_trade_counts"),
            "train_no_cost_pnl": gates.get("train_no_cost_pnl"),
            "validation_no_cost_pnl": gates.get("validation_no_cost_pnl"),
            "oos_no_cost_pnl": gates.get("oos_no_cost_pnl"),
            "oos_cost_aware_pnl": gates.get("oos_cost_aware_pnl"),
            "oos_funding_adjusted_pnl": gates.get("oos_funding_adjusted_pnl"),
            "ablation_pass_but_no_edge": ablation_facts.get("ablation_pass_but_no_edge"),
        },
        "reverse_e": reverse_facts,
        "decomposition": decomposition_facts,
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
    ablation: pd.DataFrame,
    reverse: pd.DataFrame,
    by_symbol: pd.DataFrame,
    by_direction: pd.DataFrame,
    by_timeframe: pd.DataFrame,
    horizon: pd.DataFrame,
    conflict_filter: pd.DataFrame,
    primary_timeframe: str,
) -> str:
    """Render the VSVCB-v1 postmortem Markdown report."""

    d_group = final.get("d_group") or {}
    reverse_e = final.get("reverse_e") or {}
    decomposition = final.get("decomposition") or {}
    horizon_facts = final.get("horizon_path") or {}
    conflict_facts = final.get("conflict_filter") or {}
    sanity_rows = dataframe_records(
        implementation_sanity[["check_name", "status", "value"]]
        if not implementation_sanity.empty
        else pd.DataFrame(),
        limit=20,
    )
    ablation_primary = ablation[
        (ablation["timeframe"].astype(str) == primary_timeframe)
        & (ablation["split"].astype(str) == "oos")
    ] if not ablation.empty else pd.DataFrame()
    reverse_primary = reverse[
        (reverse["dimension"].astype(str) == "timeframe_split")
        & (reverse["timeframe"].astype(str) == primary_timeframe)
        & (reverse["split"].astype(str) == "oos")
    ] if not reverse.empty else pd.DataFrame()
    return (
        "# VSVCB-v1 Postmortem Report\n\n"
        "## Executive Decision\n"
        f"- vsvcb_v1_failed={str(bool(final.get('vsvcb_v1_failed'))).lower()}\n"
        f"- trend_following_hypothesis_failed={str(bool(final.get('trend_following_hypothesis_failed'))).lower()}\n"
        f"- continue_to_phase2={str(bool(final.get('continue_to_phase2'))).lower()}\n"
        f"- parameter_plateau_allowed={str(bool(final.get('parameter_plateau_allowed'))).lower()}\n"
        f"- randomization_allowed={str(bool(final.get('randomization_allowed'))).lower()}\n"
        f"- strategy_development_allowed={str(bool(final.get('strategy_development_allowed'))).lower()}\n"
        f"- demo_live_allowed={str(bool(final.get('demo_live_allowed'))).lower()}\n\n"
        "## Required Answers\n"
        f"1. VSVCB-v1 Phase 1 是否失败？是，vsvcb_v1_failed={str(bool(final.get('vsvcb_v1_failed'))).lower()}。\n"
        f"2. 是否发现实现或数据问题？possible_implementation_issue={str(bool(final.get('possible_implementation_issue'))).lower()}，market_data_complete={final.get('market_data_complete')}，funding_data_complete={final.get('funding_data_complete')}。\n"
        f"3. D 组为什么失败？primary 15m train/validation/oos no-cost 分别为 {format_number(d_group.get('train_no_cost_pnl'), 4)} / {format_number(d_group.get('validation_no_cost_pnl'), 4)} / {format_number(d_group.get('oos_no_cost_pnl'), 4)}，cost-aware 和 funding-adjusted OOS 仍为负。\n"
        f"4. D 组是否只是少亏，而不是有 edge？是，ablation_pass_but_no_edge={str(bool(d_group.get('ablation_pass_but_no_edge'))).lower()}，D 相对 A/B/C 多数只是亏得更少，未转正。\n"
        f"5. 反向 E 组为什么优于 D？E 是 D 的机械反向；no-cost 通常接近 -D，成本后 primary OOS 仍优于 D，reverse_test_failure={str(bool(final.get('reverse_test_failure'))).lower()}。\n"
        f"6. 这是否说明 VSVCB 趋势延续假设失败？是，trend_following_hypothesis_failed={str(bool(final.get('trend_following_hypothesis_failed'))).lower()}。\n"
        f"7. 是否存在 false-breakout research 线索？possible_false_breakout_research_hypothesis={str(bool(final.get('possible_false_breakout_research_hypothesis'))).lower()}，但不能把 E 标记为趋势跟踪 edge。\n"
        f"8. 是否允许 Phase 2？否，continue_to_phase2=false，parameter_plateau_allowed=false。\n"
        f"9. 是否允许修改正式策略？否，strategy_development_allowed=false，official_strategy_modification_allowed=false。\n"
        f"10. 是否允许 demo/live？否，demo_live_allowed=false。\n"
        f"11. 下一步建议是什么？{final.get('recommended_next_step')}\n\n"
        "## Data / Implementation Sanity\n"
        + markdown_table(sanity_rows, ["check_name", "status", "value"])
        + "\n\n"
        "## D vs A/B/C Ablation\n"
        + markdown_table(
            dataframe_records(ablation_primary, limit=6),
            [
                "baseline_group",
                "d_trade_count",
                "baseline_trade_count",
                "d_no_cost_pnl",
                "baseline_no_cost_pnl",
                "d_better_than_baseline",
                "improvement_source",
            ],
        )
        + "\n\n"
        "## Reverse E\n"
        + markdown_table(
            dataframe_records(reverse_primary, limit=3),
            [
                "timeframe",
                "split",
                "d_no_cost_pnl",
                "e_no_cost_pnl",
                "d_cost_aware_pnl",
                "e_cost_aware_pnl",
                "reverse_test_failure",
            ],
        )
        + "\n\n"
        "## Symbol / Direction / Timeframe Notes\n"
        f"- BTC/ETH/SOL 拖累最大：{decomposition.get('worst_symbol')}\n"
        f"- 多头/空头拖累最大：{decomposition.get('worst_direction')}\n"
        f"- 正收益 timeframe：{decomposition.get('positive_timeframes') or []}\n"
        f"- 局部正收益 symbol/timeframe 样本需独立验证，不能据此调参。\n\n"
        "## Horizon Path\n"
        f"- best_horizon={horizon_facts.get('best_horizon')}\n"
        f"- horizon_3_better_than_10={horizon_facts.get('horizon_3_better_than_10')}\n"
        f"- horizon_5_better_than_10={horizon_facts.get('horizon_5_better_than_10')}\n"
        f"- horizon_20_worse_than_10={horizon_facts.get('horizon_20_worse_than_10')}\n"
        f"- early_reversal_likely={horizon_facts.get('early_reversal_likely')}\n"
        f"- mfe_exists_but_fixed_hold_wasted={horizon_facts.get('mfe_exists_but_fixed_hold_wasted')}\n"
        f"- mae_dominates_mfe={horizon_facts.get('mae_dominates_mfe')}\n\n"
        "## Conflict Filter\n"
        f"- d_event_count={conflict_facts.get('d_event_count')}\n"
        f"- d_trade_count={conflict_facts.get('d_trade_count')}\n"
        f"- untraded_event_count={conflict_facts.get('untraded_event_count')}\n"
        f"- untraded_event_rate={format_number(conflict_facts.get('untraded_event_rate'), 4)}\n"
        f"- single_position_filter_may_distort_result={conflict_facts.get('single_position_filter_may_distort_result')}\n\n"
        "## Guardrails\n"
        "- no_parameter_tuning_allowed=true\n"
        "- parameter_plateau_allowed=false\n"
        "- randomization_allowed=false\n"
        "- strategy_development_allowed=false\n"
        "- demo_live_allowed=false\n"
        "- E group reverse result is not a trend-following edge and is not tradable from this postmortem.\n"
    )


def run_postmortem(
    *,
    research_dir: Path,
    output_dir: Path,
    focus_group: str = DEFAULT_FOCUS_GROUP,
    reverse_group: str = DEFAULT_REVERSE_GROUP,
    primary_timeframe: str = DEFAULT_PRIMARY_TIMEFRAME,
    primary_symbols: list[str] | None = None,
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    """Run the VSVCB-v1 postmortem analysis."""

    primary_symbols = primary_symbols or DEFAULT_PRIMARY_SYMBOLS
    artifacts, load_warnings = load_artifacts(research_dir)
    events = artifacts.get("events", pd.DataFrame())
    trades = artifacts.get("trades", pd.DataFrame())
    summary = artifacts.get("summary", {})
    data_quality = artifacts.get("data_quality", {})
    funding_summary = artifacts.get("funding_summary", pd.DataFrame())
    reverse_test = artifacts.get("reverse_test", pd.DataFrame())
    warnings = collect_warnings(summary, data_quality, load_warnings)
    events = add_event_splits(events, summary)

    implementation_sanity, implementation_facts = build_implementation_sanity(
        events=events,
        trades=trades,
        summary=summary,
        data_quality=data_quality,
        funding_summary=funding_summary,
        warnings=warnings,
        focus_group=focus_group,
        reverse_group=reverse_group,
        primary_timeframe=primary_timeframe,
        primary_symbols=primary_symbols,
    )
    ablation, ablation_facts = build_ablation_postmortem(trades, focus_group, primary_timeframe)
    reverse, reverse_facts = build_reverse_directionality_postmortem(
        trades,
        events,
        reverse_test,
        focus_group,
        reverse_group,
        primary_timeframe,
    )
    by_symbol, by_direction, by_timeframe, by_symbol_direction, decomposition_facts = build_symbol_direction_timeframe_postmortems(
        trades,
        focus_group,
    )
    horizon, horizon_facts = build_horizon_path_postmortem(events, focus_group)
    feature_bins = build_feature_bin_postmortem(events, trades, focus_group)
    conflict_filter, conflict_facts = build_conflict_filter_impact(events, trades, warnings, focus_group)
    final = build_final_decision(
        summary=summary,
        implementation_facts=implementation_facts,
        ablation_facts=ablation_facts,
        reverse_facts=reverse_facts,
        decomposition_facts=decomposition_facts,
        horizon_facts=horizon_facts,
        conflict_facts=conflict_facts,
        warnings=warnings,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    write_dataframe(output_dir / OUTPUT_CSV_FILES["implementation_sanity"], implementation_sanity)
    write_dataframe(output_dir / OUTPUT_CSV_FILES["ablation"], ablation)
    write_dataframe(output_dir / OUTPUT_CSV_FILES["reverse"], reverse)
    write_dataframe(output_dir / OUTPUT_CSV_FILES["by_symbol"], by_symbol)
    write_dataframe(output_dir / OUTPUT_CSV_FILES["by_direction"], by_direction)
    write_dataframe(output_dir / OUTPUT_CSV_FILES["by_timeframe"], by_timeframe)
    write_dataframe(output_dir / OUTPUT_CSV_FILES["by_symbol_direction"], by_symbol_direction)
    write_dataframe(output_dir / OUTPUT_CSV_FILES["horizon"], horizon)
    write_dataframe(output_dir / OUTPUT_CSV_FILES["feature_bins"], feature_bins)
    write_dataframe(output_dir / OUTPUT_CSV_FILES["conflict_filter"], conflict_filter)
    (output_dir / "vsvcb_v1_postmortem_summary.json").write_text(
        json.dumps(clean_json(final), ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    (output_dir / "vsvcb_v1_postmortem_report.md").write_text(
        render_report(
            final,
            implementation_sanity,
            ablation,
            reverse,
            by_symbol,
            by_direction,
            by_timeframe,
            horizon,
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
            "VSVCB-v1 postmortem complete",
            output_dir=str(output_dir),
            vsvcb_v1_failed=final.get("vsvcb_v1_failed"),
            possible_false_breakout_research_hypothesis=final.get("possible_false_breakout_research_hypothesis"),
            possible_implementation_issue=final.get("possible_implementation_issue"),
        )
    return final


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""

    args = parse_args(argv)
    ensure_headless_runtime()
    logger = setup_logging("postmortem_vsvcb_v1", verbose=bool(args.verbose))
    run_postmortem(
        research_dir=resolve_path(args.research_dir),
        output_dir=resolve_path(args.output_dir),
        focus_group=str(args.focus_group),
        reverse_group=str(args.reverse_group),
        primary_timeframe=str(args.primary_timeframe),
        primary_symbols=parse_csv_list(args.primary_symbols),
        logger=logger,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
