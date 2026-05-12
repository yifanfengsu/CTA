#!/usr/bin/env python3
"""Postmortem and candidate audit for Trend Entry Timing research outputs."""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from common_runtime import PROJECT_ROOT, ensure_headless_runtime, log_event, setup_logging
from history_time_utils import DEFAULT_TIMEZONE


DEFAULT_RESEARCH_DIR = PROJECT_ROOT / "reports" / "research" / "trend_entry_timing"
DEFAULT_TREND_MAP_DIR = PROJECT_ROOT / "reports" / "research" / "trend_opportunity_map"
DEFAULT_FUNDING_DIR = PROJECT_ROOT / "data" / "funding" / "okx"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "reports" / "research" / "trend_entry_timing_postmortem"
DEFAULT_FOCUS_FAMILY = "cross_symbol_breadth_acceleration"

SPLITS = ["train_ext", "validation_ext", "oos_ext"]
SPLIT_LABELS = {"train_ext": "train", "validation_ext": "validation", "oos_ext": "oos"}
FIXED_NOTIONAL = 1000.0

REQUIRED_INPUT_FILES = [
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

OUTPUT_FILES = [
    "trend_entry_timing_postmortem_report.md",
    "trend_entry_timing_postmortem_summary.json",
    "candidate_gate_postmortem.csv",
    "breadth_candidate_deep_dive.csv",
    "cost_sensitivity.csv",
    "funding_dependency.csv",
    "entry_timing_concentration_postmortem.csv",
    "entry_timing_control_audit.csv",
    "entry_timing_capture_quality.csv",
    "rejected_candidate_entry_postmortem.csv",
]


class TrendEntryTimingPostmortemError(Exception):
    """Raised when the postmortem cannot be completed."""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(
        description="Audit Trend Entry Timing candidate failures without strategy development."
    )
    parser.add_argument("--research-dir", default=str(DEFAULT_RESEARCH_DIR))
    parser.add_argument("--trend-map-dir", default=str(DEFAULT_TREND_MAP_DIR))
    parser.add_argument("--funding-dir", default=str(DEFAULT_FUNDING_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--focus-family", default=DEFAULT_FOCUS_FAMILY)
    parser.add_argument("--timezone", default=DEFAULT_TIMEZONE)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def resolve_path(path_arg: str | Path) -> Path:
    """Resolve a path relative to the project root."""

    path = Path(path_arg)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def finite_number(value: Any, default: float | None = None) -> float | None:
    """Return a finite float, or default when unavailable."""

    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(number):
        return default
    return number


def clean_json(value: Any) -> Any:
    """Convert nested values into strict JSON-compatible data."""

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


def dataframe_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Return JSON-safe DataFrame records."""

    if frame.empty:
        return []
    work = frame.copy().astype(object).where(pd.notna(frame), None)
    return clean_json(json.loads(work.to_json(orient="records", date_format="iso")))


def read_csv_optional(directory: Path, filename: str, warnings: list[str]) -> pd.DataFrame:
    """Read an optional CSV, recording a warning instead of failing."""

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
    """Read an optional JSON file, recording a warning instead of failing."""

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
    """Read an optional text file, recording a warning instead of failing."""

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
    """Load all required Trend Entry Timing artifacts."""

    warnings: list[str] = []
    artifacts: dict[str, Any] = {}
    for filename in REQUIRED_INPUT_FILES:
        key = filename.rsplit(".", 1)[0]
        if filename == "trend_entry_timing_report.md":
            key = "trend_entry_timing_report_md"
        if filename.endswith(".csv"):
            artifacts[key] = read_csv_optional(research_dir, filename, warnings)
        elif filename.endswith(".json"):
            artifacts[key] = read_json_optional(research_dir, filename, warnings)
        else:
            artifacts[key] = read_text_optional(research_dir, filename, warnings)
    return artifacts, warnings


def write_dataframe(path: Path, frame: pd.DataFrame, columns: list[str] | None = None) -> None:
    """Write a DataFrame with stable columns for empty outputs."""

    output = frame.copy()
    if columns is not None:
        for column in columns:
            if column not in output.columns:
                output[column] = np.nan
        output = output.loc[:, columns]
    path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(path, index=False, encoding="utf-8")


def numeric_series(frame: pd.DataFrame, column: str) -> pd.Series:
    """Return a numeric column, or an empty float series."""

    if frame.empty or column not in frame.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def safe_sum(frame: pd.DataFrame, column: str) -> float:
    """Sum a numeric column if it exists."""

    values = numeric_series(frame, column)
    if values.empty:
        return 0.0
    return float(values.fillna(0.0).sum())


def safe_mean(frame: pd.DataFrame, column: str) -> float | None:
    """Return a numeric mean if available."""

    values = numeric_series(frame, column).dropna()
    if values.empty:
        return None
    return float(values.mean())


def bool_from_text(value: Any) -> bool | None:
    """Parse common boolean encodings."""

    if isinstance(value, bool):
        return value
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    return None


def selected_hold_for_family(family_summary: pd.DataFrame, trade_tests: pd.DataFrame, family: str) -> str | None:
    """Return selected hold label for a family."""

    if not family_summary.empty and {"family", "selected_hold_label"}.issubset(family_summary.columns):
        row = family_summary[family_summary["family"].astype(str) == family]
        if not row.empty and pd.notna(row.iloc[0]["selected_hold_label"]):
            return str(row.iloc[0]["selected_hold_label"])
    if not trade_tests.empty and {"family", "hold_label"}.issubset(trade_tests.columns):
        rows = trade_tests[trade_tests["family"].astype(str) == family]
        if not rows.empty:
            modes = rows["hold_label"].dropna().astype(str).mode()
            if not modes.empty:
                return str(modes.iloc[0])
    return None


def selected_trades(family: str, family_summary: pd.DataFrame, trade_tests: pd.DataFrame) -> pd.DataFrame:
    """Return selected-hold forward trades for one family."""

    if trade_tests.empty or "family" not in trade_tests.columns:
        return pd.DataFrame()
    rows = trade_tests[trade_tests["family"].astype(str) == family].copy()
    if "reverse" in rows.columns:
        parsed = rows["reverse"].map(bool_from_text)
        rows = rows[parsed != True].copy()  # noqa: E712
    hold = selected_hold_for_family(family_summary, trade_tests, family)
    if hold and "hold_label" in rows.columns:
        rows = rows[rows["hold_label"].astype(str) == hold].copy()
    return rows.reset_index(drop=True)


def control_trades(family: str, control_frame: pd.DataFrame) -> pd.DataFrame:
    """Return control rows for one family."""

    if control_frame.empty or "family" not in control_frame.columns:
        return pd.DataFrame()
    return control_frame[control_frame["family"].astype(str) == family].copy().reset_index(drop=True)


def split_frame(frame: pd.DataFrame, split: str | None) -> pd.DataFrame:
    """Filter a frame by split when requested."""

    if split is None or frame.empty or "split" not in frame.columns:
        return frame.copy()
    return frame[frame["split"].astype(str) == split].copy()


def funding_pnl_series(frame: pd.DataFrame) -> pd.Series:
    """Return signed funding PnL, deriving it when needed."""

    if frame.empty:
        return pd.Series(dtype=float)
    if "funding_pnl" in frame.columns:
        return numeric_series(frame, "funding_pnl").fillna(0.0)
    return numeric_series(frame, "funding_adjusted_pnl").fillna(0.0) - numeric_series(frame, "cost_aware_pnl").fillna(0.0)


def current_cost_series(frame: pd.DataFrame) -> pd.Series:
    """Return current artifact total cost per row."""

    if frame.empty:
        return pd.Series(dtype=float)
    return numeric_series(frame, "no_cost_pnl").fillna(0.0) - numeric_series(frame, "cost_aware_pnl").fillna(0.0)


def total_current_cost(frame: pd.DataFrame) -> float:
    """Return total inferred artifact cost."""

    costs = current_cost_series(frame)
    if costs.empty:
        return 0.0
    return float(costs.sum())


def cost_breakdown(frame: pd.DataFrame) -> tuple[float, float, str]:
    """Return fee and slippage cost, estimating when only total cost is present."""

    if frame.empty:
        return 0.0, 0.0, "no_trades"
    if "fee_cost" in frame.columns or "fee" in frame.columns:
        fee_col = "fee_cost" if "fee_cost" in frame.columns else "fee"
        fee = safe_sum(frame, fee_col)
    else:
        fee = math.nan
    if "slippage_cost" in frame.columns or "slippage" in frame.columns:
        slip_col = "slippage_cost" if "slippage_cost" in frame.columns else "slippage"
        slippage = safe_sum(frame, slip_col)
    else:
        slippage = math.nan
    if math.isfinite(fee) and math.isfinite(slippage):
        return fee, slippage, "explicit_columns"
    inferred = total_current_cost(frame)
    return inferred / 2.0, inferred / 2.0, "inferred_equal_split_from_no_cost_minus_cost_aware"


def group_largest_pnl_share(frame: pd.DataFrame, group_column: str, pnl_column: str = "funding_adjusted_pnl") -> float | None:
    """Return largest group PnL share using positive PnL when available, otherwise absolute PnL."""

    if frame.empty or group_column not in frame.columns or pnl_column not in frame.columns:
        return None
    grouped = frame.groupby(group_column, dropna=False)[pnl_column].sum()
    positive = grouped[grouped > 0]
    if not positive.empty and float(positive.sum()) > 0:
        return float(positive.max() / positive.sum())
    absolute = grouped.abs()
    total = float(absolute.sum())
    if total > 0:
        return float(absolute.max() / total)
    return None


def group_largest_name(frame: pd.DataFrame, group_column: str, pnl_column: str = "funding_adjusted_pnl") -> str | None:
    """Return the largest-contributing group name."""

    if frame.empty or group_column not in frame.columns or pnl_column not in frame.columns:
        return None
    grouped = frame.groupby(group_column, dropna=False)[pnl_column].sum()
    positive = grouped[grouped > 0]
    if not positive.empty:
        return str(positive.sort_values(ascending=False, kind="stable").index[0])
    if grouped.empty:
        return None
    return str(grouped.abs().sort_values(ascending=False, kind="stable").index[0])


def top_trade_contribution(frame: pd.DataFrame, top_fraction: float, pnl_column: str = "funding_adjusted_pnl") -> float | None:
    """Return top trade contribution versus total PnL."""

    if frame.empty or pnl_column not in frame.columns:
        return None
    pnl = numeric_series(frame, pnl_column).fillna(0.0)
    total = float(pnl.sum())
    if abs(total) <= 1e-12:
        return None
    top_n = max(1, int(math.ceil(len(pnl.index) * top_fraction)))
    return float(pnl.sort_values(ascending=False, kind="stable").head(top_n).sum() / total)


def positive_tail_share(frame: pd.DataFrame, top_fraction: float, pnl_column: str = "funding_adjusted_pnl") -> float | None:
    """Return top positive-tail share versus total positive PnL."""

    if frame.empty or pnl_column not in frame.columns:
        return None
    pnl = numeric_series(frame, pnl_column).fillna(0.0)
    positive = pnl[pnl > 0]
    if positive.empty or float(positive.sum()) <= 0:
        return None
    top_n = max(1, int(math.ceil(len(pnl.index) * top_fraction)))
    return float(pnl.sort_values(ascending=False, kind="stable").head(top_n).clip(lower=0.0).sum() / positive.sum())


def pnl_after_removing_top(frame: pd.DataFrame, top_fraction: float | None, pnl_column: str = "funding_adjusted_pnl") -> float:
    """Return PnL after removing top trades by PnL."""

    if frame.empty or pnl_column not in frame.columns:
        return 0.0
    pnl = numeric_series(frame, pnl_column).fillna(0.0)
    if pnl.empty:
        return 0.0
    if top_fraction is None:
        top_n = 1
    else:
        top_n = max(1, int(math.ceil(len(pnl.index) * top_fraction)))
    top_sum = float(pnl.sort_values(ascending=False, kind="stable").head(top_n).sum())
    return float(pnl.sum() - top_sum)


def trade_count_by_split(row: pd.Series) -> dict[str, int]:
    """Return split trade count dictionary from a family summary row."""

    return {
        split: int(finite_number(row.get(f"{split}_trade_count"), 0.0) or 0)
        for split in SPLITS
    }


def recalc_rejected_reasons(row: pd.Series) -> list[str]:
    """Recalculate stable-like rejection reasons when the rejected file is missing."""

    reasons: list[str] = []
    for split in SPLITS:
        if int(finite_number(row.get(f"{split}_trade_count"), 0.0) or 0) < 10:
            reasons.append(f"{split}:trade_count_lt_10")
        if (finite_number(row.get(f"{split}_no_cost_pnl"), 0.0) or 0.0) <= 0:
            reasons.append(f"{split}:no_cost_pnl_not_positive")
    if (finite_number(row.get("oos_ext_cost_aware_pnl"), 0.0) or 0.0) < 0:
        reasons.append("oos_ext:cost_aware_pnl_negative")
    if (finite_number(row.get("oos_ext_funding_adjusted_pnl"), 0.0) or 0.0) < 0:
        reasons.append("oos_ext:funding_adjusted_pnl_negative")
    if (finite_number(row.get("trend_segment_recall"), 0.0) or 0.0) < 0.20:
        reasons.append("trend_segment_recall_lt_0.20")
    if (finite_number(row.get("early_entry_rate"), 0.0) or 0.0) < 0.40:
        reasons.append("early_entry_rate_lt_0.40")
    if (finite_number(row.get("direction_match_rate"), 0.0) or 0.0) < 0.55:
        reasons.append("direction_match_rate_lt_0.55")
    forward = finite_number(row.get("funding_adjusted_pnl"), 0.0) or 0.0
    reverse = finite_number(row.get("reverse_test_result"), None)
    random = finite_number(row.get("random_time_control_result"), None)
    if reverse is None or reverse >= forward * 0.5:
        reasons.append("reverse_test_not_clearly_weaker")
    if random is None or random >= forward * 0.5:
        reasons.append("random_time_control_not_clearly_weaker")
    largest = finite_number(row.get("largest_symbol_pnl_share"), None)
    top = finite_number(row.get("top_5pct_trade_pnl_contribution"), None)
    if largest is None or largest > 0.70:
        reasons.append("largest_symbol_pnl_share_gt_0.7")
    if top is None or top > 0.80:
        reasons.append("top_5pct_trade_pnl_contribution_gt_0.8")
    return reasons


def reason_list_for_family(row: pd.Series, rejected: pd.DataFrame) -> list[str]:
    """Return rejection reasons for a family."""

    family = str(row.get("family"))
    if not rejected.empty and {"family", "rejected_reasons"}.issubset(rejected.columns):
        match = rejected[rejected["family"].astype(str) == family]
        if not match.empty and pd.notna(match.iloc[0]["rejected_reasons"]):
            text = str(match.iloc[0]["rejected_reasons"])
            return [item for item in text.split(";") if item]
    return recalc_rejected_reasons(row)


def primary_failure_category(reasons: list[str]) -> str:
    """Classify the primary gate failure."""

    if not reasons:
        return "none"
    if any("no_cost_pnl_not_positive" in reason for reason in reasons):
        return "no_cost_split_failure"
    if any("cost_aware_pnl_negative" in reason for reason in reasons):
        return "cost_or_execution_failure"
    if any("funding_adjusted_pnl_negative" in reason for reason in reasons):
        return "funding_adjusted_failure"
    if any("trend_segment_recall" in reason or "early_entry_rate" in reason or "direction_match" in reason for reason in reasons):
        return "trend_capture_quality_failure"
    if any("symbol" in reason or "top_5pct" in reason for reason in reasons):
        return "concentration_failure"
    if any("reverse" in reason or "random" in reason for reason in reasons):
        return "control_failure"
    if any("trade_count" in reason for reason in reasons):
        return "sample_size_failure"
    return "other_gate_failure"


def build_gate_postmortem(family_summary: pd.DataFrame, rejected: pd.DataFrame) -> pd.DataFrame:
    """Build gate-failure explanation for each candidate family."""

    rows: list[dict[str, Any]] = []
    if family_summary.empty or "family" not in family_summary.columns:
        return pd.DataFrame()
    for _, row in family_summary.iterrows():
        family = str(row["family"])
        reasons = reason_list_for_family(row, rejected)
        split_counts = trade_count_by_split(row)
        no_cost_values = {
            split: finite_number(row.get(f"{split}_no_cost_pnl"), 0.0) or 0.0
            for split in SPLITS
        }
        record = {
            "family": family,
            "selected_hold_label": row.get("selected_hold_label"),
            "failed_gate_count": len(reasons),
            "rejected_reasons": ";".join(reasons),
            "primary_failure_category": primary_failure_category(reasons),
            "train_no_cost": no_cost_values["train_ext"],
            "validation_no_cost": no_cost_values["validation_ext"],
            "oos_no_cost": no_cost_values["oos_ext"],
            "oos_cost": finite_number(row.get("oos_ext_cost_aware_pnl"), 0.0) or 0.0,
            "oos_funding": finite_number(row.get("oos_ext_funding_adjusted_pnl"), 0.0) or 0.0,
            "trade_count_by_split": json.dumps(split_counts, sort_keys=True),
            "trend_recall": finite_number(row.get("trend_segment_recall"), None),
            "early_entry_rate": finite_number(row.get("early_entry_rate"), None),
            "direction_match": finite_number(row.get("direction_match_rate"), None),
            "largest_symbol_pnl_share": finite_number(row.get("largest_symbol_pnl_share"), None),
            "top_5pct_trade_pnl_contribution": finite_number(row.get("top_5pct_trade_pnl_contribution"), None),
            "no_cost_all_splits_positive": all(value > 0 for value in no_cost_values.values()),
            "oos_cost_failure": (finite_number(row.get("oos_ext_cost_aware_pnl"), 0.0) or 0.0) < 0,
            "oos_funding_failure": (finite_number(row.get("oos_ext_funding_adjusted_pnl"), 0.0) or 0.0) < 0,
            "early_entry_failure": (finite_number(row.get("early_entry_rate"), 0.0) or 0.0) < 0.40,
            "direction_match_failure": (finite_number(row.get("direction_match_rate"), 0.0) or 0.0) < 0.55,
            "sample_size_failure": any(count < 10 for count in split_counts.values()),
            "concentration_failure": any("symbol" in reason or "top_5pct" in reason for reason in reasons),
            "control_failure": any("reverse" in reason or "random" in reason for reason in reasons),
        }
        rows.append(record)
    return pd.DataFrame(rows)


def concentration_metrics(family: str, scope: str, frame: pd.DataFrame) -> dict[str, Any]:
    """Build concentration and tail-risk metrics for one trade slice."""

    trade_count = int(len(frame.index))
    active_symbols = int(frame["symbol"].nunique()) if trade_count and "symbol" in frame.columns else 0
    largest_trade_count_share = None
    if trade_count and "symbol" in frame.columns:
        largest_trade_count_share = float(frame.groupby("symbol", dropna=False).size().max() / trade_count)
    pnl_total = safe_sum(frame, "funding_adjusted_pnl")
    return {
        "family": family,
        "scope": scope,
        "pnl_column": "funding_adjusted_pnl",
        "total_pnl": pnl_total,
        "largest_symbol": group_largest_name(frame, "symbol"),
        "largest_symbol_pnl_share": group_largest_pnl_share(frame, "symbol"),
        "largest_symbol_trade_count_share": largest_trade_count_share,
        "top_1_trade_contribution": top_trade_contribution(frame, 0.0),
        "top_5pct_trade_pnl_contribution": top_trade_contribution(frame, 0.05),
        "top_5pct_positive_tail_share": positive_tail_share(frame, 0.05),
        "remove_top_1_pnl": pnl_after_removing_top(frame, None),
        "remove_top_5pct_pnl": pnl_after_removing_top(frame, 0.05),
        "trade_count": trade_count,
        "active_symbol_count": active_symbols,
        "concentration_pass": bool(
            trade_count > 0
            and active_symbols >= 2
            and (group_largest_pnl_share(frame, "symbol") is not None)
            and float(group_largest_pnl_share(frame, "symbol") or 0.0) <= 0.70
            and ((top_trade_contribution(frame, 0.05) is not None and float(top_trade_contribution(frame, 0.05) or 0.0) <= 0.80) or pnl_total <= 0)
        ),
    }


def build_concentration_postmortem(
    family_summary: pd.DataFrame,
    trade_tests: pd.DataFrame,
    focus_family: str,
) -> pd.DataFrame:
    """Build concentration and tail-risk postmortem."""

    families: list[str] = []
    if not family_summary.empty and "family" in family_summary.columns:
        families.extend(family_summary["family"].dropna().astype(str).tolist())
    if not trade_tests.empty and "family" in trade_tests.columns:
        families.extend(trade_tests["family"].dropna().astype(str).tolist())
    families = list(dict.fromkeys(families))
    rows: list[dict[str, Any]] = []
    for family in families:
        trades = selected_trades(family, family_summary, trade_tests)
        rows.append(concentration_metrics(family, "all_splits", trades))
        if family == focus_family:
            rows.append(concentration_metrics(family, "oos_ext", split_frame(trades, "oos_ext")))
    return pd.DataFrame(rows)


def capture_fraction_proxy(missed_mfe: Any, remaining_mfe: Any) -> float | None:
    """Return remaining MFE share as a rough capture-quality proxy."""

    missed = finite_number(missed_mfe, None)
    remaining = finite_number(remaining_mfe, None)
    if missed is None or remaining is None:
        return None
    denominator = missed + remaining
    if denominator <= 0:
        return None
    return float(remaining / denominator)


def build_capture_quality(family_summary: pd.DataFrame) -> pd.DataFrame:
    """Build trend-capture quality diagnostics."""

    rows: list[dict[str, Any]] = []
    if family_summary.empty or "family" not in family_summary.columns:
        return pd.DataFrame()
    for _, row in family_summary.iterrows():
        early = finite_number(row.get("early_entry_rate"), None)
        lag = finite_number(row.get("median_entry_lag_pct"), None)
        truly_early = bool(early is not None and early >= 0.40 and lag is not None and lag <= 0.25)
        rows.append(
            {
                "family": str(row["family"]),
                "trend_recall": finite_number(row.get("trend_segment_recall"), None),
                "early_entry_rate": early,
                "direction_match": finite_number(row.get("direction_match_rate"), None),
                "median_entry_lag_pct": lag,
                "missed_mfe_before_entry": finite_number(row.get("average_missed_mfe_before_entry"), None),
                "remaining_mfe_after_entry": finite_number(row.get("average_remaining_mfe"), None),
                "captured_fraction_proxy": capture_fraction_proxy(
                    row.get("average_missed_mfe_before_entry"),
                    row.get("average_remaining_mfe"),
                ),
                "late_entry_problem_solved": truly_early,
                "early_entry_quality": "early_enough" if truly_early else "directionally_correct_but_still_late",
            }
        )
    return pd.DataFrame(rows)


def split_or_all_rows() -> list[tuple[str, str | None]]:
    """Return all control-audit scopes."""

    return [("all_splits", None), ("train_ext", "train_ext"), ("validation_ext", "validation_ext"), ("oos_ext", "oos_ext")]


def control_weaker(forward: float, control: float | None) -> bool:
    """Return whether a control is clearly weaker than forward."""

    if control is None:
        return False
    if forward > 0:
        return bool(control < forward * 0.5)
    return bool(control < forward)


def build_control_audit(
    focus_family: str,
    family_summary: pd.DataFrame,
    trade_tests: pd.DataFrame,
    reverse: pd.DataFrame,
    random_control: pd.DataFrame,
) -> pd.DataFrame:
    """Audit forward versus reverse and random controls for focus family."""

    forward_all = selected_trades(focus_family, family_summary, trade_tests)
    reverse_all = control_trades(focus_family, reverse)
    random_all = control_trades(focus_family, random_control)
    rows: list[dict[str, Any]] = []
    for scope, split in split_or_all_rows():
        fwd = split_frame(forward_all, split)
        rev = split_frame(reverse_all, split)
        rnd = split_frame(random_all, split)
        forward_pnl = safe_sum(fwd, "funding_adjusted_pnl")
        reverse_pnl = safe_sum(rev, "funding_adjusted_pnl")
        random_pnl = safe_sum(rnd, "funding_adjusted_pnl")
        reverse_value = reverse_pnl if len(rev.index) else None
        random_value = random_pnl if len(rnd.index) else None
        reverse_weaker = control_weaker(forward_pnl, reverse_value)
        random_weaker = control_weaker(forward_pnl, random_value)
        rows.append(
            {
                "family": focus_family,
                "scope": scope,
                "forward_trade_count": int(len(fwd.index)),
                "reverse_trade_count": int(len(rev.index)),
                "random_trade_count": int(len(rnd.index)),
                "forward_no_cost_pnl": safe_sum(fwd, "no_cost_pnl"),
                "forward_cost_aware_pnl": safe_sum(fwd, "cost_aware_pnl"),
                "forward_funding_adjusted_pnl": forward_pnl,
                "reverse_no_cost_pnl": safe_sum(rev, "no_cost_pnl"),
                "reverse_cost_aware_pnl": safe_sum(rev, "cost_aware_pnl"),
                "reverse_funding_adjusted_pnl": reverse_value,
                "random_no_cost_pnl": safe_sum(rnd, "no_cost_pnl"),
                "random_cost_aware_pnl": safe_sum(rnd, "cost_aware_pnl"),
                "random_funding_adjusted_pnl": random_value,
                "forward_gt_reverse": bool(reverse_value is not None and forward_pnl > reverse_value),
                "forward_gt_random": bool(random_value is not None and forward_pnl > random_value),
                "reverse_clearly_weaker": reverse_weaker,
                "random_clearly_weaker": random_weaker,
                "control_pass": bool(reverse_weaker and random_weaker),
                "reverse_or_random_stronger": bool(
                    (reverse_value is not None and reverse_value >= forward_pnl)
                    or (random_value is not None and random_value >= forward_pnl)
                ),
            }
        )
    return pd.DataFrame(rows)


def build_cost_sensitivity(
    focus_family: str,
    family_summary: pd.DataFrame,
    trade_tests: pd.DataFrame,
) -> pd.DataFrame:
    """Build OOS cost sensitivity for focus family without modifying strategy parameters."""

    trades = split_frame(selected_trades(focus_family, family_summary, trade_tests), "oos_ext")
    no_cost = safe_sum(trades, "no_cost_pnl")
    funding = float(funding_pnl_series(trades).sum()) if not trades.empty else 0.0
    trade_count = int(len(trades.index))
    current_cost = total_current_cost(trades)
    current_cost_aware = safe_sum(trades, "cost_aware_pnl")
    rows: list[dict[str, Any]] = []
    for fee_bps in [2, 5, 8]:
        for slippage_bps in [2, 5, 8]:
            round_trip_bps = 2.0 * (fee_bps + slippage_bps)
            fee_cost = trade_count * FIXED_NOTIONAL * 2.0 * fee_bps / 10000.0
            slippage_cost = trade_count * FIXED_NOTIONAL * 2.0 * slippage_bps / 10000.0
            total_cost = fee_cost + slippage_cost
            cost_aware = no_cost - total_cost
            rows.append(
                {
                    "family": focus_family,
                    "split": "oos_ext",
                    "fee_bps_per_side": fee_bps,
                    "slippage_bps_per_side": slippage_bps,
                    "round_trip_cost_bps": round_trip_bps,
                    "trade_count": trade_count,
                    "oos_no_cost_pnl": no_cost,
                    "fee_cost": fee_cost,
                    "slippage_cost": slippage_cost,
                    "total_cost": total_cost,
                    "oos_cost_aware_pnl": cost_aware,
                    "oos_funding_adjusted_pnl": cost_aware + funding,
                    "passes_cost_aware": bool(cost_aware > 0),
                    "current_artifact_cost": current_cost,
                    "current_artifact_cost_aware_pnl": current_cost_aware,
                    "below_current_artifact_cost": bool(total_cost < current_cost),
                }
            )
    return pd.DataFrame(rows)


def build_funding_dependency(
    focus_family: str,
    family_summary: pd.DataFrame,
    trade_tests: pd.DataFrame,
) -> pd.DataFrame:
    """Build signed and conservative funding-dependency diagnostics."""

    trades = selected_trades(focus_family, family_summary, trade_tests)
    rows: list[dict[str, Any]] = []
    for scope, split in [("all_splits", None), ("oos_ext", "oos_ext")]:
        frame = split_frame(trades, split)
        funding = funding_pnl_series(frame)
        signed_funding = float(funding.sum()) if not funding.empty else 0.0
        positive_funding = float(funding[funding > 0].sum()) if not funding.empty else 0.0
        negative_funding = float(funding[funding < 0].sum()) if not funding.empty else 0.0
        no_cost = safe_sum(frame, "no_cost_pnl")
        cost_aware = safe_sum(frame, "cost_aware_pnl")
        funding_adjusted = safe_sum(frame, "funding_adjusted_pnl")
        conservative_adjusted = cost_aware + negative_funding
        funding_dependency_ratio = float(signed_funding / abs(cost_aware)) if cost_aware < 0 and abs(cost_aware) > 1e-12 else None
        rows.append(
            {
                "family": focus_family,
                "scope": scope,
                "trade_count": int(len(frame.index)),
                "no_cost_pnl": no_cost,
                "cost_aware_pnl_without_funding": cost_aware,
                "signed_funding_pnl": signed_funding,
                "positive_funding_pnl": positive_funding,
                "negative_funding_pnl": negative_funding,
                "funding_adjusted_pnl_signed": funding_adjusted,
                "funding_adjusted_pnl_conservative": conservative_adjusted,
                "funding_events_count": int(numeric_series(frame, "funding_events_count").fillna(0).sum()) if "funding_events_count" in frame.columns else 0,
                "funding_adjusted_positive": bool(funding_adjusted > 0),
                "positive_due_to_funding": bool(cost_aware <= 0 < funding_adjusted and signed_funding > 0),
                "funding_dependency_ratio": funding_dependency_ratio,
                "funding_dependent": bool(cost_aware <= 0 < funding_adjusted and signed_funding > 0),
            }
        )
    return pd.DataFrame(rows)


def build_focus_deep_dive(
    focus_family: str,
    family_summary: pd.DataFrame,
    trade_tests: pd.DataFrame,
    concentration: pd.DataFrame,
    control_audit: pd.DataFrame,
) -> pd.DataFrame:
    """Build one-row deep dive for the focus family."""

    focus_rows = family_summary[family_summary["family"].astype(str) == focus_family] if not family_summary.empty and "family" in family_summary.columns else pd.DataFrame()
    if focus_rows.empty:
        return pd.DataFrame([{"family": focus_family, "focus_family_found": False}])
    row = focus_rows.iloc[0]
    trades = selected_trades(focus_family, family_summary, trade_tests)
    oos = split_frame(trades, "oos_ext")
    fee_cost, slippage_cost, cost_source = cost_breakdown(oos)
    oos_trade_count = int(len(oos.index))
    no_cost = safe_sum(oos, "no_cost_pnl")
    cost_aware = safe_sum(oos, "cost_aware_pnl")
    funding_pnl = float(funding_pnl_series(oos).sum()) if not oos.empty else 0.0
    funding_adjusted = safe_sum(oos, "funding_adjusted_pnl")
    avg_trade_gross = float(no_cost / oos_trade_count) if oos_trade_count else None
    avg_trade_cost = float((fee_cost + slippage_cost) / oos_trade_count) if oos_trade_count else None
    symbol_share = group_largest_pnl_share(trades, "symbol")
    timeframe_share = group_largest_pnl_share(trades, "timeframe")
    direction_column = "test_direction" if "test_direction" in trades.columns else "direction"
    direction_share = group_largest_pnl_share(trades, direction_column)
    concentration_all = concentration[(concentration["family"].astype(str) == focus_family) & (concentration["scope"].astype(str) == "all_splits")] if not concentration.empty else pd.DataFrame()
    control_all = control_audit[(control_audit["family"].astype(str) == focus_family) & (control_audit["scope"].astype(str) == "all_splits")] if not control_audit.empty else pd.DataFrame()
    total_cost = fee_cost + slippage_cost
    cost_failure_is_marginal = bool(cost_aware < 0 and abs(cost_aware) <= max(50.0, abs(no_cost) * 0.02))
    structural_cost_drag = bool(oos_trade_count > 0 and avg_trade_cost is not None and avg_trade_gross is not None and avg_trade_cost >= avg_trade_gross)
    early_entry_rate = finite_number(row.get("early_entry_rate"), None)
    median_lag = finite_number(row.get("median_entry_lag_pct"), None)
    top_5pct = finite_number(concentration_all.iloc[0].get("top_5pct_trade_pnl_contribution"), None) if not concentration_all.empty else None
    remove_top_1 = finite_number(concentration_all.iloc[0].get("remove_top_1_pnl"), None) if not concentration_all.empty else None
    control_pass = bool(control_all.iloc[0].get("control_pass")) if not control_all.empty else False
    return pd.DataFrame(
        [
            {
                "family": focus_family,
                "focus_family_found": True,
                "selected_hold_label": row.get("selected_hold_label"),
                "train_no_cost_positive": (finite_number(row.get("train_ext_no_cost_pnl"), 0.0) or 0.0) > 0,
                "validation_no_cost_positive": (finite_number(row.get("validation_ext_no_cost_pnl"), 0.0) or 0.0) > 0,
                "oos_no_cost_positive": (finite_number(row.get("oos_ext_no_cost_pnl"), 0.0) or 0.0) > 0,
                "no_cost_all_splits_positive": all((finite_number(row.get(f"{split}_no_cost_pnl"), 0.0) or 0.0) > 0 for split in SPLITS),
                "oos_trade_count": oos_trade_count,
                "oos_no_cost_pnl": no_cost,
                "oos_cost_aware_pnl": cost_aware,
                "fee_cost": fee_cost,
                "slippage_cost": slippage_cost,
                "total_current_cost": total_cost,
                "cost_breakdown_source": cost_source,
                "round_trip_turnover": float(oos_trade_count * FIXED_NOTIONAL * 2.0),
                "avg_trade_gross": avg_trade_gross,
                "avg_trade_cost": avg_trade_cost,
                "cost_failure_is_marginal": cost_failure_is_marginal,
                "structural_cost_drag": structural_cost_drag,
                "funding_pnl": funding_pnl,
                "funding_events_count": int(numeric_series(oos, "funding_events_count").fillna(0).sum()) if "funding_events_count" in oos.columns else 0,
                "oos_funding_adjusted_pnl": funding_adjusted,
                "funding_adjusted_positive_due_to_funding": bool(cost_aware <= 0 < funding_adjusted and funding_pnl > 0),
                "funding_dependent": bool(cost_aware <= 0 < funding_adjusted and funding_pnl > 0),
                "largest_symbol": group_largest_name(trades, "symbol"),
                "largest_symbol_pnl_share": symbol_share,
                "symbol_dependency": bool(symbol_share is None or symbol_share > 0.70),
                "largest_timeframe": group_largest_name(trades, "timeframe"),
                "largest_timeframe_pnl_share": timeframe_share,
                "timeframe_dependency": bool(timeframe_share is None or timeframe_share > 0.80 or (not trades.empty and "timeframe" in trades.columns and trades["timeframe"].nunique() <= 1)),
                "largest_direction": group_largest_name(trades, direction_column),
                "largest_direction_pnl_share": direction_share,
                "direction_dependency": bool(direction_share is None or direction_share > 0.80),
                "top_5pct_trade_pnl_contribution": top_5pct,
                "remove_top_1_pnl": remove_top_1,
                "top_trade_dependency": bool((top_5pct is not None and top_5pct > 0.80) or (remove_top_1 is not None and remove_top_1 < 0)),
                "trend_recall": finite_number(row.get("trend_segment_recall"), None),
                "early_entry_rate": early_entry_rate,
                "direction_match": finite_number(row.get("direction_match_rate"), None),
                "median_entry_lag_pct": median_lag,
                "missed_mfe_before_entry": finite_number(row.get("average_missed_mfe_before_entry"), None),
                "remaining_mfe_after_entry": finite_number(row.get("average_remaining_mfe"), None),
                "captured_fraction_proxy": capture_fraction_proxy(row.get("average_missed_mfe_before_entry"), row.get("average_remaining_mfe")),
                "early_entry_quality": "insufficient_early_entry" if early_entry_rate is None or early_entry_rate < 0.40 or (median_lag is not None and median_lag > 0.25) else "early_enough",
                "control_pass": control_pass,
            }
        ]
    )


def asset_condition_result(
    name: str,
    passed: bool,
    observed: Any,
    threshold: str,
) -> dict[str, Any]:
    """Return one research-asset condition result."""

    return {"condition": name, "passed": bool(passed), "observed": observed, "threshold": threshold}


def evaluate_research_asset(
    focus_family: str,
    family_summary: pd.DataFrame,
    gate_postmortem: pd.DataFrame,
    concentration: pd.DataFrame,
    control_audit: pd.DataFrame,
    cost_sensitivity: pd.DataFrame,
) -> dict[str, Any]:
    """Evaluate whether focus family is only a research asset."""

    focus_rows = family_summary[family_summary["family"].astype(str) == focus_family] if not family_summary.empty and "family" in family_summary.columns else pd.DataFrame()
    if focus_rows.empty:
        return {
            "focus_family_research_asset": False,
            "conditions": [asset_condition_result("focus_family_exists", False, "missing", "required")],
            "execution_fragile": True,
            "recommended_next_step": "pause_or_new_hypothesis",
        }
    row = focus_rows.iloc[0]
    concentration_row = concentration[(concentration["family"].astype(str) == focus_family) & (concentration["scope"].astype(str) == "all_splits")] if not concentration.empty else pd.DataFrame()
    control_row = control_audit[(control_audit["family"].astype(str) == focus_family) & (control_audit["scope"].astype(str) == "all_splits")] if not control_audit.empty else pd.DataFrame()
    gate_row = gate_postmortem[gate_postmortem["family"].astype(str) == focus_family] if not gate_postmortem.empty else pd.DataFrame()

    split_counts = trade_count_by_split(row)
    no_cost_all_positive = all((finite_number(row.get(f"{split}_no_cost_pnl"), 0.0) or 0.0) > 0 for split in SPLITS)
    oos_funding = finite_number(row.get("oos_ext_funding_adjusted_pnl"), 0.0) or 0.0
    oos_cost = finite_number(row.get("oos_ext_cost_aware_pnl"), 0.0) or 0.0
    oos_no_cost = finite_number(row.get("oos_ext_no_cost_pnl"), 0.0) or 0.0
    cost_failure_small = bool(oos_cost >= 0 or abs(oos_cost) <= max(50.0, abs(oos_no_cost) * 0.02))
    direction = finite_number(row.get("direction_match_rate"), 0.0) or 0.0
    recall = finite_number(row.get("trend_segment_recall"), 0.0) or 0.0
    largest_symbol = finite_number(concentration_row.iloc[0].get("largest_symbol_pnl_share"), None) if not concentration_row.empty else finite_number(row.get("largest_symbol_pnl_share"), None)
    top_5pct = finite_number(concentration_row.iloc[0].get("top_5pct_trade_pnl_contribution"), None) if not concentration_row.empty else finite_number(row.get("top_5pct_trade_pnl_contribution"), None)
    remove_top_1 = finite_number(concentration_row.iloc[0].get("remove_top_1_pnl"), None) if not concentration_row.empty else None
    control_pass = bool(control_row.iloc[0].get("control_pass")) if not control_row.empty else False
    lower_cost_pass = bool((cost_sensitivity["passes_cost_aware"] == True).any()) if not cost_sensitivity.empty and "passes_cost_aware" in cost_sensitivity.columns else False  # noqa: E712
    current_cost_fails = bool(oos_cost < 0)
    high_cost_fails = bool((cost_sensitivity["oos_cost_aware_pnl"] <= 0).any()) if not cost_sensitivity.empty and "oos_cost_aware_pnl" in cost_sensitivity.columns else True
    execution_fragile = bool(current_cost_fails or (lower_cost_pass and high_cost_fails))

    conditions = [
        asset_condition_result("train_validation_oos_no_cost_positive", no_cost_all_positive, {split: finite_number(row.get(f"{split}_no_cost_pnl"), None) for split in SPLITS}, "> 0 each split"),
        asset_condition_result("oos_funding_adjusted_positive", oos_funding > 0, oos_funding, "> 0"),
        asset_condition_result("oos_cost_aware_failure_small", cost_failure_small, oos_cost, ">= 0 or small negative"),
        asset_condition_result("direction_match_minimum", direction >= 0.60, direction, ">= 0.60"),
        asset_condition_result("trend_recall_minimum", recall >= 0.20, recall, ">= 0.20"),
        asset_condition_result("trade_count_each_split_minimum", all(count >= 10 for count in split_counts.values()), split_counts, ">= 10 each split"),
        asset_condition_result("largest_symbol_pnl_share_limit", largest_symbol is not None and largest_symbol <= 0.70, largest_symbol, "<= 0.70"),
        asset_condition_result("top_5pct_trade_pnl_contribution_limit", top_5pct is not None and top_5pct <= 0.80, top_5pct, "<= 0.80"),
        asset_condition_result("reverse_and_random_control_weaker", control_pass, control_row.iloc[0].to_dict() if not control_row.empty else None, "both clearly weaker"),
        asset_condition_result("remove_top_1_not_disastrous", remove_top_1 is not None and remove_top_1 >= 0.0, remove_top_1, ">= 0"),
    ]
    if not gate_row.empty:
        conditions.append(
            asset_condition_result(
                "stable_like_not_required_for_asset",
                True,
                gate_row.iloc[0].get("rejected_reasons"),
                "may fail stable-like, but must remain non-tradable",
            )
        )
    research_asset = bool(all(item["passed"] for item in conditions if item["condition"] != "stable_like_not_required_for_asset"))
    return {
        "focus_family_research_asset": research_asset,
        "conditions": conditions,
        "execution_fragile": execution_fragile,
        "recommended_next_step": "phase1_5_execution_and_threshold_diagnostics" if research_asset else "pause_or_new_hypothesis",
    }


def summarize_decision(
    *,
    focus_family: str,
    gate_postmortem: pd.DataFrame,
    deep_dive: pd.DataFrame,
    cost_sensitivity: pd.DataFrame,
    funding_dependency: pd.DataFrame,
    concentration: pd.DataFrame,
    control_audit: pd.DataFrame,
    capture_quality: pd.DataFrame,
    asset: dict[str, Any],
    warnings: list[str],
    trend_entry_summary: dict[str, Any],
    trend_map_summary: dict[str, Any],
    trend_map_quality: dict[str, Any],
    funding_dir: Path,
    timezone_name: str,
    output_dir: Path,
) -> dict[str, Any]:
    """Build final JSON summary."""

    no_cost_positive = []
    oos_cost_failures = []
    control_failures = []
    if not gate_postmortem.empty:
        no_cost_positive = gate_postmortem.loc[gate_postmortem["no_cost_all_splits_positive"] == True, "family"].astype(str).tolist()  # noqa: E712
        oos_cost_failures = gate_postmortem.loc[gate_postmortem["oos_cost_failure"] == True, "family"].astype(str).tolist()  # noqa: E712
        control_failures = gate_postmortem.loc[gate_postmortem["control_failure"] == True, "family"].astype(str).tolist()  # noqa: E712

    deep_row = deep_dive.iloc[0].to_dict() if not deep_dive.empty else {}
    funding_oos = funding_dependency[funding_dependency["scope"].astype(str) == "oos_ext"].iloc[0].to_dict() if not funding_dependency.empty and "scope" in funding_dependency.columns and not funding_dependency[funding_dependency["scope"].astype(str) == "oos_ext"].empty else {}
    concentration_all = concentration[(concentration["family"].astype(str) == focus_family) & (concentration["scope"].astype(str) == "all_splits")].iloc[0].to_dict() if not concentration.empty and not concentration[(concentration["family"].astype(str) == focus_family) & (concentration["scope"].astype(str) == "all_splits")].empty else {}
    control_all = control_audit[(control_audit["family"].astype(str) == focus_family) & (control_audit["scope"].astype(str) == "all_splits")].iloc[0].to_dict() if not control_audit.empty and not control_audit[(control_audit["family"].astype(str) == focus_family) & (control_audit["scope"].astype(str) == "all_splits")].empty else {}
    capture_focus = capture_quality[capture_quality["family"].astype(str) == focus_family].iloc[0].to_dict() if not capture_quality.empty and not capture_quality[capture_quality["family"].astype(str) == focus_family].empty else {}
    lower_cost_can_pass = bool((cost_sensitivity["passes_cost_aware"] == True).any()) if not cost_sensitivity.empty and "passes_cost_aware" in cost_sensitivity.columns else False  # noqa: E712

    primary_failure_reasons = (
        gate_postmortem["primary_failure_category"].dropna().astype(str).value_counts().to_dict()
        if not gate_postmortem.empty and "primary_failure_category" in gate_postmortem.columns
        else {}
    )

    funding_files = sorted(funding_dir.glob("*.csv")) if funding_dir.exists() else []

    return clean_json(
        {
            "mode": "trend_entry_timing_postmortem_and_candidate_audit",
            "output_dir": str(output_dir),
            "output_files": OUTPUT_FILES,
            "timezone": timezone_name,
            "warnings": sorted(dict.fromkeys(warnings)),
            "trend_entry_timing_research_failed": True,
            "source_can_enter_entry_timing_phase2": trend_entry_summary.get("can_enter_entry_timing_phase2"),
            "source_stable_like_candidate_exists": trend_entry_summary.get("stable_like_candidate_exists"),
            "trend_map_enough_opportunities": trend_map_summary.get("enough_trend_opportunities"),
            "trend_map_legacy_main_failure_mode": (trend_map_summary.get("legacy_analysis") or {}).get("main_failure_mode"),
            "trend_map_data_quality": trend_map_quality,
            "funding_dir": str(funding_dir),
            "funding_file_count": len(funding_files),
            "focus_family": focus_family,
            "primary_failure_reasons": primary_failure_reasons,
            "families_with_no_cost_positive_all_splits": no_cost_positive,
            "families_with_oos_cost_failure": oos_cost_failures,
            "families_with_control_failure": control_failures,
            "lower_cost_oos_can_pass": lower_cost_can_pass,
            "focus_family_research_asset": bool(asset.get("focus_family_research_asset")),
            "research_asset_conditions": asset.get("conditions") or [],
            "execution_fragile": bool(asset.get("execution_fragile")),
            "funding_dependent": bool(funding_oos.get("funding_dependent", False)),
            "concentration_pass": bool(concentration_all.get("concentration_pass", False)),
            "control_pass": bool(control_all.get("control_pass", False)),
            "early_entry_quality": capture_focus.get("early_entry_quality"),
            "focus_family_deep_dive": deep_row,
            "can_enter_phase1_5_diagnostic": bool(asset.get("focus_family_research_asset")),
            "can_enter_entry_timing_phase2": False,
            "strategy_development_allowed": False,
            "demo_live_allowed": False,
            "no_policy_can_be_traded": True,
            "recommended_next_step": asset.get("recommended_next_step", "pause_or_new_hypothesis"),
        }
    )


def format_number(value: Any, digits: int = 4) -> str:
    """Format a number for Markdown."""

    number = finite_number(value, None)
    if number is None:
        return "N/A"
    return f"{number:.{digits}f}"


def markdown_table(frame: pd.DataFrame, columns: list[str], limit: int = 20) -> str:
    """Render a compact Markdown table."""

    if frame.empty:
        return "- none"
    rows = frame.head(limit)
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join("---" for _ in columns) + "|"]
    for _, row in rows.iterrows():
        values: list[str] = []
        for column in columns:
            value = row.get(column)
            if isinstance(value, bool):
                values.append(str(value).lower())
            elif isinstance(value, (int, float, np.integer, np.floating)):
                values.append(format_number(value))
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def render_report(
    summary: dict[str, Any],
    gate_postmortem: pd.DataFrame,
    deep_dive: pd.DataFrame,
    funding_dependency: pd.DataFrame,
    concentration: pd.DataFrame,
    control_audit: pd.DataFrame,
    capture_quality: pd.DataFrame,
) -> str:
    """Render the required postmortem report."""

    focus = summary.get("focus_family")
    deep = deep_dive.iloc[0].to_dict() if not deep_dive.empty else {}
    funding_oos = funding_dependency[funding_dependency["scope"].astype(str) == "oos_ext"].iloc[0].to_dict() if not funding_dependency.empty and "scope" in funding_dependency.columns and not funding_dependency[funding_dependency["scope"].astype(str) == "oos_ext"].empty else {}
    concentration_all = concentration[(concentration["family"].astype(str) == str(focus)) & (concentration["scope"].astype(str) == "all_splits")].iloc[0].to_dict() if not concentration.empty and not concentration[(concentration["family"].astype(str) == str(focus)) & (concentration["scope"].astype(str) == "all_splits")].empty else {}
    control_all = control_audit[(control_audit["family"].astype(str) == str(focus)) & (control_audit["scope"].astype(str) == "all_splits")].iloc[0].to_dict() if not control_audit.empty and not control_audit[(control_audit["family"].astype(str) == str(focus)) & (control_audit["scope"].astype(str) == "all_splits")].empty else {}
    capture_focus = capture_quality[capture_quality["family"].astype(str) == str(focus)].iloc[0].to_dict() if not capture_quality.empty and not capture_quality[capture_quality["family"].astype(str) == str(focus)].empty else {}
    asset = bool(summary.get("focus_family_research_asset"))
    next_step = summary.get("recommended_next_step")
    cost_marginal = bool(deep.get("cost_failure_is_marginal"))
    structural_cost_drag = bool(deep.get("structural_cost_drag"))
    funding_dependent = bool(funding_oos.get("funding_dependent", False))
    concentration_risk = not bool(concentration_all.get("concentration_pass", False))
    control_pass = bool(control_all.get("control_pass", False))

    answers = [
        f"1. Trend Entry Timing Research failed? true. can_enter_entry_timing_phase2 is fixed to false because no candidate passed the stable-like gates.",
        f"2. Why can_enter_entry_timing_phase2=false? Gate failures remain across no-cost stability, cost/funding, early-entry quality, concentration, or controls; see candidate_gate_postmortem.csv.",
        f"3. Is {focus} a research asset? {str(asset).lower()}.",
        f"4. Why can it be no-cost positive across splits but stable_like=false? OOS cost-aware={format_number(deep.get('oos_cost_aware_pnl'))}, early_entry_rate={format_number(deep.get('early_entry_rate'))}, largest_symbol_pnl_share={format_number(deep.get('largest_symbol_pnl_share'))}.",
        f"5. Is OOS cost-aware negative only marginal? marginal={str(cost_marginal).lower()}, structural_cost_drag={str(structural_cost_drag).lower()}, avg_trade_gross={format_number(deep.get('avg_trade_gross'))}, avg_trade_cost={format_number(deep.get('avg_trade_cost'))}.",
        f"6. Does funding-adjusted turn positive because of funding? funding_dependent={str(funding_dependent).lower()}, signed_funding_pnl={format_number(funding_oos.get('signed_funding_pnl'))}, conservative_adjusted={format_number(funding_oos.get('funding_adjusted_pnl_conservative'))}.",
        f"7. Does it solve late entry? false. early_entry_rate={format_number(capture_focus.get('early_entry_rate'))}, median_entry_lag_pct={format_number(capture_focus.get('median_entry_lag_pct'))}.",
        f"8. Is it better than random/reverse control? control_pass={str(control_pass).lower()}, reverse_or_random_stronger={str(bool(control_all.get('reverse_or_random_stronger', False))).lower()}.",
        f"9. Is there concentration risk? {str(concentration_risk).lower()}; largest_symbol_pnl_share={format_number(concentration_all.get('largest_symbol_pnl_share'))}, remove_top_1_pnl={format_number(concentration_all.get('remove_top_1_pnl'))}.",
        "10. Phase 2 allowed? false.",
        "11. Formal strategy modification allowed? false.",
        "12. Demo/live allowed? false.",
        f"13. Recommended next step: {next_step}.",
    ]
    return (
        "# Trend Entry Timing Postmortem & Breadth Candidate Audit\n\n"
        "## Scope\n"
        "- This is a postmortem, not strategy development.\n"
        "- No candidate is marked tradable. Strategy modification, demo, and live remain disabled.\n"
        "- Lower-cost diagnostics are sensitivity checks only; they are not a pass.\n\n"
        "## Required Answers\n"
        + "\n".join(answers)
        + "\n\n"
        "## Gate Failure Summary\n"
        + markdown_table(
            gate_postmortem,
            [
                "family",
                "primary_failure_category",
                "failed_gate_count",
                "train_no_cost",
                "validation_no_cost",
                "oos_no_cost",
                "oos_cost",
                "oos_funding",
                "trend_recall",
                "early_entry_rate",
                "largest_symbol_pnl_share",
            ],
            limit=50,
        )
        + "\n\n"
        "## Focus Family Deep Dive\n"
        + markdown_table(
            deep_dive,
            [
                "family",
                "no_cost_all_splits_positive",
                "oos_cost_aware_pnl",
                "oos_funding_adjusted_pnl",
                "funding_dependent",
                "symbol_dependency",
                "timeframe_dependency",
                "top_trade_dependency",
                "early_entry_quality",
                "control_pass",
            ],
            limit=10,
        )
        + "\n\n"
        "## Final Gates\n"
        f"- focus_family_research_asset={str(asset).lower()}\n"
        "- can_enter_entry_timing_phase2=false\n"
        "- strategy_development_allowed=false\n"
        "- demo_live_allowed=false\n"
        f"- recommended_next_step={next_step}\n"
    )


def run_postmortem(
    *,
    research_dir: Path,
    trend_map_dir: Path,
    funding_dir: Path,
    output_dir: Path,
    focus_family: str,
    timezone_name: str,
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    """Run the Trend Entry Timing postmortem."""

    artifacts, warnings = load_artifacts(research_dir)
    trend_map_summary = read_json_optional(trend_map_dir, "trend_opportunity_summary.json", warnings)
    trend_map_quality = read_json_optional(trend_map_dir, "data_quality.json", warnings)
    if not funding_dir.exists():
        warnings.append(f"missing funding dir: {funding_dir}")

    family_summary = artifacts.get("candidate_entry_family_summary", pd.DataFrame())
    trade_tests = artifacts.get("candidate_entry_trade_tests", pd.DataFrame())
    rejected = artifacts.get("rejected_candidate_entry_families", pd.DataFrame())
    reverse = artifacts.get("candidate_entry_reverse_test", pd.DataFrame())
    random_control = artifacts.get("candidate_entry_random_control", pd.DataFrame())
    trend_entry_summary = artifacts.get("trend_entry_timing_summary", {})

    gate_postmortem = build_gate_postmortem(family_summary, rejected)
    concentration_postmortem = build_concentration_postmortem(family_summary, trade_tests, focus_family)
    control_audit = build_control_audit(focus_family, family_summary, trade_tests, reverse, random_control)
    capture_quality = build_capture_quality(family_summary)
    cost_sensitivity = build_cost_sensitivity(focus_family, family_summary, trade_tests)
    funding_dependency = build_funding_dependency(focus_family, family_summary, trade_tests)
    deep_dive = build_focus_deep_dive(focus_family, family_summary, trade_tests, concentration_postmortem, control_audit)
    asset = evaluate_research_asset(
        focus_family,
        family_summary,
        gate_postmortem,
        concentration_postmortem,
        control_audit,
        cost_sensitivity,
    )
    summary = summarize_decision(
        focus_family=focus_family,
        gate_postmortem=gate_postmortem,
        deep_dive=deep_dive,
        cost_sensitivity=cost_sensitivity,
        funding_dependency=funding_dependency,
        concentration=concentration_postmortem,
        control_audit=control_audit,
        capture_quality=capture_quality,
        asset=asset,
        warnings=warnings,
        trend_entry_summary=trend_entry_summary,
        trend_map_summary=trend_map_summary,
        trend_map_quality=trend_map_quality,
        funding_dir=funding_dir,
        timezone_name=timezone_name,
        output_dir=output_dir,
    )

    rejected_postmortem = gate_postmortem.loc[:, [column for column in gate_postmortem.columns if column in {
        "family",
        "selected_hold_label",
        "failed_gate_count",
        "primary_failure_category",
        "rejected_reasons",
        "no_cost_all_splits_positive",
        "oos_cost_failure",
        "oos_funding_failure",
        "early_entry_failure",
        "concentration_failure",
        "control_failure",
    }]].copy() if not gate_postmortem.empty else pd.DataFrame()

    output_dir.mkdir(parents=True, exist_ok=True)
    write_dataframe(output_dir / "candidate_gate_postmortem.csv", gate_postmortem)
    write_dataframe(output_dir / "breadth_candidate_deep_dive.csv", deep_dive)
    write_dataframe(output_dir / "cost_sensitivity.csv", cost_sensitivity)
    write_dataframe(output_dir / "funding_dependency.csv", funding_dependency)
    write_dataframe(output_dir / "entry_timing_concentration_postmortem.csv", concentration_postmortem)
    write_dataframe(output_dir / "entry_timing_control_audit.csv", control_audit)
    write_dataframe(output_dir / "entry_timing_capture_quality.csv", capture_quality)
    write_dataframe(output_dir / "rejected_candidate_entry_postmortem.csv", rejected_postmortem)
    (output_dir / "trend_entry_timing_postmortem_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "trend_entry_timing_postmortem_report.md").write_text(
        render_report(summary, gate_postmortem, deep_dive, funding_dependency, concentration_postmortem, control_audit, capture_quality),
        encoding="utf-8",
    )

    if logger is not None:
        log_event(
            logger,
            logging.INFO,
            "trend_entry_timing_postmortem_complete",
            "Trend Entry Timing postmortem complete",
            output_dir=str(output_dir),
            focus_family=focus_family,
            focus_family_research_asset=summary["focus_family_research_asset"],
            recommended_next_step=summary["recommended_next_step"],
        )
    return summary


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""

    args = parse_args(argv)
    ensure_headless_runtime()
    logger = setup_logging("postmortem_trend_entry_timing", verbose=bool(args.verbose))
    summary = run_postmortem(
        research_dir=resolve_path(args.research_dir),
        trend_map_dir=resolve_path(args.trend_map_dir),
        funding_dir=resolve_path(args.funding_dir),
        output_dir=resolve_path(args.output_dir),
        focus_family=str(args.focus_family),
        timezone_name=str(args.timezone),
        logger=logger,
    )
    print(json.dumps(clean_json(summary), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
