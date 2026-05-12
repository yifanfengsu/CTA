#!/usr/bin/env python3
"""Phase 1.5 diagnostics for cross-symbol breadth acceleration."""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from common_runtime import PROJECT_ROOT, ensure_headless_runtime, log_event, print_json_block, setup_logging, to_jsonable
from history_time_utils import DEFAULT_TIMEZONE


DEFAULT_ENTRY_TIMING_DIR = PROJECT_ROOT / "reports" / "research" / "trend_entry_timing"
DEFAULT_POSTMORTEM_DIR = PROJECT_ROOT / "reports" / "research" / "trend_entry_timing_postmortem"
DEFAULT_TREND_MAP_DIR = PROJECT_ROOT / "reports" / "research" / "trend_opportunity_map"
DEFAULT_FUNDING_DIR = PROJECT_ROOT / "data" / "funding" / "okx"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "reports" / "research" / "cross_symbol_breadth_phase15"
DEFAULT_FOCUS_FAMILY = "cross_symbol_breadth_acceleration"

SPLITS = ["train_ext", "validation_ext", "oos_ext"]
FIXED_NOTIONAL = 1000.0
OUTPUT_FILES = [
    "cross_symbol_breadth_phase15_report.md",
    "cross_symbol_breadth_phase15_summary.json",
    "cost_edge_margin_diagnostic.csv",
    "signal_strength_bucket_diagnostic.csv",
    "concentration_repair_diagnostic.csv",
    "funding_dependency_phase15.csv",
    "early_entry_improvement_diagnostic.csv",
    "control_robustness_phase15.csv",
    "phase15_rejected_reasons.csv",
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description="Cross-symbol breadth acceleration Phase 1.5 diagnostics.")
    parser.add_argument("--entry-timing-dir", default=str(DEFAULT_ENTRY_TIMING_DIR))
    parser.add_argument("--postmortem-dir", default=str(DEFAULT_POSTMORTEM_DIR))
    parser.add_argument("--trend-map-dir", default=str(DEFAULT_TREND_MAP_DIR))
    parser.add_argument("--funding-dir", default=str(DEFAULT_FUNDING_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--focus-family", default=DEFAULT_FOCUS_FAMILY)
    parser.add_argument("--timezone", default=DEFAULT_TIMEZONE)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def resolve_path(path_arg: str | Path) -> Path:
    """Resolve relative paths from project root."""

    path = Path(path_arg)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def finite_number(value: Any, default: float | None = None) -> float | None:
    """Return finite float or default."""

    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(number):
        return default
    return number


def read_csv_optional(directory: Path, filename: str, warnings: list[str]) -> pd.DataFrame:
    """Read optional CSV with warning on failure."""

    path = directory / filename
    if not path.exists():
        warnings.append(f"missing_input_csv:{path}")
        return pd.DataFrame()
    try:
        return pd.read_csv(path, low_memory=False)
    except Exception as exc:  # pragma: no cover - defensive branch
        warnings.append(f"failed_to_read_csv:{path}:{exc}")
        return pd.DataFrame()


def read_json_optional(directory: Path, filename: str, warnings: list[str]) -> dict[str, Any]:
    """Read optional JSON with warning on failure."""

    path = directory / filename
    if not path.exists():
        warnings.append(f"missing_input_json:{path}")
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - defensive branch
        warnings.append(f"failed_to_read_json:{path}:{exc}")
        return {}


def dataframe_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Return JSON-safe records."""

    if frame.empty:
        return []
    work = frame.copy().astype(object).where(pd.notna(frame), None)
    return json.loads(work.to_json(orient="records", force_ascii=False, date_format="iso"))


def write_dataframe(path: Path, frame: pd.DataFrame, columns: list[str] | None = None) -> None:
    """Write a DataFrame with stable columns."""

    out = frame.copy()
    if columns is not None:
        for column in columns:
            if column not in out.columns:
                out[column] = np.nan
        out = out.loc[:, columns]
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False, encoding="utf-8")


def bool_series(frame: pd.DataFrame, column: str) -> pd.Series:
    """Parse a bool-like column."""

    if frame.empty or column not in frame.columns:
        return pd.Series([False] * len(frame.index), index=frame.index)
    text = frame[column].astype(str).str.lower()
    return text.isin(["true", "1", "yes"])


def numeric_series(frame: pd.DataFrame, column: str) -> pd.Series:
    """Return numeric series or empty series."""

    if frame.empty or column not in frame.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def safe_sum(frame: pd.DataFrame, column: str) -> float:
    """Sum numeric column if available."""

    values = numeric_series(frame, column)
    if values.empty:
        return 0.0
    return float(values.fillna(0.0).sum())


def selected_hold(family_summary: pd.DataFrame, focus_family: str) -> str | None:
    """Return train-selected hold label for focus family."""

    if family_summary.empty or "family" not in family_summary.columns:
        return None
    row = family_summary[family_summary["family"].astype(str) == focus_family]
    if row.empty or "selected_hold_label" not in row.columns or pd.isna(row.iloc[0]["selected_hold_label"]):
        return None
    return str(row.iloc[0]["selected_hold_label"])


def focus_trades(trade_tests: pd.DataFrame, family_summary: pd.DataFrame, focus_family: str) -> pd.DataFrame:
    """Return selected-hold focus trades."""

    if trade_tests.empty or "family" not in trade_tests.columns:
        return pd.DataFrame()
    trades = trade_tests[trade_tests["family"].astype(str) == focus_family].copy()
    if "reverse" in trades.columns:
        trades = trades[~bool_series(trades, "reverse")].copy()
    hold = selected_hold(family_summary, focus_family)
    if hold and "hold_label" in trades.columns:
        trades = trades[trades["hold_label"].astype(str) == hold].copy()
    return trades.reset_index(drop=True)


def focus_events(events: pd.DataFrame, focus_family: str) -> pd.DataFrame:
    """Return focus events."""

    if events.empty or "family" not in events.columns:
        return pd.DataFrame()
    return events[events["family"].astype(str) == focus_family].copy().reset_index(drop=True)


def split_frame(frame: pd.DataFrame, split: str | None) -> pd.DataFrame:
    """Filter by split."""

    if split is None or frame.empty or "split" not in frame.columns:
        return frame.copy()
    return frame[frame["split"].astype(str) == split].copy()


def funding_pnl(frame: pd.DataFrame) -> pd.Series:
    """Return signed funding PnL, deriving if needed."""

    if frame.empty:
        return pd.Series(dtype=float)
    if "funding_pnl" in frame.columns:
        return numeric_series(frame, "funding_pnl").fillna(0.0)
    return numeric_series(frame, "funding_adjusted_pnl").fillna(0.0) - numeric_series(frame, "cost_aware_pnl").fillna(0.0)


def top_n_count(count: int, fraction: float) -> int:
    """Return at least one top-count for non-empty samples."""

    if count <= 0:
        return 0
    return max(1, int(math.ceil(count * fraction)))


def largest_symbol_pnl_share(frame: pd.DataFrame, pnl_column: str = "funding_adjusted_pnl") -> float | None:
    """Return largest positive symbol PnL share, or absolute share if no positive sum."""

    if frame.empty or "symbol" not in frame.columns or pnl_column not in frame.columns:
        return None
    grouped = frame.groupby("symbol", dropna=False)[pnl_column].sum()
    positive = grouped[grouped > 0]
    if not positive.empty and float(positive.sum()) > 0:
        return float(positive.max() / positive.sum())
    absolute = grouped.abs()
    if float(absolute.sum()) > 0:
        return float(absolute.max() / absolute.sum())
    return None


def top_trade_contribution(frame: pd.DataFrame, fraction: float = 0.05, pnl_column: str = "funding_adjusted_pnl") -> float | None:
    """Return top trade PnL contribution."""

    if frame.empty or pnl_column not in frame.columns:
        return None
    pnl = numeric_series(frame, pnl_column).fillna(0.0)
    total = float(pnl.sum())
    if abs(total) <= 1e-12:
        return None
    top_n = top_n_count(len(pnl.index), fraction)
    return float(pnl.sort_values(ascending=False, kind="stable").head(top_n).sum() / total)


def summarize_slice(frame: pd.DataFrame) -> dict[str, Any]:
    """Summarize a trade/event slice."""

    if frame.empty:
        return {
            "trade_count": 0,
            "no_cost_pnl": 0.0,
            "cost_aware_pnl": 0.0,
            "funding_adjusted_pnl": 0.0,
            "direction_match": None,
            "early_entry_rate": None,
            "trend_recall": None,
            "largest_symbol_pnl_share": None,
            "top_5pct_trade_pnl_contribution": None,
        }
    direction_match = None
    early_entry_rate = None
    trend_recall = None
    if "direction_matches_segment" in frame.columns:
        direction_match = float(bool_series(frame, "direction_matches_segment").mean())
    if "entry_phase" in frame.columns:
        early = frame["entry_phase"].astype(str).isin(["pre_trend", "first_10pct", "first_25pct"])
        if "direction_matches_segment" in frame.columns:
            early = early & bool_series(frame, "direction_matches_segment")
        early_entry_rate = float(early.mean())
    if "trend_segment_id" in frame.columns:
        matched = frame
        if "direction_matches_segment" in frame.columns:
            matched = frame[bool_series(frame, "direction_matches_segment")]
        if not matched.empty:
            trend_recall = int(matched["trend_segment_id"].dropna().astype(str).nunique())
    return {
        "trade_count": int(len(frame.index)),
        "no_cost_pnl": safe_sum(frame, "no_cost_pnl"),
        "cost_aware_pnl": safe_sum(frame, "cost_aware_pnl"),
        "funding_adjusted_pnl": safe_sum(frame, "funding_adjusted_pnl"),
        "direction_match": direction_match,
        "early_entry_rate": early_entry_rate,
        "trend_recall": trend_recall,
        "largest_symbol_pnl_share": largest_symbol_pnl_share(frame),
        "top_5pct_trade_pnl_contribution": top_trade_contribution(frame),
    }


def build_cost_edge_margin(trades: pd.DataFrame) -> pd.DataFrame:
    """Build OOS cost edge margin diagnostic."""

    oos = split_frame(trades, "oos_ext")
    trade_count = int(len(oos.index))
    gross = safe_sum(oos, "no_cost_pnl")
    cost_aware = safe_sum(oos, "cost_aware_pnl")
    funding_adjusted = safe_sum(oos, "funding_adjusted_pnl")
    total_cost = gross - cost_aware
    total_fee_cost = total_cost / 2.0
    total_slippage_cost = total_cost / 2.0
    avg_trade_gross = float(gross / trade_count) if trade_count else None
    avg_trade_cost = float(total_cost / trade_count) if trade_count else None
    break_even_total_per_side_bps = float(gross * 10000.0 / (2.0 * trade_count * FIXED_NOTIONAL)) if trade_count else None
    break_even_equal_fee_slippage_bps = float(gross * 10000.0 / (4.0 * trade_count * FIXED_NOTIONAL)) if trade_count else None
    current_total_per_side_bps = float(total_cost * 10000.0 / (2.0 * trade_count * FIXED_NOTIONAL)) if trade_count else None
    return pd.DataFrame(
        [
            {
                "family": DEFAULT_FOCUS_FAMILY,
                "split": "oos_ext",
                "trade_count": trade_count,
                "gross_edge_before_cost": gross,
                "oos_cost_aware_pnl": cost_aware,
                "oos_funding_adjusted_pnl": funding_adjusted,
                "total_fee_cost": total_fee_cost,
                "total_slippage_cost": total_slippage_cost,
                "total_cost": total_cost,
                "cost_to_edge_ratio": float(total_cost / gross) if abs(gross) > 1e-12 else None,
                "avg_trade_gross": avg_trade_gross,
                "avg_trade_cost": avg_trade_cost,
                "current_total_per_side_bps": current_total_per_side_bps,
                "break_even_fee_bps": break_even_total_per_side_bps,
                "break_even_slippage_bps": break_even_total_per_side_bps,
                "break_even_equal_fee_slippage_bps": break_even_equal_fee_slippage_bps,
                "cost_failure_is_marginal": bool(cost_aware < 0 and abs(cost_aware) <= max(50.0, abs(gross) * 0.02)),
                "execution_fragile": bool(cost_aware < 0 and avg_trade_cost is not None and avg_trade_gross is not None and avg_trade_cost >= avg_trade_gross),
                "trade_frequency_too_high": bool(trade_count >= 1000 and avg_trade_gross is not None and avg_trade_cost is not None and avg_trade_gross <= avg_trade_cost * 1.25),
            }
        ]
    )


def feature_direction_value(frame: pd.DataFrame, feature: str) -> pd.Series:
    """Return direction-aware feature values when useful."""

    raw = numeric_series(frame, feature)
    if raw.empty:
        return raw
    if feature in {"ret_3", "ret_6", "ret_20", "entry_price_vs_segment_start_price"} and "direction" in frame.columns:
        signs = np.where(frame["direction"].astype(str).str.lower() == "short", -1.0, 1.0)
        return raw * signs
    if feature == "funding_rate":
        return raw.abs()
    return raw


def assign_train_quantile_bucket(frame: pd.DataFrame, feature: str, warnings: list[str], bucket_count: int = 5) -> pd.Series:
    """Assign train-defined quantile buckets to all rows."""

    values = feature_direction_value(frame, feature)
    buckets = pd.Series(["missing"] * len(frame.index), index=frame.index, dtype=object)
    if values.empty or values.dropna().empty:
        warnings.append(f"signal_bucket_feature_unavailable:{feature}")
        return buckets
    train_values = values[frame["split"].astype(str) == "train_ext"].dropna() if "split" in frame.columns else values.dropna()
    if train_values.nunique() < 2:
        warnings.append(f"signal_bucket_feature_insufficient_variation:{feature}")
        buckets.loc[values.notna()] = "all"
        return buckets
    quantiles = sorted(set(float(item) for item in train_values.quantile(np.linspace(0.0, 1.0, bucket_count + 1)).to_list()))
    if len(quantiles) <= 2:
        buckets.loc[values.notna()] = "all"
        return buckets
    edges = [-np.inf] + quantiles[1:-1] + [np.inf]
    labels = [f"q{index + 1}" for index in range(len(edges) - 1)]
    buckets.loc[values.notna()] = pd.cut(values[values.notna()], bins=edges, labels=labels, include_lowest=True).astype(str)
    return buckets


def build_signal_strength_buckets(events: pd.DataFrame, trades: pd.DataFrame, warnings: list[str]) -> pd.DataFrame:
    """Build signal strength bucket diagnostics using available train-defined features."""

    desired = [
        "breadth_score",
        "breadth_count",
        "breadth_change",
        "market_return_breadth",
        "correlation",
        "dispersion",
        "direction_match_proxy",
        "funding_percentile",
        "funding_dispersion",
        "symbol_rank",
    ]
    available = [feature for feature in desired if feature in events.columns]
    if not available:
        available = [feature for feature in ["ret_3", "ret_6", "ret_20", "funding_rate", "entry_price_vs_segment_start_price"] if feature in events.columns]
        warnings.append(f"signal_strength_requested_fields_missing_using_fallback:{','.join(available)}")
    if events.empty or trades.empty or "event_id" not in events.columns or "event_id" not in trades.columns:
        return pd.DataFrame()
    joined = trades.merge(
        events.drop_duplicates("event_id"),
        on="event_id",
        how="left",
        suffixes=("", "_event"),
    )
    rows: list[dict[str, Any]] = []
    for feature in available:
        joined[f"_bucket_{feature}"] = assign_train_quantile_bucket(joined, feature, warnings)
        for bucket, group in joined.groupby(f"_bucket_{feature}", dropna=False):
            record: dict[str, Any] = {
                "feature": feature,
                "bucket": str(bucket),
                "event_count": int(group["event_id"].nunique()) if "event_id" in group.columns else int(len(group.index)),
            }
            for split in SPLITS:
                metrics = summarize_slice(split_frame(group, split))
                record[f"{split}_trade_count"] = metrics["trade_count"]
                record[f"{split}_no_cost_pnl"] = metrics["no_cost_pnl"]
                record[f"{split}_cost_aware_pnl"] = metrics["cost_aware_pnl"]
                record[f"{split}_funding_adjusted_pnl"] = metrics["funding_adjusted_pnl"]
            all_metrics = summarize_slice(group)
            train_no_cost = record["train_ext_no_cost_pnl"]
            validation_no_cost = record["validation_ext_no_cost_pnl"]
            oos_no_cost = record["oos_ext_no_cost_pnl"]
            record.update(
                {
                    "trade_count": all_metrics["trade_count"],
                    "train_no_cost": train_no_cost,
                    "validation_no_cost": validation_no_cost,
                    "oos_no_cost": oos_no_cost,
                    "oos_cost": record["oos_ext_cost_aware_pnl"],
                    "oos_funding": record["oos_ext_funding_adjusted_pnl"],
                    "direction_match": all_metrics["direction_match"],
                    "early_entry_rate": all_metrics["early_entry_rate"],
                    "trend_recall_proxy_count": all_metrics["trend_recall"],
                    "largest_symbol_pnl_share": all_metrics["largest_symbol_pnl_share"],
                    "top_5pct_trade_pnl_contribution": all_metrics["top_5pct_trade_pnl_contribution"],
                    "all_splits_no_cost_positive": bool(train_no_cost > 0 and validation_no_cost > 0 and oos_no_cost > 0),
                }
            )
            rows.append(record)
    return pd.DataFrame(rows)


def rescale_symbol_equal_weight(frame: pd.DataFrame) -> pd.DataFrame:
    """Equal-symbol rescale PnL columns without changing rows."""

    if frame.empty or "symbol" not in frame.columns:
        return frame.copy()
    out = frame.copy()
    counts = out.groupby("symbol", dropna=False).size()
    target = float(counts.mean()) if not counts.empty else 1.0
    weights = {symbol: target / count for symbol, count in counts.items() if count > 0}
    for column in ["no_cost_pnl", "cost_aware_pnl", "funding_adjusted_pnl", "funding_pnl"]:
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce").fillna(0.0) * out["symbol"].map(weights).fillna(1.0)
    return out


def cap_symbol_pnl_share(frame: pd.DataFrame, cap_share: float = 0.70) -> pd.DataFrame:
    """Downscale largest positive symbol PnL when it exceeds cap share."""

    if frame.empty or "symbol" not in frame.columns:
        return frame.copy()
    out = frame.copy()
    grouped = out.groupby("symbol", dropna=False)["funding_adjusted_pnl"].sum()
    positive = grouped[grouped > 0]
    if positive.empty or float(positive.sum()) <= 0:
        return out
    largest_symbol = positive.idxmax()
    other_positive = float(positive.sum() - positive.max())
    if other_positive <= 0:
        scale = cap_share
    else:
        target_largest = cap_share * other_positive / max(1.0 - cap_share, 1e-12)
        scale = min(1.0, target_largest / float(positive.max()))
    if scale >= 1.0:
        return out
    mask = out["symbol"] == largest_symbol
    for column in ["no_cost_pnl", "cost_aware_pnl", "funding_adjusted_pnl", "funding_pnl"]:
        if column in out.columns:
            out.loc[mask, column] = pd.to_numeric(out.loc[mask, column], errors="coerce").fillna(0.0) * scale
    return out


def cap_symbol_trade_share(frame: pd.DataFrame, cap_share: float = 0.30) -> pd.DataFrame:
    """Downscale symbols above a trade-share cap."""

    if frame.empty or "symbol" not in frame.columns:
        return frame.copy()
    out = frame.copy()
    total = len(out.index)
    counts = out.groupby("symbol", dropna=False).size()
    weights = {symbol: min(1.0, cap_share * total / count) for symbol, count in counts.items() if count > 0}
    for column in ["no_cost_pnl", "cost_aware_pnl", "funding_adjusted_pnl", "funding_pnl"]:
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce").fillna(0.0) * out["symbol"].map(weights).fillna(1.0)
    return out


def remove_largest_symbol(frame: pd.DataFrame) -> pd.DataFrame:
    """Remove largest positive PnL symbol, or largest absolute symbol if none positive."""

    if frame.empty or "symbol" not in frame.columns:
        return frame.copy()
    grouped = frame.groupby("symbol", dropna=False)["funding_adjusted_pnl"].sum()
    positive = grouped[grouped > 0]
    symbol = positive.sort_values(ascending=False, kind="stable").index[0] if not positive.empty else grouped.abs().sort_values(ascending=False, kind="stable").index[0]
    return frame[frame["symbol"] != symbol].copy()


def remove_top_trades(frame: pd.DataFrame, fraction: float | None) -> pd.DataFrame:
    """Remove top PnL trades."""

    if frame.empty or "funding_adjusted_pnl" not in frame.columns:
        return frame.copy()
    top_count = 1 if fraction is None else top_n_count(len(frame.index), fraction)
    drop_index = numeric_series(frame, "funding_adjusted_pnl").sort_values(ascending=False, kind="stable").head(top_count).index
    return frame.drop(index=drop_index).copy()


def build_concentration_repair(trades: pd.DataFrame) -> pd.DataFrame:
    """Build concentration repair diagnostics."""

    actions = {
        "original": lambda frame: frame.copy(),
        "cap_single_symbol_trade_share": cap_symbol_trade_share,
        "cap_single_symbol_pnl_share": cap_symbol_pnl_share,
        "equal_symbol_weight_rescale": rescale_symbol_equal_weight,
        "remove_largest_symbol": remove_largest_symbol,
        "remove_top_1_trade": lambda frame: remove_top_trades(frame, None),
        "remove_top_5pct_trades": lambda frame: remove_top_trades(frame, 0.05),
    }
    rows: list[dict[str, Any]] = []
    for scope, base in [("all_splits", trades), ("oos_ext", split_frame(trades, "oos_ext"))]:
        for action, func in actions.items():
            repaired = func(base)
            metrics = summarize_slice(repaired)
            rows.append(
                {
                    "scope": scope,
                    "repair_action": action,
                    "trade_count": metrics["trade_count"],
                    "no_cost_pnl": metrics["no_cost_pnl"],
                    "cost_aware_pnl": metrics["cost_aware_pnl"],
                    "funding_adjusted_pnl": metrics["funding_adjusted_pnl"],
                    "largest_symbol_pnl_share": metrics["largest_symbol_pnl_share"],
                    "top_5pct_trade_pnl_contribution": metrics["top_5pct_trade_pnl_contribution"],
                    "non_disastrous": bool(metrics["funding_adjusted_pnl"] >= -100.0),
                    "still_positive_no_cost": bool(metrics["no_cost_pnl"] > 0),
                    "still_positive_funding_adjusted": bool(metrics["funding_adjusted_pnl"] > 0),
                }
            )
    return pd.DataFrame(rows)


def build_funding_dependency(trades: pd.DataFrame, postmortem_funding: pd.DataFrame) -> pd.DataFrame:
    """Build Phase 1.5 funding dependency diagnostic."""

    rows: list[dict[str, Any]] = []
    for scope, frame in [("all_splits", trades), ("oos_ext", split_frame(trades, "oos_ext"))]:
        signed = float(funding_pnl(frame).sum()) if not frame.empty else 0.0
        positive = float(funding_pnl(frame)[funding_pnl(frame) > 0].sum()) if not frame.empty else 0.0
        negative = float(funding_pnl(frame)[funding_pnl(frame) < 0].sum()) if not frame.empty else 0.0
        no_cost = safe_sum(frame, "no_cost_pnl")
        cost_aware = safe_sum(frame, "cost_aware_pnl")
        funding_adjusted = safe_sum(frame, "funding_adjusted_pnl")
        conservative = cost_aware + negative
        positive_due_to_funding = bool(cost_aware <= 0 < funding_adjusted and signed > 0)
        rows.append(
            {
                "scope": scope,
                "trade_count": int(len(frame.index)),
                "no_cost_pnl": no_cost,
                "cost_aware_without_funding": cost_aware,
                "funding_pnl": signed,
                "positive_funding_pnl": positive,
                "negative_funding_pnl": negative,
                "funding_adjusted_minus_cost": signed,
                "funding_adjusted_pnl": funding_adjusted,
                "conservative_funding_adjusted_pnl": conservative,
                "funding_positive_contribution_share": float(positive / abs(funding_adjusted)) if abs(funding_adjusted) > 1e-12 else None,
                "positive_due_to_funding": positive_due_to_funding,
                "funding_carry_contaminated": bool(positive_due_to_funding or (funding_adjusted > 0 and positive > abs(cost_aware))),
            }
        )
    if not postmortem_funding.empty:
        rows.append(
            {
                "scope": "postmortem_reference",
                "trade_count": None,
                "no_cost_pnl": None,
                "cost_aware_without_funding": None,
                "funding_pnl": None,
                "positive_funding_pnl": None,
                "negative_funding_pnl": None,
                "funding_adjusted_minus_cost": None,
                "funding_adjusted_pnl": None,
                "conservative_funding_adjusted_pnl": None,
                "funding_positive_contribution_share": None,
                "positive_due_to_funding": None,
                "funding_carry_contaminated": bool((postmortem_funding.get("funding_dependent", pd.Series(dtype=bool)).astype(str).str.lower() == "true").any()),
            }
        )
    return pd.DataFrame(rows)


def subgroup_metrics(name: str, frame: pd.DataFrame) -> dict[str, Any]:
    """Build subgroup metrics for early-entry diagnostics."""

    record: dict[str, Any] = {"diagnostic_group": name, "event_count": int(frame["event_id"].nunique()) if "event_id" in frame.columns else int(len(frame.index))}
    for split in SPLITS:
        metrics = summarize_slice(split_frame(frame, split))
        record[f"{split}_trade_count"] = metrics["trade_count"]
        record[f"{split}_no_cost_pnl"] = metrics["no_cost_pnl"]
        record[f"{split}_cost_aware_pnl"] = metrics["cost_aware_pnl"]
        record[f"{split}_funding_adjusted_pnl"] = metrics["funding_adjusted_pnl"]
    metrics = summarize_slice(frame)
    record.update(
        {
            "trade_count": metrics["trade_count"],
            "early_entry_rate": metrics["early_entry_rate"],
            "direction_match": metrics["direction_match"],
            "median_entry_lag_pct": float(numeric_series(frame, "entry_lag_pct_of_segment").median()) if "entry_lag_pct_of_segment" in frame.columns else None,
            "oos_no_cost": record["oos_ext_no_cost_pnl"],
            "early_entry_improved_to_035": bool((metrics["early_entry_rate"] or 0.0) >= 0.35),
            "direction_and_pnl_not_sacrificed": bool((metrics["direction_match"] or 0.0) >= 0.60 and record["oos_ext_no_cost_pnl"] > 0),
        }
    )
    return record


def build_early_entry_improvement(events: pd.DataFrame, trades: pd.DataFrame, warnings: list[str]) -> pd.DataFrame:
    """Build early-entry improvement proxy diagnostics."""

    if events.empty or trades.empty or "event_id" not in events.columns:
        return pd.DataFrame()
    joined = trades.merge(events.drop_duplicates("event_id"), on="event_id", how="left", suffixes=("", "_event"))
    if "event_time" in joined.columns:
        joined["_event_time_ts"] = pd.to_datetime(joined["event_time"], errors="coerce")
    else:
        joined["_event_time_ts"] = pd.to_datetime(joined.get("entry_time"), errors="coerce")
    key_cols = [column for column in ["trend_segment_id", "symbol", "timeframe", "direction"] if column in joined.columns]
    if key_cols:
        joined["_order_in_segment"] = joined.sort_values("_event_time_ts", kind="stable").groupby(key_cols, dropna=False).cumcount() + 1
    else:
        joined["_order_in_segment"] = 1
    train = joined[joined["split"].astype(str) == "train_ext"] if "split" in joined.columns else joined
    abs_funding_threshold = float(numeric_series(train, "funding_rate").abs().median()) if "funding_rate" in train.columns and not train.empty else None
    ret20_threshold = float(numeric_series(train, "ret_20").abs().median()) if "ret_20" in train.columns and not train.empty else None
    ret3_abs_threshold = float(numeric_series(train, "ret_3").abs().median()) if "ret_3" in train.columns and not train.empty else None
    groups: dict[str, pd.DataFrame] = {
        "all_focus_events": joined,
        "first_breadth_acceleration_event": joined[joined["_order_in_segment"] == 1],
        "repeated_breadth_confirmation_event": joined[joined["_order_in_segment"] >= 2],
    }
    if abs_funding_threshold is not None and ret20_threshold is not None:
        groups["breadth_acceleration_after_compression"] = joined[(numeric_series(joined, "funding_rate").abs() <= abs_funding_threshold) & (numeric_series(joined, "ret_20").abs() <= ret20_threshold)]
    else:
        warnings.append("early_entry_compression_proxy_unavailable")
    if ret3_abs_threshold is not None:
        groups["breadth_acceleration_with_low_dispersion"] = joined[numeric_series(joined, "ret_3").abs() <= ret3_abs_threshold]
    else:
        warnings.append("early_entry_low_dispersion_proxy_unavailable")
    if "ret_3" in joined.columns:
        direction_aware_ret3 = feature_direction_value(joined, "ret_3")
        groups["breadth_acceleration_with_positive_market_return"] = joined[direction_aware_ret3 > 0]
    else:
        warnings.append("early_entry_positive_market_return_proxy_unavailable")
    return pd.DataFrame([subgroup_metrics(name, group) for name, group in groups.items()])


def build_control_robustness(
    trades: pd.DataFrame,
    reverse: pd.DataFrame,
    random_control: pd.DataFrame,
    family_summary: pd.DataFrame,
    focus_family: str,
    seeds: int = 100,
) -> pd.DataFrame:
    """Build reverse and 100-seed lightweight random-control robustness diagnostics."""

    reverse_focus = pd.DataFrame()
    if not reverse.empty and "family" in reverse.columns:
        reverse_focus = reverse[reverse["family"].astype(str) == focus_family].copy()
        if "reverse" in reverse_focus.columns:
            reverse_focus = reverse_focus[bool_series(reverse_focus, "reverse")].copy()
        hold = selected_hold(family_summary, focus_family)
        if hold and "hold_label" in reverse_focus.columns:
            reverse_focus = reverse_focus[reverse_focus["hold_label"].astype(str) == hold].copy()
    random_focus = focus_trades(random_control, family_summary, focus_family)
    rows: list[dict[str, Any]] = []
    for scope, split in [("all_splits", None), ("train_ext", "train_ext"), ("validation_ext", "validation_ext"), ("oos_ext", "oos_ext")]:
        forward = split_frame(trades, split)
        rev = split_frame(reverse_focus, split)
        rnd = split_frame(random_focus, split)
        forward_pnl = safe_sum(forward, "funding_adjusted_pnl")
        reverse_pnl = safe_sum(rev, "funding_adjusted_pnl")
        random_pnl = safe_sum(rnd, "funding_adjusted_pnl")
        bootstrap: list[float] = []
        if not rnd.empty and not forward.empty:
            group_cols = [column for column in ["symbol", "direction", "split"] if column in forward.columns and column in rnd.columns]
            for seed in range(seeds):
                rng = np.random.default_rng(seed)
                total = 0.0
                if group_cols:
                    for keys, fwd_group in forward.groupby(group_cols, dropna=False):
                        if not isinstance(keys, tuple):
                            keys = (keys,)
                        pool = rnd.copy()
                        for column, key in zip(group_cols, keys, strict=False):
                            pool = pool[pool[column] == key]
                        if pool.empty:
                            pool = rnd
                        sampled = pool.iloc[rng.integers(0, len(pool.index), size=len(fwd_group.index))]
                        total += float(pd.to_numeric(sampled["funding_adjusted_pnl"], errors="coerce").fillna(0.0).sum())
                else:
                    sampled = rnd.iloc[rng.integers(0, len(rnd.index), size=len(forward.index))]
                    total = float(pd.to_numeric(sampled["funding_adjusted_pnl"], errors="coerce").fillna(0.0).sum())
                bootstrap.append(total)
        bootstrap_array = np.array(bootstrap, dtype=float)
        rows.append(
            {
                "scope": scope,
                "forward_trade_count": int(len(forward.index)),
                "reverse_trade_count": int(len(rev.index)),
                "random_trade_count": int(len(rnd.index)),
                "forward_funding_adjusted_pnl": forward_pnl,
                "reverse_funding_adjusted_pnl": reverse_pnl if not rev.empty else None,
                "random_original_funding_adjusted_pnl": random_pnl if not rnd.empty else None,
                "reverse_weaker": bool(not rev.empty and reverse_pnl < forward_pnl),
                "random_original_weaker": bool(not rnd.empty and random_pnl < forward_pnl),
                "random_bootstrap_mean": float(bootstrap_array.mean()) if bootstrap_array.size else None,
                "random_bootstrap_p05": float(np.quantile(bootstrap_array, 0.05)) if bootstrap_array.size else None,
                "random_bootstrap_p95": float(np.quantile(bootstrap_array, 0.95)) if bootstrap_array.size else None,
                "random_share_ge_forward": float((bootstrap_array >= forward_pnl).mean()) if bootstrap_array.size else None,
                "significantly_beats_random": bool(bootstrap_array.size and float((bootstrap_array >= forward_pnl).mean()) <= 0.05),
                "control_robust": bool((not rev.empty and reverse_pnl < forward_pnl) and (bootstrap_array.size and float((bootstrap_array >= forward_pnl).mean()) <= 0.05)),
            }
        )
    return pd.DataFrame(rows)


def phase15_reasons(
    *,
    cost_edge: pd.DataFrame,
    concentration: pd.DataFrame,
    funding: pd.DataFrame,
    early_entry: pd.DataFrame,
    control: pd.DataFrame,
    family_row: pd.Series | None,
) -> tuple[pd.DataFrame, bool]:
    """Evaluate Phase 1.5 research-asset criteria."""

    reasons: list[dict[str, Any]] = []

    def add(condition: str, passed: bool, observed: Any, threshold: str) -> None:
        reasons.append({"condition": condition, "passed": bool(passed), "observed": observed, "threshold": threshold})

    if family_row is None:
        add("family_summary_available", False, "missing", "required")
    else:
        no_cost = {split: finite_number(family_row.get(f"{split}_no_cost_pnl"), 0.0) for split in SPLITS}
        add("three_splits_no_cost_positive", all((value or 0.0) > 0 for value in no_cost.values()), no_cost, "> 0 each split")
        add("oos_funding_adjusted_positive", (finite_number(family_row.get("oos_ext_funding_adjusted_pnl"), 0.0) or 0.0) > 0, family_row.get("oos_ext_funding_adjusted_pnl"), "> 0")

    cost_row = cost_edge.iloc[0].to_dict() if not cost_edge.empty else {}
    break_even = finite_number(cost_row.get("break_even_equal_fee_slippage_bps"), None)
    add(
        "cost_failure_marginal_and_realistic",
        bool(cost_row.get("cost_failure_is_marginal", False) and break_even is not None and break_even >= 2.0),
        {"cost_failure_is_marginal": cost_row.get("cost_failure_is_marginal"), "break_even_equal_fee_slippage_bps": break_even},
        "marginal and equal fee/slippage break-even >= 2 bps",
    )

    control_all = control[control["scope"].astype(str) == "all_splits"].iloc[0].to_dict() if not control.empty and not control[control["scope"].astype(str) == "all_splits"].empty else {}
    add("control_robustness", bool(control_all.get("control_robust", False)), control_all, "reverse weaker and random_share_ge_forward <= 0.05")

    repair_all = concentration[(concentration["scope"].astype(str) == "all_splits") & (concentration["repair_action"].astype(str).isin(["cap_single_symbol_pnl_share", "equal_symbol_weight_rescale", "remove_largest_symbol"]))] if not concentration.empty else pd.DataFrame()
    repair_ok = bool(not repair_all.empty and (repair_all["non_disastrous"].astype(bool) & (pd.to_numeric(repair_all["no_cost_pnl"], errors="coerce") > 0)).any())
    top_rows = concentration[(concentration["scope"].astype(str) == "all_splits") & (concentration["repair_action"].astype(str).isin(["remove_top_1_trade", "remove_top_5pct_trades"]))] if not concentration.empty else pd.DataFrame()
    top_ok = bool(not top_rows.empty and (pd.to_numeric(top_rows["no_cost_pnl"], errors="coerce") > 0).all() and (pd.to_numeric(top_rows["funding_adjusted_pnl"], errors="coerce") > -500.0).all())
    add("concentration_repair_non_disastrous", repair_ok, dataframe_records(repair_all), "constraint/rescale still no-cost positive and non-disastrous")
    add("top_trade_dependency_controllable", top_ok, dataframe_records(top_rows), "remove top 1/top 5pct remains no-cost positive and not deeply negative")

    funding_oos = funding[funding["scope"].astype(str) == "oos_ext"].iloc[0].to_dict() if not funding.empty and not funding[funding["scope"].astype(str) == "oos_ext"].empty else {}
    add("funding_not_primary_source", not bool(funding_oos.get("funding_carry_contaminated", True)), funding_oos, "funding_carry_contaminated=false")

    early_rows = early_entry[early_entry["diagnostic_group"].astype(str) != "all_focus_events"] if not early_entry.empty else pd.DataFrame()
    early_ok = bool(not early_rows.empty and ((pd.to_numeric(early_rows["early_entry_rate"], errors="coerce") >= 0.35) & early_rows["direction_and_pnl_not_sacrificed"].astype(bool)).any())
    add("early_entry_improvement_signal", early_ok, dataframe_records(early_rows), "early_entry_rate >= 0.35 without direction/PnL sacrifice")

    research_asset = all(row["passed"] for row in reasons)
    return pd.DataFrame(reasons), bool(research_asset)


def continuous_plateau_exists(bucket_diag: pd.DataFrame) -> bool:
    """Return whether any feature has two adjacent good buckets."""

    if bucket_diag.empty:
        return False
    for _, group in bucket_diag.groupby("feature", dropna=False):
        ordered = group[group["bucket"].astype(str).str.match(r"q\d+")].copy()
        if ordered.empty:
            continue
        ordered["_q"] = ordered["bucket"].astype(str).str[1:].astype(int)
        ordered = ordered.sort_values("_q")
        good = ordered["all_splits_no_cost_positive"].astype(bool) & (pd.to_numeric(ordered["direction_match"], errors="coerce") >= 0.60)
        q_values = ordered.loc[good, "_q"].to_list()
        if any((q + 1) in q_values for q in q_values):
            return True
    return False


def render_report(summary: dict[str, Any]) -> str:
    """Render markdown report."""

    return (
        "# Cross-Symbol Breadth Acceleration Phase 1.5\n\n"
        "## Scope\n"
        "- This is a near-miss lead diagnostic, not strategy development.\n"
        "- It does not modify entries, formal strategies, demo runners, live runners, or API keys.\n"
        "- `research_asset` does not mean tradable policy.\n\n"
        "## Required Answers\n"
        f"1. Why keep looking? It is the only candidate with train/validation/oos no-cost positive and high direction_match; no-cost positive all splits={str(summary.get('three_splits_no_cost_positive')).lower()}.\n"
        f"2. Why not stable? {summary.get('stable_blockers')}.\n"
        f"3. OOS cost-aware -13.86 marginal or structural? {summary.get('cost_edge_margin')}.\n"
        f"4. Funding-adjusted positive depends on funding? {str(summary.get('funding_dependency')).lower()}.\n"
        f"5. Can concentration_fail be repaired? {str(summary.get('concentration_repair')).lower()}.\n"
        f"6. Is top-trade dependency healthy? {str(summary.get('top_trade_dependency_healthy')).lower()}.\n"
        f"7. Is there a signal-strength plateau? {str(summary.get('signal_strength_plateau')).lower()}.\n"
        f"8. Is there an earlier breadth trigger? {str(summary.get('early_entry_improvement')).lower()}.\n"
        f"9. Is it significantly better than random/reverse? {str(summary.get('control_robustness')).lower()}.\n"
        f"10. Upgrade to research_asset? {str(summary.get('research_asset')).lower()}.\n"
        "11. Strategy development allowed? false.\n"
        "12. Demo/live allowed? false.\n\n"
        "## Decision\n"
        f"- research_asset={str(summary.get('research_asset')).lower()}\n"
        f"- recommended_next_step={summary.get('recommended_next_step')}\n"
        "- strategy_development_allowed=false\n"
        "- demo_live_allowed=false\n"
    )


def run_research(
    *,
    entry_timing_dir: Path,
    postmortem_dir: Path,
    trend_map_dir: Path,
    funding_dir: Path,
    output_dir: Path,
    focus_family: str,
    timezone_name: str,
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    """Run Phase 1.5 breadth diagnostics."""

    warnings: list[str] = []
    events = read_csv_optional(entry_timing_dir, "candidate_entry_events.csv", warnings)
    trade_tests = read_csv_optional(entry_timing_dir, "candidate_entry_trade_tests.csv", warnings)
    family_summary = read_csv_optional(entry_timing_dir, "candidate_entry_family_summary.csv", warnings)
    _entry_concentration = read_csv_optional(entry_timing_dir, "candidate_entry_concentration.csv", warnings)
    reverse = read_csv_optional(entry_timing_dir, "candidate_entry_reverse_test.csv", warnings)
    random_control = read_csv_optional(entry_timing_dir, "candidate_entry_random_control.csv", warnings)
    deep_dive = read_csv_optional(postmortem_dir, "breadth_candidate_deep_dive.csv", warnings)
    post_cost_sensitivity = read_csv_optional(postmortem_dir, "cost_sensitivity.csv", warnings)
    post_funding = read_csv_optional(postmortem_dir, "funding_dependency.csv", warnings)
    post_concentration = read_csv_optional(postmortem_dir, "entry_timing_concentration_postmortem.csv", warnings)
    post_control = read_csv_optional(postmortem_dir, "entry_timing_control_audit.csv", warnings)
    post_capture = read_csv_optional(postmortem_dir, "entry_timing_capture_quality.csv", warnings)
    trend_summary = read_json_optional(trend_map_dir, "trend_opportunity_summary.json", warnings)
    _trend_quality = read_json_optional(trend_map_dir, "data_quality.json", warnings)
    if not funding_dir.exists():
        warnings.append(f"missing_funding_dir:{funding_dir}")

    trades = focus_trades(trade_tests, family_summary, focus_family)
    events_focus = focus_events(events, focus_family)
    family_rows = family_summary[family_summary["family"].astype(str) == focus_family] if not family_summary.empty and "family" in family_summary.columns else pd.DataFrame()
    family_row = family_rows.iloc[0] if not family_rows.empty else None

    cost_edge = build_cost_edge_margin(trades)
    signal_buckets = build_signal_strength_buckets(events_focus, trades, warnings)
    concentration_repair = build_concentration_repair(trades)
    funding_dependency = build_funding_dependency(trades, post_funding)
    early_entry = build_early_entry_improvement(events_focus, trades, warnings)
    control = build_control_robustness(trades, reverse, random_control, family_summary, focus_family)
    rejected, research_asset = phase15_reasons(
        cost_edge=cost_edge,
        concentration=concentration_repair,
        funding=funding_dependency,
        early_entry=early_entry,
        control=control,
        family_row=family_row,
    )

    cost_row = cost_edge.iloc[0].to_dict() if not cost_edge.empty else {}
    funding_oos = funding_dependency[funding_dependency["scope"].astype(str) == "oos_ext"].iloc[0].to_dict() if not funding_dependency.empty and not funding_dependency[funding_dependency["scope"].astype(str) == "oos_ext"].empty else {}
    concentration_ok = bool(rejected[rejected["condition"] == "concentration_repair_non_disastrous"]["passed"].iloc[0]) if not rejected.empty else False
    top_ok = bool(rejected[rejected["condition"] == "top_trade_dependency_controllable"]["passed"].iloc[0]) if not rejected.empty else False
    early_ok = bool(rejected[rejected["condition"] == "early_entry_improvement_signal"]["passed"].iloc[0]) if not rejected.empty else False
    control_ok = bool(rejected[rejected["condition"] == "control_robustness"]["passed"].iloc[0]) if not rejected.empty else False
    stable_blockers = ";".join(rejected.loc[~rejected["passed"].astype(bool), "condition"].astype(str).to_list()) if not rejected.empty else "missing_diagnostics"
    summary = {
        "mode": "cross_symbol_breadth_acceleration_phase15_diagnostic",
        "focus_family": focus_family,
        "output_dir": str(output_dir),
        "output_files": OUTPUT_FILES,
        "timezone": timezone_name,
        "warnings": sorted(dict.fromkeys(warnings)),
        "trend_opportunity_map": {
            "enough_trend_opportunities": trend_summary.get("enough_trend_opportunities"),
            "legacy_main_failure_mode": (trend_summary.get("legacy_analysis") or {}).get("main_failure_mode"),
        },
        "input_reference_rows": {
            "deep_dive": dataframe_records(deep_dive),
            "post_cost_sensitivity_rows": len(post_cost_sensitivity.index) if not post_cost_sensitivity.empty else 0,
            "post_concentration_rows": len(post_concentration.index) if not post_concentration.empty else 0,
            "post_control_rows": len(post_control.index) if not post_control.empty else 0,
            "post_capture_rows": len(post_capture.index) if not post_capture.empty else 0,
        },
        "three_splits_no_cost_positive": bool(family_row is not None and all((finite_number(family_row.get(f"{split}_no_cost_pnl"), 0.0) or 0.0) > 0 for split in SPLITS)),
        "cost_edge_margin": "marginal_execution_fragile" if cost_row.get("cost_failure_is_marginal") else "not_marginal",
        "funding_dependency": bool(funding_oos.get("funding_carry_contaminated", True)),
        "concentration_repair": concentration_ok,
        "top_trade_dependency_healthy": top_ok,
        "signal_strength_plateau": continuous_plateau_exists(signal_buckets),
        "early_entry_improvement": early_ok,
        "control_robustness": control_ok,
        "stable_blockers": stable_blockers,
        "research_asset": bool(research_asset),
        "recommended_next_step": "threshold_plateau_and_execution_phase" if research_asset else "pause_or_new_hypothesis",
        "strategy_development_allowed": False,
        "demo_live_allowed": False,
        "no_policy_can_be_traded": True,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    write_dataframe(output_dir / "cost_edge_margin_diagnostic.csv", cost_edge)
    write_dataframe(output_dir / "signal_strength_bucket_diagnostic.csv", signal_buckets)
    write_dataframe(output_dir / "concentration_repair_diagnostic.csv", concentration_repair)
    write_dataframe(output_dir / "funding_dependency_phase15.csv", funding_dependency)
    write_dataframe(output_dir / "early_entry_improvement_diagnostic.csv", early_entry)
    write_dataframe(output_dir / "control_robustness_phase15.csv", control)
    write_dataframe(output_dir / "phase15_rejected_reasons.csv", rejected)
    (output_dir / "cross_symbol_breadth_phase15_summary.json").write_text(json.dumps(to_jsonable(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "cross_symbol_breadth_phase15_report.md").write_text(render_report(summary), encoding="utf-8")

    if logger is not None:
        log_event(
            logger,
            logging.INFO,
            "breadth_phase15.complete",
            "Cross-symbol breadth Phase 1.5 diagnostics complete",
            output_dir=str(output_dir),
            research_asset=summary["research_asset"],
            recommended_next_step=summary["recommended_next_step"],
        )
    return summary


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""

    args = parse_args(argv)
    ensure_headless_runtime()
    logger = setup_logging("research_cross_symbol_breadth_phase15", verbose=bool(args.verbose))
    summary = run_research(
        entry_timing_dir=resolve_path(args.entry_timing_dir),
        postmortem_dir=resolve_path(args.postmortem_dir),
        trend_map_dir=resolve_path(args.trend_map_dir),
        funding_dir=resolve_path(args.funding_dir),
        output_dir=resolve_path(args.output_dir),
        focus_family=str(args.focus_family),
        timezone_name=str(args.timezone),
        logger=logger,
    )
    print_json_block(
        "Cross-Symbol Breadth Phase 1.5 summary:",
        {
            "output_dir": summary.get("output_dir"),
            "research_asset": summary.get("research_asset"),
            "strategy_development_allowed": summary.get("strategy_development_allowed"),
            "demo_live_allowed": summary.get("demo_live_allowed"),
            "recommended_next_step": summary.get("recommended_next_step"),
            "stable_blockers": summary.get("stable_blockers"),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
