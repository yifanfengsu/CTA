#!/usr/bin/env python3
"""Compare Trend Following V3 research across train/validation/OOS splits."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from common_runtime import PROJECT_ROOT, ensure_headless_runtime, log_event, print_json_block, setup_logging, to_jsonable


SPLITS = ["train", "validation", "oos"]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "reports" / "research" / "trend_following_v3_compare"
DEFAULT_EXTENDED_OUTPUT_DIR = PROJECT_ROOT / "reports" / "research" / "trend_following_v3_extended_compare"
SPLIT_LABELS_BY_SCHEME = {
    "default": {"train": "train", "validation": "validation", "oos": "oos"},
    "extended": {"train": "train_ext", "validation": "validation_ext", "oos": "oos_ext"},
}
OUTPUT_PREFIX_BY_SCHEME = {
    "default": "trend_v3_compare",
    "extended": "trend_v3_extended_compare",
}
DEFAULT_FUNDING_BPS = [1.0, 3.0, 5.0, 10.0]
MIN_TRADE_COUNT = 10
MAX_OOS_DDPERCENT = 30.0
MAX_SYMBOL_PNL_SHARE = 0.7
MAX_TOP_5PCT_CONTRIBUTION = 0.8


class TrendFollowingV3CompareError(Exception):
    """Raised when Trend Following V3 comparison cannot continue."""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description="Compare Trend Following V3 research across sample splits.")
    parser.add_argument("--train-dir", required=True, help="Train trend_following_v3 directory.")
    parser.add_argument("--validation-dir", required=True, help="Validation trend_following_v3 directory.")
    parser.add_argument("--oos-dir", required=True, help="OOS trend_following_v3 directory.")
    parser.add_argument("--split-scheme", choices=sorted(SPLIT_LABELS_BY_SCHEME), default="default")
    parser.add_argument(
        "--output-dir",
        help="Output directory. Default depends on --split-scheme.",
    )
    parser.add_argument("--funding-bps-per-8h", default="1,3,5,10")
    parser.add_argument("--json", action="store_true", help="Print trend_v3_compare_summary.json payload.")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def resolve_path(path_arg: str | Path) -> Path:
    """Resolve path relative to project root."""

    path = Path(path_arg)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def default_output_dir(split_scheme: str) -> Path:
    """Return the default compare output directory for a split scheme."""

    if split_scheme == "extended":
        return DEFAULT_EXTENDED_OUTPUT_DIR
    return DEFAULT_OUTPUT_DIR


def output_prefix(split_scheme: str) -> str:
    """Return output filename prefix for a split scheme."""

    return OUTPUT_PREFIX_BY_SCHEME.get(split_scheme, OUTPUT_PREFIX_BY_SCHEME["default"])


def parse_funding_bps(raw_value: str | list[float] | tuple[float, ...]) -> list[float]:
    """Parse synthetic funding stress bps values."""

    if isinstance(raw_value, (list, tuple)):
        values = [float(item) for item in raw_value]
    else:
        values = [float(token) for token in str(raw_value).replace(",", " ").split() if token.strip()]
    if not values:
        raise TrendFollowingV3CompareError("--funding-bps-per-8h 不能为空")
    return values


def finite_or_none(value: Any) -> float | None:
    """Return finite float or None."""

    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number):
        return None
    return number


def read_split_leaderboard(split: str, directory: Path) -> pd.DataFrame:
    """Read one split V3 leaderboard."""

    path = directory / "trend_v3_policy_leaderboard.csv"
    if not path.exists():
        raise TrendFollowingV3CompareError(f"{split} 缺少 trend_v3_policy_leaderboard.csv: {path}")
    try:
        frame = pd.read_csv(path)
    except Exception as exc:
        raise TrendFollowingV3CompareError(f"读取 {split} leaderboard 失败: {exc!r}") from exc
    if "policy_name" not in frame.columns:
        raise TrendFollowingV3CompareError(f"{split} leaderboard 缺少 policy_name 列")
    frame["split"] = split
    frame["source_dir"] = str(directory)
    return frame


def read_split_trades(split: str, directory: Path) -> pd.DataFrame:
    """Read one split V3 trade file for funding stress diagnostics."""

    path = directory / "trend_v3_trades.csv"
    if not path.exists():
        return pd.DataFrame()
    try:
        frame = pd.read_csv(path)
    except Exception as exc:
        raise TrendFollowingV3CompareError(f"读取 {split} trades 失败: {exc!r}") from exc
    if frame.empty:
        return frame
    frame["split"] = split
    return frame


def read_split_outputs(split_dirs: dict[str, Path]) -> pd.DataFrame:
    """Read all split leaderboards."""

    frames = [read_split_leaderboard(split, split_dirs[split]) for split in SPLITS]
    return pd.concat(frames, ignore_index=True)


def split_row(group_df: pd.DataFrame, split: str) -> pd.Series | None:
    """Return one split row from a policy group."""

    subset = group_df[group_df["split"] == split]
    if subset.empty:
        return None
    return subset.iloc[0]


def numeric_from_row(row: pd.Series | None, column: str) -> float | None:
    """Return numeric row value."""

    if row is None:
        return None
    return finite_or_none(row.get(column))


def positive(value: float | None) -> bool:
    """Return true when value is strictly positive."""

    return bool(value is not None and value > 0)


def nonnegative(value: float | None) -> bool:
    """Return true when value is non-negative."""

    return bool(value is not None and value >= 0)


def dataframe_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Convert DataFrame to JSON-safe records."""

    if df.empty:
        return []
    safe_df = df.replace({np.nan: None})
    return json.loads(safe_df.to_json(orient="records", force_ascii=False))


def estimate_trade_notional(trades_df: pd.DataFrame) -> pd.Series:
    """Estimate absolute position notional for synthetic funding stress."""

    if trades_df.empty:
        return pd.Series(dtype=float)
    if {"entry_price", "volume", "contract_size"}.issubset(trades_df.columns):
        notional = (
            pd.to_numeric(trades_df["entry_price"], errors="coerce").abs()
            * pd.to_numeric(trades_df["volume"], errors="coerce").abs()
            * pd.to_numeric(trades_df["contract_size"], errors="coerce").abs()
        )
    else:
        notional = pd.Series(np.nan, index=trades_df.index, dtype=float)
    if "turnover" in trades_df.columns:
        fallback = pd.to_numeric(trades_df["turnover"], errors="coerce").abs() / 2.0
        notional = notional.where((notional.notna()) & (notional > 0), fallback)
    return notional.fillna(0.0).astype(float)


def find_policy_row(frame: pd.DataFrame, policy_name: str) -> pd.Series | None:
    """Return one policy row from a leaderboard."""

    if frame.empty or "policy_name" not in frame.columns:
        return None
    subset = frame[frame["policy_name"] == policy_name]
    if subset.empty:
        return None
    return subset.iloc[0]


def select_oos_best_policy(oos_leaderboard: pd.DataFrame) -> str | None:
    """Select the OOS best policy by cost-aware net PnL."""

    if oos_leaderboard.empty or not {"policy_name", "net_pnl"}.issubset(oos_leaderboard.columns):
        return None
    working = oos_leaderboard.copy()
    working["net_pnl"] = pd.to_numeric(working["net_pnl"], errors="coerce")
    working = working.dropna(subset=["net_pnl"]).sort_values(["net_pnl", "policy_name"], ascending=[False, True], kind="stable")
    if working.empty:
        return None
    return str(working.iloc[0]["policy_name"])


def all_no_cost_positive_policies(compare_df: pd.DataFrame) -> list[str]:
    """Return policies with train/validation/OOS no-cost PnL all positive."""

    if compare_df.empty:
        return []
    mask = (
        (compare_df["train_no_cost_positive"] == True)  # noqa: E712
        & (compare_df["validation_no_cost_positive"] == True)  # noqa: E712
        & (compare_df["oos_no_cost_positive"] == True)  # noqa: E712
    )
    return compare_df.loc[mask, "policy_name"].astype(str).tolist()


def build_funding_stress(
    *,
    oos_trades: pd.DataFrame,
    oos_leaderboard: pd.DataFrame,
    policy_names: list[str],
    funding_bps_values: list[float],
    oos_label: str,
) -> pd.DataFrame:
    """Build synthetic funding stress for selected OOS policies."""

    rows: list[dict[str, Any]] = []
    if not policy_names:
        return pd.DataFrame(
            columns=[
                "split",
                "policy_name",
                "synthetic_funding_stress",
                "funding_bps_per_8h",
                "original_net_pnl",
                "funding_cost",
                "funding_adjusted_net_pnl",
                "funding_break_even_bps",
                "remains_positive_after_funding",
                "funding_exposure_notional_periods",
                "notice",
            ]
        )
    trades = oos_trades.copy()
    if not trades.empty:
        for column in ["net_pnl", "holding_minutes", "entry_price", "volume", "contract_size", "turnover"]:
            if column in trades.columns:
                trades[column] = pd.to_numeric(trades[column], errors="coerce")
        trades["estimated_notional"] = estimate_trade_notional(trades)
    for policy_name in dict.fromkeys(policy_names):
        group = trades[trades["policy_name"] == policy_name].copy() if not trades.empty and "policy_name" in trades.columns else pd.DataFrame()
        row = find_policy_row(oos_leaderboard, policy_name)
        original = numeric_from_row(row, "net_pnl")
        if original is None:
            original = float(pd.to_numeric(group.get("net_pnl"), errors="coerce").fillna(0.0).sum()) if not group.empty else 0.0
        if group.empty:
            funding_exposure = 0.0
        else:
            holding = pd.to_numeric(group.get("holding_minutes"), errors="coerce").fillna(0.0)
            notional = pd.to_numeric(group.get("estimated_notional"), errors="coerce").fillna(0.0)
            funding_exposure = float((notional.abs() * (holding / 480.0)).sum())
        break_even = original * 10000.0 / funding_exposure if funding_exposure > 0 else None
        for funding_bps in funding_bps_values:
            funding_cost = funding_exposure * float(funding_bps) / 10000.0
            adjusted = original - funding_cost
            rows.append(
                {
                    "split": oos_label,
                    "policy_name": policy_name,
                    "synthetic_funding_stress": True,
                    "funding_bps_per_8h": float(funding_bps),
                    "original_net_pnl": original,
                    "funding_cost": funding_cost,
                    "funding_adjusted_net_pnl": adjusted,
                    "funding_break_even_bps": break_even,
                    "remains_positive_after_funding": bool(adjusted > 0),
                    "funding_exposure_notional_periods": funding_exposure,
                    "notice": "Synthetic funding stress only; not actual OKX funding fee.",
                }
            )
    return pd.DataFrame(rows)


def funding_positive_for_policy(funding_df: pd.DataFrame, policy_name: str) -> bool:
    """Return whether one policy remains positive for every funding stress row."""

    if funding_df.empty:
        return False
    subset = funding_df[funding_df["policy_name"] == policy_name]
    if subset.empty or "remains_positive_after_funding" not in subset.columns:
        return False
    return bool((subset["remains_positive_after_funding"] == True).all())  # noqa: E712


def rejection_reasons(rows: dict[str, pd.Series | None]) -> list[str]:
    """Return stable-candidate rejection reasons."""

    reasons: list[str] = []
    for split in SPLITS:
        if rows.get(split) is None:
            reasons.append(f"missing_{split}_row")
    if reasons:
        return reasons

    if not positive(numeric_from_row(rows["train"], "no_cost_net_pnl")):
        reasons.append("train_no_cost_net_pnl_not_positive")
    if not positive(numeric_from_row(rows["validation"], "no_cost_net_pnl")):
        reasons.append("validation_no_cost_net_pnl_not_positive")
    if not positive(numeric_from_row(rows["oos"], "no_cost_net_pnl")):
        reasons.append("oos_no_cost_net_pnl_not_positive")
    if not nonnegative(numeric_from_row(rows["oos"], "net_pnl")):
        reasons.append("oos_cost_aware_net_pnl_negative")
    oos_dd = numeric_from_row(rows["oos"], "max_ddpercent")
    if oos_dd is None or oos_dd > MAX_OOS_DDPERCENT:
        reasons.append("oos_max_ddpercent_over_30")
    for split in SPLITS:
        trade_count = numeric_from_row(rows[split], "trade_count")
        if trade_count is None or trade_count < MIN_TRADE_COUNT:
            reasons.append(f"{split}_trade_count_under_10")
    largest_share = numeric_from_row(rows["oos"], "largest_symbol_pnl_share")
    if largest_share is None or largest_share > MAX_SYMBOL_PNL_SHARE:
        reasons.append("oos_largest_symbol_pnl_share_over_0p7")
    top_share = numeric_from_row(rows["oos"], "top_5pct_trade_pnl_contribution")
    if top_share is None or top_share > MAX_TOP_5PCT_CONTRIBUTION:
        reasons.append("oos_top_5pct_trade_pnl_contribution_over_0p8")
    active_symbol_count = numeric_from_row(rows["oos"], "active_symbol_count")
    if active_symbol_count is None or active_symbol_count < 2:
        reasons.append("oos_not_enough_active_symbols")
    return reasons


def build_compare_leaderboard(all_df: pd.DataFrame) -> pd.DataFrame:
    """Build one cross-split row per V3 policy."""

    rows: list[dict[str, Any]] = []
    for policy_name, group_df in all_df.groupby("policy_name", dropna=False):
        split_rows = {split: split_row(group_df, split) for split in SPLITS}
        row: dict[str, Any] = {"policy_name": policy_name}
        for split in SPLITS:
            split_data = split_rows[split]
            for column in [
                "symbol_count",
                "trade_count",
                "active_symbol_count",
                "no_cost_net_pnl",
                "net_pnl",
                "cost_drag",
                "max_drawdown",
                "max_ddpercent",
                "win_rate",
                "profit_factor",
                "largest_symbol_pnl_share",
                "top_5pct_trade_pnl_contribution",
                "max_concurrent_positions",
            ]:
                row[f"{split}_{column}"] = numeric_from_row(split_data, column)
        reasons = rejection_reasons(split_rows)
        row["train_no_cost_positive"] = positive(row.get("train_no_cost_net_pnl"))
        row["validation_no_cost_positive"] = positive(row.get("validation_no_cost_net_pnl"))
        row["oos_no_cost_positive"] = positive(row.get("oos_no_cost_net_pnl"))
        row["oos_cost_aware_nonnegative"] = nonnegative(row.get("oos_net_pnl"))
        row["oos_drawdown_ok"] = bool(row.get("oos_max_ddpercent") is not None and row["oos_max_ddpercent"] <= MAX_OOS_DDPERCENT)
        row["trade_count_sufficient"] = all(
            row.get(f"{split}_trade_count") is not None and row[f"{split}_trade_count"] >= MIN_TRADE_COUNT for split in SPLITS
        )
        row["symbol_concentration_ok"] = bool(
            row.get("oos_largest_symbol_pnl_share") is not None and row["oos_largest_symbol_pnl_share"] <= MAX_SYMBOL_PNL_SHARE
        )
        row["top_trade_concentration_ok"] = bool(
            row.get("oos_top_5pct_trade_pnl_contribution") is not None
            and row["oos_top_5pct_trade_pnl_contribution"] <= MAX_TOP_5PCT_CONTRIBUTION
        )
        row["oos_not_one_symbol_or_one_trade"] = bool(
            row.get("oos_active_symbol_count") is not None
            and row["oos_active_symbol_count"] >= 2
            and row["symbol_concentration_ok"]
            and row["top_trade_concentration_ok"]
        )
        row["stable_candidate"] = bool(not reasons)
        row["rejection_reasons"] = ";".join(reasons)
        rows.append(row)

    columns = [
        "policy_name",
        "train_no_cost_net_pnl",
        "validation_no_cost_net_pnl",
        "oos_no_cost_net_pnl",
        "train_net_pnl",
        "validation_net_pnl",
        "oos_net_pnl",
        "train_trade_count",
        "validation_trade_count",
        "oos_trade_count",
        "train_active_symbol_count",
        "validation_active_symbol_count",
        "oos_active_symbol_count",
        "oos_max_ddpercent",
        "oos_largest_symbol_pnl_share",
        "oos_top_5pct_trade_pnl_contribution",
        "oos_max_concurrent_positions",
        "train_no_cost_positive",
        "validation_no_cost_positive",
        "oos_no_cost_positive",
        "oos_cost_aware_nonnegative",
        "oos_drawdown_ok",
        "trade_count_sufficient",
        "symbol_concentration_ok",
        "top_trade_concentration_ok",
        "oos_not_one_symbol_or_one_trade",
        "stable_candidate",
        "rejection_reasons",
    ]
    compare_df = pd.DataFrame(rows)
    if compare_df.empty:
        return pd.DataFrame(columns=columns)
    return compare_df[columns].sort_values(
        ["stable_candidate", "oos_net_pnl", "oos_no_cost_net_pnl"],
        ascending=[False, False, False],
        kind="stable",
    ).reset_index(drop=True)


def add_split_label_aliases(compare_df: pd.DataFrame, split_labels: dict[str, str]) -> pd.DataFrame:
    """Add extended split-name aliases while preserving V3.0 column names."""

    result = compare_df.copy()
    for split, label in split_labels.items():
        if split == label:
            continue
        prefix = f"{split}_"
        alias_prefix = f"{label}_"
        for column in list(result.columns):
            if column.startswith(prefix):
                alias_column = alias_prefix + column[len(prefix):]
                if alias_column not in result.columns:
                    result[alias_column] = result[column]
    return result


def build_rejected_candidates(compare_df: pd.DataFrame) -> list[dict[str, Any]]:
    """Build compact rejected candidate records."""

    if compare_df.empty:
        return []
    rejected = compare_df[compare_df["stable_candidate"] == False].copy()  # noqa: E712
    return dataframe_records(
        rejected[
            [
                "policy_name",
                "train_no_cost_net_pnl",
                "validation_no_cost_net_pnl",
                "oos_no_cost_net_pnl",
                "oos_net_pnl",
                "oos_max_ddpercent",
                "oos_largest_symbol_pnl_share",
                "oos_top_5pct_trade_pnl_contribution",
                "rejection_reasons",
            ]
        ]
    )


def build_summary(
    split_dirs: dict[str, Path],
    output_dir: Path,
    compare_df: pd.DataFrame,
    compare_output_df: pd.DataFrame,
    funding_df: pd.DataFrame,
    *,
    split_scheme: str,
    split_labels: dict[str, str],
    funding_bps_values: list[float],
    funding_policy_names: list[str],
) -> dict[str, Any]:
    """Build trend_v3_compare_summary.json payload."""

    stable = compare_df[compare_df["stable_candidate"] == True].copy() if not compare_df.empty else pd.DataFrame()  # noqa: E712
    stable_output = compare_output_df[compare_output_df["stable_candidate"] == True].copy() if not compare_output_df.empty else pd.DataFrame()  # noqa: E712
    stable_names = stable["policy_name"].astype(str).tolist() if not stable.empty else []
    all_positive = all_no_cost_positive_policies(compare_df)
    oos_cost_positive = (
        compare_df.loc[pd.to_numeric(compare_df["oos_net_pnl"], errors="coerce") > 0, "policy_name"].astype(str).tolist()
        if not compare_df.empty
        else []
    )
    oos_cost_nonnegative = (
        compare_df.loc[compare_df["oos_cost_aware_nonnegative"] == True, "policy_name"].astype(str).tolist()  # noqa: E712
        if not compare_df.empty
        else []
    )
    high_symbol_concentration = (
        compare_df.loc[compare_df["symbol_concentration_ok"] == False, "policy_name"].astype(str).tolist()  # noqa: E712
        if not compare_df.empty
        else []
    )
    high_top_trade_concentration = (
        compare_df.loc[compare_df["top_trade_concentration_ok"] == False, "policy_name"].astype(str).tolist()  # noqa: E712
        if not compare_df.empty
        else []
    )
    oos_dependency_risk = bool(high_symbol_concentration or high_top_trade_concentration)
    funding_rows = dataframe_records(funding_df)
    funding_robust_stable_candidates = [
        policy_name for policy_name in stable_names if funding_positive_for_policy(funding_df, policy_name)
    ]
    funding_risk_controllable = bool(stable_names and funding_robust_stable_candidates)
    can_enter_v3_1_research = bool(not stable.empty and funding_risk_controllable)
    overfit_risk = bool(
        not compare_df.empty
        and (
            (compare_df["train_no_cost_positive"] == True)  # noqa: E712
            & (
                (compare_df["validation_no_cost_positive"] == False)  # noqa: E712
                | (compare_df["oos_no_cost_positive"] == False)  # noqa: E712
            )
        ).any()
    )
    return {
        "split_scheme": split_scheme,
        "split_labels": split_labels,
        "split_dirs": {split: str(path) for split, path in split_dirs.items()},
        "output_dir": str(output_dir),
        "output_files": [
            f"{output_prefix(split_scheme)}_summary.json",
            f"{output_prefix(split_scheme)}_leaderboard.csv",
            f"{output_prefix(split_scheme)}_report.md",
            f"{output_prefix(split_scheme)}_funding_stress.csv",
        ],
        "stable_candidate_rule": {
            f"{split_labels['train']}_no_cost_net_pnl": "> 0",
            f"{split_labels['validation']}_no_cost_net_pnl": "> 0",
            f"{split_labels['oos']}_no_cost_net_pnl": "> 0",
            f"{split_labels['oos']}_cost_aware_net_pnl": ">= 0",
            f"{split_labels['oos']}_max_ddpercent": f"<= {MAX_OOS_DDPERCENT}",
            "each_split_trade_count": f">= {MIN_TRADE_COUNT}",
            "largest_symbol_pnl_share": f"<= {MAX_SYMBOL_PNL_SHARE}",
            "top_5pct_trade_pnl_contribution": f"<= {MAX_TOP_5PCT_CONTRIBUTION}",
            "oos_not_one_symbol_or_one_trade": True,
        },
        "policy_count": int(compare_df["policy_name"].nunique()) if not compare_df.empty else 0,
        "stable_candidate_exists": bool(not stable.empty),
        "stable_candidates": dataframe_records(stable_output),
        "all_no_cost_positive_policies": all_positive,
        "oos_cost_aware_positive_policies": oos_cost_positive,
        "oos_cost_aware_nonnegative_policies": oos_cost_nonnegative,
        "rejected_candidates_with_reasons": build_rejected_candidates(compare_df),
        "high_largest_symbol_pnl_share_policies": high_symbol_concentration,
        "high_top_5pct_trade_pnl_contribution_policies": high_top_trade_concentration,
        "oos_single_symbol_or_tail_trade_dependency_risk": oos_dependency_risk,
        "funding_sensitivity_required": True,
        "funding_stress_notice": "Synthetic funding stress only; not actual OKX funding fee.",
        "funding_bps_per_8h": funding_bps_values,
        "funding_stress_policy_names": funding_policy_names,
        "funding_stress": funding_rows,
        "funding_risk_controllable": funding_risk_controllable,
        "funding_robust_stable_candidates": funding_robust_stable_candidates,
        "overfit_risk": overfit_risk,
        "trend_following_v3_failed": bool(stable.empty),
        "trend_following_v3_extended_failed": bool(stable.empty) if split_scheme == "extended" else None,
        "can_enter_v3_1_research": can_enter_v3_1_research if split_scheme == "extended" else bool(not stable.empty),
        "can_enter_strategy_v3_prototype": False if split_scheme == "extended" else bool(not stable.empty),
        "demo_live_allowed": False,
    }


def format_number(value: Any, digits: int = 6) -> str:
    """Format numeric values."""

    number = finite_or_none(value)
    if number is None:
        return "N/A"
    return f"{number:.{digits}f}"


def format_candidate_rows(rows: list[dict[str, Any]], limit: int = 12) -> str:
    """Format candidate rows for Markdown."""

    if not rows:
        return "- 无"
    lines = []
    for row in rows[:limit]:
        lines.append(
            f"- {row.get('policy_name')}: train_no_cost={format_number(row.get('train_no_cost_net_pnl'), 4)}, "
            f"validation_no_cost={format_number(row.get('validation_no_cost_net_pnl'), 4)}, "
            f"oos_no_cost={format_number(row.get('oos_no_cost_net_pnl'), 4)}, "
            f"oos_net={format_number(row.get('oos_net_pnl'), 4)}, "
            f"oos_dd%={format_number(row.get('oos_max_ddpercent'), 2)}"
        )
    return "\n".join(lines)


def format_rejected_rows(rows: list[dict[str, Any]], limit: int = 20) -> str:
    """Format rejected rows for Markdown."""

    if not rows:
        return "- 无"
    lines = []
    for row in rows[:limit]:
        lines.append(f"- {row.get('policy_name')}: {row.get('rejection_reasons')}")
    return "\n".join(lines)


def markdown_table(rows: list[dict[str, Any]], columns: list[str], digits: int = 4, limit: int = 20) -> str:
    """Render a compact Markdown table."""

    if not rows:
        return "- 无"
    lines = [
        "| " + " | ".join(columns) + " |",
        "|" + "|".join("---" for _ in columns) + "|",
    ]
    for row in rows[:limit]:
        values = []
        for column in columns:
            value = row.get(column)
            if isinstance(value, (bool, np.bool_)):
                values.append(str(bool(value)).lower())
            elif isinstance(value, (int, float, np.integer, np.floating)):
                values.append(format_number(value, digits))
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def render_markdown(summary: dict[str, Any], compare_df: pd.DataFrame) -> str:
    """Render trend_v3_compare_report.md."""

    stable = summary.get("stable_candidates") or []
    rejected = summary.get("rejected_candidates_with_reasons") or []
    split_labels = summary.get("split_labels") or {"train": "train", "validation": "validation", "oos": "oos"}
    funding_rows = summary.get("funding_stress") or []
    title = "Trend Following V3 Extended 跨样本比较" if summary.get("split_scheme") == "extended" else "Trend Following V3 跨样本比较"
    return (
        f"# {title}\n\n"
        "## 核心结论\n"
        f"- split_scheme={summary.get('split_scheme')}, split_labels={split_labels}\n"
        f"- stable_candidate_exists={str(bool(summary.get('stable_candidate_exists'))).lower()}\n"
        f"- stable_candidates={[row.get('policy_name') for row in stable]}\n"
        f"- all_no_cost_positive_policies={summary.get('all_no_cost_positive_policies')}\n"
        f"- oos_cost_aware_positive_policies={summary.get('oos_cost_aware_positive_policies')}\n"
        f"- oos_cost_aware_nonnegative_policies={summary.get('oos_cost_aware_nonnegative_policies')}\n"
        f"- overfit_risk={str(bool(summary.get('overfit_risk'))).lower()}\n"
        f"- trend_following_v3_failed={str(bool(summary.get('trend_following_v3_failed'))).lower()}\n"
        f"- trend_following_v3_extended_failed={summary.get('trend_following_v3_extended_failed')}\n"
        f"- can_enter_v3_1_research={str(bool(summary.get('can_enter_v3_1_research'))).lower()}\n"
        f"- can_enter_strategy_v3_prototype={str(bool(summary.get('can_enter_strategy_v3_prototype'))).lower()}\n"
        f"- demo_live_allowed={str(bool(summary.get('demo_live_allowed'))).lower()}\n"
        f"- stable_candidate 需要 {split_labels['train']}/{split_labels['validation']}/{split_labels['oos']} no-cost 均为正、OOS 成本后不亏、OOS 回撤不超过 30%、三段交易次数均 >=10、OOS 不依赖单一 symbol 或极少数交易。\n"
        "- Funding 部分是 synthetic funding stress，不是真实 OKX funding fee。\n\n"
        "## 稳定候选\n"
        f"{format_candidate_rows(stable)}\n\n"
        "## 被拒候选与原因\n"
        f"{format_rejected_rows(rejected)}\n\n"
        "## 风险集中度\n"
        f"- high_largest_symbol_pnl_share_policies={summary.get('high_largest_symbol_pnl_share_policies')}\n"
        f"- high_top_5pct_trade_pnl_contribution_policies={summary.get('high_top_5pct_trade_pnl_contribution_policies')}\n"
        f"- oos_single_symbol_or_tail_trade_dependency_risk={str(bool(summary.get('oos_single_symbol_or_tail_trade_dependency_risk'))).lower()}\n\n"
        "## Funding Stress\n"
        f"{markdown_table(funding_rows, ['policy_name', 'funding_bps_per_8h', 'original_net_pnl', 'funding_adjusted_net_pnl', 'remains_positive_after_funding'])}\n\n"
        "## 输出文件\n"
        f"- {output_prefix(str(summary.get('split_scheme') or 'default'))}_summary.json\n"
        f"- {output_prefix(str(summary.get('split_scheme') or 'default'))}_leaderboard.csv\n"
        f"- {output_prefix(str(summary.get('split_scheme') or 'default'))}_report.md\n"
    )


def write_json(path: Path, payload: Any) -> None:
    """Write JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(payload), ensure_ascii=False, indent=2), encoding="utf-8")


def write_dataframe(path: Path, df: pd.DataFrame) -> None:
    """Write CSV."""

    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")


def run_compare(
    train_dir: Path,
    validation_dir: Path,
    oos_dir: Path,
    output_dir: Path,
    *,
    split_scheme: str = "default",
    funding_bps_values: list[float] | None = None,
) -> dict[str, Any]:
    """Run V3 cross-sample comparison."""

    if split_scheme not in SPLIT_LABELS_BY_SCHEME:
        raise TrendFollowingV3CompareError(f"unknown split scheme: {split_scheme}")
    split_labels = SPLIT_LABELS_BY_SCHEME[split_scheme]
    funding_bps = funding_bps_values or DEFAULT_FUNDING_BPS
    split_dirs = {"train": train_dir, "validation": validation_dir, "oos": oos_dir}
    all_df = read_split_outputs(split_dirs)
    compare_df = build_compare_leaderboard(all_df)
    compare_output_df = add_split_label_aliases(compare_df, split_labels)
    oos_leaderboard = read_split_leaderboard("oos", oos_dir)
    oos_trades = read_split_trades("oos", oos_dir)
    oos_best = select_oos_best_policy(oos_leaderboard)
    stable_like = all_no_cost_positive_policies(compare_df)
    stable_names = compare_df.loc[compare_df["stable_candidate"] == True, "policy_name"].astype(str).tolist() if not compare_df.empty else []  # noqa: E712
    funding_policy_names = list(dict.fromkeys(stable_names + stable_like + ([oos_best] if oos_best else [])))
    funding_df = build_funding_stress(
        oos_trades=oos_trades,
        oos_leaderboard=oos_leaderboard,
        policy_names=funding_policy_names,
        funding_bps_values=funding_bps,
        oos_label=split_labels["oos"],
    )
    summary = build_summary(
        split_dirs,
        output_dir,
        compare_df,
        compare_output_df,
        funding_df,
        split_scheme=split_scheme,
        split_labels=split_labels,
        funding_bps_values=funding_bps,
        funding_policy_names=funding_policy_names,
    )
    markdown = render_markdown(summary, compare_output_df)
    prefix = output_prefix(split_scheme)

    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / f"{prefix}_summary.json", summary)
    write_dataframe(output_dir / f"{prefix}_leaderboard.csv", compare_output_df)
    (output_dir / f"{prefix}_report.md").write_text(markdown, encoding="utf-8")
    write_dataframe(output_dir / f"{prefix}_funding_stress.csv", funding_df)
    return summary


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""

    args = parse_args(argv)
    ensure_headless_runtime()
    logger = setup_logging("compare_trend_following_v3", verbose=args.verbose)
    try:
        funding_bps = parse_funding_bps(args.funding_bps_per_8h)
        train_dir = resolve_path(args.train_dir)
        validation_dir = resolve_path(args.validation_dir)
        oos_dir = resolve_path(args.oos_dir)
        output_dir = resolve_path(args.output_dir or default_output_dir(args.split_scheme))
        summary = run_compare(
            train_dir,
            validation_dir,
            oos_dir,
            output_dir,
            split_scheme=args.split_scheme,
            funding_bps_values=funding_bps,
        )
        print_json_block(
            "Trend Following V3 comparison summary:",
            {
                "output_dir": output_dir,
                "split_scheme": args.split_scheme,
                "stable_candidate_exists": summary.get("stable_candidate_exists"),
                "stable_candidates": [row.get("policy_name") for row in summary.get("stable_candidates", [])],
                "trend_following_v3_failed": summary.get("trend_following_v3_failed"),
                "can_enter_v3_1_research": summary.get("can_enter_v3_1_research"),
                "can_enter_strategy_v3_prototype": summary.get("can_enter_strategy_v3_prototype"),
            },
        )
        if args.json:
            print(json.dumps(to_jsonable(summary), ensure_ascii=False, indent=2))
        return 0
    except TrendFollowingV3CompareError as exc:
        log_event(logger, logging.ERROR, "trend_following_v3_compare.error", str(exc))
        return 2
    except Exception:
        logger.exception(
            "Unexpected error during Trend Following V3 comparison",
            extra={"event": "trend_following_v3_compare.unexpected_error"},
        )
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
