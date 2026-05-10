#!/usr/bin/env python3
"""Analyze Trend V3 extended trades with actual OKX funding history."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from common_runtime import PROJECT_ROOT, ensure_headless_runtime, log_event, print_json_block, setup_logging, to_jsonable
from download_okx_funding_history import DEFAULT_INST_IDS, DEFAULT_OUTPUT_DIR as DEFAULT_FUNDING_DIR, parse_inst_ids
from history_time_utils import DEFAULT_TIMEZONE, resolve_timezone


DEFAULT_TREND_V3_EXTENDED_DIR = PROJECT_ROOT / "reports" / "research" / "trend_following_v3_extended"
DEFAULT_COMPARE_DIR = PROJECT_ROOT / "reports" / "research" / "trend_following_v3_extended_compare"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "reports" / "research" / "trend_following_v3_actual_funding"
DEFAULT_VERIFY_SUMMARY_PATH = PROJECT_ROOT / "reports" / "research" / "funding" / "okx_funding_verify_summary.json"
SPLITS = ["train_ext", "validation_ext", "oos_ext"]
TARGET_POLICY = "v3_1d_ema_50_200_atr5"

TRADE_OUTPUT_COLUMNS = [
    "split",
    "policy_name",
    "symbol",
    "inst_id",
    "direction",
    "entry_time",
    "exit_time",
    "holding_minutes",
    "original_net_pnl",
    "no_cost_pnl",
    "funding_events_count",
    "conservative_funding_cost",
    "signed_funding_pnl",
    "funding_adjusted_net_pnl_conservative",
    "funding_adjusted_net_pnl_signed",
    "notional",
    "notional_source",
    "funding_data_available",
    "warnings",
]


class TrendFundingAnalysisError(Exception):
    """Raised when actual funding analysis cannot continue."""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description="Analyze Trend V3 extended trades with OKX actual funding history.")
    parser.add_argument("--funding-dir", default=str(DEFAULT_FUNDING_DIR))
    parser.add_argument("--trend-v3-extended-dir", default=str(DEFAULT_TREND_V3_EXTENDED_DIR))
    parser.add_argument("--compare-dir", default=str(DEFAULT_COMPARE_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--timezone", default=DEFAULT_TIMEZONE)
    parser.add_argument("--mode", choices=("conservative", "signed"), default="conservative")
    parser.add_argument("--inst-ids", default=",".join(DEFAULT_INST_IDS))
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def resolve_path(path_arg: str | Path) -> Path:
    """Resolve a path relative to project root."""

    path = Path(path_arg)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def finite_float(value: Any, default: float = 0.0) -> float:
    """Return finite float or default."""

    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if not np.isfinite(number):
        return default
    return number


def dataframe_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Convert DataFrame to JSON-safe records."""

    if df.empty:
        return []
    safe_df = df.replace({np.nan: None})
    return json.loads(safe_df.to_json(orient="records", force_ascii=False))


def symbol_to_inst_id(symbol: str) -> str:
    """Map V3 research symbols to OKX instrument ids."""

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


def parse_trade_time(value: Any, timezone_name: str) -> pd.Timestamp:
    """Parse trade timestamp and convert to UTC."""

    tz = resolve_timezone(timezone_name)
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize(tz)
    return timestamp.tz_convert("UTC")


def load_trade_splits(trend_v3_extended_dir: Path) -> pd.DataFrame:
    """Load Trend V3 extended trade CSVs."""

    frames: list[pd.DataFrame] = []
    missing: list[str] = []
    for split in SPLITS:
        path = trend_v3_extended_dir / split / "trend_v3_trades.csv"
        if not path.exists():
            missing.append(str(path))
            continue
        frame = pd.read_csv(path)
        frame["split"] = split
        frames.append(frame)
    if not frames:
        raise TrendFundingAnalysisError(f"missing Trend V3 extended trade files: {missing}")
    return pd.concat(frames, ignore_index=True)


def load_funding_csv(path: Path) -> pd.DataFrame:
    """Load one funding CSV with normalized UTC timestamps."""

    frame = pd.read_csv(path)
    if frame.empty:
        return pd.DataFrame(columns=["funding_time_utc", "funding_rate"])
    if "funding_time_utc" in frame.columns:
        frame["funding_time_utc"] = pd.to_datetime(frame["funding_time_utc"], utc=True, errors="coerce")
    elif "funding_time" in frame.columns:
        frame["funding_time_utc"] = pd.to_datetime(pd.to_numeric(frame["funding_time"], errors="coerce"), unit="ms", utc=True)
    else:
        raise TrendFundingAnalysisError(f"funding CSV missing funding_time columns: {path}")
    frame["funding_rate"] = pd.to_numeric(frame.get("funding_rate"), errors="coerce")
    frame = frame.dropna(subset=["funding_time_utc", "funding_rate"]).copy()
    frame = frame.sort_values("funding_time_utc", kind="stable").drop_duplicates("funding_time_utc", keep="last")
    return frame.reset_index(drop=True)


def load_funding_histories(funding_dir: Path, inst_ids: list[str]) -> tuple[dict[str, pd.DataFrame], list[str], list[str]]:
    """Load funding histories for requested instruments."""

    histories: dict[str, pd.DataFrame] = {}
    missing: list[str] = []
    warnings: list[str] = []
    for inst_id in inst_ids:
        matches = sorted(funding_dir.glob(f"{inst_id}_funding_*.csv"))
        if not matches:
            missing.append(inst_id)
            warnings.append(f"{inst_id}: missing funding CSV")
            continue
        path = matches[-1]
        try:
            histories[inst_id] = load_funding_csv(path)
        except Exception as exc:
            missing.append(inst_id)
            warnings.append(f"{inst_id}: failed to read funding CSV: {exc}")
    return histories, missing, warnings


def estimate_trade_notional(trade: pd.Series) -> tuple[float, str, list[str]]:
    """Estimate absolute position notional for a trade."""

    warnings = ["mark_price_unavailable_entry_price_used_for_funding_notional"]
    entry_price = finite_float(trade.get("entry_price"), default=np.nan)
    volume = abs(finite_float(trade.get("volume"), default=np.nan))
    contract_size = abs(finite_float(trade.get("contract_size"), default=np.nan))
    if np.isfinite(entry_price) and np.isfinite(volume) and np.isfinite(contract_size) and volume > 0 and contract_size > 0:
        return abs(entry_price * volume * contract_size), "entry_price_x_volume_x_contract_size", warnings
    turnover = abs(finite_float(trade.get("turnover"), default=np.nan))
    if np.isfinite(turnover) and turnover > 0:
        warnings.append("notional_fallback_turnover_div_2")
        return turnover / 2.0, "turnover_div_2", warnings
    warnings.append("notional_unavailable_assumed_zero")
    return 0.0, "unavailable", warnings


def signed_funding_pnl(notional: float, funding_rate: float, direction: str) -> float:
    """Return funding PnL using OKX sign convention research assumption."""

    normalized = str(direction or "").strip().lower()
    if normalized == "short":
        return notional * funding_rate
    return -notional * funding_rate


def analyze_trade_row(
    trade: pd.Series,
    funding_histories: dict[str, pd.DataFrame],
    missing_inst_ids: set[str],
    timezone_name: str,
) -> dict[str, Any]:
    """Analyze funding impact for one trade row."""

    warnings: list[str] = []
    split = str(trade.get("split") or "")
    policy_name = str(trade.get("policy_name") or "")
    symbol = str(trade.get("symbol") or "")
    inst_id = symbol_to_inst_id(symbol)
    direction = str(trade.get("direction") or "")
    entry_time = str(trade.get("entry_time") or "")
    exit_time = str(trade.get("exit_time") or "")
    entry_utc = parse_trade_time(entry_time, timezone_name)
    exit_utc = parse_trade_time(exit_time, timezone_name)
    if exit_utc < entry_utc:
        warnings.append("exit_before_entry")
        entry_utc, exit_utc = exit_utc, entry_utc

    notional, notional_source, notional_warnings = estimate_trade_notional(trade)
    warnings.extend(notional_warnings)
    original_net_pnl = finite_float(trade.get("net_pnl"))
    no_cost_pnl = finite_float(trade.get("no_cost_pnl"), default=finite_float(trade.get("no_cost_net_pnl")))

    funding_df = funding_histories.get(inst_id)
    funding_data_available = funding_df is not None and not funding_df.empty
    if inst_id in missing_inst_ids or not funding_data_available:
        warnings.append("missing_funding_csv_or_empty_history")
        events = pd.DataFrame(columns=["funding_time_utc", "funding_rate"])
    else:
        first_funding_time = funding_df["funding_time_utc"].min()
        last_funding_time = funding_df["funding_time_utc"].max()
        if entry_utc < first_funding_time or exit_utc > last_funding_time:
            warnings.append(
                "funding_history_does_not_cover_trade_interval:"
                f"first={first_funding_time.isoformat()},last={last_funding_time.isoformat()}"
            )
        mask = (funding_df["funding_time_utc"] >= entry_utc) & (funding_df["funding_time_utc"] <= exit_utc)
        events = funding_df.loc[mask].copy()

    rates = pd.to_numeric(events.get("funding_rate"), errors="coerce").dropna() if not events.empty else pd.Series(dtype=float)
    conservative_cost = float((abs(notional) * rates.abs()).sum()) if not rates.empty else 0.0
    signed_pnl = float(sum(signed_funding_pnl(abs(notional), float(rate), direction) for rate in rates)) if not rates.empty else 0.0
    return {
        "split": split,
        "policy_name": policy_name,
        "symbol": symbol,
        "inst_id": inst_id,
        "direction": direction,
        "entry_time": entry_time,
        "exit_time": exit_time,
        "holding_minutes": finite_float(trade.get("holding_minutes")),
        "original_net_pnl": original_net_pnl,
        "no_cost_pnl": no_cost_pnl,
        "funding_events_count": int(len(rates)),
        "conservative_funding_cost": conservative_cost,
        "signed_funding_pnl": signed_pnl,
        "funding_adjusted_net_pnl_conservative": original_net_pnl - conservative_cost,
        "funding_adjusted_net_pnl_signed": original_net_pnl + signed_pnl,
        "notional": abs(notional),
        "notional_source": notional_source,
        "funding_data_available": bool(funding_data_available),
        "warnings": ";".join(dict.fromkeys(warnings)),
    }


def summarize_group(adjustments: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    """Summarize funding adjustments by group."""

    if adjustments.empty:
        return pd.DataFrame()
    grouped = adjustments.groupby(group_columns, dropna=False)
    summary = grouped.agg(
        trade_count=("policy_name", "size"),
        original_net_pnl=("original_net_pnl", "sum"),
        no_cost_pnl=("no_cost_pnl", "sum"),
        funding_events_count=("funding_events_count", "sum"),
        conservative_funding_cost=("conservative_funding_cost", "sum"),
        signed_funding_pnl=("signed_funding_pnl", "sum"),
        funding_adjusted_net_pnl_conservative=("funding_adjusted_net_pnl_conservative", "sum"),
        funding_adjusted_net_pnl_signed=("funding_adjusted_net_pnl_signed", "sum"),
        notional=("notional", "sum"),
    )
    return summary.reset_index()


def read_json_if_exists(path: Path) -> dict[str, Any]:
    """Read optional JSON."""

    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def select_oos_best_policy(trend_v3_extended_dir: Path) -> str | None:
    """Select OOS best policy by original cost-aware net PnL."""

    path = trend_v3_extended_dir / "oos_ext" / "trend_v3_policy_leaderboard.csv"
    if not path.exists():
        return None
    frame = pd.read_csv(path)
    if frame.empty or not {"policy_name", "net_pnl"}.issubset(frame.columns):
        return None
    frame["net_pnl"] = pd.to_numeric(frame["net_pnl"], errors="coerce")
    frame = frame.dropna(subset=["net_pnl"]).sort_values(["net_pnl", "policy_name"], ascending=[False, True], kind="stable")
    if frame.empty:
        return None
    return str(frame.iloc[0]["policy_name"])


def policies_positive_all_splits(policy_summary: pd.DataFrame, column: str) -> list[str]:
    """Return policies positive in train_ext/validation_ext/oos_ext for one metric."""

    if policy_summary.empty or column not in policy_summary.columns:
        return []
    positive: list[str] = []
    for policy_name, group in policy_summary.groupby("policy_name", dropna=False):
        split_values = {str(row["split"]): finite_float(row[column], default=np.nan) for _, row in group.iterrows()}
        if all(split in split_values and np.isfinite(split_values[split]) and split_values[split] > 0 for split in SPLITS):
            positive.append(str(policy_name))
    return sorted(positive)


def row_for_policy_split(policy_summary: pd.DataFrame, policy_name: str, split: str) -> dict[str, Any] | None:
    """Return compact summary row for one policy/split."""

    if policy_summary.empty:
        return None
    subset = policy_summary[(policy_summary["policy_name"] == policy_name) & (policy_summary["split"] == split)]
    if subset.empty:
        return None
    return dataframe_records(subset)[0]


def build_summary(
    *,
    adjustments: pd.DataFrame,
    policy_summary: pd.DataFrame,
    split_summary: pd.DataFrame,
    symbol_summary: pd.DataFrame,
    funding_missing_inst_ids: list[str],
    funding_warnings: list[str],
    verify_summary: dict[str, Any],
    compare_summary: dict[str, Any],
    trend_v3_extended_dir: Path,
    funding_dir: Path,
    output_dir: Path,
    mode: str,
) -> dict[str, Any]:
    """Build JSON summary and decision payload."""

    conservative_positive = policies_positive_all_splits(policy_summary, "funding_adjusted_net_pnl_conservative")
    signed_positive = policies_positive_all_splits(policy_summary, "funding_adjusted_net_pnl_signed")
    both_positive = sorted(set(conservative_positive).intersection(signed_positive))
    compare_stable = {
        str(row.get("policy_name"))
        for row in compare_summary.get("stable_candidates", [])
        if row.get("policy_name") is not None
    }
    funding_adjusted_stable = sorted(set(both_positive).intersection(compare_stable))
    funding_data_complete = bool(verify_summary.get("funding_data_complete")) if verify_summary else False
    symbols_with_gaps = verify_summary.get("symbols_with_large_gaps") or []
    available_data_only = not funding_data_complete
    funding_adjusted_stable_available_data_only = funding_adjusted_stable
    if not funding_data_complete:
        funding_adjusted_stable = []
    oos_best_policy = select_oos_best_policy(trend_v3_extended_dir)
    oos_best_row = row_for_policy_split(policy_summary, oos_best_policy, "oos_ext") if oos_best_policy else None
    target_rows = [row for split in SPLITS if (row := row_for_policy_split(policy_summary, TARGET_POLICY, split)) is not None]
    target_conservative_positive = bool(
        target_rows
        and len(target_rows) == len(SPLITS)
        and all(finite_float(row.get("funding_adjusted_net_pnl_conservative"), default=np.nan) > 0 for row in target_rows)
    )
    target_signed_positive = bool(
        target_rows
        and len(target_rows) == len(SPLITS)
        and all(finite_float(row.get("funding_adjusted_net_pnl_signed"), default=np.nan) > 0 for row in target_rows)
    )

    can_enter_research_only = bool(funding_data_complete and funding_adjusted_stable)
    target_conservative_final = target_conservative_positive if funding_data_complete else None
    target_signed_final = target_signed_positive if funding_data_complete else None
    weak_clue_fail = (not target_conservative_positive or not target_signed_positive) if funding_data_complete else None
    split_records = dataframe_records(split_summary)
    zero_event_splits = [
        str(row.get("split"))
        for row in split_records
        if finite_float(row.get("funding_events_count"), default=0.0) == 0.0
    ]
    funding_event_coverage_warning = (
        "likely_due_to_missing_funding_coverage"
        if available_data_only and zero_event_splits
        else None
    )
    return {
        "mode": mode,
        "funding_dir": str(funding_dir),
        "trend_v3_extended_dir": str(trend_v3_extended_dir),
        "output_dir": str(output_dir),
        "output_files": [
            "actual_funding_trade_adjustments.csv",
            "actual_funding_policy_summary.csv",
            "actual_funding_split_summary.csv",
            "actual_funding_symbol_summary.csv",
            "actual_funding_report.md",
            "actual_funding_summary.json",
        ],
        "alignment_rule": "inclusive trade holding interval: entry_time <= funding_time <= exit_time",
        "notional_rule": "entry_price * abs(volume) * abs(contract_size); fallback turnover/2; mark price is not fabricated",
        "signed_mode_assumption": (
            "OKX fundingRate sign research convention: positive rate means long pays short; "
            "negative rate means short pays long. This is a research assumption and should be rechecked before any production use."
        ),
        "funding_data_complete": funding_data_complete,
        "available_data_only": available_data_only,
        "funding_missing_inst_ids": funding_missing_inst_ids,
        "funding_warnings": funding_warnings,
        "symbols_with_funding_gaps": symbols_with_gaps,
        "verify_summary_missing": not bool(verify_summary),
        "verify_incomplete_reason": verify_summary.get("incomplete_reason") if verify_summary else [],
        "missing_before_first_available": bool(verify_summary.get("missing_before_first_available")) if verify_summary else False,
        "funding_event_coverage_warning": funding_event_coverage_warning,
        "zero_funding_event_splits_when_incomplete": zero_event_splits,
        "verify_summary_path": str(DEFAULT_VERIFY_SUMMARY_PATH),
        "trade_count": int(len(adjustments)),
        "policy_count": int(policy_summary["policy_name"].nunique()) if not policy_summary.empty else 0,
        "oos_best_policy": oos_best_policy,
        "oos_best_policy_after_funding": oos_best_row,
        "target_policy": TARGET_POLICY,
        "target_policy_rows": target_rows,
        "target_policy_conservative_all_splits_positive": target_conservative_final,
        "target_policy_signed_all_splits_positive": target_signed_final,
        "target_policy_conservative_all_splits_positive_available_data_only": target_conservative_positive,
        "target_policy_signed_all_splits_positive_available_data_only": target_signed_positive,
        "funding_adjusted_all_split_positive_policies_conservative": conservative_positive,
        "funding_adjusted_all_split_positive_policies_signed": signed_positive,
        "funding_adjusted_all_split_positive_policies_both": both_positive,
        "compare_stable_candidates": sorted(compare_stable),
        "funding_adjusted_stable_candidates_available_data_only": funding_adjusted_stable_available_data_only,
        "funding_adjusted_stable_candidates": funding_adjusted_stable,
        "funding_adjusted_stable_candidate_exists": bool(funding_adjusted_stable),
        "funding_makes_only_weak_clue_fail": weak_clue_fail,
        "funding_makes_only_weak_clue_fail_available_data_only": bool(not target_conservative_positive or not target_signed_positive),
        "can_enter_funding_aware_v3_1_research": can_enter_research_only,
        "strategy_development_allowed": False,
        "demo_live_allowed": False,
        "policy_summary": dataframe_records(policy_summary),
        "split_summary": split_records,
        "symbol_summary": dataframe_records(symbol_summary),
    }


def format_number(value: Any, digits: int = 6) -> str:
    """Format optional numeric values."""

    try:
        number = float(value)
    except (TypeError, ValueError):
        return "N/A"
    if not np.isfinite(number):
        return "N/A"
    return f"{number:.{digits}f}"


def markdown_table(rows: list[dict[str, Any]], columns: list[str], limit: int = 30) -> str:
    """Render a Markdown table."""

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
            if isinstance(value, bool):
                values.append(str(value).lower())
            elif isinstance(value, (int, float)):
                values.append(format_number(value, 6))
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def conclusion_text(value: Any, *, available_value: Any | None = None) -> str:
    """Render a conclusion, preserving incomplete-data uncertainty."""

    if value is None:
        suffix = ""
        if available_value is not None:
            suffix = f" (available_data_result={str(bool(available_value)).lower()})"
        return f"unknown_due_to_incomplete_funding_data{suffix}"
    return str(bool(value)).lower()


def render_report(summary: dict[str, Any]) -> str:
    """Render required Markdown actual funding report."""

    oos_best = summary.get("oos_best_policy_after_funding") or {}
    oos_best_conservative = finite_float(oos_best.get("funding_adjusted_net_pnl_conservative"), default=np.nan)
    oos_best_signed = finite_float(oos_best.get("funding_adjusted_net_pnl_signed"), default=np.nan)
    oos_best_available_positive = bool(oos_best and oos_best_conservative > 0 and oos_best_signed > 0)
    target_rows = summary.get("target_policy_rows") or []
    policy_rows = summary.get("policy_summary") or []
    top_policy_rows = [
        row
        for row in policy_rows
        if row.get("policy_name") in {summary.get("target_policy"), summary.get("oos_best_policy")}
    ]
    funding_complete = bool(summary.get("funding_data_complete"))
    symbols_with_gaps = summary.get("symbols_with_funding_gaps") or []
    weak_clue_failed = summary.get("funding_makes_only_weak_clue_fail")
    return (
        "# Trend V3 Actual Funding Analysis\n\n"
        "## Core Conclusion\n"
        f"- available_data_only={str(bool(summary.get('available_data_only'))).lower()}\n"
        f"- funding_data_complete={str(funding_complete).lower()}\n"
        f"- verify_summary_missing={str(bool(summary.get('verify_summary_missing'))).lower()}\n"
        f"- verify_incomplete_reason={summary.get('verify_incomplete_reason')}\n"
        f"- missing_before_first_available={str(bool(summary.get('missing_before_first_available'))).lower()}\n"
        f"- funding_event_coverage_warning={summary.get('funding_event_coverage_warning') or ''}\n"
        "- decision_rule=available_data_only results are not decision-grade and cannot unlock Strategy V3, V3.1, demo, or live.\n\n"
        "## Scope\n"
        "- This is funding-aware trend-following research only; it is not Strategy V3 development.\n"
        f"- mode={summary.get('mode')}\n"
        f"- alignment_rule={summary.get('alignment_rule')}\n"
        f"- notional_rule={summary.get('notional_rule')}\n"
        f"- signed_mode_assumption={summary.get('signed_mode_assumption')}\n"
        "- warning=mark price is unavailable in the current V3 trade files, so funding notional uses entry price approximation.\n\n"
        "## Required Answers\n"
        f"1. Funding 数据是否完整？{str(funding_complete).lower()}\n"
        f"2. 哪些 symbol funding 数据有缺口？{symbols_with_gaps or summary.get('funding_missing_inst_ids') or []}\n"
        "3. V3 extended OOS best policy 在 actual funding 后是否仍为正？"
        f"conservative={format_number(oos_best_conservative)}, "
        f"signed={format_number(oos_best_signed)}, "
        f"positive={conclusion_text(oos_best_available_positive if funding_complete else None, available_value=oos_best_available_positive)}\n"
        f"4. {TARGET_POLICY} 在 conservative mode 后是否仍为正？"
        f"{conclusion_text(summary.get('target_policy_conservative_all_splits_positive'), available_value=summary.get('target_policy_conservative_all_splits_positive_available_data_only'))}\n"
        f"5. {TARGET_POLICY} 在 signed mode 后是否仍为正？"
        f"{conclusion_text(summary.get('target_policy_signed_all_splits_positive'), available_value=summary.get('target_policy_signed_all_splits_positive_available_data_only'))}\n"
        "6. 是否有任何 policy 在 train_ext / validation_ext / oos_ext funding-adjusted 后全部为正？"
        f"conservative={summary.get('funding_adjusted_all_split_positive_policies_conservative')}, "
        f"signed={summary.get('funding_adjusted_all_split_positive_policies_signed')}\n"
        "7. Funding 是否会使当前唯一弱线索彻底失效？"
        f"{conclusion_text(weak_clue_failed, available_value=summary.get('funding_makes_only_weak_clue_fail_available_data_only'))}\n"
        "8. 是否允许进入 funding-aware V3.1 research？"
        f"{str(bool(summary.get('can_enter_funding_aware_v3_1_research'))).lower()}\n"
        "9. 是否仍禁止 Strategy V3 / demo / live？true\n\n"
        "## Target And OOS Best Policy Rows\n"
        f"{markdown_table(top_policy_rows, ['policy_name', 'split', 'original_net_pnl', 'funding_adjusted_net_pnl_conservative', 'funding_adjusted_net_pnl_signed', 'funding_events_count'])}\n\n"
        "## Funding Coverage Warning\n"
        f"- zero_funding_event_splits_when_incomplete={summary.get('zero_funding_event_splits_when_incomplete') or []}\n"
        "- funding_events_count=0 under incomplete funding coverage is interpreted as likely_due_to_missing_funding_coverage, not as evidence that no funding occurred.\n\n"
        "## Target Policy Across Splits\n"
        f"{markdown_table(target_rows, ['policy_name', 'split', 'original_net_pnl', 'funding_adjusted_net_pnl_conservative', 'funding_adjusted_net_pnl_signed', 'funding_events_count'])}\n\n"
        "## Final Gates\n"
        f"- funding_adjusted_stable_candidate_exists={str(bool(summary.get('funding_adjusted_stable_candidate_exists'))).lower()}\n"
        f"- can_enter_funding_aware_v3_1_research={str(bool(summary.get('can_enter_funding_aware_v3_1_research'))).lower()}\n"
        "- strategy_development_allowed=false\n"
        "- demo_live_allowed=false\n"
    )


def write_outputs(
    output_dir: Path,
    adjustments: pd.DataFrame,
    policy_summary: pd.DataFrame,
    split_summary: pd.DataFrame,
    symbol_summary: pd.DataFrame,
    summary: dict[str, Any],
) -> None:
    """Write CSV/JSON/Markdown outputs."""

    output_dir.mkdir(parents=True, exist_ok=True)
    adjustments.to_csv(output_dir / "actual_funding_trade_adjustments.csv", index=False, encoding="utf-8")
    policy_summary.to_csv(output_dir / "actual_funding_policy_summary.csv", index=False, encoding="utf-8")
    split_summary.to_csv(output_dir / "actual_funding_split_summary.csv", index=False, encoding="utf-8")
    symbol_summary.to_csv(output_dir / "actual_funding_symbol_summary.csv", index=False, encoding="utf-8")
    (output_dir / "actual_funding_summary.json").write_text(
        json.dumps(to_jsonable(summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "actual_funding_report.md").write_text(render_report(summary), encoding="utf-8")


def run_analysis(
    *,
    funding_dir: Path,
    trend_v3_extended_dir: Path,
    compare_dir: Path,
    output_dir: Path,
    timezone_name: str,
    mode: str,
    inst_ids: list[str],
) -> dict[str, Any]:
    """Run actual funding analysis."""

    trades = load_trade_splits(trend_v3_extended_dir)
    trade_inst_ids = sorted({symbol_to_inst_id(symbol) for symbol in trades.get("symbol", pd.Series(dtype=str)).astype(str)})
    requested_inst_ids = list(dict.fromkeys(inst_ids + trade_inst_ids))
    funding_histories, missing_inst_ids, funding_warnings = load_funding_histories(funding_dir, requested_inst_ids)
    adjustments = pd.DataFrame(
        [
            analyze_trade_row(row, funding_histories, set(missing_inst_ids), timezone_name)
            for _, row in trades.iterrows()
        ],
        columns=TRADE_OUTPUT_COLUMNS,
    )
    policy_summary = summarize_group(adjustments, ["policy_name", "split"])
    split_summary = summarize_group(adjustments, ["split"])
    symbol_summary = summarize_group(adjustments, ["symbol", "inst_id", "split"])
    verify_summary = read_json_if_exists(DEFAULT_VERIFY_SUMMARY_PATH)
    compare_summary = read_json_if_exists(compare_dir / "trend_v3_extended_compare_summary.json")
    summary = build_summary(
        adjustments=adjustments,
        policy_summary=policy_summary,
        split_summary=split_summary,
        symbol_summary=symbol_summary,
        funding_missing_inst_ids=missing_inst_ids,
        funding_warnings=funding_warnings,
        verify_summary=verify_summary,
        compare_summary=compare_summary,
        trend_v3_extended_dir=trend_v3_extended_dir,
        funding_dir=funding_dir,
        output_dir=output_dir,
        mode=mode,
    )
    write_outputs(output_dir, adjustments, policy_summary, split_summary, symbol_summary, summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""

    args = parse_args(argv)
    ensure_headless_runtime()
    logger = setup_logging("analyze_trend_v3_actual_funding", verbose=args.verbose)
    try:
        summary = run_analysis(
            funding_dir=resolve_path(args.funding_dir),
            trend_v3_extended_dir=resolve_path(args.trend_v3_extended_dir),
            compare_dir=resolve_path(args.compare_dir),
            output_dir=resolve_path(args.output_dir),
            timezone_name=args.timezone,
            mode=args.mode,
            inst_ids=parse_inst_ids(args.inst_ids),
        )
        print_json_block(
            "Trend V3 actual funding summary:",
            {
                "funding_data_complete": summary.get("funding_data_complete"),
                "target_policy_conservative_all_splits_positive": summary.get("target_policy_conservative_all_splits_positive"),
                "target_policy_signed_all_splits_positive": summary.get("target_policy_signed_all_splits_positive"),
                "funding_adjusted_stable_candidate_exists": summary.get("funding_adjusted_stable_candidate_exists"),
                "can_enter_funding_aware_v3_1_research": summary.get("can_enter_funding_aware_v3_1_research"),
                "output_dir": summary.get("output_dir"),
            },
        )
        return 0
    except TrendFundingAnalysisError as exc:
        log_event(logger, logging.ERROR, "trend_v3_actual_funding.error", str(exc))
        return 2
    except Exception:
        logger.exception("Unexpected Trend V3 actual funding failure", extra={"event": "trend_v3_actual_funding.unexpected"})
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
