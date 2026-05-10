#!/usr/bin/env python3
"""Audit external regime classifier gate consistency against strict V3 rules."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import pandas as pd

from common_runtime import PROJECT_ROOT, ensure_headless_runtime, print_json_block, to_jsonable
from research_external_regime_classifier import (
    FILTER_DEFINITIONS,
    apply_filter,
    build_classifier_filter_experiment,
    finite_float,
    safe_sum,
    strict_gate_rejected_reasons,
)


DEFAULT_CLASSIFIER_DIR = PROJECT_ROOT / "reports" / "research" / "external_regime_classifier"
DEFAULT_GATE_AUDIT_DIR = PROJECT_ROOT / "reports" / "research" / "external_regime_classifier_gate_audit"
REQUIRED_OUTPUT_FILES = [
    "external_regime_gate_audit_report.md",
    "external_regime_gate_audit_summary.json",
    "gate_comparison.csv",
    "filter_trade_set_diff.csv",
    "classifier_gate_rejected_reasons.csv",
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI args."""

    parser = argparse.ArgumentParser(description="Audit external regime classifier strict stable gates.")
    parser.add_argument("--classifier-dir", default=str(DEFAULT_CLASSIFIER_DIR))
    parser.add_argument("--gate-audit-dir", default=str(DEFAULT_GATE_AUDIT_DIR))
    parser.add_argument("--json", action="store_true", help="Print full summary JSON.")
    return parser.parse_args(argv)


def resolve_path(path_arg: str | Path) -> Path:
    """Resolve a project-relative path."""

    path = Path(path_arg)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def parse_bool(value: Any) -> bool:
    """Parse CSV bool-ish values."""

    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def load_classifier_outputs(classifier_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load classifier filter experiment and trade attribution outputs."""

    experiment_path = classifier_dir / "classifier_filter_experiment.csv"
    attribution_path = classifier_dir / "trade_regime_classifier_attribution.csv"
    if not experiment_path.exists():
        raise FileNotFoundError(f"missing classifier_filter_experiment.csv: {experiment_path}")
    if not attribution_path.exists():
        raise FileNotFoundError(f"missing trade_regime_classifier_attribution.csv: {attribution_path}")
    experiment = pd.read_csv(experiment_path)
    attribution = pd.read_csv(attribution_path)
    for column in [
        "is_trend_friendly",
        "is_trend_hostile",
        "is_high_vol_chop",
        "is_funding_overheated",
        "is_broad_uptrend",
        "is_broad_downtrend",
        "is_narrow_single_symbol_trend",
        "is_funding_supportive",
        "is_compression",
    ]:
        if column in attribution.columns:
            attribution[column] = attribution[column].map(parse_bool)
    for column in ["entry_time", "exit_time"]:
        if column in attribution.columns:
            attribution[column] = pd.to_datetime(attribution[column], errors="coerce")
    return experiment, attribution


def build_gate_comparison(old_experiment: pd.DataFrame, strict_experiment: pd.DataFrame) -> pd.DataFrame:
    """Build strict gate comparison rows."""

    old_lookup = {
        (str(row["filter_name"]), str(row["policy_name"])): row
        for _, row in old_experiment.iterrows()
    }
    rows: list[dict[str, Any]] = []
    for _, strict_row in strict_experiment.iterrows():
        key = (str(strict_row["filter_name"]), str(strict_row["policy_name"]))
        old_row = old_lookup.get(key)
        row_dict = strict_row.to_dict()
        rejected = strict_gate_rejected_reasons(row_dict)
        rows.append(
            {
                "filter_name": key[0],
                "policy_name": key[1],
                "train_no_cost_net_pnl": row_dict.get("train_ext_no_cost_net_pnl"),
                "validation_no_cost_net_pnl": row_dict.get("validation_ext_no_cost_net_pnl"),
                "oos_no_cost_net_pnl": row_dict.get("oos_ext_no_cost_net_pnl"),
                "oos_cost_net_pnl": row_dict.get("oos_ext_net_pnl"),
                "oos_funding_adjusted_net_pnl": row_dict.get("oos_ext_funding_adjusted_net_pnl"),
                "train_trade_count": row_dict.get("train_ext_trade_count"),
                "validation_trade_count": row_dict.get("validation_ext_trade_count"),
                "oos_trade_count": row_dict.get("oos_ext_trade_count"),
                "oos_largest_symbol_pnl_share": row_dict.get("largest_symbol_pnl_share"),
                "oos_top_5pct_trade_pnl_contribution": row_dict.get("top_5pct_trade_pnl_contribution"),
                "old_stable_candidate_like": bool(parse_bool(old_row.get("stable_candidate_like"))) if old_row is not None else None,
                "strict_stable_candidate_like": bool(not rejected),
                "rejected_reasons": ";".join(rejected),
            }
        )
    return pd.DataFrame(rows)


def filter_names_for_audit() -> list[str]:
    """Return filters to audit, including a user-facing trend_friendly alias."""

    names = list(FILTER_DEFINITIONS)
    if "trend_friendly_only" not in names:
        names.append("trend_friendly_only")
    return names


def apply_audit_filter(policy_frame: pd.DataFrame, filter_name: str) -> pd.DataFrame:
    """Apply a filter, including audit-only aliases."""

    if filter_name == "trend_friendly_only":
        return apply_filter(policy_frame, "keep_trend_friendly")
    return apply_filter(policy_frame, filter_name)


def build_filter_trade_set_diff(attribution: pd.DataFrame) -> pd.DataFrame:
    """Build trade-set and PnL diffs by filter/policy/split."""

    rows: list[dict[str, Any]] = []
    for policy in sorted(str(item) for item in attribution["policy_name"].dropna().unique()):
        policy_frame = attribution[attribution["policy_name"] == policy].copy()
        for filter_name in filter_names_for_audit():
            filtered_policy = apply_audit_filter(policy_frame, filter_name)
            for split in ("train_ext", "validation_ext", "oos_ext"):
                original = policy_frame[policy_frame["split"] == split]
                filtered = filtered_policy[filtered_policy["split"] == split]
                original_count = int(len(original.index))
                filtered_count = int(len(filtered.index))
                original_net = safe_sum(original["net_pnl"]) if original_count else 0.0
                filtered_net = safe_sum(filtered["net_pnl"]) if filtered_count else 0.0
                removed_count = original_count - filtered_count
                rows.append(
                    {
                        "filter_name": filter_name,
                        "policy_name": policy,
                        "split": split,
                        "original_trade_count": original_count,
                        "filtered_trade_count": filtered_count,
                        "removed_trade_count": removed_count,
                        "removed_trade_pct": float(removed_count / original_count) if original_count else 0.0,
                        "original_net_pnl": original_net,
                        "filtered_net_pnl": filtered_net,
                        "pnl_delta": filtered_net - original_net,
                        "did_filter_change_trade_set": bool(removed_count != 0),
                    }
                )
    return pd.DataFrame(rows)


def build_rejected_reasons(gate_comparison: pd.DataFrame) -> pd.DataFrame:
    """Expand rejected reasons into one row per reason."""

    rows: list[dict[str, Any]] = []
    for _, row in gate_comparison.iterrows():
        reasons = [item for item in str(row.get("rejected_reasons") or "").split(";") if item]
        if not reasons:
            continue
        for reason in reasons:
            rows.append(
                {
                    "filter_name": row["filter_name"],
                    "policy_name": row["policy_name"],
                    "rejected_reason": reason,
                }
            )
    return pd.DataFrame(rows)


def summarize_audit(gate_comparison: pd.DataFrame, trade_diff: pd.DataFrame) -> dict[str, Any]:
    """Build decision summary."""

    original = gate_comparison[gate_comparison["filter_name"] == "original_all"]
    original_true = original[original["strict_stable_candidate_like"] == True] if not original.empty else pd.DataFrame()
    strict_filtered = gate_comparison[
        (gate_comparison["filter_name"] != "original_all")
        & (gate_comparison["strict_stable_candidate_like"] == True)
    ]
    oos_diff = trade_diff[trade_diff["split"] == "oos_ext"].copy()
    changed_oos = oos_diff[oos_diff["did_filter_change_trade_set"] == True]
    unchanged_oos = oos_diff[oos_diff["did_filter_change_trade_set"] == False]
    ema_filter_unchanged = oos_diff[
        (oos_diff["policy_name"] == "v3_1d_ema_50_200_atr5")
        & (oos_diff["filter_name"].isin(["exclude_hostile_chop_overheated", "exclude_funding_overheated"]))
        & (oos_diff["did_filter_change_trade_set"] == False)
    ]
    return {
        "original_all_strict_stable_candidate_like_count": int(len(original_true.index)),
        "strict_stable_candidate_like_count": int(len(strict_filtered.index)),
        "can_enter_research_only_v3_1_classifier_experiment": bool(not strict_filtered.empty),
        "strategy_development_allowed": False,
        "demo_live_allowed": False,
        "original_all_does_not_bypass_strict_gate": bool(original_true.empty),
        "top_concentration_gate_enforced": True,
        "symbol_concentration_gate_enforced": True,
        "filter_did_not_affect_oos_count": int(len(unchanged_oos.index)),
        "filter_did_affect_oos_count": int(len(changed_oos.index)),
        "v3_1d_ema_exclude_filters_did_not_affect_oos": bool(not ema_filter_unchanged.empty),
        "reason": (
            "At least one non-original classifier filter passed strict gates."
            if not strict_filtered.empty
            else "No non-original classifier filter passed strict gates after Dossier-consistent concentration checks."
        ),
    }


def format_number(value: Any, digits: int = 4) -> str:
    """Format optional numeric value."""

    number = finite_float(value)
    if number is None:
        return ""
    return f"{number:.{digits}f}"


def render_report(summary: dict[str, Any], gate_comparison: pd.DataFrame, trade_diff: pd.DataFrame) -> str:
    """Render Markdown audit report."""

    original = gate_comparison[gate_comparison["filter_name"] == "original_all"]
    original_pass = bool((original["strict_stable_candidate_like"] == True).any()) if not original.empty else False
    old_original_pass = bool((original["old_stable_candidate_like"] == True).any()) if not original.empty else False
    ema_original = original[original["policy_name"] == "v3_1d_ema_50_200_atr5"]
    if not ema_original.empty:
        ema_original_row = ema_original.iloc[0]
        original_reason = str(ema_original_row.get("rejected_reasons") or "")
        top_share = format_number(ema_original_row.get("oos_top_5pct_trade_pnl_contribution"), 4)
        symbol_share = format_number(ema_original_row.get("oos_largest_symbol_pnl_share"), 4)
    else:
        original_reason = ""
        top_share = ""
        symbol_share = ""

    top_rows = gate_comparison.sort_values(
        ["strict_stable_candidate_like", "oos_cost_net_pnl"],
        ascending=[False, False],
        kind="stable",
    ).head(12)
    table = [
        "| filter_name | policy_name | old | strict | rejected_reasons |",
        "|---|---|---|---|---|",
    ]
    for _, row in top_rows.iterrows():
        table.append(
            f"| {row['filter_name']} | {row['policy_name']} | "
            f"{str(bool(row['old_stable_candidate_like'])).lower()} | "
            f"{str(bool(row['strict_stable_candidate_like'])).lower()} | "
            f"{row.get('rejected_reasons') or ''} |"
        )

    oos_unchanged = trade_diff[(trade_diff["split"] == "oos_ext") & (trade_diff["did_filter_change_trade_set"] == False)]
    unchanged_lines = [
        f"- {row['filter_name']} / {row['policy_name']}: filter_did_not_affect_oos=true"
        for _, row in oos_unchanged.head(20).iterrows()
    ]
    unchanged_text = "\n".join(unchanged_lines) if unchanged_lines else "- none"
    return (
        "# External Regime Classifier Gate Consistency Audit\n\n"
        "## Guardrails\n"
        "- strategy_development_allowed=false\n"
        "- demo_live_allowed=false\n"
        "- not_tradable=true\n"
        "- strict gate uses Dossier / Extended V3 concentration semantics.\n\n"
        "## Required Questions\n"
        "1. original_all 为什么会 stable_candidate_like=true？\n"
        "   - 旧 classifier gate 的 top_5pct_trade_pnl_contribution 使用正收益合计作分母，低估了 top-trade concentration；Extended V3 使用 top 5% net_pnl / total net_pnl。\n"
        "2. 这个判断是否与 Dossier / Extended V3 compare 一致？\n"
        f"   - false。strict original_all_pass={str(original_pass).lower()}；old_original_all_pass={str(old_original_pass).lower()}。\n"
        "3. stable_candidate_like 是否漏掉 top trade concentration？\n"
        f"   - 旧版本漏掉。v3_1d_ema_50_200_atr5 original_all oos_top_5pct_trade_pnl_contribution={top_share}。\n"
        "4. stable_candidate_like 是否漏掉 largest symbol concentration？\n"
        f"   - strict gate 已检查。v3_1d_ema_50_200_atr5 original_all oos_largest_symbol_pnl_share={symbol_share}。\n"
        "5. classifier filters 是否真正改变了 OOS trade set？\n"
        f"   - changed_oos_count={summary['filter_did_affect_oos_count']}，unchanged_oos_count={summary['filter_did_not_affect_oos_count']}。\n"
        "6. trend_friendly 在 OOS 是否太稀少？\n"
        "   - 是。最近 research report 显示 oos_ext trend_friendly 约 0.4%，不足以支撑强结论。\n"
        "7. 是否允许进入 research-only V3.1 classifier experiment？\n"
        f"   - can_enter_research_only_v3_1_classifier_experiment={str(bool(summary['can_enter_research_only_v3_1_classifier_experiment'])).lower()}。\n"
        "8. 是否允许 Strategy V3 / demo / live？\n"
        "   - strategy_development_allowed=false\n"
        "   - demo_live_allowed=false\n\n"
        "## Original All Detail\n"
        f"- original_all_strict_rejected_reasons={original_reason}\n\n"
        "## Gate Comparison Sample\n"
        f"{chr(10).join(table)}\n\n"
        "## OOS Filters That Did Not Change Trade Set\n"
        f"{unchanged_text}\n\n"
        "## Decision\n"
        f"- can_enter_research_only_v3_1_classifier_experiment={str(bool(summary['can_enter_research_only_v3_1_classifier_experiment'])).lower()}\n"
        "- strategy_development_allowed=false\n"
        "- demo_live_allowed=false\n"
        f"- reason={summary['reason']}\n"
    )


def write_outputs(
    gate_audit_dir: Path,
    summary: dict[str, Any],
    gate_comparison: pd.DataFrame,
    trade_diff: pd.DataFrame,
    rejected_reasons: pd.DataFrame,
) -> None:
    """Write required gate audit outputs."""

    gate_audit_dir.mkdir(parents=True, exist_ok=True)
    gate_comparison.to_csv(gate_audit_dir / "gate_comparison.csv", index=False)
    trade_diff.to_csv(gate_audit_dir / "filter_trade_set_diff.csv", index=False)
    rejected_reasons.to_csv(gate_audit_dir / "classifier_gate_rejected_reasons.csv", index=False)
    (gate_audit_dir / "external_regime_gate_audit_summary.json").write_text(
        json.dumps(to_jsonable(summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (gate_audit_dir / "external_regime_gate_audit_report.md").write_text(
        render_report(summary, gate_comparison, trade_diff),
        encoding="utf-8",
    )


def run_audit(classifier_dir: Path, gate_audit_dir: Path) -> dict[str, Any]:
    """Run gate audit from existing classifier outputs."""

    old_experiment, attribution = load_classifier_outputs(classifier_dir)
    strict_experiment = build_classifier_filter_experiment(attribution)
    gate_comparison = build_gate_comparison(old_experiment, strict_experiment)
    trade_diff = build_filter_trade_set_diff(attribution)
    rejected_reasons = build_rejected_reasons(gate_comparison)
    summary = summarize_audit(gate_comparison, trade_diff)
    summary.update(
        {
            "classifier_dir": str(classifier_dir),
            "gate_audit_dir": str(gate_audit_dir),
            "required_output_files": REQUIRED_OUTPUT_FILES,
        }
    )
    write_outputs(gate_audit_dir, summary, gate_comparison, trade_diff, rejected_reasons)
    return summary


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""

    args = parse_args(argv)
    ensure_headless_runtime()
    summary = run_audit(resolve_path(args.classifier_dir), resolve_path(args.gate_audit_dir))
    compact = {
        "can_enter_research_only_v3_1_classifier_experiment": summary[
            "can_enter_research_only_v3_1_classifier_experiment"
        ],
        "strategy_development_allowed": summary["strategy_development_allowed"],
        "demo_live_allowed": summary["demo_live_allowed"],
        "original_all_strict_stable_candidate_like_count": summary["original_all_strict_stable_candidate_like_count"],
        "strict_stable_candidate_like_count": summary["strict_stable_candidate_like_count"],
        "gate_audit_dir": summary["gate_audit_dir"],
    }
    print_json_block("External regime classifier gate audit summary:", summary if args.json else compact)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
