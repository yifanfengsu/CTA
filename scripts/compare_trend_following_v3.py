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
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Output directory. Default: reports/research/trend_following_v3_compare.",
    )
    parser.add_argument("--json", action="store_true", help="Print trend_v3_compare_summary.json payload.")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def resolve_path(path_arg: str | Path) -> Path:
    """Resolve path relative to project root."""

    path = Path(path_arg)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


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


def build_summary(split_dirs: dict[str, Path], output_dir: Path, compare_df: pd.DataFrame) -> dict[str, Any]:
    """Build trend_v3_compare_summary.json payload."""

    stable = compare_df[compare_df["stable_candidate"] == True].copy() if not compare_df.empty else pd.DataFrame()  # noqa: E712
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
        "split_dirs": {split: str(path) for split, path in split_dirs.items()},
        "output_dir": str(output_dir),
        "stable_candidate_rule": {
            "train_no_cost_net_pnl": "> 0",
            "validation_no_cost_net_pnl": "> 0",
            "oos_no_cost_net_pnl": "> 0",
            "oos_cost_aware_net_pnl": ">= 0",
            "oos_max_ddpercent": f"<= {MAX_OOS_DDPERCENT}",
            "each_split_trade_count": f">= {MIN_TRADE_COUNT}",
            "largest_symbol_pnl_share": f"<= {MAX_SYMBOL_PNL_SHARE}",
            "top_5pct_trade_pnl_contribution": f"<= {MAX_TOP_5PCT_CONTRIBUTION}",
            "oos_not_one_symbol_or_one_trade": True,
        },
        "policy_count": int(compare_df["policy_name"].nunique()) if not compare_df.empty else 0,
        "stable_candidate_exists": bool(not stable.empty),
        "stable_candidates": dataframe_records(stable),
        "rejected_candidates_with_reasons": build_rejected_candidates(compare_df),
        "overfit_risk": overfit_risk,
        "trend_following_v3_failed": bool(stable.empty),
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


def render_markdown(summary: dict[str, Any], compare_df: pd.DataFrame) -> str:
    """Render trend_v3_compare_report.md."""

    stable = summary.get("stable_candidates") or []
    rejected = summary.get("rejected_candidates_with_reasons") or []
    return (
        "# Trend Following V3 跨样本比较\n\n"
        "## 核心结论\n"
        f"- stable_candidate_exists={str(bool(summary.get('stable_candidate_exists'))).lower()}\n"
        f"- stable_candidates={[row.get('policy_name') for row in stable]}\n"
        f"- overfit_risk={str(bool(summary.get('overfit_risk'))).lower()}\n"
        f"- trend_following_v3_failed={str(bool(summary.get('trend_following_v3_failed'))).lower()}\n"
        "- stable_candidate 需要 train/validation/oos no-cost 均为正、OOS 成本后不亏、OOS 回撤不超过 30%、三段交易次数均 >=10、OOS 不依赖单一 symbol 或极少数交易。\n\n"
        "## 稳定候选\n"
        f"{format_candidate_rows(stable)}\n\n"
        "## 被拒候选与原因\n"
        f"{format_rejected_rows(rejected)}\n\n"
        "## 输出文件\n"
        "- trend_v3_compare_summary.json\n"
        "- trend_v3_compare_leaderboard.csv\n"
        "- trend_v3_compare_report.md\n"
    )


def write_json(path: Path, payload: Any) -> None:
    """Write JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(payload), ensure_ascii=False, indent=2), encoding="utf-8")


def write_dataframe(path: Path, df: pd.DataFrame) -> None:
    """Write CSV."""

    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")


def run_compare(train_dir: Path, validation_dir: Path, oos_dir: Path, output_dir: Path) -> dict[str, Any]:
    """Run V3 cross-sample comparison."""

    split_dirs = {"train": train_dir, "validation": validation_dir, "oos": oos_dir}
    all_df = read_split_outputs(split_dirs)
    compare_df = build_compare_leaderboard(all_df)
    summary = build_summary(split_dirs, output_dir, compare_df)
    markdown = render_markdown(summary, compare_df)

    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "trend_v3_compare_summary.json", summary)
    write_dataframe(output_dir / "trend_v3_compare_leaderboard.csv", compare_df)
    (output_dir / "trend_v3_compare_report.md").write_text(markdown, encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""

    args = parse_args(argv)
    ensure_headless_runtime()
    logger = setup_logging("compare_trend_following_v3", verbose=args.verbose)
    try:
        train_dir = resolve_path(args.train_dir)
        validation_dir = resolve_path(args.validation_dir)
        oos_dir = resolve_path(args.oos_dir)
        output_dir = resolve_path(args.output_dir)
        summary = run_compare(train_dir, validation_dir, oos_dir, output_dir)
        print_json_block(
            "Trend Following V3 comparison summary:",
            {
                "output_dir": output_dir,
                "stable_candidate_exists": summary.get("stable_candidate_exists"),
                "stable_candidates": [row.get("policy_name") for row in summary.get("stable_candidates", [])],
                "trend_following_v3_failed": summary.get("trend_following_v3_failed"),
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
