#!/usr/bin/env python3
"""Compare Trend Following V2 research across train/validation/OOS splits."""

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
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "reports" / "research" / "trend_following_v2_compare"
MIN_TRADE_COUNT = 10
MAX_DDPERCENT_THRESHOLD = 30.0
LOW_FREQUENCY_TRADE_COUNT = 20
MAX_EXPLAINABLE_OOS_COST_LOSS_TO_NO_COST = 0.5


class TrendFollowingV2CompareError(Exception):
    """Raised when Trend Following V2 comparison cannot continue."""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description="Compare Trend Following V2 research across sample splits.")
    parser.add_argument("--train-dir", required=True, help="Train trend_following_v2 directory.")
    parser.add_argument("--validation-dir", required=True, help="Validation trend_following_v2 directory.")
    parser.add_argument("--oos-dir", required=True, help="OOS trend_following_v2 directory.")
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Output directory. Default: reports/research/trend_following_v2_compare.",
    )
    parser.add_argument("--json", action="store_true", help="Print trend_compare_summary.json payload to stdout.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose structured logs.")
    return parser.parse_args(argv)


def resolve_path(path_arg: str | Path) -> Path:
    """Resolve a path relative to the project root."""

    path = Path(path_arg)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def finite_or_none(value: Any) -> float | None:
    """Return a finite float or None."""

    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number):
        return None
    return number


def read_split_leaderboard(split: str, directory: Path) -> pd.DataFrame:
    """Read one split trend_policy_leaderboard.csv."""

    path = directory / "trend_policy_leaderboard.csv"
    if not path.exists():
        raise TrendFollowingV2CompareError(f"{split} 缺少 trend_policy_leaderboard.csv: {path}")
    try:
        frame = pd.read_csv(path)
    except Exception as exc:
        raise TrendFollowingV2CompareError(f"读取 {split} leaderboard 失败: {exc!r}") from exc
    if "policy_name" not in frame.columns:
        raise TrendFollowingV2CompareError(f"{split} leaderboard 缺少 policy_name 列")
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
    """Return one numeric value from a row."""

    if row is None:
        return None
    return finite_or_none(row.get(column))


def positive(value: float | None) -> bool:
    """Return whether a numeric value is strictly positive."""

    return bool(value is not None and value > 0)


def nonnegative(value: float | None) -> bool:
    """Return whether a numeric value is non-negative."""

    return bool(value is not None and value >= 0)


def oos_cost_drag_explainable(oos_row: pd.Series | None) -> bool:
    """Return whether OOS cost drag is explainable by low trade frequency."""

    if oos_row is None:
        return False
    trade_count = numeric_from_row(oos_row, "trade_count")
    no_cost = numeric_from_row(oos_row, "no_cost_net_pnl")
    net = numeric_from_row(oos_row, "net_pnl")
    cost_drag = numeric_from_row(oos_row, "cost_drag")
    if trade_count is None or no_cost is None or net is None or cost_drag is None:
        return False
    return bool(
        trade_count <= LOW_FREQUENCY_TRADE_COUNT
        and no_cost > 0
        and net < 0
        and cost_drag > 0
        and abs(net) <= MAX_EXPLAINABLE_OOS_COST_LOSS_TO_NO_COST * no_cost
    )


def max_drawdown_controlled(rows: dict[str, pd.Series | None]) -> bool:
    """Return whether drawdown percent is below the comparison threshold in every split."""

    for split in SPLITS:
        value = numeric_from_row(rows.get(split), "max_ddpercent")
        if value is None or value > MAX_DDPERCENT_THRESHOLD:
            return False
    return True


def trade_count_sufficient(rows: dict[str, pd.Series | None]) -> bool:
    """Return whether every split has enough trades."""

    for split in SPLITS:
        value = numeric_from_row(rows.get(split), "trade_count")
        if value is None or value < MIN_TRADE_COUNT:
            return False
    return True


def build_compare_leaderboard(all_df: pd.DataFrame) -> pd.DataFrame:
    """Build one cross-split row per concrete policy run."""

    rows: list[dict[str, Any]] = []
    for policy_name, group_df in all_df.groupby("policy_name", dropna=False):
        split_rows = {split: split_row(group_df, split) for split in SPLITS}
        base_name = None
        atr_mult = None
        timeframe = None
        for split in SPLITS:
            row_data = split_rows[split]
            if row_data is not None:
                base_name = row_data.get("base_policy_name")
                atr_mult = finite_or_none(row_data.get("atr_mult"))
                timeframe = row_data.get("timeframe")
                break

        row: dict[str, Any] = {
            "policy_name": policy_name,
            "base_policy_name": base_name,
            "atr_mult": atr_mult,
            "timeframe": timeframe,
        }
        for split in SPLITS:
            split_data = split_rows[split]
            for column in [
                "trade_count",
                "net_pnl",
                "no_cost_net_pnl",
                "cost_drag",
                "max_drawdown",
                "max_ddpercent",
                "win_rate",
                "profit_factor",
                "top_5pct_trade_pnl_contribution",
            ]:
                row[f"{split}_{column}"] = numeric_from_row(split_data, column)

        no_cost_positive_all = all(positive(row.get(f"{split}_no_cost_net_pnl")) for split in SPLITS)
        oos_cost_ok = nonnegative(row.get("oos_net_pnl")) or oos_cost_drag_explainable(split_rows["oos"])
        drawdown_ok = max_drawdown_controlled(split_rows)
        trade_count_ok = trade_count_sufficient(split_rows)
        stable_candidate = bool(no_cost_positive_all and oos_cost_ok and drawdown_ok and trade_count_ok)
        row["train_no_cost_positive"] = positive(row.get("train_no_cost_net_pnl"))
        row["validation_no_cost_positive"] = positive(row.get("validation_no_cost_net_pnl"))
        row["oos_no_cost_positive"] = positive(row.get("oos_no_cost_net_pnl"))
        row["oos_cost_aware_nonnegative"] = nonnegative(row.get("oos_net_pnl"))
        row["oos_cost_drag_explainable"] = oos_cost_drag_explainable(split_rows["oos"])
        row["drawdown_controlled"] = drawdown_ok
        row["trade_count_sufficient"] = trade_count_ok
        row["stable_candidate"] = stable_candidate
        row["strategy_v2_candidate"] = stable_candidate
        rows.append(row)

    columns = [
        "policy_name",
        "base_policy_name",
        "atr_mult",
        "timeframe",
        "train_trade_count",
        "validation_trade_count",
        "oos_trade_count",
        "train_no_cost_net_pnl",
        "validation_no_cost_net_pnl",
        "oos_no_cost_net_pnl",
        "train_net_pnl",
        "validation_net_pnl",
        "oos_net_pnl",
        "train_cost_drag",
        "validation_cost_drag",
        "oos_cost_drag",
        "train_max_ddpercent",
        "validation_max_ddpercent",
        "oos_max_ddpercent",
        "train_top_5pct_trade_pnl_contribution",
        "validation_top_5pct_trade_pnl_contribution",
        "oos_top_5pct_trade_pnl_contribution",
        "train_no_cost_positive",
        "validation_no_cost_positive",
        "oos_no_cost_positive",
        "oos_cost_aware_nonnegative",
        "oos_cost_drag_explainable",
        "drawdown_controlled",
        "trade_count_sufficient",
        "stable_candidate",
        "strategy_v2_candidate",
    ]
    compare_df = pd.DataFrame(rows)
    if compare_df.empty:
        return pd.DataFrame(columns=columns)
    return compare_df[columns].sort_values(
        ["stable_candidate", "oos_net_pnl", "oos_no_cost_net_pnl"],
        ascending=[False, False, False],
        kind="stable",
    ).reset_index(drop=True)


def dataframe_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Convert a DataFrame to JSON-safe records."""

    if df.empty:
        return []
    safe_df = df.replace({np.nan: None})
    return json.loads(safe_df.to_json(orient="records", force_ascii=False))


def build_summary(split_dirs: dict[str, Path], output_dir: Path, compare_df: pd.DataFrame) -> dict[str, Any]:
    """Build trend_compare_summary.json payload."""

    stable = compare_df[compare_df["stable_candidate"] == True].copy() if not compare_df.empty else pd.DataFrame()  # noqa: E712
    oos_no_cost = pd.to_numeric(compare_df.get("oos_no_cost_net_pnl"), errors="coerce").fillna(0.0)
    oos_cost = pd.to_numeric(compare_df.get("oos_net_pnl"), errors="coerce").fillna(0.0)
    trend_failed = bool(not compare_df.empty and (oos_no_cost < 0).all() and (oos_cost < 0).all())
    return {
        "split_dirs": {split: str(path) for split, path in split_dirs.items()},
        "output_dir": str(output_dir),
        "min_trade_count": MIN_TRADE_COUNT,
        "max_ddpercent_threshold": MAX_DDPERCENT_THRESHOLD,
        "low_frequency_trade_count": LOW_FREQUENCY_TRADE_COUNT,
        "max_explainable_oos_cost_loss_to_no_cost": MAX_EXPLAINABLE_OOS_COST_LOSS_TO_NO_COST,
        "stable_candidate_rule": (
            "train/validation/oos no-cost > 0; every split trade_count >= min_trade_count; "
            "OOS cost-aware net_pnl >= 0 or explicitly marked low-frequency small cost-drag exception; "
            "every split max_ddpercent <= max_ddpercent_threshold."
        ),
        "oos_cost_exception_rule": (
            "trade_count <= low_frequency_trade_count, no_cost > 0, net_pnl < 0, cost_drag > 0, "
            "and abs(oos_net_pnl) <= max_explainable_oos_cost_loss_to_no_cost * oos_no_cost_net_pnl."
        ),
        "policy_count": int(compare_df["policy_name"].nunique()) if not compare_df.empty else 0,
        "stable_candidates": dataframe_records(stable),
        "stable_candidate_exists": bool(not stable.empty),
        "trend_following_v2_failed": trend_failed,
        "oos_all_policies_no_cost_and_cost_negative": trend_failed,
        "overfit_risk": bool(
            not compare_df.empty
            and (
                (compare_df["train_no_cost_positive"] == True)  # noqa: E712
                & (
                    (compare_df["validation_no_cost_positive"] == False)  # noqa: E712
                    | (compare_df["oos_no_cost_positive"] == False)  # noqa: E712
                )
            ).any()
        ),
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
            f"- {row.get('policy_name')}: "
            f"train_no_cost={format_number(row.get('train_no_cost_net_pnl'), 4)}, "
            f"validation_no_cost={format_number(row.get('validation_no_cost_net_pnl'), 4)}, "
            f"oos_no_cost={format_number(row.get('oos_no_cost_net_pnl'), 4)}, "
            f"oos_net={format_number(row.get('oos_net_pnl'), 4)}, "
            f"oos_dd%={format_number(row.get('oos_max_ddpercent'), 2)}"
        )
    return "\n".join(lines)


def render_markdown(summary: dict[str, Any], compare_df: pd.DataFrame) -> str:
    """Render trend_compare_report.md."""

    stable = summary.get("stable_candidates") or []
    overfit = compare_df[
        (compare_df["train_no_cost_positive"] == True)  # noqa: E712
        & (
            (compare_df["validation_no_cost_positive"] == False)  # noqa: E712
            | (compare_df["oos_no_cost_positive"] == False)  # noqa: E712
        )
    ].copy() if not compare_df.empty else pd.DataFrame()
    overfit_rows = dataframe_records(overfit.head(12))
    return (
        "# Trend Following V2 跨样本比较\n\n"
        "## 核心结论\n"
        f"- stable_candidate_exists={str(bool(summary.get('stable_candidate_exists'))).lower()}\n"
        f"- trend_following_v2_failed={str(bool(summary.get('trend_following_v2_failed'))).lower()}\n"
        f"- overfit_risk={str(bool(summary.get('overfit_risk'))).lower()}\n"
        f"- stable_candidate 需要 train/validation/oos no-cost 均为正、OOS 成本后不亏或低频小额成本拖累例外、回撤受控、每段 trade_count>={summary.get('min_trade_count')}。\n"
        f"- 低频成本例外阈值：trade_count<={summary.get('low_frequency_trade_count')} 且 abs(oos_net_pnl)<={summary.get('max_explainable_oos_cost_loss_to_no_cost')}*oos_no_cost_net_pnl。\n\n"
        "## 稳定候选\n"
        f"{format_candidate_rows(stable)}\n\n"
        "## Train 正但 Validation/OOS 不稳定\n"
        f"{format_candidate_rows(overfit_rows)}\n\n"
        "## 输出文件\n"
        "- trend_compare_summary.json\n"
        "- trend_compare_leaderboard.csv\n"
        "- trend_compare_report.md\n"
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
    """Run the full cross-sample comparison workflow."""

    split_dirs = {"train": train_dir, "validation": validation_dir, "oos": oos_dir}
    all_df = read_split_outputs(split_dirs)
    compare_df = build_compare_leaderboard(all_df)
    summary = build_summary(split_dirs, output_dir, compare_df)
    markdown = render_markdown(summary, compare_df)

    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "trend_compare_summary.json", summary)
    write_dataframe(output_dir / "trend_compare_leaderboard.csv", compare_df)
    (output_dir / "trend_compare_report.md").write_text(markdown, encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""

    args = parse_args(argv)
    ensure_headless_runtime()
    logger = setup_logging("compare_trend_following_v2", verbose=args.verbose)
    try:
        train_dir = resolve_path(args.train_dir)
        validation_dir = resolve_path(args.validation_dir)
        oos_dir = resolve_path(args.oos_dir)
        output_dir = resolve_path(args.output_dir)
        summary = run_compare(train_dir, validation_dir, oos_dir, output_dir)
        print_json_block(
            "Trend Following V2 comparison summary:",
            {
                "output_dir": output_dir,
                "stable_candidate_exists": summary.get("stable_candidate_exists"),
                "trend_following_v2_failed": summary.get("trend_following_v2_failed"),
                "stable_candidates": summary.get("stable_candidates"),
            },
        )
        if args.json:
            print(json.dumps(to_jsonable(summary), ensure_ascii=False, indent=2))
        return 0
    except TrendFollowingV2CompareError as exc:
        log_event(logger, logging.ERROR, "trend_following_v2_compare.error", str(exc))
        return 2
    except Exception:
        logger.exception(
            "Unexpected error during Trend Following V2 comparison",
            extra={"event": "trend_following_v2_compare.unexpected_error"},
        )
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
